#pragma once
// On-target kernel profile: every kernel captured into the batched graphs gets a timing event before and after it
// (event record nodes on the kernel's own stream: no new cross-stream dependencies). Each fherma_run's events are read
// after the run (in d2h, or at the next run / free when no d2h follows) and the last 4 calls are printed once at free as
// compact JSON lines on stdout ("kprof" key). FHERMA_KPROF=0 disables the capture-time event nodes.
#include <cuda.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

struct KProf {
    enum {P1,P2,P3,CRT,OTHER,NS};
    static constexpr const char* sname[NS]={"P1","P2","P3","CRT","other"};
    struct Name { CUfunction fn; const char* name; };
    struct Trace { CUfunction fn; unsigned gx,gy,gz,bx,smem; CUstream s; int product,chain,stage,gidx; CUevent e0,e1; };
    struct ChainStat { double busy=0,gaps=0,gap_max=0,first=0,last=0; int kernels=0,gaps10=0; };
    struct Call { unsigned long long index; int variant; double wall_ms,span_ms,union_ms,sum_ms,lead_ms,tail_ms,stage_ms[NS]; std::vector<ChainStat> chains; int sm_mhz,mem_mhz; };
    struct KStat { double sum=0; unsigned long long n=0; };
    decltype(&::cuEventCreate) ecreate=nullptr; decltype(&::cuEventRecord) erecord=nullptr;
    decltype(&::cuEventRecordWithFlags) erecordf=nullptr; decltype(&::cuEventElapsedTime) elapsed=nullptr;
    decltype(&::cuEventDestroy) edestroy=nullptr; decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor) occ=nullptr;
    decltype(&::cuFuncGetAttribute) fattr=nullptr; decltype(&::cuDeviceGetAttribute) dattr=nullptr; decltype(&::cuDeviceGetName) dname=nullptr;
    bool enabled=true,capturing=false,pending=false,reported=false;int variant=-1,product=-1,chain=-1,last_err=0;
    std::vector<Name> names; std::vector<Trace> traces[8]; std::vector<int> seen;
    CUevent ref0=nullptr,ref1=nullptr; int pending_variant=-1; double pending_wall=0; unsigned long long calls=0;
    std::chrono::steady_clock::time_point t0;
    std::deque<Call> last; double wall_min=1e30,wall_sum=0;
    std::vector<double> gsum[NS]; std::vector<unsigned> gcnt[NS]; std::vector<KStat> kstat;   // last collected call
    // NVML (optional): current SM / memory clocks after each run
    void* nvml=nullptr; void* nvdev=nullptr; int (*nv_clock)(void*,int,unsigned*)=nullptr;
    template<class D> void init(D& driver){
        if(const char* e=std::getenv("FHERMA_KPROF"))enabled=*e!='0';
        ecreate=driver.template get<decltype(ecreate)>("cuEventCreate");erecord=driver.template get<decltype(erecord)>("cuEventRecord");
        erecordf=driver.template get<decltype(erecordf)>("cuEventRecordWithFlags");elapsed=driver.template get<decltype(elapsed)>("cuEventElapsedTime_v2");
        if(!elapsed)elapsed=driver.template get<decltype(elapsed)>("cuEventElapsedTime");
        edestroy=driver.template get<decltype(edestroy)>("cuEventDestroy_v2");occ=driver.template get<decltype(occ)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
        fattr=driver.template get<decltype(fattr)>("cuFuncGetAttribute");dattr=driver.template get<decltype(dattr)>("cuDeviceGetAttribute");
        dname=driver.template get<decltype(dname)>("cuDeviceGetName");
        if(!ecreate||!erecord||!erecordf||!elapsed)enabled=false;
        if(enabled){ecreate(&ref0,CU_EVENT_DEFAULT);ecreate(&ref1,CU_EVENT_DEFAULT);
            nvml=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);
            if(nvml){auto ini=reinterpret_cast<int(*)()>(dlsym(nvml,"nvmlInit_v2"));auto h=reinterpret_cast<int(*)(unsigned,void**)>(dlsym(nvml,"nvmlDeviceGetHandleByIndex_v2"));
                nv_clock=reinterpret_cast<int(*)(void*,int,unsigned*)>(dlsym(nvml,"nvmlDeviceGetClockInfo"));
                if(!ini||!h||!nv_clock||ini()!=0||h(0,&nvdev)!=0){nv_clock=nullptr;nvdev=nullptr;}}}
    }
    void add_name(CUfunction f,const char* n){if(f)names.push_back({f,n});}
    const char* name_of(CUfunction f)const{for(const auto& x:names)if(x.fn==f)return x.name;return "?";}
    static int stage_of(const char* n){
        if(std::strstr(n,"crt"))return CRT;if(std::strstr(n,"middle"))return P2;
        if(std::strstr(n,"last")||std::strstr(n,"inverse"))return P3;
        if(std::strstr(n,"first")||std::strstr(n,"p1")||std::strstr(n,"prep")||std::strstr(n,"forward"))return P1;return OTHER;}
    void begin_variant(int v){if(!enabled||v<0||v>=8)return;capturing=true;variant=v;seen.assign(size_t(4096)*NS,0);}
    void end_variant(){capturing=false;variant=-1;product=chain=-1;}
    bool tracing(CUstream s)const{return capturing&&s;}
    CUevent pre(CUstream s){CUevent e=nullptr;ecreate(&e,CU_EVENT_DEFAULT);erecordf(e,s,CU_EVENT_RECORD_EXTERNAL);return e;}
    void post(CUevent e0,CUfunction fn,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned smem,CUstream s){
        CUevent e1=nullptr;ecreate(&e1,CU_EVENT_DEFAULT);erecordf(e1,s,CU_EVENT_RECORD_EXTERNAL);
        const int st=stage_of(name_of(fn));int gi=0;
        if(product>=0&&product<4096)gi=seen[size_t(product)*NS+st]++;
        traces[variant].push_back({fn,gx,gy,gz,bx,smem,s,product,chain,st,gi,e0,e1});}
    // run(): around the graph launch (ref events are outside the graph, on the launch stream)
    void run_begin(CUstream s,int v){if(!enabled||v<0||v>=8||traces[v].empty()){pending_variant=-1;return;}t0=std::chrono::steady_clock::now();erecord(ref0,s);pending_variant=v;}
    int run_sm=0,run_mem=0;   // clocks sampled right after the graph launch (GPU busy)
    void run_launched(CUstream s){if(pending_variant<0)return;erecord(ref1,s);unsigned m=0;run_sm=run_mem=0;if(nv_clock&&nv_clock(nvdev,1,&m)==0)run_sm=int(m);if(nv_clock&&nv_clock(nvdev,2,&m)==0)run_mem=int(m);}
    void run_end(){if(pending_variant<0)return;pending_wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();pending=true;}
    float ms(CUevent a,CUevent b){float x=0;CUresult r=elapsed(&x,a,b);if(r!=CUDA_SUCCESS){last_err=int(r);return 0.f;}return x;}
    void collect(){
        if(!pending)return;pending=false;const int v=pending_variant;pending_variant=-1;auto& tr=traces[v];
        Call c{};c.index=++calls;c.variant=v;c.wall_ms=pending_wall;c.span_ms=ms(ref0,ref1);
        wall_min=std::min(wall_min,c.wall_ms);wall_sum+=c.wall_ms;
        struct I{double s,e;int chain,stage,gidx;size_t t;};std::vector<I> iv;iv.reserve(tr.size());
        for(size_t t=0;t<tr.size();++t){const auto& x=tr[t];iv.push_back({ms(ref0,x.e0),ms(ref0,x.e1),x.chain,x.stage,x.gidx,t});}
        int maxchain=0;for(auto& x:iv)maxchain=std::max(maxchain,x.chain);c.chains.assign(size_t(maxchain)+1,{});
        for(int s=0;s<NS;++s){gsum[s].assign(64,0);gcnt[s].assign(64,0);}
        // kernel table key = trace config (fn, grid, block, smem)
        kstat.assign(tr.size(),{});
        double first=1e30,lastend=0;
        for(auto& x:iv){const double d=x.e-x.s;c.stage_ms[x.stage]+=d;c.sum_ms+=d;first=std::min(first,x.s);lastend=std::max(lastend,x.e);
            if(x.gidx<64){gsum[x.stage][x.gidx]+=d;gcnt[x.stage][x.gidx]++;}
            size_t key=x.t;for(size_t u=0;u<x.t;++u){const auto& a=tr[u];const auto& b=tr[x.t];if(a.fn==b.fn&&a.gx==b.gx&&a.gy==b.gy&&a.gz==b.gz&&a.bx==b.bx&&a.smem==b.smem){key=u;break;}}
            kstat[key].sum+=d;kstat[key].n++;}
        c.lead_ms=iv.empty()?0:first;c.tail_ms=iv.empty()?0:c.span_ms-lastend;
        {auto u=iv;std::sort(u.begin(),u.end(),[](const I& a,const I& b){return a.s<b.s;});double cs=-1,ce=-1;
         for(auto& x:u){if(x.s>ce){if(ce>cs)c.union_ms+=ce-cs;cs=x.s;ce=x.e;}else ce=std::max(ce,x.e);}if(ce>cs)c.union_ms+=ce-cs;}
        for(int ch=0;ch<=maxchain;++ch){std::vector<I> u;for(auto& x:iv)if(x.chain==ch)u.push_back(x);
            std::sort(u.begin(),u.end(),[](const I& a,const I& b){return a.s<b.s;});auto& cs=c.chains[size_t(ch)];cs.kernels=int(u.size());
            for(size_t i=0;i<u.size();++i){cs.busy+=u[i].e-u[i].s;if(i){const double g=u[i].s-u[i-1].e;if(g>0){cs.gaps+=g;cs.gap_max=std::max(cs.gap_max,g);if(g>0.01)cs.gaps10++;}}}
            if(!u.empty()){cs.first=u.front().s;cs.last=u.back().e;}}
        c.sm_mhz=run_sm;c.mem_mhz=run_mem;
        last.push_back(std::move(c));if(last.size()>4)last.pop_front();
    }
    void report(){
        if(reported)return;reported=true;collect();
        if(!enabled){std::printf("{\"kprof\":\"off\"}\n");std::fflush(stdout);return;}
        CUdevice dev=0;int sms=0,l2=0,clk=0,mclk=0,bus=0,pers=0,maj=0,mnr=0,smem_sm=0,regs_sm=0;char nm[128]={0};
        if(dattr){dattr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);dattr(&l2,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,dev);dattr(&clk,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev);
            dattr(&mclk,CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,dev);dattr(&bus,CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,dev);dattr(&pers,CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,dev);
            dattr(&maj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);dattr(&mnr,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
            dattr(&smem_sm,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,dev);dattr(&regs_sm,CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,dev);}
        if(dname)dname(nm,127,dev);for(char* p=nm;*p;++p)if(*p=='"'||*p=='\\')*p=' ';
        size_t nodes=0;for(auto& t:traces)nodes+=t.size();
        std::printf("{\"kprof\":\"dev\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2\":%d,\"clk_khz\":%d,\"memclk_khz\":%d,\"bus_bits\":%d,\"persist_l2_max\":%d,\"smem_sm\":%d,\"regs_sm\":%d,\"nvml\":%s,\"calls\":%llu,\"wall_min\":%.3f,\"wall_mean\":%.3f,\"timed_kernels_all_variants\":%zu,\"err\":%d}\n",
            nm,maj,mnr,sms,l2,clk,mclk,bus,pers,smem_sm,regs_sm,nv_clock?"true":"false",calls,calls?wall_min:0.0,calls?wall_sum/double(calls):0.0,nodes,last_err);
        if(last.empty()){std::fflush(stdout);return;}
        const auto& tr=traces[last.back().variant];int lines=0;
        for(size_t t=0;t<tr.size()&&t<kstat.size()&&lines<24;++t){if(!kstat[t].n)continue;++lines;const auto& x=tr[t];
            int regs=0,bpsm=0,lsmem=0;if(fattr){fattr(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,x.fn);fattr(&lsmem,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,x.fn);}
            if(occ)occ(&bpsm,x.fn,int(x.bx),size_t(x.smem));
            const double blocks=double(x.gx)*x.gy*x.gz;const double waves=bpsm&&sms?blocks/(double(bpsm)*sms):0;
            std::printf("{\"kprof\":\"kernel\",\"name\":\"%s\",\"stage\":\"%s\",\"grid\":[%u,%u,%u],\"block\":%u,\"dsmem\":%u,\"ssmem\":%d,\"regs\":%d,\"blocks_per_sm\":%d,\"warps_per_sm\":%d,\"waves\":%.2f,\"n\":%llu,\"ms_mean\":%.4f,\"ms_sum\":%.3f}\n",
                name_of(x.fn),sname[x.stage],x.gx,x.gy,x.gz,x.bx,x.smem,lsmem,regs,bpsm,bpsm*int((x.bx+31)/32),waves,kstat[t].n,kstat[t].sum/double(kstat[t].n),kstat[t].sum);}
        for(const auto& c:last){
            std::printf("{\"kprof\":\"call\",\"i\":%llu,\"v\":%d,\"wall_ms\":%.3f,\"gpu_span_ms\":%.3f,\"busy_union_ms\":%.3f,\"idle_ms\":%.3f,\"kern_sum_ms\":%.3f,\"lead_ms\":%.3f,\"tail_ms\":%.3f,\"sm_mhz\":%d,\"mem_mhz\":%d,\"stage_ms\":{",
                c.index,c.variant,c.wall_ms,c.span_ms,c.union_ms,c.span_ms-c.union_ms,c.sum_ms,c.lead_ms,c.tail_ms,c.sm_mhz,c.mem_mhz);
            for(int s=0;s<NS;++s)std::printf("%s\"%s\":%.3f",s?",":"",sname[s],c.stage_ms[s]);
            std::printf("},\"chains\":[");
            for(size_t i=0;i<c.chains.size();++i){const auto& h=c.chains[i];std::printf("%s{\"k\":%d,\"busy\":%.3f,\"gaps\":%.3f,\"gap_max\":%.4f,\"gaps_gt10us\":%d,\"start\":%.3f,\"end\":%.3f}",i?",":"",h.kernels,h.busy,h.gaps,h.gap_max,h.gaps10,h.first,h.last);}
            std::printf("]}\n");}
        std::printf("{\"kprof\":\"per_group_mean_ms\",\"call\":%llu",last.back().index);
        for(int s=0;s<NS;++s){int top=0;for(int g=0;g<64;++g)if(gcnt[s][g])top=g+1;if(!top)continue;std::printf(",\"%s\":[",sname[s]);
            for(int g=0;g<top;++g)std::printf("%s%.4f",g?",":"",gcnt[s][g]?gsum[s][g]/gcnt[s][g]:0.0);std::printf("]");}
        std::printf("}\n");std::fflush(stdout);
    }
};
