#pragma once
// memd diag (round 31, on 66508d6 = candidate 4; production PTX unchanged by this file): what the memory twins' time is made of.
// Exit-time lines only (after the timed run). Stage lines = the production graph (variant 6, chains, L2 schedule, all B
// products) recaptured with ONE stage enabled (stage mask) and its kernel substituted (same harness as pv97 ovl); med_us / B =
// us per product; outputs not checked. Schedule lines = the whole production graph recaptured with production-equivalent
// env schedules (outputs checked: ex). Probe lines = plain bandwidth kernels (full grid, 256 threads, uint4, grid-stride).
// Memory twin ablation bits BPO_MA (patch_memd_kern.py): 1 data loads from an L2-resident window, 2 stores off (loads kept
// live), 4 stores into an L2-resident window, 8 table loads off, 16 CRT next-input prefetch off, 32 P1 input as contiguous
// LDG.128 streams. P1 memory twin = BPO_XA 5 (input cp.async + TMA stores, no phase A/B math). GB/s = the variant's DRAM-class
// bytes per product (input, A/B, C, residues, output, CRT prefetch; tables excluded) / its time.
// FHERMA_COMBO=0 off, FHERMA_COMBO_S budget (default 240 s), FHERMA_MEMD_PROBE_MB (default 2048) DRAM probe buffer size.
#include <cuda.h>
#include <nvrtc.h>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <chrono>

struct BatchOvr { CUfunction fn[4]{}; unsigned smem[4]{}; unsigned sm=0; bool check=true; std::vector<std::pair<const char*,const char*>> env; };
struct BatchRes { bool ok=false; int ex=-1; double vs=0,med=0,prod=0; };
struct OvlTabs { CUfunction prod[4]{}; unsigned pad1=0; unsigned b=0; unsigned n=0,l=0,k=0,pf_cols=0; };

template<class C> struct ComboMods {
    struct Prog { std::string name,src; std::vector<std::string> kn; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t=0; };
    C& d; std::string arch; std::vector<Prog> progs;
    explicit ComboMods(C& c):d(c){
        int cc1=0;CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
        if(const char* e=std::getenv("FHERMA_ARCH"))arch=e;else arch=cc1>=9?"compute_90":"compute_80";}
    ~ComboMods(){d.cuCtxSynchronize();for(auto& P:progs)if(P.mod)d.cuModuleUnload(P.mod);}
    int add(const char* name,std::string src,std::vector<std::string> kn){Prog P;P.name=name;P.src=std::move(src);P.kn=std::move(kn);progs.push_back(std::move(P));return int(progs.size())-1;}
    void compile(Prog& P){
        const auto tA=std::chrono::steady_clock::now();
        try{
            auto addn=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*)>("nvrtcAddNameExpression");
            auto lown=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*,const char**)>("nvrtcGetLoweredName");
            auto setctx=d.driver.template get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");
            nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,P.src.c_str(),"combo.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("create");
            for(const auto& k:P.kn)if(k.find('<')!=std::string::npos)addn(pr,k.c_str());
            const std::string a="--gpu-architecture="+arch;const char* op[]={a.c_str(),"--std=c++17"};
            if(d.nvrtcCompileProgram(pr,2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
                const size_t e=l.find("error");throw std::runtime_error("nvrtc: "+l.substr(e==std::string::npos?0:e,160));}
            size_t n=0;d.nvrtcGetPTXSize(pr,&n);std::vector<char> img(n);d.nvrtcGetPTX(pr,img.data());
            std::vector<std::string> names(P.kn.size());
            for(size_t k=0;k<P.kn.size();++k){if(P.kn[k].find('<')==std::string::npos){names[k]=P.kn[k];continue;}const char* ln=nullptr;if(lown(pr,P.kn[k].c_str(),&ln)==NVRTC_SUCCESS&&ln)names[k]=ln;}
            d.nvrtcDestroyProgram(&pr);
            setctx(d.context);
            if(d.cuModuleLoadData(&P.mod,img.data())!=CUDA_SUCCESS){P.mod=nullptr;throw std::runtime_error("module load");}
            P.fn.assign(names.size(),nullptr);
            for(size_t k=0;k<names.size();++k){CUfunction f=nullptr;if(!names[k].empty()&&d.cuModuleGetFunction(&f,P.mod,names[k].c_str())==CUDA_SUCCESS)P.fn[k]=f;}
        }catch(const std::exception& e){P.err=e.what();for(char& ch:P.err)if(ch=='"'||ch=='\\'||ch=='\n'||ch=='\r')ch=' ';}
        P.t=std::chrono::duration<double>(std::chrono::steady_clock::now()-tA).count();}
    void build(){
        const unsigned hw=std::max(2u,std::thread::hardware_concurrency());const size_t nth=std::min<size_t>(progs.size(),std::max(1u,std::min(8u,hw-1)));
        std::atomic<size_t> next{0};std::vector<std::thread> th;
        for(size_t t=0;t<nth;++t)th.emplace_back([&]{for(size_t k;(k=next.fetch_add(1))<progs.size();)compile(progs[k]);});
        for(auto& t:th)t.join();}
    CUfunction fn(int p,size_t k=0)const{return p>=0&&size_t(p)<progs.size()&&k<progs[size_t(p)].fn.size()?progs[size_t(p)].fn[k]:nullptr;}
};

static const char* MEMD_PROBE_SRC=R"(
extern "C" __global__ void pr_rd(const uint4* s,unsigned long long n4,unsigned reps,unsigned* sink){
    unsigned acc=0;const unsigned long long st=(unsigned long long)gridDim.x*blockDim.x;
    for(unsigned r=0;r<reps;++r)
        #pragma unroll 4
        for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n4;i+=st){const uint4 v=__ldcg(s+i);acc^=v.x^v.y^v.z^v.w^r;}
    if(acc==0x9e3779b9u)sink[0]=acc;}
extern "C" __global__ void pr_wr(uint4* d,unsigned long long n4,unsigned reps){
    const unsigned long long st=(unsigned long long)gridDim.x*blockDim.x;
    for(unsigned r=0;r<reps;++r)
        #pragma unroll 4
        for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n4;i+=st)d[i]=make_uint4(unsigned(i),r,unsigned(i>>32),7u);}
extern "C" __global__ void pr_cp(const uint4* s,uint4* d,unsigned long long n4,unsigned reps){
    const unsigned long long st=(unsigned long long)gridDim.x*blockDim.x;
    for(unsigned r=0;r<reps;++r)
        #pragma unroll 4
        for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n4;i+=st){uint4 v=__ldcg(s+i);v.w^=r;d[i]=v;}}
)";

template<class C> inline void memd_probes(C& d,int pass,double el0,const std::function<double()>& el){
    (void)el0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
    auto ecreate=d.driver.template get<decltype(&::cuEventCreate)>("cuEventCreate");auto erecord=d.driver.template get<decltype(&::cuEventRecord)>("cuEventRecord");
    auto esync=d.driver.template get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");auto elapsed=d.driver.template get<decltype(&::cuEventElapsedTime)>("cuEventElapsedTime");
    auto edestroy=d.driver.template get<decltype(&::cuEventDestroy)>("cuEventDestroy_v2");auto meminfo=d.driver.template get<decltype(&::cuMemGetInfo)>("cuMemGetInfo_v2");
    auto memset8=d.driver.template get<decltype(&::cuMemsetD8)>("cuMemsetD8_v2");
    auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    static ComboMods<C>* PM=nullptr;static int pm=-1;
    if(!PM){PM=new ComboMods<C>(d);pm=PM->add("probe",MEMD_PROBE_SRC,{"pr_rd","pr_wr","pr_cp"});PM->build();
        if(!PM->fn(pm,0)){std::printf("{\"diag\":\"memd\",\"k\":\"probe\",\"error\":\"%.100s\"}\n",PM->progs[0].err.c_str());std::fflush(stdout);}}
    CUfunction f[3]={PM->fn(pm,0),PM->fn(pm,1),PM->fn(pm,2)};if(!f[0]||!f[1]||!f[2])return;
    int sms=0;CUdevice dev=0;if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);
    size_t fr=0,tot=0;meminfo(&fr,&tot);
    size_t mb=std::getenv("FHERMA_MEMD_PROBE_MB")?size_t(std::atoi(std::getenv("FHERMA_MEMD_PROBE_MB"))):2048u;
    while(mb>64&&size_t(2)*mb*(1u<<20)+(size_t(512)<<20)>fr)mb/=2;
    CUdeviceptr s=0,t=0,sk=0;
    if(d.cuMemAlloc(&s,mb<<20)!=CUDA_SUCCESS||d.cuMemAlloc(&t,mb<<20)!=CUDA_SUCCESS||d.cuMemAlloc(&sk,256)!=CUDA_SUCCESS){if(s)d.cuMemFree(s);if(t)d.cuMemFree(t);
        std::printf("{\"diag\":\"memd\",\"k\":\"probe\",\"error\":\"alloc %zu MB x2\"}\n",mb);std::fflush(stdout);return;}
    memset8(s,0x5a,mb<<20);memset8(t,0x3c,mb<<20);d.cuCtxSynchronize();   // touched once before timing
    CUevent ea=nullptr,eb=nullptr;ecreate(&ea,CU_EVENT_DEFAULT);ecreate(&eb,CU_EVENT_DEFAULT);
    for(int lvl=0;lvl<2;++lvl){   // 0: DRAM-sized (mb MB per buffer, 1 sweep), 1: 32 MB footprint (L2-resident; 32 sweeps)
        const size_t bytes=lvl?(size_t(32)<<20):(mb<<20);unsigned reps=lvl?32u:1u;
        const char* nm[3]={"rd","wr","cp"};
        for(int q=0;q<3;++q){
            int ob=0;occ(&ob,f[q],256,0);const unsigned grid=unsigned(std::max(1,sms)*std::max(1,ob));
            unsigned long long n4=(q==2?bytes/2:bytes)/16;CUdeviceptr x=s,y=t,z=sk;
            void* pr[]={&x,&n4,&reps,&z};void* pw[]={&y,&n4,&reps};void* pc[]={&x,&y,&n4,&reps};void** pp=q==0?pr:q==1?pw:pc;
            std::vector<double> v;
            for(int r=0;r<8;++r){erecord(ea,nullptr);d.cuLaunchKernel(f[q],grid,1,1,256,1,1,0,nullptr,pp,nullptr);erecord(eb,nullptr);esync(eb);float ms=0;elapsed(&ms,ea,eb);if(r)v.push_back(double(ms)*1e3);}
            std::sort(v.begin(),v.end());const double us=v[v.size()/2];const double moved=double(q==2?bytes:bytes)*reps;
            std::printf("{\"diag\":\"memd\",\"k\":\"probe\",\"pass\":%d,\"v\":\"%s_%s\",\"MB\":%.0f,\"reps\":%u,\"grid\":%u,\"us\":%.1f,\"GBps\":%.0f,\"s\":%.2f}\n",
                pass+1,lvl?"l2":"dram",nm[q],double(bytes)/(1<<20),reps,grid,us,moved/(us*1e3),el());std::fflush(stdout);}
    }
    edestroy(ea);edestroy(eb);d.cuCtxSynchronize();d.cuMemFree(s);d.cuMemFree(t);d.cuMemFree(sk);
}

template<class C> inline void run_combo(C& d,const std::string& src,OvlTabs T,
    const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();std::function<double()> el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):240.0;
    int cc1=0,cc2=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);}}
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",
        {"bpo_first_xt<0>","cb_cmp"});
    const int m_rev=M.add("p1rev","#define BPO_P1REV 1\n"+s,{"bpo_first_xt<0>"});
    // P1 memory twin variants (BPO_XA 5 + BPO_MA)
    struct V { const char* nm; unsigned ma; int m=-1; };
    std::vector<V> v1={{"M",0},{"Mld",2},{"Mst",1},{"Mstw",4},{"Mseq",32|2}};
    for(auto& x:v1){char b[40];std::snprintf(b,sizeof b,"p1%s",x.nm);x.m=M.add(b,"#define BPO_XA 5\n#define BPO_MA "+std::to_string(x.ma)+"\n"+s,{"bpo_first_xt<0>"});}
    // P2 / CRT memory twin variants (BPO_TWM wl4_p2m / wl4_crtm + BPO_MA); MA 0 also gives the P3 twin
    std::vector<V> v2={{"M",0},{"Mld",2},{"Mst",1|8|16},{"Mtab",1|2|16},{"Mldw",1},{"Mstw",4},{"Mnopf",16}};
    for(auto& x:v2){char b[40];std::snprintf(b,sizeof b,"twm%s",x.nm);x.m=M.add(b,"#define BPO_TWM 1\n#define BPO_MA "+std::to_string(x.ma)+"\n"+s,{"wl4_p2m<0>","wl4_p3m<0>","wl4_crtm<0>"});}
    M.build();
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":\"memd\",\"cc\":\"%d.%d\",\"B\":%u,\"mods\":%zu,\"compile_s\":%.1f}\n",cc1,cc2,T.b,M.progs.size(),el());
    for(const auto& P:M.progs)if(!P.mod)std::printf("{\"diag\":\"memd\",\"k\":\"mod\",\"m\":\"%s\",\"ok\":0,\"err\":\"%.120s\"}\n",P.name.c_str(),P.err.c_str());
    std::fflush(stdout);
    cmp=M.fn(m_rc,1);
    const double B=T.b?double(T.b):1.0;
    // DRAM-class bytes per product (MB)
    const double n=T.n,l=T.l,k=T.k,MBf=1.0/(1<<20);
    const double IN=2*n*l*4*MBf,AB=2*k*8*n*4*MBf,CC=k*8*n*4*MBf,OUT=n*l*4*MBf,PF=double(T.pf_cols)*256*l*4*MBf;
    auto run=[&](const char* v,BatchOvr o)->double{
        if(el()>budget){std::printf("{\"diag\":\"memd\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",v);std::fflush(stdout);return -1;}
        const BatchRes r=line(o,v);return r.ok?r.med/B:-1;};
    auto sum=[&](const char* st,int pass,const char* nm,double us,double mb,const char* what){
        std::printf("{\"diag\":\"memd\",\"k\":\"sum\",\"st\":\"%s\",\"pass\":%d,\"v\":\"%s\",\"us\":%.2f,\"MB\":%.1f,\"GBps\":%.0f,\"iso\":\"%s\"}\n",st,pass,nm,us,mb,us>0?mb*1048576.0/(us*1e3):0.0,what);std::fflush(stdout);};
    const unsigned P1=2u,P2=8u,P3=16u,CRT=32u;
    for(int pass=0;pass<2;++pass){
        char v[48];auto nm=[&](const char* a){std::snprintf(v,sizeof v,"%s%s",a,pass?"_r":"");return v;};
        {BatchOvr o;run(nm("ctrl"),o);}
        memd_probes(d,pass,0,el);
        // P1
        {BatchOvr o;o.sm=P1;o.check=false;sum("p1",pass+1,"R",run(nm("p1_R"),o),IN+AB,"production kernel");}
        {const char* what[5]={"input loads + A,B TMA stores, no math","input loads only (stores off)","A,B stores only (input from 16-column L2 window)","input loads real, stores into 16-column L2 window","input bytes as contiguous LDG.128 streams only"};
         const double mbv[5]={IN+AB,IN,AB,IN,IN};
         for(int i=0;i<5;++i){BatchOvr o;o.sm=P1;o.check=false;o.fn[0]=M.fn(v1[size_t(i)].m);char b[24];std::snprintf(b,sizeof b,"p1_%s",v1[size_t(i)].nm);if(o.fn[0])sum("p1",pass+1,v1[size_t(i)].nm,run(nm(b),o),mbv[i],what[i]);}}
        {BatchOvr o;o.sm=P1;o.check=false;o.env={{"FHERMA_PO_CHAINS","1"}};sum("p1",pass+1,"R_c1",run(nm("p1_R_c1"),o),IN+AB,"production kernel, 1 chain");}
        {BatchOvr o;o.sm=P1;o.check=false;o.fn[0]=M.fn(v1[0].m);o.env={{"FHERMA_PO_CHAINS","1"}};if(o.fn[0])sum("p1",pass+1,"M_c1",run(nm("p1_M_c1"),o),IN+AB,"memory twin, 1 chain");}
        // P2
        {BatchOvr o;o.sm=P2;o.check=false;sum("p2",pass+1,"R",run(nm("p2_R"),o),AB+CC,"production kernel");}
        {const char* what[6]={"A,B loads + tables + C stores, no math","A,B + table loads only","C stores only (A,B window, no tables)","table loads only","A,B window, tables + C stores real","A,B + tables real, C stores into window"};
         const double mbv[6]={AB+CC,AB,CC,0,CC,AB};
         for(int i=0;i<6;++i){BatchOvr o;o.sm=P2;o.check=false;o.fn[1]=M.fn(v2[size_t(i)].m,0);char b[24];std::snprintf(b,sizeof b,"p2_%s",v2[size_t(i)].nm);if(o.fn[1])sum("p2",pass+1,v2[size_t(i)].nm,run(nm(b),o),mbv[i],what[i]);}}
        // P3
        {BatchOvr o;o.sm=P3;o.check=false;sum("p3",pass+1,"R",run(nm("p3_R"),o),2*CC,"production kernel");}
        {BatchOvr o;o.sm=P3;o.check=false;o.fn[2]=M.fn(v2[0].m,1);if(o.fn[2])sum("p3",pass+1,"M",run(nm("p3_M"),o),2*CC,"C loads + tables + residue stores, no math");}
        // CRT
        {BatchOvr o;o.sm=CRT;o.check=false;sum("crt",pass+1,"R",run(nm("crt_R"),o),CC+OUT+PF,"production kernel");}
        {const char* what[7]={"residue loads + prefetch + table + out stores, no math","residue loads + prefetch + table (stores off)","out stores only (residue window, no table, no prefetch)","table only (window, stores off, no prefetch)","residue window, prefetch + out stores real","residue loads + prefetch real, out stores into window","no next-input prefetch, rest real"};
         const double mbv[7]={CC+OUT+PF,CC+PF,OUT,0,OUT+PF,CC+PF,CC+OUT};
         for(int i=0;i<7;++i){BatchOvr o;o.sm=CRT;o.check=false;o.fn[3]=M.fn(v2[size_t(i)].m,2);char b[24];std::snprintf(b,sizeof b,"crt_%s",v2[size_t(i)].nm);
            if(o.fn[3])sum("crt",pass+1,v2[size_t(i)].nm,run(nm(b),o),mbv[i],what[i]);}}
        {BatchOvr o;o.sm=CRT;o.check=false;o.env={{"FHERMA_PO_CHAINS","1"}};sum("crt",pass+1,"R_c1",run(nm("crt_R_c1"),o),CC+OUT+PF,"production kernel, 1 chain");}
        {BatchOvr o;o.sm=CRT;o.check=false;o.fn[3]=M.fn(v2[0].m,2);o.env={{"FHERMA_PO_CHAINS","1"}};if(o.fn[3])sum("crt",pass+1,"M_c1",run(nm("crt_M_c1"),o),CC+OUT+PF,"memory twin, 1 chain");}
        // whole-batch schedules (production-equivalent, outputs checked)
        {BatchOvr o;o.env={{"FHERMA_NO_M16","1"}};run(nm("s_nom16"),o);}
        {BatchOvr o;o.env={{"FHERMA_NO_M16","1"},{"FHERMA_PO_CHAINS","1"}};run(nm("s_nom16_c1"),o);}
        {BatchOvr o;o.env={{"FHERMA_PO_CHAINS","1"}};run(nm("s_c1"),o);}
        {BatchOvr o;o.env={{"FHERMA_PO_GROUP","4"}};run(nm("s_g4"),o);}
        {BatchOvr o;o.env={{"FHERMA_PO_GROUP","4"},{"FHERMA_PO_CHAINS","1"}};run(nm("s_g4_c1"),o);}
        {BatchOvr o;o.fn[0]=M.fn(m_rev);if(o.fn[0])run(nm("s_p1rev"),o);}
        {BatchOvr o;o.fn[0]=M.fn(m_rev);o.env={{"FHERMA_PO_CHAINS","1"}};if(o.fn[0])run(nm("s_p1rev_c1"),o);}
    }
    {BatchOvr o;run("ctrl_end",o);}
    std::printf("{\"diag\":\"memd\",\"k\":\"done\",\"s\":%.2f,\"MB\":[%.1f,%.1f,%.1f,%.1f,%.1f]}\n",el(),IN,AB,CC,OUT,PF);std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
