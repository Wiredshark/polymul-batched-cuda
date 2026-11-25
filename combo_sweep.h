#pragma once
// diag7 (round 37, on b745d83): exact variants. Former: abl diag (round 35, on 0640c2f = pipe diag on 5ca0049): real-kernel class removal (BPO_RM). Former: pipe diag (round 34: what binds inside the production pipeline: compute / memory twins in the production schedule,
// L2 capacity probes). Former header: sched3 diag (round 33, on 5ca0049 = candidate 5; all four stages as work-pulling resident-slot grids, depth-first groups; production PTX unchanged by this file): whole-batch schedule variants aimed at
// "GPU full AND working set in L2". Every line = the production graph (variant 6) recaptured with one change, timed in paired
// alternating reps against the production graph exec; ex = 1 when all B outputs are bit-identical (trivial-kernel lines: not
// checked). One pass + a repeat of the main lines, ctrl first and last; < 50 lines in total. FHERMA_COMBO=0 off.
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

struct BatchOvr { CUfunction fn[4]{}; unsigned smem[4]{}; unsigned sm=0; bool check=true; std::vector<std::pair<const char*,const char*>> env;
    unsigned p1grid=0,p2grid=0,p3grid=0,crtgrid=0; unsigned p3cb=0,l2g=0; bool p2g8=false,grev=false,half=false,quiet=false; bool need[4]{}; CUdeviceptr ctr=0; };
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

// ---- pipe diag probes: L2 capacity curve (copy at footprint F MB per stream, 1 / 2 concurrent streams) and persistence
static const char* PIPE_PROBE_SRC=R"(
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

// which: 1 = capacity curve (1 and 2 streams), 2 = persistence, 3 = both; tag = pass suffix
template<class C> inline void pipe_probes(C& d,int which,const char* tag){
    auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
    auto ecreate=d.driver.template get<decltype(&::cuEventCreate)>("cuEventCreate");auto erecord=d.driver.template get<decltype(&::cuEventRecord)>("cuEventRecord");
    auto esync=d.driver.template get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");auto elapsed=d.driver.template get<decltype(&::cuEventElapsedTime)>("cuEventElapsedTime");
    auto edestroy=d.driver.template get<decltype(&::cuEventDestroy)>("cuEventDestroy_v2");auto swait=d.driver.template get<decltype(&::cuStreamWaitEvent)>("cuStreamWaitEvent");
    auto screate=d.driver.template get<decltype(&::cuStreamCreate)>("cuStreamCreate");auto sdestroy=d.driver.template get<decltype(&::cuStreamDestroy)>("cuStreamDestroy_v2");
    auto memset8=d.driver.template get<decltype(&::cuMemsetD8)>("cuMemsetD8_v2");
    auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    static ComboMods<C>* PM=nullptr;static int pm=-1;
    if(!PM){PM=new ComboMods<C>(d);pm=PM->add("probe",PIPE_PROBE_SRC,{"pr_rd","pr_wr","pr_cp"});PM->build();}
    CUfunction f[3]={PM->fn(pm,0),PM->fn(pm,1),PM->fn(pm,2)};
    if(!f[0]||!f[1]||!f[2]){std::printf("{\"diag\":\"pipe\",\"k\":\"probe\",\"error\":\"%.100s\"}\n",PM->progs[0].err.c_str());std::fflush(stdout);return;}
    int sms=0;CUdevice dev=0;if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);
    int ob=0;occ(&ob,f[2],256,0);const unsigned G=unsigned(std::max(1,sms)*std::max(1,ob));
    const size_t MB=size_t(1)<<20;const size_t cap=512*MB;   // 2 streams x 256 MB footprint
    CUdeviceptr buf=0,sk=0;
    if(d.cuMemAlloc(&buf,cap)!=CUDA_SUCCESS||d.cuMemAlloc(&sk,256)!=CUDA_SUCCESS){if(buf)d.cuMemFree(buf);std::printf("{\"diag\":\"pipe\",\"k\":\"probe\",\"error\":\"alloc\"}\n");std::fflush(stdout);return;}
    memset8(buf,0x5a,cap);d.cuCtxSynchronize();
    CUstream st[2]{};screate(&st[0],CU_STREAM_NON_BLOCKING);screate(&st[1],CU_STREAM_NON_BLOCKING);
    CUevent ea=nullptr,eb=nullptr,ej=nullptr;ecreate(&ea,CU_EVENT_DEFAULT);ecreate(&eb,CU_EVENT_DEFAULT);ecreate(&ej,CU_EVENT_DISABLE_TIMING);
    auto med=[](std::vector<double> z){std::sort(z.begin(),z.end());return z[z.size()/2];};
    if(which&1){
        const unsigned FP[8]={16,32,48,64,96,128,192,256};
        for(unsigned S=1;S<=2;++S){double gb[8];
            for(int fi=0;fi<8;++fi){const size_t F=size_t(FP[fi])*MB;unsigned reps=std::max(2u,unsigned(4096/FP[fi]));
                unsigned long long n4=(F/2)/16;const unsigned g=G/S;std::vector<double> v;
                for(int r=0;r<6;++r){erecord(ea,st[0]);if(S>1){swait(st[1],ea,0);}
                    for(unsigned q=0;q<S;++q){CUdeviceptr x=buf+q*256*MB,y=x+F/2;void* pc[]={&x,&y,&n4,&reps};d.cuLaunchKernel(f[2],g,1,1,256,1,1,0,st[q],pc,nullptr);}
                    if(S>1){erecord(ej,st[1]);swait(st[0],ej,0);}
                    erecord(eb,st[0]);esync(eb);float ms=0;elapsed(&ms,ea,eb);if(r)v.push_back(double(ms)*1e3);}
                gb[fi]=double(F)*reps*S/(med(v)*1e3);}
            std::printf("{\"diag\":\"pipe\",\"k\":\"l2cp\",\"v\":\"s%u%s\",\"MB_per_stream\":[16,32,48,64,96,128,192,256],\"GBps\":[%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f],\"grid\":%u}\n",
                S,tag,gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],G/S);std::fflush(stdout);}
    }
    if(which&2){   // write X MB, write Y MB elsewhere, read the first X back (only the read is timed)
        const unsigned X=67,YS[4]={0,34,67,101};double us[4],gb[4];
        for(int yi=0;yi<4;++yi){const size_t xb=size_t(X)*MB,yb=size_t(YS[yi])*MB;std::vector<double> v;
            unsigned long long nx=xb/16,ny=yb/16;unsigned one=1;CUdeviceptr xa=buf,ya=buf+256*MB,z=sk;
            for(int r=0;r<6;++r){
                {void* pw[]={&xa,&nx,&one};d.cuLaunchKernel(f[1],G,1,1,256,1,1,0,st[0],pw,nullptr);}
                if(ny){void* pw[]={&ya,&ny,&one};d.cuLaunchKernel(f[1],G,1,1,256,1,1,0,st[0],pw,nullptr);}
                erecord(ea,st[0]);{void* pr[]={&xa,&nx,&one,&z};d.cuLaunchKernel(f[0],G,1,1,256,1,1,0,st[0],pr,nullptr);}erecord(eb,st[0]);
                esync(eb);float ms=0;elapsed(&ms,ea,eb);if(r)v.push_back(double(ms)*1e3);}
            us[yi]=med(v);gb[yi]=double(xb)/(us[yi]*1e3);}
        std::printf("{\"diag\":\"pipe\",\"k\":\"persist\",\"v\":\"x67%s\",\"Y_MB\":[0,34,67,101],\"rd_us\":[%.1f,%.1f,%.1f,%.1f],\"GBps\":[%.0f,%.0f,%.0f,%.0f]}\n",
            tag,us[0],us[1],us[2],us[3],gb[0],gb[1],gb[2],gb[3]);std::fflush(stdout);
    }
    d.cuCtxSynchronize();edestroy(ea);edestroy(eb);edestroy(ej);sdestroy(st[0]);sdestroy(st[1]);d.cuMemFree(buf);d.cuMemFree(sk);
}

template<class C> inline void run_combo(C& d,const std::string& src,OvlTabs T,
    const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    // diag7 (round 37, on b745d83 = candidate 6): exact variants paired against the production graph (variant 6), every
    // output checked (ex); three passes, ctrl first and last.
    //  p1bs      bpo_first_xq: bpt from 16 KB dynamic smem (class t_p1tw 2.4 us incl. T1 + bpt)
    //  g16       P2 and P3 over all 16 primes in one launch each (FHERMA_PO_GROUP=16 + L2 group 16): 4 launches per product
    //  g8x2_16   P2 per 8 primes twice, P3 once over 16 (5 launches)
    //  crt4      bpo_crt_lean at __launch_bounds__(128,4)
    //  p3r44/52/56  bpo_last_cm3t (2 columns) at register cap 44 / 52 / 56 (production 48)
    //  pf_lo/hi  CRT next-input prefetch items x0.5 / x1.5 (class d_in 4.7 us)
    //  allnew    p1bs + crt4 + g16
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):300.0;
    int cc1=0,cc2=0,sms=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);attr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);}}
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",{"cb_cmp"});
    const int m_bs=M.add("p1bs","#define BPO_P1BS 1\n"+s,{"bpo_first_xq<0>"});
    const int m_c4=M.add("crt4","#define BPO_CRTMB 4\n"+s,{"bpo_crt_lean<0>"});
    const int m_r44=M.add("p3r44","#define BPO_P3TR 44\n"+s,{"bpo_last_cm3t<0>"}),m_r52=M.add("p3r52","#define BPO_P3TR 52\n"+s,{"bpo_last_cm3t<0>"}),m_r56=M.add("p3r56","#define BPO_P3TR 56\n"+s,{"bpo_last_cm3t<0>"});
    M.build();
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":\"d7\",\"cc\":\"%d.%d\",\"sms\":%d,\"B\":%u,\"mods\":%zu,\"pf\":%u,\"compile_s\":%.1f}\n",cc1,cc2,sms,T.b,M.progs.size(),T.pf_cols,el());
    for(const auto& P:M.progs)if(!P.mod)std::printf("{\"diag\":\"d7\",\"k\":\"mod\",\"m\":\"%s\",\"err\":\"%.120s\"}\n",P.name.c_str(),P.err.c_str());
    std::fflush(stdout);
    cmp=M.fn(m_rc,0);
    {   // every diag module's own __constant__ p2cu table = copy of the production module's (filled by build_tabs)
        auto gglob=d.driver.template get<decltype(&::cuModuleGetGlobal)>("cuModuleGetGlobal_v2");auto cpy=d.driver.template get<decltype(&::cuMemcpyDtoD)>("cuMemcpyDtoD_v2");
        CUdeviceptr pc=0;size_t ps=0;
        if(gglob(&pc,&ps,d.module,"bpo_p2cu")==CUDA_SUCCESS)for(const auto& P:M.progs){CUdeviceptr wc=0;size_t ws=0;if(P.mod&&gglob(&wc,&ws,P.mod,"bpo_p2cu")==CUDA_SUCCESS&&ws==ps)cpy(wc,pc,ps);}
        d.cuCtxSynchronize();}
    const double B=T.b?double(T.b):1.0;
    CUfunction fbs=M.fn(m_bs),fc4=M.fn(m_c4),f44=M.fn(m_r44),f52=M.fn(m_r52),f56=M.fn(m_r56);
    {auto fsa=d.driver.template get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");if(fbs&&fsa(fbs,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(T.pad1+16384u))!=CUDA_SUCCESS)fbs=nullptr;}
    auto out=[&](const char* nm,BatchOvr o,bool need){if(!need){std::printf("{\"diag\":\"d7\",\"v\":\"%s\",\"skipped\":\"module\"}\n",nm);std::fflush(stdout);return;}
        if(el()>budget){std::printf("{\"diag\":\"d7\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",nm);std::fflush(stdout);return;}
        o.quiet=true;const BatchRes r=line(o,nm);if(!r.ok)return;
        std::printf("{\"diag\":\"d7\",\"v\":\"%s\",\"us\":%.2f,\"prod_us\":%.2f,\"vs_prod\":%.4f,\"ex\":%d}\n",nm,r.med/B,r.prod/B,r.vs,r.ex);std::fflush(stdout);};
    typedef std::vector<std::pair<const char*,const char*>> E;
    static std::string pflo,pfhi;pflo=std::to_string(std::max(1u,T.pf_cols/2));pfhi=std::to_string(std::min(512u,T.pf_cols*3/2));
    auto g16=[&](BatchOvr& o){o.env.push_back({"FHERMA_PO_GROUP","16"});o.l2g=16;};
    {BatchOvr o;out("ctrl",o,true);}
    for(int pass=0;pass<3;++pass){char nm[32];const char* sf=pass==0?"":pass==1?"_r":"_r2";auto N=[&](const char* a){std::snprintf(nm,sizeof nm,"%s%s",a,sf);return nm;};
        {BatchOvr o;o.fn[0]=fbs;o.smem[0]=16384u;out(N("p1bs"),o,fbs!=nullptr);}
        {BatchOvr o;g16(o);out(N("g16"),o,T.k==16);}
        {BatchOvr o;g16(o);o.p2g8=true;out(N("g8x2_16"),o,T.k==16);}
        {BatchOvr o;o.fn[3]=fc4;out(N("crt4"),o,fc4!=nullptr);}
        {BatchOvr o;o.fn[2]=f44;o.p3cb=2;out(N("p3r44"),o,f44!=nullptr);}
        {BatchOvr o;o.fn[2]=f52;o.p3cb=2;out(N("p3r52"),o,f52!=nullptr);}
        {BatchOvr o;o.fn[2]=f56;o.p3cb=2;out(N("p3r56"),o,f56!=nullptr);}
        {BatchOvr o;o.env=E{{"FHERMA_CRTPF_ITEMS",pflo.c_str()}};out(N("pf_lo"),o,T.pf_cols>1);}
        {BatchOvr o;o.env=E{{"FHERMA_CRTPF_ITEMS",pfhi.c_str()}};out(N("pf_hi"),o,T.pf_cols>1);}
        {BatchOvr o;o.fn[0]=fbs;o.smem[0]=16384u;o.fn[3]=fc4;g16(o);out(N("allnew"),o,fbs&&fc4&&T.k==16);}
    }
    {BatchOvr o;out("ctrl_end",o,true);}
    std::printf("{\"diag\":\"d7\",\"k\":\"done\",\"s\":%.2f}\n",el());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
