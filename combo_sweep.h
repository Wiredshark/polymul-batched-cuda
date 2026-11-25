#pragma once
// sched diag (round 32, on 5ca0049 = candidate 5; production PTX unchanged by this file): whole-batch schedule variants aimed at
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
    unsigned p1grid=0,p2grid=0; bool grev=false,half=false; CUdeviceptr ctr=0; };
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


template<class C> inline void run_combo(C& d,const std::string& src,OvlTabs T,
    const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):240.0;
    int cc1=0,cc2=0,sms=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);attr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);}}
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",{"cb_cmp"});
    const int m_wp=M.add("wp","#define BPO_P1WP 1\n#define BPO_P2WP 1\n"+s,{"bpo_first_xw<0>","bpo_middle_r16w<0>"});
    const int m_el=M.add("el","#define BPO_P1EL 1\n"+s,{"bpo_first_xr<0>"});
    const int m_tv=M.add("triv","extern \"C\" __global__ void triv_k(){}\n",{"triv_k"});
    M.build();
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":\"sched\",\"cc\":\"%d.%d\",\"sms\":%d,\"B\":%u,\"compile_s\":%.1f}\n",cc1,cc2,sms,T.b,el());
    for(const auto& P:M.progs)if(!P.mod)std::printf("{\"diag\":\"sched\",\"k\":\"mod\",\"m\":\"%s\",\"err\":\"%.120s\"}\n",P.name.c_str(),P.err.c_str());
    std::fflush(stdout);
    cmp=M.fn(m_rc,0);
    auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    auto gglob=d.driver.template get<decltype(&::cuModuleGetGlobal)>("cuModuleGetGlobal_v2");
    CUfunction w1=M.fn(m_wp,0),w2=M.fn(m_wp,1),fel=M.fn(m_el,0),tv=M.fn(m_tv,0);
    int o1=0,o2=0;if(w1)occ(&o1,w1,256,int(T.pad1));if(w2)occ(&o2,w2,128,0);
    CUdeviceptr ctr=0;size_t cb=0;if(M.progs[size_t(m_wp)].mod&&gglob(&ctr,&cb,M.progs[size_t(m_wp)].mod,"bpo_wp_ctr")!=CUDA_SUCCESS)ctr=0;
    {   // the wp module's own __constant__ p2cu table: copy of the production module's (filled by build_tabs)
        CUdeviceptr pc=0,wc=0;size_t ps=0,ws=0;auto cpy=d.driver.template get<decltype(&::cuMemcpyDtoD)>("cuMemcpyDtoD_v2");
        if(w2&&M.progs[size_t(m_wp)].mod&&gglob(&pc,&ps,d.module,"bpo_p2cu")==CUDA_SUCCESS&&gglob(&wc,&ws,M.progs[size_t(m_wp)].mod,"bpo_p2cu")==CUDA_SUCCESS&&ps==ws)
            {cpy(wc,pc,ps);d.cuCtxSynchronize();}else w2=nullptr;}
    const unsigned g1=unsigned(o1*sms),g2=unsigned(o2*sms);
    std::printf("{\"diag\":\"sched\",\"k\":\"wp\",\"p1_grid\":%u,\"p1_bpsm\":%d,\"p2_grid\":%u,\"p2_bpsm\":%d,\"ctr\":%d}\n",g1,o1,g2,o2,int(ctr!=0));std::fflush(stdout);
    auto run=[&](const char* v,BatchOvr o){if(el()>budget){std::printf("{\"diag\":\"sched\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",v);std::fflush(stdout);return;}line(o,v);};
    typedef std::vector<std::pair<const char*,const char*>> E;
    auto all=[&](int pass){
        auto nm=[&](const char* a){static char v[48];std::snprintf(v,sizeof v,"%s%s",a,pass?"_r":"");return v;};
        {BatchOvr o;o.env=E{{"FHERMA_PO_CHAINS","2"}};run(nm("c2"),o);}
        {BatchOvr o;o.env=E{{"FHERMA_PO_CHAINS","3"}};run(nm("c3"),o);}
        {BatchOvr o;o.env=E{{"FHERMA_PO_CHAINS","4"}};run(nm("c4"),o);}
        {BatchOvr o;o.half=true;run(nm("c2_half"),o);}
        {BatchOvr o;o.env=E{{"FHERMA_PO_CHAINS","1"}};run(nm("c1"),o);}
        if(w1&&ctr){BatchOvr o;o.fn[0]=w1;o.p1grid=g1;o.ctr=ctr;run(nm("wp1"),o);o.env=E{{"FHERMA_PO_CHAINS","1"}};run(nm("wp1_c1"),o);}
        if(w1&&w2&&ctr){BatchOvr o;o.fn[0]=w1;o.p1grid=g1;o.fn[1]=w2;o.p2grid=g2;o.ctr=ctr;run(nm("wp12"),o);o.env=E{{"FHERMA_PO_CHAINS","1"}};run(nm("wp12_c1"),o);}
        if(w1&&w2&&ctr){BatchOvr o;o.fn[0]=w1;o.p1grid=g1;o.fn[1]=w2;o.p2grid=g2;o.ctr=ctr;o.env=E{{"FHERMA_PO_CHAINS","3"}};run(nm("wp12_c3"),o);}
        {BatchOvr o;o.grev=true;run(nm("grev"),o);}
        if(fel){BatchOvr o;o.fn[0]=fel;run(nm("el"),o);}
        {BatchOvr o;o.env=E{{"FHERMA_CRTPF_ITEMS","0"}};run(nm("pf0"),o);}
        {BatchOvr o;o.env=E{{"FHERMA_CRTPF_ITEMS","512"}};run(nm("pf512"),o);}
        {BatchOvr o;o.env=E{{"FHERMA_NO_M16","1"}};run(nm("nom16"),o);}
        if(tv){BatchOvr o;for(int q=0;q<4;++q)o.fn[q]=tv;o.check=false;run(nm("triv"),o);o.env=E{{"FHERMA_PO_CHAINS","1"}};run(nm("triv_c1"),o);}
    };
    {BatchOvr o;run("ctrl",o);}
    all(0);all(1);
    {BatchOvr o;run("ctrl_end",o);}
    std::printf("{\"diag\":\"sched\",\"k\":\"done\",\"s\":%.2f}\n",el());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
