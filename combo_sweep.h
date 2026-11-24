#pragma once
// Round 23 (kill, on 3140af3 = production): kill-test ablation lines (clean_sweep.cuh kernels, module "var" = production source
// + CL_SWEEP_SOURCE; wrong values by design, o.check off: ex 0 + abl 1, timing only). Keys: ctrl / rc / ctrl_end controls (rc =
// production P1 / P2 / P3 / CRT recompiled in the sweep module, exact); nostage (P1 phase B minus the len-16 stage, P2 and P3
// one extra dummy radix-2 stage each: P1 -> P2 stage move estimate), nostage_p1 (P1 minus one stage only: upper bound of the P1
// saving), notma (P1 phase B without the dense TMA staging copy and its per-pair __syncwarp, results stored in place), notma_bar
// (only the per-pair __syncwarp removed); every variant line repeated once (_r, after all singles). Same launch geometry as
// production. combo_regs: JIT registers / local bytes per kernel on the device. (Round 22 p1old / p1k120 / p1k112 removed.)
// Mechanism (round 19, combo2): exit-time
// WHOLE-BATCH lines (State::combo_sweep -> run_combo). Every line recaptures the full B-product production graph (variant 6, as
// fherma_run launches it) with one set of substitutions (State::batch_line: per-stage kernel from a sweep module, environment
// overrides read at capture time, CRT prefetch items) and times it against the production graph exec in paired, alternating
// reps (vs_prod = median of pair ratios; < 1 = faster). ex = 1 when all B outputs are bit-identical to production (device compare).
// Sweep modules: the production NVRTC source (prefix + kernels, same arch option) with every extern "C" kernel turned into an
// uninstantiated template (only the named instances compile); compiled in worker threads at exit.
// FHERMA_COMBO=0 off, FHERMA_COMBO_S time budget (s, default 55, compiles included), FHERMA_BATCH_R pairs per line (default 13).
#include <thread>
#include <atomic>
#include <cmath>

template<class C> struct ComboMods {
    struct Prog { std::string name,src; std::vector<std::string> kn; bool dd=false; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t=0; };
    C& d; std::string arch; std::vector<Prog> progs;
    explicit ComboMods(C& c):d(c){
        int cc1=0;CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
        if(const char* e=std::getenv("FHERMA_ARCH"))arch=e;else arch=cc1>=9?"compute_90":"compute_80";}   // = production module option
    ~ComboMods(){d.cuCtxSynchronize();for(auto& P:progs)if(P.mod)d.cuModuleUnload(P.mod);}
    int add(const char* name,std::string src,std::vector<std::string> kn,bool dd=false){Prog P;P.name=name;P.src=std::move(src);P.kn=std::move(kn);P.dd=dd;progs.push_back(std::move(P));return int(progs.size())-1;}
    void compile(Prog& P){
        const auto tA=std::chrono::steady_clock::now();
        try{
            auto addn=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*)>("nvrtcAddNameExpression");
            auto lown=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*,const char**)>("nvrtcGetLoweredName");
            auto setctx=d.driver.template get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");
            nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,P.src.c_str(),"combo.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("create");
            for(const auto& k:P.kn)if(k.find('<')!=std::string::npos)addn(pr,k.c_str());
            const std::string a="--gpu-architecture="+arch;const char* op[]={a.c_str(),"--std=c++17","-default-device"};
            if(d.nvrtcCompileProgram(pr,P.dd?3:2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
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
    void build(){   // all programs in parallel worker threads
        const unsigned hw=std::max(2u,std::thread::hardware_concurrency());const size_t nth=std::min<size_t>(progs.size(),std::max(1u,std::min(8u,hw-1)));
        std::atomic<size_t> next{0};std::vector<std::thread> th;
        for(size_t t=0;t<nth;++t)th.emplace_back([&]{for(size_t k;(k=next.fetch_add(1))<progs.size();)compile(progs[k]);});
        for(auto& t:th)t.join();}
    CUfunction fn(int p,size_t k=0)const{return p>=0&&size_t(p)<progs.size()&&k<progs[size_t(p)].fn.size()?progs[size_t(p)].fn[k]:nullptr;}
};

// src: production NVRTC source (prefix + kernels); line: State::batch_line; cmp: State::batch_cmp (set while the lines run).
#include <cstring>
#include "clean_sweep_src.h"
template<class C> inline void run_combo(C& d,const std::string& src,unsigned p1_grid,unsigned cols,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):55.0;
    const double thr=std::getenv("FHERMA_COMBO_T")?std::atof(std::getenv("FHERMA_COMBO_T")):0.996;
    // ---- sweep modules
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",
        {"bpo_middle_r16<0>","bpo_last_cm3<0>","bpo_crt_lean<0>","cb_cmp","bpo_first_x<0>"});
    static const char* const KN[]={"cl_k1ns","cl_k1nt","cl_k1nb","cl_k2x","cl_k3x"};
    constexpr size_t NK=sizeof(KN)/sizeof(KN[0]);
    const int m_var=M.add("var",s+"\n"+CL_SWEEP_SOURCE,std::vector<std::string>(KN,KN+NK));
    M.build();
    auto vf=[&](const char* k)->CUfunction{for(size_t i=0;i<NK;++i)if(!std::strcmp(KN[i],k))return M.fn(m_var,i);return nullptr;};
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":23,\"p1_grid\":%u,\"crt_pf_prod\":%u,\"thr\":%.4f,\"compile_s\":%.1f,\"mods\":{%s}}\n",p1_grid,std::min(3*p1_grid/4,2u*cols),thr,el(),info.c_str());std::fflush(stdout);}
    {   // JIT register / local-memory (spill) footprint of every sweep kernel as loaded on this device
        auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");std::string r;
        auto one=[&](const char* nm,CUfunction f){if(!f)return;int rg=0,lc=0;ga(&rg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&lc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);char b[96];std::snprintf(b,sizeof b,"%s\"%s\":[%d,%d]",r.empty()?"":",",nm,rg,lc);r+=b;};
        one("p1",M.fn(m_rc,4));one("p2",M.fn(m_rc,0));one("p3",M.fn(m_rc,1));one("crt",M.fn(m_rc,2));
        for(const char* k:KN)one(k+3,vf(k));
        std::printf("{\"diag\":\"sw\",\"k\":\"combo_regs\",\"regs_local\":{%s}}\n",r.c_str());std::fflush(stdout);}
    cmp=M.fn(m_rc,3);
    // ---- lines (stage index: 0 P1, 1 P2, 2 P3, 3 CRT)
    struct L{std::string v;int fam=0;CUfunction f[4]{};bool chk=true;BatchRes r;};
    auto run=[&](L& x)->bool{
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return false;}
        BatchOvr o;o.check=x.chk;for(int q=0;q<4;++q)o.fn[q]=x.f[q];
        x.r=line(o,x.v.c_str());return x.r.ok;};
    std::vector<L> done;
    auto single=[&](L x,bool need){if(need&&!x.f[0]&&!x.f[1]&&!x.f[2]&&!x.f[3]){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",x.v.c_str());std::fflush(stdout);return;}if(run(x))done.push_back(x);};
    // families: 1 P1, 2 P3, 3 CRT; 0 controls (never combined)
    auto mk=[](const char* v,int fam,int stage=-1,CUfunction f=nullptr){L x;x.v=v;x.fam=fam;if(stage>=0)x.f[stage]=f;return x;};
    single(mk("ctrl",0),false);
    {L x=mk("rc",0);x.f[0]=M.fn(m_rc,4);x.f[1]=M.fn(m_rc,0);x.f[2]=M.fn(m_rc,1);x.f[3]=M.fn(m_rc,2);single(x,true);}
    {   // kill ablations (wrong by design: unchecked); a line with any missing kernel is skipped
        auto kl=[&](const char* v,CUfunction f1,CUfunction f2,CUfunction f3,bool p23){L x=mk(v,1);x.chk=false;x.f[0]=f1;x.f[1]=f2;x.f[2]=f3;
            if(!f1||(p23&&(!f2||!f3))){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",v);std::fflush(stdout);return;}single(x,true);};
        kl("nostage",vf("cl_k1ns"),vf("cl_k2x"),vf("cl_k3x"),true);
        kl("nostage_p1",vf("cl_k1ns"),nullptr,nullptr,false);
        kl("notma",vf("cl_k1nt"),nullptr,nullptr,false);
        kl("notma_bar",vf("cl_k1nb"),nullptr,nullptr,false);}
    // one repeat of every variant line (paired against production again, after all singles)
    const size_t nsingle=done.size();
    for(size_t i=0;i<nsingle;++i){if(done[i].fam<=0)continue;L x=done[i];x.v+="_r";if(run(x))done.push_back(x);}
    single(mk("ctrl_end",0),false);
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
