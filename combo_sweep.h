#pragma once
// Round 24 (p1ws, on 3140af3 = production): warp-specialized P1 lines (clean_sweep.cuh cl_p1ws / cl_p1ws_s, module "var" =
// production source + CL_SWEEP_SOURCE; exact, o.check on: ex compared). Keys: ctrl / rc / ctrl_end controls; p1ws (8 producer
// + 8 consumer warps, 1 persistent block per SM, Harvey consumers on the h30 quads 0-2), p1ws_s (same, Shoup consumers), p1ws16
// (8 producers + 16 consumers in two 8-warp teams, team t serves slot t; 768 threads -> 80-register cap, producers spill 128 B);
// iso_prod / iso_p1ws / iso_p1ws_s / iso_p1ws16 = P1-only whole batch (FHERMA_STAGE_MASK=2 at capture: the B products' P1 launches on the
// production chains, nothing else; unchecked, vs_prod is against the full production graph and meaningless: read med_us), one
// {"k":"iso"} summary line per variant (p1_us = med_us / B, vs_p1prod). Every line repeated once (_r). P1 substitutes launch
// with grid = SM count (p1_grid / 2, FHERMA_P1WS_GRID overrides), the kernel's launch-bound threads, CL_P1WS_SMEM_HOST dynamic smem.
// FHERMA_P1WS_ABL=1 adds timing ablations (wrong values, iso only): kc consumers idle, kp producers idle, kpa / kpb producers idle
// and consumers without phase A / phase B.
// combo_regs: JIT registers / local bytes per kernel on the device. (Round 23 kill lines removed.)
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
constexpr unsigned CL_P1WS_SMEM_HOST=97312u;   // = clean_sweep.cuh CL_P1WS_SMEM
inline std::string cl_sweep_source(const std::string& src){   // production source, kernels as uninstantiated templates, + round-24 kernels
    std::string s=src;const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}
    return s+"\n"+CL_SWEEP_SOURCE;}
inline unsigned cl_ws_grid(unsigned p1_grid){if(const char* e=std::getenv("FHERMA_P1WS_GRID"))return unsigned(std::atoi(e));return std::max(1u,p1_grid/2u);}   // one block per SM
template<class C> inline unsigned cl_ws_threads(C& d,CUfunction f){int v=0;auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");   // launch bound 32 (8 + NC)
    d.check(ga(&v,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f));return unsigned(v);}
template<class C> inline void cl_ws_attr(C& d,CUfunction f){auto fa=d.driver.template get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");
    d.check(fa(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(CL_P1WS_SMEM_HOST)));}
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
    static const char* const KN[]={"cl_p1ws","cl_p1ws_s","cl_p1ws16","cl_p1ws_kc","cl_p1ws_kp","cl_p1ws_kpa","cl_p1ws_kpb"};   // timing ablations (last 4) only with FHERMA_P1WS_ABL=1
    const size_t NK=std::getenv("FHERMA_P1WS_ABL")?sizeof(KN)/sizeof(KN[0]):3u;
    const int m_var=M.add("var",s+"\n"+CL_SWEEP_SOURCE,std::vector<std::string>(KN,KN+NK));
    M.build();
    auto vf=[&](const char* k)->CUfunction{for(size_t i=0;i<NK;++i)if(!std::strcmp(KN[i],k))return M.fn(m_var,i);return nullptr;};
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":24,\"p1_grid\":%u,\"crt_pf_prod\":%u,\"thr\":%.4f,\"compile_s\":%.1f,\"mods\":{%s}}\n",p1_grid,std::min(3*p1_grid/4,2u*cols),thr,el(),info.c_str());std::fflush(stdout);}
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
    const unsigned wsg=cl_ws_grid(p1_grid);
    for(size_t i=0;i<NK;++i)if(vf(KN[i]))cl_ws_attr(d,vf(KN[i]));
    auto wsl=[&](const char* v,CUfunction f,bool iso){L x=mk(v,iso?2:1);x.f[0]=f;x.chk=!iso;return x;};
    std::vector<std::pair<std::string,double>> isoms;   // iso med_us per line
    auto runl=[&](L x,bool need){
        if(need&&!x.f[0]){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",x.v.c_str());std::fflush(stdout);return;}
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return;}
        BatchOvr o;o.check=x.chk;o.fn[0]=x.f[0];
        if(x.f[0]){o.gx[0]=wsg;o.bx[0]=cl_ws_threads(d,x.f[0]);o.smem[0]=CL_P1WS_SMEM_HOST;}
        if(x.fam==2)o.env.push_back({"FHERMA_STAGE_MASK","2"});
        x.r=line(o,x.v.c_str());if(!x.r.ok)return;done.push_back(x);
        if(x.fam==2)isoms.push_back({x.v,x.r.med});};
    runl(wsl("p1ws",vf("cl_p1ws"),false),true);
    runl(wsl("p1ws_s",vf("cl_p1ws_s"),false),true);
    runl(wsl("p1ws16",vf("cl_p1ws16"),false),true);
    {L x=mk("iso_prod",2);x.chk=false;runl(x,false);}
    runl(wsl("iso_p1ws",vf("cl_p1ws"),true),true);
    runl(wsl("iso_p1ws_s",vf("cl_p1ws_s"),true),true);
    runl(wsl("iso_p1ws16",vf("cl_p1ws16"),true),true);
    if(std::getenv("FHERMA_P1WS_ABL")){runl(wsl("iso_kc",vf("cl_p1ws_kc"),true),true);runl(wsl("iso_kp",vf("cl_p1ws_kp"),true),true);runl(wsl("iso_kpa",vf("cl_p1ws_kpa"),true),true);runl(wsl("iso_kpb",vf("cl_p1ws_kpb"),true),true);}
    // one repeat of every variant line (paired against production again, after all singles)
    const size_t nsingle=done.size();
    for(size_t i=0;i<nsingle;++i){if(done[i].fam<=0)continue;L x=done[i];x.v+="_r";runl(x,false);}
    {   // iso summary: P1-only batch time per variant vs iso_prod (same repeat suffix)
        auto find=[&](const std::string& v)->double{for(auto& e:isoms)if(e.first==v)return e.second;return 0;};
        for(auto& e:isoms){if(e.first.rfind("iso_prod",0)==0)continue;const bool rr=e.first.size()>2&&e.first.compare(e.first.size()-2,2,"_r")==0;const double b0=find(rr?"iso_prod_r":"iso_prod");
            std::printf("{\"diag\":\"sw\",\"k\":\"iso\",\"v\":\"%s\",\"p1_batch_us\":%.1f,\"prod_p1_batch_us\":%.1f,\"vs_p1prod\":%.4f}\n",e.first.c_str(),e.second,b0,b0>0?e.second/b0:0.0);}
        std::fflush(stdout);}
    single(mk("ctrl_end",0),false);
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
