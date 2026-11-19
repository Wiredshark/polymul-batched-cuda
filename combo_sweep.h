#pragma once
// Round 20 (p23, on addc191 = 9758254 = pv62 production): P2 / P3 busy-SM-time lines (p23_sweep.cuh kernels, module "var" =
// production source + P23_SWEEP_SOURCE). Keys: ctrl / rc / ctrl_end controls; p2k (cb_p2k), p3k (cb_p3k), p3r16h / p3r16t
// (cb_p3r16h / cb_p3r16t, 128 threads, r16 table: BatchOvr bx[2] = 128, p3tab = 1), p3r16n (production bpo_last_r16 = the
// non-Harvey opt-in kernel, no L2 hints: reference); top-3 repeats (_r); combo = best P2 + best P3 line <= FHERMA_COMBO_T;
// c_p2k_p3r16h = that pair unconditionally.
// iso_* lines: the same batch graph with only P2 (FHERMA_STAGE_MASK 8) or only P3 (16) launched (ex 0 by design): isolated
// stage throughput under the 2-chain schedule; compare med_us within a stage (iso_p2 / iso_p2_e and iso_p3 / iso_p3_e = prod).
// combo_regs: JIT registers / local bytes per sweep kernel on the device. p2k / p3k / p3r16*: see p23_sweep.cuh.
// Mechanism (round 19, combo2): exit-time
// WHOLE-BATCH lines (State::combo_sweep -> run_combo). Every line recaptures the full B-product production graph (variant 6, as
// fherma_run launches it) with one set of substitutions (State::batch_line: per-stage kernel from a sweep module, environment
// overrides read at capture time, CRT prefetch items) and times it against the production graph exec in paired, alternating
// reps (vs_prod = median of pair ratios; < 1 = faster). ex = 1 when all B outputs are bit-identical to production (device compare).
// Sweep modules: the production NVRTC source (prefix + kernels, same arch option) with every extern "C" kernel turned into an
// uninstantiated template (only the named instances compile); compiled in worker threads at exit.
// (Round-19 keys gfl6 / gfl13 / pf* / p3all / crtef / p2el: measured in pv63, all >= 0.9968, removed.)
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
#include "p23_sweep_src.h"
template<class C> inline void run_combo(C& d,const std::string& src,unsigned p1_grid,unsigned cols,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):55.0;
    const double thr=std::getenv("FHERMA_COMBO_T")?std::atof(std::getenv("FHERMA_COMBO_T")):0.996;
    // ---- sweep modules
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const std::string P2="bpo_middle_r16<0>",P3="bpo_last_cm3<0>",CR="bpo_crt_lean<0>",P3R="bpo_last_r16<0>";
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",{P2,P3,CR,"cb_cmp",P3R});
    // var: cb_p3k instantiates the production bpo_p3_body with 2p from a lane-0 shuffle instead of the bpo_p2_of constant lookup
    // (only cb_* kernels are instantiated in this module, so the substitution reaches no other kernel)
    std::string v=s;int nsub=0;{const std::string a="bpo_p2_of(p)",b="__shfl_sync(0xffffffffu,2u*(p),0)";size_t q=0;while((q=v.find(a,q))!=std::string::npos){v.replace(q,a.size(),b);q+=b.size();++nsub;}}
    const int m_var=M.add("var",v+"\n"+P23_SWEEP_SOURCE,{"cb_p2k","cb_p3k","cb_p3r16h","cb_p3r16t"});
    M.build();
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":20,\"p2sub\":%d,\"p1_grid\":%u,\"crt_pf_prod\":%u,\"thr\":%.4f,\"compile_s\":%.1f,\"mods\":{%s}}\n",nsub,p1_grid,std::min(3*p1_grid/4,2u*cols),thr,el(),info.c_str());std::fflush(stdout);}
    {   // JIT register / local-memory (spill) footprint of every sweep kernel as loaded on this device
        auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");std::string r;
        auto one=[&](const char* nm,CUfunction f){if(!f)return;int rg=0,lc=0;ga(&rg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&lc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);char b[96];std::snprintf(b,sizeof b,"%s\"%s\":[%d,%d]",r.empty()?"":",",nm,rg,lc);r+=b;};
        one("p2",M.fn(m_rc,0));one("p3",M.fn(m_rc,1));one("p3r16n",M.fn(m_rc,4));one("p2k",M.fn(m_var,0));one("p3k",M.fn(m_var,1));one("p3r16h",M.fn(m_var,2));one("p3r16t",M.fn(m_var,3));
        std::printf("{\"diag\":\"sw\",\"k\":\"combo_regs\",\"regs_local\":{%s}}\n",r.c_str());std::fflush(stdout);}
    cmp=M.fn(m_rc,3);
    // ---- lines
    struct L{std::string v;int fam=0;CUfunction p2=nullptr,p3=nullptr,crt=nullptr;unsigned p3bx=0,p3tab=0;const char* mask=nullptr;BatchRes r;};
    auto run=[&](L& x)->bool{
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return false;}
        BatchOvr o;o.check=true;o.fn[1]=x.p2;o.fn[2]=x.p3;o.fn[3]=x.crt;o.bx[2]=x.p3bx;o.p3tab=x.p3tab;if(x.mask)o.env.push_back({"FHERMA_STAGE_MASK",x.mask});
        x.r=line(o,x.v.c_str());return x.r.ok;};
    std::vector<L> done;
    auto single=[&](L x,bool need){if(need&&!x.p2&&!x.p3&&!x.crt){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",x.v.c_str());std::fflush(stdout);return;}if(run(x))done.push_back(x);};
    auto mk=[](const char* v,int fam){L x;x.v=v;x.fam=fam;return x;};
    // families: 1 P2 kernel, 2 P3 kernel; 0 controls / references (never combined)
    single(mk("ctrl",0),false);
    {L x=mk("rc",0);x.p2=M.fn(m_rc,0);x.p3=M.fn(m_rc,1);x.crt=M.fn(m_rc,2);single(x,true);}
    {L x=mk("p2k",1);x.p2=M.fn(m_var,0);single(x,true);}
    {L x=mk("p3k",2);x.p3=M.fn(m_var,1);single(x,true);}
    {L x=mk("p3r16h",2);x.p3=M.fn(m_var,2);x.p3bx=128;x.p3tab=1;single(x,true);}
    {L x=mk("p3r16t",2);x.p3=M.fn(m_var,3);x.p3bx=128;x.p3tab=1;single(x,true);}
    {L x=mk("p3r16n",0);x.p3=M.fn(m_rc,4);x.p3bx=128;x.p3tab=1;single(x,true);}
    // repeats of the top single lines (exact, by vs_prod)
    std::vector<size_t> ord;for(size_t i=0;i<done.size();++i)if(done[i].fam>0&&done[i].r.ex==1)ord.push_back(i);
    std::sort(ord.begin(),ord.end(),[&](size_t a,size_t b){return done[a].r.vs<done[b].r.vs;});
    const size_t nsingle=done.size();
    for(size_t i=0;i<ord.size()&&i<3;++i){L x=done[ord[i]];x.v+="_r";if(run(x))done.push_back(x);}
    // per family: best exact line (mean with its repeat when repeated) <= thr
    std::vector<double> score(done.size(),0);
    for(size_t i=0;i<nsingle;++i){double s1=done[i].r.vs;int c=1;for(size_t j=nsingle;j<done.size();++j)if(done[j].v==done[i].v+"_r"&&done[j].r.ex==1){s1+=done[j].r.vs;++c;}score[i]=s1/c;}
    std::vector<int> best(3,-1);
    for(size_t i=0;i<nsingle;++i){const L& x=done[i];if(x.fam<=0||x.r.ex!=1||score[i]>thr)continue;if(best[size_t(x.fam)]<0||score[i]<score[size_t(best[size_t(x.fam)])])best[size_t(x.fam)]=int(i);}
    std::vector<int> win;for(int f=1;f<=2;++f)if(best[size_t(f)]>=0)win.push_back(best[size_t(f)]);
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_pick\",\"winners\":%zu,\"list\":\"",win.size());for(int i:win)std::printf("%s=%.4f ",done[size_t(i)].v.c_str(),score[size_t(i)]);std::printf("\"}\n");std::fflush(stdout);
    if(win.size()>=2){L c=mk("combo",-1);std::string nm="combo";
        for(int i:win){const L& x=done[size_t(i)];nm+=":"+x.v;if(x.p2)c.p2=x.p2;if(x.p3){c.p3=x.p3;c.p3bx=x.p3bx;c.p3tab=x.p3tab;}}
        c.v=nm;if(run(c)){done.push_back(c);L r=c;r.v+="_r";if(run(r))done.push_back(r);}}
    {L x=mk("c_p2k_p3r16h",-1);x.p2=M.fn(m_var,0);x.p3=M.fn(m_var,2);x.p3bx=128;x.p3tab=1;single(x,true);}   // fixed pair (local proxy is P2-memory-bound)
    // isolated stage throughput (whole-batch graph with FHERMA_STAGE_MASK: only P2 (8) or only P3 (16) launched; outputs not
    // produced -> ex 0 by design, compare med_us between iso lines of the same stage). FHERMA_P23_ISO=0 off.
    if(!(std::getenv("FHERMA_P23_ISO")&&*std::getenv("FHERMA_P23_ISO")=='0')){
        auto iso=[&](const char* v,const char* mask,CUfunction p2,CUfunction p3,unsigned bx,unsigned tab){L x=mk(v,0);x.p2=p2;x.p3=p3;x.p3bx=bx;x.p3tab=tab;x.mask=mask;single(x,false);};
        iso("iso_p2","8",nullptr,nullptr,0,0);iso("iso_p2k","8",M.fn(m_var,0),nullptr,0,0);
        iso("iso_p3","16",nullptr,nullptr,0,0);iso("iso_p3r16h","16",nullptr,M.fn(m_var,2),128,1);iso("iso_p3r16t","16",nullptr,M.fn(m_var,3),128,1);
        iso("iso_p3_e","16",nullptr,nullptr,0,0);iso("iso_p2_e","8",nullptr,nullptr,0,0);}
    single(mk("ctrl_end",0),false);
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
