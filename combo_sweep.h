#pragma once
// Round 18 (combo, on 88de7fc = pv60): exit-time WHOLE-BATCH re-test of earlier near-winners on the current best, singly and in
// combination (State::combo_sweep -> run_combo). Every line recaptures the full B-product production graph (variant 6, as
// fherma_run launches it) with one set of substitutions (State::batch_line: per-stage kernel from a sweep module, environment
// overrides read at capture time, CRT prefetch items) and times it against the production graph exec in paired, alternating
// reps (vs_prod = median of pair ratios; < 1 = faster). ex = 1 when all B outputs are bit-identical to production (device compare).
// Sweep modules: the production NVRTC source (prefix + kernels, same arch option) with every extern "C" kernel turned into an
// uninstantiated template (only the named instance compiles) plus one define per variant; compiled in worker threads at exit.
// Keys (v) and what each maps to as a production default:
//   ctrl / ctrl_end   production config recaptured (capture control)          p1rc   P1 bpo_first_x recompiled (module control)
//   bvh               P1 BPO_XG |= 4194304 (XF_BVH: B fragments of digits 0-1 loaded before the quad barrier; ladder 512bec5)
//   p3el              FHERMA_GFLAGS = gflags|128 + P3 bpo_last_cm3 compiled with BPO_P3EL (group-0 P3 residue stores evict_last; crtd 915c282)
//   gfl6 / gfl13      FHERMA_GFLAGS=6 / 13 (BIG_GFLAGS; production 5)
//   pfq / pf3q / pfg  CRT prefetch items p1_grid/4, 3*p1_grid/4, p1_grid (production min(p1_grid/2, 2 cols))
//   npf               FHERMA_NO_L2PF=1: no P1 in-kernel next-block L2 prefetch (= sched p1pf0 = ladder fx_rs_npf without the re-split)
//   npf_q / npf_3q / npf_g   npf + CRT prefetch items p1_grid/4, 3*p1_grid/4, p1_grid
//   ldsm              CRT = crtd sw_crt_ldsm (SWCD FL 3: LDS.128 A/B fragment build; method control, known loser 1.006)
//   hql               P1 in the BPO_X_QL structure, production Shoup flags (control of h1m's module)
//   h1m               P1 BPO_X_QL + BPO_XG |= 1024|2048: Harvey-lazy quads 0-2 (primes < 2^30), Shoup quad 3 (p1o 072353b)
//   combo / combo2    every single-variant family whose best line scored <= FHERMA_COMBO_T (default 0.996), merged (combo2: best two)
//   *_r               repeats (top single lines by vs_prod, combo)
// FHERMA_COMBO=0 off, FHERMA_COMBO_S time budget (s, default 55, compiles included), FHERMA_BATCH_R pairs per line (default 13).
#include "crtd_sweep_src.h"
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
template<class C> inline void run_combo(C& d,const std::string& src,unsigned p1_grid,unsigned cols,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):55.0;
    const double thr=std::getenv("FHERMA_COMBO_T")?std::atof(std::getenv("FHERMA_COMBO_T")):0.996;
    // ---- sweep modules
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const std::string P1="bpo_first_x<0>",P3="bpo_last_cm3<0>";
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",{P1,"cb_cmp"});
    const int m_bvh=M.add("bvh","#define BPO_XG 4194304u\n"+s,{P1});
    const int m_p3el=M.add("p3el","#define BPO_P3EL 1\n"+s,{P3});
    const int m_ldsm=M.add("ldsm",s+"\n"+std::string(CRTD_SWEEP_SOURCE)+"\nSWCD(cb_crt_ldsm,3,3)\n",{"cb_crt_ldsm"});
    const int m_hql=M.add("hql","#define BPO_X_QL 1\n"+s,{P1},true);
    const int m_h1m=M.add("h1m","#define BPO_X_QL 1\n#define BPO_XG (1024u|2048u)\n"+s,{P1},true);
    const int m_h1mb=M.add("h1mb","#define BPO_X_QL 1\n#define BPO_XG (1024u|2048u|4194304u)\n"+s,{P1},true);
    M.build();
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"p1_grid\":%u,\"crt_pf_prod\":%u,\"thr\":%.4f,\"compile_s\":%.1f,\"mods\":{%s}}\n",p1_grid,std::min(p1_grid/2,2u*cols),thr,el(),info.c_str());std::fflush(stdout);}
    cmp=M.fn(m_rc,1);
    // ---- lines
    struct L{std::string v;int fam=0;CUfunction p1=nullptr,p3=nullptr,crt=nullptr;int gfl=-1;bool p3el=false,npf=false,bvh=false,h1m=false;int crtpf=-1;BatchRes r;};
    const unsigned gprod=[]{const char* e=std::getenv("FHERMA_GFLAGS");return e?unsigned(std::strtoul(e,nullptr,0)):5u;}();
    auto run=[&](L& x)->bool{
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return false;}
        BatchOvr o;o.check=true;o.fn[0]=x.p1;o.fn[2]=x.p3;o.fn[3]=x.crt;o.crtpf=x.crtpf;
        if(x.gfl>=0||x.p3el)o.env.push_back({"FHERMA_GFLAGS",std::to_string((x.gfl>=0?unsigned(x.gfl):gprod)|(x.p3el?128u:0u))});
        if(x.npf)o.env.push_back({"FHERMA_NO_L2PF","1"});
        x.r=line(o,x.v.c_str());return x.r.ok;};
    std::vector<L> done;
    auto single=[&](L x,bool need){if(need&&!x.p1&&!x.p3&&!x.crt){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",x.v.c_str());std::fflush(stdout);return;}if(run(x))done.push_back(x);};
    auto mk=[](const char* v,int fam){L x;x.v=v;x.fam=fam;return x;};
    // families: 1 P1 bvh, 2 P1 h1m, 3 P3 p3el, 4 GFLAGS, 5 prefetch (CRT items / P1 npf), 0 controls (never combined)
    single(mk("ctrl",0),false);
    {L x=mk("p1rc",0);x.p1=M.fn(m_rc);single(x,true);}
    {L x=mk("bvh",1);x.p1=M.fn(m_bvh);x.bvh=true;single(x,true);}
    {L x=mk("p3el",3);x.p3=M.fn(m_p3el);x.p3el=true;single(x,true);}
    for(int g:{6,13}){L x=mk(("gfl"+std::to_string(g)).c_str(),4);x.gfl=g;single(x,false);}
    struct PF{const char* v;unsigned items;bool npf;};
    const PF pfs[]={{"pfq",p1_grid/4,false},{"pf3q",3*p1_grid/4,false},{"pfg",p1_grid,false},{"npf",0,true},{"npf_q",p1_grid/4,true},{"npf_3q",3*p1_grid/4,true},{"npf_g",p1_grid,true}};
    for(const auto& p:pfs){L x=mk(p.v,5);x.crtpf=p.items?int(p.items):-1;x.npf=p.npf;single(x,false);}
    {L x=mk("ldsm",0);x.crt=M.fn(m_ldsm);single(x,true);}
    {L x=mk("hql",0);x.p1=M.fn(m_hql);single(x,true);}
    {L x=mk("h1m",2);x.p1=M.fn(m_h1m);x.h1m=true;single(x,true);}
    // repeats of the top single lines (exact, by vs_prod)
    std::vector<size_t> ord;for(size_t i=0;i<done.size();++i)if(done[i].fam>0&&done[i].r.ex==1)ord.push_back(i);
    std::sort(ord.begin(),ord.end(),[&](size_t a,size_t b){return done[a].r.vs<done[b].r.vs;});
    const size_t nsingle=done.size();
    for(size_t i=0;i<ord.size()&&i<3;++i){L x=done[ord[i]];x.v+="_r";if(run(x))done.push_back(x);}
    // per family: best exact line (single or its repeat: mean of the two when repeated) <= thr
    std::vector<double> score(done.size(),0);
    for(size_t i=0;i<nsingle;++i){double s1=done[i].r.vs;int c=1;for(size_t j=nsingle;j<done.size();++j)if(done[j].v==done[i].v+"_r"&&done[j].r.ex==1){s1+=done[j].r.vs;++c;}score[i]=s1/c;}
    std::vector<int> best(6,-1);
    for(size_t i=0;i<nsingle;++i){const L& x=done[i];if(x.fam<=0||x.r.ex!=1||score[i]>thr)continue;if(best[size_t(x.fam)]<0||score[i]<score[size_t(best[size_t(x.fam)])])best[size_t(x.fam)]=int(i);}
    std::vector<int> win;for(int f=1;f<=5;++f)if(best[size_t(f)]>=0)win.push_back(best[size_t(f)]);
    std::sort(win.begin(),win.end(),[&](int a,int b){return score[size_t(a)]<score[size_t(b)];});
    auto merge=[&](size_t cnt,const char* tag)->L{L c=mk(tag,-1);std::string nm=tag;
        for(size_t i=0;i<cnt&&i<win.size();++i){const L& x=done[size_t(win[i])];nm+=":"+x.v;
            c.bvh|=x.bvh;c.h1m|=x.h1m;if(x.p3){c.p3=x.p3;c.p3el=true;}if(x.gfl>=0)c.gfl=x.gfl;if(x.crtpf>=0)c.crtpf=x.crtpf;c.npf|=x.npf;}
        c.p1=c.bvh&&c.h1m?M.fn(m_h1mb):c.h1m?M.fn(m_h1m):c.bvh?M.fn(m_bvh):nullptr;c.v=nm;return c;};
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_pick\",\"winners\":%zu,\"list\":\"",win.size());for(int i:win)std::printf("%s=%.4f ",done[size_t(i)].v.c_str(),score[size_t(i)]);std::printf("\"}\n");std::fflush(stdout);
    if(win.size()>=2){L c=merge(win.size(),"combo");if(!(c.bvh&&c.h1m&&!c.p1)){if(run(c)){done.push_back(c);L r=c;r.v+="_r";if(run(r))done.push_back(r);}}}
    if(win.size()>=3){L c=merge(2,"combo2");if(run(c))done.push_back(c);}
    single(mk("ctrl_end",0),false);
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
