#pragma once
// Round 19 (combo2, on 9758254 = pv62: BIG_GFLAGS 1|4|128 (group-0 P3 evict_last), CRT prefetch items 3*p1_grid/4): exit-time
// WHOLE-BATCH lines (State::combo_sweep -> run_combo). Every line recaptures the full B-product production graph (variant 6, as
// fherma_run launches it) with one set of substitutions (State::batch_line: per-stage kernel from a sweep module, environment
// overrides read at capture time, CRT prefetch items) and times it against the production graph exec in paired, alternating
// reps (vs_prod = median of pair ratios; < 1 = faster). ex = 1 when all B outputs are bit-identical to production (device compare).
// Sweep modules: the production NVRTC source (prefix + kernels, same arch option) with every extern "C" kernel turned into an
// uninstantiated template (only the named instances compile); module "var" adds text substitutions (each must match) and a P3
// wrapper; compiled in worker threads at exit.
// Keys (v) and what each maps to as a production default:
//   ctrl / ctrl_end  production config recaptured (capture control)
//   rc               P2 bpo_middle_r16 + P3 bpo_last_cm3 + CRT bpo_crt_lean recompiled unchanged (module control)
//   gfl6             FHERMA_GFLAGS=134 (6|128): P1 TMA output stores L2 evict_last (pfstep bit 30) instead of plain (bit 29)
//                    -> BIG_GFLAGS = 2|4|128 = the "P1 output evict_last for P2" line
//   gfl13            FHERMA_GFLAGS=141 (13|128): adds bit 8 (non-last-group P3 residues evict_first); with 2 groups and bit 128
//                    the group-0 P3 takes the evict_last branch first, so this is the production config again (2nd control)
//   pf58 / pf78 / pfg  CRT prefetch items 5*p1_grid/8, 7*p1_grid/8, p1_grid (solve.cpp CRT launch; production 3*p1_grid/4)
//   p3all            P3 = cb_p3el_all (bpo_last_cm3 with T0|8): residue stores evict_last for EVERY prime group
//   crtef            CRT residue cp.async with L2::cache_hint evict_first (batched_crt_lean.cuh swc_cpa16); the CRT already
//                    discards the residue lines after use (dfl 1, l2f 8, production)
//   p3all_crtef      p3all + crtef
//   p2el             P2 C stores st.global.L2::cache_hint evict_last (bpo_r16_body output loop) for the P3 reads
//   combo / combo2   best line per family scoring <= FHERMA_COMBO_T (default 0.996) merged (combo2: best two); *_r repeats
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
template<class C> inline void run_combo(C& d,const std::string& src,unsigned p1_grid,unsigned cols,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):55.0;
    const double thr=std::getenv("FHERMA_COMBO_T")?std::atof(std::getenv("FHERMA_COMBO_T")):0.996;
    // ---- sweep modules
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const std::string P2="bpo_middle_r16<0>",P3="bpo_last_cm3<0>",CR="bpo_crt_lean<0>";
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",{P2,P3,CR,"cb_cmp"});
    // variant module: substitutions (every pattern must be found; count reported)
    std::string v=s;std::string subinfo;
    auto sub=[&](const char* tag,const std::string& a,const std::string& b)->bool{size_t p=0;int c=0;while((p=v.find(a,p))!=std::string::npos){v.replace(p,a.size(),b);p+=b.size();++c;}
        subinfo+=std::string(subinfo.empty()?"":",")+"\""+tag+"\":"+std::to_string(c);return c>0;};
    const bool ok_crt=sub("crtef","__device__ __forceinline__ void swc_cpa16(u32 dst,const void* src){asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\"::\"r\"(dst),\"l\"(src):\"memory\");}",
        "__device__ __forceinline__ void swc_cpa16(u32 dst,const void* src){u64 pol_;asm(\"createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\":\"=l\"(pol_));asm volatile(\"cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\"::\"r\"(dst),\"l\"(src),\"l\"(pol_):\"memory\");}");
    const bool ok_p2=sub("p2el","for(u32 c1=0;c1<16;++c1)o[4096u*c1]=a[c1];",
        "{u64 pol_;asm(\"createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\":\"=l\"(pol_));for(u32 c1=0;c1<16;++c1)asm volatile(\"st.global.L2::cache_hint.u32 [%0], %1, %2;\"::\"l\"(o+4096u*c1),\"r\"(a[c1]),\"l\"(pol_):\"memory\");}");
    v+="\nextern \"C\" __global__ void __launch_bounds__(256) cb_p3el_all(const u32* C,u32* D,const u32* primes,const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){\n"
       "    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;\n"
       "#if BPO_NH30\n    if(primes[blockIdx.y]<(1u<<30))bpo_p3_body<(BPO_P3T|8u),true>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);else\n#endif\n"
       "    bpo_p3_body<(BPO_P3T|8u)>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);\n}\n";
    const int m_var=M.add("var",v,{P2,CR,"cb_p3el_all"});
    M.build();
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"p1_grid\":%u,\"crt_pf_prod\":%u,\"thr\":%.4f,\"compile_s\":%.1f,\"subs\":{%s},\"mods\":{%s}}\n",p1_grid,std::min(3*p1_grid/4,2u*cols),thr,el(),subinfo.c_str(),info.c_str());std::fflush(stdout);}
    cmp=M.fn(m_rc,3);
    // ---- lines
    struct L{std::string v;int fam=0;CUfunction p2=nullptr,p3=nullptr,crt=nullptr;int gfl=-1;int crtpf=-1;BatchRes r;};
    auto run=[&](L& x)->bool{
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return false;}
        BatchOvr o;o.check=true;o.fn[1]=x.p2;o.fn[2]=x.p3;o.fn[3]=x.crt;o.crtpf=x.crtpf;
        if(x.gfl>=0)o.env.push_back({"FHERMA_GFLAGS",std::to_string(x.gfl)});
        x.r=line(o,x.v.c_str());return x.r.ok;};
    std::vector<L> done;
    auto single=[&](L x,bool need){if(need&&!x.p2&&!x.p3&&!x.crt){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",x.v.c_str());std::fflush(stdout);return;}if(run(x))done.push_back(x);};
    auto mk=[](const char* v,int fam){L x;x.v=v;x.fam=fam;return x;};
    // families: 1 GFLAGS (P1 store policy), 2 CRT prefetch items, 3 P3 residue policy / CRT residue loads, 4 P2 C stores; 0 controls
    single(mk("ctrl",0),false);
    {L x=mk("rc",0);x.p2=M.fn(m_rc,0);x.p3=M.fn(m_rc,1);x.crt=M.fn(m_rc,2);single(x,true);}
    {L x=mk("gfl6",1);x.gfl=6|128;single(x,false);}
    {L x=mk("gfl13",0);x.gfl=13|128;single(x,false);}   // = production (see header): control, never combined
    struct PF{const char* v;unsigned items;};
    const PF pfs[]={{"pf58",5*p1_grid/8},{"pf78",7*p1_grid/8},{"pfg",p1_grid}};
    for(const auto& p:pfs){L x=mk(p.v,2);x.crtpf=int(std::min(p.items,2u*cols));single(x,false);}
    {L x=mk("p3all",3);x.p3=M.fn(m_var,2);single(x,true);}
    if(ok_crt){L x=mk("crtef",3);x.crt=M.fn(m_var,1);single(x,true);}
    if(ok_crt){L x=mk("p3all_crtef",3);x.p3=M.fn(m_var,2);x.crt=M.fn(m_var,1);if(x.p3)single(x,true);}
    if(ok_p2){L x=mk("p2el",4);x.p2=M.fn(m_var,0);single(x,true);}
    // repeats of the top single lines (exact, by vs_prod)
    std::vector<size_t> ord;for(size_t i=0;i<done.size();++i)if(done[i].fam>0&&done[i].r.ex==1)ord.push_back(i);
    std::sort(ord.begin(),ord.end(),[&](size_t a,size_t b){return done[a].r.vs<done[b].r.vs;});
    const size_t nsingle=done.size();
    for(size_t i=0;i<ord.size()&&i<3;++i){L x=done[ord[i]];x.v+="_r";if(run(x))done.push_back(x);}
    // per family: best exact line (mean with its repeat when repeated) <= thr
    std::vector<double> score(done.size(),0);
    for(size_t i=0;i<nsingle;++i){double s1=done[i].r.vs;int c=1;for(size_t j=nsingle;j<done.size();++j)if(done[j].v==done[i].v+"_r"&&done[j].r.ex==1){s1+=done[j].r.vs;++c;}score[i]=s1/c;}
    std::vector<int> best(5,-1);
    for(size_t i=0;i<nsingle;++i){const L& x=done[i];if(x.fam<=0||x.r.ex!=1||score[i]>thr)continue;if(best[size_t(x.fam)]<0||score[i]<score[size_t(best[size_t(x.fam)])])best[size_t(x.fam)]=int(i);}
    std::vector<int> win;for(int f=1;f<=4;++f)if(best[size_t(f)]>=0)win.push_back(best[size_t(f)]);
    std::sort(win.begin(),win.end(),[&](int a,int b){return score[size_t(a)]<score[size_t(b)];});
    auto merge=[&](size_t cnt,const char* tag)->L{L c=mk(tag,-1);std::string nm=tag;
        for(size_t i=0;i<cnt&&i<win.size();++i){const L& x=done[size_t(win[i])];nm+=":"+x.v;
            if(x.p2)c.p2=x.p2;if(x.p3)c.p3=x.p3;if(x.crt)c.crt=x.crt;if(x.gfl>=0)c.gfl=x.gfl;if(x.crtpf>=0)c.crtpf=x.crtpf;}
        c.v=nm;return c;};
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_pick\",\"winners\":%zu,\"list\":\"",win.size());for(int i:win)std::printf("%s=%.4f ",done[size_t(i)].v.c_str(),score[size_t(i)]);std::printf("\"}\n");std::fflush(stdout);
    if(win.size()>=2){L c=merge(win.size(),"combo");if(run(c)){done.push_back(c);L r=c;r.v+="_r";if(run(r))done.push_back(r);}}
    if(win.size()>=3){L c=merge(2,"combo2");if(run(c))done.push_back(c);}
    single(mk("ctrl_end",0),false);
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
