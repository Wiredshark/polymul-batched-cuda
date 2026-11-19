#pragma once
// Exit-time on-target variant sweep (after the diag replays): compiles kernel variants of P1 (bpo_first_x), P2
// (bpo_middle_r16), P3 (bpo_last_cm3) and the CRT (bpo_crt_tc_cm) with NVRTC from the production source (same prefix, same
// arch option), times each on the captured launch geometry / data against the production kernel of the same run
// (interleaved A/B reps), checks exactness against the production output, then times whole product chains (products 0
// and 1 on two streams, P1 g0 .. CRT) for the best combinations. One {"diag":"sw",...} JSON line per variant.
// Source transform: every extern "C" __global__ of the production source becomes an uninstantiated template (not
// compiled), the P2 / CRT kernels become templates over the launch-bounds min-blocks (name expressions), and the P1 / P3
// variants are extern "C" wrappers over the production device templates (bpo_first_xb<F>, bpo_p3_body<T>).
// Compiles run in worker threads started before the diag replays. FHERMA_SWEEP=0 disables; FHERMA_SWEEP_S = time budget.
// Reading a line {"diag":"sw","k":stage,"v":variant,...}: med_us/min_us = 2 chains (products 0 and 1 concurrently, as in
// production; c1_us = product 0 alone), cold (2xL2 flush), P2/P3/CRT right after their predecessor kernel (untimed);
// vs_prod / c1_vs = median over adjacent pairs of variant/production time (< 1 = faster; the "prod" line is production
// against itself = noise floor; "ctrl" = recompiled production config = recompile control); ex 1 = output identical to
// production, 0 = differs (abl = ablation, wrong values by design); r = registers, lm = local (spill) bytes per thread.
// "chain" lines: products 0+1 full chains with the named substitutions (best exact variants per stage by the geometric
// mean of vs_prod and c1_vs), vs_prod against the production chain, ex = CRT output of product 0 identical.
// Round 13 (ladder, on 565ec66): P1 only. {"diag":"lad",...} lines = the P1 morph ladder (L0..L5 cumulative, R_* single removals,
// n_* compute-only fix twins; see ladder()), fx_* = exact fix prototypes with chain lines. Traces come from kprof trace_only
// mode (kprof off by default: launch parameters copied at graph capture, no event nodes).
#include <thread>
#include <atomic>
#include <memory>
#include <cmath>
#include <sstream>

inline bool sweep_env(){const char* e=std::getenv("FHERMA_SWEEP");return !e||*e!='0';}   // round 11 (p1o): sweep on by default again (P1 Harvey / attribution round); FHERMA_SWEEP=0 off

template<class C> struct Sweep {
    C& d; KProf& kp; const unsigned L;
    std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now();double budget=40.0;   // exit-time sweep budget (s), compiles included
    struct Prog { std::string name,arch,src; std::vector<std::string> kn; std::vector<bool> expr; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t_rtc=0,t_jit=0; bool done=false; };
    std::vector<Prog> progs; std::vector<std::thread> th; std::string arch,native;
    struct PB { std::vector<unsigned char> b; std::vector<size_t> o; template<class T> void add(T x){const size_t off=(b.size()+sizeof(T)-1)/sizeof(T)*sizeof(T);b.resize(off+sizeof(T));std::memcpy(b.data()+off,&x,sizeof(T));o.push_back(off);} };
    // mk: kernel with its own parameter list, built from the production trace it replaces (fused variants; no per-stage exactness)
    struct V { std::string v; int prog=-1,ki=-1; unsigned bx=0; int pidx=-1; unsigned pxor=0,por=0; bool abl=false; CUfunction fn=nullptr; double med=0,mn=0,c1=0,c1vs=0,vs=0; int ex=-1; bool ok=false; std::function<bool(const KProf::Trace&,PB&)> mk; bool skip=false; bool wa=false; bool lad=false; bool pers=false; unsigned gx=0; int p1m=0; bool p1e=false; int gsel=-1; };   // gsel >= 0: chains substitute only the stage trace with that group index   // p1m 1: m16 chain mode, 2: m16a (see run())   // wa: P1 bpt (param 6) = shifted table
    const std::vector<KProf::Trace>* trp=nullptr;
    std::vector<V> vs[4];   // per stage (KProf::P1..CRT)
    CUevent ea=nullptr,eb=nullptr,e0=nullptr,e1=nullptr,ej=nullptr;CUdeviceptr ref=0,bad=0,wabpt=0;size_t ref_n=0;CUfunction cmp=nullptr;
    std::vector<unsigned> mprimes,mpos;   // BCRT_PRIMES / BCRT_POS parsed from the source prefix (XG_WA table)
    static std::vector<unsigned> mlist(const std::string& s,const char* name){std::vector<unsigned> v;const std::string k=std::string("#define ")+name+" ";const size_t p=s.find(k);if(p==std::string::npos)return v;
        const size_t e=s.find('\n',p);std::string x=s.substr(p+k.size(),e-p-k.size());for(char& ch:x)if(ch==','||ch=='u')ch=' ';std::istringstream is(x);unsigned long long y;while(is>>y)v.push_back(unsigned(y));return v;}
    decltype(&::cuMemcpyDtoDAsync) dtod=nullptr; decltype(&::cuMemsetD32Async) memset32=nullptr;
    double el()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();}
    Sweep(C& c,const std::string& source,unsigned l):d(c),kp(c.kp),L(l){
        if(const char* e=std::getenv("FHERMA_SWEEP_S"))budget=std::atof(e);
        mprimes=mlist(source,"BCRT_PRIMES");mpos=mlist(source,"BCRT_POS");
        int cc1=0,cc2=0;if(kp.dattr){kp.dattr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,0);kp.dattr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,0);}
        if(const char* e=std::getenv("FHERMA_ARCH"))arch=e;else arch=cc1>=9?"compute_90":"compute_80";   // = production module option
        native="sm_"+std::to_string(cc1*10+cc2);
        std::string s=source;
        // round 13 (ladder, on 565ec66 = pv54): P1 morph ladder (batched_p1x.cuh XL_* bits; all ablations, wrong values by design).
        // Rungs L0..L5 remove one more difference between the compute-only production P1 (nomem) and the isolated quad-math probe
        // (diag_p1q.h pq_ab) at each step; R_* lines remove ONE difference from L0 (reverse ladder). Each ladder line is timed in the
        // production geometry (pg: 512 blocks, c2 / c1) and in steady state (one full wave of 2 blocks/SM, 4 and 4K quads per block,
        // differenced: no prologue, no tail): see run().
        {const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
        s+="\n#define SWP1(NAME,F,G,MB) extern \"C\" __global__ void __launch_bounds__(256,MB) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){__shared__ __align__(128) u32 X[BPO_X_SMEM(F)/4];bpo_first_xb<(F),(G)>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);}\n";
        const std::string base=s;
        auto prog=[&](const char* name,const std::string& a)->int{Prog P;P.name=name;P.arch=a;P.src=base;progs.push_back(std::move(P));return int(progs.size())-1;};
        auto addk=[&](int pi,const std::string& kn,bool ex)->int{Prog& P=progs[size_t(pi)];P.kn.push_back(kn);P.expr.push_back(ex);return int(P.kn.size())-1;};
        auto addv=[&](int st,const std::string& name,int pi,int ki,int pidx=-1,unsigned pxor=0,unsigned por=0,bool abl=false,unsigned bx=0){V v;v.v=name;v.prog=pi;v.ki=ki;v.pidx=pidx;v.pxor=pxor;v.por=por;v.abl=abl;v.bx=bx;vs[st].push_back(v);};
        // programs in priority order (compiles that would start after the compile deadline are skipped)
        {const int c=prog("cmp",arch);progs[size_t(c)].src="typedef unsigned int u32;typedef unsigned long long u64;\n"   // exactness compare kernel (tiny)
            "extern \"C\" __global__ void sw_cmp(const uint4* a,const uint4* b,u64 n,u32* bad){u32 c=0;for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=u64(gridDim.x)*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n";
         addk(c,"sw_cmp",false);}
        {   // P1: ctrl (recompiled production), nost, nomem
            const int c=prog("p1",arch);auto& S=progs[size_t(c)].src;
            S+="SWP1(sw_p1_ctrl,BPO_XF,BPO_XG,BPO_TC_MINB)\nSWP1(sw_p1_nost,(BPO_XF|(8u<<20)),BPO_XG,BPO_TC_MINB)\nSWP1(sw_p1_nomem,(BPO_XF|(8u<<20)),(BPO_XG|1u),BPO_TC_MINB)\n";
            addv(KProf::P1,"ctrl",c,addk(c,"sw_p1_ctrl",false));addv(KProf::P1,"nost",c,addk(c,"sw_p1_nost",false),-1,0,0,true);addv(KProf::P1,"nomem",c,addk(c,"sw_p1_nomem",false),-1,0,0,true);}
        {   // ladder: F = nomem flags (XABL 8) or XABL 14 (no proxy fence, with NPF); G = nomem (XG_NST) | XL bits
            struct A1{const char* n;unsigned xa,g;};
            enum:unsigned{SS=16384u,NPF=32768u,HOT=65536u,NBC=131072u,NAF=262144u,NPRO=524288u,NBV=1048576u,NMMA=4u,NRED=8u,NOBAR=8388608u,TWJ=2097152u,BVH=4194304u};
            static const A1 a1[]={{"L0_ss",8,SS},{"L1_npf",14,SS|NPF},{"L2_hot",14,SS|NPF|HOT},{"L3_nbc",14,SS|NPF|HOT|NBC},{"L4_naf",14,SS|NPF|HOT|NBC|NAF},{"L5_npro",14,SS|NPF|HOT|NBC|NAF|NPRO},
                {"R_hot",8,SS|HOT},{"R_nbc",8,SS|NBC},{"R_nbcaf",8,SS|NBC|NAF},{"R_npro",8,SS|NPRO},{"R_nbv",8,SS|NBV},{"R_nmma",8,SS|NMMA},{"R_nred",8,SS|NRED},{"R_nobar",8,SS|NOBAR},
                {"n_twj",8,SS|TWJ},{"n_bvh",8,SS|BVH}};   // compute-only twins of the fix prototypes (ss comparable to L0)
            int c=-1,nk=0;
            for(const auto& x:a1){if(nk%3==0){c=prog(("lad"+std::to_string(nk/3)).c_str(),arch);}++nk;   // 3 kernels per program (parallel compiles)
                char ln[256];std::snprintf(ln,sizeof ln,"SWP1(sw_p1_%s,(BPO_XF|(%uu<<20)),(BPO_XG|1u|%uu),BPO_TC_MINB)\n",x.n,x.xa,x.g);progs[size_t(c)].src+=ln;
                addv(KProf::P1,x.n,c,addk(c,std::string("sw_p1_")+x.n,false),-1,0,0,true);vs[KProf::P1].back().lad=true;}
            // fix prototypes (exact, production memory path): pg A/B vs production, exactness, then full chains (p1=fx_*, cf: twins)
            const int f=prog("fix",arch);auto& S=progs[size_t(f)].src;
            S+="SWP1(sw_p1_fx_twj,BPO_XF,(BPO_XG|2097152u),BPO_TC_MINB)"+std::string(1,char(10))+"SWP1(sw_p1_fx_bvh,BPO_XF,(BPO_XG|4194304u),BPO_TC_MINB)"+std::string(1,char(10))+"SWP1(sw_p1_fx_twj_bvh,BPO_XF,(BPO_XG|6291456u),BPO_TC_MINB)"+std::string(1,char(10));
            for(const char* n:{"fx_twj","fx_bvh","fx_twj_bvh"})addv(KProf::P1,n,f,addk(f,std::string("sw_p1_")+n,false));
            // round 14 (geo): static tail re-split P1 (bpo_first_xs; S = blocks/SM x SMs in pfstep bits 16-25, grid RS + rem f):
            // fx_rs (prefetch of the block's tail column-op), fx_rs_npf (no in-kernel prefetch), n_rs = compute-only twin
            // (compare with nomem); whole-batch lines at the end
            const int pg=prog("pers",arch);auto& P=progs[size_t(pg)].src;const std::string nl(1,char(10));
            P+="#define SWPP(NAME,F,G,PFN) extern \"C\" __global__ void __launch_bounds__(256,BPO_TC_MINB) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){__shared__ __align__(128) u32 X[BPO_X_SMEM(F)/4];bpo_first_xs<(F),(G),(PFN)>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);}"+nl;
            P+="SWPP(sw_p1_fx_rs,BPO_XF,BPO_XG,1)"+nl+"SWPP(sw_p1_fx_rs_npf,BPO_XF,BPO_XG,0)"+nl+"SWPP(sw_p1_n_rs,(BPO_XF|(8u<<20)),(BPO_XG|1u),1)"+nl;
            for(const char* n:{"fx_rs","fx_rs_npf","n_rs"}){const bool ab_=n[0]=='n';addv(KProf::P1,n,pg,addk(pg,std::string("sw_p1_")+n,false),-1,0,0,ab_);vs[KProf::P1].back().pers=true;}}
        for(auto& P:progs)P.fn.assign(P.kn.size(),nullptr);
        // worker threads: NVRTC compile + module load (driver JIT for PTX), in program order, until the compile deadline
        const unsigned hw=std::max(2u,std::thread::hardware_concurrency());const size_t nth=std::min<size_t>(progs.size(),std::max(1u,std::min(8u,hw-1)));
        auto next=std::make_shared<std::atomic<size_t>>(0);const double cdl=budget*0.6;
        for(size_t t=0;t<nth;++t)th.emplace_back([this,next,cdl]{for(size_t k;(k=next->fetch_add(1))<progs.size();){if(el()>cdl){progs[k].err="compile deadline";progs[k].done=true;continue;}compile(progs[k]);}});
    }
    void compile(Prog& P){
        auto tA=std::chrono::steady_clock::now();
        try{
            auto addn=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*)>("nvrtcAddNameExpression");
            auto lown=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*,const char**)>("nvrtcGetLoweredName");
            auto gsz=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,size_t*)>("nvrtcGetCUBINSize");auto gcub=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,char*)>("nvrtcGetCUBIN");
            auto setctx=d.driver.template get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");
            nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,P.src.c_str(),"sweep.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("create");
            for(size_t k=0;k<P.kn.size();++k)if(P.expr[k])addn(pr,P.kn[k].c_str());
            const std::string a="--gpu-architecture="+P.arch;const char* op[]={a.c_str(),"--std=c++17","-default-device"};const int nop=P.src.rfind("#define BPO_X_QL",0)==0?3:2;   // p1h: generic-lambda quad body needs -default-device (as h30s)
            if(d.nvrtcCompileProgram(pr,nop,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
                const size_t e=l.find("error");throw std::runtime_error("nvrtc: "+l.substr(e==std::string::npos?0:e,160));}
            std::vector<char> img;const bool cub=P.arch.rfind("sm_",0)==0;
            if(cub){size_t n=0;if(gsz(pr,&n)!=NVRTC_SUCCESS||!n){d.nvrtcDestroyProgram(&pr);throw std::runtime_error("no cubin");}img.resize(n);gcub(pr,img.data());}
            else{size_t n=0;d.nvrtcGetPTXSize(pr,&n);img.resize(n);d.nvrtcGetPTX(pr,img.data());}
            std::vector<std::string> names(P.kn.size());
            for(size_t k=0;k<P.kn.size();++k){if(!P.expr[k]){names[k]=P.kn[k];continue;}const char* ln=nullptr;if(lown(pr,P.kn[k].c_str(),&ln)==NVRTC_SUCCESS&&ln)names[k]=ln;}
            d.nvrtcDestroyProgram(&pr);
            auto tB=std::chrono::steady_clock::now();P.t_rtc=std::chrono::duration<double>(tB-tA).count();
            setctx(d.context);
            if(d.cuModuleLoadData(&P.mod,img.data())!=CUDA_SUCCESS){P.mod=nullptr;throw std::runtime_error("module load");}
            for(size_t k=0;k<names.size();++k){CUfunction f=nullptr;if(!names[k].empty()&&d.cuModuleGetFunction(&f,P.mod,names[k].c_str())==CUDA_SUCCESS)P.fn[k]=f;}
            P.t_jit=std::chrono::duration<double>(std::chrono::steady_clock::now()-tB).count();
        }catch(const std::exception& e){P.err=e.what();for(char& ch:P.err)if(ch=='"'||ch=='\\'||ch=='\n'||ch=='\r')ch=' ';}
        P.done=true;
    }
    void join(){for(auto& t:th)if(t.joinable())t.join();th.clear();}
    ~Sweep(){join();d.cuCtxSynchronize();for(auto& P:progs)if(P.mod)d.cuModuleUnload(P.mod);for(auto e:{ea,eb,e0,e1,ej})if(e)kp.edestroy(e);for(auto p:{ref,bad,wabpt})if(p)d.cuMemFree(p);}
    // XG_WA B table: the production bpt ([j][lane][tile*2+reg], bytes of 2^(8m) fa mod p) with each digit factor fa times
    // 2^-(POS[j]%32); fa is read back from the table (m = 0 bytes) and the whole production table re-derived as a check.
    bool make_wabpt(CUdeviceptr src){
        constexpr unsigned J=8;if(mprimes.size()!=16||mpos.size()!=J)return false;
        std::vector<unsigned> t(J*32*16),o(J*32*16,0);if(d.cuMemcpyDtoH(t.data(),src,t.size()*4)!=CUDA_SUCCESS)return false;
        auto pw=[](unsigned long long b,unsigned long long e,unsigned long long m){unsigned long long r=1;b%=m;while(e){if(e&1)r=r*b%m;b=b*b%m;e>>=1;}return r;};
        auto fill=[&](std::vector<unsigned>& dst,unsigned j,unsigned r,unsigned fa){const unsigned long long prime=mprimes[r];unsigned wt[32];unsigned long long q=1;
            for(unsigned m=0;m<32;++m){wt[m]=unsigned(q*fa%prime);q=q*256%prime;}
            for(unsigned ln=0;ln<32;++ln){const unsigned g=ln>>2,tg=ln&3;
                for(unsigned T=0;T<8;++T){const unsigned u=T/2,half=T%2,rr=4*u+g/2,bb=2*half+g%2;if(rr!=r)continue;
                    for(unsigned h=0;h<2;++h){unsigned v=0;for(unsigned e=0;e<4;++e)v|=((wt[16*h+4*tg+e]>>(8*bb))&255u)<<(8*e);dst[(size_t(j)*32+ln)*16+T*2+h]=v;}}}};
        std::vector<unsigned> chk(t.size(),0);
        for(unsigned j=0;j<J;++j)for(unsigned r=0;r<16;++r){unsigned fa=0;
            for(unsigned bb=0;bb<4;++bb){const unsigned g=2*(r%4)+(bb&1),ln=4*g,T=2*(r/4)+(bb>>1);fa|=(t[(size_t(j)*32+ln)*16+T*2]&255u)<<(8*bb);}
            fill(chk,j,r,fa);const unsigned long long p=mprimes[r];
            fill(o,j,r,unsigned(fa*pw((p+1)/2,mpos[j]%32,p)%p));}
        if(chk!=t){std::printf("{\"diag\":\"sw\",\"k\":\"p1\",\"error\":\"wa table re-derivation mismatch\"}\n");return false;}
        if(d.cuMemAlloc(&wabpt,o.size()*4)!=CUDA_SUCCESS){wabpt=0;return false;}d.check(d.cuMemcpyHtoD(wabpt,o.data(),o.size()*4));return true;}

    // ---- measurement -------------------------------------------------------------------------------------------
    template<class D> void launch(D& g,const KProf::Trace& t,CUstream st,const V* v){
        if(v&&v->mk){PB q;if(!v->mk(t,q))throw std::runtime_error("no params");void* pp[64];for(size_t i=0;i<q.o.size();++i)pp[i]=q.b.data()+q.o[i];
            d.check(d.cuLaunchKernel(v->fn,t.gx,t.gy,t.gz,v->bx?v->bx:t.bx,1,1,t.smem,st,pp,nullptr));return;}
        std::vector<unsigned char> b=t.pb;void* pp[64];
        if(v&&v->pidx>=0&&size_t(v->pidx)<t.po.size()){unsigned x=0;std::memcpy(&x,b.data()+t.po[size_t(v->pidx)],4);x=(x^v->pxor)|v->por;std::memcpy(b.data()+t.po[size_t(v->pidx)],&x,4);}
        if(v&&v->wa){if(!wabpt||t.po.size()<7)throw std::runtime_error("no wa table");std::memcpy(b.data()+t.po[6],&wabpt,sizeof(wabpt));}
        for(size_t i=0;i<t.po.size();++i)pp[i]=b.data()+t.po[i];
        d.check(d.cuLaunchKernel(v&&v->fn?v->fn:t.fn,v&&v->gx?v->gx:t.gx,t.gy,t.gz,v&&v->bx?v->bx:t.bx,1,1,t.smem,st,pp,nullptr));(void)g;}
    // one timed sample: flush + predecessors (untimed), then k0 on s0 (and k1 on s1), ms between ev0 and the join
    template<class D> float sample(D& g,const KProf::Trace* k0,const KProf::Trace* k1,const KProf::Trace* q0,const KProf::Trace* q1,const V* v){
        CUstream a=g.s[0],b=g.s[1];g.do_flush(a);if(q0)g.replay(*q0,a);
        if(k1){kp.erecord(ea,a);g.swait(b,ea,0);if(q1)g.replay(*q1,b);kp.erecord(eb,b);g.swait(a,eb,0);}
        kp.erecord(e0,a);launch(g,*k0,a,v);
        if(k1){g.swait(b,e0,0);launch(g,*k1,b,v);kp.erecord(ej,b);g.swait(a,ej,0);}
        kp.erecord(e1,a);d.check(g.esync(e1));return kp.ms(e0,e1);}
    static double med(std::vector<float> x){if(x.empty())return 0;std::sort(x.begin(),x.end());return x[x.size()/2];}
    // interleaved A/B: production (v=nullptr) vs variant, c2 (k1 given) and c1
    template<class D> void ab(D& g,V& v,const KProf::Trace* k0,const KProf::Trace* k1,const KProf::Trace* q0,const KProf::Trace* q1,int R){
        std::vector<float> pa,va,pc,vc;std::vector<float> ra,rc;   // per-pair ratios (variant / adjacent production sample)
        for(int r=0;r<R+2;++r){
            const bool vf=r&1;float x,y;
            if(k1){x=sample(g,k0,k1,q0,q1,vf?&v:nullptr);y=sample(g,k0,k1,q0,q1,vf?nullptr:&v);if(r>=2){(vf?va:pa).push_back(x);(vf?pa:va).push_back(y);if(x>0&&y>0)ra.push_back(vf?x/y:y/x);}}
            x=sample(g,k0,nullptr,q0,nullptr,vf?&v:nullptr);y=sample(g,k0,nullptr,q0,nullptr,vf?nullptr:&v);if(r>=2){(vf?vc:pc).push_back(x);(vf?pc:vc).push_back(y);if(x>0&&y>0)rc.push_back(vf?x/y:y/x);}}
        auto& A=k1?va:vc;(void)pa;(void)pc;
        v.med=med(A)*1e3;v.mn=A.empty()?0:*std::min_element(A.begin(),A.end())*1e3;v.vs=med(k1?ra:rc);
        v.c1=med(vc)*1e3;v.c1vs=med(rc);}
    struct Reg{CUdeviceptr p;size_t bytes;};
    // exactness: prelude (product-0 traces before it) + production -> ref, prelude, poison the output, variant, compare (count of differing 16 B words)
    template<class D> int exact(D& g,const KProf::Trace& t,const std::function<void(CUstream)>& q,const std::vector<Reg>& rg,const V& v){
        if(!cmp||rg.empty())return -1;size_t tot=0;for(auto& r:rg)tot+=r.bytes;
        if(tot>ref_n){if(ref)d.cuMemFree(ref);ref=0;d.check(d.cuMemAlloc(&ref,tot));ref_n=tot;}
        CUstream a=g.s[0];q(a);launch(g,t,a,nullptr);size_t o=0;for(auto& r:rg){d.check(dtod(ref+o,r.p,r.bytes,a));o+=r.bytes;}
        q(a);for(auto& r:rg)d.check(memset32(r.p,0x5a5a5a5au,r.bytes/4,a));   // poison after the predecessor (P3 writes P2's input)
        launch(g,t,a,&v);unsigned z=0;d.check(d.cuMemcpyHtoD(bad,&z,4));o=0;
        for(auto& r:rg){CUdeviceptr x=r.p,y=ref+o;unsigned long long n=r.bytes/16;void* pp[]={&x,&y,&n,&bad};d.check(d.cuLaunchKernel(cmp,unsigned(g.sms)*4,1,1,256,1,1,0,a,pp,nullptr));o+=r.bytes;}
        d.check(d.cuCtxSynchronize());d.check(d.cuMemcpyDtoH(&z,bad,4));return z?0:1;}
    template<class P> static P param(const KProf::Trace& t,size_t i){P x{};if(i<t.po.size())std::memcpy(&x,t.pb.data()+t.po[i],sizeof(P));return x;}
    std::vector<Reg> regions(int st,const KProf::Trace& t)const{
        std::vector<Reg> r;
        if(st==KProf::P1){const unsigned N=param<unsigned>(t,7),r0=param<unsigned>(t,9),r1=param<unsigned>(t,10);const size_t off=size_t(r0)*8*N*4,sz=size_t(r1-r0)*8*N*4;
            r.push_back({param<CUdeviceptr>(t,2)+off,sz});r.push_back({param<CUdeviceptr>(t,3)+off,sz});}
        else if(st==KProf::P2)r.push_back({param<CUdeviceptr>(t,2),size_t(t.gz)*t.gy*param<unsigned>(t,5)*4});
        else if(st==KProf::P3)r.push_back({param<CUdeviceptr>(t,1),size_t(t.gy)*8*param<unsigned>(t,6)*4});
        else if(st==KProf::CRT)r.push_back({param<CUdeviceptr>(t,1),size_t(param<unsigned>(t,3))*L*4});
        return r;}
    void print(const char* k,const V& v){
        int regs=0,lm=0,bp=0;CUfunction f=v.fn;
        if(f&&kp.fattr){kp.fattr(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f);kp.fattr(&lm,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);}
        std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"med_us\":%.2f,\"min_us\":%.2f,\"vs_prod\":%.4f,\"c1_us\":%.2f,\"c1_vs\":%.4f",k,v.v.c_str(),v.med,v.mn,v.vs,v.c1,v.c1vs);
        if(v.ex>=0)std::printf(",\"ex\":%d",v.ex);if(v.abl)std::printf(",\"abl\":1");if(regs)std::printf(",\"r\":%d",regs);if(lm)std::printf(",\"lm\":%d",lm);(void)bp;
        std::printf("}\n");std::fflush(stdout);}
    // round 13 (ladder): steady-state / one-wave timing of a P1 variant on the production trace k (r0 = 0): launches of one full
    // wave (blocks/SM from the occupancy API x SMs) with r1 = 16 (4 quads per block: w1, prologue + epilogue included) and, for
    // XL_SS kernels, r1 = 16 KA / 16 KB (wA, wB). ss_us_pp = (wB - wA) / (KB - KA) scaled to the 2048 quads of one product: the
    // steady quad rate (no prologue, no wave tail; comparable to diag p1q us_per_product). pg_us_pp = production-geometry c2 / 2
    // (sweep med_us / 2); geo_us_pp = pg_us_pp - ss_us_pp (wave tail + prologue/epilogue + launch, at 2 concurrent products).
    template<class D> void ladder(D& g,const V& v,const KProf::Trace& k,double nomem_med){
        if(!v.fn||k.po.size()<12||param<unsigned>(k,9)!=0)return;
        int bps=0;if(kp.occ)kp.occ(&bps,v.fn,int(k.bx),size_t(k.smem));if(bps<=0)return;
        const unsigned gw=unsigned(bps)*unsigned(g.sms),r1=param<unsigned>(k,10);constexpr unsigned KA=4,KB=20;
        auto wave=[&](unsigned kk)->float{std::vector<unsigned char> b=k.pb;void* pp[64];const unsigned rr=r1*kk;std::memcpy(b.data()+k.po[10],&rr,4);
            for(size_t i=0;i<k.po.size();++i)pp[i]=b.data()+k.po[i];CUstream a=g.s[0];
            kp.erecord(e0,a);d.check(d.cuLaunchKernel(v.fn,gw,1,1,k.bx,1,1,k.smem,a,pp,nullptr));kp.erecord(e1,a);d.check(g.esync(e1));return kp.ms(e0,e1);};
        std::vector<float> a1,aA,aB;const int RL=11;
        for(int r=0;r<RL+2;++r){const float x=wave(1);const float y=v.lad?wave(KA):0.f;const float z=v.lad?wave(KB):0.f;if(r>=2){a1.push_back(x);aA.push_back(y);aB.push_back(z);}}
        const double q=4.0*gw,w1=med(a1)*1e3,wA=med(aA)*1e3,wB=med(aB)*1e3;
        const double ss=v.lad?(wB-wA)/double(KB-KA)*2048.0/q:0;
        int regs=0,lm=0;if(kp.fattr){kp.fattr(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,v.fn);kp.fattr(&lm,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,v.fn);}
        std::printf("{\"diag\":\"lad\",\"v\":\"%s\",\"pg_us_pp\":%.2f,\"ss_us_pp\":%.2f,\"geo_us_pp\":%.2f,\"pg_c1_us\":%.2f,\"pg_vs_nomem\":%.4f,\"w1_us\":%.2f,\"wA_us\":%.2f,\"wB_us\":%.2f,\"KA\":%u,\"KB\":%u,\"bps\":%d,\"grid_w\":%u,\"r\":%d,\"lm\":%d}\n",
            v.v.c_str(),v.med/2,ss,v.lad?v.med/2-ss:0.0,v.c1,nomem_med>0?v.med/nomem_med:0.0,w1,wA,wB,KA,KB,bps,gw,regs,lm);std::fflush(stdout);}
    std::function<void(CUfunction,unsigned,unsigned,const char*)> batch;   // round 14: whole-batch A/B of a P1 substitute (State::batch_line)
    template<class D> void run(D& g){
        join();
        {std::printf("{\"diag\":\"sw_info\",\"arch\":\"%s\",\"hw_threads\":%u,\"compile_wall_s\":%.1f,\"progs\":[",arch.c_str(),std::thread::hardware_concurrency(),el());
         for(size_t i=0;i<progs.size();++i)std::printf("%s[\"%s\",%zu,%.1f,%.1f%s%s%s]",i?",":"",progs[i].name.c_str(),progs[i].kn.size(),progs[i].t_rtc,progs[i].t_jit,
             progs[i].err.empty()?"":",\"",progs[i].err.empty()?"":progs[i].err.substr(0,120).c_str(),progs[i].err.empty()?"":"\"");
         std::printf("]}\n");std::fflush(stdout);}
        if(kp.diag_variant()<0)return;
        const auto& tr=kp.traces[kp.diag_variant()];
        dtod=d.driver.template get<decltype(dtod)>("cuMemcpyDtoDAsync_v2");memset32=d.driver.template get<decltype(memset32)>("cuMemsetD32Async");
        for(auto e:{&ea,&eb,&e0,&e1,&ej})kp.ecreate(e,CU_EVENT_DEFAULT);
        d.check(d.cuMemAlloc(&bad,16));
        for(auto& P:progs)if(P.name=="cmp"&&!P.fn.empty())cmp=P.fn[0];
        for(int st=0;st<4;++st)for(auto& v:vs[st]){if(v.prog<0){v.ok=true;continue;}const auto& P=progs[size_t(v.prog)];if(v.ki>=0&&size_t(v.ki)<P.fn.size()&&P.fn[size_t(v.ki)]){v.fn=P.fn[size_t(v.ki)];v.ok=true;}}
        // first trace of a stage for products 0 / 1 and the trace just before it in the same product (predecessor)
        auto first=[&](int prod,int st,size_t* pi)->const KProf::Trace*{for(size_t i=0;i<tr.size();++i)if(tr[i].product==prod&&tr[i].stage==st&&!tr[i].pb.empty()){if(pi)*pi=i;return &tr[i];}return nullptr;};
        auto pred=[&](int prod,size_t i)->const KProf::Trace*{for(size_t j=i;j-->0;)if(tr[j].product==prod&&!tr[j].pb.empty())return &tr[j];return nullptr;};
        CUfunction pf[4]={d.bpo_first_x,d.bpo_middle_r16,d.bpo_last_cm3,d.bpo_crt_lean};static const char* kn[4]={"p1","p2","p3","crt"};
        {const auto tw=std::chrono::steady_clock::now();   // valid product-0 data in every buffer + clock ramp after the idle compile wait (>= 0.3 s busy)
         for(int w=0;w<200&&(w<3||std::chrono::duration<double>(std::chrono::steady_clock::now()-tw).count()<0.3);++w){
             for(size_t i=0;i<tr.size();++i){const auto& t=tr[i];if(t.product==0&&!t.pb.empty())g.replay(t,g.s[0]);}d.check(d.cuCtxSynchronize());}}
        trp=&tr;
        {size_t i1=0;const KProf::Trace* t1=first(0,KProf::P1,&i1);if(t1&&t1->fn==pf[KProf::P1]&&!make_wabpt(param<CUdeviceptr>(*t1,6)))std::printf("{\"diag\":\"sw\",\"k\":\"p1\",\"error\":\"no wa table\"}\n");}
        const int RE=std::getenv("FHERMA_SWEEP_R")?std::atoi(std::getenv("FHERMA_SWEEP_R")):0;
        const KProf::Trace* k0s[4]={};const KProf::Trace* k1s[4]={};const KProf::Trace* q0s[4]={};const KProf::Trace* q1s[4]={};
        for(int st=0;st<4;++st){
            size_t i0=0,i1=0;const KProf::Trace* k0=first(0,st,&i0);const KProf::Trace* k1=first(1,st,&i1);
            if(!k0||k0->fn!=pf[st]){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"skipped\":\"prod kernel %s\"}\n",kn[st],k0?kp.name_of(k0->fn):"none");continue;}
            if(k1&&(k1->fn!=k0->fn||k1->chain==k0->chain))k1=nullptr;
            const KProf::Trace* q0=st==KProf::P1?nullptr:pred(0,i0);const KProf::Trace* q1=(st==KProf::P1||!k1)?nullptr:pred(1,i1);
            k0s[st]=k0;k1s[st]=k1;q0s[st]=q0;q1s[st]=q1;
            const auto rg=regions(st,*k0);auto prel=[&,i0](CUstream a){for(size_t j=0;j<i0;++j)if(tr[j].product==0&&!tr[j].pb.empty())g.replay(tr[j],a);};
            const int R=RE>0?RE:(st==KProf::P1?15:25);
            if(vs[st].empty())continue;   // round 13: P1 only (P2 / P3 / CRT traces kept for the chains)
            for(auto& v:vs[st])if(v.pers&&v.fn){int bps=0;if(kp.occ)kp.occ(&bps,v.fn,int(k0->bx),size_t(k0->smem));const unsigned Sl=bps>0?unsigned(bps)*unsigned(g.sms):0u;
                if(!Sl||Sl>=k0->gx||Sl>1023u||k0->gx!=512u){v.ok=false;std::printf("{\"diag\":\"sw\",\"k\":\"p1\",\"v\":\"%s\",\"skipped\":\"no tail (slots %u)\"}\n",v.v.c_str(),Sl);continue;}
                const unsigned R=512u/Sl,rem=512u-R*Sl,f=rem==0u?1u:(rem*4u<=Sl?4u:(rem*2u<=Sl?2u:1u));v.gx=R*Sl+rem*f;v.pidx=11;v.por=Sl<<16;
                std::printf("{\"diag\":\"sw\",\"k\":\"p1\",\"v\":\"%s\",\"slots\":%u,\"R\":%u,\"rem\":%u,\"f\":%u,\"grid\":%u}\n",v.v.c_str(),Sl,R,rem,f,v.gx);}
            V prod;prod.v="prod";prod.fn=nullptr;ab(g,prod,k0,k1,q0,q1,R);{V pv=prod;pv.fn=k0->fn;print(kn[st],pv);}
            double nomem_med=0;
            for(auto& v:vs[st]){
                if(el()>budget-8){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",kn[st],v.v.c_str());v.ok=false;continue;}
                if(!v.ok){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"skipped\":\"not compiled\"}\n",kn[st],v.v.c_str());continue;}
                try{ab(g,v,k0,k1,q0,q1,R);v.ex=v.mk?-1:v.abl?0:exact(g,*k0,prel,rg,v);if(v.v=="nomem")nomem_med=v.med;
                    if(st==KProf::P1&&(v.lad||v.v=="nomem"||v.v=="ctrl"))ladder(g,v,*k0,nomem_med);}catch(const std::exception& e){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"error\":\"%.60s\"}\n",kn[st],v.v.c_str(),e.what());v.ok=false;d.cuCtxSynchronize();continue;}
                print(kn[st],v);}
        }
        // product chains (products 0 and 1 on two streams from a cold flush) for the best exact combinations
        auto score=[](const V* v){return std::sqrt(v->vs*v->c1vs);};   // rank by the geometric mean of the c2 and c1 ratios
        auto best=[&](int st,size_t rank)->const V*{std::vector<const V*> c;for(auto& v:vs[st])if(v.ok&&v.ex==1&&!v.abl&&v.v!="ctrl"&&v.vs>0&&v.c1vs>0&&score(&v)<0.995)c.push_back(&v);
            std::sort(c.begin(),c.end(),[&](const V* a,const V* b){return score(a)<score(b);});return rank<c.size()?c[rank]:nullptr;};
        auto byname=[&](int st,const char* n)->const V*{for(auto& v:vs[st])if(v.ok&&v.v==n)return &v;return nullptr;};
        struct Combo{std::string name;const V* v[4];bool cf=false;};std::vector<Combo> cs;(void)best;
        auto addc=[&](const V* a,const V* b,const V* c,const V* e){Combo x{"",{a,b,c,e},false};bool any=false;for(int st=0;st<4;++st)if(x.v[st]){any=true;x.name+=(x.name.empty()?"":",")+std::string(kn[st])+"="+x.v[st]->v;}
            if(!any)x.name="prod";for(auto& y:cs)if(y.name==x.name)return;cs.push_back(x);};
        // ceiling chains (ablations, wrong by design: ex 0, abl 1): fused-pair upper bounds and per-kernel memory removal;
        // every line also as a cf: (clean read-only flush) twin.
        //   abl_p3crt_fused = P3 nost + CRT nold     abl_p1p2_fused = P1 nost + P2 nold     abl_p2p3_fused = P2 nost + P3 nold
        //   abl_allfused = P1 nost + P2 nomem + P3 nomem + CRT nold (no intermediate traffic at all; input + output stay)
        //   nomem:<k> = that kernel without global data loads/stores, the rest production; nomem:all = every kernel nomem
        auto named=[&](const std::string& n,const V* p1,const V* p2,const V* p3,const V* crt){Combo x{n,{p1,p2,p3,crt},false};for(auto& y:cs)if(y.name==x.name)return;cs.push_back(x);};
        auto nm=[&](int st,const char* n)->const V*{const V* v=byname(st,n);return v&&v->ok?v:nullptr;};
        const V *p1s=nm(KProf::P1,"nost"),*p1m=nm(KProf::P1,"nomem"),*p2l=nm(KProf::P2,"nold"),*p2s=nm(KProf::P2,"nost"),*p2m=nm(KProf::P2,"nomem"),
                *p3l=nm(KProf::P3,"nold"),*p3s=nm(KProf::P3,"nost"),*p3m=nm(KProf::P3,"nomem"),*crl=nm(KProf::CRT,"nold"),*crm=nm(KProf::CRT,"nomem");
        named("prod",nullptr,nullptr,nullptr,nullptr);
        // round 13 (ladder): one chain per exact P1 fix variant (ok, ex 1, not an ablation, not ctrl), with cf: twins
        for(const auto& v:vs[KProf::P1])if(v.ok&&v.ex==1&&!v.abl&&v.v!="ctrl")named(std::string("p1=")+v.v,&v,nullptr,nullptr,nullptr);
        (void)p1s;(void)p1m;(void)p2l;(void)p2s;(void)p2m;(void)p3l;(void)p3s;(void)p3m;(void)crl;(void)crm;
        {const size_t n0=cs.size();for(size_t i=0;i<n0;++i){Combo x=cs[i];x.name="cf:"+x.name;x.cf=true;cs.push_back(x);}}
        if(cs.size()>24)cs.resize(24);
        unsigned p1k=0;for(const auto& t:tr)if(t.product==0&&t.stage==KProf::P1&&!t.pb.empty())p1k=std::max(p1k,param<unsigned>(t,10));   // all primes (m16)
        const int nprod=[&]{int n=0;for(const auto& t:tr)n=std::max(n,t.product+1);return n;}();
        const KProf::Trace* crt0=k0s[KProf::CRT];
        auto chain=[&](int prod,CUstream a,const Combo* c){int ng[4]={0,0,0,0};for(const auto& t:tr)if(t.product==prod&&!t.pb.empty()){const V* v=nullptr;if(c&&t.stage<4&&k0s[t.stage]&&t.fn==k0s[t.stage]->fn)v=c->v[t.stage];if(v&&v->gsel>=0&&t.gidx!=v->gsel)v=nullptr;if(v&&v->skip)continue;
            if(v&&v->p1m==2&&(t.stage==KProf::P2||t.stage==KProf::P3)){if(ng[t.stage]++||!p1k)continue;   // m16a: group-0 P2 / P3 over all primes
                std::vector<unsigned char> b=t.pb;void* pp[64];for(size_t i=0;i<t.po.size();++i)pp[i]=b.data()+t.po[i];
                const unsigned gy=t.stage==KProf::P3?p1k:t.gy,gz=t.stage==KProf::P2?p1k:t.gz;d.check(d.cuLaunchKernel(t.fn,t.gx,gy,gz,t.bx,1,1,t.smem,a,pp,nullptr));continue;}
            if(v&&v->p1m>=1&&t.stage==KProf::P1){if(param<unsigned>(t,9)!=0||!p1k)continue;   // m16: later P1 groups skipped; group 0 runs r1 = all primes, input loads evict_first (no re-read)
                std::vector<unsigned char> b=t.pb;void* pp[64];std::memcpy(b.data()+t.po[10],&p1k,4);unsigned pf=0;std::memcpy(&pf,b.data()+t.po[11],4);pf&=~(1u<<28);if(v->p1e)pf&=~(3u<<29);std::memcpy(b.data()+t.po[11],&pf,4);
                for(size_t i=0;i<t.po.size();++i)pp[i]=b.data()+t.po[i];d.check(d.cuLaunchKernel(v->fn?v->fn:t.fn,t.gx,t.gy,t.gz,t.bx,1,1,t.smem,a,pp,nullptr));continue;}
            launch(g,t,a,v);}};
        auto csample=[&](const Combo* c,bool cf)->float{CUstream a=g.s[0],b=g.s[1];if(cf)g.bw(a,g.flush,g.flush,g.flush_n,0);else g.do_flush(a);kp.erecord(e0,a);g.swait(b,e0,0);chain(0,a,c);if(nprod>1)chain(1,b,c);kp.erecord(ej,b);g.swait(a,ej,0);kp.erecord(e1,a);d.check(g.esync(e1));return kp.ms(e0,e1);};
        std::vector<Reg> crg;if(crt0)crg=regions(KProf::CRT,*crt0);
        for(const auto& c:cs){
            if(el()>budget-3){std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",c.name.c_str());continue;}
            std::vector<float> pa,va,rr;const int RC=7;
            for(int r=0;r<RC+1;++r){const bool vf=r&1;const float x=csample(vf?&c:nullptr,c.cf),y=csample(vf?nullptr:&c,c.cf);if(r>=1){(vf?va:pa).push_back(x);(vf?pa:va).push_back(y);if(x>0&&y>0)rr.push_back(vf?x/y:y/x);}}
            int ex=-1;
            if(cmp&&!crg.empty()){size_t tot=crg[0].bytes;if(tot>ref_n){if(ref)d.cuMemFree(ref);d.check(d.cuMemAlloc(&ref,tot));ref_n=tot;}
                chain(0,g.s[0],nullptr);d.check(dtod(ref,crg[0].p,tot,g.s[0]));d.check(memset32(crg[0].p,0x5a5a5a5au,tot/4,g.s[0]));chain(0,g.s[0],&c);
                unsigned z=0;d.check(d.cuMemcpyHtoD(bad,&z,4));CUdeviceptr x=crg[0].p,y=ref;unsigned long long n=tot/16;void* pp[]={&x,&y,&n,&bad};
                d.check(d.cuLaunchKernel(cmp,unsigned(g.sms)*4,1,1,256,1,1,0,g.s[0],pp,nullptr));d.check(d.cuCtxSynchronize());d.check(d.cuMemcpyDtoH(&z,bad,4));ex=z?0:1;}
            const double m=med(va)*1e3,pm=med(pa)*1e3;
            bool ab_=false;for(int st=0;st<4;++st)if(c.v[st]&&c.v[st]->abl)ab_=true;
            std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"med_us\":%.1f,\"min_us\":%.1f,\"vs_prod\":%.4f,\"prod_us\":%.1f,\"ex\":%d%s}\n",c.name.c_str(),m,va.empty()?0.0:*std::min_element(va.begin(),va.end())*1e3,med(rr),pm,ex,ab_?",\"abl\":1":"");std::fflush(stdout);}
        // round 14 (geo): whole-batch lines (full B-product graph, production schedule, P1 substituted): ctrl = production kernel
        // and grid recaptured (capture noise floor), then every exact persistent P1 variant
        if(batch&&k0s[KProf::P1]){
            if(el()<budget+15){batch(k0s[KProf::P1]->fn,k0s[KProf::P1]->gx,0u,"ctrl");
                for(const auto& v:vs[KProf::P1])if(v.pers&&v.ok&&v.ex==1&&!v.abl){if(el()>budget+15){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",v.v.c_str());continue;}batch(v.fn,v.gx,v.por,v.v.c_str());}}}
        std::printf("{\"diag\":\"sw_done\",\"s\":%.1f}\n",el());std::fflush(stdout);
    }
};
