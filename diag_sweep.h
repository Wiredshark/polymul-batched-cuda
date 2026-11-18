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
#include <thread>
#include <atomic>
#include <memory>
#include <cmath>
#include <sstream>

inline bool sweep_env(){const char* e=std::getenv("FHERMA_SWEEP");return !e||*e!='0';}

template<class C> struct Sweep {
    C& d; KProf& kp; const unsigned L;
    std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now();double budget=60.0;
    struct Prog { std::string name,arch,src; std::vector<std::string> kn; std::vector<bool> expr; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t_rtc=0,t_jit=0; bool done=false; };
    std::vector<Prog> progs; std::vector<std::thread> th; std::string arch,native;
    struct PB { std::vector<unsigned char> b; std::vector<size_t> o; template<class T> void add(T x){const size_t off=(b.size()+sizeof(T)-1)/sizeof(T)*sizeof(T);b.resize(off+sizeof(T));std::memcpy(b.data()+off,&x,sizeof(T));o.push_back(off);} };
    // mk: kernel with its own parameter list, built from the production trace it replaces (fused variants; no per-stage exactness)
    struct V { std::string v; int prog=-1,ki=-1; unsigned bx=0; int pidx=-1; unsigned pxor=0,por=0; bool abl=false; CUfunction fn=nullptr; double med=0,mn=0,c1=0,c1vs=0,vs=0; int ex=-1; bool ok=false; std::function<bool(const KProf::Trace&,PB&)> mk; bool skip=false; bool wa=false; int p1m=0; bool p1e=false; };   // p1m 1: m16 chain mode, 2: m16a (see run())   // wa: P1 bpt (param 6) = shifted table
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
        // launch-bounds twins of the P2 / CRT kernels: a copy of the kernel text as a template over min-blocks (the original
        // stays, as a template with the production attributes: its <0> instance is the recompiled control)
        // subs: text substitutions applied to the copy only (each must match exactly once, else the twin is dropped)
        auto twin=[&](const char* head,const char* nhead,std::initializer_list<std::pair<const char*,const char*>> subs={})->bool{const size_t p=s.find(head);if(p==std::string::npos)return false;const size_t e=s.find("\n}",p);if(e==std::string::npos)return false;
            std::string k=s.substr(p,e+2-p);k.replace(0,std::strlen(head),nhead);for(size_t q;(q=k.find("#ifdef BPO_SWEEP"))!=std::string::npos;)k.replace(q,16,"#if 1");
            for(const auto& r:subs){const size_t q=k.find(r.first);if(q==std::string::npos||k.find(r.first,q+1)!=std::string::npos)return false;k.replace(q,std::strlen(r.first),r.second);}
            s.insert(e+2,"\n"+k+"\n");return true;};
        const bool has_p2=s.find("extern \"C\" __global__ void __launch_bounds__(128) bpo_middle_r16(")!=std::string::npos;
        const bool has_crt=s.find("extern \"C\" __global__ void __launch_bounds__(128,3) bpo_crt_lean(")!=std::string::npos;
        (void)twin;(void)has_p2;(void)has_crt;
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
        // round 7 (p1x, on 82f8da4 + round-6 scaffolding): P1 store-drain overlap. Production lane 0 of every warp waits for its
        // quad's bulk smem reads (cp.async.bulk.wait_group.read 0) at the END of the quad, before the next quad's phase-A math.
        // lb = flag 32768 alone (quad barrier just before the phase-A stores; pv26 on older kernels: neutral), lw = lb + G 512 XG_LW
        // (the drain wait moved after the next quad's phase-A math, right before that barrier: quad u's store drain overlaps quad
        // u+1's MMA / reduction / twist / J-DIF / R-DIF math), abl_nw = no per-quad drain wait (XABL 9, wrong values: upper bound
        // of what lw can hide), co_prod = compute-only reference. Chains: prod, p1=lb, p1=lw, m16 (production kernel, P1 group 0
        // runs all 16 primes in one launch, group-1 P1 skipped: one input staging + one launch per product instead of two),
        // m16+lw, and cf:* = the same chains after a clean (read-only) L2 flush instead of the dirty 2xL2 write flush.
        {struct P1V{const char* n;const char* f;unsigned g;bool abl;bool wa;};
         static const char* const CO="(BPO_XF|(8u<<20)|(1u<<24))";
         static const P1V p1v[]={
             {"ctrl","BPO_XF",0,false,false},{"lb","(BPO_XF|32768u)",0,false,false},{"lw","(BPO_XF|32768u)",512,false,false},
             {"abl_nw","(BPO_XF|(9u<<20))",0,true,false},{"co_prod",CO,0,true,false}};
         constexpr size_t NV=sizeof(p1v)/sizeof(p1v[0]),PER=3;
         int pi=-1;
         for(size_t k=0;k<NV;++k){const auto& x=p1v[k];if(k%PER==0){const char nm[4]={'p','1',char('a'+k/PER),0};pi=prog(nm,arch);}
             const std::string kn="sw_p1_"+std::to_string(k);progs[size_t(pi)].src+="SWP1("+kn+","+x.f+","+std::to_string(x.g)+"u,2)\n";
             addv(KProf::P1,x.n,pi,addk(pi,kn,false),-1,0,0,x.abl);vs[KProf::P1].back().wa=x.wa;}}
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
            const std::string a="--gpu-architecture="+P.arch;const char* op[]={a.c_str(),"--std=c++17"};
            if(d.nvrtcCompileProgram(pr,2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
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
        d.check(d.cuLaunchKernel(v&&v->fn?v->fn:t.fn,t.gx,t.gy,t.gz,v&&v->bx?v->bx:t.bx,1,1,t.smem,st,pp,nullptr));(void)g;}
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
    template<class D> void run(D& g){
        join();
        {std::printf("{\"diag\":\"sw_info\",\"arch\":\"%s\",\"hw_threads\":%u,\"compile_wall_s\":%.1f,\"progs\":[",arch.c_str(),std::thread::hardware_concurrency(),el());
         for(size_t i=0;i<progs.size();++i)std::printf("%s[\"%s\",%zu,%.1f,%.1f%s%s%s]",i?",":"",progs[i].name.c_str(),progs[i].kn.size(),progs[i].t_rtc,progs[i].t_jit,
             progs[i].err.empty()?"":",\"",progs[i].err.empty()?"":progs[i].err.substr(0,120).c_str(),progs[i].err.empty()?"":"\"");
         std::printf("]}\n");std::fflush(stdout);}
        if(kp.last.empty())return;
        const auto& tr=kp.traces[kp.last.back().variant];
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
            V prod;prod.v="prod";prod.fn=nullptr;ab(g,prod,k0,k1,q0,q1,R);{V pv=prod;pv.fn=k0->fn;print(kn[st],pv);}
            for(auto& v:vs[st]){
                if(el()>budget-8){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",kn[st],v.v.c_str());v.ok=false;continue;}
                if(!v.ok){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"skipped\":\"not compiled\"}\n",kn[st],v.v.c_str());continue;}
                try{ab(g,v,k0,k1,q0,q1,R);v.ex=v.mk?-1:exact(g,*k0,prel,rg,v);}catch(const std::exception& e){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"error\":\"%.60s\"}\n",kn[st],v.v.c_str(),e.what());v.ok=false;d.cuCtxSynchronize();continue;}
                print(kn[st],v);}
        }
        // product chains (products 0 and 1 on two streams from a cold flush) for the best exact combinations
        auto score=[](const V* v){return std::sqrt(v->vs*v->c1vs);};   // rank by the geometric mean of the c2 and c1 ratios
        auto best=[&](int st,size_t rank)->const V*{std::vector<const V*> c;for(auto& v:vs[st])if(v.ok&&v.ex==1&&!v.abl&&v.v!="ctrl"&&v.vs>0&&v.c1vs>0&&score(&v)<0.995)c.push_back(&v);
            std::sort(c.begin(),c.end(),[&](const V* a,const V* b){return score(a)<score(b);});return rank<c.size()?c[rank]:nullptr;};
        auto byname=[&](int st,const char* n)->const V*{for(auto& v:vs[st])if(v.ok&&v.v==n)return &v;return nullptr;};
        struct Combo{std::string name;const V* v[4];bool cf=false;};std::vector<Combo> cs;
        auto addc=[&](const V* a,const V* b,const V* c,const V* e){Combo x{"",{a,b,c,e},false};bool any=false;for(int st=0;st<4;++st)if(x.v[st]){any=true;x.name+=(x.name.empty()?"":",")+std::string(kn[st])+"="+x.v[st]->v;}
            if(!any)x.name="prod";for(auto& y:cs)if(y.name==x.name)return;cs.push_back(x);};
        static V skip3;skip3.v="skip";skip3.skip=true;skip3.ok=true;
        (void)skip3;
        // m16 chain mode: the production P1 kernel (or lw) with the group-0 trace's r1 = all primes, group-1 P1 traces skipped
        static V m16;m16.v="m16";m16.ok=true;m16.p1m=1;static V m16lw;m16lw.v="m16+lw";m16lw.p1m=1;
        if(const V* lw=byname(0,"lw")){m16lw.fn=lw->fn;m16lw.ok=lw->ok&&lw->ex==1;}
        // m16a: m16 plus P2 and P3 as one launch over all primes too (group-0 trace with grid z / y = all primes, later groups
        // skipped) = the FHERMA_PO_GROUP=16 schedule; P2/P3 kernels and parameters unchanged (they index by blockIdx prime)
        static V m16a=m16;m16a.v="m16a";m16a.p1m=2;static V m16alw=m16lw;m16alw.v="m16a+lw";m16alw.p1m=2;static V pas;pas.v="all";pas.ok=true;pas.p1m=2;
        // m16e: m16 with the P1 stores' L2 evict_first policy (TMA policy bits 29-30 cleared: stm 0 + flag 2048) = exactly what
        // FHERMA_PO_GROUP=16 does in production (gfl = 0 for an ungrouped P1; P2/P3 stay per 8-prime L2 group): local B=128 harness
        // 51.9 / 53.5 ms vs prod 55.9, with lw 50.8 / 50.7
        static V m16e=m16;m16e.v="m16e";m16e.p1e=true;static V m16elw=m16lw;m16elw.v="m16e+lw";m16elw.p1e=true;
        addc(nullptr,nullptr,nullptr,nullptr);
        for(const char* n:{"lb","lw"})addc(byname(0,n),nullptr,nullptr,nullptr);   // each P1 variant alone
        {Combo x{"m16",{&m16,nullptr,nullptr,nullptr}};cs.push_back(x);}
        if(m16lw.ok){Combo x{"m16+lw",{&m16lw,nullptr,nullptr,nullptr}};cs.push_back(x);}
        (void)m16alw;(void)m16a;(void)pas;   // m16a / m16a+lw measured locally +12..14% (C workspace leaves L2): not run
        {Combo x{"m16e",{&m16e,nullptr,nullptr,nullptr}};cs.push_back(x);}
        if(m16elw.ok){Combo x{"m16e+lw",{&m16elw,nullptr,nullptr,nullptr}};cs.push_back(x);}
        {const size_t n0=cs.size();for(size_t i=0;i<n0;++i){if(i==1)continue;Combo x=cs[i];x.name="cf:"+x.name;x.cf=true;cs.push_back(x);}}   // clean-flush twins
        if(cs.size()>16)cs.resize(16);
        unsigned p1k=0;for(const auto& t:tr)if(t.product==0&&t.stage==KProf::P1&&!t.pb.empty())p1k=std::max(p1k,param<unsigned>(t,10));   // all primes (m16)
        const int nprod=[&]{int n=0;for(const auto& t:tr)n=std::max(n,t.product+1);return n;}();
        const KProf::Trace* crt0=k0s[KProf::CRT];
        auto chain=[&](int prod,CUstream a,const Combo* c){int ng[4]={0,0,0,0};for(const auto& t:tr)if(t.product==prod&&!t.pb.empty()){const V* v=nullptr;if(c&&t.stage<4&&k0s[t.stage]&&t.fn==k0s[t.stage]->fn)v=c->v[t.stage];if(v&&v->skip)continue;
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
            std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"med_us\":%.1f,\"min_us\":%.1f,\"vs_prod\":%.4f,\"prod_us\":%.1f,\"ex\":%d}\n",c.name.c_str(),m,va.empty()?0.0:*std::min_element(va.begin(),va.end())*1e3,med(rr),pm,ex);std::fflush(stdout);}
        std::printf("{\"diag\":\"sw_done\",\"s\":%.1f}\n",el());std::fflush(stdout);
    }
};
