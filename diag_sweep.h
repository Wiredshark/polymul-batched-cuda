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

#define BPO_XW_SMEM (4u*8u*320u*4u)   // one P1 X buffer (bytes), = BPO_X_SMEM(BPO_XF)
inline bool sweep_env(){const char* e=std::getenv("FHERMA_SWEEP");return !e||*e!='0';}

template<class C> struct Sweep {
    C& d; KProf& kp; const unsigned L;
    std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now();double budget=60.0;
    struct Prog { std::string name,arch,src; std::vector<std::string> kn; std::vector<bool> expr; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t_rtc=0,t_jit=0; bool done=false; };
    std::vector<Prog> progs; std::vector<std::thread> th; std::string arch,native;
    struct V { std::string v; int prog=-1,ki=-1; unsigned bx=0,sm=0; int pidx=-1; unsigned pxor=0,por=0; bool abl=false; CUfunction fn=nullptr; double med=0,mn=0,c1=0,c1vs=0,vs=0; int ex=-1; bool ok=false; };
    std::vector<V> vs[4];   // per stage (KProf::P1..CRT)
    CUevent ea=nullptr,eb=nullptr,e0=nullptr,e1=nullptr,ej=nullptr;CUdeviceptr ref=0,bad=0;size_t ref_n=0;CUfunction cmp=nullptr;
    decltype(&::cuMemcpyDtoDAsync) dtod=nullptr; decltype(&::cuMemsetD32Async) memset32=nullptr;
    double el()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();}
    Sweep(C& c,const std::string& source,unsigned l):d(c),kp(c.kp),L(l){
        if(const char* e=std::getenv("FHERMA_SWEEP_S"))budget=std::atof(e);
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
        const bool has_p2=twin("extern \"C\" __global__ void __launch_bounds__(128) bpo_middle_r16(","template<int SWMB> __global__ void __launch_bounds__(128,SWMB) sw_p2r16(");
        const char* const crt_head="extern \"C\" __global__ void __launch_bounds__(128,3) bpo_crt_tc_cm(";
        const bool has_crt=s.find(crt_head)!=std::string::npos;
        // CRT twins (round 2; production 168 regs + 72 B spills at (128,3)): hl = the 36 MMA partial words h[] read from the warp's
        // smem stage inside the carry loop instead of 36 live registers (second __syncwarp after the loop); ld = residue loads
        // __ldcs (dead after the CRT). Template over the launch-bounds min-blocks.
        static const std::pair<const char*,const char*> crt_hl[]={
            {"        u32 h[SST];\n        #pragma unroll\n        for(u32 q=0;q<SST/4;++q){const uint4 v=*reinterpret_cast<const uint4*>(st+lane*SST+4*q);h[4*q]=v.x;h[4*q+1]=v.y;h[4*q+2]=v.z;h[4*q+3]=v.w;}\n        __syncwarp();\n",
             "        const u32* const h=st+lane*SST;\n"},
            {"        pend=u32(c);\n","        pend=u32(c);__syncwarp();   // stage reads of this digit done\n"}};
        static const std::pair<const char*,const char*> crt_ld={"                a[m][s][0]=base[0];a[m][s][1]=base[8];\n                a[m][s][2]=base[u64(4)*8*N];a[m][s][3]=base[u64(4)*8*N+8];\n",
             "                a[m][s][0]=__ldcs(base);a[m][s][1]=__ldcs(base+8);\n                a[m][s][2]=__ldcs(base+u64(4)*8*N);a[m][s][3]=__ldcs(base+u64(4)*8*N+8);\n"};
        const bool has_crt_hl=has_crt&&twin(crt_head,"template<int SWMB> __global__ void __launch_bounds__(128,SWMB) sw_crt_hl(",{crt_hl[0],crt_hl[1]});
        const bool has_crt_ld=has_crt&&twin(crt_head,"template<int SWMB> __global__ void __launch_bounds__(128,SWMB) sw_crt_ld(",{crt_ld});
        const bool has_crt_hlld=has_crt&&twin(crt_head,"template<int SWMB> __global__ void __launch_bounds__(128,SWMB) sw_crt_hlld(",{crt_hl[0],crt_hl[1],crt_ld});
        {const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
        s+="\n#define SWP1(NAME,F,MB) extern \"C\" __global__ void __launch_bounds__((((F)&8u)?512:256),MB) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){__shared__ __align__(128) u32 X[BPO_X_SMEM(F)/4];bpo_first_xb<(F)>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);}\n"
           "#define SWP1S(NAME,F,MB) extern \"C\" __global__ void __launch_bounds__(256,MB) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){__shared__ __align__(128) u32 X[BPO_X_SMEM(F)/4];bpo_first_xs<(F)>(a,b,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);}\n"
           "#define SWP1W(NAME,F,MB) extern \"C\" __global__ void __launch_bounds__(512,1) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){extern __shared__ __align__(128) u32 XW[];bpo_first_xw<(F)>(a,b,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,XW);}\n"
           "#define SWP3(NAME,T,LB) extern \"C\" __global__ void __launch_bounds__ LB NAME(const u32* C,u32* D,const u32* primes,const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){__shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;bpo_p3_body<(T)>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);}\n";
        const std::string base=s;
        auto prog=[&](const char* name,const std::string& a)->int{Prog P;P.name=name;P.arch=a;P.src=base;progs.push_back(std::move(P));return int(progs.size())-1;};
        auto addk=[&](int pi,const std::string& kn,bool ex)->int{Prog& P=progs[size_t(pi)];P.kn.push_back(kn);P.expr.push_back(ex);return int(P.kn.size())-1;};
        auto addv=[&](int st,const std::string& name,int pi,int ki,int pidx=-1,unsigned pxor=0,unsigned por=0,bool abl=false,unsigned bx=0){V v;v.v=name;v.prog=pi;v.ki=ki;v.pidx=pidx;v.pxor=pxor;v.por=por;v.abl=abl;v.bx=bx;vs[st].push_back(v);};
        // programs in priority order (compiles that would start after the compile deadline are skipped)
        {const int c=prog("cmp",arch);progs[size_t(c)].src="typedef unsigned int u32;typedef unsigned long long u64;\n"   // exactness compare kernel (tiny)
            "extern \"C\" __global__ void sw_cmp(const uint4* a,const uint4* b,u64 n,u32* bad){u32 c=0;for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=u64(gridDim.x)*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n";
         addk(c,"sw_cmp",false);}
        // P1: name, F (relative to the production BPO_XF), launch-bounds min blocks (production: BPO_TC_MINB = 2), ablation
        struct P1V{const char* n;const char* f;int mb;bool abl;char k;};   // k: 'b' bpo_first_xb (SWP1), 's' bpo_first_xs (SWP1S), 'w' bpo_first_xw (SWP1W)
        // Round 3 (sweep3, production = 4113c51). Round-2 knobs measured on target (pv37) and dropped: cpq, twh, mnv, q4, f4 and
        // their combinations (all >= 1.0); kept: ctrl, abl_noA, co_prod (compute-only reference).
        // New (batched_p1x.cuh bpo_first_xs / bpo_first_xw): sp = same-warp software pipelining (next quad's MMA conversion in
        // 4 hooks inside phase B); noh = the bpo_first_xs body without hooks (control for sp: phase-A math before the quad
        // barrier, else production order); ovl = noh + input half 1 in flight during quad 0's m-tile-0 conversion (bit 27);
        // sp_ovl = both; ws = warp specialised 512 threads, 1 block/SM, 2 X buffers (80 KB dynamic smem), producers MMA +
        // phase A, consumers phase B; ws_l1 = producers also phase-B layout 1 (bit 18); nfw = production without the final
        // bulk wait_group 0 (bit 26). co_* = compute-only twins (column-0 input from L2, no stores: compare against co_prod).
        // Local (RTX 5070, pre-commit): all exact; co_sp 1.04-1.05x co_prod, co_ws 1.15x, co_ws_l1 1.27x; sp_l1 (hooks in
        // layout 1 only) spills 112 B under the sm_120 JIT and is dropped.
        static const P1V p1g[][4]={
            {{"ctrl","BPO_XF",2,false,'b'},{"sp","BPO_XF",2,false,'s'},{"ovl","(BPO_XF|(1u<<27)|(1u<<25))",2,false,'s'},{"ws","BPO_XF",1,false,'w'}},
            {{"ws_l1","(BPO_XF|(1u<<18))",1,false,'w'},{"noh","(BPO_XF|(1u<<25))",2,false,'s'},{"nfw","(BPO_XF|(1u<<26))",2,false,'b'},{"ovl_nfw","(BPO_XF|(1u<<27)|(1u<<26)|(1u<<25))",2,false,'s'}},
            {{"abl_noA","(BPO_XF|(4u<<20))",2,true,'b'},{"co_prod","(BPO_XF|(8u<<20)|(1u<<24))",2,true,'b'},{"co_sp","(BPO_XF|(8u<<20)|(1u<<24))",2,true,'s'},{"co_ovl","(BPO_XF|(1u<<27)|(1u<<25)|(8u<<20)|(1u<<24))",2,true,'s'}},
            {{"co_ws","(BPO_XF|(8u<<20)|(1u<<24))",1,true,'w'},{"co_ws_l1","(BPO_XF|(1u<<18)|(8u<<20)|(1u<<24))",1,true,'w'},{"co_noh","(BPO_XF|(1u<<25)|(8u<<20)|(1u<<24))",2,true,'s'},{"sp_ovl","(BPO_XF|(1u<<27))",2,false,'s'}}};
        int kc=0;
        auto p1grp=[&](int g){const int pi=prog(("p1"+std::string(1,char('a'+g))).c_str(),arch);
            for(const auto& x:p1g[g]){if(!x.n)continue;const std::string kn="sw_p1_"+std::to_string(kc++);
                progs[size_t(pi)].src+=std::string(x.k=='s'?"SWP1S(":x.k=='w'?"SWP1W(":"SWP1(")+kn+","+x.f+","+std::to_string(x.mb)+")\n";
                addv(KProf::P1,x.n,pi,addk(pi,kn,false),-1,0,0,x.abl,x.k=='w'?512u:std::strstr(x.f,"|8u)")?512u:0u);
                if(x.k=='w')vs[KProf::P1].back().sm=2u*BPO_XW_SMEM;}};   // two X buffers (dynamic smem, opt-in above 48 KB)
        p1grp(0);
        {const int pi=prog("p3",arch);   // P3: template flags T / launch bounds
            addv(KProf::P3,"ctrl",pi,addk(pi,"bpo_last_cm3<0>",true));
            struct P3V{const char* n;const char* t;const char* lb;};
            static const P3V p3v[]={{"tma_st","(BPO_P3T|1u)","(256)"},{"tma_ldst","(BPO_P3T|3u)","(256)"},{"evict_last","(BPO_P3T|8u)","(256)"}};   // production BPO_P3T has tn2 (2048)
            int k=0;for(const auto& x:p3v){const std::string kn="sw_p3_"+std::to_string(k++);progs[size_t(pi)].src+="SWP3("+kn+","+x.t+","+x.lb+")\n";addv(KProf::P3,x.n,pi,addk(pi,kn,false));}
}
        if(has_crt){const int pi=prog("crt",arch);addv(KProf::CRT,"ctrl",pi,addk(pi,"bpo_crt_tc_cm<0>",true));
            if(has_crt_hl)for(int mb:{2,3})addv(KProf::CRT,"hl_mb"+std::to_string(mb),pi,addk(pi,"sw_crt_hl<"+std::to_string(mb)+">",true));
            if(has_crt_ld)addv(KProf::CRT,"ld_mb3",pi,addk(pi,"sw_crt_ld<3>",true));
            if(has_crt_hlld)for(int mb:{3})addv(KProf::CRT,"hlld_mb"+std::to_string(mb),pi,addk(pi,"sw_crt_hlld<"+std::to_string(mb)+">",true));}
        addv(KProf::P3,"dfl_discard_flip",-1,-1,8,1u);addv(KProf::P3,"dfl_evfirst_flip",-1,-1,8,4u);   // production kernel, runtime flag bits flipped
        addv(KProf::CRT,"dfl_discard_flip",-1,-1,8,1u);addv(KProf::CRT,"dfl_stcs_flip",-1,-1,8,2u);
        addv(KProf::P2,"ldcs_flip",-1,-1,6,4u);   // untw bit 4: A/B loads __ldcs
        if(has_p2){const int pi=prog("p2",arch);
            addv(KProf::P2,"ctrl",pi,addk(pi,"bpo_middle_r16<0>",true));
            for(int mb:{6,7,8})addv(KProf::P2,"mb"+std::to_string(mb),pi,addk(pi,"sw_p2r16<"+std::to_string(mb)+">",true));}   // (discard: production since bed0fde)
        p1grp(1);
        if(std::getenv("FHERMA_SWEEP_NATIVE")){const int pi=prog("native",native);progs[size_t(pi)].src+="SWP1(sw_n_p1,BPO_XF,BPO_TC_MINB)\n";
            addv(KProf::P1,"native_"+native,pi,addk(pi,"sw_n_p1",false));
            if(has_p2)addv(KProf::P2,"native_"+native,pi,addk(pi,"bpo_middle_r16<0>",true));
            addv(KProf::P3,"native_"+native,pi,addk(pi,"bpo_last_cm3<0>",true));
            if(has_crt)addv(KProf::CRT,"native_"+native,pi,addk(pi,"bpo_crt_tc_cm<0>",true));}
        p1grp(2);p1grp(3);
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
    ~Sweep(){join();d.cuCtxSynchronize();for(auto& P:progs)if(P.mod)d.cuModuleUnload(P.mod);for(auto e:{ea,eb,e0,e1,ej})if(e)kp.edestroy(e);for(auto p:{ref,bad})if(p)d.cuMemFree(p);}

    // ---- measurement -------------------------------------------------------------------------------------------
    template<class D> void launch(D& g,const KProf::Trace& t,CUstream st,const V* v){
        std::vector<unsigned char> b=t.pb;void* pp[64];
        if(v&&v->pidx>=0&&size_t(v->pidx)<t.po.size()){unsigned x=0;std::memcpy(&x,b.data()+t.po[size_t(v->pidx)],4);x=(x^v->pxor)|v->por;std::memcpy(b.data()+t.po[size_t(v->pidx)],&x,4);}
        for(size_t i=0;i<t.po.size();++i)pp[i]=b.data()+t.po[i];
        d.check(d.cuLaunchKernel(v&&v->fn?v->fn:t.fn,t.gx,t.gy,t.gz,v&&v->bx?v->bx:t.bx,1,1,v&&v->sm?v->sm:t.smem,st,pp,nullptr));(void)g;}
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
        {auto fsa=d.driver.template get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");   // dynamic smem above 48 KB (ws)
         for(int st=0;st<4;++st)for(auto& v:vs[st])if(v.ok&&v.fn&&v.sm>48u*1024u&&(!fsa||fsa(v.fn,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(v.sm))!=CUDA_SUCCESS))v.ok=false;}
        // first trace of a stage for products 0 / 1 and the trace just before it in the same product (predecessor)
        auto first=[&](int prod,int st,size_t* pi)->const KProf::Trace*{for(size_t i=0;i<tr.size();++i)if(tr[i].product==prod&&tr[i].stage==st&&!tr[i].pb.empty()){if(pi)*pi=i;return &tr[i];}return nullptr;};
        auto pred=[&](int prod,size_t i)->const KProf::Trace*{for(size_t j=i;j-->0;)if(tr[j].product==prod&&!tr[j].pb.empty())return &tr[j];return nullptr;};
        CUfunction pf[4]={d.bpo_first_x,d.bpo_middle_r16,d.bpo_last_cm3,d.bpo_crt_tc_cm};static const char* kn[4]={"p1","p2","p3","crt"};
        {const auto tw=std::chrono::steady_clock::now();   // valid product-0 data in every buffer + clock ramp after the idle compile wait (>= 0.3 s busy)
         for(int w=0;w<200&&(w<3||std::chrono::duration<double>(std::chrono::steady_clock::now()-tw).count()<0.3);++w){
             for(size_t i=0;i<tr.size();++i){const auto& t=tr[i];if(t.product==0&&!t.pb.empty())g.replay(t,g.s[0]);}d.check(d.cuCtxSynchronize());}}
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
                try{ab(g,v,k0,k1,q0,q1,R);v.ex=exact(g,*k0,prel,rg,v);}catch(const std::exception& e){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"error\":\"%.60s\"}\n",kn[st],v.v.c_str(),e.what());v.ok=false;d.cuCtxSynchronize();continue;}
                print(kn[st],v);}
        }
        // product chains (products 0 and 1 on two streams from a cold flush) for the best exact combinations
        auto score=[](const V* v){return std::sqrt(v->vs*v->c1vs);};   // rank by the geometric mean of the c2 and c1 ratios
        auto best=[&](int st,size_t rank)->const V*{std::vector<const V*> c;for(auto& v:vs[st])if(v.ok&&v.ex==1&&!v.abl&&v.v!="ctrl"&&v.vs>0&&v.c1vs>0&&score(&v)<0.995)c.push_back(&v);
            std::sort(c.begin(),c.end(),[&](const V* a,const V* b){return score(a)<score(b);});return rank<c.size()?c[rank]:nullptr;};
        auto byname=[&](int st,const char* n)->const V*{for(auto& v:vs[st])if(v.ok&&v.v==n)return &v;return nullptr;};
        struct Combo{std::string name;const V* v[4];};std::vector<Combo> cs;
        auto addc=[&](const V* a,const V* b,const V* c,const V* e){Combo x{"",{a,b,c,e}};bool any=false;for(int st=0;st<4;++st)if(x.v[st]){any=true;x.name+=(x.name.empty()?"":",")+std::string(kn[st])+"="+x.v[st]->v;}
            if(!any)x.name="prod";for(auto& y:cs)if(y.name==x.name)return;cs.push_back(x);};
        addc(nullptr,nullptr,nullptr,nullptr);
        addc(best(0,0),best(1,0),best(2,0),best(3,0));addc(best(0,0),nullptr,nullptr,nullptr);addc(best(0,1),best(1,0),best(2,0),best(3,0));
        addc(best(0,2),nullptr,nullptr,nullptr);addc(nullptr,best(1,0),best(2,0),best(3,0));
        {const std::string nn="native_"+native;addc(byname(0,nn.c_str()),byname(1,nn.c_str()),byname(2,nn.c_str()),byname(3,nn.c_str()));}
        for(const char* n:{"sp","ovl","ws","ws_l1","nfw","ovl_nfw"})addc(byname(0,n),nullptr,nullptr,nullptr);   // each new P1 variant alone
        // P3 store policy / CRT input path only show their effect on the CRT in a chain (residues L2-resident or not)
        addc(nullptr,nullptr,byname(2,"evict_last"),nullptr);addc(nullptr,nullptr,byname(2,"dfl_evfirst_flip"),nullptr);
        addc(nullptr,nullptr,nullptr,byname(3,"hl_mb3"));addc(nullptr,nullptr,byname(2,"evict_last"),byname(3,"hlld_mb3"));
        if(cs.size()>16)cs.resize(16);
        const int nprod=[&]{int n=0;for(const auto& t:tr)n=std::max(n,t.product+1);return n;}();
        const KProf::Trace* crt0=k0s[KProf::CRT];
        auto chain=[&](int prod,CUstream a,const Combo* c){for(const auto& t:tr)if(t.product==prod&&!t.pb.empty()){const V* v=nullptr;if(c&&t.stage<4&&k0s[t.stage]&&t.fn==k0s[t.stage]->fn)v=c->v[t.stage];launch(g,t,a,v);}};
        auto csample=[&](const Combo* c)->float{CUstream a=g.s[0],b=g.s[1];g.do_flush(a);kp.erecord(e0,a);g.swait(b,e0,0);chain(0,a,c);if(nprod>1)chain(1,b,c);kp.erecord(ej,b);g.swait(a,ej,0);kp.erecord(e1,a);d.check(g.esync(e1));return kp.ms(e0,e1);};
        std::vector<Reg> crg;if(crt0)crg=regions(KProf::CRT,*crt0);
        for(const auto& c:cs){
            if(el()>budget-3){std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",c.name.c_str());continue;}
            std::vector<float> pa,va,rr;const int RC=7;
            for(int r=0;r<RC+1;++r){const bool vf=r&1;const float x=csample(vf?&c:nullptr),y=csample(vf?nullptr:&c);if(r>=1){(vf?va:pa).push_back(x);(vf?pa:va).push_back(y);if(x>0&&y>0)rr.push_back(vf?x/y:y/x);}}
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
