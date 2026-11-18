#pragma once
// Round 10 (cmp): P1 quad-math replica probe (exit-time, own NVRTC program, never part of the production module).
// Question: is P1's transform math (twist + J-DIF + R-DIF + cross twiddle, ~49 us/product in situ by the pv43 co_noAB
// ablation) slow because of the code itself at P1's occupancy (16 warps/SM, 2 barriers per quad), or because of its
// interplay with base conversion / staging / TMA? Kernels replay one bpo_first_x quad (4 primes x 8 digits x 256 rows,
// 40 KB smem, rows 64rr+8w+gid, same xjdif / phase-A stages / xrdif6q math with flag-1 trivial twiddles) on smem-resident
// data with no global data traffic (stage / twist / cross-twiddle tables from L1/L2 as in production), no MMA, no TMA:
//   ab      phase A + barrier + phase B + barrier (production order), 256 threads, 2 blocks/SM (= P1 occupancy)
//   ab_b1   same, 1 block/SM (occupancy sensitivity: 8 warps/SM)
//   a, b    phase A only / phase B only (barriers kept)
//   ab_nobar  no __syncthreads (warp syncs only): the barrier cost
//   ab_h, a_h, b_h  the same with Harvey-lazy butterflies (p < 2^30, words in [0,2p); h30 9d2a5bd arithmetic)
//   ab512   512-thread quads (16 values/thread in phase A, len-256 stage in registers, 2 columns/warp in phase B; the
//           len-128 stage omitted, rates normalised per butterfly): 32 warps/SM in the same 40 KB
// Line: {"diag":"p1q","k":...,"bfly_per_sm_clk":r,"us_per_product":u,...}; us_per_product = 2048 quads x the quad's full
// work (45056 butterflies for ab; a 20480, b 24576) at the measured rate on all SMs. Compare: in-situ P1 phase A+B ~49
// us/product (pv43 co_prod - co_noAB), P2 in situ ~12.4 bfly/clk/SM (pv50 nomem), bf_current 9.91 (pv27).
// FHERMA_DIAG_P1Q=0 disables.
static const char* const kDiagP1qSource=R"PQ(typedef unsigned u32; typedef unsigned long long u64;
__constant__ u32 PQP[4]={2013265921u,2113929217u,2130706433u,1811939329u};
__constant__ u32 PQH[4]={1073479681u,1071513601u,1069547521u,1068236801u};   // < 2^30 (Harvey)
__device__ __forceinline__ u32 qadd(u32 a,u32 b,u32 p){const u32 s=a+b;return min(s,s-p);}
__device__ __forceinline__ u32 qsub(u32 a,u32 b,u32 p){const u32 d=a-b;return min(d,d+p);}
__device__ __forceinline__ u32 qmsh(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));const u32 r=x*u32(c)-q*p;return min(r,r-p);}
__device__ __forceinline__ void qbf(u32& x,u32& y,u64 w,u32 p){const u32 a=x,b=y;x=qadd(a,b,p);y=qmsh(a+p-b,w,p);}
__device__ __forceinline__ void qbf1(u32& x,u32& y,u32 p){const u32 a=x,b=y;x=qadd(a,b,p);y=qsub(a,b,p);}
// H (Harvey lazy, p < 2^30 as in h30 9d2a5bd): values in [0,2p), add corrected against 2p, Shoup product unreduced in [0,2p)
__device__ __forceinline__ u32 qmshl(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));return x*u32(c)-q*p;}
template<bool H>__device__ __forceinline__ void qbfH(u32& x,u32& y,u64 w,u32 p){if(H){const u32 a=x,b=y;x=qadd(a,b,2*p);y=qmshl(a+2*p-b,w,p);}else qbf(x,y,w,p);}
template<bool H>__device__ __forceinline__ void qbf1H(u32& x,u32& y,u32 p){if(H){const u32 a=x,b=y;x=qadd(a,b,2*p);y=qadd(a,2*p-b,2*p);}else qbf1(x,y,p);}
template<bool H>__device__ __forceinline__ u32 qmshH(u32 x,u64 c,u32 p){return H?qmshl(x,c,p):qmsh(x,c,p);}
#define qbf qbfH<H>
#define qbf1 qbf1H<H>
#define qmsh qmshH<H>
// xjdif<F> with flag 1 (psi^0 butterflies trivial)
template<bool H>__device__ __forceinline__ void qjdif(u32 (&x8)[8],const u64* st,u32 p){
    qbf1(x8[0],x8[4],p);qbf(x8[1],x8[5],st[32],p);qbf(x8[2],x8[6],st[64],p);qbf(x8[3],x8[7],st[96],p);
    const u64 w64=st[64];
    qbf1(x8[0],x8[2],p);qbf(x8[1],x8[3],w64,p);qbf1(x8[4],x8[6],p);qbf(x8[5],x8[7],w64,p);
    qbf1(x8[0],x8[1],p);qbf1(x8[2],x8[3],p);qbf1(x8[4],x8[5],p);qbf1(x8[6],x8[7],p);}
__device__ __forceinline__ ulonglong2 qldv(const u64* ctg,u32 lane,u32 m2){ulonglong2 w;
    asm volatile("ld.global.nc.v2.u64 {%0,%1},[%2];":"=l"(w.x),"=l"(w.y):"l"(ctg+8*lane+2*m2));return w;}
// xrdif6q<F> math (flags 1|8192|131072, no XQ4 / 1024), NCOL = 4 (production) or 2 columns; TMA replaced by nothing
template<u32 NCOL,bool H>
__device__ __forceinline__ void qrdif6q(u32* const base0,const u32 so,const u64* st,const u64* ct,const u32 p,u32 lane){
    constexpr u32 XD=320u,Q=2u,NI=NCOL/Q;
    const u32 l7=lane&7u;
    {u64 w64[4],w32[2];
        #pragma unroll
        for(u32 m=0;m<4;++m)w64[m]=st[192u+l7+8u*m];
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)w32[qq]=st[224u+l7+8u*qq];
        const u64 w16=st[240u+l7];
        u32* const pb=base0+so+l7+72u*(lane>>3);
        #pragma unroll
        for(u32 pr=0;pr<NI;++pr){u32 v[Q][8];
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)v[n][m]=pb[(Q*pr+n)*XD+8*m+4*(m>>2)];
            #pragma unroll
            for(u32 m=0;m<4;++m)
                #pragma unroll
                for(u32 n=0;n<Q;++n)qbf(v[n][m],v[n][m+4],w64[m],p);
            #pragma unroll
            for(u32 qq=0;qq<2;++qq)
                #pragma unroll
                for(u32 n=0;n<Q;++n)
                    #pragma unroll
                    for(u32 hh=0;hh<2;++hh)qbf(v[n][qq+4*hh],v[n][qq+4*hh+2],w32[qq],p);
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;m+=2)qbf(v[n][m],v[n][m+1],w16,p);
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)pb[(Q*pr+n)*XD+8*m+4*(m>>2)]=v[n][m];
            if(pr+1<NI)__syncwarp();}
    }
    __syncwarp();
    u64 w8[4];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)w8[lo]=st[248u+lo];
    const u64 w253=st[253u];
    const u32* const pr0=base0+so+8*lane+4*(lane>>2);
    const bool sw=(lane>>2)&1u;uint4* const sd=reinterpret_cast<uint4*>(base0+8*lane)+(sw?1:0);
    #pragma unroll 1
    for(u32 pr=0;pr<NI;++pr){u32 v[Q][8];
        #pragma unroll
        for(u32 n=0;n<Q;++n){const uint4* p4=reinterpret_cast<const uint4*>(pr0+(Q*pr+n)*XD);const uint4 x0=p4[0],x1=p4[1];
            v[n][0]=x0.x;v[n][1]=x0.y;v[n][2]=x0.z;v[n][3]=x0.w;v[n][4]=x1.x;v[n][5]=x1.y;v[n][6]=x1.z;v[n][7]=x1.w;}
        #pragma unroll
        for(u32 lo=0;lo<4;++lo)
            #pragma unroll
            for(u32 n=0;n<Q;++n){if(lo==0)qbf1(v[n][0],v[n][4],p);else qbf(v[n][lo],v[n][lo+4],w8[lo],p);}
        #pragma unroll
        for(u32 k=0;k<4;++k)
            #pragma unroll
            for(u32 n=0;n<Q;++n){const u32 lo=(k&1)+4*(k>>1);if(lo&1)qbf(v[n][lo],v[n][lo+2],w253,p);else qbf1(v[n][lo],v[n][lo+2],p);}
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){const ulonglong2 w=qldv(ct,lane,m2);
            #pragma unroll
            for(u32 n=0;n<Q;++n){const u32 a=v[n][2*m2],b=v[n][2*m2+1];v[n][2*m2]=qmsh(a+b,u64(w.x),p);v[n][2*m2+1]=qmsh(a+p-b,u64(w.y),p);}}
        __syncwarp();
        #pragma unroll
        for(u32 n=0;n<Q;++n){
            const uint4 e0=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]),e1=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);
            uint4* const d=sd+(Q*pr+n)*(XD/4);
            d[0]=sw?e1:e0;d[sw?-1:1]=sw?e0:e1;}
    }
    __syncwarp();
}
// MODE bits: 1 phase A, 2 phase B, 4 block barriers (else warp syncs)
template<u32 MODE,u32 NT,bool H,u32 AFR=0>
__device__ __forceinline__ void p1q_body(u32* out,u64* cyc,const u64* st,const u64* tw,const u64* ctab,u32 iters){
    __shared__ __align__(128) u32 X[4*2560];
    constexpr u32 XD=320u,XS=2560u,RR=NT==512?2u:4u,NCOL=NT==512?2u:4u;
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,gid=lane>>2,tig=lane&3u,rl=8u*warp+gid;
    u32 res[RR][8];
    #pragma unroll
    for(u32 rr=0;rr<RR;++rr)
        #pragma unroll
        for(u32 j=0;j<8;++j)res[rr][j]=(threadIdx.x*(rr*8+j+3)+blockIdx.x)&0x3fffffffu;
    for(u32 i=threadIdx.x;i<4*XS;i+=NT)X[i]=(i*2654435761u)&0x3fffffffu;
    u32 af[AFR?AFR:1];   // round 11 (p1o): AFR live registers held across the quad loop (production keeps af[8][2][4] = 64)
    #pragma unroll
    for(u32 k=0;k<(AFR?AFR:1);++k)af[k]=threadIdx.x*(2*k+1)+k;
    const u32 pA=H?PQH[tig]:PQP[tig];const u64* const stA=st+tig*256u;const u64* const twA=tw+tig*641u;
    const u32 qB=NT==512?(warp>>2):(warp>>1),pB=H?PQH[qB]:PQP[qB];const u64* const stB=st+qB*256u;
    u32* const xb=NT==512?X+qB*XS+2u*(warp&3u)*XD:X+qB*XS+4u*(warp&1u)*XD;
    __syncthreads();
    const u64 t0=clock64();
    #pragma unroll 1
    for(u32 it=0;it<iters;++it){
        const u32 c=(blockIdx.x*7u+it)&255u;
        if constexpr(AFR!=0){   // consumed once per quad (as the MMAs do), opaque so they stay live in registers
            #pragma unroll
            for(u32 k=0;k<AFR;++k)asm volatile("":"+r"(af[k]));
            #pragma unroll
            for(u32 k=0;k<AFR/2;++k){const u32 rr=(k>>3)%RR,j=k&7u;res[rr][j]^=af[k]^af[k+AFR/2];}}
        if(MODE&1){
            #pragma unroll
            for(u32 rr=0;rr<RR;++rr){const u32 row=(256u/RR)*rr+rl;
                const u64 t2=twA[1+128u+((row*256u+c)>>7)];
                #pragma unroll
                for(u32 j=0;j<8;++j)res[rr][j]=qmsh(res[rr][j],t2,pA);
                qjdif<H>(res[rr],stA,pA);}
            if(NT==256){const u64 wa=stA[rl],wb=stA[64u+rl],wc=stA[128u+rl];
                #pragma unroll
                for(u32 j=0;j<8;++j){qbf(res[0][j],res[2%RR][j],wa,pA);qbf(res[1][j],res[3%RR][j],wb,pA);qbf(res[0][j],res[1][j],wc,pA);qbf(res[2%RR][j],res[3%RR][j],wc,pA);}}
            else{const u64 wa=stA[rl];
                #pragma unroll
                for(u32 j=0;j<8;++j)qbf(res[0][j],res[1][j],wa,pA);}
            #pragma unroll
            for(u32 rr=0;rr<RR;++rr){const u32 row=(256u/RR)*rr+rl;
                #pragma unroll
                for(u32 j=0;j<8;++j)X[tig*XS+j*XD+8*tig+row+4*(row>>5)]=res[rr][j];}
        }
        if(MODE&4)__syncthreads();else __syncwarp();
        if(MODE&2){const u64* const ct=ctab+(size_t(qB)*256u+c)*256u;qrdif6q<NCOL,H>(xb,8u*qB,stB,ct,pB,lane);}
        if(MODE&4)__syncthreads();else __syncwarp();
        #pragma unroll
        for(u32 rr=0;rr<RR;++rr)
            #pragma unroll
            for(u32 j=0;j<8;++j)res[rr][j]^=it;
    }
    const u64 t1=clock64();
    u32 a=X[threadIdx.x];
    #pragma unroll
    for(u32 rr=0;rr<RR;++rr)
        #pragma unroll
        for(u32 j=0;j<8;++j)a^=res[rr][j];
    out[blockIdx.x*NT+threadIdx.x]=a;if(threadIdx.x==0)cyc[blockIdx.x]=t1-t0;
}
#undef qbf
#undef qbf1
#undef qmsh
#define PQK(NAME,MODE,NT,H) extern "C" __global__ void __launch_bounds__(NT,2) NAME(u32* out,u64* cyc,const u64* st,const u64* tw,const u64* ctab,u32 iters){p1q_body<MODE,NT,H>(out,cyc,st,tw,ctab,iters);}
#define PQKR(NAME,MODE,NT,H,AFR) extern "C" __global__ void __launch_bounds__(NT,2) NAME(u32* out,u64* cyc,const u64* st,const u64* tw,const u64* ctab,u32 iters){p1q_body<MODE,NT,H,AFR>(out,cyc,st,tw,ctab,iters);}
PQK(pq_ab,7,256,false) PQK(pq_a,5,256,false) PQK(pq_b,6,256,false) PQK(pq_ab_nobar,3,256,false) PQK(pq_ab512,7,512,false)
PQK(pq_ab_h,7,256,true) PQK(pq_a_h,5,256,true) PQK(pq_b_h,6,256,true)
PQKR(pq_ab_r32,7,256,false,32) PQKR(pq_ab_r64,7,256,false,64) PQKR(pq_ab_h_r64,7,256,true,64) PQKR(pq_ab512_r32,7,512,false,32)
)PQ";

inline bool p1q_env(){const char* e=std::getenv("FHERMA_DIAG_P1Q");return !e||*e!='0';}

template<class D> inline void run_p1q(D& g){
    auto& d=g.d;auto& kp=g.kp;const int sms=g.sms;
    CUmodule mod=nullptr;CUdeviceptr o=0,c=0,st=0,tw=0,ct=0;
    try{
        int cc1=0,cc2=0;if(kp.dattr){kp.dattr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,0);kp.dattr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,0);}
        nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,kDiagP1qSource,"diag_p1q.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("p1q nvrtc create");
        auto gsz=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,size_t*)>("nvrtcGetCUBINSize");auto gcub=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,char*)>("nvrtcGetCUBIN");
        std::vector<char> img;bool native=false;
        {std::string a="--gpu-architecture=sm_"+std::to_string(cc1*10+cc2);const char* op[]={a.c_str(),"--std=c++17"};
         if(cc1>=8&&d.nvrtcCompileProgram(pr,2,op)==NVRTC_SUCCESS){size_t n=0;if(gsz(pr,&n)==NVRTC_SUCCESS&&n){img.resize(n);if(gcub(pr,img.data())==NVRTC_SUCCESS)native=true;else img.clear();}}}
        if(!native){d.nvrtcDestroyProgram(&pr);pr=nullptr;d.nvrtcCreateProgram(&pr,kDiagP1qSource,"diag_p1q.cu",0,nullptr,nullptr);
            const char* op[]={"--gpu-architecture=compute_80","--std=c++17"};
            if(d.nvrtcCompileProgram(pr,2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);throw std::runtime_error("p1q nvrtc: "+l);}
            size_t n=0;d.nvrtcGetPTXSize(pr,&n);img.resize(n);d.nvrtcGetPTX(pr,img.data());}
        d.nvrtcDestroyProgram(&pr);
        d.check(d.cuModuleLoadData(&mod,img.data()));
        // tables: Shoup pairs (w' << 32 | w), w < p, for the 4 replica moduli: stage 4x256, twist 4x641, cross 4x256x256 (2 MB)
        static const unsigned P[2][4]={{2013265921u,2113929217u,2130706433u,1811939329u},{1073479681u,1071513601u,1069547521u,1068236801u}};   // = PQP / PQH
        unsigned long long x=0x9e3779b97f4a7c15ull;
        auto mk=[&](size_t n,size_t per){std::vector<unsigned long long> v(2*n);for(size_t i=0;i<2*n;++i){const unsigned p=P[i/n][((i%n)/per)&3];x^=x<<13;x^=x>>7;x^=x<<17;
            const unsigned w=unsigned(x%p);v[i]=((((unsigned long long)w<<32)/p)<<32)|w;}return v;};   // [set][...]: set 1 = Harvey moduli
        const size_t nst=4*256,ntw=4*641,nct=size_t(4)*256*256;
        const auto hst=mk(nst,256),htw=mk(ntw,641),hct=mk(nct,256*256);
        d.check(d.cuMemAlloc(&st,hst.size()*8));d.check(d.cuMemcpyHtoD(st,hst.data(),hst.size()*8));
        d.check(d.cuMemAlloc(&tw,htw.size()*8));d.check(d.cuMemcpyHtoD(tw,htw.data(),htw.size()*8));
        d.check(d.cuMemAlloc(&ct,hct.size()*8));d.check(d.cuMemcpyHtoD(ct,hct.data(),hct.size()*8));
        d.check(d.cuMemAlloc(&o,size_t(sms)*2*512*4));d.check(d.cuMemAlloc(&c,size_t(sms)*2*8));
        std::printf("{\"diag\":\"p1q_info\",\"sms\":%d,\"cc\":\"%d.%d\",\"native_cubin\":%s}\n",sms,cc1,cc2,native?"true":"false");std::fflush(stdout);
        struct K{const char* fn;const char* name;unsigned nt,bps;double work,full;};   // work: butterflies per quad executed; full: of the production quad part
        static const K ks[]={{"pq_ab","ab",256,2,45056,45056},{"pq_ab","ab_b1",256,1,45056,45056},{"pq_a","a",256,2,20480,20480},{"pq_b","b",256,2,24576,24576},
            {"pq_ab_nobar","ab_nobar",256,2,45056,45056},{"pq_ab512","ab512",512,2,40960,45056},
            {"pq_ab_h","ab_h",256,2,45056,45056},{"pq_a_h","a_h",256,2,20480,20480},{"pq_b_h","b_h",256,2,24576,24576},
            {"pq_ab_r32","ab_r32",256,2,45056,45056},{"pq_ab_r64","ab_r64",256,2,45056,45056},{"pq_ab_h_r64","ab_h_r64",256,2,45056,45056},{"pq_ab512_r32","ab512_r32",512,2,40960,45056}};
        const unsigned iters=64;auto fga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
        for(const auto& k:ks){
            if(std::chrono::duration<double>(std::chrono::steady_clock::now()-g.t0).count()>8.0){std::printf("{\"diag\":\"p1q\",\"k\":\"%s\",\"skipped\":\"time budget\"}\n",k.name);continue;}
            CUfunction fn=nullptr;if(d.cuModuleGetFunction(&fn,mod,k.fn)!=CUDA_SUCCESS){std::printf("{\"diag\":\"p1q\",\"k\":\"%s\",\"skipped\":\"no kernel\"}\n",k.name);continue;}
            int regs=0,lmem=0;if(fga){fga(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,fn);fga(&lmem,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fn);}
            const unsigned grid=unsigned(sms)*k.bps;unsigned long long best=~0ull;double mhz=0,bms=1e30;
            for(int r=0;r<5;++r){
                const bool hv=std::strstr(k.fn,"_h")!=nullptr;CUdeviceptr st1=st+(hv?nst*8:0),tw1=tw+(hv?ntw*8:0),ct1=ct+(hv?nct*8:0);
                unsigned it=iters;void* pp[]={&o,&c,&st1,&tw1,&ct1,&it};
                kp.erecord(g.ev0,g.s[0]);d.check(d.cuLaunchKernel(fn,grid,1,1,k.nt,1,1,0,g.s[0],pp,nullptr));kp.erecord(g.ev1,g.s[0]);d.check(g.esync(g.ev1));
                std::vector<unsigned long long> h(grid);d.check(d.cuMemcpyDtoH(h.data(),c,size_t(grid)*8));unsigned long long mx=0;for(auto v:h)mx=std::max(mx,v);
                if(r==0||mx==0)continue;const double ms=kp.ms(g.ev0,g.ev1);if(mx<best){best=mx;bms=ms;}if(ms>0)mhz=std::max(mhz,double(mx)/ms*1e-3);}
            if(best==~0ull){std::printf("{\"diag\":\"p1q\",\"k\":\"%s\",\"skipped\":\"no cycles\"}\n",k.name);continue;}
            const double rate=double(k.bps)*iters*k.work/double(best);   // grid = one co-resident wave: cycles = max over blocks   // butterflies per clock per SM (all bps blocks co-resident)
            const double us=mhz>0?2048.0*k.full/(rate*sms*mhz):0;      // whole product (2048 quads) on all SMs
            std::printf("{\"diag\":\"p1q\",\"k\":\"%s\",\"bfly_per_sm_clk\":%.2f,\"us_per_product\":%.1f,\"cyc_per_quad_blk\":%.0f,\"r\":%d,\"lm\":%d,\"ms\":%.4f,\"clk_mhz\":%.0f}\n",
                k.name,rate,us,double(best)/iters,regs,lmem,bms,mhz);std::fflush(stdout);}
    }catch(const std::exception& e){std::printf("{\"diag\":\"p1q_error\",\"what\":\"%.100s\"}\n",e.what());std::fflush(stdout);}
    d.cuCtxSynchronize();for(auto p:{o,c,st,tw,ct})if(p)d.cuMemFree(p);if(mod)d.cuModuleUnload(mod);
}
