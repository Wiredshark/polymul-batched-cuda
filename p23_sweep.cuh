// ---- Round 20 (p23, exit-time combo sweep only; appended to the sweep copy of the source, never to the production one) ----
// P2 / P3 variants with the same canonical residues as production (bit-identical batch outputs). sm_90 SASS per thread
// (both prime paths): bpo_middle_r16 3752 -> cb_p2k 3320 (H path ~-11%), 64 -> 56 regs; bpo_last_cm3 1440 -> cb_p3k 1344;
// cb_p3r16h 2112 for 16 words/thread (block: 4 warps x ~1060 vs cm3 8 warps x ~700 -> ~-24% warp instructions per block).
//   cb_p2k     bpo_middle_r16 rewritten for addressing: smem transposes as base pointer + immediate (production computes a
//              shared address per element: IADD3 + LEA each, 16 address registers), A/B/C rows as base + immediate, middle
//              twiddles as one IMAD.WIDE.U32 each, 2p from a lane-0 shuffle (bpo_p2_of = ~50 uniform instructions)
//   cb_p3k     bpo_last_cm3 with N = 65536, logR = 8 compile-time and 2p from a shuffle (module-local text substitution)
//   cb_p3r16h  P3 radix-16, 128 threads x 16 words (bpo_last_r16 structure) + Harvey-lazy J-DIT / DIT16 for p < 2^30,
//   cb_p3r16t  production L2 behaviour (dfl 1 discard of the read C lines, dfl 8 group-0 residues evict_last, dfl 4 evict_first);
//              r16t (TNH): t2n Shoup words loaded right after the J phase (56 regs) instead of before the stores (r16h, 64 regs)
#ifndef P23_SWEEP
#define P23_SWEEP 1
// opaque pointer (twiddle tables only): the per-entry address is rebuilt as one IMAD.WIDE from it, not re-folded into the index
template<class T>__device__ __forceinline__ T* p23_opq(T* p){asm("mov.b64 %0, %0;":"+l"(p));return p;}
// 2p in a plain register (bpo_p2_of = 16-way constant lookup, ~50 uniform instructions per thread; the shuffle keeps ptxas from
// folding 2p into IMAD p*-2 + s per correction, which is why production reads it from constant memory)
__device__ __forceinline__ u32 p23_p2(u32 p){return __shfl_sync(0xffffffffu,2u*p,0);}   // opaque to ptxas (a mov is not)
// Shoup word at byte offset u64(s8) * k from tb (one IMAD.WIDE.U32)
__device__ __forceinline__ u64 p23_tw(const char* tb,u32 s8,u32 k){return __ldg(reinterpret_cast<const u64*>(tb+u64(s8)*k));}
#ifndef P23_P2OF
#define P23_P2OF p23_p2
#endif
template<bool H>__device__ __forceinline__ void p23_r16_body(u32* TA,const u32* A,const u32* B,u32* C,const u32* primes,const u64* r16,u32 untw){
    constexpr u32 N=65536u;
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,s=lane&7u,c2=4u*warp+(lane>>3);
    const u64* tf=r16+u64(prime)*512u;const u64* ti=tf+256u;
    const char* tfb=reinterpret_cast<const char*>(p23_opq(tf));const char* tib=reinterpret_cast<const char*>(p23_opq(ti));
    const u64 base=u64(prime*8+j)*N;
    const u32 off=g*2048u+c2*8u+s;
    u32 a[16],b[16];
    {   const u32* a0=A+base+off;const u32* b0=B+base+off;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1){if(untw&4){a[c1]=__ldcs(a0+128u*c1);b[c1]=__ldcs(b0+128u*c1);}else{a[c1]=a0[128u*c1];b[c1]=b0[128u*c1];}}
    }
    const u32 p2=H?P23_P2OF(p):0u;(void)p2;
    r16_dif16h<H>(a,tf,p,p2);r16_dif16h<H>(b,tf,p,p2);
    u32* const tw_=TA+int(s*292u+c2);        // [17 i]: column c2 of row i
    u32* const tr_=TA+int(s*292u+17u*c2);    // [i]: row c2
    #pragma unroll
    for(u32 i=1;i<16;++i){const u64 w=p23_tw(tfb,8u*c2,r16_brev4(i));a[i]=pxmshh<H>(a[i],w,p);b[i]=pxmshh<H>(b[i],w,p);}
    #pragma unroll
    for(u32 i=0;i<16;++i)tw_[17*i]=a[i];
    __syncthreads();
    if(untw&8)bpo_discard((threadIdx.x<64u?A:B)+base+g*2048u+(threadIdx.x&63u)*32u);
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=tr_[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)tw_[17*i]=b[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)b[i]=tr_[i];
    r16_dif16h<H>(a,tf,p,p2);r16_dif16h<H>(b,tf,p,p2);
    {const u32 pinv=pxinv32(p);
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=pxmont(a[i],b[i],p,pinv);}
    r16_dit16h<H,true>(a,ti,p,p2);
    {const u32 k8=8u*r16_brev4(c2);
        #pragma unroll
        for(u32 i=1;i<16;++i)a[i]=pxmshh<H>(a[i],p23_tw(tib,k8,i),p);
        a[0]=H?pxadd(a[0],0u,p2):pzred(a[0],p);}
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)tr_[i]=a[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=tw_[17*i];
    r16_dit16h<H,true>(a,ti,p,p2);
    u32* o=C+base+(c2*256u+g*8u+s);
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)o[4096u*c1]=a[c1];
}
extern "C" __global__ void __launch_bounds__(128) cb_p2k(const u32* A,const u32* B,u32* C,const u32* primes,const u64* r16,u32 N,u32 untw){
    __shared__ u32 TA[8*292];(void)N;
#if BPO_NH30
    if(primes[blockIdx.z]<(1u<<30))p23_r16_body<true>(TA,A,B,C,primes,r16,untw);else
#endif
    p23_r16_body<false>(TA,A,B,C,primes,r16,untw);
}
extern "C" __global__ void __launch_bounds__(256) cb_p3k(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;(void)N;(void)logR;
#if BPO_NH30
    if(primes[blockIdx.y]<(1u<<30))bpo_p3_body<BPO_P3T,true>(X,&bar,C,D,primes,untwist,stage_itw,t2n,65536u,8u,dfl);else
#endif
    bpo_p3_body<BPO_P3T>(X,&bar,C,D,primes,untwist,stage_itw,t2n,65536u,8u,dfl);
}
// P3 radix-16 Harvey. Block (column c, prime), 128 threads. J phase: thread t owns slots t, t+128 (8 digit words each),
// F Shoup product (H: lazy, [0,2p)), 8-point J-DIT (H: phdit, [0,4p); else lazy pzdit, [0,2p)), tile X[d][sl + sl/16].
// R phase: thread (d = tid/16, a = tid%16): DIT16 over k2 (bit-reversed in), w^-(h2 k1), transpose, DIT16 over k1 ->
// h = 16 h1 + h2; D = Shoup(v, t2n) canonical (pxmsh accepts any u32).
template<bool H,bool TNH>__device__ __forceinline__ void p23_p3r16_body(u32* X,const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* r16,const u64* t2n,u32 dfl){
    constexpr u32 N=65536u,logR=8u;
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 tid=threadIdx.x;
    const u32 p2=H?P23_P2OF(p):0u;(void)p2;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* ti=r16+u64(prime)*512u+256u;
    {   const u32* src=C+u64(prime*8)*N+c*256u+tid;   // 8 digit rows N apart: immediates
        u32 y[2][8];
        #pragma unroll
        for(u32 hf=0;hf<2;++hf)
            #pragma unroll
            for(u32 d=0;d<8;++d)y[hf][d]=src[d*N+128u*hf];
        #pragma unroll
        for(u32 hf=0;hf<2;++hf){
            const u32 t=tid+128u*hf;
            u64 F;{const u32 k1=__brev(t)>>(32-logR);
                const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
                const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
                const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
                F=u64(wv)|(u64(q)<<32);}
            u32 (&yv)[8]=y[hf];
            #pragma unroll
            for(u32 d=0;d<8;++d)yv[d]=pxmshh<H>(yv[d],F,p);
            if(H){
                #pragma unroll
                for(u32 d0=0;d0<8;d0+=2)phdit1c(yv[d0],yv[d0+1],p2);
                #pragma unroll
                for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                    #pragma unroll
                    for(u32 d0=0;d0<8;d0+=len){
                        #pragma unroll
                        for(u32 jj=0;jj<half;++jj){if(jj==0)phdit1(yv[d0],yv[d0+half],p2);else phdit(yv[d0+jj],yv[d0+jj+half],ti[jj*(256u/len)],p,p2);}}}
            }else{
                #pragma unroll
                for(u32 d0=0;d0<8;d0+=2)pzdit1c(yv[d0],yv[d0+1],p);
                #pragma unroll
                for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                    #pragma unroll
                    for(u32 d0=0;d0<8;d0+=len){
                        #pragma unroll
                        for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(yv[d0],yv[d0+half],0,p);else pzdit(yv[d0+jj],yv[d0+jj+half],ti[jj*(256u/len)],p);}}}
            }
            u32* const xw=X+int(t+(t>>4));
            #pragma unroll
            for(u32 d=0;d<8;++d)xw[d*272u]=yv[d];
        }
    }
    __syncthreads();
    if((dfl&1)&&tid<64u)bpo_discard(C+u64(prime*8+(tid>>3))*N+c*256u+32u*(tid&7u));   // every C line of the block has been read
    const u32 d=tid>>4,a=tid&15u;
    const u64* tn=t2n+u64(prime*8+d)*512u+((c>>7)*256u+a);
    u64 tv[16];
    if(TNH){
        #pragma unroll
        for(u32 h1=0;h1<16;++h1)tv[h1]=__ldg(tn+16u*h1);}
    u32* const xr=X+int(d*272u+17u*a);   // [b]: row a of digit d
    u32* const xc=X+int(d*272u+a);       // [17 i]: column a
    u32 v[16];
    #pragma unroll
    for(u32 b=0;b<16;++b)v[b]=xr[b];
    r16_dit16h<H,false>(v,ti,p,p2);
    {const u32 k8=8u*r16_brev4(a);const char* tib=reinterpret_cast<const char*>(p23_opq(ti));
        #pragma unroll
        for(u32 i=1;i<16;++i)v[i]=pxmshh<H>(v[i],p23_tw(tib,k8,i),p);
        v[0]=H?pxadd(v[0],0u,p2):pzred(v[0],p);}
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)xr[i]=v[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)v[i]=xc[17u*i];
    r16_dit16h<H,true>(v,ti,p,p2);
    if(!TNH){
        #pragma unroll
        for(u32 h1=0;h1<16;++h1)tv[h1]=__ldg(tn+16u*h1);}
    u32* out=D+u64(prime*8+d)*N+(c*256u+a);
    if(dfl&12){
        u64 pol;if(dfl&8)asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 h1=0;h1<16;++h1){const u32 r=pxmsh(v[h1],tv[h1],p);asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+16u*h1),"r"(r),"l"(pol):"memory");}
    }else{
        #pragma unroll
        for(u32 h1=0;h1<16;++h1)out[16u*h1]=pxmsh(v[h1],tv[h1],p);
    }
}
#define P23_P3R16(NAME,TNH)                                                                                            \
extern "C" __global__ void __launch_bounds__(128) NAME(const u32* C,u32* D,const u32* primes,                          \
    const u64* untwist,const u64* r16,const u64* t2n,u32 N,u32 logR,u32 dfl){                                          \
    __shared__ __align__(16) u32 X[8*272];(void)N;(void)logR;                                                          \
    if(BPO_NH30&&primes[blockIdx.y]<(1u<<30))p23_p3r16_body<(BPO_NH30!=0),TNH>(X,C,D,primes,untwist,r16,t2n,dfl);else \
    p23_p3r16_body<false,TNH>(X,C,D,primes,untwist,r16,t2n,dfl);}
P23_P3R16(cb_p3r16h,false)
P23_P3R16(cb_p3r16t,true)
#endif
