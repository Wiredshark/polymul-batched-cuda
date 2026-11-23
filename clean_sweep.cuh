// ---- Round 23 (kill, exit-time combo sweep only; appended to the sweep copy of the source, never to the production one) ----
// Kill-test ablations on 3140af3 (wrong values by design, timing only; launched with production's arguments, grid and thread
// count, never compared for exactness):
//   cl_k1ns  P1 bpo_first_xb<BPO_XF,BPO_XG,1>: pbq phase B without the layout-1 len-16 stage (one butterfly stage fewer)
//   cl_k1nt  P1 K 2: no dense TMA staging copy and no per-pair __syncwarp (layout-2 results stored in place; same TMA bytes)
//   cl_k1nb  P1 K 3: dense staging kept, only its per-pair __syncwarp removed
//   cl_k2x   P2 bpo_r16_body<H,1>: one extra dummy radix-2 stage on both forward operands (16 butterflies + 1 twiddle load / thread)
//   cl_k3x   P3 bpo_p3b_body<H,1> at __maxnreg__(32): one extra dummy radix-2 stage after the J-DIT (4 butterflies + 1 load)
#ifndef CL_SWEEP
#define CL_SWEEP 1
#define CL_P1_SIG(NAME) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,\
    u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB)
#define CL_K1(NAME,K) extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) CL_P1_SIG(NAME){\
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];(void)logR;\
    bpo_first_xb<BPO_XF,BPO_XG,K>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);}
CL_K1(cl_k1ns,1u)
CL_K1(cl_k1nt,2u)
CL_K1(cl_k1nb,3u)
#ifndef BPO_P2_OLD
extern "C" __global__ void __launch_bounds__(128) cl_k2x(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw){
    __shared__ u32 TA[8*292];
#if BPO_NH30
    if(primes[blockIdx.z]<(1u<<30))bpo_r16_body<true,1u>(TA,A,B,C,primes,r16,N,untw);else
#endif
    bpo_r16_body<false,1u>(TA,A,B,C,primes,r16,N,untw);
}
#endif
#ifdef BPO_P3B
extern "C" __global__ void __maxnreg__(32) cl_k3x(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;
    if(N==65536u&&logR==8u&&!(dfl&2u)){
#if BPO_NH30
        if(primes[blockIdx.y]<(1u<<30))bpo_p3b_body<true,1u>(X,C,D,primes,untwist,stage_itw,t2n,dfl);else
#endif
        bpo_p3b_body<false,1u>(X,C,D,primes,untwist,stage_itw,t2n,dfl);
        return;}
#if BPO_NH30
    if(primes[blockIdx.y]<(1u<<30))bpo_p3_body<BPO_P3T,true>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);else
#endif
    bpo_p3_body<BPO_P3T>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);
}
#endif
#endif
