// ---- Round 22 (p1kd, exit-time combo sweep only; appended to the sweep copy of the source, never to the production one) ----
// P1 register-bound lines on 8175f0a (production P1 = p1k body bpo_p1k_body<0> at __launch_bounds__(256,2), 128 regs):
//   cl_p1old   old P1 body bpo_first_xb<BPO_XF,BPO_XG> at __launch_bounds__(256,2) (= FHERMA_P1_OLD=1 / 3140af3 P1)
//   cl_p1k120  p1k body, __maxnreg__(120), bpt load chain D = BPO_P1K_D (0 spill on sm_120; no extra co-residency slot)
//   cl_p1k112  p1k body, __maxnreg__(112), bpt load chain D (40 B local on sm_120 ptxas; the only cap that frees an 8K-register
//              slot = one P2 (56 x 128) or P3 (32 x 256) block beside 2 P1 blocks; P1 smem 40 KB keeps P1 at 2 blocks/SM)
// Same canonical outputs as production (device compare ex=1). The host launches these with production's arguments and thread
// count (p1_threads of the production kernel = 256); the p1k lines assume N = 65536 (trap otherwise).
#ifndef CL_SWEEP
#define CL_SWEEP 1
#define CL_P1_SIG(NAME) NAME(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,\
    u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB)
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) CL_P1_SIG(cl_p1old){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];(void)logR;
    bpo_first_xb<BPO_XF,BPO_XG>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);}
#ifdef BPO_P1K
#define CL_P1K(NAME,R)                                                                                                 \
extern "C" __global__ void __maxnreg__(R) CL_P1_SIG(NAME){                                                             \
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];(void)A;(void)B;(void)logR;                                   \
    if(N!=65536u)__trap();                                                                                              \
    bpo_p1k_body<BPO_P1K_D>(a,b,twist,stage_tw,bpt,r0,r1,pfstep,ctab,&tmA,&tmB,X);}
CL_P1K(cl_p1k120,120)
CL_P1K(cl_p1k112,112)
#endif
#endif
