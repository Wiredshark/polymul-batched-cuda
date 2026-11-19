// ---- Round 21 (clean, exit-time combo sweep only; appended to the sweep copy of the source, never to the production one) ----
// p2k techniques (pv65: fewer address / constant instructions, fewer registers -> better co-residency) applied to P3, CRT, P1.
// sm_90 SASS instructions (static, NOPs excluded) / JIT regs on cc 12.0 (local RTX 5070, compute_90 PTX):
//   P3  bpo_last_cm3 1423 / 40 -> p3a 1164 / 40, p3b 1198 / 32, p3h 1178 / 40, p3hb 1202 / 32 (no local memory)
//   CRT bpo_crt_lean 3230 / 168 -> crtn 3116 / 168, crt4 3247 / 128 (8 B local: reference only), crtn4 3108 / 128 (0 local).
//       CRT static smem 32.8 KB/block: cc 12.0 caps smem at 100 KB/SM, so 3 blocks/SM stay the limit; 128 regs only help
//       co-residency with the other stages' blocks.
//   P1  bpo_first_x 3068 / 128 -> p1n 3065, p1na 2294 (fallback LDG path gone; executed path ~unchanged), p1nr 4569 (two bodies)
// Same canonical outputs as production (bit-identical batch outputs, device compare ex=1). The host launches these kernels with
// production's arguments; they assume N = 65536 (the only size the batch graph runs them at; a wrong N would show as ex 0).
//   P3 (bpo_last_cm3 = bpo_p3_body<2308>, 256 threads, 40 regs):
//     cl_p3a   N / logR compile-time, 2p from a lane-0 shuffle, C rows / X / t2n / D as base + immediates, one t2n layout
//              (production's dfl 2 old layout is never set in the batch graph), t2n hoisted before the R-DIT as production
//     cl_p3b   cl_p3a with __maxnreg__(32) (launch bounds cannot ask for it: 8 x 256 threads > 1536) (P3 blocks pack 4 instead of 3 next to a P1 block)
//     cl_p3h   cl_p3a with half the t2n words hoisted (m < 4 before the R-DIT, m >= 4 after it): fewer live registers
//     cl_p3hb  cl_p3h with __maxnreg__(32)
//   CRT (bpo_crt_lean, 128 threads, 168 regs, 3 blocks/SM):
//     cl_crtn  N = 65536 compile-time (prefetch / discard / residue addresses as immediates), digit sums loaded per half
//     cl_crt4  production body with __launch_bounds__(128,4) (128 registers: 4 blocks/SM)
//     cl_crtn4 cl_crtn with __launch_bounds__(128,4)
//   P1 (bpo_first_x, 256 threads, 128 regs, 2 blocks/SM; algorithm unchanged):
//     cl_p1n   N = 65536 compile-time (twist stride, output plane addresses)
//     cl_p1na  cl_p1n + XG_ALN (one alignment path: cp.async staging; the batch operands are 16 B aligned)
//     cl_p1nr  cl_p1na + r0 = 0, r1 = 16 compile-time when the launch passes them (uniform branch to the generic body otherwise)
#ifndef CL_SWEEP
#define CL_SWEEP 1
__device__ __forceinline__ u32 cl_p2(u32 p){return __shfl_sync(0xffffffffu,2u*p,0);}   // opaque 2p (no IMAD p*-2 re-folding)
// ---------------- P3 ----------------
// V bits: 1 half t2n hoist (else full hoist as production BPO_P3T 2048)
template<bool H,u32 V>__device__ __forceinline__ void cl_p3_body(u32* X,const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 dfl){
    constexpr u32 N=65536u,logR=8u;
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 t=threadIdx.x,lane=t&31u,warp=t>>5;
    const u32 p2=H?cl_p2(p):0u;(void)p2;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* sit=stage_itw+prime*256u;
    u32 yv[8];
    {const u32* src=C+u64(prime*8u)*N+(c*256u+t);
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=src[d*N];}
    u64 F;{
        const u32 k1=__brev(t)>>(32-logR);
        const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
        const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
        const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
        F=u64(wv)|(u64(q)<<32);}
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
                for(u32 jj=0;jj<half;++jj){if(jj==0)phdit1(yv[d0],yv[d0+half],p2);else phdit(yv[d0+jj],yv[d0+jj+half],sit[jj*(256u/len)],p,p2);}}}
    }else{
        #pragma unroll
        for(u32 d0=0;d0<8;d0+=2)pzdit1c(yv[d0],yv[d0+1],p);
        #pragma unroll
        for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=len){
                #pragma unroll
                for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(yv[d0],yv[d0+half],sit[0],p);else pzdit(yv[d0+jj],yv[d0+jj+half],sit[jj*(256u/len)],p);}}}
    }
    {u32* const xw=X+(t+4u*(t>>5));
        #pragma unroll
        for(u32 d=0;d<8;++d)xw[d*288u]=yv[d];}
    __syncthreads();
    const u64* const tn=t2n+(u64(prime*8u+warp)*512u+((c>>7)*256u+lane));
    constexpr u32 NH=(V&1)?4u:8u;
    u64 tv[8];
    #pragma unroll
    for(u32 m=0;m<NH;++m)tv[m]=__ldg(tn+32u*m);
    if((dfl&1)&&lane<8u)bpo_discard(C+u64(prime*8u+lane)*N+(c*256u+32u*warp));   // C lines read by this warp (all used)
    u32 v[8];
    u32* const base=X+warp*288u;
    {const uint4* p4=reinterpret_cast<const uint4*>(base+(8u*lane+4u*(lane>>2)));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    if(H)phdit_smr<false>(base,v,sit,p,p2,lane);else pzdit_smr<false>(base,v,sit,p,lane);
    #pragma unroll
    for(u32 m=NH;m<8;++m)tv[m]=__ldg(tn+32u*m);
    u32* const out=D+(u64(prime*8u+warp)*N+(c*256u+lane));
    if(dfl&12){   // dfl 8: group-0 residues L2 evict_last (BPO_P3EL); dfl 4: evict_first
        u64 pol;if(dfl&8)asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=pxmsh(v[m],tv[m],p);asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+32u*m),"r"(r),"l"(pol):"memory");}
    }else{
        #pragma unroll
        for(u32 m=0;m<8;++m)out[32u*m]=pxmsh(v[m],tv[m],p);
    }
}
#define CL_P3(NAME,V,BOUNDS)                                                                                            \
extern "C" __global__ void BOUNDS NAME(const u32* C,u32* D,const u32* primes,                     \
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){                                    \
    __shared__ __align__(128) u32 X[8*288];(void)N;(void)logR;                                                         \
    if(BPO_NH30&&primes[blockIdx.y]<(1u<<30))cl_p3_body<(BPO_NH30!=0),V>(X,C,D,primes,untwist,stage_itw,t2n,dfl);else \
    cl_p3_body<false,V>(X,C,D,primes,untwist,stage_itw,t2n,dfl);}
CL_P3(cl_p3a,0,__launch_bounds__(256))
CL_P3(cl_p3b,0,__maxnreg__(32))
CL_P3(cl_p3h,1,__launch_bounds__(256))
CL_P3(cl_p3hb,1,__maxnreg__(32))
// ---------------- CRT ----------------
// SPLIT: digit column sums loaded and chained per half (even sums, then odd sums): 16 fewer live words in the digit stage
template<u32 J,bool SPLIT> __device__ __forceinline__ void cl_digit(u32 (&acc)[66],const u32* x,u32 alpha){
    if constexpr(!SPLIT){swc_digit<J,true>(acc,x,alpha);return;}
    else{
    constexpr u32 o=SWC_POS[J]/32,sh=SWC_POS[J]%32;
    u32 xe[16];
    #pragma unroll
    for(u32 i=0;i<16;++i)xe[i]=x[2*i]+(alpha*swc_np16(2*i)+swc_of16(2*i));
    swc_mac<(1u<<sh)>(acc+o,xe);
    #pragma unroll
    for(u32 i=0;i<16;++i)xe[i]=x[2*i+1]+(alpha*swc_np16(2*i+1)+swc_of16(2*i+1));
    if(sh<16)swc_mac<(1u<<((sh+16)&31))>(acc+o,xe);else swc_mac<(1u<<((sh+16)&31))>(acc+o+1,xe);
    i64 cy=0;
    #pragma unroll
    for(u32 w=16;w<=18;++w){
        const u32 ex=w==16?(alpha<<sh):(w==17&&sh?(alpha>>((32-sh)&31)):0u);
        const i64 tt=i64(acc[o+w])-i64(ex)+cy;acc[o+w]=u32(tt);cy=tt>>32;}
    }
}
template<bool NC,bool SPLIT> __device__ __forceinline__ void cl_crt_body(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 Nr,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    const u32 N=NC?65536u:Nr;
    if(pfitems){
        const u32 total=gridDim.x*blockDim.x;
        for(u32 t=blockIdx.x*blockDim.x+threadIdx.x;t<pfitems*256u;t+=total){
            const u32 it=t>>8,hh=t&255u;const u32* ps=(it>=256u?pfb:pfa)+u64(it&255u)*BCRT_L+u64(hh*256u)*BCRT_L;
            asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+BCRT_L-1));
        }
    }
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F;
    static_assert(K==16&&SW==16,"tensor CRT geometry");
    constexpr u32 WO=W/32,WS=W%32;
    constexpr u32 STW=32*L;
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    u32* const pbuf=st+32*SST;
    // NC: per-thread source base (rows r = (lane>>3)+4k: k*32N words apart) and smem destinations as base + immediates
    const u32 r_=lane>>3,ch=lane&7u;
    const u32* const rb=residues+(u64(r_)*8u*N+(i0+4u*ch));
    const u32 sd0=u32(__cvta_generic_to_shared(pbuf))+4u*(r_*32u+((4u*ch)^(8u*(r_&3u))));
    auto prefetch=[&](u32 j){
        if(NC){
            #pragma unroll
            for(u32 k=0;k<4;++k)swc_cpa16(sd0+k*512u,rb+(u64(k)*32u*N+j*N));   // r+4k: same (r&3) swizzle, +4*32 words
        }else{
            #pragma unroll
            for(u32 k=0;k<4;++k){const u32 q=lane+32*k,r=q>>3,c8=q&7;
                swc_cpa16(u32(__cvta_generic_to_shared(pbuf+r*32+((4*c8)^(8*(r&3))))),residues+(u64(r)*8+j)*N+i0+4*c8);}
        }
        swc_commit();};
    prefetch(0);
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    __syncthreads();
    u32 acc[NW+2];
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 a[2][2][4];
        swc_wait0();__syncwarp();
        {   const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(32*tig+gid)*4;
            const u32 p0=pb+32*(0^tig),p1=pb+32*(1^tig),p2=pb+32*(2^tig),p3=pb+32*(3^tig);
#define SWC_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*128>(q0);a[m][s][1]=swc_lds<(8*s)*128>(q1);a[m][s][2]=swc_lds<(8*s+4)*128>(q0);a[m][s][3]=swc_lds<(8*s+4)*128>(q1);
            SWC_FR(0,0,p0,p1) SWC_FR(0,1,p0,p1) SWC_FR(1,0,p2,p3) SWC_FR(1,1,p2,p3)
#undef SWC_FR
        }
        __syncwarp();
        if(j<7)prefetch(j+1);
        {const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
            swc_tile<0>(a,bsh,lane,sb);swc_tile<1>(a,bsh,lane,sb);swc_tile<2>(a,bsh,lane,sb);swc_tile<3>(a,bsh,lane,sb);swc_tile<4>(a,bsh,lane,sb);
            swc_tile<5>(a,bsh,lane,sb);swc_tile<6>(a,bsh,lane,sb);swc_tile<7>(a,bsh,lane,sb);swc_tile<8>(a,bsh,lane,sb);}
        __syncwarp();
        const u32* hs=st+lane*SST;
        const u64 S=u64(hs[32])+(u64(hs[33])<<16)+(u64(hs[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        switch(j){case 0:cl_digit<0,SPLIT>(acc,hs,alpha);break;case 1:cl_digit<1,SPLIT>(acc,hs,alpha);break;case 2:cl_digit<2,SPLIT>(acc,hs,alpha);break;case 3:cl_digit<3,SPLIT>(acc,hs,alpha);break;
                  case 4:cl_digit<4,SPLIT>(acc,hs,alpha);break;case 5:cl_digit<5,SPLIT>(acc,hs,alpha);break;case 6:cl_digit<6,SPLIT>(acc,hs,alpha);break;default:cl_digit<7,SPLIT>(acc,hs,alpha);break;}
        __syncwarp();
    }
    {   u32 hw[10];
        #pragma unroll
        for(u32 w=0;w<10;++w){const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;hw[w]=(a_>>WS)|(b_<<(32-WS));}
        acc[WO]&=(1u<<WS)-1;
        swc_fold(acc,hw,d);
        const u32 br=swc_subbias(acc);
        const bool hi=(acc[WO]>>WS)!=0u;
        acc[WO]&=(1u<<WS)-1;
        const u32 lt=swc_ltq(acc);
        swc_addx(acc,br?(0u-d):((hi||!lt)?d:0u),br);
        acc[WO]&=(1u<<WS)-1;
    }
    if(dfl&1){if(NC){const u32* const db=residues+(u64(lane)*N+i0);
            #pragma unroll
            for(u32 x=0;x<4;++x)bpo_discard(db+u64(x)*32u*N);}
        else for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);}
    const u32 cc_=i0>>8,h0=i0&255u;
    {   __syncwarp();
        #pragma unroll
        for(u32 w=0;w<L;++w)st[lane*L+w]=acc[w];
        __syncwarp();
        u32* const ob=output+u64(h0*256u+cc_)*L+lane;
        if(dfl&2){
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){__stcs(ob+rr*256u*L,st[rr*L+lane]);if(lane<L-32)__stcs(ob+rr*256u*L+32,st[rr*L+32+lane]);}
        }else{
            #pragma unroll
            for(u32 rr=0;rr<32;++rr){ob[rr*256u*L]=st[rr*L+lane];if(lane<L-32)ob[rr*256u*L+32]=st[rr*L+32+lane];}
        }
    }
}
#define CL_CRT(NAME,NC,SPLIT,MINB)                                                                                     \
extern "C" __global__ void __launch_bounds__(128,MINB) NAME(const u32* __restrict__ residues,u32* __restrict__ output, \
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){                         \
    cl_crt_body<NC,SPLIT>(residues,output,bfrag,N,d,pfa,pfb,pfitems,dfl);}
CL_CRT(cl_crt3,false,false,3)   // = production (control for the copy)
CL_CRT(cl_crtn,true,true,3)
CL_CRT(cl_crt4,false,false,4)
CL_CRT(cl_crtn4,true,true,4)
// ---------------- P1 ----------------
#define CL_P1(NAME,G,RC)                                                                                                \
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) NAME(const u32* a,const u32* b,u32* A,u32* B,\
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,\
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){                                               \
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];(void)N;(void)logR;                                           \
    if(RC&&r0==0u&&r1==16u)bpo_first_xb<BPO_XF,(G)>(a,b,A,B,twist,stage_tw,bpt,65536u,0u,16u,pfstep,ctab,&tmA,&tmB,X);else \
    bpo_first_xb<BPO_XF,(G)>(a,b,A,B,twist,stage_tw,bpt,65536u,r0,r1,pfstep,ctab,&tmA,&tmB,X);}
CL_P1(cl_p1n,0u,false)
CL_P1(cl_p1na,16u,false)
CL_P1(cl_p1nr,16u,true)
#endif
