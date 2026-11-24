// ---- Round 24 (sfix, stall fixes of STALL_PROFILE_20261123; appended to the combo sweep copy of the source, and to the
// production module only when FHERMA_SFIX is set; never part of the default production source) ----
// EXACT variants (same arithmetic, same values; only schedule / request shape changes), production launch geometry/arguments:
//   cs_crt_ov   CRT bpo_crt_lean with the IMMA tiles of digit j+1 software-pipelined under the carry chain of digit j: the
//               lane's h row of digit j is read into registers first, the (single) h stage is then free for tiles(j+1),
//               which are issued in three groups between the AIN adds / even / odd mac chains / final carries of digit j.
//               Same smem (no second stage: 3 blocks/SM kept), launch bounds (128,3).
//   cs_p2<TWB,ST>  P2 bpo_middle_r16 (p2k body), __launch_bounds__(128,8) (= production's 64 regs):
//               TWB: the 8 distinct stage twiddles t[16j] of the forward / inverse r16 tables and the block's 16 x 15 middle
//               twiddles (forward, then overwritten by the inverse ones after the first barrier) staged in smem (+2 KB),
//               cp.async-copied once per block (no registers held) instead of 56 + 30 per-warp 64-bit LDGs with 1-4 distinct addresses;
//               ST: output stores as STG.128: 4x4 transposes over the lane quads (4 shuffles each) -> 4 instead of 16
//               store requests per thread (each request writes 16 full 32 B sectors).
//   cs_p3t      P3 bpo_last_cm3 (p3b body) with the 256-entry stage_itw table of the prime staged in smem (+2 KB, one LDG.64 per
//               thread) for the smr R-DIT (29 per-thread 64-bit twiddle loads -> LDS); J-DIT twiddles unchanged; __maxnreg__(32).
#ifndef CS_SWEEP
#define CS_SWEEP 1
#ifdef BTC_F
// digit J (AIN form of swc_digit<J,true>) from the lane's h row already in registers; F0/F1/F2 run between the chains
template<u32 J,class F0,class F1,class F2> __device__ __forceinline__ void cs_digit(u32 (&acc)[66],u32 (&xe)[16],u32 (&xo)[16],u32 alpha,F0 f0,F1 f1,F2 f2){
    constexpr u32 o=SWC_POS[J]/32,sh=SWC_POS[J]%32;
    #pragma unroll
    for(u32 i=0;i<16;++i){xe[i]+=alpha*swc_np16(2*i)+swc_of16(2*i);xo[i]+=alpha*swc_np16(2*i+1)+swc_of16(2*i+1);}
    f0();
    swc_mac<(1u<<sh)>(acc+o,xe);
    f1();
    if(sh<16)swc_mac<(1u<<((sh+16)&31))>(acc+o,xo);else swc_mac<(1u<<((sh+16)&31))>(acc+o+1,xo);
    f2();
    i64 c=0;
    #pragma unroll
    for(u32 w=16;w<=18;++w){
        const u32 ex=w==16?(alpha<<sh):(w==17&&sh?(alpha>>((32-sh)&31)):0u);
        const i64 t=i64(acc[o+w])-i64(ex)+c;acc[o+w]=u32(t);c=t>>32;}
}
extern "C" __global__ void __launch_bounds__(128,3) cs_crt_ov(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
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
    static_assert(WO+1==L&&WS>0,"q = 2^W - d with W in the top limb");
    constexpr u32 STW=32*L;
    static_assert(32*SST+512<=STW,"prefetch buffer aliases the output stage");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    u32* const pbuf=st+32*SST;
    auto prefetch=[&](u32 j){
        #pragma unroll
        for(u32 k=0;k<4;++k){const u32 q=lane+32*k,r=q>>3,ch=q&7;
            swc_cpa16(u32(__cvta_generic_to_shared(pbuf+r*32+((4*ch)^(8*(r&3))))),residues+(u64(r)*8+j)*N+i0+4*ch);}
        swc_commit();};
    prefetch(0);
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    __syncthreads();
    u32 acc[NW+2];
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    u32 a[2][2][4];
    const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
    auto frag=[&](){
        swc_wait0();__syncwarp();
        const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(32*tig+gid)*4;
        const u32 p0=pb+32*(0^tig),p1=pb+32*(1^tig),p2=pb+32*(2^tig),p3=pb+32*(3^tig);
#define SWC_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*128>(q0);a[m][s][1]=swc_lds<(8*s)*128>(q1);a[m][s][2]=swc_lds<(8*s+4)*128>(q0);a[m][s][3]=swc_lds<(8*s+4)*128>(q1);
        SWC_FR(0,0,p0,p1) SWC_FR(0,1,p0,p1) SWC_FR(1,0,p2,p3) SWC_FR(1,1,p2,p3)
#undef SWC_FR
        __syncwarp();};
    frag();prefetch(1);
    swc_tile<0>(a,bsh,lane,sb);swc_tile<1>(a,bsh,lane,sb);swc_tile<2>(a,bsh,lane,sb);swc_tile<3>(a,bsh,lane,sb);swc_tile<4>(a,bsh,lane,sb);
    swc_tile<5>(a,bsh,lane,sb);swc_tile<6>(a,bsh,lane,sb);swc_tile<7>(a,bsh,lane,sb);swc_tile<8>(a,bsh,lane,sb);
    #pragma unroll
    for(u32 j=0;j<8;++j){
        __syncwarp();   // tiles(j) stores visible
        u32 xe[16],xo[16];
        const u32* hs=st+lane*SST;
        #pragma unroll
        for(u32 i=0;i<16;++i){const uint2 v=*reinterpret_cast<const uint2*>(hs+2*i);xe[i]=v.x;xo[i]=v.y;}
        const u64 S=u64(hs[32])+(u64(hs[33])<<16)+(u64(hs[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        __syncwarp();   // every lane's h row of digit j is in registers: the stage is free for tiles(j+1)
        const bool nx=j<7;
        if(nx){frag();if(j<6)prefetch(j+2);}
        auto g0=[&]{if(nx){swc_tile<0>(a,bsh,lane,sb);swc_tile<1>(a,bsh,lane,sb);swc_tile<2>(a,bsh,lane,sb);}};
        auto g1=[&]{if(nx){swc_tile<3>(a,bsh,lane,sb);swc_tile<4>(a,bsh,lane,sb);swc_tile<5>(a,bsh,lane,sb);}};
        auto g2=[&]{if(nx){swc_tile<6>(a,bsh,lane,sb);swc_tile<7>(a,bsh,lane,sb);swc_tile<8>(a,bsh,lane,sb);}};
        switch(j){case 0:cs_digit<0>(acc,xe,xo,alpha,g0,g1,g2);break;case 1:cs_digit<1>(acc,xe,xo,alpha,g0,g1,g2);break;case 2:cs_digit<2>(acc,xe,xo,alpha,g0,g1,g2);break;case 3:cs_digit<3>(acc,xe,xo,alpha,g0,g1,g2);break;
                  case 4:cs_digit<4>(acc,xe,xo,alpha,g0,g1,g2);break;case 5:cs_digit<5>(acc,xe,xo,alpha,g0,g1,g2);break;case 6:cs_digit<6>(acc,xe,xo,alpha,g0,g1,g2);break;default:cs_digit<7>(acc,xe,xo,alpha,g0,g1,g2);break;}
    }
    {   // fold / final (= bpo_crt_lean)
        u32 hw[10];
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
    if(dfl&1)for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);
    const u32 cc_=i0>>8,h0=i0&255u;
    {
        __syncwarp();
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
#endif
#ifndef BPO_P2_OLD
// r16 stage twiddles from a compact table: q[j] = t[16 j] (t[32 j] = q[2 j], t[64] = q[4]); the r16_* bodies index t[16 j]
struct CsTw{const u64* q;__device__ __forceinline__ u64 operator[](u32 i)const{return q[i>>4];}};
template<class T>__device__ __forceinline__ void cs_dif16(u32 (&v)[16],T t,u32 p){
    #pragma unroll
    for(u32 j=0;j<8;++j){const u32 a=v[j],b=v[j+8];v[j]=pxadd(a,b,p);v[j+8]=j?pxmsh(a+p-b,t[16*j],p):pxsub(a,b,p);}
    #pragma unroll
    for(u32 h=0;h<16;h+=8)
        #pragma unroll
        for(u32 j=0;j<4;++j){const u32 a=v[h+j],b=v[h+j+4];v[h+j]=pxadd(a,b,p);v[h+j+4]=j?pxmsh(a+p-b,t[32*j],p):pxsub(a,b,p);}
    #pragma unroll
    for(u32 h=0;h<16;h+=4){
        {const u32 a=v[h],b=v[h+2];v[h]=pxadd(a,b,p);v[h+2]=pxsub(a,b,p);}
        {const u32 a=v[h+1],b=v[h+3];v[h+1]=pxadd(a,b,p);v[h+3]=pxmsh(a+p-b,t[64],p);}}
    #pragma unroll
    for(u32 h=0;h<16;h+=2){const u32 a=v[h],b=v[h+1];v[h]=pxadd(a,b,p);v[h+1]=pxsub(a,b,p);}
}
template<bool CI,class T>__device__ __forceinline__ void cs_dit16(u32 (&v)[16],T t,u32 p){
    #pragma unroll
    for(u32 h=0;h<16;h+=2){if(CI)pzdit1c(v[h],v[h+1],p);else pzdit1(v[h],v[h+1],0,p);}
    #pragma unroll
    for(u32 h=0;h<16;h+=4){pzdit1(v[h],v[h+2],0,p);pzdit(v[h+1],v[h+3],t[64],p);}
    #pragma unroll
    for(u32 h=0;h<16;h+=8)
        #pragma unroll
        for(u32 j=0;j<4;++j){if(j)pzdit(v[h+j],v[h+j+4],t[32*j],p);else pzdit1(v[h],v[h+4],0,p);}
    #pragma unroll
    for(u32 j=0;j<8;++j){if(j)pzdit(v[j],v[j+8],t[16*j],p);else pzdit1(v[0],v[8],0,p);}
}
template<bool H,class T>__device__ __forceinline__ void cs_dif16h(u32 (&v)[16],T t,u32 p,u32 p2){
    if(!H){cs_dif16(v,t,p);return;}
    #pragma unroll
    for(u32 j=0;j<8;++j){const u32 a=v[j],b=v[j+8];v[j]=pxadd(a,b,p2);v[j+8]=j?pxmshl(a+p2-b,t[16*j],p):pxadd(a,p2-b,p2);}
    #pragma unroll
    for(u32 h=0;h<16;h+=8)
        #pragma unroll
        for(u32 j=0;j<4;++j){const u32 a=v[h+j],b=v[h+j+4];v[h+j]=pxadd(a,b,p2);v[h+j+4]=j?pxmshl(a+p2-b,t[32*j],p):pxadd(a,p2-b,p2);}
    #pragma unroll
    for(u32 h=0;h<16;h+=4){
        {const u32 a=v[h],b=v[h+2];v[h]=pxadd(a,b,p2);v[h+2]=pxadd(a,p2-b,p2);}
        {const u32 a=v[h+1],b=v[h+3];v[h+1]=pxadd(a,b,p2);v[h+3]=pxmshl(a+p2-b,t[64],p);}}
    #pragma unroll
    for(u32 h=0;h<16;h+=2){const u32 a=v[h],b=v[h+1];v[h]=pxadd(a,b,p2);v[h+1]=pxadd(a,p2-b,p2);}
}
template<bool H,bool CI,class T>__device__ __forceinline__ void cs_dit16h(u32 (&v)[16],T t,u32 p,u32 p2){
    if(!H){cs_dit16<CI>(v,t,p);return;}
    #pragma unroll
    for(u32 h=0;h<16;h+=2){if(CI)phdit1c(v[h],v[h+1],p2);else phdit1(v[h],v[h+1],p2);}
    #pragma unroll
    for(u32 h=0;h<16;h+=4){phdit1(v[h],v[h+2],p2);phdit(v[h+1],v[h+3],t[64],p,p2);}
    #pragma unroll
    for(u32 h=0;h<16;h+=8)
        #pragma unroll
        for(u32 j=0;j<4;++j){if(j)phdit(v[h+j],v[h+j+4],t[32*j],p,p2);else phdit1(v[h],v[h+4],p2);}
    #pragma unroll
    for(u32 j=0;j<8;++j){if(j)phdit(v[j],v[j+8],t[16*j],p,p2);else phdit1(v[0],v[8],p2);}
}
__device__ __forceinline__ void cs_cpa8(u64* dst,const u64* src){asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"::"r"(u32(__cvta_generic_to_shared(dst))),"l"(src):"memory");}
// 4x4 transpose over the lane quad (lane bits 0..1 = q): lane q ends with r[k] = lane k's r[q]
__device__ __forceinline__ void cs_tr4(u32& r0,u32& r1,u32& r2,u32& r3,u32 q){
    {const bool up=(q&2u)!=0u;u32 x=up?r0:r2,y=up?r1:r3;x=__shfl_xor_sync(0xffffffffu,x,2);y=__shfl_xor_sync(0xffffffffu,y,2);if(up){r0=x;r1=y;}else{r2=x;r3=y;}}
    {const bool up=(q&1u)!=0u;u32 x=up?r0:r1,y=up?r2:r3;x=__shfl_xor_sync(0xffffffffu,x,1);y=__shfl_xor_sync(0xffffffffu,y,1);if(up){r0=x;r2=y;}else{r1=x;r3=y;}}
}
// bpo_r16_body (p2k) with TWB (smem twiddles) / ST (128-bit output stores); same values
template<bool H,bool TWB,bool ST>__device__ __forceinline__ void cs_r16_body(u32* TA,u64* TW,u64* TS,const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 untw){
    constexpr u32 N=65536u;
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,s=lane&7u,c2=4u*warp+(lane>>3);
    const u64* tf=r16+u64(prime)*512u;const u64* ti=tf+256u;
    const char* tfb=reinterpret_cast<const char*>(bpo_opq(tf));const char* tib=reinterpret_cast<const char*>(bpo_opq(ti));
    const u64 base=u64(prime*8+j)*N;
    const u32 off=g*2048u+c2*8u+s;
    u32 a[16],b[16];
    {   const u32* a0=A+base+off;const u32* b0=B+base+off;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1){if(untw&4){a[c1]=__ldcs(a0+128u*c1);b[c1]=__ldcs(b0+128u*c1);}else{a[c1]=a0[128u*c1];b[c1]=b0[128u*c1];}}
    }
    const u32 t=threadIdx.x;
    // TW[c*15 + i-1]: forward middle twiddle w^(c brev4(i)) (inverse after the first barrier: w^-(i brev4(c)))
    // (cp.async 8-byte copies: no registers held across the A/B loads / transposes)
    auto mid_cp=[&](u32 e,bool inv){const u32 c=e/15u,i=e-15u*c+1u;cs_cpa8(TW+e,inv?ti+i*r16_brev4(c):tf+c*r16_brev4(i));};
    if constexpr(TWB){
        if(t<16u)cs_cpa8(TS+t,(t<8u?tf:ti)+16u*(t&7u));
        mid_cp(t,false);if(t<112u)mid_cp(t+128u,false);
        asm volatile("cp.async.commit_group;":::"memory");asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
    }
    const CsTw tsf{TS},tsi{TS+8};
    const u32 p2=H?__shfl_sync(0xffffffffu,2u*p,0):0u;(void)p2;
    if constexpr(TWB){cs_dif16h<H>(a,tsf,p,p2);cs_dif16h<H>(b,tsf,p,p2);}else{r16_dif16h<H>(a,tf,p,p2);r16_dif16h<H>(b,tf,p,p2);}
    u32* const tw_=TA+int(s*292u+c2);
    u32* const tr_=TA+int(s*292u+17u*c2);
    #pragma unroll
    for(u32 i=1;i<16;++i){const u64 w=TWB?TW[c2*15u+i-1u]:bpo_tw8(tfb,8u*c2,r16_brev4(i));a[i]=pxmshh<H>(a[i],w,p);b[i]=pxmshh<H>(b[i],w,p);}
    #pragma unroll
    for(u32 i=0;i<16;++i)tw_[17*i]=a[i];
    __syncthreads();
    if constexpr(TWB){mid_cp(t,true);if(t<112u)mid_cp(t+128u,true);asm volatile("cp.async.commit_group;":::"memory");}   // forward TW reads are done (barrier above)
    if(untw&8)bpo_discard((threadIdx.x<64u?A:B)+base+g*2048u+(threadIdx.x&63u)*32u);
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=tr_[i];
    if constexpr(TWB)asm volatile("cp.async.wait_group 0;":::"memory");   // inverse TW visible after the next barrier
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)tw_[17*i]=b[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)b[i]=tr_[i];
    if constexpr(TWB){cs_dif16h<H>(a,tsf,p,p2);cs_dif16h<H>(b,tsf,p,p2);}else{r16_dif16h<H>(a,tf,p,p2);r16_dif16h<H>(b,tf,p,p2);}
    {const u32 pinv=pxinv32(p);
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=pxmont(a[i],b[i],p,pinv);}
    if constexpr(TWB)cs_dit16h<H,true>(a,tsi,p,p2);else r16_dit16h<H,true>(a,ti,p,p2);
    {const u32 k8=8u*r16_brev4(c2);(void)k8;
        #pragma unroll
        for(u32 i=1;i<16;++i)a[i]=pxmshh<H>(a[i],TWB?TW[c2*15u+i-1u]:bpo_tw8(tib,k8,i),p);
        a[0]=H?pxadd(a[0],0u,p2):pzred(a[0],p);}
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)tr_[i]=a[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=tw_[17*i];
    if constexpr(TWB)cs_dit16h<H,true>(a,tsi,p,p2);else r16_dit16h<H,true>(a,ti,p,p2);
    if constexpr(ST){   // lane (s = 4h + q, c2): after the quad transpose k, row c1 = 4k + q, words s = 4h .. 4h+3
        const u32 q=s&3u,h=s>>2;
        uint4* o4=reinterpret_cast<uint4*>(C+base+(c2*256u+g*8u+4u*h+4096u*q));
        #pragma unroll
        for(u32 k=0;k<4;++k){u32 r0=a[4*k],r1=a[4*k+1],r2=a[4*k+2],r3=a[4*k+3];cs_tr4(r0,r1,r2,r3,q);o4[4096u*k]=make_uint4(r0,r1,r2,r3);}   // 4 rows of 4096 words = 4096 uint4
    }else{
        u32* o=C+base+(c2*256u+g*8u+s);
        #pragma unroll
        for(u32 c1=0;c1<16;++c1)o[4096u*c1]=a[c1];
    }
}
#define CS_P2K(NAME,TWB,ST) extern "C" __global__ void __launch_bounds__(128,8) NAME(const u32* A,const u32* B,u32* C,\
    const u32* primes,const u64* r16,u32 N,u32 untw){\
    __shared__ u32 TA[8*292];__shared__ u64 TW[TWB?240:1];__shared__ u64 TS[TWB?16:1];(void)N;\
    if(BPO_NH30&&primes[blockIdx.z]<(1u<<30))cs_r16_body<true,TWB,ST>(TA,TW,TS,A,B,C,primes,r16,untw);else\
    cs_r16_body<false,TWB,ST>(TA,TW,TS,A,B,C,primes,r16,untw);}
#ifndef BPO_NH30
#define BPO_NH30 0
#endif
CS_P2K(cs_p2t,true,false)
CS_P2K(cs_p2s,false,true)
CS_P2K(cs_p2ts,true,true)
#endif
#ifdef BPO_P3B
// bpo_p3b_body with the stage_itw table of the prime in smem for the smr R-DIT (TS[t] = sit[t], 256 threads)
template<bool H>__device__ __forceinline__ void cs_p3t_body(u32* X,u64* TS,const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 dfl){
    constexpr u32 N=65536u,logR=8u;
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 t=threadIdx.x,lane=t&31u,warp=t>>5;
    const u32 p2=H?__shfl_sync(0xffffffffu,2u*p,0):0u;(void)p2;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* sit=stage_itw+prime*256u;
    u32 yv[8];
    {const u32* src=C+u64(prime*8u)*N+(c*256u+t);
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=src[d*N];}
    TS[t]=__ldg(sit+t);   // visible after the X barrier
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
    u64 tv[8];
    #pragma unroll
    for(u32 m=0;m<8;++m)tv[m]=__ldg(tn+32u*m);
    if((dfl&1)&&lane<8u)bpo_discard(C+u64(prime*8u+lane)*N+(c*256u+32u*warp));
    u32 v[8];
    u32* const base=X+warp*288u;
    {const uint4* p4=reinterpret_cast<const uint4*>(base+(8u*lane+4u*(lane>>2)));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    if(H)phdit_smr<false>(base,v,TS,p,p2,lane);else pzdit_smr<false>(base,v,TS,p,lane);
    u32* const out=D+(u64(prime*8u+warp)*N+(c*256u+lane));
#ifdef BPO_P3EL
    const u32 pm=12u;
#else
    const u32 pm=4u;
#endif
    if(dfl&pm){
        u64 pol;if(dfl&pm&8u)asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=pxmsh(v[m],tv[m],p);asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+32u*m),"r"(r),"l"(pol):"memory");}
    }else{
        #pragma unroll
        for(u32 m=0;m<8;++m)out[32u*m]=pxmsh(v[m],tv[m],p);
    }
}
extern "C" __global__ void __maxnreg__(32) cs_p3t(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;__shared__ __align__(16) u64 TS[256];
    if(N==65536u&&logR==8u&&!(dfl&2u)){
#if BPO_NH30
        if(primes[blockIdx.y]<(1u<<30))cs_p3t_body<true>(X,TS,C,D,primes,untwist,stage_itw,t2n,dfl);else
#endif
        cs_p3t_body<false>(X,TS,C,D,primes,untwist,stage_itw,t2n,dfl);
        return;}
#if BPO_NH30
    if(primes[blockIdx.y]<(1u<<30))bpo_p3_body<BPO_P3T,true>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);else
#endif
    bpo_p3_body<BPO_P3T>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);
}
#endif
// ---- Round 25 (p1x2): EXACT P1 variants (bpo_first_xb K bits >= 16, see batched_p1x.cuh XK_*), production launch geometry /
// arguments / smem; cs_p1ct and cs_p1all read the permuted cross-twiddle copy cs_ctp_buf (filled once by cs_ctp_fill from ctab).
#if defined(BPO_X_SMEM)&&(!defined(BPO_LEAN)||BPO_LEAN_P1X)
__device__ u64 cs_ctp_buf[BCRT_K*65536u];   // ct'[(r 256 + c) 256 + 64 m2 + 2 lane + e] = ctab[(r 256 + c) 256 + 8 lane + 2 m2 + e]
extern "C" __global__ void __launch_bounds__(256) cs_ctp_fill(const u64* __restrict__ ctab){
    for(u32 i=blockIdx.x*blockDim.x+threadIdx.x;i<BCRT_K*65536u;i+=gridDim.x*blockDim.x){const u32 s=i&255u;
        cs_ctp_buf[i]=ctab[(i&~255u)+8u*((s>>1)&31u)+2u*(s>>6)+(s&1u)];}}
#define CS_P1(NAME,KB,CTP) extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) NAME(const u32* a,const u32* b,u32* A,u32* B,\
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,\
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){\
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];(void)logR;(void)ctab;\
    bpo_first_xb<BPO_XF,BPO_XG,(KB)>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,(CTP),&tmA,&tmB,X);}
CS_P1(cs_p1il,1u<<16,ctab)
CS_P1(cs_p1pr,1u<<17,ctab)
CS_P1(cs_p1ld,1u<<18,ctab)
CS_P1(cs_p1ct,1u<<19,ctab?cs_ctp_buf:ctab)
CS_P1(cs_p1bk,1u<<20,ctab)
CS_P1(cs_p1all,31u<<16,ctab?cs_ctp_buf:ctab)
#endif
#endif
