// bpo_crt_pipe (opt-in, FHERMA_CRT_PIPE=<flags>; the kernel is emitted only when BCP_FL is defined,
// the templates below are not instantiated otherwise: production PTX unchanged): bpo_crt_lean restructured for issue
// efficiency. Same launch geometry / arguments / smem / output, same arithmetic (bit-exact by construction):
//  1  tile phase with independent IMMA chains: two n8 tiles in flight (4 accumulators), non-volatile mma, so the fixed
//     IMMA->IMMA accumulate latency is covered by the other chains instead of ~220 never-executed UIADD3 fillers.
//  (default) software pipeline over the 8 digits: digit j's h row is read into registers right after its tiles, so the IMMA
//     tile phase of digit j+1 (writing the same warp stage) is interleaved with the carry chains of digit j;
//  2  no pipeline: production order (chains of digit j, then the tiles of digit j+1).
//  4  B fragments as one LDS.128 per tile (bsh re-laid out [T][lane][4] while it is copied in) instead of two LDS.64.
// Timing-only ablations (wrong values): 64 no digit chains, 128 no tiles, 256 no final reduction, 512 no output stores.
#ifdef BTC_F
__device__ __forceinline__ void bcp_mma(u32 (&c)[4],const u32 (&a)[4],u32 b0,u32 b1){
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// B fragments of tile T (b0 = k-half 0, b1 = k-half 1)
template<u32 FL,u32 T> __device__ __forceinline__ uint4 bcp_bfr(const u32* bsh,u32 lane){
    if(FL&4)return *reinterpret_cast<const uint4*>(bsh+(T*32+lane)*4);
    const uint2 x=*reinterpret_cast<const uint2*>(bsh+((T*2+0)*32+lane)*2),y=*reinterpret_cast<const uint2*>(bsh+((T*2+1)*32+lane)*2);
    return make_uint4(x.x,x.y,y.x,y.y);
}
// tiles T and U (U = T+1, or U = T for a single tile) of both m tiles -> h stage
template<u32 FL,u32 T,u32 U> __device__ __forceinline__ void bcp_tiles(const u32 (&a)[2][2][4],const u32* bsh,u32 lane,u32 sb){
    const uint4 bt=bcp_bfr<FL,T>(bsh,lane);
    if(T==U){
        u32 c[4]={0,0,0,0},e[4]={0,0,0,0};
        bcp_mma(c,a[0][0],bt.x,bt.y);bcp_mma(e,a[1][0],bt.x,bt.y);bcp_mma(c,a[0][1],bt.z,bt.w);bcp_mma(e,a[1][1],bt.z,bt.w);
        swc_sts<(0*36+4*T)*4>(sb,c[0]+(c[1]<<8));swc_sts<(8*36+4*T)*4>(sb,c[2]+(c[3]<<8));
        swc_sts<(16*36+4*T)*4>(sb,e[0]+(e[1]<<8));swc_sts<(24*36+4*T)*4>(sb,e[2]+(e[3]<<8));
    }else{
        const uint4 bu=bcp_bfr<FL,U>(bsh,lane);
        u32 c[4]={0,0,0,0},e[4]={0,0,0,0},f[4]={0,0,0,0},g[4]={0,0,0,0};
        bcp_mma(c,a[0][0],bt.x,bt.y);bcp_mma(e,a[1][0],bt.x,bt.y);bcp_mma(f,a[0][0],bu.x,bu.y);bcp_mma(g,a[1][0],bu.x,bu.y);
        bcp_mma(c,a[0][1],bt.z,bt.w);bcp_mma(e,a[1][1],bt.z,bt.w);bcp_mma(f,a[0][1],bu.z,bu.w);bcp_mma(g,a[1][1],bu.z,bu.w);
        swc_sts<(0*36+4*T)*4>(sb,c[0]+(c[1]<<8));swc_sts<(8*36+4*T)*4>(sb,c[2]+(c[3]<<8));
        swc_sts<(16*36+4*T)*4>(sb,e[0]+(e[1]<<8));swc_sts<(24*36+4*T)*4>(sb,e[2]+(e[3]<<8));
        swc_sts<(0*36+4*U)*4>(sb,f[0]+(f[1]<<8));swc_sts<(8*36+4*U)*4>(sb,f[2]+(f[3]<<8));
        swc_sts<(16*36+4*U)*4>(sb,g[0]+(g[1]<<8));swc_sts<(24*36+4*U)*4>(sb,g[2]+(g[3]<<8));
    }
}
template<u32 FL,u32 P> __device__ __forceinline__ void bcp_tphase(const u32 (&a)[2][2][4],const u32* bsh,u32 lane,u32 sb){   // P: 0 tiles 0-2, 1 tiles 3-5, 2 tiles 6-8
    if(FL&128)return;   // ablation (timing only)
    if(FL&1){
        if(P==0){bcp_tiles<FL,0,1>(a,bsh,lane,sb);bcp_tiles<FL,2,2>(a,bsh,lane,sb);}
        if(P==1){bcp_tiles<FL,3,4>(a,bsh,lane,sb);bcp_tiles<FL,5,5>(a,bsh,lane,sb);}
        if(P==2){bcp_tiles<FL,6,7>(a,bsh,lane,sb);bcp_tiles<FL,8,8>(a,bsh,lane,sb);}
    }else{
        if(P==0){bcp_tiles<FL,0,0>(a,bsh,lane,sb);bcp_tiles<FL,1,1>(a,bsh,lane,sb);bcp_tiles<FL,2,2>(a,bsh,lane,sb);}
        if(P==1){bcp_tiles<FL,3,3>(a,bsh,lane,sb);bcp_tiles<FL,4,4>(a,bsh,lane,sb);bcp_tiles<FL,5,5>(a,bsh,lane,sb);}
        if(P==2){bcp_tiles<FL,6,6>(a,bsh,lane,sb);bcp_tiles<FL,7,7>(a,bsh,lane,sb);bcp_tiles<FL,8,8>(a,bsh,lane,sb);}
    }
}
// swc_digit<J,true> split into its three dependent pieces (P 0: even chain, 1: odd chain, 2: carry tail); x = the h row
template<u32 J,u32 P> __device__ __forceinline__ void bcp_digit(u32 (&acc)[66],const u32 (&x)[36],u32 alpha){
    constexpr u32 o=SWC_POS[J]/32,sh=SWC_POS[J]%32;
    if(P==0){u32 xe[16];
        #pragma unroll
        for(u32 i=0;i<16;++i)xe[i]=x[2*i]+alpha*swc_np16(2*i)+swc_of16(2*i);
        swc_mac<(1u<<sh)>(acc+o,xe);}
    if(P==1){u32 xo[16];
        #pragma unroll
        for(u32 i=0;i<16;++i)xo[i]=x[2*i+1]+alpha*swc_np16(2*i+1)+swc_of16(2*i+1);
        if(sh<16)swc_mac<(1u<<((sh+16)&31))>(acc+o,xo);else swc_mac<(1u<<((sh+16)&31))>(acc+o+1,xo);}
    if(P==2){
        i64 c=0;
        #pragma unroll
        for(u32 w=16;w<=18;++w){
            const u32 ex=w==16?(alpha<<sh):(w==17&&sh?(alpha>>((32-sh)&31)):0u);
            const i64 t=i64(acc[o+w])-i64(ex)+c;acc[o+w]=u32(t);c=t>>32;}}
}
template<u32 FL,u32 J,u32 P> __device__ __forceinline__ void bcp_dig(u32 (&acc)[66],const u32 (&x)[36],u32 alpha){
    if(FL&64){if(P==0){_Pragma("unroll") for(u32 i=0;i<16;++i)acc[(J*7+i)&63]+=x[2*i]^alpha;}return;}   // ablation: no chains
    bcp_digit<J,P>(acc,x,alpha);
}
template<u32 FL> __device__ __forceinline__ void bcp_body(const u32* __restrict__ residues,u32* __restrict__ output,
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
    auto frags=[&](u32 (&a)[2][2][4]){
        swc_wait0();__syncwarp();
        const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(32*tig+gid)*4;
        const u32 p0=pb+32*(0^tig),p1=pb+32*(1^tig),p2=pb+32*(2^tig),p3=pb+32*(3^tig);
#define SWC_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*128>(q0);a[m][s][1]=swc_lds<(8*s)*128>(q1);a[m][s][2]=swc_lds<(8*s+4)*128>(q0);a[m][s][3]=swc_lds<(8*s+4)*128>(q1);
        SWC_FR(0,0,p0,p1) SWC_FR(0,1,p0,p1) SWC_FR(1,0,p2,p3) SWC_FR(1,1,p2,p3)
#undef SWC_FR
        __syncwarp();};
    prefetch(0);
    if(FL&4){for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128){const u32 h=x&1,ln=(x>>1)&31,s2=(x>>6)&1,T=x>>7;bsh[(T*32+ln)*4+s2*2+h]=bfrag[x];}}
    else for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    __syncthreads();
    u32 acc[NW+2];
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
    const u32* hs=st+lane*SST;
    u32 a[2][2][4];
    u32 x[36];
    auto readrow=[&](){__syncwarp();
        #pragma unroll
        for(u32 i=0;i<9;++i){const uint4 v=*reinterpret_cast<const uint4*>(hs+4*i);x[4*i]=v.x;x[4*i+1]=v.y;x[4*i+2]=v.z;x[4*i+3]=v.w;}
        __syncwarp();};
    // prologue: digit 0 tiles
    frags(a);prefetch(1);
    bcp_tphase<FL,0>(a,bsh,lane,sb);bcp_tphase<FL,1>(a,bsh,lane,sb);bcp_tphase<FL,2>(a,bsh,lane,sb);
    readrow();
#define BCP_STEP(J) {\
        const u64 S=u64(x[32])+(u64(x[33])<<16)+(u64(x[34])<<32);const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);\
        if(J<7&&!(FL&2)){frags(a);if(J<6)prefetch(J+2);            bcp_tphase<FL,0>(a,bsh,lane,sb);bcp_dig<FL,J,0>(acc,x,alpha);bcp_tphase<FL,1>(a,bsh,lane,sb);bcp_dig<FL,J,1>(acc,x,alpha);            bcp_tphase<FL,2>(a,bsh,lane,sb);bcp_dig<FL,J,2>(acc,x,alpha);readrow();}        else{bcp_dig<FL,J,0>(acc,x,alpha);bcp_dig<FL,J,1>(acc,x,alpha);bcp_dig<FL,J,2>(acc,x,alpha);            if(J<7){frags(a);if(J<6)prefetch(J+2);bcp_tphase<FL,0>(a,bsh,lane,sb);bcp_tphase<FL,1>(a,bsh,lane,sb);bcp_tphase<FL,2>(a,bsh,lane,sb);readrow();}}}
    BCP_STEP(0) BCP_STEP(1) BCP_STEP(2) BCP_STEP(3) BCP_STEP(4) BCP_STEP(5) BCP_STEP(6) BCP_STEP(7)
#undef BCP_STEP
    if(!(FL&256)){   // final reduction: as bpo_crt_lean (FL 256: ablation, skipped)
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
    if(dfl&1)for(u32 x_=lane;x_<K*8u;x_+=32u)bpo_discard(residues+u64(x_)*N+i0);
    const u32 cc_=i0>>8,h0=i0&255u;
    if(FL&512){u32 z=0;for(u32 w=0;w<L;++w)z^=acc[w];if(z==0x9e3779b9u)output[i0+lane]=z;}   // ablation: no output stores
    else{
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
#ifdef BCP_FL
extern "C" __global__ void __launch_bounds__(128,3) bpo_crt_pipe(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    bcp_body<BCP_FL>(residues,output,bfrag,N,d,pfa,pfb,pfitems,dfl);
}
#endif
// ---- bpo_crt_q4 (opt-in, FHERMA_CRT_Q4=<flags>; kernel emitted only when BCQ_FL is defined): 4 blocks/SM CRT.
// <= 128 registers (__launch_bounds__(128,4)) and 23 KB static smem (4 x 24 KB with the per-block reserve fits 100 KB/SM):
//  * no cp.async residue buffer: MMA row R = 16m + 8h + gid of the warp holds coefficient 16m + 2gid + h, so each A fragment
//    is two 8 B LDG.64 (8 per digit, issued one digit ahead into registers);
//  * B fragments one LDS.128 per tile ([T][lane][4] layout), h stage 32 x 36 words per warp as before;
//  * output staged in two halves of 16 coefficients (16 x 55 words fit the h stage), stores in coefficient order.
// Flags: 1 paired tiles (independent IMMA chains, as bpo_crt_pipe 1), 2 next-digit fragment loads issued after the tiles
// (instead of before), 4 output stores via __stcs regardless of dfl.
template<u32 FL> __device__ __forceinline__ void bcq_body(const u32* __restrict__ residues,u32* __restrict__ output,
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
    constexpr u32 STW=32*SST;
    static_assert(16*L<=STW,"half output stage fits the h stage");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    // fragment loads of digit j: MMA row R = 16m + 8h + gid holds coefficient pi(R) = 16m + 2gid + h, so a fragment's two rows
    // are 2 consecutive coefficients: a[m][s] = {LDG.64 prime 8s+tig, LDG.64 prime 8s+4+tig} at coefficients i0+16m+2gid
    const u32* rb=residues+u64(tig)*8*N+i0+2*gid;
    auto ldfr=[&](u32 (&a)[2][2][4],u32 j){
        #pragma unroll
        for(u32 m=0;m<2;++m)
        #pragma unroll
        for(u32 s=0;s<2;++s)
        #pragma unroll
        for(u32 h=0;h<2;++h){const uint2 v=__ldcg(reinterpret_cast<const uint2*>(rb+(u64(8*s+4*h)*8+j)*N+16*m));
            a[m][s][2*h]=v.x;a[m][s][2*h+1]=v.y;}};
    u32 a[2][2][4],an[2][2][4];
    ldfr(an,0);
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128){const u32 h=x&1,ln=(x>>1)&31,s2=(x>>6)&1,T=x>>7;bsh[(T*32+ln)*4+s2*2+h]=bfrag[x];}
    __syncthreads();
    u32 acc[NW+2];
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
    const u32* hs=st+lane*SST;
    constexpr u32 TF=4|(FL&1);
#define BCQ_STEP(J) {\
        _Pragma("unroll") for(u32 m=0;m<2;++m) _Pragma("unroll") for(u32 s=0;s<2;++s) _Pragma("unroll") for(u32 i=0;i<4;++i)a[m][s][i]=an[m][s][i];\
        if(J<7&&!(FL&2))ldfr(an,J+1);\
        bcp_tphase<TF,0>(a,bsh,lane,sb);bcp_tphase<TF,1>(a,bsh,lane,sb);bcp_tphase<TF,2>(a,bsh,lane,sb);\
        if(J<7&&(FL&2))ldfr(an,J+1);\
        u32 x[36];__syncwarp();\
        _Pragma("unroll") for(u32 i=0;i<9;++i){const uint4 v=*reinterpret_cast<const uint4*>(hs+4*i);x[4*i]=v.x;x[4*i+1]=v.y;x[4*i+2]=v.z;x[4*i+3]=v.w;}\
        __syncwarp();\
        const u64 S=u64(x[32])+(u64(x[33])<<16)+(u64(x[34])<<32);const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);\
        bcp_digit<J,0>(acc,x,alpha);bcp_digit<J,1>(acc,x,alpha);bcp_digit<J,2>(acc,x,alpha);}
    BCQ_STEP(0) BCQ_STEP(1) BCQ_STEP(2) BCQ_STEP(3) BCQ_STEP(4) BCQ_STEP(5) BCQ_STEP(6) BCQ_STEP(7)
#undef BCQ_STEP
    {
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
    if(dfl&1)for(u32 x_=lane;x_<K*8u;x_+=32u)bpo_discard(residues+u64(x_)*N+i0);
    const u32 cc_=i0>>8,h0=i0&255u;
    u32* const ob=output+u64(h0*256u+cc_)*L+lane;
    // lane = stage row R holds coefficient pi(R) = 16(R/16) + 2(R%8) + (R/8)%2; coefficient c sits in row 16(c/16) + 8(c%2) + (c%16)/2
    #pragma unroll
    for(u32 hf=0;hf<2;++hf){
        __syncwarp();
        if((lane>>4)==hf){
            #pragma unroll
            for(u32 w=0;w<L;++w)st[(lane&15)*L+w]=acc[w];}
        __syncwarp();
        #pragma unroll
        for(u32 rr=0;rr<32;++rr){
            if(rr/16!=hf)continue;
            const u32 row=8*(rr&1)+(rr&15)/2;
            if((dfl&2)||(FL&4)){__stcs(ob+rr*256u*L,st[row*L+lane]);if(lane<L-32)__stcs(ob+rr*256u*L+32,st[row*L+32+lane]);}
            else{ob[rr*256u*L]=st[row*L+lane];if(lane<L-32)ob[rr*256u*L+32]=st[row*L+32+lane];}
        }
    }
}
#ifdef BCQ_FL
extern "C" __global__ void __launch_bounds__(128,4) bpo_crt_q4(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    bcq_body<BCQ_FL>(residues,output,bfrag,N,d,pfa,pfb,pfitems,dfl);
}
#endif
#endif
