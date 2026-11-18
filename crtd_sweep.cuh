// Exit-time sweep only (diag_sweep.h round 8, crtd): CRT smem diet variants of bpo_crt_lean and the P3 evict_last twin.
// Appended to the sweep copy of the production source, never to the production module. Same parameters / launch geometry /
// output as bpo_crt_lean (and the same swc_* digit / final chains); only the fragment build, the smem layout and the launch
// bounds differ. Flags FL:
//   1 (A128) residue A fragments by 4 LDS.128 per digit instead of 16 LDS.32. ldmatrix itself does not apply (LDSM hands a
//     lane (row = f(gid), word = tig) of the [prime][coefficient] buffer, the transpose of the A fragment; a coefficient-major
//     buffer would need 4-byte cp.async), so the MMA's free permutations are used instead: the k dimension (primes) is
//     permuted - lane tig's 4 k-groups (s,tig),(s,4+tig) carry primes 4tig+2s+u - and the m dimension (coefficients) too -
//     C row (m tile, half h) of lane gid = coefficient 4gid+2m+h - so a lane's 4 words per prime are one 16 B chunk. B is
//     permuted to match (in-kernel rebuild of the bfrag table), the h stage keeps its rows (slot gid+8k holds coefficient
//     4gid+k) and the per-lane digit math runs coefficient sigma(lane) = 4(lane&7)+(lane>>3), un-permuted at the output
//     stage. Chunk swizzle ch ^ 2(prime>>2): conflict-free LDS.128 quarter-warps.
//   2 (B128) B fragments: one LDS.128 per tile ([T][lane][s][2] table) instead of two LDS.64.
//   4 (ALIAS) residue prefetch buffer inside the h stage (prefetch of digit j+1 issued after the h reads of digit j), output
//     stage in two 16-coefficient halves: 4.5 KB/warp instead of 6.9 KB -> 23 KB/block (4 blocks/SM fit in 100 KB).
#if defined(BTC_F)
template<u32 FL> __device__ __forceinline__ void swcd_tile(const u32 (&a)[2][2][4],const u32* bsh,u32 lane,u32 sb,u32 T){
    uint2 b0,b1;
    if(FL&2){const uint4 b=*reinterpret_cast<const uint4*>(bsh+(T*32+lane)*4);b0=make_uint2(b.x,b.y);b1=make_uint2(b.z,b.w);}
    else{b0=*reinterpret_cast<const uint2*>(bsh+((T*2+0)*32+lane)*2);b1=*reinterpret_cast<const uint2*>(bsh+((T*2+1)*32+lane)*2);}
    u32 c[4]={0,0,0,0};
    imma16832(c,a[0][0][0],a[0][0][1],a[0][0][2],a[0][0][3],b0.x,b0.y);imma16832(c,a[0][1][0],a[0][1][1],a[0][1][2],a[0][1][3],b1.x,b1.y);
    asm volatile("st.shared.u32 [%0], %1;"::"r"(sb+(0*36+4*T)*4),"r"(c[0]+(c[1]<<8)):"memory");asm volatile("st.shared.u32 [%0], %1;"::"r"(sb+(8*36+4*T)*4),"r"(c[2]+(c[3]<<8)):"memory");
    u32 e[4]={0,0,0,0};
    imma16832(e,a[1][0][0],a[1][0][1],a[1][0][2],a[1][0][3],b0.x,b0.y);imma16832(e,a[1][1][0],a[1][1][1],a[1][1][2],a[1][1][3],b1.x,b1.y);
    asm volatile("st.shared.u32 [%0], %1;"::"r"(sb+(16*36+4*T)*4),"r"(e[0]+(e[1]<<8)):"memory");asm volatile("st.shared.u32 [%0], %1;"::"r"(sb+(24*36+4*T)*4),"r"(e[2]+(e[3]<<8)):"memory");
}
template<u32 FL> __device__ __forceinline__ void swcd_body(const u32* __restrict__ residues,u32* __restrict__ output,
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
    constexpr bool AL=(FL&4)!=0;
    constexpr u32 STW=AL?32*SST:32*L;   // per-warp stage words
    static_assert(AL?(512<=STW&&16*L<=STW):(32*SST+512<=STW),"stage layout");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    u32* const pbuf=AL?st:st+32*SST;
    auto prefetch=[&](u32 j){
        #pragma unroll
        for(u32 k=0;k<4;++k){const u32 q=lane+32*k,r=q>>3,ch=q&7;
            const u32 dw=(FL&1)?r*32+4*(ch^(2*(r>>2))):r*32+((4*ch)^(8*(r&3)));
            swc_cpa16(u32(__cvta_generic_to_shared(pbuf+dw)),residues+(u64(r)*8+j)*N+i0+4*ch);}
        swc_commit();};
    prefetch(0);
    // B table: entry (T, s, lane, hh) = production entry of the prime it carries (prime map of FL&1), layout of FL&2
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128){
        u32 T,s,ln,hh;
        if(FL&2){hh=x&1;s=(x>>1)&1;ln=(x>>2)&31;T=x>>7;}else{hh=x&1;ln=(x>>1)&31;s=(x>>6)&1;T=x>>7;}
        const u32 P=(FL&1)?4*(ln&3)+2*s+hh:8*s+4*hh+(ln&3);
        const u32 src=((T*2+(P>>3))*32+(ln&~3u)+(P&3))*2+((P>>2)&1);
        bsh[x]=bfrag[src];}
    __syncthreads();
    u32 acc[NW+2];
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 a[2][2][4];
        swc_wait0();__syncwarp();
        if(FL&1){
            const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(4*tig*32+4*(gid^(2*tig)))*4;
            uint4 v[4];
            #pragma unroll
            for(u32 x=0;x<4;++x)asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(v[x].x),"=r"(v[x].y),"=r"(v[x].z),"=r"(v[x].w):"r"(pb+x*128):"memory");
            #pragma unroll
            for(u32 s=0;s<2;++s){const uint4 e0=v[2*s],e1=v[2*s+1];
                a[0][s][0]=e0.x;a[0][s][1]=e0.y;a[1][s][0]=e0.z;a[1][s][1]=e0.w;
                a[0][s][2]=e1.x;a[0][s][3]=e1.y;a[1][s][2]=e1.z;a[1][s][3]=e1.w;}
        }else{
            const u32 pb=u32(__cvta_generic_to_shared(pbuf))+(32*tig+gid)*4;
            const u32 p0=pb+32*(0^tig),p1=pb+32*(1^tig),p2=pb+32*(2^tig),p3=pb+32*(3^tig);
#define SWC_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*128>(q0);a[m][s][1]=swc_lds<(8*s)*128>(q1);a[m][s][2]=swc_lds<(8*s+4)*128>(q0);a[m][s][3]=swc_lds<(8*s+4)*128>(q1);
            SWC_FR(0,0,p0,p1) SWC_FR(0,1,p0,p1) SWC_FR(1,0,p2,p3) SWC_FR(1,1,p2,p3)
#undef SWC_FR
        }
        __syncwarp();
        if(!AL&&j<7)prefetch(j+1);
        {const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
            #pragma unroll
            for(u32 T=0;T<NT;++T)swcd_tile<FL>(a,bsh,lane,sb,T);}
        __syncwarp();
        const u32* hs=st+lane*SST;
        alignas(8) u32 hx[AL?32:2];   // AL: the h words in registers before the prefetch may overwrite them
        if(AL){
            #pragma unroll
            for(u32 i=0;i<16;++i){const uint2 q=*reinterpret_cast<const uint2*>(hs+2*i);hx[2*i]=q.x;hx[2*i+1]=q.y;}}
        const u32* xs=AL?hx:hs;
        const u64 S=u64(hs[32])+(u64(hs[33])<<16)+(u64(hs[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        if(AL){__syncwarp();if(j<7)prefetch(j+1);}   // h stage reads done: the next digit's residues may land on it
        switch(j){case 0:swc_digit<0,true>(acc,xs,alpha);break;case 1:swc_digit<1,true>(acc,xs,alpha);break;case 2:swc_digit<2,true>(acc,xs,alpha);break;case 3:swc_digit<3,true>(acc,xs,alpha);break;
                  case 4:swc_digit<4,true>(acc,xs,alpha);break;case 5:swc_digit<5,true>(acc,xs,alpha);break;case 6:swc_digit<6,true>(acc,xs,alpha);break;default:swc_digit<7,true>(acc,xs,alpha);break;}
        __syncwarp();
    }
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
    if(dfl&1)for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);
    const u32 cc_=i0>>8,h0=i0&255u;
    const u32 co=(FL&1)?4*(lane&7)+(lane>>3):lane;   // coefficient (within the warp's 32) of this lane's digit math
    u32* const ob=output+u64(h0*256u+cc_)*L+lane;
    #pragma unroll
    for(u32 hf=0;hf<(AL?2u:1u);++hf){
        __syncwarp();
        if(!AL||(co>>4)==hf){
            #pragma unroll
            for(u32 w=0;w<L;++w)st[(co-16*hf)*L+w]=acc[w];}
        __syncwarp();
        #pragma unroll
        for(u32 rr=0;rr<(AL?16u:32u);++rr){u32* o=ob+(rr+16*hf)*256u*L;
            if(dfl&2){__stcs(o,st[rr*L+lane]);if(lane<L-32)__stcs(o+32,st[rr*L+32+lane]);}
            else{o[0]=st[rr*L+lane];if(lane<L-32)o[32]=st[rr*L+32+lane];}}
    }
}
#define SWCD(NAME,FL,MB) extern "C" __global__ void __launch_bounds__(128,MB) NAME(const u32* __restrict__ residues,u32* __restrict__ output,\
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){swcd_body<FL>(residues,output,bfrag,N,d,pfa,pfb,pfitems,dfl);}
#endif
// P3 with evict_last residue stores (bpo_p3_body flag 8): the CRT reads group-0 residues from L2
extern "C" __global__ void __launch_bounds__(256) sw_p3el(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;
    bpo_p3_body<(BPO_P3T|8u)>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);
}
