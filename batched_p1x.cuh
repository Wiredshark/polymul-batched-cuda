// bpo_first_x / bpo_first_xv: P1 (TC base conversion + twist + J-DIF + R-DIF + cross twiddle, one block per
// column) with the values of bpo_first_tc2g, leaner arithmetic and optional TMA output stores.
// bpo_first_x uses flags BPO_XF (default set), bpo_first_xv flags BPO_XV (A/B experiments, graph variant 7).
// Flags: 1 skip psi^0 multiplies (7 of 12 J-DIF, 6 of 40 R-DIF+ct), 2 TMA tensor stores (compute_90+),
//        4 stage twiddles of the quad in smem, 8 512-thread blocks (16 warps x 16 rows, af 32 registers),
//        32 bank-spread regions (320 words, skewed layout at +8q: the four primes of a phase-A store hit distinct banks),
//        64 R-DIF stages len 256/128 in phase A (lane rows 64rr+8w+gid; implies 32), phase B two layouts,
//        128 (with 64) phase B processes its columns in interleaved pairs (ILP).
//        256 (with 64) input rows loaded coalesced (__ldcs, 128-row halves) into X, A fragments built from smem
//            (the 2.3x redundant per-thread LDGs of the funnel-shift windows missed L1).
//        512 (with 256) input halves staged with cp.async 16 B chunks (row stride 60 words, all chunks in flight).
//        2048 (with 64) TMA output stores with an L2 evict_first policy (P2 re-reads them only after all primes are written).
//  * corrections as unsigned min (s-p wraps above s when s<p): add = IADD3+IADD3+IMNMX, no ISETP/SEL;
//  * DIF butterfly y' = shoup(x+p-y, w): Shoup accepts any 32-bit input, so the difference needs no reduction;
//  * base-conversion reduction of x = Y0+Y1 2^8+Y2 2^16+Y3 2^24 (< 2^46) with a 32-bit Barrett step on
//    hi = floor-ish(x/2^14) and m = floor(2^62/p) = mu>>2: q in {floor(x/p)-1, floor(x/p)}, r = x-qp in [0,2p)
//    computed mod 2^32, fed unreduced to the Shoup twist multiply (any 32-bit input).
//  * TMA: each R-DIF column is written back densely (slot order) into its own 128 B aligned smem region and
//    stored with one 3-D tensor copy (box 8 words x 32 slot octets x 1 plane) by lane 0; before the next quad
//    overwrites X, lane 0 waits for the bulk reads (wait_group.read 0).
#if defined(BPT_ENABLE)&&defined(BPR_DT)
#ifndef BPO_XA
#define BPO_XA 0
#endif
#if defined(__CUDA_ARCH__)&&__CUDA_ARCH__>=900
#define BPO_X_HAS_TMA 1
#else
#define BPO_X_HAS_TMA 0
#endif
#ifndef BPO_XF
#define BPO_XF (BPO_X_HAS_TMA?(1|2|64|128|256|512|2048):1)   // flags 64/128 assume the TMA path; 256 coalesced input staging; 2048 evict_first stores
#endif
#ifndef BPO_XV
#define BPO_XV BPO_XF
#endif
struct __align__(64) XTMap{u64 w[16];};   // CUtensorMap (128 B) passed as __grid_constant__
__device__ __forceinline__ u32 xadd(u32 a,u32 b,u32 p){const u32 s=a+b;return min(s,s-p);}
__device__ __forceinline__ u32 xsub(u32 a,u32 b,u32 p){const u32 d=a-b;return min(d,d+p);}
__device__ __forceinline__ u32 xmsh(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));const u32 r=x*u32(c)-q*p;return min(r,r-p);}
__device__ __forceinline__ void xbf(u32& x,u32& y,u64 w,u32 p){const u32 a=x,b=y;x=xadd(a,b,p);y=xmsh(a+p-b,w,p);}
__device__ __forceinline__ void xbf1(u32& x,u32& y,u32 p){const u32 a=x,b=y;x=xadd(a,b,p);y=xsub(a,b,p);}   // twiddle psi^0
template<bool TRIV>__device__ __forceinline__ void xbft(u32& x,u32& y,const u64& w,u32 p){if(TRIV)xbf1(x,y,p);else xbf(x,y,w,p);}
// 8-point cyclic DIF over the digit axis (= BPO_JDIF): entry e of the stage table = psi^(2eN/256).
template<u32 F>__device__ __forceinline__ void xjdif(u32 (&x8)[8],const u64* st,u32 p){
    constexpr bool T=(F&1)!=0;
    xbft<T>(x8[0],x8[4],st[0],p);xbf(x8[1],x8[5],st[32],p);xbf(x8[2],x8[6],st[64],p);xbf(x8[3],x8[7],st[96],p);
    const u64 w0=st[0],w64=st[64];
    xbft<T>(x8[0],x8[2],w0,p);xbf(x8[1],x8[3],w64,p);xbft<T>(x8[4],x8[6],w0,p);xbf(x8[5],x8[7],w64,p);
    xbft<T>(x8[0],x8[1],w0,p);xbft<T>(x8[2],x8[3],w0,p);xbft<T>(x8[4],x8[5],w0,p);xbft<T>(x8[6],x8[7],w0,p);
}
// R-DIF of one (prime,digit) column in its 288-word skewed region + cross twiddle (ctab row from L1/L2).
template<u32 F>
__device__ __forceinline__ void xrdif(u32* const base0,const u32 so,const u64* st,const u64* ct,u32 p,u32* outp,u32 lane,const XTMap* tm,u32 c,u32 plane){
    constexpr bool T=(F&1)!=0;
    u32* const base=base0+so;   // skewed working layout; the TMA dense copy goes to base0 (128 B aligned)
    u32 v[8];
    const ulonglong2* c2=reinterpret_cast<const ulonglong2*>(ct+8*lane);
    ulonglong2 cw[4];
    if(F&16){
        #pragma unroll
        for(u32 m2=0;m2<4;++m2)cw[m2]=__ldg(c2+m2);   // cross twiddles issued early (L2 latency off the chain)
    }
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[lane+36*m];
    #pragma unroll
    for(u32 m=0;m<4;++m)xbf(v[m],v[m+4],st[lane+32u*m],p);
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=st[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh)xbf(v[qq+4*hh],v[qq+4*hh+2],w,p);}
    {const u64 w=st[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2)xbf(v[m],v[m+1],w,p);}
    #pragma unroll
    for(u32 m=0;m<8;++m)base[lane+36*m]=v[m];
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)xbf(v[lo],v[lo+4],st[224u+(lane&3u)+4*lo],p);
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);xbf(v[lo],v[lo+2],st[240u+(lane&3u)+4*(lo&1)],p);}
    {const u64 w=st[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)xbf(v[lo],v[lo+1],w,p);}
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    {const u64 w252=st[252u],w253=st[253u],w254=st[254u];   // st[252]=st[254]=psi^0
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);if(lo&1)xbf(v[lo],v[lo+2],w253,p);else xbft<T>(v[lo],v[lo+2],w252,p);}
    #pragma unroll
    for(u32 lo=0;lo<8;lo+=2)xbft<T>(v[lo],v[lo+1],w254,p);}
    #pragma unroll
    for(u32 m2=0;m2<4;++m2){const ulonglong2 w=(F&16)?cw[m2]:__ldg(c2+m2);v[2*m2]=xmsh(v[2*m2],u64(w.x),p);v[2*m2+1]=xmsh(v[2*m2+1],u64(w.y),p);}
#if BPO_X_HAS_TMA
    if(F&2){
        __syncwarp();   // layout-3 reads of the region done
        const uint4 e0=make_uint4(v[0],v[1],v[2],v[3]),e1=make_uint4(v[4],v[5],v[6],v[7]);
        const bool sw=(lane>>2)&1u;uint4* sd=reinterpret_cast<uint4*>(base0+8*lane);
        sd[sw?1:0]=sw?e1:e0;sd[sw?0:1]=sw?e0:e1;   // conflict-free quarter-warp phases
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        __syncwarp();
        if(lane==0&&BPO_XA!=1){const u32 sa=u32(__cvta_generic_to_shared(base0));
            asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane),"r"(sa):"memory");
            asm volatile("cp.async.bulk.commit_group;":::"memory");}
        return;
    }
#endif
    uint4* o=reinterpret_cast<uint4*>(outp+lane*2048u);
    if(BPO_XA==1&&(v[0]^v[1]^v[2]^v[3]^v[4]^v[5]^v[6]^v[7])!=0xffffffffu)return;   // ablation: compute only
    o[0]=make_uint4(v[0],v[1],v[2],v[3]);o[1]=make_uint4(v[4],v[5],v[6],v[7]);
}
// Flag 64 phase B: stages len 64..2 of one column (len 256/128 done in phase A). Layout 1: row = (lane&7)+8m+64(lane>>3)
// (regs = row bits 3..5), layout 2: row = 8 lane+m (regs = bits 0..2); skewed address row+4(row>>5).
template<u32 F,u32 NC>
__device__ __forceinline__ void xrdif6(u32* const (&base0)[NC],const u32 (&so)[NC],const u64* const (&st)[NC],const u64* const (&ct)[NC],const u32 (&pp)[NC],u32 lane,const XTMap* tm,u32 c,const u32 (&plane)[NC],u32 stm=0){
    constexpr bool T=(F&1)!=0;
    u32 v[NC][8];
    const u32 l7=lane&7u,b1=l7+72u*(lane>>3);   // skew(l7+64g) = l7+72g; +8m (m<4) / +8m+4 (m>=4)
    if(BPO_XA==5){   // ablation: memory only (TMA of the region as is)
#if BPO_X_HAS_TMA
        if(lane==0){
            #pragma unroll
            for(u32 n=0;n<NC;++n){const u32 sa=u32(__cvta_generic_to_shared(base0[n]));
                asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane[n]),"r"(sa):"memory");}
            asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
        return;
    }
    ulonglong2 cw[(F&16)?NC:1][4];
    if(F&16){   // cross twiddles issued first (L2 latency off the dependency chain)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const ulonglong2* c2=reinterpret_cast<const ulonglong2*>(ct[n]+8*lane);
            #pragma unroll
            for(u32 m2=0;m2<4;++m2)cw[(F&16)?n:0][m2]=__ldg(c2+m2);}
    }
    #pragma unroll
    for(u32 n=0;n<NC;++n)
        #pragma unroll
        for(u32 m=0;m<8;++m)v[n][m]=base0[n][so[n]+b1+8*m+4*(m>>2)];
    #pragma unroll
    for(u32 m=0;m<4;++m)
        #pragma unroll
        for(u32 n=0;n<NC;++n)xbf(v[n][m],v[n][m+4],st[n][192u+l7+8u*m],pp[n]);   // len 64: j = row&31
    #pragma unroll
    for(u32 qq=0;qq<2;++qq)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u64 w=st[n][224u+l7+8u*qq];     // len 32: j = row&15
            #pragma unroll
            for(u32 hh=0;hh<2;++hh)xbf(v[n][qq+4*hh],v[n][qq+4*hh+2],w,pp[n]);}
    #pragma unroll
    for(u32 n=0;n<NC;++n){const u64 w=st[n][240u+l7];                // len 16: j = row&7
        #pragma unroll
        for(u32 m=0;m<8;m+=2)xbf(v[n][m],v[n][m+1],w,pp[n]);}
    #pragma unroll
    for(u32 n=0;n<NC;++n)
        #pragma unroll
        for(u32 m=0;m<8;++m)base0[n][so[n]+b1+8*m+4*(m>>2)]=v[n][m];
    __syncwarp();
    #pragma unroll
    for(u32 n=0;n<NC;++n){const uint4* p4=reinterpret_cast<const uint4*>(base0[n]+so[n]+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[n][0]=x0.x;v[n][1]=x0.y;v[n][2]=x0.z;v[n][3]=x0.w;v[n][4]=x1.x;v[n][5]=x1.y;v[n][6]=x1.z;v[n][7]=x1.w;}
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)
        #pragma unroll
        for(u32 n=0;n<NC;++n){if(lo==0)xbft<T>(v[n][0],v[n][4],st[n][248u],pp[n]);else xbf(v[n][lo],v[n][lo+4],st[n][248u+lo],pp[n]);}   // len 8 (st[248]=psi^0)
    #pragma unroll
    for(u32 k=0;k<4;++k)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 lo=(k&1)+4*(k>>1);if(lo&1)xbf(v[n][lo],v[n][lo+2],st[n][253u],pp[n]);else xbft<T>(v[n][lo],v[n][lo+2],st[n][252u],pp[n]);}
    #pragma unroll
    for(u32 lo=0;lo<8;lo+=2)
        #pragma unroll
        for(u32 n=0;n<NC;++n)xbft<T>(v[n][lo],v[n][lo+1],st[n][254u],pp[n]);
    #pragma unroll
    for(u32 n=0;n<NC;++n){const ulonglong2* c2=reinterpret_cast<const ulonglong2*>(ct[n]+8*lane);
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){const ulonglong2 w=(F&16)?cw[(F&16)?n:0][m2]:__ldg(c2+m2);v[n][2*m2]=xmsh(v[n][2*m2],u64(w.x),pp[n]);v[n][2*m2+1]=xmsh(v[n][2*m2+1],u64(w.y),pp[n]);}}
#if BPO_X_HAS_TMA
    __syncwarp();
    #pragma unroll
    for(u32 n=0;n<NC;++n){
        const uint4 e0=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]),e1=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);
        const bool sw=(lane>>2)&1u;uint4* sd=reinterpret_cast<uint4*>(base0[n]+8*lane);
        sd[sw?1:0]=sw?e1:e0;sd[sw?0:1]=sw?e0:e1;}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncwarp();
    if(lane==0&&BPO_XA!=8){   // ablation 8: compute only (no output stores)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 sa=u32(__cvta_generic_to_shared(base0[n]));
            if((F&2048)&&stm!=1u){   // flag 2048: L2 evict_first policy (P1 output is re-read only after all primes are written);
                // stm (runtime, prime-group P1 whose output P2 reads next): 1 plain stores, 2 evict_last
                u64 pol;if(stm==2u)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
                asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%1,%2,%3}],[%4],%5;"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane[n]),"r"(sa),"l"(pol):"memory");
            }else
            asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane[n]),"r"(sa):"memory");}
        asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
}
template<u32 F>
__device__ __forceinline__ void bpo_first_xb(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 uq0,u32 uq1,u32 pfstep,const u64* ctab,
    const XTMap* tmA,const XTMap* tmB,u32* X,const u32 id,const u32 pfid,const bool lastseg){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr bool S6=(F&64)!=0;
    constexpr u32 XD=(F&96)?320u:288u,XS=(F&96)?8*320:8*288+32;   // 128 B aligned (prime,digit) regions
    const u32 stm=(pfstep>>29)&3u;const bool ldn=(pfstep>>28)&1u;pfstep&=0x0fffffffu;   // bits 29-30 TMA store policy (xrdif6 stm), 28 input loads normal (L2-retained for the next prime group)
    const u32 c=id&255u,op=id>>8;   // id: (operand, column) of this segment; quads [uq0,uq1); pfid: column to L2-prefetch (>=512 none)
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(c)*L;
    u32* dst=op?B:A;
    if(BPO_XA==6||BPO_XA==7){   // ablation: streaming reference (6: input column + 128 KB contiguous out, 7: out only)
        u32 acc=0;
        if(BPO_XA==6)for(u32 x=h;x<256u*L;x+=256){const u32 rr=x/L,w=x-rr*L;acc^=__ldg(src+u64(rr*256u)*L+w);}
        uint4* o=reinterpret_cast<uint4*>(dst+u64(c)*32768u);
        #pragma unroll 4
        for(u32 x=h;x<8192u;x+=256)o[x]=make_uint4(acc,x,acc^x,id);
        return;
    }
    constexpr u32 MT=(F&8)?1:2,WR=16*MT,NW=256/WR;   // m-tiles per warp, rows per warp, warps
    u32 af[8][MT][4];
    constexpr u32 XRS=60;   // flag 512 smem row stride (words): 15 aligned 16 B chunks, conflict-free fragment reads
    const u32 xoff=(c*L)&3u;   // flag 512: word offset of every row start in its 16 B chunk (256 L words per row step)
    const bool ca=(F&512)&&((reinterpret_cast<unsigned long long>(op?b:a)&15ull)==0);   // flag 512 needs a 16 B aligned operand base
    if((F&256)&&S6&&MT==2){   // flag 256: coalesced row loads of 128-row halves into X, fragments built from smem
        #pragma unroll
        for(u32 mt=0;mt<MT;++mt){
            if(mt)__syncthreads();
            const u32* hs=src+u64(mt*128u*256u)*L;
            if(ca){   // flag 512: all 16 B chunks of the half in flight at once (cp.async, one DRAM/L2 round trip)
                const u32 q=h&15u,nch=(xoff+L+3u)/4u;const u32* g0=hs-xoff+4u*q;
                u64 pol=0;if(!ldn)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
                if(q<nch){
                    #pragma unroll
                    for(u32 i=0;i<8;++i){const u32 rr=(h>>4)+16u*i;
                        const u32 sa=u32(__cvta_generic_to_shared(X+rr*XRS+4u*q));const u32* g=g0+u64(rr*256u)*L;
                        if(ldn)asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(g):"memory");
                        else asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(sa),"l"(g),"l"(pol):"memory");}}
                asm volatile("cp.async.commit_group;":::"memory");
                asm volatile("cp.async.wait_group 0;":::"memory");
            }else
            #pragma unroll 4
            for(u32 x=h;x<128u*L;x+=256u){const u32 rr=x/L,w=x-rr*L;X[x]=ldn?__ldg(hs+u64(rr*256u)*L+w):__ldcs(hs+u64(rr*256u)*L+w);}
            __syncthreads();
            #pragma unroll
            for(u32 j=0;j<8;++j){
                const u32 wb=POS[j]/32,sh=POS[j]%32;
                #pragma unroll
                for(u32 x=0;x<4;++x){
                    const u32 lrow=64*(x&1)+8*warp+gid,t=tig+4*(x>>1);
                    const u32* rp=ca?X+lrow*XRS+xoff:X+lrow*L;
                    const u32 w0=wb+t,w1=min(wb+t+1,L-1);
                    const u32 v=__funnelshift_r(rp[min(w0,L-1)],rp[w1],sh);
                    const int width=int(POS[j+1])-int(POS[j]+32*t);
                    af[j][mt][x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
                }
            }
        }
    }else
    #pragma unroll
    for(u32 j=0;j<8;++j){
        const u32 wb=POS[j]/32,sh=POS[j]%32;
        #pragma unroll
        for(u32 mt=0;mt<MT;++mt)
            #pragma unroll
            for(u32 x=0;x<4;++x){
                const u32 row=S6?64*(2*mt+(x&1))+8*warp+gid:WR*warp+16*mt+gid+8*(x&1),t=tig+4*(x>>1);
                const u32* rp=src+u64(row*256u)*L;
                const u32 w0=wb+t,w1=min(wb+t+1,L-1);
                const u32 v=__funnelshift_r(__ldg(rp+min(w0,L-1)),__ldg(rp+w1),sh);
                const int width=int(POS[j+1])-int(POS[j]+32*t);
                af[j][mt][x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
            }
    }
    if(pfid<512u&&h<256u){
        const u32 pf=pfid;const u32* ps=(pf>=256u?b:a)+u64(pf&255u)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    const u32 stride=weighted_root_stride(N);
    u64* STS=reinterpret_cast<u64*>(X+4*XS);   // flag 4: the quad's stage tables (4 x 256 u64)
    #pragma unroll 1
    for(u32 u=uq0;u<uq1;++u){
        __syncthreads();   // previous quad's R-DIF reads of X (and STS) done
        if(F&4){
            #pragma unroll
            for(u32 q=0;q<4;++q)if(h<256u)STS[q*256u+h]=stage_tw[size_t(4*u+q)*256+h];
        }
        if(BPO_XA!=4&&BPO_XA!=5){   // ablation 4/5: no phase A
            const u32 r=4*u+tig,p=bpr_pr[r];const u32 m62=u32(bpr_mu[r]>>2);   // floor(2^62/p)
            const u64* tw=twist+size_t(r)*stride;const u64* st=stage_tw+size_t(r)*256;
            u32 res[2*MT][8];   // [row 2mt+hf][digit], unreduced in [0,2p)
            #pragma unroll
            for(u32 j=0;j<8;++j){
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+u);
                #pragma unroll
                for(u32 mt=0;mt<MT;++mt){
                    u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                    pimma16832(c0,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                    pimma16832(c1,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);
                    #pragma unroll
                    for(u32 hf=0;hf<2;++hf){
                        const u32 y0=c0[2*hf],y1=c0[2*hf+1],y2=c1[2*hf],y3=c1[2*hf+1];   // each < 2^21
                        const u32 lo=y0+(y1<<8)+(y2<<16)+(y3<<24);
                        const u32 hi=(y0>>14)+(y1>>6)+(y2<<2)+(y3<<10);
                        const u32 q=__umulhi(hi,m62)>>16;
                        res[2*mt+hf][j]=lo-q*p;
                    }
                }
            }
            #pragma unroll
            if(S6){   // rows 64rr+8w+gid: twist, J-DIF, R-DIF stages len 256 (rr^2) and 128 (rr^1) in registers
                const u32 rl=8*warp+gid;
                #pragma unroll
                for(u32 rr=0;rr<4;++rr){
                    const u64 t2=tw[1+128u+(((64*rr+rl)*256u+c)>>7)];
                    #pragma unroll
                    for(u32 j=0;j<8;++j)res[rr][j]=xmsh(res[rr][j],t2,p);
                    xjdif<F>(res[rr],st,p);
                }
                const u64 wa=st[rl],wb=st[64u+rl],wc=st[128u+rl];
                #pragma unroll
                for(u32 j=0;j<8;++j){xbf(res[0][j],res[2][j],wa,p);xbf(res[1][j],res[3][j],wb,p);xbf(res[0][j],res[1][j],wc,p);xbf(res[2][j],res[3][j],wc,p);}
                #pragma unroll
                for(u32 rr=0;rr<4;++rr){const u32 row=64*rr+rl;
                    #pragma unroll
                    for(u32 j=0;j<8;++j)X[tig*XS+j*XD+8*tig+row+4*(row>>5)]=res[rr][j];}
            }else
            for(u32 rr=0;rr<2*MT;++rr){
                const u32 row=WR*warp+8*rr+gid;
                const u64 t2=tw[1+128u+((row*256u+c)>>7)];
                #pragma unroll
                for(u32 j=0;j<8;++j)res[rr][j]=xmsh(res[rr][j],t2,p);
                xjdif<F>(res[rr],st,p);
                #pragma unroll
                for(u32 j=0;j<8;++j)X[tig*XS+j*XD+((F&32)?8*tig:0)+row+4*(row>>5)]=res[rr][j];
            }
        }
        __syncthreads();
        #pragma unroll 1
        for(u32 n=0;n<(BPO_XA==3?0u:32/NW);++n){   // (prime q, digit j) columns of the quad (ablation 3: none)
            const u32 ci=warp+NW*n,j=ci&7u,q=ci>>3,r=4*u+q;
            if(S6&&!(F&128)){u32* const bs[1]={X+q*XS+j*XD};const u32 so[1]={8*q};const u64* const sts[1]={(F&4)?STS+q*256u:stage_tw+size_t(r)*256};
                const u64* const cts[1]={ctab+(size_t(r)*256u+c)*256u};const u32 ps[1]={bpr_pr[r]},pl[1]={r*8+j};
                xrdif6<F,1>(bs,so,sts,cts,ps,lane,op?tmB:tmA,c,pl,stm);}else
            if(S6){if(n&1)continue;   // pairs (n, n+1)
                const u32 ci2=ci+NW,j2=ci2&7u,q2=ci2>>3,r2=4*u+q2;
                u32* const bs[2]={X+q*XS+j*XD,X+q2*XS+j2*XD};const u32 so[2]={8*q,8*q2};
                const u64* const sts[2]={(F&4)?STS+q*256u:stage_tw+size_t(r)*256,(F&4)?STS+q2*256u:stage_tw+size_t(r2)*256};
                const u64* const cts[2]={ctab+(size_t(r)*256u+c)*256u,ctab+(size_t(r2)*256u+c)*256u};const u32 ps[2]={bpr_pr[r],bpr_pr[r2]},pl[2]={r*8+j,r2*8+j2};
                xrdif6<F,2>(bs,so,sts,cts,ps,lane,op?tmB:tmA,c,pl,stm);}else
            xrdif<F>(X+q*XS+j*XD,(F&32)?8*q:0,(F&4)?STS+q*256u:stage_tw+size_t(r)*256,ctab+(size_t(r)*256u+c)*256u,bpr_pr[r],dst+u64(r*8+j)*N+c*8u,lane,op?tmB:tmA,c,r*8+j);
        }
#if BPO_X_HAS_TMA
        if((F&2)&&lane==0&&BPO_XA!=9)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");   // X reusable (ablation 9: no wait)
#endif
    }
#if BPO_X_HAS_TMA
    if((F&2)&&lane==0&&lastseg)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
#endif
}
// Grid: 512 blocks = one (operand, column) each, all quads [r0/4,r1/4); or (flag-free runtime choice) a persistent grid
// G < 512: the 512*nq (column, quad) items split evenly into G contiguous ranges, so G = SMs x blocks/SM fills
// exactly one wave (no 1.36-wave tail); each column segment of a range restages its input (A fragments).
template<u32 F_>
__device__ __forceinline__ void bpo_first_xg(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const XTMap* tmA,const XTMap* tmB,u32* X){
    const u32 G=gridDim.x,q0=r0/4,nq=(r1-r0)/4,ps=pfstep&0x0fffffffu;
    if(G>=512u){const u32 id=blockIdx.x;
        bpo_first_xb<F_>(a,b,A,B,twist,stage_tw,bpt,N,q0,q0+nq,pfstep,ctab,tmA,tmB,X,id,ps?id+ps:~0u,true);return;}
    // whole columns first (column t*G+b in round t: concurrent blocks read adjacent input columns, which share
    // L2 lines), then the R = 512 mod G leftover columns' R*nq items split evenly into G contiguous ranges.
    const u32 b0=blockIdx.x,F=512u/G,R=512u-F*G,ext=R*nq,i0=u32((u64(b0)*ext)/G),i1=u32((u64(b0+1)*ext)/G);
    #pragma unroll 1
    for(u32 t=0;t<F;++t){const u32 col=t*G+b0;
        if(t)__syncthreads();
        bpo_first_xb<F_>(a,b,A,B,twist,stage_tw,bpt,N,q0,q0+nq,pfstep,ctab,tmA,tmB,X,col,ps?(t+1<F?col+G:(i0<i1?F*G+i0/nq:~0u)):~0u,i0==i1&&t+1==F);}
    #pragma unroll 1
    for(u32 i=i0;i<i1;){
        const u32 col=F*G+i/nq,ie=min(i1,(i/nq+1)*nq);
        if(F||i!=i0)__syncthreads();   // previous segment's phase-B reads of X done before the input restaging
        bpo_first_xb<F_>(a,b,A,B,twist,stage_tw,bpt,N,q0+(i%nq),q0+(ie-(i/nq)*nq),pfstep,ctab,tmA,tmB,X,col,(ps&&ie<i1)?col+1:~0u,ie==i1);
        i=ie;
    }
}
#define BPO_X_SMEM(F) (4*(((F)&96)?8*320:8*288+32)*4+(((F)&4)?4*256*8:0))
#if !defined(BPO_LEAN)||BPO_LEAN_P1X
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) bpo_first_x(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];
    bpo_first_xg<BPO_XF>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#endif // bpo_first_x
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__((BPO_XV&8)?512:256,BPO_TC_MINB) bpo_first_xv(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XV)/4];
    bpo_first_xg<BPO_XV>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#endif // bpo_first_xv
#endif
