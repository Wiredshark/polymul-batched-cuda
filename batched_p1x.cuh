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
//        4096 lean base conversion (x = w + 2^16 z, hi = floor(x/2^14) exact, r = lo + q(-p) with -p from a constant table)
//             and phase-A stores / cp.async / fragment reads through hoisted pointers with immediate offsets.
//        8192 (with 64) phase B: warp w owns prime w>>1, digits 4(w&1)..+3 in groups of BPO_X_NC columns that share
//             the stage and cross twiddles (xrdif6s); len-2 psi^0 stage fused into the cross twiddle (with 1).
//        16384 (with 8192) L1 prefetch of the warp's phase-B cross twiddles at the quad start (off: no gain measured).
//        32768 (with 64, without 4) quad barrier just before the phase-A stores: phase-A math overlaps the previous quad's phase B.
//        65536 (with 8192) phase-B TMA staging stores without the e0/e1 value swap (2-way bank conflict instead of 16 SELs per pair).
//        131072 (with 8192, default) phase B in one pass per warp (xrdif6q): one fence + one 4-box TMA burst per quad, twiddles
//             once per quad (sweep a26c406 "pbq", on target 0.938x P1, exact); 65536 then only affects the xrdif6s path.
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
#ifndef BPO_MIXED
#define BPO_MIXED 0   // 1 with the h30 prime set (primes < 2^30 present: per-prime P1 Barrett shift)
#endif
// Diagnostic ablations in template flag bits >= 20 (BPO_DIAG kernels / exit-time sweep only; production F has none set;
// XABL 12 = no phase A + no stores, 13 = no phase A + no phase B: sweep round 5 compute-only breakdown):
// bits 20-23 override BPO_XA, bit 24 inputs from column 0 of a (L2-resident).
#define XABL(F) ((((F)>>20)&15u)?(((F)>>20)&15u):u32(BPO_XA))
#define XDIN(F) (((F)>>24)&1u)
// Sweep-only variant bits (exit-time sweep round 2; production BPO_XF sets none of them):
//   bit 28 XCPQ input staging in 64-row quarters, two cp.async groups in flight (fragments of quarter k built while k+1 lands)
//   bit 29 XTWH phase-A twist / J-DIF / R-DIF-256/128 twiddles loaded before the MMA digit loop (latency off the chain)
//   bit 30 XMNV phase-A MMAs as non-volatile asm (ptxas may interleave them with the reduction / integer work)
//   bit 31 XQ4  pbq phase B layout-2 stages on all 4 columns at once (no rolled pair loop, 4x8 live values)
//   1024        pbq phase B layout-1 groups as a rolled loop too (smaller code; sweep a26c406 "r1" on xrdif6q)
#define XCPQ(F) (((F)>>28)&1u)
#define XTWH(F) (((F)>>29)&1u)
#define XMNV(F) (((F)>>30)&1u)
#define XQ4(F) (((F)>>31)&1u)
#if defined(__CUDA_ARCH__)&&__CUDA_ARCH__>=900
#define BPO_X_HAS_TMA 1
#else
#define BPO_X_HAS_TMA 0
#endif
#ifndef BPO_XF
#ifndef BPO_X_NC
#define BPO_X_NC 2
#endif
#define BPO_XF (BPO_X_HAS_TMA?(1|2|64|128|256|512|2048|4096|8192|131072):1)   // flags 64/128 assume the TMA path; 256 coalesced input staging; 2048 evict_first stores
#endif
#ifndef BPO_XV
#define BPO_XV BPO_XF
#endif
#ifndef BPO_XG
#define BPO_XG 0u   // second template word of bpo_first_x (production 0; FHERMA_EXTRA_DEFINES test hook for sweep variants, e.g. lw = 512u with BPO_XF|32768)
#endif
struct __align__(64) XTMap{u64 w[16];};
struct XNegP{u32 v[BCRT_K];};
__host__ __device__ constexpr XNegP xnegp_make(){XNegP a{};constexpr u32 P[BCRT_K]={BCRT_PRIMES};for(u32 i=0;i<BCRT_K;++i)a.v[i]=0u-P[i];return a;}
__constant__ XNegP bpr_np=xnegp_make();
struct XFMask{u32 v[8][2][4];};   // fragment word masks [digit j][x>>1][tig]: width POS[j+1]-POS[j]-32(tig+4(x>>1)) clamped to [0,32]
__host__ __device__ constexpr XFMask xfmask_make(){XFMask a{};constexpr u32 P[9]={BCRT_POS,BCRT_W};
    for(u32 j=0;j<8;++j)for(u32 xh=0;xh<2;++xh)for(u32 t=0;t<4;++t){const int wd=int(P[j+1])-int(P[j])-32*int(t+4*xh);
        a.v[j][xh][t]=wd>=32?~0u:(wd<=0?0u:((1u<<wd)-1u));}return a;}
__constant__ XFMask bpr_fmask=xfmask_make();   // -p mod 2^32 as an opaque constant: r = lo + q(-p) is one IMAD (ptxas re-derives -q from 0-p)
   // CUtensorMap (128 B) passed as __grid_constant__
__device__ __forceinline__ void pimma16832nv(u32 (&c)[4],u32 a0,u32 a1,u32 a2,u32 a3,u32 b0,u32 b1){   // sweep XMNV: schedulable
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ u32 xadd(u32 a,u32 b,u32 p){const u32 s=a+b;return min(s,s-p);}
__device__ __forceinline__ u32 xsub(u32 a,u32 b,u32 p){const u32 d=a-b;return min(d,d+p);}
__device__ __forceinline__ u32 xmsh(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));const u32 r=x*u32(c)-q*p;return min(r,r-p);}
__device__ __forceinline__ void xbf(u32& x,u32& y,u64 w,u32 p){const u32 a=x,b=y;x=xadd(a,b,p);y=xmsh(a+p-b,w,p);}
__device__ __forceinline__ void xbf1(u32& x,u32& y,u32 p){const u32 a=x,b=y;x=xadd(a,b,p);y=xsub(a,b,p);}   // twiddle psi^0
template<bool TRIV>__device__ __forceinline__ void xbft(u32& x,u32& y,const u64& w,u32 p){if(TRIV)xbf1(x,y,p);else xbf(x,y,w,p);}
// Round 11 (p1o) Harvey-lazy butterflies (as h30 2297ca6 / h30s 246d72e; H only in sweep instantiations, production H=false
// reduces to xbf / xbf1 / xmsh): words in [0,2p) between stages (needs p < 2^30 for exact values; with the production primes the
// H instantiations are timing twins with wrong values by design). hbf: x' = min(a+b, a+b-2p), y' = Shoup(a+2p-b) unreduced.
__device__ __forceinline__ u32 xmshl(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));return x*u32(c)-q*p;}
// 2p and -2p come from constant tables (opaque to ptxas): with 2u*p in registers ptxas rematerialises every s-2p as IMAD
// p*-2+s and a+2p-b as IMAD p*2+a (measured in the round-11 SASS: the in-situ Harvey P1 had +221 IMAD, -320 VIADDMNMX, which is
// why the pv49 h30 P1 lost 1.049 while the register-only probe gained 1.094); x' = min(s, s+(-2p)) is then one VIADDMNMX.
__device__ __forceinline__ u32 xaddh(u32 a,u32 b,u32 n2){const u32 s=a+b;return min(s,s+n2);}
template<bool H>__device__ __forceinline__ void xbfh(u32& x,u32& y,u64 w,u32 p,u32 p2,u32 n2=0){if(H){const u32 a=x,b=y;x=xaddh(a,b,n2);y=xmshl(a+p2-b,w,p);}else xbf(x,y,w,p);}
template<bool H>__device__ __forceinline__ void xbf1h(u32& x,u32& y,u32 p,u32 p2,u32 n2=0){if(H){const u32 a=x,b=y;x=xaddh(a,b,n2);y=xaddh(a,p2-b,n2);}else xbf1(x,y,p);}
template<bool TRIV,bool H>__device__ __forceinline__ void xbfth(u32& x,u32& y,const u64& w,u32 p,u32 p2,u32 n2=0){if(TRIV)xbf1h<H>(x,y,p,p2,n2);else xbfh<H>(x,y,w,p,p2,n2);}
template<bool H>__device__ __forceinline__ u32 xmshh(u32 x,u64 c,u32 p){return H?xmshl(x,c,p):xmsh(x,c,p);}
#ifdef BPO_X_QL
__host__ __device__ constexpr XNegP xp2_make(bool neg){XNegP a{};constexpr u32 P[BCRT_K]={BCRT_PRIMES};for(u32 i=0;i<BCRT_K;++i)a.v[i]=neg?0u-2u*P[i]:2u*P[i];return a;}
__constant__ XNegP bpr_p2=xp2_make(false);__constant__ XNegP bpr_n2=xp2_make(true);
#define BPR_P2(r) bpr_p2.v[r]
#define BPR_N2(r) bpr_n2.v[r]
#else
#define BPR_P2(r) (2u*bpr_pr[r])
#define BPR_N2(r) (0u-2u*bpr_pr[r])
#endif
struct XQS{static constexpr bool v=false;};struct XQS2{static constexpr bool v=false;};struct XQH{static constexpr bool v=true;};   // quad-body tags (BPO_X_QL sweep module)
// 8-point cyclic DIF over the digit axis (= BPO_JDIF): entry e of the stage table = psi^(2eN/256).
template<u32 F,bool H=false>__device__ __forceinline__ void xjdif(u32 (&x8)[8],const u64* st,u32 p,u32 p2=0,u32 n2=0){
    constexpr bool T=(F&1)!=0;
    xbfth<T,H>(x8[0],x8[4],st[0],p,p2,n2);xbfh<H>(x8[1],x8[5],st[32],p,p2,n2);xbfh<H>(x8[2],x8[6],st[64],p,p2,n2);xbfh<H>(x8[3],x8[7],st[96],p,p2,n2);
    const u64 w0=st[0],w64=st[64];
    xbfth<T,H>(x8[0],x8[2],w0,p,p2,n2);xbfh<H>(x8[1],x8[3],w64,p,p2,n2);xbfth<T,H>(x8[4],x8[6],w0,p,p2,n2);xbfh<H>(x8[5],x8[7],w64,p,p2,n2);
    xbfth<T,H>(x8[0],x8[1],w0,p,p2,n2);xbfth<T,H>(x8[2],x8[3],w0,p,p2,n2);xbfth<T,H>(x8[4],x8[5],w0,p,p2,n2);xbfth<T,H>(x8[6],x8[7],w0,p,p2,n2);
}
template<u32 F,bool H=false>__device__ __forceinline__ void xjdifv(u32 (&x8)[8],const u64 w0,const u64 w32,const u64 w64,const u64 w96,u32 p,u32 p2=0,u32 n2=0){   // = xjdif, twiddles in registers
    constexpr bool T=(F&1)!=0;
    xbfth<T,H>(x8[0],x8[4],w0,p,p2,n2);xbfh<H>(x8[1],x8[5],w32,p,p2,n2);xbfh<H>(x8[2],x8[6],w64,p,p2,n2);xbfh<H>(x8[3],x8[7],w96,p,p2,n2);
    xbfth<T,H>(x8[0],x8[2],w0,p,p2,n2);xbfh<H>(x8[1],x8[3],w64,p,p2,n2);xbfth<T,H>(x8[4],x8[6],w0,p,p2,n2);xbfh<H>(x8[5],x8[7],w64,p,p2,n2);
    xbfth<T,H>(x8[0],x8[1],w0,p,p2,n2);xbfth<T,H>(x8[2],x8[3],w0,p,p2,n2);xbfth<T,H>(x8[4],x8[5],w0,p,p2,n2);xbfth<T,H>(x8[6],x8[7],w0,p,p2,n2);
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
        if(lane==0&&XABL(F)!=1){const u32 sa=u32(__cvta_generic_to_shared(base0));
            asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane),"r"(sa):"memory");
            asm volatile("cp.async.bulk.commit_group;":::"memory");}
        return;
    }
#endif
    uint4* o=reinterpret_cast<uint4*>(outp+lane*2048u);
    if(XABL(F)==1&&(v[0]^v[1]^v[2]^v[3]^v[4]^v[5]^v[6]^v[7])!=0xffffffffu)return;   // ablation: compute only
    o[0]=make_uint4(v[0],v[1],v[2],v[3]);o[1]=make_uint4(v[4],v[5],v[6],v[7]);
}
// Flag 64 phase B: stages len 64..2 of one column (len 256/128 done in phase A). Layout 1: row = (lane&7)+8m+64(lane>>3)
// (regs = row bits 3..5), layout 2: row = 8 lane+m (regs = bits 0..2); skewed address row+4(row>>5).
template<u32 F,u32 NC>
__device__ __forceinline__ void xrdif6(u32* const (&base0)[NC],const u32 (&so)[NC],const u64* const (&st)[NC],const u64* const (&ct)[NC],const u32 (&pp)[NC],u32 lane,const XTMap* tm,u32 c,const u32 (&plane)[NC],u32 stm=0){
    constexpr bool T=(F&1)!=0;
    u32 v[NC][8];
    const u32 l7=lane&7u,b1=l7+72u*(lane>>3);   // skew(l7+64g) = l7+72g; +8m (m<4) / +8m+4 (m>=4)
    if(XABL(F)==5){   // ablation: memory only (TMA of the region as is)
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
    if(lane==0&&XABL(F)!=8){   // ablation 8: compute only (no output stores)
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
// Flag 8192 phase B: NC columns of ONE prime (digits j..j+NC-1): stage twiddles and cross twiddles are loaded once and
// shared by the NC columns; smem addresses as one hoisted pointer + immediates. Same values as xrdif6.
template<u32 F,u32 NC>
__device__ __forceinline__ void xrdif6s(u32* const base0,const u32 so,const u64* st,const u64* ct,const u32 p,u32 lane,const XTMap* tm,u32 c,const u32 plane0,u32 stm=0){
    constexpr bool T=(F&1)!=0;
    constexpr u32 XD=320u;
    u32 v[NC][8];
    const u32 l7=lane&7u;
    {u32* const pb=base0+so+l7+72u*(lane>>3);   // skew(l7+64g) = l7+72g; +8m (m<4) / +8m+4 (m>=4)
        #pragma unroll
        for(u32 n=0;n<NC;++n)
            #pragma unroll
            for(u32 m=0;m<8;++m)v[n][m]=pb[n*XD+8*m+4*(m>>2)];
        u64 w64[4],w32[2];
        #pragma unroll
        for(u32 m=0;m<4;++m)w64[m]=st[192u+l7+8u*m];
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)w32[qq]=st[224u+l7+8u*qq];
        const u64 w16=st[240u+l7];
        #pragma unroll
        for(u32 m=0;m<4;++m)
            #pragma unroll
            for(u32 n=0;n<NC;++n)xbf(v[n][m],v[n][m+4],w64[m],p);   // len 64: j = row&31
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)
            #pragma unroll
            for(u32 n=0;n<NC;++n)
                #pragma unroll
                for(u32 hh=0;hh<2;++hh)xbf(v[n][qq+4*hh],v[n][qq+4*hh+2],w32[qq],p);   // len 32: j = row&15
        #pragma unroll
        for(u32 n=0;n<NC;++n)
            #pragma unroll
            for(u32 m=0;m<8;m+=2)xbf(v[n][m],v[n][m+1],w16,p);   // len 16: j = row&7
        #pragma unroll
        for(u32 n=0;n<NC;++n)
            #pragma unroll
            for(u32 m=0;m<8;++m)pb[n*XD+8*m+4*(m>>2)]=v[n][m];
    }
    __syncwarp();
    {const u32* const pr=base0+so+8*lane+4*(lane>>2);
        #pragma unroll
        for(u32 n=0;n<NC;++n){const uint4* p4=reinterpret_cast<const uint4*>(pr+n*XD);const uint4 x0=p4[0],x1=p4[1];
            v[n][0]=x0.x;v[n][1]=x0.y;v[n][2]=x0.z;v[n][3]=x0.w;v[n][4]=x1.x;v[n][5]=x1.y;v[n][6]=x1.z;v[n][7]=x1.w;}}
    {u64 w8[4];
        #pragma unroll
        for(u32 lo=0;lo<4;++lo)w8[lo]=st[248u+lo];
        const u64 w253=st[253u],w252=st[252u],w254=st[254u];
        #pragma unroll
        for(u32 lo=0;lo<4;++lo)
            #pragma unroll
            for(u32 n=0;n<NC;++n){if(lo==0)xbft<T>(v[n][0],v[n][4],w8[0],p);else xbf(v[n][lo],v[n][lo+4],w8[lo],p);}   // len 8 (st[248]=psi^0)
        #pragma unroll
        for(u32 k=0;k<4;++k)
            #pragma unroll
            for(u32 n=0;n<NC;++n){const u32 lo=(k&1)+4*(k>>1);if(lo&1)xbf(v[n][lo],v[n][lo+2],w253,p);else xbft<T>(v[n][lo],v[n][lo+2],w252,p);}
        if(!T){
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)
            #pragma unroll
            for(u32 n=0;n<NC;++n)xbf(v[n][lo],v[n][lo+1],w254,p);}}
    {const ulonglong2* c2=reinterpret_cast<const ulonglong2*>(ct+8*lane);
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){const ulonglong2 w=__ldg(c2+m2);
            #pragma unroll
            for(u32 n=0;n<NC;++n){const u32 a=v[n][2*m2],b=v[n][2*m2+1];
                if(T){v[n][2*m2]=xmsh(a+b,u64(w.x),p);v[n][2*m2+1]=xmsh(a+p-b,u64(w.y),p);}   // len-2 psi^0 stage fused into the cross twiddle
                else{v[n][2*m2]=xmsh(a,u64(w.x),p);v[n][2*m2+1]=xmsh(b,u64(w.y),p);}}}}
#if BPO_X_HAS_TMA
    __syncwarp();
    {const bool sw=(lane>>2)&1u;uint4* const sd=reinterpret_cast<uint4*>(base0+8*lane)+(sw?1:0);   // conflict-free quarter-warp phases
        #pragma unroll
        for(u32 n=0;n<NC;++n){
            const uint4 e0=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]),e1=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);
            uint4* const d=sd+n*(XD/4);
            if(F&65536){uint4* const d0=reinterpret_cast<uint4*>(base0+8*lane)+n*(XD/4);d0[0]=e0;d0[1]=e1;}   // flag 65536: no value swap (2-way bank conflict, no SELs)
            else{d[0]=sw?e1:e0;d[sw?-1:1]=sw?e0:e1;}}}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncwarp();
    if(lane==0&&XABL(F)!=8&&XABL(F)!=11){
        u64 pol=0;const bool hint=(F&2048)&&stm!=1u;
        if(hint){if(stm==2u)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));}
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 sa=u32(__cvta_generic_to_shared(base0+n*XD));
            if(hint)asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%1,%2,%3}],[%4],%5;"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa),"l"(pol):"memory");
            else asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa):"memory");}
        asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
}
__device__ __forceinline__ ulonglong2 xct_ldv(const u64* ctg,u32 lane,u32 m2){   // cross twiddles 2 m2, 2 m2+1 of a lane, never CSE'd across pairs (registers)
    ulonglong2 w;
    asm volatile("ld.global.nc.v2.u64 {%0,%1},[%2];":"=l"(w.x),"=l"(w.y):"l"(ctg+8*lane+2*m2));
    return w;
}
// Flag 131072 (with 8192) phase B, xrdif6q (sweep a26c406 "pbq"; on target 0.938x P1 time, exact): the warp's 4 columns
// in ONE pass: layout-1 stages of both column pairs, one __syncwarp, layout-2 stages + cross twiddle per pair (rolled pair
// loop), one proxy fence and one lane-0 TMA burst (4 boxes, 1 commit) per quad instead of 2 x (fence, syncwarp, TMA
// section); stage and cross twiddles loaded once per quad, not per pair. Same values as xrdif6s (flag 65536 not used here).
// Sweep ablations of the pbq phase B (wrong values by design): bit 18 XNTW stage twiddles from registers (no loads), bit 27
// XNCT cross twiddles from registers (no loads). Production F sets neither (loads unchanged).
#define XNTW(F) (((F)>>18)&1u)
#define XNCT(F) (((F)>>27)&1u)
#define XR16(F) (((F)>>25)&3u)   // sweep phase-B modes (xrdif16 / cts, see below)
#define XCA(F) (XR16(F)==1?4u:XR16(F)==2?16u:0u)   // phase-A store offset of odd-digit columns (words)
template<u32 F>__device__ __forceinline__ u64 xstw(const u64* st,u32 i){if constexpr(XNTW(F)!=0)return (u64(0x9e3779b9u*(i+1u))<<32)|u64(i*0x85ebca6bu);else return st[i];}
// Round 23 (kill, exit-time sweep only; production K = 0 compiles none of it) kill-test ablations of the pbq phase B, wrong
// values by design: K 1 (nostage) no layout-1 len-16 stage (its 8Q butterflies per pair and the w16 twiddle load);
// K 2 (notma) no dense staging: the layout-2 results go back in place to the lane's own skewed working-layout words (same
// STS.128 count, no cross-lane hazard: no per-pair __syncwarp, no sw swizzle) and the TMA boxes read base0 + n XD as
// production (same bytes stored, stale / mixed words); K 3 (notma_bar) dense staging kept, only the per-pair __syncwarp removed.
template<u32 F,bool H=false,u32 XW=0,u32 K=0>   // XW (sweep XG_XW, round 16 p1b): output run length L (TMA coords (0, c, (256/L) plane)); 0 = production map
__device__ __forceinline__ void xrdif6q(u32* const base0,const u32 so,const u64* st,const u64* ct,const u32 p,u32 lane,const XTMap* tm,u32 c,const u32 plane0,u32 stm,u32 p2=0,u32 n2=0){
    constexpr bool T=(F&1)!=0;
    constexpr u32 XD=320u;
    constexpr u32 Q=2u,NI=4u/Q;   // column pairs
    const u32 l7=lane&7u;
    {u64 w64[4],w32[2];
        #pragma unroll
        for(u32 m=0;m<4;++m)w64[m]=xstw<F>(st,192u+l7+8u*m);
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)w32[qq]=xstw<F>(st,224u+l7+8u*qq);
        const u64 w16=K==1u?0ull:xstw<F>(st,240u+l7);
        u32* const pb=base0+so+l7+72u*(lane>>3);
        auto l1=[&](const u32 pr){u32 v[Q][8];
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)v[n][m]=pb[(Q*pr+n)*XD+8*m+4*(m>>2)];
            #pragma unroll
            for(u32 m=0;m<4;++m)
                #pragma unroll
                for(u32 n=0;n<Q;++n)xbfh<H>(v[n][m],v[n][m+4],w64[m],p,p2,n2);
            #pragma unroll
            for(u32 qq=0;qq<2;++qq)
                #pragma unroll
                for(u32 n=0;n<Q;++n)
                    #pragma unroll
                    for(u32 hh=0;hh<2;++hh)xbfh<H>(v[n][qq+4*hh],v[n][qq+4*hh+2],w32[qq],p,p2,n2);
            if constexpr(K!=1u){
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;m+=2)xbfh<H>(v[n][m],v[n][m+1],w16,p,p2,n2);}
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)pb[(Q*pr+n)*XD+8*m+4*(m>>2)]=v[n][m];
        };
        if(F&1024u){   // sweep flag 1024: layout-1 groups as a rolled loop
            #pragma unroll 1
            for(u32 pr=0;pr<NI;++pr)l1(pr);
        }else{
        #pragma unroll
        for(u32 pr=0;pr<NI;++pr){l1(pr);if(pr+1<NI)__syncwarp();}}   // groups kept sequential (8Q live values)
    }
    __syncwarp();
    u64 w8[4];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)w8[lo]=xstw<F>(st,248u+lo);
    const u64 w253=xstw<F>(st,253u),w252=xstw<F>(st,252u),w254=xstw<F>(st,254u);
    const u32* const pr0=base0+so+8*lane+4*(lane>>2);
    const bool sw=(lane>>2)&1u;uint4* const sd=reinterpret_cast<uint4*>(base0+8*lane)+(sw?1:0);
    constexpr u32 Q2=XQ4(F)?4u:Q,NI2=4u/Q2;   // sweep bit 31: all 4 columns in one layout-2 pass
    #pragma unroll 1
    for(u32 pr=0;pr<NI2;++pr){u32 v[Q2][8];
        #pragma unroll
        for(u32 n=0;n<Q2;++n){const uint4* p4=reinterpret_cast<const uint4*>(pr0+(Q2*pr+n)*XD);const uint4 x0=p4[0],x1=p4[1];
            v[n][0]=x0.x;v[n][1]=x0.y;v[n][2]=x0.z;v[n][3]=x0.w;v[n][4]=x1.x;v[n][5]=x1.y;v[n][6]=x1.z;v[n][7]=x1.w;}
        #pragma unroll
        for(u32 lo=0;lo<4;++lo)
            #pragma unroll
            for(u32 n=0;n<Q2;++n){if(lo==0)xbfth<T,H>(v[n][0],v[n][4],w8[0],p,p2,n2);else xbfh<H>(v[n][lo],v[n][lo+4],w8[lo],p,p2,n2);}
        #pragma unroll
        for(u32 k=0;k<4;++k)
            #pragma unroll
            for(u32 n=0;n<Q2;++n){const u32 lo=(k&1)+4*(k>>1);if(lo&1)xbfh<H>(v[n][lo],v[n][lo+2],w253,p,p2,n2);else xbfth<T,H>(v[n][lo],v[n][lo+2],w252,p,p2,n2);}
        if(!T){
            #pragma unroll
            for(u32 lo=0;lo<8;lo+=2)
                #pragma unroll
                for(u32 n=0;n<Q2;++n)xbfh<H>(v[n][lo],v[n][lo+1],w254,p,p2,n2);}
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){ulonglong2 w;if constexpr(XNCT(F)!=0){w.x=(u64(lane*0x27d4eb2fu+m2)<<32)|u64(p^m2);w.y=w.x^0x165667b1u;}else if constexpr(XR16(F)==3)w=*reinterpret_cast<const ulonglong2*>(ct+2u*(4u*lane+((m2+(lane>>1))&3u)));else w=xct_ldv(ct,lane,m2);
            #pragma unroll
            for(u32 n=0;n<Q2;++n){const u32 a=v[n][2*m2],b=v[n][2*m2+1];
                if(T){v[n][2*m2]=xmsh(a+b,u64(w.x),p);v[n][2*m2+1]=xmsh(a+(H?p2:p)-b,u64(w.y),p);}
                else{v[n][2*m2]=xmsh(a,u64(w.x),p);v[n][2*m2+1]=xmsh(b,u64(w.y),p);}}}
#if BPO_X_HAS_TMA
        if constexpr(K==2u){   // kill notma: in place, own words only
            #pragma unroll
            for(u32 n=0;n<Q2;++n){uint4* const p4=reinterpret_cast<uint4*>(const_cast<u32*>(pr0)+(Q2*pr+n)*XD);
                p4[0]=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]);p4[1]=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);}
        }else{
        if constexpr(K!=3u)__syncwarp();   // layout-2 reads of this pair's regions done (the other pair's regions are disjoint)
        #pragma unroll
        for(u32 n=0;n<Q2;++n){
            const uint4 e0=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]),e1=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);
            uint4* const d=sd+(Q2*pr+n)*(XD/4);
            d[0]=sw?e1:e0;d[sw?-1:1]=sw?e0:e1;}}
#endif
    }
#if BPO_X_HAS_TMA
    if constexpr(XABL(F)!=14)asm volatile("fence.proxy.async.shared::cta;":::"memory");   // ladder XABL 14: compute only without the proxy fence
    __syncwarp();
    if(lane==0&&XABL(F)!=8&&XABL(F)!=11&&XABL(F)!=12&&XABL(F)!=14){
        u64 pol=0;const bool hint=(F&2048)&&stm!=1u;
        if(hint){if(stm==2u)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));}
        #pragma unroll
        for(u32 n=0;n<4;++n){const u32 sa=u32(__cvta_generic_to_shared(base0+n*XD));
            if(hint)asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%1,%2,%3}],[%4],%5;"::"l"(reinterpret_cast<u64>(tm)),"r"(XW?0u:c*8u),"r"(XW?c:0u),"r"(XW?(256u/(XW?XW:1u))*(plane0+n):plane0+n),"r"(sa),"l"(pol):"memory");
            else asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(XW?0u:c*8u),"r"(XW?c:0u),"r"(XW?(256u/(XW?XW:1u))*(plane0+n):plane0+n),"r"(sa):"memory");}
        asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
}
// Sweep-only radix-16 phase B (XR16 = template bits 25-26; production BPO_XF sets neither, its code generation is untouched):
// one column per half-warp (lanes 16hw..16hw+15 own column 2pr+hw of the warp's 4 digits, two passes), 16 values per lane.
//   XR16 1 "b16": phase A as production (R-DIF len 256/128; hw=1 columns at +4 words). B1 rows (l&3)+4m+64(l>>2) (regs =
//     row bits 2..5): stages len 64,32,16,8 (lane twiddles), stored out of place in a line-skewed layout
//     pos(r) = r+4((L+2)>>2) (L = r>>4; hw=1 columns at +16 words). B2 rows 16l+4c+i (regs = bits 0..3; chunk c = (k+rot)&3,
//     rot = (l>>1)&3, so loads and the dense staging stores are conflict-free): stage len 4, len 2 fused into the cross twiddle.
//   XR16 2 "r16a": phase A without R stages (MMA, twist, J-DIF; 352-word regions, hw=1 columns at +16). B1 rows l+16m (regs =
//     bits 4..7): stages len 256..32 in place (lane twiddles); B2 rows 16l+m (regs = bits 0..3): stages len 16,8,4 (uniform
//     twiddles), len 2 fused into the cross twiddle; dense staging stores unrotated (4-way conflicts).
//   XR16 3 "cts" (pbq phase B, not xrdif16): the quad's cross-twiddle rows are staged in smem by cp.async at the quad start
//     (+8 KB, 48 KB static), phase B reads them with conflict-free LDS.128 instead of L2/L1 loads.
// Bank-conflict-free patterns checked offline. Exchanges per quad as production: phase A -> B1 (block barrier), B1 -> B2
// (__syncwarp), B2 -> TMA staging.
template<u32 F>
__device__ __forceinline__ void xrdif16(u32* const base0,const u32 so,const u64* st,const u64* ct,const u32 p,u32 lane,const XTMap* tm,u32 c,const u32 plane0,u32 stm){
    constexpr bool T=(F&1)!=0;
    constexpr u32 MODE=XR16(F);
    constexpr u32 XD=MODE==2?352u:320u;
    const u32 hw=lane>>4,l=lane&15u,rot=(l>>1)&3u;
    #pragma unroll 1
    for(u32 pr=0;pr<2;++pr){
        u32* const cb=base0+(2u*pr+hw)*XD;   // this half-warp's column
        u32 v[16];
        if constexpr(MODE==1){
            const u32* const pa=cb+so+4u*hw+(l&3u)+72u*(l>>2);   // skew(r) = r+4(r>>5)
            #pragma unroll
            for(u32 m=0;m<16;++m)v[m]=pa[4*m+4*(m>>3)];
            const u64* const sl=st+(l&3u);
            #pragma unroll
            for(u32 s=0;s<4;++s){const u32 nt=8u>>s,h=4u*nt;   // len 2h = 64..8: pairs (m, m+nt), twiddle st[256-2h+(l&3)+4(m mod nt)]
                u64 tw[8];
                #pragma unroll
                for(u32 t=0;t<8;++t)if(t<nt)tw[t]=sl[256u-2u*h+4u*t];
                #pragma unroll
                for(u32 m=0;m<16;++m)if(!(m&nt))xbf(v[m],v[m+nt],tw[m&(nt-1u)],p);}
            __syncwarp();   // B1 reads of the column done (the store is out of place)
            u32* const pb=cb+so+16u*hw+(l&3u)+68u*(l>>2);   // pos(r) for r = (l&3)+4m+64(l>>2)
            #pragma unroll
            for(u32 m=0;m<16;++m)pb[4*m+4*(m>>3)]=v[m];
        }else{
            u32* const pa=cb+so+16u*hw+l;
            #pragma unroll
            for(u32 m=0;m<16;++m)v[m]=pa[16*m+4*(m>>1)];
            const u64* const sl=st+l;
            #pragma unroll
            for(u32 s=0;s<4;++s){const u32 nt=8u>>s,h=16u*nt;   // len 2h = 256..32: twiddle st[256-2h+l+16(m mod nt)]
                u64 tw[8];
                #pragma unroll
                for(u32 t=0;t<8;++t)if(t<nt)tw[t]=sl[256u-2u*h+16u*t];
                #pragma unroll
                for(u32 m=0;m<16;++m)if(!(m&nt))xbf(v[m],v[m+nt],tw[m&(nt-1u)],p);}
            #pragma unroll
            for(u32 m=0;m<16;++m)pa[16*m+4*(m>>1)]=v[m];   // in place
        }
        __syncwarp();
        u32 cc[4];   // dense chunk held in register chunk k
        #pragma unroll
        for(u32 k=0;k<4;++k)cc[k]=MODE==1?((k+rot)&3u):k;
        {const u32* const pc=MODE==1?cb+so+16u*hw+16u*l+4u*((l+2u)>>2):cb+so+16u*hw+16u*l+4u*(l>>1);
            #pragma unroll
            for(u32 k=0;k<4;++k){const uint4 x=*reinterpret_cast<const uint4*>(pc+4u*cc[k]);v[4*k]=x.x;v[4*k+1]=x.y;v[4*k+2]=x.z;v[4*k+3]=x.w;}}
        if constexpr(MODE==2){
            #pragma unroll
            for(u32 s=0;s<2;++s){const u32 nt=s?4u:8u,tb=s?248u:240u;   // len 16, 8 (twiddles uniform: row&7, row&3 = m&7, m&3)
                u64 tw[8];
                #pragma unroll
                for(u32 t=0;t<8;++t)if(t<nt)tw[t]=st[tb+t];
                #pragma unroll
                for(u32 m=0;m<16;++m)if(!(m&nt)){const u32 t=m&(nt-1u);if(s==1&&t==0)xbft<T>(v[m],v[m+nt],tw[0],p);else xbf(v[m],v[m+nt],tw[t],p);}}   // st[248]=psi^0
        }
        {const u64 w252=st[252u],w253=st[253u];   // len 4 (st[252]=psi^0)
            #pragma unroll
            for(u32 m=0;m<16;++m)if(!(m&2u)){if(m&1u)xbf(v[m],v[m+2],w253,p);else xbft<T>(v[m],v[m+2],w252,p);}}
        if(!T){const u64 w254=st[254u];
            #pragma unroll
            for(u32 m=0;m<16;m+=2)xbf(v[m],v[m+1],w254,p);}
        {const u64* const cl=ct+16u*l;   // cross twiddles (len-2 psi^0 stage fused with flag 1)
            #pragma unroll
            for(u32 k=0;k<4;++k)
                #pragma unroll
                for(u32 h2=0;h2<2;++h2){const ulonglong2 w=__ldg(reinterpret_cast<const ulonglong2*>(cl+4u*cc[k]+2u*h2));
                    const u32 m=4*k+2*h2;const u32 a=v[m],b=v[m+1];
                    if(T){v[m]=xmsh(a+b,u64(w.x),p);v[m+1]=xmsh(a+p-b,u64(w.y),p);}else{v[m]=xmsh(a,u64(w.x),p);v[m+1]=xmsh(b,u64(w.y),p);}}}
#if BPO_X_HAS_TMA
        __syncwarp();   // B2 reads of this column done
        u32* const sd=cb+16u*l;
        #pragma unroll
        for(u32 k=0;k<4;++k)*reinterpret_cast<uint4*>(sd+4u*cc[k])=make_uint4(v[4*k],v[4*k+1],v[4*k+2],v[4*k+3]);
        // (r16a: the stages on bits 2, 3 forbid the chunk rotation: 4-way conflicts; a 32-SEL rotation spilled 12 B and was slower locally)
#endif
    }
#if BPO_X_HAS_TMA
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncwarp();
    if(lane==0&&XABL(F)!=8&&XABL(F)!=11){
        u64 pol=0;const bool hint=(F&2048)&&stm!=1u;
        if(hint){if(stm==2u)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));}
        #pragma unroll
        for(u32 n=0;n<4;++n){const u32 sa=u32(__cvta_generic_to_shared(base0+n*XD));
            if(hint)asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%1,%2,%3}],[%4],%5;"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa),"l"(pol):"memory");
            else asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa):"memory");}
        asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
}
// Sweep round 6 (prologue / staging / fragment / MMA / reduction split): second template word G (production G = 0, its
// code generation is untouched). Ablations (wrong values, compute-only twins in the sweep):
//   G 1   XG_NST  no input staging (no cp.async issued: fragments built from smem garbage)
//   G 2   XG_NFB  no fragment build (af = thread-varying constants; the staging still lands and is waited for)
//   G 4   XG_NMMA no MMA (accumulators = af ^ bv: same reduction work on unbounded values)
//   G 8   XG_NRED no reduction to residues (res = y0^y1^y2^y3)
// Variants (exact):
//   G 16  XG_ALN  one alignment path: cp.async staging assumed (operand base 16 B aligned; production checks and compiles
//                 the coalesced-LDG fallback too)
//   G 32  XG_WA   word-aligned digit windows: fragment word t of digit j = row word POS[j]/32+t, masked to the digit's bits
//                 (plain LDS + LOP, no funnel shift). Window value = digit * 2^(POS[j]%32): the sweep passes a B table whose
//                 digit factor carries 2^-(POS[j]%32) (same digit values and positions, same byte bounds: exact)
//   G 128 XG_WG   (with XG_WA) fragments loaded straight from global memory (one LDG per fragment word into af, all 64 in
//                 flight; no smem staging, no cp.async wait, no barriers before the quad loop; L2 policy as production)
//   G 256 XG_PF2  L2 prefetch of the second input half's rows right after the first half's cp.async (its DRAM fetch overlaps
//                 the first half's wait + fragment build)
//   G 64  XG_RED5 base-conversion reduction in 5 ops: x = w + 2^16 z, q = umulhi(z, floor(2^48/p)) (w/p + z/2^32 < 1 for
//                 2^30 < p < 2^31: q in {floor(x/p)-1, floor(x/p)}, r = x-qp in [0,2p) as production; twist output canonical)
#define XG_NST(G) ((G)&1u)
#define XG_NFB(G) (((G)>>1)&1u)
#define XG_NMMA(G) (((G)>>2)&1u)
#define XG_NRED(G) (((G)>>3)&1u)
#define XG_ALN(G) (((G)>>4)&1u)
#define XG_WA(G) (((G)>>5)&1u)
#define XG_RED5(G) (((G)>>6)&1u)
#define XG_WG(G) (((G)>>7)&1u)
#define XG_PF2(G) (((G)>>8)&1u)
// Sweep round 7 (p1x): G 512 XG_LW (with F flag 32768 = LB): late TMA drain wait. Production waits (lane 0, wait_group.read 0)
//   at the END of each quad, before the next quad's phase-A math; with LB the only consumer of that wait is the barrier just
//   before the next quad's phase-A stores into X, so XG_LW moves the wait there: quad u's bulk smem reads (store drain) overlap
//   quad u+1's MMA / reduction / twist / J-DIF / R-DIF-256/128 math. Exact (X is rewritten only after every warp's lane 0 waited
//   and the block barrier); the last quad's drain is covered by the final wait_group 0 as in production.
#define XG_LW(G) (((G)>>9)&1u)
// Round 11 (p1o) Harvey / code-size bits (sweep only; production G = 0 -> H = false, one quad body, code generation untouched):
//   G 1024 XG_H    Harvey-lazy P1 math (twist, J-DIF, R-DIF len 256/128, phase B stages; cross twiddle canonical): with the
//                  production primes (> 2^30) a timing twin, values wrong by design (exact only for p < 2^30)
//   G 2048 XG_HQ3  (BPO_X_QL module) quads u < 3 with the G-selected arithmetic, quad 3 canonical Shoup: two quad bodies
//                  (the h30 12 + 4 prime structure)
//   G 4096 XG_DUP  (BPO_X_QL module) two Shoup quad bodies (u < 3 / u = 3): exact, isolates the code-size effect of HQ3
#define XG_H(G) (((G)>>10)&1u)
#define XG_HQ3(G) (((G)>>11)&1u)
#define XG_DUP(G) (((G)>>12)&1u)
//   G 8192 XG_PFA  L1 prefetch of the NEXT quad's phase-A tables at the start of phase B (twist rows 64rr+rl of prime 4(u+1)+tig,
//                  R-DIF st[rl], st[64+rl], st[128+rl], J-DIF st[32e]): no registers held (twh hoisting holds 22), exact. The
//                  production SASS loads the 4 twist rows serially into one register pair after the MMAs (L2 latency per row
//                  per quad: the tables of a new prime quad are L1-cold)
#define XG_PFA(G) (((G)>>13)&1u)
// Round 13 (ladder) morph-ladder bits (sweep only; production G = 0 compiles none of them): steps between the isolated quad-math
// probe (diag_p1q.h pq_ab) and the compute-only production P1 (sweep "nomem" = XABL 8 + XG_NST), one difference per bit:
//   G 16384  XL_SS   steady-state addressing: r1 may exceed 16 (prime quad uq = u&3, c-dependent table rows (twist, cross twiddle)
//                    at column (c+7(u>>2))&255, B fragments of quad uq): the sweep runs one full wave (2 blocks/SM) for many quads
//                    and differences two lengths (no prologue / tail in the rate); with r1 = 16 the addresses are production's
//   G 32768  XL_NPF  no next-block L2 prefetch (pfstep), no bulk-group drain waits; with XABL 14 also no proxy fence (xrdif6q)
//   G 65536  XL_HOT  twist and stage tables of prime quad 0 for every quad (L1-resident as in the probe; p and the cross twiddles stay)
//   G 131072 XL_NBC  no base conversion: no B-fragment loads, no MMA, no reduction; res = af words (af opaque per quad: stays live)
//   G 262144 XL_NAF  (with NBC) no af registers at all: res = per-quad hash (as the probe's register-resident values)
//   G 524288 XL_NPRO no prologue: no staging wait / fragment build / barriers before the quad loop (af = hash when still used)
#define XL_SS(G) (((G)>>14)&1u)
#define XL_NPF(G) (((G)>>15)&1u)
#define XL_HOT(G) (((G)>>16)&1u)
#define XL_NBC(G) (((G)>>17)&1u)
#define XL_NAF(G) (((G)>>18)&1u)
#define XL_NPRO(G) (((G)>>19)&1u)
//   G 1048576 XL_NBV no B-fragment loads (bv = per-quad register hash; MMA and reduction kept)
#define XL_NBV(G) (((G)>>20)&1u)
//   G 8388608 XL_NOBAR no block barriers in the quad loop (warp syncs only; the probe's ab_nobar twin)
#define XL_NOBAR(G) (((G)>>23)&1u)
// Round 13 fix prototypes (exact; sweep only, production G = 0):
//   G 2097152 XF_TWJ twist inside the base-conversion digit loop: the quad's 4 twist rows are loaded before the MMAs and each
//                    residue is twisted as soon as its reduction is done (independent mulmods under the LDG -> MMA -> reduction
//                    latency chain; the rr loop keeps J-DIF and the R stages). Same values.
//   G 4194304 XF_BVH B fragments of digits 0-1 loaded before the quad barrier (their latency under the barrier wait)
#define XF_TWJ(G) (((G)>>21)&1u)
#define XF_BVH(G) (((G)>>22)&1u)
//   G 16777216 XP_PERS (set by bpo_first_xs only) column-op index from the wrapper instead of blockIdx.x
#define XP_PERS(G) (((G)>>24)&1u)
//   G bits 25-30 XG_XW (round 16 p1b, sweep only; production G = 0): P1 output layout with row runs of L = XG_XW words per column:
//                  plane word (row/L)*256L + c*L + row%L (production = L 8 with the old 2048 x 32 map); TMA box {L, 1, 256/L}
//                  (host map State::p1x_map with ovr_xw = L); consumer = sweep P2 bpo_middle_r16w<L> (16L threads, L slots/block)
#define XG_XW(G) (((G)>>25)&63u)
template<u32 G>__device__ __forceinline__ u32 xl_hotq(u32 u){   // XL_HOT: table quad 0 through an opaque per-quad zero (loads stay in the loop)
    if constexpr(XL_HOT(G)!=0){u32 z;asm volatile("mov.u32 %0, 0;":"=r"(z));return z;}else return u;}
struct XFMaskW{u32 v[8][2][4];};   // XG_WA word masks [digit j][xh][tig]: bits [POS[j]%32-32t, POS[j]%32+D-32t) of word t = tig+4xh
__host__ __device__ constexpr XFMaskW xfmaskw_make(){XFMaskW a{};constexpr u32 P[9]={BCRT_POS,BCRT_W};
    for(u32 j=0;j<8;++j)for(u32 xh=0;xh<2;++xh)for(u32 t=0;t<4;++t){const int sh=int(P[j]%32u),e=sh+int(P[j+1])-int(P[j]),w0=32*int(t+4*xh);
        const int lo=sh-w0<0?0:(sh-w0>32?32:sh-w0),hi=e-w0<0?0:(e-w0>32?32:e-w0);
        const u32 mh=hi>=32?~0u:((1u<<hi)-1u),ml=lo>=32?~0u:((1u<<lo)-1u);a.v[j][xh][t]=hi>lo?(mh&~ml):0u;}return a;}
__constant__ XFMaskW bpr_fmaskw=xfmaskw_make();
__host__ __device__ constexpr bool xwa_full(u32 j,u32 xh){const XFMaskW m=xfmaskw_make();for(u32 t=0;t<4;++t)if(m.v[j][xh][t]!=~0u)return false;return true;}
__host__ __device__ constexpr bool xred5_ok(){constexpr u32 P[BCRT_K]={BCRT_PRIMES};for(u32 i=0;i<BCRT_K;++i)if(P[i]<=(1u<<30)||P[i]>=(1u<<31))return false;return true;}
template<u32 F,u32 G=0,u32 K=0>   // K: round-23 kill ablation of the pbq phase B (xrdif6q K; production 0)
__device__ __forceinline__ void bpo_first_xb(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const XTMap* tmA,const XTMap* tmB,u32* X,u32 idov=0){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr bool S6=(F&64)!=0;
    constexpr u32 XD=XR16(F)==2?352u:((F&96)?320u:288u),XS=XR16(F)==2?8*352:((F&96)?8*320:8*288+32);   // 128 B aligned (prime,digit) regions
    const u32 stm=(pfstep>>29)&3u;const bool ldn=(pfstep>>28)&1u;pfstep&=0x0fffffffu;   // bits 29-30 TMA store policy (xrdif6 stm), 28 input loads normal (L2-retained for the next prime group)
    const u32 id=XP_PERS(G)?idov:blockIdx.x,c=id&255u,op=id>>8;   // XP_PERS: column-op from the re-split wrapper (bpo_first_xs)
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(XABL(F)==11?0u:c)*L;if(XDIN(F))src=a;   // ablation 11: compute proxy (column-0 input, L2-resident; no stores)
    u32* dst=op?B:A;
    if(XABL(F)==6||XABL(F)==7){   // ablation: streaming reference (6: input column + 128 KB contiguous out, 7: out only)
        u32 acc=0;
        if(XABL(F)==6)for(u32 x=h;x<256u*L;x+=256){const u32 rr=x/L,w=x-rr*L;acc^=__ldg(src+u64(rr*256u)*L+w);}
        uint4* o=reinterpret_cast<uint4*>(dst+u64(c)*32768u);
        #pragma unroll 4
        for(u32 x=h;x<8192u;x+=256)o[x]=make_uint4(acc,x,acc^x,id);
        return;
    }
    constexpr u32 MT=(F&8)?1:2,WR=16*MT,NW=256/WR;   // m-tiles per warp, rows per warp, warps
    u32 af[8][MT][4];
    constexpr u32 XRS=60;   // flag 512 smem row stride (words): 15 aligned 16 B chunks, conflict-free fragment reads
    const u32 xoff=XDIN(F)?0u:((XABL(F)==11?0u:c)*L)&3u;   // flag 512: word offset of every row start in its 16 B chunk (256 L words per row step)
    const bool ca=(F&512)&&(XG_ALN(G)||((reinterpret_cast<unsigned long long>(op?b:a)&15ull)==0));   // XG_ALN: one path   // flag 512 needs a 16 B aligned operand base
    bool cpqd=false;   // sweep XCPQ path taken (if constexpr: production code generation untouched)
    if constexpr(XCPQ(F)&&(F&256)&&(F&512)&&(F&4096)&&S6&&MT==2)if(ca){   // sweep XCPQ: 64-row quarters, one cp.async group in flight while building
        cpqd=true;
        u32 mk[8][2];
        #pragma unroll
        for(u32 j=0;j<8;++j)
            #pragma unroll
            for(u32 xh=0;xh<2;++xh)mk[j][xh]=bpr_fmask.v[j][xh][tig];
        const u32 q=h&15u,nch=(xoff+L+3u)/4u;
        u64 pol=0;if(!ldn)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        auto issue=[&](const u32 k){   // quarter k = rows 64(k&1)..+63 of half k>>1 -> X rows 64(k&1)..
            if(q<nch){const u32* const g0=src+u64((k>>1)*128u*256u)*L-xoff+4u*q+u64((64u*(k&1u)+(h>>4))*256u)*L;
                const u32 sa0=u32(__cvta_generic_to_shared(X+(64u*(k&1u)+(h>>4))*XRS+4u*q));
                #pragma unroll
                for(u32 i=0;i<4;++i){const u32 sa=sa0+i*(16u*XRS*4u);const u32* g=g0+i*(16u*256u*L);
                    if(ldn)asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(g):"memory");
                    else asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(sa),"l"(g),"l"(pol):"memory");}}
            asm volatile("cp.async.commit_group;":::"memory");};
        auto build=[&](const u32 k){const u32 mt=k>>1,xo=k&1u;const u32* const rpt=X+(8*warp+gid)*XRS+xoff+tig;
            #pragma unroll
            for(u32 j=0;j<8;++j){
                const u32 wb=POS[j]/32,sh=POS[j]%32;
                const int D=int(POS[j+1])-int(POS[j]);
                #pragma unroll
                for(u32 xh=0;xh<2;++xh){
                    const u32* rp=rpt+64*xo*XRS+wb+4*xh;
                    const u32 v=__funnelshift_r(rp[0],rp[1],sh);
                    if(D-32*(3+4*int(xh))>=32)af[j][mt][xo+2*xh]=v;else af[j][mt][xo+2*xh]=v&mk[j][xh];}}};
        issue(0);issue(1);
        asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();build(0);__syncthreads();issue(2);
        asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();build(1);__syncthreads();issue(3);
        asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();build(2);
        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();build(3);
    }
    if constexpr(XL_NPRO(G)!=0){   // ladder: no prologue
        if constexpr(!XL_NAF(G)){
        #pragma unroll
        for(u32 j=0;j<8;++j)
            #pragma unroll
            for(u32 mt=0;mt<MT;++mt)
                #pragma unroll
                for(u32 x=0;x<4;++x)af[j][mt][x]=(h*0x9e3779b9u)^(0x01010101u*(8u*j+x+1u)+mt);}
    }else
    if(cpqd){}else
    if constexpr(XG_WG(G)!=0&&XG_WA(G)!=0&&(F&256)!=0&&(F&4096)!=0&&S6&&MT==2){   // sweep wg: word-aligned windows by LDG
        u64 pol=0;if(!ldn)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        const u32* const rp0=src+u64((8u*warp+gid)*256u)*L+tig;
        #pragma unroll
        for(u32 mt=0;mt<MT;++mt)
            #pragma unroll
            for(u32 j=0;j<8;++j)
                #pragma unroll
                for(u32 x=0;x<4;++x){const u32* const g=rp0+u64((64u*(2u*mt+(x&1u)))*256u)*L+POS[j]/32u+4u*(x>>1);u32 v;
                    if(ldn)asm("ld.global.nc.u32 %0,[%1];":"=r"(v):"l"(g));
                    else asm("ld.global.nc.L2::cache_hint.u32 %0,[%1],%2;":"=r"(v):"l"(g),"l"(pol));
                    af[j][mt][x]=v;}
        #pragma unroll
        for(u32 mt=0;mt<MT;++mt)
            #pragma unroll
            for(u32 j=0;j<8;++j)
                #pragma unroll
                for(u32 x=0;x<4;++x)if(!xwa_full(j,x>>1))af[j][mt][x]&=bpr_fmaskw.v[j][x>>1][tig];
    }else
    if((F&256)&&S6&&MT==2){   // flag 256: coalesced row loads of 128-row halves into X, fragments built from smem
        u32 mk[8][2];   // flag 4096: per-thread masks of the partial fragment words (clamped funnel shift: 2^wd-1, 0 or ~0)
        #pragma unroll
        for(u32 j=0;j<8;++j)
            #pragma unroll
            for(u32 xh=0;xh<2;++xh){const int wd=int(POS[j+1])-int(POS[j])-32*int(tig+4*xh);
                (void)wd;mk[j][xh]=bpr_fmask.v[j][xh][tig];}   // opaque constants: ptxas otherwise rebuilds per-fragment SELs
        #pragma unroll
        for(u32 mt=0;mt<MT;++mt){
            if(mt)__syncthreads();
            const u32* hs=src+u64(mt*128u*256u)*L;
            if(ca){   // flag 512: all 16 B chunks of the half in flight at once (cp.async, one DRAM/L2 round trip)
                const u32 q=h&15u,nch=(xoff+L+3u)/4u;const u32* g0=hs-xoff+4u*q;
                u64 pol=0;if(!ldn)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
                if(!XG_NST(G)&&q<nch){
                    const u32 sa0=u32(__cvta_generic_to_shared(X+(h>>4)*XRS+4u*q));const u32* const gq=g0+u64((h>>4)*256u)*L;
                    #pragma unroll
                    for(u32 i=0;i<8;++i){
                        const u32 sa=(F&4096)?sa0+i*(16u*XRS*4u):u32(__cvta_generic_to_shared(X+((h>>4)+16u*i)*XRS+4u*q));
                        const u32* g=(F&4096)?gq+i*(16u*256u*L):g0+u64(((h>>4)+16u*i)*256u)*L;
                        if(ldn)asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(g):"memory");
                        else asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(sa),"l"(g),"l"(pol):"memory");}}
                asm volatile("cp.async.commit_group;":::"memory");
                if constexpr(XG_PF2(G)!=0)if(mt==0){   // sweep pf2: second half's rows -> L2 while the first half lands
                    const u32* const ps=src+u64((128u+(h&127u))*256u)*L;
                    if(h<128u){asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));}
                    else asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));}
                asm volatile("cp.async.wait_group 0;":::"memory");
            }else
            #pragma unroll 4
            for(u32 x=h;x<128u*L;x+=256u){const u32 rr=x/L,w=x-rr*L;X[x]=ldn?__ldg(hs+u64(rr*256u)*L+w):__ldcs(hs+u64(rr*256u)*L+w);}
            __syncthreads();
            if((F&4096)&&ca){
                static_assert(3u+POS[7]/32u+8u<XRS,"flag 4096: unclamped fragment windows stay inside the 60-word row");
                const u32* const rpt=X+(8*warp+gid)*XRS+xoff+tig;
                if constexpr(XG_NFB(G)!=0){   // ablation: no fragment build
                    #pragma unroll
                    for(u32 j=0;j<8;++j)
                        #pragma unroll
                        for(u32 x=0;x<4;++x)af[j][mt][x]=(h*0x9e3779b9u)^(0x01010101u*(8u*j+x+1u)+mt);
                }else if constexpr(XG_WA(G)!=0){   // word-aligned windows (B table carries 2^-(POS[j]%32))
                    static_assert(3u+POS[7]/32u+8u<=XRS,"XG_WA: windows stay inside the 60-word row");
                    #pragma unroll
                    for(u32 j=0;j<8;++j){
                        const u32 wb=POS[j]/32;
                        #pragma unroll
                        for(u32 x=0;x<4;++x){
                            const u32 v=rpt[64*(x&1)*XRS+wb+4*(x>>1)];
                            if(xwa_full(j,x>>1))af[j][mt][x]=v;else af[j][mt][x]=v&bpr_fmaskw.v[j][x>>1][tig];}
                    }
                }else
                #pragma unroll
                for(u32 j=0;j<8;++j){
                    const u32 wb=POS[j]/32,sh=POS[j]%32;
                    const int D=int(POS[j+1])-int(POS[j]);
                    #pragma unroll
                    for(u32 x=0;x<4;++x){
                        const u32* rp=rpt+64*(x&1)*XRS+wb+4*(x>>1);
                        const u32 v=__funnelshift_r(rp[0],rp[1],sh);
                        if(D-32*(3+4*int(x>>1))>=32)af[j][mt][x]=v;   // full word for every tig
                        else af[j][mt][x]=v&mk[j][x>>1];
                    }
                }
            }else
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
    if(!XL_NPF(G)&&pfstep&&h<256u&&id+pfstep<512u){
        const u32 pf=id+pfstep;const u32* ps=(pf>=256u?b:a)+u64(pf&255u)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    const u32 stride=weighted_root_stride(N);
    u64* STS=reinterpret_cast<u64*>(X+4*XS);   // flag 4: the quad's stage tables (4 x 256 u64)
    #pragma unroll 1
    // flag 32768 (S6, without 4): the quad barrier moves from the loop top to just before the phase-A stores into X, so a
    // warp's phase-A math (MMA, reduction, twist, J-DIF, R-DIF 256/128) overlaps the other warps' phase B of the previous quad.
    constexpr bool LB=(F&32768)&&!(F&4)&&S6&&XABL(F)!=4&&XABL(F)!=5;
    for(u32 u=r0/4;u<r1/4;++u){
#ifdef BPO_X_QL
      auto quad=[&](auto hv){constexpr bool H=decltype(hv)::v;   // sweep module only: the quad body as a generic lambda (HQ3 / DUP: two bodies)
#else
      {constexpr bool H=XG_H(G)!=0;
#endif
        const u32 uq=XL_SS(G)?(u&3u):u,cq=XL_SS(G)?((c+7u*(u>>2))&255u):c;   // ladder XL_SS: prime quad / table column (= u, c in production)
        uint4 bvh[2];   // fix XF_BVH
        if constexpr(XF_BVH(G)!=0){
            #pragma unroll
            for(u32 j=0;j<2;++j)bvh[j]=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+uq);}
        if constexpr(XL_NOBAR(G)==0)
        if(!LB)__syncthreads();   // previous quad's R-DIF reads of X (and STS) done
        if constexpr(XR16(F)==3&&!(F&524288u)){   // sweep cts (bit 19: ablation, no staging: reads garbage): the quad's cross-twiddle rows (4 primes x 2 KB) -> smem by cp.async, chunk k = 4L+m2 at 4L+((m2+(L>>1))&3)
            u64* const cts=reinterpret_cast<u64*>(X+4*XS);
            #pragma unroll
            for(u32 i=0;i<2;++i){const u32 g=h+256u*i,qq=g>>7,k=g&127u,L=k>>2,m2=k&3u;
                const u64* const src=ctab+(size_t(4*u+qq)*256u+c)*256u+2u*k;
                const u32 sa=u32(__cvta_generic_to_shared(cts+qq*256u+2u*(4u*L+((m2+(L>>1))&3u))));
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(src):"memory");}
            asm volatile("cp.async.commit_group;":::"memory");
        }
        if(F&4){
            #pragma unroll
            for(u32 q=0;q<4;++q)if(h<256u)STS[q*256u+h]=stage_tw[size_t(4*u+q)*256+h];
        }
        if((F&16384)&&(F&8192)&&S6&&NW==8){   // flag 16384: phase-B cross twiddles of this warp's prime pulled into L1 now (L2 latency off phase B)
            const u64* const ctp=ctab+(size_t(4*u+(warp>>1))*256u+c)*256u+8u*lane;
            asm volatile("prefetch.global.L1 [%0];"::"l"(ctp));
        }
        if(XABL(F)!=4&&XABL(F)!=5&&XABL(F)!=12&&XABL(F)!=13){   // ablation 4/5: no phase A (12: and no stores, 13: no phase A, no phase B)
            const u32 r=4*uq+tig,p=bpr_pr[r];const u32 msh=(BPO_MIXED&&p<(1u<<30))?3u:2u;const u32 m62=u32(bpr_mu[r]>>msh),qsh=18u-msh;   // floor(2^(64-msh)/p) < 2^32 (BPO_MIXED: h30 prime set)
            const u32 np=(F&4096)?bpr_np.v[r]:0u-p;
            const u32 m48=XG_RED5(G)?u32(bpr_mu[r]>>16):0u;   // floor(2^48/p)
            const u32 rt=XL_HOT(G)?4u*xl_hotq<G>(u)+tig:r;   // ladder XL_HOT: prime-quad-0 tables
            const u64* tw=twist+size_t(rt)*stride;const u64* st=stage_tw+size_t(rt)*256;
            u32 res[2*MT][8];   // [row 2mt+hf][digit], unreduced in [0,2p)
            u64 hw[4+4+3];   // sweep XTWH: twist (4 rows), J-DIF st[0,32,64,96], R-DIF st[rl], st[64+rl], st[128+rl]
            if(XTWH(F)&&S6){const u32 rl=8*warp+gid;
                #pragma unroll
                for(u32 rr=0;rr<4;++rr)hw[rr]=tw[1+128u+(((64*rr+rl)*256u+cq)>>7)];
                #pragma unroll
                for(u32 e=0;e<4;++e)hw[4+e]=st[32u*e];
                hw[8]=st[rl];hw[9]=st[64u+rl];hw[10]=st[128u+rl];}
            u64 tj[4];   // fix XF_TWJ: the quad's twist rows, loaded before the base conversion
            if constexpr(XF_TWJ(G)!=0&&S6&&MT==2){const u32 rl=8*warp+gid;
                #pragma unroll
                for(u32 rr=0;rr<4;++rr)tj[rr]=tw[1+128u+(((64*rr+rl)*256u+cq)>>7)];}
            if constexpr(XL_NBC(G)!=0){   // ladder: no base conversion
                if constexpr(!XL_NAF(G)){
                    #pragma unroll
                    for(u32 j=0;j<8;++j)
                        #pragma unroll
                        for(u32 mt=0;mt<MT;++mt)
                            #pragma unroll
                            for(u32 x=0;x<4;++x)asm volatile("":"+r"(af[j][mt][x]));   // af live and opaque, as the MMA operands
                    #pragma unroll
                    for(u32 j=0;j<8;++j)
                        #pragma unroll
                        for(u32 mt=0;mt<MT;++mt)
                            #pragma unroll
                            for(u32 hf=0;hf<2;++hf)res[2*mt+hf][j]=af[j][mt][hf]^af[j][mt][hf+2];
                }else{const u32 hb=(h*0x9e3779b9u)^(u*0x85ebca6bu);
                    #pragma unroll
                    for(u32 j=0;j<8;++j)
                        #pragma unroll
                        for(u32 i=0;i<2*MT;++i)res[i][j]=hb+0x27d4eb2fu*(8u*i+j+1u);}
            }else{   // (braces: EDG rejects a pragma'd loop as a discarded if-constexpr else branch)
            #pragma unroll
            for(u32 j=0;j<8;++j){
                uint4 bv;if constexpr(XL_NBV(G)!=0){const u32 hb=(lane*0x9e3779b9u)^(u*0x85ebca6bu)^(j*0x27d4eb2fu);bv=make_uint4(hb,hb^0x01010101u,hb+0x5bd1e995u,hb^0x7f4a7c15u);}
                else if constexpr(XF_BVH(G)!=0){if(j<2)bv=bvh[j<2?j:0];else bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+uq);}
                else bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+uq);
                #pragma unroll
                for(u32 mt=0;mt<MT;++mt){
                    u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                    if constexpr(XG_NMMA(G)!=0){const u32* const a4=af[j][mt];c0[0]=a4[0]^bv.x;c0[1]=a4[1]^bv.y;c0[2]=a4[2]^bv.x;c0[3]=a4[3]^bv.y;
                        c1[0]=a4[0]^bv.z;c1[1]=a4[1]^bv.w;c1[2]=a4[2]^bv.z;c1[3]=a4[3]^bv.w;}else
                    if(XMNV(F)){pimma16832nv(c0,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                        pimma16832nv(c1,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);}else{
                    pimma16832(c0,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                    pimma16832(c1,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);}
                    #pragma unroll
                    for(u32 hf=0;hf<2;++hf){
                        const u32 y0=c0[2*hf],y1=c0[2*hf+1],y2=c1[2*hf],y3=c1[2*hf+1];   // each < 2^21
                        if constexpr(XG_NRED(G)!=0)res[2*mt+hf][j]=y0^y1^y2^y3;else
                        if constexpr(XG_RED5(G)!=0&&(F&4096)!=0){static_assert(XG_RED5(G)==0||xred5_ok(),"XG_RED5 needs 2^30 < p < 2^31");
                            const u32 w=y0+(y1<<8),z=y2+(y3<<8);
                            res[2*mt+hf][j]=(w+(z<<16))+__umulhi(z,m48)*np;
                        }else
                        if(F&4096){   // x = w + 2^16 z (w, z < 2^30): hi = floor(x/2^14) exactly, r = lo + q(-p)
                            const u32 w=y0+(y1<<8),z=y2+(y3<<8);
                            const u32 lo=w+(z<<16),hi=(w>>14)+(z<<2);
                            const u32 q=__umulhi(hi,m62)>>qsh;
                            res[2*mt+hf][j]=lo+q*np;
                        }else{
                        const u32 lo=y0+(y1<<8)+(y2<<16)+(y3<<24);
                        const u32 hi=(y0>>14)+(y1>>6)+(y2<<2)+(y3<<10);
                        const u32 q=__umulhi(hi,m62)>>qsh;
                        res[2*mt+hf][j]=lo-q*p;}
                        if constexpr(XF_TWJ(G)!=0&&S6&&MT==2)res[2*mt+hf][j]=xmshh<H>(res[2*mt+hf][j],tj[2*mt+hf],p);   // fix XF_TWJ
                    }
                }
            }
            }
            #pragma unroll
            if(S6){   // rows 64rr+8w+gid: twist, J-DIF, R-DIF stages len 256 (rr^2) and 128 (rr^1) in registers
                const u32 rl=8*warp+gid;
                #pragma unroll
                for(u32 rr=0;rr<4;++rr){
                    if constexpr(XF_TWJ(G)==0){const u64 t2=XTWH(F)?hw[rr]:tw[1+128u+(((64*rr+rl)*256u+cq)>>7)];
                    #pragma unroll
                    for(u32 j=0;j<8;++j)res[rr][j]=xmshh<H>(res[rr][j],t2,p);}
                    if(XTWH(F))xjdifv<F,H>(res[rr],hw[4],hw[5],hw[6],hw[7],p,H?BPR_P2(r):0u,H?BPR_N2(r):0u);else
                    xjdif<F,H>(res[rr],st,p,H?BPR_P2(r):0u,H?BPR_N2(r):0u);
                }
                const u64 wa=XTWH(F)?hw[8]:st[rl],wb=XTWH(F)?hw[9]:st[64u+rl],wc=XTWH(F)?hw[10]:st[128u+rl];
                if constexpr(XR16(F)!=2)   // sweep r16a: R stages all in phase B
                #pragma unroll
                for(u32 j=0;j<8;++j){const u32 p2=H?BPR_P2(r):0u,n2=H?BPR_N2(r):0u;xbfh<H>(res[0][j],res[2][j],wa,p,p2,n2);xbfh<H>(res[1][j],res[3][j],wb,p,p2,n2);xbfh<H>(res[0][j],res[1][j],wc,p,p2,n2);xbfh<H>(res[2][j],res[3][j],wc,p,p2,n2);}
                if(LB){
#if BPO_X_HAS_TMA
                    if constexpr(XG_LW(G)!=0)if((F&2)&&lane==0&&XABL(F)!=9)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");   // sweep lw: drain wait after the math
#endif
                    __syncthreads();}   // previous quad's phase-B / TMA reads of X done (flag 32768)
                if(F&4096){u32* const xa=X+tig*XS+8*tig+rl+4*(rl>>5);   // row 64rr+rl -> skewed 72rr+rl+4(rl>>5): immediate offsets
                    #pragma unroll
                    for(u32 rr=0;rr<4;++rr)
                        #pragma unroll
                        for(u32 j=0;j<8;++j)xa[j*XD+72*rr+XCA(F)*(j&1u)]=res[rr][j];
                }else
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
        if constexpr(XR16(F)==3)asm volatile("cp.async.wait_group 0;":::"memory");   // sweep cts: cross twiddles landed (visible after the barrier)
        if constexpr(XL_NOBAR(G)!=0)__syncwarp();else
        __syncthreads();
        if constexpr(XG_PFA(G)!=0&&S6)if(u+1u<r1/4u){const u32 rn=4u*(u+1u)+tig,rl=8u*warp+gid;
            const u64* const twn=twist+size_t(rn)*stride+1u+128u;const u64* const stn=stage_tw+size_t(rn)*256u;
            #pragma unroll
            for(u32 rr=0;rr<4;++rr)asm volatile("prefetch.global.L1 [%0];"::"l"(twn+(((64u*rr+rl)*256u+c)>>7)));
            #pragma unroll
            for(u32 e=0;e<3;++e)asm volatile("prefetch.global.L1 [%0];"::"l"(stn+64u*e+rl));
            asm volatile("prefetch.global.L1 [%0];"::"l"(stn+32u*(gid&3u)));}
        if((F&8192)&&S6&&NW==8&&XABL(F)!=3&&XABL(F)!=13){   // warp w: prime q = w>>1, digits 4(w&1)..+3 as NC-column groups sharing twiddles
            constexpr u32 NC=BPO_X_NC;
            const u32 q=warp>>1,r=4*uq+q,p=bpr_pr[r];
            const u64* const st=(F&4)?STS+q*256u:stage_tw+size_t(XL_HOT(G)?4u*xl_hotq<G>(u)+q:r)*256;const u64* const ct=XR16(F)==3?reinterpret_cast<const u64*>(X+4*XS)+q*256u:ctab+(size_t(r)*256u+cq)*256u;
            if constexpr(XR16(F)==1||XR16(F)==2)xrdif16<F>(X+q*XS+4*(warp&1u)*XD,8*q,st,ct,p,lane,op?tmB:tmA,c,r*8+4*(warp&1u),stm);   // sweep radix-16 phase B
            else if(F&131072u)xrdif6q<F,H,XG_XW(G),K>(X+q*XS+4*(warp&1u)*XD,8*q,st,ct,p,lane,op?tmB:tmA,c,r*8+4*(warp&1u),stm,H?BPR_P2(r):0u,H?BPR_N2(r):0u);   // flag 131072: one pass
            else{
            #pragma unroll
            for(u32 g=0;g<4/NC;++g){const u32 j=4*(warp&1u)+NC*g;
                xrdif6s<F,NC>(X+q*XS+j*XD,8*q,st,ct,p,lane,op?tmB:tmA,c,r*8+j,stm);}}
        }else
        #pragma unroll 1
        for(u32 n=0;n<((XABL(F)==3||XABL(F)==13)?0u:32/NW);++n){   // (13: no phase B either; the pbq branch above skips it)   // (prime q, digit j) columns of the quad (ablation 3: none)
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
        if constexpr(!(XG_LW(G)!=0&&LB))   // sweep lw: the wait moves to the LB barrier of the next quad (final wait_group 0 after the loop)
        if((F&2)&&lane==0&&XABL(F)!=9&&!XL_NPF(G))asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");   // X reusable (ablation 9: no wait)
#endif
#ifdef BPO_X_QL
      };
      if constexpr(XG_HQ3(G)!=0){if(u<3u){if constexpr(XG_H(G)!=0)quad(XQH{});else quad(XQS{});}else quad(XQS2{});}
      else if constexpr(XG_DUP(G)!=0){if(u<3u)quad(XQS{});else quad(XQS2{});}
      else if constexpr(XG_H(G)!=0)quad(XQH{});else quad(XQS{});
#else
      }
#endif
    }
#if BPO_X_HAS_TMA
    if((F&2)&&lane==0&&!XL_NPF(G))asm volatile("cp.async.bulk.wait_group 0;":::"memory");
#endif
}
#define BPO_X_SMEM(F) (4*(XR16(F)==2?8*352:(((F)&96)?8*320:8*288+32))*4+(((F)&4)?4*256*8:0)+(XR16(F)==3?4*256*8:0))
// Round 14 (geo) P1 geometry fix, sweep only (never instantiated by the production module): bpo_first_xs = static tail
// re-split. S = slots (blocks/SM x SMs) arrives in pfstep bits 16-25 (the sweep / whole-batch launch ORs it in; 0 = no split,
// production behaviour). The 512 column-ops are run as R = 512/S full rounds (block b < RS: column-op b, all quads, exactly
// production's first waves) plus the rem = 512-RS remaining column-ops split into f quad ranges (f = 4, 2 or 1, largest with
// rem f <= S): block RS+i (i < rem f) = column-op RS+i/f, quads [(i%f)Q/f, (i%f+1)Q/f). Grid = RS + rem f (target S = 376:
// 376 + 272 = 648 blocks, makespan 4+2 quads instead of 2 x 4; +136 prologues). One segment per block (no loop: no extra
// registers). Same per-element math as production (exact). L2 prefetch (PFN 1): a last-round full block b < rem f prefetches
// its tail column-op RS+b/f, earlier rounds b+S (production semantics); PFN 0: none. The CRT-side prefetch (items 0..p1_grid/2-1
// of the next product) matches the first round.
template<u32 F,u32 G,u32 PFN>
__device__ __forceinline__ void bpo_first_xs(const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,
    u32 N,u32 r0,u32 r1,u32 pfstep,const u64* ctab,const XTMap* tmA,const XTMap* tmB,u32* X){
    constexpr u32 T=512u;
    const u32 Sx=(pfstep>>16)&1023u,S=Sx?Sx:T,R=T/S,rem=T-R*S,f=rem==0u?1u:(rem*4u<=S?4u:(rem*2u<=S?2u:1u)),Q=(r1-r0)/4u,bk=blockIdx.x;
    const u32 pfs=pfstep&~(1023u<<16),pfl=pfs&0xf0000000u;const bool pfon=PFN!=0u&&(pfs&0x0fffffffu)!=0u;
    u32 id,q0=0,q1=Q,nx=0;
    if(bk<R*S){id=bk;if(id+S<R*S)nx=id+S;else if(bk-(R-1u)*S<rem*f)nx=R*S+(bk-(R-1u)*S)/f;}
    else{const u32 i=bk-R*S;id=R*S+i/f;q0=(i%f)*Q/f;q1=(i%f+1u)*Q/f;}
    const u32 pf=pfon&&nx>id?pfl|(nx-id):pfl;
    bpo_first_xb<F,G|(1u<<24)>(a,b,A,B,twist,stage_tw,bpt,N,r0+4u*q0,r0+4u*q1,pf,ctab,tmA,tmB,X,id);
}
#if !defined(BPO_LEAN)||BPO_LEAN_P1X
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) bpo_first_x(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];
    bpo_first_xb<BPO_XF,BPO_XG>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#endif // bpo_first_x
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__((BPO_XV&8)?512:256,BPO_TC_MINB) bpo_first_xv(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XV)/4];
    bpo_first_xb<BPO_XV>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#endif // bpo_first_xv
#if defined(BPO_DIAG)&&(!defined(BPO_LEAN)||BPO_LEAN_P1X)
// Exit-time diagnostics (solve.cpp diag): bpo_first_x ablations with the production signature, and bandwidth probes.
#define BPO_XD_KERNEL(NAME,FL) extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) NAME(const u32* a,const u32* b,u32* A,u32* B,    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];    bpo_first_xb<(FL)>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);}
BPO_XD_KERNEL(bpo_xd_nostore,BPO_XF|(8u<<20))
BPO_XD_KERNEL(bpo_xd_l2in,BPO_XF|(1u<<24))
BPO_XD_KERNEL(bpo_xd_compute,BPO_XF|(8u<<20)|(1u<<24))
extern "C" __global__ void diag_bw(const uint4* __restrict__ src,uint4* dst,u64 nr,u64 nw,u32 seed){   // read nr / write nw uint4s, grid-stride
    const u64 t=u64(blockIdx.x)*blockDim.x+threadIdx.x,S=u64(gridDim.x)*blockDim.x;u32 acc=0;
    const u64 m=nr>nw?nr:nw;
    for(u64 i=t;i<m;i+=S){if(i<nr){const uint4 x=__ldcs(src+i);acc^=x.x^x.y^x.z^x.w;}if(i<nw)dst[i]=make_uint4(u32(i),seed,acc,u32(i>>32));}
    if(acc==0x9e3779b9u&&seed==1u)dst[0].x=acc;}
#endif
#endif
