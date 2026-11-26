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
    if(lane==0&&BPO_XA!=8&&BPO_XA!=11){
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
#ifdef BPO_P1WL   // wline p1wl (FHERMA_P1WL=1): host table ctab_wl[r][c][64 m2 + 2 lane + e] = ctab[r][c][8 lane + 2 m2 + e]: one request = 512 contiguous bytes (4 whole lines)
    asm volatile("ld.global.nc.v2.u64 {%0,%1},[%2];":"=l"(w.x),"=l"(w.y):"l"(ctg+2*lane+64*m2));
#else
    asm volatile("ld.global.nc.v2.u64 {%0,%1},[%2];":"=l"(w.x),"=l"(w.y):"l"(ctg+8*lane+2*m2));
#endif
    return w;
}
// Flag 131072 (with 8192) phase B, xrdif6q (sweep a26c406 "pbq"; on target 0.938x P1 time, exact): the warp's 4 columns
// in ONE pass: layout-1 stages of both column pairs, one __syncwarp, layout-2 stages + cross twiddle per pair (rolled pair
// loop), one proxy fence and one lane-0 TMA burst (4 boxes, 1 commit) per quad instead of 2 x (fence, syncwarp, TMA
// section); stage and cross twiddles loaded once per quad, not per pair. Same values as xrdif6s (flag 65536 not used here).
template<u32 F>
__device__ __forceinline__ void xrdif6q(u32* const base0,const u32 so,const u64* st,const u64* ct,const u32 p,u32 lane,const XTMap* tm,u32 c,const u32 plane0,u32 stm){
    constexpr bool T=(F&1)!=0;
    constexpr u32 XD=320u;
    constexpr u32 Q=2u,NI=4u/Q;   // column pairs
    const u32 l7=lane&7u;
    {u64 w64[4],w32[2];
        #pragma unroll
        for(u32 m=0;m<4;++m)w64[m]=st[192u+l7+8u*m];
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)w32[qq]=st[224u+l7+8u*qq];
        const u64 w16=st[240u+l7];
        u32* const pb=base0+so+l7+72u*(lane>>3);
        auto l1=[&](const u32 pr){u32 v[Q][8];
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)v[n][m]=pb[(Q*pr+n)*XD+8*m+4*(m>>2)];
            #pragma unroll
            for(u32 m=0;m<4;++m)
                #pragma unroll
                for(u32 n=0;n<Q;++n)xbf(v[n][m],v[n][m+4],w64[m],p);
            #pragma unroll
            for(u32 qq=0;qq<2;++qq)
                #pragma unroll
                for(u32 n=0;n<Q;++n)
                    #pragma unroll
                    for(u32 hh=0;hh<2;++hh)xbf(v[n][qq+4*hh],v[n][qq+4*hh+2],w32[qq],p);
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;m+=2)xbf(v[n][m],v[n][m+1],w16,p);
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)pb[(Q*pr+n)*XD+8*m+4*(m>>2)]=v[n][m];
        };
        #pragma unroll
        for(u32 pr=0;pr<NI;++pr){l1(pr);if(pr+1<NI)__syncwarp();}   // groups kept sequential (8Q live values)
    }
    __syncwarp();
    u64 w8[4];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)w8[lo]=st[248u+lo];
    const u64 w253=st[253u],w252=st[252u],w254=st[254u];
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
            for(u32 n=0;n<Q;++n){if(lo==0)xbft<T>(v[n][0],v[n][4],w8[0],p);else xbf(v[n][lo],v[n][lo+4],w8[lo],p);}
        #pragma unroll
        for(u32 k=0;k<4;++k)
            #pragma unroll
            for(u32 n=0;n<Q;++n){const u32 lo=(k&1)+4*(k>>1);if(lo&1)xbf(v[n][lo],v[n][lo+2],w253,p);else xbft<T>(v[n][lo],v[n][lo+2],w252,p);}
        if(!T){
            #pragma unroll
            for(u32 lo=0;lo<8;lo+=2)
                #pragma unroll
                for(u32 n=0;n<Q;++n)xbf(v[n][lo],v[n][lo+1],w254,p);}
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){const ulonglong2 w=xct_ldv(ct,lane,m2);
            #pragma unroll
            for(u32 n=0;n<Q;++n){const u32 a=v[n][2*m2],b=v[n][2*m2+1];
                if(T){v[n][2*m2]=xmsh(a+b,u64(w.x),p);v[n][2*m2+1]=xmsh(a+p-b,u64(w.y),p);}
                else{v[n][2*m2]=xmsh(a,u64(w.x),p);v[n][2*m2+1]=xmsh(b,u64(w.y),p);}}}
#if BPO_X_HAS_TMA
        __syncwarp();   // layout-2 reads of this pair's regions done (the other pair's regions are disjoint)
        #pragma unroll
        for(u32 n=0;n<Q;++n){
            const uint4 e0=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]),e1=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);
            uint4* const d=sd+(Q*pr+n)*(XD/4);
            d[0]=sw?e1:e0;d[sw?-1:1]=sw?e0:e1;}
#endif
    }
#if BPO_X_HAS_TMA
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncwarp();
    if(lane==0&&BPO_XA!=8&&BPO_XA!=11){
        u64 pol=0;const bool hint=(F&2048)&&stm!=1u;
        if(hint){if(stm==2u)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));}
        #pragma unroll
        for(u32 n=0;n<4;++n){const u32 sa=u32(__cvta_generic_to_shared(base0+n*XD));
            if(hint)asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%1,%2,%3}],[%4],%5;"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa),"l"(pol):"memory");
            else asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa):"memory");}
        asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
}
template<u32 F>
__device__ __forceinline__ void bpo_first_xb(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const XTMap* tmA,const XTMap* tmB,u32* X){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr bool S6=(F&64)!=0;
    constexpr u32 XD=(F&96)?320u:288u,XS=(F&96)?8*320:8*288+32;   // 128 B aligned (prime,digit) regions
    const u32 stm=(pfstep>>29)&3u;const bool ldn=(pfstep>>28)&1u;pfstep&=0x0fffffffu;   // bits 29-30 TMA store policy (xrdif6 stm), 28 input loads normal (L2-retained for the next prime group)
    const u32 id=blockIdx.x,c=id&255u,op=id>>8;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(BPO_XA==11?0u:c)*L;   // ablation 11: compute proxy (column-0 input, L2-resident; no stores)
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
    const u32 xoff=((BPO_XA==11?0u:c)*L)&3u;   // flag 512: word offset of every row start in its 16 B chunk (256 L words per row step)
    const bool ca=(F&512)&&((reinterpret_cast<unsigned long long>(op?b:a)&15ull)==0);   // flag 512 needs a 16 B aligned operand base
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
                if(q<nch&&!ABX(1)){
                    const u32 sa0=u32(__cvta_generic_to_shared(X+(h>>4)*XRS+4u*q));const u32* const gq=g0+u64((h>>4)*256u)*L;
                    #pragma unroll
                    for(u32 i=0;i<8;++i){
                        const u32 sa=(F&4096)?sa0+i*(16u*XRS*4u):u32(__cvta_generic_to_shared(X+((h>>4)+16u*i)*XRS+4u*q));
                        const u32* g=(F&4096)?gq+i*(16u*256u*L):g0+u64(((h>>4)+16u*i)*256u)*L;
                        if(ldn)asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(g):"memory");
                        else asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(sa),"l"(g),"l"(pol):"memory");}}
                asm volatile("cp.async.commit_group;":::"memory");
                asm volatile("cp.async.wait_group 0;":::"memory");
            }else
            #pragma unroll 4
            for(u32 x=h;x<128u*L;x+=256u){const u32 rr=x/L,w=x-rr*L;X[x]=ABX(1)?x:(ldn?__ldg(hs+u64(rr*256u)*L+w):__ldcs(hs+u64(rr*256u)*L+w));}
            __syncthreads();
            if((F&4096)&&ca){
                static_assert(3u+POS[7]/32u+8u<XRS,"flag 4096: unclamped fragment windows stay inside the 60-word row");
                const u32* const rpt=X+(8*warp+gid)*XRS+xoff+tig;
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
    if(pfstep&&h<256u&&id+pfstep<512u&&!ABX(1)){
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
    constexpr bool LB=(F&32768)&&!(F&4)&&S6&&BPO_XA!=4&&BPO_XA!=5;
#ifdef BPO_P1RV   // p1rev (flag 0x800000, kernel bpo_first_xr): prime quads in reverse order (exact)
    for(u32 uu=r0/4;uu<r1/4;++uu){const u32 u=(F&0x800000u)?r1/4-1u-(uu-r0/4):uu;
#else
    for(u32 u=r0/4;u<r1/4;++u){
#endif
        if(!LB)__syncthreads();   // previous quad's R-DIF reads of X (and STS) done
        if(F&4){
            #pragma unroll
            for(u32 q=0;q<4;++q)if(h<256u)STS[q*256u+h]=stage_tw[size_t(4*u+q)*256+h];
        }
        if((F&16384)&&(F&8192)&&S6&&NW==8){   // flag 16384: phase-B cross twiddles of this warp's prime pulled into L1 now (L2 latency off phase B)
            const u64* const ctp=ctab+(size_t(4*u+(warp>>1))*256u+c)*256u+8u*lane;
            asm volatile("prefetch.global.L1 [%0];"::"l"(ctp));
        }
        if(BPO_XA!=4&&BPO_XA!=5){   // ablation 4/5: no phase A
            const u32 r=4*u+tig,p=bpr_pr[r];const u32 msh=(BPO_MIXED&&p<(1u<<30))?3u:2u;const u32 m62=u32(bpr_mu[r]>>msh),qsh=18u-msh;   // floor(2^(64-msh)/p) < 2^32 (BPO_MIXED: h30 prime set)
            const u32 np=(F&4096)?bpr_np.v[r]:0u-p;
            const u64* tw=twist+size_t(r)*stride;const u64* st=stage_tw+size_t(ABX(2)?0u:r)*256;
            u32 res[2*MT][8];   // [row 2mt+hf][digit], unreduced in [0,2p)
            #pragma unroll
            for(u32 j=0;j<8;++j){
#ifdef BPO_P1WL   // wline p1wl: host table bpt_wl[(4 j + u) 32 + lane] (uint4) = bpt[(32 j + lane) 16 + 4 u ..]: 512 contiguous bytes per request
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt)+((size_t(j)*4+(ABX(2)?0u:u))*32+lane));
#else
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+(ABX(2)?0u:u));
#endif
                #pragma unroll
                for(u32 mt=0;mt<MT;++mt){
                    u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                    pimma16832(c0,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                    pimma16832(c1,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);
                    #pragma unroll
                    for(u32 hf=0;hf<2;++hf){
                        const u32 y0=c0[2*hf],y1=c0[2*hf+1],y2=c1[2*hf],y3=c1[2*hf+1];   // each < 2^21
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
                    }
                }
            }
            #pragma unroll
            if(S6){   // rows 64rr+8w+gid: twist, J-DIF, R-DIF stages len 256 (rr^2) and 128 (rr^1) in registers
                const u32 rl=8*warp+gid;
#ifdef BPO_P1TW   // p1tw (flag 0x400000, kernel bpo_first_xt): twist argument = T1[((u*2 + c/128)*256 + 64 rr + rl)*4 + tig]
                constexpr bool TW=(F&0x400000u)!=0;
                const u64* const t1=twist+((size_t(ABX(2)?0u:u)*2u+(ABX(2)?0u:(c>>7)))*256u+rl)*4u+tig;(void)t1;
#endif
                #pragma unroll
                for(u32 rr=0;rr<4;++rr){
#ifdef BPO_P1TW
                    const u64 t2=TW?t1[256u*rr]:tw[1+128u+(((64*rr+rl)*256u+c)>>7)];
#else
                    const u64 t2=tw[1+128u+(((64*rr+rl)*256u+c)>>7)];
#endif
                    #pragma unroll
                    for(u32 j=0;j<8;++j)res[rr][j]=xmsh(res[rr][j],t2,p);
                    xjdif<F>(res[rr],st,p);
                }
                const u64 wa=st[rl],wb=st[64u+rl],wc=st[128u+rl];
                #pragma unroll
                for(u32 j=0;j<8;++j){xbf(res[0][j],res[2][j],wa,p);xbf(res[1][j],res[3][j],wb,p);xbf(res[0][j],res[1][j],wc,p);xbf(res[2][j],res[3][j],wc,p);}
                if(LB)__syncthreads();   // previous quad's phase-B / TMA reads of X done (flag 32768)
                if(F&4096){u32* const xa=X+tig*XS+8*tig+rl+4*(rl>>5);   // row 64rr+rl -> skewed 72rr+rl+4(rl>>5): immediate offsets
                    #pragma unroll
                    for(u32 rr=0;rr<4;++rr)
                        #pragma unroll
                        for(u32 j=0;j<8;++j)xa[j*XD+72*rr]=res[rr][j];
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
        __syncthreads();
        if((F&8192)&&S6&&NW==8&&BPO_XA!=3){   // warp w: prime q = w>>1, digits 4(w&1)..+3 as NC-column groups sharing twiddles
            constexpr u32 NC=BPO_X_NC;
            const u32 q=warp>>1,r=4*u+q,p=bpr_pr[r];
            const u64* const st=(F&4)?STS+q*256u:stage_tw+size_t(r)*256;const u64* const ct=ctab+(size_t(r)*256u+c)*256u;
            if(F&131072u)xrdif6q<F>(X+q*XS+4*(warp&1u)*XD,8*q,st,ct,p,lane,op?tmB:tmA,c,r*8+4*(warp&1u),stm);   // flag 131072: one pass
            else{
            #pragma unroll
            for(u32 g=0;g<4/NC;++g){const u32 j=4*(warp&1u)+NC*g;
                xrdif6s<F,NC>(X+q*XS+j*XD,8*q,st,ct,p,lane,op?tmB:tmA,c,r*8+j,stm);}}
        }else
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
    if((F&2)&&lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
#endif
}
#define BPO_X_SMEM(F) (4*(((F)&96)?8*320:8*288+32)*4+(((F)&4)?4*256*8:0))
#if !defined(BPO_LEAN)||BPO_LEAN_P1X
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) bpo_first_x(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];
    bpo_first_xb<BPO_XF>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#if defined(BPO_P1TW)&&BPO_X_HAS_TMA&&((BPO_XF)&64)
// p1tw entry: bpo_first_x with the S6 twist from T1 (host launches it only when T1 was built; else bpo_first_x)
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) bpo_first_xt(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];
    bpo_first_xb<BPO_XF|0x400000u>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#ifdef BPO_P1RV
// p1rev entry (candidate 5, pv99 on-target s_p1rev 0.9693 / 0.9710): bpo_first_xt with the prime quads in reverse order, so
// the P2 group read first (primes 0-7) is the most recently written; host launches it only when present (else bpo_first_xt)
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) bpo_first_xr(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];
    bpo_first_xb<BPO_XF|0x400000u|0x800000u>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#endif
#if defined(BPO_P1LZ)&&defined(BPO_P1RV)
// ---- p1lz (kernel bpo_first_xz; host switch FHERMA_P1LZ=1, default off): bpo_first_xr with lazy (Harvey) arithmetic on
// the prime quads whose four primes are all < 2^30 (4u + 3 < BPO_NH30: quads 0-2 of the h30 set); the 31-bit quad runs the
// canonical code (same template, LZ = false). Bounds for LZ (p < 2^30, so 4p < 2^32; values in units of p):
//  Z0 base conversion r = lo - q p in [0,2p) (unchanged; header comment of this file).
//  Z1 lazy Shoup zmsh(x, w) = x w - q p in [0,2p) for ANY 32-bit x (w' = floor(w 2^32 / p) in the high word). No correction:
//     the twist product drops xmsh's min(r, r-p).
//  Z2 GS butterfly, a, b in [0,2p): X' = min(s, s-2p), s = a+b < 4p -> [0,2p) (one correction, as xadd's);
//     Y' = zmsh(a + 2p - b, w), argument in (0,4p) < 2^32 -> [0,2p): xmsh's correction dropped.
//  Z3 trivial (psi^0) butterfly: X' as Z2; Y' = min(d, d-2p), d = a + 2p - b in (0,4p) -> [0,2p) (as xsub's: no change).
//  Z4 invariant: every value kept in registers, written to X (smem) or read back is in [0,2p) (Z0-Z3 preserve it).
//  Z5 last R-DIF stage without the fused cross twiddle (!(F&1)): X' = a+b < 4p unreduced (consumer is a Shoup product).
//  Z6 cross twiddle (with the fused psi^0 stage, flag 1): xmsh(a+b, w) and xmsh(a+2p-b, w), arguments < 4p < 2^32;
//     canonical output min(r, r-p), r in [0,2p) -> [0,p): the A/B words are bit-identical to bpo_first_xr's (canonical
//     residues), so P2's input contract is unchanged.
// Dropped per (prime, column): the Y-side correction of every twiddled butterfly and the twist correction.
#ifndef BPO_NH30
#define BPO_NH30 0
#endif
struct ZP2{u32 v[BCRT_K];};
__host__ __device__ constexpr ZP2 zp2_make(){ZP2 a{};constexpr u32 P[BCRT_K]={BCRT_PRIMES};for(u32 i=0;i<BCRT_K;++i)a.v[i]=2u*P[i];return a;}
__constant__ ZP2 bpr_p2=zp2_make();   // 2p per prime (wraps for 31-bit primes: used only by LZ quads, p < 2^30)
__device__ __forceinline__ u32 zr2(u32 s,u32 p2){return min(s,s-p2);}
__device__ __forceinline__ u32 zmsh(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));return x*u32(c)-q*p;}   // Z1
template<bool LZ>__device__ __forceinline__ void zbf(u32& x,u32& y,u64 w,u32 p,u32 p2){
    if(LZ){const u32 a=x,b=y;x=zr2(a+b,p2);y=zmsh(a+p2-b,w,p);}else xbf(x,y,w,p);}   // Z2
template<bool LZ,bool TRIV>__device__ __forceinline__ void zbft(u32& x,u32& y,const u64& w,u32 p,u32 p2){
    if(!TRIV)zbf<LZ>(x,y,w,p,p2);else if(LZ){const u32 a=x,b=y;x=zr2(a+b,p2);y=zr2(a+p2-b,p2);}else xbf1(x,y,p);}   // Z3
template<u32 F,bool LZ>__device__ __forceinline__ void zjdif(u32 (&x8)[8],const u64* st,u32 p,u32 p2){
    constexpr bool T=(F&1)!=0;
    zbft<LZ,T>(x8[0],x8[4],st[0],p,p2);zbf<LZ>(x8[1],x8[5],st[32],p,p2);zbf<LZ>(x8[2],x8[6],st[64],p,p2);zbf<LZ>(x8[3],x8[7],st[96],p,p2);
    const u64 w0=st[0],w64=st[64];
    zbft<LZ,T>(x8[0],x8[2],w0,p,p2);zbf<LZ>(x8[1],x8[3],w64,p,p2);zbft<LZ,T>(x8[4],x8[6],w0,p,p2);zbf<LZ>(x8[5],x8[7],w64,p,p2);
    zbft<LZ,T>(x8[0],x8[1],w0,p,p2);zbft<LZ,T>(x8[2],x8[3],w0,p,p2);zbft<LZ,T>(x8[4],x8[5],w0,p,p2);zbft<LZ,T>(x8[6],x8[7],w0,p,p2);
}
// xrdif6q with LZ (same layout, twiddles and TMA burst; arithmetic per Z1-Z6)
template<u32 F,bool LZ>
__device__ __forceinline__ void zrdif6q(u32* const base0,const u32 so,const u64* st,const u64* ct,const u32 p,u32 lane,const XTMap* tm,u32 c,const u32 plane0,u32 stm,const u32 p2){
    constexpr bool T=(F&1)!=0;
    constexpr u32 XD=320u;
    constexpr u32 Q=2u,NI=4u/Q;
    (void)p2;
    const u32 l7=lane&7u;
    {u64 w64[4],w32[2];
        #pragma unroll
        for(u32 m=0;m<4;++m)w64[m]=st[192u+l7+8u*m];
        #pragma unroll
        for(u32 qq=0;qq<2;++qq)w32[qq]=st[224u+l7+8u*qq];
        const u64 w16=st[240u+l7];
        u32* const pb=base0+so+l7+72u*(lane>>3);
        auto l1=[&](const u32 pr){u32 v[Q][8];
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)v[n][m]=pb[(Q*pr+n)*XD+8*m+4*(m>>2)];
            #pragma unroll
            for(u32 m=0;m<4;++m)
                #pragma unroll
                for(u32 n=0;n<Q;++n)zbf<LZ>(v[n][m],v[n][m+4],w64[m],p,p2);
            #pragma unroll
            for(u32 qq=0;qq<2;++qq)
                #pragma unroll
                for(u32 n=0;n<Q;++n)
                    #pragma unroll
                    for(u32 hh=0;hh<2;++hh)zbf<LZ>(v[n][qq+4*hh],v[n][qq+4*hh+2],w32[qq],p,p2);
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;m+=2)zbf<LZ>(v[n][m],v[n][m+1],w16,p,p2);
            #pragma unroll
            for(u32 n=0;n<Q;++n)
                #pragma unroll
                for(u32 m=0;m<8;++m)pb[(Q*pr+n)*XD+8*m+4*(m>>2)]=v[n][m];
        };
        #pragma unroll
        for(u32 pr=0;pr<NI;++pr){l1(pr);if(pr+1<NI)__syncwarp();}
    }
    __syncwarp();
    u64 w8[4];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)w8[lo]=st[248u+lo];
    const u64 w253=st[253u],w252=st[252u],w254=st[254u];
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
            for(u32 n=0;n<Q;++n){if(lo==0)zbft<LZ,T>(v[n][0],v[n][4],w8[0],p,p2);else zbf<LZ>(v[n][lo],v[n][lo+4],w8[lo],p,p2);}
        #pragma unroll
        for(u32 k=0;k<4;++k)
            #pragma unroll
            for(u32 n=0;n<Q;++n){const u32 lo=(k&1)+4*(k>>1);if(lo&1)zbf<LZ>(v[n][lo],v[n][lo+2],w253,p,p2);else zbft<LZ,T>(v[n][lo],v[n][lo+2],w252,p,p2);}
        if(!T){
            #pragma unroll
            for(u32 lo=0;lo<8;lo+=2)
                #pragma unroll
                for(u32 n=0;n<Q;++n){if(LZ){const u32 a=v[n][lo],b=v[n][lo+1];v[n][lo]=a+b;v[n][lo+1]=zmsh(a+p2-b,w254,p);}   // Z5
                    else xbf(v[n][lo],v[n][lo+1],w254,p);}}
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){const ulonglong2 w=xct_ldv(ct,lane,m2);
            #pragma unroll
            for(u32 n=0;n<Q;++n){const u32 a=v[n][2*m2],b=v[n][2*m2+1];
                if(T){v[n][2*m2]=xmsh(a+b,u64(w.x),p);v[n][2*m2+1]=xmsh(a+(LZ?p2:p)-b,u64(w.y),p);}   // Z6
                else{v[n][2*m2]=xmsh(a,u64(w.x),p);v[n][2*m2+1]=xmsh(b,u64(w.y),p);}}}
#if BPO_X_HAS_TMA
        __syncwarp();
        #pragma unroll
        for(u32 n=0;n<Q;++n){
            const uint4 e0=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]),e1=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);
            uint4* const d=sd+(Q*pr+n)*(XD/4);
            d[0]=sw?e1:e0;d[sw?-1:1]=sw?e0:e1;}
#endif
    }
#if BPO_X_HAS_TMA
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncwarp();
    if(lane==0&&!ABX(4)){
        u64 pol=0;const bool hint=(F&2048)&&stm!=1u;
        if(hint){if(stm==2u)asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));}
        #pragma unroll
        for(u32 n=0;n<4;++n){const u32 sa=u32(__cvta_generic_to_shared(base0+n*XD));
            if(hint)asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%1,%2,%3}],[%4],%5;"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa),"l"(pol):"memory");
            else asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%1,%2,%3}],[%4];"::"l"(reinterpret_cast<u64>(tm)),"r"(c*8u),"r"(0u),"r"(plane0+n),"r"(sa):"memory");}
        asm volatile("cp.async.bulk.commit_group;":::"memory");}
#endif
}
// one prime quad of bpo_first_xbz (phase A + phase B), LZ per Z1-Z6
template<u32 F,bool LZ>
__device__ __forceinline__ void xz_quad(const u32 u,const u32 (&af)[8][2][4],u32* const X,const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,
    const u64* ctab,const XTMap* tmA,const XTMap* tmB,const u32 c,const u32 op,const u32 lane,const u32 warp,const u32 gid,const u32 tig,const u32 stm){
    constexpr u32 MT=2,XD=320u,XS=8*320;
        __syncthreads();   // previous quad's R-DIF reads of X done
        {
            const u32 r=4*u+tig,p=bpr_pr[r];const u32 msh=(BPO_MIXED&&p<(1u<<30))?3u:2u;const u32 m62=u32(bpr_mu[r]>>msh),qsh=18u-msh;
            const u32 np=bpr_np.v[r];const u32 p2=bpr_p2.v[r];(void)p2;   // 2p from a table: IADD3 operands (ptxas refolds 2*p into IMAD)
            const u64* st=stage_tw+size_t(ABX(2)?0u:r)*256;
            u32 res[2*MT][8];   // Z0: [0,2p)
            #pragma unroll
            for(u32 j=0;j<8;++j){
#ifdef BPO_P1WL
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt)+((size_t(j)*4+(ABX(2)?0u:u))*32+lane));
#else
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+(ABX(2)?0u:u));
#endif
                #pragma unroll
                for(u32 mt=0;mt<MT;++mt){
                    u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                    pimma16832(c0,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                    pimma16832(c1,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);
                    #pragma unroll
                    for(u32 hf=0;hf<2;++hf){
                        const u32 y0=c0[2*hf],y1=c0[2*hf+1],y2=c1[2*hf],y3=c1[2*hf+1];
                        const u32 w=y0+(y1<<8),z=y2+(y3<<8);
                        const u32 lo=w+(z<<16),hi=(w>>14)+(z<<2);
                        const u32 q=__umulhi(hi,m62)>>qsh;
                        res[2*mt+hf][j]=lo+q*np;
                    }
                }
            }
            const u32 rl=8*warp+gid;
            const u64* const t1=twist+((size_t(ABX(2)?0u:u)*2u+(ABX(2)?0u:(c>>7)))*256u+rl)*4u+tig;
            #pragma unroll
            for(u32 rr=0;rr<4;++rr){
                const u64 t2=t1[256u*rr];
                #pragma unroll
                for(u32 j=0;j<8;++j)res[rr][j]=LZ?zmsh(res[rr][j],t2,p):xmsh(res[rr][j],t2,p);   // Z1 (LZ: [0,2p), no correction)
                zjdif<F,LZ>(res[rr],st,p,p2);
            }
            const u64 wa=st[rl],wb=st[64u+rl],wc=st[128u+rl];
            #pragma unroll
            for(u32 j=0;j<8;++j){zbf<LZ>(res[0][j],res[2][j],wa,p,p2);zbf<LZ>(res[1][j],res[3][j],wb,p,p2);zbf<LZ>(res[0][j],res[1][j],wc,p,p2);zbf<LZ>(res[2][j],res[3][j],wc,p,p2);}
            u32* const xa=X+tig*XS+8*tig+rl+4*(rl>>5);   // Z4: stored words in [0,2p) (LZ) / [0,p)
            #pragma unroll
            for(u32 rr=0;rr<4;++rr)
                #pragma unroll
                for(u32 j=0;j<8;++j)xa[j*XD+72*rr]=res[rr][j];
        }
        __syncthreads();
        {const u32 q=warp>>1,r=4*u+q,p=bpr_pr[r];
            const u64* const st=stage_tw+size_t(ABX(2)?0u:r)*256;const u64* const ct=ctab+(size_t(ABX(2)?0u:r)*256u+(ABX(2)?0u:c))*256u;
            zrdif6q<F,LZ>(X+q*XS+4*(warp&1u)*XD,8*q,st,ct,p,lane,op?tmB:tmA,c,r*8+4*(warp&1u),stm,bpr_p2.v[r]);}
#if BPO_X_HAS_TMA
        if(lane==0)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");   // X reusable
#endif
}
// bpo_first_xb restricted to the production flag set (S6, flags 256/512/4096/8192/131072, p1tw, p1rev), loop body per quad
// instantiated twice (LZ for quads of primes < 2^30, canonical otherwise).
template<u32 F>
__device__ __forceinline__ void bpo_first_xbz(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const XTMap* tmA,const XTMap* tmB,u32* X){
    static_assert((F&64)&&(F&256)&&(F&512)&&(F&4096)&&(F&8192)&&(F&131072u)&&(F&0x400000u)&&(F&0x800000u)&&!(F&8)&&!(F&4)&&!(F&32768u),"p1lz: production flag set only");
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr u32 XD=320u,XS=8*320;
    const u32 stm=(pfstep>>29)&3u;const bool ldn=(pfstep>>28)&1u;pfstep&=0x0fffffffu;
    const u32 id=blockIdx.x,c=id&255u,op=id>>8;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(c)*L;
    constexpr u32 MT=2;
    u32 af[8][MT][4];
    constexpr u32 XRS=60;
    const u32 xoff=(c*L)&3u;
    const bool ca=((reinterpret_cast<unsigned long long>(op?b:a)&15ull)==0);
    {
        u32 mk[8][2];
        #pragma unroll
        for(u32 j=0;j<8;++j)
            #pragma unroll
            for(u32 xh=0;xh<2;++xh)mk[j][xh]=bpr_fmask.v[j][xh][tig];
        #pragma unroll
        for(u32 mt=0;mt<MT;++mt){
            if(mt)__syncthreads();
            const u32* hs=src+u64(mt*128u*256u)*L;
            if(ca){
                const u32 q=h&15u,nch=(xoff+L+3u)/4u;const u32* g0=hs-xoff+4u*q;
                u64 pol=0;if(!ldn)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
                if(q<nch&&!ABX(1)){
                    const u32 sa0=u32(__cvta_generic_to_shared(X+(h>>4)*XRS+4u*q));const u32* const gq=g0+u64((h>>4)*256u)*L;
                    #pragma unroll
                    for(u32 i=0;i<8;++i){
                        const u32 sa=sa0+i*(16u*XRS*4u);
                        const u32* g=gq+i*(16u*256u*L);
                        if(ldn)asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sa),"l"(g):"memory");
                        else asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(sa),"l"(g),"l"(pol):"memory");}}
                asm volatile("cp.async.commit_group;":::"memory");
                asm volatile("cp.async.wait_group 0;":::"memory");
            }else
            #pragma unroll 4
            for(u32 x=h;x<128u*L;x+=256u){const u32 rr=x/L,w=x-rr*L;X[x]=ABX(1)?x:(ldn?__ldg(hs+u64(rr*256u)*L+w):__ldcs(hs+u64(rr*256u)*L+w));}
            __syncthreads();
            if(ca){
                const u32* const rpt=X+(8*warp+gid)*XRS+xoff+tig;
                #pragma unroll
                for(u32 j=0;j<8;++j){
                    const u32 wb=POS[j]/32,sh=POS[j]%32;
                    const int D=int(POS[j+1])-int(POS[j]);
                    #pragma unroll
                    for(u32 x=0;x<4;++x){
                        const u32* rp=rpt+64*(x&1)*XRS+wb+4*(x>>1);
                        const u32 v=__funnelshift_r(rp[0],rp[1],sh);
                        if(D-32*(3+4*int(x>>1))>=32)af[j][mt][x]=v;
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
                    const u32* rp=X+lrow*L;
                    const u32 w0=wb+t,w1=min(wb+t+1,L-1);
                    const u32 v=__funnelshift_r(rp[min(w0,L-1)],rp[w1],sh);
                    const int width=int(POS[j+1])-int(POS[j]+32*t);
                    af[j][mt][x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
                }
            }
        }
    }
    if(pfstep&&h<256u&&id+pfstep<512u&&!ABX(1)){
        const u32 pf=id+pfstep;const u32* ps=(pf>=256u?b:a)+u64(pf&255u)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    #pragma unroll 1
    for(u32 uu=r0/4;uu<r1/4;++uu){const u32 u=r1/4-1u-(uu-r0/4);   // p1rev: quads in reverse order
#ifdef BPO_P1LZ_ALL   // census-only build (JIT_DEFS): every quad on the lazy body, so the function holds one quad body
        if(true)xz_quad<F,true>(u,af,X,twist,stage_tw,bpt,ctab,tmA,tmB,c,op,lane,warp,gid,tig,stm);else
#endif
        if(4u*u+3u<u32(BPO_NH30))xz_quad<F,true>(u,af,X,twist,stage_tw,bpt,ctab,tmA,tmB,c,op,lane,warp,gid,tig,stm);
        else xz_quad<F,false>(u,af,X,twist,stage_tw,bpt,ctab,tmA,tmB,c,op,lane,warp,gid,tig,stm);}
#if BPO_X_HAS_TMA
    if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
#endif
}
// p1lz entry (host launches it only when FHERMA_P1LZ=1 and the entry exists; else bpo_first_xr)
extern "C" __global__ void __launch_bounds__((BPO_XF&8)?512:256,BPO_TC_MINB) bpo_first_xz(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XF)/4];
    bpo_first_xbz<BPO_XF|0x400000u|0x800000u>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#endif
// T1 init (once): out[((u*2+hh)*256 + x)*4 + tg] = twist[(4u+tg)*641 + 129 + 2x + hh], grid K/4, 512 threads (hh, x)
extern "C" __global__ void __launch_bounds__(512) bpo_t1_init(const u64* twist,u64* out){
    const u32 u=blockIdx.x,hh=threadIdx.x>>8,x=threadIdx.x&255u;
    #pragma unroll
    for(u32 tg=0;tg<4;++tg)out[((u*2u+hh)*256u+x)*4u+tg]=twist[(4u*u+tg)*641u+129u+2u*x+hh];
}
#endif
#endif // bpo_first_x
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__((BPO_XV&8)?512:256,BPO_TC_MINB) bpo_first_xv(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab,
    const __grid_constant__ XTMap tmA,const __grid_constant__ XTMap tmB){
    __shared__ __align__(128) u32 X[BPO_X_SMEM(BPO_XV)/4];
    bpo_first_xb<BPO_XV>(a,b,A,B,twist,stage_tw,bpt,N,r0,r1,pfstep,ctab,&tmA,&tmB,X);
}
#endif // bpo_first_xv
#endif
