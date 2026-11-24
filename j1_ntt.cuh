// J=1 pure-RNS integer transform chain (N = 65536, 118 primes < 2^30), bit-exact with j1/ref.py:
//   ntt_fwd (merged-twist Cooley-Tukey, Harvey lazy, [0,4p)) of both operands, pointwise Montgomery ([0,2p)),
//   ntt_inv (merged-twist Gentleman-Sande, Harvey lazy, [0,2p)).  All arrays [K][N] u32, natural coefficient order
//   n = 256 r + c (r = row, c = column); spectra stay in the reference's bit-reversed positions, so every butterfly
//   is the reference butterfly with the same operands.
// Three kernels (production P1 / P2 / P3 pass structure, 16 stages = 8 column + 8 row):
//   j1_colfwd : stages 1-8  (index bits 15..8 = r) of one operand per 128-thread group, 8 columns x 256 rows, 16 x 16
//               in registers with one smem transpose; twiddles psi^brev16(j), j < 256, from a 256-entry Shoup table.
//   j1_row    : stages 9-16 (bits 7..0 = c) of a and b, pointwise, inverse stages 16-9; one warp per row pair
//               (r, 255 - r), warp-local smem transposes (no block barrier).  Twiddles j >= 256 are generated exactly
//               (canonical w and Shoup word floor(w 2^32/p)) from two 256-entry factor tables (j1_twgen).
//   j1_colinv : inverse stages 8-1, store y.
// Every kernel may run in place (output pointer = input pointer).
#ifndef J1_NTT_CUH
#define J1_NTT_CUH
typedef unsigned int u32;
typedef unsigned long long u64;
// Harvey CT butterfly (ref ntt_fwd): x, y < 4p -> x' = u + Shoup(y), y' = u - Shoup(y) + 2p, u = x mod< 2p; both < 4p.
__device__ __forceinline__ void j1_ct(u32& x,u32& y,u64 w,u32 p,u32 p2){
    const u32 a=min(x,x-p2);const u32 t=y*u32(w)-__umulhi(y,u32(w>>32))*p;x=a+t;y=a+p2-t;}
// Harvey GS butterfly (ref ntt_inv): x, y < 2p -> x' = (x + y) mod< 2p, y' = Shoup(x - y + 2p); both < 2p.
__device__ __forceinline__ void j1_gs(u32& x,u32& y,u64 w,u32 p,u32 p2){
#ifdef J1_PROBE_NOARITH   // timing probe only: memory/transposes without butterfly arithmetic
    x=x+y+u32(w);y=y^u32(w>>32);return;
#endif
    const u32 s=x+y,d=x+p2-y;x=min(s,s-p2);y=d*u32(w)-__umulhi(d,u32(w>>32))*p;}
// ref pointwise: x, y < 4p -> (z + (z_lo * pinv mod 2^32) p) >> 32, z = (x mod< 2p)(y mod< 2p), pinv = -p^-1 mod 2^32.
__device__ __forceinline__ u32 j1_pw(u32 x,u32 y,u32 p,u32 p2,u32 pinv){
    x=min(x,x-p2);y=min(y,y-p2);const u64 z=u64(x)*y;const u32 m=u32(z)*pinv;return u32((z+u64(m)*p)>>32);}
// Exact twiddle w = A B mod p with ws = floor(w 2^32 / p), from A (canonical) and Bs = floor(B 2^64 / p):
//   q = floor(A Bs / 2^32) mod 2^32 is ws or ws - 1 (A Bs / 2^32 > A B 2^32/p - 1/4 and floor(A B 2^32/p) = ws mod 2^32);
//   rem = w 2^32 - q p in (0, 2p) is -q p mod 2^32; rem >= p -> q + 1;  w = umulhi(ws, p) + 1 (0 < w 2^32 - ws p < p).
// one = 1 in an opaque register (j1_opq1): ptxas otherwise forms umulhi + 1 as IMAD.HI with a 64-bit 2^32 addend and
// rematerializes the MOV.64 constant per twiddle.
__device__ __forceinline__ u64 j1_twgen(u32 A,u64 Bs,u32 p,u32 one){
    u32 q=A*u32(Bs>>32)+__umulhi(A,u32(Bs));
    const u32 rem=0u-q*p;q+=(rem>=p)?one:0u;
    return u64(__umulhi(q,p)+one)|(u64(q)<<32);}
__device__ __forceinline__ u32 j1_opq1(){return __shfl_sync(0xffffffffu,1u,0);}
// 2p as an opaque register (lane-0 shuffle, as production p2k): ptxas otherwise folds every x - 2p / a + 2p - t into
// IMAD p, -2 / IMAD p, 2 (+1 instruction per butterfly).  All lanes of the warp hold the same prime.
__device__ __forceinline__ u32 j1_opq2(u32 p){return __shfl_sync(0xffffffffu,2u*p,0);}
__device__ __forceinline__ u32 j1_brev16(u32 j){return __brev(j)>>16;}
template<u32 D>__device__ __forceinline__ constexpr u32 j1_brevd(u32 k){u32 r=0;for(u32 b=0;b<D;++b)r|=((k>>b)&1u)<<(D-1-b);return r;}

// ---- column passes (stages 1-8 / 8-1): one group of 128 threads (4 warps) per 8 columns of one prime ----
// Thread t: hi = t / 8, cl = t % 8; column c = 8 colblk + cl.  Load layout: thread owns r = 16 r1 + hi (r1 = 0..15);
// after the transpose it owns r = 16 hi + r2.  Warp loads/stores: 4 rows x 32 B sectors.  smem tile (per group,
// J1_COL_SMEM = 2176 words = 8.5 KB): S[r][cl] at 8 r + cl + 8 (r >> 4), conflict-free both ways.
// Group barrier: named barrier bar_id over the group's 128 threads (fused kernels pass their own id).
// tw1: per prime [2][256] Shoup words: [0][j] = psi^brev16(j), [1][j] = psi^-brev16(j) (j = 1..255).
#define J1_COL_SMEM 2176
__device__ __forceinline__ void j1_bar128(u32 id){asm volatile("bar.sync %0, 128;"::"r"(id):"memory");}
// two consecutive Shoup words (16-byte aligned pair j, j+1)
__device__ __forceinline__ void j1_ld2(const u64* t,u64& w0,u64& w1){const ulonglong2 v=__ldg(reinterpret_cast<const ulonglong2*>(t));w0=v.x;w1=v.y;}
// forward stages over the 16 registers v[i], i = 16-point index: stage d (0..3) pairs bit 3-d, block k = i >> (4-d),
// twiddle tw[2^d + k] (CT); INV: Gentleman-Sande with stages d = 3..0.  tw[1..15] loaded as 8 aligned pairs (tw + 2m).
template<bool INV>__device__ __forceinline__ void j1_r16(u32 (&v)[16],const u64* tw,u32 p,u32 p2){
    u64 w[16];
    #pragma unroll
    for(u32 m=0;m<8;++m)j1_ld2(tw+2u*m,w[2*m],w[2*m+1]);
    #pragma unroll
    for(u32 dd=0;dd<4;++dd){const u32 d=INV?3u-dd:dd,half=8u>>d;
        #pragma unroll
        for(u32 i=0;i<16;++i)if(!(i&half)){if(INV)j1_gs(v[i],v[i+half],w[(1u<<d)+(i>>(4-d))],p,p2);else j1_ct(v[i],v[i+half],w[(1u<<d)+(i>>(4-d))],p,p2);}}
}
// tile index of S[r][cl] = 8 r + cl + 8 (r >> 4):  r = 16 r1 + r2  ->  136 r1 + 8 r2 + cl
// Shared-memory staging (for fusion with the conversion epilogue / back conversion): the group's tile S, J1_COL_SMEM
// words, holds one prime's 256 x 8 block of coefficients n = 256 r + 8 colblk + cl at word 8 r + cl + 8 (r >> 4)
// (= j1_sidx(r, cl)), one u32 residue each.
//   SIN  (colfwd): input from the tile, values < 4p (conversion output < 2p).  The producer fills the tile and passes a
//        group barrier (or any barrier covering the 128 threads) before the call; X is not read.
//   SOUT (colinv): output y ([0, 2p), same tile layout) is left in the tile, ending with a group barrier; Y is not written.
// The tile is otherwise the transpose buffer, so the body owns it for its duration.
__device__ __forceinline__ u32 j1_sidx(u32 r,u32 cl){return 8u*r+cl+8u*(r>>4);}
template<bool SIN=false>__device__ __forceinline__ void j1_colfwd_body(u32 colblk,u32 prime,u32 tid,u32 bar_id,u32* S,
    const u32* X,u32* Y,const u32* primes,const u64* tw1){
    const u32 p=primes[prime],p2=j1_opq2(p);
    const u32 hi=tid>>3,cl=tid&7u,c=colblk*8u+cl;
    const u64* t=tw1+u64(prime)*512u;
    const u32* x=X+u64(prime)*65536u+c+256u*hi;u32* y=Y+u64(prime)*65536u+c+4096u*hi;
    u32* const sA=S+8u*hi+cl;      // r2 = hi, step r1: +136
    u32* const sB=S+136u*hi+cl;    // r1 = hi, step r2: +8
    u32 v[16];
    #pragma unroll
    for(u32 r1=0;r1<16;++r1)v[r1]=SIN?sA[136u*r1]:x[4096u*r1];   // r2 = hi
    // stages 1-4 over r1: stage d+1 pairs r1 bit 3-d, block i = r1 >> (4-d), twiddle j = 2^d + i
    j1_r16<false>(v,t,p,p2);
    #pragma unroll
    for(u32 r1=0;r1<16;++r1)sA[136u*r1]=v[r1];
    j1_bar128(bar_id);
    #pragma unroll
    for(u32 r2=0;r2<16;++r2)v[r2]=sB[8u*r2];   // thread now owns r1 = hi
    // stages 5-8 over r2: stage 5+d pairs r2 bit 3-d, i = (r1 << d) | (r2 >> (4-d)), j = 2^(4+d) + (r1 << d) + k
    {   u64 w[15];   // w[2^d - 1 + k]
        w[0]=__ldg(t+16u+hi);
        j1_ld2(t+32u+2u*hi,w[1],w[2]);
        j1_ld2(t+64u+4u*hi,w[3],w[4]);j1_ld2(t+66u+4u*hi,w[5],w[6]);
        #pragma unroll
        for(u32 m=0;m<4;++m)j1_ld2(t+128u+8u*hi+2u*m,w[7+2*m],w[8+2*m]);
        #pragma unroll
        for(u32 d=0;d<4;++d){const u32 half=8u>>d;
            #pragma unroll
            for(u32 r2=0;r2<16;++r2)if(!(r2&half))j1_ct(v[r2],v[r2+half],w[(1u<<d)-1u+(r2>>(4-d))],p,p2);}
    }
    j1_bar128(bar_id);   // tile reads done (the group may be reused for its next column block)
    #pragma unroll
    for(u32 r2=0;r2<16;++r2)y[256u*r2]=v[r2];
}
template<bool SOUT=false>__device__ __forceinline__ void j1_colinv_body(u32 colblk,u32 prime,u32 tid,u32 bar_id,u32* S,
    const u32* X,u32* Y,const u32* primes,const u64* tw1){
    const u32 p=primes[prime],p2=j1_opq2(p);
    const u32 hi=tid>>3,cl=tid&7u,c=colblk*8u+cl;
    const u64* t=tw1+u64(prime)*512u+256u;
    const u32* x=X+u64(prime)*65536u+c+4096u*hi;u32* y=Y+u64(prime)*65536u+c+256u*hi;
    u32* const sA=S+8u*hi+cl;      // r2 = hi
    u32* const sB=S+136u*hi+cl;    // r1 = hi
    u32 v[16];
    #pragma unroll
    for(u32 r2=0;r2<16;++r2)v[r2]=x[256u*r2];   // r1 = hi
    {   u64 w[15];
        w[0]=__ldg(t+16u+hi);
        j1_ld2(t+32u+2u*hi,w[1],w[2]);
        j1_ld2(t+64u+4u*hi,w[3],w[4]);j1_ld2(t+66u+4u*hi,w[5],w[6]);
        #pragma unroll
        for(u32 m=0;m<4;++m)j1_ld2(t+128u+8u*hi+2u*m,w[7+2*m],w[8+2*m]);
        #pragma unroll
        for(u32 dd=0;dd<4;++dd){const u32 d=3u-dd,half=8u>>d;
            #pragma unroll
            for(u32 r2=0;r2<16;++r2)if(!(r2&half))j1_gs(v[r2],v[r2+half],w[(1u<<d)-1u+(r2>>(4-d))],p,p2);}
    }
    #pragma unroll
    for(u32 r2=0;r2<16;++r2)sB[8u*r2]=v[r2];
    j1_bar128(bar_id);
    #pragma unroll
    for(u32 r1=0;r1<16;++r1)v[r1]=sA[136u*r1];   // thread now owns r2 = hi
    j1_r16<true>(v,t,p,p2);
    if(SOUT){   // own tile positions (read above by this thread only): no barrier before, one after for the consumer
        #pragma unroll
        for(u32 r1=0;r1<16;++r1)sA[136u*r1]=v[r1];
        j1_bar128(bar_id);return;}
    j1_bar128(bar_id);
    #pragma unroll
    for(u32 r1=0;r1<16;++r1)y[4096u*r1]=v[r1];
}
// standalone wrappers: grid (32 column blocks, K[, 2 operands]), 128 threads, 64 registers (8 blocks/SM)
extern "C" __global__ void __launch_bounds__(128,8) j1_colfwd(const u32* X0,const u32* X1,u32* Y0,u32* Y1,
    const u32* primes,const u64* tw1){
    __shared__ u32 S[J1_COL_SMEM];
    j1_colfwd_body(blockIdx.x,blockIdx.y,threadIdx.x,1u,S,blockIdx.z?X1:X0,blockIdx.z?Y1:Y0,primes,tw1);
}
extern "C" __global__ void __launch_bounds__(128,8) j1_colinv(const u32* X,u32* Y,const u32* primes,const u64* tw1){
    __shared__ u32 S[J1_COL_SMEM];
    j1_colinv_body(blockIdx.x,blockIdx.y,threadIdx.x,1u,S,X,Y,primes,tw1);
}
// staged test wrappers (J1_STAGED harness mode): a stand-in producer fills the tile from global memory in a different
// thread order (as a conversion epilogue would), and a stand-in consumer copies the tile out; same grids as above.
extern "C" __global__ void __launch_bounds__(128,8) j1_colfwd_s(const u32* X0,const u32* X1,u32* Y0,u32* Y1,
    const u32* primes,const u64* tw1){
    __shared__ u32 S[J1_COL_SMEM];
    const u32* X=(blockIdx.z?X1:X0)+u64(blockIdx.y)*65536u+blockIdx.x*8u;
    for(u32 i=threadIdx.x;i<2048u;i+=128u){const u32 r=i>>3,cl=i&7u;S[j1_sidx(r,cl)]=X[256u*r+cl];}
    j1_bar128(1u);
    j1_colfwd_body<true>(blockIdx.x,blockIdx.y,threadIdx.x,1u,S,nullptr,blockIdx.z?Y1:Y0,primes,tw1);
}
extern "C" __global__ void __launch_bounds__(128,8) j1_colinv_s(const u32* X,u32* Y,const u32* primes,const u64* tw1){
    __shared__ u32 S[J1_COL_SMEM];
    j1_colinv_body<true>(blockIdx.x,blockIdx.y,threadIdx.x,1u,S,X,nullptr,primes,tw1);
    u32* y=Y+u64(blockIdx.y)*65536u+blockIdx.x*8u;
    for(u32 i=threadIdx.x;i<2048u;i+=128u){const u32 r=i>>3,cl=i&7u;y[256u*r+cl]=S[j1_sidx(r,cl)];}
}
// ---- stages 9-16 of a and b, pointwise, inverse stages 16-9: one warp per row pair (r, 255 - r) ----
// Lane (h, x): h = lane / 16 picks the row, x = lane % 16.  Phase 1 (stages 9-12 / 12-9): x = c2, register c1
// (c = 16 c1 + c2); phase 2 (stages 13-16): x = c1, register c2.  Twiddle j = 2^(s-1) + (n >> (17-s)), value
// psi^e, e = brev16(j) = 256 e_hi + e_lo: A = thi[dir][e_hi] (psi^(+-256 e_hi)), Bs = tlo[dir][e_lo]
// (floor(psi^(+-e_lo) 2^64 / p)); per prime thi [2][256] u32, tlo [2][256] u64.
// Phase-1 twiddles (15 per row and direction, uniform over c2) are made by lanes x < 15 and read from smem.
// Phase-2 thread twiddles: j = 2^(12+d) + (r << (4+d)) + (c1 << d) + k, k < 2^d: e = e0 + (brev_d(k) << (16-d)),
// so e_lo = e0 & 255 is shared by the 2^d twiddles of a stage (one tlo load per stage).
// Per integer warp smem: J1_ROW_SMEM_WORDS = 544 transpose words + 64 u64 twiddle slots = 2688 B.
#define J1_ROW_SMEM_WORDS 544
template<u32 D>__device__ __forceinline__ u32 j1_p2e0(u32 r,u32 c1){return j1_brev16((4096u<<D)+(r<<(4+D))+(c1<<D));}
// phase-2 stage D over c2 (half = 8 >> D, block k = c2 >> (4-D)): each twiddle is made just before its butterflies
template<u32 D>__device__ __forceinline__ void j1_p2fwd(u32 (&a)[16],u32 (&b)[16],u32 r,u32 c1,const u32* thi,const u64* tlo,u32 p,u32 p2,u32 one){
    const u32 e0=j1_p2e0<D>(r,c1);const u64 bs=tlo[e0&255u];const u32* th=thi+(e0>>8);
    constexpr u32 half=8u>>D;
    #pragma unroll
    for(u32 k=0;k<(1u<<D);++k){const u64 w=j1_twgen(th[j1_brevd<D>(k)<<(8-D)],bs,p,one);
        #pragma unroll
        for(u32 c2=k<<(4-D);c2<((k+1u)<<(4-D));++c2)if(!(c2&half)){j1_ct(a[c2],a[c2+half],w,p,p2);j1_ct(b[c2],b[c2+half],w,p,p2);}}
}
template<u32 D>__device__ __forceinline__ void j1_p2inv(u32 (&a)[16],u32 r,u32 c1,const u32* thi,const u64* tlo,u32 p,u32 p2,u32 one){
    const u32 e0=j1_p2e0<D>(r,c1);const u64 bs=tlo[256u+(e0&255u)];const u32* th=thi+256u+(e0>>8);
    constexpr u32 half=8u>>D;
    #pragma unroll
    for(u32 k=0;k<(1u<<D);++k){const u64 w=j1_twgen(th[j1_brevd<D>(k)<<(8-D)],bs,p,one);
        #pragma unroll
        for(u32 c2=k<<(4-D);c2<((k+1u)<<(4-D));++c2)if(!(c2&half))j1_gs(a[c2],a[c2+half],w,p,p2);}
}
// Warp body: row pair rp (rows rp and 255 - rp, rp < 128) of one prime.  S: J1_ROW_SMEM_WORDS words, TW: 64 u64.
__device__ __forceinline__ void j1_row_body(u32 rp,u32 prime,u32 lane,u32* S,u64* TW,
    const u32* A,const u32* B,u32* C,const u32* primes,const u32* pinvs,const u32* thi_,const u64* tlo_){
    const u32 p=primes[prime],p2=j1_opq2(p),pinv=pinvs[prime],one=j1_opq1();
    const u32 h=lane>>4,x=lane&15u,r=h?255u-rp:rp;
    const u32* thi=thi_+u64(prime)*512u;const u64* tlo=tlo_+u64(prime)*512u;
    const u64 base=u64(prime)*65536u+r*256u+x;
    u32 a[16],b[16];
    #pragma unroll
    for(u32 c1=0;c1<16;++c1){a[c1]=A[base+16u*c1];b[c1]=B[base+16u*c1];}
    u64* const tf=TW+32u*h;u64* const ti=tf+16u;
    if(x<15u){const u32 d=31u-__clz(x+1u),k=x+1u-(1u<<d),e=j1_brev16((256u<<d)+(r<<d)+k);
        tf[x]=j1_twgen(thi[e>>8],tlo[e&255u],p,one);
        ti[x]=j1_twgen(thi[256u+(e>>8)],tlo[256u+(e&255u)],p,one);}
    __syncwarp();
    // stages 9-12 over c1: stage 9+d pairs c1 bit 3-d, i = (r << d) | (c1 >> (4-d)), table slot 2^d - 1 + (c1 >> (4-d))
    #pragma unroll
    for(u32 d=0;d<4;++d){const u32 half=8u>>d;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1)if(!(c1&half)){const u64 w=tf[(1u<<d)-1u+(c1>>(4-d))];j1_ct(a[c1],a[c1+half],w,p,p2);j1_ct(b[c1],b[c1+half],w,p,p2);}}
    u32* const sw=S+h*272u+x;        // [17 c1 + c2] at c2 = x (phase-1 write / phase-1 read)
    u32* const sr=S+h*272u+17u*x;    // row c1 = x (phase-2 read / write)
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)sw[17u*c1]=a[c1];
    __syncwarp();
    #pragma unroll
    for(u32 c2=0;c2<16;++c2)a[c2]=sr[c2];
    __syncwarp();
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)sw[17u*c1]=b[c1];
    __syncwarp();
    #pragma unroll
    for(u32 c2=0;c2<16;++c2)b[c2]=sr[c2];
    // stages 13-16 over c2 (thread c1 = x)
    j1_p2fwd<0>(a,b,r,x,thi,tlo,p,p2,one);j1_p2fwd<1>(a,b,r,x,thi,tlo,p,p2,one);
    j1_p2fwd<2>(a,b,r,x,thi,tlo,p,p2,one);j1_p2fwd<3>(a,b,r,x,thi,tlo,p,p2,one);
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=j1_pw(a[i],b[i],p,p2,pinv);
    j1_p2inv<3>(a,r,x,thi,tlo,p,p2,one);j1_p2inv<2>(a,r,x,thi,tlo,p,p2,one);
    j1_p2inv<1>(a,r,x,thi,tlo,p,p2,one);j1_p2inv<0>(a,r,x,thi,tlo,p,p2,one);
    __syncwarp();
    #pragma unroll
    for(u32 c2=0;c2<16;++c2)sr[c2]=a[c2];
    __syncwarp();
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)a[c1]=sw[17u*c1];
    #pragma unroll
    for(u32 dd=0;dd<4;++dd){const u32 d=3u-dd,half=8u>>d;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1)if(!(c1&half))j1_gs(a[c1],a[c1+half],ti[(1u<<d)-1u+(c1>>(4-d))],p,p2);}
    __syncwarp();   // smem (tile, twiddles) free for the warp's next row pair
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)C[base+16u*c1]=a[c1];
}
// standalone wrapper: grid (32, K), 128 threads = 4 row-pair warps, 64 registers
extern "C" __global__ void __launch_bounds__(128,8) j1_row(const u32* A,const u32* B,u32* C,
    const u32* primes,const u32* pinvs,const u32* thi,const u64* tlo){
    __shared__ u32 S[4][J1_ROW_SMEM_WORDS];
    __shared__ u64 TW[4][64];
    const u32 wl=threadIdx.x>>5;
    j1_row_body(blockIdx.x*4u+wl,blockIdx.y,threadIdx.x&31u,S[wl],TW[wl],A,B,C,primes,pinvs,thi,tlo);
}
#endif
