// Pass-order restructure (N=65536, R=S=256, J=8), NVRTC-specialized.
// Same exact modular values as prepare2 -> forward -> middle -> inverse; only the
// placement of the (commuting) J-axis transforms and the intermediate layouts differ:
//   P1 bpo_first : digit split + twist + J-DIF + R-DIF + cross twiddle (one block per column)
//                  out layout W1[r*8+j][slot/8][c][slot%8]
//   P2 bpo_middle: S-DIF (A,B), pointwise, S-DIT, cross untwiddle (block per (8 slots, j, r))
//                  out layout W2[r*8+j][c][slot]
//   P3 bpo_last  : J-DIT + R-DIT + untwist + normalization (block per (8 columns, r))
//                  out layout natural [r*8+j][i] for batched_crt.
#ifdef BPR_DT
// Stage twiddle tables (host-built, per prime, 256 u64): entry (len,j) = psi^(2j*N/len) at 256-len+j.
#define BPO_RDIF(v,st)                                                                    \
    _Pragma("unroll") for(u32 m=0;m<4;++m){const u64 w=st[lane+32u*m];                    \
        const u32 x=v[m],y=v[m+4];v[m]=add(x,y,p);v[m+4]=mul_shoup(sub(x,y,p),w,p);}       \
    _Pragma("unroll") for(u32 q=0;q<2;++q){const u64 w=st[128u+lane+32u*q];               \
        _Pragma("unroll") for(u32 hh=0;hh<2;++hh){const u32 m=q+4*hh;                     \
            const u32 x=v[m],y=v[m+2];v[m]=add(x,y,p);v[m+2]=mul_shoup(sub(x,y,p),w,p);}} \
    {   const u64 w=st[192u+lane];                                                        \
        _Pragma("unroll") for(u32 m=0;m<8;m+=2){                                          \
            const u32 x=v[m],y=v[m+1];v[m]=add(x,y,p);v[m+1]=mul_shoup(sub(x,y,p),w,p);}}  \
    _Pragma("unroll") for(u32 len=32;len>=2;len>>=1){                                     \
        const u32 half=len/2;const bool upper=(lane&half)!=0;                             \
        const u64 w=st[256u-len+(lane&(half-1))];                                         \
        _Pragma("unroll") for(u32 m=0;m<8;++m){                                           \
            const u32 x=v[m],y=__shfl_xor_sync(0xffffffffu,x,half);                       \
            v[m]=upper?mul_shoup(sub(y,x,p),w,p):add(x,y,p);}}
#define BPO_RDIT(v,sit)                                                                   \
    _Pragma("unroll") for(u32 len=2;len<=32;len<<=1){                                     \
        const u32 half=len/2;const bool upper=(lane&half)!=0;                             \
        const u64 w=sit[256u-len+(lane&(half-1))];                                        \
        _Pragma("unroll") for(u32 m=0;m<8;++m){                                           \
            const u32 x=upper?mul_shoup(v[m],w,p):v[m];                                   \
            const u32 y=__shfl_xor_sync(0xffffffffu,x,half);                              \
            v[m]=upper?sub(y,x,p):add(x,y,p);}}                                           \
    {   const u64 w=sit[192u+lane];                                                       \
        _Pragma("unroll") for(u32 m=0;m<8;m+=2){                                          \
            const u32 x=v[m],y=mul_shoup(v[m+1],w,p);v[m]=add(x,y,p);v[m+1]=sub(x,y,p);}}  \
    _Pragma("unroll") for(u32 q=0;q<2;++q){const u64 w=sit[128u+lane+32u*q];              \
        _Pragma("unroll") for(u32 hh=0;hh<2;++hh){const u32 m=q+4*hh;                     \
            const u32 x=v[m],y=mul_shoup(v[m+2],w,p);v[m]=add(x,y,p);v[m+2]=sub(x,y,p);}} \
    _Pragma("unroll") for(u32 m=0;m<4;++m){const u64 w=sit[lane+32u*m];                   \
        const u32 x=v[m],y=mul_shoup(v[m+4],w,p);v[m]=add(x,y,p);v[m+4]=sub(x,y,p);}
// length-8 cyclic DIF / DIT over the digit axis: psi^(2j*N/len) = table entry j*256/len
#define BPO_JDIF(x8,st)                                                                   \
    _Pragma("unroll") for(u32 len=8;len>=2;len>>=1){const u32 half=len/2;                 \
        _Pragma("unroll") for(u32 d0=0;d0<8;d0+=len){                                     \
            _Pragma("unroll") for(u32 jj=0;jj<half;++jj){                                 \
                const u64 w=st[jj*(256u/len)];                                            \
                const u32 x=x8[d0+jj],y=x8[d0+jj+half];                                   \
                x8[d0+jj]=add(x,y,p);x8[d0+jj+half]=mul_shoup(sub(x,y,p),w,p);}}}
#define BPO_JDIT(x8,sit)                                                                  \
    _Pragma("unroll") for(u32 len=2;len<=8;len<<=1){const u32 half=len/2;                 \
        _Pragma("unroll") for(u32 d0=0;d0<8;d0+=len){                                     \
            _Pragma("unroll") for(u32 jj=0;jj<half;++jj){                                 \
                const u32 x=x8[d0+jj],y=mul_shoup(x8[d0+jj+half],sit[jj*(256u/len)],p);   \
                x8[d0+jj]=add(x,y,p);x8[d0+jj+half]=sub(x,y,p);}}}

#ifndef BPO_ABL
#define BPO_ABL 0
#endif
#ifndef BPO_FOLDW
#define BPO_FOLDW 0
#endif
#ifndef BPO_P1_MINB
#define BPO_P1_MINB 2
#endif
__constant__ u32 bpo_wt[BCRT_K*8*BPR_DT]={BPO_WEIGHTS};   // digit weight * per-digit factor (mod p)
#ifndef BPO_NH30
#define BPO_NH30 0   // 12 with the default h30 prime set (Harvey-lazy P2 / P3 butterflies for the primes < 2^30); 0 with FHERMA_NO_H30=1
#endif
#if BPO_NH30
struct BpoP2{u32 v[BCRT_K];};__host__ __device__ constexpr BpoP2 bpo_p2_make(){BpoP2 a{};constexpr u32 P[BCRT_K]={BCRT_PRIMES};for(u32 i=0;i<BCRT_K;++i)a.v[i]=2u*P[i];return a;}
__constant__ BpoP2 bpo_p2c=bpo_p2_make();   // 2p from constant memory: computed 2p is folded by ptxas into IMAD p*-2+s + VIMNMX per correction
// 2p of a block-uniform prime p (the kernel's prime pointer may be group-offset): constant-table lookup by value
__device__ __forceinline__ u32 bpo_p2_of(u32 p){u32 r=0;
    #pragma unroll
    for(u32 i=0;i<BCRT_K;++i)if(bpr_pr[i]==p)r=bpo_p2c.v[i];
    return r;}
#else
__device__ __forceinline__ u32 bpo_p2_of(u32){return 0u;}   // H paths not instantiated
#endif
// grid (256 columns, 2 operands), 256 threads: thread h owns coefficient i=256h+c.
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256,BPO_P1_MINB) bpo_first(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,u32 N,u32 logR,u32 r0,u32 r1){
    constexpr u32 W=BCRT_W,L=BCRT_L,DT=BPR_DT;
    constexpr u32 POS[8]={BCRT_POS};
#ifndef BPO_STAGE_ROWS
#define BPO_STAGE_ROWS 128
#endif
    constexpr u32 SR=BPO_STAGE_ROWS;
    __shared__ __align__(16) u32 SG[(SR*L>4096?SR*L:4096)];   // staging buffer, then the exchange tiles
    __shared__ u64 CT[2][256];
    u32 (*X)[8*256]=reinterpret_cast<u32(*)[8*256]>(SG);
    const u32 c=blockIdx.x,op=blockIdx.y,h=threadIdx.x,lane=h&31u,warp=h>>5;
    const u32 i=h*256u+c;
    const u32* src=(op?b:a)+u64(c)*L;   // row h at src + h*256*L
    u32* dst=op?B:A;
    u32 row[L];
#if BPO_ABL==7
    #pragma unroll
    for(u32 w=0;w<L;++w)row[w]=(h*0x9e3779b9u)^(w*c+op);
    if(0)
#endif
    {   // coalesced staging, 64 rows (3520 words) at a time through X
        u32* sg=SG;
        #pragma unroll 1
        for(u32 q=0;q<256/SR;++q){
            __syncthreads();
            for(u32 x=h;x<SR*L;x+=256){const u32 rr=x/L,w=x-rr*L;sg[x]=__ldg(src+u64((SR*q+rr)*256u)*L+w);}
            __syncthreads();
            if(h/SR==q){
                #pragma unroll
                for(u32 w=0;w<L;++w)row[w]=sg[(h%SR)*L+w];
            }
        }
        __syncthreads();
    }
    const u32 stride=weighted_root_stride(N);
    const u32 k1h=__brev(h)>>(32-logR);
    #pragma unroll 1
    for(u32 r=r0;r<r1;++r){
        const u32 p=bpr_pr[r];const u64 mu=bpr_mu[r];
        const u64* tw=twist+size_t(r)*stride;
        const u64* st=stage_tw+size_t(r)*256;
#if WROOT_COMPACT
        // psi^i = t1(i&127) t2(i>>7); t1 = t1(c&127) is constant over the column, so it commutes
        // with the R-DIF and is folded into the cross twiddle of slot h (shared by all digits).
        const u64 t2=tw[1+128u+(i>>7)];
        {
            const u32 wv=mul_shoup(u32(bm_root(tw,2u*c*k1h,N,p)),tw[1+(c&127u)],p);
            const u64 m0=tw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
            const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
            CT[r&1][h]=u64(wv)|(u64(q)<<32);
        }
#else
        const u64 t1=tw[i];
        CT[r&1][h]=bm_root(tw,2u*c*k1h,N,p);
#endif
        u32 x8[8];
        #pragma unroll
        for(u32 j=0;j<8;++j){
            const u32 begin=POS[j],end=j==7?W:POS[j+1];
            u64 sum=0;
            #pragma unroll
            for(u32 k=0;k<DT;++k){
                const u32 bit=begin+16*k;
                if(bit<end&&BPO_ABL!=6){
                    const u32 word=bit/32,shift=bit%32,count=end-bit<16?end-bit:16;
                    const u32 lo=row[word],hi=word+1<L?row[word+1]:0;
                    const u32 v=__funnelshift_r(lo,hi,shift)&((1u<<count)-1);
#if BPO_FOLDW
                    sum+=u64(v)*bpo_wt[(r*8+j)*DT+k];
#else
                    sum+=u64(v)*bpr_wt[r*DT+k];
#endif
                }
            }
#if BPO_ABL==6
            u32 x=row[j*6]+(u32)sum;
#else
            u32 x=reduce(sum,p,mu);
#endif
#if !BPO_FOLDW
            x=mul_shoup(x,bpr_fa[r*8+j],p);
#endif
#if WROOT_COMPACT
            x8[j]=mul_shoup(x,t2,p);
#else
            x8[j]=mul_shoup(x,t1,p);
#endif
        }
        BPO_JDIF(x8,st)
        u32* Xb=X[r&1];
        #pragma unroll
        for(u32 j=0;j<8;++j)Xb[j*256+h]=x8[j];
        __syncthreads();
        u32 v[8];
        #pragma unroll
        for(u32 m=0;m<8;++m)v[m]=Xb[warp*256+lane+32*m];
#if BPO_ABL!=1
        BPO_RDIF(v,st)
#endif
        u32* out=dst+u64(r*8+warp)*N+c*8u;
        #pragma unroll
        for(u32 m=0;m<8;++m){
            const u32 slot=lane+32*m;
            out[(slot>>3)*2048u+(slot&7u)]=mul_shoup(v[m],CT[r&1][slot],p);
        }
    }
}
#endif // bpo_first

// grid (32 slot groups, 8 digits, primes), 256 threads: warp s owns slot 8g+s, lane+32m = column.
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256) bpo_middle(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* ratios,const u64* untwist,const u64* stage_tw,const u64* stage_itw,u32 N,u32 logR,u32 untw){
    __shared__ u32 sa[256*9],sb[256*9];
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const u64 mu=ratios[prime];
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* st=stage_tw+prime*256u;const u64* sit=stage_itw+prime*256u;
    const u64 base=u64(prime*8+j)*N;
    {
        const uint4* a4=reinterpret_cast<const uint4*>(A+base+g*2048u);
        const uint4* b4=reinterpret_cast<const uint4*>(B+base+g*2048u);
        #pragma unroll
        for(u32 q=threadIdx.x;q<512;q+=256){
            const uint4 va=a4[q],vb=b4[q];const u32 cc=q>>1,s=(q&1u)*4;u32* xa=sa+cc*9+s;u32* xb=sb+cc*9+s;
            xa[0]=va.x;xa[1]=va.y;xa[2]=va.z;xa[3]=va.w;xb[0]=vb.x;xb[1]=vb.y;xb[2]=vb.z;xb[3]=vb.w;
        }
    }
    __syncthreads();
    u32 a[8],b[8];
    #pragma unroll
    for(u32 m=0;m<8;++m){a[m]=sa[(lane+32*m)*9+warp];b[m]=sb[(lane+32*m)*9+warp];}
    if(untw&2){
        #pragma unroll
        for(u32 m=0;m<8;++m)a[m]^=b[m];
    }else{
    {   // S-axis DIF for A and B (same butterflies as batched_middle phase 1)
        #pragma unroll
        for(u32 m=0;m<4;++m){
            const u64 w=st[lane+32u*m];
            u32 x=a[m],y=a[m+4];a[m]=add(x,y,p);a[m+4]=mul_shoup(sub(x,y,p),w,p);
            x=b[m];y=b[m+4];b[m]=add(x,y,p);b[m+4]=mul_shoup(sub(x,y,p),w,p);
        }
        #pragma unroll
        for(u32 q=0;q<2;++q){
            const u64 w=st[128u+lane+32u*q];
            #pragma unroll
            for(u32 hh=0;hh<2;++hh){const u32 m=q+4*hh;
                u32 x=a[m],y=a[m+2];a[m]=add(x,y,p);a[m+2]=mul_shoup(sub(x,y,p),w,p);
                x=b[m];y=b[m+2];b[m]=add(x,y,p);b[m+2]=mul_shoup(sub(x,y,p),w,p);}
        }
        {
            const u64 w=st[192u+lane];
            #pragma unroll
            for(u32 m=0;m<8;m+=2){
                u32 x=a[m],y=a[m+1];a[m]=add(x,y,p);a[m+1]=mul_shoup(sub(x,y,p),w,p);
                x=b[m];y=b[m+1];b[m]=add(x,y,p);b[m+1]=mul_shoup(sub(x,y,p),w,p);
            }
        }
        #pragma unroll
        for(u32 len=32;len>=2;len>>=1){
            const u32 half=len/2;const bool upper=(lane&half)!=0;
            const u64 w=st[256u-len+(lane&(half-1))];
            #pragma unroll
            for(u32 m=0;m<8;++m){
                const u32 x=a[m],y=__shfl_xor_sync(0xffffffffu,x,half);
                a[m]=upper?mul_shoup(sub(y,x,p),w,p):add(x,y,p);
                const u32 u=b[m],vv=__shfl_xor_sync(0xffffffffu,u,half);
                b[m]=upper?mul_shoup(sub(vv,u,p),w,p):add(u,vv,p);
            }
        }
    }
    #pragma unroll
    for(u32 m=0;m<8;++m)a[m]=mul(a[m],b[m],p,mu);
    BPO_RDIT(a,sit)
    }
    const u32 k1=__brev(g*8u+warp)>>(32-logR);
    if(untw&1)
    #pragma unroll
    for(u32 m=0;m<8;++m){
        const u32 col=lane+32*m;
#ifdef TW_PACKED
        a[m]=mul_shoup(a[m],bm_root(itw,2u*col*k1,N,p),p);
#else
        a[m]=weighted_root_action(a[m],itw,2u*col*k1,N,p);
#endif
    }
    __syncthreads();   // all tile reads done
    #pragma unroll
    for(u32 m=0;m<8;++m)sa[(lane+32*m)*9+warp]=a[m];
    __syncthreads();
    {
        const u32 cc=threadIdx.x;const u32* s=sa+cc*9;
        uint4* o=reinterpret_cast<uint4*>(C+base+cc*256u+g*8u);
        o[0]=make_uint4(s[0],s[1],s[2],s[3]);o[1]=make_uint4(s[4],s[5],s[6],s[7]);
    }
}
#endif // bpo_middle

// P2 compute+store of one tile already in sa/sb (caller synced); leaves smem free on return.
__device__ __forceinline__ void bpo_mid_tile(u32* C,u32* sa,const u32* sb,u32 p,u64 mu,
    const u64* itw,const u64* st,const u64* sit,u32 N,u32 logR,u32 untw,u32 g,u64 base,u32 lane,u32 warp){
    u32 a[8],b[8];
    #pragma unroll
    for(u32 m=0;m<8;++m){a[m]=sa[(lane+32*m)*9+warp];b[m]=sb[(lane+32*m)*9+warp];}
    if(untw&2){
        #pragma unroll
        for(u32 m=0;m<8;++m)a[m]^=b[m];
    }else{
    {   // S-axis DIF for A and B (same butterflies as batched_middle phase 1)
        #pragma unroll
        for(u32 m=0;m<4;++m){
            const u64 w=st[lane+32u*m];
            u32 x=a[m],y=a[m+4];a[m]=add(x,y,p);a[m+4]=mul_shoup(sub(x,y,p),w,p);
            x=b[m];y=b[m+4];b[m]=add(x,y,p);b[m+4]=mul_shoup(sub(x,y,p),w,p);
        }
        #pragma unroll
        for(u32 q=0;q<2;++q){
            const u64 w=st[128u+lane+32u*q];
            #pragma unroll
            for(u32 hh=0;hh<2;++hh){const u32 m=q+4*hh;
                u32 x=a[m],y=a[m+2];a[m]=add(x,y,p);a[m+2]=mul_shoup(sub(x,y,p),w,p);
                x=b[m];y=b[m+2];b[m]=add(x,y,p);b[m+2]=mul_shoup(sub(x,y,p),w,p);}
        }
        {
            const u64 w=st[192u+lane];
            #pragma unroll
            for(u32 m=0;m<8;m+=2){
                u32 x=a[m],y=a[m+1];a[m]=add(x,y,p);a[m+1]=mul_shoup(sub(x,y,p),w,p);
                x=b[m];y=b[m+1];b[m]=add(x,y,p);b[m+1]=mul_shoup(sub(x,y,p),w,p);
            }
        }
        #pragma unroll
        for(u32 len=32;len>=2;len>>=1){
            const u32 half=len/2;const bool upper=(lane&half)!=0;
            const u64 w=st[256u-len+(lane&(half-1))];
            #pragma unroll
            for(u32 m=0;m<8;++m){
                const u32 x=a[m],y=__shfl_xor_sync(0xffffffffu,x,half);
                a[m]=upper?mul_shoup(sub(y,x,p),w,p):add(x,y,p);
                const u32 u=b[m],vv=__shfl_xor_sync(0xffffffffu,u,half);
                b[m]=upper?mul_shoup(sub(vv,u,p),w,p):add(u,vv,p);
            }
        }
    }
    #pragma unroll
    for(u32 m=0;m<8;++m)a[m]=mul(a[m],b[m],p,mu);
    BPO_RDIT(a,sit)
    }
    const u32 k1=__brev(g*8u+warp)>>(32-logR);
    if(untw&1)
    #pragma unroll
    for(u32 m=0;m<8;++m){
        const u32 col=lane+32*m;
#ifdef TW_PACKED
        a[m]=mul_shoup(a[m],bm_root(itw,2u*col*k1,N,p),p);
#else
        a[m]=weighted_root_action(a[m],itw,2u*col*k1,N,p);
#endif
    }
    __syncthreads();   // all tile reads done
    #pragma unroll
    for(u32 m=0;m<8;++m)sa[(lane+32*m)*9+warp]=a[m];
    __syncthreads();
    {
        const u32 cc=threadIdx.x;const u32* s=sa+cc*9;
        uint4* o=reinterpret_cast<uint4*>(C+base+cc*256u+g*8u);
        o[0]=make_uint4(s[0],s[1],s[2],s[3]);o[1]=make_uint4(s[4],s[5],s[6],s[7]);
    }
}
// Double-buffered P2: grid (32 slot groups, 4 digit pairs, primes); tile (g, 2jb+1) global loads
// are in flight while tile (g, 2jb) is computed.
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256) bpo_middle_db(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* ratios,const u64* untwist,const u64* stage_tw,const u64* stage_itw,u32 N,u32 logR,u32 untw){
    __shared__ u32 sa[256*9],sb[256*9];
    const u32 g=blockIdx.x,j0=2*blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const u64 mu=ratios[prime];
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* st=stage_tw+prime*256u;const u64* sit=stage_itw+prime*256u;
    uint4 va[2],vb[2];
    #pragma unroll
    for(u32 t=0;t<2;++t){
        const u64 base=u64(prime*8+j0+t)*N;
        if(t==0){
            const uint4* a4=reinterpret_cast<const uint4*>(A+base+g*2048u);const uint4* b4=reinterpret_cast<const uint4*>(B+base+g*2048u);
            #pragma unroll
            for(u32 x=0;x<2;++x){va[x]=a4[threadIdx.x+256*x];vb[x]=b4[threadIdx.x+256*x];}
        }
        __syncthreads();   // previous tile's smem reads done
        #pragma unroll
        for(u32 x=0;x<2;++x){
            const u32 q=threadIdx.x+256*x,cc=q>>1,s=(q&1u)*4;u32* xa=sa+cc*9+s;u32* xb=sb+cc*9+s;
            xa[0]=va[x].x;xa[1]=va[x].y;xa[2]=va[x].z;xa[3]=va[x].w;xb[0]=vb[x].x;xb[1]=vb[x].y;xb[2]=vb[x].z;xb[3]=vb[x].w;
        }
        __syncthreads();
        if(t==0){   // prefetch the second tile
            const u64 nb=base+N;
            const uint4* a4=reinterpret_cast<const uint4*>(A+nb+g*2048u);const uint4* b4=reinterpret_cast<const uint4*>(B+nb+g*2048u);
            #pragma unroll
            for(u32 x=0;x<2;++x){va[x]=a4[threadIdx.x+256*x];vb[x]=b4[threadIdx.x+256*x];}
        }
        bpo_mid_tile(C,sa,sb,p,mu,itw,st,sit,N,logR,untw,g,base,lane,warp);
    }
}
#endif // bpo_middle_db
// grid (32 column groups, primes), 512 threads. Half hf owns columns c0+4hf..+3. Per column:
// thread t = slot does the J-DIT, then warp j owns digit j (lane+32m = row) for the R-DIT,
// untwist and normalization; each lane finally stores 4 consecutive columns (16 B).
#ifndef BPO_P3_H
#define BPO_P3_H 2
#endif
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256*BPO_P3_H) bpo_last(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* normalized_untwist,u32 N){
    __shared__ u32 X[BPO_P3_H][2][8*256];
    const u32 hf=threadIdx.x>>8,t=threadIdx.x&255u,lane=t&31u,warp=t>>5;
    const u32 c0=blockIdx.x*(4*BPO_P3_H)+4*hf,prime=blockIdx.y,p=primes[prime];
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* sit=stage_itw+prime*256u;
    const u64 nrm=normalized_untwist[prime*8+warp];
    const u32* src=C+u64(prime*8)*N+c0*256u+t;
    u32 res[4][8];
    u32 nx[8];
    #pragma unroll
    for(u32 d=0;d<8;++d)nx[d]=src[u64(d)*N];
    #pragma unroll
    for(u32 cc=0;cc<4;++cc){
        u32 yv[8];
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=nx[d];
        if(cc<3){
            #pragma unroll
            for(u32 d=0;d<8;++d)nx[d]=src[u64(d)*N+(cc+1)*256u];
        }
        BPO_JDIT(yv,sit)
        u32* Xb=X[hf][cc&1];
        #pragma unroll
        for(u32 d=0;d<8;++d)Xb[d*256+t]=yv[d];
        __syncthreads();
        u32 v[8];
        #pragma unroll
        for(u32 m=0;m<8;++m)v[m]=Xb[warp*256+lane+32*m];
        BPO_RDIT(v,sit)
        #pragma unroll
        for(u32 m=0;m<8;++m){
            const u32 ii=(lane+32*m)*256+c0+cc;
            res[cc][m]=mul_shoup(weighted_root_action(v[m],itw,ii,N,p),nrm,p);
        }
    }
    u32* out=D+u64(prime*8+warp)*N+c0;
    #pragma unroll
    for(u32 m=0;m<8;++m)
        *reinterpret_cast<uint4*>(out+(lane+32*m)*256u)=make_uint4(res[0][m],res[1][m],res[2][m],res[3][m]);
}
#endif // bpo_last

// Column-major variant of P3: output [r*8+j][c][h] (rows of one column contiguous), consumed by
// bpo_crt_cm. grid (256 columns, primes), 256 threads, one sync per block, 128 B stores per warp.
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256) bpo_last_cm(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* normalized_untwist,u32 N){
    __shared__ u32 X[8*256];
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 t=threadIdx.x,lane=t&31u,warp=t>>5;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* sit=stage_itw+prime*256u;
    const u64 nrm=normalized_untwist[prime*8+warp];
    const u32* src=C+u64(prime*8)*N+c*256u+t;
    u32 yv[8];
    #pragma unroll
    for(u32 d=0;d<8;++d)yv[d]=src[u64(d)*N];
    BPO_JDIT(yv,sit)
    #pragma unroll
    for(u32 d=0;d<8;++d)X[d*256+t]=yv[d];
    __syncthreads();
    u32 v[8];
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=X[warp*256+lane+32*m];
    BPO_RDIT(v,sit)
    u32* out=D+u64(prime*8+warp)*N+c*256u+lane;
    #pragma unroll
    for(u32 m=0;m<8;++m){
        const u32 ii=(lane+32*m)*256+c;
        out[32*m]=mul_shoup(weighted_root_action(v[m],itw,ii,N,p),nrm,p);
    }
}
#endif // bpo_last_cm

// Variant-4 P3: the cross untwiddle psi^-(2 c k1) moved here from P2 (constant over the digits of
// one (column,slot), so built once per thread) with the column-constant untwist factor folded in;
// after the R-DIT one table factor t2n[r][j][(i>>7)&1][i>>8] = psi^-(128(i>>7)) * nrm_j.
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256) bpo_last_cm2(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR){
    __shared__ u32 X[8*256];
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 t=threadIdx.x,lane=t&31u,warp=t>>5;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* sit=stage_itw+prime*256u;
    const u32* src=C+u64(prime*8)*N+c*256u+t;
    u32 yv[8];
    #pragma unroll
    for(u32 d=0;d<8;++d)yv[d]=src[u64(d)*N];
    u64 F;
    {
        const u32 k1=__brev(t)>>(32-logR);
        const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
        const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
        const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
        F=u64(wv)|(u64(q)<<32);
    }
    #pragma unroll
    for(u32 d=0;d<8;++d)yv[d]=mul_shoup(yv[d],F,p);
    BPO_JDIT(yv,sit)
    #pragma unroll
    for(u32 d=0;d<8;++d)X[d*256+t]=yv[d];
    __syncthreads();
    u32 v[8];
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=X[warp*256+lane+32*m];
    BPO_RDIT(v,sit)
    const u64* tn=t2n+u64(prime*8+warp)*512u+(c>>7)*256u;
    u32* out=D+u64(prime*8+warp)*N+c*256u+lane;
    #pragma unroll
    for(u32 m=0;m<8;++m)out[32*m]=mul_shoup(v[m],tn[lane+32*m],p);
}
#endif // bpo_last_cm2
// 256-point DIF / DIT of one column held by a warp, without shuffles: three register layouts
// (row bits {5,6,7} -> {2,3,4} -> {0,1,2}) connected by intra-warp transposes through a private
// 288-word region (row -> row + 4*(row>>5)). DIF: layout 1 in, layout 3 out; DIT: the reverse.
__device__ __forceinline__ void bpo_dif_smr(u32* base,u32 (&v)[8],const u64* st,u32 p,u32 lane){
    #pragma unroll
    for(u32 m=0;m<4;++m){const u64 w=st[lane+32u*m];
        const u32 x=v[m],y=v[m+4];v[m]=add(x,y,p);v[m+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=st[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;
            const u32 x=v[m],y=v[m+2];v[m]=add(x,y,p);v[m+2]=mul_shoup(sub(x,y,p),w,p);}}
    {const u64 w=st[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=v[m],y=v[m+1];v[m]=add(x,y,p);v[m+1]=mul_shoup(sub(x,y,p),w,p);}}
    #pragma unroll
    for(u32 m=0;m<8;++m)base[lane+36*m]=v[m];
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo){const u64 w=st[224u+(lane&3u)+4*lo];
        const u32 x=v[lo],y=v[lo+4];v[lo]=add(x,y,p);v[lo+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[240u+(lane&3u)+4*(lo&1)];
        const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
    {const u64 w=st[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[252u+(lo&1)];
        const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
    {const u64 w=st[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
}
__device__ __forceinline__ void bpo_dit_smr(u32* base,u32 (&v)[8],const u64* sit,u32 p,u32 lane){
    {const u64 w=sit[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=mul_shoup(v[lo+1],w,p);v[lo]=add(x,y,p);v[lo+1]=sub(x,y,p);}}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=sit[252u+(lo&1)];
        const u32 x=v[lo],y=mul_shoup(v[lo+2],w,p);v[lo]=add(x,y,p);v[lo+2]=sub(x,y,p);}
    __syncwarp();   // previous readers of the region are done
    {uint4* p4=reinterpret_cast<uint4*>(base+8*lane+4*(lane>>2));
     p4[0]=make_uint4(v[0],v[1],v[2],v[3]);p4[1]=make_uint4(v[4],v[5],v[6],v[7]);}
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    {const u64 w=sit[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=mul_shoup(v[lo+1],w,p);v[lo]=add(x,y,p);v[lo+1]=sub(x,y,p);}}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=sit[240u+(lane&3u)+4*(lo&1)];
        const u32 x=v[lo],y=mul_shoup(v[lo+2],w,p);v[lo]=add(x,y,p);v[lo+2]=sub(x,y,p);}
    #pragma unroll
    for(u32 lo=0;lo<4;++lo){const u64 w=sit[224u+(lane&3u)+4*lo];
        const u32 x=v[lo],y=mul_shoup(v[lo+4],w,p);v[lo]=add(x,y,p);v[lo+4]=sub(x,y,p);}
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[lane+36*m];
    {const u64 w=sit[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=v[m],y=mul_shoup(v[m+1],w,p);v[m]=add(x,y,p);v[m+1]=sub(x,y,p);}}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=sit[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;
            const u32 x=v[m],y=mul_shoup(v[m+2],w,p);v[m]=add(x,y,p);v[m+2]=sub(x,y,p);}}
    #pragma unroll
    for(u32 m=0;m<4;++m){const u64 w=sit[lane+32u*m];
        const u32 x=v[m],y=mul_shoup(v[m+4],w,p);v[m]=add(x,y,p);v[m+4]=sub(x,y,p);}
}
// L2 discard of one dead 128 B line (no write-back of its dirty data); caller guarantees every read
// of the line has completed and nothing reads it before the next full overwrite.
__device__ __forceinline__ void bpo_discard(const void* p){asm volatile("discard.global.L2 [%0], 128;"::"l"(p):"memory");}
#ifndef BPO_P2ABL
#define BPO_P2ABL 0
#endif
// Lean modular helpers (same canonical values as add/sub/mul_shoup; see batched_p1x.cuh):
// unsigned-min corrections, DIF difference fed unreduced (x+p-y < 2p < 2^32) to Shoup.
__device__ __forceinline__ u32 pxadd(u32 a,u32 b,u32 p){const u32 s=a+b;return min(s,s-p);}
__device__ __forceinline__ u32 pxsub(u32 a,u32 b,u32 p){const u32 d=a-b;return min(d,d+p);}
__device__ __forceinline__ u32 pxmsh(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));const u32 r=x*u32(c)-q*p;return min(r,r-p);}
template<bool L>__device__ __forceinline__ void pxdif(u32& x,u32& y,u64 w,u32 p){
    if(L){const u32 a=x,b=y;x=pxadd(a,b,p);y=pxmsh(a+p-b,w,p);}else{const u32 a=x,b=y;x=add(a,b,p);y=mul_shoup(sub(a,b,p),w,p);}}
template<bool L>__device__ __forceinline__ void pxdif1(u32& x,u32& y,u64 w,u32 p){   // w = psi^0
    if(L){const u32 a=x,b=y;x=pxadd(a,b,p);y=pxsub(a,b,p);}else pxdif<false>(x,y,w,p);}
template<bool L>__device__ __forceinline__ void pxdit(u32& x,u32& y,u64 w,u32 p){
    if(L){const u32 a=x,t=pxmsh(y,w,p);x=pxadd(a,t,p);y=pxsub(a,t,p);}else{const u32 a=x,t=mul_shoup(y,w,p);x=add(a,t,p);y=sub(a,t,p);}}
template<bool L>__device__ __forceinline__ void pxdit1(u32& x,u32& y,u64 w,u32 p){
    if(L){const u32 a=x,t=y;x=pxadd(a,t,p);y=pxsub(a,t,p);}else pxdit<false>(x,y,w,p);}
// Montgomery product a*b*2^-32 mod p (p odd < 2^31, pinv = p^-1 mod 2^32): (ab - m p)/2^32 with m = ab*pinv mod 2^32.
__device__ __forceinline__ u32 pxmont(u32 a,u32 b,u32 p,u32 pinv){
    const u32 lo=a*b,hi=__umulhi(a,b);const u32 m=lo*pinv;const u32 mh=__umulhi(m,p);return pxsub(hi,mh,p);}
__device__ __forceinline__ u32 pxinv32(u32 p){u32 x=p;
    #pragma unroll
    for(int i=0;i<5;++i)x*=2u-p*x;return x;}   // p^-1 mod 2^32 (Newton)
// bpo_dif_smr / bpo_dit_smr with lean arithmetic (L) and the psi^0 stages (len 2, half of len 4) as add/sub.
template<bool L>__device__ __forceinline__ void pxdif_smr(u32* base,u32 (&v)[8],const u64* st,u32 p,u32 lane){
    #pragma unroll
    for(u32 m=0;m<4;++m)pxdif<L>(v[m],v[m+4],st[lane+32u*m],p);
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=st[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;pxdif<L>(v[m],v[m+2],w,p);}}
    {const u64 w=st[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2)pxdif<L>(v[m],v[m+1],w,p);}
    #pragma unroll
    for(u32 m=0;m<8;++m)base[lane+36*m]=v[m];
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)pxdif<L>(v[lo],v[lo+4],st[224u+(lane&3u)+4*lo],p);
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);pxdif<L>(v[lo],v[lo+2],st[240u+(lane&3u)+4*(lo&1)],p);}
    {const u64 w=st[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)pxdif<L>(v[lo],v[lo+1],w,p);}
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    {const u64 w0=st[252u],w1=st[253u];
     pxdif1<L>(v[0],v[2],w0,p);pxdif<L>(v[1],v[3],w1,p);pxdif1<L>(v[4],v[6],w0,p);pxdif<L>(v[5],v[7],w1,p);}
    {const u64 w=st[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)pxdif1<L>(v[lo],v[lo+1],w,p);}
}
// pxdif_smr for two operands at once (A and B of P2): same values, each step interleaved for ILP,
// one __syncwarp per layout change for both.
template<bool L>__device__ __forceinline__ void pxdif2_smr(u32* const (&base)[2],u32 (&v)[2][8],const u64* st,u32 p,u32 lane){
    #pragma unroll
    for(u32 m=0;m<4;++m){const u64 w=st[lane+32u*m];
        #pragma unroll
        for(u32 o=0;o<2;++o)pxdif<L>(v[o][m],v[o][m+4],w,p);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=st[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;
            #pragma unroll
            for(u32 o=0;o<2;++o)pxdif<L>(v[o][m],v[o][m+2],w,p);}}
    {const u64 w=st[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2){
            #pragma unroll
            for(u32 o=0;o<2;++o)pxdif<L>(v[o][m],v[o][m+1],w,p);}}
    #pragma unroll
    for(u32 o=0;o<2;++o){
        #pragma unroll
        for(u32 m=0;m<8;++m)base[o][lane+36*m]=v[o][m];}
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 o=0;o<2;++o){
        #pragma unroll
        for(u32 m=0;m<8;++m)v[o][m]=base[o][b2+4*m];}
    #pragma unroll
    for(u32 lo=0;lo<4;++lo){const u64 w=st[224u+(lane&3u)+4*lo];
        #pragma unroll
        for(u32 o=0;o<2;++o)pxdif<L>(v[o][lo],v[o][lo+4],w,p);}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[240u+(lane&3u)+4*(lo&1)];
        #pragma unroll
        for(u32 o=0;o<2;++o)pxdif<L>(v[o][lo],v[o][lo+2],w,p);}
    {const u64 w=st[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){
            #pragma unroll
            for(u32 o=0;o<2;++o)pxdif<L>(v[o][lo],v[o][lo+1],w,p);}}
    __syncwarp();
    #pragma unroll
    for(u32 o=0;o<2;++o){
        #pragma unroll
        for(u32 m=0;m<8;++m)base[o][b2+4*m]=v[o][m];}
    __syncwarp();
    #pragma unroll
    for(u32 o=0;o<2;++o){const uint4* p4=reinterpret_cast<const uint4*>(base[o]+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[o][0]=x0.x;v[o][1]=x0.y;v[o][2]=x0.z;v[o][3]=x0.w;v[o][4]=x1.x;v[o][5]=x1.y;v[o][6]=x1.z;v[o][7]=x1.w;}
    {const u64 w0=st[252u],w1=st[253u];
        #pragma unroll
        for(u32 o=0;o<2;++o){pxdif1<L>(v[o][0],v[o][2],w0,p);pxdif<L>(v[o][1],v[o][3],w1,p);pxdif1<L>(v[o][4],v[o][6],w0,p);pxdif<L>(v[o][5],v[o][7],w1,p);}}
    {const u64 w=st[254u];
        #pragma unroll
        for(u32 o=0;o<2;++o){
            #pragma unroll
            for(u32 lo=0;lo<8;lo+=2)pxdif1<L>(v[o][lo],v[o][lo+1],w,p);}}
}
template<bool L>__device__ __forceinline__ void pxdit_smr(u32* base,u32 (&v)[8],const u64* sit,u32 p,u32 lane){
    {const u64 w=sit[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)pxdit1<L>(v[lo],v[lo+1],w,p);}
    {const u64 w0=sit[252u],w1=sit[253u];
     pxdit1<L>(v[0],v[2],w0,p);pxdit<L>(v[1],v[3],w1,p);pxdit1<L>(v[4],v[6],w0,p);pxdit<L>(v[5],v[7],w1,p);}
    __syncwarp();
    {uint4* p4=reinterpret_cast<uint4*>(base+8*lane+4*(lane>>2));
     p4[0]=make_uint4(v[0],v[1],v[2],v[3]);p4[1]=make_uint4(v[4],v[5],v[6],v[7]);}
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    {const u64 w=sit[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)pxdit<L>(v[lo],v[lo+1],w,p);}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);pxdit<L>(v[lo],v[lo+2],sit[240u+(lane&3u)+4*(lo&1)],p);}
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)pxdit<L>(v[lo],v[lo+4],sit[224u+(lane&3u)+4*lo],p);
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[lane+36*m];
    {const u64 w=sit[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2)pxdit<L>(v[m],v[m+1],w,p);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=sit[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;pxdit<L>(v[m],v[m+2],w,p);}}
    #pragma unroll
    for(u32 m=0;m<4;++m)pxdit<L>(v[m],v[m+4],sit[lane+32u*m],p);
}
// Lazy DIT (p < 2^31): butterfly outputs stay in [0,2p) (sum a+t, difference a+p-t with a,t in [0,p));
// x inputs are reduced on entry (min), y inputs feed Shoup unreduced (any y < 2^32 -> [0,p) after its min).
// Saves the two output corrections of each multiplied butterfly; consumers must accept [0,2p) (Shoup does).
__device__ __forceinline__ u32 pzred(u32 x,u32 p){return min(x,x-p);}
__device__ __forceinline__ void pzdit(u32& x,u32& y,u64 w,u32 p){const u32 a=pzred(x,p),t=pxmsh(y,w,p);x=a+t;y=a+p-t;}
__device__ __forceinline__ void pzdit1(u32& x,u32& y,u64,u32 p){const u32 a=pzred(x,p),t=pzred(y,p);x=a+t;y=a+p-t;}
__device__ __forceinline__ void pzdit1c(u32& x,u32& y,u32 p){const u32 a=x,t=y;x=a+t;y=a+p-t;}   // canonical inputs
template<bool CI>__device__ __forceinline__ void pzdit_smr(u32* base,u32 (&v)[8],const u64* sit,u32 p,u32 lane){
    {const u64 w=sit[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){if(CI)pzdit1c(v[lo],v[lo+1],p);else pzdit1(v[lo],v[lo+1],w,p);}}
    {const u64 w0=sit[252u],w1=sit[253u];
     pzdit1(v[0],v[2],w0,p);pzdit(v[1],v[3],w1,p);pzdit1(v[4],v[6],w0,p);pzdit(v[5],v[7],w1,p);}
    __syncwarp();
    {uint4* p4=reinterpret_cast<uint4*>(base+8*lane+4*(lane>>2));
     p4[0]=make_uint4(v[0],v[1],v[2],v[3]);p4[1]=make_uint4(v[4],v[5],v[6],v[7]);}
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    {const u64 w=sit[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)pzdit(v[lo],v[lo+1],w,p);}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);pzdit(v[lo],v[lo+2],sit[240u+(lane&3u)+4*(lo&1)],p);}
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)pzdit(v[lo],v[lo+4],sit[224u+(lane&3u)+4*lo],p);
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[lane+36*m];
    {const u64 w=sit[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2)pzdit(v[m],v[m+1],w,p);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=sit[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;pzdit(v[m],v[m+2],w,p);}}
    #pragma unroll
    for(u32 m=0;m<4;++m)pzdit(v[m],v[m+4],sit[lane+32u*m],p);
}
__device__ __forceinline__ u32 pxmshl(u32 x,u64 c,u32 p){const u32 q=__umulhi(x,u32(c>>32));return x*u32(c)-q*p;}
template<bool H>__device__ __forceinline__ u32 pxmshh(u32 x,u64 c,u32 p){return H?pxmshl(x,c,p):pxmsh(x,c,p);}
__device__ __forceinline__ void phdit(u32& x,u32& y,u64 w,u32 p,u32 p2){const u32 a=pxadd(x,0u,p2),t=pxmshl(y,w,p);x=a+t;y=a+p2-t;}
__device__ __forceinline__ void phdit1(u32& x,u32& y,u32 p2){const u32 a=pxadd(x,0u,p2),t=pxadd(y,0u,p2);x=a+t;y=a+p2-t;}
__device__ __forceinline__ void phdit1c(u32& x,u32& y,u32 p2){const u32 a=x,t=y;x=a+t;y=a+p2-t;}   // inputs < 2p
// Harvey-lazy variant of pzdit_smr (p < 2^30): words in [0,4p) (phdit: one correction per butterfly); inputs < 4p
template<bool CI>__device__ __forceinline__ void phdit_smr(u32* base,u32 (&v)[8],const u64* sit,u32 p,u32 p2,u32 lane){
    {const u64 w=sit[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){(void)w;if(CI)phdit1c(v[lo],v[lo+1],p2);else phdit1(v[lo],v[lo+1],p2);}}
    {const u64 w0=sit[252u],w1=sit[253u];
     (void)w0;phdit1(v[0],v[2],p2);phdit(v[1],v[3],w1,p,p2);phdit1(v[4],v[6],p2);phdit(v[5],v[7],w1,p,p2);}
    __syncwarp();
    {uint4* p4=reinterpret_cast<uint4*>(base+8*lane+4*(lane>>2));
     p4[0]=make_uint4(v[0],v[1],v[2],v[3]);p4[1]=make_uint4(v[4],v[5],v[6],v[7]);}
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    {const u64 w=sit[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2)phdit(v[lo],v[lo+1],w,p,p2);}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);phdit(v[lo],v[lo+2],sit[240u+(lane&3u)+4*(lo&1)],p,p2);}
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)phdit(v[lo],v[lo+4],sit[224u+(lane&3u)+4*lo],p,p2);
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[lane+36*m];
    {const u64 w=sit[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2)phdit(v[m],v[m+1],w,p,p2);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=sit[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;phdit(v[m],v[m+2],w,p,p2);}}
    #pragma unroll
    for(u32 m=0;m<4;++m)phdit(v[m],v[m+4],sit[lane+32u*m],p,p2);
}
// P2 flags F (bpo_middle_smr: BPO_P2F, A/B twin bpo_middle_smrv: BPO_P2FV):
//   1 lean arithmetic + psi^0 stages as add/sub (S-DIF, S-DIT), 2 (with 1) A and B S-DIF interleaved,
//   4 (with 1) lazy S-DIT: C words in [0,2p) (consumer: bpo_last_cm3 Shoup F product, exact for any u32)
#ifndef BPO_P2F
#define BPO_P2F 5   // 1: 62.57 vs 64.39 ms min A/B (median 63.70 vs 66.07); 5 (+lazy S-DIT, with P3T 260)
#endif
#ifndef BPO_P2FV
#define BPO_P2FV BPO_P2F
#endif
// P2 with shuffle-free S transforms (same values as bpo_middle). The 18 KB tile becomes eight
// private per-warp regions (288 words for A in sa, 288 for B in sb) once the columns are in registers.
template<u32 F>__device__ __forceinline__ void bpo_p2_body(u32* sa,u32* sb,const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* ratios,const u64* untwist,const u64* stage_tw,const u64* stage_itw,u32 N,u32 logR,u32 untw){
    constexpr bool L=(F&1)!=0;
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const u64 mu=ratios[prime];
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* st=stage_tw+prime*256u;const u64* sit=stage_itw+prime*256u;
    const u64 base=u64(prime*8+j)*N;
    {
        const uint4* a4=reinterpret_cast<const uint4*>(A+base+g*2048u);
        const uint4* b4=reinterpret_cast<const uint4*>(B+base+g*2048u);
        #pragma unroll
        for(u32 q=threadIdx.x;q<512;q+=256){
            uint4 va,vb;if(BPO_P2ABL==2){va=make_uint4(q,q+1,q+2,q+3);vb=make_uint4(q*3,q,j,g);}else if(untw&4){va=__ldcs(a4+q);vb=__ldcs(b4+q);}else{va=a4[q];vb=b4[q];}const u32 cc=q>>1,s=(q&1u)*4;u32* xa=sa+cc*9+s;u32* xb=sb+cc*9+s;
            xa[0]=va.x;xa[1]=va.y;xa[2]=va.z;xa[3]=va.w;xb[0]=vb.x;xb[1]=vb.y;xb[2]=vb.z;xb[3]=vb.w;
        }
    }
    __syncthreads();
    if((untw&8)&&threadIdx.x<128u)bpo_discard((threadIdx.x<64u?A:B)+base+g*2048u+(threadIdx.x&63u)*32u);   // tile is in smem
    u32 ab[2][8];u32 (&a)[8]=ab[0];u32 (&b)[8]=ab[1];
    #pragma unroll
    for(u32 m=0;m<8;++m){a[m]=sa[(lane+32*m)*9+warp];b[m]=sb[(lane+32*m)*9+warp];}
    __syncthreads();   // tile -> private regions
    u32* ra=sa+warp*288;u32* rb=sb+warp*288;
    if(F&16){   // ablation: no arithmetic (wrong values)
        #pragma unroll
        for(u32 m=0;m<8;++m)a[m]^=b[m];
    }else{
    if(L&&(F&2)){u32* const rr[2]={ra,rb};pxdif2_smr<true>(rr,ab,st,p,lane);}
    else if(L){pxdif_smr<true>(ra,a,st,p,lane);pxdif_smr<true>(rb,b,st,p,lane);}
    else{bpo_dif_smr(ra,a,st,p,lane);bpo_dif_smr(rb,b,st,p,lane);}
    if(untw&16){   // Montgomery pointwise (a*b*2^-32; host folds 2^32 into the P3 table t2n)
        const u32 pinv=pxinv32(p);
        #pragma unroll
        for(u32 m=0;m<8;++m)a[m]=pxmont(a[m],b[m],p,pinv);
    }else{
        #pragma unroll
        for(u32 m=0;m<8;++m)a[m]=mul(a[m],b[m],p,mu);
    }
    if(L&&(F&4))pzdit_smr<true>(ra,a,sit,p,lane);else if(L)pxdit_smr<true>(ra,a,sit,p,lane);else bpo_dit_smr(ra,a,sit,p,lane);
    }
    const u32 k1=__brev(g*8u+warp)>>(32-logR);
    if(untw&1)
    #pragma unroll
    for(u32 m=0;m<8;++m)a[m]=weighted_root_action(a[m],itw,2u*(lane+32*m)*k1,N,p);
    __syncthreads();   // private regions -> output tile
    #pragma unroll
    for(u32 m=0;m<8;++m)sa[(lane+32*m)*9+warp]=a[m];
    __syncthreads();
    {
        const u32 cc=threadIdx.x;const u32* s=sa+cc*9;
        uint4* o=reinterpret_cast<uint4*>(C+base+cc*256u+g*8u);
        if(BPO_P2ABL!=1||s[0]==0xffffffffu){o[0]=make_uint4(s[0],s[1],s[2],s[3]);o[1]=make_uint4(s[4],s[5],s[6],s[7]);}
    }
}
extern "C" __global__ void __launch_bounds__(256) bpo_middle_smr(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* ratios,const u64* untwist,const u64* stage_tw,const u64* stage_itw,u32 N,u32 logR,u32 untw){
    __shared__ __align__(16) u32 sa[256*9];
    __shared__ __align__(16) u32 sb[256*9];
    bpo_p2_body<BPO_P2F>(sa,sb,A,B,C,primes,ratios,untwist,stage_tw,stage_itw,N,logR,untw);
}
// P2 radix-16 (bpo_middle_r16, FHERMA_P2R16): the S = 256-point cyclic DFTs as 16 x 16 in registers.
// 128 threads = 8 slots s x 16 threads; thread (s, c2) owns x[16 c1 + c2], c1 = 0..15 (coalesced 128 B warp loads:
// warp w = c2 in 4w..4w+3, lane&7 = s). Forward per operand: DIF16 over c1 (twiddles w^(16e) from the r16 table),
// middle twiddle w^(c2 k1), one 16 x 16 smem transpose (slot stride 292 = 4 mod 32, row stride 17: conflict-free both ways),
// DIF16 over c2 -> thread t holds X[brev4(t) + 16 brev4(i)] in register i for both operands. Pointwise (Montgomery
// with untw&16), inverse = DIT16 (bit-reversed in) + w^-(c2 k1) + transpose + DIT16: thread (s, c2) gets 256 x[16 c1 + c2]
// (unnormalized, lazy words in [0,2p)) and stores C[c][g*8+s]. Same residues as bpo_middle_smr (P3 consumes any u32).
// r16: per prime 2 x 256 Shoup words, [0][e] = w^e, [1][e] = w^-e, w = psi^512 (the S root of the stage tables).
__device__ __forceinline__ void r16_dif16(u32 (&v)[16],const u64* t,u32 p){
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
// lazy DIT16, bit-reversed input -> natural output; CI: inputs canonical (first stage without input reductions)
template<bool CI>__device__ __forceinline__ void r16_dit16(u32 (&v)[16],const u64* t,u32 p){
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
__device__ __forceinline__ constexpr u32 r16_brev4(u32 i){return ((i&1u)<<3)|((i&2u)<<1)|((i&4u)>>1)|((i&8u)>>3);}
// Harvey-lazy radix-16 (H, p < 2^30): DIF words in [0,2p) (sum min(s,s-2p), difference Shoup(a+2p-b) unreduced);
// DIT words in [0,4p): a = min(x,x-2p), t = Shoup(y) lazy, x' = a+t, y' = a+2p-t (one correction per multiplied butterfly).
template<bool H>__device__ __forceinline__ void r16_dif16h(u32 (&v)[16],const u64* t,u32 p,u32 p2){
    if(!H){r16_dif16(v,t,p);return;}
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
template<bool H,bool CI>__device__ __forceinline__ void r16_dit16h(u32 (&v)[16],const u64* t,u32 p,u32 p2){
    if(!H){r16_dit16<CI>(v,t,p);return;}
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
#ifndef BPO_P2_OLD
// p2k (default; pv65 on-target whole batch 0.985, round-20 sweep line p2k): same arithmetic as the old body, addressing only:
// smem transposes as one base pointer + immediates (old: IADD3 + LEA per element, 16 live address registers), A/B/C rows as
// base + immediates (N = 65536 compile-time; the host runs this kernel only for n = 65536), middle twiddles as one IMAD.WIDE
// from an opaque table pointer (c2, brev4(i), k1 <= 15: index <= 225, the old "& 255" is a no-op), 2p from a lane-0 shuffle
// (ptxas cannot fold it into IMAD p*-2; replaces the 16-way bpo_p2_of constant lookup). JIT regs 80 -> 64 (6 -> 8 blocks/SM).
// FHERMA_P2_OLD=1 compiles the old body (BPO_P2_OLD).
template<class T>__device__ __forceinline__ T* bpo_opq(T* p){asm("mov.b64 %0, %0;":"+l"(p));return p;}
__device__ __forceinline__ u64 bpo_tw8(const char* tb,u32 s8,u32 k){return __ldg(reinterpret_cast<const u64*>(tb+u64(s8)*k));}
#ifdef BPO_P2CU   // p2cu (kernel bpo_middle_r16c): per prime 256 words, [0,128) = r16[0,128), [128,256) = r16[256,384) (the
// radix-16 stage twiddles tf / ti actually read: t[16 j], t[32 j], t[64]); 32 KB for K = 16; filled by the host (build_tabs)
__constant__ unsigned long long bpo_p2cu[BCRT_K*256];
template<bool H,bool MT=false,bool CU=false>
#elif defined(BPO_P2MT)
template<bool H,bool MT=false>
#else
template<bool H>
#endif
__device__ __forceinline__ void bpo_r16_body(u32* TA,const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32,u32 untw){
    constexpr u32 N=65536u;
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,s=lane&7u,c2=4u*warp+(lane>>3);
#ifdef BPO_P2MT   // p2mt (MT, kernel bpo_middle_r16m): r16 argument = per prime 1024 words: [0,512) the r16 table, [512 + 16 i + c2] = w^(c2 brev4(i)), [768 + 16 i + c2] = w^-(brev4(c2) i)
    const u64* tf=r16+u64(prime)*(MT?1024u:512u);const u64* ti=tf+256u;const u64* const tmf=tf+(512u+c2);const u64* const tmi=tf+(768u+c2);(void)tmf;(void)tmi;
#else
    const u64* tf=r16+u64(prime)*512u;const u64* ti=tf+256u;
#endif
    const char* tfb=reinterpret_cast<const char*>(bpo_opq(tf));const char* tib=reinterpret_cast<const char*>(bpo_opq(ti));
#ifdef BPO_P2CU   // CU: stage twiddles from the constant table (untw bits 24-31 = global index of the group's first prime)
    const u64* const cuf=CU?bpo_p2cu+((untw>>24)+prime)*256u:tf;const u64* const cui=CU?cuf+128u:ti;
#define BPO_P2TF cuf
#define BPO_P2TI cui
#else
#define BPO_P2TF tf
#define BPO_P2TI ti
#endif
#ifdef BPO_TWC   // ovl compute twin: tiles (prime, g mod 16) only (4 MB, L2-resident); stores predicated off (untw bit 30)
#define P2A_G (g&15u)
#define P2ST if(untw&0x40000000u)
    const u64 base=u64(prime*8)*N;
#else
#define P2A_G g
#define P2ST
    const u64 base=u64(prime*8+j)*N;
#endif
    const u32 off=P2A_G*2048u+c2*8u+s;
    u32 a[16],b[16];
    {   const u32* a0=A+base+off;const u32* b0=B+base+off;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1){if(untw&4){a[c1]=__ldcs(a0+128u*c1);b[c1]=__ldcs(b0+128u*c1);}else{a[c1]=a0[128u*c1];b[c1]=b0[128u*c1];}}
    }
    const u32 p2=H?__shfl_sync(0xffffffffu,2u*p,0):0u;(void)p2;
    r16_dif16h<H>(a,BPO_P2TF,p,p2);r16_dif16h<H>(b,BPO_P2TF,p,p2);
    u32* const tw_=TA+int(s*292u+c2);        // [17 i]: column c2 of row i
    u32* const tr_=TA+int(s*292u+17u*c2);    // [i]: row c2
    #pragma unroll
#ifdef BPO_P2MT
    for(u32 i=1;i<16;++i){const u64 w=MT?__ldg(tmf+16u*i):bpo_tw8(tfb,8u*c2,r16_brev4(i));a[i]=pxmshh<H>(a[i],w,p);b[i]=pxmshh<H>(b[i],w,p);}
#else
    for(u32 i=1;i<16;++i){const u64 w=bpo_tw8(tfb,8u*c2,r16_brev4(i));a[i]=pxmshh<H>(a[i],w,p);b[i]=pxmshh<H>(b[i],w,p);}
#endif
    #pragma unroll
    for(u32 i=0;i<16;++i)tw_[17*i]=a[i];
    __syncthreads();   // every thread's A/B loads have completed (values consumed above)
    if(untw&8)bpo_discard((threadIdx.x<64u?A:B)+base+P2A_G*2048u+(threadIdx.x&63u)*32u);   // dead A/B lines: no write-back (P3 / next P1 overwrite them)
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=tr_[i];   // thread row t = c2: k1 = brev4(c2)
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)tw_[17*i]=b[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)b[i]=tr_[i];
    r16_dif16h<H>(a,BPO_P2TF,p,p2);r16_dif16h<H>(b,BPO_P2TF,p,p2);
    {const u32 pinv=pxinv32(p);   // Montgomery pointwise only (host selects r16 with the 2^32-folded t2nM table)
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=pxmont(a[i],b[i],p,pinv);}
    r16_dit16h<H,true>(a,BPO_P2TI,p,p2);
    {const u32 k8=8u*r16_brev4(c2);
        #pragma unroll
#ifdef BPO_P2MT
        for(u32 i=1;i<16;++i)a[i]=pxmshh<H>(a[i],MT?__ldg(tmi+16u*i):bpo_tw8(tib,k8,i),p);
#else
        for(u32 i=1;i<16;++i)a[i]=pxmshh<H>(a[i],bpo_tw8(tib,k8,i),p);   // natural c2' = i
#endif
        a[0]=H?pxadd(a[0],0u,p2):pzred(a[0],p);}   // all canonical (H: < 2p) for the second DIT16 (CI)
    __syncthreads();   // forward reads of TA done
    #pragma unroll
    for(u32 i=0;i<16;++i)tr_[i]=a[i];   // row t = c2 (k1 = brev4(t)), column c2' = i
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=tw_[17*i];   // register i = row i -> k1 = brev4(i): bit-reversed input
    r16_dit16h<H,true>(a,BPO_P2TI,p,p2);   // H: C words in [0,4p) (P3 starts with a Shoup product: any u32)
    u32* o=C+base+(c2*256u+P2A_G*8u+s);
#undef P2A_G
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)P2ST o[4096u*c1]=a[c1];
#undef P2ST
}
#undef BPO_P2TF
#undef BPO_P2TI
#ifdef BPO_P2MT
// p2mt entry: bpo_middle_r16 with the r16x middle twiddles (host launches it only when r16x was built; else bpo_middle_r16)
#ifdef BPO_TWC
#define P2TWC_LB __launch_bounds__(128,9)   // ovl compute twin: R's register count (56, 9 blocks/SM)
#else
#define P2TWC_LB __launch_bounds__(128)
#endif
extern "C" __global__ void P2TWC_LB bpo_middle_r16m(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw){
    __shared__ u32 TA[8*292];
#if BPO_NH30
    if(primes[blockIdx.z]<(1u<<30))bpo_r16_body<true,true>(TA,A,B,C,primes,r16,N,untw);else
#endif
    bpo_r16_body<false,true>(TA,A,B,C,primes,r16,N,untw);
}
#ifdef BPO_P2CU
// p2cu entry: bpo_middle_r16m with the stage twiddles from bpo_p2cu (host launches it only when the table was filled; else r16m)
extern "C" __global__ void __launch_bounds__(128) bpo_middle_r16c(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw){
    __shared__ u32 TA[8*292];
#if BPO_NH30
    if(primes[blockIdx.z]<(1u<<30))bpo_r16_body<true,true,true>(TA,A,B,C,primes,r16,N,untw);else
#endif
    bpo_r16_body<false,true,true>(TA,A,B,C,primes,r16,N,untw);
}
#endif
// r16x init (once): per prime [0,512) = r16, [512+16i+c2] = r16[(c2 brev4(i))&255], [768+16i+c2] = r16[256+((brev4(c2) i)&255)]; grid K, 1024 threads
extern "C" __global__ void __launch_bounds__(1024) bpo_r16x_init(const u64* r16,u64* out){
    const u32 r=blockIdx.x,x=threadIdx.x;const u64* h=r16+u64(r)*512u;u64* o=out+u64(r)*1024u;
    if(x<512u)o[x]=h[x];
    else if(x<768u){const u32 i=(x-512u)>>4,c2=x&15u;o[x]=h[(c2*r16_brev4(i))&255u];}
    else{const u32 i=(x-768u)>>4,c2=x&15u;o[x]=h[256u+((r16_brev4(c2)*i)&255u)];}
}
#endif
#else
template<bool H>__device__ __forceinline__ void bpo_r16_body(u32* TA,const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw){
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,s=lane&7u,c2=4u*warp+(lane>>3);
    const u64* tf=r16+u64(prime)*512u;const u64* ti=tf+256u;
    const u64 base=u64(prime*8+j)*N;
    u32 a[16],b[16];
    {   const u32* a0=A+base+g*2048u+c2*8u+s;const u32* b0=B+base+g*2048u+c2*8u+s;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1){if(untw&4){a[c1]=__ldcs(a0+128u*c1);b[c1]=__ldcs(b0+128u*c1);}else{a[c1]=a0[128u*c1];b[c1]=b0[128u*c1];}}
    }
    const u32 p2=H?bpo_p2_of(p):0u;(void)p2;
    r16_dif16h<H>(a,tf,p,p2);r16_dif16h<H>(b,tf,p,p2);
    u32* const ta=TA+s*292u;
    #pragma unroll
    for(u32 i=1;i<16;++i){const u64 w=__ldg(tf+((c2*r16_brev4(i))&255u));a[i]=pxmshh<H>(a[i],w,p);b[i]=pxmshh<H>(b[i],w,p);}
    #pragma unroll
    for(u32 i=0;i<16;++i)ta[17*i+c2]=a[i];
    __syncthreads();   // every thread's A/B loads have completed (values consumed above)
    if(untw&8)bpo_discard((threadIdx.x<64u?A:B)+base+g*2048u+(threadIdx.x&63u)*32u);   // dead A/B lines: no write-back (P3 / next P1 overwrite them)
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=ta[17*c2+i];   // thread row t = c2: k1 = brev4(c2)
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)ta[17*i+c2]=b[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)b[i]=ta[17*c2+i];
    r16_dif16h<H>(a,tf,p,p2);r16_dif16h<H>(b,tf,p,p2);
    {const u32 pinv=pxinv32(p);   // Montgomery pointwise only (host selects r16 with the 2^32-folded t2nM table)
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=pxmont(a[i],b[i],p,pinv);}
    r16_dit16h<H,true>(a,ti,p,p2);
    {const u32 k1=r16_brev4(c2);
        #pragma unroll
        for(u32 i=1;i<16;++i)a[i]=pxmshh<H>(a[i],__ldg(ti+((i*k1)&255u)),p);   // natural c2' = i
        a[0]=H?pxadd(a[0],0u,p2):pzred(a[0],p);}   // all canonical (H: < 2p) for the second DIT16 (CI)
    __syncthreads();   // forward reads of TA done
    #pragma unroll
    for(u32 i=0;i<16;++i)ta[17*c2+i]=a[i];   // row t = c2 (k1 = brev4(t)), column c2' = i
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=ta[17*i+c2];   // register i = row i -> k1 = brev4(i): bit-reversed input
    r16_dit16h<H,true>(a,ti,p,p2);   // H: C words in [0,4p) (P3 starts with a Shoup product: any u32)
    u32* o=C+base+c2*256u+g*8u+s;
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)o[4096u*c1]=a[c1];
}
#endif
extern "C" __global__ void __launch_bounds__(128) bpo_middle_r16(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw){
    __shared__ u32 TA[8*292];   // one transpose buffer (A, then B, then the inverse): 9.3 KB, occupancy reg-limited
#if BPO_NH30   // h30 prime set (default): Harvey-lazy body for the primes < 2^30
    if(primes[blockIdx.z]<(1u<<30))bpo_r16_body<true>(TA,A,B,C,primes,r16,N,untw);else
#endif
    bpo_r16_body<false>(TA,A,B,C,primes,r16,N,untw);
}
// P3 radix-16 (bpo_last_r16, FHERMA_P3R16): same values as bpo_last_cm3 (BPO_P3T 260). Block (column c, prime), 128 threads.
// Phase J: thread t owns slots t and t+128: 8 digit words each (coalesced), F (cross untwiddle * t1) Shoup product,
// lazy 8-point J-DIT, tile X[d][sl + sl/16] (digit stride 272 = 16 mod 32). Phase R: thread (d = tid/16, a = tid%16)
// takes slots 16a+b (b = 0..15; slot holds frequency brev8, so k1 = brev4(a), k2 = brev4(b): bit-reversed DIT input),
// DIT16 over k2, w^-(h2 k1), transpose T[d][a][h2] (row stride 17), DIT16 over k1 -> natural h1: h = 16 h1 + h2.
// Output D[(prime*8+d)*N + c*256 + h] = Shoup(v, t2n[...][(c>>7)*256 + h]) canonical. R and J roots = r16 inverse table.
extern "C" __global__ void __launch_bounds__(128) bpo_last_r16(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* r16,const u64* t2n,u32 N,u32 logR){
    __shared__ u32 X[8*272];u32* const T=X;   // 8.7 KB: the transpose reuses the tile (one more barrier)
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 tid=threadIdx.x;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* ti=r16+u64(prime)*512u+256u;
    {   const u32* src=C+u64(prime*8)*N+c*256u+tid;
        u32 y[2][8];
        #pragma unroll
        for(u32 hf=0;hf<2;++hf)
            #pragma unroll
            for(u32 d=0;d<8;++d)y[hf][d]=src[u64(d)*N+128u*hf];
        #pragma unroll
        for(u32 hf=0;hf<2;++hf){
            const u32 t=tid+128u*hf;
            u64 F;{const u32 k1=__brev(t)>>(32-logR);
                const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
                const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
                const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
                F=u64(wv)|(u64(q)<<32);}
            u32 (&yv)[8]=y[hf];
            #pragma unroll
            for(u32 d=0;d<8;++d)yv[d]=pxmsh(yv[d],F,p);
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=2)pzdit1c(yv[d0],yv[d0+1],p);
            #pragma unroll
            for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                #pragma unroll
                for(u32 d0=0;d0<8;d0+=len){
                    #pragma unroll
                    for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(yv[d0],yv[d0+half],0,p);else pzdit(yv[d0+jj],yv[d0+jj+half],ti[jj*(256u/len)],p);}}}
            #pragma unroll
            for(u32 d=0;d<8;++d)X[d*272u+t+(t>>4)]=yv[d];
        }
    }
    __syncthreads();
    const u32 d=tid>>4,a=tid&15u;
    u32 v[16];
    #pragma unroll
    for(u32 b=0;b<16;++b)v[b]=X[d*272u+17u*a+b];
    r16_dit16<false>(v,ti,p);
    {const u32 k1=r16_brev4(a);
        #pragma unroll
        for(u32 i=1;i<16;++i)v[i]=pxmsh(v[i],__ldg(ti+((i*k1)&255u)),p);
        v[0]=pzred(v[0],p);}
    __syncthreads();   // all tile reads done before the transpose overwrites it
    #pragma unroll
    for(u32 i=0;i<16;++i)T[d*272u+17u*a+i]=v[i];
    __syncthreads();
    const u32 h2=a;
    #pragma unroll
    for(u32 i=0;i<16;++i)v[i]=T[d*272u+17u*i+h2];
    r16_dit16<true>(v,ti,p);
    const u64* tn=t2n+u64(prime*8+d)*512u+(c>>7)*256u+h2;
    u32* out=D+u64(prime*8+d)*N+c*256u+h2;
    #pragma unroll
    for(u32 h1=0;h1<16;++h1)out[16u*h1]=pxmsh(v[h1],tn[16u*h1],p);
}
// ---- Fused inverse tail (FHERMA_P3CRT_X=1): P2 bpo_middle_r16x + bpo_p3crt_x replace P2 + P3 + CRT ----
// Inverse R-transform of 256 = 32 (k_lo) x 8 (k_hi), k = k_lo + 32 k_hi; slot 8g+b holds k = 32 brev3(b) + brev5(g).
// P2x (block = 4 slot octets g = 4G..4G+3, i.e. 4 k_lo values, all 8 k_hi) does, per octet, the r16 S transforms, then
// F (P3's cross untwiddle ipsi^(2 c k + (c & 127)) as two Shoup factors TH[k_hi][c] TL[k_lo][c]), the 8-point DIT over
// k_hi (-> h2 < 8) and the twiddle w^-(h2 k_lo); out C'[ch][G][c][h2][q] (g = 4G + q; block = 512 threads, one
// thread group per octet, 32 KB contiguous out).
// Status: exact but slower locally (sequential profile P2x 26.3 vs P2 14.2 + P3 12.8 ms; P3CRT 38.9 vs CRT 9.0 ms): the
// smem residue tile (16.5 KB per 32 coefficients) caps the CRT at 4 warps/SM vs 12 from L2. Opt-in only.
// P3CRT (block = (c, h2), 128 threads = 128 channels): 32-point DIT over k_lo (g order = bit-reversed k_lo) -> h1,
// J-DIT + t2n (as bpo_last_cm3), canonical residues of the 32 coefficients i = (h2 + 8 h1) 256 + c in smem, then the
// tensor CRT of bpo_crt_tc_cm (warp 0 of 2) reading them from smem: no D round trip, no separate CRT launch.
// fxt: per prime [TH 8 x 256][TL 32 x 256] Shoup words.
extern "C" __global__ void __launch_bounds__(512) bpo_middle_r16x(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,const u64* fxt,u32 N,u32 untw){
    __shared__ __align__(16) u32 TA[4*8*292];   // one r16 tile per octet q (thread group tid/128)
    const u32 G=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 tid=threadIdx.x,q=tid>>7,lane=tid&31u,warp=(tid>>5)&3u,s=lane&7u,c2=4u*warp+(lane>>3);
    const u64* tf=r16+u64(prime)*512u;const u64* ti=tf+256u;
    const u64* TH=fxt+u64(prime)*(40u*256u);const u64* TL=TH+8u*256u;
    const u64 base=u64(prime*8+j)*N;
    u32* const TQ=TA+q*(8u*292u);u32* const ta=TQ+s*292u;
    const u32 pinv=pxinv32(p);
    {
        const u32 g=4u*G+q,klo=r16_brev4(g&15u)*2u+(g>>4);   // brev5(g)
        u32 a[16],b[16];
        {   const u32* a0=A+base+g*2048u+c2*8u+s;const u32* b0=B+base+g*2048u+c2*8u+s;
            #pragma unroll
            for(u32 c1=0;c1<16;++c1){if(untw&4){a[c1]=__ldcs(a0+128u*c1);b[c1]=__ldcs(b0+128u*c1);}else{a[c1]=a0[128u*c1];b[c1]=b0[128u*c1];}}
        }
        r16_dif16(a,tf,p);r16_dif16(b,tf,p);
        #pragma unroll
        for(u32 i=1;i<16;++i){const u64 w=__ldg(tf+((c2*r16_brev4(i))&255u));a[i]=pxmsh(a[i],w,p);b[i]=pxmsh(b[i],w,p);}
        #pragma unroll
        for(u32 i=0;i<16;++i)ta[17*i+c2]=a[i];
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=ta[17*c2+i];
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)ta[17*i+c2]=b[i];
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)b[i]=ta[17*c2+i];
        r16_dif16(a,tf,p);r16_dif16(b,tf,p);
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=pxmont(a[i],b[i],p,pinv);
        r16_dit16<true>(a,ti,p);
        {const u32 k1=r16_brev4(c2);
            #pragma unroll
            for(u32 i=1;i<16;++i)a[i]=pxmsh(a[i],__ldg(ti+((i*k1)&255u)),p);
            a[0]=pzred(a[0],p);}
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)ta[17*c2+i]=a[i];
        __syncthreads();
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=ta[17*i+c2];
        r16_dit16<true>(a,ti,p);   // a[c1] = 256 x_s[16 c1 + c2] in [0,2p)
        {   const u32 khi=((s&1u)<<2)|(s&2u)|((s&4u)>>2);   // brev3(s)
            const u64* th=TH+khi*256u+c2;const u64* tl=TL+klo*256u+c2;
            #pragma unroll
            for(u32 c1=0;c1<16;++c1)a[c1]=pxmsh(pxmsh(a[c1],__ldg(th+16u*c1),p),__ldg(tl+16u*c1),p);   // canonical
        }
        __syncthreads();   // transpose reads done
        #pragma unroll
        for(u32 c1=0;c1<16;++c1)TQ[(16u*c1+c2)*8u+s]=a[c1];
    }
    __syncthreads();
    {   // 8-point DIT over k_hi (slot b, k_hi = brev3(b): bit-reversed input) -> natural h2, then w^-(h2 k_lo); in place
        const u32 c=tid&255u;
        #pragma unroll
        for(u32 qq=0;qq<2;++qq){
            const u32 qo=2u*(tid>>8)+qq,g=4u*G+qo,klo=r16_brev4(g&15u)*2u+(g>>4);
            u32* const cell=TA+qo*(8u*292u)+c*8u;u32 v[8];
            {const uint4 x0=*reinterpret_cast<const uint4*>(cell),x1=*reinterpret_cast<const uint4*>(cell+4);
             v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
            #pragma unroll
            for(u32 h=0;h<8;h+=2)pzdit1c(v[h],v[h+1],p);
            #pragma unroll
            for(u32 h=0;h<8;h+=4){pzdit1(v[h],v[h+2],0,p);pzdit(v[h+1],v[h+3],ti[64],p);}
            pzdit1(v[0],v[4],0,p);
            #pragma unroll
            for(u32 jj=1;jj<4;++jj)pzdit(v[jj],v[jj+4],ti[32u*jj],p);
            #pragma unroll
            for(u32 h2=1;h2<8;++h2)v[h2]=pxmsh(v[h2],__ldg(ti+h2*klo),p);
            *reinterpret_cast<uint4*>(cell)=make_uint4(v[0],v[1],v[2],v[3]);*reinterpret_cast<uint4*>(cell+4)=make_uint4(v[4],v[5],v[6],v[7]);
        }
    }
    __syncthreads();
    {   // out C'[ch][G][c][h2][q]: the block's 32 KB are contiguous
        #pragma unroll
        for(u32 i=0;i<4;++i){const u32 x=tid+512u*i;   // = c*8 + h2
            *reinterpret_cast<uint4*>(C+base+G*8192u+4u*x)=make_uint4(TA[x],TA[8u*292u+x],TA[16u*292u+x],TA[24u*292u+x]);}
    }
}
// fxt init (once): per prime row r < 8: TH[r][c] = ipsi^(64 c r); row 8 + kl: TL[kl][c] = ipsi^(2 c kl + (c & 127)) (Shoup words)
extern "C" __global__ void __launch_bounds__(256) bpo_fxt_init(const u64* untwist,const u32* primes,u64* fxt,u32 N){
    const u32 row=blockIdx.x,prime=blockIdx.y,c=threadIdx.x,p=primes[prime];
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u32 wv=row<8u?u32(bm_root(itw,64u*c*row,N,p)):mul_shoup(u32(bm_root(itw,2u*c*(row-8u),N,p)),itw[1+(c&127u)],p);
    const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
    const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
    fxt[(u64(prime)*40u+row)*256u+c]=u64(wv)|(u64(q)<<32);
}
#ifdef BTC_F
#define FX_RS 33u
extern "C" __global__ void __launch_bounds__(64) bpo_p3crt_x(const u32* __restrict__ Cx,u32* __restrict__ output,
    const u32* __restrict__ bfrag,const u64* r16,const u64* t2n,u32 N,u32 d){
    __shared__ __align__(16) u32 R[128*FX_RS];
    __shared__ __align__(16) u32 stage_all[32*36];   // IMMA staging; the output staging reuses R
    const u32 cc=blockIdx.y,h2=blockIdx.x,tid=threadIdx.x,warp=tid>>5;
    const u32* const bsh=bfrag;   // B fragments via L1
    {   // phase D: thread = channel ch (prime ch/8, digit ch%8): 32-point DIT over k_lo -> h1
      #pragma unroll 1
      for(u32 ch=tid;ch<128u;ch+=64u){
        const u32 pr=ch>>3,p=bpr_pr[pr];const u64* ti=r16+u64(pr)*512u+256u;
        u32 v[32];
        const uint4* src=reinterpret_cast<const uint4*>(Cx+u64(ch)*N+cc*32u+h2*4u);
        #pragma unroll
        for(u32 x=0;x<8;++x){const uint4 t=src[x*2048u];v[4*x]=t.x;v[4*x+1]=t.y;v[4*x+2]=t.z;v[4*x+3]=t.w;}
        #pragma unroll
        for(u32 len=2;len<=32;len<<=1){const u32 half=len/2;
            #pragma unroll
            for(u32 b0=0;b0<32;b0+=len){
                #pragma unroll
                for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(v[b0],v[b0+half],0,p);else pzdit(v[b0+jj],v[b0+jj+half],ti[jj*(256u/len)],p);}}}
        #pragma unroll
        for(u32 h1=0;h1<32;++h1)R[ch*FX_RS+h1]=v[h1];
      }
    }
    __syncthreads();
    {   // phase J: thread (h1 = lane, primes 4 warp..4 warp+3): lazy J-DIT over the 8 digits, t2n -> canonical residues
        const u32 h1=tid&31u,h=h2+8u*h1;
        #pragma unroll 1
        for(u32 pq=0;pq<8;++pq){const u32 pr=8u*warp+pq,p=bpr_pr[pr];const u64* ti=r16+u64(pr)*512u+256u;
            u32 yv[8];
            #pragma unroll
            for(u32 jd=0;jd<8;++jd)yv[jd]=R[(pr*8u+jd)*FX_RS+h1];
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=2)pzdit1(yv[d0],yv[d0+1],0,p);
            #pragma unroll
            for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                #pragma unroll
                for(u32 d0=0;d0<8;d0+=len){
                    #pragma unroll
                    for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(yv[d0],yv[d0+half],0,p);else pzdit(yv[d0+jj],yv[d0+jj+half],ti[jj*(256u/len)],p);}}}
            #pragma unroll
            for(u32 jd=0;jd<8;++jd)R[(pr*8u+jd)*FX_RS+h1]=pxmsh(yv[jd],__ldg(t2n+u64(pr*8u+jd)*512u+(cc>>7)*256u+h),p);
        }
    }
    __syncthreads();
    if(warp!=0)return;
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F;
    static_assert(K==16&&SW==16,"tensor CRT geometry");
    constexpr u32 POS[8]={BCRT_POS};
    constexpr u32 QW[L]={BCRT_Q},BW[L]={BCRT_BIAS};
    constexpr u32 NPW[SW]={BTC_NP},OFW[SW]={BTC_OFF};
    const u32 lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    u32* st=stage_all;
    u32 acc[NW];
    #pragma unroll
    for(u32 w=0;w<NW;++w)acc[w]=0;
    u32 pend=0;
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 a[2][2][4];
        #pragma unroll
        for(u32 m=0;m<2;++m)
            #pragma unroll
            for(u32 s=0;s<2;++s){
                const u32 r=8*s+tig;const u32* base=R+(r*8u+j)*FX_RS+16*m+gid;
                a[m][s][0]=base[0];a[m][s][1]=base[8];
                a[m][s][2]=base[4u*8u*FX_RS];a[m][s][3]=base[4u*8u*FX_RS+8];
            }
        #pragma unroll
        for(u32 T=0;T<NT;++T){
            const uint2 b0=*reinterpret_cast<const uint2*>(bsh+((T*2+0)*32+lane)*2);
            const uint2 b1=*reinterpret_cast<const uint2*>(bsh+((T*2+1)*32+lane)*2);
            #pragma unroll
            for(u32 m=0;m<2;++m){
                u32 c[4]={0,0,0,0};
                imma16832(c,a[m][0][0],a[m][0][1],a[m][0][2],a[m][0][3],b0.x,b0.y);
                imma16832(c,a[m][1][0],a[m][1][1],a[m][1][2],a[m][1][3],b1.x,b1.y);
                st[(16*m+gid)*SST+4*T+tig]=c[0]+(c[1]<<8);
                st[(16*m+gid+8)*SST+4*T+tig]=c[2]+(c[3]<<8);
            }
        }
        __syncwarp();
        u32 h[SST];
        #pragma unroll
        for(u32 q=0;q<SST/4;++q){const uint4 v=*reinterpret_cast<const uint4*>(st+lane*SST+4*q);h[4*q]=v.x;h[4*q+1]=v.y;h[4*q+2]=v.z;h[4*q+3]=v.w;}
        __syncwarp();
        const u64 S=u64(h[32])+(u64(h[33])<<16)+(u64(h[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        const u32 o=POS[j]/32,sh=POS[j]%32;
        constexpr u32 dummy=0;(void)dummy;
        const u32 inj=j?(POS[j-1]/32+SW+1-o):0xffffffffu;
        u64 zc=0,c=0;u32 prevZ=0;
        #pragma unroll
        for(u32 w=0;w<=SW;++w){
            u32 z=0;
            if(w<SW){const u64 v=u64(h[2*w])+(u64(h[2*w+1])<<16)+u64(alpha)*NPW[w]+OFW[w]+zc;z=u32(v);zc=v>>32;}
            const u32 part=sh?((z<<sh)|(prevZ>>(32-sh))):z;
            prevZ=z;
            if(o+w<NW){const u64 s=u64(acc[o+w])+part+c+(w==inj?pend:0u);acc[o+w]=u32(s);c=s>>32;}
        }
        pend=u32(c);
    }
    {   // final pending carry enters word POS[7]/32+SW+1
        constexpr u32 start=POS[7]/32+SW+1;u64 c=pend;
        #pragma unroll
        for(u32 w=0;w<NW;++w)if(w>=start){const u64 s=u64(acc[w])+c;acc[w]=u32(s);c=s>>32;}
    }
    constexpr u32 WO=W/32,WS=W%32;
    #pragma unroll
    for(u32 f=0;f<2;++f){
        u64 c=0;
        #pragma unroll
        for(u32 w=0;w<NW;++w){
            const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;
            const u32 hw=WS?((a_>>WS)|(b_<<(32-WS))):a_;
            const u32 low=w<WO?acc[w]:(w==WO&&WS?(acc[w]&((1u<<WS)-1)):0);
            const u64 s=u64(low)+u64(hw)*d+c;acc[w]=u32(s);c=s>>32;
        }
    }
    u32 borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;borrow=u32(s>>63);}
    if(!borrow){
        borrow=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    }
    borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-BW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    if(borrow){
        u64 c=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])+QW[w]+c;acc[w]=u32(s);c=s>>32;}
    }
    st=R;   // all residue reads of this (only) CRT warp are done
    #pragma unroll
    for(u32 w=0;w<L;++w)st[lane*L+w]=acc[w];
    __syncwarp();
    for(u32 x=lane;x<32*L;x+=32){const u32 rr=x/L,w=x-rr*L;__stcs(output+u64((h2+8u*rr)*256u+cc)*L+w,st[x]);}
}
#endif
#if !defined(BPO_LEAN)
extern "C" __global__ void __launch_bounds__(256) bpo_middle_smrv(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* ratios,const u64* untwist,const u64* stage_tw,const u64* stage_itw,u32 N,u32 logR,u32 untw){
    __shared__ __align__(16) u32 sa[256*9];
    __shared__ __align__(16) u32 sb[256*9];
    bpo_p2_body<BPO_P2FV>(sa,sb,A,B,C,primes,ratios,untwist,stage_tw,stage_itw,N,logR,untw);
}
#endif
#ifndef BPO_P3ABL
#define BPO_P3ABL 0
#endif
// Bulk-copy (TMA) helpers, compute_90+ only (no-ops otherwise; callers gate on BPO_HAS_BULK).
#if defined(__CUDA_ARCH__)&&__CUDA_ARCH__>=900
#define BPO_HAS_BULK 1
#else
#define BPO_HAS_BULK 0
#endif
__device__ __forceinline__ u32 bpo_sa(const void* s){return u32(__cvta_generic_to_shared(s));}
__device__ __forceinline__ void bpo_bulk_store(void* g,const void* s,u32 bytes){
#if BPO_HAS_BULK
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0],[%1],%2;"::"l"(g),"r"(bpo_sa(s)),"r"(bytes):"memory");
#endif
}
__device__ __forceinline__ void bpo_bulk_commit(){
#if BPO_HAS_BULK
    asm volatile("cp.async.bulk.commit_group;":::"memory");
#endif
}
__device__ __forceinline__ void bpo_bulk_wait_read(){
#if BPO_HAS_BULK
    asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
#endif
}
__device__ __forceinline__ void bpo_fence_async(){
#if BPO_HAS_BULK
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
}
__device__ __forceinline__ void bpo_bulk_load(void* s,const void* g,u32 bytes,u64* bar){
#if BPO_HAS_BULK
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(bpo_sa(s)),"l"(g),"r"(bytes),"r"(bpo_sa(bar)):"memory");
#endif
}
__device__ __forceinline__ void bpo_bar_init(u64* bar,u32 count){
#if BPO_HAS_BULK
    asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(bpo_sa(bar)),"r"(count):"memory");
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
#endif
}
__device__ __forceinline__ void bpo_bar_expect(u64* bar,u32 bytes){
#if BPO_HAS_BULK
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bpo_sa(bar)),"r"(bytes):"memory");
#endif
}
__device__ __forceinline__ void bpo_bar_wait(u64* bar,u32 parity){
#if BPO_HAS_BULK
    asm volatile("{\n.reg .pred P;\nBPO_W:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra BPO_W;\n}"::"r"(bpo_sa(bar)),"r"(parity):"memory");
#endif
}
// P3 flags T (template; bpo_last_cm3 uses BPO_P3T, A/B twin bpo_last_cm3v uses BPO_P3TV):
//   1 TMA bulk store of each warp's 1 KB output row, 2 TMA bulk loads of the 8 C rows (both compute_90+ only),
//   4 lean arithmetic (J-DIT psi^0 butterflies and R-DIT psi^0 stages as add/sub),
//   256 (with 4) lazy J-DIT + R-DIT: words in [0,2p) between stages, the t2n Shoup product canonicalizes
//   1024 / 2048 (exit-time sweep; = stall 2fe0785 BPO_P3TN 1 / 2): t2n words loaded ahead of the D stores / before the R-DIT (2048 default)
#ifndef BPO_P3T
#define BPO_P3T 2308   // 4 lean ~neutral A/B (63.22 vs 63.64); 260 = 4 + lazy J-DIT/R-DIT (P2F 5 lazy S-DIT feeds it); +2048 t2n hoist (sweep a26c406 "tn2": on target 0.886x P3, exact)
#endif
#ifndef BPO_P3TV
#define BPO_P3TV BPO_P3T
#endif
// P3 with the shuffle-free DIT (skewed exchange tile, layout-3 entry); same values as bpo_last_cm2.
template<u32 T0,bool H=false>__device__ __forceinline__ void bpo_p3_body(u32* X,u64* bar,const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    constexpr u32 T=BPO_HAS_BULK?T0:0u;   // T0&16: ablation, no arithmetic (wrong values)
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 t=threadIdx.x,lane=t&31u,warp=t>>5;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* sit=stage_itw+prime*256u;
    const u32* src=C+u64(prime*8)*N+c*256u+t;
    u32 yv[8];
    if(T&2){
        if(t==0){bpo_bar_init(bar,1);bpo_bar_expect(bar,8192u);
            #pragma unroll
            for(u32 d=0;d<8;++d)bpo_bulk_load(X+256*d,C+u64(prime*8+d)*N+c*256u,1024u,bar);}
        __syncthreads();   // barrier initialized before anyone waits on it
    }
    u64 F;
    if(T0&32)F=itw[1+(t&127u)];else   // ablation 32: F from a plain table entry
    {
        const u32 k1=__brev(t)>>(32-logR);
        const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
        const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
        const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
        F=u64(wv)|(u64(q)<<32);
    }
    if(T&2){
        bpo_bar_wait(bar,0);
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=X[256*d+t];
        __syncthreads();
    }else if(BPO_P3ABL==3){
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=(t*8u+d+c)&0x3fffffffu;
    }else{
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=src[u64(d)*N];
    }
    if((BPO_P3ABL!=1&&!(T0&16))&&(T0&4)){
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=pxmshh<H&&(T0&256)!=0>(yv[d],F,p);   // H: [0,2p)
        if(H&&(T0&256)){   // Harvey J-DIT: len 2 on inputs < 2p, words in [0,4p)
            const u32 p2=bpo_p2_of(p);
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=2)phdit1c(yv[d0],yv[d0+1],p2);
            #pragma unroll
            for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                #pragma unroll
                for(u32 d0=0;d0<8;d0+=len){
                    #pragma unroll
                    for(u32 jj=0;jj<half;++jj){if(jj==0)phdit1(yv[d0],yv[d0+half],p2);else phdit(yv[d0+jj],yv[d0+jj+half],sit[jj*(256u/len)],p,p2);}}}
        }else if(T0&256){   // lazy J-DIT: len 2 on canonical inputs (no corrections), outputs in [0,2p)
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=2)pzdit1c(yv[d0],yv[d0+1],p);
            #pragma unroll
            for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                #pragma unroll
                for(u32 d0=0;d0<8;d0+=len){
                    #pragma unroll
                    for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(yv[d0],yv[d0+half],sit[0],p);else pzdit(yv[d0+jj],yv[d0+jj+half],sit[jj*(256u/len)],p);}}}
        }else if(!(T0&64))   // ablation 64: no J-DIT
        #pragma unroll
        for(u32 len=2;len<=8;len<<=1){const u32 half=len/2;
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=len){
                #pragma unroll
                for(u32 jj=0;jj<half;++jj){if(jj==0)pxdit1<true>(yv[d0],yv[d0+half],sit[0],p);else pxdit<true>(yv[d0+jj],yv[d0+jj+half],sit[jj*(256u/len)],p);}}}
    }else if((BPO_P3ABL!=1&&!(T0&16))){
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=mul_shoup(yv[d],F,p);
        BPO_JDIT(yv,sit)
    }
    #pragma unroll
    for(u32 d=0;d<8;++d)X[d*288+t+4*(t>>5)]=yv[d];
    __syncthreads();
    constexpr u32 TN=(T0&1024)?1u:(T0&2048)?2u:0u;u64 tv[8];
    if(TN==2){const u64* tn2=(dfl&2)?t2n+u64(prime*8+warp)*512u+(c>>7)+2u*lane:t2n+u64(prime*8+warp)*512u+(c>>7)*256u+lane;const u32 ts2=(dfl&2)?64u:32u;
        #pragma unroll
        for(u32 m=0;m<8;++m)tv[m]=__ldg(tn2+ts2*m);}
    if((dfl&1)&&lane<8u)bpo_discard(C+u64(prime*8+lane)*N+c*256u+32u*warp);   // C lines read by this warp (all used)
    u32 v[8];
    u32* base=X+warp*288;
    {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    if((BPO_P3ABL!=1&&!(T0&16))){if(T0&128){}else if(H&&(T0&4)&&(T0&256))phdit_smr<false>(base,v,sit,p,bpo_p2_of(p),lane);else if((T0&4)&&(T0&256))pzdit_smr<false>(base,v,sit,p,lane);else if(T0&4)pxdit_smr<true>(base,v,sit,p,lane);else bpo_dit_smr(base,v,sit,p,lane);}
    const u64* tn=(dfl&2)?t2n+u64(prime*8+warp)*512u+(c>>7)+2u*lane:t2n+u64(prime*8+warp)*512u+(c>>7)*256u+lane;const u32 ts=(dfl&2)?64u:32u;   // dfl 2: old interleaved layout (A/B only)
    if(TN==1){
        #pragma unroll
        for(u32 m=0;m<8;++m)tv[m]=__ldg(tn+ts*m);}
    u32* out=D+u64(prime*8+warp)*N+c*256u+lane;
    if(T&1){
        __syncwarp();
        #pragma unroll
        for(u32 m=0;m<8;++m)base[lane+32*m]=mul_shoup(v[m],(TN?tv[m]:tn[ts*m]),p);
        bpo_fence_async();
        __syncwarp();
        if(lane==0){bpo_bulk_store(out,base,1024u);bpo_bulk_commit();bpo_bulk_wait_read();}
    }else if(BPO_P3ABL==1||(T0&16)){
        #pragma unroll
        for(u32 m=0;m<8;++m)out[32*m]=v[m];
    }else if(BPO_P3ABL==2){
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=mul_shoup(v[m],(TN?tv[m]:tn[ts*m]),p);if(r==0xffffffffu)out[32*m]=r;}
    }else if(T0&8){   // residues stay in L2 for the CRT (evict_last policy)
        u64 pol;asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=(T0&4)?pxmsh(v[m],(TN?tv[m]:tn[ts*m]),p):mul_shoup(v[m],(TN?tv[m]:tn[ts*m]),p);
            asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+32*m),"r"(r),"l"(pol):"memory");}
#ifdef BPO_P3EL
    }else if((T0&4)&&(dfl&8)){   // dfl 8 (BIG_GFLAGS bit 128, default on; group-0 P3 of a prime-grouped product only): residues L2 evict_last, the CRT reads them from L2
        u64 pol;asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=pxmsh(v[m],(TN?tv[m]:tn[ts*m]),p);
            asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+32*m),"r"(r),"l"(pol):"memory");}
#endif
    }else if((T0&4)&&(dfl&4)){   // dfl 4 (runtime, prime-group P1): residues L2 evict_first (the CRT reads them only after all groups)
        u64 pol;asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=pxmsh(v[m],(TN?tv[m]:tn[ts*m]),p);
            asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+32*m),"r"(r),"l"(pol):"memory");}
    }else if(T0&4){
        #pragma unroll
        for(u32 m=0;m<8;++m)out[32*m]=pxmsh(v[m],(TN?tv[m]:tn[ts*m]),p);
    }else{
        #pragma unroll
        for(u32 m=0;m<8;++m)out[32*m]=mul_shoup(v[m],(TN?tv[m]:tn[ts*m]),p);
    }
}
#if !defined(BPO_P3_OLD)&&BPO_P3T==2308&&BPO_P3ABL==0
#define BPO_P3B 1
// p3b (default; pv67 on-target whole batch p3b 0.9941 / repeat 0.9934, round-21 sweep line cl_p3b): the BPO_P3T 2308 values of
// bpo_p3_body (lean, lazy / Harvey J-DIT + R-DIT, t2n hoisted before the R-DIT, BPO_P3EL group-0 evict_last) with p2k
// addressing: N = 65536, logR = 8 compile-time, C rows / X / t2n / D as base + immediates, 2p from a lane-0 shuffle (no 16-way
// bpo_p2_of lookup), one t2n layout; kernel __maxnreg__(32) (40 -> 32 regs, 0 spill; P3 blocks pack better next to P1 / P2
// blocks). sm_90 SASS 1423 -> 1198 instr. The kernel falls back to bpo_p3_body when N != 65536, logR != 8 or dfl 2 (old t2n
// layout). FHERMA_P3_OLD=1 compiles the old kernel (BPO_P3_OLD).
#ifdef BPO_P3F
template<bool H,bool PF=false>
#else
template<bool H>
#endif
__device__ __forceinline__ void bpo_p3b_body(u32* X,const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 dfl){
    constexpr u32 N=65536u,logR=8u;
    const u32 c=blockIdx.x,prime=blockIdx.y,p=primes[prime];
    const u32 t=threadIdx.x,lane=t&31u,warp=t>>5;
#ifdef BPO_TWC   // ovl compute twin: tiles (prime, c mod 16) only (2.5 MB, L2-resident); stores predicated off (dfl bit 30)
#define P3A_C (c&15u)
#define P3A_P prime
#define P3ST if(dfl&0x40000000u)
#else
#define P3ST
#define P3A_C c
#define P3A_P prime
#endif
    const u32 p2=H?__shfl_sync(0xffffffffu,2u*p,0):0u;(void)p2;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64* sit=stage_itw+prime*256u;
    u32 yv[8];
    {const u32* src=C+u64(P3A_P*8u)*N+(P3A_C*256u+t);
        #pragma unroll
        for(u32 d=0;d<8;++d)yv[d]=src[d*N];}
    u64 F;
#ifdef BPO_P3F   // p3f (PF, kernel bpo_last_cm3f): untwist argument = F table [prime][c][t] (bpo_ftab_init: the values of the block below)
    if(PF)F=__ldg(untwist+(u64(P3A_P)*65536u+(P3A_C*256u+t)));else
#endif
    {
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
    const u64* const tn=t2n+(u64(P3A_P*8u+warp)*512u+((P3A_C>>7)*256u+lane));
    u64 tv[8];
    #pragma unroll
    for(u32 m=0;m<8;++m)tv[m]=__ldg(tn+32u*m);
    if((dfl&1)&&lane<8u)bpo_discard(C+u64(P3A_P*8u+lane)*N+(P3A_C*256u+32u*warp));   // C lines read by this warp (all used)
    u32 v[8];
    u32* const base=X+warp*288u;
    {const uint4* p4=reinterpret_cast<const uint4*>(base+(8u*lane+4u*(lane>>2)));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    if(H)phdit_smr<false>(base,v,sit,p,p2,lane);else pzdit_smr<false>(base,v,sit,p,lane);
    u32* const out=D+(u64(P3A_P*8u+warp)*N+(P3A_C*256u+lane));
#undef P3A_C
#undef P3A_P
#ifdef BPO_P3EL
    const u32 pm=12u;   // dfl 8: group-0 residues L2 evict_last; dfl 4: evict_first (bpo_p3_body order)
#else
    const u32 pm=4u;
#endif
    P3ST if(dfl&pm){
        u64 pol;if(dfl&pm&8u)asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(pol));else asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pol));
        #pragma unroll
        for(u32 m=0;m<8;++m){const u32 r=pxmsh(v[m],tv[m],p);asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;"::"l"(out+32u*m),"r"(r),"l"(pol):"memory");}
    }else{
        #pragma unroll
        for(u32 m=0;m<8;++m)out[32u*m]=pxmsh(v[m],tv[m],p);
    }
#undef P3ST
}
#ifdef BPO_P3F
// p3f entry: the p3b body with F from the table (host launches it only when the table was built and the p3b conditions hold:
// N 65536, logR 8, no dfl 2; else bpo_last_cm3)
extern "C" __global__ void __maxnreg__(32) bpo_last_cm3f(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];(void)N;(void)logR;
#if BPO_NH30
    if(primes[blockIdx.y]<(1u<<30))bpo_p3b_body<true,true>(X,C,D,primes,untwist,stage_itw,t2n,dfl);else
#endif
    bpo_p3b_body<false,true>(X,C,D,primes,untwist,stage_itw,t2n,dfl);
}
// F table init (once): F[prime][c][t] = the bpo_p3b_body F factor (same formula), grid (256, K), 256 threads
extern "C" __global__ void __launch_bounds__(256) bpo_ftab_init(const u64* untwist,const u32* primes,u64* out){
    constexpr u32 N=65536u,logR=8u;
    const u32 c=blockIdx.x,prime=blockIdx.y,t=threadIdx.x,p=primes[prime];
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u32 k1=__brev(t)>>(32-logR);
    const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
    const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
    const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
    out[(u64(prime)*256u+c)*256u+t]=u64(wv)|(u64(q)<<32);
}
#endif
extern "C" __global__ void __maxnreg__(32) bpo_last_cm3(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;
    if(N==65536u&&logR==8u&&!(dfl&2u)){
#if BPO_NH30
        if(primes[blockIdx.y]<(1u<<30))bpo_p3b_body<true>(X,C,D,primes,untwist,stage_itw,t2n,dfl);else
#endif
        bpo_p3b_body<false>(X,C,D,primes,untwist,stage_itw,t2n,dfl);
        return;}
#if BPO_NH30
    if(primes[blockIdx.y]<(1u<<30))bpo_p3_body<BPO_P3T,true>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);else
#endif
    bpo_p3_body<BPO_P3T>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);
}
#else
extern "C" __global__ void __launch_bounds__(256) bpo_last_cm3(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;
#if BPO_NH30   // h30 prime set (default): Harvey-lazy J-DIT / R-DIT for the primes < 2^30
    if(primes[blockIdx.y]<(1u<<30))bpo_p3_body<BPO_P3T,true>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);else
#endif
    bpo_p3_body<BPO_P3T>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);
}
#endif
#if !defined(BPO_LEAN)
extern "C" __global__ void __launch_bounds__(256) bpo_last_cm3v(const u32* C,u32* D,const u32* primes,
    const u64* untwist,const u64* stage_itw,const u64* t2n,u32 N,u32 logR,u32 dfl){
    __shared__ __align__(128) u32 X[8*288];__shared__ __align__(8) u64 bar;
    bpo_p3_body<BPO_P3TV>(X,&bar,C,D,primes,untwist,stage_itw,t2n,N,logR,dfl);
}
#endif
// Copy of batched_crt (same arithmetic) reading column-major residues.
#ifdef BCRT_W
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(128,3) bpo_crt_cm(const u32* residues,u32* output,
 const u32* primes,const u64* ratios,const u32* partial_matrix,const u32* correction_table,
 const u32* q,const u32* bias,u32 N,u32 d,u64 res_stride,u64 out_stride) {
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64;
    constexpr u32 POS[8]={BCRT_POS};
    constexpr u32 PR[K]={BCRT_PRIMES};constexpr u64 MU[K]={BCRT_MUS};constexpr u32 PM[K*SW]={BCRT_PM};
    constexpr u32 QW[L]={BCRT_Q},BW[L]={BCRT_BIAS};
    __shared__ u32 stage[128*L];
    // column-major residues [r*8+j][c][h]: block = (column c, row half hb), thread = row h
    const u32 c=blockIdx.x&255u,hb=blockIdx.x>>8,h=hb*128+threadIdx.x,i=c*256+h;
    residues+=blockIdx.y*res_stride;output+=blockIdx.y*out_stride;
    u32 acc[NW];
    #pragma unroll
    for(u32 w=0;w<NW;++w)acc[w]=0;
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 t[K];u64 fixed=0;
        #pragma unroll
        for(u32 r=0;r<K;++r){
            t[r]=residues[(u64(r)*8+j)*N+i];
            const u32 p=PR[r];const u64 mu=MU[r];
            const u64 numerator=u64(t[r])<<32;
            u64 quotient=__umul64hi(numerator,mu);
            quotient+=(numerator-quotient*p)>=p;
            fixed+=quotient;
        }
        const u32 alpha=u32((fixed+(u64(1)<<31))>>32);
        // Z_j words are produced low to high and folded straight into acc
        // (shifted by pos_j); only the previous word is kept for the shift.
        const u32 o=POS[j]/32,sh=POS[j]%32;
        u64 carry=0,prev_high=0;u32 prevZ=0,c=0;
        #pragma unroll
        for(u32 w=0;w<=SW;++w){
            u32 z=0;
            if(w<SW){
                u64 low=0,high=0;
                #pragma unroll
                for(u32 r=0;r<K;++r){const u64 product=u64(t[r])*PM[r*SW+w];low+=u32(product);high+=product>>32;}
                low+=correction_table[alpha*SW+w];
                const u64 s=low+prev_high+carry;z=u32(s);carry=s>>32;prev_high=high;
            }
            const u32 part=sh?((z<<sh)|(prevZ>>(32-sh))):z;
            prevZ=z;
            if(o+w<NW){const u64 s=u64(acc[o+w])+part+c;acc[o+w]=u32(s);c=u32(s>>32);}
        }
        #pragma unroll
        for(u32 w=0;w<NW;++w)if(w>o+SW&&w<NW){const u64 s=u64(acc[w])+c;acc[w]=u32(s);c=u32(s>>32);}
    }
    // two folds: x = (x mod 2^W) + d*(x >> W), in place: the high words read
    // for word w live at w+WO >= WO > w, so they are read before overwritten.
    constexpr u32 WO=W/32,WS=W%32;
    #pragma unroll
    for(u32 f=0;f<2;++f){
        u64 c=0;
        #pragma unroll
        for(u32 w=0;w<NW;++w){
            const u32 a=w+WO<NW?acc[w+WO]:0,b=w+WO+1<NW?acc[w+WO+1]:0;
            const u32 hw=WS?((a>>WS)|(b<<(32-WS))):a;
            const u32 low=w<WO?acc[w]:(w==WO&&WS?(acc[w]&((1u<<WS)-1)):0);
            const u64 s=u64(low)+u64(hw)*d+c;acc[w]=u32(s);c=s>>32;
        }
    }
    // Now x < 2^W + small: words >= L are zero. if x >= q: x -= q
    u32 borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;borrow=u32(s>>63);}
    if(!borrow){
        borrow=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    }
    // x -= bias; if negative x += q
    borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-BW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    if(borrow){
        u64 c=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])+QW[w]+c;acc[w]=u32(s);c=s>>32;}
    }
    #pragma unroll
    for(u32 w=0;w<L;++w)stage[threadIdx.x*L+w]=acc[w];
    __syncthreads();
    for(u32 x=threadIdx.x;x<128*L;x+=128){const u32 rr=x/L,w=x-rr*L;output[u64((hb*128+rr)*256u+c)*L+w]=stage[x];}
}
#endif // bpo_crt_cm
#endif
// Copy of batched_crt_tc (same arithmetic) reading column-major residues [r*8+j][c][h].
#ifdef BTC_F
extern "C" __global__ void __launch_bounds__(128,3) bpo_crt_tc_cm(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* pfa,const u32* pfb,u32 pfitems,u32 dfl){
    if(pfitems){   // L2 prefetch of the input columns of this chain's next product's first P1 wave
        const u32 total=gridDim.x*blockDim.x;
        for(u32 t=blockIdx.x*blockDim.x+threadIdx.x;t<pfitems*256u;t+=total){
            const u32 it=t>>8,hh=t&255u;const u32* ps=(it>=256u?pfb:pfa)+u64(it&255u)*BCRT_L+u64(hh*256u)*BCRT_L;
            asm volatile("prefetch.global.L2 [%0];"::"l"(ps));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));asm volatile("prefetch.global.L2 [%0];"::"l"(ps+BCRT_L-1));
        }
    }
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F;
    static_assert(K==16&&SW==16,"tensor CRT geometry");
    constexpr u32 POS[8]={BCRT_POS};
    constexpr u32 QW[L]={BCRT_Q},BW[L]={BCRT_BIAS};
    constexpr u32 NPW[SW]={BTC_NP},OFW[SW]={BTC_OFF};
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 stage_all[4][32*L];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    __syncthreads();
    u32* st=stage_all[warp];
    const u32 i0=blockIdx.x*128+warp*32;
    u32 acc[NW];
    #pragma unroll
    for(u32 w=0;w<NW;++w)acc[w]=0;
    u32 pend=0;
    #pragma unroll
    for(u32 j=0;j<8;++j){
        u32 a[2][2][4];
        #pragma unroll
        for(u32 m=0;m<2;++m)
            #pragma unroll
            for(u32 s=0;s<2;++s){
                const u32 r=8*s+tig;const u32* base=residues+(u64(r)*8+j)*N+i0+16*m+gid;
                a[m][s][0]=base[0];a[m][s][1]=base[8];
                a[m][s][2]=base[u64(4)*8*N];a[m][s][3]=base[u64(4)*8*N+8];
            }
        #pragma unroll
        for(u32 T=0;T<NT;++T){
            const uint2 b0=*reinterpret_cast<const uint2*>(bsh+((T*2+0)*32+lane)*2);
            const uint2 b1=*reinterpret_cast<const uint2*>(bsh+((T*2+1)*32+lane)*2);
            #pragma unroll
            for(u32 m=0;m<2;++m){
                u32 c[4]={0,0,0,0};
                imma16832(c,a[m][0][0],a[m][0][1],a[m][0][2],a[m][0][3],b0.x,b0.y);
                imma16832(c,a[m][1][0],a[m][1][1],a[m][1][2],a[m][1][3],b1.x,b1.y);
                st[(16*m+gid)*SST+4*T+tig]=c[0]+(c[1]<<8);
                st[(16*m+gid+8)*SST+4*T+tig]=c[2]+(c[3]<<8);
            }
        }
        __syncwarp();
        u32 h[SST];
        #pragma unroll
        for(u32 q=0;q<SST/4;++q){const uint4 v=*reinterpret_cast<const uint4*>(st+lane*SST+4*q);h[4*q]=v.x;h[4*q+1]=v.y;h[4*q+2]=v.z;h[4*q+3]=v.w;}
        __syncwarp();
        const u64 S=u64(h[32])+(u64(h[33])<<16)+(u64(h[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        const u32 o=POS[j]/32,sh=POS[j]%32;
        constexpr u32 dummy=0;(void)dummy;
        const u32 inj=j?(POS[j-1]/32+SW+1-o):0xffffffffu;
        u64 zc=0,c=0;u32 prevZ=0;
        #pragma unroll
        for(u32 w=0;w<=SW;++w){
            u32 z=0;
            if(w<SW){const u64 v=u64(h[2*w])+(u64(h[2*w+1])<<16)+u64(alpha)*NPW[w]+OFW[w]+zc;z=u32(v);zc=v>>32;}
            const u32 part=sh?((z<<sh)|(prevZ>>(32-sh))):z;
            prevZ=z;
            if(o+w<NW){const u64 s=u64(acc[o+w])+part+c+(w==inj?pend:0u);acc[o+w]=u32(s);c=s>>32;}
        }
        pend=u32(c);
    }
    {   // final pending carry enters word POS[7]/32+SW+1
        constexpr u32 start=POS[7]/32+SW+1;u64 c=pend;
        #pragma unroll
        for(u32 w=0;w<NW;++w)if(w>=start){const u64 s=u64(acc[w])+c;acc[w]=u32(s);c=s>>32;}
    }
    constexpr u32 WO=W/32,WS=W%32;
    #pragma unroll
    for(u32 f=0;f<2;++f){
        u64 c=0;
        #pragma unroll
        for(u32 w=0;w<NW;++w){
            const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;
            const u32 hw=WS?((a_>>WS)|(b_<<(32-WS))):a_;
            const u32 low=w<WO?acc[w]:(w==WO&&WS?(acc[w]&((1u<<WS)-1)):0);
            const u64 s=u64(low)+u64(hw)*d+c;acc[w]=u32(s);c=s>>32;
        }
    }
    u32 borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;borrow=u32(s>>63);}
    if(!borrow){
        borrow=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-QW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    }
    borrow=0;
    #pragma unroll
    for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])-BW[w]-borrow;acc[w]=u32(s);borrow=u32(s>>63);}
    if(borrow){
        u64 c=0;
        #pragma unroll
        for(u32 w=0;w<L;++w){const u64 s=u64(acc[w])+QW[w]+c;acc[w]=u32(s);c=s>>32;}
    }
    if(dfl&1)for(u32 x=lane;x<K*8u;x+=32u)bpo_discard(residues+u64(x)*N+i0);   // residues consumed (acc built)
    #pragma unroll
    for(u32 w=0;w<L;++w)st[lane*L+w]=acc[w];
    __syncwarp();
    // column-major positions: i0 = c*256+h0 holds coefficients (h0+lane)*256+c
    {const u32 c=i0>>8,h0=i0&255u;
     for(u32 x=lane;x<32*L;x+=32){const u32 rr=x/L,w=x-rr*L;u32* o=output+u64((h0+rr)*256u+c)*L+w;if(dfl&2)__stcs(o,st[x]);else *o=st[x];}}
}
#endif
// Tensor-core P1: same values as bpo_first; residues via the byte IMMA of batched_prepare_tc
// (fa folded into byte weights). Warp w owns rows 32w..32w+31 as A fragments for all digits.
// Per prime quad u: lane (gid,tig) gets prime 4u+tig for rows 16mt+8hf+gid, all 8 digits.
#if defined(BPT_ENABLE)
#ifndef BPO_TC_MINB
#define BPO_TC_MINB 2
#endif
#define BPO_TC_SR 128
#define BPO_TC_XS (2048+8)
#define BPO_TC_SMEM (4*BPO_TC_XS>BPO_TC_SR*BCRT_L+8?4*BPO_TC_XS:BPO_TC_SR*BCRT_L+8)
// NC independent (prime,digit) columns per warp, interleaved for ILP (same butterflies as
// bpo_smr_rdif). base[n]: skewed 288-word regions; outputs 32 B vector stores.
template<u32 NC,bool GCT=false,u32 XA=0>
__device__ __forceinline__ void bpo_smr_rdifn(u32* const (&base)[NC],const u64* const (&st)[NC],const u64* const (&ct)[NC],
    const u32 (&pp)[NC],u32* const (&outp)[NC],u32 lane){
    u32 v[NC][8];
    if(XA==2){   // ablation: memory only (no butterflies / twiddles)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const uint4* p4=reinterpret_cast<const uint4*>(base[n]+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
            uint4* o=reinterpret_cast<uint4*>(outp[n]+lane*2048u);o[0]=x0;o[1]=x1;}
        return;
    }
    #pragma unroll
    for(u32 n=0;n<NC;++n)
        #pragma unroll
        for(u32 m=0;m<8;++m)v[n][m]=base[n][lane+36*m];
    #pragma unroll
    for(u32 m=0;m<4;++m)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u64 w=st[n][lane+32u*m];
            const u32 x=v[n][m],y=v[n][m+4];v[n][m]=add(x,y,p);v[n][m+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u64 w=st[n][128u+lane+32u*qq];
            #pragma unroll
            for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;
                const u32 x=v[n][m],y=v[n][m+2];v[n][m]=add(x,y,p);v[n][m+2]=mul_shoup(sub(x,y,p),w,p);}}
    #pragma unroll
    for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u64 w=st[n][192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=v[n][m],y=v[n][m+1];v[n][m]=add(x,y,p);v[n][m+1]=mul_shoup(sub(x,y,p),w,p);}}
    #pragma unroll
    for(u32 n=0;n<NC;++n)
        #pragma unroll
        for(u32 m=0;m<8;++m)base[n][lane+36*m]=v[n][m];
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 n=0;n<NC;++n)
        #pragma unroll
        for(u32 m=0;m<8;++m)v[n][m]=base[n][b2+4*m];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u64 w=st[n][224u+(lane&3u)+4*lo];
            const u32 x=v[n][lo],y=v[n][lo+4];v[n][lo]=add(x,y,p);v[n][lo+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 k=0;k<4;++k)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u32 lo=(k&1)+4*(k>>1);const u64 w=st[n][240u+(lane&3u)+4*(lo&1)];
            const u32 x=v[n][lo],y=v[n][lo+2];v[n][lo]=add(x,y,p);v[n][lo+2]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u64 w=st[n][248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[n][lo],y=v[n][lo+1];v[n][lo]=add(x,y,p);v[n][lo+1]=mul_shoup(sub(x,y,p),w,p);}}
    __syncwarp();
    #pragma unroll
    for(u32 n=0;n<NC;++n)
        #pragma unroll
        for(u32 m=0;m<8;++m)base[n][b2+4*m]=v[n][m];
    __syncwarp();
    #pragma unroll
    for(u32 n=0;n<NC;++n){const uint4* p4=reinterpret_cast<const uint4*>(base[n]+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
        v[n][0]=x0.x;v[n][1]=x0.y;v[n][2]=x0.z;v[n][3]=x0.w;v[n][4]=x1.x;v[n][5]=x1.y;v[n][6]=x1.z;v[n][7]=x1.w;}
    #pragma unroll
    for(u32 k=0;k<4;++k)
        #pragma unroll
        for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u32 lo=(k&1)+4*(k>>1);const u64 w=st[n][252u+(lo&1)];
            const u32 x=v[n][lo],y=v[n][lo+2];v[n][lo]=add(x,y,p);v[n][lo+2]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 n=0;n<NC;++n){const u32 p=pp[n];const u64 w=st[n][254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[n][lo],y=v[n][lo+1];v[n][lo]=add(x,y,p);v[n][lo+1]=mul_shoup(sub(x,y,p),w,p);}}
    #pragma unroll
    for(u32 n=0;n<NC;++n){const u32 p=pp[n];
        if(GCT){   // ct = ctab row in global (L1/L2): slot 8*lane+m, 16 B loads
            const ulonglong2* c2=reinterpret_cast<const ulonglong2*>(ct[n]+8*lane);
            #pragma unroll
            for(u32 m2=0;m2<4;++m2){const ulonglong2 w=__ldg(c2+m2);
                v[n][2*m2]=mul_shoup(v[n][2*m2],u64(w.x),p);v[n][2*m2+1]=mul_shoup(v[n][2*m2+1],u64(w.y),p);}
        }else{
        const u64* c8=ct[n]+lane;   // skewed: slot 8*lane+m at m*33+lane
        #pragma unroll
        for(u32 m=0;m<8;++m)v[n][m]=mul_shoup(v[n][m],c8[m*33],p);}
        uint4* o=reinterpret_cast<uint4*>(outp[n]+lane*2048u);
        if(XA==1){if((v[n][0]^v[n][1]^v[n][2]^v[n][3]^v[n][4]^v[n][5]^v[n][6]^v[n][7])!=0xffffffffu)continue;}   // ablation: compute only
        o[0]=make_uint4(v[n][0],v[n][1],v[n][2],v[n][3]);o[1]=make_uint4(v[n][4],v[n][5],v[n][6],v[n][7]);}
}
#ifndef BPO_RDNC
#define BPO_RDNC 1
#endif
#define BPO_TC_XS2 (8*288+8)
#define BPO_TC_SMEM2 (4*BPO_TC_XS2>BPO_TC_SR*BCRT_L+8?4*BPO_TC_XS2:BPO_TC_SR*BCRT_L+8)
template<bool SMR,u32 NC=0,bool GCT=false,u32 XA=0>
__device__ __forceinline__ void bpo_tc_item(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 ua,u32 ub,
    const u32 c,const u32 op,u32* SG,u64 (*CT)[264],u32 pf=0xffffffffu,const u64* ctab=nullptr,bool ef=false){
    constexpr u32 W=BCRT_W,L=BCRT_L;
    u64 efp=0;if(ef)asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(efp));
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr u32 SR=BPO_TC_SR,XS=SMR?BPO_TC_XS2:BPO_TC_XS,XD=SMR?288u:256u;
    // SMR: digit region of 288 words, row -> row + 4*(row>>5) (bank skew for the transposes)
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(c)*L;
    u32* dst=op?B:A;
    u32 af[8][2][4];
#if BPO_ABL==7
    #pragma unroll
    for(u32 j=0;j<8;++j)
        #pragma unroll
        for(u32 mt=0;mt<2;++mt)
            #pragma unroll
            for(u32 x=0;x<4;++x)af[j][mt][x]=(h*0x9e3779b9u)^(j*7+mt*3+x+c);
    if(0)
#endif
#ifndef BPO_AFD
#define BPO_AFD 1
#endif
    if(SMR&&BPO_AFD){   // A fragments straight from global (L1/L2): no smem staging pass
        #pragma unroll
        for(u32 j=0;j<8;++j){
            const u32 wb=POS[j]/32,sh=POS[j]%32;
            #pragma unroll
            for(u32 mt=0;mt<2;++mt)
                #pragma unroll
                for(u32 x=0;x<4;++x){
                    const u32 row=32*warp+16*mt+gid+8*(x&1),t=tig+4*(x>>1);
                    const u32* rp=src+u64(row*256u)*L;
                    const u32 w0=wb+t,w1=min(wb+t+1,L-1);
                    u32 x0,x1;if(ef){asm("ld.global.nc.L2::cache_hint.u32 %0, [%1], %2;":"=r"(x0):"l"(rp+min(w0,L-1)),"l"(efp));asm("ld.global.nc.L2::cache_hint.u32 %0, [%1], %2;":"=r"(x1):"l"(rp+w1),"l"(efp));}
                    else{x0=__ldg(rp+min(w0,L-1));x1=__ldg(rp+w1);}
                    const u32 v=__funnelshift_r(x0,x1,sh);
                    const int width=int(POS[j+1])-int(POS[j]+32*t);
                    af[j][mt][x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
                }
        }
    }else
    #pragma unroll 1
    for(u32 q=0;q<256/SR;++q){
        __syncthreads();
#if BPO_ABL==8
        for(u32 x=h;x<SR*L;x+=256)SG[x]=x*0x9e3779b9u+c;
#else
        for(u32 x=h;x<SR*L;x+=256){const u32 rr=x/L,w=x-rr*L;SG[x]=__ldg(src+u64((SR*q+rr)*256u)*L+w);}
#endif
        if(h<8)SG[SR*L+h]=0;
        __syncthreads();
        if(warp*32/SR==q){
            #pragma unroll
            for(u32 j=0;j<8;++j){
                const u32 wb=POS[j]/32,sh=POS[j]%32;
                #pragma unroll
                for(u32 mt=0;mt<2;++mt)
                    #pragma unroll
                    for(u32 x=0;x<4;++x){
                        const u32 row=(32*warp+16*mt+gid+8*(x&1))%SR,t=tig+4*(x>>1);
                        const u32* rp=SG+row*L+wb+t;
                        const u32 v=__funnelshift_r(rp[0],rp[1],sh);
                        const int width=int(POS[j+1])-int(POS[j]+32*t);
                        af[j][mt][x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
                    }
            }
        }
    }
    if(pf<512u){   // L2 prefetch of the next wave's column: its staging then hits L2
        const u32* ps=(pf>=256u?b:a)+u64(pf&255u)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    const u32 stride=weighted_root_stride(N);
    const u32 k1h=__brev(h)>>(32-logR);
    u32* X=SG;
    #pragma unroll 1
    for(u32 u=ua;u<ub;++u){
        __syncthreads();   // previous quad's X / CT reads done (and staging reads)
        if(GCT){}else
        if(ctab){   // host-built cross twiddles [r][c][slot] (t1 folded), coalesced
            #pragma unroll
            for(u32 q=0;q<4;++q)CT[q][NC>0?(h&7u)*33u+(h>>3):h]=ctab[(size_t(4*u+q)*256u+c)*256u+h];
        }else
        if(!(SMR&&BPO_ABL==14))
        #pragma unroll
        for(u32 q=0;q<4;++q){   // cross twiddle (with the column-constant t1 folded in) of slot h
            const u32 r=4*u+q,p=bpr_pr[r];const u64* tw=twist+size_t(r)*stride;
            const u32 wv=mul_shoup(u32(bm_root(tw,2u*c*k1h,N,p)),tw[1+(c&127u)],p);
            const u64 m0=tw[0];u32 qq=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
            const u64 rem=(u64(wv)<<32)-u64(qq)*p;if(rem>=p)++qq;
            CT[q][NC>0?(h&7u)*33u+(h>>3):h]=u64(wv)|(u64(qq)<<32);
        }
        {
            const u32 r=4*u+tig,p=bpr_pr[r];const u64 mu=bpr_mu[r];
            const u64* tw=twist+size_t(r)*stride;const u64* st=stage_tw+size_t(r)*256;
            u32 res[4][8];   // [row 2mt+hf][digit]
            #pragma unroll
            for(u32 j=0;j<8;++j){
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+u);
                #pragma unroll
                for(u32 mt=0;mt<2;++mt){
                    u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                    if(XA==2){c0[0]=af[j][mt][0];c0[2]=af[j][mt][2];c1[0]=af[j][mt][1];c1[2]=af[j][mt][3]^bv.x;}else
                    if(SMR&&BPO_ABL==11){c0[0]=af[j][mt][0]^bv.x;c0[1]=af[j][mt][1];c0[2]=af[j][mt][2]^bv.z;c0[3]=af[j][mt][3];c1[0]=bv.y;c1[1]=bv.w;c1[2]=af[j][mt][1]^bv.x;c1[3]=u;}else{
                    pimma16832(c0,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                    pimma16832(c1,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);}
                    #pragma unroll
                    for(u32 hf=0;hf<2;++hf){
                        if(XA==2){res[2*mt+hf][j]=c0[2*hf]^c1[2*hf];continue;}
                        const u64 x=u64(c0[2*hf])+(u64(c0[2*hf+1])<<8)+(u64(c1[2*hf])<<16)+(u64(c1[2*hf+1])<<24);
                        res[2*mt+hf][j]=reduce(x,p,mu);
                    }
                }
            }
            #pragma unroll
            for(u32 rr=0;rr<4;++rr){
                const u32 row=32*warp+8*rr+gid;   // rr = 2mt+hf -> 16mt+8hf
                const u64 t2=tw[1+128u+((row*256u+c)>>7)];
                #pragma unroll
                if(XA!=2&&!(SMR&&BPO_ABL==12)){
                for(u32 j=0;j<8;++j)res[rr][j]=mul_shoup(res[rr][j],t2,p);
                BPO_JDIF(res[rr],st)
                }
                #pragma unroll
                for(u32 j=0;j<8;++j)X[tig*XS+j*XD+(SMR?row+4*(row>>5):row)]=res[rr][j];
            }
        }
        __syncthreads();
        if(SMR&&NC>0){
        #pragma unroll 1
        constexpr u32 NCA=NC>0?NC:1;
        for(u32 q0=0;q0<4;q0+=NCA){
            u32* bs[NCA];const u64* sts[NCA];const u64* cts[NCA];u32 ps[NCA];u32* os[NCA];
            #pragma unroll
            for(u32 n=0;n<NCA;++n){const u32 q=q0+n,r=4*u+q;bs[n]=X+q*XS+warp*XD;sts[n]=stage_tw+size_t(r)*256;cts[n]=GCT?ctab+(size_t(r)*256u+c)*256u:CT[q];ps[n]=bpr_pr[r];os[n]=dst+u64(r*8+warp)*N+c*8u;}
            bpo_smr_rdifn<NCA,GCT,XA>(bs,sts,cts,ps,os,lane);
        }
        }else if(SMR){
        #pragma unroll 1
        for(u32 q=0;q<4;++q){
            const u32 r=4*u+q,p=bpr_pr[r];const u64* st=stage_tw+size_t(r)*256;
            u32* base=X+q*XS+warp*XD;
            u32 v[8];
            if(BPO_ABL==13){
                {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
                 uint4* o=reinterpret_cast<uint4*>(dst+u64(r*8+warp)*N+lane*2048u+c*8u);o[0]=x0;o[1]=x1;}
                continue;
            }
            // layout 1: row = lane+32m (regs = row bits 5..7); stages len 256,128,64
            #pragma unroll
            for(u32 m=0;m<8;++m)v[m]=base[lane+36*m];
            #pragma unroll
            for(u32 m=0;m<4;++m){const u64 w=st[lane+32u*m];
                const u32 x=v[m],y=v[m+4];v[m]=add(x,y,p);v[m+4]=mul_shoup(sub(x,y,p),w,p);}
            #pragma unroll
            for(u32 qq=0;qq<2;++qq){const u64 w=st[128u+lane+32u*qq];
                #pragma unroll
                for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;
                    const u32 x=v[m],y=v[m+2];v[m]=add(x,y,p);v[m+2]=mul_shoup(sub(x,y,p),w,p);}}
            {const u64 w=st[192u+lane];
                #pragma unroll
                for(u32 m=0;m<8;m+=2){const u32 x=v[m],y=v[m+1];v[m]=add(x,y,p);v[m+1]=mul_shoup(sub(x,y,p),w,p);}}
            #pragma unroll
            for(u32 m=0;m<8;++m)base[lane+36*m]=v[m];
            __syncwarp();
            // layout 2: row = (lane&3) | m<<2 | (lane>>2)<<5 (regs = row bits 2..4); stages len 32,16,8
            const u32 b2=(lane&3u)+36u*(lane>>2);
            #pragma unroll
            for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
            #pragma unroll
            for(u32 lo=0;lo<4;++lo){const u64 w=st[224u+(lane&3u)+4*lo];
                const u32 x=v[lo],y=v[lo+4];v[lo]=add(x,y,p);v[lo+4]=mul_shoup(sub(x,y,p),w,p);}
            #pragma unroll
            for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[240u+(lane&3u)+4*(lo&1)];
                const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
            {const u64 w=st[248u+(lane&3u)];
                #pragma unroll
                for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
            __syncwarp();
            #pragma unroll
            for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
            __syncwarp();
            // layout 3: row = 8*lane+m (regs = row bits 0..2); stages len 4,2
            {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
             v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
            #pragma unroll
            for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[252u+(lo&1)];
                const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
            {const u64 w=st[254u];
                #pragma unroll
                for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
            const u64* ct=CT[q]+8*lane;
            #pragma unroll
            for(u32 m=0;m<8;++m)v[m]=mul_shoup(v[m],ct[m],p);
            uint4* o=reinterpret_cast<uint4*>(dst+u64(r*8+warp)*N+lane*2048u+c*8u);
            o[0]=make_uint4(v[0],v[1],v[2],v[3]);o[1]=make_uint4(v[4],v[5],v[6],v[7]);
        }
        }else{
        #pragma unroll 1
        for(u32 q=0;q<4;++q){
            const u32 r=4*u+q,p=bpr_pr[r];const u64* st=stage_tw+size_t(r)*256;
            u32 v[8];
            #pragma unroll
            for(u32 m=0;m<8;++m)v[m]=X[q*XS+warp*256+lane+32*m];
            #if BPO_ABL!=1
            BPO_RDIF(v,st)
#endif

            u32* out=dst+u64(r*8+warp)*N+c*8u;
            #pragma unroll
            for(u32 m=0;m<8;++m){
                const u32 slot=lane+32*m;
                out[(slot>>3)*2048u+(slot&7u)]=mul_shoup(v[m],CT[q][slot],p);
            }
        }
        }
    }
}
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256,BPO_TC_MINB) bpo_first_tc(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep){
    __shared__ __align__(16) u32 SG[BPO_TC_SMEM];
    __shared__ u64 CT[4][264];
    const u32 id=blockIdx.x;
    bpo_tc_item<false>(a,b,A,B,twist,stage_tw,bpt,N,logR,r0/4,r1/4,id&255u,id>>8,SG,CT,pfstep?id+pfstep:0xffffffffu);
}
#endif // bpo_first_tc
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256,BPO_TC_MINB) bpo_first_tc2(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep){
    __shared__ __align__(16) u32 SG[BPO_TC_SMEM2];
    __shared__ u64 CT[4][264];
    const u32 id=blockIdx.x;
    bpo_tc_item<true>(a,b,A,B,twist,stage_tw,bpt,N,logR,r0/4,r1/4,id&255u,id>>8,SG,CT,pfstep?id+pfstep:0xffffffffu);
}
#endif // bpo_first_tc2
// R-DIF of one (prime,digit) column in the warp's 288-word region (skewed rows), cross twiddle,
// 32 B vector stores; same butterflies as bpo_tc_item<true>.
__device__ __forceinline__ void bpo_smr_rdif(u32* base,const u64* st,const u64* ct,u32 p,u32* outp,u32 lane){
    u32 v[8];
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[lane+36*m];
    #pragma unroll
    for(u32 m=0;m<4;++m){const u64 w=st[lane+32u*m];
        const u32 x=v[m],y=v[m+4];v[m]=add(x,y,p);v[m+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=st[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;
            const u32 x=v[m],y=v[m+2];v[m]=add(x,y,p);v[m+2]=mul_shoup(sub(x,y,p),w,p);}}
    {const u64 w=st[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=v[m],y=v[m+1];v[m]=add(x,y,p);v[m+1]=mul_shoup(sub(x,y,p),w,p);}}
    #pragma unroll
    for(u32 m=0;m<8;++m)base[lane+36*m]=v[m];
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo){const u64 w=st[224u+(lane&3u)+4*lo];
        const u32 x=v[lo],y=v[lo+4];v[lo]=add(x,y,p);v[lo+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[240u+(lane&3u)+4*(lo&1)];
        const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
    {const u64 w=st[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[252u+(lo&1)];
        const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
    {const u64 w=st[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
    const u64* c8=ct+8*lane;
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=mul_shoup(v[m],c8[m],p);
    uint4* o=reinterpret_cast<uint4*>(outp+lane*2048u);
    o[0]=make_uint4(v[0],v[1],v[2],v[3]);o[1]=make_uint4(v[4],v[5],v[6],v[7]);
}
#ifndef BPO_TC3_MINB
#define BPO_TC3_MINB 3
#endif
// Low-register P1 (3 blocks/SM): A fragments rebuilt per (quad,digit) from global (L1 hits),
// exchange tile for 2 primes; each prime quad is exchanged in two halves (tig<2, tig>=2).
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256,BPO_TC3_MINB) bpo_first_tc3(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr u32 XS=8*288+8;
    __shared__ __align__(16) u32 X[2*XS];
    __shared__ u64 CT[4][256];
    const u32 id=blockIdx.x,c=id&255u,op=id>>8;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(c)*L;
    u32* dst=op?B:A;
    if(pfstep&&id+pfstep<512u){
        const u32 pf=id+pfstep;const u32* ps=(pf>=256u?b:a)+u64(pf&255u)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    const u32 stride=weighted_root_stride(N);
    const u32 k1h=__brev(h)>>(32-logR);
    #pragma unroll 1
    for(u32 u=r0/4;u<r1/4;++u){
        __syncthreads();
        #pragma unroll
        for(u32 q=0;q<4;++q){
            const u32 r=4*u+q,p=bpr_pr[r];const u64* tw=twist+size_t(r)*stride;
            const u32 wv=mul_shoup(u32(bm_root(tw,2u*c*k1h,N,p)),tw[1+(c&127u)],p);
            const u64 m0=tw[0];u32 qq=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
            const u64 rem=(u64(wv)<<32)-u64(qq)*p;if(rem>=p)++qq;
            CT[q][h]=u64(wv)|(u64(qq)<<32);
        }
        const u32 r=4*u+tig,p=bpr_pr[r];const u64 mu=bpr_mu[r];
        const u64* tw=twist+size_t(r)*stride;const u64* st=stage_tw+size_t(r)*256;
        u32 res[4][8];
        #pragma unroll
        for(u32 j=0;j<8;++j){
            const u32 wb=POS[j]/32,sh=POS[j]%32;
            const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+u);
            #pragma unroll
            for(u32 mt=0;mt<2;++mt){
                u32 af[4];
                #pragma unroll
                for(u32 x=0;x<4;++x){
                    const u32 row=32*warp+16*mt+gid+8*(x&1),t=tig+4*(x>>1);
                    const u32* rp=src+u64(row*256u)*L;
                    const u32 v=__funnelshift_r(__ldg(rp+min(wb+t,L-1)),__ldg(rp+min(wb+t+1,L-1)),sh);
                    const int width=int(POS[j+1])-int(POS[j]+32*t);
                    af[x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
                }
                u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                pimma16832(c0,af[0],af[1],af[2],af[3],bv.x,bv.y);
                pimma16832(c1,af[0],af[1],af[2],af[3],bv.z,bv.w);
                #pragma unroll
                for(u32 hf=0;hf<2;++hf){
                    const u64 x=u64(c0[2*hf])+(u64(c0[2*hf+1])<<8)+(u64(c1[2*hf])<<16)+(u64(c1[2*hf+1])<<24);
                    res[2*mt+hf][j]=reduce(x,p,mu);
                }
            }
        }
        #pragma unroll
        for(u32 rr=0;rr<4;++rr){
            const u32 row=32*warp+8*rr+gid;
            const u64 t2=tw[1+128u+((row*256u+c)>>7)];
            #pragma unroll
            for(u32 j=0;j<8;++j)res[rr][j]=mul_shoup(res[rr][j],t2,p);
            BPO_JDIF(res[rr],st)
        }
        #pragma unroll
        for(u32 half=0;half<2;++half){
            if(half)__syncthreads();   // first half's R-DIF reads done
            if((tig>>1)==half){
                #pragma unroll
                for(u32 rr=0;rr<4;++rr){
                    const u32 row=32*warp+8*rr+gid;
                    #pragma unroll
                    for(u32 j=0;j<8;++j)X[(tig&1)*XS+j*288+row+4*(row>>5)]=res[rr][j];
                }
            }
            __syncthreads();
            #pragma unroll 1
            for(u32 q2=0;q2<2;++q2){
                const u32 q=2*half+q2,rq=4*u+q;
                bpo_smr_rdif(X+q2*XS+warp*288,stage_tw+size_t(rq)*256,CT[q],bpr_pr[rq],dst+u64(rq*8+warp)*N+c*8u,lane);
            }
        }
    }
}
#endif // bpo_first_tc3
#if !defined(BPO_LEAN)||(!BPO_LEAN_SPLIT&&!BPO_LEAN_GCT)
extern "C" __global__ void __launch_bounds__(256,BPO_TC_MINB) bpo_first_tc2n(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab){
    __shared__ __align__(16) u32 SG[BPO_TC_SMEM2];
    __shared__ u64 CT[4][264];
    const u32 id=blockIdx.x;
    bpo_tc_item<true,BPO_RDNC>(a,b,A,B,twist,stage_tw,bpt,N,logR,r0/4,r1/4,id&255u,id>>8,SG,CT,pfstep?id+pfstep:0xffffffffu,ctab);
}
#endif // bpo_first_tc2n
// Same as bpo_first_tc2n with the R-DIF reading the cross twiddles straight from ctab (no CT smem staging).
#if !defined(BPO_LEAN)||(!BPO_LEAN_SPLIT&&BPO_LEAN_GCT&&BPO_LEAN_TC2)
extern "C" __global__ void __launch_bounds__(256,BPO_TC_MINB) bpo_first_tc2g(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab){
    __shared__ __align__(16) u32 SG[BPO_TC_SMEM2];
    const u32 id=blockIdx.x;
    const bool ef=pfstep>>31;pfstep&=0x7fffffffu;   // bit 31: input loads L2 evict-first (L2 schedule)
    bpo_tc_item<true,BPO_RDNC,true>(a,b,A,B,twist,stage_tw,bpt,N,logR,r0/4,r1/4,id&255u,id>>8,SG,nullptr,pfstep?id+pfstep:0xffffffffu,ctab,ef);
}
#endif // bpo_first_tc2g
// bpo_first_w: 512-thread P1 (16 warps x 16 rows): A fragments af[8][4] = 32 registers per lane, so two
// 512-thread blocks share an SM (32 warps vs 16 for tc2g). Same values as bpo_first_tc2g.
#ifndef BPO_W_MINB
#define BPO_W_MINB 2
#endif
#if !defined(BPO_LEAN)||BPO_LEAN_P1W
extern "C" __global__ void __launch_bounds__(512,BPO_W_MINB) bpo_first_w(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 pfstep,const u64* ctab){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr u32 XS=BPO_TC_XS2,XD=288u;
    __shared__ __align__(16) u32 X[4*XS];
    pfstep&=0x7fffffffu;
    const u32 id=blockIdx.x,c=id&255u,op=id>>8;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(c)*L;
    u32* dst=op?B:A;
    u32 af[8][4];
    #pragma unroll
    for(u32 j=0;j<8;++j){
        const u32 wb=POS[j]/32,sh=POS[j]%32;
        #pragma unroll
        for(u32 x=0;x<4;++x){
            const u32 row=16*warp+gid+8*(x&1),t=tig+4*(x>>1);
            const u32* rp=src+u64(row*256u)*L;
            const u32 w0=wb+t,w1=min(wb+t+1,L-1);
            const u32 v=__funnelshift_r(__ldg(rp+min(w0,L-1)),__ldg(rp+w1),sh);
            const int width=int(POS[j+1])-int(POS[j]+32*t);
            af[j][x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
        }
    }
    if(pfstep&&h<256u&&id+pfstep<512u){
        const u32 pf=id+pfstep;const u32* ps=(pf>=256u?b:a)+u64(pf&255u)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    const u32 stride=weighted_root_stride(N);
    #pragma unroll 1
    for(u32 u=r0/4;u<r1/4;++u){
        __syncthreads();   // previous quad's R-DIF reads of X done
        {
            const u32 r=4*u+tig,p=bpr_pr[r];const u64 mu=bpr_mu[r];
            const u64* tw=twist+size_t(r)*stride;const u64* st=stage_tw+size_t(r)*256;
            u32 res[2][8];   // [hf][digit], row 16warp+8hf+gid
            #pragma unroll
            for(u32 j=0;j<8;++j){
                const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+u);
                u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                pimma16832(c0,af[j][0],af[j][1],af[j][2],af[j][3],bv.x,bv.y);
                pimma16832(c1,af[j][0],af[j][1],af[j][2],af[j][3],bv.z,bv.w);
                #pragma unroll
                for(u32 hf=0;hf<2;++hf){
                    const u64 x=u64(c0[2*hf])+(u64(c0[2*hf+1])<<8)+(u64(c1[2*hf])<<16)+(u64(c1[2*hf+1])<<24);
                    res[hf][j]=reduce(x,p,mu);
                }
            }
            #pragma unroll
            for(u32 hf=0;hf<2;++hf){
                const u32 row=16*warp+8*hf+gid;
                const u64 t2=tw[1+128u+((row*256u+c)>>7)];
                #pragma unroll
                for(u32 j=0;j<8;++j)res[hf][j]=mul_shoup(res[hf][j],t2,p);
                BPO_JDIF(res[hf],st)
                #pragma unroll
                for(u32 j=0;j<8;++j)X[tig*XS+j*XD+row+4*(row>>5)]=res[hf][j];
            }
        }
        __syncthreads();
        #pragma unroll 1
        for(u32 n=0;n<2;++n){
            const u32 j=warp&7u,q=2*(warp>>3)+n,r=4*u+q;
            u32* bs[1]={X+q*XS+j*XD};const u64* sts[1]={stage_tw+size_t(r)*256};const u64* cts[1]={ctab+(size_t(r)*256u+c)*256u};
            const u32 ps[1]={bpr_pr[r]};u32* os[1]={dst+u64(r*8+j)*N+c*8u};
            bpo_smr_rdifn<1,true>(bs,sts,cts,ps,os,lane);
        }
    }
}
#endif // bpo_first_w
// Split P1 (variant 4 default unless FHERMA_V5_OFF bit 2048): (a) bpo_p1a = TC base conversion + twist
// + J-DIF for columns c0..c0+ncol-1 and prime quads ua..ub-1, no smem, residues straight to an
// L2-resident intermediate T[op][r-4ua][j][cl][row]; (b) bpo_p1b = R-DIF + cross twiddle, one warp per
// (prime,digit,column), high occupancy. Same exact values as bpo_first_tc2n.
#if !defined(BPO_LEAN)||BPO_LEAN_SPLIT
extern "C" __global__ void __launch_bounds__(256,BPO_TC_MINB) bpo_p1a(const u32* a,const u32* b,u32* T,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 c0,u32 ncol,u32 ua,u32 ub,u32 pfstep){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    const u32 id=blockIdx.x,cl=id%ncol,op=id/ncol,c=c0+cl;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,gid=lane>>2,tig=lane&3;
    const u32* src=(op?b:a)+u64(c)*L;
    u32 af[8][2][4];
    #pragma unroll
    for(u32 j=0;j<8;++j){
        const u32 wb=POS[j]/32,sh=POS[j]%32;
        #pragma unroll
        for(u32 mt=0;mt<2;++mt)
            #pragma unroll
            for(u32 x=0;x<4;++x){
                const u32 row=32*warp+16*mt+gid+8*(x&1),t=tig+4*(x>>1);
                const u32* rp=src+u64(row*256u)*L;
                const u32 w0=wb+t,w1=min(wb+t+1,L-1);
                const u32 v=__funnelshift_r(__ldg(rp+min(w0,L-1)),__ldg(rp+w1),sh);
                const int width=int(POS[j+1])-int(POS[j]+32*t);
                af[j][mt][x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
            }
    }
    if(pfstep&&id+pfstep<2u*ncol){   // L2 prefetch of the next wave's column
        const u32 pf=id+pfstep;
        const u32* ps=(pf>=ncol?b:a)+u64(c0+pf%ncol)*L+u64(h*256u)*L;
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+32));
        asm volatile("prefetch.global.L2 [%0];"::"l"(ps+L-1));
    }
    const u32 stride=weighted_root_stride(N),nr=4u*(ub-ua);
    #pragma unroll 1
    for(u32 u=ua;u<ub;++u){
        const u32 r=4*u+tig,p=bpr_pr[r];const u64 mu=bpr_mu[r];
        const u64* tw=twist+size_t(r)*stride;const u64* st=stage_tw+size_t(r)*256;
        u32 res[4][8];
        #pragma unroll
        for(u32 j=0;j<8;++j){
            const uint4 bv=__ldg(reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16)+u);
            #pragma unroll
            for(u32 mt=0;mt<2;++mt){
                u32 c0_[4]={0,0,0,0},c1_[4]={0,0,0,0};
                pimma16832(c0_,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.x,bv.y);
                pimma16832(c1_,af[j][mt][0],af[j][mt][1],af[j][mt][2],af[j][mt][3],bv.z,bv.w);
                #pragma unroll
                for(u32 hf=0;hf<2;++hf){
                    const u64 x=u64(c0_[2*hf])+(u64(c0_[2*hf+1])<<8)+(u64(c1_[2*hf])<<16)+(u64(c1_[2*hf+1])<<24);
                    res[2*mt+hf][j]=reduce(x,p,mu);
                }
            }
        }
        u32* to=T+((size_t(op)*nr+(r-4*ua))*8u*ncol+cl)*256u;
        #pragma unroll
        for(u32 rr=0;rr<4;++rr){
            const u32 row=32*warp+8*rr+gid;
            const u64 t2=tw[1+128u+((row*256u+c)>>7)];
            #pragma unroll
            for(u32 j=0;j<8;++j)res[rr][j]=mul_shoup(res[rr][j],t2,p);
            BPO_JDIF(res[rr],st)
            #pragma unroll
            for(u32 j=0;j<8;++j)to[size_t(j)*ncol*256u+row]=res[rr][j];
        }
    }
}
#endif // bpo_p1a
#ifndef BPO_P1B_MINB
#define BPO_P1B_MINB 4
#endif
// grid (ncol, nr, 2 operands), 256 threads: warp j = digit of prime r0+blockIdx.y, column c0+blockIdx.x.
// flags bit0: stage the cross-twiddle row through smem (else read ctab directly, L1/L2).
#if !defined(BPO_LEAN)||BPO_LEAN_SPLIT
extern "C" __global__ void __launch_bounds__(256,BPO_P1B_MINB) bpo_p1b(const u32* __restrict__ T,u32* A,u32* B,
    const u64* __restrict__ stage_tw,const u64* __restrict__ ctab,u32 N,u32 c0,u32 ncol,u32 r0,u32 flags){
    __shared__ __align__(16) u32 X[8*288];
    __shared__ __align__(16) u64 CS[264];   // skewed: slot 8*lane+m at m*33+lane
    const u32 cl=blockIdx.x,rl=blockIdx.y,op=blockIdx.z,nr=gridDim.y,c=c0+cl,r=r0+rl;
    const u32 h=threadIdx.x,lane=h&31u,warp=h>>5,p=bpr_pr[r];
    const u64* st=stage_tw+size_t(r)*256;
    const u64* ct=ctab+(size_t(r)*256u+c)*256u;
    if(flags&1u){CS[(h&7u)*33u+(h>>3)]=ct[h];__syncthreads();}
    const u32* srcp=T+((size_t(op)*nr+rl)*8u*ncol+size_t(warp)*ncol+cl)*256u;
    u32* base=X+warp*288u;
    u32 v[8];
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=srcp[lane+32*m];
    #pragma unroll
    for(u32 m=0;m<4;++m){const u64 w=st[lane+32u*m];
        const u32 x=v[m],y=v[m+4];v[m]=add(x,y,p);v[m+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 qq=0;qq<2;++qq){const u64 w=st[128u+lane+32u*qq];
        #pragma unroll
        for(u32 hh=0;hh<2;++hh){const u32 m=qq+4*hh;
            const u32 x=v[m],y=v[m+2];v[m]=add(x,y,p);v[m+2]=mul_shoup(sub(x,y,p),w,p);}}
    {const u64 w=st[192u+lane];
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=v[m],y=v[m+1];v[m]=add(x,y,p);v[m+1]=mul_shoup(sub(x,y,p),w,p);}}
    #pragma unroll
    for(u32 m=0;m<8;++m)base[lane+36*m]=v[m];
    __syncwarp();
    const u32 b2=(lane&3u)+36u*(lane>>2);
    #pragma unroll
    for(u32 m=0;m<8;++m)v[m]=base[b2+4*m];
    #pragma unroll
    for(u32 lo=0;lo<4;++lo){const u64 w=st[224u+(lane&3u)+4*lo];
        const u32 x=v[lo],y=v[lo+4];v[lo]=add(x,y,p);v[lo+4]=mul_shoup(sub(x,y,p),w,p);}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[240u+(lane&3u)+4*(lo&1)];
        const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
    {const u64 w=st[248u+(lane&3u)];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
    __syncwarp();
    #pragma unroll
    for(u32 m=0;m<8;++m)base[b2+4*m]=v[m];
    __syncwarp();
    {const uint4* p4=reinterpret_cast<const uint4*>(base+8*lane+4*(lane>>2));const uint4 x0=p4[0],x1=p4[1];
     v[0]=x0.x;v[1]=x0.y;v[2]=x0.z;v[3]=x0.w;v[4]=x1.x;v[5]=x1.y;v[6]=x1.z;v[7]=x1.w;}
    #pragma unroll
    for(u32 k=0;k<4;++k){const u32 lo=(k&1)+4*(k>>1);const u64 w=st[252u+(lo&1)];
        const u32 x=v[lo],y=v[lo+2];v[lo]=add(x,y,p);v[lo+2]=mul_shoup(sub(x,y,p),w,p);}
    {const u64 w=st[254u];
        #pragma unroll
        for(u32 lo=0;lo<8;lo+=2){const u32 x=v[lo],y=v[lo+1];v[lo]=add(x,y,p);v[lo+1]=mul_shoup(sub(x,y,p),w,p);}}
    if(flags&1u){
        #pragma unroll
        for(u32 m=0;m<8;++m)v[m]=mul_shoup(v[m],CS[m*33u+lane],p);
    }else{
        const ulonglong2* c2=reinterpret_cast<const ulonglong2*>(ct+8*lane);
        #pragma unroll
        for(u32 m2=0;m2<4;++m2){const ulonglong2 w=__ldg(c2+m2);
            v[2*m2]=mul_shoup(v[2*m2],u64(w.x),p);v[2*m2+1]=mul_shoup(v[2*m2+1],u64(w.y),p);}
    }
    uint4* o=reinterpret_cast<uint4*>((op?B:A)+u64(r*8+warp)*N+lane*2048u+c*8u);
    o[0]=make_uint4(v[0],v[1],v[2],v[3]);o[1]=make_uint4(v[4],v[5],v[6],v[7]);
}
#endif // bpo_p1b
// Persistent variant: G blocks loop over the 512 (column,operand) items, item = b + G*t.
// Blocks b >= G/2 (normally the second block resident on each SM) start with the first half
// of their first item and finish its second half last, so the two blocks of an SM are out of
// phase and one block's input staging overlaps the other's compute.
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256,BPO_TC_MINB) bpo_first_tcp(const u32* a,const u32* b,u32* A,u32* B,
    const u64* twist,const u64* stage_tw,const u32* __restrict__ bpt,u32 N,u32 logR,u32 r0,u32 r1,u32 G,u32 items){
    __shared__ __align__(16) u32 SG[BPO_TC_SMEM];
    __shared__ u64 CT[4][264];
    const u32 bb=blockIdx.x,u0=r0/4,u1=r1/4,um=(u0+u1)/2;
    const bool lag=bb>=G/2&&u1-u0>=2;
    #pragma unroll 1
    for(u32 it=bb,t=0;it<items;it+=G,++t){
        const u32 c=it%256u,op=it/256u;
        if(lag&&t==0)bpo_tc_item<false>(a,b,A,B,twist,stage_tw,bpt,N,logR,u0,um,c,op,SG,CT);
        else bpo_tc_item<false>(a,b,A,B,twist,stage_tw,bpt,N,logR,u0,u1,c,op,SG,CT);
    }
    if(lag&&bb<items){const u32 c=bb%256u,op=bb/256u;bpo_tc_item<false>(a,b,A,B,twist,stage_tw,bpt,N,logR,um,u1,c,op,SG,CT);}
}
#endif // bpo_first_tcp
#endif
#endif
