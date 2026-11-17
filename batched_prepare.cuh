// Batched: transpose + weighted digit extraction fused. One thread owns one
// coefficient of one product: its 55 row-major limbs are read once (L1 keeps
// the 220-byte row hot), each 16-bit digit term is extracted once and reused
// for all primes. Arithmetic per (prime,digit) output is identical to
// weighted_prepare. grid = (N/128, products), block = 128.
#ifndef BATCHED_ONLY
extern "C" __global__ void batched_prepare_fused(
    const u32* a,const u32* b,u32* A,u32* B,
    const u32* primes,const u64* ratios,const u64* prepared_twist,const u64* twist,
    const u32* weights,const u32* positions,u32 N,u32 L,u32 W,u32 digit_terms,u32 K,
    u64 in_stride,u64 ws_stride){
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N)return;
    const u64 z=blockIdx.y;
    const u32* ar=a+z*in_stride+size_t(i)*L;const u32* br=b+z*in_stride+size_t(i)*L;
    u32* Az=A+z*ws_stride;u32* Bz=B+z*ws_stride;
    const u32 stride=weighted_root_stride(N);
    for(u32 j=0;j<8;++j){
        const u32 begin=positions[j],end=j==7?W:positions[j+1];
        u32 ta[16],tb[16];
        #pragma unroll
        for(u32 k=0;k<16;++k){
            const u32 bit=begin+16*k;ta[k]=0;tb[k]=0;
            if(k<digit_terms&&bit<end){
                const u32 word=bit/32,shift=bit%32,count=min(16u,end-bit),mask=(1u<<count)-1;
                const u32 al=__ldg(ar+word),bl=__ldg(br+word);
                const u32 ah=word+1<L?__ldg(ar+word+1):0,bh=word+1<L?__ldg(br+word+1):0;
                ta[k]=__funnelshift_r(al,ah,shift)&mask;tb[k]=__funnelshift_r(bl,bh,shift)&mask;
            }
        }
        for(u32 r=0;r<K;++r){
            const u32 p=primes[r];const u64 mu=ratios[r];const u32* wr=weights+size_t(r)*digit_terms;
            u64 sa=0,sb=0;
            #pragma unroll
            for(u32 k=0;k<16;++k)if(k<digit_terms){const u32 wt=wr[k];sa+=u64(ta[k])*wt;sb+=u64(tb[k])*wt;}
            const u32 batch=r*8+j;
            const u64* rootset=twist+size_t(r)*stride;const u64 factor=prepared_twist[batch];
            Az[size_t(batch)*N+i]=mul_shoup(weighted_root_action(reduce(sa,p,mu),rootset,i,N,p),factor,p);
            Bz[size_t(batch)*N+i]=mul_shoup(weighted_root_action(reduce(sb,p,mu),rootset,i,N,p),factor,p);
        }
    }
}

#endif // BATCHED_ONLY

// Compile-time specialization (NVRTC macros from the point): row kept in
// registers, digit terms and weights are immediates, psi^i loaded once per
// prime. Same modular values as weighted_prepare (exact, canonical).
#ifdef BPR_DT
__constant__ u32 bpr_pr[BCRT_K]={BCRT_PRIMES};__constant__ u64 bpr_mu[BCRT_K]={BCRT_MUS};
__constant__ u32 bpr_wt[BCRT_K*BPR_DT]={BPR_WEIGHTS};__constant__ u64 bpr_fa[BCRT_K*8]={BPR_FACT};
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(128) batched_prepare2(
    const u32* a,const u32* b,u32* A,u32* B,const u64* twist,u32 N,u64 in_stride,u64 ws_stride,u32 r0,u32 r1){
    constexpr u32 K=BCRT_K,W=BCRT_W,L=BCRT_L,DT=BPR_DT;
    constexpr u32 POS[8]={BCRT_POS};
    __shared__ u32 rows[128*L];
    const u32 i0=blockIdx.x*128,i=i0+threadIdx.x;
    const u32 stride=weighted_root_stride(N);
    #pragma unroll 1
    for(u32 op=0;op<2;++op){
        const u32* src=(op?b:a)+blockIdx.y*in_stride+u64(i0)*L;
        u32* dst=(op?B:A)+blockIdx.y*ws_stride;
        __syncthreads();
        for(u32 x=threadIdx.x;x<128*L;x+=128)rows[x]=src[x];
        __syncthreads();
        u32 row[L];
        #pragma unroll
        for(u32 w=0;w<L;++w)row[w]=rows[threadIdx.x*L+w];
        #pragma unroll 1
        for(u32 r=r0;r<r1;++r){
            const u32 p=bpr_pr[r];const u64 mu=bpr_mu[r];
#if WROOT_COMPACT
            const u64 t1=twist[size_t(r)*stride+1+(i&127u)],t2=twist[size_t(r)*stride+1+128u+(i>>7)];
#else
            const u64 t1=twist[size_t(r)*stride+i];
#endif
            #pragma unroll
            for(u32 j=0;j<8;++j){
                constexpr u32 dummy=0;(void)dummy;
                const u32 begin=POS[j],end=j==7?W:POS[j+1];
                u64 sum=0;
                #pragma unroll
                for(u32 k=0;k<DT;++k){
                    const u32 bit=begin+16*k;
                    if(bit<end){
                        const u32 word=bit/32,shift=bit%32,count=end-bit<16?end-bit:16;
                        const u32 lo=row[word],hi=word+1<L?row[word+1]:0;
                        const u32 v=__funnelshift_r(lo,hi,shift)&((1u<<count)-1);
                        sum+=u64(v)*bpr_wt[r*DT+k];
                    }
                }
                u32 x=reduce(sum,p,mu);
                x=mul_shoup(x,t1,p);
#if WROOT_COMPACT
                x=mul_shoup(x,t2,p);
#endif
                x=mul_shoup(x,bpr_fa[r*8+j],p);
                dst[size_t(r*8+j)*N+i]=x;
            }
        }
    }
}
#endif // batched_prepare2
#endif
// prepare3: one operand per block (blockIdx.y = 0:a, 1:b); the 128-row input
// tile is fetched with all 16-byte loads issued before any shared store, so
// the whole 28 KB tile is in flight at once. Arithmetic identical to prepare2.
#ifdef BPR_DT
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(128) batched_prepare3(
    const u32* a,const u32* b,u32* A,u32* B,const u64* twist,u32 N,u64 in_stride,u64 ws_stride,u32 r0,u32 r1){
    constexpr u32 W=BCRT_W,L=BCRT_L,DT=BPR_DT;
    constexpr u32 POS[8]={BCRT_POS};
    constexpr u32 V4=128*L/4,PER=(V4+127)/128;
    static_assert((128*L)%4==0,"tile not uint4 aligned");
    __shared__ __align__(16) u32 rows[128*L];
    const u32 i0=blockIdx.x*128,i=i0+threadIdx.x;
    const u32 stride=weighted_root_stride(N);
    const u32 op=blockIdx.y;
    const uint4* src=reinterpret_cast<const uint4*>((op?b:a)+u64(i0)*L);
    // evict-first only when this is the input's last use (prime groups re-read it from L2)
    const bool single_pass=r0==0&&r1==BCRT_K;
    u32* dst=op?B:A;
    {
        uint4 v[PER];
        #pragma unroll
        for(u32 k=0;k<PER;++k){const u32 x=threadIdx.x+128*k;if(x<V4)v[k]=single_pass?__ldcs(src+x):__ldg(src+x);}
        #pragma unroll
        for(u32 k=0;k<PER;++k){const u32 x=threadIdx.x+128*k;if(x<V4)reinterpret_cast<uint4*>(rows)[x]=v[k];}
    }
    __syncthreads();
    u32 row[L];
    #pragma unroll
    for(u32 w=0;w<L;++w)row[w]=rows[threadIdx.x*L+w];
    #pragma unroll 1
    for(u32 r=r0;r<r1;++r){
        const u32 p=bpr_pr[r];const u64 mu=bpr_mu[r];
#if WROOT_COMPACT
        const u64 t1=twist[size_t(r)*stride+1+(i&127u)],t2=twist[size_t(r)*stride+1+128u+(i>>7)];
#else
        const u64 t1=twist[size_t(r)*stride+i];
#endif
        #pragma unroll
        for(u32 j=0;j<8;++j){
            const u32 begin=POS[j],end=j==7?W:POS[j+1];
            u64 sum=0;
            #pragma unroll
            for(u32 k=0;k<DT;++k){
                const u32 bit=begin+16*k;
                if(bit<end){
                    const u32 word=bit/32,shift=bit%32,count=end-bit<16?end-bit:16;
                    const u32 lo=row[word],hi=word+1<L?row[word+1]:0;
                    const u32 v=__funnelshift_r(lo,hi,shift)&((1u<<count)-1);
                    sum+=u64(v)*bpr_wt[r*DT+k];
                }
            }
            u32 x=reduce(sum,p,mu);
            x=mul_shoup(x,t1,p);
#if WROOT_COMPACT
            x=mul_shoup(x,t2,p);
#endif
            x=mul_shoup(x,bpr_fa[r*8+j],p);
            dst[size_t(r*8+j)*N+i]=x;
        }
    }
}
#endif // batched_prepare3
#endif

// Tensor-core base conversion (IMMA u8 x u8 -> s32). For digit j the residues
// of all 16 primes are one byte matrix product per 16 coefficients:
//   A[coef][m] = byte m of digit j (m < 32; bytes past the digit are zero),
//   B[m][(r,b)] = byte b of (2^(8m) * fa[r][j] mod p_r)  (fa folded in),
// so x = sum_b Y_b 2^(8b) = sum_m byte_m * (2^(8m) fa mod p) exactly
// (Y_b < 28*255^2 < 2^21, x < 2^46), then x mod p, *psi^i as prepare2.
// Tiles 2u,2u+1 hold primes 4u..4u+3: lane (g,t) gets prime 4u+t, bytes 0..3.
// Same modular value as prepare2 (canonical). grid=(N/128,2 operands), 128 thr.
#if defined(BPT_ENABLE)&&defined(BPR_DT)
__device__ __forceinline__ void pimma16832(u32 (&c)[4],u32 a0,u32 a1,u32 a2,u32 a3,u32 b0,u32 b1){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(128) batched_prepare_tc(
    const u32* a,const u32* b,u32* A,u32* B,const u64* twist,const u32* __restrict__ bpt,u32 N,u32 r0,u32 r1){
    constexpr u32 W=BCRT_W,L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    constexpr u32 V4=128*L/4,PER=(V4+127)/128;
    __shared__ __align__(16) u32 rows[128*L+8];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    const u32 i0=blockIdx.x*128;
    const u32 stride=weighted_root_stride(N);
    const u32 op=blockIdx.y;
    const uint4* src=reinterpret_cast<const uint4*>((op?b:a)+u64(i0)*L);
    // evict-first only when this is the input's last use (prime groups re-read it from L2)
    const bool single_pass=r0==0&&r1==BCRT_K;
    u32* dst=op?B:A;
    {
        uint4 v[PER];
        #pragma unroll
        for(u32 k=0;k<PER;++k){const u32 x=threadIdx.x+128*k;if(x<V4)v[k]=single_pass?__ldcs(src+x):__ldg(src+x);}
        #pragma unroll
        for(u32 k=0;k<PER;++k){const u32 x=threadIdx.x+128*k;if(x<V4)reinterpret_cast<uint4*>(rows)[x]=v[k];}
        if(threadIdx.x<8)rows[128*L+threadIdx.x]=0;
    }
    // per-lane prime constants and twist factors (prime 4u+tig, rows of both m-tiles)
    const u32 u0=r0/4,u1=r1/4;
    u32 pr[4];u64 mu[4],t2[4],t1[4][4];
    #pragma unroll
    for(u32 u=0;u<4;++u)if(u>=u0&&u<u1){
        const u32 r=4*u+tig;pr[u]=bpr_pr[r];mu[u]=bpr_mu[r];
        const u64* tw=twist+size_t(r)*stride+1;
        t2[u]=tw[128u+(i0>>7)];
        #pragma unroll
        for(u32 q=0;q<4;++q)t1[u][q]=tw[(32*warp+8*q+gid)&127u];  // q: m-tile*2+half
    }
    __syncthreads();
    #pragma unroll
    for(u32 j=0;j<8;++j){
        const uint4* bj=reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16);
        u32 bf[16];
        #pragma unroll
        for(u32 q=0;q<4;++q){const uint4 v=__ldg(bj+q);bf[4*q]=v.x;bf[4*q+1]=v.y;bf[4*q+2]=v.z;bf[4*q+3]=v.w;}
        constexpr u32 dummy=0;(void)dummy;
        const u32 wb=POS[j]/32,sh=POS[j]%32;
        #pragma unroll
        for(u32 mt=0;mt<2;++mt){
            u32 af[4];
            #pragma unroll
            for(u32 x=0;x<4;++x){
                const u32 row=32*warp+16*mt+gid+8*(x&1),t=tig+4*(x>>1);
                const u32* rp=rows+row*L+wb+t;
                const u32 v=__funnelshift_r(rp[0],rp[1],sh);
                const int width=int(POS[j+1])-int(POS[j]+32*t);
                af[x]=width>=32?v:(width<=0?0u:(v&((1u<<width)-1)));
            }
            #pragma unroll
            for(u32 u=0;u<4;++u)if(u>=u0&&u<u1){
                u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
                pimma16832(c0,af[0],af[1],af[2],af[3],bf[(2*u)*2],bf[(2*u)*2+1]);
                pimma16832(c1,af[0],af[1],af[2],af[3],bf[(2*u+1)*2],bf[(2*u+1)*2+1]);
                const u32 r=4*u+tig;
                #pragma unroll
                for(u32 hf=0;hf<2;++hf){
                    const u64 x=u64(c0[2*hf])+(u64(c0[2*hf+1])<<8)+(u64(c1[2*hf])<<16)+(u64(c1[2*hf+1])<<24);
                    u32 y=reduce(x,pr[u],mu[u]);
                    y=mul_shoup(y,t1[u][2*mt+hf],pr[u]);
                    y=mul_shoup(y,t2[u],pr[u]);
                    dst[size_t(r*8+j)*N+i0+32*warp+16*mt+8*hf+gid]=y;
                }
            }
        }
    }
}
#endif // batched_prepare_tc
#endif
