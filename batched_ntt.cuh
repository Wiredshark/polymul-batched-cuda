// Register-resident middle step for N=65536, R=S=256, J=8, 256 threads.
// Same exact arithmetic as weighted_middle (DIF over S, DIF over J, pointwise,
// DIT over J, DIT over S, cross twiddle); only the data placement differs:
// warp w owns digit w of A and B with element e=lane+32m in register m.
// Three block barriers instead of ~20; S stages >=64 are pure register ops.
__device__ __forceinline__ u64 bm_root(const u64* t,u32 exponent,u32 N,u32 p){return weighted_root_pack(t,exponent,N,p);}
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256) batched_middle(const u32* A,const u32* B,u32* C,
 const u32* primes,const u64* ratios,const u64* twist,const u64* untwist,
 u32 N,u32 R,u32 S,u32 logR,u32 J) {
    __shared__ u32 sa[8*256],sb[8*256];
    const u32 prime=blockIdx.y,p=primes[prime],slot=blockIdx.x;
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const u32 k1=__brev(slot)>>(32-logR);
    const u64 mu=ratios[prime];
    const u64* tw=twist+(u64)prime*weighted_root_stride(N);
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    const u64 base=(u64)prime*8*N+(u64)slot*256;
    // ---- phase 1: S-axis DIF of digit `warp` for A and B, in registers ----
    u32 a[8],b[8];
    {
        const u64 off=base+(u64)warp*N+lane;
        #pragma unroll
        for(u32 m=0;m<8;++m){a[m]=A[off+32*m];b[m]=B[off+32*m];}
    }
    // len=256: m <-> m+4, j=lane+32m
    #pragma unroll
    for(u32 m=0;m<4;++m){
        const u64 w=bm_root(tw,2u*(lane+32u*m)*(N/256u),N,p);
        u32 x=a[m],y=a[m+4];a[m]=add(x,y,p);a[m+4]=mul_shoup(sub(x,y,p),w,p);
        x=b[m];y=b[m+4];b[m]=add(x,y,p);b[m+4]=mul_shoup(sub(x,y,p),w,p);
    }
    // len=128: m <-> m+2 for m in {0,1,4,5}, j=lane+32(m&1)
    #pragma unroll
    for(u32 q=0;q<2;++q){
        const u64 w=bm_root(tw,2u*(lane+32u*q)*(N/128u),N,p);
        #pragma unroll
        for(u32 h=0;h<2;++h){const u32 m=q+4*h;
            u32 x=a[m],y=a[m+2];a[m]=add(x,y,p);a[m+2]=mul_shoup(sub(x,y,p),w,p);
            x=b[m];y=b[m+2];b[m]=add(x,y,p);b[m+2]=mul_shoup(sub(x,y,p),w,p);}
    }
    // len=64: m <-> m+1 (even m), j=lane
    {
        const u64 w=bm_root(tw,2u*lane*(N/64u),N,p);
        #pragma unroll
        for(u32 m=0;m<8;m+=2){
            u32 x=a[m],y=a[m+1];a[m]=add(x,y,p);a[m+1]=mul_shoup(sub(x,y,p),w,p);
            x=b[m];y=b[m+1];b[m]=add(x,y,p);b[m+1]=mul_shoup(sub(x,y,p),w,p);
        }
    }
    // len=32..2 across lanes
    #pragma unroll
    for(u32 len=32;len>=2;len>>=1){
        const u32 half=len/2,j=lane&(half-1);const bool upper=(lane&half)!=0;
        const u64 w=bm_root(tw,2u*j*(N/len),N,p);
        #pragma unroll
        for(u32 m=0;m<8;++m){
            const u32 x=a[m],y=__shfl_xor_sync(0xffffffffu,x,half);
            a[m]=upper?mul_shoup(sub(y,x,p),w,p):add(x,y,p);
            const u32 u=b[m],v=__shfl_xor_sync(0xffffffffu,u,half);
            b[m]=upper?mul_shoup(sub(v,u,p),w,p):add(u,v,p);
        }
    }
    #pragma unroll
    for(u32 m=0;m<8;++m){sa[warp*256+lane+32*m]=a[m];sb[warp*256+lane+32*m]=b[m];}
    __syncthreads();
    // ---- phase 2: J-axis per column (thread = column), pointwise, inverse J ----
    {
        const u32 c=threadIdx.x;
        u32 x8[8],y8[8];
        #pragma unroll
        for(u32 d=0;d<8;++d){x8[d]=sa[d*256+c];y8[d]=sb[d*256+c];}
        #pragma unroll
        for(u32 len=8;len>=2;len>>=1){
            const u32 half=len/2;
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=len){
                #pragma unroll
                for(u32 j=0;j<half;++j){
                    const u64 w=bm_root(tw,2u*j*(N/len),N,p);
                    u32 x=x8[d0+j],y=x8[d0+j+half];x8[d0+j]=add(x,y,p);x8[d0+j+half]=mul_shoup(sub(x,y,p),w,p);
                    x=y8[d0+j];y=y8[d0+j+half];y8[d0+j]=add(x,y,p);y8[d0+j+half]=mul_shoup(sub(x,y,p),w,p);
                }
            }
        }
        #pragma unroll
        for(u32 d=0;d<8;++d)x8[d]=mul(x8[d],y8[d],p,mu);
        #pragma unroll
        for(u32 len=2;len<=8;len<<=1){
            const u32 half=len/2;
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=len){
                #pragma unroll
                for(u32 j=0;j<half;++j){
                    const u32 x=x8[d0+j],y=mul_shoup(x8[d0+j+half],bm_root(itw,2u*j*(N/len),N,p),p);
                    x8[d0+j]=add(x,y,p);x8[d0+j+half]=sub(x,y,p);
                }
            }
        }
        __syncthreads();  // every phase-2 read of sa is done before it is overwritten
        #pragma unroll
        for(u32 d=0;d<8;++d)sa[d*256+c]=x8[d];
    }
    __syncthreads();
    // ---- phase 3: S-axis DIT of digit `warp`, twiddle, store ----
    #pragma unroll
    for(u32 m=0;m<8;++m)a[m]=sa[warp*256+lane+32*m];
    #pragma unroll
    for(u32 len=2;len<=32;len<<=1){
        const u32 half=len/2,j=lane&(half-1);const bool upper=(lane&half)!=0;
        const u64 w=bm_root(itw,2u*j*(N/len),N,p);
        #pragma unroll
        for(u32 m=0;m<8;++m){
            const u32 x=upper?mul_shoup(a[m],w,p):a[m];
            const u32 y=__shfl_xor_sync(0xffffffffu,x,half);
            a[m]=upper?sub(y,x,p):add(x,y,p);
        }
    }
    {   // len=64: m <-> m+1, j=lane
        const u64 w=bm_root(itw,2u*lane*(N/64u),N,p);
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=a[m],y=mul_shoup(a[m+1],w,p);a[m]=add(x,y,p);a[m+1]=sub(x,y,p);}
    }
    #pragma unroll
    for(u32 q=0;q<2;++q){   // len=128
        const u64 w=bm_root(itw,2u*(lane+32u*q)*(N/128u),N,p);
        #pragma unroll
        for(u32 h=0;h<2;++h){const u32 m=q+4*h;const u32 x=a[m],y=mul_shoup(a[m+2],w,p);a[m]=add(x,y,p);a[m+2]=sub(x,y,p);}
    }
    #pragma unroll
    for(u32 m=0;m<4;++m){   // len=256
        const u64 w=bm_root(itw,2u*(lane+32u*m)*(N/256u),N,p);
        const u32 x=a[m],y=mul_shoup(a[m+4],w,p);a[m]=add(x,y,p);a[m+4]=sub(x,y,p);
    }
    const u64 off=base+(u64)warp*N+lane;
    #pragma unroll
    for(u32 m=0;m<8;++m){
        const u32 col=lane+32*m;
#ifdef TW_PACKED
        C[off+32*m]=mul_shoup(a[m],bm_root(itw,2u*col*k1,N,p),p);
#else
        C[off+32*m]=weighted_root_action(a[m],itw,2u*col*k1,N,p);
#endif
    }
}
#endif // batched_middle
