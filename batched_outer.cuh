// Outer (row-axis) passes, N=65536, R=S=256, G=8 columns per block, 256 threads:
// warp w owns column first+w; row e=lane+32m lives in register m. Same exact
// arithmetic as weighted_forward / weighted_inverse. Shared tiles padded to 9.
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256) batched_forward(u32* A,u32* B,const u32* primes,
 const u64* twist,u32 N,u32 logR) {
    __shared__ u32 ta[256*9],tb[256*9];
    const u32 batch=blockIdx.y,prime=batch>>3,p=primes[prime],first=blockIdx.x*8;
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const u64 base=(u64)batch*N;
    const u64* tw=twist+(u64)prime*weighted_root_stride(N);
    for(u32 t=threadIdx.x;t<2048;t+=256){
        const u32 row=t>>3,col=t&7u;const u64 g=base+(u64)row*256+first+col;
        ta[row*9+col]=A[g];tb[row*9+col]=B[g];
    }
    __syncthreads();
    u32 a[8],b[8];
    #pragma unroll
    for(u32 m=0;m<8;++m){a[m]=ta[(lane+32*m)*9+warp];b[m]=tb[(lane+32*m)*9+warp];}
    #pragma unroll
    for(u32 m=0;m<4;++m){
        const u64 w=bm_root(tw,2u*(lane+32u*m)*(N/256u),N,p);
        u32 x=a[m],y=a[m+4];a[m]=add(x,y,p);a[m+4]=mul_shoup(sub(x,y,p),w,p);
        x=b[m];y=b[m+4];b[m]=add(x,y,p);b[m+4]=mul_shoup(sub(x,y,p),w,p);
    }
    #pragma unroll
    for(u32 q=0;q<2;++q){
        const u64 w=bm_root(tw,2u*(lane+32u*q)*(N/128u),N,p);
        #pragma unroll
        for(u32 h=0;h<2;++h){const u32 m=q+4*h;
            u32 x=a[m],y=a[m+2];a[m]=add(x,y,p);a[m+2]=mul_shoup(sub(x,y,p),w,p);
            x=b[m];y=b[m+2];b[m]=add(x,y,p);b[m+2]=mul_shoup(sub(x,y,p),w,p);}
    }
    {
        const u64 w=bm_root(tw,2u*lane*(N/64u),N,p);
        #pragma unroll
        for(u32 m=0;m<8;m+=2){
            u32 x=a[m],y=a[m+1];a[m]=add(x,y,p);a[m+1]=mul_shoup(sub(x,y,p),w,p);
            x=b[m];y=b[m+1];b[m]=add(x,y,p);b[m+1]=mul_shoup(sub(x,y,p),w,p);
        }
    }
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
    const u32 column=first+warp;
    #pragma unroll
    for(u32 m=0;m<8;++m){
        const u32 slot=lane+32*m,k1=__brev(slot)>>(32-logR);
#ifdef TW_PACKED
        const u64 w=bm_root(tw,2u*column*k1,N,p);
        a[m]=mul_shoup(a[m],w,p);b[m]=mul_shoup(b[m],w,p);
#else   // two table factors applied directly: no Shoup-quotient reconstruction
        a[m]=weighted_root_action(a[m],tw,2u*column*k1,N,p);b[m]=weighted_root_action(b[m],tw,2u*column*k1,N,p);
#endif
    }
    __syncthreads();
    #pragma unroll
    for(u32 m=0;m<8;++m){ta[(lane+32*m)*9+warp]=a[m];tb[(lane+32*m)*9+warp]=b[m];}
    __syncthreads();
    for(u32 t=threadIdx.x;t<2048;t+=256){
        const u32 row=t>>3,col=t&7u;const u64 g=base+(u64)row*256+first+col;
        A[g]=ta[row*9+col];B[g]=tb[row*9+col];
    }
}
#endif // batched_forward

#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256) batched_inverse(u32* C,const u32* primes,
 const u64* untwist,const u64* normalized_untwist,u32 N) {
    __shared__ u32 tc[256*9];
    const u32 batch=blockIdx.y,prime=batch>>3,p=primes[prime],first=blockIdx.x*8;
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5;
    const u64 base=(u64)batch*N;
    const u64* itw=untwist+(u64)prime*weighted_root_stride(N);
    for(u32 t=threadIdx.x;t<2048;t+=256){
        const u32 row=t>>3,col=t&7u;tc[row*9+col]=C[base+(u64)row*256+first+col];
    }
    __syncthreads();
    u32 a[8];
    #pragma unroll
    for(u32 m=0;m<8;++m)a[m]=tc[(lane+32*m)*9+warp];
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
    {
        const u64 w=bm_root(itw,2u*lane*(N/64u),N,p);
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=a[m],y=mul_shoup(a[m+1],w,p);a[m]=add(x,y,p);a[m+1]=sub(x,y,p);}
    }
    #pragma unroll
    for(u32 q=0;q<2;++q){
        const u64 w=bm_root(itw,2u*(lane+32u*q)*(N/128u),N,p);
        #pragma unroll
        for(u32 h=0;h<2;++h){const u32 m=q+4*h;const u32 x=a[m],y=mul_shoup(a[m+2],w,p);a[m]=add(x,y,p);a[m+2]=sub(x,y,p);}
    }
    #pragma unroll
    for(u32 m=0;m<4;++m){
        const u64 w=bm_root(itw,2u*(lane+32u*m)*(N/256u),N,p);
        const u32 x=a[m],y=mul_shoup(a[m+4],w,p);a[m]=add(x,y,p);a[m+4]=sub(x,y,p);
    }
    const u32 column=first+warp;const u64 nrm=normalized_untwist[batch];
    #pragma unroll
    for(u32 m=0;m<8;++m){
        const u32 i=(lane+32*m)*256+column;
        a[m]=mul_shoup(weighted_root_action(a[m],itw,i,N,p),nrm,p);
    }
    __syncthreads();
    #pragma unroll
    for(u32 m=0;m<8;++m)tc[(lane+32*m)*9+warp]=a[m];
    __syncthreads();
    for(u32 t=threadIdx.x;t<2048;t+=256){
        const u32 row=t>>3,col=t&7u;C[base+(u64)row*256+first+col]=tc[row*9+col];
    }
}
#endif // batched_inverse
