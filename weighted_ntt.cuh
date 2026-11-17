// XW32_BEGIN_ACTUAL_HELPERS
// One warp owns independent 32-row tails. STRIDE=8: eight columns and four
// row lanes, with eight register values/lane. STRIDE=1: 32 row lanes and
// one register value/lane. Every call has all 32 active lanes; no shared
// location is used for exchange inside these functions.
__device__ __forceinline__ u32 xw32_root_index(u32 exponent) {
#ifdef XW32_TEST_WRONG_ROOT
    return exponent^2u; // Test-only arithmetic activation fault; never promoted.
#else
    return exponent;
#endif
}
template<unsigned STRIDE>
__device__ __forceinline__ void xw32_dif_pair(u32 (&a)[STRIDE],u32 (&b)[STRIDE],
 const u64* roots,u32 N,u32 p,u32 row_lane) {
    const unsigned ROWS=32/STRIDE;
    // The long part of this 32-row tail is wholly private to each lane.
    #pragma unroll
    for(unsigned length=32;length>ROWS;length>>=1) {
        const unsigned h=length/(2*ROWS),step=N/length;
        #pragma unroll
        for(unsigned z=0;z<STRIDE/2;++z) {
            const unsigned j=z&(h-1),i=((z&~(h-1))<<1)+j;
            const u64 w=weighted_root_pack(roots,xw32_root_index(2*(row_lane+ROWS*j)*step),N,p);
            u32 x=a[i],y=a[i+h];a[i]=add(x,y,p);a[i+h]=mul_shoup(sub(x,y,p),w,p);
            x=b[i];y=b[i+h];b[i]=add(x,y,p);b[i+h]=mul_shoup(sub(x,y,p),w,p);
        }
    }
    #pragma unroll
    for(unsigned length=ROWS;length>=2;length>>=1) {
        const unsigned half=length/2,step=N/length,j=row_lane&(half-1);
        const bool upper=(row_lane&half)!=0;
        const u64 w=weighted_root_pack(roots,xw32_root_index(2*j*step),N,p);
        #pragma unroll
        for(unsigned z=0;z<STRIDE;++z) {
            const u32 x=a[z],y=__shfl_xor_sync(0xffffffffu,x,half*STRIDE);
            a[z]=upper?mul_shoup(sub(y,x,p),w,p):add(x,y,p);
            const u32 u=b[z],v=__shfl_xor_sync(0xffffffffu,u,half*STRIDE);
            b[z]=upper?mul_shoup(sub(v,u,p),w,p):add(u,v,p);
        }
    }
}
template<unsigned STRIDE>
__device__ __forceinline__ void xw32_dit(u32 (&a)[STRIDE],const u64* roots,
 u32 N,u32 p,u32 row_lane) {
    const unsigned ROWS=32/STRIDE;
    #pragma unroll
    for(unsigned length=2;length<=ROWS;length<<=1) {
        const unsigned half=length/2,step=N/length,j=row_lane&(half-1);
        const bool upper=(row_lane&half)!=0;
        const u64 w=weighted_root_pack(roots,xw32_root_index(2*j*step),N,p);
        #pragma unroll
        for(unsigned z=0;z<STRIDE;++z) {
            const u32 x=upper?mul_shoup(a[z],w,p):a[z];
            const u32 y=__shfl_xor_sync(0xffffffffu,x,half*STRIDE);
            a[z]=upper?sub(y,x,p):add(x,y,p);
        }
    }
    #pragma unroll
    for(unsigned length=2*ROWS;length<=32;length<<=1) {
        const unsigned h=length/(2*ROWS),step=N/length;
        #pragma unroll
        for(unsigned z=0;z<STRIDE/2;++z) {
            const unsigned j=z&(h-1),i=((z&~(h-1))<<1)+j;
            const u64 w=weighted_root_pack(roots,xw32_root_index(2*(row_lane+ROWS*j)*step),N,p);
            const u32 x=a[i],y=mul_shoup(a[i+h],w,p);
            a[i]=add(x,y,p);a[i+h]=sub(x,y,p);
        }
    }
}
// XW32_END_ACTUAL_HELPERS

// These shared-memory boundaries are deliberately outside the helpers.
// Every warp completes all loads/exchanges/stores before the unconditional
// caller block barrier permits another tile or transform phase to consume it.
__device__ __forceinline__ void xw32_outer_dif(u32* sa,u32* sb,
 const u64* roots,u32 N,u32 p) {
    for(u32 t=threadIdx.x;t<256;t+=128) {
        const u32 col=t&7u,row_lane=(t>>3)&3u,group=t>>5;
        const u32 base=(32*group+row_lane)*8+col;
        u32 a[8],b[8];
        #pragma unroll
        for(unsigned z=0;z<8;++z){a[z]=sa[base+32*z];b[z]=sb[base+32*z];}
        xw32_dif_pair<8>(a,b,roots,N,p,row_lane);
        #pragma unroll
        for(unsigned z=0;z<8;++z){sa[base+32*z]=a[z];sb[base+32*z]=b[z];}
    }
}
__device__ __forceinline__ void xw32_outer_dit(u32* sc,const u64* roots,u32 N,u32 p) {
    for(u32 t=threadIdx.x;t<256;t+=128) {
        const u32 col=t&7u,row_lane=(t>>3)&3u,group=t>>5;
        const u32 base=(32*group+row_lane)*8+col;
        u32 a[8];
        #pragma unroll
        for(unsigned z=0;z<8;++z)a[z]=sc[base+32*z];
        xw32_dit<8>(a,roots,N,p,row_lane);
        #pragma unroll
        for(unsigned z=0;z<8;++z)sc[base+32*z]=a[z];
    }
}
__device__ __forceinline__ void xw32_middle_dif(u32* sa,u32* sb,
 const u64* roots,u32 N,u32 p) {
    #pragma unroll
    for(unsigned digit=0;digit<8;++digit) {
        const u32 at=digit*256+threadIdx.x;
        u32 a[1]={sa[at]},b[1]={sb[at]};
        xw32_dif_pair<1>(a,b,roots,N,p,threadIdx.x&31u);
        sa[at]=a[0];sb[at]=b[0];
    }
}
__device__ __forceinline__ void xw32_middle_dit(u32* sa,const u64* roots,u32 N,u32 p) {
    #pragma unroll
    for(unsigned digit=0;digit<8;++digit) {
        const u32 at=digit*256+threadIdx.x;
        u32 a[1]={sa[at]};
        xw32_dit<1>(a,roots,N,p,threadIdx.x&31u);
        sa[at]=a[0];
    }
}

// Exact weighted bivariate transform. Host preconditions: N=R*S, J,R,S,G
// powers of two, J divides N, G divides S, and dynamic shared memory fits.
// Global layout is [prime][digit][coefficient]; roots are [prime][coefficient].
// Inputs already include digit weights u_d and the negacyclic twist psi^i.
// The final normalization table is [prime][digit][coefficient] and contains
// (N*J)^-1 * psi^-i * u_d^-1. This file relies on kernels.cu's exact helpers.

extern "C" __global__ void weighted_forward(u32* A,u32* B,const u32* primes,
 const u64* ratios,const u64* twist,u32 N,u32 R,u32 S,u32 G,u32 logR,u32 J) {
    extern __shared__ u32 workspace[];
    u32* sa=workspace;u32* sb=workspace+R*G;
    u32 batch=blockIdx.y,prime=batch/J,p=primes[prime],first=blockIdx.x*G;
    u32 logG=__ffs((int)G)-1;
    u64 base=(u64)batch*N,root_base=(u64)prime*weighted_root_stride(N);
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x){
        u32 row=t>>logG,column=first+(t&(G-1));
        sa[t]=A[base+row*S+column];sb[t]=B[base+row*S+column];
    }
    __syncthreads();
    const bool xw32=(N==65536&&R==256&&S==256&&G==8&&J==8&&blockDim.x==128);
    for(u32 length=R,step=S;length>=(xw32?64u:2u);length/=2,step*=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<(R/2)*G;t+=blockDim.x){
            u32 col=t&(G-1),butterfly=t>>logG,j=butterfly&(half-1);
            u32 row=((butterfly&~(half-1))<<1)+j,i=row*G+col,h=half*G;
            u64 w=weighted_root_pack(twist+root_base,2*j*step,N,p);
            u32 x=sa[i],y=sa[i+h];sa[i]=add(x,y,p);sa[i+h]=mul_shoup(sub(x,y,p),w,p);
            x=sb[i];y=sb[i+h];sb[i]=add(x,y,p);sb[i+h]=mul_shoup(sub(x,y,p),w,p);
        }
        __syncthreads();
    }
    if(xw32){
        xw32_outer_dif(sa,sb,twist+root_base,N,p);
        __syncthreads();
    }
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x){
        u32 slot=t>>logG,column=first+(t&(G-1)),k1=logR?__brev(slot)>>(32-logR):0;
        u64 w=weighted_root_pack(twist+root_base,2*column*k1,N,p);
        A[base+slot*S+column]=mul_shoup(sa[t],w,p);
        B[base+slot*S+column]=mul_shoup(sb[t],w,p);
    }
}

// One block owns all J digits for one four-step row slot and one prime.
// x-DIF produces bit-reversed x frequencies within each digit; J-DIF then
// bit-reverses digit frequencies. Both inputs share that ordering. Matching
// J-DIT and x-DIT remove the two permutations without any explicit reorder.
extern "C" __global__ void weighted_middle(const u32* A,const u32* B,u32* C,
 const u32* primes,const u64* ratios,const u64* twist,const u64* untwist,
 u32 N,u32 R,u32 S,u32 logR,u32 J) {
    extern __shared__ u32 workspace[];
    u32* sa=workspace;u32* sb=workspace+J*S;
    u32 prime=blockIdx.y,p=primes[prime],slot=blockIdx.x,logS=__ffs((int)S)-1;
    u32 k1=logR?__brev(slot)>>(32-logR):0;
    u64 root_base=(u64)prime*weighted_root_stride(N),base=(u64)prime*J*N+slot*S,mu=ratios[prime];
    for(u32 t=threadIdx.x;t<J*S;t+=blockDim.x){
        u64 offset=base+(u64)(t>>logS)*N+(t&(S-1));
        sa[t]=A[offset];sb[t]=B[offset];
    }
    __syncthreads();
#ifndef MX_SKIP_SFWD
    // Independently transform the S contiguous columns of every digit.
    const bool xw32=(N==65536&&R==256&&S==256&&J==8&&blockDim.x==256);
    for(u32 length=S,step=R;length>=(xw32?64u:2u);length/=2,step*=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<J*(S/2);t+=blockDim.x){
            u32 j=t&(half-1),i=((t&~(half-1))<<1)+j;
            u64 w=weighted_root_pack(twist+root_base,2*j*step,N,p);
            u32 x=sa[i],y=sa[i+half];sa[i]=add(x,y,p);sa[i+half]=mul_shoup(sub(x,y,p),w,p);
            x=sb[i];y=sb[i+half];sb[i]=add(x,y,p);sb[i+half]=mul_shoup(sub(x,y,p),w,p);
        }
        __syncthreads();
    }
    if(xw32){
        xw32_middle_dif(sa,sb,twist+root_base,N,p);
        __syncthreads();
    }
#endif
#ifndef MX_SKIP_J
    // Transform the digit axis separately for each x-frequency slot.
    for(u32 length=J,step=N/J;length>=2;length/=2,step*=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<(J/2)*S;t+=blockDim.x){
            u32 col=t&(S-1),butterfly=t>>logS,j=butterfly&(half-1);
            u32 digit=((butterfly&~(half-1))<<1)+j,i=digit*S+col,h=half*S;
            u64 w=weighted_root_pack(twist+root_base,2*j*step,N,p);
            u32 x=sa[i],y=sa[i+h];sa[i]=add(x,y,p);sa[i+h]=mul_shoup(sub(x,y,p),w,p);
            x=sb[i];y=sb[i+h];sb[i]=add(x,y,p);sb[i+h]=mul_shoup(sub(x,y,p),w,p);
        }
        __syncthreads();
    }
#endif
    for(u32 t=threadIdx.x;t<J*S;t+=blockDim.x)sa[t]=mul(sa[t],sb[t],p,mu);
#ifndef MX_SKIP_J
    __syncthreads();
    for(u32 length=2,step=N/2;length<=J;length*=2,step/=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<(J/2)*S;t+=blockDim.x){
            u32 col=t&(S-1),butterfly=t>>logS,j=butterfly&(half-1);
            u32 digit=((butterfly&~(half-1))<<1)+j,i=digit*S+col,h=half*S;
            u32 x=sa[i],y=mul_shoup(sa[i+h],weighted_root_pack(untwist+root_base,2*j*step,N,p),p);
            sa[i]=add(x,y,p);sa[i+h]=sub(x,y,p);
        }
        __syncthreads();
    }
#endif
#ifndef MX_SKIP_SINV
    if(xw32){
        xw32_middle_dit(sa,untwist+root_base,N,p);
        __syncthreads();
    }
    for(u32 length=xw32?64u:2u,step=xw32?N/64:N/2;length<=S;length*=2,step/=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<J*(S/2);t+=blockDim.x){
            u32 j=t&(half-1),i=((t&~(half-1))<<1)+j;
            u32 x=sa[i],y=mul_shoup(sa[i+half],weighted_root_pack(untwist+root_base,2*j*step,N,p),p);
            sa[i]=add(x,y,p);sa[i+half]=sub(x,y,p);
        }
        __syncthreads();
    }
#endif
    for(u32 t=threadIdx.x;t<J*S;t+=blockDim.x){
        u32 col=t&(S-1);u64 offset=base+(u64)(t>>logS)*N+col;
        u64 w=weighted_root_pack(untwist+root_base,2*col*k1,N,p);
        C[offset]=mul_shoup(sa[t],w,p);
    }
}

extern "C" __global__ void weighted_inverse(u32* C,const u32* primes,
 const u64* ratios,const u64* untwist,const u64* normalized_untwist,
 u32 N,u32 R,u32 S,u32 G,u32 J) {
    extern __shared__ u32 sc[];
    u32 batch=blockIdx.y,prime=batch/J,p=primes[prime],first=blockIdx.x*G;
    u32 logG=__ffs((int)G)-1;
    u64 base=(u64)batch*N,root_base=(u64)prime*weighted_root_stride(N);
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x)sc[t]=C[base+(t>>logG)*S+first+(t&(G-1))];
    __syncthreads();
    const bool xw32=(N==65536&&R==256&&S==256&&G==8&&J==8&&blockDim.x==128);
    if(xw32){
        xw32_outer_dit(sc,untwist+root_base,N,p);
        __syncthreads();
    }
    for(u32 length=xw32?64u:2u,step=xw32?N/64:N/2;length<=R;length*=2,step/=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<(R/2)*G;t+=blockDim.x){
            u32 col=t&(G-1),butterfly=t>>logG,j=butterfly&(half-1);
            u32 row=((butterfly&~(half-1))<<1)+j,i=row*G+col,h=half*G;
            u32 x=sc[i],y=mul_shoup(sc[i+h],weighted_root_pack(untwist+root_base,2*j*step,N,p),p);
            sc[i]=add(x,y,p);sc[i+h]=sub(x,y,p);
        }
        __syncthreads();
    }
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x){
        u32 i=(t>>logG)*S+first+(t&(G-1));
        // Exact inverse factorization shares the existing prime root array.
        C[base+i]=mul_shoup(weighted_root_action(sc[t],untwist+root_base,i,N,p),normalized_untwist[batch],p);
    }
}
