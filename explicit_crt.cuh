// Explicit CRT, with an exact fixed-point rounding proof. Requires M>4B,
// K<=128, 128<=W<=1747 and q=2^W-d, 0<d<2^32. One warp owns a coefficient.
// All arithmetic including quotient determination is integral.
__device__ __forceinline__ u32 warp_word(u32 x0,u32 x1,u32 j) {
    return __shfl_sync(0xffffffffu,j<32?x0:x1,j&31);
}
__device__ __forceinline__ u64 warp_high(u32 x0,u32 x1,u32 W) {
    u32 word=W/32,s=W%32;
    u64 a=warp_word(x0,x1,word),b=warp_word(x0,x1,word+1),c=warp_word(x0,x1,word+2);
    if(!s)return a|(b<<32);
    return (a>>s)|(b<<(32-s))|(c<<(64-s));
}
// Add an at-most-96-bit integer to the low W bits; preserve one overflow limb.
__device__ __forceinline__ void warp_fold_add(u32& x0,u32& x1,u32 W,u32 L,u32 a0,u32 a1,u32 a2,u32 lane) {
    u32 word=W/32,s=W%32;
    if(lane>word)x0=0;else if(lane==word)x0=s?(x0&((1u<<s)-1)):0;
    if(lane+32>word)x1=0;else if(lane+32==word)x1=s?(x1&((1u<<s)-1)):0;
    u64 carry=0;
    for(u32 j=0;j<L+1;++j){
        u32 addend=j==0?a0:(j==1?a1:(j==2?a2:0));
        u64 v=u64(warp_word(x0,x1,j))+addend+carry;
        if(j==lane)x0=u32(v);if(j==lane+32)x1=u32(v);carry=v>>32;
    }
}
extern "C" __global__ void reconstruct_explicit(const u32* residues,u32* output,
 const u32* primes,const u64* ratios,const u32* crt_inverse,const u32* crt_matrix,
 const u32* neg_pmodq,const u32* q,u32 N,u32 K,u32 W,u32 L,u32 d) {
    u32 lane=threadIdx.x&31,warp=threadIdx.x>>5,i=blockIdx.x*4+warp;
    if(i>=N)return;
    __shared__ u32 digits[4][128];
    u64 fixed=0;
    for(u32 r=lane;r<K;r+=32){
        u32 p=primes[r];u64 mu=ratios[r];
        u32 t=mul(residues[(u64)r*N+i],crt_inverse[r],p,mu);
        digits[warp][r]=t;
        u64 numerator=u64(t)<<32,quotient=__umul64hi(numerator,mu);
        quotient+=(numerator-quotient*p)>=p;
        fixed+=quotient;
    }
    for(u32 off=16;off;off>>=1)fixed+=__shfl_down_sync(0xffffffffu,fixed,off);
    u32 alpha=u32((__shfl_sync(0xffffffffu,fixed,0)+(u64(1)<<31))>>32);
    __syncwarp();
    u64 lo0=0,hi0=0,lo1=0,hi1=0;
    for(u32 r=0;r<K;++r){
        u32 t=digits[warp][r];
        if(lane<L){u64 v=u64(t)*crt_matrix[(u64)r*L+lane];lo0+=u32(v);hi0+=v>>32;}
        if(lane+32<L){u64 v=u64(t)*crt_matrix[(u64)r*L+lane+32];lo1+=u32(v);hi1+=v>>32;}
    }
    // Add alpha*(q-M mod q), congruent to the negative CRT correction.
    if(lane<L){u64 v=u64(alpha)*neg_pmodq[lane];lo0+=u32(v);hi0+=v>>32;}
    if(lane+32<L){u64 v=u64(alpha)*neg_pmodq[lane+32];lo1+=u32(v);hi1+=v>>32;}
    u64 prev0=__shfl_up_sync(0xffffffffu,hi0,1),prev1=__shfl_up_sync(0xffffffffu,hi1,1);
    u64 bridge=__shfl_sync(0xffffffffu,hi0,31);
    if(!lane){prev0=0;prev1=bridge;}
    u64 v0=lo0+prev0,v1=lo1+prev1;
    u32 x0=0,x1=0;u64 carry=0;
    for(u32 j=0;j<L+2;++j){
        u64 v=__shfl_sync(0xffffffffu,j<32?v0:v1,j&31)+carry;
        if(j==lane)x0=u32(v);if(j==lane+32)x1=u32(v);carry=v>>32;
    }
    u64 high=warp_high(x0,x1,W);
    u64 product0=u64(u32(high))*d,product1=(high>>32)*d+(product0>>32);
    warp_fold_add(x0,x1,W,L,u32(product0),u32(product1),u32(product1>>32),lane);
    high=warp_high(x0,x1,W); // At most one after the first fold, W>=128.
    warp_fold_add(x0,x1,W,L,u32(high*d),u32((high*d)>>32),0,lane);
    int greater=0;
    for(u32 j=L;j--;){u32 v=warp_word(x0,x1,j),qj=q[j];if(!greater&&v!=qj)greater=v>qj?1:-1;}
    if(greater>=0){u64 borrow=0;for(u32 j=0;j<L;++j){u32 v=warp_word(x0,x1,j);u64 t=u64(q[j])+borrow;
        if(j==lane)x0=u32(u64(v)-t);if(j==lane+32)x1=u32(u64(v)-t);borrow=u64(v)<t;}}
    if(lane<L)output[(u64)i*L+lane]=x0;
    if(lane+32<L)output[(u64)i*L+lane+32]=x1;
}
