// Original exact weighted-CRT reconstruction. Host validates J=8, K<=128,
// shortWords<=32, 32*shortWords>=G+1, P>4B, G=B.bit_length(), Z<2^2047 and
// (Z>>W)*d < 2^(W-2), with W>=128. No input-dependent host arithmetic.
__device__ __forceinline__ u32 wcrt_prefix(u32 g,u32 p,u32 lane,u32 seed) {
    // All callers have binary, disjoint generate/propagate and binary seed.
    // The last lane <= this lane that does not propagate determines its carry.
    const u32 generated=__ballot_sync(0xffffffffu,g!=0);
    const u32 stopped=__ballot_sync(0xffffffffu,p==0);
    const u32 eligible=stopped&(0xffffffffu>>(31u-lane));
    if(!eligible)return seed;
    return (generated>>(31u-u32(__clz(eligible))))&1u;
}
__device__ __forceinline__ void wcrt_normalize(u64 v0,u64 v1,u32& x0,u32& x1,u32 lane) {
    const u32 h0=u32(v0>>32),h1=u32(v1>>32);
    u32 prev0=__shfl_up_sync(0xffffffffu,h0,1),prev1=__shfl_up_sync(0xffffffffu,h1,1);
    u32 bridge=__shfl_sync(0xffffffffu,h0,31);
    if(!lane){prev0=0;prev1=bridge;}
    const u64 z0=u64(u32(v0))+prev0,z1=u64(u32(v1))+prev1;
    const u32 c0=wcrt_prefix(u32(z0>>32),u32(z0)==0xffffffffu,lane,0);
    bridge=__shfl_sync(0xffffffffu,c0,31);
    const u32 c1=wcrt_prefix(u32(z1>>32),u32(z1)==0xffffffffu,lane,bridge);
    prev0=__shfl_up_sync(0xffffffffu,c0,1);prev1=__shfl_up_sync(0xffffffffu,c1,1);
    if(!lane){prev0=0;prev1=bridge;}
    x0=u32(z0)+prev0;x1=u32(z1)+prev1;
}
__device__ __forceinline__ u32 wcrt_word(u32 x0,u32 x1,u32 word) {
    // All lanes execute both shuffles, including requests outside capacity.
    const u32 low=__shfl_sync(0xffffffffu,x0,word&31);
    const u32 high=__shfl_sync(0xffffffffu,x1,word&31);
    return word<32?low:(word<64?high:0);
}
__device__ __forceinline__ u32 wcrt_high_word(u32 x0,u32 x1,u32 word,u32 W) {
    const u32 offset=W/32,shift=W%32;
    const u32 a=wcrt_word(x0,x1,word+offset);
    const u32 b=wcrt_word(x0,x1,word+offset+1);
    return shift?(a>>shift)|(b<<(32-shift)):a;
}
__device__ __forceinline__ u32 wcrt_low_word(u32 value,u32 word,u32 W) {
    const u32 offset=W/32,shift=W%32;
    return word<offset?value:(word==offset&&shift?value&((1u<<shift)-1):0);
}
__device__ __forceinline__ void wcrt_fold(u32& x0,u32& x1,u32 W,u32 d,u32 lane) {
    const u32 high0=wcrt_high_word(x0,x1,lane,W);
    const u32 high1=wcrt_high_word(x0,x1,lane+32,W);
    // Each redundant digit <= (2^32-1)^2+(2^32-1) < 2^64.
    const u64 v0=u64(high0)*d+wcrt_low_word(x0,lane,W);
    const u64 v1=u64(high1)*d+wcrt_low_word(x1,lane+32,W);
    wcrt_normalize(v0,v1,x0,x1,lane);
}
__device__ __forceinline__ u32 wcrt_subtract(u32 x0,u32 x1,u32 y0,u32 y1,
 u32& z0,u32& z1,u32 lane) {
    const u32 b0=wcrt_prefix(x0<y0,x0==y0,lane,0);
    const u32 bridge=__shfl_sync(0xffffffffu,b0,31);
    const u32 b1=wcrt_prefix(x1<y1,x1==y1,lane,bridge);
    u32 prev0=__shfl_up_sync(0xffffffffu,b0,1),prev1=__shfl_up_sync(0xffffffffu,b1,1);
    if(!lane){prev0=0;prev1=bridge;}
    z0=x0-y0-prev0;z1=x1-y1-prev1;
    return __shfl_sync(0xffffffffu,b1,31);
}
// One 256-thread block per coefficient. Input: [(prime*8+digit)*N+i].
// partial_matrix: [K][shortWords], full P/p. correction_table: [K+1][shortWords],
// (2^G-alpha*P) modulo 2^(32*shortWords). positions[8] = ceil(W*j/8).
// bias[L] is canonical (sum_j 2^(G+positions[j])) modulo q.
extern "C" __global__ void reconstruct_weighted8(const u32* residues,u32* output,
 const u32* primes,const u64* ratios,const u32* crt_inverse,const u32* partial_matrix,
 const u32* correction_table,const u32* positions,const u32* q,const u32* bias,
 u32 N,u32 K,u32 W,u32 L,u32 shortWords,u32 d) {
    const u32 lane=threadIdx.x&31,digit=threadIdx.x>>5,i=blockIdx.x;
    if(i>=N)return;
    __shared__ u32 residues_scaled[8][128];
    __shared__ u32 positive[8][32];
    u64 fixed=0;
    for(u32 r=lane;r<K;r+=32) {
        const u32 p=primes[r];const u64 mu=ratios[r];
        const u32 t=residues[((u64)r*8+digit)*N+i];
        residues_scaled[digit][r]=t;
        const u64 numerator=u64(t)<<32;
        u64 quotient=__umul64hi(numerator,mu);
        quotient+=(numerator-quotient*p)>=p;
        fixed+=quotient;
    }
    for(u32 offset=16;offset;offset>>=1)fixed+=__shfl_down_sync(0xffffffffu,fixed,offset);
    // sum(t_r/p_r)=alpha+T/P with |T/P|<1/4. Flooring each 32-bit fixed-point
    // term underestimates by <K/2^32<=128/2^32, so nearest-integer rounding
    // still recovers alpha exactly, including negative T.
    const u32 alpha=u32((__shfl_sync(0xffffffffu,fixed,0)+(u64(1)<<31))>>32);
    __syncwarp();
    u64 low=0,high=0;
    if(lane<shortWords) {
        for(u32 r=0;r<K;++r) {
            const u64 product=u64(residues_scaled[digit][r])*partial_matrix[(u64)r*shortWords+lane];
            low+=u32(product);high+=product>>32;
        }
        low+=correction_table[(u64)alpha*shortWords+lane];
    }
    u64 previous=__shfl_up_sync(0xffffffffu,high,1);
    if(!lane)previous=0;
    u32 word=0,unused=0;
    wcrt_normalize(low+previous,0,word,unused,lane);
    // Carry beyond shortWords is discarded intentionally: short CRT is modulo
    // 2^(32*shortWords), and its unique positive answer T_digit+2^G fits there.
    positive[digit][lane]=lane<shortWords?word:0;
    __syncthreads();
    if(digit)return;
    u64 assembled0=0,assembled1=0;
    for(u32 j=0;j<8;++j) {
        const u32 offset=positions[j]/32,shift=positions[j]%32;
        const int index0=int(lane)-int(offset),index1=int(lane+32)-int(offset);
        const u32 a0=index0>=0&&index0<32?positive[j][index0]:0;
        const u32 a1=index1>=0&&index1<32?positive[j][index1]:0;
        const u32 b0=index0>0&&index0<=32?positive[j][index0-1]:0;
        const u32 b1=index1>0&&index1<=32?positive[j][index1-1]:0;
        assembled0+=shift?(a0<<shift)|(b0>>(32-shift)):a0;
        assembled1+=shift?(a1<<shift)|(b1>>(32-shift)):a1;
    }
    u32 x0=0,x1=0;
    wcrt_normalize(assembled0,assembled1,x0,x1,lane);
    wcrt_fold(x0,x1,W,d,lane);
    // First fold is <2^W+2^(W-2), so high<=1. If high==1, its low part is
    // <2^(W-2); another fold is strictly below 2^W for W>=128 and d<2^32.
    wcrt_fold(x0,x1,W,d,lane);
    const u32 q0=lane<L?q[lane]:0,q1=lane+32<L?q[lane+32]:0;
    u32 z0,z1;
    const u32 underflow=wcrt_subtract(x0,x1,q0,q1,z0,z1,lane);
    if(!underflow){x0=z0;x1=z1;}
    const u32 bias0=lane<L?bias[lane]:0,bias1=lane+32<L?bias[lane+32]:0;
    const u32 negative=wcrt_subtract(x0,x1,bias0,bias1,z0,z1,lane);
    wcrt_normalize(u64(z0)+(negative?q0:0),u64(z1)+(negative?q1:0),x0,x1,lane);
    if(lane<L)output[(u64)i*L+lane]=x0;
    if(lane+32<L)output[(u64)i*L+lane+32]=x1;
}

__device__ __forceinline__ u32 wcrt16_prefix(u32 g,u32 p,u32 lane16) {
    const u32 generated=__ballot_sync(0xffffffffu,g!=0);
    const u32 stopped=__ballot_sync(0xffffffffu,p==0);
    const u32 warpLane=(threadIdx.x&16u)+lane16;
    const u32 halfMask=0xffffu<<(warpLane&16u);
    const u32 eligible=stopped&halfMask&(0xffffffffu>>(31u-warpLane));
    if(!eligible)return 0;
    return (generated>>(31u-u32(__clz(eligible))))&1u;
}
__device__ __forceinline__ u32 wcrt16_normalize(u64 value,u32 lane16) {
    const u32 high=u32(value>>32);
    u32 previous=__shfl_up_sync(0xffffffffu,high,1,16);
    if(!lane16)previous=0;
    const u64 z=u64(u32(value))+previous;
    const u32 carry=wcrt16_prefix(u32(z>>32),u32(z)==0xffffffffu,lane16);
    u32 prior=__shfl_up_sync(0xffffffffu,carry,1,16);
    if(!lane16)prior=0;
    return u32(z)+prior;
}
// Exact LARGE specialization: two 16-lane digit groups per warp, four warps.
extern "C" __global__ void reconstruct_weighted8_subwarp16(const u32* residues,u32* output,
 const u32* primes,const u64* ratios,const u32* crt_inverse,const u32* partial_matrix,
 const u32* correction_table,const u32* positions,const u32* q,const u32* bias,
 u32 N,u32 K,u32 W,u32 L,u32 shortWords,u32 d) {
    const u32 lane16=threadIdx.x&15u,digit=threadIdx.x>>4,i=blockIdx.x;
    if(N!=65536||W!=1747||L!=55||K!=16||!shortWords||shortWords>16||blockDim.x!=128||i>=N)return;
    __shared__ u32 residues_scaled[8][16];
    __shared__ u32 positive[8][16];
    const u32 prime=primes[lane16];
    const u64 mu=ratios[lane16];
    const u32 scaled=residues[((u64)lane16*8+digit)*N+i];
    residues_scaled[digit][lane16]=scaled;
    const u64 numerator=u64(scaled)<<32;
    u64 fixed=__umul64hi(numerator,mu);
    fixed+=(numerator-fixed*prime)>=prime;
    for(u32 offset=8;offset;offset>>=1)fixed+=__shfl_down_sync(0xffffffffu,fixed,offset,16);
    const u32 alpha=u32((__shfl_sync(0xffffffffu,fixed,0,16)+(u64(1)<<31))>>32);
    __syncwarp();
    u64 low=0,high=0;
    if(lane16<shortWords) {
        for(u32 r=0;r<16;++r) {
            const u64 product=u64(residues_scaled[digit][r])*partial_matrix[(u64)r*shortWords+lane16];
            low+=u32(product);high+=product>>32;
        }
        low+=correction_table[(u64)alpha*shortWords+lane16];
    }
    u64 previous=__shfl_up_sync(0xffffffffu,high,1,16);
    if(!lane16)previous=0;
    const u32 word=wcrt16_normalize(low+previous,lane16);
    positive[digit][lane16]=lane16<shortWords?word:0;
    __syncthreads();
    if(threadIdx.x>=32)return;
    const u32 lane=threadIdx.x;
    u64 assembled0=0,assembled1=0;
    for(u32 j=0;j<8;++j) {
        const u32 offset=positions[j]/32,shift=positions[j]%32;
        const int index0=int(lane)-int(offset),index1=int(lane+32)-int(offset);
        const u32 a0=index0>=0&&index0<16?positive[j][index0]:0;
        const u32 a1=index1>=0&&index1<16?positive[j][index1]:0;
        const u32 b0=index0>0&&index0<=16?positive[j][index0-1]:0;
        const u32 b1=index1>0&&index1<=16?positive[j][index1-1]:0;
        assembled0+=shift?(a0<<shift)|(b0>>(32-shift)):a0;
        assembled1+=shift?(a1<<shift)|(b1>>(32-shift)):a1;
    }
    u32 x0=0,x1=0;
    wcrt_normalize(assembled0,assembled1,x0,x1,lane);
    wcrt_fold(x0,x1,W,d,lane);
    // First fold is <2^W+2^(W-2), so high<=1. If high==1, its low part is
    // <2^(W-2); another fold is strictly below 2^W for W>=128 and d<2^32.
    wcrt_fold(x0,x1,W,d,lane);
    const u32 q0=lane<L?q[lane]:0,q1=lane+32<L?q[lane+32]:0;
    u32 z0,z1;
    const u32 underflow=wcrt_subtract(x0,x1,q0,q1,z0,z1,lane);
    if(!underflow){x0=z0;x1=z1;}
    const u32 bias0=lane<L?bias[lane]:0,bias1=lane+32<L?bias[lane+32]:0;
    const u32 negative=wcrt_subtract(x0,x1,bias0,bias1,z0,z1,lane);
    wcrt_normalize(u64(z0)+(negative?q0:0),u64(z1)+(negative?q1:0),x0,x1,lane);
    if(lane<L)output[(u64)i*L+lane]=x0;
    if(lane+32<L)output[(u64)i*L+lane+32]=x1;
}
