// Parameter-only two-factor roots. All coefficient actions remain timed.
#ifndef WROOT_COMPACT
#define WROOT_COMPACT 1
#endif
__device__ __forceinline__ u32 weighted_root_stride(u32 N) {return (WROOT_COMPACT&&N==65536)?641u:N;}
__device__ __forceinline__ u64 weighted_root_pack(const u64* table,u32 exponent,u32 N,u32 p) {
    exponent&=2*N-1;u32 e=exponent&(N-1);u64 packed;
    if(!WROOT_COMPACT||N!=65536)packed=table[e];
    else if((e&127u)==0)packed=table[1+128u+(e>>7)];
    else {
        u32 w=mul_shoup(u32(table[1+(e&127u)]),table[1+128u+(e>>7)],p);
        u64 mu=table[0];u32 q=w*u32(mu>>32)+__umulhi(w,u32(mu));
        u64 rem=(u64(w)<<32)-u64(q)*p;if(rem>=p)++q;
        packed=u64(w)|(u64(q)<<32);
    }
    if(exponent&N)packed=u64(p-u32(packed))|(u64(0xffffffffu-u32(packed>>32))<<32);
    return packed;
}
__device__ __forceinline__ u32 weighted_root_action(u32 x,const u64* table,u32 exponent,u32 N,u32 p) {
    if(!WROOT_COMPACT||N!=65536)return mul_shoup(x,weighted_root_pack(table,exponent,N,p),p);
    exponent&=2*N-1;u32 e=exponent&(N-1);
    x=mul_shoup(x,table[1+(e&127u)],p);
    x=mul_shoup(x,table[1+128u+(e>>7)],p);
    return (exponent&N)&&x?p-x:x;
}

#ifndef BATCHED_ONLY
// Exact variable-width digit extraction. All input-dependent work is timed.
extern "C" __global__ void weighted_prepare(
    const u32* a,const u32* b,u32* A,u32* B,
    const u32* primes,const u64* ratios,const u64* prepared_twist,const u64* twist,
    const u32* weights,const u32* positions,u32 N,u32 L,u32 W,u32 digit_terms){
    u32 i=blockIdx.x*blockDim.x+threadIdx.x,batch=blockIdx.y;
    if(i>=N)return;
    u32 r=batch/8,j=batch%8,p=primes[r],begin=positions[j];
    u32 end=j==7?W:positions[j+1];u64 mu=ratios[r];
    u64 ae=0,ao=0,be=0,bo=0;
    for(u32 k=0;k<digit_terms;++k){
        u32 bit=begin+16*k;
        if(bit<end){
            u32 word=bit/32,shift=bit%32,count=min(16u,end-bit);
            u32 al=a[size_t(word)*N+i],bl=b[size_t(word)*N+i];
            u32 ah=word+1<L?a[size_t(word+1)*N+i]:0;
            u32 bh=word+1<L?b[size_t(word+1)*N+i]:0;
            u32 mask=(1u<<count)-1,weight=weights[size_t(r)*digit_terms+k];
            u32 av=__funnelshift_r(al,ah,shift)&mask,bv=__funnelshift_r(bl,bh,shift)&mask;
            if(k&1){ao+=u64(av)*weight;bo+=u64(bv)*weight;}
            else{ae+=u64(av)*weight;be+=u64(bv)*weight;}
        }
    }
    // Recover the product action through two exact Shoup multiplications.
    const u64* rootset=twist+size_t(r)*weighted_root_stride(N);u64 factor=prepared_twist[batch];
    A[size_t(batch)*N+i]=mul_shoup(weighted_root_action(reduce(ae+ao,p,mu),rootset,i,N,p),factor,p);
    B[size_t(batch)*N+i]=mul_shoup(weighted_root_action(reduce(be+bo,p,mu),rootset,i,N,p),factor,p);
}

#endif // BATCHED_ONLY
