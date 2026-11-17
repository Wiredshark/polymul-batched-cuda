// Fused prepare + outer forward pass (N=65536, R=S=256, J=8), NVRTC-specialized.
// Block = (8 columns, digit j, operand): warp w owns column first+w, register m
// holds row lane+32m. Each thread reads its 8 coefficients' digit-j words once,
// then for every prime: residues (same arithmetic as batched_prepare2), the
// 256-point row DIF (same as batched_forward), cross twiddle, padded-tile
// transpose and a coalesced store. A,B are written once and never re-read here.
#if defined(BPR_DT) && defined(BCRT_W) && defined(BPF_ENABLE)
template<u32 JD>
__device__ __forceinline__ void bpf_digit(const u32* __restrict__ src,u32* __restrict__ dst,
    const u32* primes,const u64* twist,u32 N,u32 logR,u32 first,u32* tile){
    constexpr u32 W=BCRT_W,L=BCRT_L,K=BCRT_K,DT=BPR_DT;
    constexpr u32 POS[8]={BCRT_POS};
    constexpr u32 BEGIN=POS[JD],END=JD==7?W:POS[JD+1];
    constexpr u32 W0=BEGIN/32,W1=(END-1)/32;           // words holding this digit
    constexpr u32 NWD=(W1+1<L?W1+2:W1+1)-W0;           // + one high word for funnel shifts
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,column=first+warp;
    const u32 stride=weighted_root_stride(N);
    #pragma unroll 1
    for(u32 r=0;r<K;++r){
        const u32 p=bpr_pr[r];const u64 mu=bpr_mu[r];
        const u64* tw=twist+(u64)r*stride;
        const u64 fa=bpr_fa[r*8+JD];
        u32 a[8];
#if WROOT_COMPACT
        const u64 t1=tw[1+(column&127u)];
#endif
        #pragma unroll
        for(u32 m=0;m<8;++m){
            // digit words re-read per prime: the block's 74 KB input slice stays in L1/L2
            u32 words[NWD];
            {const u32* row=src+(u64)((lane+32*m)*256+column)*L+W0;
             #pragma unroll
             for(u32 x=0;x<NWD;++x)words[x]=__ldg(row+x);}
            u64 sum=0;
            #pragma unroll
            for(u32 k=0;k<DT;++k){
                constexpr u32 dummy=0;(void)dummy;
                const u32 bit=BEGIN+16*k;
                if(bit<END){
                    const u32 wl=bit/32-W0,shift=bit%32,count=END-bit<16?END-bit:16;
                    const u32 lo=words[wl],hi=wl+1<NWD?words[wl+1]:0;
                    sum+=u64(__funnelshift_r(lo,hi,shift)&((1u<<count)-1))*bpr_wt[r*DT+k];
                }
            }
            const u32 i=(lane+32*m)*256+column;
            u32 x=reduce(sum,p,mu);
#if WROOT_COMPACT
            x=mul_shoup(x,t1,p);x=mul_shoup(x,tw[1+128u+(i>>7)],p);
#else
            x=mul_shoup(x,tw[i],p);
#endif
            a[m]=mul_shoup(x,fa,p);
        }
        // 256-point DIF along rows (identical to batched_forward)
        #pragma unroll
        for(u32 m=0;m<4;++m){
            const u64 w=bm_root(tw,2u*(lane+32u*m)*(N/256u),N,p);
            const u32 x=a[m],y=a[m+4];a[m]=add(x,y,p);a[m+4]=mul_shoup(sub(x,y,p),w,p);
        }
        #pragma unroll
        for(u32 q=0;q<2;++q){
            const u64 w=bm_root(tw,2u*(lane+32u*q)*(N/128u),N,p);
            #pragma unroll
            for(u32 h=0;h<2;++h){const u32 m=q+4*h;const u32 x=a[m],y=a[m+2];a[m]=add(x,y,p);a[m+2]=mul_shoup(sub(x,y,p),w,p);}
        }
        {
            const u64 w=bm_root(tw,2u*lane*(N/64u),N,p);
            #pragma unroll
            for(u32 m=0;m<8;m+=2){const u32 x=a[m],y=a[m+1];a[m]=add(x,y,p);a[m+1]=mul_shoup(sub(x,y,p),w,p);}
        }
        #pragma unroll
        for(u32 len=32;len>=2;len>>=1){
            const u32 half=len/2,j=lane&(half-1);const bool upper=(lane&half)!=0;
            const u64 w=bm_root(tw,2u*j*(N/len),N,p);
            #pragma unroll
            for(u32 m=0;m<8;++m){
                const u32 x=a[m],y=__shfl_xor_sync(0xffffffffu,x,half);
                a[m]=upper?mul_shoup(sub(y,x,p),w,p):add(x,y,p);
            }
        }
        #pragma unroll
        for(u32 m=0;m<8;++m){
            const u32 slot=lane+32*m,k1=__brev(slot)>>(32-logR);
            a[m]=mul_shoup(a[m],bm_root(tw,2u*column*k1,N,p),p);
        }
        __syncthreads();   // previous prime's tile fully stored
        #pragma unroll
        for(u32 m=0;m<8;++m)tile[(lane+32*m)*9+warp]=a[m];
        __syncthreads();
        u32* out=dst+(u64)(r*8+JD)*N;
        for(u32 t=threadIdx.x;t<2048;t+=256){
            const u32 row=t>>3,col=t&7u;out[(u64)row*256+first+col]=tile[row*9+col];
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) batched_prepfwd(const u32* a,const u32* b,u32* A,u32* B,
    const u32* primes,const u64* twist,u32 N,u32 logR,u64 in_stride,u64 ws_stride){
    __shared__ u32 tile[256*9];
    const u32 first=blockIdx.x*8,jd=blockIdx.y,op=blockIdx.z&1u;const u64 prod=blockIdx.z>>1;
    const u32* src=(op?b:a)+prod*in_stride;u32* dst=(op?B:A)+prod*ws_stride;
    switch(jd){
        case 0:bpf_digit<0>(src,dst,primes,twist,N,logR,first,tile);break;
        case 1:bpf_digit<1>(src,dst,primes,twist,N,logR,first,tile);break;
        case 2:bpf_digit<2>(src,dst,primes,twist,N,logR,first,tile);break;
        case 3:bpf_digit<3>(src,dst,primes,twist,N,logR,first,tile);break;
        case 4:bpf_digit<4>(src,dst,primes,twist,N,logR,first,tile);break;
        case 5:bpf_digit<5>(src,dst,primes,twist,N,logR,first,tile);break;
        case 6:bpf_digit<6>(src,dst,primes,twist,N,logR,first,tile);break;
        default:bpf_digit<7>(src,dst,primes,twist,N,logR,first,tile);break;
    }
}
#endif
