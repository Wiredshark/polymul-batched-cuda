// Fused tensor-core base conversion + outer forward pass (current R=S=256 tiling).
// Block = (8 columns, digit j, operand), 256 threads. For each group of 4 primes:
//  1) MMA phase: the block's 2048 coefficients i=row*256+first+col (16-coef m-tiles
//     = 2 rows x 8 cols) are converted exactly as batched_prepare_tc (same byte
//     weights bpt, reduce, psi^i) into a padded shared tile per prime;
//  2) NTT phase: warp w runs batched_forward's 256-point column DIF + cross
//     twiddle for (prime, column w), then the tile is stored like batched_forward.
// Output equals batched_prepare_tc followed by batched_forward (bit-exact); A/B
// are written once and never read. Opt-in on the host (FHERMA_TCFUSE=1).
#if defined(BPT_ENABLE)&&defined(BPR_DT)
#ifndef BPFT_MINB
#define BPFT_MINB 2
#endif
__device__ __forceinline__ void imma_pf(u32 (&c)[4],const u32 (&a)[4],uint2 b){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b.x),"r"(b.y));
}
// rw: this lane's 12 row-stage roots (host table, identical to bm_root values): 4,2,1,5 per stage group.
__device__ __forceinline__ void bpft_column(u32 (&a)[8],const u64* tw,const u64* __restrict__ rw,u32 N,u32 p,u32 lane,u32 column,u32 logR){
    u64 R[12];
    #pragma unroll
    for(u32 x=0;x<12;++x)R[x]=__ldg(rw+x);
    #pragma unroll
    for(u32 m=0;m<4;++m){
        const u64 w=R[m];
        const u32 x=a[m],y=a[m+4];a[m]=add(x,y,p);a[m+4]=mul_shoup(sub(x,y,p),w,p);
    }
    #pragma unroll
    for(u32 q=0;q<2;++q){
        const u64 w=R[4+q];
        #pragma unroll
        for(u32 h=0;h<2;++h){const u32 m=q+4*h;const u32 x=a[m],y=a[m+2];a[m]=add(x,y,p);a[m+2]=mul_shoup(sub(x,y,p),w,p);}
    }
    {
        const u64 w=R[6];
        #pragma unroll
        for(u32 m=0;m<8;m+=2){const u32 x=a[m],y=a[m+1];a[m]=add(x,y,p);a[m+1]=mul_shoup(sub(x,y,p),w,p);}
    }
    #pragma unroll
    for(u32 len=32,st=7;len>=2;len>>=1,++st){
        const u32 half=len/2;const bool upper=(lane&half)!=0;
        const u64 w=R[st];
        #pragma unroll
        for(u32 m=0;m<8;++m){
            const u32 x=a[m],y=__shfl_xor_sync(0xffffffffu,x,half);
            a[m]=upper?mul_shoup(sub(y,x,p),w,p):add(x,y,p);
        }
    }
    #pragma unroll
    for(u32 m=0;m<8;++m){
        const u32 slot=lane+32*m,k1=__brev(slot)>>(32-logR);
#ifdef TW_PACKED
        a[m]=mul_shoup(a[m],bm_root(tw,2u*column*k1,N,p),p);
#else
        a[m]=weighted_root_action(a[m],tw,2u*column*k1,N,p);
#endif
    }
}
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(256,BPFT_MINB) batched_prepfwd_tc(
    const u32* __restrict__ a,const u32* __restrict__ b,u32* A,u32* B,const u64* __restrict__ twist,
    const u32* __restrict__ bpt,const u64* __restrict__ rtw,u32 N,u32 logR,u32 r0,u32 r1){
    constexpr u32 L=BCRT_L;
    constexpr u32 POS[9]={BCRT_POS,BCRT_W};
    __shared__ u32 tile[4][256*9];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    const u32 first=blockIdx.x*8,j=blockIdx.y,op=blockIdx.z;
    const u32* src=op?b:a;u32* dst=op?B:A;
    const u32 stride=weighted_root_stride(N);
    u32 wb=0,sh=0;int wbits=0;
    #pragma unroll
    for(u32 x=0;x<8;++x)if(x==j){wb=POS[x]/32;sh=POS[x]%32;wbits=int(POS[x+1])-int(POS[x]);}
    const uint4* bj=reinterpret_cast<const uint4*>(bpt+(size_t(j)*32+lane)*16);
    u32 bf[16];
    #pragma unroll
    for(u32 q=0;q<4;++q){const uint4 v=__ldg(bj+q);bf[4*q]=v.x;bf[4*q+1]=v.y;bf[4*q+2]=v.z;bf[4*q+3]=v.w;}
    // digit-word masks for t=tig and t=tig+4
    u32 mask[2];
    #pragma unroll
    for(u32 e=0;e<2;++e){const int width=wbits-int(32*(tig+4*e));mask[e]=width>=32?~0u:(width<=0?0u:((1u<<width)-1));}
    // digit-j A fragments of the block's 128 m-tiles (16 per warp), extracted once
    u32 afs[16][4];
    #pragma unroll
    for(u32 q=0;q<16;++q){
        const u32 mt=warp+8*q;
        #pragma unroll
        for(u32 x=0;x<4;++x){
            const u32 row=2*mt+(x&1),t=tig+4*(x>>1);
            const u32* rp=src+u64(row*256u+first+gid)*L+wb+t;
            const u32 lo=__ldg(rp),hi=wb+t+1<L?__ldg(rp+1):0u;
            afs[q][x]=__funnelshift_r(lo,hi,sh)&mask[x>>1];
        }
    }
    #pragma unroll
    for(u32 u=0;u<4;++u){
        if(u<r0/4||u>=r1/4)continue;
        const u32 r=4*u+tig,p=bpr_pr[r];const u64 mu=bpr_mu[r];
        const u64* tw=twist+size_t(r)*stride+1;
        const u64 t1=tw[(first+gid)&127u];
        const uint2 b00=make_uint2(bf[(2*u)*2],bf[(2*u)*2+1]),b10=make_uint2(bf[(2*u+1)*2],bf[(2*u+1)*2+1]);
        #pragma unroll
        for(u32 q=0;q<16;++q){
            const u32 mt=warp+8*q;
            const u32 (&af)[4]=afs[q];
            u32 c0[4]={0,0,0,0},c1[4]={0,0,0,0};
            imma_pf(c0,af,b00);imma_pf(c1,af,b10);
            #pragma unroll
            for(u32 hf=0;hf<2;++hf){
                const u32 row=2*mt+hf;
                const u64 x=u64(c0[2*hf])+(u64(c0[2*hf+1])<<8)+(u64(c1[2*hf])<<16)+(u64(c1[2*hf+1])<<24);
                u32 y=reduce(x,p,mu);
                y=mul_shoup(y,t1,p);
                y=mul_shoup(y,tw[128u+2u*row+(first>>7)],p);
                tile[tig][row*9+gid]=y;
            }
        }
        __syncthreads();
        #pragma unroll 1
        for(u32 pl=0;pl<4;++pl){
            const u32 rr=4*u+pl,pp=bpr_pr[rr];
            const u64* twr=twist+size_t(rr)*stride;
            u32 v[8];
            #pragma unroll
            for(u32 m=0;m<8;++m)v[m]=tile[pl][(lane+32*m)*9+warp];
            bpft_column(v,twr,rtw+(size_t(rr)*32+lane)*12,N,pp,lane,first+warp,logR);
            #pragma unroll
            for(u32 m=0;m<8;++m)tile[pl][(lane+32*m)*9+warp]=v[m];
        }
        __syncthreads();
        #pragma unroll 1
        for(u32 pl=0;pl<4;++pl){
            u32* d=dst+u64((4*u+pl)*8+j)*N;
            for(u32 t=threadIdx.x;t<2048;t+=256){const u32 row=t>>3,col=t&7u;d[row*256+first+col]=tile[pl][row*9+col];}
        }
        __syncthreads();
    }
}
#endif // batched_prepfwd_tc
#endif
