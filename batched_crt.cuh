// Batched exact weighted CRT, one thread per coefficient. Point constants come
// from NVRTC macros (BCRT_*) so every limb array lives in registers with
// compile-time indices. Arithmetic is the same integer computation as
// reconstruct_weighted8: per digit Z_j = sum_r t_r*P/p_r + corr[alpha_j]
// (mod 2^(32*SW)), x = sum_j Z_j*2^pos_j (mod 2^2048), two folds by d,
// conditional -q, -bias (+q). grid = (N/128, products), block = 128.
#ifdef BCRT_W
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(128,3) batched_crt(const u32* residues,u32* output,
 const u32* primes,const u64* ratios,const u32* partial_matrix,const u32* correction_table,
 const u32* q,const u32* bias,u32 N,u32 d,u64 res_stride,u64 out_stride) {
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64;
    constexpr u32 POS[8]={BCRT_POS};
    constexpr u32 PR[K]={BCRT_PRIMES};constexpr u64 MU[K]={BCRT_MUS};constexpr u32 PM[K*SW]={BCRT_PM};
    constexpr u32 QW[L]={BCRT_Q},BW[L]={BCRT_BIAS};
    __shared__ u32 stage[128*L];
    const u32 i0=blockIdx.x*128,i=i0+threadIdx.x;
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
    u32* dst=output+u64(i0)*L;
    for(u32 x=threadIdx.x;x<128*L;x+=128)dst[x]=stage[x];
}
#endif // batched_crt
#endif

// Tensor-core exact CRT (IMMA u8 x u8 -> s32). Per digit j and coefficient:
//   U_j = sum_r t_r * (P/p_r) is an integer matrix product over bytes:
//   A[coef][(r,a)] = byte a of t_r (the residue word itself is the fragment),
//   B[(r,a)][c]    = byte (c-a) of P/p_r, column c = output byte position.
// Column sums are < 2^22 (host-checked), so s32 accumulation is exact; byte
// pairs are merged into 16-bit-digit sums H_k = Y_2k + 256*Y_2k+1 < 2^32.
// A ninth tile computes S = sum_r t_r*R_r, R_r = floor(2^F/p_r), from which
// alpha = round(sum t_r/p_r) (error < 2^-11, true distance from .5 > 1/4).
// Z_j = (U_j + 2^G - alpha*P) mod 2^512 as in batched_crt; the rest (shifted
// accumulation, folds, -q, -bias) is the same integer computation.
// One warp = 32 coefficients (two m16 tiles); H goes through a warp-private
// shared stage so that each lane then owns one coefficient.
#ifdef BTC_F
__device__ __forceinline__ void imma16832(u32 (&c)[4],u32 a0,u32 a1,u32 a2,u32 a3,u32 b0,u32 b1){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
#ifndef BPO_LEAN
extern "C" __global__ void __launch_bounds__(128,3) batched_crt_tc(const u32* __restrict__ residues,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d){
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
    #pragma unroll
    for(u32 w=0;w<L;++w)st[lane*L+w]=acc[w];
    __syncwarp();
    static_assert((32*L)%4==0,"stage uint4");
    uint4* dst=reinterpret_cast<uint4*>(output+u64(i0)*L);const uint4* src=reinterpret_cast<const uint4*>(st);
    #pragma unroll
    for(u32 x=lane;x<32*L/4;x+=32)dst[x]=src[x];
}
#endif // batched_crt_tc
#endif
