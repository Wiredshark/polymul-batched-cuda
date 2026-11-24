// PLACEHOLDER J=1 base conversions (plain integer CUDA, no tensor cores, not tuned): exactly ref.py forward_convert
// (uncentred) and back_convert, for one product.  To be replaced by the tensor-core kernels of wt-j1conv behind the same
// host interface (j1_path.h: forward_convert / back_convert).
// Boundary layouts (ref.py): limbs [N][55] u32 (row-major, one coefficient per row); residues [K][N] u32 prime-major.
#ifndef J1_CONV_REF_CUH
#define J1_CONV_REF_CUH
#define J1C_L 55u
// forward: block = 16 coefficients, thread k < K owns prime k (4 byte-plane sums per coefficient, s32 = ref imma).
// tbl = fwdA or fwdB [224][4K] u8: row b, column 4k + j = byte j of 2^(8b) f_k mod p_k (row 219: -q f_k mod p_k).
// Epilogue (ref): lo = S0 + S1<<8 + S2<<16 + S3<<24 mod 2^32, hi = S0>>16 + S1>>8 + S2 + S3<<8,
// qh = umulhi(hi, floor(2^61/p)) >> 13, r = lo - qh p in [0, 2p).
extern "C" __global__ void __launch_bounds__(128) j1c_fwd_ref(const u32* __restrict__ x,u32* __restrict__ r,
    const unsigned char* __restrict__ tbl,const u32* __restrict__ primes,const u32* __restrict__ bm,u32 K,u32 N){
    __shared__ u32 lim[16][56];
    const u32 n0=blockIdx.x*16u,k=threadIdx.x;
    for(u32 i=threadIdx.x;i<16u*56u;i+=blockDim.x){const u32 m=i/56u,j=i%56u;lim[m][j]=j<J1C_L?x[size_t(n0+m)*J1C_L+j]:0u;}
    __syncthreads();
    if(k>=K)return;
    u32 S[16][4];
    #pragma unroll
    for(u32 m=0;m<16;++m){S[m][0]=S[m][1]=S[m][2]=S[m][3]=0u;}
    const u32 stride=4u*K;
    for(u32 b=0;b<220u;++b){
        const u32 w=*reinterpret_cast<const u32*>(tbl+size_t(b)*stride+4u*k);
        const u32 w0=w&255u,w1=(w>>8)&255u,w2=(w>>16)&255u,w3=w>>24;
        #pragma unroll
        for(u32 m=0;m<16;++m){const u32 a=(lim[m][b>>2]>>(8u*(b&3u)))&255u;S[m][0]+=a*w0;S[m][1]+=a*w1;S[m][2]+=a*w2;S[m][3]+=a*w3;}
    }
    const u32 p=primes[k],mk=bm[k];
    #pragma unroll
    for(u32 m=0;m<16;++m){
        const u32 lo=S[m][0]+(S[m][1]<<8)+(S[m][2]<<16)+(S[m][3]<<24);
        const u32 hi=(S[m][0]>>16)+(S[m][1]>>8)+S[m][2]+(S[m][3]<<8);
        const u32 qh=__umulhi(hi,mk)>>13;
        r[size_t(k)*N+n0+m]=lo-qh*p;
    }
}
// back: block = 32 coefficients, 224 threads.  v = (sum_k y_k R_k + 2^44) >> 45 (= the ref fraction pass: its Toeplitz
// byte columns sum to exactly sum_k y_k R_k); main column c < 219: S_c = sum_{k,i} byte_i(y_k) backM[4k+i][c] +
// v backM[4K][c]; then (one thread per coefficient) carries into 56 limbs, one fold of H = Z >> 1747 times delta,
// one conditional subtraction of q -> canonical 55 limbs.
extern "C" __global__ void __launch_bounds__(224) j1c_back_ref(const u32* __restrict__ y,u32* __restrict__ out,
    const unsigned char* __restrict__ bmat,const u32* __restrict__ R,u32 K,u32 N,u32 delta){
    __shared__ u32 ys[32][121];
    __shared__ u32 S[32][225];
    const u32 n0=blockIdx.x*32u,t=threadIdx.x;
    for(u32 i=t;i<32u*K;i+=224u){const u32 k=i/32u,m=i%32u;ys[m][k]=y[size_t(k)*N+n0+m];}
    __syncthreads();
    if(t<32u){unsigned long long T=0;for(u32 k=0;k<K;++k)T+=(unsigned long long)ys[t][k]*R[k];ys[t][K]=u32((T+(1ull<<44))>>45);}
    __syncthreads();
    {   const u32 c=t;u32 acc[32];
        #pragma unroll
        for(u32 m=0;m<32;++m)acc[m]=0u;
        for(u32 row=0;row<=4u*K;++row){
            const u32 b=bmat[size_t(row)*224u+c];
            const u32 kk=row>>2,sh=8u*(row&3u);
            #pragma unroll
            for(u32 m=0;m<32;++m){const u32 a=row<4u*K?(ys[m][kk]>>sh)&255u:ys[m][K];acc[m]+=a*b;}
        }
        #pragma unroll
        for(u32 m=0;m<32;++m)S[m][c]=acc[m];
    }
    __syncthreads();
    if(t>=32u)return;
    const u32 m=t;u32 lim[56];
    unsigned long long carry=0;
    #pragma unroll
    for(u32 j=0;j<56;++j){unsigned long long acc=carry;
        #pragma unroll
        for(u32 e=0;e<4;++e){const u32 c=4u*j+e;if(c<219u)acc+=(unsigned long long)S[m][c]<<(8u*e);}
        lim[j]=u32(acc);carry=acc>>32;}
    // fold: Z = H 2^1747 + Lo -> Lo + H delta
    const u32 sh=19u;   // 1747 - 32*54
    const unsigned long long H=(lim[54]>>sh)|((unsigned long long)lim[55]<<(32u-sh));
    lim[54]&=(1u<<sh)-1u;
    carry=H*delta;
    #pragma unroll
    for(u32 j=0;j<55;++j){const unsigned long long acc=(unsigned long long)lim[j]+carry;lim[j]=u32(acc);carry=acc>>32;}
    // conditional subtraction: Tz >= q  <=>  Tz + delta >= 2^1747
    u32 tt[55];carry=delta;
    #pragma unroll
    for(u32 j=0;j<55;++j){const unsigned long long acc=(unsigned long long)lim[j]+carry;tt[j]=u32(acc);carry=acc>>32;}
    const bool ge=((tt[54]>>sh)!=0u)||carry!=0;
    tt[54]&=(1u<<sh)-1u;
    u32* o=out+size_t(n0+m)*J1C_L;
    #pragma unroll
    for(u32 j=0;j<55;++j)o[j]=ge?tt[j]:lim[j];
}
// diag helper: count of differing uint4 groups
extern "C" __global__ void __launch_bounds__(256) j1d_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){
    unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){
        const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}
    if(c)atomicAdd(bad,c);}
#endif
