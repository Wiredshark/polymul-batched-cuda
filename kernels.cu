// Original diagnostic CUDA implementation. Exact integer arithmetic throughout.
typedef unsigned int u32;
typedef unsigned long long u64;
// Exit-time timeline probe (tl_probe.h): the TL module defines TL_PARAM / TL_GUARD (per-block %globaltimer start/end + %smid);
// production: both empty (PTX unchanged).
#ifndef TL_PARAM
#define TL_PARAM
#define TL_GUARD
#endif

// Packed constant: low word w; high word floor(w*2^32/p). For x,w<p<2^31,
// the approximate quotient is exact or one low, hence0<=r<2p<2^32. Unsigned
// low-word products preserve this exact remainder even when the products wrap.
__device__ __forceinline__ u32 mul_shoup(u32 x,u64 constant,u32 p) {
    u32 w=(u32)constant,q=__umulhi(x,(u32)(constant>>32));
    u32 r=x*w-q*p;return r>=p?r-p:r;
}
__device__ __forceinline__ u64 cross_shoup(const u64* powers,u32 exponent,u32 N,u32 p) {
    u64 packed=powers[exponent&(N-1)];
    // Roots are nonzero and p is odd, so w*2^32/p is never integral.
    // floor((p-w)*2^32/p) = 2^32-1-floor(w*2^32/p).
    if(exponent&N)return ((u64)(~(u32)(packed>>32))<<32)|(p-(u32)packed);
    return packed;
}

// The precomputed array contains psi^0 ... psi^(N-1), with psi^N=-1.
// Four-step cross factors use exponent2*column*frequency, less than2N.
__device__ __forceinline__ u32 cross_root(const u32* powers,u32 exponent,u32 N,u32 p) {
    u32 v=powers[exponent&(N-1)];return (exponent&N)?p-v:v;
}
#ifndef FAST_MOD
#define FAST_MOD 1
#endif
__device__ __forceinline__ u32 reduce(u64 v,u32 p,u64 reciprocal) {
#if FAST_MOD
    // floor(2^64/p) underestimates the quotient by at most one.
    u64 quotient=__umul64hi(v,reciprocal);
    u64 remainder=v-quotient*p;
    return (u32)(remainder>=p?remainder-p:remainder);
#else
    return v%p;
#endif
}
__device__ __forceinline__ u32 mul(u32 a, u32 b, u32 p,u64 reciprocal) { return reduce((u64)a*b,p,reciprocal); }
__device__ __forceinline__ u32 add(u32 a,u32 b,u32 p) {u32 s=a+b; return s>=p?s-p:s;}
__device__ __forceinline__ u32 sub(u32 a,u32 b,u32 p) {return a>=b?a-b:a+p-b;}

// Transpose the two canonical coefficient-major inputs once. The tiled copies
// have coalesced reads and writes; the padding avoids shared-memory bank conflicts.
// All subsequent prime conversions reuse contiguous coefficient lanes.
#ifndef BATCHED_ONLY
extern "C" __global__ void transpose_inputs(const u32* a,const u32* b,u32* at,u32* bt,u32 N,u32 L) {
    __shared__ u32 ta[32][33],tb[32][33];
    u32 tx=threadIdx.x%32,ty=threadIdx.x/32;
    u32 x=blockIdx.x*32+tx,y=blockIdx.y*32+ty;
    #pragma unroll
    for(u32 j=0;j<32;j+=8) if(x<L && y+j<N) {
        ta[ty+j][tx]=a[(u64)(y+j)*L+x];
        tb[ty+j][tx]=b[(u64)(y+j)*L+x];
    }
    __syncthreads();
    x=blockIdx.y*32+tx;y=blockIdx.x*32+ty;
    #pragma unroll
    for(u32 j=0;j<32;j+=8) if(x<N && y+j<L) {
        at[(u64)(y+j)*N+x]=ta[tx][ty+j];
        bt[(u64)(y+j)*N+x]=tb[tx][ty+j];
    }
}
extern "C" __global__ void transpose_output(const u32* source,u32* output,u32 N,u32 L) {
    __shared__ u32 tile[32][33];
    u32 tx=threadIdx.x%32,ty=threadIdx.x/32;
    u32 x=blockIdx.x*32+tx,y=blockIdx.y*32+ty;
    #pragma unroll
    for(u32 j=0;j<32;j+=8) if(x<N && y+j<L) tile[ty+j][tx]=source[(u64)(y+j)*N+x];
    __syncthreads();
    x=blockIdx.y*32+tx;y=blockIdx.x*32+ty;
    #pragma unroll
    for(u32 j=0;j<32;j+=8) if(x<L && y+j<N) output[(u64)(y+j)*L+x]=tile[tx][ty+j];
}

// Convert limb-major u32 limbs to prime-major residues, apply twist,
// in natural order. Forward DIF and inverse DIT cancel the frequency permutation.
extern "C" __global__ void prepare(const u32* a,const u32* b,u32* A,u32* B,
 const u32* primes,const u64* ratios,const u64* twist,const u32* weights,u32 N,u32 L,u32 logn) {
    u32 i=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y;
    if(i>=N) return;
    u32 p=primes[r];u64 mu=ratios[r],alo=0,ahi=0,blo=0,bhi=0;
    // x mod p = sum_j digit16(x,j) * (2^(16*j) mod p) mod p.
    // Setup checks 2*L*65535*(p-1) <= UINT64_MAX. Four independent
    // accumulation chains remove the serial modular-Horner dependency.
    for(u32 k=0;k<L;++k) {
        u32 av=a[(u64)k*N+i],bv=b[(u64)k*N+i];
        u32 wl=weights[(u64)r*(2*L)+2*k],wh=weights[(u64)r*(2*L)+2*k+1];
        alo+=(u64)(av&65535)*wl;ahi+=(u64)(av>>16)*wh;
        blo+=(u64)(bv&65535)*wl;bhi+=(u64)(bv>>16)*wh;
    }
    u32 x=reduce(alo+ahi,p,mu),y=reduce(blo+bhi,p,mu);
    u32 j=i;u64 w=twist[(u64)r*N+i];
    A[(u64)r*N+j]=mul_shoup(x,w,p); B[(u64)r*N+j]=mul_shoup(y,w,p);
}
extern "C" __global__ void forward(u32* A,u32* B,const u32* primes,
 const u64* ratios,const u32* twist,u32 N,u32 length) {
    u32 t=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(t>=N/2) return;
    u32 half=length/2,j=t%half,i=(t/half)*length+j,p=primes[r];
    u64 base=(u64)r*N,mu=ratios[r];
    u32 w=twist[base+2*j*(N/length)];
    u32 x=A[base+i],y=A[base+i+half];
    A[base+i]=add(x,y,p); A[base+i+half]=mul(sub(x,y,p),w,p,mu);
    x=B[base+i]; y=B[base+i+half];
    B[base+i]=add(x,y,p); B[base+i+half]=mul(sub(x,y,p),w,p,mu);
}
extern "C" __global__ void pointwise(const u32* A,const u32* B,u32* C,
 const u32* primes,const u64* ratios,u32 N,u32 logn) {
    u32 i=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(i>=N) return;
    u64 base=(u64)r*N;
    C[base+(__brev(i)>>(32-logn))]=mul(A[base+i],B[base+i],primes[r],ratios[r]);
}
extern "C" __global__ void inverse(u32* C,const u32* primes,const u64* ratios,const u32* untwist,
 u32 N,u32 length) {
    u32 t=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(t>=N/2) return;
    u32 half=length/2,j=t%half,i=(t/half)*length+j,p=primes[r];
    u64 base=(u64)r*N;
    u32 x=C[base+i],y=mul(C[base+i+half],untwist[base+2*j*(N/length)],p,ratios[r]);
    C[base+i]=add(x,y,p); C[base+i+half]=sub(x,y,p);
}
extern "C" __global__ void finish(u32* C,const u32* primes,const u64* ratios,const u32* untwist,
 const u32* invn,u32 N) {
    u32 i=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(i>=N) return;
    u64 offset=(u64)r*N+i; u32 p=primes[r];
    C[offset]=mul(mul(C[offset],invn[r],p,ratios[r]),untwist[offset],p,ratios[r]);
}

// Incremental exact CRT followed by reduction for q=2^W-d, 0<d<2^32.
// The host proves product(primes)>2*N*(q-1)^2. Hence centered CRT is unique.
// The fixed scratch capacity is validated at setup before any launch.
#define WORDS 128
__device__ bool ge(const u32* x,const u32* y,u32 size) {
    for(int j=(int)size-1;j>=0;--j) {if(x[j]!=y[j]) return x[j]>y[j];}
    return true;
}
__device__ u32 subtract(u32* x,const u32* y,u32 size) {
    u64 borrow=0;
    for(u32 j=0;j<size;++j) {u64 v=(u64)y[j]+borrow;u32 old=x[j];x[j]=(u32)((u64)old-v);borrow=(u64)old<v;}
    return (u32)borrow;
}
extern "C" __global__ void reconstruct(const u32* residues,u32* output,
 const u32* primes,const u64* ratios,const u32* prefixes,const u32* lengths,const u32* inverses,
 const u32* midpoint,const u32* q,const u32* pmodq,u32 N,u32 K,u32 W,u32 L,u32 d) {
    u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    u32 x[WORDS]={0},high[WORDS];
    for(u32 r=0;r<K;++r) {
        u32 p=primes[r],length=lengths[r],rem=0;u64 mu=ratios[r];
        for(int j=(int)length-1;j>=0;--j) rem=reduce(((u64)rem<<32)|x[j],p,mu);
        u32 digit=mul(sub(residues[(u64)r*N+i],rem,p),inverses[r],p,mu);
        u64 carry=0;
        for(u32 j=0;j<length;++j) {
            u64 value=(u64)prefixes[r*WORDS+j]*digit+x[j]+carry;
            x[j]=(u32)value;carry=value>>32;
        }
        for(u32 j=length;carry && j<WORDS;++j) {
            u64 value=(u64)x[j]+carry;x[j]=(u32)value;carry=value>>32;
        }
    }
    // P is odd; midpoint stores floor(P/2)+1, the first negative representative.
    bool negative=ge(x,midpoint,WORDS);
    u32 word=W/32,shift=W%32;
    for(;;) {
        u32 nonzero=0;
        for(u32 j=0;j<WORDS;++j) {
            u32 v=0;
            if(j+word<WORDS) {
                v=x[j+word]>>shift;
                if(shift && j+word+1<WORDS) v|=x[j+word+1]<<(32-shift);
            }
            high[j]=v;nonzero|=v;
        }
        if(!nonzero) break;
        if(shift) x[word]&=((1u<<shift)-1);
        for(u32 j=word+(shift!=0);j<WORDS;++j) x[j]=0;
        u64 carry=0;
        for(u32 j=0;j<WORDS;++j) {
            u64 value=(u64)high[j]*d+x[j]+carry;
            x[j]=(u32)value;carry=value>>32;
        }
    }
    if(ge(x,q,L)) subtract(x,q,L);
    if(negative && subtract(x,pmodq,L)) {
        u64 carry=0;
        for(u32 j=0;j<L;++j) {u64 v=(u64)x[j]+q[j]+carry;x[j]=(u32)v;carry=v>>32;}
    }
    for(u32 j=0;j<L;++j) output[(u64)j*N+i]=x[j];
}

extern "C" __global__ void check_reduction(const u64* values,u32* output,
 const u32* primes,const u64* ratios,u32 count) {
    u32 i=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(i<count) output[r*count+i]=reduce(values[i],primes[r],ratios[r]);
}

// Complete forward DIF, multiply at identical bit-reversed frequency indices,
// then begin inverse DIT, all in one shared-memory tile.
extern "C" __global__ void middle_local(const u32* A,const u32* B,u32* C,
 const u32* primes,const u64* ratios,const u32* twist,const u32* untwist,u32 N,u32 tile) {
    __shared__ u32 sa[1024],sb[1024];
    u32 r=blockIdx.y,start=blockIdx.x*tile,p=primes[r];
    u64 base=(u64)r*N,mu=ratios[r];
    for(u32 j=threadIdx.x;j<tile;j+=blockDim.x){sa[j]=A[base+start+j];sb[j]=B[base+start+j];}
    __syncthreads();
    for(u32 length=tile;length>=2;length/=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<tile/2;t+=blockDim.x){
            u32 j=t%half,i=(t/half)*length+j,w=twist[base+2*j*(N/length)];
            u32 x=sa[i],y=sa[i+half];sa[i]=add(x,y,p);sa[i+half]=mul(sub(x,y,p),w,p,mu);
            x=sb[i];y=sb[i+half];sb[i]=add(x,y,p);sb[i+half]=mul(sub(x,y,p),w,p,mu);
        }
        __syncthreads();
    }
    for(u32 j=threadIdx.x;j<tile;j+=blockDim.x)sa[j]=mul(sa[j],sb[j],p,mu);
    __syncthreads();
    for(u32 length=2;length<=tile;length*=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<tile/2;t+=blockDim.x){
            u32 j=t%half,i=(t/half)*length+j;
            u32 x=sa[i],y=mul(sa[i+half],untwist[base+2*j*(N/length)],p,mu);
            sa[i]=add(x,y,p);sa[i+half]=sub(x,y,p);
        }
        __syncthreads();
    }
    for(u32 j=threadIdx.x;j<tile;j+=blockDim.x)C[base+start+j]=sa[j];
}

// Final inverse DIT stage and coefficient normalization in one coalesced pass.
extern "C" __global__ void inverse_finish(u32* C,const u32* primes,
 const u64* ratios,const u32* untwist,const u32* invn,u32 N) {
    u32 j=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
    if(j>=N/2)return;
    u64 base=(u64)r*N,mu=ratios[r];u32 p=primes[r],half=N/2;
    u32 x=C[base+j],y=mul(C[base+j+half],untwist[base+2*j],p,mu);
    C[base+j]=mul(mul(add(x,y,p),invn[r],p,mu),untwist[base+j],p,mu);
    C[base+j+half]=mul(mul(sub(x,y,p),invn[r],p,mu),untwist[base+j+half],p,mu);
}

// Cooley-Tukey N=R*S, j=row*S+column, k=k1+R*k2.
// First: R-point transforms down G adjacent columns. Output row slots retain
// bit-reversed k1 order; cross factor uses the natural k1, not the row slot.
extern "C" __global__ void fourstep_forward(u32* A,u32* B,const u32* primes,
 const u64* ratios,const u64* twist,u32 N,u32 R,u32 S,u32 G,u32 logR) {
    extern __shared__ u32 workspace[];
    u32* sa=workspace;u32* sb=workspace+R*G;
    u32 prime=blockIdx.y,p=primes[prime],first=blockIdx.x*G;
    u32 logG=__ffs((int)G)-1;
    u64 base=(u64)prime*N;
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x){
        u32 row=t>>logG,column=first+(t&(G-1));
        sa[t]=A[base+row*S+column];sb[t]=B[base+row*S+column];
    }
    __syncthreads();
    for(u32 length=R,step=S;length>=2;length/=2,step*=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<(R/2)*G;t+=blockDim.x){
            u32 col=t&(G-1),butterfly=t>>logG,j=butterfly&(half-1),row=((butterfly&~(half-1))<<1)+j;
            u32 i=row*G+col,h=half*G;u64 w=twist[base+2*j*step];
            u32 x=sa[i],y=sa[i+h];sa[i]=add(x,y,p);sa[i+h]=mul_shoup(sub(x,y,p),w,p);
            x=sb[i];y=sb[i+h];sb[i]=add(x,y,p);sb[i+h]=mul_shoup(sub(x,y,p),w,p);
        }
        __syncthreads();
    }
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x){
        u32 slot=t>>logG,column=first+(t&(G-1)),k1=logR?__brev(slot)>>(32-logR):0;
        u64 w=cross_shoup(twist+base,2*column*k1,N,p);
        A[base+slot*S+column]=mul_shoup(sa[t],w,p);
        B[base+slot*S+column]=mul_shoup(sb[t],w,p);
    }
}

// Second: S-point transforms across rows, product, inverse S transforms.
// Both spectra have identical doubly-permuted order. No permutation is needed.
extern "C" __global__ void fourstep_middle(const u32* A,const u32* B,u32* C,
 const u32* primes,const u64* ratios,const u64* twist,const u64* untwist,
 u32 N,u32 R,u32 S,u32 logR) {
    extern __shared__ u32 workspace[];
    u32* sa=workspace;u32* sb=workspace+S;
    u32 prime=blockIdx.y,p=primes[prime],slot=blockIdx.x;
    u32 k1=logR?__brev(slot)>>(32-logR):0;
    u64 base=(u64)prime*N,offset=base+slot*S,mu=ratios[prime];
    for(u32 j=threadIdx.x;j<S;j+=blockDim.x){sa[j]=A[offset+j];sb[j]=B[offset+j];}
    __syncthreads();
    for(u32 length=S,step=R;length>=2;length/=2,step*=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<S/2;t+=blockDim.x){
            u32 j=t&(half-1),i=((t&~(half-1))<<1)+j;u64 w=twist[base+2*j*step];
            u32 x=sa[i],y=sa[i+half];sa[i]=add(x,y,p);sa[i+half]=mul_shoup(sub(x,y,p),w,p);
            x=sb[i];y=sb[i+half];sb[i]=add(x,y,p);sb[i+half]=mul_shoup(sub(x,y,p),w,p);
        }
        __syncthreads();
    }
    for(u32 j=threadIdx.x;j<S;j+=blockDim.x)sa[j]=mul(sa[j],sb[j],p,mu);
    __syncthreads();
    for(u32 length=2,step=N/2;length<=S;length*=2,step/=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<S/2;t+=blockDim.x){
            u32 j=t&(half-1),i=((t&~(half-1))<<1)+j;
            u32 x=sa[i],y=mul_shoup(sa[i+half],untwist[base+2*j*step],p);
            sa[i]=add(x,y,p);sa[i+half]=sub(x,y,p);
        }
        __syncthreads();
    }
    for(u32 j=threadIdx.x;j<S;j+=blockDim.x){
        u64 w=cross_shoup(untwist+base,2*j*k1,N,p);
        C[offset+j]=mul_shoup(sa[j],w,p);
    }
}

// Last: inverse R transforms down columns, then divide by N and undo psi^j.
// Every block owns disjoint columns, so the global buffer is safely in-place.
extern "C" __global__ void fourstep_inverse(u32* C,const u32* primes,
 const u64* ratios,const u64* untwist,const u64* normalized_untwist,u32 N,u32 R,u32 S,u32 G) {
    extern __shared__ u32 sc[];
    u32 prime=blockIdx.y,p=primes[prime],first=blockIdx.x*G;
    u32 logG=__ffs((int)G)-1;
    u64 base=(u64)prime*N;
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x)sc[t]=C[base+(t>>logG)*S+first+(t&(G-1))];
    __syncthreads();
    for(u32 length=2,step=N/2;length<=R;length*=2,step/=2){
        u32 half=length/2;
        for(u32 t=threadIdx.x;t<(R/2)*G;t+=blockDim.x){
            u32 col=t&(G-1),butterfly=t>>logG,j=butterfly&(half-1),row=((butterfly&~(half-1))<<1)+j;
            u32 i=row*G+col,h=half*G;
            u32 x=sc[i],y=mul_shoup(sc[i+h],untwist[base+2*j*step],p);
            sc[i]=add(x,y,p);sc[i+h]=sub(x,y,p);
        }
        __syncthreads();
    }
    for(u32 t=threadIdx.x;t<R*G;t+=blockDim.x){
        u32 j=(t>>logG)*S+first+(t&(G-1));
        C[base+j]=mul_shoup(sc[t],normalized_untwist[base+j],p);
    }
}

#endif // BATCHED_ONLY
