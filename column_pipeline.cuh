// Column partitions of the frozen parent kernels. Global R,S,N,roots and
// arithmetic are unchanged; only block/coefficient selection is restricted.
// Host domain: first_column is aligned to column_count, column_count in
// {32,64,128}, S=256, R=256, N=65536, G=8. No host pipeline is wired yet.

extern "C" __global__ void transpose_inputs_columns(const u32* a,const u32* b,u32* at,u32* bt,u32 N,u32 L,u32 S,u32 first_column,u32 log_column_count) {
    const u32 log_tiles=log_column_count-5;
    const u32 original_y=(blockIdx.y>>log_tiles)*(S/32)+first_column/32+(blockIdx.y&((1u<<log_tiles)-1));
    __shared__ u32 ta[32][33],tb[32][33];
    u32 tx=threadIdx.x%32,ty=threadIdx.x/32;
    u32 x=blockIdx.x*32+tx,y=original_y*32+ty;
    #pragma unroll
    for(u32 j=0;j<32;j+=8) if(x<L && y+j<N) {
        ta[ty+j][tx]=a[(u64)(y+j)*L+x];
        tb[ty+j][tx]=b[(u64)(y+j)*L+x];
    }
    __syncthreads();
    x=original_y*32+tx;y=blockIdx.x*32+ty;
    #pragma unroll
    for(u32 j=0;j<32;j+=8) if(x<N && y+j<L) {
        at[(u64)(y+j)*N+x]=ta[tx][ty+j];
        bt[(u64)(y+j)*N+x]=tb[tx][ty+j];
    }
}

extern "C" __global__ void weighted_prepare_columns(
    const u32* a,const u32* b,u32* A,u32* B,
    const u32* primes,const u64* ratios,const u64* prepared_twist,const u64* twist,
    const u32* weights,const u32* positions,u32 N,u32 L,u32 W,u32 digit_terms,u32 S,u32 first_column,u32 log_column_count){
    const u32 column_count=1u<<log_column_count;
    const u32 local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=(N/S)*column_count)return;
    const u32 i=(local>>log_column_count)*S+first_column+(local&(column_count-1));
    const u32 batch=blockIdx.y;
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

extern "C" __global__ void weighted_forward_columns(u32* A,u32* B,const u32* primes,
 const u64* ratios,const u64* twist,u32 N,u32 R,u32 S,u32 G,u32 logR,u32 J,u32 first_column) {
    extern __shared__ u32 workspace[];
    u32* sa=workspace;u32* sb=workspace+R*G;
    u32 batch=blockIdx.y,prime=batch/J,p=primes[prime],first=first_column+blockIdx.x*G;
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

extern "C" __global__ void weighted_inverse_columns(u32* C,const u32* primes,
 const u64* ratios,const u64* untwist,const u64* normalized_untwist,
 u32 N,u32 R,u32 S,u32 G,u32 J,u32 first_column) {
    extern __shared__ u32 sc[];
    u32 batch=blockIdx.y,prime=batch/J,p=primes[prime],first=first_column+blockIdx.x*G;
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

// CA1_BEGIN_ASSEMBLY_HELPER
__device__ __forceinline__ u64 ca1_aggregate_word(const u64* columns,u32 wraps,
 const u32* positions,u32 short_words,u32 word) {
    u64 total=0;
    u32 magnitude_word=0,borrow=0;
    for(u32 j=0;j<8;++j) {
        const u32 offset=positions[j]/32,shift=positions[j]%32;
        const int at=int(word)-int(offset);
        const u64 a=at>=0&&at<int(short_words)?columns[j*16+at]:0;
        const u64 b=at>0&&at<=int(short_words)?columns[j*16+at-1]:0;
        // Keep the high part of the redundant previous column. A u32 cast
        // of b here would silently discard carries before digit assembly.
        total+=u64(u32(u32(a)<<shift))+(b>>(32u-shift));
        const u32 wrap_word=short_words+offset;
        if((wraps>>j)&1u) {
            // positions are ceil(W*j/8), so their word offsets are distinct.
            if(word==wrap_word)magnitude_word|=1u<<shift;
            if(word>wrap_word)borrow=1;
        }
    }
    // One word of -sum_j wraps_j*2^(32*short_words+positions[j]), modulo
    // 2^2048. 'borrow' says some nonzero magnitude word lies below this word.
    return total+u64(u32(0u-magnitude_word-borrow));
}
// CA1_END_ASSEMBLY_HELPER

extern "C" __global__ void reconstruct_weighted8_subwarp16_columns(const u32* residues,u32* output,
 const u32* primes,const u64* ratios,const u32* crt_inverse,const u32* partial_matrix,
 const u32* correction_table,const u32* positions,const u32* q,const u32* bias,
 u32 N,u32 K,u32 W,u32 L,u32 shortWords,u32 d,u32 S,u32 first_column,u32 log_column_count) {
    const u32 column_count=1u<<log_column_count;
    const u32 lane16=threadIdx.x&15u,digit=threadIdx.x>>4;
    const u32 i=(blockIdx.x>>log_column_count)*S+first_column+(blockIdx.x&(column_count-1));
    if(N!=65536||W!=1747||L!=55||K!=16||!shortWords||shortWords>16||blockDim.x!=128||i>=N)return;
    __shared__ u32 residues_scaled[8][16];
    __shared__ u64 redundant[128];
    __shared__ u32 wrapped[8];
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
    // U_digit = S_digit + correction_mod_capacity. Do not normalize its
    // sixteen word columns; carry them only after all eight weights combine.
    redundant[digit*16+lane16]=lane16<shortWords?low+previous:0;
    if(!lane16)wrapped[digit]=alpha!=0;
    __syncthreads();
    if(threadIdx.x>=32)return;
    const u32 lane=threadIdx.x;
    const u32 wraps=__ballot_sync(0xffffffffu,lane<8?wrapped[lane]!=0:false);
    const u64 assembled0=ca1_aggregate_word(redundant,wraps,positions,shortWords,lane);
    const u64 assembled1=ca1_aggregate_word(redundant,wraps,positions,shortWords,lane+32);
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
