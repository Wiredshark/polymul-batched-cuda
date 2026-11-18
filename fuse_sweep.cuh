// Exit-time sweep only (appended to the sweep copy of the source, never to the production one): P3 folded away.
//   sw_p2j : bpo_middle_r16 + P3's F (cross untwiddle * t1) and lazy J-DIT. The 8 digit CTAs of one (slot octet g, prime)
//            form a cluster (1,8,1); after the S transforms each CTA leaves its 8 slots x 256 columns in smem, CTA rank r
//            then reads columns 32r..32r+31 of all 8 digits through DSMEM, applies F + J-DIT and writes C'[prime*8+j][c][slot]
//            for all 8 digits j (same layout as C).
//   sw_crtr: bpo_crt_lean whose residue load stage is P3's R-DIT + t2n on C'. Block = (column c, parity b) = the 128
//            coefficients h = 2k+b of column c (k = 32 warp + lane). Per digit j, warp w transforms the rows of primes
//            4w..4w+3: 256 slots -> the 128 outputs h = b mod 2 (first DIT stage on (2u, 2u+1) keeps only the b half, the
//            remaining 7 stages never mix the position LSB), t2n, then the 16 x 128 residue tile (block smem) feeds the
//            unchanged MMA / lean digit stage. Output stores per coefficient h*256 + c.
// FZ flags (sw_crtr template): 1 = R-phase loads of digit j+1 issued before the MMA / digit stage of digit j (two rows);
// 8 = the 4 rows of the warp transformed interleaved (fz_rrow4, all 4 rows loaded first; with 1: rows 0,1 prefetched);
// ablations (wrong values): 2 = no R arithmetic (v = slot pairs, no t2n), 4 = no C' loads (synthetic slots).
#if defined(__CUDA_ARCH__)&&__CUDA_ARCH__>=900&&defined(BTC_F)
__device__ __forceinline__ void fz_cluster_sync(){asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;":::"memory");}
__device__ __forceinline__ u32 fz_mapa(u32 a,u32 rank){u32 r;asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"(a),"r"(rank));return r;}
__device__ __forceinline__ u32 fz_ldc(u32 a){u32 v;asm volatile("ld.shared::cluster.u32 %0, [%1];":"=r"(v):"r"(a):"memory");return v;}
// PJ ablations (wrong values): 1 = tile reads from the own smem (no DSMEM traffic), 2 = no cluster barriers
template<u32 PJ> __device__ __forceinline__ void fz_p2j_body(u32* TA,const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw,const u64* untwist,u32 logR){
    const u32 g=blockIdx.x,j=blockIdx.y,prime=blockIdx.z,p=primes[prime];
    const u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5,s=lane&7u,c2=4u*warp+(lane>>3);
    const u64* tf=r16+u64(prime)*512u;const u64* ti=tf+256u;
    const u64 base=u64(prime*8+j)*N;
    u32 a[16],b[16];
    {   const u32* a0=A+base+g*2048u+c2*8u+s;const u32* b0=B+base+g*2048u+c2*8u+s;
        #pragma unroll
        for(u32 c1=0;c1<16;++c1){if(untw&4){a[c1]=__ldcs(a0+128u*c1);b[c1]=__ldcs(b0+128u*c1);}else{a[c1]=a0[128u*c1];b[c1]=b0[128u*c1];}}
    }
    r16_dif16(a,tf,p);r16_dif16(b,tf,p);
    u32* const ta=TA+s*292u;
    #pragma unroll
    for(u32 i=1;i<16;++i){const u64 w=__ldg(tf+((c2*r16_brev4(i))&255u));a[i]=pxmsh(a[i],w,p);b[i]=pxmsh(b[i],w,p);}
    #pragma unroll
    for(u32 i=0;i<16;++i)ta[17*i+c2]=a[i];
    __syncthreads();
    if(untw&8)bpo_discard((threadIdx.x<64u?A:B)+base+g*2048u+(threadIdx.x&63u)*32u);
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=ta[17*c2+i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)ta[17*i+c2]=b[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)b[i]=ta[17*c2+i];
    r16_dif16(a,tf,p);r16_dif16(b,tf,p);
    {const u32 pinv=pxinv32(p);
        #pragma unroll
        for(u32 i=0;i<16;++i)a[i]=pxmont(a[i],b[i],p,pinv);}
    r16_dit16<true>(a,ti,p);
    {const u32 k1=r16_brev4(c2);
        #pragma unroll
        for(u32 i=1;i<16;++i)a[i]=pxmsh(a[i],__ldg(ti+((i*k1)&255u)),p);
        a[0]=pzred(a[0],p);}
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)ta[17*c2+i]=a[i];
    __syncthreads();
    #pragma unroll
    for(u32 i=0;i<16;++i)a[i]=ta[17*i+c2];
    r16_dit16<true>(a,ti,p);   // a[c1] = C value of (column 16 c1 + c2, slot 8g + s), digit j, in [0,2p)
    __syncthreads();   // transpose reads done
    #pragma unroll
    for(u32 c1=0;c1<16;++c1)TA[(16u*c1+c2)*8u+s]=a[c1];   // tile T[c][s]
    if(PJ&2)__syncthreads();else fz_cluster_sync();
    {   const u32 tid=threadIdx.x,sl=tid&7u,slot=g*8u+sl,k1=__brev(slot)>>(32-logR);
        const u64* itw=untwist+u64(prime)*weighted_root_stride(N);
        const u32 la=u32(__cvta_generic_to_shared(TA));
        #pragma unroll
        for(u32 u=0;u<2;++u){
            const u32 c=32u*j+16u*u+(tid>>3);
            u32 yv[8];
            #pragma unroll
            for(u32 d=0;d<8;++d)yv[d]=(PJ&1)?TA[(c*8u+sl+d*8u)&2047u]:fz_ldc(fz_mapa(la+(c*8u+sl)*4u,d));
            u64 F;{const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);
                const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));
                const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;
                F=u64(wv)|(u64(q)<<32);}
            #pragma unroll
            for(u32 d=0;d<8;++d)yv[d]=pxmsh(yv[d],F,p);
            #pragma unroll
            for(u32 d0=0;d0<8;d0+=2)pzdit1c(yv[d0],yv[d0+1],p);
            #pragma unroll
            for(u32 len=4;len<=8;len<<=1){const u32 half=len/2;
                #pragma unroll
                for(u32 d0=0;d0<8;d0+=len){
                    #pragma unroll
                    for(u32 jj=0;jj<half;++jj){if(jj==0)pzdit1(yv[d0],yv[d0+half],0,p);else pzdit(yv[d0+jj],yv[d0+jj+half],ti[jj*(256u/len)],p);}}}
            #pragma unroll
            for(u32 d=0;d<8;++d)C[u64(prime*8+d)*N+c*256u+slot]=yv[d];   // [0,2p)
        }
    }
    if(!(PJ&2))fz_cluster_sync();   // remote tile reads done before any CTA of the cluster exits
}
extern "C" __global__ void __cluster_dims__(1,8,1) __launch_bounds__(128) sw_p2j(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw,const u64* untwist,u32 logR){
    __shared__ __align__(16) u32 TA[8*292];fz_p2j_body<0>(TA,A,B,C,primes,r16,N,untw,untwist,logR);}
extern "C" __global__ void __cluster_dims__(1,8,1) __launch_bounds__(128) sw_p2j_nodsm(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw,const u64* untwist,u32 logR){
    __shared__ __align__(16) u32 TA[8*292];fz_p2j_body<1>(TA,A,B,C,primes,r16,N,untw,untwist,logR);}
extern "C" __global__ void __launch_bounds__(128) sw_p2j_nocl(const u32* A,const u32* B,u32* C,
    const u32* primes,const u64* r16,u32 N,u32 untw,const u64* untwist,u32 logR){
    __shared__ __align__(16) u32 TA[8*292];fz_p2j_body<3>(TA,A,B,C,primes,r16,N,untw,untwist,logR);}
// R-DIT scratch swizzle (row of 128 words): conflict-free for every layout of fz_rrow (uint4 writes of 4 consecutive q,
// reads/writes q = 16(l>>2)+(l&3)+4i, (l&15)+16i+64(l>>4), l+32i)
__device__ __forceinline__ u32 fz_sw(u32 q){return q^((q>>3)&4u)^((q>>1)&8u)^((q>>2)&16u);}
// one row: x = 8 slots 8 lane..8 lane+7 (values in [0,2p)), parity b -> v[i] = canonical residue of h = 2(lane + 32 i) + b
__device__ __forceinline__ void fz_rrow(const u32 (&x)[8],u32 (&v)[4],u32* scr,const u64* sit,const u64* tn,u32 b,u32 p,u32 lane){
    #pragma unroll
    for(u32 i=0;i<4;++i){const u32 a=pzred(x[2*i],p),t=pzred(x[2*i+1],p);v[i]=b?a+p-t:a+t;}   // q = 4 lane + i
    pzdit(v[0],v[1],sit[252u+b],p);pzdit(v[2],v[3],sit[252u+b],p);
    pzdit(v[0],v[2],sit[248u+b],p);pzdit(v[1],v[3],sit[250u+b],p);
    __syncwarp();
    *reinterpret_cast<uint4*>(scr+fz_sw(4u*lane))=make_uint4(v[0],v[1],v[2],v[3]);
    __syncwarp();
    {const u32 q0=16u*(lane>>2)+(lane&3u);
        #pragma unroll
        for(u32 i=0;i<4;++i)v[i]=scr[fz_sw(q0+4u*i)];
        const u32 jl=2u*(lane&3u)+b;
        pzdit(v[0],v[1],sit[240u+jl],p);pzdit(v[2],v[3],sit[240u+jl],p);
        pzdit(v[0],v[2],sit[224u+jl],p);pzdit(v[1],v[3],sit[232u+jl],p);
        __syncwarp();
        #pragma unroll
        for(u32 i=0;i<4;++i)scr[fz_sw(q0+4u*i)]=v[i];}
    __syncwarp();
    {const u32 q0=(lane&15u)+64u*(lane>>4);
        #pragma unroll
        for(u32 i=0;i<4;++i)v[i]=scr[fz_sw(q0+16u*i)];
        const u32 jl=2u*(lane&15u)+b;
        pzdit(v[0],v[1],sit[192u+jl],p);pzdit(v[2],v[3],sit[192u+jl],p);
        pzdit(v[0],v[2],sit[128u+jl],p);pzdit(v[1],v[3],sit[160u+jl],p);
        __syncwarp();
        #pragma unroll
        for(u32 i=0;i<4;++i)scr[fz_sw(q0+16u*i)]=v[i];}
    __syncwarp();
    #pragma unroll
    for(u32 i=0;i<4;++i)v[i]=scr[fz_sw(lane+32u*i)];
    {const u32 jl=2u*lane+b;pzdit(v[0],v[2],sit[jl],p);pzdit(v[1],v[3],sit[64u+jl],p);}
    #pragma unroll
    for(u32 i=0;i<4;++i)v[i]=pxmsh(v[i],__ldg(tn+2u*(lane+32u*i)),p);
}
// the 4 rows of a warp interleaved (ILP 8 per stage); x[r] = slots of row r, scratch row r at scr + 128 r
__device__ __forceinline__ void fz_rrow4(const u32 (&x)[4][8],u32 (&v)[4][4],u32* scr,const u64* const (&sit)[4],const u64* const (&tn)[4],const u32 (&p)[4],u32 b,u32 lane){
    #pragma unroll
    for(u32 r=0;r<4;++r){
        #pragma unroll
        for(u32 i=0;i<4;++i){const u32 a=pzred(x[r][2*i],p[r]),t=pzred(x[r][2*i+1],p[r]);v[r][i]=b?a+p[r]-t:a+t;}}
    #pragma unroll
    for(u32 r=0;r<4;++r){const u64 w4=sit[r][252u+b],w8a=sit[r][248u+b],w8b=sit[r][250u+b];
        pzdit(v[r][0],v[r][1],w4,p[r]);pzdit(v[r][2],v[r][3],w4,p[r]);pzdit(v[r][0],v[r][2],w8a,p[r]);pzdit(v[r][1],v[r][3],w8b,p[r]);}
    __syncwarp();
    #pragma unroll
    for(u32 r=0;r<4;++r)*reinterpret_cast<uint4*>(scr+128u*r+fz_sw(4u*lane))=make_uint4(v[r][0],v[r][1],v[r][2],v[r][3]);
    __syncwarp();
    {const u32 q0=16u*(lane>>2)+(lane&3u),jl=2u*(lane&3u)+b;
        #pragma unroll
        for(u32 r=0;r<4;++r){
            #pragma unroll
            for(u32 i=0;i<4;++i)v[r][i]=scr[128u*r+fz_sw(q0+4u*i)];}
        #pragma unroll
        for(u32 r=0;r<4;++r){const u64 w16=sit[r][240u+jl],w32a=sit[r][224u+jl],w32b=sit[r][232u+jl];
            pzdit(v[r][0],v[r][1],w16,p[r]);pzdit(v[r][2],v[r][3],w16,p[r]);pzdit(v[r][0],v[r][2],w32a,p[r]);pzdit(v[r][1],v[r][3],w32b,p[r]);}
        __syncwarp();
        #pragma unroll
        for(u32 r=0;r<4;++r){
            #pragma unroll
            for(u32 i=0;i<4;++i)scr[128u*r+fz_sw(q0+4u*i)]=v[r][i];}}
    __syncwarp();
    {const u32 q0=(lane&15u)+64u*(lane>>4),jl=2u*(lane&15u)+b;
        #pragma unroll
        for(u32 r=0;r<4;++r){
            #pragma unroll
            for(u32 i=0;i<4;++i)v[r][i]=scr[128u*r+fz_sw(q0+16u*i)];}
        #pragma unroll
        for(u32 r=0;r<4;++r){const u64 w64=sit[r][192u+jl],w128a=sit[r][128u+jl],w128b=sit[r][160u+jl];
            pzdit(v[r][0],v[r][1],w64,p[r]);pzdit(v[r][2],v[r][3],w64,p[r]);pzdit(v[r][0],v[r][2],w128a,p[r]);pzdit(v[r][1],v[r][3],w128b,p[r]);}
        __syncwarp();
        #pragma unroll
        for(u32 r=0;r<4;++r){
            #pragma unroll
            for(u32 i=0;i<4;++i)scr[128u*r+fz_sw(q0+16u*i)]=v[r][i];}}
    __syncwarp();
    {const u32 jl=2u*lane+b;
        #pragma unroll
        for(u32 r=0;r<4;++r){
            #pragma unroll
            for(u32 i=0;i<4;++i)v[r][i]=scr[128u*r+fz_sw(lane+32u*i)];}
        #pragma unroll
        for(u32 r=0;r<4;++r){pzdit(v[r][0],v[r][2],sit[r][jl],p[r]);pzdit(v[r][1],v[r][3],sit[r][64u+jl],p[r]);
            #pragma unroll
            for(u32 i=0;i<4;++i)v[r][i]=pxmsh(v[r][i],__ldg(tn[r]+2u*(lane+32u*i)),p[r]);}}
}
template<u32 FZ> __global__ void __launch_bounds__(128,3) sw_crtr(const u32* __restrict__ C,u32* __restrict__ output,
 const u32* __restrict__ bfrag,u32 N,u32 d,const u32* __restrict__ primes,const u64* __restrict__ stage_itw,const u64* __restrict__ t2n){
    constexpr u32 K=BCRT_K,SW=BCRT_SW,W=BCRT_W,L=BCRT_L,NW=64,NT=9,SST=36,F=BTC_F,RS=136;
    static_assert(K==16&&SW==16,"tensor CRT geometry");
    constexpr u32 WO=W/32,WS=W%32;
    static_assert(WO+1==L&&WS>0,"q = 2^W - d with W in the top limb");
    constexpr u32 STW=32*L;
    static_assert(4*32*SST+16*RS<=4*STW,"h stages + residue tile alias the output stages");
    __shared__ __align__(16) u32 bsh[NT*2*32*2];
    __shared__ __align__(16) u32 U[4*STW];
    const u32 warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,tig=lane&3;
    const u32 c=blockIdx.x>>1,b=blockIdx.x&1u;
    u32* const st=U+warp*(32*SST);   // h stage / R scratch of this warp
    u32* const pt=U+4*32*SST;        // residue tile [16 primes][RS]: coefficient k at word k
    for(u32 x=threadIdx.x;x<NT*2*32*2;x+=128)bsh[x]=bfrag[x];
    u32 acc[NW+2];
    #pragma unroll
    for(u32 w=0;w<NW+2;++w)acc[w]=0;
    const u32* const cb=C+u64(c)*256u+8u*lane;
    auto ldrow=[&](u32 pr,u32 j,u32 (&x)[8]){if(FZ&4){
            #pragma unroll
            for(u32 m=0;m<8;++m)x[m]=((pr*8u+j)*2654435761u+(lane*8u+m)*40503u+c)&0x3fffffffu;return;}
        const uint4* q=reinterpret_cast<const uint4*>(cb+u64(pr*8u+j)*N);const uint4 x0=q[0],x1=q[1];
        x[0]=x0.x;x[1]=x0.y;x[2]=x0.z;x[3]=x0.w;x[4]=x1.x;x[5]=x1.y;x[6]=x1.z;x[7]=x1.w;};
    auto rrow=[&](u32 pr,u32 j,const u32 (&x)[8]){
        u32 v[4];const u32 p=primes[pr];
        if(FZ&2){
            #pragma unroll
            for(u32 i=0;i<4;++i)v[i]=(b?x[2*i+1]:x[2*i])&0x3fffffffu;}
        else fz_rrow(x,v,st+128u*(pr&3u),stage_itw+pr*256u,t2n+u64(pr*8u+j)*512u+(c>>7)*256u+b,b,p,lane);
        #pragma unroll
        for(u32 i=0;i<4;++i)pt[pr*RS+lane+32u*i]=v[i];};
    u32 xa[8],xb[8];
    if(FZ&1){ldrow(4*warp,0,xa);ldrow(4*warp+1,0,xb);}
    #pragma unroll 1
    for(u32 j=0;j<8;++j){
        {   // R phase: rows of primes 4 warp .. 4 warp + 3, digit j
            if(FZ&8){u32 x4[4][8];
                if(FZ&1){
                    #pragma unroll
                    for(u32 m=0;m<8;++m){x4[0][m]=xa[m];x4[1][m]=xb[m];}
                    ldrow(4*warp+2,j,x4[2]);ldrow(4*warp+3,j,x4[3]);}
                else{
                    #pragma unroll
                    for(u32 r=0;r<4;++r)ldrow(4*warp+r,j,x4[r]);}
                u32 v4[4][4];const u64* sit4[4];const u64* tn4[4];u32 p4[4];
                #pragma unroll
                for(u32 r=0;r<4;++r){const u32 pr=4*warp+r;sit4[r]=stage_itw+pr*256u;tn4[r]=t2n+u64(pr*8u+j)*512u+(c>>7)*256u+b;p4[r]=primes[pr];}
                fz_rrow4(x4,v4,st,sit4,tn4,p4,b,lane);
                #pragma unroll
                for(u32 r=0;r<4;++r){
                    #pragma unroll
                    for(u32 i=0;i<4;++i)pt[(4*warp+r)*RS+lane+32u*i]=v4[r][i];}}
            else if(FZ&1){rrow(4*warp,j,xa);ldrow(4*warp+2,j,xa);rrow(4*warp+1,j,xb);ldrow(4*warp+3,j,xb);rrow(4*warp+2,j,xa);rrow(4*warp+3,j,xb);}
            else{ldrow(4*warp,j,xa);ldrow(4*warp+1,j,xb);rrow(4*warp,j,xa);ldrow(4*warp+2,j,xa);rrow(4*warp+1,j,xb);ldrow(4*warp+3,j,xb);rrow(4*warp+2,j,xa);rrow(4*warp+3,j,xb);}
        }
        __syncthreads();
        u32 a[2][2][4];
        {   const u32 pb=u32(__cvta_generic_to_shared(pt))+(RS*tig+32*warp+gid)*4;
            const u32 p0=pb,p1=pb+32,p2=pb+64,p3=pb+96;
#define FZ_FR(m,s,q0,q1) a[m][s][0]=swc_lds<(8*s)*RS*4>(q0);a[m][s][1]=swc_lds<(8*s)*RS*4>(q1);a[m][s][2]=swc_lds<(8*s+4)*RS*4>(q0);a[m][s][3]=swc_lds<(8*s+4)*RS*4>(q1);
            FZ_FR(0,0,p0,p1) FZ_FR(0,1,p0,p1) FZ_FR(1,0,p2,p3) FZ_FR(1,1,p2,p3)
#undef FZ_FR
        }
        __syncthreads();   // residue tile reads done (next R phase rewrites it)
        if((FZ&1)&&j<7){ldrow(4*warp,j+1,xa);ldrow(4*warp+1,j+1,xb);}
        {const u32 sb=u32(__cvta_generic_to_shared(st))+(gid*SST+tig)*4;
            swc_tile<0>(a,bsh,lane,sb);swc_tile<1>(a,bsh,lane,sb);swc_tile<2>(a,bsh,lane,sb);swc_tile<3>(a,bsh,lane,sb);swc_tile<4>(a,bsh,lane,sb);
            swc_tile<5>(a,bsh,lane,sb);swc_tile<6>(a,bsh,lane,sb);swc_tile<7>(a,bsh,lane,sb);swc_tile<8>(a,bsh,lane,sb);}
        __syncwarp();
        const u32* hs=st+lane*SST;
        const u64 S=u64(hs[32])+(u64(hs[33])<<16)+(u64(hs[34])<<32);
        const u32 alpha=u32((S+(u64(1)<<(F-1)))>>F);
        switch(j){case 0:swc_digit<0,true>(acc,hs,alpha);break;case 1:swc_digit<1,true>(acc,hs,alpha);break;case 2:swc_digit<2,true>(acc,hs,alpha);break;case 3:swc_digit<3,true>(acc,hs,alpha);break;
                  case 4:swc_digit<4,true>(acc,hs,alpha);break;case 5:swc_digit<5,true>(acc,hs,alpha);break;case 6:swc_digit<6,true>(acc,hs,alpha);break;default:swc_digit<7,true>(acc,hs,alpha);break;}
        __syncwarp();
    }
    {   u32 hw[10];
        #pragma unroll
        for(u32 w=0;w<10;++w){const u32 a_=w+WO<NW?acc[w+WO]:0,b_=w+WO+1<NW?acc[w+WO+1]:0;hw[w]=(a_>>WS)|(b_<<(32-WS));}
        acc[WO]&=(1u<<WS)-1;
        swc_fold(acc,hw,d);
        const u32 br=swc_subbias(acc);
        const bool hi=(acc[WO]>>WS)!=0u;
        acc[WO]&=(1u<<WS)-1;
        const u32 lt=swc_ltq(acc);
        swc_addx(acc,br?(0u-d):((hi||!lt)?d:0u),br);
        acc[WO]&=(1u<<WS)-1;
    }
    __syncthreads();   // all warps done with the h stages / residue tile (aliased by the output stages)
    {   u32* const so=U+warp*STW;
        #pragma unroll
        for(u32 w=0;w<L;++w)so[lane*L+w]=acc[w];
        __syncwarp();
        u32* const ob=output+(u64(64u*warp+b)*256u+c)*L+lane;   // coefficient rr: h = 2(32 warp + rr) + b
        #pragma unroll
        for(u32 rr=0;rr<32;++rr){ob[rr*512u*L]=so[rr*L+lane];if(lane<L-32)ob[rr*512u*L+32]=so[rr*L+32+lane];}
    }
}
#endif
