// ---- Round 24 (ceil2): self-contained clean NTT pipeline (exit-time ntt_clean lines only; its own NVRTC module, never part
// of the production source). Per (product, prime): negacyclic convolution of length L = 2^19 of two residue vectors (p < 2^30,
// p = 1 mod 2^20), i.e. 2 forward + 1 inverse length-2^19 NTTs + pointwise, Harvey lazy butterflies with Shoup twiddles,
// Montgomery pointwise. Four-step L = 512 (rows r) x 1024 (columns c), element i = c + 1024 r:
//   nck1  column pass, forward (a and b): per column the 512-point DFT over r (nested 8 x 8 x 8, three register rounds of
//         radix-8, two smem exchanges), 4 adjacent columns per thread (16-byte accesses), then the L-level twiddle w^{c a} psi^c
//         (two factors: TG[m][c/4] shared by the 4 columns, TQ[m][c%4]); negacyclic twist folded (phi^{64 j} on input,
//         phi^t in the round-1 twiddle, psi^c in the L twiddle). Output row m holds column-DFT index a = bitrev9(m).
//   nck2  row pass, fused: one warp per stored row (1024 points = 32 x 32, lane t holds c = t + 32 j): forward a, forward b
//         (radix-32 in registers, 32x33 smem transpose, radix-32), Montgomery pointwise, inverse (mirror), store.
//   nck3  column pass, inverse: inverse L twiddle (scale 2^32 / L folded in) then the mirror of nck1, output in [0, p).
//   nck1s  nck1 with synthesized inputs (29-bit hash residues, no input DRAM read; production makes its residues on chip too).
//   nck1f / nck1fs / nck3f: the L twiddle from full tables [m][c] (one multiply per point, 8 B / point of L2 table reads) instead of two factors.
//   nck1w / nck1ws / nck3w (structure B): 512-point columns as 32 x 16 (radix-16 then radix-32 rounds, one smem exchange), full L tables.
//   nck2h  variant of nck2 with 16-byte row loads / stores through the warp's smem tile.
// 2p is read from the table (T.pp[32 + k]) so ptxas keeps it in a register: IADD3 u - v + 2p and VIADDMNMX reductions (-15..-19%
// SASS vs 2 * p re-derived by LEA).
#ifndef NC_CLEAN
#define NC_CLEAN 1
#define NCF __device__ __forceinline__
#define NC_L 524288u
#ifndef NC_K1MB
#define NC_K1MB 2
#endif
#ifndef NC_K2MB
#define NC_K2MB 2
#endif
#ifndef NC_KWMB
#define NC_KWMB 4
#endif
#ifndef NC_K3MB
#define NC_K3MB 3
#endif
struct NcTab {
    const uint2 *tw8f,*tw8i,*tws8f,*tws8i,*tw32f,*tw32i,*t512f,*t512i,*t64f,*t64i,*t1024f,*t1024i,*tgf,*tgi,*tqf,*tqi;
    const uint2 *tw16f,*tw16i,*t512wf,*t512wi,*tws16f,*tws16i;   // structure B columns: 16-pt DFT twiddles, round-1 twiddle w_512^{t a} phi^t (t<32), twist phi^{32 j}
    const unsigned* pp;   // [k] = p, [16 + k] = p^-1 mod 2^32, [32 + k] = 2p (loaded: keeps 2p in its own register, IADD3 u - v + 2p)
    const unsigned *tfw,*tfwp,*tiw,*tiwp;   // full L twiddle tables [k][m][c] (w, Shoup w'): forward w^{c a} psi^c, inverse w^{-c a} psi^{-c} 2^32 / L (TWF kernels)
};
struct NcGeo { unsigned long long in_pstride,out_pstride; unsigned k0,kc,rpw,pad; };   // product stride of the input / output slots (words), first prime, primes per launch, nck2 rows per warp
NCF unsigned nc_mulsh(unsigned a,uint2 w,unsigned p){const unsigned q=__umulhi(a,w.y);return a*w.x-q*p;}   // [0,2p) for any a
NCF unsigned nc_r2(unsigned x,unsigned p2){return min(x,x-p2);}
NCF void nc_gs(unsigned& u,unsigned& v,uint2 w,unsigned p,unsigned p2){const unsigned s=u+v,d=u-v+p2;u=nc_r2(s,p2);v=nc_mulsh(d,w,p);}   // in/out [0,2p)
NCF void nc_gs1(unsigned& u,unsigned& v,unsigned p2){const unsigned s=u+v,d=u-v+p2;u=nc_r2(s,p2);v=nc_r2(d,p2);}
NCF void nc_ct(unsigned& u,unsigned& v,uint2 w,unsigned p,unsigned p2){const unsigned a=nc_r2(u,p2),t=nc_mulsh(v,w,p);u=a+t;v=a-t+p2;}   // in/out [0,4p)
NCF void nc_ct1(unsigned& u,unsigned& v,unsigned p2){const unsigned a=nc_r2(u,p2),t=nc_r2(v,p2);u=a+t;v=a-t+p2;}
NCF unsigned nc_mont(unsigned a,unsigned b,unsigned p,unsigned pinv){const unsigned lo=a*b,hi=__umulhi(a,b),m=lo*pinv;return hi-__umulhi(m,p)+p;}   // a b 2^-32, a,b < 2p -> (0,2p)
// DIF (natural in, bit-reversed out) / DIT (bit-reversed in, natural out, x E) over E registers, Q independent columns sharing twiddles.
template<int E,int Q,int LEN,class TW> NCF void nc_dif_st(unsigned (&v)[E][Q],TW& tw,unsigned p,unsigned p2){
#pragma unroll
    for(int s=0;s<E;s+=2*LEN)
#pragma unroll
    for(int j=0;j<LEN;++j){
        if(j==0){
#pragma unroll
            for(int q=0;q<Q;++q)nc_gs1(v[s][q],v[s+LEN][q],p2);}
        else{const uint2 w=tw(j*(E/(2*LEN)));
#pragma unroll
            for(int q=0;q<Q;++q)nc_gs(v[s+j][q],v[s+j+LEN][q],w,p,p2);}}
    if constexpr(LEN>1)nc_dif_st<E,Q,LEN/2>(v,tw,p,p2);
}
template<int E,int Q,int LEN,class TW> NCF void nc_dit_st(unsigned (&v)[E][Q],TW& tw,unsigned p,unsigned p2){
#pragma unroll
    for(int s=0;s<E;s+=2*LEN)
#pragma unroll
    for(int j=0;j<LEN;++j){
        if(j==0){
#pragma unroll
            for(int q=0;q<Q;++q)nc_ct1(v[s][q],v[s+LEN][q],p2);}
        else{const uint2 w=tw(j*(E/(2*LEN)));
#pragma unroll
            for(int q=0;q<Q;++q)nc_ct(v[s+j][q],v[s+j+LEN][q],w,p,p2);}}
    if constexpr(2*LEN<E)nc_dit_st<E,Q,2*LEN>(v,tw,p,p2);
}
template<int E,int Q,class TW> NCF void nc_dif(unsigned (&v)[E][Q],TW tw,unsigned p,unsigned p2){nc_dif_st<E,Q,E/2>(v,tw,p,p2);}
template<int E,int Q,class TW> NCF void nc_dit(unsigned (&v)[E][Q],TW tw,unsigned p,unsigned p2){nc_dit_st<E,Q,1>(v,tw,p,p2);}
NCF uint2 nc_lds2(const uint2* p){uint2 r;asm volatile("ld.shared.v2.u32 {%0,%1},[%2];":"=r"(r.x),"=r"(r.y):"r"(unsigned(__cvta_generic_to_shared(p))));return r;}   // not hoisted / CSE'd (register pressure)
NCF unsigned nc_sw(unsigned slot){return slot^((slot>>3)&1u);}   // uint4 slot swizzle (rounds 1/2 touch slots s, s+1; round 3 s, s+8)
NCF uint4 nc_ld4(const unsigned* p){uint4 r;asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w):"l"(p));return r;}   // streamed input
NCF void nc_st4(unsigned* p,uint4 v){asm volatile("st.global.cs.v4.u32 [%0],{%1,%2,%3,%4};"::"l"(p),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w):"memory");}   // streamed output
// ---------------- structure A columns: 512 = 8 x 8 x 8, 4 columns / thread, 256 threads = 16 columns (blockIdx.x = strip 0..63)
// grid (64, 2 operands, units); unit u = gi * kc + ki: input a/b + gi * in_pstride + (k0 + ki) * L, output A/B + u * L.
template<int SYN,int TWF> NCF void nc_k1_body(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,const NcTab& T,const NcGeo& g){
    __shared__ __align__(16) uint4 S[512*4];
    const unsigned u=blockIdx.z,gi=u/g.kc,k=g.k0+u%g.kc;
    const unsigned* X=(blockIdx.y?b:a)+gi*g.in_pstride+size_t(k)*NC_L;unsigned* Y=(blockIdx.y?B:A)+size_t(u)*NC_L;
    const unsigned p=T.pp[k],p2=T.pp[32+k],tid=threadIdx.x,gq=tid&3,tq=tid>>2,G=blockIdx.x*4+gq;
    unsigned v[8][4];
    {   // round 1: rows tq + 64 j, twist phi^{64 j}, 8-pt DIF over j, twiddle w_512^{tq a1} phi^tq
#pragma unroll
        if(SYN){   // synthesized residues (no input read): 29-bit hash of (unit, operand, index) < 2^29 < p
            const unsigned h0=(u*2+blockIdx.y)*0x9E3779B9u;
#pragma unroll
            for(int j=0;j<8;++j)
#pragma unroll
                for(int q=0;q<4;++q){unsigned x=(h0^((tq+64*j)*1024+4*G+q))*0x85EBCA6Bu;x^=x>>15;v[j][q]=x>>3;}
        }else
#pragma unroll
        for(int j=0;j<8;++j){const uint4 x=nc_ld4(X+size_t(tq+64*j)*1024+4*G);v[j][0]=x.x;v[j][1]=x.y;v[j][2]=x.z;v[j][3]=x.w;}
        const uint2* ts=T.tws8f+k*8;
#pragma unroll
        for(int j=1;j<8;++j){const uint2 w=ts[j];
#pragma unroll
            for(int q=0;q<4;++q)v[j][q]=nc_mulsh(v[j][q],w,p);}
        const uint2* t8=T.tw8f+k*4;nc_dif<8,4>(v,[&](int i){return t8[i];},p,p2);
        const uint2* tw=T.t512f+size_t(k)*512+tq;
#pragma unroll
        for(int p1=0;p1<8;++p1){const uint2 w=tw[p1*64];
#pragma unroll
            for(int q=0;q<4;++q)v[p1][q]=nc_mulsh(v[p1][q],w,p);
            S[nc_sw(tq+64*p1)*4+gq]=make_uint4(v[p1][0],v[p1][1],v[p1][2],v[p1][3]);}
    }
    __syncthreads();
    {   // round 2: thread (t0, p1) over t1, twiddle w_64^{t0 a2}; in place (same slot set)
        const unsigned t0=tq&7,p1=tq>>3;
#pragma unroll
        for(int t1=0;t1<8;++t1){const uint4 x=S[nc_sw(t0+8*t1+64*p1)*4+gq];v[t1][0]=x.x;v[t1][1]=x.y;v[t1][2]=x.z;v[t1][3]=x.w;}
        const uint2* t8=T.tw8f+k*4;nc_dif<8,4>(v,[&](int i){return t8[i];},p,p2);
        const uint2* tw=T.t64f+k*64+t0;
#pragma unroll
        for(int p2_=0;p2_<8;++p2_){if(p2_){const uint2 w=tw[p2_*8];
#pragma unroll
            for(int q=0;q<4;++q)v[p2_][q]=nc_mulsh(v[p2_][q],w,p);}
            S[nc_sw(t0+8*p2_+64*p1)*4+gq]=make_uint4(v[p2_][0],v[p2_][1],v[p2_][2],v[p2_][3]);}
    }
    __syncthreads();
    {   // round 3: thread (p2, p1) over t0; stored row m = p3 + 8 p2 + 64 p1; L twiddle TQ[m][q] TG[m][G]
        const unsigned p2_=tq&7,p1=tq>>3;
#pragma unroll
        for(int t0=0;t0<8;++t0){const uint4 x=S[nc_sw(t0+8*p2_+64*p1)*4+gq];v[t0][0]=x.x;v[t0][1]=x.y;v[t0][2]=x.z;v[t0][3]=x.w;}
        const uint2* t8=T.tw8f+k*4;nc_dif<8,4>(v,[&](int i){return t8[i];},p,p2);
#pragma unroll
        for(int p3=0;p3<8;++p3){const unsigned m=p3+8*p2_+64*p1;
            if(TWF){const size_t o=(size_t(k)*512+m)*1024+4*G;const uint4 w=*reinterpret_cast<const uint4*>(T.tfw+o),wp=*reinterpret_cast<const uint4*>(T.tfwp+o);
                v[p3][0]=nc_mulsh(v[p3][0],make_uint2(w.x,wp.x),p);v[p3][1]=nc_mulsh(v[p3][1],make_uint2(w.y,wp.y),p);v[p3][2]=nc_mulsh(v[p3][2],make_uint2(w.z,wp.z),p);v[p3][3]=nc_mulsh(v[p3][3],make_uint2(w.w,wp.w),p);
            }else{const uint2* tqp=T.tqf+(size_t(k)*512+m)*4;const uint2 wg=T.tgf[(size_t(k)*512+m)*256+G];
#pragma unroll
            for(int q=1;q<4;++q)v[p3][q]=nc_mulsh(v[p3][q],tqp[q],p);
#pragma unroll
            for(int q=0;q<4;++q)v[p3][q]=nc_mulsh(v[p3][q],wg,p);}
            *reinterpret_cast<uint4*>(Y+size_t(m)*1024+4*G)=make_uint4(v[p3][0],v[p3][1],v[p3][2],v[p3][3]);}
    }
}
extern "C" __global__ void __launch_bounds__(256,NC_K1MB) nck1(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,NcTab T,NcGeo g){nc_k1_body<0,0>(a,b,A,B,T,g);}
extern "C" __global__ void __launch_bounds__(256,NC_K1MB) nck1s(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,NcTab T,NcGeo g){nc_k1_body<1,0>(a,b,A,B,T,g);}
extern "C" __global__ void __launch_bounds__(256,NC_K1MB) nck1f(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,NcTab T,NcGeo g){nc_k1_body<0,1>(a,b,A,B,T,g);}
extern "C" __global__ void __launch_bounds__(256,NC_K1MB) nck1fs(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,NcTab T,NcGeo g){nc_k1_body<1,1>(a,b,A,B,T,g);}
// grid (64, units): inverse column pass, C = unit buffers (u * L), out + gi * out_pstride + (k0 + ki) * L, result in [0, p).
template<int TWF> NCF void nc_k3_body(const unsigned* __restrict__ C,unsigned* __restrict__ out,const NcTab& T,const NcGeo& g){
    __shared__ __align__(16) uint4 S[512*4];
    const unsigned u=blockIdx.y,gi=u/g.kc,k=g.k0+u%g.kc;
    const unsigned* X=C+size_t(u)*NC_L;unsigned* Y=out+gi*g.out_pstride+size_t(k)*NC_L;
    const unsigned p=T.pp[k],p2=T.pp[32+k],tid=threadIdx.x,gq=tid&3,tq=tid>>2,G=blockIdx.x*4+gq;
    unsigned v[8][4];
    {   // inverse round 3: rows m = p3 + 8 p2 + 64 p1, inverse L twiddle, 8-pt DIT -> natural t0
        const unsigned p2_=tq&7,p1=tq>>3;
#pragma unroll
        for(int p3=0;p3<8;++p3){const unsigned m=p3+8*p2_+64*p1;const uint4 x=*reinterpret_cast<const uint4*>(X+size_t(m)*1024+4*G);v[p3][0]=x.x;v[p3][1]=x.y;v[p3][2]=x.z;v[p3][3]=x.w;
            if(TWF){const size_t o=(size_t(k)*512+m)*1024+4*G;const uint4 w=*reinterpret_cast<const uint4*>(T.tiw+o),wp=*reinterpret_cast<const uint4*>(T.tiwp+o);
                v[p3][0]=nc_mulsh(v[p3][0],make_uint2(w.x,wp.x),p);v[p3][1]=nc_mulsh(v[p3][1],make_uint2(w.y,wp.y),p);v[p3][2]=nc_mulsh(v[p3][2],make_uint2(w.z,wp.z),p);v[p3][3]=nc_mulsh(v[p3][3],make_uint2(w.w,wp.w),p);
            }else{const uint2* tqp=T.tqi+(size_t(k)*512+m)*4;const uint2 wg=T.tgi[(size_t(k)*512+m)*256+G];
#pragma unroll
            for(int q=1;q<4;++q)v[p3][q]=nc_mulsh(v[p3][q],tqp[q],p);
#pragma unroll
            for(int q=0;q<4;++q)v[p3][q]=nc_mulsh(v[p3][q],wg,p);}}
        const uint2* t8=T.tw8i+k*4;nc_dit<8,4>(v,[&](int i){return t8[i];},p,p2);
#pragma unroll
        for(int t0=0;t0<8;++t0)S[nc_sw(t0+8*p2_+64*p1)*4+gq]=make_uint4(v[t0][0],v[t0][1],v[t0][2],v[t0][3]);
    }
    __syncthreads();
    {   // inverse round 2: thread (t0, p1) over p2, twiddle w_64^{-t0 a2}, DIT -> natural t1, in place
        const unsigned t0=tq&7,p1=tq>>3;const uint2* tw=T.t64i+k*64+t0;
#pragma unroll
        for(int p2_=0;p2_<8;++p2_){const uint4 x=S[nc_sw(t0+8*p2_+64*p1)*4+gq];v[p2_][0]=x.x;v[p2_][1]=x.y;v[p2_][2]=x.z;v[p2_][3]=x.w;
            if(p2_){const uint2 w=tw[p2_*8];
#pragma unroll
                for(int q=0;q<4;++q)v[p2_][q]=nc_mulsh(v[p2_][q],w,p);}}
        const uint2* t8=T.tw8i+k*4;nc_dit<8,4>(v,[&](int i){return t8[i];},p,p2);
#pragma unroll
        for(int t1=0;t1<8;++t1)S[nc_sw(t0+8*t1+64*p1)*4+gq]=make_uint4(v[t1][0],v[t1][1],v[t1][2],v[t1][3]);
    }
    __syncthreads();
    {   // inverse round 1: thread tq over p1, twiddle w_512^{-tq a1} phi^-tq, DIT -> natural j, untwist phi^{-64 j}, reduce
        const uint2* tw=T.t512i+size_t(k)*512+tq;
#pragma unroll
        for(int p1=0;p1<8;++p1){const uint4 x=S[nc_sw(tq+64*p1)*4+gq];const uint2 w=tw[p1*64];v[p1][0]=nc_mulsh(x.x,w,p);v[p1][1]=nc_mulsh(x.y,w,p);v[p1][2]=nc_mulsh(x.z,w,p);v[p1][3]=nc_mulsh(x.w,w,p);}
        const uint2* t8=T.tw8i+k*4;nc_dit<8,4>(v,[&](int i){return t8[i];},p,p2);
        const uint2* ts=T.tws8i+k*8;
#pragma unroll
        for(int j=0;j<8;++j){
#pragma unroll
            for(int q=0;q<4;++q){unsigned x=j?nc_mulsh(v[j][q],ts[j],p):nc_r2(v[j][q],p2);x=x>=p?x-p:x;v[j][q]=x;}
            nc_st4(Y+size_t(tq+64*j)*1024+4*G,make_uint4(v[j][0],v[j][1],v[j][2],v[j][3]));}
    }
}
extern "C" __global__ void __launch_bounds__(256,NC_K3MB) nck3(const unsigned* __restrict__ C,unsigned* __restrict__ out,NcTab T,NcGeo g){nc_k3_body<0>(C,out,T,g);}
extern "C" __global__ void __launch_bounds__(256,NC_K3MB) nck3f(const unsigned* __restrict__ C,unsigned* __restrict__ out,NcTab T,NcGeo g){nc_k3_body<1>(C,out,T,g);}
// ---------------- structure B columns: 512 = 32 (t) x 16 (j), row r = t + 32 j; two register rounds, one smem exchange, full L
// twiddle tables. 128 threads = 16 columns (grid.x = 64 strips). Round 1: thread (gq 0..3 column quad, t 0..31) holds 4 columns x
// 16 rows, twist phi^{32 j}, 16-pt DIF over j, twiddle w_512^{t a} phi^t (a = bitrev4(pos)), smem slot t + 32 pos. Round 2:
// thread (cp 0..7 column pair, pos 0..15) holds 2 columns x 32 slots t + 32 pos, 32-pt DIF over t -> stored row m = q + 32 pos
// (column-DFT index bitrev9(m), as structure A). Smem [slot][16 columns], slot pairs swapped when bit 5 is set (conflict-free).
NCF unsigned nc_swb(unsigned slot){return slot^((slot>>5)&1u);}
template<int SYN> NCF void nc_k1w_body(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,const NcTab& T,const NcGeo& g){
    __shared__ __align__(16) unsigned S[512*16];
    const unsigned u=blockIdx.z,gi=u/g.kc,k=g.k0+u%g.kc;
    const unsigned* X=(blockIdx.y?b:a)+gi*g.in_pstride+size_t(k)*NC_L;unsigned* Y=(blockIdx.y?B:A)+size_t(u)*NC_L;
    const unsigned p=T.pp[k],p2=T.pp[32+k],tid=threadIdx.x;
    {   // round 1
        const unsigned gq=tid&3,t=tid>>2,G=blockIdx.x*4+gq;unsigned v[16][4];
        if(SYN){const unsigned h0=(u*2+blockIdx.y)*0x9E3779B9u;
#pragma unroll
            for(int j=0;j<16;++j)
#pragma unroll
                for(int q=0;q<4;++q){unsigned x=(h0^((t+32*j)*1024+4*G+q))*0x85EBCA6Bu;x^=x>>15;v[j][q]=x>>3;}
        }else
#pragma unroll
        for(int j=0;j<16;++j){const uint4 x=nc_ld4(X+size_t(t+32*j)*1024+4*G);v[j][0]=x.x;v[j][1]=x.y;v[j][2]=x.z;v[j][3]=x.w;}
        const uint2* ts=T.tws16f+k*32;
#pragma unroll
        for(int j=1;j<16;++j){const uint2 w=ts[j];
#pragma unroll
            for(int q=0;q<4;++q)v[j][q]=nc_mulsh(v[j][q],w,p);}
        const uint2* t16=T.tw16f+k*8;nc_dif<16,4>(v,[&](int i){return t16[i];},p,p2);
        const uint2* tw=T.t512wf+size_t(k)*512+t;
#pragma unroll
        for(int q1=0;q1<16;++q1){const uint2 w=tw[q1*32];
#pragma unroll
            for(int q=0;q<4;++q)v[q1][q]=nc_mulsh(v[q1][q],w,p);
            *reinterpret_cast<uint4*>(S+nc_swb(t+32*q1)*16+4*gq)=make_uint4(v[q1][0],v[q1][1],v[q1][2],v[q1][3]);}
    }
    __syncthreads();
    {   // round 2
        const unsigned cp=tid&7,pos=tid>>3,c0=blockIdx.x*16+2*cp;unsigned v[32][2];
#pragma unroll
        for(int t=0;t<32;++t){const uint2 x=*reinterpret_cast<const uint2*>(S+nc_swb(t+32*pos)*16+2*cp);v[t][0]=x.x;v[t][1]=x.y;}
        const uint2* t32=T.tw32f+k*16;nc_dif<32,2>(v,[&](int i){return t32[i];},p,p2);
#pragma unroll
        for(int q=0;q<32;++q){const unsigned m=q+32*pos;const size_t o=(size_t(k)*512+m)*1024+c0;
            const uint2 w=*reinterpret_cast<const uint2*>(T.tfw+o),wp=*reinterpret_cast<const uint2*>(T.tfwp+o);
            *reinterpret_cast<uint2*>(Y+size_t(m)*1024+c0)=make_uint2(nc_mulsh(v[q][0],make_uint2(w.x,wp.x),p),nc_mulsh(v[q][1],make_uint2(w.y,wp.y),p));}
    }
}
extern "C" __global__ void __launch_bounds__(128,NC_KWMB) nck1w(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,NcTab T,NcGeo g){nc_k1w_body<0>(a,b,A,B,T,g);}
extern "C" __global__ void __launch_bounds__(128,NC_KWMB) nck1ws(const unsigned* __restrict__ a,const unsigned* __restrict__ b,unsigned* __restrict__ A,unsigned* __restrict__ B,NcTab T,NcGeo g){nc_k1w_body<1>(a,b,A,B,T,g);}
extern "C" __global__ void __launch_bounds__(128,NC_KWMB) nck3w(const unsigned* __restrict__ C,unsigned* __restrict__ out,NcTab T,NcGeo g){
    __shared__ __align__(16) unsigned S[512*16];
    const unsigned u=blockIdx.y,gi=u/g.kc,k=g.k0+u%g.kc;
    const unsigned* X=C+size_t(u)*NC_L;unsigned* Y=out+gi*g.out_pstride+size_t(k)*NC_L;
    const unsigned p=T.pp[k],p2=T.pp[32+k],tid=threadIdx.x;
    {   // inverse round 2: rows m = q + 32 pos, inverse L twiddle, 32-pt DIT -> natural t
        const unsigned cp=tid&7,pos=tid>>3,c0=blockIdx.x*16+2*cp;unsigned v[32][2];
#pragma unroll
        for(int q=0;q<32;++q){const unsigned m=q+32*pos;const size_t o=(size_t(k)*512+m)*1024+c0;const uint2 x=*reinterpret_cast<const uint2*>(X+size_t(m)*1024+c0);
            const uint2 w=*reinterpret_cast<const uint2*>(T.tiw+o),wp=*reinterpret_cast<const uint2*>(T.tiwp+o);
            v[q][0]=nc_mulsh(x.x,make_uint2(w.x,wp.x),p);v[q][1]=nc_mulsh(x.y,make_uint2(w.y,wp.y),p);}
        const uint2* t32=T.tw32i+k*16;nc_dit<32,2>(v,[&](int i){return t32[i];},p,p2);
#pragma unroll
        for(int t=0;t<32;++t)*reinterpret_cast<uint2*>(S+nc_swb(t+32*pos)*16+2*cp)=make_uint2(v[t][0],v[t][1]);
    }
    __syncthreads();
    {   // inverse round 1: thread (gq, t) over pos, twiddle w_512^{-t a} phi^-t, 16-pt DIT -> natural j, untwist phi^{-32 j}, reduce
        const unsigned gq=tid&3,t=tid>>2,G=blockIdx.x*4+gq;unsigned v[16][4];const uint2* tw=T.t512wi+size_t(k)*512+t;
#pragma unroll
        for(int q1=0;q1<16;++q1){const uint4 x=*reinterpret_cast<const uint4*>(S+nc_swb(t+32*q1)*16+4*gq);const uint2 w=tw[q1*32];
            v[q1][0]=nc_mulsh(x.x,w,p);v[q1][1]=nc_mulsh(x.y,w,p);v[q1][2]=nc_mulsh(x.z,w,p);v[q1][3]=nc_mulsh(x.w,w,p);}
        const uint2* t16=T.tw16i+k*8;nc_dit<16,4>(v,[&](int i){return t16[i];},p,p2);
        const uint2* ts=T.tws16i+k*32;
#pragma unroll
        for(int j=0;j<16;++j){
#pragma unroll
            for(int q=0;q<4;++q){unsigned x=j?nc_mulsh(v[j][q],ts[j],p):nc_r2(v[j][q],p2);x=x>=p?x-p:x;v[j][q]=x;}
            nc_st4(Y+size_t(t+32*j)*1024+4*G,make_uint4(v[j][0],v[j][1],v[j][2],v[j][3]));}
    }
}
// ---------------- row pass (fused): grid (512 / (8 rpw), units), 256 threads = 8 warps, warp w handles rows blockIdx.x * 8 RPW + i * 8 + w.
// HV = 1: rows loaded / stored with 16-byte accesses through the warp's smem tile (structure B), else 4-byte coalesced.
template<int HV> NCF void nc_row_fwd(const unsigned* __restrict__ R,unsigned (&x)[32][1],unsigned* sx,const uint2* s32,const uint2* s1024,unsigned lane,unsigned p,unsigned p2){
    if(HV){   // 8 x LDG.128 per lane: lane holds c = 4 lane + 128 i + e; scatter to sx[c%32 (row t)][c/32 (j)]
#pragma unroll
        for(int i=0;i<8;++i){const uint4 q=*reinterpret_cast<const uint4*>(R+4*lane+128*i);const unsigned c=4*lane+128*i;
            sx[((c+0)&31)*33+((c+0)>>5)]=q.x;sx[((c+1)&31)*33+((c+1)>>5)]=q.y;sx[((c+2)&31)*33+((c+2)>>5)]=q.z;sx[((c+3)&31)*33+((c+3)>>5)]=q.w;}
        __syncwarp();
#pragma unroll
        for(int j=0;j<32;++j)x[j][0]=sx[lane*33+j];
        __syncwarp();
    }else{
#pragma unroll
        for(int j=0;j<32;++j)x[j][0]=R[lane+32*j];
    }
    nc_dif<32,1>(x,[&](int i){return nc_lds2(s32+i);},p,p2);
#pragma unroll
    for(int q=1;q<32;++q)x[q][0]=nc_mulsh(x[q][0],s1024[q*32+lane],p);
#pragma unroll
    for(int q=0;q<32;++q)sx[q*33+lane]=x[q][0];
    __syncwarp();
#pragma unroll
    for(int t=0;t<32;++t)x[t][0]=sx[lane*33+t];
    __syncwarp();
    nc_dif<32,1>(x,[&](int i){return nc_lds2(s32+i);},p,p2);
}
template<int HV> NCF void nc_k2_body(const unsigned* __restrict__ A,const unsigned* __restrict__ B,unsigned* __restrict__ C,const NcTab& T,const NcGeo& g){
    extern __shared__ __align__(16) uint2 nc_dyn[];   // dynamic: s1f[1024] s1i[1024] s32f[16] s32i[16] sX[8][32*33] (50432 B)
    uint2* s1f=nc_dyn;uint2* s1i=nc_dyn+1024;uint2* s32f=nc_dyn+2048;uint2* s32i=nc_dyn+2064;unsigned* sX0=reinterpret_cast<unsigned*>(nc_dyn+2080);
    const unsigned u=blockIdx.y,k=g.k0+u%g.kc,tid=threadIdx.x,lane=tid&31,w=tid>>5;
    for(unsigned i=tid;i<1024;i+=256){s1f[i]=T.t1024f[size_t(k)*1024+i];s1i[i]=T.t1024i[size_t(k)*1024+i];}
    if(tid<16)s32f[tid]=T.tw32f[k*16+tid];else if(tid<32)s32i[tid-16]=T.tw32i[k*16+tid-16];
    const unsigned p=T.pp[k],p2=T.pp[32+k],pinv=T.pp[16+k];
    __syncthreads();
    unsigned* sx=sX0+w*(32*33);
    for(unsigned r=0;r<g.rpw;++r){
        const unsigned m=blockIdx.x*8*g.rpw+r*8+w;const size_t off=size_t(u)*NC_L+size_t(m)*1024;
        unsigned x[32][1],y[32][1];
        nc_row_fwd<HV>(A+off,x,sx,s32f,s1f,lane,p,p2);
        nc_row_fwd<HV>(B+off,y,sx,s32f,s1f,lane,p,p2);
#pragma unroll
        for(int q=0;q<32;++q)x[q][0]=nc_mont(x[q][0],y[q][0],p,pinv);
        nc_dit<32,1>(x,[&](int i){return nc_lds2(s32i+i);},p,p2);
#pragma unroll
        for(int t=0;t<32;++t)sx[lane*33+t]=x[t][0];
        __syncwarp();
#pragma unroll
        for(int q=0;q<32;++q)x[q][0]=sx[q*33+lane];
        __syncwarp();
#pragma unroll
        for(int q=1;q<32;++q)x[q][0]=nc_mulsh(x[q][0],s1i[q*32+lane],p);
        nc_dit<32,1>(x,[&](int i){return nc_lds2(s32i+i);},p,p2);
        unsigned* Cr=C+off;
        if(HV){
#pragma unroll
            for(int j=0;j<32;++j)sx[lane*33+j]=x[j][0];
            __syncwarp();
#pragma unroll
            for(int i=0;i<8;++i){const unsigned c=4*lane+128*i;
                *reinterpret_cast<uint4*>(Cr+c)=make_uint4(sx[((c+0)&31)*33+((c+0)>>5)],sx[((c+1)&31)*33+((c+1)>>5)],sx[((c+2)&31)*33+((c+2)>>5)],sx[((c+3)&31)*33+((c+3)>>5)]);}
            __syncwarp();
        }else{
#pragma unroll
            for(int j=0;j<32;++j)Cr[lane+32*j]=x[j][0];
        }
    }
}
extern "C" __global__ void __launch_bounds__(256,NC_K2MB) nck2(const unsigned* __restrict__ A,const unsigned* __restrict__ B,unsigned* __restrict__ C,NcTab T,NcGeo g){nc_k2_body<0>(A,B,C,T,g);}
extern "C" __global__ void __launch_bounds__(256,NC_K2MB) nck2h(const unsigned* __restrict__ A,const unsigned* __restrict__ B,unsigned* __restrict__ C,NcTab T,NcGeo g){nc_k2_body<1>(A,B,C,T,g);}
#endif
