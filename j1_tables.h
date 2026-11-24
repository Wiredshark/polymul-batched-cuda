#pragma once
// J=1 host tables (init, untimed), regenerated from the formulas of j1/params.py / gen_tables.py (no shipped binaries):
//   per prime: p (the 118 largest p = 1 mod 2^17 below 2^30, descending), psi (smallest g >= 2 with x = g^((p-1)/2N),
//   x^N = -1), s = (M/p)^-1 mod p, fB = 2^32 N^-1 s mod p, pinv = -p^-1 mod 2^32, bm = floor(2^61/p), R = floor(2^45/p);
//   NTT: tw1 [2][256] Shoup words psi^(+-brev16(j)), thi [2][256] psi^(+-256 h), tlo [2][256] floor(psi^(+-l) 2^64/p);
//   conversion: fwdA / fwdB [224][472] u8, backM [480][224] u8, backF [480][8] u8, D = (-M) mod q (55 limbs).
// j1/j1_tabletest.cpp checks every table against the shipped j1/tables binaries.
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace j1t {
typedef uint32_t u32;typedef uint64_t u64;typedef unsigned __int128 u128;
inline u32 pw(u64 b,u64 e,u64 p){u64 r=1;b%=p;while(e){if(e&1)r=r*b%p;b=b*b%p;e>>=1;}return u32(r);}
inline bool isprime(u32 n){if(n<2)return false;for(u32 a:{2u,3u,5u,7u})if(n%a==0)return n==a;u32 d=n-1,s=0;while(!(d&1)){d>>=1;++s;}
    for(u32 a:{2u,3u,5u,7u}){u64 x=pw(a,d,n);if(x==1||x==n-1)continue;bool ok=false;for(u32 r=1;r<s;++r){x=x*x%n;if(x==n-1){ok=true;break;}}if(!ok)return false;}return true;}
inline u32 brev16(u32 j){u32 r=0;for(int b=0;b<16;++b)r|=((j>>b)&1u)<<(15-b);return r;}
// residues mod q = 2^W - delta as little-endian u32 vectors
struct ModQ{
    u32 W,delta,L;std::vector<u32> q;
    ModQ(u32 W_,u32 delta_):W(W_),delta(delta_),L((W_+31)/32){q.assign(L,0xffffffffu);q[L-1]=(W%32)?((1u<<(W%32))-1u):0xffffffffu;
        q[0]-=delta-1u;}   // 2^W - 1 - (delta - 1), no borrow (0 < delta < 2^32)
    static void trim(std::vector<u32>& x){while(x.size()>1&&!x.back())x.pop_back();}
    static u32 bits(const std::vector<u32>& x){for(size_t i=x.size();i--;)if(x[i])return u32(i*32+32-__builtin_clz(x[i]));return 0;}
    bool geq(const std::vector<u32>& a,const std::vector<u32>& b)const{size_t n=std::max(a.size(),b.size());for(size_t i=n;i--;){u32 x=i<a.size()?a[i]:0,y=i<b.size()?b[i]:0;if(x!=y)return x>y;}return true;}
    void sub(std::vector<u32>& a,const std::vector<u32>& b)const{u64 br=0;for(size_t i=0;i<a.size();++i){u64 d=u64(a[i])-(i<b.size()?b[i]:0)-br;a[i]=u32(d);br=(d>>63)&1;}}
    // x -> x mod q (canonical, L limbs), by pseudo-Mersenne folding
    std::vector<u32> reduce(std::vector<u32> x)const{
        trim(x);
        while(bits(x)>W){std::vector<u32> lo(L,0),hi;for(u32 i=0;i<L&&i<x.size();++i)lo[i]=x[i];lo[L-1]&=(W%32)?((1u<<(W%32))-1u):0xffffffffu;
            for(u32 i=W;i<bits(x);i+=32){u32 w=0;for(u32 b=0;b<32&&i+b<bits(x);++b)w|=((x[(i+b)/32]>>((i+b)%32))&1u)<<b;hi.push_back(w);}
            std::vector<u32> r(std::max<size_t>(L,hi.size()+1)+1,0);for(u32 i=0;i<L;++i)r[i]=lo[i];
            u64 c=0;for(size_t i=0;i<r.size();++i){u64 v=u64(r[i])+(i<hi.size()?u64(hi[i])*delta:0)+c;r[i]=u32(v);c=v>>32;}
            x=r;trim(x);}
        x.resize(L,0);while(geq(x,q))sub(x,q);return x;}
    std::vector<u32> mul(const std::vector<u32>& a,const std::vector<u32>& b)const{std::vector<u32> r(a.size()+b.size()+1,0);
        for(size_t i=0;i<a.size();++i){u64 c=0;for(size_t j=0;j<b.size();++j){u64 v=u64(a[i])*b[j]+r[i+j]+c;r[i+j]=u32(v);c=v>>32;}r[i+b.size()]+=u32(c);}return reduce(r);}
    u32 mod_small(u32 p)const{u64 r=0;for(size_t i=q.size();i--;)r=((r<<32)|q[i])%p;return u32(r);}
};
struct Tables{
    u32 N=0,K=0;std::vector<u32> primes,psi,s,fB,pinv,bm,R,thi,D;std::vector<u64> tw1,tlo;
    std::vector<uint8_t> fwdA,fwdB,backM,backF;   // [224][4K] / [480][224] / [480][8]
    // m16n8k32 .u8 B fragments (gen_tables.py pack_b; fwd* of the column-permuted matrices fwd*_perm [224][480]):
    // fragA / fragB [7][60][32][2], fragM [15][28][32][2], fragF [15][1][32][2] u32
    std::vector<u32> fragA,fragB,fragM,fragF;
};
// gen_tables.py pack_b: out[ks][nt][lane][h] = bytes B[32 ks + 16 h + 4 tg + e][8 nt + g], e = 0..3 (lane = 4 g + tg)
inline std::vector<u32> pack_b(const std::vector<uint8_t>& B,u32 kdim,u32 ndim){
    std::vector<u32> o(size_t(kdim/32)*(ndim/8)*64);
    for(u32 ks=0;ks<kdim/32;++ks)for(u32 nt=0;nt<ndim/8;++nt)for(u32 lane=0;lane<32;++lane)for(u32 h=0;h<2;++h){
        const u32 g=lane>>2,tg=lane&3,kb=32*ks+16*h+4*tg;u32 v=0;
        for(u32 e=0;e<4;++e)v|=u32(B[size_t(kb+e)*ndim+8*nt+g])<<(8*e);
        o[((size_t(ks)*(ndim/8)+nt)*32+lane)*2+h]=v;}
    return o;}
// gen_tables.py fwd_permuted: prime k, plane j -> column 8 (2 (k/4) + (j>>1)) + 2 (k%4) + (j&1), 480 columns
inline std::vector<uint8_t> fwd_permuted(const std::vector<uint8_t>& B,u32 K){
    const u32 nq=(K+3)/4;std::vector<uint8_t> P(size_t(224)*nq*16,0);
    for(u32 r=0;r<224;++r)for(u32 k=0;k<K;++k)for(u32 j=0;j<4;++j)P[size_t(r)*nq*16+8*(2*(k/4)+(j>>1))+2*(k%4)+(j&1)]=B[size_t(r)*4*K+4*k+j];
    return P;}
inline Tables make_tables(u32 N,u32 K,u32 W,u32 delta){
    constexpr u32 KF=224,NB=219,KB=480,NBACK=224,NFRAC=8;
    if(N!=65536u||W!=1747u||4*K+1>KB)throw std::runtime_error("j1 tables: unsupported point");
    Tables T;T.N=N;T.K=K;ModQ mq(W,delta);
    for(u32 c=((1u<<30)-2u)/(1u<<17)*(1u<<17)+1u;T.primes.size()<K&&c>(1u<<29);c-=1u<<17)if(isprime(c))T.primes.push_back(c);
    if(T.primes.size()!=K)throw std::runtime_error("j1 tables: prime pool");
    T.psi.resize(K);T.s.resize(K);T.fB.resize(K);T.pinv.resize(K);T.bm.resize(K);T.R.resize(K);
    T.tw1.resize(size_t(K)*512);T.thi.resize(size_t(K)*512);T.tlo.resize(size_t(K)*512);
    T.fwdA.assign(size_t(KF)*4*K,0);T.fwdB.assign(size_t(KF)*4*K,0);T.backM.assign(size_t(KB)*NBACK,0);T.backF.assign(size_t(KB)*NFRAC,0);
    for(u32 k=0;k<K;++k){const u32 p=T.primes[k];
        for(u64 g=2;;++g){u32 x=pw(g,(p-1)/(2*N),p);if(pw(x,N,p)==p-1){T.psi[k]=x;break;}}
        u64 mk=1;for(u32 j=0;j<K;++j)if(j!=k)mk=mk*(T.primes[j]%p)%p;
        T.s[k]=pw(mk,p-2,p);T.fB[k]=u32(u64(pw(2,32,p))*pw(N,p-2,p)%p*T.s[k]%p);
        u32 iv=p;for(int i=0;i<5;++i)iv*=2u-p*iv;T.pinv[k]=0u-iv;T.bm[k]=u32((u64(1)<<61)/p);T.R[k]=u32((u64(1)<<45)/p);
        const u32 ps=T.psi[k],pi=pw(ps,p-2,p);
        for(u32 dd=0;dd<2;++dd){const u32 g=dd?pi:ps;
            for(u32 j=0;j<256;++j){u32 w=pw(g,brev16(j),p);T.tw1[k*512+dd*256+j]=u64(w)|(((u64(w)<<32)/p)<<32);}
            for(u32 i=0;i<256;++i){T.thi[k*512+dd*256+i]=pw(g,256ull*i,p);u32 w=pw(g,i,p);T.tlo[k*512+dd*256+i]=u64((u128(w)<<64)/p);}}
        const u32 negq=(p-mq.mod_small(p))%p;
        for(u32 op=0;op<2;++op){const u64 f=op?T.fB[k]:1u;auto& B=op?T.fwdB:T.fwdA;
            for(u32 b=0;b<=NB;++b){const u32 c=b<NB?u32(u64(pw(2,8ull*b,p))*f%p):u32(u64(negq)*f%p);
                for(u32 j=0;j<4;++j)B[size_t(b)*4*K+4*k+j]=uint8_t(c>>(8*j));}}
        for(u32 i=0;i<4;++i)for(u32 j=0;j<2;++j)T.backF[size_t(4*k+i)*NFRAC+i+j]=uint8_t(T.R[k]>>(8*j));
    }
    // M/p_k mod q from prefix / suffix products; C_{k,i} = (M/p_k) 256^i mod q; D = (-M) mod q
    std::vector<std::vector<u32>> pre(K+1),suf(K+1);pre[0]={1};suf[K]={1};
    for(u32 k=0;k<K;++k)pre[k+1]=mq.mul(pre[k],{T.primes[k]});
    for(u32 k=K;k--;)suf[k]=mq.mul(suf[k+1],{T.primes[k]});
    auto put=[&](u32 row,const std::vector<u32>& v){for(u32 c=0;c<NB;++c)T.backM[size_t(row)*NBACK+c]=uint8_t(v[c/4]>>(8*(c%4)));};
    for(u32 k=0;k<K;++k){std::vector<u32> C=mq.mul(pre[k],suf[k+1]);
        for(u32 i=0;i<4;++i){put(4*k+i,C);C=mq.mul(C,{256u});}}
    std::vector<u32> Mq=pre[K];T.D=Mq;bool zero=true;for(u32 x:Mq)if(x)zero=false;
    if(zero)T.D.assign(mq.L,0);else{T.D=mq.q;mq.sub(T.D,Mq);}
    put(4*K,T.D);
    T.fragA=pack_b(fwd_permuted(T.fwdA,K),KF,(K+3)/4*16);T.fragB=pack_b(fwd_permuted(T.fwdB,K),KF,(K+3)/4*16);
    T.fragM=pack_b(T.backM,KB,NBACK);T.fragF=pack_b(T.backF,KB,NFRAC);
    return T;
}
}  // namespace j1t
