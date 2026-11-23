#pragma once
// Round 24 (ceil2): exit-time ntt_clean lines (ntt_clean.cuh, own NVRTC module). Host side: 16 primes p = k 2^20 + 1 < 2^30,
// twiddle tables, residue inputs (random, NC_SIN product slots per operand, products i -> slot i % NC_SIN, output slot
// i % NC_SOUT), one CUDA graph for the whole batch (B = 128 products x 16 primes: nck1 -> nck2 -> nck3 per launch group of
// G products x KC primes, groups round-robin over S streams, each stream with its own A / B / C unit buffers), timed as the
// median of R graph launches. Correctness: direct negacyclic convolution sums (no NTT) at 48 output indices for 4
// (slot, prime) pairs. Per-kernel split: the same graph with only nck1 / nck2 / nck3 launches.
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cuda.h>

namespace nc {
typedef unsigned long long u64;typedef unsigned u32;
inline u64 mulm(u64 a,u64 b,u64 p){return a*b%p;}
inline u64 powm(u64 a,u64 e,u64 p){u64 r=1;a%=p;while(e){if(e&1)r=r*a%p;a=a*a%p;e>>=1;}return r;}
inline bool isprime(u64 n){if(n<2)return false;for(u64 d=2;d*d<=n;++d)if(n%d==0)return false;return true;}
inline u32 brev(u32 x,int bits){u32 r=0;for(int i=0;i<bits;++i)r|=((x>>i)&1u)<<(bits-1-i);return r;}
struct HTab { u64 tw8f,tw8i,tws8f,tws8i,tw32f,tw32i,t512f,t512i,t64f,t64i,t1024f,t1024i,tgf,tgi,tqf,tqi,tw16f,tw16i,t512wf,t512wi,tws16f,tws16i,pp,tfw,tfwp,tiw,tiwp; };   // = NcTab
struct HGeo { u64 in_pstride,out_pstride; u32 k0,kc,rpw,pad; };   // = NcGeo
constexpr u32 L=1u<<19,K=16;
struct Api {
    decltype(&::cuMemAlloc) alloc;decltype(&::cuMemFree) mfree;decltype(&::cuMemcpyHtoD) h2d;decltype(&::cuMemcpyDtoH) d2h;decltype(&::cuLaunchKernel) launch;
    decltype(&::cuStreamCreate) screate;decltype(&::cuStreamDestroy) sdestroy;decltype(&::cuEventCreate) ecreate;decltype(&::cuEventRecord) erecord;
    decltype(&::cuEventSynchronize) esync;decltype(&::cuEventElapsedTime) elapsed;decltype(&::cuEventDestroy) edestroy;decltype(&::cuStreamWaitEvent) swait;
    decltype(&::cuStreamBeginCapture) bcap;decltype(&::cuStreamEndCapture) ecap;decltype(&::cuGraphInstantiateWithFlags) ginst;decltype(&::cuGraphLaunch) glaunch;
    decltype(&::cuGraphDestroy) gdestroy;decltype(&::cuGraphExecDestroy) gxdestroy;decltype(&::cuCtxSynchronize) csync;decltype(&::cuFuncSetAttribute) fattr;
    decltype(&::cuFuncGetAttribute) fgattr;decltype(&::cuStreamSynchronize) ssync;
    template<class G> explicit Api(G get){
        alloc=(decltype(alloc))get("cuMemAlloc_v2");mfree=(decltype(mfree))get("cuMemFree_v2");h2d=(decltype(h2d))get("cuMemcpyHtoD_v2");d2h=(decltype(d2h))get("cuMemcpyDtoH_v2");
        launch=(decltype(launch))get("cuLaunchKernel");screate=(decltype(screate))get("cuStreamCreate");sdestroy=(decltype(sdestroy))get("cuStreamDestroy_v2");
        ecreate=(decltype(ecreate))get("cuEventCreate");erecord=(decltype(erecord))get("cuEventRecord");esync=(decltype(esync))get("cuEventSynchronize");
        elapsed=(decltype(elapsed))get("cuEventElapsedTime");edestroy=(decltype(edestroy))get("cuEventDestroy_v2");swait=(decltype(swait))get("cuStreamWaitEvent");
        bcap=(decltype(bcap))get("cuStreamBeginCapture_v2");ecap=(decltype(ecap))get("cuStreamEndCapture");ginst=(decltype(ginst))get("cuGraphInstantiateWithFlags");
        glaunch=(decltype(glaunch))get("cuGraphLaunch");gdestroy=(decltype(gdestroy))get("cuGraphDestroy");gxdestroy=(decltype(gxdestroy))get("cuGraphExecDestroy");
        csync=(decltype(csync))get("cuCtxSynchronize");fattr=(decltype(fattr))get("cuFuncSetAttribute");fgattr=(decltype(fgattr))get("cuFuncGetAttribute");
        ssync=(decltype(ssync))get("cuStreamSynchronize");}
};
struct Fns { CUfunction k1=nullptr,k2=nullptr,k2h=nullptr,k3=nullptr,k1s=nullptr,k1f=nullptr,k1fs=nullptr,k3f=nullptr,k1w=nullptr,k1ws=nullptr,k3w=nullptr; };
struct Cfg { std::string name; u32 G=4,KC=2,S=2,rpw=2; bool k2h=false,syn=false,twf=false,pmaj=false; u32 colw=0; };   // colw: structure B columns, bit 0 nck1w, bit 1 nck3w (full tables)   // twf: full L twiddle tables (nck1f / nck3f); pmaj: prime-major group order   // syn: nck1s (inputs synthesized in-kernel, no input DRAM read)
struct Res { bool ok=false; double med=0,min=0,k1=0,k2=0,k3=0; int bad=-1; };

struct Pipe {
    Api& A;Fns F;u32 B=128,SIN=8,SOUT=8;std::vector<u32> P;
    CUdeviceptr tab=0,ftab=0,in_a=0,in_b=0,out=0;std::vector<CUdeviceptr> work;HTab ht{};u64 tab_bytes=0;std::string err;
    std::vector<CUstream> st;
    Pipe(Api& a,Fns f,u32 b):A(a),F(f),B(b){}
    ~Pipe(){A.csync();for(auto w:work)if(w)A.mfree(w);if(tab)A.mfree(tab);if(ftab)A.mfree(ftab);if(in_a)A.mfree(in_a);if(in_b)A.mfree(in_b);if(out)A.mfree(out);for(auto s:st)A.sdestroy(s);}
    static uint64_t shoup(u64 w,u64 p){return ((w<<32)/p);}
    bool init(){
        for(u64 k=1023;k>0&&P.size()<K;--k){const u64 p=(k<<20)+1;if(p<(1ull<<30)&&isprime(p))P.push_back(u32(p));}
        // tables: offsets in uint2 units
        struct Seg{u64* field;u64 n;};std::vector<Seg> segs={{&ht.tw8f,4},{&ht.tw8i,4},{&ht.tws8f,8},{&ht.tws8i,8},{&ht.tw32f,16},{&ht.tw32i,16},{&ht.t512f,512},{&ht.t512i,512},
            {&ht.t64f,64},{&ht.t64i,64},{&ht.t1024f,1024},{&ht.t1024i,1024},{&ht.tgf,512*256},{&ht.tgi,512*256},{&ht.tqf,512*4},{&ht.tqi,512*4},{&ht.tw16f,8},{&ht.tw16i,8},
            {&ht.t512wf,512},{&ht.t512wi,512},{&ht.tws16f,32},{&ht.tws16i,32}};
        u64 tot=0;std::vector<u64> off;for(auto& s:segs){off.push_back(tot);tot+=s.n*K;}
        const u64 pp_off=tot;tot+=2*K;   // 4K u32: p, p^-1 mod 2^32, 2p
        std::vector<u32> h(size_t(tot)*2,0);
        auto put=[&](size_t seg,u32 k,u64 idx,u64 w,u64 p){const u64 o=(off[seg]+u64(k)*segs[seg].n+idx)*2;h[o]=u32(w);h[o+1]=u32(shoup(w,p));};
        for(u32 k=0;k<K;++k){const u64 p=P[k];
            u64 g=2;while(powm(g,(p-1)/2,p)!=p-1)++g;   // quadratic non-residue -> psi = g^{(p-1)/2^20} has order 2^20
            const u64 psi=powm(g,(p-1)>>20,p),ipsi=powm(psi,p-2,p),w=psi*psi%p,iw=ipsi*ipsi%p;
            auto W=[&](long long e){e%=(long long)L;if(e<0)e+=L;return powm(w,u64(e),p);};auto Ps=[&](long long e){e%=(long long)(2*L);if(e<0)e+=2*L;return powm(psi,u64(e),p);};
            (void)iw;
            for(int i=0;i<4;++i){put(0,k,i,W(65536ll*i),p);put(1,k,i,W(-65536ll*i),p);}
            for(int j=0;j<8;++j){put(2,k,j,Ps(65536ll*j),p);put(3,k,j,Ps(-65536ll*j),p);}
            for(int i=0;i<16;++i){put(4,k,i,W(16384ll*i),p);put(5,k,i,W(-16384ll*i),p);}
            for(int i=0;i<8;++i){put(16,k,i,W(32768ll*i),p);put(17,k,i,W(-32768ll*i),p);}   // structure B: w_16^i
            for(int q1=0;q1<16;++q1)for(int t=0;t<32;++t){const long long e=1024ll*t*brev(q1,4);put(18,k,q1*32+t,mulm(W(e),Ps(1024ll*t),p),p);put(19,k,q1*32+t,mulm(W(-e),Ps(-1024ll*t),p),p);}
            for(int j=0;j<16;++j){put(20,k,j,Ps(32768ll*j),p);put(21,k,j,Ps(-32768ll*j),p);}
            for(int p1=0;p1<8;++p1)for(int t=0;t<64;++t){const long long e=1024ll*t*brev(p1,3);put(6,k,p1*64+t,mulm(W(e),Ps(1024ll*t),p),p);put(7,k,p1*64+t,mulm(W(-e),Ps(-1024ll*t),p),p);}
            for(int p2=0;p2<8;++p2)for(int t0=0;t0<8;++t0){const long long e=8192ll*t0*brev(p2,3);put(8,k,p2*8+t0,W(e),p);put(9,k,p2*8+t0,W(-e),p);}
            for(int q=0;q<32;++q)for(int t=0;t<32;++t){const long long e=512ll*t*brev(q,5);put(10,k,q*32+t,W(e),p);put(11,k,q*32+t,W(-e),p);}
            const u64 sc=mulm(powm((1ull<<32)%p,1,p),powm(L%p,p-2,p),p);   // 2^32 / L (Montgomery pointwise + inverse scale)
            {   // TG[m][G] = w^{4 G a} psi^{4G}, TQ[m][q] = w^{q a} psi^q, a = bitrev9(m); powers by running products (fast)
                for(u32 m=0;m<512;++m){const u64 a=brev(m,9);const u64 st=mulm(powm(w,4*a,p),Ps(4),p),sti=mulm(powm(w,(L-(4*a)%L)%L,p),Ps(-4),p);u64 x=1,y=sc;
                    for(u32 G=0;G<256;++G){put(12,k,u64(m)*256+G,x,p);put(13,k,u64(m)*256+G,y,p);x=x*st%p;y=y*sti%p;}
                    for(int q=0;q<4;++q){put(14,k,m*4+q,mulm(W((long long)(q*a)),Ps(q),p),p);put(15,k,m*4+q,mulm(W(-(long long)(q*a)),Ps(-q),p),p);}}
            }
            h[pp_off*2+k]=u32(p);u32 inv=u32(p);for(int i=0;i<5;++i)inv*=2u-u32(p)*inv;h[pp_off*2+16+k]=inv;h[pp_off*2+32+k]=u32(2*p);
        }
        tab_bytes=tot*8;if(A.alloc(&tab,tab_bytes)!=CUDA_SUCCESS){err="alloc tab";return false;}
        if(A.h2d(tab,h.data(),tab_bytes)!=CUDA_SUCCESS){err="h2d tab";return false;}
        for(size_t s=0;s<segs.size();++s)*segs[s].field=tab+off[s]*8;ht.pp=tab+pp_off*8;
        {   // full L twiddle tables (TWF): [k][m][c] = TG[m][c/4] TQ[m][c%4], as separate w / w' arrays (forward, inverse)
            const u64 n=u64(K)*L;if(A.alloc(&ftab,n*16)!=CUDA_SUCCESS){err="alloc ftab";return false;}
            std::vector<u32> fw(n),fp(n);
            for(int inv=0;inv<2;++inv){
                for(u32 k=0;k<K;++k){const u64 p=P[k];const size_t og=size_t(off[12+inv]+u64(k)*512*256)*2,oq=size_t(off[14+inv]+u64(k)*512*4)*2;
                    for(u32 m=0;m<512;++m)for(u32 c=0;c<1024;++c){const u64 w=u64(h[og+(size_t(m)*256+c/4)*2])*h[oq+(size_t(m)*4+c%4)*2]%p;const size_t o=(size_t(k)*512+m)*1024+c;fw[o]=u32(w);fp[o]=u32(shoup(w,p));}}
                if(A.h2d(ftab+u64(inv)*n*8,fw.data(),n*4)!=CUDA_SUCCESS||A.h2d(ftab+u64(inv)*n*8+n*4,fp.data(),n*4)!=CUDA_SUCCESS){err="h2d ftab";return false;}}
            ht.tfw=ftab;ht.tfwp=ftab+n*4;ht.tiw=ftab+n*8;ht.tiwp=ftab+n*12;}
        // inputs: random residues
        const u64 slot=u64(K)*L;
        if(A.alloc(&in_a,SIN*slot*4)!=CUDA_SUCCESS||A.alloc(&in_b,SIN*slot*4)!=CUDA_SUCCESS||A.alloc(&out,SOUT*slot*4)!=CUDA_SUCCESS){err="alloc io";return false;}
        std::vector<u32> buf(slot);u64 x=0x9E3779B97F4A7C15ull;
        for(int op=0;op<2;++op)for(u32 s=0;s<SIN;++s){for(u32 k=0;k<K;++k)for(u32 i=0;i<L;++i){x^=x<<13;x^=x>>7;x^=x<<17;buf[size_t(k)*L+i]=u32((x>>16)%P[k]);}
            if(A.h2d((op?in_b:in_a)+s*slot*4,buf.data(),slot*4)!=CUDA_SUCCESS){err="h2d in";return false;}}
        if(F.k2)A.fattr(F.k2,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,50432);
        if(F.k2h)A.fattr(F.k2h,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,50432);
        if((!F.k1f||!F.k1fs||!F.k3f)){F.k1f=F.k1fs=F.k3f=nullptr;}
        return true;}
    bool ensure(u32 S,u32 units){   // per stream: A, B, C unit buffers
        const u64 need=u64(units)*L*4*3;
        if(work.size()>=S&&wunits>=units)return true;
        A.csync();for(auto w:work)if(w)A.mfree(w);work.clear();for(auto s:st)A.sdestroy(s);st.clear();
        for(u32 s=0;s<S;++s){CUdeviceptr w=0;if(A.alloc(&w,need)!=CUDA_SUCCESS){err="alloc work";return false;}work.push_back(w);CUstream q=nullptr;A.screate(&q,CU_STREAM_NON_BLOCKING);st.push_back(q);}
        wunits=units;return true;}
    u32 wunits=0;
    // which: bit0 k1, bit1 k2, bit2 k3
    CUgraphExec build(const Cfg& c,int which){
        const u32 units=c.G*c.KC;if(!ensure(c.S,units))return nullptr;
        if(B%c.G||K%c.KC||SIN%c.G||SOUT%c.G||!c.rpw||64%c.rpw){err="bad cfg";return nullptr;}
        CUstream s0=st[0];if(A.bcap(s0,CU_STREAM_CAPTURE_MODE_RELAXED)!=CUDA_SUCCESS){err="capture";return nullptr;}
        std::vector<CUevent> evs;auto ev=[&]{CUevent e=nullptr;A.ecreate(&e,CU_EVENT_DISABLE_TIMING);evs.push_back(e);return e;};
        {CUevent f=ev();A.erecord(f,s0);for(u32 i=1;i<c.S;++i)A.swait(st[i],f,0);}
        const u64 slot=u64(K)*L;u32 gi=0;bool okl=true;
        const u32 ng=(B/c.G)*(K/c.KC);
        for(u32 q=0;q<ng;++q,++gi){const u32 p0=c.pmaj?(q%(B/c.G))*c.G:(q/(K/c.KC))*c.G,k0=c.pmaj?(q/(B/c.G))*c.KC:(q%(K/c.KC))*c.KC;
            CUstream s=st[gi%c.S];CUdeviceptr w=work[gi%c.S],Aw=w,Bw=w+u64(units)*L*4,Cw=w+2*u64(units)*L*4;
            CUdeviceptr ia=in_a+u64(p0%SIN)*slot*4,ib=in_b+u64(p0%SIN)*slot*4,oo=out+u64(p0%SOUT)*slot*4;
            HGeo g{slot,slot,k0,c.KC,c.rpw,0};HTab t=ht;
            if(which&1){void* pr[]={&ia,&ib,&Aw,&Bw,&t,&g};okl&=(c.colw&1)?A.launch(c.syn?F.k1ws:F.k1w,64,2,units,128,1,1,0,s,pr,nullptr)==CUDA_SUCCESS:A.launch(c.twf?(c.syn?F.k1fs:F.k1f):(c.syn?F.k1s:F.k1),64,2,units,256,1,1,0,s,pr,nullptr)==CUDA_SUCCESS;}
            if(which&2){void* pr[]={&Aw,&Bw,&Cw,&t,&g};okl&=A.launch(c.k2h?F.k2h:F.k2,512/(8*c.rpw),units,1,256,1,1,50432,s,pr,nullptr)==CUDA_SUCCESS;}
            if(which&4){void* pr[]={&Cw,&oo,&t,&g};okl&=(c.colw&2)?A.launch(F.k3w,64,units,1,128,1,1,0,s,pr,nullptr)==CUDA_SUCCESS:A.launch(c.twf?F.k3f:F.k3,64,units,1,256,1,1,0,s,pr,nullptr)==CUDA_SUCCESS;}
        }
        for(u32 i=1;i<c.S;++i){CUevent j=ev();A.erecord(j,st[i]);A.swait(s0,j,0);}
        CUgraph gr=nullptr;const CUresult r=A.ecap(s0,&gr);CUgraphExec ge=nullptr;
        if(r==CUDA_SUCCESS&&gr){if(A.ginst(&ge,gr,0)!=CUDA_SUCCESS)ge=nullptr;A.gdestroy(gr);}
        for(auto e:evs)A.edestroy(e);
        if(!okl||!ge){err=okl?"instantiate":"launch";if(ge)A.gxdestroy(ge);return nullptr;}
        return ge;}
    double time1(CUgraphExec ge,int R,double* mn=nullptr){
        CUevent a=nullptr,b=nullptr;A.ecreate(&a,CU_EVENT_DEFAULT);A.ecreate(&b,CU_EVENT_DEFAULT);std::vector<double> v;
        A.glaunch(ge,st[0]);A.ssync(st[0]);
        for(int i=0;i<R;++i){A.erecord(a,st[0]);A.glaunch(ge,st[0]);A.erecord(b,st[0]);A.esync(b);float ms=0;A.elapsed(&ms,a,b);v.push_back(ms*1e3);}
        A.edestroy(a);A.edestroy(b);std::sort(v.begin(),v.end());if(mn)*mn=v[0];return v[v.size()/2];}
    // direct negacyclic convolution sums at 48 indices for (slot, prime) pairs; returns number of mismatches
    int verify(){
        const u64 slot=u64(K)*L;int bad=0;std::vector<u32> a(L),b(L),o(L);
        const u32 ss[2]={0,SOUT-1},kk[2]={0,K-1};
        for(u32 si:ss)for(u32 k:kk){
            A.d2h(a.data(),in_a+(u64(si%SIN)*slot+u64(k)*L)*4,L*4);A.d2h(b.data(),in_b+(u64(si%SIN)*slot+u64(k)*L)*4,L*4);A.d2h(o.data(),out+(u64(si)*slot+u64(k)*L)*4,L*4);
            const u64 p=P[k];std::mt19937 rng(si*131+k);
            for(int t=0;t<48;++t){const u32 j=t<4?(t==0?0:t==1?L-1:t==2?1:L/2):rng()%L;u64 s=0;
                for(u32 i=0;i<L;++i){const u32 d=j>=i?j-i:j+L-i;u64 m=u64(a[i])*b[d]%p;if(j<i)m=m?p-m:0;s+=m;if(s>=(1ull<<62))s%=p;}
                s%=p;if(s!=o[j])++bad;}}
        return bad;}
    Res run(const Cfg& c,int R){
        Res r;CUgraphExec ge=build(c,7);if(!ge)return r;
        r.med=time1(ge,R,&r.min);A.gxdestroy(ge);
        for(int w=0;w<3;++w){CUgraphExec gk=build(c,1<<w);if(!gk)return r;const double t=time1(gk,std::max(3,R/2));A.gxdestroy(gk);(w==0?r.k1:w==1?r.k2:r.k3)=t;}
        ge=build(c,7);if(!ge)return r;time1(ge,1);A.gxdestroy(ge);A.csync();   // restore outputs of the full pipeline
        r.ok=true;return r;}
};
inline void info(Pipe& pp,int l2_bytes,int sms){
    Api& A=pp.A;const Fns& F=pp.F;
    auto rl=[&](CUfunction f,int& r,int& l){r=l=0;if(f){A.fgattr(&r,CU_FUNC_ATTRIBUTE_NUM_REGS,f);A.fgattr(&l,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);}};
    int r1,l1,r2,l2,r2h,l2h,r3,l3,r1s,l1s,r1f,l1f,r3f,l3f,r1w,l1w,r3w,l3w;rl(F.k1,r1,l1);rl(F.k2,r2,l2);rl(F.k2h,r2h,l2h);rl(F.k3,r3,l3);rl(F.k1s,r1s,l1s);rl(F.k1f,r1f,l1f);rl(F.k3f,r3f,l3f);rl(F.k1w,r1w,l1w);rl(F.k3w,r3w,l3w);
    // DRAM floor per product: inputs 2 x 16 x 2 MB read + output 16 x 2 MB written (syn: output only). L2 traffic (intermediates):
    // nck1 writes A, B (4 MB / prime), nck2 reads 4 + writes 2, nck3 reads 2. Ops per conv point: butterflies 3 x 19 / 2 = 28.5
    // (each 1 Shoup multiply except the w = 1 ones); extra Shoup multiplies: twist 2 x 7/8 + untwist 7/8, round twiddles, L twiddle
    // 3 x 7/4, pointwise 1 (Montgomery). add/sub ~2 per butterfly.
    const double pts=double(K)*L,bfly=pts*28.5,trivial=pts*3*(3*7.0/8+2*31.0/32);   // w = 1 butterflies per transform: 7 of 12 per 8-pt DFT (3 rounds), 31 of 80 per 32-pt (2 rounds)
    const double mm=bfly-trivial+pts*(3*7.0/8+3*(1+7.0/8)+3*(31.0/32)+3*7.0/4+1);
    std::printf("{\"diag\":\"sw\",\"k\":\"ntt_clean_info\",\"B\":%u,\"primes\":%u,\"L\":%u,\"in_slots\":%u,\"out_slots\":%u,\"l2_mb\":%.0f,\"sms\":%d,"
        "\"regs_local\":{\"k1\":[%d,%d],\"k1s\":[%d,%d],\"k2\":[%d,%d],\"k2h\":[%d,%d],\"k3\":[%d,%d],\"k1f\":[%d,%d],\"k3f\":[%d,%d],\"k1w\":[%d,%d],\"k3w\":[%d,%d]},\"dram_mb_per_product\":%.1f,\"dram_mb_per_product_syn\":%.1f,"
        "\"l2_mb_per_product\":%.1f,\"bfly_per_product_M\":%.1f,\"mulmod_per_product_M\":%.1f,\"mulmod_per_product_M_fulltab\":%.1f,\"tab_mb\":%.1f}\n",
        pp.B,K,L,pp.SIN,pp.SOUT,l2_bytes/1048576.0,sms,r1,l1,r1s,l1s,r2,l2,r2h,l2h,r3,l3,r1f,l1f,r3f,l3f,r1w,l1w,r3w,l3w,pts*4*3/1e6,pts*4/1e6,pts*4*(2+3+1)/1e6,bfly/1e6,mm/1e6,(mm-pts*3*0.75)/1e6,(pp.tab_bytes+pts*16)/1e6);
    std::fflush(stdout);}
inline Res line(Pipe& pp,const Cfg& c,int R,bool check){
    const auto t0=std::chrono::steady_clock::now();
    Res r=pp.run(c,R);
    if(!r.ok){std::printf("{\"diag\":\"sw\",\"k\":\"ntt_clean\",\"v\":\"%s\",\"error\":\"%s\"}\n",c.name.c_str(),pp.err.c_str());std::fflush(stdout);return r;}
    if(check&&!c.syn)r.bad=pp.verify();
    std::printf("{\"diag\":\"sw\",\"k\":\"ntt_clean\",\"v\":\"%s\",\"G\":%u,\"KC\":%u,\"S\":%u,\"rpw\":%u,\"k2h\":%d,\"twf\":%d,\"pmaj\":%d,\"colw\":%u,\"syn\":%d,\"med_us\":%.1f,\"min_us\":%.1f,\"us_per_product\":%.2f,\"k1_us\":%.1f,\"k2_us\":%.1f,\"k3_us\":%.1f,\"bad\":%d,\"n\":%d,\"s\":%.2f}\n",
        c.name.c_str(),c.G,c.KC,c.S,c.rpw,int(c.k2h),int(c.twf),int(c.pmaj),c.colw,int(c.syn),r.med,r.min,r.med/pp.B,r.k1,r.k2,r.k3,r.bad,R,std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());std::fflush(stdout);
    return r;}
inline Cfg mkcfg(u32 G,u32 KC,u32 S,bool h,bool y,bool f=false,bool pm=false,u32 rpw=2,u32 w=0){Cfg x;x.G=G;x.KC=KC;x.S=S;x.k2h=h;x.syn=y;x.twf=f||w;x.pmaj=pm;x.rpw=rpw;x.colw=w;x.name="g"+std::to_string(G)+"k"+std::to_string(KC)+"s"+std::to_string(S)+"r"+std::to_string(rpw)+(h?"h":"")+(w==3?"w":w==2?"fw":w==1?"wf":f?"f":"")+(pm?"p":"")+(y?"y":"");return x;}
// Sweep: every (G, KC, S) with residue inputs (structure A, full L tables, prime-major, rpw 4; first line verified), the same
// set synthesized (syn), then structure variants on the best of each (verified when the kernels differ); returns the best medians {residue, syn} in us (0 = none).
inline std::pair<double,double> sweep(Pipe& pp,const std::vector<Cfg>& base,int R,const std::function<bool()>& over={}){
    double best[2]={0,0};Cfg bc[2];bool first=true;
    for(int y=0;y<2;++y)for(Cfg c:base){
        if(over&&over()){std::printf("{\"diag\":\"sw\",\"k\":\"ntt_clean\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",c.name.c_str());std::fflush(stdout);continue;}
        c=mkcfg(c.G,c.KC,c.S,false,y,true,true,4);Res r=line(pp,c,R,first&&!y);if(!y)first=false;
        if(r.ok&&r.bad<=0&&(best[y]==0||r.med<best[y])){best[y]=r.med;bc[y]=c;}}
    // structure variants on the best config of each family: full L twiddle tables (f), + prime-major order (fp), 16-byte rows (h)
    for(int y=0;y<2;++y){if(best[y]<=0)continue;const Cfg b=bc[y];
        // k2h, twf, pmaj, rpw, colw (bit 0: structure B nck1w, bit 1: structure B nck3w)
        const unsigned vv[][5]={{0,0,0,2,0},{0,0,1,4,0},{0,1,0,4,0},{0,1,1,2,0},{0,1,1,1,0},{1,1,1,4,0},{0,1,1,4,3},{0,1,1,4,2},{0,1,1,4,1}};
        for(const auto& x:vv){if(over&&over())break;if(x[0]&&!pp.F.k2h)continue;if(x[1]&&!pp.F.k1f)continue;if(((x[4]&1)&&!pp.F.k1w)||((x[4]&2)&&!pp.F.k3w))continue;
            Cfg c=mkcfg(b.G,b.KC,b.S,x[0],y,x[1],x[2],x[3],x[4]);Res r=line(pp,c,R,!y&&(x[0]||x[1]||x[4]));if(r.ok&&r.bad<=0&&r.med<best[y]){best[y]=r.med;bc[y]=c;}}
        std::printf("{\"diag\":\"sw\",\"k\":\"ntt_clean_best\",\"syn\":%d,\"v\":\"%s\",\"med_us\":%.1f,\"us_per_product\":%.2f}\n",y,bc[y].name.c_str(),best[y],best[y]/pp.B);std::fflush(stdout);}
    return {best[0],best[1]};}
}   // namespace nc
