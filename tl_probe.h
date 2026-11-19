// Exit-time timeline probe (branch tl): host-side analysis + the TL module's device prefix.
// The TL module = g_kernel_prefix + tl_device_defs() + KERNEL_SOURCE: the 4 production stage kernels (bpo_first_x,
// bpo_middle_r16, bpo_last_cm3, bpo_crt_lean) gain one trailing parameter (tl_base, the first record slot of the launch)
// and a guard object: thread 0 stores %globaltimer + %smid at block entry; at kernel exit (every return path, destructor)
// one lane per active warp set does atomicMax(end, %globaltimer). Production source has TL_PARAM / TL_GUARD empty.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

inline const char* tl_device_defs(){return R"TLDEFS(
__device__ unsigned long long* g_tl_t;   // [2*slot] start ns, [2*slot+1] end ns (globaltimer)
__device__ unsigned* g_tl_sm;            // [slot] smid
__device__ __forceinline__ unsigned long long tl_now(){unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t));return t;}
// slot / buffer pointers re-read through volatile asm / loads at both ends: nothing is CSE'd across the kernel body, so the
// guard keeps at most the base argument live (register count reported per kernel in the hdr line)
__device__ __forceinline__ unsigned tl_slot(unsigned b){unsigned x,y,z,nx,ny;asm volatile("mov.u32 %0,%%ctaid.x;":"=r"(x));asm volatile("mov.u32 %0,%%ctaid.y;":"=r"(y));
    asm volatile("mov.u32 %0,%%ctaid.z;":"=r"(z));asm volatile("mov.u32 %0,%%nctaid.x;":"=r"(nx));asm volatile("mov.u32 %0,%%nctaid.y;":"=r"(ny));return b+x+nx*(y+ny*z);}
#ifndef TL_MODE
#define TL_MODE 5   // 1 start stamp (thread 0), 4 end stamp (barrier + thread 0; local B=128 TL/prod 1.011), 2 per-warp atomicMax end stamp (1.057; FHERMA_TL_MODE=3 to compare). The analysis needs a start and an end stamp.
#endif
struct TlGuard{
    unsigned b;
    __device__ __forceinline__ explicit TlGuard(unsigned base):b(base){
        if((TL_MODE&1)&&threadIdx.x==0){unsigned sm;asm volatile("mov.u32 %0,%%smid;":"=r"(sm));const unsigned s=tl_slot(b);
            (*(unsigned long long* volatile*)&g_tl_t)[2ull*s]=tl_now();(*(unsigned* volatile*)&g_tl_sm)[s]=sm;}}
    __device__ __forceinline__ ~TlGuard(){const unsigned m=__activemask();
        if((TL_MODE&2)&&(threadIdx.x&31u)==unsigned(__ffs(m)-1))atomicMax(&(*(unsigned long long* volatile*)&g_tl_t)[2ull*tl_slot(b)+1],tl_now());
        if(TL_MODE&4){__syncthreads();if(threadIdx.x==0)(*(unsigned long long* volatile*)&g_tl_t)[2ull*tl_slot(b)+1]=tl_now();}}   // 4: end = all threads at exit (one barrier, one store)
};
#define TL_PARAM ,unsigned tl_base
#define TL_GUARD TlGuard tl_guard_(tl_base);
// order-independent 2x64-bit hash of a u32 array (exactness check: production vs instrumented outputs)
extern "C" __global__ void tl_hash(const unsigned* x,unsigned long long n,unsigned long long* out){
    unsigned long long h1=0,h2=0;
    for(unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){
        unsigned long long v=((unsigned long long)x[i]<<32)^(i*0x9e3779b97f4a7c15ull);
        v^=v>>31;v*=0x7fb5d329728ea185ull;v^=v>>27;v*=0x81dadef4bc2dd44dull;v^=v>>33;h1+=v;h2+=v*(2*i+1);}
    for(int o=16;o;o>>=1){h1+=__shfl_xor_sync(~0u,h1,o);h2+=__shfl_xor_sync(~0u,h2,o);}
    if((threadIdx.x&31)==0){atomicAdd(out,h1);atomicAdd(out+1,h2);}
}
)TLDEFS";}

struct TlLaunch{int stage,product,chain,group;unsigned base,nblk,bx,smem;};

// One JSON line per item; all times in us from the first block start of the batch.
inline void tl_analyze(const std::vector<TlLaunch>& L,const std::vector<unsigned long long>& T,const std::vector<unsigned>& SM,
                       int sms,const double occ[4],int S,int B){
    static const char* SN[5]={"P1","P2","P3","CRT","-"};
    const size_t nb=SM.size();
    unsigned long long t0=~0ull,t1=0;size_t miss=0;
    for(size_t i=0;i<nb;++i){const auto s=T[2*i],e=T[2*i+1];if(!s||!e||e<s){++miss;continue;}t0=std::min(t0,s);t1=std::max(t1,e);}
    if(t0>=t1){std::printf("{\"tl\":\"error\",\"what\":\"no valid records\",\"miss\":%zu}\n",miss);std::fflush(stdout);return;}
    auto us=[&](unsigned long long x){return double(x-t0)*1e-3;};
    const double span=us(t1);
    std::string o;char b[512];
    auto put=[&](const char* fmt,auto... a){std::snprintf(b,sizeof b,fmt,a...);o+=b;};
    auto line=[&](){o+="}\n";std::fputs(o.c_str(),stdout);std::fflush(stdout);o.clear();};
    // timer resolution: smallest positive step between sorted block starts
    unsigned long long res=~0ull;{std::vector<unsigned long long> st;st.reserve(nb);for(size_t i=0;i<nb;++i)if(T[2*i]&&T[2*i+1]>=T[2*i])st.push_back(T[2*i]);
        std::sort(st.begin(),st.end());for(size_t i=1;i<st.size();++i)if(st[i]>st[i-1])res=std::min(res,st[i]-st[i-1]);}
    // per launch
    struct LS{double s=1e30,e=0,lst=0,dsum=0;int n=0;};std::vector<LS> ls(L.size());
    std::vector<float> dur[4];
    for(size_t j=0;j<L.size();++j){auto& x=ls[j];for(unsigned q=0;q<L[j].nblk;++q){const size_t i=L[j].base+q;const auto s=T[2*i],e=T[2*i+1];if(!s||!e||e<s)continue;
        const double a=us(s),z=us(e);x.s=std::min(x.s,a);x.e=std::max(x.e,z);x.lst=std::max(x.lst,a);x.dsum+=z-a;++x.n;dur[L[j].stage].push_back(float(z-a));}
        if(!x.n)x.s=x.e=x.lst=0;}
    auto pct=[](std::vector<float>& v,double p){if(v.empty())return 0.0;size_t k=size_t(p*(v.size()-1));std::nth_element(v.begin(),v.begin()+k,v.end());return double(v[k]);};
    // stage unions (launch intervals)
    auto uni=[&](auto pred){std::vector<std::pair<double,double>> iv;for(size_t j=0;j<L.size();++j)if(ls[j].n&&pred(j))iv.push_back({ls[j].s,ls[j].e});
        std::sort(iv.begin(),iv.end());double tot=0,cs=-1,ce=-1;for(auto& p:iv){if(p.first>ce){if(ce>cs)tot+=ce-cs;cs=p.first;ce=p.second;}else ce=std::max(ce,p.second);}if(ce>cs)tot+=ce-cs;return tot;};
    // ---- sweep over block events: weighted SM-slot occupancy per stage, idle SMs, low-occupancy intervals, histogram
    struct Ev{double t;int sm,st,d;};std::vector<Ev> ev;ev.reserve(2*nb);
    int maxsm=sms;
    for(size_t j=0;j<L.size();++j)for(unsigned q=0;q<L[j].nblk;++q){const size_t i=L[j].base+q;const auto s=T[2*i],e=T[2*i+1];if(!s||!e||e<s)continue;
        const int sm=int(SM[i]);maxsm=std::max(maxsm,sm+1);ev.push_back({us(s),sm,L[j].stage,1});ev.push_back({us(e),sm,L[j].stage,-1});}
    std::sort(ev.begin(),ev.end(),[](const Ev& x,const Ev& y){return x.t<y.t||(x.t==y.t&&x.d<y.d);});
    double bw=100;if(span/bw>170)bw=std::ceil(span/170/50)*50;   // ~<=170 bins
    const int nbin=int(span/bw)+1;std::vector<double> hs[4],hidle(nbin,0);for(auto& h:hs)h.assign(nbin,0);
    auto addb=[&](std::vector<double>& h,double a,double z,double v){while(a<z){int k=std::min(nbin-1,int(a/bw));double e=std::min(z,(k+1)*bw);if(e<=a)e=z;h[k]+=v*(e-a);a=e;}};
    std::vector<int> cnt(maxsm,0);int idle=sms;double w[4]={0,0,0,0},busy[4]={0,0,0,0},idle_t=0,lt50=0,lt25=0;
    struct Gap{double s,e,wi;};std::vector<Gap> gaps;bool in=false;Gap cur{0,0,0};
    double tp=0;
    for(size_t k=0;k<=ev.size();++k){const double t=k<ev.size()?ev[k].t:span;const double dt=t-tp;
        if(dt>0){const double W=w[0]+w[1]+w[2]+w[3];for(int s=0;s<4;++s){busy[s]+=w[s]*dt;if(w[s]>0)addb(hs[s],tp,t,w[s]);}
            idle_t+=idle*dt;if(idle)addb(hidle,tp,t,idle);
            if(W<0.5*sms){lt50+=dt;if(W<0.25*sms)lt25+=dt;if(!in){in=true;cur={tp,t,0};}cur.e=t;cur.wi+=W*dt;}
            else if(in){in=false;if(!gaps.empty()&&cur.s-gaps.back().e<1.0){gaps.back().e=cur.e;gaps.back().wi+=cur.wi;}else gaps.push_back(cur);}}
        if(k==ev.size())break;
        const Ev& x=ev[k];w[x.st]+=x.d/occ[x.st];if(x.sm<maxsm){if(x.d>0){if(cnt[x.sm]++==0&&x.sm<sms)--idle;}else{if(--cnt[x.sm]==0&&x.sm<sms)++idle;}}
        tp=t;}
    if(in)gaps.push_back(cur);
    // launch concurrency
    std::vector<std::pair<double,int>> le;for(size_t j=0;j<L.size();++j)if(ls[j].n){le.push_back({ls[j].s,1});le.push_back({ls[j].e,-1});}
    std::sort(le.begin(),le.end());double conc[5]={0,0,0,0,0};{int c=0;double p=0;for(auto& x:le){conc[std::min(c,4)]+=x.first-p;p=x.first;c+=x.second;}conc[0]+=span-p;}
    // ---- stage lines
    for(int s=0;s<4;++s){int n=0;double sp=0,spmax=0,disp=0,blk=0;std::vector<float> spv;double gsp[2]={0,0},gbl[2]={0,0};int gn[2]={0,0};
        for(size_t j=0;j<L.size();++j)if(L[j].stage==s&&ls[j].n){++n;const double x=ls[j].e-ls[j].s;sp+=x;spmax=std::max(spmax,x);spv.push_back(float(x));disp+=ls[j].lst-ls[j].s;blk=L[j].nblk;
            const int g=std::min(L[j].group,1);gsp[g]+=x;gbl[g]+=ls[j].dsum/ls[j].n;++gn[g];}
        if(!n)continue;auto& dv=dur[s];double dm=0;for(float x:dv)dm+=x;dm/=std::max<size_t>(1,dv.size());
        const double act=uni([&](size_t j){return L[j].stage==s;});
        put("{\"tl\":\"st\",\"s\":\"%s\",\"n\":%d,\"blk\":%.0f,\"occ\":%.0f,\"blk_us\":[%.1f,%.1f,%.1f,%.1f,%.1f]",SN[s],n,blk,occ[s],dm,pct(dv,0.1),pct(dv,0.5),pct(dv,0.9),pct(dv,1.0));
        put(",\"span_us\":[%.1f,%.1f,%.1f],\"disp_us\":%.1f,\"busy_sm_ms\":%.2f,\"act_ms\":%.2f,\"fill\":%.3f",sp/n,pct(spv,0.5),spmax,disp/n,busy[s]*1e-3,act*1e-3,act>0?busy[s]/(sms*act):0.0);
        if(gn[1])put(",\"g\":[[%.1f,%.1f],[%.1f,%.1f]]",gsp[0]/std::max(1,gn[0]),gbl[0]/std::max(1,gn[0]),gsp[1]/gn[1],gbl[1]/gn[1]);
        line();}
    const double cap=sms*span,ball=busy[0]+busy[1]+busy[2]+busy[3];
    put("{\"tl\":\"sum\",\"span_us\":%.1f,\"res_ns\":%llu,\"miss\":%zu,\"cap_sm_ms\":%.1f,\"busy_sm_ms\":%.1f,\"idle_sm_ms\":%.1f,\"part_sm_ms\":%.1f",span,res,miss,cap*1e-3,ball*1e-3,idle_t*1e-3,(cap-ball-idle_t)*1e-3);
    put(",\"lt50_ms\":%.2f,\"lt25_ms\":%.2f,\"conc_ms\":[%.2f,%.2f,%.2f,%.2f,%.2f]",lt50*1e-3,lt25*1e-3,conc[0]*1e-3,conc[1]*1e-3,conc[2]*1e-3,conc[3]*1e-3,conc[4]*1e-3);
    line();
    // ---- histogram lines (percent of SM capacity per bin)
    for(int s=0;s<5;++s){auto& h=s<4?hs[s]:hidle;put("{\"tl\":\"h\",\"s\":\"%s\",\"bin_us\":%.0f,\"v\":[",s<4?SN[s]:"idle",bw);
        for(int k=0;k<nbin;++k){const double width=std::min(bw,span-k*bw);put(k?",%d":"%d",int(std::lround(width>0?100.0*h[k]/(sms*width):0)));}
        o+="]";line();}
    // ---- chains
    std::vector<std::vector<size_t>> ch(S);for(size_t j=0;j<L.size();++j)if(ls[j].n&&L[j].chain>=0&&L[j].chain<S)ch[L[j].chain].push_back(j);
    for(auto& c:ch)std::sort(c.begin(),c.end(),[&](size_t x,size_t y){return ls[x].s<ls[y].s;});
    auto lab=[&](size_t j){std::string r=SN[L[j].stage];if(L[j].stage==1||L[j].stage==2)r+="g"+std::to_string(L[j].group);return r;};
    auto state=[&](int c,double t)->int{auto& v=ch[c];size_t lo=0,hi=v.size();while(lo<hi){size_t m=(lo+hi)/2;if(ls[v[m]].s<=t)lo=m+1;else hi=m;}
        if(lo==0)return -1;const size_t j=v[lo-1];return t<=ls[j].e?int(j):-1;};
    for(int c=0;c<S;++c){auto& v=ch[c];if(v.empty())continue;std::map<std::string,std::array<double,3>> tr;double wait=0,busyc=0;std::vector<double> p1s;
        for(size_t q=0;q<v.size();++q){busyc+=ls[v[q]].e-ls[v[q]].s;if(L[v[q]].stage==0)p1s.push_back(ls[v[q]].s);
            if(q){const double g=ls[v[q]].s-ls[v[q-1]].e;wait+=g;auto& a=tr[lab(v[q-1])+">"+lab(v[q])];a[0]+=g;a[1]=std::max(a[1],g);a[2]+=1;}}
        put("{\"tl\":\"ch\",\"c\":%d,\"n\":%zu,\"first_us\":%.1f,\"last_us\":%.1f,\"run_ms\":%.2f,\"wait_ms\":%.2f",c,v.size(),ls[v.front()].s,ls[v.back()].e,busyc*1e-3,wait*1e-3);
        if(p1s.size()>1)put(",\"period_us\":%.1f",(p1s.back()-p1s.front())/(p1s.size()-1));
        o+=",\"gap_us\":{";bool f=true;for(auto& kv:tr){put("%s\"%s\":[%.1f,%.1f]",f?"":",",kv.first.c_str(),kv.second[0]/kv.second[2],kv.second[1]);f=false;}o+="}";
        o+=",\"p1s\":[";for(size_t q=0;q<p1s.size();++q)put(q?",%.0f":"%.0f",p1s[q]);o+="]";line();}
    // ---- chain-pair state matrix (ms): rows chain 0 state, cols chain 1 state
    if(S==2){std::vector<double> bp{0,span};for(size_t j=0;j<L.size();++j)if(ls[j].n){bp.push_back(ls[j].s);bp.push_back(ls[j].e);}std::sort(bp.begin(),bp.end());
        double m[5][5]={};for(size_t q=1;q<bp.size();++q){const double a=bp[q-1],z=bp[q];if(z<=a)continue;const double t=(a+z)/2;const int x=state(0,t),y=state(1,t);
            m[x<0?4:L[x].stage][y<0?4:L[y].stage]+=z-a;}
        o+="{\"tl\":\"pair\",\"rc\":\"c0 x c1: P1,P2,P3,CRT,-\",\"ms\":[";for(int r=0;r<5;++r){o+=r?",[":"[";for(int c=0;c<5;++c)put(c?",%.2f":"%.2f",m[r][c]*1e-3);o+="]";}o+="]";line();}
    // ---- critical gaps (weighted occupancy < 50% of SM slots), longest first
    {std::vector<size_t> byS(L.size());for(size_t j=0;j<L.size();++j)byS[j]=j;std::sort(byS.begin(),byS.end(),[&](size_t x,size_t y){return ls[x].s<ls[y].s;});
     std::vector<Gap> g=gaps;std::sort(g.begin(),g.end(),[](const Gap& x,const Gap& y){return x.e-x.s>y.e-y.s;});
     double tot=0;int n2=0;for(auto& x:gaps){tot+=x.e-x.s;if(x.e-x.s>=2)++n2;}
     put("{\"tl\":\"gaps\",\"n\":%zu,\"n2us\":%d,\"tot_ms\":%.2f,\"f\":\"start,dur,occ%%,c0,c1,next\",\"top\":[",gaps.size(),n2,tot*1e-3);
     for(size_t q=0;q<g.size()&&q<14;++q){const double mid=(g[q].s+g[q].e)/2;std::string st[2];for(int c=0;c<2;++c){const int j=c<S?state(c,mid):-1;st[c]=j<0?"-":lab(size_t(j))+"p"+std::to_string(L[j].product);}
        std::string nx="none";for(size_t j:byS)if(ls[j].n&&ls[j].s>=g[q].s-0.01){nx=lab(j)+"p"+std::to_string(L[j].product)+"c"+std::to_string(L[j].chain)+"@"+std::to_string(int(std::lround(ls[j].s-g[q].s)));break;}
        put("%s[%.1f,%.1f,%d,\"%s\",\"%s\",\"%s\"]",q?",":"",g[q].s,g[q].e-g[q].s,int(std::lround(100*g[q].wi/std::max(1e-9,(g[q].e-g[q].s)*sms))),st[0].c_str(),st[1].c_str(),nx.c_str());}
     o+="]";line();}
    // ---- launch detail for a few products
    {const int pick[6]={0,1,B/2,B/2+1,B-2,B-1};o+="{\"tl\":\"lz\",\"f\":\"p,c,k,start,end,blk_us\",\"v\":[";bool f=true;
     for(int pi=0;pi<6;++pi){if(pi&&pick[pi]==pick[pi-1])continue;for(size_t j=0;j<L.size();++j)if(L[j].product==pick[pi]&&ls[j].n){
        put("%s[%d,%d,\"%s\",%.1f,%.1f,%.1f]",f?"":",",L[j].product,L[j].chain,lab(j).c_str(),ls[j].s,ls[j].e,ls[j].dsum/ls[j].n);f=false;}}
     o+="]";line();}
}
