#pragma once
// Exit-time on-target variant sweep, round h30s (on 590b89d = pv45 + m16 default). After the timed calls, kernel variants
// are compiled with NVRTC (worker threads, same arch option as the production module) and timed on the captured launch
// geometry / data of the last timed call against the production kernels (interleaved A/B reps). The timed path, the
// production module and its PTX are untouched (production kernels are only replayed).
//  A) h30: the mixed prime set of 9d2a5bd (12 primes < 2^30 + 4 of 31 bits, tight CRT bound, tensor CRT F = 45) with the
//     Harvey-lazy butterflies of 2297ca6 (ported onto the current P1 pbq / P2 r16 / P3 tn2 kernels) for the < 2^30 primes.
//     The h30 module = h30 prefix (BPO_MIXED 1, BPO_NH30 12) + the production kernel source; its tables are a second table
//     set (State::build_h30) on the production context. h30 kernels replay the production traces with every table pointer
//     remapped to the h30 tables (workspaces unchanged). "h30sh" = same primes and tables, Shoup butterflies (BPO_NH30 0):
//     separates the prime-set effect from the Harvey effect. Per-kernel lines are timing only (other residues: no ex);
//     the chain lines check the CRT output of products 0 and 1 against the production chain (ex = 1: bit-identical).
//  B) occupancy twins (launch-bounds min-blocks; same code): P2 bpo_middle_r16 mb12 / mb10 / mb9, P3 bpo_last_cm3 mb8.
//     "ctrl" = production kernel recompiled here (recompile control). occ = blocks/SM from the occupancy API.
// Line keys: {"diag":"sw","k":stage,"v":variant,...}: med_us/min_us = 2 chains (products 0 and 1 concurrently, as in
// production; c1_us = product 0 alone), cold (2xL2 dirty flush), P2/P3/CRT right after their predecessor kernel (untimed);
// vs_prod / c1_vs = median over adjacent pairs of variant/production time (< 1 = faster; "prod" = production against itself =
// noise floor); ex 1 = output identical to production; r = registers, lm = local (spill) bytes per thread, occ = blocks/SM.
// "chain" lines: products 0+1 full chains (P1 .. CRT, current m16 + CRT-prefetch schedule = the captured traces) with the
// named substitutions; cf: = after a clean (read-only) L2 flush; x: = plus products 2/3's P1 (benefits from the CRT prefetch).
// FHERMA_SWEEP=0 disables; FHERMA_SWEEP_H30=0 skips the h30 part; FHERMA_SWEEP_S = time budget (s, default 50).
#include <thread>
#include <atomic>
#include <memory>
#include <cmath>
#include <sstream>

inline bool sweep_env(){const char* e=std::getenv("FHERMA_SWEEP");return !e||*e!='0';}
inline bool sweep_h30_env(){const char* e=std::getenv("FHERMA_SWEEP_H30");return !e||*e!='0';}

template<class C> struct Sweep {
    C& d; KProf& kp; const unsigned L;
    std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now();double budget=50.0;
    struct Prog { std::string name,arch,src; std::vector<std::string> kn; bool dd=false; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t_rtc=0,t_jit=0; };
    std::vector<Prog> progs; std::vector<std::thread> th; std::string arch,h30info;
    struct V { std::string v; int prog=-1,ki=-1; bool h30=false; CUfunction fn=nullptr; double med=0,mn=0,c1=0,c1vs=0,vs=0; int ex=-1; bool ok=false; };
    std::vector<V> vs[4];   // per stage (KProf::P1..CRT)
    std::vector<DiagTab> pt,ht;   // production / h30 tables (same order)
    CUevent ea=nullptr,eb=nullptr,e0=nullptr,e1=nullptr,ej=nullptr;CUdeviceptr ref=0,bad=0;size_t ref_n=0;CUfunction cmp=nullptr,bwf=nullptr;
    decltype(&::cuMemcpyDtoDAsync) dtod=nullptr; decltype(&::cuMemsetD32Async) memset32=nullptr;
    double el()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();}
    static std::vector<unsigned> mlist(const std::string& s,const char* name){std::vector<unsigned> v;const std::string k=std::string("#define ")+name+" ";const size_t p=s.find(k);if(p==std::string::npos)return v;
        const size_t e=s.find('\n',p);std::string x=s.substr(p+k.size(),e-p-k.size());for(char& ch:x)if(ch==','||ch=='u')ch=' ';std::istringstream is(x);unsigned long long y;while(is>>y)v.push_back(unsigned(y));return v;}
    // every extern "C" kernel of a source copy becomes an uninstantiated template (only name-expression instances compile)
    static void templatize(std::string& s){const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    // launch-bounds twin: a copy of the kernel text (head .. first "\n}") right after it, as a template over min-blocks
    static bool twin(std::string& s,const char* head,const char* nhead){const size_t p=s.find(head);if(p==std::string::npos)return false;const size_t e=s.find("\n}",p);if(e==std::string::npos)return false;
        std::string k=s.substr(p,e+2-p);k.replace(0,std::strlen(head),nhead);s.insert(e+2,"\n"+k+"\n");return true;}
    Sweep(C& c,const std::string& source,const std::string& h30src,unsigned l,const std::vector<DiagTab>& ptabs,const std::vector<DiagTab>& htabs):d(c),kp(c.kp),L(l),pt(ptabs),ht(htabs){
        if(const char* e=std::getenv("FHERMA_SWEEP_S"))budget=std::atof(e);
        int cc1=0;if(kp.dattr)kp.dattr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,0);
        if(const char* e=std::getenv("FHERMA_ARCH"))arch=e;else arch=cc1>=9?"compute_90":"compute_80";   // = production module option
        auto prog=[&](const char* name,std::string src,std::initializer_list<const char*> kn,bool dd=false)->int{Prog P;P.name=name;P.arch=arch;P.src=std::move(src);for(auto k:kn)P.kn.push_back(k);P.dd=dd;P.fn.assign(P.kn.size(),nullptr);progs.push_back(std::move(P));return int(progs.size())-1;};
        auto addv=[&](int st,const char* name,int pi,int ki,bool h30=false){V v;v.v=name;v.prog=pi;v.ki=ki;v.h30=h30;vs[st].push_back(v);};
        // helpers: exactness compare + L2 flush / bandwidth kernel
        prog("cmp","typedef unsigned int u32;typedef unsigned long long u64;\n"
            "extern \"C\" __global__ void sw_cmp(const uint4* a,const uint4* b,u64 n,u32* bad){u32 c=0;for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=u64(gridDim.x)*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n"
            "extern \"C\" __global__ void diag_bw(const uint4* __restrict__ src,uint4* dst,u64 nr,u64 nw,u32 seed){const u64 t=u64(blockIdx.x)*blockDim.x+threadIdx.x,S=u64(gridDim.x)*blockDim.x;u32 acc=0;const u64 m=nr>nw?nr:nw;"
            "for(u64 i=t;i<m;i+=S){if(i<nr){const uint4 x=__ldcs(src+i);acc^=x.x^x.y^x.z^x.w;}if(i<nw)dst[i]=make_uint4(u32(i),seed,acc,u32(i>>32));}if(acc==0x9e3779b9u&&seed==1u)dst[0].x=acc;}\n",{"sw_cmp","diag_bw"});
        // B) occupancy twins (P2 single wave would need 2048 blocks on 188 x 12 = 2256 slots)
        {std::string s=source;
         const bool t2=twin(s,"extern \"C\" __global__ void __launch_bounds__(128) bpo_middle_r16(","template<int MB_> __global__ void __launch_bounds__(128,MB_) sw_p2_mb(");
         const bool t3=twin(s,"extern \"C\" __global__ void __launch_bounds__(256) bpo_last_cm3(","template<int MB_> __global__ void __launch_bounds__(256,MB_) sw_p3_mb(");
         templatize(s);
         const int pi=prog("occ",s,{"bpo_middle_r16<0>","sw_p2_mb<12>","sw_p2_mb<10>","sw_p2_mb<9>","bpo_last_cm3<0>","sw_p3_mb<8>"});
         addv(KProf::P2,"ctrl",pi,0);
         if(t2){addv(KProf::P2,"mb12",pi,1);addv(KProf::P2,"mb10",pi,2);addv(KProf::P2,"mb9",pi,3);}
         addv(KProf::P3,"ctrl",pi,4);if(t3)addv(KProf::P3,"mb8",pi,5);}
        // A) h30 (Harvey) and h30sh (same primes, Shoup butterflies): production kernels from the h30 prefix
        if(!h30src.empty()){
            {const auto pr=mlist(h30src,"BCRT_PRIMES"),f=mlist(h30src,"BTC_F");unsigned n30=0;for(auto p:pr)n30+=p<(1u<<30);
             h30info="\"h30_primes\":"+std::to_string(pr.size())+",\"lt2^30\":"+std::to_string(n30)+",\"btc_f\":"+std::to_string(f.empty()?0u:f[0]);}
            std::string s=h30src;templatize(s);
            const int ph=prog("h30",s,{"bpo_first_x<0>","bpo_middle_r16<0>","bpo_last_cm3<0>","bpo_crt_lean<0>"},true);
            const std::string k="#define BPO_NH30 ";const size_t q=s.find(k);
            int ps=-1;if(q!=std::string::npos){std::string s0=s;const size_t e=s0.find('\n',q);s0.replace(q,e-q,k+"0");ps=prog("h30sh",s0,{"bpo_first_x<0>","bpo_middle_r16<0>","bpo_last_cm3<0>","bpo_crt_lean<0>"},true);}
            for(int st=0;st<4;++st){addv(st,"h30",ph,st,true);if(ps>=0)addv(st,"h30sh",ps,st,true);}
        }
        // worker threads: NVRTC compile + module load (driver JIT for PTX), heaviest (h30) first
        const unsigned hw=std::max(2u,std::thread::hardware_concurrency());const size_t nth=std::min<size_t>(progs.size(),std::max(1u,std::min(8u,hw-1)));
        std::vector<size_t> order;for(size_t i=progs.size();i-->0;)order.push_back(i);
        auto next=std::make_shared<std::atomic<size_t>>(0);const double cdl=budget*0.6;
        for(size_t t=0;t<nth;++t)th.emplace_back([this,next,cdl,order]{for(size_t k;(k=next->fetch_add(1))<order.size();){Prog& P=progs[order[k]];if(el()>cdl){P.err="compile deadline";continue;}compile(P);}});
    }
    void compile(Prog& P){
        auto tA=std::chrono::steady_clock::now();
        try{
            auto addn=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*)>("nvrtcAddNameExpression");
            auto lown=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*,const char**)>("nvrtcGetLoweredName");
            auto setctx=d.driver.template get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");
            nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,P.src.c_str(),"sweep.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("create");
            for(const auto& k:P.kn)if(k.find('<')!=std::string::npos)addn(pr,k.c_str());
            const std::string a="--gpu-architecture="+P.arch;const char* op[]={a.c_str(),"--std=c++17","-default-device"};
            if(d.nvrtcCompileProgram(pr,P.dd?3:2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
                const size_t e=l.find("error");throw std::runtime_error("nvrtc: "+l.substr(e==std::string::npos?0:e,160));}
            std::vector<char> img;{size_t n=0;d.nvrtcGetPTXSize(pr,&n);img.resize(n);d.nvrtcGetPTX(pr,img.data());}
            std::vector<std::string> names(P.kn.size());
            for(size_t k=0;k<P.kn.size();++k){if(P.kn[k].find('<')==std::string::npos){names[k]=P.kn[k];continue;}const char* ln=nullptr;if(lown(pr,P.kn[k].c_str(),&ln)==NVRTC_SUCCESS&&ln)names[k]=ln;}
            d.nvrtcDestroyProgram(&pr);
            auto tB=std::chrono::steady_clock::now();P.t_rtc=std::chrono::duration<double>(tB-tA).count();
            setctx(d.context);
            if(d.cuModuleLoadData(&P.mod,img.data())!=CUDA_SUCCESS){P.mod=nullptr;throw std::runtime_error("module load");}
            for(size_t k=0;k<names.size();++k){CUfunction f=nullptr;if(!names[k].empty()&&d.cuModuleGetFunction(&f,P.mod,names[k].c_str())==CUDA_SUCCESS)P.fn[k]=f;}
            P.t_jit=std::chrono::duration<double>(std::chrono::steady_clock::now()-tB).count();
        }catch(const std::exception& e){P.err=e.what();for(char& ch:P.err)if(ch=='"'||ch=='\\'||ch=='\n'||ch=='\r')ch=' ';}
    }
    void join(){for(auto& t:th)if(t.joinable())t.join();th.clear();}
    ~Sweep(){join();d.cuCtxSynchronize();for(auto& P:progs)if(P.mod)d.cuModuleUnload(P.mod);for(auto e:{ea,eb,e0,e1,ej})if(e)kp.edestroy(e);for(auto p:{ref,bad})if(p)d.cuMemFree(p);}

    // ---- measurement -------------------------------------------------------------------------------------------
    // h30: every 8-byte parameter pointing into a production table is moved to the same offset of the h30 table
    int remap(const KProf::Trace& t,std::vector<unsigned char>& b)const{int n=0;
        for(size_t i=0;i<t.po.size();++i){const size_t sz=(i+1<t.po.size()?t.po[i+1]:t.pb.size())-t.po[i];if(sz<8||sz>=64||t.po[i]%8)continue;   // 8-byte slot (+ alignment padding before a 64 B-aligned tensor map)
            unsigned long long x=0;std::memcpy(&x,b.data()+t.po[i],8);
            for(size_t j=0;j<pt.size()&&j<ht.size();++j)if(pt[j].p&&x>=pt[j].p&&x<pt[j].p+pt[j].bytes){x=x-pt[j].p+ht[j].p;std::memcpy(b.data()+t.po[i],&x,8);++n;break;}}
        return n;}
    template<class D> void launch(D& g,const KProf::Trace& t,CUstream st,const V* v){
        std::vector<unsigned char> b=t.pb;void* pp[64];if(v&&v->h30)remap(t,b);
        for(size_t i=0;i<t.po.size();++i)pp[i]=b.data()+t.po[i];
        d.check(d.cuLaunchKernel(v&&v->fn?v->fn:t.fn,t.gx,t.gy,t.gz,t.bx,1,1,t.smem,st,pp,nullptr));(void)g;}
    // one timed sample: flush + predecessors (untimed), then k0 on s0 (and k1 on s1), ms between e0 and the join
    template<class D> float sample(D& g,const KProf::Trace* k0,const KProf::Trace* k1,const KProf::Trace* q0,const KProf::Trace* q1,const V* v){
        CUstream a=g.s[0],b=g.s[1];g.do_flush(a);if(q0)g.replay(*q0,a);
        if(k1){kp.erecord(ea,a);g.swait(b,ea,0);if(q1)g.replay(*q1,b);kp.erecord(eb,b);g.swait(a,eb,0);}
        kp.erecord(e0,a);launch(g,*k0,a,v);
        if(k1){g.swait(b,e0,0);launch(g,*k1,b,v);kp.erecord(ej,b);g.swait(a,ej,0);}
        kp.erecord(e1,a);d.check(g.esync(e1));return kp.ms(e0,e1);}
    static double med(std::vector<float> x){if(x.empty())return 0;std::sort(x.begin(),x.end());return x[x.size()/2];}
    // interleaved A/B: production (v=nullptr) vs variant, c2 (k1 given) and c1
    template<class D> void ab(D& g,V& v,const KProf::Trace* k0,const KProf::Trace* k1,const KProf::Trace* q0,const KProf::Trace* q1,int R){
        std::vector<float> va,vc,ra,rc;
        for(int r=0;r<R+2;++r){
            const bool vf=r&1;float x,y;
            if(k1){x=sample(g,k0,k1,q0,q1,vf?&v:nullptr);y=sample(g,k0,k1,q0,q1,vf?nullptr:&v);if(r>=2){va.push_back(vf?x:y);if(x>0&&y>0)ra.push_back(vf?x/y:y/x);}}
            x=sample(g,k0,nullptr,q0,nullptr,vf?&v:nullptr);y=sample(g,k0,nullptr,q0,nullptr,vf?nullptr:&v);if(r>=2){vc.push_back(vf?x:y);if(x>0&&y>0)rc.push_back(vf?x/y:y/x);}}
        auto& A=k1?va:vc;
        v.med=med(A)*1e3;v.mn=A.empty()?0:*std::min_element(A.begin(),A.end())*1e3;v.vs=med(k1?ra:rc);v.c1=med(vc)*1e3;v.c1vs=med(rc);}
    struct Reg{CUdeviceptr p;size_t bytes;};
    void ensure_ref(size_t tot){if(tot>ref_n){if(ref)d.cuMemFree(ref);ref=0;d.check(d.cuMemAlloc(&ref,tot));ref_n=tot;}}
    unsigned compare(CUstream a,const std::vector<Reg>& rg,int sms){unsigned z=0;d.check(d.cuMemcpyHtoD(bad,&z,4));size_t o=0;
        for(auto& r:rg){CUdeviceptr x=r.p,y=ref+o;unsigned long long n=r.bytes/16;void* pp[]={&x,&y,&n,&bad};d.check(d.cuLaunchKernel(cmp,unsigned(sms)*4,1,1,256,1,1,0,a,pp,nullptr));o+=r.bytes;}
        d.check(d.cuCtxSynchronize());d.check(d.cuMemcpyDtoH(&z,bad,4));return z;}
    // exactness: prelude + production -> ref, prelude, poison the output, variant, compare (count of differing 16 B words)
    template<class D> int exact(D& g,const KProf::Trace& t,const std::function<void(CUstream)>& q,const std::vector<Reg>& rg,const V& v){
        if(!cmp||rg.empty())return -1;size_t tot=0;for(auto& r:rg)tot+=r.bytes;ensure_ref(tot);
        CUstream a=g.s[0];q(a);launch(g,t,a,nullptr);size_t o=0;for(auto& r:rg){d.check(dtod(ref+o,r.p,r.bytes,a));o+=r.bytes;}
        q(a);for(auto& r:rg)d.check(memset32(r.p,0x5a5a5a5au,r.bytes/4,a));
        launch(g,t,a,&v);return compare(a,rg,g.sms)?0:1;}
    template<class P> static P param(const KProf::Trace& t,size_t i){P x{};if(i<t.po.size())std::memcpy(&x,t.pb.data()+t.po[i],sizeof(P));return x;}
    std::vector<Reg> regions(int st,const KProf::Trace& t)const{
        std::vector<Reg> r;
        if(st==KProf::P1){const unsigned N=param<unsigned>(t,7),r0=param<unsigned>(t,9),r1=param<unsigned>(t,10);const size_t off=size_t(r0)*8*N*4,sz=size_t(r1-r0)*8*N*4;
            r.push_back({param<CUdeviceptr>(t,2)+off,sz});r.push_back({param<CUdeviceptr>(t,3)+off,sz});}
        else if(st==KProf::P2)r.push_back({param<CUdeviceptr>(t,2),size_t(t.gz)*t.gy*param<unsigned>(t,5)*4});
        else if(st==KProf::P3)r.push_back({param<CUdeviceptr>(t,1),size_t(t.gy)*8*param<unsigned>(t,6)*4});
        else if(st==KProf::CRT)r.push_back({param<CUdeviceptr>(t,1),size_t(param<unsigned>(t,3))*L*4});
        return r;}
    void print(const char* k,const V& v,const KProf::Trace& t){
        int regs=0,lm=0,oc=0;CUfunction f=v.fn?v.fn:t.fn;
        if(f&&kp.fattr){kp.fattr(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f);kp.fattr(&lm,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);}
        if(f&&kp.occ)kp.occ(&oc,f,int(t.bx),size_t(t.smem));
        std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"med_us\":%.2f,\"min_us\":%.2f,\"vs_prod\":%.4f,\"c1_us\":%.2f,\"c1_vs\":%.4f",k,v.v.c_str(),v.med,v.mn,v.vs,v.c1,v.c1vs);
        if(v.ex>=0)std::printf(",\"ex\":%d",v.ex);std::printf(",\"r\":%d,\"lm\":%d,\"occ\":%d}\n",regs,lm,oc);std::fflush(stdout);}
    template<class D> void run(D& g){
        join();
        {std::printf("{\"diag\":\"sw_info\",\"round\":\"h30s\",\"arch\":\"%s\",\"hw_threads\":%u,\"compile_wall_s\":%.1f,%s%s\"progs\":[",arch.c_str(),std::thread::hardware_concurrency(),el(),h30info.c_str(),h30info.empty()?"":",");
         for(size_t i=0;i<progs.size();++i)std::printf("%s[\"%s\",%zu,%.1f,%.1f%s%s%s]",i?",":"",progs[i].name.c_str(),progs[i].kn.size(),progs[i].t_rtc,progs[i].t_jit,
             progs[i].err.empty()?"":",\"",progs[i].err.empty()?"":progs[i].err.substr(0,120).c_str(),progs[i].err.empty()?"":"\"");
         std::printf("]}\n");std::fflush(stdout);}
        if(kp.last.empty())return;
        for(auto& P:progs)if(P.name=="cmp"&&P.fn.size()==2){cmp=P.fn[0];bwf=P.fn[1];}
        if(!cmp||!bwf){std::printf("{\"diag\":\"sw\",\"error\":\"no helper module\"}\n");return;}
        g.bwf=bwf;
        const auto& tr=kp.traces[kp.last.back().variant];
        dtod=d.driver.template get<decltype(dtod)>("cuMemcpyDtoDAsync_v2");memset32=d.driver.template get<decltype(memset32)>("cuMemsetD32Async");
        for(auto e:{&ea,&eb,&e0,&e1,&ej})kp.ecreate(e,CU_EVENT_DEFAULT);
        d.check(d.cuMemAlloc(&bad,16));
        for(int st=0;st<4;++st)for(auto& v:vs[st]){const auto& P=progs[size_t(v.prog)];if(v.ki>=0&&size_t(v.ki)<P.fn.size()&&P.fn[size_t(v.ki)]){v.fn=P.fn[size_t(v.ki)];v.ok=true;}}
        auto first=[&](int prod,int st,size_t* pi)->const KProf::Trace*{for(size_t i=0;i<tr.size();++i)if(tr[i].product==prod&&tr[i].stage==st&&!tr[i].pb.empty()){if(pi)*pi=i;return &tr[i];}return nullptr;};
        auto pred=[&](int prod,size_t i)->const KProf::Trace*{for(size_t j=i;j-->0;)if(tr[j].product==prod&&!tr[j].pb.empty())return &tr[j];return nullptr;};
        CUfunction pf[4]={d.bpo_first_x,d.bpo_middle_r16,d.bpo_last_cm3,d.bpo_crt_lean};static const char* kn[4]={"p1","p2","p3","crt"};
        {int nk[KProf::NS]={};int bad_stage=0,remapped=0;for(const auto& t:tr)if(t.product==0&&!t.pb.empty()){++nk[t.stage];if(t.stage>=4||t.fn!=pf[t.stage])++bad_stage;else if(!ht.empty()){std::vector<unsigned char> b=t.pb;remapped+=remap(t,b);}}
         std::printf("{\"diag\":\"sw_trace\",\"variant\":%d,\"traces\":%zu,\"prod0\":[%d,%d,%d,%d,%d],\"non_prod_kernels\":%d,\"h30_remapped_ptrs\":%d}\n",kp.last.back().variant,tr.size(),nk[0],nk[1],nk[2],nk[3],nk[4],bad_stage,remapped);}
        {const auto tw=std::chrono::steady_clock::now();   // valid product-0 data in every buffer + clock ramp after the idle compile wait (>= 0.3 s busy)
         for(int w=0;w<200&&(w<3||std::chrono::duration<double>(std::chrono::steady_clock::now()-tw).count()<0.3);++w){
             for(const auto& t:tr)if(t.product<2&&!t.pb.empty())g.replay(t,g.s[0]);d.check(d.cuCtxSynchronize());}}
        const int RE=std::getenv("FHERMA_SWEEP_R")?std::atoi(std::getenv("FHERMA_SWEEP_R")):0;
        const KProf::Trace* k0s[4]={};
        for(int st=0;st<4;++st){
            size_t i0=0,i1=0;const KProf::Trace* k0=first(0,st,&i0);const KProf::Trace* k1=first(1,st,&i1);
            if(!k0||k0->fn!=pf[st]){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"skipped\":\"prod kernel %s\"}\n",kn[st],k0?kp.name_of(k0->fn):"none");continue;}
            if(k1&&(k1->fn!=k0->fn||k1->chain==k0->chain))k1=nullptr;
            const KProf::Trace* q0=st==KProf::P1?nullptr:pred(0,i0);const KProf::Trace* q1=(st==KProf::P1||!k1)?nullptr:pred(1,i1);
            k0s[st]=k0;
            const auto rg=regions(st,*k0);auto prel=[&,i0](CUstream a){for(size_t j=0;j<i0;++j)if(tr[j].product==0&&!tr[j].pb.empty())g.replay(tr[j],a);};
            const int R=RE>0?RE:(st==KProf::P1?15:25);
            V prod;prod.v="prod";ab(g,prod,k0,k1,q0,q1,R);print(kn[st],prod,*k0);
            for(auto& v:vs[st]){
                if(el()>budget-8){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",kn[st],v.v.c_str());v.ok=false;continue;}
                if(!v.ok){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"skipped\":\"not compiled\"}\n",kn[st],v.v.c_str());continue;}
                try{ab(g,v,k0,k1,q0,q1,R);v.ex=v.h30?-1:exact(g,*k0,prel,rg,v);}catch(const std::exception& e){std::printf("{\"diag\":\"sw\",\"k\":\"%s\",\"v\":\"%s\",\"error\":\"%.60s\"}\n",kn[st],v.v.c_str(),e.what());v.ok=false;d.cuCtxSynchronize();continue;}
                print(kn[st],v,*k0);}
        }
        // product chains (products 0 and 1 on two streams from a cold flush)
        auto byname=[&](int st,const char* n)->const V*{for(auto& v:vs[st])if(v.ok&&v.v==n)return &v;return nullptr;};
        auto exok=[&](int st,const char* n)->const V*{const V* v=byname(st,n);return v&&v->ex==1?v:nullptr;};
        struct Combo{std::string name;const V* v[4];bool cf=false,ext=false;};std::vector<Combo> cs;
        cs.push_back({"prod",{nullptr,nullptr,nullptr,nullptr}});
        for(const char* n:{"mb12","mb10","mb9"})if(const V* v=exok(KProf::P2,n))cs.push_back({std::string("p2=")+n,{nullptr,v,nullptr,nullptr}});
        if(const V* v=exok(KProf::P3,"mb8")){cs.push_back({"p3=mb8",{nullptr,nullptr,v,nullptr}});
            for(const char* n:{"mb12","mb10"})if(const V* w=exok(KProf::P2,n))cs.push_back({std::string("p2=")+n+",p3=mb8",{nullptr,w,v,nullptr}});}
        for(const char* n:{"h30","h30sh"}){const V* a[4];bool all=true;for(int st=0;st<4;++st){a[st]=byname(st,n);all=all&&a[st]&&k0s[st];}
            if(!all){std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"skipped\":\"stage kernel missing\"}\n",n);continue;}
            cs.push_back({n,{a[0],a[1],a[2],a[3]}});Combo x=cs.back();x.name=std::string("cf:")+n;x.cf=true;cs.push_back(x);x.name=std::string("x:")+n;x.cf=false;x.ext=true;cs.push_back(x);}
        const int nprod=[&]{int n=0;for(const auto& t:tr)n=std::max(n,t.product+1);return n;}();
        auto chain=[&](int prod,CUstream a,const Combo* c){
            for(const auto& t:tr)if(t.product==prod&&!t.pb.empty()){const V* v=nullptr;if(c&&t.stage<4&&k0s[t.stage]&&t.fn==k0s[t.stage]->fn)v=c->v[t.stage];launch(g,t,a,v);}
            if(c&&c->ext)for(const auto& t:tr)if(t.product==prod+2&&t.stage==KProf::P1&&!t.pb.empty()){launch(g,t,a,c->v[KProf::P1]);break;}};   // x: next same-chain product's P1
        auto csample=[&](const Combo* c,bool cf)->float{CUstream a=g.s[0],b=g.s[1];if(cf)g.bw(a,g.flush,g.flush,g.flush_n,0);else g.do_flush(a);kp.erecord(e0,a);g.swait(b,e0,0);chain(0,a,c);if(nprod>1)chain(1,b,c);kp.erecord(ej,b);g.swait(a,ej,0);kp.erecord(e1,a);d.check(g.esync(e1));return kp.ms(e0,e1);};
        std::vector<Reg> crg;for(int pr=0;pr<std::min(nprod,2);++pr){size_t i=0;if(const KProf::Trace* t=first(pr,KProf::CRT,&i)){auto r=regions(KProf::CRT,*t);crg.insert(crg.end(),r.begin(),r.end());}}
        for(const auto& c:cs){
            if(el()>budget-3){std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",c.name.c_str());continue;}
            Combo base{"base",{nullptr,nullptr,nullptr,nullptr},c.cf,c.ext};
            std::vector<float> pa,va,rr;const int RC=9;
            try{
            for(int r=0;r<RC+1;++r){const bool vf=r&1;const float x=csample(vf?&c:&base,c.cf),y=csample(vf?&base:&c,c.cf);if(r>=1){(vf?va:pa).push_back(x);(vf?pa:va).push_back(y);if(x>0&&y>0)rr.push_back(vf?x/y:y/x);}}
            int ex=-1;
            if(!crg.empty()){size_t tot=0;for(auto& r:crg)tot+=r.bytes;ensure_ref(tot);CUstream a=g.s[0];
                chain(0,a,nullptr);if(nprod>1)chain(1,a,nullptr);size_t o=0;for(auto& r:crg){d.check(dtod(ref+o,r.p,r.bytes,a));o+=r.bytes;}
                for(auto& r:crg)d.check(memset32(r.p,0x5a5a5a5au,r.bytes/4,a));chain(0,a,&c);if(nprod>1)chain(1,a,&c);ex=compare(a,crg,g.sms)?0:1;}
            std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"med_us\":%.1f,\"min_us\":%.1f,\"vs_prod\":%.4f,\"prod_us\":%.1f,\"ex\":%d}\n",c.name.c_str(),med(va)*1e3,va.empty()?0.0:*std::min_element(va.begin(),va.end())*1e3,med(rr),med(pa)*1e3,ex);std::fflush(stdout);
            }catch(const std::exception& e){std::printf("{\"diag\":\"sw\",\"k\":\"chain\",\"v\":\"%s\",\"error\":\"%.60s\"}\n",c.name.c_str(),e.what());d.cuCtxSynchronize();}}
        std::printf("{\"diag\":\"sw_done\",\"s\":%.1f}\n",el());std::fflush(stdout);
    }
};
