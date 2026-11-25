#pragma once
// wl2 diag (round 27, on p1wl 530fb86 = production candidate; production PTX unchanged by this file): exit-time paired
// whole-batch lines (State::batch_line: the full B-product production graph, variant 6, recaptured with per-stage substitutes
// and timed against the production graph exec in paired, alternating reps; vs_prod = median of pair ratios, < 1 = faster;
// ex = 1 when all B outputs are bit-identical to the production graph's, device compare) and per-kernel replay lines.
// Variants = re-laid host / init-built tables read with a changed index expression only (same grid, block, registers):
//   p1tw  P1 bpo_first_x BPO_P1TW: phase-A twist t2 from T1[((u*2 + c/128)*256 + 64 rr + rl)*4 + tig] (one warp request =
//         256 contiguous bytes = 2 lines; production: 4 primes x 8 rows, 7.5 lines x 34 B)
//   p1st  P1 BPO_P1ST: phase-A stage twiddles (J-DIF st[32/64/96], R-DIF st[rl], st[64+rl], st[128+rl]) from a quad-interleaved
//         copy (entry e of prime 4u+tig at 4e+tig) appended to the stage table (production: 4 lines per request)
//   p2mt  P2 bpo_middle_r16 BPO_P2MT: middle twiddles w^(c2 brev4(i)) / w^-(brev4(c2) i) from [i][c2] tables appended to r16
//         (per prime 1024 words): 4 lanes groups in one 32 B sector, base + immediate offsets (production: bpo_tw8, 2.9 lines)
//   p3f   P3 bpo_last_cm3 BPO_P3F: F (untwist root x cross untwiddle, Shoup pair) from a table [prime][c][t] built at exit by
//         wl2_ftab with the kernel's own formula (production: bm_root gathers, 8.9 lines per request, + the math)
//   all   p1tw + p1st (one P1 module) + p2mt + p3f.
// Lines: ctrl, rc (production P1/P2/P3/CRT recompiled), p1tw, p1st, p2mt, p3f, all, one repeat of each (_r), ctrl_end.
// kcyc: one captured production launch per stage (product 0) replayed back to back: p1 prod / rc / p1tw / p1st / p1all,
// p2 prod / rc / p2mt, p3 prod / rc / p3f, crt prod (table pointers patched in the replayed parameters).
// FHERMA_COMBO=0 off, FHERMA_COMBO_S budget (s, default 50), FHERMA_BATCH_R pairs per line (default 13), FHERMA_KCYC_R rounds.
#include <cuda.h>
#include <nvrtc.h>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

// Per stage (0 P1, 1 P2, 2 P3, 3 CRT): fn = kernel substitute (null = production); table substitutes (0 = production):
// p1tw = P1 twist argument, p1st = P1 stage_tw argument, r16x = P2 r16 base (stride 1024 per prime), ftab = P3 untwist base
// (stride 65536 per prime); check = exactness vs the production graph.
struct BatchOvr { CUfunction fn[4]{}; CUdeviceptr p1tw=0,p1st=0,r16x=0,ftab=0; bool check=true; };
struct BatchRes { bool ok=false; int ex=-1; double vs=0,med=0,prod=0; };
struct Wl2Tabs { CUdeviceptr tw=0,st=0,r16=0,itw=0,primes=0,t1=0,stq=0,r16x=0; unsigned k=0; };   // production pointers + host-built variant tables

template<class C> struct ComboMods {
    struct Prog { std::string name,src; std::vector<std::string> kn; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t=0; };
    C& d; std::string arch; std::vector<Prog> progs;
    explicit ComboMods(C& c):d(c){
        int cc1=0;CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
        if(const char* e=std::getenv("FHERMA_ARCH"))arch=e;else arch=cc1>=9?"compute_90":"compute_80";}
    ~ComboMods(){d.cuCtxSynchronize();for(auto& P:progs)if(P.mod)d.cuModuleUnload(P.mod);}
    int add(const char* name,std::string src,std::vector<std::string> kn){Prog P;P.name=name;P.src=std::move(src);P.kn=std::move(kn);progs.push_back(std::move(P));return int(progs.size())-1;}
    void compile(Prog& P){
        const auto tA=std::chrono::steady_clock::now();
        try{
            auto addn=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*)>("nvrtcAddNameExpression");
            auto lown=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*,const char**)>("nvrtcGetLoweredName");
            auto setctx=d.driver.template get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");
            nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,P.src.c_str(),"combo.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("create");
            for(const auto& k:P.kn)if(k.find('<')!=std::string::npos)addn(pr,k.c_str());
            const std::string a="--gpu-architecture="+arch;const char* op[]={a.c_str(),"--std=c++17"};
            if(d.nvrtcCompileProgram(pr,2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
                const size_t e=l.find("error");throw std::runtime_error("nvrtc: "+l.substr(e==std::string::npos?0:e,160));}
            size_t n=0;d.nvrtcGetPTXSize(pr,&n);std::vector<char> img(n);d.nvrtcGetPTX(pr,img.data());
            std::vector<std::string> names(P.kn.size());
            for(size_t k=0;k<P.kn.size();++k){if(P.kn[k].find('<')==std::string::npos){names[k]=P.kn[k];continue;}const char* ln=nullptr;if(lown(pr,P.kn[k].c_str(),&ln)==NVRTC_SUCCESS&&ln)names[k]=ln;}
            d.nvrtcDestroyProgram(&pr);
            setctx(d.context);
            if(d.cuModuleLoadData(&P.mod,img.data())!=CUDA_SUCCESS){P.mod=nullptr;throw std::runtime_error("module load");}
            P.fn.assign(names.size(),nullptr);
            for(size_t k=0;k<names.size();++k){CUfunction f=nullptr;if(!names[k].empty()&&d.cuModuleGetFunction(&f,P.mod,names[k].c_str())==CUDA_SUCCESS)P.fn[k]=f;}
        }catch(const std::exception& e){P.err=e.what();for(char& ch:P.err)if(ch=='"'||ch=='\\'||ch=='\n'||ch=='\r')ch=' ';}
        P.t=std::chrono::duration<double>(std::chrono::steady_clock::now()-tA).count();}
    void build(){
        const unsigned hw=std::max(2u,std::thread::hardware_concurrency());const size_t nth=std::min<size_t>(progs.size(),std::max(1u,std::min(8u,hw-1)));
        std::atomic<size_t> next{0};std::vector<std::thread> th;
        for(size_t t=0;t<nth;++t)th.emplace_back([&]{for(size_t k;(k=next.fetch_add(1))<progs.size();)compile(progs[k]);});
        for(auto& t:th)t.join();}
    CUfunction fn(int p,size_t k=0)const{return p>=0&&size_t(p)<progs.size()&&k<progs[size_t(p)].fn.size()?progs[size_t(p)].fn[k]:nullptr;}
};

template<class C> inline void run_combo(C& d,const std::string& src,const CUfunction (&prodfn)[4],Wl2Tabs T,
    const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):50.0;
    int cc1=0,cc2=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);}}
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n"
        // F table of P3 (bpo_p3b_body formula, N = 65536, logR = 8): out[(prime*256 + c)*256 + t]
        "extern \"C\" __global__ void wl2_ftab(const u64* untwist,const u32* primes,u64* out){const u32 c=blockIdx.x,prime=blockIdx.y,t=threadIdx.x,p=primes[prime];constexpr u32 N=65536u,logR=8u;"
        "const u64* itw=untwist+(u64)prime*weighted_root_stride(N);const u32 k1=__brev(t)>>(32-logR);const u32 wv=mul_shoup(u32(bm_root(itw,2u*c*k1,N,p)),itw[1+(c&127u)],p);"
        "const u64 m0=itw[0];u32 q=wv*u32(m0>>32)+__umulhi(wv,u32(m0));const u64 rem=(u64(wv)<<32)-u64(q)*p;if(rem>=p)++q;out[(u64(prime)*256u+c)*256u+t]=u64(wv)|(u64(q)<<32);}\n",
        {"bpo_middle_r16<0>","bpo_last_cm3<0>","bpo_crt_lean<0>","cb_cmp","bpo_first_x<0>","wl2_ftab"});
    const int m_tw=M.add("p1tw","#define BPO_P1TW 1\n"+s,{"bpo_first_x<0>"});
    const int m_st=M.add("p1st","#define BPO_P1ST 1\n"+s,{"bpo_first_x<0>"});
    const int m_mt=M.add("p2mt","#define BPO_P2MT 1\n"+s,{"bpo_middle_r16<0>"});
    const int m_f=M.add("p3f","#define BPO_P3F 1\n"+s,{"bpo_last_cm3<0>"});
    const int m_all=M.add("all","#define BPO_P1TW 1\n#define BPO_P1ST 1\n#define BPO_P2MT 1\n#define BPO_P3F 1\n"+s,{"bpo_first_x<0>","bpo_middle_r16<0>","bpo_last_cm3<0>"});
    M.build();
    // P3 F table (8 MB for 16 primes), built on device with the production formula
    CUdeviceptr ftab=0;
    if(CUfunction ff=M.fn(m_rc,5))if(T.itw&&T.primes&&T.k&&d.cuMemAlloc(&ftab,size_t(T.k)*65536*8)==CUDA_SUCCESS){
        CUdeviceptr a=T.itw,b=T.primes,o=ftab;void* pp[]={&a,&b,&o};
        if(d.cuLaunchKernel(ff,256,T.k,1,256,1,1,0,nullptr,pp,nullptr)!=CUDA_SUCCESS||d.cuCtxSynchronize()!=CUDA_SUCCESS){d.cuMemFree(ftab);ftab=0;}}
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":\"wl2\",\"cc\":\"%d.%d\",\"tabs\":[%d,%d,%d,%d],\"compile_s\":%.1f,\"mods\":{%s}}\n",cc1,cc2,int(T.t1!=0),int(T.stq!=0),int(T.r16x!=0),int(ftab!=0),el(),info.c_str());std::fflush(stdout);}
    {   // JIT registers / local bytes / static smem / blocks per SM per kernel as loaded on this device
        auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");std::string r;
        auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
        auto one=[&](const char* nm,CUfunction f,int bt){if(!f)return;int rg=0,lc=0,sm=0,ob=0;ga(&rg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&lc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);ga(&sm,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);if(occ)occ(&ob,f,bt,0);
            char b[96];std::snprintf(b,sizeof b,"%s\"%s\":[%d,%d,%d,%d]",r.empty()?"":",",nm,rg,lc,sm,ob);r+=b;};
        one("p1",prodfn[0],256);one("p2",prodfn[1],128);one("p3",prodfn[2],256);one("crt",prodfn[3],128);
        std::printf("{\"diag\":\"sw\",\"k\":\"combo_regs\",\"regs_local_smem_bpsm\":{%s}}\n",r.c_str());std::fflush(stdout);r.clear();   // two lines (< 300 chars each)
        one("p1tw",M.fn(m_tw),256);one("p1st",M.fn(m_st),256);one("p2mt",M.fn(m_mt),128);one("p3f",M.fn(m_f),256);one("p1all",M.fn(m_all,0),256);one("p2all",M.fn(m_all,1),128);one("p3all",M.fn(m_all,2),256);
        std::printf("{\"diag\":\"sw\",\"k\":\"combo_regs\",\"regs_local_smem_bpsm\":{%s}}\n",r.c_str());std::fflush(stdout);}
    cmp=M.fn(m_rc,3);
    // ---- whole-batch lines
    struct L{std::string v;BatchOvr o;bool var=false;BatchRes r;};std::vector<L> done;
    auto run=[&](L x,bool rep){
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return;}
        if(x.var&&!x.o.fn[0]&&!x.o.fn[1]&&!x.o.fn[2]&&!x.o.fn[3]){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module or table missing\"}\n",x.v.c_str());std::fflush(stdout);return;}
        x.r=line(x.o,x.v.c_str());if(x.r.ok&&!rep)done.push_back(x);};
    auto mk=[](const char* v){L x;x.v=v;x.var=std::string(v).rfind("ctrl",0)!=0;return x;};
    run(mk("ctrl"),false);
    {L x=mk("rc");x.o.fn[0]=M.fn(m_rc,4);x.o.fn[1]=M.fn(m_rc,0);x.o.fn[2]=M.fn(m_rc,1);x.o.fn[3]=M.fn(m_rc,2);if(!x.o.fn[0]||!x.o.fn[1]||!x.o.fn[2]||!x.o.fn[3])x.o=BatchOvr{};run(x,false);}
    {L x=mk("p1tw");if(M.fn(m_tw)&&T.t1){x.o.fn[0]=M.fn(m_tw);x.o.p1tw=T.t1;}run(x,false);}
    {L x=mk("p1st");if(M.fn(m_st)&&T.stq){x.o.fn[0]=M.fn(m_st);x.o.p1st=T.stq;}run(x,false);}
    {L x=mk("p2mt");if(M.fn(m_mt)&&T.r16x){x.o.fn[1]=M.fn(m_mt);x.o.r16x=T.r16x;}run(x,false);}
    {L x=mk("p3f");if(M.fn(m_f)&&ftab){x.o.fn[2]=M.fn(m_f);x.o.ftab=ftab;}run(x,false);}
    {L x=mk("all");if(M.fn(m_all,0)&&M.fn(m_all,1)&&M.fn(m_all,2)&&T.t1&&T.stq&&T.r16x&&ftab){x.o.fn[0]=M.fn(m_all,0);x.o.fn[1]=M.fn(m_all,1);x.o.fn[2]=M.fn(m_all,2);x.o.p1tw=T.t1;x.o.p1st=T.stq;x.o.r16x=T.r16x;x.o.ftab=ftab;}run(x,false);}
    {const size_t n=done.size();for(size_t i=0;i<n;++i){if(done[i].v=="ctrl")continue;L x=done[i];x.v+="_r";run(x,true);}}
    run(mk("ctrl_end"),true);
    // ---- kernel-level replay lines
    {auto& kp=d.kp;const int var=kp.diag_variant();
     if(var<0||kp.traces[var].empty()){std::printf("{\"diag\":\"kcyc\",\"skipped\":\"no captured launches (FHERMA_DIAG=0 or no cuFuncGetParamInfo)\"}\n");std::fflush(stdout);}
     else{const auto& tr=kp.traces[var];
        auto screate=d.driver.template get<decltype(&::cuStreamCreate)>("cuStreamCreate");auto sdestroy=d.driver.template get<decltype(&::cuStreamDestroy)>("cuStreamDestroy_v2");
        auto esync=d.driver.template get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");
        CUstream st=nullptr;d.check(screate(&st,CU_STREAM_NON_BLOCKING));CUevent e0=nullptr,e1=nullptr;kp.ecreate(&e0,CU_EVENT_DEFAULT);kp.ecreate(&e1,CU_EVENT_DEFAULT);
        int clk=0;CUdevice dev=0;d.cuDeviceGet(&dev,0);auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");attr(&clk,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev);
        const int R=std::getenv("FHERMA_KCYC_R")?std::max(3,std::atoi(std::getenv("FHERMA_KCYC_R"))):15,NL=8;
        const int sst[4]={KProf::P1,KProf::P2,KProf::P3,KProf::CRT};const char* SN[4]={"p1","p2","p3","crt"};
        struct P{int idx;CUdeviceptr from,to;};struct V{const char* mod;CUfunction f;std::vector<P> pt;};
        for(int q=0;q<4;++q){
            const KProf::Trace* t=nullptr;for(const auto& x:tr)if(x.product==0&&x.stage==sst[q]&&!x.pb.empty()&&x.fn==prodfn[q]){t=&x;break;}
            if(!t){std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"skipped\":\"no trace\"}\n",SN[q]);std::fflush(stdout);continue;}
            std::vector<V> vs={{"prod",prodfn[q],{}}};
            if(q==0){vs.push_back({"rc",M.fn(m_rc,4),{}});vs.push_back({"p1tw",T.t1?M.fn(m_tw):nullptr,{{4,T.tw,T.t1}}});vs.push_back({"p1st",T.stq?M.fn(m_st):nullptr,{{5,T.st,T.stq}}});
                vs.push_back({"p1all",T.t1&&T.stq?M.fn(m_all,0):nullptr,{{4,T.tw,T.t1},{5,T.st,T.stq}}});}
            if(q==1){vs.push_back({"rc",M.fn(m_rc,0),{}});vs.push_back({"p2mt",T.r16x?M.fn(m_mt):nullptr,{{4,T.r16,T.r16x}}});}
            if(q==2){vs.push_back({"rc",M.fn(m_rc,1),{}});vs.push_back({"p3f",ftab?M.fn(m_f):nullptr,{{3,T.itw,ftab}}});}
            const size_t np=t->po.size();
            std::vector<std::vector<unsigned char>> pbs(vs.size(),t->pb);std::vector<std::vector<void*>> pps(vs.size(),std::vector<void*>(np));std::vector<bool> bad(vs.size(),false);
            for(size_t m=0;m<vs.size();++m){for(size_t i=0;i<np;++i)pps[m][i]=pbs[m].data()+t->po[i];
                for(const auto& x:vs[m].pt){CUdeviceptr y=0;if(size_t(x.idx)>=np){bad[m]=true;continue;}std::memcpy(&y,pps[m][size_t(x.idx)],sizeof y);
                    if(y!=x.from){bad[m]=true;continue;}std::memcpy(pps[m][size_t(x.idx)],&x.to,sizeof y);}}   // trace of group 0: the table base itself
            std::vector<std::vector<float>> v(vs.size());
            for(int r=0;r<R+1;++r)for(size_t m=0;m<vs.size();++m){if(!vs[m].f||bad[m])continue;
                kp.erecord(e0,st);for(int k=0;k<NL;++k)if(d.cuLaunchKernel(vs[m].f,t->gx,t->gy,t->gz,t->bx,1,1,t->smem,st,pps[m].data(),nullptr)!=CUDA_SUCCESS)bad[m]=true;
                kp.erecord(e1,st);if(esync(e1)!=CUDA_SUCCESS)bad[m]=true;if(r>0&&!bad[m])v[m].push_back(kp.ms(e0,e1)*1e3f/NL);}
            double ref=0;
            for(size_t m=0;m<vs.size();++m){if(!vs[m].f)continue;
                if(bad[m]||v[m].empty()){std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"mod\":\"%s\",\"error\":\"launch or table patch\"}\n",SN[q],vs[m].mod);std::fflush(stdout);continue;}
                std::sort(v[m].begin(),v[m].end());const double med=v[m][v[m].size()/2];if(m==0)ref=med;
                std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"mod\":\"%s\",\"us\":%.3f,\"us_min\":%.3f,\"vs_prod\":%.4f,\"cyc_nominal\":%.0f,\"grid\":[%u,%u,%u],\"block\":%u}\n",
                    SN[q],vs[m].mod,med,v[m].front(),ref>0?med/ref:0.0,med*clk*1e-3,t->gx,t->gy,t->gz,t->bx);std::fflush(stdout);}}
        kp.edestroy(e0);kp.edestroy(e1);d.cuCtxSynchronize();sdestroy(st);}}
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;if(ftab)d.cuMemFree(ftab);
}
