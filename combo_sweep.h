#pragma once
// wl3 diag (round 29, on candidate 3 = production default path; production PTX unchanged by this file): exit-time paired
// whole-batch lines (State::batch_line: the full B-product production graph, variant 6, recaptured with per-stage substitutes
// and timed against the production graph exec in paired, alternating reps; vs_prod = median of pair ratios, < 1 = faster;
// ex = 1 when all B outputs are bit-identical to the production graph's, device compare). Theme: issue slots into tables and
// immediates. Variants (same grid / block; registers reported per module):
//   p2o8  production P2 (bpo_middle_r16m, 56 regs, 9 blocks/SM) pinned to 8 blocks/SM with dynamic smem (layout vs occupancy)
//   p2cu  P2 radix-16 stage twiddles (tf/ti [16 j], [32 j], [64]: 28 LDG.64 per thread) from __constant__ (ULDC, uniform)
//   p3cu  P3 p3b uniform stage twiddles (sit 0/32/64/96/252/253/254) from __constant__
//   p3lt  P3 p3b 14 lane twiddles of the R-DIT as 7 whole-line uint4 loads (base + immediates) instead of 14 LDG.64 + index math
//   p3cl  p3cu + p3lt (one P3 module)
//   p1pc  P1 phase-A per-prime constants {p, m62, qsh, np} as one uint4 (appended to T1) instead of 3 divergent LDC + math
//   (crtn  CRT residue addressing with N = 65536 compile-time: +0.50 slots per point in the SASS census, dropped; no line)
//   all   p2cu + p3lt + p1pc = the variants that remove slots in the SASS census (p3cl < p3lt alone; p2o8 off)
// Two passes (_r), ctrl line at the start of each pass and ctrl_end. FHERMA_COMBO=0 off, FHERMA_COMBO_S budget (s, default 60),
// FHERMA_BATCH_R pairs per line (default 13).
#include <cuda.h>
#include <cstdint>
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

// Per stage (0 P1, 1 P2, 2 P3, 3 CRT): fn = kernel substitute (null = production); p1tw = P1 T1 argument; p3sit = P3 stage_itw
// base (stride 704 per prime); p2smem = P2 dynamic smem; p2hi / p3hi: OR s0 << 24 into the P2 untw / P3 dfl word (global prime
// index for the __constant__ tables); check = exactness vs the production graph.
struct BatchOvr { CUfunction fn[4]{}; CUdeviceptr p1tw=0,p3sit=0; unsigned p2smem=0; bool p2hi=false,p3hi=false,check=true; };
struct BatchRes { bool ok=false; int ex=-1; double vs=0,med=0,prod=0; };
struct Wl3Tabs { CUdeviceptr t1=0,r16=0,sit=0,primes=0; unsigned k=0; CUfunction p1=nullptr,p2=nullptr,p3=nullptr,crt=nullptr; };

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

template<class V> inline std::string wl3_list(const char* decl,const V& v){
    std::string s=std::string("__constant__ unsigned long long ")+decl+"["+std::to_string(v.size())+"]={";
    for(size_t i=0;i<v.size();++i){s+=std::to_string(v[i]);s+="ull";if(i+1<v.size())s+=',';}
    return s+"};\n";}

template<class C> inline void run_combo(C& d,const std::string& src,Wl3Tabs T,
    const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):60.0;
    int cc1=0,cc2=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);}}
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    const unsigned K=T.k;
    // __constant__ tables (from the production device tables) and the P3 lane table / P1 constants
    std::string c2,c3;CUdeviceptr lt=0,t1pc=0;
    try{
        if(T.r16&&K){std::vector<uint64_t> h(size_t(K)*512),o(size_t(K)*256);d.check(d.cuMemcpyDtoH(h.data(),T.r16,h.size()*8));
            for(unsigned r=0;r<K;++r)for(unsigned e=0;e<128;++e){o[r*256+e]=h[r*512+e];o[r*256+128+e]=h[r*512+256+e];}c2=wl3_list("bpo_p2cu",o);}
        if(T.sit&&K){std::vector<uint64_t> h(size_t(K)*256),o(size_t(K)*8,0),q(size_t(K)*704);d.check(d.cuMemcpyDtoH(h.data(),T.sit,h.size()*8));
            const unsigned us[7]={0,32,64,96,252,253,254};
            for(unsigned r=0;r<K;++r){for(unsigned e=0;e<7;++e)o[r*8+e]=h[r*256+us[e]];
                for(unsigned x=0;x<256;++x)q[size_t(r)*704+x]=h[r*256+x];
                for(unsigned l=0;l<32;++l){const unsigned l3=l&3u;
                    const unsigned pr[7][2]={{248+l3,240+l3},{244+l3,224+l3},{228+l3,232+l3},{236+l3,192+l},{128+l,160+l},{l,32+l},{64+l,96+l}};
                    for(unsigned qi=0;qi<7;++qi)for(unsigned hh=0;hh<2;++hh)q[size_t(r)*704+256+2*(32*qi+l)+hh]=h[r*256+pr[qi][hh]];}}
            c3=wl3_list("bpo_p3cu",o);lt=d.upload(q);}
    }catch(...){c2.clear();c3.clear();lt=0;}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n"
        // P1 per-prime constants of phase A (the production expressions): out[r] = {p, m62, qsh, np}
        "extern \"C\" __global__ void wl3_pc(uint4* out){const u32 r=threadIdx.x;if(r>=BCRT_K)return;const u32 p=bpr_pr[r];const u32 msh=(BPO_MIXED&&p<(1u<<30))?3u:2u;out[r]=make_uint4(p,u32(bpr_mu[r]>>msh),18u-msh,bpr_np.v[r]);}\n",
        {"bpo_first_xt<0>","bpo_middle_r16m<0>","bpo_last_cm3f<0>","bpo_crt_lean<0>","cb_cmp","wl3_pc"});
    const int m_p2cu=c2.empty()?-1:M.add("p2cu","#define BPO_P2CU 1\n"+c2+s,{"bpo_middle_r16m<0>"});
    const int m_p3cu=c3.empty()?-1:M.add("p3cu","#define BPO_P3CU 1\n"+c3+s,{"bpo_last_cm3f<0>"});
    const int m_p3lt=M.add("p3lt","#define BPO_P3LT 1\n"+s,{"bpo_last_cm3f<0>"});
    const int m_p3cl=c3.empty()?-1:M.add("p3cl","#define BPO_P3CU 1\n#define BPO_P3LT 1\n"+c3+s,{"bpo_last_cm3f<0>"});
    const int m_p1pc=M.add("p1pc","#define BPO_P1PC 1\n"+s,{"bpo_first_xt<0>"});
    const int m_all=c2.empty()?-1:M.add("all","#define BPO_P2CU 1\n#define BPO_P3LT 1\n#define BPO_P1PC 1\n"+c2+s,
        {"bpo_first_xt<0>","bpo_middle_r16m<0>","bpo_last_cm3f<0>"});
    M.build();
    // P1 T1 + appended constants (16 B per prime)
    if(CUfunction pf=M.fn(m_rc,5))if(T.t1&&K%4==0){const size_t tw=size_t(K/4)*2048;
        if(d.cuMemAlloc(&t1pc,(tw+2*K)*8)==CUDA_SUCCESS){CUdeviceptr o=t1pc+tw*8;void* pp[]={&o};
            auto dtod=d.driver.template get<decltype(&::cuMemcpyDtoD)>("cuMemcpyDtoD_v2");if(dtod(t1pc,T.t1,tw*8)!=CUDA_SUCCESS||d.cuLaunchKernel(pf,1,1,1,32,1,1,0,nullptr,pp,nullptr)!=CUDA_SUCCESS||d.cuCtxSynchronize()!=CUDA_SUCCESS){d.cuMemFree(t1pc);t1pc=0;}}else t1pc=0;}
    // p2o8: dynamic smem that caps the production P2 at 8 blocks/SM
    auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    int ob0=0,ob8=0;unsigned dyn8=0;
    if(occ&&T.p2){occ(&ob0,T.p2,128,0);for(unsigned dy=256;dy<=40960&&ob0>8;dy+=256){int ob=0;if(occ(&ob,T.p2,128,dy)!=CUDA_SUCCESS)break;if(ob<=8){dyn8=dy;ob8=ob;break;}}}
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":\"wl3\",\"cc\":\"%d.%d\",\"tabs\":[%d,%d,%d,%d],\"p2o8\":[%d,%u,%d],\"compile_s\":%.1f}\n",cc1,cc2,int(!c2.empty()),int(!c3.empty()),int(lt!=0),int(t1pc!=0),ob0,dyn8,ob8,el());
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_mods\",\"mods\":{%s}}\n",info.c_str());std::fflush(stdout);}
    {   // JIT registers / local bytes / static smem / blocks per SM per kernel as loaded on this device (lines < 300 chars)
        auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");std::string r;
        auto one=[&](const char* nm,CUfunction f,int bt){if(!f)return;int rg=0,lc=0,sm=0,ob=0;ga(&rg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&lc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);ga(&sm,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);if(occ)occ(&ob,f,bt,0);
            char b[96];std::snprintf(b,sizeof b,"%s\"%s\":[%d,%d,%d,%d]",r.empty()?"":",",nm,rg,lc,sm,ob);r+=b;};
        auto flush=[&]{std::printf("{\"diag\":\"sw\",\"k\":\"combo_regs\",\"regs_local_smem_bpsm\":{%s}}\n",r.c_str());std::fflush(stdout);r.clear();};
        one("p1",T.p1,256);one("p2",T.p2,128);one("p3",T.p3,256);one("crt",T.crt,128);one("p2cu",M.fn(m_p2cu),128);one("p3cu",M.fn(m_p3cu),256);flush();
        one("p3lt",M.fn(m_p3lt),256);one("p3cl",M.fn(m_p3cl),256);one("p1pc",M.fn(m_p1pc),256);flush();
        one("p1all",M.fn(m_all,0),256);one("p2all",M.fn(m_all,1),128);one("p3all",M.fn(m_all,2),256);flush();}
    cmp=M.fn(m_rc,4);
    // ---- whole-batch lines, two passes
    struct L{std::string v;BatchOvr o;bool var=false;};std::vector<L> vs;
    auto mk=[](const char* v){L x;x.v=v;x.var=std::string(v).rfind("ctrl",0)!=0;return x;};
    {L x=mk("rc");x.o.fn[0]=M.fn(m_rc,0);x.o.fn[1]=M.fn(m_rc,1);x.o.fn[2]=M.fn(m_rc,2);x.o.fn[3]=M.fn(m_rc,3);if(!x.o.fn[0]||!x.o.fn[1]||!x.o.fn[2]||!x.o.fn[3])x.o=BatchOvr{};vs.push_back(x);}
    {L x=mk("p2o8");if(dyn8){x.o.fn[1]=T.p2;x.o.p2smem=dyn8;}vs.push_back(x);}
    {L x=mk("p2cu");if(M.fn(m_p2cu)){x.o.fn[1]=M.fn(m_p2cu);x.o.p2hi=true;}vs.push_back(x);}
    {L x=mk("p3cu");if(M.fn(m_p3cu)){x.o.fn[2]=M.fn(m_p3cu);x.o.p3hi=true;}vs.push_back(x);}
    {L x=mk("p3lt");if(M.fn(m_p3lt)&&lt){x.o.fn[2]=M.fn(m_p3lt);x.o.p3sit=lt;}vs.push_back(x);}
    {L x=mk("p3cl");if(M.fn(m_p3cl)&&lt){x.o.fn[2]=M.fn(m_p3cl);x.o.p3sit=lt;x.o.p3hi=true;}vs.push_back(x);}
    {L x=mk("p1pc");if(M.fn(m_p1pc)&&t1pc){x.o.fn[0]=M.fn(m_p1pc);x.o.p1tw=t1pc;}vs.push_back(x);}
    {L x=mk("all");if(M.fn(m_all,0)&&M.fn(m_all,1)&&M.fn(m_all,2)&&lt&&t1pc){for(int q=0;q<3;++q)x.o.fn[q]=M.fn(m_all,q);x.o.p1tw=t1pc;x.o.p3sit=lt;x.o.p2hi=true;}vs.push_back(x);}
    auto run=[&](L x){
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return;}
        if(x.var&&!x.o.fn[0]&&!x.o.fn[1]&&!x.o.fn[2]&&!x.o.fn[3]){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module or table missing\"}\n",x.v.c_str());std::fflush(stdout);return;}
        line(x.o,x.v.c_str());};
    run(mk("ctrl"));for(const auto& x:vs)run(x);
    run(mk("ctrl_r"));for(auto x:vs){x.v+="_r";run(x);}
    run(mk("ctrl_end"));
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f}\n",el());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;if(t1pc)d.cuMemFree(t1pc);
}
