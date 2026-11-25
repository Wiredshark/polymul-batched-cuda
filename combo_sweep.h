#pragma once
// wline diag (round 26, on wline = 3140af3 + opt-in p1wl / p2w; production PTX byte-identical to 3140af3): exit-time paired
// whole-batch lines (State::batch_line: the full B-product production graph, variant 6, recaptured with per-stage substitutes
// and timed against the production graph exec in paired, alternating reps; vs_prod = median of pair ratios, < 1 = faster;
// ex = 1 when all B outputs are bit-identical to the production graph's, device compare) and per-kernel replay lines.
// Variants (whole-line global requests, see I:\fherma-work\wline\reqshape_*.txt):
//   p1wl  P1 bpo_first_x with BPO_P1WL: cross twiddles (ctab, 8 MB) and IMMA B fragments (bpt) read through permuted
//         host tables (ctab_wl / bpt_wl, built here at exit from the production tables) so every LDG.128 request covers
//         512 contiguous bytes = 4 whole lines (production: 16 lines x 32 B each); same grid / threads / registers.
//   p2w   P2 bpo_middle_r16 with BPO_P2W: 512-thread blocks (4 slot octets, named barrier per 128-thread group), the C
//         output through a 256 x 32 smem tile, every STG.32 one whole 128 B line (production: 4 x 32 B); grid.x 8, 512 thr.
//   p2w_direct  ablation: the p2w 512-thread structure with the production 4 x 32 B C stores (BPO_P2W 2): isolates the tile.
//   wl    both (p1wl + p2w).
// Lines: ctrl, rc (production P1/P2/P3/CRT recompiled), p1wl, p2w, wl, one repeat of each (_r), ctrl_end.
// kcyc lines: one captured production launch per stage (product 0, kprof trace) replayed back to back, NL launches per sample,
// R alternating rounds: p1 prod / rc / p1wl (tables patched), p2 prod / rc / p2w (grid 8 x 512 threads), p3 prod, crt prod.
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

// Per stage (0 P1, 1 P2, 2 P3, 3 CRT): fn = kernel substitute (null = production), gx / bx = grid.x / block.x substitute
// (0 = production); ctab / bpt = P1 table substitutes (0 = production); check = exactness vs the production graph.
struct BatchOvr { CUfunction fn[4]{}; unsigned gx[4]{}; unsigned bx[4]{}; CUdeviceptr ctab=0,bpt=0; bool check=true; };
struct BatchRes { bool ok=false; int ex=-1; double vs=0,med=0,prod=0; };

template<class C> struct ComboMods {
    struct Prog { std::string name,src; std::vector<std::string> kn; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t=0; };
    C& d; std::string arch; std::vector<Prog> progs;
    explicit ComboMods(C& c):d(c){
        int cc1=0;CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
        if(const char* e=std::getenv("FHERMA_ARCH"))arch=e;else arch=cc1>=9?"compute_90":"compute_80";}   // = production module option
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

// src: production NVRTC source (prefix + kernels); line: State::batch_line; cmp: State::batch_cmp (set while the lines run);
// prodfn: production P1 / P2 / P3 / CRT functions; ctab_wl / bpt_wl: whole-line P1 tables (device, built by the caller).
template<class C> inline void run_combo(C& d,const std::string& src,const CUfunction (&prodfn)[4],CUdeviceptr ctab_wl,CUdeviceptr bpt_wl,
    CUdeviceptr ctab,CUdeviceptr bpt,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):50.0;
    int cc1=0,cc2=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);}}
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",
        {"bpo_middle_r16<0>","bpo_last_cm3<0>","bpo_crt_lean<0>","cb_cmp","bpo_first_x<0>"});
    const int m_p1=M.add("p1wl","#define BPO_P1WL 1\n"+s,{"bpo_first_x<0>"});
    const int m_p2=M.add("p2w","#define BPO_P2W 1\n"+s,{"bpo_middle_r16<0>"});
    const int m_p2d=M.add("p2wd","#define BPO_P2W 2\n"+s,{"bpo_middle_r16<0>"});   // ablation: p2w structure, production stores
    M.build();
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":\"wline\",\"cc\":\"%d.%d\",\"tables\":%d,\"compile_s\":%.1f,\"mods\":{%s}}\n",cc1,cc2,int(ctab_wl&&bpt_wl),el(),info.c_str());std::fflush(stdout);}
    {   // JIT registers / local bytes / static smem / max threads per kernel as loaded on this device
        auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");std::string r;
        auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
        auto one=[&](const char* nm,CUfunction f){if(!f)return;int rg=0,lc=0,sm=0,mt=0,ob=0;ga(&rg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&lc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);ga(&sm,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);ga(&mt,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f);
            const int bt=std::string(nm).rfind("p2w",0)==0?512:std::string(nm)=="p2"?128:std::string(nm)=="crt"?128:256;if(occ)occ(&ob,f,bt,0);char b[112];std::snprintf(b,sizeof b,"%s\"%s\":[%d,%d,%d,%d,%d]",r.empty()?"":",",nm,rg,lc,sm,mt,ob);r+=b;};
        one("p1",prodfn[0]);one("p2",prodfn[1]);one("p3",prodfn[2]);one("crt",prodfn[3]);one("p1wl",M.fn(m_p1));one("p2w",M.fn(m_p2));one("p2wd",M.fn(m_p2d));
        std::printf("{\"diag\":\"sw\",\"k\":\"combo_regs\",\"regs_local_smem_thr_bpsm\":{%s}}\n",r.c_str());std::fflush(stdout);}
    cmp=M.fn(m_rc,3);
    // ---- whole-batch lines
    struct L{std::string v;BatchOvr o;bool var=false;BatchRes r;};std::vector<L> done;
    auto run=[&](L x,bool rep){
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return;}
        if(x.var&&!x.o.fn[0]&&!x.o.fn[1]&&!x.o.fn[2]&&!x.o.fn[3]){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",x.v.c_str());std::fflush(stdout);return;}
        x.r=line(x.o,x.v.c_str());if(x.r.ok&&!rep)done.push_back(x);};
    auto mk=[](const char* v){L x;x.v=v;return x;};
    run(mk("ctrl"),false);
    {L x=mk("rc");x.var=true;x.o.fn[0]=M.fn(m_rc,4);x.o.fn[1]=M.fn(m_rc,0);x.o.fn[2]=M.fn(m_rc,1);x.o.fn[3]=M.fn(m_rc,2);
     if(!x.o.fn[0]||!x.o.fn[1]||!x.o.fn[2]||!x.o.fn[3])x.o=BatchOvr{};run(x,false);}
    auto p1wl=[&](L& x){x.var=true;if(M.fn(m_p1)&&ctab_wl&&bpt_wl){x.o.fn[0]=M.fn(m_p1);x.o.ctab=ctab_wl;x.o.bpt=bpt_wl;}};
    auto p2w=[&](L& x){x.var=true;if(M.fn(m_p2)){x.o.fn[1]=M.fn(m_p2);x.o.gx[1]=8;x.o.bx[1]=512;}};
    {L x=mk("p1wl");p1wl(x);run(x,false);}
    {L x=mk("p2w");p2w(x);run(x,false);}
    {L x=mk("p2w_direct");x.var=true;if(M.fn(m_p2d)){x.o.fn[1]=M.fn(m_p2d);x.o.gx[1]=8;x.o.bx[1]=512;}run(x,false);}
    {L x=mk("wl");p1wl(x);p2w(x);if(!x.o.fn[0]||!x.o.fn[1])x.o=BatchOvr{};run(x,false);}
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
        struct V{const char* mod;CUfunction f;unsigned gx,bx;CUdeviceptr ct,bp;};
        for(int q=0;q<4;++q){
            const KProf::Trace* t=nullptr;for(const auto& x:tr)if(x.product==0&&x.stage==sst[q]&&!x.pb.empty()&&x.fn==prodfn[q]){t=&x;break;}
            if(!t){std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"skipped\":\"no trace\"}\n",SN[q]);std::fflush(stdout);continue;}
            std::vector<V> vs={{"prod",prodfn[q],t->gx,t->bx,0,0}};
            if(q==0){vs.push_back({"rc",M.fn(m_rc,4),t->gx,t->bx,0,0});if(ctab_wl&&bpt_wl)vs.push_back({"p1wl",M.fn(m_p1),t->gx,t->bx,ctab_wl,bpt_wl});}
            if(q==1){vs.push_back({"rc",M.fn(m_rc,0),t->gx,t->bx,0,0});vs.push_back({"p2w",M.fn(m_p2),t->gx/4u,512u,0,0});vs.push_back({"p2w_direct",M.fn(m_p2d),t->gx/4u,512u,0,0});}
            const size_t np=t->po.size();
            std::vector<std::vector<unsigned char>> pbs(vs.size(),t->pb);std::vector<std::vector<void*>> pps(vs.size(),std::vector<void*>(np));
            for(size_t m=0;m<vs.size();++m){for(size_t i=0;i<np;++i)pps[m][i]=pbs[m].data()+t->po[i];
                if(vs[m].ct&&np>12){   // bpo_first_x params: 6 = bpt, 12 = ctab (patched only when they hold the production tables)
                    CUdeviceptr x=0;std::memcpy(&x,pps[m][6],sizeof x);if(x==bpt)std::memcpy(pps[m][6],&vs[m].bp,sizeof x);
                    std::memcpy(&x,pps[m][12],sizeof x);if(x==ctab)std::memcpy(pps[m][12],&vs[m].ct,sizeof x);}}
            std::vector<std::vector<float>> v(vs.size());std::vector<bool> bad(vs.size(),false);
            for(int r=0;r<R+1;++r)for(size_t m=0;m<vs.size();++m){if(!vs[m].f||bad[m])continue;
                kp.erecord(e0,st);for(int k=0;k<NL;++k)if(d.cuLaunchKernel(vs[m].f,vs[m].gx,t->gy,t->gz,vs[m].bx,1,1,t->smem,st,pps[m].data(),nullptr)!=CUDA_SUCCESS)bad[m]=true;
                kp.erecord(e1,st);if(esync(e1)!=CUDA_SUCCESS)bad[m]=true;if(r>0&&!bad[m])v[m].push_back(kp.ms(e0,e1)*1e3f/NL);}
            double ref=0;
            for(size_t m=0;m<vs.size();++m){if(!vs[m].f)continue;
                if(bad[m]||v[m].empty()){std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"mod\":\"%s\",\"error\":\"launch\"}\n",SN[q],vs[m].mod);std::fflush(stdout);continue;}
                std::sort(v[m].begin(),v[m].end());const double med=v[m][v[m].size()/2];if(m==0)ref=med;
                std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"mod\":\"%s\",\"us\":%.3f,\"us_min\":%.3f,\"vs_prod\":%.4f,\"cyc_nominal\":%.0f,\"grid\":[%u,%u,%u],\"block\":%u}\n",
                    SN[q],vs[m].mod,med,v[m].front(),ref>0?med/ref:0.0,med*clk*1e-3,vs[m].gx,t->gy,t->gz,vs[m].bx);std::fflush(stdout);}}
        kp.edestroy(e0);kp.edestroy(e1);d.cuCtxSynchronize();sdestroy(st);}}
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
