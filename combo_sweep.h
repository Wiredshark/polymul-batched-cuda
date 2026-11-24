#pragma once
// Round 23 (kill, on 3140af3 = production): kill-test ablation lines (clean_sweep.cuh kernels, module "var" = production source
// + CL_SWEEP_SOURCE; wrong values by design, o.check off: ex 0 + abl 1, timing only). Keys: ctrl / rc / ctrl_end controls (rc =
// production P1 / P2 / P3 / CRT recompiled in the sweep module, exact); nostage (P1 phase B minus the len-16 stage, P2 and P3
// one extra dummy radix-2 stage each: P1 -> P2 stage move estimate), nostage_p1 (P1 minus one stage only: upper bound of the P1
// saving), notma (P1 phase B without the dense TMA staging copy and its per-pair __syncwarp, results stored in place), notma_bar
// (only the per-pair __syncwarp removed); every variant line repeated once (_r, after all singles). Same launch geometry as
// production. combo_regs: JIT registers / local bytes per kernel on the device. (Round 22 p1old / p1k120 / p1k112 removed.)
// Mechanism (round 19, combo2): exit-time
// WHOLE-BATCH lines (State::combo_sweep -> run_combo). Every line recaptures the full B-product production graph (variant 6, as
// fherma_run launches it) with one set of substitutions (State::batch_line: per-stage kernel from a sweep module, environment
// overrides read at capture time, CRT prefetch items) and times it against the production graph exec in paired, alternating
// reps (vs_prod = median of pair ratios; < 1 = faster). ex = 1 when all B outputs are bit-identical to production (device compare).
// Sweep modules: the production NVRTC source (prefix + kernels, same arch option) with every extern "C" kernel turned into an
// uninstantiated template (only the named instances compile); compiled in worker threads at exit.
// FHERMA_COMBO=0 off, FHERMA_COMBO_S time budget (s, default 55, compiles included), FHERMA_BATCH_R pairs per line (default 13).
#include <thread>
#include <atomic>
#include <cmath>

template<class C> struct ComboMods {
    struct Prog { std::string name,src; std::vector<std::string> kn; bool dd=false; CUmodule mod=nullptr; std::vector<CUfunction> fn; std::string err; double t=0; };
    C& d; std::string arch; std::vector<Prog> progs;
    explicit ComboMods(C& c):d(c){
        int cc1=0;CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
        if(const char* e=std::getenv("FHERMA_ARCH"))arch=e;else arch=cc1>=9?"compute_90":"compute_80";}   // = production module option
    ~ComboMods(){d.cuCtxSynchronize();for(auto& P:progs)if(P.mod)d.cuModuleUnload(P.mod);}
    int add(const char* name,std::string src,std::vector<std::string> kn,bool dd=false){Prog P;P.name=name;P.src=std::move(src);P.kn=std::move(kn);P.dd=dd;progs.push_back(std::move(P));return int(progs.size())-1;}
    void compile(Prog& P){
        const auto tA=std::chrono::steady_clock::now();
        try{
            auto addn=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*)>("nvrtcAddNameExpression");
            auto lown=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*,const char**)>("nvrtcGetLoweredName");
            auto setctx=d.driver.template get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");
            nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,P.src.c_str(),"combo.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("create");
            for(const auto& k:P.kn)if(k.find('<')!=std::string::npos)addn(pr,k.c_str());
            const std::string a="--gpu-architecture="+arch;const char* op[]={a.c_str(),"--std=c++17","-default-device"};
            if(d.nvrtcCompileProgram(pr,P.dd?3:2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
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
    void build(){   // all programs in parallel worker threads
        const unsigned hw=std::max(2u,std::thread::hardware_concurrency());const size_t nth=std::min<size_t>(progs.size(),std::max(1u,std::min(8u,hw-1)));
        std::atomic<size_t> next{0};std::vector<std::thread> th;
        for(size_t t=0;t<nth;++t)th.emplace_back([&]{for(size_t k;(k=next.fetch_add(1))<progs.size();)compile(progs[k]);});
        for(auto& t:th)t.join();}
    CUfunction fn(int p,size_t k=0)const{return p>=0&&size_t(p)<progs.size()&&k<progs[size_t(p)].fn.size()?progs[size_t(p)].fn[k]:nullptr;}
};

// src: production NVRTC source (prefix + kernels); line: State::batch_line; cmp: State::batch_cmp (set while the lines run).
#include <cstring>
// ---- Round 25 (cubd, on 3140af3): target-JIT code identification. Whole-batch paired lines where the P1 / P2 / P3 / CRT stage
// kernels (bpo_first_x, bpo_middle_r16, bpo_last_cm3, bpo_crt_lean; all other kernels stay the production module's) come from
// precompiled sm_120 cubins (cubin_diag.h): cub_jit616 = the local RTX 5070 driver 616.56 JIT cubin of the production PTX,
// cub_nvrtc128 = the offline NVRTC 12.8 sm_120 cubin of the same source; mix_p1 / mix_p23 = jit616 for P1 only / P2+P3 only.
// A refused cubin load prints one JSON line and its lines are skipped (production path untouched). Then kernel-level lines
// ("kcyc": one captured production launch per stage replayed back to back, us per launch, per module, local 5070 value beside)
// and identification lines ("cubd_id": driver / NVRTC / NVML versions, PTX header + hash, per-kernel attributes, and the
// target JIT's own code fingerprint: the production PTX JIT-linked with cuLink* on this driver, per-kernel .text size + hash,
// next to the same fingerprint of both embedded cubins).
#include "cubin_diag.h"
inline std::string g_prod_ptx;   // copy of the production module PTX (set in Cuda::Cuda, host only), for the exit-time JIT fingerprint
#include <dlfcn.h>
inline unsigned long long cubd_fnv(const void* p,size_t n){const unsigned char* b=static_cast<const unsigned char*>(p);unsigned long long h=0xcbf29ce484222325ull;for(size_t i=0;i<n;++i){h^=b[i];h*=0x100000001b3ull;}return h;}
struct CubdText { size_t size=0; unsigned long long h=0,hm=0; };   // hm: hash with the 32-bit immediate half of every instruction zeroed (relocation-insensitive)
inline CubdText cubd_text(const unsigned char* b,size_t n,const char* kern){   // ELF64: .text.<kern> section size + FNV hash
    CubdText r;if(!b||n<64||b[0]!=0x7f||b[1]!='E'||b[4]!=2)return r;
    unsigned long long shoff=0;unsigned short shes=0,shn=0,shstr=0;std::memcpy(&shoff,b+0x28,8);std::memcpy(&shes,b+0x3a,2);std::memcpy(&shn,b+0x3c,2);std::memcpy(&shstr,b+0x3e,2);
    if(shoff+size_t(shn)*shes>n||shstr>=shn)return r;
    auto sh=[&](unsigned i,unsigned o,unsigned w)->unsigned long long{unsigned long long v=0;std::memcpy(&v,b+shoff+size_t(i)*shes+o,w);return v;};
    const unsigned long long so=sh(shstr,0x18,8);const std::string want=std::string(".text.")+kern;
    for(unsigned i=0;i<shn;++i){const unsigned long long nm=sh(i,0,4),off=sh(i,0x18,8),sz=sh(i,0x20,8);
        if(so+nm>=n)continue;const char* s=reinterpret_cast<const char*>(b+so+nm);
        if(want==s&&off+sz<=n){r.size=size_t(sz);r.h=cubd_fnv(b+off,size_t(sz));std::vector<unsigned char> m(b+off,b+off+sz);for(size_t w=0;w+16<=m.size();w+=16)std::memset(m.data()+w+4,0,4);r.hm=cubd_fnv(m.data(),m.size());return r;}}
    return r;}
template<class C> inline void run_cubd(C& d,const std::string& src,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction rc[4],const std::function<bool()>& over){
    static const char* const KN[4]={"bpo_first_x","bpo_middle_r16","bpo_last_cm3","bpo_crt_lean"};
    static const char* const SN[4]={"p1","p2","p3","crt"};
    auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
    if(const char* o=std::getenv("FHERMA_CUBD_PTX_OUT"))if(FILE* f=std::fopen(o,"wb")){std::fwrite(g_prod_ptx.data(),1,g_prod_ptx.size(),f);std::fclose(f);}   // local tool: dump the production PTX
    // ---- identification
    {int drv=0;d.driver.template get<decltype(&::cuDriverGetVersion)>("cuDriverGetVersion")(&drv);int rm=0,rn=0;
     try{d.rtc.template get<nvrtcResult(*)(int*,int*)>("nvrtcVersion")(&rm,&rn);}catch(...){}
     char nvs[96]="";if(void* h=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL)){auto ini=reinterpret_cast<int(*)()>(dlsym(h,"nvmlInit_v2"));auto dv=reinterpret_cast<int(*)(char*,unsigned)>(dlsym(h,"nvmlSystemGetDriverVersion"));
         if(ini&&dv&&ini()==0)dv(nvs,sizeof nvs);}
     for(char* c=nvs;*c;++c)if(*c=='"'||*c=='\\')*c='_';
     std::string pv,pt;{const std::string& p=g_prod_ptx;size_t a=p.find(".version ");if(a!=std::string::npos)pv=p.substr(a+9,p.find('\n',a)-a-9);a=p.find(".target ");if(a!=std::string::npos)pt=p.substr(a+8,p.find('\n',a)-a-8);}
     for(auto* s:{&pv,&pt})for(char& c:*s)if(c=='"'||c=='\\'||c=='\r')c=' ';
     std::printf("{\"diag\":\"cubd_id\",\"k\":\"versions\",\"driver_api\":%d,\"nvml_driver\":\"%s\",\"nvrtc\":\"%d.%d\",\"ptx_version\":\"%s\",\"ptx_target\":\"%s\",\"ptx_bytes\":%zu,\"ptx_fnv\":\"%016llx\",\"ptx_fnv_local\":\"%016llx\",\"src_bytes\":%zu,\"src_fnv\":\"%016llx\",\"src_fnv_local\":\"%016llx\",\"src_match\":%d}\n",
         drv,nvs,rm,rn,pv.c_str(),pt.c_str(),g_prod_ptx.size(),cubd_fnv(g_prod_ptx.data(),g_prod_ptx.size()),CUBD_PTX_FNV,src.size(),cubd_fnv(src.data(),src.size()),CUBD_SRC_FNV,int(cubd_fnv(src.data(),src.size())==CUBD_SRC_FNV));std::fflush(stdout);}
    // ---- cubin modules (failure-tolerant)
    struct Mod{const char* name;const unsigned char* img;size_t n;CUmodule m=nullptr;CUfunction f[4]{};};
    Mod mods[2]={{"jit616",CUBD_JIT616,sizeof(CUBD_JIT616)},{"nvrtc128",CUBD_NVRTC128,sizeof(CUBD_NVRTC128)}};
    for(auto& M:mods){const auto t=std::chrono::steady_clock::now();const CUresult r=d.cuModuleLoadData(&M.m,M.img);
        if(r!=CUDA_SUCCESS){M.m=nullptr;const char* es=nullptr;d.cuGetErrorString(r,&es);std::printf("{\"diag\":\"cubd_id\",\"k\":\"load\",\"mod\":\"%s\",\"ok\":0,\"rc\":%d,\"err\":\"%.60s\"}\n",M.name,int(r),es?es:"");std::fflush(stdout);continue;}
        int nf=0;for(int q=0;q<4;++q)if(d.cuModuleGetFunction(&M.f[q],M.m,KN[q])!=CUDA_SUCCESS)M.f[q]=nullptr;else ++nf;
        std::printf("{\"diag\":\"cubd_id\",\"k\":\"load\",\"mod\":\"%s\",\"ok\":1,\"kernels\":%d,\"bytes\":%zu,\"s\":%.3f}\n",M.name,nf,M.n,std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());std::fflush(stdout);}
    const CUfunction prod[4]={d.bpo_first_x,d.bpo_middle_r16,d.bpo_last_cm3,d.bpo_crt_lean};
    {   // per-kernel attributes as loaded: regs, local bytes, static smem, binaryVersion, ptxVersion
        std::string r;auto one=[&](const char* mod,int q,CUfunction f){if(!f)return;int a[5]={};ga(&a[0],CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&a[1],CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);ga(&a[2],CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);ga(&a[3],CU_FUNC_ATTRIBUTE_BINARY_VERSION,f);ga(&a[4],CU_FUNC_ATTRIBUTE_PTX_VERSION,f);
            char b[128];std::snprintf(b,sizeof b,"%s\"%s/%s\":[%d,%d,%d,%d,%d]",r.empty()?"":",",SN[q],mod,a[0],a[1],a[2],a[3],a[4]);r+=b;};
        for(int q=0;q<4;++q){one("target_jit",q,prod[q]);for(auto& M:mods)one(M.name,q,M.f[q]);}
        std::printf("{\"diag\":\"cubd_id\",\"k\":\"attrs\",\"regs_local_smem_bin_ptx\":{%s}}\n",r.c_str());std::fflush(stdout);}
    // ---- whole-batch lines
    struct L{std::string v;CUfunction f[4]{};BatchRes r;};std::vector<L> done;
    auto run=[&](L x,bool rep){
        if(over()){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return;}
        if(!x.f[0]&&!x.f[1]&&!x.f[2]&&!x.f[3]&&x.v.rfind("ctrl",0)!=0){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"module missing\"}\n",x.v.c_str());std::fflush(stdout);return;}
        BatchOvr o;o.check=true;for(int q=0;q<4;++q)o.fn[q]=x.f[q];x.r=line(o,x.v.c_str());if(x.r.ok&&!rep)done.push_back(x);};
    auto mk=[&](const char* v,const Mod* m,unsigned mask)->L{L x;x.v=v;for(int q=0;q<4;++q)if(mask>>q&1){x.f[q]=m?m->f[q]:rc[q];if(!x.f[q]){x.f[0]=x.f[1]=x.f[2]=x.f[3]=nullptr;break;}}return x;};
    run(mk("ctrl",nullptr,0),false);
    run(mk("rc",nullptr,15),false);
    run(mk("cub_jit616",&mods[0],15),false);
    run(mk("cub_nvrtc128",&mods[1],15),false);
    run(mk("mix_p1",&mods[0],1),false);
    run(mk("mix_p23",&mods[0],6),false);
    {const size_t n=done.size();for(size_t i=0;i<n;++i){if(done[i].v=="ctrl")continue;L x=done[i];x.v+="_r";run(x,true);}}
    run(mk("ctrl_end",nullptr,0),true);
    // ---- kernel-level lines: one captured production launch per stage, replayed back to back (R alternating rounds)
    {auto& kp=d.kp;const int var=kp.diag_variant();
     if(var<0||kp.traces[var].empty()){std::printf("{\"diag\":\"kcyc\",\"skipped\":\"no captured launches (FHERMA_DIAG=0 or no cuFuncGetParamInfo)\"}\n");std::fflush(stdout);}
     else{const auto& tr=kp.traces[var];
        auto screate=d.driver.template get<decltype(&::cuStreamCreate)>("cuStreamCreate");auto sdestroy=d.driver.template get<decltype(&::cuStreamDestroy)>("cuStreamDestroy_v2");
        auto esync=d.driver.template get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");
        CUstream st=nullptr;d.check(screate(&st,CU_STREAM_NON_BLOCKING));CUevent e0=nullptr,e1=nullptr;kp.ecreate(&e0,CU_EVENT_DEFAULT);kp.ecreate(&e1,CU_EVENT_DEFAULT);
        int clk=0;CUdevice dev=0;d.cuDeviceGet(&dev,0);auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");attr(&clk,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev);
        const int R=std::getenv("FHERMA_KCYC_R")?std::max(3,std::atoi(std::getenv("FHERMA_KCYC_R"))):15,NL=8;   // NL launches per timed sample
        const int sst[4]={KProf::P1,KProf::P2,KProf::P3,KProf::CRT};
        for(int q=0;q<4;++q){
            const KProf::Trace* t=nullptr;for(const auto& x:tr)if(x.product==0&&x.stage==sst[q]&&!x.pb.empty()&&x.fn==prod[q]){t=&x;break;}
            if(!t){std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"skipped\":\"no trace\"}\n",SN[q]);std::fflush(stdout);continue;}
            std::vector<unsigned char> pb=t->pb;void* pp[64];for(size_t i=0;i<t->po.size()&&i<64;++i)pp[i]=pb.data()+t->po[i];
            const char* mn[3]={"target_jit","jit616","nvrtc128"};const CUfunction fs[3]={prod[q],mods[0].f[q],mods[1].f[q]};
            std::vector<float> v[3];bool bad[3]={};
            for(int r=0;r<R+1;++r)for(int m=0;m<3;++m){if(!fs[m]||bad[m])continue;
                kp.erecord(e0,st);for(int k=0;k<NL;++k)if(d.cuLaunchKernel(fs[m],t->gx,t->gy,t->gz,t->bx,1,1,t->smem,st,pp,nullptr)!=CUDA_SUCCESS)bad[m]=true;
                kp.erecord(e1,st);if(esync(e1)!=CUDA_SUCCESS)bad[m]=true;if(r>0&&!bad[m])v[m].push_back(kp.ms(e0,e1)*1e3f/NL);}
            for(int m=0;m<3;++m){if(!fs[m])continue;
                if(bad[m]||v[m].empty()){std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"mod\":\"%s\",\"error\":\"launch\"}\n",SN[q],mn[m]);std::fflush(stdout);continue;}
                std::sort(v[m].begin(),v[m].end());const double med=v[m][v[m].size()/2];
                char key[48];std::snprintf(key,sizeof key,"%s/%s",SN[q],mn[m]);double loc=-1;for(const CubdLocal* c=CUBD_LOCAL;c->key;++c)if(!std::strcmp(c->key,key))loc=c->us;
                std::printf("{\"diag\":\"kcyc\",\"k\":\"%s\",\"mod\":\"%s\",\"us\":%.3f,\"us_min\":%.3f,\"cyc_nominal\":%.0f,\"grid\":[%u,%u,%u],\"block\":%u,\"local5070_us\":%.3f}\n",
                    SN[q],mn[m],med,v[m].front(),med*clk*1e-3,t->gx,t->gy,t->gz,t->bx,loc);std::fflush(stdout);}}
        kp.edestroy(e0);kp.edestroy(e1);d.cuCtxSynchronize();sdestroy(st);}}
    // ---- target JIT code fingerprint: the production PTX JIT-linked on this driver (cuLink*), per-kernel .text size + hash
    if(!(std::getenv("FHERMA_CUBD_LINK")&&*std::getenv("FHERMA_CUBD_LINK")=='0')&&!g_prod_ptx.empty()){
        const auto t=std::chrono::steady_clock::now();std::vector<unsigned char> img;std::string err;
        try{auto lc=d.driver.template get<decltype(&::cuLinkCreate)>("cuLinkCreate_v2");auto la=d.driver.template get<decltype(&::cuLinkAddData)>("cuLinkAddData_v2");
            auto lcomp=d.driver.template get<decltype(&::cuLinkComplete)>("cuLinkComplete");auto ld=d.driver.template get<decltype(&::cuLinkDestroy)>("cuLinkDestroy");
            CUlinkState ls=nullptr;if(lc(0,nullptr,nullptr,&ls)!=CUDA_SUCCESS)throw std::runtime_error("cuLinkCreate");
            CUresult r=la(ls,CU_JIT_INPUT_PTX,const_cast<char*>(g_prod_ptx.data()),g_prod_ptx.size(),"prod.ptx",0,nullptr,nullptr);
            void* cub=nullptr;size_t cn=0;if(r==CUDA_SUCCESS)r=lcomp(ls,&cub,&cn);
            if(r==CUDA_SUCCESS&&cub&&cn)img.assign(static_cast<unsigned char*>(cub),static_cast<unsigned char*>(cub)+cn);else err="link rc "+std::to_string(int(r));
            ld(ls);}catch(const std::exception& e){err=e.what();}
        std::string s;for(int q=0;q<4;++q){const CubdText a=cubd_text(img.data(),img.size(),KN[q]),b=cubd_text(CUBD_JIT616,sizeof(CUBD_JIT616),KN[q]),c=cubd_text(CUBD_NVRTC128,sizeof(CUBD_NVRTC128),KN[q]);
            const CubdLink* L=nullptr;for(const CubdLink* x=CUBD_LINK_LOCAL;x->k;++x)if(!std::strcmp(x->k,KN[q]))L=x;
            char x[420];std::snprintf(x,sizeof x,"%s\"%s\":{\"target\":[%zu,\"%016llx\",\"%016llx\"],\"local616\":[%zu,\"%016llx\",\"%016llx\"],\"jit616_cubin_masked\":\"%016llx\",\"nvrtc128\":[%zu,\"%016llx\"],\"same_as_local\":%d,\"same_masked\":%d}",s.empty()?"":",",SN[q],a.size,a.h,a.hm,
                L?L->size:size_t(0),L?L->h:0ull,L?L->hm:0ull,b.hm,c.size,c.hm,int(L&&a.size&&a.size==L->size&&a.h==L->h),int(L&&a.size&&a.size==L->size&&a.hm==L->hm));s+=x;}
        std::printf("{\"diag\":\"cubd_id\",\"k\":\"jit_text\",\"link_bytes\":%zu,\"cubin_fnv\":\"%016llx\",%s\"s\":%.2f,\"text_size_fnv\":{%s}}\n",img.size(),cubd_fnv(img.data(),img.size()),err.empty()?"":("\"err\":\""+err+"\",").c_str(),
            std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count(),s.c_str());std::fflush(stdout);}
    for(auto& M:mods)if(M.m){d.cuCtxSynchronize();d.cuModuleUnload(M.m);}
}
template<class C> inline void run_combo(C& d,const std::string& src,unsigned p1_grid,unsigned cols,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):40.0;
    int cc1=0,cc2=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);}}
    // ---- rc sweep module (production source recompiled, every kernel an uninstantiated template except the named ones)
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",
        {"bpo_middle_r16<0>","bpo_last_cm3<0>","bpo_crt_lean<0>","cb_cmp","bpo_first_x<0>"});
    M.build();
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,80).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":25,\"cc\":\"%d.%d\",\"p1_grid\":%u,\"cols\":%u,\"compile_s\":%.1f,\"mods\":{%s}}\n",cc1,cc2,p1_grid,cols,el(),info.c_str());std::fflush(stdout);}
    cmp=M.fn(m_rc,3);
    CUfunction rc[4]={M.fn(m_rc,4),M.fn(m_rc,0),M.fn(m_rc,1),M.fn(m_rc,2)};
    if(cc1==12&&cc2==0){try{run_cubd(d,src,line,rc,[&]{return el()>budget;});}catch(const std::exception& e){std::printf("{\"diag\":\"cubd\",\"error\":\"%.100s\"}\n",e.what());std::fflush(stdout);}}
    else{std::printf("{\"diag\":\"cubd\",\"skipped\":\"not sm_120\"}\n");std::fflush(stdout);}
    std::printf("{\"diag\":\"sw\",\"k\":\"combo_done\",\"s\":%.2f}\n",el());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
