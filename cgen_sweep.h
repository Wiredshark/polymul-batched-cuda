#pragma once
// Round 24 (cgen, on 3140af3 = production): exit-time whole-batch CODE-GENERATION sweep. The production NVRTC source (prefix +
// kernels, unchanged, plus one appended compare kernel cg_cmp) is compiled into a separate CUmodule per variant; the batch path's
// four kernels (P1 bpo_first_x, P2 bpo_middle_r16, P3 bpo_last_cm3, CRT bpo_crt_lean) are rebound to that module (BatchOvr fn[0..3],
// production grid / threads / arguments) and the whole B-product graph is timed paired against the production graph exec
// (State::batch_line, vs_prod < 1 = faster) with every output compared exactly against production (ex 1 = all B outputs
// bit-identical; a winner would ship). Variants:
//   t80 / t86 / t89  --gpu-architecture=compute_80 / 86 / 89 (no TMA: BPO_X_HAS_TMA 0, P1 stores its layout-2 results directly =
//                    the existing compute_80 fallback path; tma 0 in cgen_info)
//   t90              compute_90 = production rebuilt (sanity; should match ctrl)
//   t100 / t120      compute_100 / compute_120 (skipped with the NVRTC reason if rejected)
//   jo3 / jo2        compute_90 PTX of t90 loaded with cuModuleLoadDataEx CU_JIT_OPTIMIZATION_LEVEL 3 / 2 (driver default 4)
//   dopt / edv / dd  compute_90 + NVRTC "-dopt=on" / "--extra-device-vectorization" / "-default-device"
// Lines: ctrl, rc (production kernels recompiled as template instances, as in e01e292), every built variant, one repeat of each
// (_r, after all singles), ctrl_end. cgen_regs: JIT registers / local (spill) bytes per kernel per variant on this device.
// FHERMA_CGEN=0 off, FHERMA_CGEN_S time budget (s, default 170, compiles included), FHERMA_BATCH_R pairs per line (default 13),
// FHERMA_CGEN_ONLY=comma list of variant names (default all).
#include <thread>
#include <atomic>
#include <cstring>
#include <cstdint>

template<class C> inline void run_cgen(C& d,const std::string& src,const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_CGEN"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_CGEN_S")?std::atof(std::getenv("FHERMA_CGEN_S")):170.0;
    const char* only=std::getenv("FHERMA_CGEN_ONLY");
    auto want=[&](const char* v){if(!only||!*only)return true;const std::string o=std::string(",")+only+",";return o.find(std::string(",")+v+",")!=std::string::npos;};
    auto addn=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*)>("nvrtcAddNameExpression");
    auto lown=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,const char*,const char**)>("nvrtcGetLoweredName");
    auto setctx=d.driver.template get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");
    auto loadex=d.driver.template get<decltype(&::cuModuleLoadDataEx)>("cuModuleLoadDataEx");
    auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
    std::string prod_arch;{int cc1=0;CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS)attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
        if(const char* e=std::getenv("FHERMA_ARCH"))prod_arch=e;else prod_arch=cc1>=9?"compute_90":"compute_80";}   // = production module option
    static const char* const CMP="\nextern \"C\" __global__ void cg_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n";
    // ---- programs (NVRTC) and modules (driver JIT)
    struct Prog { std::string name,src,arch; std::vector<std::string> opts,kn; std::vector<char> ptx; std::vector<std::string> lowered; std::string err; double t=0; };
    struct Mod { std::string name; int prog=-1,olev=-1; CUmodule mod=nullptr; CUfunction fn[5]{}; std::string err; double t=0; bool tma=true; };
    std::vector<Prog> P;std::vector<Mod> M;
    static const char* const KN[]={"bpo_first_x","bpo_middle_r16","bpo_last_cm3","bpo_crt_lean","cg_cmp"};   // fn[0..3] = stage P1, P2, P3, CRT
    auto addp=[&](const char* nm,std::string s,const char* arch,std::vector<std::string> opts,std::vector<std::string> kn){Prog p;p.name=nm;p.src=std::move(s);p.arch=arch;p.opts=std::move(opts);p.kn=std::move(kn);P.push_back(std::move(p));return int(P.size())-1;};
    auto addm=[&](const char* nm,int prog,int olev=-1){Mod m;m.name=nm;m.prog=prog;m.olev=olev;M.push_back(m);return int(M.size())-1;};
    // rc: production kernels as template instances (every extern "C" kernel an uninstantiated template), as e01e292's rc line
    {std::string s=src;const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}
     addm("rc",addp("rc",s+CMP,prod_arch.c_str(),{},{"bpo_first_x<0>","bpo_middle_r16<0>","bpo_last_cm3<0>","bpo_crt_lean<0>","cg_cmp"}));}
    const std::vector<std::string> kn(KN,KN+5);const std::string full=src+CMP;
    struct T{const char* v;const char* arch;const char* opt;};
    static const T TG[]={{"t80","compute_80",nullptr},{"t86","compute_86",nullptr},{"t89","compute_89",nullptr},{"t90","compute_90",nullptr},{"t100","compute_100",nullptr},{"t120","compute_120",nullptr},
        {"dopt","compute_90","-dopt=on"},{"edv","compute_90","--extra-device-vectorization"},{"dd","compute_90","-default-device"}};
    int p90=-1;
    for(const T& t:TG){const bool need90=!std::strcmp(t.v,"t90")&&(want("jo3")||want("jo2"));if(!want(t.v)&&!need90)continue;
        const int pi=addp(t.v,full,t.arch,t.opt?std::vector<std::string>{t.opt}:std::vector<std::string>{},kn);if(!std::strcmp(t.v,"t90"))p90=pi;
        if(want(t.v)){const int mi=addm(t.v,pi);M[size_t(mi)].tma=std::atoi(t.arch+8)>=90;}}
    if(p90>=0){if(want("jo3"))addm("jo3",p90,3);if(want("jo2"))addm("jo2",p90,2);}
    auto pool=[&](size_t cnt,const std::function<void(size_t)>& f){const unsigned hw=std::max(2u,std::thread::hardware_concurrency());const size_t nth=std::min<size_t>(cnt,std::max(1u,std::min(8u,hw-1)));
        std::atomic<size_t> next{0};std::vector<std::thread> th;for(size_t t=0;t<nth;++t)th.emplace_back([&]{for(size_t k;(k=next.fetch_add(1))<cnt;)f(k);});for(auto& t:th)t.join();};
    auto clean=[](std::string s){for(char& ch:s)if(ch=='"'||ch=='\\'||ch=='\n'||ch=='\r'||ch=='\t')ch=' ';return s;};
    pool(P.size(),[&](size_t i){Prog& p=P[i];const auto tA=std::chrono::steady_clock::now();
        try{nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,p.src.c_str(),"kernels.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("create");
            for(const auto& k:p.kn)if(k.find('<')!=std::string::npos)addn(pr,k.c_str());
            std::vector<std::string> os{"--gpu-architecture="+p.arch,"--std=c++17"};for(const auto& o:p.opts)os.push_back(o);   // = production options (+ the variant's)
            std::vector<const char*> op;for(const auto& o:os)op.push_back(o.c_str());
            const nvrtcResult rr=d.nvrtcCompileProgram(pr,int(op.size()),op.data());
            if(rr!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);
                const size_t e=l.find("rror");throw std::runtime_error("nvrtc "+std::to_string(int(rr))+": "+l.substr(e==std::string::npos?0:(e>1?e-1:0),160));}
            size_t n=0;d.nvrtcGetPTXSize(pr,&n);p.ptx.resize(n);d.nvrtcGetPTX(pr,p.ptx.data());
            p.lowered.assign(p.kn.size(),std::string());
            for(size_t k=0;k<p.kn.size();++k){if(p.kn[k].find('<')==std::string::npos){p.lowered[k]=p.kn[k];continue;}const char* ln=nullptr;if(lown(pr,p.kn[k].c_str(),&ln)==NVRTC_SUCCESS&&ln)p.lowered[k]=ln;}
            d.nvrtcDestroyProgram(&pr);
        }catch(const std::exception& e){p.err=clean(e.what());p.ptx.clear();}
        p.t=std::chrono::duration<double>(std::chrono::steady_clock::now()-tA).count();});
    const double nvrtc_s=el();
    pool(M.size(),[&](size_t i){Mod& m=M[i];const Prog& p=P[size_t(m.prog)];const auto tA=std::chrono::steady_clock::now();
        if(p.ptx.empty()){m.err=p.err.empty()?"no ptx":p.err;return;}
        setctx(d.context);CUresult r;
        if(m.olev>=0){CUjit_option jo[1]={CU_JIT_OPTIMIZATION_LEVEL};void* jv[1]={reinterpret_cast<void*>(uintptr_t(m.olev))};r=loadex(&m.mod,p.ptx.data(),1,jo,jv);}
        else r=d.cuModuleLoadData(&m.mod,p.ptx.data());
        if(r!=CUDA_SUCCESS){const char* s=nullptr;d.cuGetErrorString(r,&s);m.err=std::string("module load: ")+(s?s:"?");m.mod=nullptr;}
        else for(size_t k=0;k<5;++k){CUfunction f=nullptr;if(!p.lowered[k].empty()&&d.cuModuleGetFunction(&f,m.mod,p.lowered[k].c_str())==CUDA_SUCCESS)m.fn[k]=f;}
        m.t=std::chrono::duration<double>(std::chrono::steady_clock::now()-tA).count();});
    {std::string info;for(const auto& m:M){const Prog& p=P[size_t(m.prog)];std::string opt;for(const auto& o:p.opts)opt+=(opt.empty()?"":" ")+o;
            char b[480];std::snprintf(b,sizeof b,"%s\"%s\":{\"arch\":\"%s\",\"opt\":\"%s\",\"jit_O\":%d,\"tma\":%d,\"nvrtc_s\":%.1f,\"load_s\":%.1f,\"ptx_kb\":%zu,\"ok\":%d%s%s%s}",info.empty()?"":",",m.name.c_str(),p.arch.c_str(),opt.c_str(),m.olev,int(m.tma),p.t,m.t,p.ptx.size()/1024,int(m.mod!=nullptr&&m.fn[0]&&m.fn[1]&&m.fn[2]&&m.fn[3]),m.err.empty()?"":",\"err\":\"",m.err.substr(0,200).c_str(),m.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"cgen_info\",\"round\":24,\"prod_arch\":\"%s\",\"nvrtc_s\":%.1f,\"compile_s\":%.1f,\"mods\":{%s}}\n",prod_arch.c_str(),nvrtc_s,el(),info.c_str());std::fflush(stdout);}
    {   // JIT register / local-memory (spill) footprint per kernel per variant as loaded on this device ([regs, local bytes])
        auto one=[&](std::string& r,const char* nm,CUfunction f){if(!f)return;int rg=0,lc=0;ga(&rg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&lc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);char b[64];std::snprintf(b,sizeof b,"%s\"%s\":[%d,%d]",r.empty()?"":",",nm,rg,lc);r+=b;};
        std::string all;
        {std::string r;one(r,"p1",d.bpo_first_x);one(r,"p2",d.bpo_middle_r16);one(r,"p3",d.bpo_last_cm3);one(r,"crt",d.bpo_crt_lean);all+="\"prod\":{"+r+"}";}
        for(const auto& m:M){if(!m.mod)continue;std::string r;one(r,"p1",m.fn[0]);one(r,"p2",m.fn[1]);one(r,"p3",m.fn[2]);one(r,"crt",m.fn[3]);all+=",\""+m.name+"\":{"+r+"}";}
        std::printf("{\"diag\":\"sw\",\"k\":\"cgen_regs\",\"regs_local\":{%s}}\n",all.c_str());std::fflush(stdout);}
    // device compare kernel (exactness of every line against the production outputs): rc's, else any built module's
    cmp=nullptr;for(const auto& m:M)if(m.fn[4]){cmp=m.fn[4];break;}
    // ---- lines
    struct L{std::string v;CUfunction f[4]{};bool var=false;};
    std::vector<L> done;
    auto run=[&](L x){
        if(el()>budget){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",x.v.c_str());std::fflush(stdout);return;}
        BatchOvr o;o.check=true;for(int q=0;q<4;++q)o.fn[q]=x.f[q];
        if(line(o,x.v.c_str()).ok)done.push_back(x);};
    run(L{"ctrl"});
    for(const auto& m:M){
        if(!m.mod||!m.fn[0]||!m.fn[1]||!m.fn[2]||!m.fn[3]){std::printf("{\"diag\":\"sw\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"%s\"}\n",m.name.c_str(),m.err.empty()?"kernel missing":m.err.substr(0,200).c_str());std::fflush(stdout);continue;}
        L x;x.v=m.name;x.var=m.name!="rc";for(int q=0;q<4;++q)x.f[q]=m.fn[q];run(x);}
    const size_t nsingle=done.size();
    for(size_t i=0;i<nsingle;++i){if(!done[i].var)continue;L x=done[i];x.v+="_r";x.var=false;run(x);}
    run(L{"ctrl_end"});
    std::printf("{\"diag\":\"sw\",\"k\":\"cgen_done\",\"s\":%.2f,\"lines\":%zu}\n",el(),done.size());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
    for(auto& m:M)if(m.mod)d.cuModuleUnload(m.mod);
}
