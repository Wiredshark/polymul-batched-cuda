#pragma once
// ovl diag (round 30, on 6b772c1 = production; production PTX unchanged by this file): compute / memory overlap per kernel in the
// real batch configuration. Each line = the production graph (variant 6: pass order, chains, L2 schedule, all B products)
// recaptured with ONE stage enabled (stage mask) and that stage's kernel substituted; med_us / B = us per product of that stage
// as production schedules it (same grid, block, registers, data sizes). Paired, alternating reps against the full production
// graph (vs_prod meaningless here); outputs not checked (twins compute wrong values on purpose).
// Per stage (p1 bpo_first_xt, p2 bpo_middle_r16m, p3 bpo_last_cm3f, crt bpo_crt_lean):
//   R    production kernel            C  compute-only twin (BPO_TWC: loads from an L2-resident window of tiles, stores predicated off; P1: ablation BPO_XA 12, columns c mod 16, stores predicated off)
//   M    memory-only twin (BPO_TWM wl4_*; P1: ablation BPO_XA 5)    Rlo  production kernel at one block/SM less (dynamic smem)
// Summary line per stage and pass: e = (C + M - R) / min(C, M), pot = R - max(C, M) (us per product). Two passes, ctrl line
// (full production graph, one stage-free reference) in each. FHERMA_COMBO=0 off, FHERMA_COMBO_S budget (default 120 s).
#include <cuda.h>
#include <nvrtc.h>
#include <cstdint>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

struct BatchOvr { CUfunction fn[4]{}; unsigned smem[4]{}; unsigned sm=0; bool check=true; };
struct BatchRes { bool ok=false; int ex=-1; double vs=0,med=0,prod=0; };
struct OvlTabs { CUfunction prod[4]{}; unsigned pad1=0; unsigned b=0; };

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

template<class C> inline void run_combo(C& d,const std::string& src,OvlTabs T,
    const std::function<BatchRes(const BatchOvr&,const char*)>& line,CUfunction& cmp){
    if(const char* e=std::getenv("FHERMA_COMBO"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double budget=std::getenv("FHERMA_COMBO_S")?std::atof(std::getenv("FHERMA_COMBO_S")):120.0;
    int cc1=0,cc2=0;{CUdevice dev=0;auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);}}
    std::string s=src;{const std::string a="extern \"C\" __global__",b="template<int SWD_> __global__";size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}}
    ComboMods<C> M(d);
    const int m_rc=M.add("rc",s+"\nextern \"C\" __global__ void cb_cmp(const uint4* a,const uint4* b,unsigned long long n,unsigned* bad){unsigned c=0;for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(unsigned long long)gridDim.x*blockDim.x){const uint4 x=a[i],y=b[i];c+=(x.x!=y.x)|(x.y!=y.y)|(x.z!=y.z)|(x.w!=y.w);}if(c)atomicAdd(bad,c);}\n",
        {"bpo_first_xt<0>","bpo_middle_r16m<0>","bpo_last_cm3f<0>","bpo_crt_lean<0>","cb_cmp"});
    const int m_c=M.add("twc","#define BPO_TWC 1\n"+s,{"bpo_middle_r16m<0>","bpo_last_cm3f<0>","bpo_crt_lean<0>"});
    const int m_m=M.add("twm","#define BPO_TWM 1\n"+s,{"wl4_p2m<0>","wl4_p3m<0>","wl4_crtm<0>"});
    const int m_x11=M.add("xa12","#define BPO_XA 12\n"+s,{"bpo_first_xt<0>"});
    const int m_x5=M.add("xa5","#define BPO_XA 5\n"+s,{"bpo_first_xt<0>"});
    M.build();
    {std::string info;for(const auto& P:M.progs){char b[160];std::snprintf(b,sizeof b,"%s\"%s\":{\"s\":%.1f,\"ok\":%d%s%s%s}",info.empty()?"":",",P.name.c_str(),P.t,int(P.mod!=nullptr),P.err.empty()?"":",\"err\":\"",P.err.substr(0,60).c_str(),P.err.empty()?"":"\"");info+=b;}
     std::printf("{\"diag\":\"sw\",\"k\":\"combo_info\",\"round\":\"ovl\",\"cc\":\"%d.%d\",\"B\":%u,\"compile_s\":%.1f,\"mods\":{%s}}\n",cc1,cc2,T.b,el(),info.c_str());std::fflush(stdout);}
    const char* SN[4]={"p1","p2","p3","crt"};const unsigned MASK[4]={2u,8u,16u,32u},BT[4]={256u,128u,256u,128u};
    CUfunction RC[4]={M.fn(m_rc,0),M.fn(m_rc,1),M.fn(m_rc,2),M.fn(m_rc,3)};
    CUfunction CF[4]={M.fn(m_x11),M.fn(m_c,0),M.fn(m_c,1),M.fn(m_c,2)},MF[4]={M.fn(m_x5),M.fn(m_m,0),M.fn(m_m,1),M.fn(m_m,2)};
    auto ga=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
    auto fsa=d.driver.template get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");
    auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    unsigned lo_dyn[4]{};int ob[4]{},oblo[4]{};
    for(int q=0;q<4;++q){   // registers / local / static smem / blocks per SM of R, C, M; dynamic smem for one block/SM less
        const unsigned b0=q==0?T.pad1:0u;std::string r;
        auto one=[&](const char* nm,CUfunction f){if(!f)return;int rg=0,lc=0,sm=0,o=0;ga(&rg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);ga(&lc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f);ga(&sm,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);occ(&o,f,int(BT[q]),b0);
            char b[80];std::snprintf(b,sizeof b,"%s\"%s\":[%d,%d,%d,%d]",r.empty()?"":",",nm,rg,lc,sm,o);r+=b;};
        one("R",T.prod[q]);one("C",CF[q]);one("M",MF[q]);
        if(T.prod[q]&&occ(&ob[q],T.prod[q],int(BT[q]),b0)==CUDA_SUCCESS&&ob[q]>1){int ssm=0;ga(&ssm,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,T.prod[q]);
            for(unsigned dy=256;dy<=96u*1024u;dy+=256){if(ssm+b0+dy>48u*1024u&&fsa(T.prod[q],CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(b0+dy))!=CUDA_SUCCESS)break;
                int o=0;if(occ(&o,T.prod[q],int(BT[q]),b0+dy)!=CUDA_SUCCESS)break;if(o<ob[q]){if(o==ob[q]-1){lo_dyn[q]=dy;oblo[q]=o;}break;}}}
        std::printf("{\"diag\":\"ovl\",\"k\":\"regs\",\"st\":\"%s\",\"regs_local_smem_bpsm\":{%s},\"lo\":[%d,%u]}\n",SN[q],r.c_str(),oblo[q],lo_dyn[q]);std::fflush(stdout);}
    cmp=M.fn(m_rc,4);
    const double B=T.b?double(T.b):1.0;
    auto run=[&](const char* v,BatchOvr o)->double{
        if(el()>budget){std::printf("{\"diag\":\"ovl\",\"k\":\"batch\",\"v\":\"%s\",\"skipped\":\"time budget\"}\n",v);std::fflush(stdout);return -1;}
        const BatchRes r=line(o,v);return r.ok?r.med/B:-1;};
    for(int pass=0;pass<2;++pass){
        {BatchOvr o;char v[32];std::snprintf(v,sizeof v,"ctrl%s",pass?"_r":"");run(v,o);}
        for(int q=0;q<4;++q){
            double R=-1,Cc=-1,Mm=-1,Rl=-1,Rc=-1;char v[32];
            {BatchOvr o;o.sm=MASK[q];o.check=false;std::snprintf(v,sizeof v,"%s_R%s",SN[q],pass?"_r":"");R=run(v,o);}
            if(RC[q]){BatchOvr o;o.sm=MASK[q];o.check=false;o.fn[q]=RC[q];std::snprintf(v,sizeof v,"%s_Rrc%s",SN[q],pass?"_r":"");Rc=run(v,o);}
            if(CF[q]){BatchOvr o;o.sm=MASK[q];o.check=false;o.fn[q]=CF[q];std::snprintf(v,sizeof v,"%s_C%s",SN[q],pass?"_r":"");Cc=run(v,o);}
            if(MF[q]){BatchOvr o;o.sm=MASK[q];o.check=false;o.fn[q]=MF[q];std::snprintf(v,sizeof v,"%s_M%s",SN[q],pass?"_r":"");Mm=run(v,o);}
            if(lo_dyn[q]){BatchOvr o;o.sm=MASK[q];o.check=false;o.smem[q]=lo_dyn[q];std::snprintf(v,sizeof v,"%s_Rlo%s",SN[q],pass?"_r":"");Rl=run(v,o);}
            const double mn=std::min(Cc,Mm),mx=std::max(Cc,Mm);
            std::printf("{\"diag\":\"ovl\",\"k\":\"sum\",\"st\":\"%s\",\"pass\":%d,\"R\":%.2f,\"Rrc\":%.2f,\"C\":%.2f,\"M\":%.2f,\"Rlo\":%.2f,\"bpsm\":[%d,%d],\"e\":%.3f,\"pot\":%.2f}\n",
                SN[q],pass+1,R,Rc,Cc,Mm,Rl,ob[q],oblo[q],(R>0&&mn>0)?(Cc+Mm-R)/mn:0.0,(R>0&&mx>0)?R-mx:0.0);std::fflush(stdout);}
    }
    {BatchOvr o;run("ctrl_end",o);}
    std::printf("{\"diag\":\"ovl\",\"k\":\"done\",\"s\":%.2f}\n",el());std::fflush(stdout);
    d.cuCtxSynchronize();cmp=nullptr;
}
