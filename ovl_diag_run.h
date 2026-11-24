#pragma once
// Exit-time tensor / integer overlap diag (OVERLAP_20261124; FHERMA_OVL_DIAG=0: off). Never touches the production module,
// streams or buffers: JIT-loads its own small module (ovl_diag.h OVL_PTX: 3140af3 device helpers + the experiment kernels,
// NVRTC compute_90 PTX -> cuModuleLoadData, the production load path) and three experiment-C tl.py cubins, then reruns the
// best configurations of experiments A (two kernels, two streams, co-residency gate) and B (one fused kernel, warp roles)
// plus their alone references, timed with per-warp %globaltimer / %clock64 records (cycles = wall ns x SM GHz, best of 3).
// Work is per SMSP (independent of the SM count): T = 4 x 6720 m16n8k32 IMMA per SMSP (4 products' worth of base conversion),
// I = 192 virtual P2 blocks (production p2k bpo_r16_body, H path) per SM. Prints JSON lines with the local RTX 5070 values.
#include "ovl_diag.h"
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <chrono>
inline double ovl_local(const char* k){for(const OvlLocal& l:OVL_LOCAL)if(std::strcmp(l.key,k)==0)return l.v;return -1.0;}
template<class C> void run_ovl_diag(C& d){
    if(const char* e=std::getenv("FHERMA_OVL_DIAG"))if(std::atoi(e)==0)return;
    const auto t0=std::chrono::steady_clock::now();
    auto& dr=d.driver;
    auto devget=dr.template get<decltype(&::cuDeviceGet)>("cuDeviceGet");
    auto attr=dr.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
    auto dname=dr.template get<decltype(&::cuDeviceGetName)>("cuDeviceGetName");
    auto dver=dr.template get<decltype(&::cuDriverGetVersion)>("cuDriverGetVersion");
    auto fattr=dr.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
    auto fset=dr.template get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");
    auto occf=dr.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    auto mload=dr.template get<decltype(&::cuModuleLoadData)>("cuModuleLoadData");
    auto mget=dr.template get<decltype(&::cuModuleGetFunction)>("cuModuleGetFunction");
    auto munload=dr.template get<decltype(&::cuModuleUnload)>("cuModuleUnload");
    auto malloc_=dr.template get<decltype(&::cuMemAlloc_v2)>("cuMemAlloc_v2");
    auto mfree=dr.template get<decltype(&::cuMemFree_v2)>("cuMemFree_v2");
    auto mset=dr.template get<decltype(&::cuMemsetD8_v2)>("cuMemsetD8_v2");
    auto h2d=dr.template get<decltype(&::cuMemcpyHtoD_v2)>("cuMemcpyHtoD_v2");
    auto d2h=dr.template get<decltype(&::cuMemcpyDtoH_v2)>("cuMemcpyDtoH_v2");
    auto launch=dr.template get<decltype(&::cuLaunchKernel)>("cuLaunchKernel");
    auto sync=dr.template get<decltype(&::cuCtxSynchronize)>("cuCtxSynchronize");
    auto screate=dr.template get<decltype(&::cuStreamCreate)>("cuStreamCreate");
    auto sdestroy=dr.template get<decltype(&::cuStreamDestroy_v2)>("cuStreamDestroy_v2");
    CUdevice dev=0;if(devget(&dev,0)!=CUDA_SUCCESS)return;
    int mj=0,mn=0,sms=0,clk=0,drv=0,optin=0;char name[128]={0};
    attr(&mj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);attr(&mn,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
    attr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);attr(&clk,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev);
    attr(&optin,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,dev);dname(name,127,dev);dver(&drv);
    for(char* c=name;*c;++c)if(*c=='"'||*c=='\\')*c='_';
    if(sync()!=CUDA_SUCCESS)return;
    CUmodule mod=nullptr;CUresult lr=(mj==12&&mn==0)?mload(&mod,OVL_PTX):CUDA_ERROR_NOT_SUPPORTED;
    CUfunction fs=nullptr,fi=nullptr,ft=nullptr,fb=nullptr;int rg[3]={0,0,0};
    if(lr==CUDA_SUCCESS&&(mget(&fs,mod,"ovl_spin")||mget(&fi,mod,"ovl_i")||mget(&ft,mod,"ovl_t")||mget(&fb,mod,"ovl_b")))lr=CUDA_ERROR_NOT_FOUND;
    if(lr==CUDA_SUCCESS){fattr(&rg[0],CU_FUNC_ATTRIBUTE_NUM_REGS,fi);fattr(&rg[1],CU_FUNC_ATTRIBUTE_NUM_REGS,ft);fattr(&rg[2],CU_FUNC_ATTRIBUTE_NUM_REGS,fb);}
    std::printf("{\"diag\":\"ovl\",\"k\":\"info\",\"dev\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"clock_khz\":%d,\"driver\":%d,\"load\":%d,\"regs_i_t_b\":[%d,%d,%d],\"local_regs\":[56,63,64]}\n",
        name,mj,mn,sms,clk,drv,int(lr),rg[0],rg[1],rg[2]);std::fflush(stdout);
    if(lr!=CUDA_SUCCESS){if(mod)munload(mod);return;}
    fset(fb,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,optin);
    // buffers (all L2-resident: 3 x 4 MB P2 operands, 2 MB conversion data, 2 MB outputs)
    const unsigned N=65536u,G=2u;const unsigned P0=1070727169u,P1=1065484289u;const unsigned mu=unsigned((1ull<<48)/P0);
    std::vector<CUdeviceptr> bufs;bool okb=true;
    auto al=[&](size_t nb,unsigned bits)->CUdeviceptr{CUdeviceptr p=0;if(malloc_(&p,nb)!=CUDA_SUCCESS){okb=false;return 0;}bufs.push_back(p);
        if(bits){std::vector<unsigned> h(nb/4);unsigned long long x=0x9e3779b97f4a7c15ull^nb;
            for(auto& v:h){x=x*6364136223846793005ull+1442695040888963407ull;v=unsigned(x>>32)&(bits>=32?0xffffffffu:((1u<<bits)-1u));}h2d(p,h.data(),nb);}
        else mset(p,0,nb);return p;};
    CUdeviceptr A=al(size_t(G)*8*N*4,29),B=al(size_t(G)*8*N*4,29),Cb=al(size_t(G)*8*N*4,0),pr=al(64,0),r16=al(size_t(G)*512*8,30);
    CUdeviceptr D=al((0x4000u*32+28*32)*4,32),Bc=al(8*224*8,32),O=al((0x4000u*32+64*30)*4,0);
    const size_t ntm=size_t(sms)*32*5;CUdeviceptr TM0=al(ntm*8,0),TM1=al(ntm*8,0),GO=al(64,0);
    CUstream s0=nullptr,s1=nullptr;
    if(okb){unsigned hp[2]={P0,P1};h2d(pr,hp,8);}
    if(!okb||screate(&s0,CU_STREAM_NON_BLOCKING)!=CUDA_SUCCESS||screate(&s1,CU_STREAM_NON_BLOCKING)!=CUDA_SUCCESS){
        std::printf("{\"diag\":\"ovl\",\"error\":\"alloc\"}\n");std::fflush(stdout);
        for(auto p:bufs)mfree(p);if(s0)sdestroy(s0);if(s1)sdestroy(s1);munload(mod);return;}
    const unsigned X=4,RPS=16*X,NVS=48*X,nvb=unsigned(sms)*NVS;const double T16=double(RPS)*420.0*16.0,NIMMA=double(RPS)*420.0;
    bool bad=false;
    auto L=[&](CUfunction f,unsigned gx,unsigned bx,unsigned sm,CUstream s,void** a){if(!bad&&launch(f,gx,1,1,bx,1,1,sm,s,a,nullptr)!=CUDA_SUCCESS)bad=true;};
    auto warm=[&](){unsigned long long ns=20000000ull;void* a[]={&O,&ns};L(fs,unsigned(sms)*4,256,0,s0,a);if(sync()!=CUDA_SUCCESS)bad=true;};
    struct Sum{double cyc=-1,s1=-1,s2=-1,ghz=0;unsigned to=0;};
    auto rd=[&](CUdeviceptr tm,unsigned nw,std::vector<unsigned long long>& v){size_t o=v.size();v.resize(o+size_t(nw)*5);d2h(v.data()+o,tm,size_t(nw)*5*8);};
    auto summ=[&](const std::vector<unsigned long long>& v)->Sum{Sum r;unsigned long long a0=~0ull,a1=0,m0[3]={~0ull,~0ull,~0ull},m1[3]={0,0,0};std::vector<double> gz;
        for(size_t i=0;i+4<v.size();i+=5){if(v[i+2]<=v[i+1])continue;const unsigned role=unsigned(v[i]>>16)&3u;
            a0=std::min(a0,v[i+1]);a1=std::max(a1,v[i+2]);m0[role]=std::min(m0[role],v[i+1]);m1[role]=std::max(m1[role],v[i+2]);gz.push_back(double(v[i+4]-v[i+3])/double(v[i+2]-v[i+1]));}
        if(gz.empty())return r;std::sort(gz.begin(),gz.end());r.ghz=gz[gz.size()/2];r.cyc=double(a1-a0)*r.ghz;
        if(m1[1]>m0[1])r.s1=double(m1[1]-m0[1])*r.ghz;if(m1[2]>m0[2])r.s2=double(m1[2]-m0[2])*r.ghz;return r;};
    // A: mode 1 = I only, 2 = T only, 3 = both (I launched first; both kernels wait at a device gate until all blocks are resident)
    auto runA=[&](unsigned wI,unsigned wT,unsigned mode)->Sum{
        mset(TM0,0,ntm*8);mset(TM1,0,ntm*8);mset(GO,0,64);warm();
        unsigned ex=((mode&1)?unsigned(sms)*wI:0u)+((mode&2)?unsigned(sms):0u),G_=G,nv=nvb,nrb=wT?RPS/wT:0u,pp=P0,mm=mu;
        void* ai[]={&A,&B,&Cb,&pr,&r16,&nv,&G_,&TM0,&GO,&ex};void* at[]={&D,&Bc,&O,&nrb,&pp,&mm,&TM1,&GO,&ex};
        if(mode&1)L(fi,unsigned(sms)*wI,128,0,s0,ai);
        if(mode&2)L(ft,unsigned(sms),128*wT,0,s1,at);
        if(sync()!=CUDA_SUCCESS)bad=true;
        std::vector<unsigned long long> v;if(mode&1)rd(TM0,unsigned(sms)*wI*4,v);if(mode&2)rd(TM1,unsigned(sms)*wT*4,v);
        Sum r=summ(v);unsigned g[2]={0,0};d2h(g,GO,8);r.to=g[1];return r;};
    auto runB=[&](unsigned nIg,unsigned wT,bool doI,bool doT)->Sum{
        mset(TM0,0,ntm*8);warm();
        unsigned G_=G,nv=doI?nvb:0u,nrb=doT?RPS/wT:0u,pp=P0,mm=mu,ng=nIg,tf=0;
        void* ab[]={&A,&B,&Cb,&pr,&r16,&nv,&G_,&D,&Bc,&O,&nrb,&pp,&mm,&ng,&TM0,&tf};
        L(fb,unsigned(sms),128*(nIg+wT),14336u+9344u*nIg,s0,ab);if(sync()!=CUDA_SUCCESS)bad=true;
        std::vector<unsigned long long> v;rd(TM0,unsigned(sms)*(nIg+wT)*4,v);return summ(v);};
    auto best=[&](auto f)->Sum{Sum b;for(int t=0;t<3&&!bad;++t){Sum r=f();if(r.cyc>0&&(b.cyc<0||r.cyc<b.cyc))b=r;}return b;};
    auto ln=[&](const char* k,const char* fmt,auto... a){if(bad)return;std::printf("{\"diag\":\"ovl\",\"k\":\"%s\",",k);std::printf(fmt,a...);std::printf("}\n");std::fflush(stdout);};
    // references
    const Sum t4=best([&]{return runA(0,4,2);}),i4=best([&]{return runA(4,0,1);}),i8=best([&]{return runA(8,0,1);});
    ln("ref","\"name\":\"T_wT4\",\"cyc\":%.0f,\"cyc_per_imma\":%.3f,\"ghz\":%.3f,\"local5070\":%.0f",t4.cyc,t4.cyc/NIMMA,t4.ghz,ovl_local("T_wT4"));
    ln("ref","\"name\":\"I_wI4\",\"cyc\":%.0f,\"cyc_per_vblock\":%.1f,\"local5070\":%.0f",i4.cyc,i4.cyc/NVS,ovl_local("I_wI4"));
    ln("ref","\"name\":\"I_wI8\",\"cyc\":%.0f,\"cyc_per_vblock\":%.1f,\"I8_over_T16\":%.3f,\"local5070\":%.0f",i8.cyc,i8.cyc/NVS,i8.cyc/T16,ovl_local("I_wI8"));
    auto eff=[](double t,double i,double b){return (t+i-b)/std::min(t,i);};
    auto cfgline=[&](const char* k,const char* key,unsigned wI,unsigned wT,const Sum& ta,const Sum& ia,const Sum& bo){
        char kb[64],ke[64];std::snprintf(kb,64,"%s.both",key);std::snprintf(ke,64,"%s.eff",key);
        ln(k,"\"wI\":%u,\"wT\":%u,\"T_alone\":%.0f,\"I_alone\":%.0f,\"both\":%.0f,\"span_I\":%.0f,\"span_T\":%.0f,\"eff\":%.3f,\"eff_vs_best_alone\":%.3f,\"both_over_T16\":%.3f,"
           "\"cyc_per_imma_on_I8\":%.2f,\"gate_timeouts\":%u,\"local_both\":%.0f,\"local_eff\":%.3f",
           wI,wT,ta.cyc,ia.cyc,bo.cyc,bo.s1,bo.s2,eff(ta.cyc,ia.cyc,bo.cyc),(t4.cyc+i8.cyc-bo.cyc)/t4.cyc,bo.cyc/T16,(bo.cyc-i8.cyc)/NIMMA,bo.to,ovl_local(kb),ovl_local(ke));};
    {const Sum b=best([&]{return runA(4,4,3);});cfgline("A","A44",4,4,t4,i4,b);}
    for(unsigned wT:{4u,2u}){
        const Sum bt=best([&]{return runB(4,wT,false,true);}),bi=best([&]{return runB(4,wT,true,false);}),bb=best([&]{return runB(4,wT,true,true);});
        cfgline("B",wT==4?"B44":"B42",4,wT,bt,bi,bb);}
    // experiment C: one warp doing both (tl.py loop, 8 warps per SMSP, cycles per period per SMSP)
    {CUdeviceptr out=al(1u<<16,0),clkb=al(size_t(sms)*8*4*3*8,0);double cp[3]={-1,-1,-1};std::vector<unsigned long long> h(size_t(sms)*8*4*3);
     for(unsigned p=0;p<3&&okb&&!bad;++p){const OvlCPat& P=OVL_CPATS[p];CUmodule m2=nullptr;if(mload(&m2,P.cubin)!=CUDA_SUCCESS){ln("C","\"p\":\"%s\",\"error\":\"load\"",P.name);continue;}
        CUfunction f=nullptr;if(mget(&f,m2,"tl_t")!=CUDA_SUCCESS){munload(m2);continue;}
        int occ=0;occf(&occ,f,128,0);const unsigned bps=std::min<unsigned>(8u,unsigned(occ>0?occ:1)),nb=unsigned(sms)*bps;unsigned rep=20,k7=7;void* a[]={&out,&rep,&clkb,&k7};double bst=0;
        for(int t=0;t<3;++t){if(launch(f,nb,1,1,128,1,1,0,nullptr,a,nullptr)!=CUDA_SUCCESS||sync()!=CUDA_SUCCESS){bad=true;break;}
            d2h(h.data(),clkb,size_t(nb)*4*3*8);std::map<unsigned long long,std::pair<unsigned long long,std::pair<unsigned long long,unsigned>>> per;
            for(unsigned g=0;g<nb*4;++g){auto it=per.find(h[3*g]);if(it==per.end())per[h[3*g]]={h[3*g+1],{h[3*g+2],1u}};else{it->second.first=std::min(it->second.first,h[3*g+1]);it->second.second.first=std::max(it->second.second.first,h[3*g+2]);++it->second.second.second;}}
            std::vector<double> v;for(auto& kv:per){const auto s_=kv.second.first,e_=kv.second.second.first;if(e_>s_)v.push_back(double(kv.second.second.second)*rep*P.nloop/double(e_-s_)/4.0);}
            if(!v.empty()){std::sort(v.begin(),v.end());bst=std::max(bst,v[v.size()/2]);}}
        munload(m2);cp[p]=bst>0?double(P.per)/bst:-1;char kk[64];std::snprintf(kk,64,"%s",P.name);
        ln("C","\"p\":\"%s\",\"w\":%u,\"cyc_per_period\":%.3f,\"local5070\":%.3f",P.name,bps,cp[p],ovl_local(kk));}
     if(cp[0]>0&&cp[1]>0&&cp[2]>0)ln("C","\"p\":\"eff\",\"eff\":%.3f,\"imma_cost_on_int\":%.2f,\"local_eff\":%.3f,\"local_imma_cost_on_int\":%.2f",
        eff(cp[0],cp[1],cp[2]),cp[2]-cp[1],ovl_local("c_eff"),ovl_local("c_cost"));}
    if(bad){std::printf("{\"diag\":\"ovl\",\"error\":\"launch\"}\n");std::fflush(stdout);}
    sync();for(auto p:bufs)mfree(p);sdestroy(s0);sdestroy(s1);munload(mod);
    std::printf("{\"diag\":\"ovl\",\"k\":\"done\",\"s\":%.2f}\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());std::fflush(stdout);
}
