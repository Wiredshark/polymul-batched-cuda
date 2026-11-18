#pragma once
// Exit-time on-device experiments (after all timed calls, from ~State after the kprof report): replays kernels captured
// by kprof (launch parameters copied at capture time) on two fresh streams. The variant sweep itself is diag_sweep.h;
// this file holds the replay / flush / stream helpers and the entry point run_diag. Output: compact {"diag":...} JSON
// lines on stdout. FHERMA_DIAG=0 disables everything (nothing is compiled or allocated). Clobbers the workspaces only,
// after the last timed call; the production module, graphs and timed path are untouched.
#include "kprof.h"
#include <nvrtc.h>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

struct DiagTab { CUdeviceptr p; size_t bytes; };   // a device table the production kernels take by pointer
inline bool diag_env(){const char* e=std::getenv("FHERMA_DIAG");return !e||*e!='0';}

template<class C> struct Diag {
    C& d; KProf& kp; CUstream s[2]{};
    decltype(&::cuStreamWaitEvent) swait; decltype(&::cuEventSynchronize) esync; decltype(&::cuStreamCreate) screate;
    decltype(&::cuStreamDestroy) sdestroy;
    CUdeviceptr flush=0;size_t flush_n=0;int sms=1;CUfunction bwf=nullptr;   // bwf: diag_bw from the sweep's helper module
    explicit Diag(C& c):d(c),kp(c.kp){
        swait=d.driver.template get<decltype(swait)>("cuStreamWaitEvent");esync=d.driver.template get<decltype(esync)>("cuEventSynchronize");
        screate=d.driver.template get<decltype(screate)>("cuStreamCreate");sdestroy=d.driver.template get<decltype(sdestroy)>("cuStreamDestroy_v2");
        for(auto& x:s)d.check(screate(&x,CU_STREAM_NON_BLOCKING));
        CUdevice dev=0;int l2=0;if(kp.dattr){kp.dattr(&l2,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,dev);kp.dattr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);}
        flush_n=size_t(std::max(l2,1<<20))*2/16;d.check(d.cuMemAlloc(&flush,flush_n*16));
    }
    ~Diag(){d.cuCtxSynchronize();for(auto x:s)if(x)sdestroy(x);if(flush)d.cuMemFree(flush);}
    void bw(CUstream st,CUdeviceptr src,CUdeviceptr dst,unsigned long long nr,unsigned long long nw){
        unsigned seed=0;void* pp[]={&src,&dst,&nr,&nw,&seed};d.check(d.cuLaunchKernel(bwf,unsigned(sms)*4,1,1,256,1,1,0,st,pp,nullptr));}
    void do_flush(CUstream st){bw(st,0,flush,0,flush_n);}   // dirty flush: 2 x L2 of writes
    void replay(const KProf::Trace& t,CUstream st){
        std::vector<unsigned char> b=t.pb;void* pp[64];
        for(size_t i=0;i<t.po.size();++i)pp[i]=b.data()+t.po[i];
        d.check(d.cuLaunchKernel(t.fn,t.gx,t.gy,t.gz,t.bx,1,1,t.smem,st,pp,nullptr));}
};
#include "diag_sweep.h"
// src: the production NVRTC source (prefix + kernels); prod_tabs: the production device tables (DiagTab order of
// State::tables()); h30: builds the mixed-prime (h30) prefix + kernels and its tables in the same order (false: not available).
template<class C,class H> inline void run_diag(C& c,const std::string& src,unsigned l,const std::vector<DiagTab>& prod_tabs,H&& h30){
    if(!diag_env()||!c.kp.enabled||!c.kp.pinfo)return;
    std::string hsrc;std::vector<DiagTab> htabs;
    if(sweep_env()&&sweep_h30_env()){
        const auto th=std::chrono::steady_clock::now();
        try{const bool ok=h30(hsrc,htabs);std::printf("{\"diag\":\"sw_h30\",\"tables_s\":%.2f,\"ok\":%d}\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-th).count(),int(ok));
            if(!ok){std::printf("{\"diag\":\"sw\",\"k\":\"h30\",\"error\":\"mixed prime set does not fit the batched layout\"}\n");hsrc.clear();}}
        catch(const std::exception& e){std::printf("{\"diag\":\"sw\",\"k\":\"h30\",\"error\":\"%.100s\"}\n",e.what());hsrc.clear();}
        if(htabs.size()!=prod_tabs.size())hsrc.clear();std::fflush(stdout);}
    std::unique_ptr<Sweep<C>> sw;
    try{if(sweep_env())sw=std::make_unique<Sweep<C>>(c,src,hsrc,l,prod_tabs,htabs);}catch(const std::exception& e){std::printf("{\"diag\":\"sw_error\",\"what\":\"%.100s\"}\n",e.what());}
    try{if(sw){Diag<C> dg(c);sw->run(dg);}}catch(const std::exception& e){std::printf("{\"diag\":\"error\",\"what\":\"%.100s\"}\n",e.what());std::fflush(stdout);}
    sw.reset();
}
