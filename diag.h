#pragma once
// Exit-time on-device micro-experiments (after all timed calls, from ~State after the kprof report): replays kernels
// captured by kprof (launch parameters copied at capture) on two fresh streams, plus ablation variants of bpo_first_x
// (BPO_DIAG kernels) and bandwidth probes; prints compact {"diag":...} JSON lines on stdout. Each line: median and
// min of R timed reps (cudaEvent), after 2 warm-up reps. "cold" = a 2xL2 write flush before every rep.
// FHERMA_DIAG=0 disables (also drops the BPO_DIAG kernels from the module). Clobbers the workspaces only.
#include "kprof.h"
#include "diag_mb.h"
#include <nvrtc.h>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

inline bool diag_env(){const char* e=std::getenv("FHERMA_DIAG");return !e||*e!='0';}
// round 15 (bceil): whole-batch substitution set for State::batch_line. Per stage (0 P1, 1 P2, 2 P3, 3 CRT): fn = kernel
// substitute (null = production), gx = grid.x substitute (0 = production), smem0 = launch with no dynamic smem (trivial
// kernel); por = OR-ed into the P1 pfstep argument; check = exactness vs the production graph (controls only; else ex 0).
// round 16 (p1b): bx = block.x substitute (0 = production); xw = P1 output run length L for the P1 TMA map (0 = production map).
// round 18 (combo, on 88de7fc: h30 prime set is production, the round-17 h30 table swap is gone): env = environment overrides
// set for the capture only (empty value = unset), crtpf = CRT prefetch items (-1 = production min(p1_grid/2, 2 cols)).
struct BatchOvr { CUfunction fn[4]{}; unsigned gx[4]{}; bool smem0[4]{}; unsigned por=0; bool check=false; unsigned bx[4]{}; unsigned xw=0; std::vector<std::pair<std::string,std::string>> env; int crtpf=-1; unsigned p3tab=0; };   // round 20: p3tab 1 = P3 substitute gets the r16 table
struct BatchRes { bool ok=false; int ex=-1; double vs=0,med=0,prod=0; };   // State::batch_line result (vs = median pair ratio)

template<class C> struct Diag {
    C& d; KProf& kp; CUstream s[2]{}; CUevent ev0=nullptr,ev1=nullptr,evj=nullptr;
    decltype(&::cuStreamWaitEvent) swait; decltype(&::cuEventSynchronize) esync; decltype(&::cuStreamCreate) screate;
    decltype(&::cuStreamDestroy) sdestroy;
    CUdeviceptr flush=0,bsrc=0,bdst=0;size_t flush_n=0;int sms=1;int lines=0;double t_start=0;
    std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now();
    explicit Diag(C& c):d(c),kp(c.kp){
        swait=d.driver.template get<decltype(swait)>("cuStreamWaitEvent");esync=d.driver.template get<decltype(esync)>("cuEventSynchronize");
        screate=d.driver.template get<decltype(screate)>("cuStreamCreate");sdestroy=d.driver.template get<decltype(sdestroy)>("cuStreamDestroy_v2");
        for(auto& x:s)d.check(screate(&x,CU_STREAM_NON_BLOCKING));
        kp.ecreate(&ev0,CU_EVENT_DEFAULT);kp.ecreate(&ev1,CU_EVENT_DEFAULT);kp.ecreate(&evj,CU_EVENT_DEFAULT);
        CUdevice dev=0;int l2=0;if(kp.dattr){kp.dattr(&l2,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,dev);kp.dattr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);}
        flush_n=size_t(std::max(l2,1<<20))*2/16;d.check(d.cuMemAlloc(&flush,flush_n*16));
    }
    ~Diag(){d.cuCtxSynchronize();for(auto x:s)if(x)sdestroy(x);for(auto e:{ev0,ev1,evj})if(e)kp.edestroy(e);
        for(auto p:{flush,bsrc,bdst})if(p)d.cuMemFree(p);}
    void bw(CUstream st,CUdeviceptr src,CUdeviceptr dst,unsigned long long nr,unsigned long long nw){
        unsigned seed=0;void* pp[]={&src,&dst,&nr,&nw,&seed};d.check(d.cuLaunchKernel(d.diag_bw,unsigned(sms)*4,1,1,256,1,1,0,st,pp,nullptr));}
    void do_flush(CUstream st){bw(st,0,flush,0,flush_n);}
    // replay one captured kernel (fn override, pfstep override for bpo_first_x-signature kernels: param 11)
    void replay(const KProf::Trace& t,CUstream st,CUfunction fn=nullptr,long long pf=-1){
        std::vector<unsigned char> b=t.pb;void* pp[64];
        if(pf>=0&&t.po.size()>11)std::memcpy(b.data()+t.po[11],&pf,4);
        for(size_t i=0;i<t.po.size();++i)pp[i]=b.data()+t.po[i];
        d.check(d.cuLaunchKernel(fn?fn:t.fn,t.gx,t.gy,t.gz,t.bx,1,1,t.smem,st,pp,nullptr));}
    // time body(s0,s1) (both streams start together, joined): median/min over R reps
    void time(const char* name,const std::function<void(CUstream,CUstream)>& body,bool cold,const std::function<void(CUstream)>& pre={},double bytes=0,int R=10){
        if(lines>=36)return;
        if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>12.0){std::printf("{\"diag\":\"%s\",\"skipped\":\"time budget\"}\n",name);++lines;return;}
        std::vector<float> v;
        for(int r=0;r<R+2;++r){
            if(pre)pre(s[0]);if(cold)do_flush(s[0]);
            kp.erecord(ev0,s[0]);swait(s[1],ev0,0);body(s[0],s[1]);kp.erecord(evj,s[1]);swait(s[0],evj,0);kp.erecord(ev1,s[0]);
            d.check(esync(ev1));if(r>=2)v.push_back(kp.ms(ev0,ev1));}
        std::sort(v.begin(),v.end());const double med=v[v.size()/2],mn=v.front();
        if(bytes>0)std::printf("{\"diag\":\"%s\",\"med_ms\":%.4f,\"min_ms\":%.4f,\"GBps_med\":%.1f}\n",name,med,mn,bytes/med*1e-6);
        else std::printf("{\"diag\":\"%s\",\"med_ms\":%.4f,\"min_ms\":%.4f}\n",name,med,mn);
        ++lines;std::fflush(stdout);}
    // Instruction / butterfly micro-benchmarks (diag_mb.h): one block of 1024 threads per SM, cycles from clock64 (max over
    // SMs), SM clock from cycles/event-time. Prefers a native cubin for the device cc, falls back to compute_80 PTX.
    // corr=true: only the round-8 correction kernels cr*/ci* (default on): corr_per_smsp_clk = corrections per clock per SMSP,
    // ipc_smsp = warp instructions per clock per SMSP implied by the SASS instruction count per correction (ins, +1 IMAD for ci*)
    void run_mb(bool corr=false){
        if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>6.0){std::printf("{\"diag\":\"mb\",\"skipped\":\"time budget\"}\n");return;}
        CUmodule mod=nullptr;CUdeviceptr o=0,c=0;
        try{
            int cc1=0,cc2=0;if(kp.dattr){kp.dattr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,0);kp.dattr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,0);}
            nvrtcProgram pr=nullptr;if(d.nvrtcCreateProgram(&pr,kDiagMbSource,"diag_mb.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("mb nvrtc create");
            auto gsz=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,size_t*)>("nvrtcGetCUBINSize");auto gcub=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,char*)>("nvrtcGetCUBIN");
            std::vector<char> img;bool native=false;
            {std::string a="--gpu-architecture=sm_"+std::to_string(cc1*10+cc2);const char* op[]={a.c_str(),"--std=c++17"};
             if(cc1>=8&&d.nvrtcCompileProgram(pr,2,op)==NVRTC_SUCCESS){size_t n=0;if(gsz(pr,&n)==NVRTC_SUCCESS&&n){img.resize(n);if(gcub(pr,img.data())==NVRTC_SUCCESS)native=true;else img.clear();}}}
            if(!native){d.nvrtcDestroyProgram(&pr);pr=nullptr;d.nvrtcCreateProgram(&pr,kDiagMbSource,"diag_mb.cu",0,nullptr,nullptr);
                const char* op[]={"--gpu-architecture=compute_80","--std=c++17"};
                if(d.nvrtcCompileProgram(pr,2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);throw std::runtime_error("mb nvrtc: "+l);}
                size_t n=0;d.nvrtcGetPTXSize(pr,&n);img.resize(n);d.nvrtcGetPTX(pr,img.data());}
            d.nvrtcDestroyProgram(&pr);
            d.check(d.cuModuleLoadData(&mod,img.data()));
            d.check(d.cuMemAlloc(&o,size_t(sms)*1024*4));d.check(d.cuMemAlloc(&c,size_t(sms)*8));
            std::printf("{\"diag\":\"mb_info\",\"sms\":%d,\"cc\":\"%d.%d\",\"native_cubin\":%s,\"threads\":1024}\n",sms,cc1,cc2,native?"true":"false");
            struct K{const char* fn;const char* name;const char* key;double ops;int kind;};   // kind 0: thread-ops, 1: int8 mma, 2: butterflies
            static const K ks[]={{"mb0","mb_imad","per_sm_clk",8,0},{"mb1","mb_imad_hi","per_sm_clk",8,0},{"mb2","mb_iadd3","per_sm_clk",8,0},{"mb3","mb_shoup_mulmod","per_sm_clk",8,0},
                {"mb4","mb_ffma","per_sm_clk",4,0},{"mb5","mb_imma_s8_m16n8k32","int8_mac_per_sm_clk",0,1},{"mb6","mb_min_add_sub","per_sm_clk",8,0},{"mb7","mb_imad_wide_xor","per_sm_clk",4,0},
                {"mb8","mb_imad4_lop3_4","per_sm_clk",8,0},{"mb9","mb_lop3","per_sm_clk",8,0},{"mb10","mb_add_asm","per_sm_clk",8,0},{"mb11","mb_min_asm","per_sm_clk",8,0},
                {"mb12","mb_ffma_asm","per_sm_clk",8,0},{"mb13","mb_ffma4_lop3_4","per_sm_clk",8,0},{"mb14","mb_viaddmnmx","per_sm_clk",8,0},{"mb15","mb_add_then_min","per_sm_clk",8,0},
                {"mb16","mb_xadd","per_sm_clk",8,0},{"mb17","mb_imad4_min4","per_sm_clk",8,0},{"mb18","mb_imad4_ffma4","per_sm_clk",8,0},{"mb19","mb_add_imm","per_sm_clk",8,0},
                {"mb20","mb_shl_imm","per_sm_clk",8,0},{"mb21","mb_setp_selp","per_sm_clk",8,0},{"mb22","mb_prmt","per_sm_clk",8,0},
                {"bf0","bf_current","bfly_per_sm_clk",8,2},{"bf1","bf_harvey","bfly_per_sm_clk",8,2},
                {"cr0","corr_viaddmnmx","corr_per_smsp_clk",8,3},{"cr1","corr_iadd3_imnmx","corr_per_smsp_clk",8,3},{"cr2","corr_isetp_iadd_sel","corr_per_smsp_clk",8,3},{"cr3","corr_isetp_piadd","corr_per_smsp_clk",8,3},
                {"cr4","corr_lop3_mask","corr_per_smsp_clk",8,3},{"cr5","corr_imad_imnmx","corr_per_smsp_clk",8,3},{"cr6","corr_isetp_pimad","corr_per_smsp_clk",8,3},
                {"ci0","corr_viaddmnmx+imad","corr_per_smsp_clk",8,4},{"ci1","corr_iadd3_imnmx+imad","corr_per_smsp_clk",8,4},{"ci2","corr_isetp_iadd_sel+imad","corr_per_smsp_clk",8,4},{"ci3","corr_isetp_piadd+imad","corr_per_smsp_clk",8,4},
                {"ci4","corr_lop3_mask+imad","corr_per_smsp_clk",8,4},{"ci5","corr_imad_imnmx+imad","corr_per_smsp_clk",8,4},{"ci6","corr_isetp_pimad+imad","corr_per_smsp_clk",8,4}};
            static const int cins[7]={1,3,2,2,4,2,2};   // SASS instructions per correction (sm_90 SASS dump; no sm_120 disassembler here, see diag_mb.h)
            for(const auto& k:ks){
                if((k.kind>=3)!=corr)continue;
                if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>9.0){std::printf("{\"diag\":\"%s\",\"skipped\":\"time budget\"}\n",k.name);continue;}
                CUfunction fn=nullptr;if(d.cuModuleGetFunction(&fn,mod,k.fn)!=CUDA_SUCCESS){std::printf("{\"diag\":\"%s\",\"skipped\":\"no kernel\"}\n",k.name);continue;}
                const double iters=k.kind==2?2048.0:4096.0;unsigned long long best=~0ull;double mhz=0;
                for(int r=0;r<6;++r){
                    unsigned seed=3+r;void* pp[]={&o,&c,&seed};
                    kp.erecord(ev0,s[0]);d.check(d.cuLaunchKernel(fn,unsigned(sms),1,1,1024,1,1,0,s[0],pp,nullptr));kp.erecord(ev1,s[0]);d.check(esync(ev1));
                    std::vector<unsigned long long> h(sms);d.check(d.cuMemcpyDtoH(h.data(),c,size_t(sms)*8));unsigned long long mx=0;for(auto v:h)mx=std::max(mx,v);
                    if(r==0||mx==0)continue;best=std::min(best,mx);const double ms=kp.ms(ev0,ev1);if(ms>0)mhz=std::max(mhz,double(mx)/ms*1e-3);}
                if(best==~0ull){std::printf("{\"diag\":\"%s\",\"skipped\":\"no cycles\"}\n",k.name);continue;}
                const double v=k.kind==1?1024.0/32*4096*4*(16*8*32)/double(best):1024.0*iters*k.ops/double(best);
                if(k.kind>=3){const double cs=v/4;const int ins=cins[k.fn[2]-'0']+(k.kind==4?1:0);
                    std::printf("{\"diag\":\"%s\",\"%s\":%.3f,\"ins\":%d,\"ipc_smsp\":%.3f,\"clk_mhz\":%.0f}\n",k.name,k.key,cs,ins,cs*ins/32,mhz);std::fflush(stdout);continue;}
                std::printf("{\"diag\":\"%s\",\"%s\":%.2f,\"clk_mhz\":%.0f}\n",k.name,k.key,v,mhz);std::fflush(stdout);}
        }catch(const std::exception& e){std::printf("{\"diag\":\"mb_error\",\"what\":\"%.100s\"}\n",e.what());std::fflush(stdout);}
        d.cuCtxSynchronize();if(c)d.cuMemFree(c);if(o)d.cuMemFree(o);if(mod)d.cuModuleUnload(mod);
    }
    void run(size_t in_bytes,size_t out_bytes){
        if(kp.last.empty()||!d.diag_bw)return;
        const auto& tr=kp.traces[kp.last.back().variant];
        auto find=[&](int prod,int stage,int gidx)->const KProf::Trace*{for(const auto& t:tr)if(t.product==prod&&t.stage==stage&&t.gidx==gidx&&!t.pb.empty())return &t;return nullptr;};
        const KProf::Trace* p0=find(0,KProf::P1,0);const KProf::Trace* p1=find(1,KProf::P1,0);const KProf::Trace* p0g1=find(0,KProf::P1,1);
        if(!p0||p0->fn!=d.bpo_first_x||p0->po.size()!=15){std::printf("{\"diag\":\"off\",\"why\":\"no bpo_first_x trace with params\"}\n");return;}
        unsigned pf0=0;std::memcpy(&pf0,p0->pb.data()+p0->po[11],4);unsigned r0=0,r1=0;std::memcpy(&r0,p0->pb.data()+p0->po[9],4);std::memcpy(&r1,p0->pb.data()+p0->po[10],4);
        int nprod=0,ng=0;for(const auto& t:tr){nprod=std::max(nprod,t.product+1);if(t.product==0&&t.stage==KProf::P1)++ng;}
        in_bytes*=size_t(std::max(ng,1));   // P1 reads the inputs once per prime group
        std::printf("{\"diag\":\"info\",\"variant\":%d,\"traces\":%zu,\"products\":%d,\"p1_grid\":%u,\"p1_block\":%u,\"p1_r0\":%u,\"p1_r1\":%u,\"pfstep\":\"0x%08x\",\"p1_chain_of_prod1\":%d,\"p1_groups\":%d,\"p1_in_MB\":%.2f,\"out_MB\":%.2f,\"flush_MB\":%.1f,\"note\":\"replays lack graph L2 access-policy windows\"}\n",
            kp.last.back().variant,tr.size(),nprod,p0->gx,p0->bx,r0,r1,pf0,p1?p1->chain:-1,ng,in_bytes/1e6,out_bytes/1e6,flush_n*16/1e6);++lines;
        // 1. production P1 group 0 of product 0 (cold / warm), alone and with product 1's group 0 concurrently
        auto p1run=[&](CUfunction fn,long long pf,bool two){return [=,this](CUstream a,CUstream b){replay(*p0,a,fn,pf);if(two&&p1)replay(*p1,b,fn,pf);};};
        time("p1_g0_c1_cold",p1run(nullptr,-1,false),true);
        time("p1_g0_c1_warm",p1run(nullptr,-1,false),false);
        if(p1)time("p1_g0_c2_cold",p1run(nullptr,-1,true),true);
        if(p0g1)time("p1_g1_c1_after_g0",[&](CUstream a,CUstream){replay(*p0g1,a);},false,[&](CUstream a){do_flush(a);replay(*p0,a);});
        // 2/3. ablations (same params, different kernel) and TMA store / input-load policy (pfstep bits 29-30 / 28)
        struct V{const char* n;CUfunction f;long long pf;bool c2;};
        const unsigned base=pf0&0x0fffffffu,ld=pf0&(1u<<28);
        const V vs[]={{"nostore",d.bpo_xd_nostore,-1,true},{"l2in",d.bpo_xd_l2in,-1,true},{"compute",d.bpo_xd_compute,-1,true},{"st_global",d.bpo_xd_stg,-1,true},{"st_global_cs",d.bpo_xd_stcs,-1,false},
            {"tma_nohint",nullptr,base|ld|(1u<<29),true},{"tma_evict_first",nullptr,base|ld,true},{"tma_evict_last",nullptr,base|ld|(2u<<29),true},{"ld_flip",nullptr,(pf0^(1u<<28)),false}};
        char nm[96];
        const bool full=std::getenv("FHERMA_DIAG_FULL")&&*std::getenv("FHERMA_DIAG_FULL")=='1';   // TMA hints / ld_flip / st_global_cs: measured in pv25 (no effect), off by default
        for(const auto& x:vs){if(!x.f&&x.pf<0)continue;   // ablation kernel missing
            if(!full&&(!x.f||x.f==d.bpo_xd_stcs))continue;
            std::snprintf(nm,sizeof nm,"p1_%s_c1_cold",x.n);time(nm,p1run(x.f,x.pf,false),true);
            if(p1&&x.c2){std::snprintf(nm,sizeof nm,"p1_%s_c2_cold",x.n);time(nm,p1run(x.f,x.pf,true),true);}}
        // exactness of the per-lane store path vs TMA (A,B planes of group 0)
        if(d.bpo_xd_stg){CUdeviceptr A=0,B=0;std::memcpy(&A,p0->pb.data()+p0->po[2],8);std::memcpy(&B,p0->pb.data()+p0->po[3],8);
            const size_t off=size_t(r0)*8*65536*4,sz=size_t(r1-r0)*8*65536*4;std::vector<unsigned char> h0(2*sz),h1(2*sz);
            auto get=[&](std::vector<unsigned char>& h){d.check(d.cuCtxSynchronize());d.check(d.cuMemcpyDtoH(h.data(),A+off,sz));d.check(d.cuMemcpyDtoH(h.data()+sz,B+off,sz));};
            replay(*p0,s[0]);get(h0);replay(*p0,s[0],d.bpo_xd_stg);get(h1);
            std::printf("{\"diag\":\"st_global_matches_tma\",\"ok\":%s}\n",h0==h1?"true":"false");++lines;}
        // 4. bandwidth probes at P1 byte volumes (measured in pv25/pv33: FHERMA_DIAG_FULL=1 only)
        if(full){const size_t big=std::max(in_bytes,out_bytes);d.check(d.cuMemAlloc(&bsrc,big));d.check(d.cuMemAlloc(&bdst,big));
         const unsigned long long nr=in_bytes/16,nw=out_bytes/16;
         time("bw_read_cold",[&](CUstream a,CUstream){bw(a,bsrc,bdst,nr,0);},true,{},double(in_bytes));
         time("bw_write_cold",[&](CUstream a,CUstream){bw(a,bsrc,bdst,0,nw);},true,{},double(out_bytes));
         time("bw_copy_cold",[&](CUstream a,CUstream){bw(a,bsrc,bdst,nr,nw);},true,{},double(in_bytes+out_bytes));
         time("bw_copy_c2_cold",[&](CUstream a,CUstream b){bw(a,bsrc,bdst,nr/2,nw/2);bw(b,bsrc+nr/2*16,bdst+nw/2*16,nr-nr/2,nw-nw/2);},true,{},double(in_bytes+out_bytes));}
        // 5. P2 / P3 / CRT (first of product 0, after its prelude replayed untimed), and the serial product-0 chain
        for(int st:{int(KProf::P2),int(KProf::P3),int(KProf::CRT)}){
            size_t idx=tr.size();for(size_t i=0;i<tr.size();++i)if(tr[i].product==0&&tr[i].stage==st&&!tr[i].pb.empty()){idx=i;break;}
            if(idx==tr.size())continue;
            auto pre=[&,idx](CUstream a){do_flush(a);for(size_t i=0;i<idx;++i)if(tr[i].product==0)replay(tr[i],a);};
            std::snprintf(nm,sizeof nm,"%s_first_c1",KProf::sname[st]);time(nm,[&,idx](CUstream a,CUstream){replay(tr[idx],a);},false,pre);}
        auto chain=[&](int prod,CUstream a){for(const auto& t:tr)if(t.product==prod&&!t.pb.empty())replay(t,a);};
        time("product0_serial_c1_cold",[&](CUstream a,CUstream){chain(0,a);},true,{},0,5);
        if(p1&&p1->chain!=p0->chain)time("products01_c2_cold",[&](CUstream a,CUstream b){chain(0,a);chain(1,b);},true,{},0,5);
        std::printf("{\"diag\":\"done\",\"s\":%.2f}\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());std::fflush(stdout);
    }
};
#include "diag_sweep.h"
#include "diag_p1q.h"
// src: the production NVRTC source (prefix + kernels) for the variant sweep (diag_sweep.h; FHERMA_SWEEP=0 off), l: limbs.
// The instruction micro-benchmarks (run_mb, measured on the target in pv27) only run with FHERMA_DIAG_MB=1.
// batch (round 14): whole-batch A/B callback for P1 substitutes (State::batch_line), optional.
// h30 (round 17, h30b): builds the mixed-prime table set + its NVRTC source (prefix + kernels); only called when the sweep runs
// (FHERMA_SWEEP_H30=0 skips it).
template<class C> inline void run_diag(C& c,size_t in_bytes,size_t out_bytes,const std::string& src,unsigned l,const std::function<void(const BatchOvr&,const char*)>& batch={},const std::function<bool(std::string&)>& h30={}){
    if(!diag_env()||(!c.kp.enabled&&!c.kp.trace_only)||!c.kp.pinfo||!c.diag_bw||c.kp.diag_variant()<0)return;
    std::string hsrc;
    if(sweep_env()&&h30&&!(std::getenv("FHERMA_SWEEP_H30")&&*std::getenv("FHERMA_SWEEP_H30")=='0')){const auto th=std::chrono::steady_clock::now();
        try{const bool ok=h30(hsrc);if(!ok)hsrc.clear();std::printf("{\"diag\":\"sw_h30\",\"tables_s\":%.2f,\"ok\":%d}\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-th).count(),int(ok));}
        catch(const std::exception& e){hsrc.clear();std::printf("{\"diag\":\"sw_h30\",\"error\":\"%.100s\"}\n",e.what());}std::fflush(stdout);}
    std::unique_ptr<Sweep<C>> sw;
    try{if(sweep_env())sw=std::make_unique<Sweep<C>>(c,src,l,hsrc);}catch(const std::exception& e){std::printf("{\"diag\":\"sw_error\",\"what\":\"%.100s\"}\n",e.what());}
    try{Diag<C> dg(c);if(p1q_env())run_p1q(dg);   // round 15: FHERMA_DIAG_P1Q=1 only (default off)
          // round 10 (cmp): P1 quad-math replica probe (diag_p1q.h), default on
        const char* mb=std::getenv("FHERMA_DIAG_MB");if(mb&&*mb=='1')dg.run_mb();if(mb&&*mb=='2')dg.run_mb(true);   // round 8 correction kernels (measured in pv48): FHERMA_DIAG_MB=2 onlydg.run(in_bytes,out_bytes);
        if(sw){sw->batch=batch;sw->run(dg);}}catch(const std::exception& e){std::printf("{\"diag\":\"error\",\"what\":\"%.100s\"}\n",e.what());std::fflush(stdout);}
    sw.reset();
}
