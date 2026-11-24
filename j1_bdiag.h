#pragma once
// Exit-time J=1 whole-batch diag (FHERMA_J1_BDIAG=0: off; FHERMA_BATCH_R pairs per line, default 9).  Paired-line
// pattern of wt-stackd 28f6279: every line times a full B-product batch against the production graph (the exact launch
// fherma_run makes) in alternating pairs; vs_prod = median of the pair ratios (< 1: faster than production); ex = 1 when
// all B outputs are bit-identical to the production outputs (device compare).  Lines (k "batch"):
//   ctrl (production vs production), j1_if1 / j1_if2 / j1_if3 (J=1 path, 1 / 2 / 3 products in flight), one repeat of
//   each J=1 line (_r), ctrl_end.  Then per-stage lines for one product (k "stage", 3 repeats): fwd_a, fwd_b, colfwd,
//   row, colinv, back (us, SMSP cycles, IMMA x 16 share of the SMSP cycles for the conversions).
//   Fused lines (j1ntt-fdiag, FHERMA_J1_FUSE kernels of j1_fused.cuh; own J1Path per split, built and freed in turn):
//   j1f16 / j1f12 / j1f20 (+ _r) = fused path with X split 16+16 / 12+20 / 20+12 tensor+integer warps (Y always 16+16),
//   then k "cycles" lines per split and kernel (X, Y): median block span in SM clock64 cycles (= SMSP cycles) with only
//   the tensor role, only the integer role, both (and each role's span inside both), eff = (T + I - both) / min(T, I),
//   cycles per IMMA (per SMSP) and cycles per residue point (per SMSP); 3 repeats, best.  Local RTX 5070 values
//   are embedded (local_*).  The J=1 path uses its own module and buffers; the timed production path is untouched.
#include "j1_path.h"
#include "j1_bdiag_local.h"
#include <chrono>
#include <functional>
#include <algorithm>
#include <cstring>
namespace j1bd {
inline double local(const char* k){for(const J1BLocal& l:J1B_LOCAL)if(std::strcmp(l.key,k)==0)return l.v;return -1.0;}
}
// prod(): launch the production batch on `gs` (no sync).  Buffers: the production a_all / b_all / out_all.
template<class C> void run_j1_bdiag(C& d,CUstream gs,const std::function<void()>& prod,CUdeviceptr a_all,CUdeviceptr b_all,CUdeviceptr out_all,
                                     uint32_t n,uint32_t w,uint32_t l,uint32_t delta,uint32_t B){
    if(const char* e=std::getenv("FHERMA_J1_BDIAG"))if(*e=='0')return;
    const auto t0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    if(d.cuCtxSynchronize()!=CUDA_SUCCESS)return;
    auto ecreate=d.driver.template get<decltype(&::cuEventCreate)>("cuEventCreate");auto erec=d.driver.template get<decltype(&::cuEventRecord)>("cuEventRecord");
    auto esync=d.driver.template get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");auto etime=d.driver.template get<decltype(&::cuEventElapsedTime)>("cuEventElapsedTime");
    auto edes=d.driver.template get<decltype(&::cuEventDestroy_v2)>("cuEventDestroy_v2");auto dtod=d.driver.template get<decltype(&::cuMemcpyDtoD_v2)>("cuMemcpyDtoD_v2");
    auto mset=d.driver.template get<decltype(&::cuMemsetD32_v2)>("cuMemsetD32_v2");
    std::unique_ptr<J1Path<C>> j1;
    try{j1=std::make_unique<J1Path<C>>(d,n,w,l,delta,B,3u,0);}catch(const std::exception& e){std::printf("{\"diag\":\"j1b\",\"error\":\"init %.100s\"}\n",e.what());std::fflush(stdout);return;}
    const size_t count=size_t(B)*n*l;
    CUdeviceptr ref=0,bad=0;
    if(d.cuMemAlloc(&ref,count*4)!=CUDA_SUCCESS||d.cuMemAlloc(&bad,16)!=CUDA_SUCCESS){std::printf("{\"diag\":\"j1b\",\"error\":\"alloc\"}\n");std::fflush(stdout);if(ref)d.cuMemFree(ref);return;}
    std::printf("{\"diag\":\"j1b\",\"k\":\"info\",\"B\":%u,\"sms\":%u,\"clock_khz\":%u,\"conv\":\"%s\",\"fwd_grid\":[%u,2],\"back_grid\":%u,\"init_s\":%.2f}\n",
        B,j1->nsm,j1->clk,j1->tc?"tc":"ref",j1->fgx,j1->bgx,el());std::fflush(stdout);
    CUevent ea=nullptr,eb=nullptr;d.check(ecreate(&ea,0));d.check(ecreate(&eb,0));
    auto timed=[&](const std::function<void()>& f)->double{d.check(erec(ea,gs));f();d.check(erec(eb,gs));d.check(esync(eb));float ms=0;d.check(etime(&ms,ea,eb));return double(ms)*1e3;};
    auto prodf=[&]{prod();};
    // reference outputs = production
    timed(prodf);d.check(d.cuCtxSynchronize());d.check(dtod(ref,out_all,count*4));
    const int R=std::max(3,std::getenv("FHERMA_BATCH_R")?std::atoi(std::getenv("FHERMA_BATCH_R")):9);
    J1Path<C>* fz=nullptr;   // fused instance for the j1f lines
    auto line=[&](const char* name,uint32_t sflight){   // sflight 0 = production (control)
        const auto tl=std::chrono::steady_clock::now();
        J1Path<C>* jp=fz?fz:j1.get();const uint32_t sf=fz?0u:sflight;
        std::function<void()> var=(sflight||fz)?std::function<void()>([&]{jp->run(a_all,b_all,out_all,gs,sf);}):std::function<void()>(prodf);
        d.check(mset(out_all,0xA5A5A5A5u,count));d.check(d.cuCtxSynchronize());timed(var);
        const unsigned nb=j1->compare(out_all,ref,count,bad,gs);const int ex=nb?0:1;
        std::vector<double> a,b,q;timed(prodf);timed(var);
        for(int i=0;i<R;++i){double x,y;if(i&1){y=timed(var);x=timed(prodf);}else{x=timed(prodf);y=timed(var);}a.push_back(x);b.push_back(y);q.push_back(y/x);}
        auto med=[](std::vector<double> z){std::sort(z.begin(),z.end());return z[z.size()/2];};
        char lk[64],lv[64];std::snprintf(lk,64,"%s.vs",name);std::snprintf(lv,64,"%s.med",name);
        std::printf("{\"diag\":\"j1b\",\"k\":\"batch\",\"v\":\"%s\",\"med_us\":%.1f,\"min_us\":%.1f,\"prod_us\":%.1f,\"vs_prod\":%.4f,\"ex\":%d,\"bad_groups\":%u,\"n\":%d,\"local_vs_prod\":%.4f,\"local_med_us\":%.1f,\"s\":%.2f}\n",
            name,med(b),*std::min_element(b.begin(),b.end()),med(a),med(q),ex,nb,R,j1bd::local(lk),j1bd::local(lv),std::chrono::duration<double>(std::chrono::steady_clock::now()-tl).count());std::fflush(stdout);};
    try{
        line("ctrl",0);line("j1_if1",1);line("j1_if2",2);line("j1_if3",3);
        line("j1_if1_r",1);line("j1_if2_r",2);line("j1_if3_r",3);
        {   std::vector<typename J1Path<C>::FuseCyc> cy[3];const uint32_t xts[3]={16,12,20};uint32_t qgs[3]={0,0,0};
            for(int k=0;k<3;++k){char nm[32];std::unique_ptr<J1Path<C>> f;
                try{f=std::make_unique<J1Path<C>>(d,n,w,l,delta,B,2u,1,xts[k]);}catch(const std::exception& e){std::printf("{\"diag\":\"j1b\",\"error\":\"fused init %u: %.100s\"}\n",xts[k],e.what());std::fflush(stdout);continue;}
                fz=f.get();std::snprintf(nm,32,"j1f%u",xts[k]);line(nm,0);std::snprintf(nm,32,"j1f%u_r",xts[k]);line(nm,0);
                fz=nullptr;std::snprintf(nm,32,"j1_if1_x%u",xts[k]);line(nm,1);   // unfused line next to it (same pairing)
                cy[k]=f->fuse_cycle_data(a_all,b_all,out_all,3);qgs[k]=f->qg;f.reset();}
            line("j1_if1_r2",1);
            for(int k=0;k<3;++k)for(const auto& r:cy[k]){const double* c=r.c;char key[48],sp[16];
                std::snprintf(sp,16,"%u+%u",r.kind?16u:xts[k],r.kind?16u:32u-xts[k]);std::snprintf(key,48,"%s%u",r.kind?"Y":"X",xts[k]);
                auto lk=[&](const char* f){static char b[64];std::snprintf(b,64,"%s.%s",key,f);return j1bd::local(b);};
                const double lT=lk("tensor_alone"),lI=lk("int_alone"),lB=lk("both"),lE=lk("eff");
                std::printf("{\"diag\":\"j1b\",\"k\":\"cycles\",\"kernel\":\"%s\",\"x_split\":\"%u+%u\",\"split\":\"%s\",\"qg\":%u,\"tensor_alone\":%.0f,\"int_alone\":%.0f,\"both\":%.0f,"
                    "\"tensor_in_both\":%.0f,\"int_in_both\":%.0f,\"eff\":%.3f,\"cyc_per_imma_alone\":%.2f,\"cyc_per_imma_both\":%.2f,\"cyc_per_pt_int_alone\":%.3f,\"cyc_per_pt_int_both\":%.3f,"
                    "\"local_tensor_alone\":%.0f,\"local_int_alone\":%.0f,\"local_both\":%.0f,\"local_eff\":%.3f}\n",
                    r.kind?"Y":"X",xts[k],32u-xts[k],sp,qgs[k],c[0],c[1],c[2],c[3],c[4],r.eff,c[0]/(r.imma/r.per),c[2]/(r.imma/r.per),c[1]/(r.ipt/r.per),c[2]/(r.ipt/r.per),lT,lI,lB,lE);}
            std::fflush(stdout);}
        line("ctrl_end",0);
        // per-stage lines, one product (inputs: product 0 of the batch)
        const char* nm[6]={"fwd_a","fwd_b","colfwd","row","colinv","back"};
        for(int r=0;r<3;++r){auto t=j1->stage_us(a_all,b_all,out_all);
            for(int i=0;i<6;++i){const double im=i<2?j1->imma_fwd:i==5?j1->imma_back:0.0;const double cyc=t[i]*j1->clk*1e-3;   // per-SMSP cycles = wall cycles
                char lk[64];std::snprintf(lk,64,"%s.us",nm[i]);
                std::printf("{\"diag\":\"j1b\",\"k\":\"stage\",\"stage\":\"%s\",\"repeat\":%d,\"us\":%.2f,\"smsp_cycles\":%.0f,\"imma16_share\":%.3f,\"local_us\":%.2f}\n",
                    nm[i],r,t[i],cyc,im>0?im*16.0/(cyc*4.0*j1->nsm):0.0,j1bd::local(lk));}
            std::fflush(stdout);}
    }catch(const std::exception& e){std::printf("{\"diag\":\"j1b\",\"error\":\"%.100s\"}\n",e.what());std::fflush(stdout);}
    d.cuCtxSynchronize();edes(ea);edes(eb);d.cuMemFree(ref);d.cuMemFree(bad);j1.reset();
    std::printf("{\"diag\":\"j1b\",\"k\":\"done\",\"s\":%.2f}\n",el());std::fflush(stdout);
}
