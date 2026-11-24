#pragma once
// J=1 pure-RNS batched path (FHERMA_J1=1; default off, the production path and its module are untouched).
// Own NVRTC module (j1_kernel_source.h: j1_ntt.cuh + j1_conv_ref.cuh, compute_90 PTX -> cuModuleLoadData), own tables
// (j1_tables.h, generated at init), own residue buffers; reads the production batch inputs a_all / b_all and writes
// out_all, so h2d / d2h are the production byte copies.
//
// Per product (stream st[i % S], S products in flight, default 2 = FHERMA_J1_INFLIGHT):
//   forward_convert(a, ra, 0), forward_convert(b, rb, 1)  ->  ntt_chain(ra, rb) (y in ra)  ->  back_convert(ra, out)
// Interface the tensor-core conversions (wt-j1conv) must implement, one product, all device pointers, async on `s`:
//   forward_convert(limbs, res, operand, s): limbs [N][55] u32 (a < q canonical) -> res [K][N] u32 in [0, 2p_k),
//       = ref.py forward_convert(.., BfA if operand == 0 else BfB) bit for bit (operand 1 carries 2^32 N^-1 s_k).
//   back_convert(y, out, s): y [K][N] u32 in [0, 2p_k) (= ref.py ntt_inv output) -> out [N][55] u32 canonical mod q,
//       = ref.py back_convert bit for bit.
// Implementations: tensor-core j1_fwd / j1_back (j1_conv_tc.cuh, wt-j1conv 4e0ea38; default) or the integer placeholders
// j1c_fwd_ref / j1c_back_ref (j1_conv_ref.cuh; FHERMA_J1_CONV=ref).  Tables (device, built here at init): fragment-packed
// fwd tables [30 quads][7 ks][32 lanes] uint4 (A, B), pm [120] uint2 (p, floor(2^61/p)), backM pairs [14][15][32] uint4,
// backF [15][32] uint2; plain fwdA / fwdB [224][472] u8, backM [480][224] u8, R [K], barrett_m [K], primes.
#include "j1_tables.h"
#include "j1_kernel_source.h"
#include <cuda.h>
#include <nvrtc.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
template<class C> struct J1Path {
    typedef uint32_t u32;
    C& d;u32 N,K,L,delta,B,S;j1t::Tables T;
    CUmodule mod=nullptr;CUfunction fF=nullptr,fR=nullptr,fI=nullptr,fCF=nullptr,fCB=nullptr,fTF=nullptr,fTB=nullptr,fCmp=nullptr;
    CUdeviceptr dP=0,dPi=0,dT1=0,dTh=0,dTl=0,dBmB=0,dR=0,dFA=0,dFB=0,dBM=0,dTabA=0,dTabB=0,dPm=0,dTm=0,dTf=0;
    bool tc=true;u32 fsm=0,bsm=0,fgx=1,bgx=1,nsm=0,clk=0;
    // fused path (FHERMA_J1_FUSE=1; j1_fused.cuh): module compiled with J1X_T / J1F_QG, X / Y kernels, counters, clock records
    bool fuse=false,xtf=false;u32 xT=16,qg=15,xsm=0,ysm=0;CUmodule fmod=nullptr;CUfunction fX=nullptr,fY=nullptr;CUdeviceptr dDone=0,dRec=0;
    CUmodule compile(const std::vector<std::string>& defs){
        nvrtcProgram prog=nullptr;d.rcheck(d.nvrtcCreateProgram(&prog,J1_KERNEL_SOURCE,"j1_kernels.cu",0,nullptr,nullptr));
        std::vector<std::string> o={"--gpu-architecture=compute_90","--std=c++17"};for(auto& x:defs)o.push_back("-D"+x);
        std::vector<const char*> op;for(auto& x:o)op.push_back(x.c_str());
        if(d.nvrtcCompileProgram(prog,int(op.size()),op.data())!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(prog,&n);std::string log(n,'\0');d.nvrtcGetProgramLog(prog,log.data());d.nvrtcDestroyProgram(&prog);throw std::runtime_error("J1 NVRTC: "+log.substr(0,600));}
        size_t n=0;d.rcheck(d.nvrtcGetPTXSize(prog,&n));std::vector<char> ptx(n);d.rcheck(d.nvrtcGetPTX(prog,ptx.data()));d.rcheck(d.nvrtcDestroyProgram(&prog));
        CUmodule m=nullptr;d.check(d.cuModuleLoadData(&m,ptx.data()));return m;}
    std::vector<CUdeviceptr> ra,rb;std::vector<CUstream> st;
    decltype(&::cuStreamCreate) screate;decltype(&::cuStreamDestroy_v2) sdestroy;
    J1Path(C& d_,u32 N_,u32 W,u32 L_,u32 delta_,u32 B_,u32 S_force=0,int fuse_force=-1,u32 xt_force=0):d(d_),N(N_),K(118),L(L_),delta(delta_),B(B_){
        if(N!=65536u||W!=1747u||L!=55u)throw std::runtime_error("FHERMA_J1: only the N=65536, W=1747 point");
        const char* e=std::getenv("FHERMA_J1_INFLIGHT");S=S_force?S_force:e?u32(std::atoi(e)):2u;if(S<1)S=1;if(S>4)S=4;if(S>B)S=B;
        T=j1t::make_tables(N,K,W,delta);
        // module
        nvrtcProgram prog=nullptr;d.rcheck(d.nvrtcCreateProgram(&prog,J1_KERNEL_SOURCE,"j1_kernels.cu",0,nullptr,nullptr));
        const char* opts[]={"--gpu-architecture=compute_90","--std=c++17"};
        if(d.nvrtcCompileProgram(prog,2,opts)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(prog,&n);std::string log(n,'\0');d.nvrtcGetProgramLog(prog,log.data());d.nvrtcDestroyProgram(&prog);throw std::runtime_error("J1 NVRTC: "+log);}
        size_t n=0;d.rcheck(d.nvrtcGetPTXSize(prog,&n));std::vector<char> ptx(n);d.rcheck(d.nvrtcGetPTX(prog,ptx.data()));d.rcheck(d.nvrtcDestroyProgram(&prog));
        d.check(d.cuModuleLoadData(&mod,ptx.data()));
        d.check(d.cuModuleGetFunction(&fF,mod,"j1_colfwd"));d.check(d.cuModuleGetFunction(&fR,mod,"j1_row"));d.check(d.cuModuleGetFunction(&fI,mod,"j1_colinv"));
        d.check(d.cuModuleGetFunction(&fCF,mod,"j1c_fwd_ref"));d.check(d.cuModuleGetFunction(&fCB,mod,"j1c_back_ref"));
        d.check(d.cuModuleGetFunction(&fTF,mod,"j1_fwd"));d.check(d.cuModuleGetFunction(&fCmp,mod,"j1d_cmp"));d.check(d.cuModuleGetFunction(&fTB,mod,"j1_back"));
        if(const char* c=std::getenv("FHERMA_J1_CONV"))tc=std::string(c)!="ref";
        {   auto fset=d.driver.template get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");
            auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
            auto attr=d.driver.template get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
            CUdevice dev=0;d.check(d.driver.template get<decltype(&::cuDeviceGet)>("cuDeviceGet")(&dev,0));int v=0;
            attr(&v,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);nsm=u32(v);attr(&v,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev);clk=u32(v);
            fsm=15u*7u*512u+15u*32u;bsm=2u*7680u+3840u+8u*(3u*16u*20u*4u+512u);   // J1F_SMEM (QG 15), J1B_SMEM
            d.check(fset(fTF,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(fsm)));d.check(fset(fTB,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(bsm)));
            int bf=0,bb=0;d.check(occ(&bf,fTF,512,fsm));d.check(occ(&bb,fTB,512,bsm));
            fgx=std::max(1u,nsm*u32(std::max(1,bf))/2u);bgx=nsm*u32(std::max(1,bb));}   // as wt-j1conv test_conv.py (2 prime groups of 15 quads)
        // tables and buffers
        dP=d.upload(T.primes);dPi=d.upload(T.pinv);dT1=d.upload(T.tw1);dTh=d.upload(T.thi);dTl=d.upload(T.tlo);dBmB=d.upload(T.bm);dR=d.upload(T.R);
        dFA=d.upload(T.fwdA);dFB=d.upload(T.fwdB);dBM=d.upload(T.backM);
        {   auto ftab=[&](const std::vector<u32>& f){std::vector<u32> o(size_t(30)*7*32*4);   // [7][60][32][2] -> [30][7][32][4]
                for(u32 q=0;q<30;++q)for(u32 ks=0;ks<7;++ks)for(u32 ln=0;ln<32;++ln)for(u32 t=0;t<2;++t)for(u32 h=0;h<2;++h)
                    o[((size_t(q)*7+ks)*32+ln)*4+2*t+h]=f[((size_t(ks)*60+2*q+t)*32+ln)*2+h];return o;};
            dTabA=d.upload(ftab(T.fragA));dTabB=d.upload(ftab(T.fragB));
            std::vector<u32> m(size_t(14)*15*32*4);   // [15][28][32][2] -> [14][15][32][4]
            for(u32 c=0;c<14;++c)for(u32 ks=0;ks<15;++ks)for(u32 ln=0;ln<32;++ln)for(u32 t=0;t<2;++t)for(u32 h=0;h<2;++h)
                m[((size_t(c)*15+ks)*32+ln)*4+2*t+h]=T.fragM[((size_t(ks)*28+2*c+t)*32+ln)*2+h];
            dTm=d.upload(m);dTf=d.upload(T.fragF);
            std::vector<u32> pm(240,0);for(u32 k=0;k<K;++k){pm[2*k]=T.primes[k];pm[2*k+1]=T.bm[k];}dPm=d.upload(pm);}
        screate=d.driver.template get<decltype(&::cuStreamCreate)>("cuStreamCreate");sdestroy=d.driver.template get<decltype(&::cuStreamDestroy_v2)>("cuStreamDestroy_v2");
        if(const char* f=std::getenv("FHERMA_J1_FUSE"))fuse=std::atoi(f)==1;
        if(fuse_force>=0)fuse=fuse_force==1;
        if(xt_force){xT=xt_force;xtf=true;}
        if(fuse)init_fused();
        for(u32 c=0;c<std::max(S,fuse?2u:1u);++c){ra.push_back(d.alloc(size_t(K)*N*4));rb.push_back(d.alloc(size_t(K)*N*4));CUstream s=nullptr;d.check(screate(&s,CU_STREAM_NON_BLOCKING));st.push_back(s);}
        std::fprintf(stderr,"{\"j1\":true,\"K\":%u,\"inflight\":%u,\"conv\":\"%s\",\"fwd_grid\":[%u,2],\"back_grid\":%u}\n",K,S,tc?"tc":"ref",fgx,bgx);
    }
    ~J1Path(){for(auto s:st)if(s)sdestroy(s);if(mod)d.cuModuleUnload(mod);if(fmod)d.cuModuleUnload(fmod);}
    void init_fused(){
        if(const char* e=std::getenv("FHERMA_J1_XT"))if(!xtf)xT=u32(std::atoi(e));if(xT<4||xT>28||(32-xT)%4)throw std::runtime_error("FHERMA_J1_XT: 4..28, 32 - XT a multiple of 4");
        if(const char* e=std::getenv("FHERMA_J1_QG"))qg=u32(std::atoi(e));
        auto xs=[&](u32 g){return g*7u*512u+g*32u+(32u-xT)*2688u;};
        if(!std::getenv("FHERMA_J1_QG")&&xs(qg)>101376u)qg=10;   // 99 KB block limit: QG 10 = 3 prime groups of 10 quads
        if(qg!=15&&qg!=10&&qg!=6&&qg!=5)throw std::runtime_error("FHERMA_J1_QG must divide 30 (15, 10, 6, 5)");
        xsm=xs(qg);ysm=2u*7680u+3840u+8u*(3u*16u*20u*4u+512u)+4u*8704u;
        fmod=compile({"J1X_T="+std::to_string(xT),"J1F_QG="+std::to_string(qg)});
        d.check(d.cuModuleGetFunction(&fX,fmod,"j1_fuse_x"));d.check(d.cuModuleGetFunction(&fY,fmod,"j1_fuse_y"));
        auto fset=d.driver.template get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");
        auto occ=d.driver.template get<decltype(&::cuOccupancyMaxActiveBlocksPerMultiprocessor)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
        auto fattr=d.driver.template get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
        d.check(fset(fX,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(xsm)));d.check(fset(fY,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(ysm)));
        int ox=0,oy=0,rx=0,ry=0,lx=0,ly=0;d.check(occ(&ox,fX,1024,xsm));d.check(occ(&oy,fY,1024,ysm));
        fattr(&rx,CU_FUNC_ATTRIBUTE_NUM_REGS,fX);fattr(&ry,CU_FUNC_ATTRIBUTE_NUM_REGS,fY);fattr(&lx,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fX);fattr(&ly,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fY);
        if(ox<1||oy<1)throw std::runtime_error("J1 fused kernels do not fit an SM");
        dDone=d.alloc(K*4);dRec=d.alloc(size_t(nsm)*32*2*8);
        std::fprintf(stderr,"{\"j1_fuse\":true,\"xT\":%u,\"xI\":%u,\"qg\":%u,\"x_smem\":%u,\"y_smem\":%u,\"x_regs_local\":[%d,%d],\"y_regs_local\":[%d,%d],\"blocks_per_sm\":[%d,%d]}\n",xT,32-xT,qg,xsm,ysm,rx,lx,ry,ly,ox,oy);}
    // fused launches: X(fp = product whose forward conversion runs, ip = product whose row + colinv run), Y(bp back, cp colfwd); -1 = role idle
    void launch_x(CUdeviceptr a_all,CUdeviceptr b_all,int fp,int ip,CUstream s,bool recd=false){
        const size_t stride=size_t(N)*L*4;auto msa=d.driver.template get<decltype(&::cuMemsetD32Async)>("cuMemsetD32Async");
        CUdeviceptr xa=fp>=0?a_all+fp*stride:0,xb=fp>=0?b_all+fp*stride:0,fra=fp>=0?ra[fp%2]:0,frb=fp>=0?rb[fp%2]:0;
        CUdeviceptr A=ip>=0?ra[ip%2]:0,Bv=ip>=0?rb[ip%2]:0,rec=recd?dRec:0;
        if(ip>=0)d.check(msa(dDone,0u,K,s));
        void* a[]={&xa,&xb,&fra,&frb,&dTabA,&dTabB,&dPm,&A,&Bv,&dP,&dPi,&dTh,&dTl,&dT1,&dDone,&rec};
        d.check(d.cuLaunchKernel(fX,nsm,1,1,1024,1,1,xsm,s,a,nullptr));}
    void launch_y(CUdeviceptr out_all,int bp,int cp,CUstream s,bool recd=false){
        const size_t stride=size_t(N)*L*4;
        CUdeviceptr y=bp>=0?ra[bp%2]:0,out=bp>=0?out_all+bp*stride:0,fA=cp>=0?ra[cp%2]:0,fB=cp>=0?rb[cp%2]:0,rec=recd?dRec:0;
        void* a[]={&y,&out,&dTm,&dTf,&fA,&fB,&dP,&dT1,&rec};
        d.check(d.cuLaunchKernel(fY,nsm,1,1,1024,1,1,ysm,s,a,nullptr));}
    // whole batch, one stream: X(F0) Y(C0) then per i: X(F i+1 | I i) Y(B i | C i+1)
    void run_fused(CUdeviceptr a_all,CUdeviceptr b_all,CUdeviceptr out_all,CUstream join=nullptr){
        auto erec=d.driver.template get<decltype(&::cuEventRecord)>("cuEventRecord");auto swait=d.driver.template get<decltype(&::cuStreamWaitEvent)>("cuStreamWaitEvent");
        CUstream s=st[0];const int nb=int(B);
        if(join){if(jev.empty()){auto ec=d.driver.template get<decltype(&::cuEventCreate)>("cuEventCreate");jev.resize(S+1);for(auto& e:jev)d.check(ec(&e,CU_EVENT_DISABLE_TIMING));}
            d.check(erec(jev[S],join));d.check(swait(s,jev[S],0));}
        launch_x(a_all,b_all,0,-1,s);launch_y(out_all,-1,0,s);
        for(int i=0;i<nb;++i){launch_x(a_all,b_all,i+1<nb?i+1:-1,i,s);launch_y(out_all,i,i+1<nb?i+1:-1,s);}
        if(join){d.check(erec(jev[0],s));d.check(swait(join,jev[0],0));}}
    // clock64 overlap of the fused kernels (product 0 / 1 data): per role and together, median block span over SMs
    struct FuseCyc{int kind;double c[5]={0,0,0,0,0};double imma,ipt,per,eff;};   // c: tensor alone, int alone, both, tensor in both, int in both
    std::vector<FuseCyc> fuse_cycle_data(CUdeviceptr a_all,CUdeviceptr b_all,CUdeviceptr out_all,int reps=3){
        std::vector<unsigned long long> h(size_t(nsm)*64);
        auto span=[&](int which)->double{   // 0 tensor warps, 1 integer warps, 2 all; median over blocks
            const u32 t=0;(void)t;std::vector<double> v;
            for(u32 b=0;b<nsm;++b){unsigned long long lo=~0ull,hi=0;
                for(u32 w=0;w<32;++w){const bool ten=w<(cur_kind?16u:xT);if(which==0&&!ten)continue;if(which==1&&ten)continue;
                    lo=std::min(lo,h[(b*32+w)*2]);hi=std::max(hi,h[(b*32+w)*2+1]);}
                v.push_back(double(hi-lo));}
            std::sort(v.begin(),v.end());return v[v.size()/2];};
        // prepare product 0 / 1 intermediates (unfused-equivalent sequence on the fused kernels)
        CUstream s=st[0];launch_x(a_all,b_all,0,-1,s);launch_y(out_all,-1,0,s);launch_x(a_all,b_all,1,0,s);d.check(d.cuCtxSynchronize());
        std::vector<FuseCyc> res;
        for(int kind=0;kind<2;++kind){cur_kind=kind;FuseCyc r;r.kind=kind;double* c=r.c;
            for(int mode=0;mode<3;++mode){double best=-1;   // mode 0 tensor alone, 1 integer alone, 2 both
                for(int rp=0;rp<reps;++rp){
                    if(kind==0)launch_x(a_all,b_all,mode!=1?1:-1,mode!=0?0:-1,s,true);   // X: F(1) | row+colinv(0)
                    else launch_y(out_all,mode!=1?0:-1,mode!=0?1:-1,s,true);              // Y: B(0) | colfwd(1)
                    d.check(d.cuCtxSynchronize());d.check(d.cuMemcpyDtoH(h.data(),dRec,h.size()*8));
                    const double sp=span(mode==0?0:mode==1?1:2);if(best<0||sp<best){best=sp;if(mode==2){c[3]=span(0);c[4]=span(1);}}}
                c[mode]=best;}
            r.imma=kind==0?2.0*imma_fwd:imma_back;r.ipt=double(K)*N;r.per=4.0*nsm;r.eff=(c[0]+c[1]-c[2])/std::min(c[0],c[1]);res.push_back(r);}
        cur_kind=0;return res;}
    void fuse_cycles(CUdeviceptr a_all,CUdeviceptr b_all,CUdeviceptr out_all,FILE* f,const char* tag,int reps=3){
        for(const FuseCyc& r:fuse_cycle_data(a_all,b_all,out_all,reps)){const double* c=r.c;const double imma=r.imma,ipt=r.ipt,per=r.per,eff=r.eff;const int kind=r.kind;
            std::fprintf(f,"{\"%s\":\"j1_fuse_cycles\",\"kernel\":\"%s\",\"split\":\"%u+%u\",\"qg\":%u,\"tensor_alone\":%.0f,\"int_alone\":%.0f,\"both\":%.0f,\"tensor_in_both\":%.0f,\"int_in_both\":%.0f,\"eff\":%.3f,"
                "\"cyc_per_imma_alone\":%.2f,\"cyc_per_imma_both\":%.2f,\"cyc_per_pt_int_alone\":%.3f,\"cyc_per_pt_int_both\":%.3f}\n",
                tag,kind?"Y":"X",kind?16u:xT,kind?16u:32u-xT,qg,c[0],c[1],c[2],c[3],c[4],eff,c[0]/(imma/per),c[2]/(imma/per),c[1]/(ipt/per),c[2]/(ipt/per));std::fflush(f);}
    }
    int cur_kind=0;
    void L_(CUfunction f,u32 gx,u32 gy,u32 gz,u32 bx,CUstream s,void** a){d.check(d.cuLaunchKernel(f,gx,gy,gz,bx,1,1,0,s,a,nullptr));}
    // ---- interface (replaceable conversions) ----
    void forward_convert(CUdeviceptr limbs,CUdeviceptr res,int operand,CUstream s){
        if(tc){CUdeviceptr tb=operand?dTabB:dTabA;u32 n=N;void* a[]={&limbs,&res,&tb,&dPm,&n};d.check(d.cuLaunchKernel(fTF,fgx,2,1,512,1,1,fsm,s,a,nullptr));return;}
        CUdeviceptr tb=operand?dFB:dFA;u32 k=K,n=N;void* a[]={&limbs,&res,&tb,&dP,&dBmB,&k,&n};L_(fCF,N/16u,1,1,128,s,a);}
    void back_convert(CUdeviceptr y,CUdeviceptr out,CUstream s){
        if(tc){u32 n=N;void* a[]={&y,&out,&dTm,&dTf,&n};d.check(d.cuLaunchKernel(fTB,bgx,1,1,512,1,1,bsm,s,a,nullptr));return;}
        u32 k=K,n=N,dl=delta;void* a[]={&y,&out,&dBM,&dR,&k,&n,&dl};L_(fCB,N/32u,1,1,224,s,a);}
    // ---- integer transform chain: ra, rb (conversion outputs) -> y in ra ----
    void ntt_chain(CUdeviceptr A,CUdeviceptr Bv,CUstream s){
        {void* a[]={&A,&Bv,&A,&Bv,&dP,&dT1};L_(fF,32,K,2,128,s,a);}
        {void* a[]={&A,&Bv,&A,&dP,&dPi,&dTh,&dTl};L_(fR,32,K,1,128,s,a);}
        {void* a[]={&A,&A,&dP,&dT1};L_(fI,32,K,1,128,s,a);}
    }
    // per-stage time of one product (us, best of `reps` repeats of `it` launches; stream st[0]): [fwd a, fwd b, colfwd, row, colinv, back]
    std::vector<double> stage_us(CUdeviceptr a,CUdeviceptr b,CUdeviceptr out,u32 reps=3,u32 it=10){
        auto ecreate=d.driver.template get<decltype(&::cuEventCreate)>("cuEventCreate");auto erec=d.driver.template get<decltype(&::cuEventRecord)>("cuEventRecord");
        auto esync=d.driver.template get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");auto etime=d.driver.template get<decltype(&::cuEventElapsedTime)>("cuEventElapsedTime");
        auto edes=d.driver.template get<decltype(&::cuEventDestroy_v2)>("cuEventDestroy_v2");
        CUevent e0,e1;d.check(ecreate(&e0,0));d.check(ecreate(&e1,0));CUstream s=st[0];CUdeviceptr A=ra[0],Bv=rb[0];
        std::vector<double> r(6,-1);
        for(int stg=0;stg<6;++stg)for(u32 rep=0;rep<=reps;++rep){
            d.check(erec(e0,s));
            for(u32 i=0;i<it;++i){switch(stg){
                case 0:forward_convert(a,A,0,s);break;case 1:forward_convert(b,Bv,1,s);break;
                case 2:{void* x[]={&A,&Bv,&A,&Bv,&dP,&dT1};L_(fF,32,K,2,128,s,x);}break;
                case 3:{void* x[]={&A,&Bv,&A,&dP,&dPi,&dTh,&dTl};L_(fR,32,K,1,128,s,x);}break;
                case 4:{void* x[]={&A,&A,&dP,&dT1};L_(fI,32,K,1,128,s,x);}break;
                default:back_convert(A,out,s);}}
            d.check(erec(e1,s));d.check(esync(e1));float ms=0;d.check(etime(&ms,e0,e1));const double us=ms*1e3/it;
            if(rep&&(r[stg]<0||us<r[stg]))r[stg]=us;}
        edes(e0);edes(e1);return r;}
    // IMMAs per product: forward 4096 m-tiles x 30 quads x 14 per operand; back 4096 x (15 + 14 x 30)
    static constexpr double imma_fwd=4096.0*30*14,imma_back=4096.0*(15+14*30);
    // one JSON line per stage; share = IMMAs x 16 cycles / SMSP cycles of the stage
    void report_stages(CUdeviceptr a,CUdeviceptr b,CUdeviceptr out,const char* tag,FILE* f=stderr){
        const char* nm[6]={"fwd_a","fwd_b","colfwd","row","colinv","back"};
        for(int r=0;r<3;++r){auto t=stage_us(a,b,out);double tot=0;for(double x:t)tot+=x;
            std::fprintf(f,"{\"%s\":\"j1_stages\",\"conv\":\"%s\",\"repeat\":%d,\"total_us\":%.1f",tag,tc?"tc":"ref",r,tot);
            for(int i=0;i<6;++i){const double im=i<2?imma_fwd:i==5?imma_back:0.0;const double cyc=t[i]*clk*1e-3*4.0*nsm;
                std::fprintf(f,",\"%s_us\":%.2f",nm[i],t[i]);if(im>0)std::fprintf(f,",\"%s_imma16_share\":%.3f",nm[i],im*16.0/cyc);}
            std::fprintf(f,"}\n");std::fflush(f);}}
    // all B products (asynchronous; the caller synchronises once, as the production run does)
    // join (diag timing): the chains start after the work queued on `join` and `join` waits for all of them;
    // s_use (diag): products in flight (<= S)
    std::vector<CUevent> jev;
    void run(CUdeviceptr a_all,CUdeviceptr b_all,CUdeviceptr out_all,CUstream join=nullptr,u32 s_use=0){
        if(fuse&&!s_use){run_fused(a_all,b_all,out_all,join);return;}
        const size_t stride=size_t(N)*L*4;const u32 SU=s_use?std::min(s_use,S):S;
        auto erec=d.driver.template get<decltype(&::cuEventRecord)>("cuEventRecord");auto swait=d.driver.template get<decltype(&::cuStreamWaitEvent)>("cuStreamWaitEvent");
        if(join){if(jev.empty()){auto ec=d.driver.template get<decltype(&::cuEventCreate)>("cuEventCreate");jev.resize(S+1);for(auto& e:jev)d.check(ec(&e,CU_EVENT_DISABLE_TIMING));}
            d.check(erec(jev[S],join));for(u32 c=0;c<SU;++c)d.check(swait(st[c],jev[S],0));}
        for(u32 i=0;i<B;++i){const u32 c=i%SU;CUstream s=st[c];
            forward_convert(a_all+i*stride,ra[c],0,s);forward_convert(b_all+i*stride,rb[c],1,s);
            ntt_chain(ra[c],rb[c],s);
            back_convert(ra[c],out_all+i*stride,s);}
        if(join)for(u32 c=0;c<SU;++c){d.check(erec(jev[c],st[c]));d.check(swait(join,jev[c],0));}
    }
    // device compare of n words (diag): returns the number of differing uint4 groups
    unsigned compare(CUdeviceptr x,CUdeviceptr y,size_t words,CUdeviceptr bad,CUstream s){
        unsigned z=0;d.check(d.cuMemcpyHtoD(bad,&z,4));unsigned long long n4=words/4;void* a[]={&x,&y,&n4,&bad};
        d.check(d.cuLaunchKernel(fCmp,1024,1,1,256,1,1,0,s,a,nullptr));d.check(d.cuCtxSynchronize());d.check(d.cuMemcpyDtoH(&z,bad,4));return z;}
};
