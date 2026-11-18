#pragma once
// Exit-time thread-block-cluster / DSMEM feasibility probe (after all timed calls, from ~State after the kprof report).
// Prints {"diag":"cl_*",...} JSON lines on stdout: device limits, max potential cluster size and max active clusters for
// the shapes a single-pass cluster NTT would use, remote-DSMEM read/store/bulk-copy throughput (B/clk per CTA from
// clock64, aggregate GB/s from events) for one isolated cluster and for a full grid, remote/local load latency, and an
// L2-resident global read reference. Separate NVRTC program (native cubin when NVRTC knows the arch, else compute_90 PTX).
// FHERMA_CLDIAG=0 disables. Touches no solver state. Local RTX 5070 reference (sm_120, 48 SMs): remote ld 2-5 B/clk/CTA,
// bulk exchange ~2 B/clk/CTA, L2 read ~20 B/clk/SM, remote latency 223 clk vs local 36.
#include <cuda.h>
#include <nvrtc.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* kClusterDiagSource = R"CLD(
typedef unsigned int u32; typedef unsigned long long u64;
__device__ __forceinline__ u32 crank(){u32 r;asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(r));return r;}
__device__ __forceinline__ u32 csize(){u32 r;asm volatile("mov.u32 %0, %%cluster_nctarank;":"=r"(r));return r;}
__device__ __forceinline__ void csync(){asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;":::"memory");}
__device__ __forceinline__ u32 mapa(u32 sa,u32 rank){u32 r;asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"(sa),"r"(rank));return r;}
__device__ __forceinline__ u32 sa_of(const void* p){return u32(__cvta_generic_to_shared(p));}
extern __shared__ __align__(16) u32 S[];
// mode 0 local v4, 1 remote v4 (rank+1), 2 remote scalar (rank+1), 3 remote v4 stores, 4 remote latency, 5 local latency
extern "C" __global__ void cl_bw(u32* out,u64* clk,u32 words,u32 iters,u32 mode){
    const u32 t=threadIdx.x,nt=blockDim.x,r=crank(),cs=csize();
    for(u32 i=t;i<words;i+=nt)S[i]=(mode>=4)?((i+561u)%words):i*2654435761u+r;
    csync();
    const u32 base=sa_of(S),rb=mapa(base,(r+1)%cs);
    u32 acc=0;u64 t0=clock64();
    if(mode==0||mode==1){const u32 n4=words/4;
        for(u32 it=0;it<iters;++it)
            for(u32 i0=t;i0<n4;i0+=4*nt){uint4 v[4];
                #pragma unroll
                for(u32 k=0;k<4;++k){const u32 i=(i0+k*nt)%n4;
                    if(mode==0)v[k]=*reinterpret_cast<const uint4*>(S+4*i);
                    else asm("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(v[k].x),"=r"(v[k].y),"=r"(v[k].z),"=r"(v[k].w):"r"(rb+16*i));}
                #pragma unroll
                for(u32 k=0;k<4;++k)acc+=v[k].x^v[k].y^v[k].z^v[k].w;}
    }else if(mode==2){
        for(u32 it=0;it<iters;++it)
            for(u32 i0=t;i0<words;i0+=8*nt){u32 v[8];
                #pragma unroll
                for(u32 k=0;k<8;++k)asm("ld.shared::cluster.u32 %0, [%1];":"=r"(v[k]):"r"(rb+4*((i0+k*nt)%words)));
                #pragma unroll
                for(u32 k=0;k<8;++k)acc+=v[k];}
    }else if(mode==3){const u32 n4=words/8,wb=rb+16*n4;
        for(u32 it=0;it<iters;++it)
            for(u32 i=t;i<n4;i+=nt)asm volatile("st.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};"::"r"(wb+16*i),"r"(i),"r"(it),"r"(r),"r"(acc):"memory");
    }else if(t==0){u32 x=0;
        for(u32 it=0;it<iters;++it){if(mode==4)asm volatile("ld.shared::cluster.u32 %0, [%1];":"=r"(x):"r"(rb+4*x):"memory");else x=*(volatile u32*)(S+x);}
        acc=x;
    }
    u64 t1=clock64();
    csync();
    if(t==0)clk[blockIdx.x]=t1-t0;
    if(acc==0x12345678u)out[blockIdx.x]=acc;
}
// all-to-all exchange by bulk copies: CTA r pushes chunk q of src (words/C) into rank q's dst slot r (mbarrier complete_tx)
extern "C" __global__ void cl_xbulk(u32* out,u64* clk,u32 words,u32 iters,u32 mode){
    const u32 t=threadIdx.x,nt=blockDim.x,r=crank(),cs=csize();
    u32* src=S;u32* dst=S+words;__shared__ __align__(8) u64 bar;
    for(u32 i=t;i<words;i+=nt)src[i]=i+r;
    const u32 chunk=words/cs,bytes=chunk*4,bsa=sa_of(&bar);
    if(t==0){asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(bsa):"memory");asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    csync();
    u64 t0=clock64();
    for(u32 it=0;it<iters;++it){
        if(t==0)asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(bsa),"r"(bytes*cs):"memory");
        csync();
        if(t<cs){const u32 q=(r+t)%cs;const u32 d=mapa(sa_of(dst+r*chunk),q),b=mapa(bsa,q),s=sa_of(src+q*chunk);
            asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"::"r"(d),"r"(s),"r"(bytes),"r"(b):"memory");}
        asm volatile("{\n.reg .pred P;\nCLW:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@!P bra CLW;\n}"::"r"(bsa),"r"(it&1u):"memory");
        csync();
    }
    u64 t1=clock64();
    if(t==0)clk[blockIdx.x]=t1-t0;
    if(dst[t]==0x12345678u)out[blockIdx.x]=1;
}
// L2-resident global reads (reference)
extern "C" __global__ void cl_l2(const uint4* g,u64* clk,u32 n4,u32 iters,u32* out){
    const u32 t=threadIdx.x,nt=blockDim.x;u32 acc=0;u64 t0=clock64();
    for(u32 it=0;it<iters;++it)
        for(u32 i0=t+blockIdx.x*nt*8;i0<n4;i0+=8*nt*gridDim.x){uint4 v[8];
            #pragma unroll
            for(u32 k=0;k<8;++k)v[k]=__ldcg(g+((i0+k*nt)%n4));
            #pragma unroll
            for(u32 k=0;k<8;++k)acc+=v[k].x^v[k].y^v[k].z^v[k].w;}
    u64 t1=clock64();
    if(t==0)clk[blockIdx.x]=t1-t0;
    if(acc==0x12345678u)out[blockIdx.x]=acc;
}
)CLD";

inline bool cluster_diag_env(){const char* e=std::getenv("FHERMA_CLDIAG");return !e||*e!='0';}

// C: the solver's Cuda wrapper (driver/rtc libraries + NVRTC entry points).
template<class C> void run_cluster_diag(C& d){
    if(!cluster_diag_env())return;
    CUmodule m=nullptr;CUdeviceptr o=0,c=0,g=0;
    auto drv=[&](const char* n){return d.driver.template get<void*>(n);};
    auto ev_create=reinterpret_cast<decltype(&::cuEventCreate)>(drv("cuEventCreate"));
    auto ev_record=reinterpret_cast<decltype(&::cuEventRecord)>(drv("cuEventRecord"));
    auto ev_elapsed=reinterpret_cast<decltype(&::cuEventElapsedTime)>(drv("cuEventElapsedTime"));
    auto ev_destroy=reinterpret_cast<decltype(&::cuEventDestroy)>(drv("cuEventDestroy_v2"));
    auto dattr=reinterpret_cast<decltype(&::cuDeviceGetAttribute)>(drv("cuDeviceGetAttribute"));
    auto fattr=reinterpret_cast<decltype(&::cuFuncSetAttribute)>(drv("cuFuncSetAttribute"));
    auto memset8=reinterpret_cast<decltype(&::cuMemsetD8)>(drv("cuMemsetD8_v2"));
    CUevent e0=nullptr,e1=nullptr;
    try{
        auto lex=reinterpret_cast<decltype(&::cuLaunchKernelEx)>(drv("cuLaunchKernelEx"));
        auto occ_cl=reinterpret_cast<decltype(&::cuOccupancyMaxActiveClusters)>(drv("cuOccupancyMaxActiveClusters"));
        auto occ_pot=reinterpret_cast<decltype(&::cuOccupancyMaxPotentialClusterSize)>(drv("cuOccupancyMaxPotentialClusterSize"));
        CUdevice dev=0;int sms=0,cc1=0,cc2=0,smpb=0,smpsm=0,clk_khz=0,l2=0;
        dattr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);dattr(&cc1,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);dattr(&cc2,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
        dattr(&smpb,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,dev);dattr(&smpsm,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,dev);
        dattr(&clk_khz,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev);dattr(&l2,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,dev);
        if(cc1<9){std::printf("{\"diag\":\"cl_skip\",\"cc\":\"%d.%d\"}\n",cc1,cc2);return;}
        // NVRTC: native cubin (sm_XY) if this NVRTC knows the arch, else compute_90 PTX (driver JIT)
        std::vector<char> img;bool native=false;
        {   nvrtcProgram pr=nullptr;
            auto gsz=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,size_t*)>("nvrtcGetCUBINSize");auto gcub=d.rtc.template get<nvrtcResult(*)(nvrtcProgram,char*)>("nvrtcGetCUBIN");
            const std::string a1="--gpu-architecture=sm_"+std::to_string(cc1)+std::to_string(cc2);
            if(d.nvrtcCreateProgram(&pr,kClusterDiagSource,"cl_diag.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("nvrtc create");
            {const char* op[]={a1.c_str(),"--std=c++17"};if(d.nvrtcCompileProgram(pr,2,op)==NVRTC_SUCCESS){size_t n=0;if(gsz(pr,&n)==NVRTC_SUCCESS&&n){img.resize(n);native=gcub(pr,img.data())==NVRTC_SUCCESS;}}}
            if(!native){d.nvrtcDestroyProgram(&pr);pr=nullptr;d.nvrtcCreateProgram(&pr,kClusterDiagSource,"cl_diag.cu",0,nullptr,nullptr);
                const char* op[]={"--gpu-architecture=compute_90","--std=c++17"};
                if(d.nvrtcCompileProgram(pr,2,op)!=NVRTC_SUCCESS){size_t n=0;d.nvrtcGetProgramLogSize(pr,&n);std::string l(n,'\0');d.nvrtcGetProgramLog(pr,l.data());d.nvrtcDestroyProgram(&pr);throw std::runtime_error("nvrtc: "+l.substr(0,80));}
                size_t n=0;d.nvrtcGetPTXSize(pr,&n);img.assign(n,0);d.nvrtcGetPTX(pr,img.data());}
            d.nvrtcDestroyProgram(&pr);}
        d.check(d.cuModuleLoadData(&m,img.data()));
        CUfunction fbw,fxb,fl2;d.check(d.cuModuleGetFunction(&fbw,m,"cl_bw"));d.check(d.cuModuleGetFunction(&fxb,m,"cl_xbulk"));d.check(d.cuModuleGetFunction(&fl2,m,"cl_l2"));
        for(CUfunction f:{fbw,fxb}){d.check(fattr(f,CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,1));d.check(fattr(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,smpb-1024));}
        d.check(d.cuMemAlloc(&o,1<<16));d.check(d.cuMemAlloc(&c,8<<12));
        d.check(ev_create(&e0,CU_EVENT_DEFAULT));d.check(ev_create(&e1,CU_EVENT_DEFAULT));
        std::printf("{\"diag\":\"cl_info\",\"sms\":%d,\"cc\":\"%d.%d\",\"native_cubin\":%s,\"smem_block_optin\":%d,\"smem_sm\":%d,\"clock_khz\":%d,\"l2\":%d}\n",sms,cc1,cc2,native?"true":"false",smpb,smpsm,clk_khz,l2);
        auto cfg_of=[&](unsigned grid,unsigned th,unsigned sm,unsigned cl,CUlaunchAttribute* at){CUlaunchConfig k{};k.gridDimX=grid;k.gridDimY=1;k.gridDimZ=1;k.blockDimX=th;k.blockDimY=1;k.blockDimZ=1;k.sharedMemBytes=sm;
            at[0].id=CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;at[0].value.clusterDim.x=cl;at[0].value.clusterDim.y=1;at[0].value.clusterDim.z=1;k.attrs=at;k.numAttrs=1;return k;};
        for(unsigned th:{256u,512u})for(unsigned sm:{16384u,32768u,49152u,65536u,98304u}){
            CUlaunchAttribute at[1];CUlaunchConfig k=cfg_of(64,th,sm,1,at);k.numAttrs=0;int pot=0;CUresult rp=occ_pot(&pot,fbw,&k);
            std::printf("{\"diag\":\"cl_occ\",\"threads\":%u,\"smem\":%u,\"max_potential_cluster\":%d",th,sm,rp==CUDA_SUCCESS?pot:-1);
            for(unsigned cl:{1u,2u,4u,8u,16u}){CUlaunchConfig k2=cfg_of(cl*64,th,sm,cl,at);int ac=0;CUresult r=occ_cl(&ac,fbw,&k2);std::printf(",\"active_c%u\":%d",cl,r==CUDA_SUCCESS?ac:-1);}
            std::printf("}\n");
        }
        auto timed=[&](CUfunction f,unsigned grid,unsigned th,unsigned sm,unsigned cl,void** args,float& ms,unsigned long long& mx)->bool{
            CUlaunchAttribute at[1];CUlaunchConfig k=cfg_of(grid,th,sm,cl,at);
            if(lex(&k,f,args,nullptr)!=CUDA_SUCCESS||d.cuCtxSynchronize()!=CUDA_SUCCESS)return false;
            ms=1e30f;for(int rep=0;rep<3;++rep){ev_record(e0,nullptr);if(lex(&k,f,args,nullptr)!=CUDA_SUCCESS)return false;ev_record(e1,nullptr);if(d.cuCtxSynchronize()!=CUDA_SUCCESS)return false;float t=0;ev_elapsed(&t,e0,e1);ms=std::min(ms,t);}
            std::vector<unsigned long long> h(grid);d.check(d.cuMemcpyDtoH(h.data(),c,grid*8));mx=*std::max_element(h.begin(),h.end());return true;};
        static const char* mn[]={"local_v4","remote_v4","remote_u32","remote_st_v4","remote_lat","local_lat"};
        for(unsigned cl:{2u,4u,8u,16u}){
            const unsigned th=512,sm=32768;unsigned words=sm/4;
            CUlaunchAttribute at[1];CUlaunchConfig k=cfg_of(cl,th,sm,cl,at);int ac=0;
            if(occ_cl(&ac,fbw,&k)!=CUDA_SUCCESS||ac<=0){std::printf("{\"diag\":\"cl_bw\",\"cluster\":%u,\"launchable\":false}\n",cl);continue;}
            for(int full=0;full<2;++full){const unsigned grid=full?cl*unsigned(ac):cl;
                for(unsigned mode=0;mode<6;++mode){if(!full&&mode==0)continue;if(full&&mode>=4)continue;
                    unsigned iters=mode>=4?2048:32;void* args[]={&o,&c,&words,&iters,&mode};float ms;unsigned long long mx;
                    if(!timed(fbw,grid,th,sm,cl,args,ms,mx)){std::printf("{\"diag\":\"cl_bw\",\"cluster\":%u,\"mode\":\"%s\",\"error\":true}\n",cl,mn[mode]);continue;}
                    if(mode>=4)std::printf("{\"diag\":\"cl_lat\",\"cluster\":%u,\"mode\":\"%s\",\"clk_per_load\":%.0f}\n",cl,mn[mode],double(mx)/iters);
                    else{const double bytes=double(mode==3?words/2:words)*4*iters;
                        std::printf("{\"diag\":\"cl_bw\",\"cluster\":%u,\"clusters\":%u,\"mode\":\"%s\",\"B_per_clk_cta\":%.2f,\"agg_GBps\":%.0f}\n",cl,grid/cl,mn[mode],bytes/double(mx),bytes*grid/(ms*1e-3)/1e9);}}
                {   unsigned w=4096,iters=32,mode=0;void* args[]={&o,&c,&w,&iters,&mode};const unsigned sm2=2*w*4;
                    CUlaunchConfig k2=cfg_of(cl,th,sm2,cl,at);int ac2=0;if(occ_cl(&ac2,fxb,&k2)!=CUDA_SUCCESS||ac2<=0)continue;
                    const unsigned grid2=full?cl*unsigned(ac2):cl;float ms;unsigned long long mx;
                    if(!timed(fxb,grid2,th,sm2,cl,args,ms,mx)){std::printf("{\"diag\":\"cl_xbulk\",\"cluster\":%u,\"error\":true}\n",cl);continue;}
                    std::printf("{\"diag\":\"cl_xbulk\",\"cluster\":%u,\"clusters\":%u,\"kb_per_cta\":16,\"clk_per_exchange\":%.0f,\"B_per_clk_cta\":%.2f,\"agg_GBps\":%.0f}\n",cl,grid2/cl,double(mx)/iters,double(w)*4*iters/double(mx),double(w)*4*iters*grid2/(ms*1e-3)/1e9);}
            }
        }
        {   const unsigned n4=(16u<<20)/16;d.check(d.cuMemAlloc(&g,size_t(n4)*16));d.check(memset8(g,1,size_t(n4)*16));
            unsigned nn=n4,iters=8;void* args[]={&g,&c,&nn,&iters,&o};
            for(unsigned mul:{1u,2u}){const unsigned grid=unsigned(sms)*mul;
                if(d.cuLaunchKernel(fl2,grid,1,1,512,1,1,0,nullptr,args,nullptr)!=CUDA_SUCCESS||d.cuCtxSynchronize()!=CUDA_SUCCESS)break;
                float ms=1e30f;for(int rep=0;rep<3;++rep){ev_record(e0,nullptr);d.check(d.cuLaunchKernel(fl2,grid,1,1,512,1,1,0,nullptr,args,nullptr));ev_record(e1,nullptr);d.check(d.cuCtxSynchronize());float t=0;ev_elapsed(&t,e0,e1);ms=std::min(ms,t);}
                const double bytes=double(n4)*16*iters;
                std::printf("{\"diag\":\"cl_l2ref\",\"grid\":%u,\"agg_GBps\":%.0f,\"B_per_clk_sm\":%.1f}\n",grid,bytes/(ms*1e-3)/1e9,bytes/(ms*1e-3)/(double(clk_khz)*1e3)/sms);}
        }
    }catch(const std::exception& e){std::printf("{\"diag\":\"cl_error\",\"what\":\"%.100s\"}\n",e.what());}
    std::fflush(stdout);
    if(e0)ev_destroy(e0);if(e1)ev_destroy(e1);
    for(auto p:{o,c,g})if(p)d.cuMemFree(p);
    if(m)d.cuModuleUnload(m);
}
