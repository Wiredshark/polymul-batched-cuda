// Exact RNS/NTT multiplication. Setup sees parameters only; all input-dependent
// Batched answer: fherma_h2d/fherma_d2h move bytes only; fherma_run is device compute for all B products.
#include "fherma.h"
#include "host_registration.h"
#include "owned_pinned_vector_pool.h"
#include "output_thp_hint.h"
#include "kernel_source.h"
#include "column_pipeline_graph.h"
#include "root_cache_policy.h"
#include "kprof.h"
#include <cuda.h>
#include <nvrtc.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <type_traits>
#include <dlfcn.h>
#include <unistd.h>
#include <thread>
#include <functional>
#include <atomic>
#include <sched.h>
#include <pthread.h>
#include <sys/resource.h>
#include <mutex>
#include <condition_variable>
#if defined(FHERMA_POOL_NEW)
static constexpr bool pool_new_built=true;
#else
static constexpr bool pool_new_built=false;   // candidate 8 default: no allocator interposition at all
#endif
#ifndef STG_REG_DEFAULT
#define STG_REG_DEFAULT false
#endif

namespace {
using u32=uint32_t; using u64=uint64_t; using Big=std::vector<u32>;
void norm(Big& a) { while(a.size()>1 && !a.back()) a.pop_back(); }
int cmp(const Big& a,const Big& b) {
    if(a.size()!=b.size()) return a.size()>b.size()?1:-1;
    for(size_t i=a.size();i--;) if(a[i]!=b[i]) return a[i]>b[i]?1:-1;
    return 0;
}
u32 bits(const Big& a) { u32 x=a.back(),k=0;while(x){++k;x>>=1;}return u32(32*(a.size()-1))+k; }
Big minus(Big a,const Big& b) {
    if(cmp(a,b)<0) throw std::runtime_error("negative unsigned subtraction");
    u64 borrow=0;
    for(size_t i=0;i<a.size();++i){u64 v=(i<b.size()?u64(b[i]):0)+borrow;u32 old=a[i];a[i]=u32(u64(old)-v);borrow=u64(old)<v;}
    norm(a);return a;
}
Big multiply(const Big& a,const Big& b) {
    Big c(a.size()+b.size(),0);
    for(size_t i=0;i<a.size();++i){u64 carry=0;
        for(size_t j=0;j<b.size();++j){u64 v=u64(a[i])*b[j]+c[i+j]+carry;c[i+j]=u32(v);carry=v>>32;}
        c[i+b.size()]=u32(carry);
    }norm(c);return c;
}
u32 small_mod(const Big& a,u32 p) {u64 r=0;for(size_t i=a.size();i--;)r=((r<<32)|a[i])%p;return u32(r);}
Big exact_divide(const Big& a,u32 p) {Big out(a.size());u64 r=0;for(size_t i=a.size();i--;){u64 v=(r<<32)|a[i];out[i]=u32(v/p);r=v%p;}if(r)throw std::runtime_error("nonexact CRT division");norm(out);return out;}
void shift_add(Big& a,u32 bit) {u64 carry=bit;for(auto& v:a){u64 x=u64(v)*2+carry;v=u32(x);carry=x>>32;}if(carry)a.push_back(u32(carry));}
Big big_mod(const Big& a,const Big& q) {Big r{0};for(u32 i=bits(a);i--;){shift_add(r,(a[i/32]>>(i%32))&1);if(cmp(r,q)>=0)r=minus(r,q);}return r;}
Big half_plus_one(Big a) {u32 carry=0;for(size_t i=a.size();i--;){u32 old=a[i];a[i]=(old>>1)|(carry<<31);carry=old&1;}u64 c=1;for(auto& v:a){u64 x=u64(v)+c;v=u32(x);c=x>>32;}if(c)a.push_back(u32(c));norm(a);return a;}
Big padded(Big a,size_t words){if(a.size()>words)throw std::runtime_error("limb capacity exceeded");a.resize(words,0);return a;}
Big plus(Big a,const Big& b){a.resize(std::max(a.size(),b.size()),0);u64 carry=0;for(size_t i=0;i<a.size();++i){u64 v=u64(a[i])+(i<b.size()?b[i]:0)+carry;a[i]=u32(v);carry=v>>32;}if(carry)a.push_back(u32(carry));norm(a);return a;}
Big shift_left(const Big& a,u32 shift){u32 off=shift/32,s=shift%32;Big out(a.size()+off+1,0);u64 carry=0;for(size_t i=0;i<a.size();++i){u64 v=(u64(a[i])<<s)|carry;out[i+off]=u32(v);carry=v>>32;}out[a.size()+off]=u32(carry);norm(out);return out;}
Big shift_right(const Big& a,u32 shift){u32 off=shift/32,s=shift%32;if(off>=a.size())return Big{0};Big out(a.size()-off);for(size_t i=off;i<a.size();++i){out[i-off]=a[i]>>s;if(s&&i+1<a.size())out[i-off]|=a[i+1]<<(32-s);}norm(out);return out;}
Big two_power(u32 bit){return shift_left(Big{1},bit);}
u32 power(u32 a,u64 e,u32 p){u64 r=1,x=a;while(e){if(e&1)r=r*x%p;x=x*x%p;e>>=1;}return u32(r);}
u64 pack_shoup(u32 value,u32 prime){if(value>=prime||prime>=(u32(1)<<31))throw std::runtime_error("Shoup constant outside supported range");return (((u64(value)<<32)/prime)<<32)|value;}
bool prime32(u32 p) {
    if(p<2)return false;
    for(u32 a:{2u,3u,5u,7u,11u})if(p%a==0)return p==a;
    u32 d=p-1,s=0;while(!(d&1)){d>>=1;++s;}
    // These bases are deterministic over the supported p < 2^31 range.
    for(u32 a:{2u,3u,5u,7u,11u}){u32 x=power(a,d,p);if(x==1||x==p-1)continue;bool ok=false;
        for(u32 r=1;r<s;++r){x=u32(u64(x)*x%p);if(x==p-1){ok=true;break;}}
        if(!ok)return false;
    }return true;
}
u32 square_root(u32 a,u32 p){
    if(!a)return 0;
    if(power(a,(p-1)/2,p)!=1)throw std::runtime_error("missing square root");
    u32 odd=p-1,s=0;while(!(odd&1)){odd>>=1;++s;}
    u32 z=2;while(power(z,(p-1)/2,p)!=p-1)++z;
    u32 c=power(z,odd,p),x=power(a,(odd+1)/2,p),t=power(a,odd,p),m=s;
    while(t!=1){u32 i=0,v=t;while(v!=1&&i<m){v=u32(u64(v)*v%p);++i;}if(i==m)throw std::runtime_error("Tonelli-Shanks failure");u32 b=power(c,u64(1)<<(m-i-1),p);x=u32(u64(x)*b%p);c=u32(u64(b)*b%p);t=u32(u64(t)*c%p);m=i;}
    if(u32(u64(x)*x%p)!=a)throw std::runtime_error("invalid square root");
    return x;
}
u32 eighth_root(u32 value,u32 p){u32 beta=value;for(u32 i=0;i<3;++i)beta=square_root(beta,p);if(power(beta,8,p)!=value)throw std::runtime_error("invalid eighth root");return beta;}
struct Library {
    void* handle;
    explicit Library(const char* name):handle(dlopen(name,RTLD_NOW|RTLD_LOCAL)){if(!handle)throw std::runtime_error(std::string("dlopen ")+name+": "+dlerror());}
    ~Library(){dlclose(handle);}
    template<class T>T get(const char* name){void* p=dlsym(handle,name);if(!p)throw std::runtime_error(std::string("missing symbol ")+name);return reinterpret_cast<T>(p);}
};
inline std::string g_kernel_prefix;  // point-specific NVRTC macros, set before Cuda is built
// p1wl (default; pv91 on-target whole batch 0.9873 / repeat 0.9883, P1 kernel replay 0.9813): bpo_first_x reads the cross
// twiddles and the IMMA B fragments through whole-line table layouts (every LDG.128 request = 512 contiguous bytes = 4 full
// lines; production 3140af3: 16 lines x 32 B). Separate device copies (ctab_wl, tc_bpt_wl) feed only the bpo_first_x launch;
// every other kernel keeps the original tables. FHERMA_P1WL=0 restores the 3140af3 module (no BPO_P1WL) and tables.
static bool p1wl_env(){const char* e=std::getenv("FHERMA_P1WL");return !(e&&*e=='0');}
// Candidate 3 (levers of pv93 on-target wl2 diag, paired whole batch: p1tw 0.9897, p2mt 0.9898, p3f 0.9927): input-independent
// per-(prime, index) values read as whole-line tables. Each lever is its own kernel entry (bpo_first_xt, bpo_middle_r16m,
// bpo_last_cm3f) next to the unchanged production kernel; its table is built on the device once, before the graph capture
// (outside run). If a lever's table cannot be built or its precondition fails, its launches keep the production kernel (the
// lever's off path, identical results) and the reason is printed in one exit line {"diag":"tabs",...}; nothing aborts.
// Switches (module): FHERMA_P1TW=0 / FHERMA_P2MT=0 / FHERMA_P3F=0; all three off + FHERMA_P1WL=0 compiles the 3140af3 module.
//  p1tw: bpo_first_x S6 twist t2 = tw[1+128+((64rr+rl)256+c)/128] (2 primes x 32 rows per warp request, 8 lines) from
//        T1[((u*2+c/128)*256+64rr+rl)*4+tig] (256 contiguous bytes per request); 64 KB.
//  p2mt: bpo_middle_r16 middle twiddles w^(c2 brev4(i)) / w^-(brev4(c2) i) (index math + 8-byte gathers) from per-prime
//        [512+16i+c2] / [768+16i+c2] next to the r16 table (stride 1024 per prime); 128 KB; JIT 64 -> 56 regs (9 blocks/SM).
//  p2cu: (candidate 4, pv96 on-target wl3 diag p2cu 0.9856 / 0.9850) bpo_middle_r16m radix-16 stage twiddles (28 LDG.64 per
//        thread) from a __constant__ table bpo_p2cu (uniform: ULDC), kernel bpo_middle_r16c; 32 KB; FHERMA_P2CU=0 off (module
//        without BPO_P2CU = the candidate 3 module); needs p2mt. Table: device-to-device copies of r16tab before the capture.
//  p1rev: (candidate 5, pv99 on-target whole batch 0.9693 / 0.9710) bpo_first_xt with the prime quads in reverse order
//        (kernel bpo_first_xr): P2 group 0 reads the newest A,B lines; FHERMA_P1REV=0 off (module without BPO_P1RV); needs p1tw.
//  p3f:  bpo_last_cm3 (p3b body) F factor (bm_root gather + Shoup product + companion) from F[prime][c][t]; 8 MB, built
//        by bpo_ftab_init with the kernel's own formula.
static bool tabs_env(const char* n){const char* e=std::getenv(n);return !(e&&*e=='0');}
static bool lever_env(const char* n,bool def){const char* e=std::getenv(n);return e?std::atoi(e)!=0:def;}   // FHERMA_X=0/1 overrides the default
static bool p3t2_env(){return lever_env("FHERMA_P3T2",true);}   // p3t2 default on (candidate 6); FHERMA_P3T2=0 = the 5ca0049 module
static unsigned p3cb_env(){const char* e=std::getenv("FHERMA_P3CB");return e&&std::atoi(e)==4?4u:2u;}   // p3t2 columns per block
// ctab[r][c][8 lane + 2 m2 + e] -> [r][c][64 m2 + 2 lane + e] (xct_ldv request m2 = 512 contiguous bytes)
static std::vector<u64> ctab_wl_of(const std::vector<u64>& ct){std::vector<u64> o(ct.size());
    for(size_t rc=0;rc+256<=ct.size();rc+=256)for(unsigned l=0;l<32;++l)for(unsigned m2=0;m2<4;++m2)for(unsigned e=0;e<2;++e)o[rc+64*m2+2*l+e]=ct[rc+8*l+2*m2+e];return o;}
// bpt[(32 j + lane) 16 + 4 u + x] -> [((4 j + u) 32 + lane) 4 + x] (uint4 request (j, u) = 512 contiguous bytes)
static std::vector<u32> bpt_wl_of(const std::vector<u32>& b){std::vector<u32> o(b.size());const size_t J=b.size()/(32*16);
    for(size_t j=0;j<J;++j)for(unsigned l=0;l<32;++l)for(unsigned u=0;u<4;++u)for(unsigned x=0;x<4;++x)o[((j*4+u)*32+l)*4+x]=b[(j*32+l)*16+4*u+x];return o;}
struct Cuda {
    Library driver{"libcuda.so.1"};
    Library rtc{std::getenv("NVRTC_LIBRARY")?std::getenv("NVRTC_LIBRARY"):"libnvrtc.so.12"};
    // Optional symbols permit an explicit, counted pageable-copy fallback.
    decltype(&::cuMemHostRegister) cuMemHostRegister=reinterpret_cast<decltype(&::cuMemHostRegister)>(dlsym(driver.handle,"cuMemHostRegister_v2"));
    decltype(&::cuMemHostUnregister) cuMemHostUnregister=reinterpret_cast<decltype(&::cuMemHostUnregister)>(dlsym(driver.handle,"cuMemHostUnregister"));
    HostRegistrationStats registration_stats;
#define DRIVER(name,symbol) decltype(&::name) name=driver.get<decltype(&::name)>(symbol)
    DRIVER(cuInit,"cuInit");DRIVER(cuDeviceGet,"cuDeviceGet");
    DRIVER(cuCtxCreate,"cuCtxCreate_v2");DRIVER(cuCtxDestroy,"cuCtxDestroy_v2");
    DRIVER(cuModuleLoadData,"cuModuleLoadData");DRIVER(cuModuleUnload,"cuModuleUnload");DRIVER(cuModuleGetFunction,"cuModuleGetFunction");
    DRIVER(cuMemAlloc,"cuMemAlloc_v2");DRIVER(cuMemFree,"cuMemFree_v2");
    DRIVER(cuMemcpyHtoD,"cuMemcpyHtoD_v2");DRIVER(cuMemcpyDtoH,"cuMemcpyDtoH_v2");
    DRIVER(cuMemcpyHtoDAsync,"cuMemcpyHtoDAsync_v2");DRIVER(cuMemcpyDtoHAsync,"cuMemcpyDtoHAsync_v2");
    DRIVER(cuCtxSynchronize,"cuCtxSynchronize");DRIVER(cuLaunchKernel,"cuLaunchKernel");DRIVER(cuGetErrorString,"cuGetErrorString");
#undef DRIVER
#define RTC(name) decltype(&::name) name=rtc.get<decltype(&::name)>(#name)
    RTC(nvrtcCreateProgram);RTC(nvrtcCompileProgram);RTC(nvrtcGetProgramLogSize);RTC(nvrtcGetProgramLog);RTC(nvrtcGetPTXSize);RTC(nvrtcGetPTX);RTC(nvrtcDestroyProgram);
#undef RTC
    CUstream stream=nullptr;  // capture stream for the batched graph (null = legacy)
    CUcontext context=nullptr;CUmodule module=nullptr;std::vector<CUdeviceptr> buffers;
    std::unique_ptr<L2RootCachePolicy> root_cache;
    CUfunction prepare,forward,pointwise,inverse,finish,reconstruct,transpose_inputs,transpose_output,reconstruct_explicit,middle_local,inverse_finish,fourstep_forward,fourstep_middle,fourstep_inverse;
    CUfunction weighted_prepare,weighted_forward,weighted_middle,weighted_inverse,reconstruct_weighted8,reconstruct_weighted8_subwarp16;
    CUfunction batched_prepfwd=nullptr,batched_forward=nullptr,batched_inverse=nullptr,batched_middle=nullptr,batched_crt=nullptr,batched_prepare2=nullptr,batched_prepare3=nullptr,batched_crt_tc=nullptr,batched_prepare_tc=nullptr,batched_prepfwd_tc=nullptr,bpo_first=nullptr,bpo_middle=nullptr,bpo_last=nullptr,bpo_last_cm=nullptr,bpo_crt_cm=nullptr,bpo_crt_tc_cm=nullptr,bpo_crt_lean=nullptr,bpo_first_tc=nullptr,bpo_first_tcp=nullptr,bpo_last_cm2=nullptr,bpo_middle_db=nullptr,bpo_first_tc2=nullptr,bpo_first_tc3=nullptr,bpo_middle_smr=nullptr,bpo_last_cm3=nullptr,bpo_first_tc2n=nullptr,bpo_first_tc2g=nullptr,bpo_p1a=nullptr,bpo_p1b=nullptr,bpo_first_x=nullptr,bpo_first_xv=nullptr,bpo_last_cm3v=nullptr,bpo_middle_smrv=nullptr,bpo_crt_tc_cmv=nullptr,bpo_first_w=nullptr,bpo_middle_r16=nullptr,bpo_last_r16=nullptr,bpo_middle_r16x=nullptr,bpo_fxt_init=nullptr,bpo_first_xt=nullptr,bpo_first_xr=nullptr,bpo_middle_r16m=nullptr,bpo_middle_r16c=nullptr,bpo_last_cm3f=nullptr,bpo_last_cm3t=nullptr,bpo_t1_init=nullptr,bpo_r16x_init=nullptr,bpo_ftab_init=nullptr,bpo_p3crt_x=nullptr,batched_prepare_fused,transpose_inputs_columns,weighted_prepare_columns,weighted_forward_columns,weighted_inverse_columns,reconstruct_weighted8_subwarp16_columns;
    void getfn(CUfunction* f,CUmodule m,const char* name){CUresult r=cuModuleGetFunction(f,m,name);if(r==CUDA_ERROR_NOT_FOUND){*f=nullptr;return;}check(r);kp.add_name(*f,name);}
    void check(CUresult r){if(r!=CUDA_SUCCESS){const char* message=nullptr;cuGetErrorString(r,&message);throw std::runtime_error(message?message:"CUDA error");}}
    void rcheck(nvrtcResult r){if(r!=NVRTC_SUCCESS)throw std::runtime_error("NVRTC error "+std::to_string(int(r)));}
    Cuda(){try{
        check(cuInit(0));CUdevice device;check(cuDeviceGet(&device,0));check(cuCtxCreate(&context,0,device));kp.init(driver);
        if(const char* dump=std::getenv("FHERMA_DUMP_SOURCE")){FILE* f=std::fopen(dump,"wb");if(f){std::string all=g_kernel_prefix+KERNEL_SOURCE;std::fwrite(all.data(),1,all.size(),f);std::fclose(f);}}
        // NVRTC target: compute_90 on cc >= 9.0 (P1 TMA stores), compute_80 otherwise or as fallback; FHERMA_ARCH overrides.
        int ccmaj=0;{auto attr=driver.get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");if(attr)attr(&ccmaj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device);}
        auto build=[&](const char* target){
        nvrtcProgram program=nullptr;rcheck(nvrtcCreateProgram(&program,(g_kernel_prefix+KERNEL_SOURCE).c_str(),"kernels.cu",0,nullptr,nullptr));
        try{const std::string arch=std::string("--gpu-architecture=")+target;const char* options[]={arch.c_str(),"--std=c++17"};auto t_rtc0=std::chrono::steady_clock::now();auto status=nvrtcCompileProgram(program,2,options);auto t_rtc1=std::chrono::steady_clock::now();
            if(status!=NVRTC_SUCCESS){size_t size=0;nvrtcGetProgramLogSize(program,&size);std::string log(size,'\0');nvrtcGetProgramLog(program,log.data());throw std::runtime_error(log);}
            size_t size=0;rcheck(nvrtcGetPTXSize(program,&size));std::vector<char> ptx(size);rcheck(nvrtcGetPTX(program,ptx.data()));check(cuModuleLoadData(&module,ptx.data()));if(std::getenv("FHERMA_INIT_TIMING"))std::fprintf(stderr,"{\"arch\":\"%s\",\"nvrtc_s\":%.2f,\"module_load_s\":%.2f,\"ptx_bytes\":%zu}\n",target,std::chrono::duration<double>(t_rtc1-t_rtc0).count(),std::chrono::duration<double>(std::chrono::steady_clock::now()-t_rtc1).count(),size);
        }catch(...){nvrtcDestroyProgram(&program);throw;}rcheck(nvrtcDestroyProgram(&program));
        };
        if(const char* e=std::getenv("FHERMA_ARCH"))build(e);
        else if(ccmaj>=9){try{build("compute_90");}catch(const std::exception&){module=nullptr;build("compute_80");}}
        else build("compute_80");
        getfn(&prepare,module,"prepare");getfn(&forward,module,"forward");getfn(&pointwise,module,"pointwise");
        getfn(&inverse,module,"inverse");getfn(&finish,module,"finish");getfn(&reconstruct,module,"reconstruct");
        getfn(&transpose_inputs,module,"transpose_inputs");getfn(&transpose_output,module,"transpose_output");
        getfn(&reconstruct_explicit,module,"reconstruct_explicit");
        getfn(&middle_local,module,"middle_local");getfn(&inverse_finish,module,"inverse_finish");
        getfn(&fourstep_forward,module,"fourstep_forward");getfn(&fourstep_middle,module,"fourstep_middle");getfn(&fourstep_inverse,module,"fourstep_inverse");
        getfn(&weighted_prepare,module,"weighted_prepare");getfn(&weighted_forward,module,"weighted_forward");getfn(&weighted_middle,module,"weighted_middle");getfn(&weighted_inverse,module,"weighted_inverse");getfn(&reconstruct_weighted8,module,"reconstruct_weighted8");getfn(&reconstruct_weighted8_subwarp16,module,"reconstruct_weighted8_subwarp16");
        getfn(&batched_prepare_fused,module,"batched_prepare_fused");
        getfn(&batched_middle,module,"batched_middle");
        getfn(&batched_forward,module,"batched_forward");getfn(&batched_inverse,module,"batched_inverse");
        if(g_kernel_prefix.find("BCRT_W")!=std::string::npos){getfn(&batched_crt,module,"batched_crt");getfn(&batched_prepare2,module,"batched_prepare2");getfn(&batched_prepare3,module,"batched_prepare3");getfn(&batched_crt_tc,module,"batched_crt_tc");getfn(&batched_prepare_tc,module,"batched_prepare_tc");getfn(&batched_prepfwd_tc,module,"batched_prepfwd_tc");getfn(&batched_prepfwd,module,"batched_prepfwd");getfn(&bpo_first,module,"bpo_first");getfn(&bpo_middle,module,"bpo_middle");getfn(&bpo_last,module,"bpo_last");getfn(&bpo_last_cm,module,"bpo_last_cm");getfn(&bpo_crt_cm,module,"bpo_crt_cm");getfn(&bpo_crt_tc_cm,module,"bpo_crt_tc_cm");getfn(&bpo_crt_lean,module,"bpo_crt_lean");getfn(&bpo_first_tc,module,"bpo_first_tc");getfn(&bpo_first_tcp,module,"bpo_first_tcp");getfn(&bpo_last_cm2,module,"bpo_last_cm2");getfn(&bpo_middle_db,module,"bpo_middle_db");getfn(&bpo_first_tc2,module,"bpo_first_tc2");getfn(&bpo_first_tc3,module,"bpo_first_tc3");getfn(&bpo_middle_smr,module,"bpo_middle_smr");getfn(&bpo_last_cm3,module,"bpo_last_cm3");getfn(&bpo_first_tc2n,module,"bpo_first_tc2n");getfn(&bpo_first_tc2g,module,"bpo_first_tc2g");getfn(&bpo_p1a,module,"bpo_p1a");getfn(&bpo_first_x,module,"bpo_first_x");getfn(&bpo_first_xv,module,"bpo_first_xv");getfn(&bpo_last_cm3v,module,"bpo_last_cm3v");getfn(&bpo_middle_smrv,module,"bpo_middle_smrv");getfn(&bpo_crt_tc_cmv,module,"bpo_crt_tc_cmv");getfn(&bpo_first_w,module,"bpo_first_w");getfn(&bpo_p1b,module,"bpo_p1b");getfn(&bpo_middle_r16,module,"bpo_middle_r16");getfn(&bpo_last_r16,module,"bpo_last_r16");getfn(&bpo_middle_r16x,module,"bpo_middle_r16x");getfn(&bpo_fxt_init,module,"bpo_fxt_init");getfn(&bpo_first_xt,module,"bpo_first_xt");getfn(&bpo_first_xr,module,"bpo_first_xr");getfn(&bpo_middle_r16m,module,"bpo_middle_r16m");getfn(&bpo_middle_r16c,module,"bpo_middle_r16c");getfn(&bpo_last_cm3f,module,"bpo_last_cm3f");getfn(&bpo_last_cm3t,module,"bpo_last_cm3t");getfn(&bpo_t1_init,module,"bpo_t1_init");getfn(&bpo_r16x_init,module,"bpo_r16x_init");getfn(&bpo_ftab_init,module,"bpo_ftab_init");getfn(&bpo_p3crt_x,module,"bpo_p3crt_x");}
        getfn(&transpose_inputs_columns,module,"transpose_inputs_columns");
        getfn(&weighted_prepare_columns,module,"weighted_prepare_columns");
        getfn(&weighted_forward_columns,module,"weighted_forward_columns");
        getfn(&weighted_inverse_columns,module,"weighted_inverse_columns");
        getfn(&reconstruct_weighted8_subwarp16_columns,module,"reconstruct_weighted8_subwarp16_columns");
    }catch(...){close();throw;}}
    void close(){root_cache.reset();for(auto p:buffers)cuMemFree(p);buffers.clear();if(module)cuModuleUnload(module);module=nullptr;if(context)cuCtxDestroy(context);context=nullptr;}
    ~Cuda(){registration_stats.report();close();}
    CUdeviceptr alloc(size_t bytes){CUdeviceptr p;check(cuMemAlloc(&p,bytes));buffers.push_back(p);return p;}
    template<class T>CUdeviceptr upload(const std::vector<T>& v){auto p=alloc(v.size()*sizeof(T));check(cuMemcpyHtoD(p,v.data(),v.size()*sizeof(T)));return p;}
    void enable_root_cache(CUdeviceptr base,size_t extent){
        L2RootCacheApi a{};
        a.device_attribute=driver.get<decltype(a.device_attribute)>("cuDeviceGetAttribute");
        a.get_limit=driver.get<decltype(a.get_limit)>("cuCtxGetLimit");
        a.set_limit=driver.get<decltype(a.set_limit)>("cuCtxSetLimit");
        a.get_stream=driver.get<decltype(a.get_stream)>("cuStreamGetAttribute");
        a.set_stream=driver.get<decltype(a.set_stream)>("cuStreamSetAttribute");
        a.get_node=driver.get<decltype(a.get_node)>("cuGraphKernelNodeGetAttribute");
        a.set_node=driver.get<decltype(a.set_node)>("cuGraphKernelNodeSetAttribute");
        a.synchronize=cuCtxSynchronize;
        CUdevice dev;check(cuDeviceGet(&dev,0));
        root_cache=std::make_unique<L2RootCachePolicy>(a,dev,base,extent);
    }
    KProf kp;
    void klaunch(CUfunction fn,u32 gx,u32 gy,u32 gz,u32 bx,u32 smem,void** params){
        if(kp.tracing(stream)){CUevent e0=kp.pre(stream);check(cuLaunchKernel(fn,gx,gy,gz,bx,1,1,smem,stream,params,nullptr));kp.post(e0,fn,gx,gy,gz,bx,smem,stream);}
        else check(cuLaunchKernel(fn,gx,gy,gz,bx,1,1,smem,stream,params,nullptr));}
    template<class...Args>void launch(CUfunction fn,u32 gx,u32 gy,Args...args){void* params[]={static_cast<void*>(&args)...};klaunch(fn,gx,gy,1,128,0,params);}
    template<class...Args>void launch_ntt(CUfunction fn,u32 gx,u32 gy,u32 threads,u32 shared_bytes,Args...args){void* params[]={static_cast<void*>(&args)...};klaunch(fn,gx,gy,1,threads,shared_bytes,params);}
    template<class...Args>void launch3(CUfunction fn,u32 gx,u32 gy,u32 gz,u32 threads,Args...args){void* params[]={static_cast<void*>(&args)...};klaunch(fn,gx,gy,gz,threads,0,params);}
    template<class...Args>void launch256(CUfunction fn,u32 gx,u32 gy,Args...args){void* params[]={static_cast<void*>(&args)...};klaunch(fn,gx,gy,1,256,0,params);}
};
// Give a vector whose capacity already holds n elements size n without
// value-initialising them (the D2H writes every element). libstdc++ only:
// std::vector is {start,finish,end_of_storage}; the layout is verified at
// runtime and any mismatch falls back to resize().
template<class T> static void set_size_no_init(std::vector<T>& v,size_t n){
#if defined(__GLIBCXX__)
    static_assert(std::is_trivially_default_constructible_v<T>&&sizeof(std::vector<T>)==3*sizeof(T*));
    if(n>=v.size()&&n<=v.capacity()&&v.data()){
        T* start=v.data();T** impl=reinterpret_cast<T**>(&v);
        if(impl[0]==start&&impl[1]==start+v.size()&&impl[2]==start+v.capacity()){impl[1]=start+n;if(v.size()==n)return;}
    }
#endif
    v.resize(n);
}
// Candidate 8 hook helper: host threads (created in init) that only memcpy one contiguous byte range, split in
// equal parts; the calling thread copies part 0. Blocking waits, so the crew is idle (asleep) during fherma_run.
class CopyCrew {
    std::vector<std::thread> th;std::mutex m;std::condition_variable go,fin;
    char* dst=nullptr;const char* src=nullptr;size_t len=0;unsigned parts=1,left=0;uint64_t gen=0;bool quit=false;
    void part(unsigned i){const size_t a=len/parts*i,b=i+1==parts?len:len/parts*(i+1);if(b>a)std::memcpy(dst+a,src+a,b-a);}
    void loop(unsigned i){uint64_t seen=0;for(;;){{std::unique_lock<std::mutex> lk(m);go.wait(lk,[&]{return quit||gen!=seen;});if(quit)return;seen=gen;}
        part(i);{std::lock_guard<std::mutex> lk(m);if(--left==0)fin.notify_one();}}}
public:
    explicit CopyCrew(unsigned n):parts(std::max(1u,n)){for(unsigned i=1;i<parts;++i)th.emplace_back([this,i]{loop(i);});}
    ~CopyCrew(){{std::lock_guard<std::mutex> lk(m);quit=true;}go.notify_all();for(auto& t:th)t.join();}
    unsigned size()const{return parts;}
    void copy(void* d,const void* s,size_t n){
        if(parts==1||n<(size_t(1)<<20)){std::memcpy(d,s,n);return;}
        {std::lock_guard<std::mutex> lk(m);dst=static_cast<char*>(d);src=static_cast<const char*>(s);len=n;left=parts-1;++gen;}
        go.notify_all();part(0);
        std::unique_lock<std::mutex> lk(m);fin.wait(lk,[&]{return left==0;});}
};
struct HostTimes {   // candidate 8: host-side timing of the hooks and of run's host steps (us), one stderr line at exit
    static constexpr int M=64;double v[8][M]{};int c=0;bool on=true;
    void add(const double* x,int nx){if(!on||c>=M)return;for(int i=0;i<nx&&i<8;++i)v[i][c]=x[i];++c;}
    double med(int i)const{double t[M];for(int j=0;j<c;++j)t[j]=v[i][j];std::sort(t,t+c);return c?t[c/2]:0;}
};
struct State {
    u32 n,w,l,k,delta,logn=0,rows,cols,group=8,logrows=0,ntt_threads=128;std::unique_ptr<Cuda> cuda;
    std::unique_ptr<owning_pinned_vector::Pool> output_pool;
    std::unique_ptr<ColumnStripPipeline<Cuda>> column_pipeline;
    OutputThpHintStats output_thp_stats;
    // candidate 8 staging (setup: sized by B, N, L only): page-locked chunks, a copy stream, one event per chunk, the crew
    struct Stage{bool reg=false;void* buf[4]{};CUevent ev[4]{};CUstream s=nullptr;size_t chunk=0;unsigned nb=0;std::unique_ptr<CopyCrew> crew;bool ok=false;
        decltype(&::cuMemcpyHtoDAsync) h2da=nullptr;decltype(&::cuMemcpyDtoHAsync) d2ha=nullptr;decltype(&::cuEventRecord) erec=nullptr;
        decltype(&::cuEventSynchronize) esync=nullptr;decltype(&::cuStreamSynchronize) ssync=nullptr;} stg;
    HostTimes ht_run,ht_hook;double ht_h2d_last=0,ht_cp=0,ht_wait=0,ht_h2d_cp=0,ht_h2d_wait=0;
    static double us_since(std::chrono::steady_clock::time_point t){return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();}
    void init_stage(){
        stg.ok=false;if(!lever_env("FHERMA_STAGE",true))return;
        auto& d=*cuda;auto& D=d.driver;
        auto mah=D.get<decltype(&::cuMemAllocHost)>("cuMemAllocHost_v2");auto ecr=D.get<decltype(&::cuEventCreate)>("cuEventCreate");
        auto scr=D.get<decltype(&::cuStreamCreate)>("cuStreamCreate");
        stg.h2da=D.get<decltype(&::cuMemcpyHtoDAsync)>("cuMemcpyHtoDAsync_v2");stg.d2ha=D.get<decltype(&::cuMemcpyDtoHAsync)>("cuMemcpyDtoHAsync_v2");
        stg.erec=D.get<decltype(&::cuEventRecord)>("cuEventRecord");stg.esync=D.get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");
        stg.ssync=D.get<decltype(&::cuStreamSynchronize)>("cuStreamSynchronize");
        if(!mah||!ecr||!scr||!stg.h2da||!stg.d2ha||!stg.erec||!stg.esync||!stg.ssync)return;
        const char* e=std::getenv("FHERMA_STG_MB");const size_t mb=e&&std::atoi(e)>0?size_t(std::atoi(e)):32;
        stg.chunk=mb<<20;stg.nb=4;
        stg.reg=lever_env("FHERMA_STG_REG",STG_REG_DEFAULT);
        for(unsigned k=0;k<stg.nb;++k){
            bool okb;if(stg.reg){void* q=nullptr;okb=posix_memalign(&q,4096,stg.chunk)==0;if(okb){std::memset(q,0,stg.chunk);if(!d.cuMemHostRegister||d.cuMemHostRegister(q,stg.chunk,0)!=CUDA_SUCCESS){std::free(q);okb=false;}else stg.buf[k]=q;}}
            else okb=mah(&stg.buf[k],stg.chunk)==CUDA_SUCCESS;
            if(!okb||ecr(&stg.ev[k],CU_EVENT_DISABLE_TIMING)!=CUDA_SUCCESS){free_stage();return;}}
        if(scr(&stg.s,CU_STREAM_NON_BLOCKING)!=CUDA_SUCCESS){free_stage();return;}
        const unsigned hw=std::max(1u,std::thread::hardware_concurrency());
        e=std::getenv("FHERMA_STG_T");const unsigned t=e&&std::atoi(e)>0?unsigned(std::atoi(e)):std::min(16u,std::max(2u,hw/2));
        try{stg.crew=std::make_unique<CopyCrew>(t);}catch(...){free_stage();return;}
        stg.ok=true;
    }
    void free_stage(){
        stg.crew.reset();if(!cuda)return;auto& D=cuda->driver;
        auto mfh=D.get<decltype(&::cuMemFreeHost)>("cuMemFreeHost");auto ed=D.get<decltype(&::cuEventDestroy)>("cuEventDestroy_v2");
        auto sd=D.get<decltype(&::cuStreamDestroy)>("cuStreamDestroy_v2");
        if(stg.s&&stg.ssync)stg.ssync(stg.s);
        for(unsigned k=0;k<4;++k){if(stg.buf[k]){if(stg.reg){if(cuda->cuMemHostUnregister)cuda->cuMemHostUnregister(stg.buf[k]);std::free(stg.buf[k]);}else if(mfh)mfh(stg.buf[k]);}if(stg.ev[k]&&ed)ed(stg.ev[k]);stg.buf[k]=nullptr;stg.ev[k]=nullptr;}
        if(stg.s&&sd){sd(stg.s);}
        stg.s=nullptr;stg.ok=false;
    }
    // Byte copies through the staging chunks. Chunk k is refilled only after its previous DMA finished (its event).
    void stage_up(CUdeviceptr dst,const char* src,size_t bytes){
        auto& d=*cuda;for(size_t off=0,i=0;off<bytes;++i){const unsigned k=unsigned(i%stg.nb);const size_t len=std::min(stg.chunk,bytes-off);
            auto t=std::chrono::steady_clock::now();d.check(stg.esync(stg.ev[k]));ht_wait+=us_since(t);
            t=std::chrono::steady_clock::now();stg.crew->copy(stg.buf[k],src+off,len);ht_cp+=us_since(t);
            d.check(stg.h2da(dst+off,stg.buf[k],len,stg.s));d.check(stg.erec(stg.ev[k],stg.s));off+=len;}
    }
    void stage_down(char* dst,CUdeviceptr src,size_t bytes){
        auto& d=*cuda;const size_t nc=(bytes+stg.chunk-1)/stg.chunk;
        auto issue=[&](size_t i){const unsigned k=unsigned(i%stg.nb);const size_t off=i*stg.chunk,len=std::min(stg.chunk,bytes-off);
            d.check(stg.esync(stg.ev[k]));d.check(stg.d2ha(stg.buf[k],src+off,len,stg.s));d.check(stg.erec(stg.ev[k],stg.s));};
        for(size_t i=0;i<std::min<size_t>(stg.nb,nc);++i)issue(i);
        for(size_t i=0;i<nc;++i){const unsigned k=unsigned(i%stg.nb);const size_t off=i*stg.chunk,len=std::min(stg.chunk,bytes-off);
            auto t=std::chrono::steady_clock::now();d.check(stg.esync(stg.ev[k]));ht_wait+=us_since(t);
            t=std::chrono::steady_clock::now();stg.crew->copy(dst+off,stg.buf[k],len);ht_cp+=us_since(t);if(i+stg.nb<nc)issue(i+stg.nb);}
        d.check(stg.ssync(stg.s));
    }
    uint64_t arena_calls=0,arena_inputs_owned=0,arena_mixed=0,arena_inputs_unowned=0;
    static bool pin_output(void* opaque,void* address,size_t bytes)noexcept{
        auto& driver=*static_cast<Cuda*>(opaque);
        return driver.cuMemHostRegister&&driver.cuMemHostRegister(address,bytes,0)==CUDA_SUCCESS;
    }
    static bool unpin_output(void* opaque,void* address,size_t)noexcept{
        auto& driver=*static_cast<Cuda*>(opaque);
        return driver.cuMemHostUnregister&&driver.cuMemHostUnregister(address)==CUDA_SUCCESS;
    }
    ~State(){
        if(cuda)cuda->kp.report();
        if(ht_run.on&&ht_run.c)std::fprintf(stderr,"{\"hostt\":true,\"n\":%d,\"stage\":%d,\"crew\":%u,\"chunk_mb\":%zu,\"run_pre_us\":%.1f,\"launch_us\":%.1f,\"out_us\":%.1f,\"sync_us\":%.1f,\"run_us\":%.1f,\"h2d_us\":%.0f,\"d2h_us\":%.0f,\"h2d_cp\":%.0f,\"h2d_wait\":%.0f,\"d2h_cp\":%.0f,\"d2h_wait\":%.0f}\n",
            ht_run.c,int(stg.ok),stg.crew?stg.crew->size():0u,stg.chunk>>20,ht_run.med(0),ht_run.med(1),ht_run.med(2),ht_run.med(3),ht_run.med(4),ht_hook.med(0),ht_hook.med(1),ht_hook.med(2),ht_hook.med(3),ht_hook.med(4),ht_hook.med(5));
        try{run_ld3();}catch(const std::exception& e){std::printf("{\"diag\":\"ld3\",\"err\":\"%.120s\"}\n",e.what());}
        try{run_launch_diag();}catch(const std::exception& e){std::printf("{\"diag\":\"ld\",\"err\":\"%.120s\"}\n",e.what());}
        free_stage();
        if(tabs_done&&tabs_any)std::printf("{\"diag\":\"tabs\",\"p1tw\":%d,\"p1rev\":%d,\"p2mt\":%d,\"p2cu\":%d,\"p3f\":%d,\"p3t2\":%d,\"ms\":%.2f,\"fallback\":\"%.120s\"}\n",int(p1tw_t1!=0),int(p1rev_ok),int(p2mt_r16x!=0),int(p2cu_ok),int(p3f_ftab!=0),p3t2_ok?int(p3cb_env()):0,tabs_ms,tabs_why.c_str());
        if(prof){pcalls/= (bcount?bcount:1);prof_report();}
        // Destroy graphs before retiring their registered host storage.
        column_pipeline.reset();
        if(!output_pool)return;
        const bool quiescent=cuda&&cuda->cuCtxSynchronize()==CUDA_SUCCESS;
        output_pool->retire(quiescent);
        const auto& stats=output_pool->stats;
        std::fprintf(stderr,"{\"output_pool\":true,\"borrows\":%llu,\"hits\":%llu,\"growth\":%llu,\"pin_failures\":%llu,\"unpin_failures\":%llu,\"quarantines\":%llu,\"quiescent\":%s}\n",
            (unsigned long long)stats.borrows,(unsigned long long)stats.hits,(unsigned long long)stats.growth,
            (unsigned long long)stats.pin_failures,(unsigned long long)stats.unpin_failures,
            (unsigned long long)stats.quarantined_slots,quiescent?"true":"false");
        std::fprintf(stderr,"{\"prepared_input_arena\":true,\"calls\":%llu,\"both_inputs_owned\":%llu,\"mixed_inputs\":%llu,\"both_inputs_unowned\":%llu,\"prepared_hits\":%llu,\"prepared_misses\":%llu}\n",
            (unsigned long long)arena_calls,(unsigned long long)arena_inputs_owned,
            (unsigned long long)arena_mixed,(unsigned long long)arena_inputs_unowned,
            (unsigned long long)stats.prepared_hits,(unsigned long long)stats.prepared_misses);
        output_pool.reset();
    }
    size_t host_page_size=0;u32 bcount=1;CUdeviceptr a_all=0,b_all=0,out_all=0;
    void alloc_batch(){auto& d=*cuda;const size_t bytes=size_t(bcount)*n*l*4;a_all=d.alloc(bytes);b_all=d.alloc(bytes);out_all=d.alloc(bytes);
        // Three page-locked host slots of exactly one batch tensor (candidate 7: OFF by default, FHERMA_PINNED_SLOTS=1 on):
        // the harness's input vectors and our output vector land in them, so h2d/d2h are DMA copies. Off: no Pool, the
        // replaced operator new only forwards to malloc (ordinary pageable vectors).
        if(pool_new_built&&host_page_size==4096&&!std::getenv("FHERMA_NO_PINNED_SLOTS")&&lever_env("FHERMA_PINNED_SLOTS",false))
            output_pool=std::make_unique<owning_pinned_vector::Pool>(
                owning_pinned_vector::Backend{cuda.get(),&State::pin_output,&State::unpin_output},bytes,3);
        if(!output_pool)init_stage();
        ht_run.on=ht_hook.on=lever_env("FHERMA_HOSTT",true);}
    CUdeviceptr p,mu,tw,itw,invn,weights,a,b,at,bt,A,B,C,prefix,lengths,inverses,midpoint,qmod,pmodq,out,outt,crt_inverses,crt_matrix,neg_pmodq;
    // h30 (default, pv59 whole-batch h30p23 0.993 on target): mixed prime set of 9d2a5bd (12 primes < 2^30 + 31-bit tail, tight
    // CRT bound, tensor CRT F = 45); P2 / P3 Harvey-lazy butterflies for the primes < 2^30 (BPO_NH30), P1 / CRT Shoup (P1 per-prime
    // Barrett shift BPO_MIXED). FHERMA_NO_H30=1: the previous all-31-bit prime set (loose bound, F = 46).
    bool h30=!std::getenv("FHERMA_NO_H30");u32 nh30=0;
    u32 root_stride_host=0;bool weighted=false;u32 short_words=0,digit_terms=0,weighted_middle_threads=128;
    CUdeviceptr weighted_twist,positions,corrections,bias,tc_bfrag=0,tc_bpt=0,tc_rtw=0,r16tab=0,fxt=0,stage_tw=0,stage_itw=0,t2n=0,t2nM=0,t2nO=0,t2nMO=0,ctab=0,ctab_wl=0,tc_bpt_wl=0,p1tw_t1=0,p2mt_r16x=0,p3f_ftab=0;bool p2cu_ok=false,p1rev_ok=false,p3t2_ok=false;std::vector<u32> tc_bfrag_host,tc_bpt_host;std::vector<u64> tc_rtw_host;
    void init_weighted(const Big& q){
        constexpr u32 J=8;weighted=true;
        if(n<J||n%J||w<128)throw std::runtime_error("weighted domain violation");
        if(8u*J*cols>49152)throw std::runtime_error("weighted middle exceeds48KiB dynamic shared capacity");
        // Scheduling uses only the public transform length. Keep the smaller
        // point at 128 threads; independently override the middle for comparison.
        weighted_middle_threads=n>=65536?256:128;
        if(const char* config=std::getenv("FHERMA_WEIGHTED_MIDDLE_THREADS")){
            unsigned long value=std::stoul(config);
            if(value!=128&&value!=256)throw std::runtime_error("FHERMA_WEIGHTED_MIDDLE_THREADS must be 128 or 256");
            weighted_middle_threads=u32(value);
        }
        const u32 digit_bits=(w+J-1)/J;digit_terms=(digit_bits+15)/16;
        const Big vmax=minus(two_power(digit_bits),Big{1});
        // FHERMA_NO_H30=1: 31-bit primes until P > 4 * loose bound (vmax^2 2NJ delta). h30 (default, 9d2a5bd): NH30 = 12
        // primes < 2^30 (Harvey-lazy NTT arithmetic) first, then 31-bit primes until P > 2 * tight bound * (1 + 2^-7). Tight
        // bound: |c_kj| <= N S_j, S_j = sum_{j1+j2 = j mod J} (delta on wrap) 2^e (2^b_j1-1)(2^b_j2-1), e in {0,1}.
        Big bound;
        if(!h30)bound=multiply(multiply(multiply(vmax,vmax),Big{2*n*J}),Big{delta});
        else{u32 pz[J+1];for(u32 j=0;j<J;++j)pz[j]=(w*j+J-1)/J;pz[J]=w;Big smax{0};
            for(u32 j=0;j<J;++j){Big s{0};for(u32 j1=0;j1<J;++j1){const u32 j2=(j+J-j1)%J;
                Big t=multiply(minus(two_power(pz[j1+1]-pz[j1]),Big{1}),minus(two_power(pz[j2+1]-pz[j2]),Big{1}));
                t=shift_left(t,pz[j1]+pz[j2]-pz[(j1+j2)%J]-(j1+j2>=J?w:0u));if(j1+j2>=J)t=multiply(t,Big{delta});s=plus(s,t);}
                if(cmp(s,smax)>0)smax=s;}
            bound=multiply(smax,Big{n});}
        const u32 G=bits(bound);const Big target_bound=!h30?shift_left(bound,2):plus(shift_left(bound,1),shift_right(bound,7));Big P{1};
        nh30=h30?12u:0u;
        u32 candidate=((u32(1)<<(nh30?30:31))-2)/(2*n)*(2*n)+1;
        std::vector<u32> primes,roots,betas;
        while(cmp(P,target_bound)<=0||primes.size()<nh30){
            if(primes.size()==nh30&&nh30&&candidate<(u32(1)<<30))candidate=((u32(1)<<31)-2)/(2*n)*(2*n)+1;   // 31-bit tail
            bool accepted=false;u32 beta=0;
            while(!accepted){
                if(candidate<=2*n)throw std::runtime_error("weighted prime search exhausted");
                if(prime32(candidate)&&delta%candidate){
                    u32 target=u32(u64(power(2,w,candidate))*power(delta%candidate,candidate-2,candidate)%candidate);
                    if(power(target,(candidate-1)/J,candidate)==1){beta=eighth_root(target,candidate);accepted=true;}
                }
                if(!accepted)candidate-=2*n;
            }
            u32 psi=0;for(u32 base=2;!psi;++base){u32 x=power(base,(candidate-1)/(2*n),candidate);if(power(x,n,candidate)==candidate-1)psi=x;}
            primes.push_back(candidate);roots.push_back(psi);betas.push_back(beta);
            P=multiply(P,Big{candidate});candidate-=2*n;
        }
        k=u32(primes.size());if(k>128)throw std::runtime_error("weighted prime capacity exceeded");
        u32 ceil_log_k=0;for(u32 v=k-1;v;v>>=1)++ceil_log_k;
        short_words=(bits(P)+ceil_log_k+1+31)/32;
        if(short_words>32||32*short_words<G+1)throw std::runtime_error("weighted short CRT capacity exceeded");
        const Big capacity=two_power(32*short_words),offset=two_power(G);
        std::vector<u32> digit_positions(J);Big sum_positions{0};
        for(u32 j=0;j<J;++j){digit_positions[j]=(w*j+J-1)/J;sum_positions=plus(sum_positions,two_power(digit_positions[j]));}
        const Big z_bound=shift_left(sum_positions,G+1);
        if(cmp(z_bound,two_power(2047))>=0||cmp(multiply(shift_right(z_bound,w),Big{delta}),two_power(w-2))>=0)throw std::runtime_error("weighted final fold capacity exceeded");
        const Big bias_value=big_mod(shift_left(sum_positions,G),q);
        std::vector<u32> crtInv(k),crtMatrix,crtCorrections;
        for(u32 r=0;r<k;++r){Big partial=exact_divide(P,primes[r]);crtInv[r]=power(small_mod(partial,primes[r]),primes[r]-2,primes[r]);Big row=padded(partial,short_words);crtMatrix.insert(crtMatrix.end(),row.begin(),row.end());}
        for(u32 alpha=0;alpha<=k;++alpha){Big product=multiply(P,Big{alpha});Big correction=cmp(offset,product)>=0?minus(offset,product):minus(capacity,minus(product,offset));Big row=padded(correction,short_words);crtCorrections.insert(crtCorrections.end(),row.begin(),row.end());}
        // Batched throughput: compact two-factor roots by default; FHERMA_FULL_ROOTS=1 selects full psi^i tables (slower locally).
        const bool compact_roots=n==65536&&!std::getenv("FHERMA_FULL_ROOTS");
        const u32 root_stride=compact_roots?641u:n;root_stride_host=root_stride;
        std::vector<u64> ratios(k),twists(size_t(k)*root_stride),untwists(size_t(k)*root_stride),prepared(size_t(k)*J),normalized(size_t(k)*J);
        std::vector<u32> digit_weights(size_t(k)*digit_terms);
        for(u32 r=0;r<k;++r){
            u32 prime=primes[r],psi=roots[r],ipsi=power(psi,prime-2,prime),x=1,y=1;
            ratios[r]=std::numeric_limits<u64>::max()/prime;
            if(u64(digit_terms)>std::numeric_limits<u64>::max()/65535/(prime-1))throw std::runtime_error("weighted digit dot exceeds uint64 capacity");
            u32 dw=1;for(u32 j=0;j<digit_terms;++j){digit_weights[size_t(r)*digit_terms+j]=dw;dw=u32(u64(dw)*65536%prime);}
            if(compact_roots){
                for(unsigned direction=0;direction<2;++direction){
                    auto& roots_out=direction?untwists:twists;u32 base_root=direction?ipsi:psi;
                    size_t row=size_t(r)*root_stride;roots_out[row]=ratios[r];
                    u32 value=1;for(u32 i=0;i<128;++i){roots_out[row+1+i]=pack_shoup(value,prime);value=u32(u64(value)*base_root%prime);}
                    u32 step=power(base_root,128,prime);value=1;
                    for(u32 i=0;i<512;++i){roots_out[row+1+128+i]=pack_shoup(value,prime);value=u32(u64(value)*step%prime);}
                }
            }else{for(u32 i=0;i<n;++i){twists[size_t(r)*n+i]=pack_shoup(x,prime);untwists[size_t(r)*n+i]=pack_shoup(y,prime);x=u32(u64(x)*psi%prime);y=u32(u64(y)*ipsi%prime);}}
            u32 invbeta=power(betas[r],prime-2,prime),inv_nj=power(n*J,prime-2,prime),invbeta_j=1;
            for(u32 j=0;j<J;++j){
                u32 digit_factor=u32(u64(power(2,digit_positions[j],prime))*invbeta_j%prime);
                u32 inverse_factor=u32(u64(power(digit_factor,prime-2,prime))*inv_nj%prime);
                inverse_factor=u32(u64(inverse_factor)*crtInv[r]%prime);
                // Exact separable factors: shared psi^i roots remain in tw/itw.
                size_t ix=size_t(r)*J+j;prepared[ix]=pack_shoup(digit_factor,prime);normalized[ix]=pack_shoup(inverse_factor,prime);
                invbeta_j=u32(u64(invbeta_j)*invbeta%prime);
            }
        }
        {   // Thread-per-coefficient CRT needs the point's limb geometry at compile time.
            const std::string nl(1,'\n');
            std::string m="#define BCRT_W "+std::to_string(w)+nl+"#define BCRT_L "+std::to_string(l)+nl+"#define BCRT_K "+std::to_string(k)+nl+
                "#define BCRT_SW "+std::to_string(short_words)+nl+"#define BCRT_POS ";
            for(u32 j=0;j<J;++j)m+=std::to_string(digit_positions[j])+(j+1<J?",":nl.c_str());
            auto list=[&](const char* name,const auto& v,const char* suffix){m+=std::string("#define ")+name+" ";
                for(size_t x=0;x<v.size();++x)m+=std::to_string(v[x])+suffix+(x+1<v.size()?",":"");m+=nl;};
            list("BCRT_PRIMES",primes,"u");list("BCRT_MUS",ratios,"ull");list("BCRT_PM",crtMatrix,"u");
            list("BCRT_Q",padded(q,l),"u");list("BPR_WEIGHTS",digit_weights,"u");list("BPR_FACT",prepared,"ull");
            {std::vector<u32> fw(size_t(k)*J*digit_terms);   // pass-order: digit weight * per-digit factor
             for(u32 r=0;r<k;++r)for(u32 j=0;j<J;++j)for(u32 t=0;t<digit_terms;++t)
                 fw[(size_t(r)*J+j)*digit_terms+t]=u32(u64(digit_weights[size_t(r)*digit_terms+t])*u32(prepared[size_t(r)*J+j])%primes[r]);
             list("BPO_WEIGHTS",fw,"u");}
            m+="#define BPR_DT "+std::to_string(digit_terms)+nl;list("BCRT_BIAS",padded(bias_value,l),"u");
            if(h30)m+="#define BPO_MIXED 1"+nl+"#define BPO_NH30 "+std::to_string(nh30)+nl;
            if(std::getenv("FHERMA_P2_OLD"))m+="#define BPO_P2_OLD 1"+nl;   // opt-out: pre-p2k bpo_middle_r16 addressing (pv65 p2k 0.985 on target)
            if(p1wl_env())m+="#define BPO_P1WL 1"+nl;
            if(n==65536&&compact_roots&&k%4==0&&tabs_env("FHERMA_P1TW"))m+="#define BPO_P1TW 1"+nl;   // p1tw (T1 twist table, kernel bpo_first_xt)
            if(n==65536&&compact_roots&&k%4==0&&tabs_env("FHERMA_P1TW")&&tabs_env("FHERMA_P1REV"))m+="#define BPO_P1RV 1"+nl;   // p1rev (reverse quad order, kernel bpo_first_xr)
            if(n==65536&&tabs_env("FHERMA_P2MT"))m+="#define BPO_P2MT 1"+nl;   // p2mt (r16x middle twiddles, kernel bpo_middle_r16m)
            if(n==65536&&k&&k<=16&&tabs_env("FHERMA_P2MT")&&tabs_env("FHERMA_P2CU"))m+="#define BPO_P2CU 1"+nl;   // p2cu (constant stage twiddles, kernel bpo_middle_r16c)
            if(n==65536&&tabs_env("FHERMA_P3F"))m+="#define BPO_P3F 1"+nl;
            if(n==65536&&tabs_env("FHERMA_P3F")&&p3t2_env()){m+="#define BPO_P3T2 1"+nl;m+="#define BPO_P3CB ";m+=char('0'+p3cb_env());m+=nl;}   // p3t2 (kernel bpo_last_cm3t)   // p3f (F factor table, kernel bpo_last_cm3f)
            if(std::getenv("FHERMA_P3_OLD"))m+="#define BPO_P3_OLD 1"+nl;   // opt-out: pre-p3b bpo_last_cm3 (pv67 p3b 0.994 on target)
            if(!compact_roots)m+="#define WROOT_COMPACT 0"+nl;
            if(k==16&&short_words==16&&!std::getenv("FHERMA_NO_TCCRT")){
                // Tensor-core CRT constants (batched_crt_tc): B fragments, alpha reciprocals, 2^512-P, 2^G.
                // h30: F = largest <= 46 with every R = floor(2^F/p) < 2^16 (45 with primes < 2^30), plus the alpha rounding
                // slack check 2 bound 2^F + 2 K pmax P < P 2^F (9d2a5bd). Production: F = 46.
                u32 F=46;bool ok=true;u32 pmax=0;if(h30)for(u32 r=0;r<k;++r){pmax=std::max(pmax,primes[r]);while(F>32&&((u64(1)<<F)/primes[r])>>16)--F;}
                std::vector<u32> R(k);for(u32 r=0;r<k;++r){R[r]=u32((u64(1)<<F)/primes[r]);if(R[r]>>16)ok=false;}
                if(h30&&cmp(plus(shift_left(bound,F+1),multiply(P,Big{2*k*pmax})),shift_left(P,F))>=0)ok=false;
                auto pm_byte=[&](u32 r,int c)->u32{if(c<0||c>=int(4*short_words))return 0;return (crtMatrix[size_t(r)*short_words+c/4]>>(8*(c%4)))&255u;};
                auto r_byte=[&](u32 r,int c)->u32{if(c<0||c>=2)return 0;return (R[r]>>(8*c))&255u;};
                tc_bfrag_host.assign(9*2*32*2,0);
                for(u32 T=0;T<9;++T)for(u32 s2=0;s2<2;++s2)for(u32 ln=0;ln<32;++ln)for(u32 h=0;h<2;++h){
                    const u32 g=ln>>2,tg=ln&3,r=8*s2+4*h+tg;const int c=int(8*T+g);u32 v=0;
                    for(int e=0;e<4;++e)v|=(T<8?pm_byte(r,c-e):r_byte(r,int(g)-e))<<(8*e);
                    tc_bfrag_host[((T*2+s2)*32+ln)*2+h]=v;}
                // worst-case column sums: H_k=Y_2k+256*Y_2k+1 must fit u32 (and Y in s32).
                for(u32 T=0;T<9&&ok;++T)for(u32 c2=0;c2<8;c2+=2){u64 y[2]={0,0};
                    for(u32 e2=0;e2<2;++e2)for(u32 r=0;r<k;++r)for(int a2=0;a2<4;++a2){
                        const u64 tmax=a2<3?255:((primes[r]-1)>>24);const int c=int(c2+e2);
                        y[e2]+=tmax*(T<8?pm_byte(r,int(8*T)+c-a2):r_byte(r,c-a2));}
                    if(y[0]+256*y[1]>0xffffffffull||y[0]>0x7fffffffull)ok=false;}
                // pending carry of digit j enters digit j+1's chain (host check of the kernel's assumption)
                for(u32 j=1;j<J;++j){const u32 inj=digit_positions[j-1]/32+short_words+1-digit_positions[j]/32;if(inj>short_words||digit_positions[j]/32+inj>=64)ok=false;}
                if(ok){
                    const Big np=padded(minus(capacity,P),short_words),off=padded(offset,short_words);
                    m+="#define BTC_F "+std::to_string(F)+nl;list("BTC_NP",np,"u");list("BTC_OFF",off,"u");
                }else tc_bfrag_host.clear();
            }
            if(k==16&&compact_roots&&!std::getenv("FHERMA_NO_TCPREP")){
                // Tensor-core base conversion table (batched_prepare_tc): [j][lane][tile*2+reg].
                bool ok=true;for(u32 j=0;j<J;++j){const u32 e=j+1<J?digit_positions[j+1]:w;if(e-digit_positions[j]>224)ok=false;}
                if(ok){
                    tc_bpt_host.assign(size_t(J)*32*16,0);
                    for(u32 j=0;j<J;++j)for(u32 r=0;r<k;++r){
                        const u32 prime=primes[r],fa=u32(prepared[size_t(r)*J+j]);u32 wt[32];u64 pw=1;
                        for(u32 m=0;m<32;++m){wt[m]=u32(pw*fa%prime);pw=pw*256%prime;}
                        for(u32 ln=0;ln<32;++ln){const u32 g=ln>>2,tg=ln&3;
                            for(u32 T=0;T<8;++T){const u32 u=T/2,half=T%2,rr=4*u+g/2,bb=2*half+g%2;if(rr!=r)continue;
                                for(u32 h=0;h<2;++h){u32 v=0;for(u32 e=0;e<4;++e)v|=((wt[16*h+4*tg+e]>>(8*bb))&255u)<<(8*e);
                                    tc_bpt_host[(size_t(j)*32+ln)*16+T*2+h]=v;}}}
                    }
                    m+="#define BPT_ENABLE 1"+nl;
                    // fused prepare+forward: per-lane row-stage roots psi^e (e as in batched_forward's bm_root calls)
                    if(n==65536){tc_rtw_host.assign(size_t(k)*32*12,0);
                        for(u32 r=0;r<k;++r)for(u32 ln=0;ln<32;++ln){u32 ex[12];
                            for(u32 m2=0;m2<4;++m2)ex[m2]=2u*(ln+32u*m2)*(n/256u);
                            for(u32 q2=0;q2<2;++q2)ex[4+q2]=2u*(ln+32u*q2)*(n/128u);
                            ex[6]=2u*ln*(n/64u);
                            for(u32 st=7,len=32;len>=2;len>>=1,++st)ex[st]=2u*(ln&(len/2-1))*(n/len);
                            for(u32 x=0;x<12;++x)tc_rtw_host[(size_t(r)*32+ln)*12+x]=pack_shoup(power(roots[r],ex[x]%(2*n),primes[r]),primes[r]);}}
                }
            }
            if(n==65536&&rows==256&&cols==256&&!std::getenv("FHERMA_LEGACY_KERNELS"))m+="#define BATCHED_ONLY 1"+nl;
            if(std::getenv("FHERMA_PREPFWD"))m+="#define BPF_ENABLE 1"+nl;
            if(lean_jit()&&n==65536&&rows==256&&cols==256)m+="#define BPO_LEAN 1"+nl+"#define BPO_LEAN_SPLIT "+std::string(std::getenv("FHERMA_P1SPLIT")?"1":"0")+nl+"#define BPO_LEAN_GCT "+std::string(std::getenv("FHERMA_P1_CTSMEM")?"0":"1")+nl+"#define BPO_LEAN_P1X "+std::string(std::getenv("FHERMA_P1_TC2G")?"0":"1")+nl+"#define BPO_LEAN_TC2 "+std::string(std::getenv("FHERMA_P1_TC2G")?"1":"0")+nl+"#define BPO_LEAN_P1W "+std::string(std::getenv("FHERMA_P1W")?"1":"0")+nl+((group_flags()&128)?"#define BPO_P3EL 1"+nl:std::string());   // GFLAGS 128 (default, pv61 whole-batch p3el 0.986): group-0 P3 residue stores evict_last   // default graph kernels only (init time)
            if(const char* x=std::getenv("FHERMA_EXTRA_DEFINES")){for(const char* c=x;*c;++c)m+=(*c==';'?nl[0]:*c);m+=nl;}  // experiments only
            if(w>=1024&&l<=64&&short_words<=32)g_kernel_prefix=m;
        }
        cuda=std::make_unique<Cuda>();auto& device=*cuda;
        {int v=0;CUdevice dev=0;auto attr=device.driver.get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
         if(device.cuDeviceGet(&dev,0)==CUDA_SUCCESS&&attr(&v,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,dev)==CUDA_SUCCESS)l2_bytes=u32(v);}
        p=device.upload(primes);mu=device.upload(ratios);
        if(compact_roots&&w==1747&&l==55&&k==16){
            const size_t half=twists.size()*sizeof(u64);
            if(half!=L2RootCachePolicy::root_bytes/2||untwists.size()!=twists.size())
                throw std::runtime_error("L2R1 root table size");
            tw=device.alloc(2*half);itw=tw+half;
            device.check(device.cuMemcpyHtoD(tw,twists.data(),half));
            device.check(device.cuMemcpyHtoD(itw,untwists.data(),half));
#ifdef L2R1_TEST_CORRUPT_ROOT_COPY
            // Negative-control build only: destroy one used forward root.
            // Swapping both tables would merely permute a valid NTT basis.
            const u64 corrupt_root=0;
            device.check(device.cuMemcpyHtoD(tw+1064u,&corrupt_root,sizeof(corrupt_root)));
#endif
            // L2R1 (legacy single-product policy) off for the batched path: it set a CU_LIMIT_PERSISTING_L2_CACHE_SIZE
            // set-aside (164096 B requested, rounded up by the driver: 3 MB on the 5070) plus a persisting access-policy
            // window on the legacy default stream for the root tables; the batched graphs run on their own streams, so the
            // window never applied, only the set-aside carved L2. FHERMA_L2R1=1 restores it (A/B).
            if(const char* e=std::getenv("FHERMA_L2R1");e&&*e=='1')device.enable_root_cache(tw,2*half);
        }else{tw=device.upload(twists);itw=device.upload(untwists);}
        if(n==65536){   // pass-order kernels: 256-point stage twiddles, entry (len,j) at 256-len+j
            std::vector<u64> st(size_t(k)*256),sit(size_t(k)*256);
            for(u32 r=0;r<k;++r){const u32 prime=primes[r],psi=roots[r],ipsi=power(psi,prime-2,prime);
                st[size_t(r)*256+255]=pack_shoup(1,prime);sit[size_t(r)*256+255]=pack_shoup(1,prime);
                for(u32 len=256;len>=2;len>>=1)for(u32 j=0;j<len/2;++j){const u64 e=u64(2)*j*(n/len);
                    st[size_t(r)*256+256-len+j]=pack_shoup(power(psi,e,prime),prime);sit[size_t(r)*256+256-len+j]=pack_shoup(power(ipsi,e,prime),prime);}}
            stage_tw=device.upload(st);stage_itw=device.upload(sit);
            {   // P2 radix-16 table: per prime [0][e] = w^e, [1][e] = w^-e (Shoup), w = psi^512 (S root), e < 256
                std::vector<u64> rt(size_t(k)*512);
                for(u32 r=0;r<k;++r){const u32 prime=primes[r],wf=power(roots[r],512,prime),wi=power(wf,prime-2,prime);u32 x=1,y=1;
                    for(u32 e=0;e<256;++e){rt[size_t(r)*512+e]=pack_shoup(x,prime);rt[size_t(r)*512+256+e]=pack_shoup(y,prime);x=u32(u64(x)*wf%prime);y=u32(u64(y)*wi%prime);}}
                r16tab=device.upload(rt);}
            if(compact_roots){   // P3 (variant 4): t2n[r][j][x&1][x>>1] = ipsi^(128x) * normalized factor j (x = 2 row + (c>>7): row-contiguous, coalesced)
                std::vector<u64> tn(size_t(k)*J*512);
                for(u32 r=0;r<k;++r){const u32 prime=primes[r],ipsi=power(roots[r],prime-2,prime),step=power(ipsi,128,prime);
                    for(u32 j=0;j<J;++j){u32 v=u32(normalized[size_t(r)*J+j]);for(u32 x=0;x<512;++x){tn[(size_t(r)*J+j)*512+(x&1u)*256u+(x>>1)]=pack_shoup(v,prime);v=u32(u64(v)*step%prime);}}}
                t2n=device.upload(tn);
                // Montgomery-pointwise P2 (untw bit 16): same table times 2^32 mod p
                for(u32 r=0;r<k;++r){const u32 prime=primes[r];const u64 R=(u64(1)<<32)%prime;for(size_t x=0;x<size_t(J)*512;++x){u64& e=tn[size_t(r)*J*512+x];e=pack_shoup(u32(u64(u32(e))*R%prime),prime);}}
                t2nM=device.upload(tn);
                if(std::getenv("FHERMA_V7_T2NOLD")){std::vector<u64> to(tn.size());   // A/B twin: old interleaved layouts
                    for(size_t rj=0;rj<size_t(k)*J;++rj)for(u32 x=0;x<512;++x)to[rj*512+x]=tn[rj*512+(x&1u)*256u+(x>>1)];t2nMO=device.upload(to);
                    for(u32 r=0;r<k;++r){const u32 prime=primes[r];const u64 Ri=power(u32((u64(1)<<32)%prime),prime-2,prime);for(size_t x=0;x<size_t(J)*512;++x){u64& e=to[size_t(r)*J*512+x];e=pack_shoup(u32(u64(u32(e))*Ri%prime),prime);}}
                    t2nO=device.upload(to);}
                // P1 cross twiddles (variant 4): ctab[r][c][slot] = psi^(2 c brev8(slot)) * psi^(c&127)
                std::vector<u64> ct(size_t(k)*256*256);
                for(u32 r=0;r<k;++r){const u32 prime=primes[r],psi=roots[r];
                    u32 pw[256];
                    for(u32 c=0;c<256;++c){const u32 t1=power(psi,c&127u,prime),g=power(psi,u64(2)*c,prime);
                        pw[0]=t1;for(u32 k1=1;k1<256;++k1)pw[k1]=u32(u64(pw[k1-1])*g%prime);   // psi^(2ck1)*t1
                        for(u32 sl=0;sl<256;++sl){u32 k1=0;for(u32 bt=0;bt<8;++bt)if(sl>>bt&1)k1|=1u<<(7-bt);
                            ct[(size_t(r)*256+c)*256+sl]=pack_shoup(pw[k1],prime);}}}
                ctab=device.upload(ct);if(p1wl_env())ctab_wl=device.upload(ctab_wl_of(ct));   // p1wl: whole-line copy for bpo_first_x
            }
        }
        invn=device.upload(normalized);weighted_twist=device.upload(prepared);weights=device.upload(digit_weights);
        positions=device.upload(digit_positions);crt_inverses=device.upload(crtInv);crt_matrix=device.upload(crtMatrix);corrections=device.upload(crtCorrections);qmod=device.upload(padded(q,l));bias=device.upload(padded(bias_value,l));
        if(!tc_bfrag_host.empty())tc_bfrag=device.upload(tc_bfrag_host);
        if(!tc_bpt_host.empty()){tc_bpt=device.upload(tc_bpt_host);if(p1wl_env())tc_bpt_wl=device.upload(bpt_wl_of(tc_bpt_host));}   // p1wl: whole-line copy for bpo_first_x
        if(!tc_rtw_host.empty())tc_rtw=device.upload(tc_rtw_host);
        a=device.alloc(size_t(n)*l*4);b=device.alloc(size_t(n)*l*4);out=device.alloc(size_t(n)*l*4);at=device.alloc(size_t(n)*l*4);bt=device.alloc(size_t(n)*l*4);
        A=device.alloc(size_t(k)*J*n*4);B=device.alloc(size_t(k)*J*n*4);C=device.alloc(size_t(k)*J*n*4);
        if(n==65536&&k==16&&r16tab&&device.bpo_fxt_init&&device.bpo_middle_r16x&&device.bpo_p3crt_x){   // fused P3+CRT: F factor tables TH/TL
            fxt=device.alloc(size_t(k)*40*256*8);device.launch3(device.bpo_fxt_init,40u,k,1u,256u,itw,p,fxt,n);device.check(device.cuCtxSynchronize());}
        // Three empty parameter-only slots for ordinary inputs and output.
        // Generic allocation can only lease a free prepared slot; retained
        // vectors remain exclusive, and only explicit run Scope may grow.
        if(false)
            output_pool=std::make_unique<owning_pinned_vector::Pool>(
                owning_pinned_vector::Backend{cuda.get(),&State::pin_output,&State::unpin_output},size_t(n)*l*4,3);
        (void)0;
        init_column_pipeline();
        // batched: no graphs, so no graph-node coverage requirement
    }
    void init_column_pipeline(){
        if(!(output_pool&&n==65536&&w==1747&&l==55&&k==16&&weighted&&
             rows==256&&cols==256&&group==8&&ntt_threads==128&&weighted_middle_threads==256&&
             short_words>=1&&short_words<=16))return;
        void* slots[3]{};
        if(!output_pool->snapshot_empty_pinned_slots(slots,3))
            throw std::runtime_error("initial prepared slot snapshot failed");
        ColumnPipelineConfig config{};
        config.n=n;config.w=w;config.l=l;config.k=k;config.rows=rows;config.cols=cols;
        config.group=group;config.logrows=logrows;config.outer_threads=ntt_threads;
        config.middle_threads=weighted_middle_threads;config.short_words=short_words;
        config.delta=delta;config.digit_terms=digit_terms;
        config.a=a;config.b=b;config.at=at;config.bt=bt;config.A=A;config.B=B;config.C=C;
        config.out=out;config.p=p;config.mu=mu;config.weighted_twist=weighted_twist;
        config.tw=tw;config.itw=itw;config.invn=invn;config.weights=weights;
        config.positions=positions;config.crt_inverses=crt_inverses;config.crt_matrix=crt_matrix;
        config.corrections=corrections;config.qmod=qmod;config.bias=bias;
        column_pipeline=std::make_unique<ColumnStripPipeline<Cuda>>(*cuda,config,slots);
    }
    explicit State(const fherma::Point& point):n(point.N),w(point.W),l(point.L){
        bcount=point.B;if(!bcount)throw std::runtime_error("B must be positive");
        long page_size=sysconf(_SC_PAGESIZE);if(page_size<=0||(page_size&(page_size-1)))throw std::runtime_error("unsupported host page size");host_page_size=size_t(page_size);
        if(n<2||n>65536||(n&(n-1))||w<2||w>1747||l!=(w+31)/32||point.q.data.size()!=l)throw std::runtime_error("unsupported point");
        Big q=point.q.data;norm(q);if(bits(q)!=w||small_mod(q,2*n)!=1)throw std::runtime_error("invalid modulus");
        Big pow2(w/32+1,0);pow2[w/32]=u32(1)<<(w%32);Big diff=minus(pow2,q);
        if(diff.size()!=1||!diff[0])throw std::runtime_error("requires q=2^W-d, 0<d<2^32");
        delta=diff[0];
        for(u32 x=n;x>1;x>>=1)++logn;
        rows=1u<<(logn/2);
        if(const char* config=std::getenv("FHERMA_FOURSTEP_ROWS")){unsigned long value=std::stoul(config);if(!value||value>n||(value&(value-1)))throw std::runtime_error("FHERMA_FOURSTEP_ROWS must be power of two <=N");rows=u32(value);}
        cols=n/rows;
        if(const char* config=std::getenv("FHERMA_COLUMN_GROUP")){unsigned long value=std::stoul(config);if(!value||value>32||(value&(value-1)))throw std::runtime_error("FHERMA_COLUMN_GROUP must be power of two <=32");group=u32(value);}
        group=std::min(group,cols);
        if(const char* config=std::getenv("FHERMA_NTT_THREADS")){unsigned long value=std::stoul(config);if(value!=128&&value!=256)throw std::runtime_error("FHERMA_NTT_THREADS must be128 or256");ntt_threads=u32(value);}
        if(8u*rows*group>49152||8u*cols>49152)throw std::runtime_error("fourstep decomposition exceeds48KiB dynamic shared capacity");
        for(u32 v=rows;v>1;v>>=1)++logrows;
        if(w>=128&&n>=8){init_weighted(q);alloc_batch();return;}
        // Explicit CRT rounds a fixed-point quotient: require the stronger M>4B.
        Big qm1=minus(q,Big{1}),bound=multiply(multiply(qm1,qm1),Big{4*n}),P{1};
        u32 candidate=((u32(1)<<31)-2)/(2*n)*(2*n)+1;
        std::vector<u32> primes,roots,lens,invs,prefixes;
        while(cmp(P,bound)<=0){
            while(!prime32(candidate)){if(candidate<=2*n)throw std::runtime_error("prime search exhausted");candidate-=2*n;}
            u32 psi=0;for(u32 base=2;!psi;++base){u32 x=power(base,(candidate-1)/(2*n),candidate);if(power(x,n,candidate)==candidate-1)psi=x;}
            primes.push_back(candidate);roots.push_back(psi);lens.push_back(u32(P.size()));invs.push_back(power(small_mod(P,candidate),candidate-2,candidate));
            Big row=padded(P,128);prefixes.insert(prefixes.end(),row.begin(),row.end());P=multiply(P,Big{candidate});candidate-=2*n;
        }
        if(bits(P)>=4096)throw std::runtime_error("CRT capacity exceeded");
        k=u32(primes.size());
        if(k>128)throw std::runtime_error("explicit CRT prime capacity exceeded");
        std::vector<u32> crtInv(k),crtMatrix;
        for(u32 r=0;r<k;++r){Big partial=exact_divide(P,primes[r]);crtInv[r]=power(small_mod(partial,primes[r]),primes[r]-2,primes[r]);Big row=padded(big_mod(partial,q),l);crtMatrix.insert(crtMatrix.end(),row.begin(),row.end());}
        std::vector<u64> ratios(k),twists(size_t(k)*n),untwists(size_t(k)*n),normalized_untwists(size_t(k)*n);std::vector<u32> ninvs(k),digit_weights(size_t(k)*2*l);
        for(u32 r=0;r<k;++r){u32 prime=primes[r],psi=roots[r],ipsi=power(psi,prime-2,prime),x=1,y=1;
            ratios[r]=std::numeric_limits<u64>::max()/prime;ninvs[r]=power(n,prime-2,prime);
            // This stronger bound also covers arbitrary values in the unused
            // top-limb bits, and bounds every partial sum because terms >= 0.
            if(u64(2)*l>std::numeric_limits<u64>::max()/65535/(prime-1))throw std::runtime_error("digit dot product exceeds uint64 capacity");
            u32 digit_weight=1;
            for(u32 j=0;j<2*l;++j){digit_weights[size_t(r)*2*l+j]=digit_weight;digit_weight=u32(u64(digit_weight)*65536%prime);}
            for(u32 j=0;j<n;++j){twists[size_t(r)*n+j]=pack_shoup(x,prime);untwists[size_t(r)*n+j]=pack_shoup(y,prime);normalized_untwists[size_t(r)*n+j]=pack_shoup(u32(u64(y)*ninvs[r]%prime),prime);x=u32(u64(x)*psi%prime);y=u32(u64(y)*ipsi%prime);}
        }
        cuda=std::make_unique<Cuda>();auto& d=*cuda;
        p=d.upload(primes);mu=d.upload(ratios);tw=d.upload(twists);itw=d.upload(untwists);invn=d.upload(normalized_untwists);weights=d.upload(digit_weights);
        crt_inverses=d.upload(crtInv);crt_matrix=d.upload(crtMatrix);neg_pmodq=d.upload(padded(minus(q,big_mod(P,q)),l));
        prefix=d.upload(prefixes);lengths=d.upload(lens);inverses=d.upload(invs);midpoint=d.upload(padded(half_plus_one(P),128));qmod=d.upload(padded(q,l));pmodq=d.upload(padded(big_mod(P,q),l));
        a=d.alloc(size_t(n)*l*4);b=d.alloc(size_t(n)*l*4);out=d.alloc(size_t(n)*l*4);
        at=d.alloc(size_t(n)*l*4);bt=d.alloc(size_t(n)*l*4);outt=d.alloc(size_t(n)*l*4);
        A=d.alloc(size_t(k)*n*4);B=d.alloc(size_t(k)*n*4);C=d.alloc(size_t(k)*n*4);
        alloc_batch();
    }
    void check_shape(const fherma::Tensor<uint32_t>& t,const char* name)const{
        std::vector<int64_t> shape{int64_t(bcount),int64_t(n),int64_t(l)};
        if(t.shape!=shape||t.data.size()!=size_t(bcount)*n*l)throw std::runtime_error(std::string(name)+": shape mismatch");
    }
    // Byte copies only: caller vectors -> buffers allocated in init.
    void h2d(const fherma::Inputs& in){
        const auto t0=std::chrono::steady_clock::now();ht_cp=ht_wait=0;
        check_shape(in.a,"a");check_shape(in.b,"b");
        auto& d=*cuda;const size_t bytes=size_t(bcount)*n*l*4;
        if(stg.ok){stage_up(a_all,reinterpret_cast<const char*>(in.a.data.data()),bytes);stage_up(b_all,reinterpret_cast<const char*>(in.b.data.data()),bytes);d.check(stg.ssync(stg.s));}
        else{d.check(d.cuMemcpyHtoD(a_all,in.a.data.data(),bytes));d.check(d.cuMemcpyHtoD(b_all,in.b.data.data(),bytes));}
        ht_h2d_last=us_since(t0);ht_h2d_cp=ht_cp;ht_h2d_wait=ht_wait;
    }
    fherma::Outputs d2h(fherma::Outputs answer){
        auto& d=*cuda;d.kp.collect();const size_t bytes=size_t(bcount)*n*l*4;
        if(answer.c.data.size()!=size_t(bcount)*n*l)throw std::runtime_error("d2h: output not sized by run");
        const auto t0=std::chrono::steady_clock::now();ht_cp=ht_wait=0;
        if(stg.ok)stage_down(reinterpret_cast<char*>(answer.c.data.data()),out_all,bytes);
        else d.check(d.cuMemcpyDtoH(answer.c.data.data(),out_all,bytes));
        {const double x[6]={ht_h2d_last,us_since(t0),ht_h2d_cp,ht_h2d_wait,ht_cp,ht_wait};ht_hook.add(x,6);}
        return answer;
    }
    // One product: device a,b (row-major N x L) -> device out (row-major N x L).
    // FHERMA_STAGE_PROFILE=1: events between stages (serializes nothing; per-product sync at end of run).
    int prof=(std::getenv("FHERMA_STAGE_PROFILE")&&*std::getenv("FHERMA_STAGE_PROFILE"))?1:0;std::vector<CUevent> pev;std::vector<int> ptag;double pacc[16]{};uint64_t pcalls=0;
    void mark(int tag){if(!prof)return;auto& d=*cuda;static auto rec=d.driver.get<decltype(&::cuEventRecord)>("cuEventRecord");static auto cre=d.driver.get<decltype(&::cuEventCreate)>("cuEventCreate");
        if(pev.size()<=ptag.size()){CUevent e;d.check(cre(&e,0));pev.push_back(e);}d.check(rec(pev[ptag.size()],nullptr));ptag.push_back(tag);}
    void prof_collect(){if(!prof)return;auto& d=*cuda;static auto el=d.driver.get<decltype(&::cuEventElapsedTime)>("cuEventElapsedTime");
        for(size_t i=1;i<ptag.size();++i){float ms=0;d.check(el(&ms,pev[i-1],pev[i]));pacc[ptag[i]]+=ms;}ptag.clear();++pcalls;}
    void prof_report(){if(!prof||!pcalls)return;const char* nm[8]={"start","prepare(+transpose)","forward","middle","inverse","crt","other","x"};
        std::fprintf(stderr,"{\"stage_ms_per_run\":{");for(int i=1;i<6;++i)std::fprintf(stderr,"%s\"%s\":%.2f",i>1?",":"",nm[i],pacc[i]/pcalls);std::fprintf(stderr,"}}\n");}
    // Prime groups: run prepare/forward/middle/inverse for pg primes at a time so the
    // per-group working set (input + pg/K of A,B + C) can stay in L2; CRT last.
    // Default: group primes only when L2 can hold a group's working set
    // (input + group A,B + C + output ~ 94 MB at g=4 for the LARGE point).
    u32 l2_bytes=0;
    u32 prime_group()const{const char* e=std::getenv("FHERMA_PRIME_GROUP");
        u32 g=e?u32(std::atoi(e)):(l2_bytes>=(96u<<20)?4u:k);return (g&&k%g==0)?g:k;}
    bool use_prepfwd()const{return cuda->batched_prepfwd&&weighted&&n==65536&&rows==256&&cols==256&&std::getenv("FHERMA_PREPFWD");}  // experimental: slower (177 vs 138 ms)
    // Fused tensor-core base conversion + first forward pass on the current 256x256 tiling (opt-in).
    bool use_tcfuse()const{return cuda->batched_prepfwd_tc&&tc_bpt&&tc_rtw&&weighted&&n==65536&&rows==256&&cols==256&&std::getenv("FHERMA_TCFUSE")&&!std::getenv("FHERMA_V14");}
    void launch_tcfuse(CUdeviceptr a,CUdeviceptr b,u32 r0,u32 r1,CUdeviceptr A,CUdeviceptr B){cuda->launch3(cuda->batched_prepfwd_tc,cols/8,8u,2u,256u,a,b,A,B,tw,tc_bpt,tc_rtw,n,logrows,r0,r1);}
    bool use_bouter()const{return n==65536&&rows==256&&cols==256&&!std::getenv("FHERMA_OLD_OUTER");}
    bool use_bmiddle()const{return n==65536&&rows==256&&cols==256&&!std::getenv("FHERMA_OLD_MIDDLE");}
    // A,B: workspace bases indexed by absolute prime (row r*8+j); group reuse passes base-r0*J*n.
    void launch_prepare(CUdeviceptr a,CUdeviceptr b,u32 r0,u32 r1){launch_prepare(a,b,r0,r1,this->A,this->B);}
    void launch_prepare(CUdeviceptr a,CUdeviceptr b,u32 r0,u32 r1,CUdeviceptr A,CUdeviceptr B){
        auto& d=*cuda;constexpr u32 J=8;
        if(d.batched_prepare_tc&&tc_bpt&&r0%4==0&&r1%4==0&&!std::getenv("FHERMA_OLD_TCPREP")&&!std::getenv("FHERMA_V14"))d.launch(d.batched_prepare_tc,n/128,2,a,b,A,B,tw,tc_bpt,n,r0,r1);
        else if(d.batched_prepare3&&!std::getenv("FHERMA_PREPARE2")&&!std::getenv("FHERMA_V14"))d.launch(d.batched_prepare3,n/128,2,a,b,A,B,tw,n,u64(n)*l,u64(k)*J*n,r0,r1);
        else d.launch(d.batched_prepare2,n/128,1,a,b,A,B,tw,n,u64(n)*l,u64(k)*J*n,r0,r1);
    }
    void launch_crt(CUdeviceptr out){
        auto& d=*cuda;
        if(d.batched_crt_tc&&tc_bfrag&&!std::getenv("FHERMA_OLD_TCCRT")&&!std::getenv("FHERMA_V14"))d.launch(d.batched_crt_tc,n/128,1,C,out,tc_bfrag,n,delta);
        else d.launch(d.batched_crt,n/128,1,C,out,p,mu,crt_matrix,corrections,qmod,bias,n,delta,u64(0),u64(0));
    }
    bool compute_grouped(CUdeviceptr a,CUdeviceptr b,CUdeviceptr out){
        auto& d=*cuda;const u32 pg=prime_group();
        if(!(d.batched_prepare2&&d.batched_crt&&weighted&&n==65536&&pg<k&&!std::getenv("FHERMA_NO_GROUP")))return false;
        constexpr u32 J=8;mark(0);
        for(u32 r0=0;r0<k;r0+=pg){
            // Group reuse (default): every group's A/B intermediates live in one group-sized slot,
            // so a large L2 keeps them resident instead of writing back 64 MB/product of dead groups.
            const bool reuse=!std::getenv("FHERMA_NO_GROUP_REUSE");
            const size_t goff=size_t(r0)*J*n*4;
            const CUdeviceptr Ag=reuse?A:A+goff,Bg=reuse?B:B+goff,Cg=C+goff;
            const CUdeviceptr pg_=p+size_t(r0)*4,mug=mu+size_t(r0)*8,twg=tw+size_t(r0)*root_stride_host*8,itwg=itw+size_t(r0)*root_stride_host*8,invg=invn+size_t(r0)*J*8;
            const bool tcf=use_tcfuse()&&r0%4==0&&pg%4==0;
            if(tcf)launch_tcfuse(a,b,r0,r0+pg,Ag-goff,Bg-goff);else launch_prepare(a,b,r0,r0+pg,Ag-goff,Bg-goff);mark(1);
            if(tcf){}else if(use_bouter())d.launch_ntt(d.batched_forward,cols/8,pg*J,256,0,Ag,Bg,pg_,twg,n,logrows);else d.launch_ntt(d.weighted_forward,cols/group,pg*J,ntt_threads,8u*rows*group,Ag,Bg,pg_,mug,twg,n,rows,cols,group,logrows,J);mark(2);
            if(use_bmiddle())d.launch_ntt(d.batched_middle,rows,pg,256,0,Ag,Bg,Cg,pg_,mug,twg,itwg,n,rows,cols,logrows,J);else d.launch_ntt(d.weighted_middle,rows,pg,weighted_middle_threads,8u*J*cols,Ag,Bg,Cg,pg_,mug,twg,itwg,n,rows,cols,logrows,J);mark(3);
            if(use_bouter())d.launch_ntt(d.batched_inverse,cols/8,pg*J,256,0,Cg,pg_,itwg,invg,n);else d.launch_ntt(d.weighted_inverse,cols/group,pg*J,ntt_threads,4u*rows*group,Cg,pg_,mug,itwg,invg,n,rows,cols,group,J);mark(4);
        }
        launch_crt(out);mark(5);
        return true;
    }
    // Pass-order pipeline: P1 (prepare+J+R fwd) -> P2 (S fwd, pointwise, S inv) -> P3 (J+R inv) -> CRT.
    CUdeviceptr next_a=0,next_b=0;   // input of the next product on the same chain (i+S, S = chains; 0 for the last S products): CRT prefetch, default on
    u32 xoff=0;   // variant 5 = variant 4 minus features in FHERMA_V5_OFF (1 CRT prefetch, 2 P1 L2 prefetch, 4 P3v2)
    bool off(u32 bit)const{return (xoff&bit)!=0;}
    // FHERMA_P1_PAD=bytes (variant 4/5, xoff bit 8 disables): extra dynamic smem so fewer P1 blocks
    // share an SM, leaving room for other chains' P2/P3/CRT blocks.
    u32 p1_pad(){if(!pox||off(8))return 0;const char* e=std::getenv("FHERMA_P1_PAD");if(!e)return 0;const u32 v=u32(std::atoi(e));
        static bool set=false;if(!set&&v){auto fa=cuda->driver.get<decltype(&::cuFuncSetAttribute)>("cuFuncSetAttribute");cuda->check(fa(cuda->bpo_first_tc,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(v)));set=true;}return v;}
    // Split P1 (variant 4, opt-in FHERMA_P1SPLIT=1 - slower locally; FHERMA_V5_OFF bit 2048 disables).
    // V5_OFF bit 4096: R-DIF stages ctab through smem (split: flag; fused: bpo_first_tc2n instead of the
    // default bpo_first_tc2g that reads ctab directly; FHERMA_P1_CTSMEM=1 also selects tc2n).
    // FHERMA_P1S_COLS columns x FHERMA_P1S_QUADS prime quads per group; intermediate wsT per chain.
    u32 p1s_cols=64,p1s_quads=4,p1s_skip=std::getenv("FHERMA_P1S_SKIP")?u32(std::atoi(std::getenv("FHERMA_P1S_SKIP"))):0u;   // skip: ablation only (1 a, 2 b)
   CUdeviceptr wsT[4]{};
    bool p1split()const{return pox&&!off(2048)&&cuda->bpo_p1a&&cuda->bpo_p1b&&ctab&&tc_bpt&&wsT[cur_ws]&&std::getenv("FHERMA_P1SPLIT");}
    bool pox=false;u32 p1x=0,p1_grid=96;   // p1x: P1 kernel 1 bpo_first_x, 2 bpo_first_xv (graph variant 7, A/B FHERMA_P1X), 3 bpo_first_w
      // variant 4: persistent out-of-phase P1
    bool p3r16()const{return cuda->bpo_last_r16&&r16tab&&std::getenv("FHERMA_P3R16")&&std::atoi(std::getenv("FHERMA_P3R16"))!=0;}   // opt-in: neutral-to-slower locally (P3 is latency-bound, not issue-bound)
    bool p3crtx()const{return fxt&&cuda->bpo_middle_r16x&&cuda->bpo_p3crt_x&&std::getenv("FHERMA_P3CRT_X")&&std::atoi(std::getenv("FHERMA_P3CRT_X"))!=0;}   // fused P2x + P3/CRT (opt-in)
    bool p2r16()const{return cuda->bpo_middle_r16&&r16tab&&!(std::getenv("FHERMA_P2R16")&&std::atoi(std::getenv("FHERMA_P2R16"))==0);}   // default on; FHERMA_P2R16=0 restores the old kernel
    bool p3v2()const{return pox&&!off(4)&&(cuda->bpo_last_cm2||cuda->bpo_last_cm3)&&t2n&&!std::getenv("FHERMA_NO_P3V2");}
    int po=0;   // 0 v14 pipeline, 1 pass-order (natural CRT layout), 2 pass-order column-major residues
    // FHERMA_STAGE_MASK (bench only): bit1 prepare/P1, bit2 forward, bit3 middle/P2, bit4 inverse/P3, bit5 CRT.
    u32 stage_mask()const{const char* e=std::getenv("FHERMA_STAGE_MASK");return e?u32(std::strtoul(e,nullptr,0)):~0u;}
    // Lean JIT (default): NVRTC compiles only the kernels of the default graph (variant 4); any experiment
    // switch below, or FHERMA_FULL_JIT=1, compiles everything and builds all graph variants.
    static bool lean_jit(){for(const char* e:{"FHERMA_FULL_JIT","FHERMA_NO_GRAPH","FHERMA_STAGE_PROFILE","FHERMA_V5_OFF","FHERMA_V14","FHERMA_PO_DEFAULT","BENCH_AB","FHERMA_LEGACY_KERNELS","FHERMA_PREPFWD","FHERMA_TCFUSE"})if(std::getenv(e))return false;return true;}
    bool lean=lean_jit();
    bool po_available()const{if(lean)return cuda->bpo_middle_smr&&cuda->bpo_last_cm3&&cuda->bpo_crt_tc_cm&&(cuda->bpo_first_x||cuda->bpo_first_tc2n||cuda->bpo_first_tc2g||(cuda->bpo_p1a&&cuda->bpo_p1b))&&weighted&&n==65536&&rows==256&&cols==256;
        return cuda->bpo_first&&cuda->bpo_middle&&cuda->bpo_last&&cuda->batched_crt&&weighted&&n==65536&&rows==256&&cols==256;}
    CUdeviceptr wsA[4]{},wsB[4]{},wsC[4]{};u32 cur_ws=0;
    // L2 schedule (FHERMA_L2G=g): after a full-prime P1, P2+P3 run per group of g primes through a fixed
    // per-chain C workspace wsG (g*8*N words) that is rewritten every group, so C stays in L2.
    CUdeviceptr wsG[4]{},wsG2[4]{};u32 l2g=0;bool l2on=false;bool p23db_now=false;CUstream side_streams[4]{};std::vector<CUevent> cap_events;
    // FHERMA_L2F feature bits (L2 schedule only): 1 P2 evict-first A/B loads, 2 P2 discards A/B lines,
    // 4 P3 discards C lines, 8 CRT discards residue lines, 16 CRT streaming output stores, 32 P1 evict-first input loads
    u32 l2f=std::getenv("FHERMA_L2F")?u32(std::strtoul(std::getenv("FHERMA_L2F"),nullptr,0)):31u;   // per-chain workspaces (chain 0 = A,B,C)
    // bpo_first_x TMA stores: 3-D map over one operand's P1 output, dims {N/32 words (c*8+slot%8), 32 slot octets, k*8 planes}.
    // default P1 kernel: bpo_first_x (FHERMA_P1_TC2G=1 restores bpo_first_tc2g; FHERMA_P1W=1 bpo_first_w)
    bool t2n_old()const{return (p1x==2)!=(std::getenv("FHERMA_T2N_SWAP")!=nullptr);}   // A/B twin: V7 (or with SWAP, V6) uses the old t2n layout
    u32 p1x_default()const{return std::getenv("FHERMA_P1_TC2G")?0u:std::getenv("FHERMA_P1W")?3u:1u;}
    u32 p1_threads(CUfunction f){int v=256;static auto ga=cuda->driver.get<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
        if(f&&ga)cuda->check(ga(&v,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f));return u32(v>=512?512:256);}   // launch bound 512 -> 512-thread P1
    CUtensorMap p1x_map(CUdeviceptr base){CUtensorMap m{};auto& d=*cuda;
        static auto enc=d.driver.get<decltype(&::cuTensorMapEncodeTiled)>("cuTensorMapEncodeTiled");
        const cuuint64_t dims[3]={cuuint64_t(n/32),32,cuuint64_t(k)*8},strides[2]={cuuint64_t(n/32)*4,cuuint64_t(n)*4};const cuuint32_t box[3]={8,32,1},es[3]={1,1,1};
        if(enc)d.check(enc(&m,CU_TENSOR_MAP_DATA_TYPE_UINT32,3,reinterpret_cast<void*>(base),dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        return m;}
    void compute_po(CUdeviceptr a,CUdeviceptr b,CUdeviceptr out){
        const CUdeviceptr A=cur_ws?wsA[cur_ws]:this->A,B=cur_ws?wsB[cur_ws]:this->B,C=cur_ws?wsC[cur_ws]:this->C;
        auto& d=*cuda;constexpr u32 J=8;// ungrouped by default: P1 reads the input once for all primes; FHERMA_PO_GROUP=g opts in
        u32 pg=k;if(const char* e=std::getenv("FHERMA_PO_GROUP")){const u32 g=u32(std::atoi(e));if(g&&k%g==0)pg=g;}else if(pox&&big_part()&&k%BIG_GROUP==0)pg=BIG_GROUP;
        const u32 gfl=pg<k?group_flags():0u;
        // m16 (default on large parts, pv46 sweep chain m16 0.996 / clean flush 0.965 on target): ONE P1 launch over all k primes
        // (input staged once, one launch per product) while P2/P3 keep the pg-prime L2 groups; P1 stores keep the group policy
        // (gfl&1: plain; evict_first stores = m16e measured 1.17x); input loads evict_first (no re-read). FHERMA_NO_M16=1 off.
        const bool m16=pox&&pg<k&&!std::getenv("FHERMA_PO_GROUP")&&!std::getenv("FHERMA_NO_M16");
        mark(0);const u32 sm=stage_mask();
        const u32 p3h=std::getenv("FHERMA_EXTRA_DEFINES")&&std::strstr(std::getenv("FHERMA_EXTRA_DEFINES"),"BPO_P3_H 1")?1u:2u;
        CUfunction p1k=nullptr;const u32 g2=(l2on&&wsG[cur_ws]&&l2g<k)?l2g:k;
        bool fxon=false;
        for(u32 r0=0;r0<k;r0+=pg){
            const CUdeviceptr Ag=A+size_t(r0)*J*n*4,Bg=B+size_t(r0)*J*n*4,Cg=C+size_t(r0)*J*n*4;
            const CUdeviceptr pg_=p+size_t(r0)*4,mug=mu+size_t(r0)*8,twg=tw+size_t(r0)*root_stride_host*8,itwg=itw+size_t(r0)*root_stride_host*8,invg=invn+size_t(r0)*J*8;
            const u32 p1r1=m16?k:r0+pg;   // P1 prime range end (m16: all primes at r0 = 0, no P1 at r0 > 0)
            if(!(sm&2)||(m16&&r0)){}else if(p1split()&&r0==0&&pg==k){   // split P1: per (column group, prime-quad group) TC+J-DIF then R-DIF via L2
                const CUdeviceptr T=wsT[cur_ws];const u32 pfs=u32(!off(2)&&!std::getenv("FHERMA_NO_L2PF")?(std::getenv("FHERMA_PF_STEP")?std::atoi(std::getenv("FHERMA_PF_STEP")):int(p1_grid)):0);
                for(u32 c0=0;c0<cols;c0+=p1s_cols)for(u32 u0=0;u0<k/4;u0+=p1s_quads){
                    if(!(p1s_skip&1))d.launch_ntt(d.bpo_p1a,2u*p1s_cols,1u,256u,0u,a,b,T,tw,stage_tw,tc_bpt,n,c0,p1s_cols,u0,u0+p1s_quads,pfs);
                    if(!(p1s_skip&2))d.launch3(d.bpo_p1b,p1s_cols,4u*p1s_quads,2u,256u,T,A,B,stage_tw,ctab,n,c0,p1s_cols,4u*u0,u32(off(4096)?1u:0u));}}
            else if(!(sm&2)){}else if(pox&&!off(32)&&!off(512)&&(p1k=(p1x==2&&d.bpo_first_xv&&ctab)?d.bpo_first_xv:(p1x==1&&d.bpo_first_x&&ctab)?d.bpo_first_x:(p1x==3&&d.bpo_first_w&&ctab)?d.bpo_first_w:(!off(4096)&&!off(1024)&&ctab&&d.bpo_first_tc2g&&!std::getenv("FHERMA_P1_CTSMEM"))?d.bpo_first_tc2g:d.bpo_first_tc2n)&&tc_bpt&&r0%4==0&&pg%4==0){if(p1k&&(p1k==d.bpo_first_x||p1k==d.bpo_first_xv)){const bool wl=(p1k==d.bpo_first_x||p1k==d.bpo_first_xv)&&ctab_wl&&tc_bpt_wl;const CUtensorMap ta=p1x_map(A),tb=p1x_map(B);const bool t1=p1tw_t1&&p1k==d.bpo_first_x;d.launch_ntt(t1?(p1rev_ok&&d.bpo_first_xr?d.bpo_first_xr:d.bpo_first_xt):p1k,cols*2u,1u,p1_threads(p1k),p1_pad(),a,b,A,B,t1?p1tw_t1:tw,stage_tw,wl?tc_bpt_wl:tc_bpt,n,logrows,r0,p1r1,u32(!off(2)&&!std::getenv("FHERMA_NO_L2PF")?(std::getenv("FHERMA_PF_STEP")?std::atoi(std::getenv("FHERMA_PF_STEP")):int(p1_grid)):0)|((gfl&1)?(1u<<29):(gfl&2)?(2u<<29):0u)|((gfl&4)&&p1r1<k?(1u<<28):0u)|(l2on&&(l2f&32)&&p1k==d.bpo_first_tc2g?0x80000000u:0u),(off(1024)||!ctab)?CUdeviceptr(0):(wl?ctab_wl:ctab),ta,tb);}else d.launch_ntt(p1k,cols*2u,1u,p1_threads(p1k),p1_pad(),a,b,A,B,tw,stage_tw,tc_bpt,n,logrows,r0,p1r1,u32(!off(2)&&!std::getenv("FHERMA_NO_L2PF")?(std::getenv("FHERMA_PF_STEP")?std::atoi(std::getenv("FHERMA_PF_STEP")):int(p1_grid)):0)|(l2on&&(l2f&32)&&p1k==d.bpo_first_tc2g?0x80000000u:0u),(off(1024)||!ctab)?CUdeviceptr(0):ctab);}
            else if(pox&&std::getenv("FHERMA_P1P")&&d.bpo_first_tcp&&tc_bpt&&r0%4==0&&pg%4==0)d.launch3(d.bpo_first_tcp,p1_grid,1u,1u,256u,a,b,A,B,tw,stage_tw,tc_bpt,n,logrows,r0,p1r1,p1_grid,2u*cols);
            else if(d.bpo_first_tc&&tc_bpt&&r0%4==0&&pg%4==0&&!std::getenv("FHERMA_PO_NOTC"))d.launch_ntt(pox&&!off(64)&&std::getenv("FHERMA_TC3")&&d.bpo_first_tc3?d.bpo_first_tc3:pox&&!off(32)&&!off(512)&&d.bpo_first_tc2n?d.bpo_first_tc2n:pox&&!off(32)&&d.bpo_first_tc2?d.bpo_first_tc2:d.bpo_first_tc,cols*2u,1u,256u,p1_pad(),a,b,A,B,tw,stage_tw,tc_bpt,n,logrows,r0,p1r1,u32(pox&&!off(2)&&!std::getenv("FHERMA_NO_L2PF")?(std::getenv("FHERMA_PF_STEP")?std::atoi(std::getenv("FHERMA_PF_STEP")):int(p1_grid)):0));
            else d.launch3(d.bpo_first,cols,2u,1u,256u,a,b,A,B,tw,stage_tw,n,logrows,r0,p1r1);mark(1);
            // Montgomery pointwise only on the smr P2 + cm3 P3 pair (default on; FHERMA_P2_NOMONT off; V7 twin FHERMA_V7_P2MONT)
            const bool mont=t2nM&&po==2&&p3v2()&&pox&&!off(128)&&(p1x==2?(std::getenv("FHERMA_V7_P2MONT")?std::atoi(std::getenv("FHERMA_V7_P2MONT"))!=0:!std::getenv("FHERMA_P2_NOMONT")):!std::getenv("FHERMA_P2_NOMONT"))&&(p1x==2?(d.bpo_middle_smrv&&d.bpo_last_cm3v):(d.bpo_middle_smr&&!off(256)&&d.bpo_last_cm3))&&!(pox&&std::getenv("FHERMA_P2DB"));
            fxon=mont&&p3crtx()&&p2r16()&&p3v2()&&!t2n_old()&&tc_bfrag&&d.bpo_crt_tc_cm;
            auto p23=[&](u32 s0,u32 gg,CUdeviceptr Cg,u32 which=3u){const CUdeviceptr Ag=A+size_t(s0)*J*n*4,Bg=B+size_t(s0)*J*n*4;
            const CUdeviceptr pgs_=p+size_t(s0)*4,mug=mu+size_t(s0)*8,itwg=itw+size_t(s0)*root_stride_host*8,invg=invn+size_t(s0)*J*8;
            if(!(which&1)||!(sm&8)){}else if(pox&&!off(16)&&d.bpo_middle_db&&std::getenv("FHERMA_P2DB"))d.launch3(d.bpo_middle_db,32u,J/2,gg,256u,Ag,Bg,Cg,pgs_,mug,itwg,stage_tw+size_t(s0)*256*8,stage_itw+size_t(s0)*256*8,n,logrows,u32(!(p3v2()))|(pox&&std::getenv("FHERMA_P2ABL")?2u:0u)|(l2on&&(l2f&1)?4u:0u)|(l2on&&(l2f&2)?8u:0u));
            else if(fxon)d.launch3(d.bpo_middle_r16x,8u,J,gg,512u,Ag,Bg,C+size_t(s0)*J*n*4,pgs_,r16tab+size_t(s0)*512*8,fxt+size_t(s0)*40*256*8,n,(l2on&&(l2f&1)?4u:0u)|16u);
            else if(p2r16()&&p3v2()&&mont)d.launch3(p2mt_r16x?(p2cu_ok&&d.bpo_middle_r16c?d.bpo_middle_r16c:d.bpo_middle_r16m):d.bpo_middle_r16,32u,J,gg,128u,Ag,Bg,Cg,pgs_,p2mt_r16x?p2mt_r16x+size_t(s0)*1024*8:r16tab+size_t(s0)*512*8,n,(l2on&&(l2f&1)?4u:0u)|(l2on&&(l2f&2)&&!std::getenv("FHERMA_P2_NODISCARD")?8u:0u)|(mont?16u:0u)|(p2mt_r16x&&p2cu_ok&&d.bpo_middle_r16c?u32(s0)<<24:0u));
            else d.launch3(p1x==2&&d.bpo_middle_smrv?d.bpo_middle_smrv:pox&&!off(128)&&d.bpo_middle_smr?d.bpo_middle_smr:d.bpo_middle,32u,J,gg,256u,Ag,Bg,Cg,pgs_,mug,itwg,stage_tw+size_t(s0)*256*8,stage_itw+size_t(s0)*256*8,n,logrows,u32(!(p3v2()))|(pox&&std::getenv("FHERMA_P2ABL")?2u:0u)|(l2on&&(l2f&1)?4u:0u)|(l2on&&(l2f&2)?8u:0u)|(mont?16u:0u));mark(3);
            if(!(which&2)||!(sm&16)||fxon){}else if(po==2&&p3v2()&&p3r16()&&!t2n_old())d.launch3(d.bpo_last_r16,cols,gg,1u,128u,Cg,Ag,pgs_,itwg,r16tab+size_t(s0)*512*8,(mont?t2nM:t2n)+size_t(s0)*J*512*8,n,logrows);
            else if(po==2&&p3v2()){const CUfunction p3k=p1x==2&&d.bpo_last_cm3v?d.bpo_last_cm3v:!off(256)&&d.bpo_last_cm3?d.bpo_last_cm3:d.bpo_last_cm2;
                const bool f3=p3f_ftab&&p3k==d.bpo_last_cm3&&n==65536&&logrows==8&&!(t2n_old()&&t2nO);
                d.launch3(f3?(p3t2_ok?d.bpo_last_cm3t:d.bpo_last_cm3f):p3k,f3&&p3t2_ok?cols/p3cb_env():cols,gg,1u,256u,Cg,Ag,pgs_,f3?p3f_ftab+size_t(s0)*65536*8:itwg,stage_itw+size_t(s0)*256*8,(t2n_old()&&t2nO?(mont?t2nMO:t2nO):(mont?t2nM:t2n))+size_t(s0)*J*512*8,n,logrows,u32(l2on&&(l2f&4)?1u:0u)|(t2n_old()&&t2nO?2u:0u)|((gfl&8)&&r0+pg<k?4u:0u)|((gfl&128)&&r0==0&&pg<k?8u:0u));}
            else if(po==2)d.launch3(d.bpo_last_cm,cols,gg,1u,256u,Cg,Ag,pgs_,itwg,stage_itw+size_t(s0)*256*8,invg,n);
            else d.launch3(d.bpo_last,64u/p3h,gg,1u,256u*p3h,Cg,Ag,pgs_,itwg,stage_itw+size_t(s0)*256*8,invg,n);mark(4);};
            if(g2<pg&&p23db_now&&wsG2[cur_ws]&&side_streams[cur_ws]){   // P3(g) on a side stream overlaps P2(g+1); C double-buffered
                static auto ecreate=d.driver.get<decltype(&::cuEventCreate)>("cuEventCreate");static auto erecord=d.driver.get<decltype(&::cuEventRecord)>("cuEventRecord");static auto swait=d.driver.get<decltype(&::cuStreamWaitEvent)>("cuStreamWaitEvent");
                const CUstream s=d.stream,s2=side_streams[cur_ws];const CUdeviceptr buf[2]={wsG[cur_ws],wsG2[cur_ws]};std::vector<CUevent> e3;u32 i=0;
                auto ev=[&](){CUevent e=nullptr;d.check(ecreate(&e,CU_EVENT_DISABLE_TIMING));cap_events.push_back(e);return e;};
                for(u32 s0=r0;s0<r0+pg;s0+=g2,++i){
                    if(i>=2)d.check(swait(s,e3[i-2],0));
                    d.stream=s;p23(s0,g2,buf[i&1],1u);
                    CUevent e2=ev();d.check(erecord(e2,s));d.check(swait(s2,e2,0));
                    d.stream=s2;p23(s0,g2,buf[i&1],2u);
                    CUevent e=ev();d.check(erecord(e,s2));e3.push_back(e);}
                d.stream=s;d.check(swait(s,e3.back(),0));}
            else if(g2<pg){for(u32 s0=r0;s0<r0+pg;s0+=g2)p23(s0,g2,wsG[cur_ws]);}else p23(r0,pg,Cg);
        }
        if(!(sm&32)){}else if(fxon)d.launch3(d.bpo_p3crt_x,8u,256u,1u,64u,C,out,tc_bfrag,r16tab,t2nM,n,delta);
        else if(po==2&&d.bpo_crt_tc_cm&&tc_bfrag&&!std::getenv("FHERMA_OLD_TCCRT")){const bool pf=pox&&next_a&&!off(1)&&!std::getenv("FHERMA_NO_CRTPF");d.launch(d.bpo_crt_lean&&!std::getenv("FHERMA_CRT_OLD")?d.bpo_crt_lean:d.bpo_crt_tc_cm,n/128,1,A,out,tc_bfrag,n,delta,pf?next_a:CUdeviceptr(0),pf?next_b:CUdeviceptr(0),pf?std::min(std::getenv("FHERMA_CRTPF_ITEMS")?u32(std::atoi(std::getenv("FHERMA_CRTPF_ITEMS"))):3u*p1_grid/4,2u*cols):0u,u32(l2on&&(l2f&8)?1u:0u)|u32(l2on&&(l2f&16)?2u:0u));}   // default (pv44 sweep x:crt=pf 0.983; items = 3*p1_grid/4 capped at 2 cols (pv61 whole-batch pf3q 0.994 vs p1_grid/2; FHERMA_CRTPF_ITEMS=n overrides): L2 prefetch of the next same-chain product's P1 input; none for the last product of each chain (next_a=0); FHERMA_NO_CRTPF=1 off
        else if(po==2)d.launch(d.bpo_crt_cm,n/128,1,A,out,p,mu,crt_matrix,corrections,qmod,bias,n,delta,u64(0),u64(0));
        else if(d.batched_crt_tc&&tc_bfrag&&!std::getenv("FHERMA_OLD_TCCRT"))d.launch(d.batched_crt_tc,n/128,1,A,out,tc_bfrag,n,delta);
        else d.launch(d.batched_crt,n/128,1,A,out,p,mu,crt_matrix,corrections,qmod,bias,n,delta,u64(0),u64(0));mark(5);
    }
    void compute_one(CUdeviceptr a,CUdeviceptr b,CUdeviceptr out){
        if(po){compute_po(a,b,out);return;}
        if(compute_grouped(a,b,out))return;
        auto& d=*cuda;mark(0);const u32 sm=stage_mask();
        u32 full=(n+127)/128;
        const bool tcf=use_tcfuse();
        const bool fused=tcf||use_prepfwd();
        if(tcf)launch_tcfuse(a,b,0,k,A,B);
        else if(fused){
            constexpr u32 J=8;
            d.launch3(d.batched_prepfwd,cols/8,J,2u,256u,a,b,A,B,p,tw,n,logrows,u64(n)*l,u64(k)*J*n);
        }else if(d.batched_prepare2&&n==65536&&!std::getenv("FHERMA_OLD_PREPARE")&&!std::getenv("FHERMA_PREPARE_V1")){
            constexpr u32 J=8;
            if(sm&2)launch_prepare(a,b,0,k);
        }else if(weighted&&digit_terms<=16&&!std::getenv("FHERMA_OLD_PREPARE")){
            constexpr u32 J=8;
            d.launch(d.batched_prepare_fused,full,1,a,b,A,B,p,mu,weighted_twist,tw,weights,positions,n,l,w,digit_terms,k,u64(n)*l,u64(k)*J*n);
        }else{
            d.launch256(d.transpose_inputs,(l+31)/32,(n+31)/32,a,b,at,bt,n,l);
            if(weighted){constexpr u32 J=8;d.launch(d.weighted_prepare,full,k*J,at,bt,A,B,p,mu,weighted_twist,tw,weights,positions,n,l,w,digit_terms);}
        }
        mark(1);
        if(weighted){
            constexpr u32 J=8;
            if(fused||!(sm&4)){}else if(use_bouter())d.launch_ntt(d.batched_forward,cols/8,k*J,256,0,A,B,p,tw,n,logrows);else d.launch_ntt(d.weighted_forward,cols/group,k*J,ntt_threads,8u*rows*group,A,B,p,mu,tw,n,rows,cols,group,logrows,J);
            mark(2);
            if(!(sm&8)){}else if(use_bmiddle())d.launch_ntt(d.batched_middle,rows,k,256,0,A,B,C,p,mu,tw,itw,n,rows,cols,logrows,J);else d.launch_ntt(d.weighted_middle,rows,k,weighted_middle_threads,8u*J*cols,A,B,C,p,mu,tw,itw,n,rows,cols,logrows,J);mark(3);
            if(!(sm&16)){}else if(use_bouter())d.launch_ntt(d.batched_inverse,cols/8,k*J,256,0,C,p,itw,invn,n);else d.launch_ntt(d.weighted_inverse,cols/group,k*J,ntt_threads,4u*rows*group,C,p,mu,itw,invn,n,rows,cols,group,J);mark(4);
            if(!(sm&32)){}else if(d.batched_crt&&n%128==0&&!std::getenv("FHERMA_OLD_CRT"))
                launch_crt(out);
            else if(n==65536&&w==1747&&l==55&&k==16&&short_words>=1&&short_words<=16)
                d.launch(d.reconstruct_weighted8_subwarp16,n,1,C,out,p,mu,crt_inverses,crt_matrix,corrections,positions,qmod,bias,n,k,w,l,short_words,delta);
            else
                d.launch256(d.reconstruct_weighted8,n,1,C,out,p,mu,crt_inverses,crt_matrix,corrections,positions,qmod,bias,n,k,w,l,short_words,delta);
            mark(5);
        }else{
            d.launch(d.prepare,full,k,at,bt,A,B,p,mu,tw,weights,n,l,logn);
            d.launch_ntt(d.fourstep_forward,cols/group,k,ntt_threads,8u*rows*group,A,B,p,mu,tw,n,rows,cols,group,logrows);
            d.launch_ntt(d.fourstep_middle,rows,k,ntt_threads,8u*cols,A,B,C,p,mu,tw,itw,n,rows,cols,logrows);
            d.launch_ntt(d.fourstep_inverse,cols/group,k,ntt_threads,4u*rows*group,C,p,mu,itw,invn,n,rows,cols,group);
            if(w>=128)d.launch(d.reconstruct_explicit,(n+3)/4,1,C,out,p,mu,crt_inverses,crt_matrix,neg_pmodq,qmod,n,k,w,l,delta);
            else {
                d.launch(d.reconstruct,full,1,C,outt,p,mu,prefix,lengths,inverses,midpoint,qmod,pmodq,n,k,w,l,delta);
                d.launch256(d.transpose_output,(n+31)/32,(l+31)/32,outt,out,n,l);
            }
        }
    }
    // Product chains in the graph (FHERMA_PO_CHAINS overrides): 2 on <=64-SM parts (5070: 2 chains x g=4 beat
    // 4 x g=2, 68.1 vs 71.5 ms A/B), 4 otherwise (unchanged).
    // P3/P2 overlap through a double-buffered C workspace (FHERMA_P23DB=1; A/B twin FHERMA_V7_P23DB)
    static bool p23db_env(bool v7){const char* e=std::getenv(v7?"FHERMA_V7_P23DB":"FHERMA_P23DB");return v7&&!e?p23db_env(false):e&&*e=='1';}
    // Large parts (>=128 SMs or >=96 MB L2, e.g. RTX PRO 6000 Blackwell: 188 SMs, 128 MB L2; FHERMA_BIG=0/1 forces):
    // BIG_CHAINS product chains, P1 per BIG_GROUP primes (P2 consumes each group's A,B from L2), group flags BIG_GFLAGS
    // (FHERMA_GFLAGS overrides): 1 plain P1 stores (no evict_first), 2 P1 stores evict_last, 4 P1 input loads normal on
    // all but the last group (input kept in L2 for the re-read), 8 P3 residues of all but the last group stored evict_first
    // (the CRT reads them only after the last group; the last group's residues stay normal so the CRT can hit them in L2).
    static constexpr u32 BIG_CHAINS=2,BIG_GROUP=8,BIG_GFLAGS=1|4|128;   // 128 (default, pv61 whole-batch p3el 0.986 / combo2 p3el+pf3q 0.979 on target): group-0 P3 residue stores evict_last (the CRT reads group 0 from L2; adds BPO_P3EL to the module); FHERMA_NO_P3EL=1 off   // 4: group-0 input loads keep L2 normal priority (re-read by group 1)
    int big_cache=-1;
    bool big_part(){if(big_cache>=0)return big_cache!=0;if(const char* e=std::getenv("FHERMA_BIG")){big_cache=*e=='1';return big_cache!=0;}
        int sms=0,l2=0;CUdevice dev=0;auto& d=*cuda;auto attr=d.driver.get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS){attr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);attr(&l2,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,dev);}
        big_cache=(sms>=128||l2>=(96<<20))?1:0;return big_cache!=0;}
    u32 group_flags(){if(const char* e=std::getenv("FHERMA_GFLAGS"))return u32(std::strtoul(e,nullptr,0));return std::getenv("FHERMA_NO_P3EL")?BIG_GFLAGS&~128u:BIG_GFLAGS;}
    u32 default_chains(){if(const char* e=std::getenv("FHERMA_PO_CHAINS"))return u32(std::atoi(e));if(big_part())return BIG_CHAINS;int v=0;CUdevice dev=0;auto& d=*cuda;
        auto attr=d.driver.get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
        if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS&&attr(&v,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev)==CUDA_SUCCESS&&v>0&&v<=64)return 2u;return 4u;}
    CUgraphExec graph_exec=nullptr,graph_execs[8]{};CUstream chain_streams[4]{};CUstream graph_stream=nullptr;
    // Whole batch as one CUDA graph (pointers are fixed at init): one launch per run.
    // Candidate 3 tables (once, before the graph capture, outside run; device-side init kernels, no host copies). Any failed
    // precondition / allocation / launch leaves that lever's pointer 0: its launch sites keep the production kernel.
    // FHERMA_TABS_FAIL=mask (test only): forced failure, 1 p1tw, 2 p2mt, 4 p3f, 8 p2cu, 16 p1rev, 32 p3t2.
    bool tabs_done=false,tabs_any=false;double tabs_ms=0;std::string tabs_why;
    void build_tabs(){
        if(tabs_done||!cuda){return;}tabs_done=true;auto& d=*cuda;
        tabs_any=d.bpo_first_xt||d.bpo_first_xr||d.bpo_middle_r16m||d.bpo_middle_r16c||d.bpo_last_cm3f;if(!tabs_any){return;}
        const auto t0=std::chrono::steady_clock::now();
        const char* fe=std::getenv("FHERMA_TABS_FAIL");const unsigned long fail=fe?std::strtoul(fe,nullptr,0):0ul;
        auto why=[&](const char* x){if(!tabs_why.empty())tabs_why+=' ';tabs_why+=x;};
        auto make=[&](CUfunction f,u32 gx,u32 gy,u32 bx,size_t words,CUdeviceptr src,bool two,const char* nm)->CUdeviceptr{
            CUdeviceptr o=0;
            try{
                if(d.cuMemAlloc(&o,words*8)!=CUDA_SUCCESS){o=0;why(nm);return 0;}
                CUdeviceptr x=src,y=p,z=o;void* p3[]={&x,&y,&z};void* p2[]={&x,&z};
                if(d.cuLaunchKernel(f,gx,gy,1,bx,1,1,0,nullptr,two?p2:p3,nullptr)!=CUDA_SUCCESS||d.cuCtxSynchronize()!=CUDA_SUCCESS){d.cuMemFree(o);why(nm);return 0;}
                d.buffers.push_back(o);return o;
            }catch(...){if(o)d.cuMemFree(o);why(nm);return 0;}};
        if(d.bpo_first_xt){if(!d.bpo_t1_init||!tw||root_stride_host!=641||k%4||!k||(fail&1))why("p1tw:pre");else p1tw_t1=make(d.bpo_t1_init,k/4,1,512,size_t(k/4)*2*256*4,tw,true,"p1tw:cuda");}
        if(d.bpo_middle_r16m){if(!d.bpo_r16x_init||!r16tab||n!=65536||(fail&2))why("p2mt:pre");else p2mt_r16x=make(d.bpo_r16x_init,k,1,1024,size_t(k)*1024,r16tab,true,"p2mt:cuda");}
        if(d.bpo_first_xr){if(!p1tw_t1||(fail&16))why("p1rev:pre");else p1rev_ok=true;}   // p1rev: no table; needs the p1tw launch
        if(d.bpo_middle_r16c){   // p2cu (after p2mt): bpo_p2cu[r][0,128) = r16tab[r][0,128), [r][128,256) = r16tab[r][256,384)
            try{
                auto gg=d.driver.get<decltype(&::cuModuleGetGlobal)>("cuModuleGetGlobal_v2");auto cp=d.driver.get<decltype(&::cuMemcpyDtoD)>("cuMemcpyDtoD_v2");
                CUdeviceptr cb=0;size_t cs=0;
                if(!gg||!cp||!p2mt_r16x||!r16tab||n!=65536||!k||(fail&8)||gg(&cb,&cs,d.module,"bpo_p2cu")!=CUDA_SUCCESS||cs!=size_t(k)*256*8)why("p2cu:pre");
                else{bool ok=true;
                    for(u32 r=0;r<k&&ok;++r)ok=cp(cb+size_t(r)*256*8,r16tab+size_t(r)*512*8,128*8)==CUDA_SUCCESS&&cp(cb+(size_t(r)*256+128)*8,r16tab+(size_t(r)*512+256)*8,128*8)==CUDA_SUCCESS;
                    if(ok&&d.cuCtxSynchronize()==CUDA_SUCCESS)p2cu_ok=true;else why("p2cu:cuda");}
            }catch(...){p2cu_ok=false;why("p2cu:cuda");}}
        if(d.bpo_last_cm3f){if(!d.bpo_ftab_init||!itw||!p||n!=65536||(fail&4))why("p3f:pre");else p3f_ftab=make(d.bpo_ftab_init,256,k,256,size_t(k)*65536,itw,false,"p3f:cuda");}
        if(d.bpo_last_cm3t){if(!p3f_ftab||!d.bpo_last_cm3f||n!=65536||(n/256)%p3cb_env()||(fail&32))why("p3t2:pre");else p3t2_ok=true;}   // p3t2: needs the p3f launch; grid cols / P3CB
        tabs_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();}
    void build_graph(){
        build_tabs();ld3_init();
        auto& d=*cuda;
        // L2 schedule unit (default on for the pass-order point; FHERMA_L2G=0 off, =g forces g, =k hints only):
        // one P2+P3 unit = (product, g primes) whose C workspace is g*8*N words per chain; g is the largest
        // power of two < k with chains*g*8*N*4 <= L2/2 (5070 48 MB: g=2, 4x4 MB; RTX PRO 6000 128 MB: g=8, 4x16 MB).
        {const u32 ch=std::max(1u,std::min(std::min(default_chains(),4u),bcount));
         u32 l2=l2_bytes;{int v=0;CUdevice dev=0;auto attr=d.driver.get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
          if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS&&attr(&v,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,dev)==CUDA_SUCCESS&&v>0)l2=u32(v);}
         const char* e=std::getenv("FHERMA_L2G");
         if(e){const u32 g=u32(std::atoi(e));if(g&&g<=k&&k%g==0)l2g=g;}
         else if(n==65536&&k==16){l2g=k;for(u32 g=k/2;g>=1;g/=2)if(size_t(ch)*g*8*n*4<=size_t(l2)/2){l2g=g;break;}}
         if(lean&&std::getenv("FHERMA_L2OFF"))l2g=0;
         if(std::getenv("FHERMA_INIT_TIMING"))std::fprintf(stderr,"{\"l2_bytes\":%u,\"l2g\":%u,\"chains\":%u}\n",l2,l2g,ch);}
        // A/B twin overrides for graph variant 7 (bench only): FHERMA_V7_CHAINS, FHERMA_V7_L2G
        const u32 l2g6=l2g;u32 l2g7=l2g;if(const char* e7=std::getenv("FHERMA_V7_L2G")){const u32 g=u32(std::atoi(e7));if(g&&g<=k&&k%g==0)l2g7=g;}
        {const u32 gm=std::max(l2g6<k?l2g6:0u,l2g7<k?l2g7:0u);if(gm)for(u32 c=0;c<std::min(4u,bcount);++c){wsG[c]=d.alloc(size_t(gm)*8*n*4);if(p23db_env(false)||p23db_env(true))wsG2[c]=d.alloc(size_t(gm)*8*n*4);}}
        if(prof||std::getenv("FHERMA_NO_GRAPH")){l2on=l2g&&!std::getenv("FHERMA_L2OFF");pox=true;p1x=p1x_default();return;}
        auto screate=d.driver.get<decltype(&::cuStreamCreate)>("cuStreamCreate");
        auto begin=d.driver.get<decltype(&::cuStreamBeginCapture)>("cuStreamBeginCapture_v2");
        auto endc=d.driver.get<decltype(&::cuStreamEndCapture)>("cuStreamEndCapture");
        auto inst=d.driver.get<decltype(&::cuGraphInstantiateWithFlags)>("cuGraphInstantiateWithFlags");
        auto gdestroy=d.driver.get<decltype(&::cuGraphDestroy)>("cuGraphDestroy");
        d.check(screate(&graph_stream,CU_STREAM_NON_BLOCKING));
        d.stream=graph_stream;
        const size_t stride=size_t(n)*l*4;
        auto ecreate=d.driver.get<decltype(&::cuEventCreate)>("cuEventCreate");
        auto erecord=d.driver.get<decltype(&::cuEventRecord)>("cuEventRecord");
        auto swait=d.driver.get<decltype(&::cuStreamWaitEvent)>("cuStreamWaitEvent");
        // Variant 3: pass-order column-major with independent products on parallel chains
        // (FHERMA_PO_CHAINS, default 2), each chain with its own A,B,C workspace.
        u32 chains=default_chains();chains=std::max(1u,std::min(chains,4u));
        {int v=0;CUdevice dev=0;auto attr=d.driver.get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");
         if(d.cuDeviceGet(&dev,0)==CUDA_SUCCESS&&attr(&v,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev)==CUDA_SUCCESS&&v>0)p1_grid=2u*u32(v);
         if(const char* e=std::getenv("FHERMA_P1_GRID"))p1_grid=u32(std::atoi(e));}
        if(const char* e=std::getenv("FHERMA_P1S_COLS")){const u32 v=u32(std::atoi(e));if(v&&v<=cols&&cols%v==0)p1s_cols=v;}
        if(const char* e=std::getenv("FHERMA_P1S_QUADS")){const u32 v=u32(std::atoi(e));if(v&&v<=k/4&&(k/4)%v==0)p1s_quads=v;}
        if(cuda->bpo_p1a&&cuda->bpo_p1b)for(u32 c=0;c<std::min(chains,bcount);++c)wsT[c]=d.alloc(size_t(2)*4*p1s_quads*8*p1s_cols*256*4);
        for(int variant=0;variant<(po_available()?8:1);++variant){
            l2g=variant==7?l2g7:l2g6;
            p23db_now=p23db_env(variant==7);
            {static const u32 l2f6=l2f;l2f=variant==7&&std::getenv("FHERMA_V7_L2F")?u32(std::strtoul(std::getenv("FHERMA_V7_L2F"),nullptr,0)):l2f6;}
            if(variant==6&&!l2g)continue;   // variant 6 = variant 4 + L2 schedule
            if(variant==7&&(lean||!cuda->bpo_first_xv))continue;   // variant 7 = default graph with bpo_first_xv (A/B: FHERMA_P1X)
            l2on=variant==6||(variant==7&&l2g);const int vb=variant>=6?4:variant;p1x=variant==7?2u:p1x_default();
            po=vb>=2?2:vb;pox=vb>=4;xoff=variant==5&&std::getenv("FHERMA_V5_OFF")?u32(std::atoi(std::getenv("FHERMA_V5_OFF"))):0u;
            if(lean&&(vb!=4||(variant==4&&l2g)))continue;   // lean: only the default graph (6 when the L2 schedule is on)
            if(!lean&&variant>=2&&!(cuda->bpo_last_cm&&cuda->bpo_crt_cm))continue;
            const u32 ch7=std::getenv("FHERMA_V7_CHAINS")?std::max(1u,std::min(4u,u32(std::atoi(std::getenv("FHERMA_V7_CHAINS"))))):chains;
            const u32 S=vb>=3?std::min(variant==7?ch7:chains,bcount):1u;
            for(u32 c=1;c<S;++c)if(!wsA[c]){wsA[c]=d.alloc(size_t(k)*8*n*4);wsB[c]=d.alloc(size_t(k)*8*n*4);wsC[c]=d.alloc(size_t(k)*8*n*4);}
            CUstream cs[4]{graph_stream};CUevent fork=nullptr,join[4]{};
            for(u32 c=1;c<S;++c){if(!chain_streams[c])d.check(screate(&chain_streams[c],CU_STREAM_NON_BLOCKING));cs[c]=chain_streams[c];}
            if(p23db_now)for(u32 c=0;c<S;++c)if(!side_streams[c])d.check(screate(&side_streams[c],CU_STREAM_NON_BLOCKING));
            d.check(begin(graph_stream,CU_STREAM_CAPTURE_MODE_RELAXED));d.kp.begin_variant(variant);
            if(variant==6)ld3_node(false);   // ld3 head stamp
            if(S>1){d.check(ecreate(&fork,CU_EVENT_DISABLE_TIMING));d.check(erecord(fork,graph_stream));for(u32 c=1;c<S;++c)d.check(swait(cs[c],fork,0));}
            for(u32 i=0;i<bcount;++i){d.stream=cs[i%S];cur_ws=i%S;d.kp.product=int(i);d.kp.chain=int(i%S);next_a=i+S<bcount?a_all+(i+S)*stride:0;next_b=i+S<bcount?b_all+(i+S)*stride:0;compute_one(a_all+i*stride,b_all+i*stride,out_all+i*stride);}
            d.stream=graph_stream;cur_ws=0;next_a=next_b=0;
            for(u32 c=1;c<S;++c){d.check(ecreate(&join[c],CU_EVENT_DISABLE_TIMING));d.check(erecord(join[c],cs[c]));d.check(swait(graph_stream,join[c],0));}
            if(variant==6)ld3_node(true);   // ld3 tail stamp + slot bump
            d.kp.end_variant();CUgraph g=nullptr;d.check(endc(graph_stream,&g));
            d.check(inst(&graph_execs[variant],g,0));if(variant==6&&ld3_on)ld3_graph=g;else d.check(gdestroy(g));   // ld3 keeps the graph (re-instantiation line)
        }
        l2g=l2g6;p23db_now=false;po=0;pox=false;p1x=0;xoff=0;l2on=false;graph_exec=graph_execs[0]?graph_execs[0]:graph_execs[4]?graph_execs[4]:graph_execs[6];
    }
    // FHERMA_PO=1 selects the pass-order pipeline (default off until it wins A/B).
    // FHERMA_POCM=1 -> variant 2, FHERMA_PO=1 -> variant 1, else FHERMA_PO_DEFAULT (A/B: BENCH_AB toggles one of them).
    int want_po()const{const char* e=std::getenv("FHERMA_POY");if(e&&*e=='1')return 5;e=std::getenv("FHERMA_POX");if(e&&*e=='1')return 4;e=std::getenv("FHERMA_POCH");if(e&&*e=='1')return 3;e=std::getenv("FHERMA_POCM");if(e&&*e=='1')return 2;e=std::getenv("FHERMA_PO");if(e&&*e=='1')return 1;
        e=std::getenv("FHERMA_PO_DEFAULT");if(e)return std::atoi(e);
        return std::getenv("FHERMA_V14")?0:4;}   // default: pass-order, column-major, chains, P2->P3 untwiddle, P1 L2 prefetch
    // ---- launch diag ld1 (on 6a3f4ed = candidate 8; diag build only): how the production work is SUBMITTED, at exit
    // on the real batch. Same kernels, same per-chain order, same chains/workspaces as graph variant 6; only the
    // number of graphs, which streams/threads launch them, the context sync mode and the wait call change.
    std::vector<CUevent> ld_events;
    CUevent ld_ev(bool timing){auto& D=cuda->driver;static auto ecreate=D.get<decltype(&::cuEventCreate)>("cuEventCreate");
        CUevent e=nullptr;cuda->check(ecreate(&e,timing?0u:unsigned(CU_EVENT_DISABLE_TIMING)));ld_events.push_back(e);return e;}
    u32 ld_chains(){return std::min(std::max(1u,std::min(default_chains(),4u)),bcount);}
    void ld_state(bool on){
        if(on){l2on=true;pox=true;po=2;p1x=p1x_default();xoff=0;p23db_now=p23db_env(false);}
        else{l2on=false;pox=false;po=0;p1x=0;xoff=0;p23db_now=false;cur_ws=0;next_a=next_b=0;cuda->stream=graph_stream;}
    }
    void ld_one(u32 i,u32 S,const CUstream* cs){const size_t stride=size_t(n)*l*4;auto& d=*cuda;
        d.stream=cs[i%S];cur_ws=i%S;next_a=i+S<bcount?a_all+(i+S)*stride:0;next_b=i+S<bcount?b_all+(i+S)*stride:0;
        compute_one(a_all+i*stride,b_all+i*stride,out_all+i*stride);}
    // products [i0,i1) of the variant-6 schedule; only>=0: just chain `only`'s products, captured on that chain's stream
    CUgraphExec ld_cap(u32 i0,u32 i1,int only,CUevent head=nullptr,CUevent tail=nullptr){
        auto& d=*cuda;auto& D=d.driver;
        static auto begin=D.get<decltype(&::cuStreamBeginCapture)>("cuStreamBeginCapture_v2");
        static auto endc=D.get<decltype(&::cuStreamEndCapture)>("cuStreamEndCapture");
        static auto inst=D.get<decltype(&::cuGraphInstantiateWithFlags)>("cuGraphInstantiateWithFlags");
        static auto gdestroy=D.get<decltype(&::cuGraphDestroy)>("cuGraphDestroy");
        static auto erecord=D.get<decltype(&::cuEventRecord)>("cuEventRecord");
        static auto swait=D.get<decltype(&::cuStreamWaitEvent)>("cuStreamWaitEvent");
        static auto erecf=D.get<decltype(&::cuEventRecordWithFlags)>("cuEventRecordWithFlags");   // external record nodes (timing)
        const u32 S=ld_chains();CUstream cs[4]{graph_stream};for(u32 c=1;c<S;++c)cs[c]=chain_streams[c];
        const CUstream cap=only>=0?cs[only]:graph_stream;CUgraphExec ge=nullptr;bool capt=false;
        ld_state(true);
        try{
            d.stream=cap;d.check(begin(cap,CU_STREAM_CAPTURE_MODE_RELAXED));capt=true;
            if(head)d.check(erecf(head,cap,CU_EVENT_RECORD_EXTERNAL));
            if(only<0&&S>1){CUevent f=ld_ev(false);d.check(erecord(f,graph_stream));for(u32 c=1;c<S;++c)d.check(swait(cs[c],f,0));}
            for(u32 i=i0;i<i1;++i){if(only>=0&&int(i%S)!=only)continue;ld_one(i,S,cs);}
            d.stream=cap;
            if(only<0)for(u32 c=1;c<S;++c){CUevent j=ld_ev(false);d.check(erecord(j,cs[c]));d.check(swait(graph_stream,j,0));}
            if(tail)d.check(erecf(tail,cap,CU_EVENT_RECORD_EXTERNAL));
            CUgraph g=nullptr;capt=false;d.check(endc(cap,&g));const CUresult r=inst(&ge,g,0);gdestroy(g);d.check(r);
        }catch(...){if(capt){CUgraph g=nullptr;endc(cap,&g);if(g)gdestroy(g);}ge=nullptr;}
        ld_state(false);return ge;
    }
    // two parked launcher threads (created once for the diag), each launches its own (graph, stream) job
    struct LdCrew {
        std::vector<std::thread> th;std::mutex m;std::condition_variable go,fin;uint64_t gen=0;unsigned left=0;bool quit=false;
        std::vector<std::pair<CUgraphExec,CUstream>> jobs;CUresult err=CUDA_SUCCESS;
        decltype(&::cuGraphLaunch) gl=nullptr;
        void start(unsigned nt,CUcontext ctx,decltype(&::cuCtxSetCurrent) setcur,decltype(&::cuGraphLaunch) g){gl=g;
            for(unsigned t=0;t<nt;++t)th.emplace_back([=,this]{setcur(ctx);uint64_t seen=0;for(;;){{std::unique_lock<std::mutex> lk(m);go.wait(lk,[&]{return quit||gen!=seen;});if(quit)return;seen=gen;}
                CUresult r=t<jobs.size()?gl(jobs[t].first,jobs[t].second):CUDA_SUCCESS;{std::lock_guard<std::mutex> lk(m);if(r!=CUDA_SUCCESS)err=r;if(--left==0)fin.notify_one();}}});}
        void launch(){{std::lock_guard<std::mutex> lk(m);left=unsigned(th.size());++gen;}go.notify_all();std::unique_lock<std::mutex> lk(m);fin.wait(lk,[&]{return left==0;});}
        ~LdCrew(){{std::lock_guard<std::mutex> lk(m);quit=true;}go.notify_all();for(auto& t:th)t.join();}
    };
    struct LdSub { std::vector<std::pair<CUgraphExec,CUstream>> g; u32 direct=0; bool thr=false; int wait=0; CUevent pre=nullptr,head=nullptr,tail=nullptr; };
    double ld_once(const LdSub& s,LdCrew* crew,double* lau){
        auto& d=*cuda;auto& D=d.driver;
        static auto glaunch=D.get<decltype(&::cuGraphLaunch)>("cuGraphLaunch");static auto erecord=D.get<decltype(&::cuEventRecord)>("cuEventRecord");
        static auto esync=D.get<decltype(&::cuEventSynchronize)>("cuEventSynchronize");static auto sq=D.get<decltype(&::cuStreamQuery)>("cuStreamQuery");
        static auto swait=D.get<decltype(&::cuStreamWaitEvent)>("cuStreamWaitEvent");
        static CUevent fin_ev=ld_ev(false);
        const auto t0=std::chrono::steady_clock::now();
        if(s.pre)d.check(erecord(s.pre,graph_stream));
        if(s.direct){const u32 S=ld_chains();CUstream cs[4]{graph_stream};for(u32 c=1;c<S;++c)cs[c]=chain_streams[c];
            ld_state(true);for(u32 i=0;i<s.direct;++i)ld_one(i,S,cs);ld_state(false);
            for(u32 c=1;c<S;++c){static CUevent dj[4]{};if(!dj[c])dj[c]=ld_ev(false);d.check(erecord(dj[c],cs[c]));d.check(swait(graph_stream,dj[c],0));}}
        if(s.thr&&crew){crew->jobs=s.g;crew->launch();d.check(crew->err);}
        else for(const auto& x:s.g)d.check(glaunch(x.first,x.second));
        const auto t1=std::chrono::steady_clock::now();
        if(s.wait==0)d.check(d.cuCtxSynchronize());
        else{for(const auto& x:s.g)if(x.second!=graph_stream){static CUevent cj[4]{};const u32 c=x.second==chain_streams[1]?1:x.second==chain_streams[2]?2:3;if(!cj[c])cj[c]=ld_ev(false);d.check(erecord(cj[c],x.second));d.check(swait(graph_stream,cj[c],0));}
            d.check(erecord(fin_ev,graph_stream));
            if(s.wait==1)d.check(esync(fin_ev));else{CUresult r;while((r=sq(graph_stream))==CUDA_ERROR_NOT_READY){}d.check(r);}}
        const auto t2=std::chrono::steady_clock::now();
        *lau=std::chrono::duration<double,std::micro>(t1-t0).count();return std::chrono::duration<double,std::micro>(t2-t0).count();
    }
    void run_launch_diag(){
        if(!lever_env("FHERMA_LDIAG",false)||!graph_execs[6]||bcount<2)return;
        auto& d=*cuda;auto& D=d.driver;
        auto ged=D.get<decltype(&::cuGraphExecDestroy)>("cuGraphExecDestroy");auto eel=D.get<decltype(&::cuEventElapsedTime)>("cuEventElapsedTime_v2");if(!eel)eel=D.get<decltype(&::cuEventElapsedTime)>("cuEventElapsedTime");
        auto dtod=D.get<decltype(&::cuMemcpyDtoD)>("cuMemcpyDtoD_v2");auto mset=D.get<decltype(&::cuMemsetD8)>("cuMemsetD8_v2");
        typedef CUresult (*FlagsSet)(unsigned);typedef CUresult (*FlagsGet)(unsigned*);
        auto fset=reinterpret_cast<FlagsSet>(dlsym(D.handle,"cuCtxSetFlags"));auto fget=reinterpret_cast<FlagsGet>(dlsym(D.handle,"cuCtxGetFlags"));
        auto setcur=D.get<decltype(&::cuCtxSetCurrent)>("cuCtxSetCurrent");auto glaunch=D.get<decltype(&::cuGraphLaunch)>("cuGraphLaunch");
        const auto T0=std::chrono::steady_clock::now();auto el=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-T0).count();};
        const size_t bytes=size_t(bcount)*n*l*4;const u32 S=ld_chains(),B=bcount;
        const int R=std::getenv("FHERMA_LD_R")?std::max(3,std::atoi(std::getenv("FHERMA_LD_R"))):15;
        unsigned flags0=0;if(fget)fget(&flags0);
        std::printf("{\"diag\":\"ld\",\"k\":\"info\",\"B\":%u,\"chains\":%u,\"R\":%d,\"ctx_flags\":%u,\"setflags\":%d,\"hostt_launch_us\":%.1f,\"hostt_run_us\":%.1f}\n",
            B,S,R,flags0,fset?1:0,ht_run.med(1),ht_run.med(4));std::fflush(stdout);
        CUdeviceptr ref=0;try{ref=d.alloc(bytes);}catch(...){ref=0;}
        const CUgraphExec prod=graph_execs[6];LdSub P;P.g={{prod,graph_stream}};double lau=0;
        ld_once(P,nullptr,&lau);if(ref){d.check(dtod(ref,out_all,bytes));d.check(d.cuCtxSynchronize());}
        std::vector<char> h1(size_t(32)<<20),h2(size_t(32)<<20);
        auto exact=[&](const LdSub& s,LdCrew* crew)->int{if(!ref)return -1;double x;d.check(mset(out_all,0,bytes));d.check(d.cuCtxSynchronize());ld_once(s,crew,&x);d.check(d.cuCtxSynchronize());
            for(size_t off=0;off<bytes;off+=h1.size()){const size_t len=std::min(h1.size(),bytes-off);d.check(d.cuMemcpyDtoH(h1.data(),ref+off,len));d.check(d.cuMemcpyDtoH(h2.data(),out_all+off,len));if(std::memcmp(h1.data(),h2.data(),len))return 0;}
            return 1;};
        LdCrew crew2;if(setcur)crew2.start(2,d.context,setcur,glaunch);
        auto med=[](std::vector<double> v){std::sort(v.begin(),v.end());return v.empty()?0.0:v[v.size()/2];};
        auto line=[&](const char* nm,LdSub s,unsigned flags,bool evs){
            if(el()>240){std::printf("{\"diag\":\"ld\",\"v\":\"%s\",\"skipped\":\"time\"}\n",nm);return;}
            for(const auto& x:s.g)if(!x.first){std::printf("{\"diag\":\"ld\",\"v\":\"%s\",\"err\":\"capture\"}\n",nm);return;}
            bool fl=flags!=~0u;if(fl&&(!fset||fset(flags)!=CUDA_SUCCESS)){std::printf("{\"diag\":\"ld\",\"v\":\"%s\",\"err\":\"setflags\"}\n",nm);return;}
            std::vector<double> tp,tv,lp,lv,hd,bd,tl;double a=0,b2=0;
            for(int w=0;w<2;++w){ld_once(P,nullptr,&a);ld_once(s,&crew2,&a);}
            for(int r=0;r<R;++r){tp.push_back(ld_once(P,nullptr,&a));lp.push_back(a);const double t=ld_once(s,&crew2,&b2);tv.push_back(t);lv.push_back(b2);
                if(evs){float x=0,y=0;if(eel(&x,s.pre,s.head)==CUDA_SUCCESS&&eel(&y,s.head,s.tail)==CUDA_SUCCESS){hd.push_back(1e3*x);bd.push_back(1e3*y);tl.push_back(t-1e3*(x+y));}}}
            const int ex=exact(s,&crew2);
            if(fl)fset(flags0);
            const double mv=med(tv),mp=med(tp);
            if(evs)std::printf("{\"diag\":\"ld\",\"v\":\"%s\",\"us\":%.1f,\"prod_us\":%.1f,\"vs\":%.4f,\"ex\":%d,\"lau\":%.1f,\"lau_p\":%.1f,\"head_us\":%.1f,\"body_us\":%.1f,\"tail_us\":%.1f}\n",nm,mv,mp,mp>0?mv/mp:0,ex,med(lv),med(lp),med(hd),med(bd),med(tl));
            else std::printf("{\"diag\":\"ld\",\"v\":\"%s\",\"us\":%.1f,\"prod_us\":%.1f,\"vs\":%.4f,\"ex\":%d,\"lau\":%.1f,\"lau_p\":%.1f,\"n\":%zu}\n",nm,mv,mp,mp>0?mv/mp:0,ex,med(lv),med(lp),s.g.size());
            std::fflush(stdout);
            for(const auto& x:s.g)if(x.first!=prod)ged(x.first);};
        auto split=[&](u32 parts){LdSub s;for(u32 j=0;j<parts;++j)s.g.push_back({ld_cap(j*B/parts,(j+1)*B/parts,-1),graph_stream});return s;};
        auto headg=[&](u32 h){LdSub s;s.g={{ld_cap(0,h,-1),graph_stream},{ld_cap(h,B,-1),graph_stream}};return s;};
        auto perch=[&](bool thr){LdSub s;s.thr=thr;for(u32 c=0;c<S;++c)s.g.push_back({ld_cap(0,B,int(c)),c?chain_streams[c]:graph_stream});return s;};
        auto direct=[&](u32 h){LdSub s;s.direct=h;s.g={{ld_cap(h,B,-1),graph_stream}};return s;};
        auto prodw=[&](int w){LdSub s=P;s.wait=w;return s;};
        {LdSub s=P;line("ctrl",s,~0u,false);}
        for(int pass=0;pass<2;++pass){char nm[32];auto N=[&](const char* b){std::snprintf(nm,sizeof nm,"%s%s",b,pass?"_r":"");return nm;};
            {LdSub s;s.g={{ld_cap(0,B,-1),graph_stream}};line(N("recap"),s,~0u,false);}                       // recaptured production graph: capture itself
            {LdSub s;s.pre=ld_ev(true);s.head=ld_ev(true);s.tail=ld_ev(true);s.g={{ld_cap(0,B,-1,s.head,s.tail),graph_stream}};line(N("ev"),s,~0u,true);}   // head/body/tail
            for(u32 p:{2u,4u,8u,16u,32u,64u,128u})if(p<=B){char b[16];std::snprintf(b,sizeof b,"split%u",p);line(N(b),split(p),~0u,false);}
            for(u32 h:{2u,4u})if(h<B){char b[16];std::snprintf(b,sizeof b,"head%u",h);line(N(b),headg(h),~0u,false);}
            if(S>1){line(N("perchain"),perch(false),~0u,false);if(setcur)line(N("perch_thr"),perch(true),~0u,false);}
            for(u32 h:{2u,4u})if(h<B){char b[16];std::snprintf(b,sizeof b,"direct%u",h);line(N(b),direct(h),~0u,false);}
            line(N("spin"),P,unsigned(CU_CTX_SCHED_SPIN),false);line(N("yield"),P,unsigned(CU_CTX_SCHED_YIELD),false);line(N("block"),P,unsigned(CU_CTX_SCHED_BLOCKING_SYNC),false);
            line(N("wait_ev"),prodw(1),~0u,false);line(N("wait_poll"),prodw(2),~0u,false);}
        {LdSub s=P;line("ctrl_end",s,~0u,false);}
        std::printf("{\"diag\":\"ld\",\"k\":\"done\",\"s\":%.1f}\n",el());std::fflush(stdout);
        (void)ref;   // freed with the other device buffers
        auto edes=D.get<decltype(&::cuEventDestroy)>("cuEventDestroy_v2");for(auto e:ld_events)edes(e);ld_events.clear();
    }
    // ---- ld3 (diag build on c89e973 = 6a3f4ed + ld1 helpers; nothing in fherma_h2d / fherma_d2h): the slow GPU-body state.
    // Graph variant 6 carries a head and a tail stamp kernel (one writer block per SM: %clock64 + %globaltimer into slot
    // *ctr) and a 1-thread bump of ctr: per launch, body = max tail time - min head time, effective SM clock = per-SM
    // cycles / ns over the whole body. run() takes NVML / host snapshots before the launch and after the synchronize.
    struct Ld3Nvml {
        void* h=nullptr;void* dev=nullptr;bool ok=false;size_t pstride=24;
        int (*init)()=nullptr;int (*byPci)(const char*,void**)=nullptr;int (*clock)(void*,int,unsigned*)=nullptr;int (*power)(void*,unsigned*)=nullptr;
        int (*temp)(void*,int,unsigned*)=nullptr;int (*pstate)(void*,int*)=nullptr;int (*reasons)(void*,unsigned long long*)=nullptr;
        int (*util)(void*,unsigned*)=nullptr;int (*procs)(void*,unsigned*,void*)=nullptr;int (*viol)(void*,int,unsigned long long*)=nullptr;
        template<class F> void sym(F& f,const char* a,const char* b=nullptr){void* p=dlsym(h,a);if(!p&&b)p=dlsym(h,b);f=reinterpret_cast<F>(p);}
        void open(const char* pci){h=dlopen("libnvidia-ml.so.1",RTLD_NOW);if(!h)return;
            sym(init,"nvmlInit_v2","nvmlInit");sym(byPci,"nvmlDeviceGetHandleByPciBusId_v2","nvmlDeviceGetHandleByPciBusId");sym(clock,"nvmlDeviceGetClockInfo");
            sym(power,"nvmlDeviceGetPowerUsage");sym(temp,"nvmlDeviceGetTemperature");sym(pstate,"nvmlDeviceGetPerformanceState");
            sym(reasons,"nvmlDeviceGetCurrentClocksEventReasons","nvmlDeviceGetCurrentClocksThrottleReasons");sym(util,"nvmlDeviceGetUtilizationRates");
            procs=reinterpret_cast<decltype(procs)>(dlsym(h,"nvmlDeviceGetComputeRunningProcesses_v3"));if(!procs){procs=reinterpret_cast<decltype(procs)>(dlsym(h,"nvmlDeviceGetComputeRunningProcesses_v2"));}
            if(!procs){procs=reinterpret_cast<decltype(procs)>(dlsym(h,"nvmlDeviceGetComputeRunningProcesses"));pstride=16;}
            sym(viol,"nvmlDeviceGetViolationStatus");
            if(!init||!byPci||init()!=0||byPci(pci,&dev)!=0)return;ok=true;}
    } ld3_nv;
    struct Ld3Snap { unsigned clk=0,pw=0,tmp=0,ut=0,np=0,no=0;int ps=-1;unsigned long long rs=0,pv=0,tv=0;double cost=0; };
    void ld3_snap(Ld3Snap& s,bool full){
        const auto t=std::chrono::steady_clock::now();auto& N=ld3_nv;
        if(N.ok){if(N.clock)N.clock(N.dev,1,&s.clk);if(N.reasons)N.reasons(N.dev,&s.rs);   // pre-launch: 2 calls only
            if(full){if(N.pstate)N.pstate(N.dev,&s.ps);unsigned long long v[2]{};if(N.viol&&N.viol(N.dev,0,v)==0)s.pv=v[1];if(N.viol&&N.viol(N.dev,1,v)==0)s.tv=v[1];if(N.power)N.power(N.dev,&s.pw);if(N.temp)N.temp(N.dev,0,&s.tmp);unsigned u[2]{};if(N.util&&N.util(N.dev,u)==0)s.ut=u[0];
                if(N.procs){alignas(8) unsigned char buf[64*24];unsigned cnt=64;if(N.procs(N.dev,&cnt,buf)==0){s.np=cnt;const unsigned me=unsigned(getpid());
                    for(unsigned i=0;i<cnt&&i<64;++i){unsigned pid;std::memcpy(&pid,buf+i*N.pstride,4);if(pid!=me)++s.no;}}}}}
        s.cost=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();}
    struct Ld3Rec { int cond=-1,slot=-1,cpu0=-1,cpu1=-1;double lau=0,host=0,load=0;long ivcsw=0;Ld3Snap a,b; };
    std::vector<Ld3Rec> ld3_recs;int ld3_slots=0;bool ld3_on=false,ld3_live=false;Ld3Rec ld3_cur;
    CUmodule ld3_mod=nullptr;CUfunction ld3_stamp=nullptr,ld3_bump=nullptr;CUdeviceptr ld3_buf=0,ld3_flag=0,ld3_ctr=0;CUgraph ld3_graph=nullptr;
    static constexpr int LD3_SLOTS=160;
    void ld3_init(){
        if(ld3_mod||!lever_env("FHERMA_LD3",true))return;auto& d=*cuda;
        const char* src="extern \"C\" __global__ void ld3_stamp(unsigned long long* buf,unsigned* flag,const unsigned* ctr,unsigned which){\n"
            "  if(threadIdx.x)return;unsigned sm;asm volatile(\"mov.u32 %0, %%smid;\":\"=r\"(sm));if(sm>=256)return;const unsigned slot=*ctr;if(slot>=160)return;\n"
            "  const unsigned i=(slot*2+which)*256+sm;if(atomicCAS(&flag[i],0u,1u)!=0u)return;unsigned long long g;\n"
            "  asm volatile(\"mov.u64 %0, %%globaltimer;\":\"=l\"(g));buf[2*i]=clock64();buf[2*i+1]=g;}\n"
            "extern \"C\" __global__ void ld3_bump(unsigned* ctr){*ctr+=1u;}\n";
        nvrtcProgram p=nullptr;if(d.nvrtcCreateProgram(&p,src,"ld3.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)return;
        const char* opt[]={"--gpu-architecture=compute_80"};
        if(d.nvrtcCompileProgram(p,1,opt)!=NVRTC_SUCCESS){d.nvrtcDestroyProgram(&p);return;}
        size_t sz=0;d.nvrtcGetPTXSize(p,&sz);std::vector<char> ptx(sz);d.nvrtcGetPTX(p,ptx.data());d.nvrtcDestroyProgram(&p);
        if(d.cuModuleLoadData(&ld3_mod,ptx.data())!=CUDA_SUCCESS){ld3_mod=nullptr;return;}
        d.check(d.cuModuleGetFunction(&ld3_stamp,ld3_mod,"ld3_stamp"));d.check(d.cuModuleGetFunction(&ld3_bump,ld3_mod,"ld3_bump"));
        ld3_buf=d.alloc(size_t(LD3_SLOTS)*2*256*16);ld3_flag=d.alloc(size_t(LD3_SLOTS)*2*256*4);ld3_ctr=d.alloc(256);
        auto ms=d.driver.get<decltype(&::cuMemsetD8)>("cuMemsetD8_v2");d.check(ms(ld3_flag,0,size_t(LD3_SLOTS)*2*256*4));d.check(ms(ld3_ctr,0,256));d.check(ms(ld3_buf,0,size_t(LD3_SLOTS)*2*256*16));
        char pci[32]{};CUdevice dev=0;d.cuDeviceGet(&dev,0);auto gp=d.driver.get<decltype(&::cuDeviceGetPCIBusId)>("cuDeviceGetPCIBusId");gp(pci,32,dev);ld3_nv.open(pci);
        ld3_recs.reserve(256);ld3_on=true;}
    void ld3_node(bool tail){   // inside the variant-6 capture, on graph_stream
        if(!ld3_on)return;auto& d=*cuda;int sms=0;CUdevice dev=0;d.cuDeviceGet(&dev,0);
        auto attr=d.driver.get<decltype(&::cuDeviceGetAttribute)>("cuDeviceGetAttribute");attr(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);
        unsigned which=tail?1u:0u;void* a[]={&ld3_buf,&ld3_flag,&ld3_ctr,&which};
        d.check(d.cuLaunchKernel(ld3_stamp,unsigned(std::max(1,sms)*4),1,1,32,1,1,0,graph_stream,a,nullptr));
        if(tail){void* b[]={&ld3_ctr};d.check(d.cuLaunchKernel(ld3_bump,1,1,1,1,1,1,0,graph_stream,b,nullptr));}}
    struct Ld3Host { int cpu=-1;long iv=0; };
    static Ld3Host ld3_host(){Ld3Host h;h.cpu=sched_getcpu();struct rusage u{};if(getrusage(RUSAGE_THREAD,&u)==0)h.iv=u.ru_nivcsw;return h;}
    void ld3_rec_before(Ld3Rec& r){static const bool pre=lever_env("FHERMA_LD3_PRE",true);if(pre)ld3_snap(r.a,false);const Ld3Host h=ld3_host();r.cpu0=h.cpu;r.ivcsw=-h.iv;}
    void ld3_rec_after(Ld3Rec& r){const Ld3Host h=ld3_host();r.cpu1=h.cpu;r.ivcsw+=h.iv;ld3_snap(r.b,true);double la[1]{};if(getloadavg(la,1)==1)r.load=la[0];}
    // the exit loop: conditions around production launches (5 reps each)
    void run_ld3(){
        if(!ld3_on||!graph_execs[6])return;auto& d=*cuda;auto& D=d.driver;
        auto glaunch=D.get<decltype(&::cuGraphLaunch)>("cuGraphLaunch");auto ginst=D.get<decltype(&::cuGraphInstantiateWithFlags)>("cuGraphInstantiateWithFlags");
        auto ged=D.get<decltype(&::cuGraphExecDestroy)>("cuGraphExecDestroy");
        const size_t bytes=size_t(bcount)*n*l*4,count=size_t(bcount)*n*l;const int ncase=int(ld3_recs.size());
        std::vector<uint32_t> ha,hb;ha.reserve(count);hb.reserve(count);set_size_no_init(ha,count);set_size_no_init(hb,count);
        d.check(d.cuMemcpyDtoH(ha.data(),a_all,bytes));d.check(d.cuMemcpyDtoH(hb.data(),b_all,bytes));
        auto h2d_staged=[&]{if(stg.ok){stage_up(a_all,reinterpret_cast<const char*>(ha.data()),bytes);stage_up(b_all,reinterpret_cast<const char*>(hb.data()),bytes);d.check(stg.ssync(stg.s));}
            else{d.check(d.cuMemcpyHtoD(a_all,ha.data(),bytes));d.check(d.cuMemcpyHtoD(b_all,hb.data(),bytes));}};
        auto plain2g=[&]{const size_t two=size_t(2)<<30;d.check(d.cuMemcpyHtoD(a_all,ha.data(),bytes));if(two>bytes)d.check(d.cuMemcpyHtoD(b_all,hb.data(),std::min(bytes,two-bytes)));};
        auto once=[&](int cond,CUgraphExec ge,bool outalloc){Ld3Rec r;r.cond=cond;r.slot=ld3_slots++;ld3_rec_before(r);
            const auto t1=std::chrono::steady_clock::now();d.check(glaunch(ge,graph_stream));const auto t2=std::chrono::steady_clock::now();
            {std::vector<uint32_t> o;if(outalloc){o.reserve(count);advise_unused_output_thp(output_thp_stats,o.data(),1,count,o.capacity(),host_page_size);set_size_no_init(o,count);}
             d.check(d.cuCtxSynchronize());}
            const auto t3=std::chrono::steady_clock::now();ld3_rec_after(r);
            r.lau=std::chrono::duration<double,std::micro>(t2-t1).count();r.host=std::chrono::duration<double,std::micro>(t3-t1).count();ld3_recs.push_back(r);};
        std::atomic<bool> stop{false};std::atomic<int> up{0};
        auto with_threads=[&](int nt,bool copy,const std::function<void()>& f){std::vector<std::thread> th;stop=false;up=0;
            for(int t=0;t<nt;++t)th.emplace_back([&,copy]{std::vector<char> s,dd;if(copy){s.assign(size_t(64)<<20,1);dd.assign(size_t(64)<<20,0);}
                ++up;volatile unsigned long x=0;while(!stop.load(std::memory_order_relaxed)){if(copy)std::memcpy(dd.data(),s.data(),s.size());else x=x+1;}});
            while(up.load()<nt)std::this_thread::yield();f();stop=true;for(auto& t:th)t.join();};
        const char* cn[]={"c_ref","c_spin16","c_memcpy16","c_pin","c_plain2g","c_plain2g_st","c_fresh1","c_fresh2","c_b2b","c_outalloc","c_noout","c_ref_end"};
        const int R=5;CUgraphExec P=graph_execs[6];
        for(int r=0;r<R;++r){h2d_staged();once(0,P,false);}
        for(int r=0;r<R;++r){h2d_staged();with_threads(16,false,[&]{once(1,P,false);});}
        for(int r=0;r<R;++r){h2d_staged();with_threads(16,true,[&]{once(2,P,false);});}
        {cpu_set_t old;CPU_ZERO(&old);pthread_getaffinity_np(pthread_self(),sizeof old,&old);cpu_set_t one;CPU_ZERO(&one);CPU_SET(std::max(0,sched_getcpu()),&one);
         pthread_setaffinity_np(pthread_self(),sizeof one,&one);for(int r=0;r<R;++r){h2d_staged();once(3,P,false);}pthread_setaffinity_np(pthread_self(),sizeof old,&old);}
        for(int r=0;r<R;++r){plain2g();once(4,P,false);}
        for(int r=0;r<R;++r){plain2g();h2d_staged();once(5,P,false);}
        if(ld3_graph)for(int r=0;r<R;++r){CUgraphExec ge=nullptr;if(ginst(&ge,ld3_graph,0)!=CUDA_SUCCESS)break;h2d_staged();once(6,ge,false);h2d_staged();once(7,ge,false);ged(ge);}
        for(int r=0;r<R;++r)once(8,P,false);
        for(int r=0;r<R;++r){h2d_staged();once(9,P,true);h2d_staged();once(10,P,false);}
        for(int r=0;r<R;++r){h2d_staged();once(11,P,false);}
        // stamps -> body and effective SM clock per slot
        std::vector<unsigned long long> sb(size_t(LD3_SLOTS)*2*256*2);std::vector<unsigned> fl(size_t(LD3_SLOTS)*2*256);
        d.check(d.cuCtxSynchronize());d.check(d.cuMemcpyDtoH(sb.data(),ld3_buf,sb.size()*8));d.check(d.cuMemcpyDtoH(fl.data(),ld3_flag,fl.size()*4));
        auto slotv=[&](int s,double& body,double& clk,int& nsm){body=-1;clk=-1;nsm=0;if(s<0||s>=LD3_SLOTS)return;unsigned long long g0=~0ull,g1=0;std::vector<double> c;
            for(int m=0;m<256;++m){const size_t h=size_t(s*2)*256+m,t=size_t(s*2+1)*256+m;
                if(fl[h]){g0=std::min(g0,sb[2*h+1]);}if(fl[t]){g1=std::max(g1,sb[2*t+1]);}
                if(fl[h]&&fl[t]&&sb[2*t+1]>sb[2*h+1])c.push_back(double(sb[2*t]-sb[2*h])/double(sb[2*t+1]-sb[2*h+1])*1e3);}
            if(g1>g0&&g0!=~0ull)body=double(g1-g0)/1e3;nsm=int(c.size());if(!c.empty()){std::sort(c.begin(),c.end());clk=c[c.size()/2];}};
        auto med=[](std::vector<double> v){if(v.empty())return -1.0;std::sort(v.begin(),v.end());return v[v.size()/2];};
        auto line=[&](const char* nm,int i0,const std::vector<int>& idx){std::vector<double> bo,ck,ca,cb,pw,tm,ut,lau,cost,costb,iv,pv,tv;unsigned long long rs=0;int ps=-1,np=0,no=0,mig=0,nsm=0;double ld=0,bmin=1e30,bmax=0;
            for(int k:idx){const auto& r=ld3_recs[k];double b,c;int ns;slotv(r.slot,b,c,ns);bo.push_back(b);ck.push_back(c);nsm=ns;bmin=std::min(bmin,b);bmax=std::max(bmax,b);
                ca.push_back(r.a.clk);cb.push_back(r.b.clk);pw.push_back(r.b.pw/1000.0);tm.push_back(r.b.tmp);ut.push_back(r.b.ut);lau.push_back(r.lau);cost.push_back(r.a.cost);costb.push_back(r.b.cost);
                iv.push_back(double(r.ivcsw));if(k>0){pv.push_back(double(r.b.pv-ld3_recs[k-1].b.pv)/1e3);tv.push_back(double(r.b.tv-ld3_recs[k-1].b.tv)/1e3);}rs|=r.a.rs|r.b.rs;ps=r.b.ps;np=std::max(np,int(r.b.np));no=std::max(no,int(r.b.no));mig+=r.cpu0!=r.cpu1;ld=r.load;}
            std::printf("{\"diag\":\"ld3\",\"v\":\"%s\",%s\"body\":%.1f,\"bmin\":%.1f,\"bmax\":%.1f,\"clk\":%.0f,\"nclk\":\"%.0f/%.0f\",\"ps\":%d,\"rs\":\"%llx\",\"pviol\":%.0f,\"tviol\":%.0f,\"W\":%.0f,\"C\":%.0f,\"ut\":%.0f,\"procs\":\"%d/%d\",\"mig\":%d,\"ivcs\":%.0f,\"ld\":%.1f,\"lau\":%.0f,\"nva\":%.0f,\"nvb\":%.0f,\"nsm\":%d}\n",
                nm,i0>=0?(std::string("\"i\":")+std::to_string(i0)+",").c_str():"",med(bo),bmin,bmax,med(ck),med(ca),med(cb),ps,rs,med(pv),med(tv),med(pw),med(tm),med(ut),np,no,mig,med(iv),ld,med(lau),med(cost),med(costb),nsm);std::fflush(stdout);};
        unsigned devslots=0;d.check(d.cuMemcpyDtoH(&devslots,ld3_ctr,4));std::printf("{\"diag\":\"ld3\",\"k\":\"info\",\"devslots\":%u,",devslots);std::printf("\"nvml\":%d,\"pstride\":%zu,\"cases\":%d,\"slots\":%d,\"B\":%u}\n",ld3_nv.ok?1:0,ld3_nv.pstride,ncase,ld3_slots,bcount);
        for(int c=0;c<12;++c){std::vector<int> idx;for(int k=ncase;k<int(ld3_recs.size());++k)if(ld3_recs[k].cond==c)idx.push_back(k);if(!idx.empty())line(cn[c],-1,idx);}
        for(int k=0;k<ncase&&k<16;++k)line("case",k,{k});   // the timed cases LAST (the site keeps the log tail)
    }
    fherma::Outputs run(const fherma::Inputs& in){
        (void)in;  // inputs already on the device (h2d)
        const auto r0=std::chrono::steady_clock::now();auto r1=r0,r2=r0;
        auto& d=*cuda;const size_t stride=size_t(n)*l*4,count=size_t(bcount)*n*l;
        d.kp.collect();  // previous call not yet read (no d2h in between)
        if(graph_exec&&!std::getenv("FHERMA_NO_GRAPH")){
            static auto glaunch=d.driver.get<decltype(&::cuGraphLaunch)>("cuGraphLaunch");
            int v=want_po();if(v==4&&graph_execs[6]&&!std::getenv("FHERMA_L2OFF"))v=6;if(v==6&&graph_execs[7]&&std::getenv("FHERMA_P1X"))v=7;if(!graph_execs[v])v=graph_exec==graph_execs[0]?0:graph_exec==graph_execs[4]?4:6;d.kp.run_begin(graph_stream,v);if(ld3_on&&v==6){ld3_cur=Ld3Rec{};ld3_cur.slot=ld3_slots++;ld3_rec_before(ld3_cur);ld3_live=true;}r1=std::chrono::steady_clock::now();d.check(glaunch(graph_execs[v]?graph_execs[v]:graph_exec,graph_stream));r2=std::chrono::steady_clock::now();d.kp.run_launched(graph_stream);
        }else for(u32 i=0;i<bcount;++i){po=po_available()?std::min(want_po(),2):0;compute_one(a_all+i*stride,b_all+i*stride,out_all+i*stride);if(prof){d.check(d.cuCtxSynchronize());prof_collect();}}
        if(prof)pcalls+=0;
        fherma::Outputs result;result.c.shape={int64_t(bcount),int64_t(n),int64_t(l)};
        if(output_pool){owning_pinned_vector::Scope select(*output_pool);result.c.data.reserve(count);}
        else result.c.data.reserve(count);
        // candidate 8: no first word written here (that first touch cost a page fault, or a 2 MB huge-page zeroing, inside
        // the ranked call); FHERMA_OUT_TOUCH=1 restores it. Huge pages for the untouched output: fewer d2h first-touch faults.
        static const bool out_touch=lever_env("FHERMA_OUT_TOUCH",false);
        if(out_touch)result.c.data.emplace_back(0);
        advise_unused_output_thp(output_thp_stats,result.c.data.data(),out_touch?result.c.data.size():1,count,result.c.data.capacity(),host_page_size);
        set_size_no_init(result.c.data,count);  // filled by d2h
        const auto r3=std::chrono::steady_clock::now();
        d.check(d.cuCtxSynchronize());
        const auto r4=std::chrono::steady_clock::now();
        if(ld3_live){ld3_live=false;ld3_rec_after(ld3_cur);ld3_cur.lau=std::chrono::duration<double,std::micro>(r2-r1).count();ld3_cur.host=std::chrono::duration<double,std::micro>(r4-r1).count();ld3_recs.push_back(ld3_cur);}
        d.kp.run_end();
        {auto us=[](std::chrono::steady_clock::time_point a,std::chrono::steady_clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();};
         const double x[5]={us(r0,r1),us(r1,r2),us(r2,r3),us(r3,r4),us(r0,r4)};ht_run.add(x,5);}
        return result;
    }
    // Bench-only helpers (not part of the official entry points).
    void bench_replicate(u32 present){
        auto& d=*cuda;const size_t stride=size_t(n)*l*4;
        auto copy=d.driver.get<decltype(&::cuMemcpyDtoD)>("cuMemcpyDtoD_v2");
        for(u32 i=present;i<bcount;++i){d.check(copy(a_all+i*stride,a_all+(i%present)*stride,stride));d.check(copy(b_all+i*stride,b_all+(i%present)*stride,stride));}
        d.check(d.cuCtxSynchronize());
    }
    void bench_upload(const uint32_t* a,const uint32_t* b,u32 products){
        auto& d=*cuda;const size_t bytes=size_t(products)*n*l*4;
        d.check(d.cuMemcpyHtoD(a_all,a,bytes));d.check(d.cuMemcpyHtoD(b_all,b,bytes));
    }
    void bench_fetch(u32 index,u32 products,void* dst){
        auto& d=*cuda;const size_t stride=size_t(n)*l*4;
        d.check(d.cuMemcpyDtoH(dst,out_all+index*stride,products*stride));
    }
};
}
void* fherma_init(const fherma::Point& point){
    owning_pinned_vector::Bypass bypass;
    auto state=std::make_unique<State>(point);
    state->build_graph();
    if(state->output_pool)state->output_pool->activate_prepared_reserve();
    return state.release();
}
fherma::Outputs fherma_run(void* state,const fherma::Inputs& in){return static_cast<State*>(state)->run(in);}
void fherma_h2d(void* state,const fherma::Inputs& in){static_cast<State*>(state)->h2d(in);}
fherma::Outputs fherma_d2h(void* state,fherma::Outputs answer){return static_cast<State*>(state)->d2h(std::move(answer));}
extern "C" void fherma_bench_replicate(void* state,unsigned present){static_cast<State*>(state)->bench_replicate(present);}
extern "C" void fherma_bench_upload(void* state,const uint32_t* a,const uint32_t* b,unsigned products){static_cast<State*>(state)->bench_upload(a,b,products);}
extern "C" void fherma_bench_fetch(void* state,unsigned index,unsigned products,void* dst){static_cast<State*>(state)->bench_fetch(index,products,dst);}
void fherma_free(void* state){delete static_cast<State*>(state);}
