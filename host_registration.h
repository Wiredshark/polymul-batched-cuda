#pragma once
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

// No ownership transfer: pinning is limited to this timed call. Page unions
// avoid duplicate registration of adjacent heap allocations sharing a page.
struct HostRegistrationStats {
    uint64_t runs=0,fully_registered_runs=0,fallback_runs=0;
    uint64_t attempts=0,successes=0,bytes=0,unregister_attempts=0,unregister_successes=0;
    uint64_t missing_symbols=0,register_failures=0,unregister_failures=0,sync_failures=0;
    int last_register_error=0,last_unregister_error=0;
    void report() const {
        if(!runs)return;
        std::fprintf(stderr,"{\"host_registration\":true,\"runs\":%llu,\"fully_registered_runs\":%llu,\"fallback_runs\":%llu,\"attempts\":%llu,\"successes\":%llu,\"bytes\":%llu,\"unregister_attempts\":%llu,\"unregister_successes\":%llu,\"missing_symbols\":%llu,\"register_failures\":%llu,\"unregister_failures\":%llu,\"sync_failures\":%llu,\"last_register_error\":%d,\"last_unregister_error\":%d}\n",
            (unsigned long long)runs,(unsigned long long)fully_registered_runs,(unsigned long long)fallback_runs,
            (unsigned long long)attempts,(unsigned long long)successes,(unsigned long long)bytes,
            (unsigned long long)unregister_attempts,(unsigned long long)unregister_successes,
            (unsigned long long)missing_symbols,(unsigned long long)register_failures,
            (unsigned long long)unregister_failures,(unsigned long long)sync_failures,
            last_register_error,last_unregister_error);
    }
};

struct HostRange {uintptr_t begin=0,end=0;};
inline HostRange host_page_range(const void* pointer,size_t bytes,size_t page_size){
    uintptr_t start=reinterpret_cast<uintptr_t>(pointer);
    if(!bytes||!page_size||(page_size&(page_size-1))||start>std::numeric_limits<uintptr_t>::max()-bytes)
        throw std::runtime_error("invalid host registration range");
    uintptr_t end=start+bytes;
    if(end>std::numeric_limits<uintptr_t>::max()-(page_size-1))throw std::runtime_error("host registration alignment overflow");
    return {start&~uintptr_t(page_size-1),(end+page_size-1)&~uintptr_t(page_size-1)};
}
inline size_t merge_host_ranges(std::array<HostRange,3>& ranges){
    std::sort(ranges.begin(),ranges.end(),[](auto a,auto b){return a.begin<b.begin;});
    size_t count=0;
    for(auto range:ranges){
        if(count&&range.begin<=ranges[count-1].end)ranges[count-1].end=std::max(ranges[count-1].end,range.end);
        else ranges[count++]=range;
    }
    return count;
}

template<class Driver>class RegisteredHostRanges {
    Driver& driver;
    HostRegistrationStats& stats;
    std::array<HostRange,3> ranges;
    std::array<bool,3> owned{};
    size_t count=0;
    bool fully_registered_=false;
    bool release_noexcept() noexcept {
        bool ok=true;
        for(size_t i=count;i--;){
            if(!owned[i])continue;
            ++stats.unregister_attempts;
            CUresult result=driver.cuMemHostUnregister(reinterpret_cast<void*>(ranges[i].begin));
            if(result==CUDA_SUCCESS){owned[i]=false;++stats.unregister_successes;}
            else{ok=false;++stats.unregister_failures;stats.last_unregister_error=int(result);}
        }
        return ok;
    }
public:
    RegisteredHostRanges(Driver& d,HostRegistrationStats& s,const void* a,const void* b,void* output,size_t bytes,size_t page_size):
        driver(d),stats(s),ranges{host_page_range(a,bytes,page_size),host_page_range(b,bytes,page_size),host_page_range(output,bytes,page_size)}{
        ++stats.runs;count=merge_host_ranges(ranges);
        if(!driver.cuMemHostRegister||!driver.cuMemHostUnregister){++stats.missing_symbols;++stats.fallback_runs;return;}
        for(size_t i=0;i<count;++i){
            ++stats.attempts;
            CUresult result=driver.cuMemHostRegister(reinterpret_cast<void*>(ranges[i].begin),ranges[i].end-ranges[i].begin,0);
            if(result!=CUDA_SUCCESS){
                ++stats.register_failures;stats.last_register_error=int(result);++stats.fallback_runs;
                // Never unregister a range we did not successfully register.
                // Rollback all of our ranges before attempting pageable copies.
                if(!release_noexcept()){
                    // Constructor failure has no destructor. Retry outstanding
                    // cleanup once, then report a hard failure to the harness.
                    release_noexcept();
                    throw std::runtime_error("CUDA host registration rollback failed");
                }
                return;
            }
            owned[i]=true;++stats.successes;stats.bytes+=ranges[i].end-ranges[i].begin;
        }
        fully_registered_=true;++stats.fully_registered_runs;
    }
    RegisteredHostRanges(const RegisteredHostRanges&)=delete;
    RegisteredHostRanges& operator=(const RegisteredHostRanges&)=delete;
    bool fully_registered() const noexcept{return fully_registered_;}
    void release(){
        if(!release_noexcept())throw std::runtime_error("CUDA host unregister failed");
        fully_registered_=false;
    }
    ~RegisteredHostRanges(){
        if(std::any_of(owned.begin(),owned.end(),[](bool value){return value;})){
            // This path is exception cleanup. Complete outstanding work before
            // caller-owned vectors can be destroyed or reused.
            if(driver.cuCtxSynchronize()!=CUDA_SUCCESS)++stats.sync_failures;
            release_noexcept();
        }
    }
};

// Declare after the registration guard and output vector. On exception this
// drains submitted work before those objects release any host storage.
template<class Driver>class CurrentCallCompletion {
    Driver& driver;
    HostRegistrationStats& stats;
    bool pending=true;
public:
    CurrentCallCompletion(Driver& d,HostRegistrationStats& s):driver(d),stats(s){}
    CurrentCallCompletion(const CurrentCallCompletion&)=delete;
    CurrentCallCompletion& operator=(const CurrentCallCompletion&)=delete;
    void finish(){
        // Leave pending set if check throws, so unwinding attempts completion.
        driver.check(driver.cuCtxSynchronize());pending=false;
    }
    ~CurrentCallCompletion(){
        if(pending&&driver.cuCtxSynchronize()!=CUDA_SUCCESS)++stats.sync_failures;
    }
};
