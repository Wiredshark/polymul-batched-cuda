#pragma once
#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

// L2R1 revision 2: one fixed, public-parameter-only cache policy. It never launches a
// kernel, reads a coefficient, or changes a transform constant. The State owns
// a private CUDA context. Its original limit/default-stream attributes are
// restored before that context and the root allocation are destroyed.
struct L2RootCacheApi {
    decltype(&::cuDeviceGetAttribute) device_attribute=nullptr;
    decltype(&::cuCtxGetLimit) get_limit=nullptr;
    decltype(&::cuCtxSetLimit) set_limit=nullptr;
    decltype(&::cuStreamGetAttribute) get_stream=nullptr;
    decltype(&::cuStreamSetAttribute) set_stream=nullptr;
    decltype(&::cuGraphKernelNodeGetAttribute) get_node=nullptr;
    decltype(&::cuGraphKernelNodeSetAttribute) set_node=nullptr;
    decltype(&::cuCtxSynchronize) synchronize=nullptr;
};

class L2RootCachePolicy {
public:
    static constexpr size_t root_bytes=164096u;
    struct Observation { uintptr_t node; unsigned role; CUaccessPolicyWindow window; };
private:
    L2RootCacheApi api_;
    CUdeviceptr base_=0;
    size_t old_limit_=0,actual_limit_=0,maximum_actual_limit_=0;
    CUstreamAttrValue old_stream_{};
    bool limit_touched_=false,stream_touched_=false,closed_=false;
    std::vector<Observation> nodes_;
    int maximum_set_aside_=0,maximum_window_=0,l2_bytes_=0;
    static void check(CUresult r) {
        if(r!=CUDA_SUCCESS)throw std::runtime_error("L2R1 CUDA policy error "+std::to_string(int(r)));
    }
    static void require(bool yes,const char* message) {
        if(!yes)throw std::runtime_error(message);
    }
    CUaccessPolicyWindow wanted()const {
        CUaccessPolicyWindow w{};
        w.base_ptr=reinterpret_cast<void*>(static_cast<uintptr_t>(base_));
        w.num_bytes=root_bytes;w.hitRatio=1.0f;
        w.hitProp=CU_ACCESS_PROPERTY_PERSISTING;w.missProp=CU_ACCESS_PROPERTY_NORMAL;
        return w;
    }
    void verify(const CUaccessPolicyWindow& w)const {
        const auto want=wanted();
        require(w.base_ptr==want.base_ptr&&w.num_bytes==want.num_bytes&&
            w.hitRatio==1.0f&&w.hitProp==want.hitProp&&w.missProp==want.missProp,
            "L2R1 policy readback mismatch");
    }
public:
    L2RootCachePolicy(L2RootCacheApi api,CUdevice device,CUdeviceptr base,size_t extent):api_(api),base_(base) {
        require(base_&&extent==root_bytes,"L2R1 root extent mismatch");
        require(api_.device_attribute&&api_.get_limit&&api_.set_limit&&api_.get_stream&&
            api_.set_stream&&api_.get_node&&api_.set_node&&api_.synchronize,
            "L2R1_CAPABILITY_EXCLUSION_missing_driver_symbol");
        check(api_.device_attribute(&maximum_set_aside_,CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,device));
        check(api_.device_attribute(&maximum_window_,CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,device));
        check(api_.device_attribute(&l2_bytes_,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,device));
        require(maximum_set_aside_>=int(root_bytes)&&maximum_window_>=int(root_bytes)&&l2_bytes_>int(root_bytes),
            "ROOT_FACTOR_CAPABILITY_EXCLUSION_compact_policy_unavailable");
        // cuCtxSetLimit is a request. The driver may round or clamp it.
        // Keep the root window fixed; accept only an actual set-aside that
        // covers that window and stays within BOTH reported byte capacities.
        // Cast only after the signed device attributes passed the checks above.
        maximum_actual_limit_=size_t(maximum_set_aside_<l2_bytes_?maximum_set_aside_:l2_bytes_);
        check(api_.get_limit(&old_limit_,CU_LIMIT_PERSISTING_L2_CACHE_SIZE));
        check(api_.get_stream(nullptr,CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,&old_stream_));
        try {
#ifndef L2R1_TEST_POLICY_OFF
            limit_touched_=true;check(api_.set_limit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE,root_bytes));
#endif
            check(api_.get_limit(&actual_limit_,CU_LIMIT_PERSISTING_L2_CACHE_SIZE));
#ifndef L2R1_TEST_POLICY_OFF
            require(actual_limit_>=root_bytes&&actual_limit_<=maximum_actual_limit_,
                "L2R1_CAPABILITY_EXCLUSION_set_aside_out_of_bounds_v2");
            CUstreamAttrValue desired{};desired.accessPolicyWindow=wanted();
            stream_touched_=true;check(api_.set_stream(nullptr,CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,&desired));
            verify(default_window());
#endif
            nodes_.reserve(66);
        }catch(...){close();throw;}
    }
    L2RootCachePolicy(const L2RootCachePolicy&)=delete;
    L2RootCachePolicy& operator=(const L2RootCachePolicy&)=delete;
    ~L2RootCachePolicy(){close();}
    void graph_node(CUgraphNode node,unsigned role) {
        require(!closed_&&role>=1&&role<=4&&nodes_.size()<66,"L2R1 graph role/count");
#ifndef L2R1_TEST_POLICY_OFF
        CUkernelNodeAttrValue desired{};desired.accessPolicyWindow=wanted();
        check(api_.set_node(node,CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW,&desired));
#endif
        CUkernelNodeAttrValue observed{};
        check(api_.get_node(node,CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW,&observed));
#ifndef L2R1_TEST_POLICY_OFF
        verify(observed.accessPolicyWindow);
#endif
        nodes_.push_back({reinterpret_cast<uintptr_t>(node),role,observed.accessPolicyWindow});
    }
    void require_graph_coverage()const {
        unsigned counts[5]{};
        for(const auto& x:nodes_)++counts[x.role];
        require(nodes_.size()==66&&counts[1]==24&&counts[2]==24&&counts[3]==6&&counts[4]==12,
            "L2R1 graph policy coverage incomplete");
    }
    CUaccessPolicyWindow default_window()const {
        CUstreamAttrValue v{};
        check(api_.get_stream(nullptr,CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,&v));
        return v.accessPolicyWindow;
    }
    size_t current_limit()const {
        size_t v=0;check(api_.get_limit(&v,CU_LIMIT_PERSISTING_L2_CACHE_SIZE));return v;
    }
    const std::vector<Observation>& observations()const{return nodes_;}
    CUdeviceptr base()const{return base_;}
    int maximum_set_aside()const{return maximum_set_aside_;}
    int maximum_window()const{return maximum_window_;}
    int l2_bytes()const{return l2_bytes_;}
    size_t maximum_actual_limit()const{return maximum_actual_limit_;}
    size_t applied_limit()const{return actual_limit_;}
    size_t previous_limit()const{return old_limit_;}
    CUaccessPolicyWindow previous_window()const{return old_stream_.accessPolicyWindow;}
    void close()noexcept {
        if(closed_)return;
        closed_=true;
        // Never unwind into freeing buffers after an unsuccessful drain.
        if((limit_touched_||stream_touched_)&&api_.synchronize()!=CUDA_SUCCESS)std::terminate();
        if(stream_touched_&&api_.set_stream(nullptr,CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,&old_stream_)!=CUDA_SUCCESS)std::terminate();
        if(limit_touched_&&api_.set_limit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE,old_limit_)!=CUDA_SUCCESS)std::terminate();
        size_t restored_limit=0;CUstreamAttrValue restored_stream{};
        if(api_.get_limit(&restored_limit,CU_LIMIT_PERSISTING_L2_CACHE_SIZE)!=CUDA_SUCCESS||restored_limit!=old_limit_)std::terminate();
        if(api_.get_stream(nullptr,CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,&restored_stream)!=CUDA_SUCCESS)std::terminate();
        const auto& a=restored_stream.accessPolicyWindow;const auto& b=old_stream_.accessPolicyWindow;
        if(a.base_ptr!=b.base_ptr||a.num_bytes!=b.num_bytes||a.hitRatio!=b.hitRatio||a.hitProp!=b.hitProp||a.missProp!=b.missProp)std::terminate();
        std::fprintf(stderr,"{\"l2r1_policy\":true,\"policy_version\":2,\"requested_bytes\":%zu,\"actual_bytes\":%zu,\"minimum_bytes\":%zu,\"maximum_bytes\":%zu,\"maximum_set_aside\":%d,\"maximum_window\":%d,\"l2_bytes\":%d,\"graph_nodes\":%zu,\"previous_limit\":%zu,\"restored_limit\":%zu,\"previous_stream\":{\"base\":%llu,\"bytes\":%zu,\"hit_ratio\":%.9g,\"hit_property\":%d,\"miss_property\":%d},\"restored_stream\":{\"base\":%llu,\"bytes\":%zu,\"hit_ratio\":%.9g,\"hit_property\":%d,\"miss_property\":%d},\"stream_restored\":true,\"restored\":true}\n",
            root_bytes,actual_limit_,root_bytes,maximum_actual_limit_,maximum_set_aside_,maximum_window_,l2_bytes_,
            nodes_.size(),old_limit_,restored_limit,
            (unsigned long long)reinterpret_cast<uintptr_t>(b.base_ptr),b.num_bytes,double(b.hitRatio),int(b.hitProp),int(b.missProp),
            (unsigned long long)reinterpret_cast<uintptr_t>(a.base_ptr),a.num_bytes,double(a.hitRatio),int(a.hitProp),int(a.missProp));
        // XW32 compact transport: six complete records below the observed
        // 300-character UI row limit. The original full row remains above.
        // This is teardown reporting after exact restoration, outside run().
        const auto context=(unsigned long long)base_;
        std::fprintf(stderr,"{\"xw32p\":1,\"c\":%llu,\"i\":0,\"v\":2,\"req\":%zu,\"act\":%zu,\"min\":%zu,\"max\":%zu}\n",
            context,root_bytes,actual_limit_,root_bytes,maximum_actual_limit_);
        std::fprintf(stderr,"{\"xw32p\":1,\"c\":%llu,\"i\":1,\"set\":%d,\"win\":%d,\"l2\":%d,\"nodes\":%zu}\n",
            context,maximum_set_aside_,maximum_window_,l2_bytes_,nodes_.size());
        std::fprintf(stderr,"{\"xw32p\":1,\"c\":%llu,\"i\":2,\"old\":%zu,\"new\":%zu}\n",
            context,old_limit_,restored_limit);
        std::fprintf(stderr,"{\"xw32p\":1,\"c\":%llu,\"i\":3,\"base\":%llu,\"bytes\":%zu,\"ratio\":%.9g,\"hit\":%d,\"miss\":%d}\n",
            context,(unsigned long long)reinterpret_cast<uintptr_t>(b.base_ptr),b.num_bytes,double(b.hitRatio),int(b.hitProp),int(b.missProp));
        std::fprintf(stderr,"{\"xw32p\":1,\"c\":%llu,\"i\":4,\"base\":%llu,\"bytes\":%zu,\"ratio\":%.9g,\"hit\":%d,\"miss\":%d}\n",
            context,(unsigned long long)reinterpret_cast<uintptr_t>(a.base_ptr),a.num_bytes,double(a.hitRatio),int(a.hitProp),int(a.missProp));
        std::fprintf(stderr,"{\"xw32p\":1,\"c\":%llu,\"i\":5,\"parts\":6,\"stream_restored\":true,\"restored\":true}\n",context);
    }
};
