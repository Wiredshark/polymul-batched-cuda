#pragma once
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <dlfcn.h>

// Rectangular executable memcpy nodes cannot have their operands updated with
// cuGraphExecMemcpyNodeSetParams. Every graph here permanently names empty host
// storage owned by the State's pool. No arbitrary caller pointer is retained.
struct ColumnPipelineConfig {
    uint32_t n,w,l,k,rows,cols,group,logrows,outer_threads,middle_threads;
    uint32_t short_words,delta,digit_terms;
    CUdeviceptr a,b,at,bt,A,B,C,out,p,mu,weighted_twist,tw,itw,invn;
    CUdeviceptr weights,positions,crt_inverses,crt_matrix,corrections,qmod,bias;
};

template<class Driver> class ColumnStripPipeline {
    using u32=uint32_t;
    static constexpr u32 columns=64,log_columns=6;
    Driver& d;
    ColumnPipelineConfig c;
    void* slots[3]{};
    CUgraphExec front[3][3]{},back[3]{};
    CUstream stream=nullptr;
    bool available_=false,closed=false;
    uint64_t nodes=0,instantiations=0,calls=0,launches=0,fallbacks=0;
    uint64_t api_failures=0,sync_failures=0,cleanup_failures=0;
    decltype(&::cuGraphCreate) create=nullptr;
    decltype(&::cuGraphAddMemcpyNode) add_memcpy=nullptr;
    decltype(&::cuGraphAddKernelNode) add_kernel=nullptr;
    decltype(&::cuGraphInstantiate) instantiate=nullptr;
    decltype(&::cuGraphLaunch) launch=nullptr;
    decltype(&::cuGraphExecDestroy) destroy_exec=nullptr;
    decltype(&::cuGraphDestroy) destroy_graph=nullptr;
    decltype(&::cuStreamCreate) create_stream=nullptr;
    decltype(&::cuStreamDestroy) destroy_stream=nullptr;

    template<class F> F symbol(const char* name) noexcept {
        return reinterpret_cast<F>(dlsym(d.driver.handle,name));
    }
    void check(CUresult status){
        if(status!=CUDA_SUCCESS)++api_failures;
        d.check(status);
    }
    void cleanup_check(CUresult status) noexcept {
        if(status!=CUDA_SUCCESS){++cleanup_failures;report();std::terminate();}
    }
    // A failed drain cannot certify DMA completion. Terminate rather than let
    // exception unwinding release or reuse any storage that CUDA might access.
    void drain_noexcept() noexcept {
        if(d.cuCtxSynchronize()!=CUDA_SUCCESS){++sync_failures;report();std::terminate();}
    }
    CUgraphNode copy(CUgraph graph,const CUgraphNode* deps,size_t count,
                     bool upload,void* host,CUdeviceptr device,u32 first){
        CUDA_MEMCPY3D x{};
        const size_t pitch=size_t(c.cols)*c.l*sizeof(u32);
        const size_t offset=size_t(first)*c.l*sizeof(u32);
        x.srcMemoryType=upload?CU_MEMORYTYPE_HOST:CU_MEMORYTYPE_DEVICE;
        x.dstMemoryType=upload?CU_MEMORYTYPE_DEVICE:CU_MEMORYTYPE_HOST;
        if(upload){x.srcHost=static_cast<const unsigned char*>(host)+offset;x.dstDevice=device+offset;}
        else{x.srcDevice=device+offset;x.dstHost=static_cast<unsigned char*>(host)+offset;}
        x.srcPitch=pitch;x.dstPitch=pitch;x.srcHeight=c.rows;x.dstHeight=c.rows;
        x.WidthInBytes=size_t(columns)*c.l*sizeof(u32);x.Height=c.rows;x.Depth=1;
        CUgraphNode node=nullptr;
        check(add_memcpy(&node,graph,deps,count,&x,d.context));++nodes;return node;
    }
    template<class... Args> CUgraphNode kernel(CUgraph graph,const CUgraphNode* deps,size_t count,
            CUfunction fn,u32 gx,u32 gy,u32 threads,u32 shared,Args... args){
        void* params[]={static_cast<void*>(&args)...};
        CUDA_KERNEL_NODE_PARAMS x{};
        x.func=fn;x.gridDimX=gx;x.gridDimY=gy;x.gridDimZ=1;
        x.blockDimX=threads;x.blockDimY=1;x.blockDimZ=1;
        x.sharedMemBytes=shared;x.kernelParams=params;
        CUgraphNode node=nullptr;
        check(add_kernel(&node,graph,deps,count,&x));++nodes;
        // Set the graph-node attribute itself before instantiation. A launch
        // stream attribute is not used as evidence of graph-node coverage.
        if(d.root_cache){
            unsigned role=fn==d.weighted_prepare_columns?1:
                fn==d.weighted_forward_columns?2:fn==d.weighted_middle?3:
                fn==d.weighted_inverse_columns?4:0;
            if(role)d.root_cache->graph_node(node,role);
        }
        return node;
    }
    void make_front(unsigned ia,unsigned ib){
        CUgraph graph=nullptr;
        try{
            check(create(&graph,0));
            CUgraphNode copy_tail=nullptr,forwards[8]{};
            for(u32 s=0;s<c.cols/columns;++s){
                const u32 first=s*columns;
                auto ca=copy(graph,copy_tail?&copy_tail:nullptr,copy_tail?1:0,true,slots[ia],c.a,first);
                auto cb=copy(graph,&ca,1,true,slots[ib],c.b,first);copy_tail=cb;
                auto t=kernel(graph,&cb,1,d.transpose_inputs_columns,(c.l+31)/32,c.rows*(columns/32),256,0,
                    c.a,c.b,c.at,c.bt,c.n,c.l,c.cols,first,log_columns);
                auto p=kernel(graph,&t,1,d.weighted_prepare_columns,(c.rows*columns+127)/128,c.k*8,128,0,
                    c.at,c.bt,c.A,c.B,c.p,c.mu,c.weighted_twist,c.tw,c.weights,c.positions,
                    c.n,c.l,c.w,c.digit_terms,c.cols,first,log_columns);
                forwards[s]=kernel(graph,&p,1,d.weighted_forward_columns,columns/c.group,c.k*8,c.outer_threads,
                    8u*c.rows*c.group,c.A,c.B,c.p,c.mu,c.tw,c.n,c.rows,c.cols,c.group,c.logrows,u32(8),first);
            }
            kernel(graph,forwards,c.cols/columns,d.weighted_middle,c.rows,c.k,c.middle_threads,8u*8*c.cols,
                c.A,c.B,c.C,c.p,c.mu,c.tw,c.itw,c.n,c.rows,c.cols,c.logrows,u32(8));
            check(instantiate(&front[ia][ib],graph,0));++instantiations;
            cleanup_check(destroy_graph(graph));graph=nullptr;
        }catch(...){if(graph)cleanup_check(destroy_graph(graph));throw;}
    }
    void make_back(unsigned io){
        CUgraph graph=nullptr;
        try{
            check(create(&graph,0));
            CUgraphNode compute_tail=nullptr,copy_tail=nullptr;
            for(u32 s=0;s<c.cols/columns;++s){
                const u32 first=s*columns;
                auto inv=kernel(graph,compute_tail?&compute_tail:nullptr,compute_tail?1:0,
                    d.weighted_inverse_columns,columns/c.group,c.k*8,c.outer_threads,4u*c.rows*c.group,
                    c.C,c.p,c.mu,c.itw,c.invn,c.n,c.rows,c.cols,c.group,u32(8),first);
                auto crt=kernel(graph,&inv,1,d.reconstruct_weighted8_subwarp16_columns,c.rows*columns,1,128,0,
                    c.C,c.out,c.p,c.mu,c.crt_inverses,c.crt_matrix,c.corrections,c.positions,c.qmod,c.bias,
                    c.n,c.k,c.w,c.l,c.short_words,c.delta,c.cols,first,log_columns);
                compute_tail=crt;
                CUgraphNode deps[2]{crt,copy_tail};
                copy_tail=copy(graph,deps,copy_tail?2:1,false,slots[io],c.out,first);
            }
            check(instantiate(&back[io],graph,0));++instantiations;
            cleanup_check(destroy_graph(graph));graph=nullptr;
        }catch(...){if(graph)cleanup_check(destroy_graph(graph));throw;}
    }
    int index(const void* pointer) const noexcept {
        for(int i=0;i<3;++i)if(slots[i]==pointer)return i;
        return -1;
    }
    void close() noexcept {
        if(closed)return;
        closed=true;
        if(stream){
            drain_noexcept();
            for(auto& row:front)for(auto& e:row)if(e){cleanup_check(destroy_exec(e));e=nullptr;}
            for(auto& e:back)if(e){cleanup_check(destroy_exec(e));e=nullptr;}
            cleanup_check(destroy_stream(stream));stream=nullptr;
        }
        available_=false;
    }
    void report() const noexcept {
        std::fprintf(stderr,"{\"column_pipeline\":true,\"columns\":64,\"nodes\":%llu,\"execs\":%llu,\"calls\":%llu,\"launches\":%llu,\"fallbacks\":%llu,\"api_fail\":%llu,\"sync_fail\":%llu,\"cleanup_fail\":%llu}\n",
            (unsigned long long)nodes,(unsigned long long)instantiations,(unsigned long long)calls,
            (unsigned long long)launches,(unsigned long long)fallbacks,(unsigned long long)api_failures,
            (unsigned long long)sync_failures,(unsigned long long)cleanup_failures);
    }
public:
    ColumnStripPipeline(Driver& driver,const ColumnPipelineConfig& config,void* const* prepared):d(driver),c(config){
        for(unsigned i=0;i<3;++i)slots[i]=prepared[i];
        if(c.n!=65536||c.w!=1747||c.l!=55||c.k!=16||c.rows!=256||c.cols!=256||c.group!=8||
           c.logrows!=8||c.outer_threads!=128||c.middle_threads!=256||!c.short_words||c.short_words>16||
           !slots[0]||!slots[1]||!slots[2]||slots[0]==slots[1]||slots[0]==slots[2]||slots[1]==slots[2])
            throw std::runtime_error("column pipeline configuration mismatch");
        create=symbol<decltype(create)>("cuGraphCreate");
        add_memcpy=symbol<decltype(add_memcpy)>("cuGraphAddMemcpyNode");
        add_kernel=symbol<decltype(add_kernel)>("cuGraphAddKernelNode_v2");
        instantiate=symbol<decltype(instantiate)>("cuGraphInstantiateWithFlags");
        launch=symbol<decltype(launch)>("cuGraphLaunch");
        destroy_exec=symbol<decltype(destroy_exec)>("cuGraphExecDestroy");
        destroy_graph=symbol<decltype(destroy_graph)>("cuGraphDestroy");
        create_stream=symbol<decltype(create_stream)>("cuStreamCreate");
        destroy_stream=symbol<decltype(destroy_stream)>("cuStreamDestroy_v2");
        // Only absent optional capability may select fallback at setup. An API
        // failure with present symbols is retained and fails initialization.
        if(!(create&&add_memcpy&&add_kernel&&instantiate&&launch&&destroy_exec&&destroy_graph&&create_stream&&destroy_stream))return;
        try{
            check(create_stream(&stream,CU_STREAM_NON_BLOCKING));
            for(unsigned ia=0;ia<3;++ia)for(unsigned ib=0;ib<3;++ib)if(ia!=ib)make_front(ia,ib);
            for(unsigned io=0;io<3;++io)make_back(io);
            available_=true;
        }catch(...){close();report();throw;}
    }
    ColumnStripPipeline(const ColumnStripPipeline&)=delete;
    ColumnStripPipeline& operator=(const ColumnStripPipeline&)=delete;
    ~ColumnStripPipeline(){close();report();}
    bool accepts(const void* a,const void* b,const void* output){
        const int ia=index(a),ib=index(b),io=index(output);
        if(available_&&ia>=0&&ib>=0&&io>=0&&ia!=ib&&ia!=io&&ib!=io)return true;
        ++fallbacks;return false;
    }
    template<class InitializeOutput> void run(const void* a,const void* b,void* output,InitializeOutput initialize){
        const int ia=index(a),ib=index(b),io=index(output);
        if(!available_||ia<0||ib<0||io<0||ia==ib||ia==io||ib==io)
            throw std::runtime_error("invalid column pipeline slot selection");
        ++calls;
        try{
            check(launch(front[ia][ib],stream));++launches;
            initialize();
            check(launch(back[io],stream));++launches;
            check(d.cuCtxSynchronize());
        }catch(...){drain_noexcept();throw;}
    }
};
