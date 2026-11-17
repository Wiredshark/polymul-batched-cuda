#pragma once
// This header defines replaceable allocation functions in exactly ONE TU.
// Explicit Scope may grow storage in a current call. A public-init reserve
// may also supply ordinary exact-size allocations, using FREE prepared slots
// only. That generic path never allocates/pins/copies or examines payloads.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
namespace owning_pinned_vector {
struct Backend {
    void* opaque=nullptr;
    bool (*pin)(void*,void*,std::size_t) noexcept=nullptr;
    bool (*unpin)(void*,void*,std::size_t) noexcept=nullptr;
};
struct Stats;
struct Slot {
    void* address=nullptr;std::size_t size=0,requested_bytes=0;
    std::uint64_t owner_identity=0;Stats* owner_stats=nullptr;
    bool in_use=false,pinned=false,retiring=false,retired=false,quarantined=false;
    Slot* registry_next=nullptr;Slot* pool_next=nullptr;
};
inline std::mutex registry_mutex;
inline Slot* registry=nullptr;
class Pool;
inline thread_local Pool* selected_pool=nullptr;
inline thread_local std::uint64_t prepared_owner_identity=0;
inline thread_local unsigned prepared_bypass_depth=0;
inline std::uint64_t next_pool_identity=0;
inline std::uint64_t reserve_pool_identity(){
    std::lock_guard<std::mutex> guard(registry_mutex);
    if(next_pool_identity==std::numeric_limits<std::uint64_t>::max())throw std::bad_alloc();
    return ++next_pool_identity;
}
class Bypass {
    Pool* previous;
public:
    Bypass() noexcept:previous(selected_pool){selected_pool=nullptr;++prepared_bypass_depth;}
    ~Bypass(){--prepared_bypass_depth;selected_pool=previous;}
};
inline void unlink_locked(Slot* target) noexcept {
    Slot** link=&registry;while(*link&&*link!=target)link=&(*link)->registry_next;
    if(*link!=target)std::abort();
    *link=target->registry_next;
}
inline void destroy_slot(Slot* slot) noexcept {std::free(slot->address);std::free(slot);}
struct Stats {std::uint64_t borrows=0,hits=0,growth=0,pin_failures=0,unpin_failures=0,quarantined_slots=0,prepared_hits=0,prepared_misses=0;};
class Pool {
    const std::size_t requested,rounded;
    Backend backend;Slot* slots=nullptr;bool active=true;
    const std::uint64_t identity;
    static std::size_t rounded_size(std::size_t bytes){
        if(!bytes||bytes>std::numeric_limits<std::size_t>::max()-4095)throw std::bad_alloc();
        return (bytes+4095)&~std::size_t(4095);
    }
    Slot* create_slot(bool in_use){
        Bypass bypass;
        Slot* node=static_cast<Slot*>(std::malloc(sizeof(Slot)));if(!node)throw std::bad_alloc();
        ::new(static_cast<void*>(node))Slot{};
        void* address=nullptr;
        if(posix_memalign(&address,4096,rounded)!=0){std::free(node);throw std::bad_alloc();}
        node->address=address;node->size=rounded;node->in_use=in_use;
        node->requested_bytes=requested;node->owner_identity=identity;node->owner_stats=&stats;
        if(!backend.pin(backend.opaque,address,rounded)){
            ++stats.pin_failures;destroy_slot(node);throw std::bad_alloc();
        }
        node->pinned=true;
        {std::lock_guard<std::mutex> guard(registry_mutex);
            node->registry_next=registry;registry=node;
            node->pool_next=slots;slots=node;
        }
        return node;
    }
public:
    Stats stats;
    Pool(Backend driver,std::size_t bytes,std::size_t initial_slots=1)
        :requested(bytes),rounded(rounded_size(bytes)),backend(driver),identity(reserve_pool_identity()){
        if(!backend.pin||!backend.unpin||!initial_slots)throw std::invalid_argument("invalid pinned pool backend");
        try{for(std::size_t i=0;i<initial_slots;++i)create_slot(false);}
        catch(...){retire();throw;}
    }
    Pool(const Pool&)=delete;Pool& operator=(const Pool&)=delete;
    ~Pool(){retire();}
    // Setup-only metadata snapshot; never reads or acquires operand storage.
    bool snapshot_empty_pinned_slots(void** addresses,std::size_t count)const noexcept{
        std::lock_guard<std::mutex> guard(registry_mutex);
        if(!active||!addresses||!count)return false;
        std::size_t i=0;
        for(Slot* s=slots;s;s=s->pool_next){
            if(i>=count||s->in_use||!s->pinned||s->retiring||s->retired||s->quarantined)return false;
            addresses[i++]=s->address;
        }
        return i==count;
    }
    bool matches(std::size_t bytes)const noexcept{return bytes==requested;}
    void activate_prepared_reserve(){
        {
            std::lock_guard<std::mutex> guard(registry_mutex);
            if(active){prepared_owner_identity=identity;return;}
        }
        // Constructing an exception can call replacement operator new.
        // Never allocate while holding the allocation registry mutex.
        throw std::runtime_error("inactive prepared arena");
    }
    bool owns_live_pinned(const void* address,std::size_t bytes)const noexcept{
        std::lock_guard<std::mutex> guard(registry_mutex);
        if(!active||bytes!=requested)return false;
        for(Slot* s=slots;s;s=s->pool_next)
            if(s->address==address&&s->in_use&&s->pinned&&!s->retiring&&!s->retired&&!s->quarantined)return true;
        return false;
    }
    void* acquire(){
        {std::lock_guard<std::mutex> guard(registry_mutex);
            if(!active)throw std::bad_alloc();
            ++stats.borrows;
            for(Slot* s=slots;s;s=s->pool_next)if(!s->in_use){
                s->in_use=true;++stats.hits;return s->address;
            }
            ++stats.growth;
        }
        // Concurrent use/free of one solver state is outside its existing API.
        // Growth is input-independent allocation work INSIDE the current call.
        return create_slot(true)->address;
    }
    void retire(bool quiescent=true) noexcept {
        {std::lock_guard<std::mutex> guard(registry_mutex);
            if(!active)return;
            active=false;
            if(prepared_owner_identity==identity)prepared_owner_identity=0;
            for(Slot* s=slots;s;s=s->pool_next){s->retiring=true;s->owner_stats=nullptr;}
        }
        // CUDA completion is the caller's required precondition before retire.
        // Never hold the registry mutex across driver calls or malloc/free.
        Bypass bypass;Slot* s=slots;slots=nullptr;
        while(s){
            Slot* next=s->pool_next;
            const bool unpinned=quiescent&&backend.unpin(backend.opaque,s->address,s->size);
            if(quiescent&&!unpinned)++stats.unpin_failures;
            if(!unpinned)++stats.quarantined_slots;
            bool reclaim=false;
            {std::lock_guard<std::mutex> guard(registry_mutex);
                s->pinned=!unpinned;s->quarantined=!unpinned;
                s->retiring=false;s->retired=true;
                if(!s->in_use&&unpinned){unlink_locked(s);reclaim=true;}
            }
            if(reclaim)destroy_slot(s);
            s=next;
        }
        // Live ordinary vectors retain their storage. Their later delete frees
        // it without touching the now-destroyable CUDA owner or this Pool.
    }
};
class Scope {
    Pool* previous;
public:
    explicit Scope(Pool& pool)noexcept:previous(selected_pool){selected_pool=&pool;}
    ~Scope(){selected_pool=previous;}
    Scope(const Scope&)=delete;
};
inline void release(void* address) noexcept {
    if(!address)return;
    Slot* found=nullptr;bool reclaim=false;
    {std::lock_guard<std::mutex> guard(registry_mutex);
        for(Slot* s=registry;s;s=s->registry_next)if(s->address==address){found=s;break;}
        if(found){
            if(!found->in_use)std::abort();
            found->in_use=false;
            if(found->retired&&!found->retiring&&!found->pinned&&!found->quarantined){
                unlink_locked(found);reclaim=true;
            }
        }
    }
    if(reclaim)destroy_slot(found);else if(!found)std::free(address);
}
inline void* try_prepared_allocate(std::size_t bytes)noexcept{
    if(!prepared_owner_identity||prepared_bypass_depth)return nullptr;
    std::lock_guard<std::mutex> guard(registry_mutex);
    Stats* matching=nullptr;
    for(Slot* s=registry;s;s=s->registry_next){
        if(s->owner_identity!=prepared_owner_identity||s->requested_bytes!=bytes||!s->owner_stats||!s->pinned||s->retiring||s->retired||s->quarantined)continue;
        matching=s->owner_stats;
        if(!s->in_use){
            s->in_use=true;++matching->borrows;++matching->hits;++matching->prepared_hits;
            return s->address;
        }
    }
    if(matching)++matching->prepared_misses;
    return nullptr;
}
inline void* ordinary_allocate(std::size_t bytes){
    for(;;){if(void* p=std::malloc(bytes?bytes:1))return p;
        auto handler=std::get_new_handler();if(!handler)throw std::bad_alloc();handler();}
}
}
void* operator new(std::size_t bytes){
    auto* pool=owning_pinned_vector::selected_pool;
    if(pool&&pool->matches(bytes))return pool->acquire();
    if(void* prepared=owning_pinned_vector::try_prepared_allocate(bytes))return prepared;
    return owning_pinned_vector::ordinary_allocate(bytes);
}
void operator delete(void* address) noexcept {owning_pinned_vector::release(address);}
void operator delete(void* address,std::size_t) noexcept {owning_pinned_vector::release(address);}
void* operator new[](std::size_t bytes){return ::operator new(bytes);}
void operator delete[](void* address) noexcept {owning_pinned_vector::release(address);}
void operator delete[](void* address,std::size_t) noexcept {owning_pinned_vector::release(address);}
