#pragma once
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#if defined(__linux__)
#include <sys/mman.h>
#endif

// A fixed alignment opportunity, not a claim about the host's actual THP size
// or backing. Bounds refer only to the requested output, never spare capacity.
constexpr std::uintptr_t OUTPUT_THP_ALIGNMENT=std::uintptr_t(2)*1024*1024;
struct OutputThpInterior {
    std::uintptr_t first=0;
    std::size_t bytes=0;
    bool valid=false;
};
inline OutputThpInterior output_thp_unused_interior(const void* data,
    std::size_t live_words,std::size_t required_words,std::size_t capacity_words)noexcept{
    static_assert(sizeof(std::uint32_t)==4,"output words must have four bytes");
    if(!data||!live_words||live_words>required_words||required_words>capacity_words||
       required_words>std::numeric_limits<std::size_t>::max()/4)return {};
    const std::size_t live_bytes=live_words*4,required_bytes=required_words*4;
    const auto base=reinterpret_cast<std::uintptr_t>(data);
    const auto maximum=std::numeric_limits<std::uintptr_t>::max();
    if(required_bytes>maximum||base>maximum-required_bytes)return {};
    const std::uintptr_t live_end=base+live_bytes,end=base+required_bytes;
    const std::uintptr_t mask=OUTPUT_THP_ALIGNMENT-1;
    if(live_end>maximum-mask)return {};
    const std::uintptr_t first=(live_end+mask)&~mask,last=end&~mask;
    if(first>=last)return {0,0,true};
    if(last-first>std::numeric_limits<std::size_t>::max())return {};
    return {first,std::size_t(last-first),true};
}

struct OutputThpHintStats {
    std::uint64_t calls=0,attempts=0,successes=0,failures=0,attempted_bytes=0;
    std::uint64_t empty=0,invalid=0,unsupported=0;
    int last_error=0;
    ~OutputThpHintStats(){
        if(!calls)return;
        // Metadata only; advice success never means huge pages were created.
        std::fprintf(stderr,"{\"output_thp\":true,\"c\":%llu,\"a\":%llu,\"ok\":%llu,\"fail\":%llu,\"b\":%llu,\"empty\":%llu,\"invalid\":%llu,\"unsupported\":%llu,\"err\":%d}\n",
            (unsigned long long)calls,(unsigned long long)attempts,(unsigned long long)successes,
            (unsigned long long)failures,(unsigned long long)attempted_bytes,
            (unsigned long long)empty,(unsigned long long)invalid,(unsigned long long)unsupported,last_error);
    }
};

inline void advise_unused_output_thp(OutputThpHintStats& stats,void* data,
    std::size_t live_words,std::size_t required_words,std::size_t capacity_words,
    std::size_t base_page_bytes)noexcept{
    ++stats.calls;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if(!base_page_bytes||base_page_bytes>OUTPUT_THP_ALIGNMENT||OUTPUT_THP_ALIGNMENT%base_page_bytes){
        ++stats.unsupported;return;
    }
    const auto interior=output_thp_unused_interior(data,live_words,required_words,capacity_words);
    if(!interior.valid){++stats.invalid;return;}
    if(!interior.bytes){++stats.empty;return;}
    ++stats.attempts;stats.attempted_bytes+=interior.bytes;
    const int saved_errno=errno;
    const int status=::madvise(reinterpret_cast<void*>(interior.first),interior.bytes,MADV_HUGEPAGE);
    const int advice_error=errno;errno=saved_errno;
    if(status==0)++stats.successes;
    else{++stats.failures;stats.last_error=advice_error;}
    // Even a failed advice can have changed part of the mapping policy. Keep
    // ordinary registration/copies and do not pretend NOHUGEPAGE restores the
    // unknown prior policy. No payload is read, written or retained here.
#else
    (void)data;(void)live_words;(void)required_words;(void)capacity_words;(void)base_page_bytes;
    ++stats.unsupported;
#endif
}
