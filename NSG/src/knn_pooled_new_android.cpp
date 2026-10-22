// Linked only into Android nsg_knn. Batch small C++ allocations to avoid
// exhausting vm.max_map_count in Scudo during million-node initialization.
// Slabs are retained for reuse until process exit; large allocations use malloc.
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <pthread.h>

namespace {
constexpr std::size_t kStep = 64;
constexpr std::size_t kClasses = 128;
constexpr std::size_t kSlabBytes = 1024 * 1024;
struct alignas(std::max_align_t) Header {
    Header* next;
    std::size_t bucket;
};
static_assert(kStep % alignof(Header) == 0, "Allocation alignment");
struct Bucket {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    Header* free = nullptr;
};
// No dynamic C++ allocation or destructor is needed by this allocator itself.
Bucket buckets[kClasses];

void* try_allocate(std::size_t size) noexcept {
    if (size == 0) size = 1;
    if (size > std::numeric_limits<std::size_t>::max() - sizeof(Header))
        return nullptr;
    const std::size_t bytes = size + sizeof(Header);
    if (bytes > kStep * kClasses) {
        auto* header = static_cast<Header*>(std::malloc(bytes));
        if (!header) return nullptr;
        header->bucket = kClasses;
        return header + 1;
    }
    const std::size_t slot = (bytes - 1) / kStep;
    Bucket& bucket = buckets[slot];
    pthread_mutex_lock(&bucket.mutex);
    if (!bucket.free) {
        const std::size_t stride = (slot + 1) * kStep;
        const std::size_t count = kSlabBytes / stride;
        auto* slab = static_cast<unsigned char*>(std::malloc(count * stride));
        if (!slab) {
            pthread_mutex_unlock(&bucket.mutex);
            return nullptr;
        }
        for (std::size_t i = 0; i < count; ++i) {
            auto* header = reinterpret_cast<Header*>(slab + i * stride);
            header->bucket = slot;
            header->next = bucket.free;
            bucket.free = header;
        }
    }
    Header* header = bucket.free;
    bucket.free = header->next;
    pthread_mutex_unlock(&bucket.mutex);
    return header + 1;
}

void release(void* pointer) noexcept {
    if (!pointer) return;
    auto* header = static_cast<Header*>(pointer) - 1;
    if (header->bucket == kClasses) {
        std::free(header);
        return;
    }
    Bucket& bucket = buckets[header->bucket];
    pthread_mutex_lock(&bucket.mutex);
    header->next = bucket.free;
    bucket.free = header;
    pthread_mutex_unlock(&bucket.mutex);
}
} // namespace

void* operator new(std::size_t size) {
    for (;;) {
        if (void* pointer = try_allocate(size)) return pointer;
        std::new_handler handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t) noexcept { release(p); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try { return ::operator new(size); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return ::operator new(size, std::nothrow);
}
void operator delete(void* p, const std::nothrow_t&) noexcept { release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { release(p); }
