#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <cstring>
#include <mutex>
// Scheduler control words (GdrPool) live in one of three places:
// - default (gdr): device memory mapped to the CPU with GDRCopy.
// - MORPHX_SYNC_BACKEND_HOSTMAPPED: pinned, device-mapped host memory. No
//   gdrdrv needed, but every per-block atomic crosses PCIe.
// - MORPHX_SYNC_BACKEND_MEMCPY: device memory; the CPU reads and writes it with
//   small async copies on a private control stream. No gdrdrv needed and the
//   per-block protocol stays in device memory.
#if !defined(MORPHX_SYNC_BACKEND_HOSTMAPPED) && !defined(MORPHX_SYNC_BACKEND_MEMCPY)
#include <gdrapi.h>
#include <immintrin.h>
#else
#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#endif
#include <chrono>
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <boost/lockfree/queue.hpp>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            assert(0); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CU_ERROR(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char *err_name; \
            cuGetErrorName(err, &err_name); \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << err_name << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            auto cuda_err = cudaGetLastError(); \
            std::cerr << "CUBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(cuda_err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define GDR_POOL_SIZE (GPU_PAGE_SIZE << 10) // 1MB
#define GDR_ENTRY_SIZE 8
#define GDR_BUFFER_SIZE (512) // TODO: 512

struct GdrEntry {
    CUdeviceptr d;
    uintptr_t h;
    GdrEntry half() {
        return {d + GDR_ENTRY_SIZE / 2, h + GDR_ENTRY_SIZE / 2};
    }
};

// Control words of one patched launch: agents, minSM|maxSM and fetched|finished
// (minSM.half() is maxSM, fetched.half() is finished). The memcpy backend keeps
// them contiguous so a launch is initialized with one copy.
struct LaunchSlot {
    GdrEntry agents;
    GdrEntry minSM;
    GdrEntry fetched;
};

// Slot handling for the backends whose host writes land immediately (gdr,
// hostmapped): three independent entries, written before the launch.
#define MORPHX_IMMEDIATE_LAUNCH_SLOTS \
    inline LaunchSlot launch_alloc() { \
        return {gdr_malloc(), gdr_malloc(), gdr_malloc()}; \
    } \
    inline void launch_free(LaunchSlot &slot) { \
        gdr_free(slot.agents); \
        gdr_free(slot.minSM); \
        gdr_free(slot.fetched); \
    } \
    /* withFetched=false (reclaim launches) leaves fetched|finished alone. */ \
    inline void launch_init(LaunchSlot &slot, uint64_t agents, uint64_t minmax, bool withFetched, CUstream) { \
        if(withFetched) set(slot.fetched, 0); \
        set(slot.agents, agents); \
        set(slot.minSM, minmax); \
    } \
    inline void get_many(const GdrEntry *entries, uint64_t *out, int n) { \
        for(int i = 0; i < n; ++i) { \
            GdrEntry entry = entries[i]; \
            out[i] = get(entry); \
        } \
    }

#if !defined(MORPHX_SYNC_BACKEND_HOSTMAPPED) && !defined(MORPHX_SYNC_BACKEND_MEMCPY)
class GdrPool {
private:
    CUdeviceptr d_pool;
    uintptr_t h_pool;
    CUdeviceptr d_alloc;  // cudaMalloc result (d_pool is rounded up from it)
    void *h_map;          // gdr_map result (h_pool is offset from it)
    gdr_t g;
    gdr_mh_t g_mh;
    boost::lockfree::queue<GdrEntry> buffer{GDR_POOL_SIZE / GDR_ENTRY_SIZE};
public:
    GdrPool() {
        g = gdr_open();
        if(g == nullptr) {
            std::cerr << "Failed to open gdr (is /dev/gdrdrv present and accessible?)\n";
            exit(EXIT_FAILURE);
        }
        // gdr_pin_buffer needs a GPU-page (64 KB) aligned range: over-allocate
        // by one page so the rounded-up pool stays inside the allocation.
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_alloc, GDR_POOL_SIZE + GPU_PAGE_SIZE));
        CHECK_CUDA_ERROR(cudaMemset((void*)d_alloc, 0, GDR_POOL_SIZE + GPU_PAGE_SIZE));
        d_pool = (d_alloc + (GPU_PAGE_SIZE - 1)) & ~(CUdeviceptr)(GPU_PAGE_SIZE - 1);
        if(gdr_pin_buffer(g, (unsigned long)d_pool, GDR_POOL_SIZE, 0, 0, &g_mh) != 0) {
            std::cerr << "Failed to pin input buffer\n";
            exit(EXIT_FAILURE);
        }
        if (gdr_map(g, g_mh, &h_map, GDR_POOL_SIZE) != 0) {
            std::cerr << "Failed to map input GPU buffer\n";
            exit(EXIT_FAILURE);
        }
        gdr_info_t info;
        if(gdr_get_info(g, g_mh, &info) != 0) {
            std::cerr << "Failed to get info\n";
            exit(EXIT_FAILURE);
        }
        // The mapping starts at the pinned page (info.va); d_pool is page
        // aligned, so off is 0, but keep the general form.
        long off = (long)(d_pool - info.va);
        h_pool = (uintptr_t)h_map + off;
        // printf("h_pool: %lx, d_pool: %llx\n", h_pool, d_pool);
        for(size_t i = 0; i < GDR_POOL_SIZE; i += GDR_ENTRY_SIZE) {
            GdrEntry entry({d_pool + i, h_pool + i});
            if(!buffer.push(entry)) {
                std::cerr << "GDR pool initialization failed\n";
                exit(EXIT_FAILURE);
            }
        }
    }
    inline GdrEntry gdr_malloc() {
        GdrEntry entry;
        if(!buffer.pop(entry)) {
            std::cerr << "GDR pool is empty\n";
            exit(EXIT_FAILURE);
        }
        return entry;
    }
    inline void gdr_free(GdrEntry entry) {
        if(!buffer.push(entry)) {
            std::cerr << "GDR pool push failed\n";
            exit(EXIT_FAILURE);
        } 
    }

    inline void get_buffer(int bufferId, void *h_buffer) {
        if(gdr_copy_from_mapping(g_mh, (void*)(h_buffer), (void*)(h_pool + bufferId * GDR_BUFFER_SIZE), GDR_BUFFER_SIZE) != 0) {
            std::cerr << "Failed to copy from mapping\n";
            exit(EXIT_FAILURE);
        }
    }
    inline void set_buffer(int bufferId, void *h_buffer) {
        if(gdr_copy_to_mapping(g_mh, (void*)(h_pool + bufferId * GDR_BUFFER_SIZE), (void*)(h_buffer), GDR_BUFFER_SIZE) != 0) {
            std::cerr << "Failed to copy to mapping\n";
            exit(EXIT_FAILURE);
        }
    }

    inline int get_index(GdrEntry &entry) {
        return ((entry.h - h_pool) % GDR_BUFFER_SIZE) / sizeof(int);
    }
    inline int get_bufferId(GdrEntry &entry) {
        return (entry.h - h_pool) / GDR_BUFFER_SIZE;
    }

    // The BAR mapping is write-combining: one aligned 8-byte store, then sfence
    // so it leaves the WC buffer (and, PCIe posted writes being ordered, lands
    // before the doorbell of a following kernel launch). Same effect as
    // gdr_copy_to_mapping/gdr_copy_from_mapping for 8 bytes, minus the call.
    inline void set(GdrEntry &entry, uint64_t value) {
        *(volatile uint64_t*)entry.h = value;
        _mm_sfence();
    }
    inline uint64_t get(GdrEntry &entry) {
        return *(volatile uint64_t*)entry.h;
    }

    // Host writes are visible immediately; nothing to order before a launch.
    inline void launch_fence(CUstream) {}
    MORPHX_IMMEDIATE_LAUNCH_SLOTS

    ~GdrPool() {
        if(gdr_unmap(g, g_mh, h_map, GDR_POOL_SIZE) != 0) {
            std::cerr << "Failed to unmap input GPU buffer";
            exit(EXIT_FAILURE);
        }
        if(gdr_unpin_buffer(g, g_mh) != 0) {
            std::cerr << "Failed to unpin input buffer";
            exit(EXIT_FAILURE);
        }
        if(gdr_close(g) != 0) {
            std::cerr << "Failed to close gdr";
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA_ERROR(cudaFree((void*)d_alloc));
        while(!buffer.empty()) {
            GdrEntry entry;
            buffer.pop(entry);
        }
    }
};
#elif defined(MORPHX_SYNC_BACKEND_HOSTMAPPED)
class GdrPool {
private:
    uintptr_t h_pool;
    CUdeviceptr d_pool;
    boost::lockfree::queue<GdrEntry> buffer{GDR_POOL_SIZE / GDR_ENTRY_SIZE};
public:
    GdrPool() {
        void *h = nullptr;
        CHECK_CUDA_ERROR(cudaHostAlloc(&h, GDR_POOL_SIZE, cudaHostAllocMapped | cudaHostAllocPortable));
        memset(h, 0, GDR_POOL_SIZE);
        void *d = nullptr;
        CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&d, h, 0));
        h_pool = (uintptr_t)h;
        d_pool = (CUdeviceptr)d;
        for(size_t i = 0; i < GDR_POOL_SIZE; i += GDR_ENTRY_SIZE) {
            GdrEntry entry({d_pool + i, h_pool + i});
            if(!buffer.push(entry)) {
                std::cerr << "GDR pool initialization failed\n";
                exit(EXIT_FAILURE);
            }
        }
    }
    inline GdrEntry gdr_malloc() {
        GdrEntry entry;
        if(!buffer.pop(entry)) {
            std::cerr << "GDR pool is empty\n";
            exit(EXIT_FAILURE);
        }
        return entry;
    }
    inline void gdr_free(GdrEntry entry) {
        if(!buffer.push(entry)) {
            std::cerr << "GDR pool push failed\n";
            exit(EXIT_FAILURE);
        }
    }

    inline void get_buffer(int bufferId, void *h_buffer) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        memcpy(h_buffer, (void*)(h_pool + bufferId * GDR_BUFFER_SIZE), GDR_BUFFER_SIZE);
    }
    inline void set_buffer(int bufferId, void *h_buffer) {
        memcpy((void*)(h_pool + bufferId * GDR_BUFFER_SIZE), h_buffer, GDR_BUFFER_SIZE);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    inline int get_index(GdrEntry &entry) {
        return ((entry.h - h_pool) % GDR_BUFFER_SIZE) / sizeof(int);
    }
    inline int get_bufferId(GdrEntry &entry) {
        return (entry.h - h_pool) / GDR_BUFFER_SIZE;
    }

    // Entries are 8-byte aligned, so these are single 64-bit accesses on x86.
    inline void set(GdrEntry &entry, uint64_t value) {
        *(volatile uint64_t*)entry.h = value;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    inline uint64_t get(GdrEntry &entry) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return *(volatile uint64_t*)entry.h;
    }

    // Host writes are visible immediately; nothing to order before a launch.
    inline void launch_fence(CUstream) {}
    MORPHX_IMMEDIATE_LAUNCH_SLOTS

    ~GdrPool() {
        CHECK_CUDA_ERROR(cudaFreeHost((void*)h_pool));
        while(!buffer.empty()) {
            GdrEntry entry;
            buffer.pop(entry);
        }
    }
};
#else
// Device-memory pool driven by async copies on a private, non-blocking control
// stream. All host writes are enqueued there in host order (last writer wins,
// as with GDRCopy); reads copy device -> pinned slot and sync only that stream.
// Only copies are used, never memset: a memset may run as a kernel, which could
// not start while persistent agents occupy every SM.
//
// h_pool is a pinned host mirror of the pool: every write updates the mirror
// before its copy is enqueued, and launch initialization copies straight from
// the mirror, so a copy that executes late still carries the newest value.
//
// A launch's control words (LaunchSlot) are one 32-byte block initialized with
// one copy. MORPHX_MEMCPY_INIT selects where that copy goes:
// - control (default): on the control stream, followed by an event the launch
//   stream waits on (MORPHX_MEMCPY_LAUNCH_SYNC=1 syncs on the host instead).
// - stream: on the launch stream itself, right before the kernel. No event and
//   no lock, but a short copy sits between consecutive kernels. Reads of a
//   block whose copy hasn't run yet must not see the previous launch's
//   counters, so freed blocks get fetched|finished zeroed on the control
//   stream and are only reused once that write is known to have completed.
class GdrPool {
private:
    static constexpr size_t kSlots = 1 << 14;
    static constexpr size_t kBlockSize = 32;
    static constexpr size_t kBlockRegion = GDR_POOL_SIZE / 2;  // blocks live above this offset
    static constexpr size_t kBlocks = (GDR_POOL_SIZE - kBlockRegion) / kBlockSize;
    CUdeviceptr d_pool;
    uintptr_t h_pool;        // pinned host mirror
    uint64_t *write_slots;   // pinned sources of in-flight H2D copies
    uint64_t *read_slots;    // pinned targets of D2H copies
    char *buffer_scratch;    // pinned, for get_buffer/set_buffer
    size_t write_next = 0, read_next = 0;
    uint64_t write_seq = 0;                   // control-stream writes enqueued (under mu)
    std::atomic<uint64_t> completed_seq{0};   // writes known to have landed
    std::vector<uint64_t> block_free_seq;     // write_seq of a block's zeroing
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    CUevent fence_event = nullptr;
    bool launch_sync = false;
    bool init_on_launch_stream = false;
    std::mutex mu;
    boost::lockfree::queue<GdrEntry> buffer{kBlockRegion / GDR_ENTRY_SIZE};
    boost::lockfree::queue<uint32_t> blocks{kBlocks};

    inline void bind_context() {
        CUcontext cur = nullptr;
        cuCtxGetCurrent(&cur);
        if(cur == nullptr) {
            CHECK_CU_ERROR(cuCtxSetCurrent(ctx));
        }
    }
    inline void note_completed(uint64_t seq) {
        uint64_t cur = completed_seq.load(std::memory_order_relaxed);
        while(cur < seq && !completed_seq.compare_exchange_weak(cur, seq)) {}
    }
    // Caller holds mu. A write slot is reused only after a full lap, and the
    // stream is drained at the start of every lap.
    inline void enqueue_write(CUdeviceptr dst, uint64_t value) {
        if(write_next % kSlots == 0 && write_next != 0) {
            CHECK_CU_ERROR(cuStreamSynchronize(stream));
            note_completed(write_seq);
        }
        uint64_t *src = &write_slots[write_next++ % kSlots];
        *src = value;
        CHECK_CU_ERROR(cuMemcpyHtoDAsync(dst, src, sizeof(uint64_t), stream));
        write_seq++;
    }
    inline GdrEntry entry_at(size_t offset) {
        return {d_pool + offset, h_pool + offset};
    }
public:
    GdrPool() {
        void *d = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d, GDR_POOL_SIZE));
        CHECK_CUDA_ERROR(cudaMemset(d, 0, GDR_POOL_SIZE));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        d_pool = (CUdeviceptr)d;
        void *mirror = nullptr;
        CHECK_CUDA_ERROR(cudaHostAlloc(&mirror, GDR_POOL_SIZE, cudaHostAllocPortable));
        memset(mirror, 0, GDR_POOL_SIZE);
        h_pool = (uintptr_t)mirror;
        CHECK_CUDA_ERROR(cudaHostAlloc((void**)&write_slots, kSlots * sizeof(uint64_t), cudaHostAllocPortable));
        CHECK_CUDA_ERROR(cudaHostAlloc((void**)&read_slots, kSlots * sizeof(uint64_t), cudaHostAllocPortable));
        CHECK_CUDA_ERROR(cudaHostAlloc((void**)&buffer_scratch, GDR_BUFFER_SIZE, cudaHostAllocPortable));
        CHECK_CU_ERROR(cuCtxGetCurrent(&ctx));
        int least = 0, greatest = 0;
        CHECK_CU_ERROR(cuCtxGetStreamPriorityRange(&least, &greatest));
        CHECK_CU_ERROR(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, greatest));
        CHECK_CU_ERROR(cuEventCreate(&fence_event, CU_EVENT_DISABLE_TIMING));
        const char *sync = getenv("MORPHX_MEMCPY_LAUNCH_SYNC");
        launch_sync = sync && atoi(sync);
        const char *init = getenv("MORPHX_MEMCPY_INIT");
        init_on_launch_stream = init && strcmp(init, "stream") == 0;
        for(size_t i = 0; i < kBlockRegion; i += GDR_ENTRY_SIZE) {
            if(!buffer.push(entry_at(i))) {
                std::cerr << "GDR pool initialization failed\n";
                exit(EXIT_FAILURE);
            }
        }
        block_free_seq.assign(kBlocks, 0);
        for(uint32_t i = 0; i < kBlocks; ++i) {
            if(!blocks.push(i)) {
                std::cerr << "GDR block pool initialization failed\n";
                exit(EXIT_FAILURE);
            }
        }
    }
    inline GdrEntry gdr_malloc() {
        GdrEntry entry;
        if(!buffer.pop(entry)) {
            std::cerr << "GDR pool is empty\n";
            exit(EXIT_FAILURE);
        }
        return entry;
    }
    inline void gdr_free(GdrEntry entry) {
        if(!buffer.push(entry)) {
            std::cerr << "GDR pool push failed\n";
            exit(EXIT_FAILURE);
        }
    }

    inline LaunchSlot launch_alloc() {
        uint32_t idx;
        if(!blocks.pop(idx)) {
            std::cerr << "GDR block pool is empty\n";
            exit(EXIT_FAILURE);
        }
        // Blocks are recycled FIFO, so the zeroing has almost always landed.
        if(completed_seq.load(std::memory_order_acquire) < block_free_seq[idx]) {
            uint64_t seq;
            {
                std::lock_guard<std::mutex> lock(mu);
                bind_context();
                seq = write_seq;
            }
            CHECK_CU_ERROR(cuStreamSynchronize(stream));
            note_completed(seq);
        }
        size_t base = kBlockRegion + idx * kBlockSize;
        return {entry_at(base), entry_at(base + 8), entry_at(base + 16)};
    }
    inline void launch_free(LaunchSlot &slot) {
        uint32_t idx = (uint32_t)((slot.agents.d - d_pool - kBlockRegion) / kBlockSize);
        if(init_on_launch_stream) {
            std::lock_guard<std::mutex> lock(mu);
            bind_context();
            *(volatile uint64_t*)slot.fetched.h = 0;
            enqueue_write(slot.fetched.d, 0);
            block_free_seq[idx] = write_seq;
        }
        if(!blocks.push(idx)) {
            std::cerr << "GDR block pool push failed\n";
            exit(EXIT_FAILURE);
        }
    }
    // Initialize a launch's words before it starts; withFetched=false (reclaim
    // launches) leaves fetched|finished alone. Later set()s still win: they
    // update the mirror this copy reads from.
    inline void launch_init(LaunchSlot &slot, uint64_t agents, uint64_t minmax, bool withFetched, CUstream launchStream) {
        *(volatile uint64_t*)slot.agents.h = agents;
        *(volatile uint64_t*)slot.minSM.h = minmax;
        if(withFetched) {
            *(volatile uint64_t*)slot.fetched.h = 0;
        }
        size_t bytes = withFetched ? 3 * GDR_ENTRY_SIZE : 2 * GDR_ENTRY_SIZE;
        if(init_on_launch_stream) {
            bind_context();
            CHECK_CU_ERROR(cuMemcpyHtoDAsync(slot.agents.d, (void*)slot.agents.h, bytes, launchStream));
            return;
        }
        std::lock_guard<std::mutex> lock(mu);
        bind_context();
        CHECK_CU_ERROR(cuMemcpyHtoDAsync(slot.agents.d, (void*)slot.agents.h, bytes, stream));
        write_seq++;
        if(launch_sync) {
            CHECK_CU_ERROR(cuStreamSynchronize(stream));
            note_completed(write_seq);
            return;
        }
        CHECK_CU_ERROR(cuEventRecord(fence_event, stream));
        CHECK_CU_ERROR(cuStreamWaitEvent(launchStream, fence_event, 0));
    }

    inline void get_buffer(int bufferId, void *h_buffer) {
        std::lock_guard<std::mutex> lock(mu);
        bind_context();
        CHECK_CU_ERROR(cuMemcpyDtoHAsync(buffer_scratch, d_pool + bufferId * GDR_BUFFER_SIZE, GDR_BUFFER_SIZE, stream));
        CHECK_CU_ERROR(cuStreamSynchronize(stream));
        note_completed(write_seq);
        memcpy(h_buffer, buffer_scratch, GDR_BUFFER_SIZE);
    }
    inline void set_buffer(int bufferId, void *h_buffer) {
        std::lock_guard<std::mutex> lock(mu);
        bind_context();
        memcpy((void*)(h_pool + bufferId * GDR_BUFFER_SIZE), h_buffer, GDR_BUFFER_SIZE);
        memcpy(buffer_scratch, h_buffer, GDR_BUFFER_SIZE);
        CHECK_CU_ERROR(cuMemcpyHtoDAsync(d_pool + bufferId * GDR_BUFFER_SIZE, buffer_scratch, GDR_BUFFER_SIZE, stream));
        write_seq++;
        CHECK_CU_ERROR(cuStreamSynchronize(stream));
        note_completed(write_seq);
    }

    inline int get_index(GdrEntry &entry) {
        return ((entry.h - h_pool) % GDR_BUFFER_SIZE) / sizeof(int);
    }
    inline int get_bufferId(GdrEntry &entry) {
        return (entry.h - h_pool) / GDR_BUFFER_SIZE;
    }

    // Asynchronous: lands on the device in host order after earlier writes.
    inline void set(GdrEntry &entry, uint64_t value) {
        std::lock_guard<std::mutex> lock(mu);
        bind_context();
        *(volatile uint64_t*)entry.h = value;
        enqueue_write(entry.d, value);
    }
    // Sees every write issued before it; the lock is dropped while waiting.
    inline uint64_t get(GdrEntry &entry) {
        uint64_t out;
        get_many(&entry, &out, 1);
        return out;
    }
    // n reads, one sync. Each value sees every write issued before the call.
    inline void get_many(const GdrEntry *entries, uint64_t *out, int n) {
        static constexpr int kMaxReads = 64;
        assert(n <= kMaxReads);
        if(n <= 0) return;
        uint64_t *dst[kMaxReads];
        uint64_t seq;
        {
            std::lock_guard<std::mutex> lock(mu);
            bind_context();
            for(int i = 0; i < n; ++i) {
                dst[i] = &read_slots[read_next++ % kSlots];
                CHECK_CU_ERROR(cuMemcpyDtoHAsync(dst[i], entries[i].d, sizeof(uint64_t), stream));
            }
            seq = write_seq;
        }
        CHECK_CU_ERROR(cuStreamSynchronize(stream));
        note_completed(seq);
        for(int i = 0; i < n; ++i) {
            out[i] = *(volatile uint64_t*)dst[i];
        }
    }
    // Make work later enqueued on launchStream wait for all writes issued so far.
    inline void launch_fence(CUstream launchStream) {
        std::lock_guard<std::mutex> lock(mu);
        bind_context();
        if(launch_sync) {
            CHECK_CU_ERROR(cuStreamSynchronize(stream));
            note_completed(write_seq);
            return;
        }
        CHECK_CU_ERROR(cuEventRecord(fence_event, stream));
        CHECK_CU_ERROR(cuStreamWaitEvent(launchStream, fence_event, 0));
    }

    ~GdrPool() {
        cuStreamSynchronize(stream);
        cuEventDestroy(fence_event);
        cuStreamDestroy(stream);
        cudaFreeHost(write_slots);
        cudaFreeHost(read_slots);
        cudaFreeHost(buffer_scratch);
        cudaFree((void*)d_pool);
        cudaFreeHost((void*)h_pool);
        while(!buffer.empty()) {
            GdrEntry entry;
            buffer.pop(entry);
        }
    }
};
#endif

// Pin the calling (scheduler) thread to one CPU the process may run on. The
// k-th call takes the k-th CPU from the end of the allowed set, so under a batch
// scheduler's cpuset it stays inside the job's CPUs (pinning to CPU 0, 1, ...
// fails there). MORPHX_AGENT_CPU=<n> pins to CPU n, -1 disables pinning.
inline void pinThreadToAllowedCpu(int k) {
    int cpu = -1;
    const char *env = getenv("MORPHX_AGENT_CPU");
    if(env != nullptr && env[0] != '\0') {
        cpu = atoi(env);
        if(cpu < 0) return;
    } else {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if(sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
        int n = CPU_COUNT(&allowed);
        if(n < 2) return;  // don't take the only CPU away from the application
        int want = n - 1 - (k % n);
        for(int c = 0, seen = 0; c < CPU_SETSIZE; ++c) {
            if(CPU_ISSET(c, &allowed) && seen++ == want) {
                cpu = c;
                break;
            }
        }
        if(cpu < 0) return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
    }
}

// SM count of the current device; replaces the A100's 108 that used to be hardcoded.
inline int getDeviceSMCount() {
    static int nrSM = 0;
    if(nrSM == 0) {
        int dev = 0;
        CHECK_CUDA_ERROR(cudaGetDevice(&dev));
        CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&nrSM, cudaDevAttrMultiProcessorCount, dev));
    }
    return nrSM;
}

// Default split hints were tuned on a 108-SM A100; scale them to this GPU.
inline int scaleSplitHint(int a100Hint, int nrSM) {
    return (a100Hint * nrSM + 54) / 108;
}

typedef void (*fixSMHandle)(cudaStream_t, int, int);
typedef void (*suggestSMHandle)(cudaStream_t, int, int);

fixSMHandle getFixSMHandle() {
    void *handle = dlopen(nullptr, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error opening handle: " << dlerror() << std::endl;
        return nullptr;
    }
    fixSMHandle func = (fixSMHandle)dlsym(handle, "fix_SM");
    if (!func) {
        std::cerr << "Error loading symbol: " << dlerror() << std::endl;
        return nullptr;
    }
    return func;
}

suggestSMHandle getSuggestSMHandle() {
    void *handle = dlopen(nullptr, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error opening handle: " << dlerror() << std::endl;
        return nullptr;
    }
    suggestSMHandle func = (suggestSMHandle)dlsym(handle, "suggest_SM");
    if (!func) {
        std::cerr << "Error loading symbol: " << dlerror() << std::endl;
        return nullptr;
    }
    return func;
}

bool fixSMForStream(cudaStream_t stream, int minSM, int maxSM) {
    fixSMHandle func = getFixSMHandle();
    if (func) {
        func(stream, minSM, maxSM);
        return true;
    }
    return false;
}

bool suggestSMForStream(cudaStream_t stream, int minSM, int maxSM) {
    suggestSMHandle func = getSuggestSMHandle();
    if (func) {
        func(stream, minSM, maxSM);
        return true;
    }
    return false;
}