// Checks a GdrPool sync backend (include/common.h) on a real GPU.
// Mimics the per-block protocol the MorphX pass inserts (SMSched.cpp):
// atomic sub on agents, volatile loads of minSM/maxSM, atomic add on fetched
// and finished. Compares control words in device memory vs the pool.
//
// Build (pick one backend define):
//   nvcc -O2 -arch=sm_90 -DMORPHX_SYNC_BACKEND_HOSTMAPPED -I include -lcuda \
//        -o sync_backend_test scripts/ae/sync_backend_test.cu
//   nvcc -O2 -arch=sm_90 -DMORPHX_SYNC_BACKEND_MEMCPY -I include -lcuda ...
// With the memcpy backend, run it with MORPHX_MEMCPY_INIT=control (default) and
// MORPHX_MEMCPY_INIT=stream to cover both launch_init() paths.
#include "common.h"
#include <cstdio>
#include <thread>
#include <vector>

struct Ctl {
    int *agents, *minSM, *maxSM, *fetched, *finished;
};

__global__ void protocol(Ctl c, long long spin, int *skipped) {
    if(threadIdx.x == 0) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        atomicSub(c.agents, 1);
        int lo = *(volatile int*)c.minSM, hi = *(volatile int*)c.maxSM;
        if((int)smid < lo || (int)smid >= hi) {
            atomicAdd(skipped, 1);
        }
        atomicAdd(c.fetched, 1);
        long long t0 = clock64();
        while(clock64() - t0 < spin) {}
        atomicAdd(c.finished, 1);
    }
    __syncthreads();
}

// One block waits for the host to write a value into *in, then echoes it to *out.
__global__ void pingpong(volatile int *in, volatile int *out, int rounds) {
    for(int r = 1; r <= rounds; ++r) {
        while(*in != r) {}
        __threadfence_system();
        *out = r;
    }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t pack(uint32_t lo, uint32_t hi) {
    return (uint64_t)hi << 32 | lo;
}

// Every pool write the next launch depends on goes through set() + launch_fence().
static Ctl ctl_from_pool(GdrPool &pool, GdrEntry *e) {
    for(int i = 0; i < 4; ++i) e[i] = pool.gdr_malloc();
    GdrEntry maxE = e[1].half();
    pool.set(e[0], 1 << 30);
    pool.set(e[1], pack(0, 1 << 30));
    pool.set(e[2], 0);
    pool.set(e[3], 0);
    return {(int*)e[0].d, (int*)e[1].d, (int*)maxE.d, (int*)e[2].d, (int*)e[3].d};
}

static Ctl ctl_device() {
    int *d;
    CHECK_CUDA_ERROR(cudaMalloc(&d, 5 * sizeof(int)));
    int h[5] = {1 << 30, 0, 1 << 30, 0, 0};
    CHECK_CUDA_ERROR(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice));
    return {d, d + 1, d + 2, d + 3, d + 4};
}

// Reads the value the host wrote right before the launch.
__global__ void read_back(const int *src, int *dst) {
    *dst = *(volatile const int*)src;
}

// Reads a LaunchSlot the way a patched kernel would at its first block.
__global__ void read_slot(const uint64_t *agents, const uint64_t *minmax, const uint64_t *fetched, uint64_t *dst) {
    dst[0] = *(volatile const uint64_t*)agents;
    dst[1] = *(volatile const uint64_t*)minmax;
    dst[2] = *(volatile const uint64_t*)fetched;
}

// Leaves non-zero counters behind, like a finished patched launch.
__global__ void dirty_slot(int *fetched, int *finished, int n) {
    atomicAdd(fetched, n);
    atomicAdd(finished, n);
}

static float time_protocol(Ctl c, int blocks, int *skipped, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    protocol<<<blocks, 128, 0, s>>>(c, 0, skipped);   // warm-up
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaEventRecord(a, s);
    protocol<<<blocks, 128, 0, s>>>(c, 0, skipped);
    cudaEventRecord(b, s);
    CHECK_CUDA_ERROR(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

int main() {
    int fails = 0;
    cudaDeviceProp p;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&p, 0));
    int native = 0, pageable = 0;
    cudaDeviceGetAttribute(&native, cudaDevAttrHostNativeAtomicSupported, 0);
    cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0);
    printf("[env] %s cc=%d.%d sms=%d hostNativeAtomic=%d pageableMemoryAccess=%d\n",
           p.name, p.major, p.minor, p.multiProcessorCount, native, pageable);

    GdrPool pool;
    cudaStream_t s;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int *skipped;
    CHECK_CUDA_ERROR(cudaMallocManaged(&skipped, sizeof(int)));
    *skipped = 0;

    // 1. Counter correctness under contention.
    GdrEntry e[4];
    Ctl hc = ctl_from_pool(pool, e);
    int blocks = p.multiProcessorCount * 64;
    pool.launch_fence(s);
    protocol<<<blocks, 128, 0, s>>>(hc, 0, skipped);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    int fetched = (int)pool.get(e[2]), finished = (int)pool.get(e[3]);
    int agents = (int)(uint32_t)pool.get(e[0]);
    bool ok = fetched == blocks && finished == blocks && agents == (1 << 30) - blocks;
    fails += !ok;
    printf("[1 counters] blocks=%d fetched=%d finished=%d agents_delta=%d -> %s\n",
           blocks, fetched, finished, (1 << 30) - agents, ok ? "PASS" : "FAIL");

    // 2. Host sees progress while the kernel runs. Use many short waves: blocks
    //    that all fit at once would finish together and look like one step.
    pool.set(e[3], 0);
    pool.launch_fence(s);
    const int live_blocks = p.multiProcessorCount * 128;
    protocol<<<live_blocks, 128, 0, s>>>(hc, 2000000, skipped);
    int seen = 0, last = -1;
    while(cudaStreamQuery(s) == cudaErrorNotReady) {
        int f = (int)pool.get(e[3]);
        if(f != last) { ++seen; last = f; }
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    ok = seen > 2;
    fails += !ok;
    printf("[2 live progress] distinct finished values observed while running=%d final=%d/%d -> %s\n",
           seen, (int)pool.get(e[3]), live_blocks, ok ? "PASS" : "FAIL");

    // 3. Host writes reach a running kernel (minSM/maxSM resize path); round-trip latency.
    GdrEntry in = pool.gdr_malloc(), out = pool.gdr_malloc();
    pool.set(in, 0); pool.set(out, 0);
    const int rounds = 2000;
    pool.launch_fence(s);
    pingpong<<<1, 1, 0, s>>>((int*)in.d, (int*)out.d, rounds);
    double t0 = now_us();
    for(int r = 1; r <= rounds; ++r) {
        pool.set(in, (uint32_t)r);
        while((int)(uint32_t)pool.get(out) != r) {}
    }
    double rtt = (now_us() - t0) / rounds;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("[3 host<->kernel] %d round trips, avg %.2f us -> PASS\n", rounds, rtt);

    // 4. Per-block cost of the protocol: device memory vs the pool.
    Ctl dc = ctl_device();
    for(int mult : {1, 16, 128}) {
        int nb = p.multiProcessorCount * mult;
        float dms = time_protocol(dc, nb, skipped, s);
        float hms = time_protocol(hc, nb, skipped, s);
        printf("[4 overhead] blocks=%-6d device=%.3f ms  pool=%.3f ms  (+%.2f us/block)\n",
               nb, dms, hms, (hms - dms) * 1000.0f / nb);
    }

    // 5. maxSM restricts work: with maxSM=half, blocks on SMs >= half are marked skipped.
    *skipped = 0;
    pool.set(e[1], pack(0, p.multiProcessorCount / 2));
    pool.launch_fence(s);
    protocol<<<p.multiProcessorCount * 4, 128, 0, s>>>(hc, 100000, skipped);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    ok = *skipped > 0 && *skipped < p.multiProcessorCount * 4;
    fails += !ok;
    printf("[5 maxSM read] maxSM=%d skipped=%d of %d -> %s\n",
           p.multiProcessorCount / 2, *skipped, p.multiProcessorCount * 4, ok ? "PASS" : "FAIL");

    // 6. Launch ordering: a kernel sees at least the write issued right before
    //    its launch on a busy stream, with no host sync in between. Later writes
    //    may already have landed too (last writer wins, as with GDRCopy).
    const int iters = 200;
    int *seen_vals;
    CHECK_CUDA_ERROR(cudaMallocManaged(&seen_vals, iters * sizeof(int)));
    GdrEntry v = pool.gdr_malloc();
    pool.set(e[3], 0);
    pool.launch_fence(s);
    protocol<<<p.multiProcessorCount * 8, 128, 0, s>>>(hc, 2000000, skipped);  // keep s busy
    double tl = now_us();
    for(int i = 0; i < iters; ++i) {
        pool.set(v, 1000 + i);
        pool.launch_fence(s);
        read_back<<<1, 1, 0, s>>>((int*)v.d, &seen_vals[i]);
    }
    double per_launch = (now_us() - tl) / iters;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    int wrong = 0;
    for(int i = 0; i < iters; ++i) wrong += seen_vals[i] < 1000 + i;
    ok = wrong == 0;
    fails += !ok;
    printf("[6 launch order] %d set+fence+launch, wrong=%d, host cost %.2f us/launch -> %s\n",
           iters, wrong, per_launch, ok ? "PASS" : "FAIL");

    // 7. Host cost of one read and one write.
    const int n = 5000;
    double tr = now_us();
    for(int i = 0; i < n; ++i) pool.get(v);
    tr = (now_us() - tr) / n;
    double tw = now_us();
    for(int i = 0; i < n; ++i) pool.set(v, i);
    pool.get(v);
    tw = (now_us() - tw) / n;
    printf("[7 host cost] get %.2f us, set %.2f us\n", tr, tw);

    // 8. launch_init(): every launch on a busy stream sees exactly its own
    //    slot's agents, minSM|maxSM and fetched|finished=0, even when the slot
    //    was used and freed with non-zero counters just before.
    const int nslots = 200;
    std::vector<LaunchSlot> slots(nslots);
    for(int i = 0; i < nslots; ++i) {
        slots[i] = pool.launch_alloc();
        pool.launch_init(slots[i], 0, 0, true, s);
        dirty_slot<<<1, 1, 0, s>>>((int*)slots[i].fetched.d, (int*)slots[i].fetched.half().d, 1000 + i);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    for(int i = 0; i < nslots; ++i) pool.launch_free(slots[i]);
    // MORPHX_MEMCPY_INIT=stream zeroes fetched|finished on free (its reads can
    // run ahead of a reused block's init copy); the other modes don't need to.
    int zeroed = 0;
    for(int i = 0; i < nslots; ++i) zeroed += pool.get(slots[i].fetched) == 0;
    const char *init_mode = getenv("MORPHX_MEMCPY_INIT");
    bool want_zeroed = init_mode && strcmp(init_mode, "stream") == 0;
#ifndef MORPHX_SYNC_BACKEND_MEMCPY
    want_zeroed = false;
#endif
    uint64_t *slot_vals;
    CHECK_CUDA_ERROR(cudaMallocManaged(&slot_vals, nslots * 3 * sizeof(uint64_t)));
    pool.set(e[3], 0);
    pool.launch_fence(s);
    protocol<<<p.multiProcessorCount * 8, 128, 0, s>>>(hc, 2000000, skipped);  // keep s busy
    int stale = 0;
    double ti = now_us();
    for(int i = 0; i < nslots; ++i) {
        slots[i] = pool.launch_alloc();
        pool.launch_init(slots[i], pack(1000 + i, 0), pack(i, i + 1), true, s);
        read_slot<<<1, 1, 0, s>>>((uint64_t*)slots[i].agents.d, (uint64_t*)slots[i].minSM.d,
                                  (uint64_t*)slots[i].fetched.d, &slot_vals[3 * i]);
    }
    double per_init = (now_us() - ti) / nslots;
    // Before the busy kernel ends, no reader may see the previous counters.
    for(int i = 0; i < nslots; i += 20) stale += pool.get(slots[i].fetched) != 0;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    int bad = 0;
    for(int i = 0; i < nslots; ++i) {
        bad += slot_vals[3 * i] != pack(1000 + i, 0) || slot_vals[3 * i + 1] != pack(i, i + 1) || slot_vals[3 * i + 2] != 0;
    }
    ok = bad == 0 && stale == 0 && (!want_zeroed || zeroed == nslots);
    fails += !ok;
    printf("[8 launch_init] %d slots, wrong=%d stale_reads=%d zeroed_on_free=%d/%d, host cost %.2f us/(init+launch) -> %s\n",
           nslots, bad, stale, zeroed, nslots, per_init, ok ? "PASS" : "FAIL");

    // 9. A set() issued after launch_init() but before the kernel starts wins.
    pool.launch_fence(s);
    protocol<<<p.multiProcessorCount * 8, 128, 0, s>>>(hc, 2000000, skipped);  // keep s busy
    for(int i = 0; i < nslots; ++i) {
        pool.launch_init(slots[i], pack(7, 0), pack(0, 1), true, s);
        pool.set(slots[i].minSM, pack(i, 2 * i + 2));
        read_slot<<<1, 1, 0, s>>>((uint64_t*)slots[i].agents.d, (uint64_t*)slots[i].minSM.d,
                                  (uint64_t*)slots[i].fetched.d, &slot_vals[3 * i]);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    bad = 0;
    for(int i = 0; i < nslots; ++i) bad += slot_vals[3 * i + 1] != pack(i, 2 * i + 2);
    ok = bad == 0;
    fails += !ok;
    printf("[9 late set wins] %d slots, wrong=%d -> %s\n", nslots, bad, ok ? "PASS" : "FAIL");
    for(int i = 0; i < nslots; ++i) pool.launch_free(slots[i]);

    // 10. Host cost of one batched read of n entries.
    for(int nr : {1, 2, 4}) {
        std::vector<GdrEntry> es(nr, v);
        std::vector<uint64_t> outs(nr);
        double tg = now_us();
        for(int i = 0; i < n; ++i) pool.get_many(es.data(), outs.data(), nr);
        tg = (now_us() - tg) / n;
        printf("[10 get_many] n=%d %.2f us\n", nr, tg);
    }

    printf("%s\n", fails ? "RESULT: FAIL" : "RESULT: PASS");
    return fails != 0;
}
