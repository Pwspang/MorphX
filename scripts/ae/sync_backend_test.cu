// Checks the hostmapped GdrPool backend (include/common.h) on a real GPU.
// Mimics the per-block protocol the MorphX pass inserts (SMSched.cpp):
// atomic sub on agents, volatile loads of minSM/maxSM, atomic add on fetched
// and finished. Compares control words in device memory vs the pool.
//
// Build: nvcc -O2 -arch=sm_90 -DMORPHX_SYNC_BACKEND_HOSTMAPPED -I include \
//            -o sync_backend_test scripts/ae/sync_backend_test.cu
#include "common.h"
#include <cstdio>
#include <thread>

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

static Ctl ctl_from_pool(GdrPool &pool, GdrEntry *e) {
    for(int i = 0; i < 4; ++i) e[i] = pool.gdr_malloc();
    GdrEntry maxE = e[1].half();
    pool.set(e[0], 1 << 30);
    pool.set(e[1], 0);
    *(volatile int*)maxE.h = 1 << 30;
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

static float time_protocol(Ctl c, int blocks, int *skipped) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    protocol<<<blocks, 128>>>(c, 0, skipped);   // warm-up
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaEventRecord(a);
    protocol<<<blocks, 128>>>(c, 0, skipped);
    cudaEventRecord(b);
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
    int *skipped;
    CHECK_CUDA_ERROR(cudaMallocManaged(&skipped, sizeof(int)));
    *skipped = 0;

    // 1. Counter correctness under contention.
    GdrEntry e[4];
    Ctl hc = ctl_from_pool(pool, e);
    int blocks = p.multiProcessorCount * 64;
    protocol<<<blocks, 128>>>(hc, 0, skipped);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    int fetched = (int)pool.get(e[2]), finished = (int)pool.get(e[3]);
    int agents = (int)(uint32_t)pool.get(e[0]);
    bool ok = fetched == blocks && finished == blocks && agents == (1 << 30) - blocks;
    fails += !ok;
    printf("[1 counters] blocks=%d fetched=%d finished=%d agents_delta=%d -> %s\n",
           blocks, fetched, finished, (1 << 30) - agents, ok ? "PASS" : "FAIL");

    // 2. Host sees progress while the kernel runs.
    pool.set(e[3], 0);
    protocol<<<p.multiProcessorCount * 8, 128>>>(hc, 20000000, skipped);
    int seen = 0, last = -1;
    while(cudaStreamQuery(0) == cudaErrorNotReady) {
        int f = (int)pool.get(e[3]);
        if(f != last) { ++seen; last = f; }
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    ok = seen > 2;
    fails += !ok;
    printf("[2 live progress] distinct finished values observed while running=%d final=%d -> %s\n",
           seen, (int)pool.get(e[3]), ok ? "PASS" : "FAIL");

    // 3. Host writes reach a running kernel (minSM/maxSM resize path); round-trip latency.
    GdrEntry in = pool.gdr_malloc(), out = pool.gdr_malloc();
    pool.set(in, 0); pool.set(out, 0);
    const int rounds = 2000;
    pingpong<<<1, 1>>>((int*)in.d, (int*)out.d, rounds);
    double t0 = now_us();
    for(int r = 1; r <= rounds; ++r) {
        pool.set(in, (uint32_t)r);
        while((int)(uint32_t)pool.get(out) != r) {}
    }
    double rtt = (now_us() - t0) / rounds;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("[3 host<->kernel] %d round trips, avg %.2f us -> PASS\n", rounds, rtt);

    // 4. Per-block cost of the protocol: device memory vs hostmapped pool.
    Ctl dc = ctl_device();
    for(int mult : {1, 16, 128}) {
        int nb = p.multiProcessorCount * mult;
        float dms = time_protocol(dc, nb, skipped);
        float hms = time_protocol(hc, nb, skipped);
        printf("[4 overhead] blocks=%-6d device=%.3f ms  hostmapped=%.3f ms  (+%.2f us/block)\n",
               nb, dms, hms, (hms - dms) * 1000.0f / nb);
    }

    // 5. maxSM restricts work: with maxSM=half, blocks on SMs >= half are marked skipped.
    *skipped = 0;
    *(volatile int*)e[1].half().h = p.multiProcessorCount / 2;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    protocol<<<p.multiProcessorCount * 4, 128>>>(hc, 100000, skipped);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    ok = *skipped > 0 && *skipped < p.multiProcessorCount * 4;
    fails += !ok;
    printf("[5 maxSM read] maxSM=%d skipped=%d of %d -> %s\n",
           p.multiProcessorCount / 2, *skipped, p.multiProcessorCount * 4, ok ? "PASS" : "FAIL");

    printf("%s\n", fails ? "RESULT: FAIL" : "RESULT: PASS");
    return fails != 0;
}
