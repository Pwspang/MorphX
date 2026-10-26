// Checks the Orion-style hook (runtime/src/cuda-orion.cpp): kernels on two
// best-effort streams must not overlap (they are serialized), a high-priority
// stream must still overlap with them, and results stay correct through both
// <<<>>> and cudaLaunchKernelExC. Each kernel records its first-block start and
// last-block end with %globaltimer.
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Small grid so two kernels could easily co-run without the hook.
__global__ void spin(float *out, unsigned long long *span, long long cycles) {
    unsigned long long t0 = now_ns();
    long long c0 = clock64();
    while(clock64() - c0 < cycles) {}
    if(threadIdx.x == 0) {
        out[blockIdx.x] = blockIdx.x + 1.0f;
        atomicMin(&span[0], t0);
        atomicMax(&span[1], now_ns());
    }
}

struct Span { unsigned long long start, end; };

static bool overlap(Span a, Span b) { return a.start < b.end && b.start < a.end; }

int main() {
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStream_t be1, be2, hp;
    cudaStreamCreateWithPriority(&be1, cudaStreamNonBlocking, lo);
    cudaStreamCreateWithPriority(&be2, cudaStreamNonBlocking, lo);
    cudaStreamCreateWithPriority(&hp, cudaStreamNonBlocking, hi);
    const int blocks = 8, n = 3;
    const long long cycles = 200000000;  // ~100 ms
    float *out[n];
    unsigned long long *span[n];
    for(int i = 0; i < n; ++i) {
        cudaMalloc(&out[i], blocks * sizeof(float));
        cudaMemset(out[i], 0, blocks * sizeof(float));
        cudaMalloc(&span[i], 2 * sizeof(unsigned long long));
        unsigned long long init[2] = {~0ull, 0};
        cudaMemcpy(span[i], init, sizeof(init), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    // be1: <<<>>>; be2: cudaLaunchKernelExC; hp: <<<>>>, launched last.
    spin<<<blocks, 32, 0, be1>>>(out[0], span[0], cycles);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(32);
    cfg.stream = be2;
    void *args[] = {&out[1], &span[1], (void *)&cycles};
    cudaLaunchKernelExC(&cfg, (const void *)spin, args);
    spin<<<blocks, 32, 0, hp>>>(out[2], span[2], cycles);
    cudaError_t err = cudaDeviceSynchronize();

    Span s[n];
    int wrong = 0;
    for(int i = 0; i < n; ++i) {
        cudaMemcpy(&s[i], span[i], sizeof(Span), cudaMemcpyDeviceToHost);
        std::vector<float> h(blocks);
        cudaMemcpy(h.data(), out[i], blocks * sizeof(float), cudaMemcpyDeviceToHost);
        for(int b = 0; b < blocks; ++b) wrong += h[b] != b + 1.0f;
    }
    bool be_overlap = overlap(s[0], s[1]);
    bool hp_overlap = overlap(s[2], s[0]) || overlap(s[2], s[1]);
    printf("err=%s wrong=%d\n", cudaGetErrorString(err), wrong);
    for(int i = 0; i < n; ++i)
        printf("kernel %d (%s): %.1f-%.1f ms\n", i, i == 2 ? "high" : "best-effort",
               (s[i].start - s[0].start) / 1e6, (s[i].end - s[0].start) / 1e6);
    printf("best-effort kernels overlap: %s\n", be_overlap ? "yes" : "no");
    printf("high-priority overlaps best-effort: %s\n", hp_overlap ? "yes" : "no");
    // Expected under the hook: no BE overlap, HP overlap. Without it: both overlap.
    const char *expect = getenv("ORION_EXPECT");
    bool hooked = expect && expect[0] == 'h';
    bool ok = err == cudaSuccess && wrong == 0 && hp_overlap && (hooked ? !be_overlap : be_overlap);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
