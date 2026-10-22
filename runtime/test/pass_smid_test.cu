// End-to-end check of the MorphX pass + runtime: build with clang++ and
// -fpass-plugin=libsmsched.so, run with the runtime hook preloaded. Each block
// records %smid. With PASS_TEST_MIN/MAX set, fix_SM() feeds [min, max) into the
// runtime's first (profiling) launch, which then runs on [0, max) only; later
// launches of a lone stream get the whole GPU from the scheduler, so the range
// check is only enforced for launch 0 (PASS_TEST_STRICT=1 enforces it always).
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <set>
#include <vector>

__global__ void spin_add(const float *a, float *c, unsigned *smid, int n, long long spin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long t0 = clock64();
    while(clock64() - t0 < spin) {}
    if(i < n) c[i] = a[i] + 1.0f;
    if(threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        smid[blockIdx.x] = s;
    }
}

int main() {
    const int threads = 256, blocks = 132 * 16, n = blocks * threads;
    int lo = getenv("PASS_TEST_MIN") ? atoi(getenv("PASS_TEST_MIN")) : -1;
    int hi = getenv("PASS_TEST_MAX") ? atoi(getenv("PASS_TEST_MAX")) : -1;
    long long spin = getenv("PASS_TEST_SPIN") ? atoll(getenv("PASS_TEST_SPIN")) : 200000;
    bool strict = getenv("PASS_TEST_STRICT") && atoi(getenv("PASS_TEST_STRICT"));
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if(lo >= 0) {
        // cuda.cpp defines fix_SM outside its extern "C" block, so it is C++-mangled.
        void *sym = dlsym(RTLD_DEFAULT, "fix_SM");
        if(!sym) sym = dlsym(RTLD_DEFAULT, "_Z6fix_SMP11CUstream_stii");
        auto fix = (void (*)(cudaStream_t, int, int))sym;
        printf("fix_SM %s, range [%d,%d)\n", fix ? "found" : "NOT FOUND (runtime not preloaded?)", lo, hi);
        if(fix) fix(s, lo, hi);
    }
    float *a, *c;
    unsigned *smid;
    cudaMalloc(&a, n * 4); cudaMalloc(&c, n * 4); cudaMalloc(&smid, blocks * 4);
    std::vector<float> ha(n, 1.0f), hc(n);
    std::vector<unsigned> hs(blocks);
    cudaMemcpy(a, ha.data(), n * 4, cudaMemcpyHostToDevice);
    int fails = 0;
    for(int it = 0; it < 6; ++it) {
        cudaMemsetAsync(c, 0, n * 4, s);
        cudaMemsetAsync(smid, 0xff, blocks * 4, s);
        spin_add<<<blocks, threads, 0, s>>>(a, c, smid, n, spin);
        cudaError_t e = cudaStreamSynchronize(s);
        cudaMemcpy(hc.data(), c, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hs.data(), smid, blocks * 4, cudaMemcpyDeviceToHost);
        int wrong = 0, ran = 0, outside = 0;
        std::set<unsigned> ids;
        for(int i = 0; i < n; ++i) wrong += hc[i] != 2.0f;
        for(unsigned v : hs) {
            if(v == 0xffffffffu) continue;
            ++ran; ids.insert(v);
            if(lo >= 0 && (int)v >= hi) ++outside;
        }
        bool ok = e == cudaSuccess && wrong == 0 && ran == blocks && (outside == 0 || (it > 0 && !strict));
        fails += !ok;
        printf("[launch %d] err=%s wrong=%d blocks_ran=%d/%d sms=%zu min=%d max=%d outside_range=%d -> %s\n",
               it, cudaGetErrorString(e), wrong, ran, blocks, ids.size(),
               ids.empty() ? -1 : (int)*ids.begin(), ids.empty() ? -1 : (int)*ids.rbegin(), outside, ok ? "PASS" : "FAIL");
    }
    printf("RESULT: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
