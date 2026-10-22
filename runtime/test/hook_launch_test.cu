// End-to-end check of the MorphX runtime hook (libcuda.so + libpreload.so) on a
// real GPU, without the LLVM pass: vec_add_pk is a hand-written stand-in for
// what smsched-pass generates (same trailing arguments and block protocol).
//
// Checks, run under LD_PRELOAD with SMSCHED_DEBUG=1:
//   1. <<<>>> launch of vec_add is redirected to vec_add_pk and still correct
//   2. cudaLaunchKernelExC without attributes takes the same path
//   3. cudaLaunchKernelExC with a cluster attribute keeps its cluster size
//   4. a kernel without a _pk variant runs unmodified
// Build and run: runtime/test/run_hook_test.sh
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void vec_add(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) a[i] += 1.0f;
}

// Mirrors smsched-pass: original args, then dim3 grid, agents, fetched,
// finished, minSM, maxSM. Blocks pull logical block ids from *fetched and
// leave when their SM falls outside [minSM, maxSM).
extern "C" __global__ void vec_add_pk(float *a, int n, dim3 grid, int *agents, int *fetched,
                                      int *finished, const int *minSM, const int *maxSM) {
    extern __shared__ int smem[];
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    int nBlocks = grid.x * grid.y * grid.z;
    bool leader = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
    if(leader) atomicSub(agents, 1);
    while(true) {
        if(leader) {
            unsigned lo = *(volatile const int*)minSM, hi = *(volatile const int*)maxSM;
            if(smid < lo || smid >= hi) {
                smem[0] = nBlocks;
                while(atomicAdd(agents, 0) != 0) {}
            } else {
                smem[0] = atomicAdd(fetched, 1);
            }
        }
        __syncthreads();
        int bid = smem[0];
        if(bid >= nBlocks) return;
        int i = bid * blockDim.x + threadIdx.x;
        if(i < n) a[i] += 1.0f;
        __syncthreads();
        if(leader) atomicAdd(finished, 1);
    }
}

extern "C" __global__ void cluster_size(unsigned *out) {
    unsigned n;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n));
    if(threadIdx.x == 0) out[blockIdx.x] = n;
}

extern "C" __global__ void no_pk(int *out) {
    if(threadIdx.x == 0) atomicAdd(out, 1);
}

#define CHECK(call) do { cudaError_t e = (call); if(e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

static int check_vec(const char *tag, float *d, int n, float expect) {
    std::vector<float> h(n);
    CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for(int i = 0; i < n; ++i) bad += h[i] != expect;
    printf("[%s] %s (%d/%d wrong)\n", tag, bad ? "FAIL" : "PASS", bad, n);
    return bad != 0;
}

int main() {
    int fails = 0;
    const int n = 1 << 20, threads = 256, blocks = (n + threads - 1) / threads;
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float *a;
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMemset(a, 0, n * sizeof(float)));

    // 1. triple-chevron launch
    vec_add<<<blocks, threads, 0, s>>>(a, n);
    CHECK(cudaStreamSynchronize(s));
    fails += check_vec("1 <<<>>> vec_add", a, n, 1.0f);

    // 2. cudaLaunchKernelExC, no attributes
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(threads);
    cfg.stream = s;
    int nArg = n;
    void *args[] = {&a, &nArg};
    CHECK(cudaLaunchKernelExC(&cfg, (const void*)vec_add, args));
    CHECK(cudaStreamSynchronize(s));
    fails += check_vec("2 cudaLaunchKernelExC vec_add", a, n, 2.0f);

    // 3. cluster attribute must survive the hook
    unsigned *cl;
    const int clBlocks = 64;
    CHECK(cudaMalloc(&cl, clBlocks * sizeof(unsigned)));
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t ccfg = {};
    ccfg.gridDim = dim3(clBlocks);
    ccfg.blockDim = dim3(32);
    ccfg.stream = s;
    ccfg.attrs = &attr;
    ccfg.numAttrs = 1;
    void *cargs[] = {&cl};
    CHECK(cudaLaunchKernelExC(&ccfg, (const void*)cluster_size, cargs));
    CHECK(cudaStreamSynchronize(s));
    std::vector<unsigned> hcl(clBlocks);
    CHECK(cudaMemcpy(hcl.data(), cl, clBlocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
    int badCl = 0;
    for(unsigned v : hcl) badCl += v != 2;
    printf("[3 cluster attr kept] %s (cluster size %u)\n", badCl ? "FAIL" : "PASS", hcl[0]);
    fails += badCl != 0;

    // 4. kernel without a _pk variant
    int *cnt;
    CHECK(cudaMalloc(&cnt, sizeof(int)));
    CHECK(cudaMemset(cnt, 0, sizeof(int)));
    no_pk<<<100, 32, 0, s>>>(cnt);
    CHECK(cudaStreamSynchronize(s));
    int hcnt = 0;
    CHECK(cudaMemcpy(&hcnt, cnt, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[4 unpatched no_pk] %s (%d/100 blocks)\n", hcnt == 100 ? "PASS" : "FAIL", hcnt);
    fails += hcnt != 100;

    printf("RESULT: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
