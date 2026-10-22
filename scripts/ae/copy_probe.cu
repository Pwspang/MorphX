// Does a small copy on a side stream complete while every SM is busy?
// Times several flavours with a spin kernel filling the GPU (sync backend design).
// Build: nvcc -O2 -arch=sm_90 -o copy_probe scripts/ae/copy_probe.cu -lcuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { CUresult r = (x); if(r) { const char *n; cuGetErrorName(r, &n); printf("%s:%d %s\n", __FILE__, __LINE__, n); exit(1);} } while(0)

__global__ void spin(unsigned long long ns, int *ctr) {
    unsigned long long t0 = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while(true) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if(t - t0 > ns) break;
    }
    if(threadIdx.x == 0) atomicAdd(ctr, 1);
}

static double us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    cudaFree(0);
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, spin, 1024, 0);
    CUstream work, ctl;
    int lo, hi;
    CK(cuCtxGetStreamPriorityRange(&lo, &hi));
    CK(cuStreamCreateWithPriority(&work, CU_STREAM_NON_BLOCKING, 0));
    CK(cuStreamCreateWithPriority(&ctl, CU_STREAM_NON_BLOCKING, hi));
    int *ctr;
    cudaMalloc(&ctr, 4096);
    cudaMemset(ctr, 0, 4096);
    uint64_t *pin;
    cudaHostAlloc((void**)&pin, 1 << 20, 0);
    cudaDeviceSynchronize();
    printf("sms=%d occ(1024 thr)=%d\n", sms, occ);
    for(int full = 0; full < 2; ++full) {
        int blocks = full ? sms * occ * 2 : 1;   // full: 2 waves that fill every SM
        cudaMemset(ctr, 0, 4);
        cudaDeviceSynchronize();
        spin<<<blocks, 1024, 0, (cudaStream_t)work>>>(200000000ULL, ctr);   // 200 ms per wave
        double t0 = us();
        while(us() - t0 < 20000) {}
        double a0 = us(); CK(cuMemcpyDtoHAsync(pin, (CUdeviceptr)ctr, 8, ctl)); CK(cuStreamSynchronize(ctl)); double a = us() - a0;
        double b0 = us(); CK(cuMemcpyDtoHAsync(pin + 64, (CUdeviceptr)ctr, 4096, ctl)); CK(cuStreamSynchronize(ctl)); double b = us() - b0;
        pin[1000] = 7;
        double c0 = us(); CK(cuMemcpyHtoDAsync((CUdeviceptr)(ctr + 512), &pin[1000], 8, ctl)); CK(cuStreamSynchronize(ctl)); double c = us() - c0;
        double d0 = us(); CK(cuStreamWriteValue64(ctl, (CUdeviceptr)(ctr + 600), 42, 0)); CK(cuStreamSynchronize(ctl)); double d = us() - d0;
        double f0 = us(); CK(cuMemcpyAsync((CUdeviceptr)(pin + 2000), (CUdeviceptr)ctr, 8, ctl)); CK(cuStreamSynchronize(ctl)); double f = us() - f0;
        CUmemcpyAttributes attr = {};
        attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        attr.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
        CUdeviceptr dsts[1] = {(CUdeviceptr)(pin + 3000)};
        CUdeviceptr srcs[1] = {(CUdeviceptr)ctr};
        size_t sizes[1] = {8};
        size_t attrIdx[1] = {0};
        double g0 = us();
        CUresult gr = cuMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrIdx, 1, ctl);
        CK(cuStreamSynchronize(ctl));
        double g = us() - g0;
        double elapsed_ms = (us() - t0) / 1000;
        CK(cuStreamSynchronize(work));
        printf("[%s] blocks=%d ctr_seen=%llu (copies done %.1f ms after launch)\n"
               "  DtoH8=%.1fus DtoH4K=%.1fus HtoD8=%.1fus writeValue64=%.1fus generic8=%.1fus batch_overlap8=%.1fus(rc=%d)\n",
               full ? "GPU full" : "GPU idle", blocks, (unsigned long long)(pin[0] & 0xffffffff), elapsed_ms,
               a, b, c, d, f, g, (int)gr);
    }
    return 0;
}
