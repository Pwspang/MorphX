// Host and GPU cost of patched launches under the MorphX runtime. Build it with
// clang + -fpass-plugin=libsmsched.so (see run_launch_overhead.sh) and run it
// with and without the runtime preloaded. Set SMSCHED_TARGET_PATTERNS=spin_work
// so every launch takes the patched path even when the kernel is short.
//   alone:   GPU time of one launch on an idle stream (events around it; with
//            the runtime this includes waiting for its control words).
//   burst:   N launches back to back, then one sync (best of 3). Reports host
//            enqueue time and wall time per launch.
//   latency: launch + stream sync, N/4 times. Reports the median per iteration.
//   span:    first logical block start to last logical block end (%globaltimer),
//            i.e. the kernel's own execution without launch and queueing delays.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ unsigned long long g_span[2];

__device__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void spin_work(float *a, int n, int iters) {
    if(threadIdx.x == 0) atomicMin(&g_span[0], globaltimer());
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        float v = a[i];
        for(int k = 0; k < iters; ++k) v = v * 0.999f + 0.001f;
        a[i] = v;
    }
    __syncthreads();
    if(threadIdx.x == 0) atomicMax(&g_span[1], globaltimer());
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    int n_launch = argc > 1 ? atoi(argv[1]) : 2000;
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    const int threads = 256;
    struct Shape { const char *name; int blocks; int iters; };
    // tiny: a few us, like decode-time elementwise/GEMV kernels; medium: ~50 us.
    Shape shapes[] = {{"tiny", 64, 64}, {"medium", 1056, 4000}};
    float *a;
    cudaMalloc(&a, 1056 * threads * sizeof(float));
    cudaMemset(a, 0, 1056 * threads * sizeof(float));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for(Shape sh : shapes) {
        int n = sh.blocks * threads;
        for(int w = 0; w < 20; ++w) spin_work<<<sh.blocks, threads, 0, s>>>(a, n, sh.iters);  // warm-up + profile
        cudaStreamSynchronize(s);

        std::vector<double> alone, span;
        for(int r = 0; r < 50; ++r) {
            unsigned long long init[2] = {~0ull, 0};
            cudaMemcpyToSymbol(g_span, init, sizeof(init));
            cudaEventRecord(e0, s);
            spin_work<<<sh.blocks, threads, 0, s>>>(a, n, sh.iters);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            alone.push_back(ms * 1000.0);
            unsigned long long got[2];
            cudaMemcpyFromSymbol(got, g_span, sizeof(got));
            span.push_back((got[1] - got[0]) / 1000.0);
        }

        double best_enq = 1e30, best_wall = 1e30;
        for(int r = 0; r < 3; ++r) {
            double t0 = now_us();
            for(int i = 0; i < n_launch; ++i) spin_work<<<sh.blocks, threads, 0, s>>>(a, n, sh.iters);
            double t1 = now_us();
            cudaStreamSynchronize(s);
            double t2 = now_us();
            best_enq = std::min(best_enq, (t1 - t0) / n_launch);
            best_wall = std::min(best_wall, (t2 - t0) / n_launch);
        }

        std::vector<double> lat;
        for(int i = 0; i < n_launch / 4; ++i) {
            double t3 = now_us();
            spin_work<<<sh.blocks, threads, 0, s>>>(a, n, sh.iters);
            cudaStreamSynchronize(s);
            lat.push_back(now_us() - t3);
        }
        cudaError_t err = cudaGetLastError();
        printf("[%s] blocks=%d alone=%.2f us span=%.2f us | burst: enqueue %.2f us/launch, wall %.2f us/launch | latency median %.2f us | %s\n",
               sh.name, sh.blocks, median(alone), median(span), best_enq, best_wall, median(lat), cudaGetErrorString(err));
    }
    return 0;
}
