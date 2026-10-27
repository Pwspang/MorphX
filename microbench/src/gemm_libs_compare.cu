// Correctness and timing of libmycutlass.so builds against cuBLAS (fp32
// accumulate), for the myCutlassHgemm contract: fp16, all column-major,
// C = alpha * A * B + beta * C. Libraries are dlopen'ed, so the Sm80 and sm90
// builds (same symbol) can be compared in one process. Build with nvcc:
//   nvcc -O2 -arch=sm_90 -o gemm_libs_compare microbench/src/gemm_libs_compare.cu -lcublas -ldl
//
//   gemm_libs_compare compare <dir1>[,<dir2>...] m,n,k [m,n,k ...]
//       one line per shape: cuBLAS ms, then ms / max error / PASS|FAIL / host us
//       per call per library
//   gemm_libs_compare tune <sm90 dir> m,n,k [m,n,k ...]
//       CSV, every config of the sm90 build (myCutlassSm90Run) on every shape
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

typedef void (*HgemmFn)(int, int, int, const void *, long long, const void *, long long, void *, long long,
                        float, float, cudaStream_t);
typedef int (*Sm90RunFn)(int, int, int, int, const void *, long long, const void *, long long, void *, long long,
                         float, float, cudaStream_t);
typedef int (*Sm90NumFn)();
typedef const char *(*Sm90NameFn)(int);
typedef int (*Sm90SelectFn)(int, int, int);

__global__ void fill(__half *p, size_t n, unsigned seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    p[i] = __float2half(((h & 0xffff) / 65535.f - 0.5f) * scale);
  }
}

// out[0] = max |c - ref| (as float bits, NaN -> +inf), out[1] = max |ref|
__global__ void maxdiff(const __half *c, const __half *ref, size_t n, unsigned *out) {
  float e = 0.f, r = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float x = __half2float(c[i]), y = __half2float(ref[i]);
    float d = fabsf(x - y);
    e = fmaxf(e, isnan(d) ? INFINITY : d);
    r = fmaxf(r, fabsf(y));
  }
  atomicMax(&out[0], __float_as_uint(e));
  atomicMax(&out[1], __float_as_uint(r));
}

struct Problem {
  int m, n, k;
  __half *a, *b, *c, *ref;
  unsigned *diff;
};

static cudaStream_t gStream;
static cublasHandle_t gHandle;

static void cublas_gemm(const Problem &p) {
  const float one = 1.f, zero = 0.f;
  CHECK_CUBLAS(cublasGemmEx(gHandle, CUBLAS_OP_N, CUBLAS_OP_N, p.m, p.n, p.k, &one, p.a, CUDA_R_16F, p.m,
                            p.b, CUDA_R_16F, p.k, &zero, p.ref, CUDA_R_16F, p.m, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

static void check(const Problem &p, float *err, float *ref) {
  CHECK_CUDA(cudaMemsetAsync(p.diff, 0, 8, gStream));
  maxdiff<<<1024, 256, 0, gStream>>>(p.c, p.ref, (size_t)p.m * p.n, p.diff);
  unsigned h[2];
  CHECK_CUDA(cudaMemcpyAsync(h, p.diff, 8, cudaMemcpyDeviceToHost, gStream));
  CHECK_CUDA(cudaStreamSynchronize(gStream));
  memcpy(err, &h[0], 4);
  memcpy(ref, &h[1], 4);
}

template <class F>
static float time_ms(F fn) {
  cudaEvent_t e0, e1;
  CHECK_CUDA(cudaEventCreate(&e0));
  CHECK_CUDA(cudaEventCreate(&e1));
  for (int i = 0; i < 2; ++i) fn();
  CHECK_CUDA(cudaEventRecord(e0, gStream));
  fn();
  CHECK_CUDA(cudaEventRecord(e1, gStream));
  CHECK_CUDA(cudaEventSynchronize(e1));
  float once;
  CHECK_CUDA(cudaEventElapsedTime(&once, e0, e1));
  int iters = once > 0 ? (int)(20.f / once) : 50;
  iters = iters < 5 ? 5 : iters > 50 ? 50 : iters;
  CHECK_CUDA(cudaEventRecord(e0, gStream));
  for (int i = 0; i < iters; ++i) fn();
  CHECK_CUDA(cudaEventRecord(e1, gStream));
  CHECK_CUDA(cudaEventSynchronize(e1));
  float ms;
  CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
  CHECK_CUDA(cudaEventDestroy(e0));
  CHECK_CUDA(cudaEventDestroy(e1));
  return ms / iters;
}

static void *open_lib(const std::string &dir) {
  std::string path = dir + "/libmycutlass.so";
  void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "dlopen %s: %s\n", path.c_str(), dlerror()); exit(1); }
  return h;
}

static bool ok(float err, float ref) { return err <= 1e-2f * ref + 1e-3f; }

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s compare|tune <lib dirs> m,n,k ...\n", argv[0]);
    return 2;
  }
  std::string mode = argv[1];
  std::vector<std::string> dirs;
  for (char *s = strtok(argv[2], ","); s; s = strtok(nullptr, ",")) dirs.push_back(s);
  CHECK_CUDA(cudaStreamCreateWithFlags(&gStream, cudaStreamNonBlocking));
  CHECK_CUBLAS(cublasCreate(&gHandle));
  CHECK_CUBLAS(cublasSetStream(gHandle, gStream));

  std::vector<void *> libs;
  for (auto &d : dirs) libs.push_back(open_lib(d));
  if (mode == "tune") {
    printf("m,n,k,config,name,ms,max_err,max_ref,status,cublas_ms\n");
  } else {
    printf("%-22s %10s", "m,n,k", "cuBLAS_ms");
    for (auto &d : dirs) printf(" | %-52s", d.substr(d.find_last_of('/') + 1).c_str());
    printf("\n");
  }
  fflush(stdout);
  for (int s = 3; s < argc; ++s) {
    Problem p{};
    if (sscanf(argv[s], "%d,%d,%d", &p.m, &p.n, &p.k) != 3) { fprintf(stderr, "bad shape %s\n", argv[s]); return 2; }
    size_t sa = (size_t)p.m * p.k, sb = (size_t)p.k * p.n, sc = (size_t)p.m * p.n;
    CHECK_CUDA(cudaMalloc(&p.a, sa * 2));
    CHECK_CUDA(cudaMalloc(&p.b, sb * 2));
    CHECK_CUDA(cudaMalloc(&p.c, sc * 2));
    CHECK_CUDA(cudaMalloc(&p.ref, sc * 2));
    CHECK_CUDA(cudaMalloc(&p.diff, 8));
    fill<<<1024, 256, 0, gStream>>>(p.a, sa, 1u, 1.f);
    fill<<<1024, 256, 0, gStream>>>(p.b, sb, 2u, 1.f);
    cublas_gemm(p);
    float tCublas = time_ms([&] { cublas_gemm(p); });
    if (mode == "tune") {
      auto run = (Sm90RunFn)dlsym(libs[0], "myCutlassSm90Run");
      auto num = (Sm90NumFn)dlsym(libs[0], "myCutlassSm90NumConfigs");
      auto name = (Sm90NameFn)dlsym(libs[0], "myCutlassSm90ConfigName");
      if (!run || !num || !name) { fprintf(stderr, "%s is not an sm90 build\n", dirs[0].c_str()); return 2; }
      for (int cfg = 0; cfg < num(); ++cfg) {
        CHECK_CUDA(cudaMemsetAsync(p.c, 0xff, sc * 2, gStream));  // NaN: unwritten output fails
        int rc = run(cfg, p.m, p.n, p.k, p.a, p.m, p.b, p.k, p.c, p.m, 1.f, 0.f, gStream);
        if (rc != 0) {
          printf("%d,%d,%d,%d,%s,,,,unsupported,%.4f\n", p.m, p.n, p.k, cfg, name(cfg), tCublas);
          continue;
        }
        float err, ref;
        check(p, &err, &ref);
        float ms = time_ms([&] { run(cfg, p.m, p.n, p.k, p.a, p.m, p.b, p.k, p.c, p.m, 1.f, 0.f, gStream); });
        printf("%d,%d,%d,%d,%s,%.4f,%.4g,%.4g,%s,%.4f\n", p.m, p.n, p.k, cfg, name(cfg), ms, err, ref,
               ok(err, ref) ? "PASS" : "FAIL", tCublas);
        fflush(stdout);
      }
    } else {
      printf("%-22s %10.4f", argv[s], tCublas);
      for (size_t l = 0; l < libs.size(); ++l) {
        auto hgemm = (HgemmFn)dlsym(libs[l], "myCutlassHgemm");
        auto select = (Sm90SelectFn)dlsym(libs[l], "myCutlassSm90SelectConfig");
        auto name = (Sm90NameFn)dlsym(libs[l], "myCutlassSm90ConfigName");
        CHECK_CUDA(cudaMemsetAsync(p.c, 0xff, sc * 2, gStream));
        hgemm(p.m, p.n, p.k, p.a, p.m, p.b, p.k, p.c, p.m, 1.f, 0.f, gStream);
        float err, ref;
        check(p, &err, &ref);
        float ms = time_ms([&] { hgemm(p.m, p.n, p.k, p.a, p.m, p.b, p.k, p.c, p.m, 1.f, 0.f, gStream); });
        // host cost of one call: issue 50 without waiting (the queue absorbs them)
        auto h0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 50; ++i) hgemm(p.m, p.n, p.k, p.a, p.m, p.b, p.k, p.c, p.m, 1.f, 0.f, gStream);
        double host_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count() / 50;
        CHECK_CUDA(cudaStreamSynchronize(gStream));
        char cfg[64] = "";
        if (select && name) snprintf(cfg, sizeof(cfg), " %s", name(select(p.m, p.n, p.k)));
        char cell[128];
        snprintf(cell, sizeof(cell), "%.4f err=%.3g %s host=%.1fus%s", ms, err, ok(err, ref) ? "PASS" : "FAIL", host_us, cfg);
        printf(" | %-52s", cell);
      }
      printf("\n");
      fflush(stdout);
    }
    CHECK_CUDA(cudaFree(p.a));
    CHECK_CUDA(cudaFree(p.b));
    CHECK_CUDA(cudaFree(p.c));
    CHECK_CUDA(cudaFree(p.ref));
    CHECK_CUDA(cudaFree(p.diff));
  }
  return 0;
}
