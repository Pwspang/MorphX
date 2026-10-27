// Spike: a Hopper-native (WGMMA + TMA) CUTLASS 3 GEMM built with clang and the
// SMSched pass, checked against cuBLAS, with and without the MorphX runtime.
// Same contract as myCutlassHgemm: fp16 in/out, fp32 accumulate, all matrices
// column-major, D = alpha * A * B + beta * C on a caller-provided stream.
// Non-warp-specialized mainloop (KernelTma) and a 1x1x1 cluster, so the kernel
// is an ordinary grid of independent CTAs, which is what the pass assumes.
//
//   ./sm90_spike M N K [iters]          (LD_PRELOAD the runtime to test _pk)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

template <class TileShape>
struct Sm90Gemm {
  using ClusterShape = Shape<_1, _1, _1>;
  using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;
  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      ElementC, StrideC, StrideC,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAcc, ElementAcc>,
      cutlass::gemm::EpilogueDefault>;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTma>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static void run(int m, int n, int k, const ElementA *a, const ElementB *b, ElementC *c,
                  float alpha, float beta, cudaStream_t stream) {
    auto sA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, cute::make_shape(m, k, 1));
    auto sB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, cute::make_shape(n, k, 1));
    auto sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {a, sA, b, sB},
        {{alpha, beta}, c, sC, c, sC}};
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { fprintf(stderr, "can_implement failed\n"); exit(1); }
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) { fprintf(stderr, "initialize failed\n"); exit(1); }
    if (gemm.run(stream) != cutlass::Status::kSuccess) { fprintf(stderr, "run failed\n"); exit(1); }
  }
};

using Gemm128x128 = Sm90Gemm<Shape<_128, _128, _64>>;

int main(int argc, char **argv) {
  int m = argc > 1 ? atoi(argv[1]) : 4096;
  int n = argc > 2 ? atoi(argv[2]) : 4096;
  int k = argc > 3 ? atoi(argv[3]) : 4096;
  int iters = argc > 4 ? atoi(argv[4]) : 20;
  size_t sa = (size_t)m * k, sb = (size_t)k * n, sc = (size_t)m * n;

  std::vector<ElementA> hA(sa), hB(sb);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto &x : hA) x = ElementA(dist(rng));
  for (auto &x : hB) x = ElementB(dist(rng));
  ElementA *dA; ElementB *dB; ElementC *dC, *dRef;
  CHECK_CUDA(cudaMalloc(&dA, sa * sizeof(ElementA)));
  CHECK_CUDA(cudaMalloc(&dB, sb * sizeof(ElementB)));
  CHECK_CUDA(cudaMalloc(&dC, sc * sizeof(ElementC)));
  CHECK_CUDA(cudaMalloc(&dRef, sc * sizeof(ElementC)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sa * sizeof(ElementA), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sb * sizeof(ElementB), cudaMemcpyHostToDevice));
  // MorphX doesn't support patched kernels on the default stream.
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetStream(handle, stream));
  const __half one = __float2half(1.f), zero = __float2half(0.f);
  auto cublas = [&]() {
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one,
                             (const __half *)dA, m, (const __half *)dB, k, &zero, (__half *)dRef, m));
  };
  auto ours = [&]() { Gemm128x128::run(m, n, k, dA, dB, dC, 1.f, 0.f, stream); };

  cublas();
  ours();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<ElementC> hC(sc), hRef(sc);
  CHECK_CUDA(cudaMemcpy(hC.data(), dC, sc * sizeof(ElementC), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hRef.data(), dRef, sc * sizeof(ElementC), cudaMemcpyDeviceToHost));
  double maxErr = 0, maxRef = 0;
  for (size_t i = 0; i < sc; ++i) {
    maxErr = std::max(maxErr, (double)std::fabs(float(hC[i]) - float(hRef[i])));
    maxRef = std::max(maxRef, (double)std::fabs(float(hRef[i])));
  }

  cudaEvent_t e0, e1;
  CHECK_CUDA(cudaEventCreate(&e0));
  CHECK_CUDA(cudaEventCreate(&e1));
  auto timeit = [&](auto fn) {
    for (int i = 0; i < 3; ++i) fn();
    CHECK_CUDA(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; ++i) fn();
    CHECK_CUDA(cudaEventRecord(e1, stream));
    CHECK_CUDA(cudaEventSynchronize(e1));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    return ms / iters;
  };
  float tCublas = timeit(cublas), tOurs = timeit(ours);
  double tflops = 2.0 * m * n * k / (tOurs * 1e-3) / 1e12;
  printf("M=%d N=%d K=%d | sm90 128x128x64 %.3f ms (%.0f TFLOP/s) | cuBLAS %.3f ms | max|err| %.4g (max|ref| %.4g) -> %s\n",
         m, n, k, tOurs, tflops, tCublas, maxErr, maxRef, maxErr <= 0.02 * maxRef + 0.05 ? "PASS" : "FAIL");
  return 0;
}
