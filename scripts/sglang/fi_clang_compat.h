// Force-included by scripts/sglang/fi_clang_nvcc.py into every FlashInfer JIT
// source built with clang (MORPHX_FLASHINFER_PASS=1).
//
// nvcc's math_functions.hpp declares the global ::min/::max overloads (int,
// unsigned, mixed signed/unsigned, 64-bit, float) __host__ __device__, and
// FlashInfer's host-side launch code relies on that (e.g. page.cuh,
// cascade.cuh, prefill.cuh: min(num_blocks_per_sm, ...), max(..., 256)).
// Under clang they are __device__ only. Add host-only versions; device code
// keeps resolving to CUDA's device overloads (clang prefers same-side
// candidates), so kernels are unchanged.
#pragma once
#if defined(__CUDA__) && defined(__clang__)
#include <type_traits>

template <typename A, typename B,
          typename = std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>>>
__host__ inline std::common_type_t<A, B> min(A a, B b) {
  using T = std::common_type_t<A, B>;
  return static_cast<T>(a) < static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}

template <typename A, typename B,
          typename = std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>>>
__host__ inline std::common_type_t<A, B> max(A a, B b) {
  using T = std::common_type_t<A, B>;
  return static_cast<T>(a) > static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}
#endif
