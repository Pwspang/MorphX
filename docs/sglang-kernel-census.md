# SGLang kernel census (stock wheels, H200)

Input to the SGLang go/no-go (plan §9 step 1): which kernels does SGLang run
for a dense model, and how much of the GPU time is in kernels that the MorphX
LLVM pass could instrument?

**Setup:** hpcc `hpc-gaas-g25`, NVIDIA H200 (132 SMs), driver 580.126.20,
job 91232 (2026-10-22). Env `~/morphx/envs/sglang-stock`
(`scripts/sglang/install_stock_env.sh`): SGLang from `~/sglang-main`
(snapshot 2026-04-29), torch 2.9.1+cu130, flashinfer-python/-cubin
0.6.8.post1, sglang-kernel 0.4.1.post1+cu130, transformers 5.6.0, triton
3.5.1. Model Qwen3-4B (bf16). In-process runner (`scripts/sglang/runner.py`):
attention backend `flashinfer`, CUDA graphs, radix cache, chunked prefill and
the overlap scheduler off. Measured with `torch.profiler` over one prefill
batch (8 × 512 tokens) and 16 decode steps (batch 8), via
`scripts/sglang/smoke_runner.py`. **No MorphX pass or runtime involved.**

## Summary

| | Prefill | Decode |
|---|---:|---:|
| cuBLAS/cuBLASLt GEMMs (`nvjet_*`, closed source) | **80.2%** | **69.6%** |
| CUDA C++ from source (FlashInfer attention, sglang jit kernels, PyTorch native) | 16.9% | 25.1% |
| CuTe DSL (FlashInfer RMSNorm / fused add-RMSNorm, Python → MLIR JIT) | 2.9% | 5.2% |
| Triton | 0.0% | 0.1% |
| Kernel launches | 464 per batch | ~554 per step |

All LLM kernels ran on the caller's CUDA stream (CUPTI stream 13, the same as
a marker kernel launched on it); none on the legacy default stream.

## What this means for MorphX

1. **The GEMMs are the problem.** With stock wheels, every linear layer is
   `aten::mm` → cuBLASLt `nvjet_*` kernels: 70–80% of GPU time, closed
   source, so the pass cannot add the SM-range checks MorphX needs. The
   original artifact's runtime targets CUTLASS kernels by name
   (`cutlassL6Kernel` = mangled `cutlass::Kernel`, plus mangled CUTLASS
   template fragments in `runtime/src/cuda-llm.cpp:646-656`), which suggests
   the authors' stack ran the LLM GEMMs as CUTLASS kernels compiled from
   source. For SGLang the bf16 linear layers must be routed to a GEMM built
   with clang + the pass (e.g. a CUTLASS GEMM op replacing `aten::mm` in
   SGLang's `UnquantizedLinearMethod`, or a PyTorch build whose `mm` uses
   CUTLASS). **This is the main go/no-go item**; without it MorphX controls at
   most ~20–30% of the LLM's GPU time.
2. **FlashInfer attention** (`BatchPrefillWithPagedKVCacheKernel`,
   `PrefillWithKVCacheKernel`, `PersistentVariableLengthMergeStatesKernel`,
   13.6–22%) is CUDA C++ JIT-compiled by FlashInfer → can be built with clang
   + the pass (point FlashInfer's JIT at clang, or AOT build). These are the
   same kernel names `cuda-llm.cpp` already targets.
3. **sglang `jit_kernel`** ops (`act_and_mul`, `fused_rope`, `fused_qknorm`,
   `store_kvcache`; 5–8%) are CUDA C++ compiled at run time by tvm-ffi with
   nvcc → same treatment as FlashInfer's JIT. They need gcc ≥ 13 as host
   compiler (gcc 12.3 fails to parse `activation.cuh`); on hpcc use
   `source /opt/rh/gcc-toolset-13/enable`.
4. **FlashInfer norms are CuTe DSL** (`kernel_cutlass_kernel_flashinfernorm…`,
   3–5%): generated through CUTLASS's Python DSL/MLIR, not clang. Either
   accept them as uncontrolled or switch SGLang to a C++ RMSNorm (sgl-kernel's
   or PyTorch's) — small share either way.
5. **Launch count:** ~554 kernel launches per decode step for a 36-layer
   model, all through `cuLaunchKernel(Ex)` (no CUDA graphs). MorphX's
   per-launch interception cost matters at this rate (Fig 15).

## Raw tables

Groups follow the kernel's library (by kernel name, then the torch op that
launched it); "How the kernel binary is built" is what decides whether the
MorphX pass can instrument it.

### Prefill (1 batch, batch 8, input 512, 464 kernel launches, 46.26 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| FlashInfer | 8.4 | 3.867 |
| sgl-kernel | 8.2 | 3.791 |
| Triton | 0.0 | 0.003 |
| cuBLAS/cuBLASLt | 80.2 | 37.081 |
| PyTorch native | 3.3 | 1.519 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| cuBLAS/cuBLASLt (closed source) | 80.2 |
| CUDA C++ source (rebuildable with clang + pass) | 16.9 |
| CuTe DSL (Python -> MLIR JIT) | 2.9 |
| Triton JIT | 0.0 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 108 | 67.1 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 36 | 12.6 |
| `void flashinfer::PrefillWithKVCacheKernel<flashinfer::CollectiveMainloop<RaggedParams::AdditionalParams, fl...` | `-` | FlashInfer | 36 | 5.4 |
| `void (anonymous namespace)::act_and_mul_kernel<__nv_bfloat16, ((anonymous namespace)::ActivationKind)0, tru...` | `sglang::_run_activation_inplace` | sgl-kernel | 36 | 4.3 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 36 | 3.1 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | `-` | FlashInfer | 72 | 2.9 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 36 | 1.8 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u>((anonymous namesp...` | `sglang::apply_rope_inplace` | sgl-kernel | 36 | 1.5 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 36 | 0.6 |
| `nvjet_tst_384x8_64x4_2x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 1 | 0.4 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 11 | 0.1 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128o256025...` | `-` | FlashInfer | 1 | 0.0 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | `aten::gather` | PyTorch native | 2 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 1 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 1 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true>...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::nati...` | `aten::sum` | PyTorch native | 1 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | `aten::sub` | PyTorch native | 2 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 2 | 0.0 |

### Decode (16 steps, batch 8, input 512, 8864 kernel launches, 60.12 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| FlashInfer | 22.1 | 13.301 |
| sgl-kernel | 5.2 | 3.116 |
| Triton | 0.0 | 0.000 |
| cuBLAS/cuBLASLt | 69.6 | 41.842 |
| PyTorch native | 3.1 | 1.859 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| cuBLAS/cuBLASLt (closed source) | 69.6 |
| CUDA C++ source (rebuildable with clang + pass) | 25.1 |
| CuTe DSL (Python -> MLIR JIT) | 5.2 |
| Triton JIT | 0.1 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `nvjet_tst_192x8_64x8_4x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 28.6 |
| `nvjet_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 14.6 |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, ...` | `-` | FlashInfer | 576 | 13.8 |
| `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 10.3 |
| `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 7.8 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | `-` | FlashInfer | 1152 | 5.1 |
| `nvjet_tst_384x8_64x4_2x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 16 | 5.0 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, _...` | `aten::mm` | cuBLAS/cuBLASLt | 1152 | 3.4 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __nv_bfloat16, __nv_bfloat16, i...` | `-` | FlashInfer | 576 | 3.0 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 576 | 2.3 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u>((anonymous namesp...` | `sglang::apply_rope_inplace` | sgl-kernel | 576 | 1.5 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 576 | 1.4 |
| `void (anonymous namespace)::act_and_mul_kernel<__nv_bfloat16, ((anonymous namespace)::ActivationKind)0, tru...` | `sglang::_run_activation_inplace` | sgl-kernel | 576 | 1.3 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 576 | 1.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 16 | 0.2 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 16 | 0.2 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(...` | `aten::index_select` | PyTorch native | 16 | 0.1 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128o256025...` | `-` | FlashInfer | 16 | 0.1 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 32 | 0.1 |
| `create_flashinfer_kv_indices_triton` | `-` | FlashInfer | 16 | 0.1 |

