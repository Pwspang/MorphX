# SGLang kernel census (H200)

Input to the SGLang go/no-go (plan §9 step 1): which kernels does SGLang run
for a dense model, and how much of the GPU time is in kernels that the MorphX
LLVM pass could instrument?

Two rounds:
- **J1 (current): linear layers through `mycutlass`** (`MORPHX_LLM_GEMM=mycutlass`,
  fp16), with and without the MorphX runtime. Sections 1–4.
- **J0: stock wheels, bf16, cuBLAS GEMMs.** Section 5, kept for reference.

## 1. GO / NO-GO

**GO, with two blocking items before any performance experiment.**

- **Functionally GO.** Routing SGLang's dense linear layers (QKV/O/gate-up/down
  and the tied lm_head) through the pass-built CUTLASS GEMM in
  `libmycutlass.so` gives the same greedy output as cuBLAS (192/192 tokens on
  6 prompts, prefill logits max |diff| 0.034, argmax 100% equal), also under
  the MorphX runtime, where the GEMMs run as their `_pk` clones. 88% (prefill)
  and 78% (decode) of the LLM's GPU time is now in pass-built kernels (was
  0%); building FlashInfer's and SGLang's run-time-compiled CUDA C++ kernels
  with clang + the pass as well would raise that to ~98% / ~96%. The rest is
  FlashInfer's CuTe-DSL RMSNorm (2–4%).
- **Blocking 1: the `hostmapped` sync backend is far too slow for real
  numbers.** Under the runtime, GPU time per decode step goes from 5.2 ms to
  226 ms (~44x) and the prefill batch from 74 ms to 1085 ms (~15x), in line
  with the +4.7 µs serialized cost per thread block measured in plan §5
  (~20k CUTLASS blocks per decode step). Needs GDRCopy or the `memcpy`
  backend (plan §5, option C).
- **Blocking 2 (fairness, not correctness): mycutlass is slower than cuBLAS on
  Hopper.** Its kernels are CUTLASS 2.x Sm80 tensor-op GEMMs with an
  A100-tuned shape table: GEMM time is 1.70x (prefill) and 1.54x (decode) that
  of cuBLAS, end-to-end TBT +56%. All baselines must therefore run with the
  same mycutlass GEMMs, as the authors' setup implies; re-tuning the shape
  table for 132 SMs is optional.
- Smaller items: the runner keeps a transposed copy of every dense weight
  (+~7.5 GB for Qwen3-4B; could transpose in place), fp16 only, and ~6k
  launches per run use `cuLaunchKernelEx` attributes (PDL from SGLang's
  run-time-compiled kernels), which the runtime passes through unmodified.

## 2. Setup (J1)

hpcc `hpc-gaas-g25`, NVIDIA H200 (132 SMs), driver 580.126.20, job 91312
(2026-10-22). Env `~/morphx/envs/sglang-stock` as in J0. Model Qwen3-4B in
**float16** for both GEMM paths. `libmycutlass.so` built from h100-port
`c64e56b` with clang (LLVM 23) + the MorphX pass, sm_90: 28 `_pk` kernels,
4640 HMMA instructions, no traps (`~/morphx/build/mycutlass-c64e56b`; the
earlier `dd5b06f` build trapped in every MMA). Runtime: `runtime/src/cuda.cpp`
built with `-DMORPHX_SYNC_BACKEND_HOSTMAPPED` (`~/morphx/build/hook`),
`LD_PRELOAD=libcuda.so:libpreload.so`, single LLM stream (so the runtime
assigns the full GPU, `range=[0,132)`; no co-location split exercised).
Scripts: `scripts/sglang/mycutlass_check.pbs` (runs
`debug_mycutlass.py`, `check_mycutlass.py`, `smoke_runner.py`).

## 3. Results (J1)

**Correctness**

| Check | Result |
|---|---|
| Raw `myCutlassHgemm`, 7 weight shapes (incl. 151936×2560 lm_head) × T ∈ {1, 7, 64, 512}, both layouts | all OK, max abs err vs fp32 ≤ 0.008 |
| Per layer in the model vs fp32 reference | error ratio mycutlass/cuBLAS = 1.00 |
| Prefill logits (6 prompts × 151936), mycutlass vs cuBLAS | max abs diff 0.034, mean 0.0053, argmax 100% |
| Greedy decode, 6 prompts × 32 tokens | 192/192 tokens equal, 6/6 sequences identical |
| Same, **under the MorphX runtime** (patched `_pk` GEMMs) | identical numbers: 192/192, max abs diff 0.034 |
| Fallbacks to torch during generation | 0 (4608 linear + 32 lm_head calls through mycutlass) |

**GPU time by kernel origin** (one prefill batch 8 × 512, 16 decode steps, batch 8)

| | torch fp16 prefill | mycutlass prefill | torch fp16 decode | mycutlass decode |
|---|---:|---:|---:|---:|
| GPU time | 47.0 ms | 73.6 ms | 60.3 ms (3.77 ms/step) | 82.8 ms (5.17 ms/step) |
| cuBLAS/cuBLASLt (closed) | 80.5% | 0% | 69.8% | 0% |
| **CUTLASS 2.x, pass-built (mycutlass)** | 0% | **87.6%** | 0% | **78.3%** |
| CUDA C++ source, not yet pass-built (FlashInfer attention, SGLang JIT, PyTorch native) | 16.6% | 10.5% | 25.0% | 17.8% |
| CuTe DSL (FlashInfer RMSNorm) | 2.9% | 1.9% | 5.2% | 3.8% |
| GEMM time only | 37.9 ms | 64.5 ms (1.70x) | 42.1 ms | 64.8 ms (1.54x) |
| Kernel launches | 464 | 464 | 8864 | 7712 |

**End to end** (Poisson, 4 req/s, input 512, output 64; under the runtime: 8
requests, output 32, `SMSCHED_DEBUG=1` logging every launch)

| | requests | output tok/s | TTFT mean / max | TBT mean / p50 / p99 | latency mean |
|---|---:|---:|---:|---:|---:|
| torch fp16 | 20 | 175.3 | 20.6 / 42.2 ms | 11.7 / 11.3 / 24.0 ms | 756 ms |
| mycutlass fp16 | 20 | 166.1 | 30.0 / 49.8 ms | 18.3 / 17.4 / 36.4 ms | 1180 ms |
| mycutlass fp16 + MorphX runtime (hostmapped) | 8 | 28.0 | 822 / 1329 ms | 252 / 227 / 1290 ms | 8622 ms |

Throughput is arrival-bound at 4 req/s, so latency is the meaningful
comparison. GPU time under the runtime (CUPTI): prefill batch 1085 ms, decode
226 ms/step, 99% of it in the CUTLASS GEMMs.

**Runtime interception** (all 33 722 launches of the runtime run)

| Debug event | Count | Meaning |
|---|---:|---|
| `assign` | 12 028 | patched launch: `_pk` clone launched with SM range (all CUTLASS GEMMs) |
| `profile` | 7 | first launch of a kernel name: profiled before use |
| `bypass-unpatched` | 21 687 | no `_pk` clone (FlashInfer, SGLang JIT, PyTorch kernels) |
| `bypass-launch-attrs` | 6 059 | `cuLaunchKernelEx` with launch attributes (PDL), passed through |

Every CUTLASS GEMM launch went through `assign`/`profile` (e.g. 8100 launches
of the 64×64×64 tile config, 8099 `assign` + 1 `profile`).

## 4. Next steps for the SGLang path

1. Faster sync backend (GDRCopy on hpcc, or plan §5 option C) — re-measure
   the runtime overhead.
2. Build FlashInfer's and SGLang's run-time-compiled CUDA C++ kernels with
   clang + the pass (FlashInfer AOT or JIT compiler override; tvm-ffi
   compiler override for `sglang.jit_kernel`).
3. Use `MORPHX_LLM_GEMM=mycutlass` (fp16) for every baseline.

## 5. J0: stock wheels, bf16, cuBLAS (job 91232)

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

### Summary

| | Prefill | Decode |
|---|---:|---:|
| cuBLAS/cuBLASLt GEMMs (`nvjet_*`, closed source) | **80.2%** | **69.6%** |
| CUDA C++ from source (FlashInfer attention, sglang jit kernels, PyTorch native) | 16.9% | 25.1% |
| CuTe DSL (FlashInfer RMSNorm / fused add-RMSNorm, Python → MLIR JIT) | 2.9% | 5.2% |
| Triton | 0.0% | 0.1% |
| Kernel launches | 464 per batch | ~554 per step |

All LLM kernels ran on the caller's CUDA stream (CUPTI stream 13, the same as
a marker kernel launched on it); none on the legacy default stream.

### What this meant for MorphX (J0)

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

### J0 raw tables (bf16, cuBLAS)

Groups follow the kernel's library (by kernel name, then the torch op that
launched it); "How the kernel binary is built" is what decides whether the
MorphX pass can instrument it.

#### Prefill (1 batch, batch 8, input 512, 464 kernel launches, 46.26 ms GPU)

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

#### Decode (16 steps, batch 8, input 512, 8864 kernel launches, 60.12 ms GPU)

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

## 6. J1 raw tables

Rows labelled "mycutlass CUTLASS (incl. ≤0.003 ms Triton)" were reported as "Triton" by the job (the classifier missed the mangled `_ZN7cutlassL6KernelI…` names, fixed since); the only real Triton kernels there are three bookkeeping kernels totalling ≤0.003 ms per prefill.

### torch fp16, no runtime

#### Prefill (1 batch, batch 8, input 512, 464 kernel launches, 47.02 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| mycutlass (CUTLASS) | 0.0 | 0.000 |
| FlashInfer | 8.2 | 3.873 |
| sgl-kernel | 8.0 | 3.778 |
| Triton | 0.0 | 0.003 |
| cuBLAS/cuBLASLt | 80.5 | 37.853 |
| PyTorch native | 3.2 | 1.516 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| cuBLAS/cuBLASLt (closed source) | 80.5 |
| CUDA C++ source (rebuildable with clang + pass) | 16.6 |
| CuTe DSL (Python -> MLIR JIT) | 2.9 |
| Triton JIT | 0.0 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `nvjet_hsh_320x128_64x3_1x2_h_bz_coopB_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 108 | 67.5 |
| `nvjet_hsh_256x128_64x4_1x2_h_bz_coopA_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 36 | 12.6 |
| `void flashinfer::PrefillWithKVCacheKernel<flashinfer::CollectiveMainloop<RaggedParams::AdditionalParams, fl...` | `-` | FlashInfer | 36 | 5.3 |
| `void (anonymous namespace)::act_and_mul_kernel<__half, ((anonymous namespace)::ActivationKind)0, true, fals...` | `sglang::_run_activation_inplace` | sgl-kernel | 36 | 4.2 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 36 | 3.1 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrf16gm...` | `-` | FlashInfer | 72 | 2.9 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __half>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 36 | 1.7 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __half, long, 16u>((anonymous namespace)::F...` | `sglang::apply_rope_inplace` | sgl-kernel | 36 | 1.5 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 36 | 0.6 |
| `nvjet_hsh_384x8_64x4_2x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 1 | 0.4 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 11 | 0.1 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | `aten::gather` | PyTorch native | 2 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 1 | 0.0 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrf16gmemalign128o2560256...` | `-` | FlashInfer | 1 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 1 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true>...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 2 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | `aten::sub` | PyTorch native | 2 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::nati...` | `aten::sum` | PyTorch native | 1 | 0.0 |

#### Decode (16 steps, batch 8, input 512, 8864 kernel launches, 60.29 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| mycutlass (CUTLASS) | 0.0 | 0.000 |
| FlashInfer | 21.9 | 13.185 |
| sgl-kernel | 5.2 | 3.136 |
| Triton | 0.0 | 0.000 |
| cuBLAS/cuBLASLt | 69.8 | 42.059 |
| PyTorch native | 3.2 | 1.914 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| cuBLAS/cuBLASLt (closed source) | 69.8 |
| CUDA C++ source (rebuildable with clang + pass) | 25.0 |
| CuTe DSL (Python -> MLIR JIT) | 5.2 |
| Triton JIT | 0.1 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `nvjet_hsh_192x8_64x8_4x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 28.5 |
| `nvjet_hsh_128x8_64x12_4x1_v_bz_splitK_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 14.6 |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, ...` | `-` | FlashInfer | 576 | 13.8 |
| `nvjet_hsh_64x8_64x16_4x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 10.4 |
| `nvjet_hsh_64x8_64x16_4x1_v_bz_splitK_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 576 | 7.9 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrf16gm...` | `-` | FlashInfer | 1152 | 5.1 |
| `nvjet_hsh_384x8_64x4_2x1_v_bz_TNT` | `aten::mm` | cuBLAS/cuBLASLt | 16 | 4.9 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __half, float, __half, false, float, __half, __half,...` | `aten::mm` | cuBLAS/cuBLASLt | 1152 | 3.4 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __half, __half, int>(__half*, f...` | `-` | FlashInfer | 576 | 2.8 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 576 | 2.4 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __half, long, 16u>((anonymous namespace)::F...` | `sglang::apply_rope_inplace` | sgl-kernel | 576 | 1.5 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __half>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 576 | 1.4 |
| `void (anonymous namespace)::act_and_mul_kernel<__half, ((anonymous namespace)::ActivationKind)0, true, fals...` | `sglang::_run_activation_inplace` | sgl-kernel | 576 | 1.3 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 576 | 1.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 16 | 0.3 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 16 | 0.2 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::Half, long, unsigned int, 2, 2, -2>(at::...` | `aten::index_select` | PyTorch native | 16 | 0.1 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrf16gmemalign128o2560256...` | `-` | FlashInfer | 16 | 0.1 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 32 | 0.1 |
| `create_flashinfer_kv_indices_triton` | `-` | FlashInfer | 16 | 0.1 |


### mycutlass fp16, no runtime

#### Prefill (1 batch, batch 8, input 512, 464 kernel launches, 73.62 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| FlashInfer | 5.3 | 3.916 |
| sgl-kernel | 5.1 | 3.722 |
| mycutlass CUTLASS (incl. ≤0.003 ms Triton) | 87.6 | 64.466 |
| cuBLAS/cuBLASLt | 0.0 | 0.000 |
| PyTorch native | 2.1 | 1.515 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| CUTLASS 2.x from source (libmycutlass, pass-built) (incl. ≤0.1% Triton) | 87.6 |
| CUDA C++ source (rebuildable with clang + pass) | 10.5 |
| CuTe DSL (Python -> MLIR JIT) | 1.9 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi256ELi32EE...` | `-` | mycutlass (CUTLASS) | 36 | 42.4 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi32EE...` | `-` | mycutlass (CUTLASS) | 72 | 35.2 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi32EE...` | `-` | mycutlass (CUTLASS) | 36 | 9.6 |
| `void flashinfer::PrefillWithKVCacheKernel<flashinfer::CollectiveMainloop<RaggedParams::AdditionalParams, fl...` | `-` | FlashInfer | 36 | 3.4 |
| `void (anonymous namespace)::act_and_mul_kernel<__half, ((anonymous namespace)::ActivationKind)0, true, fals...` | `sglang::_run_activation_inplace` | sgl-kernel | 36 | 2.6 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 36 | 2.0 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrf16gm...` | `-` | FlashInfer | 72 | 1.9 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __half>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 36 | 1.1 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __half, long, 16u>((anonymous namespace)::F...` | `sglang::apply_rope_inplace` | sgl-kernel | 36 | 1.0 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 36 | 0.4 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi256ELi32EEE...` | `-` | mycutlass (CUTLASS) | 1 | 0.3 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 11 | 0.0 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | `aten::gather` | PyTorch native | 2 | 0.0 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrf16gmemalign128o2560256...` | `-` | FlashInfer | 1 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 1 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 1 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | `aten::sub` | PyTorch native | 2 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true>...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::nati...` | `aten::sum` | PyTorch native | 1 | 0.0 |

#### Decode (16 steps, batch 8, input 512, 7712 kernel launches, 82.79 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| FlashInfer | 15.7 | 13.028 |
| sgl-kernel | 3.8 | 3.108 |
| mycutlass CUTLASS (incl. ≤0.003 ms Triton) | 78.3 | 64.841 |
| cuBLAS/cuBLASLt | 0.0 | 0.000 |
| PyTorch native | 2.2 | 1.816 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| CUTLASS 2.x from source (libmycutlass, pass-built) (incl. ≤0.1% Triton) | 78.4 |
| CUDA C++ source (rebuildable with clang + pass) | 17.8 |
| CuTe DSL (Python -> MLIR JIT) | 3.8 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi64ELi64EEEN...` | `-` | mycutlass (CUTLASS) | 1728 | 50.3 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi256ELi32EEE...` | `-` | mycutlass (CUTLASS) | 592 | 28.0 |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, ...` | `-` | FlashInfer | 576 | 9.9 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrf16gm...` | `-` | FlashInfer | 1152 | 3.8 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __half, __half, int>(__half*, f...` | `-` | FlashInfer | 576 | 2.0 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 576 | 1.6 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __half, long, 16u>((anonymous namespace)::F...` | `sglang::apply_rope_inplace` | sgl-kernel | 576 | 1.1 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __half>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 576 | 1.0 |
| `void (anonymous namespace)::act_and_mul_kernel<__half, ((anonymous namespace)::ActivationKind)0, true, fals...` | `sglang::_run_activation_inplace` | sgl-kernel | 576 | 0.9 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 576 | 0.7 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 16 | 0.2 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 16 | 0.1 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::Half, long, unsigned int, 2, 2, -2>(at::...` | `aten::index_select` | PyTorch native | 16 | 0.1 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrf16gmemalign128o2560256...` | `-` | FlashInfer | 16 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 32 | 0.0 |
| `create_flashinfer_kv_indices_triton` | `-` | FlashInfer | 16 | 0.0 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel...` | `aten::_index_put_impl_` | PyTorch native | 16 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long...` | `aten::cumsum` | PyTorch native | 16 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*,...` | `aten::add_` | PyTorch native | 16 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | `aten::add_` | PyTorch native | 16 | 0.0 |


### mycutlass fp16, MorphX runtime (hostmapped), 4 decode steps

#### Prefill (1 batch, batch 8, input 512, 464 kernel launches, 1084.89 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| FlashInfer | 0.4 | 4.016 |
| sgl-kernel | 0.3 | 3.666 |
| mycutlass CUTLASS (incl. ≤0.003 ms Triton) | 99.2 | 1075.684 |
| cuBLAS/cuBLASLt | 0.0 | 0.000 |
| PyTorch native | 0.1 | 1.528 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| CUTLASS 2.x from source (libmycutlass, pass-built) (incl. ≤0.1% Triton) | 99.2 |
| CUDA C++ source (rebuildable with clang + pass) | 0.7 |
| CuTe DSL (Python -> MLIR JIT) | 0.1 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi32EE...` | `-` | mycutlass (CUTLASS) | 72 | 42.6 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi256ELi32EE...` | `-` | mycutlass (CUTLASS) | 36 | 41.5 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi32EE...` | `-` | mycutlass (CUTLASS) | 36 | 14.7 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi256ELi32EEE...` | `-` | mycutlass (CUTLASS) | 1 | 0.4 |
| `void flashinfer::PrefillWithKVCacheKernel<flashinfer::CollectiveMainloop<RaggedParams::AdditionalParams, fl...` | `-` | FlashInfer | 36 | 0.2 |
| `void (anonymous namespace)::act_and_mul_kernel<__half, ((anonymous namespace)::ActivationKind)0, true, fals...` | `sglang::_run_activation_inplace` | sgl-kernel | 36 | 0.2 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 36 | 0.1 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrf16gm...` | `-` | FlashInfer | 72 | 0.1 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __half>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 36 | 0.1 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __half, long, 16u>((anonymous namespace)::F...` | `sglang::apply_rope_inplace` | sgl-kernel | 36 | 0.1 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 36 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 11 | 0.0 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | `aten::gather` | PyTorch native | 2 | 0.0 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrf16gmemalign128o2560256...` | `-` | FlashInfer | 1 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 1 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 1 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | `aten::sub` | PyTorch native | 2 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true>...` | `aten::cumsum` | PyTorch native | 3 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::nati...` | `aten::sum` | PyTorch native | 1 | 0.0 |

#### Decode (4 steps, batch 8, input 512, 1928 kernel launches, 905.74 ms GPU)

| Group | % GPU time | ms |
|---|---:|---:|
| FlashInfer | 0.4 | 3.440 |
| sgl-kernel | 0.1 | 0.804 |
| mycutlass CUTLASS (incl. ≤0.003 ms Triton) | 99.5 | 901.042 |
| cuBLAS/cuBLASLt | 0.0 | 0.000 |
| PyTorch native | 0.0 | 0.449 |
| other | 0.0 | 0.000 |

| How the kernel binary is built | % GPU time |
|---|---:|
| CUTLASS 2.x from source (libmycutlass, pass-built) (incl. ≤0.1% Triton) | 99.5 |
| CUDA C++ source (rebuildable with clang + pass) | 0.4 |
| CuTe DSL (Python -> MLIR JIT) | 0.1 |

Top kernels:

| Kernel | Launched from (torch op) | Group | Launches | % GPU time |
|---|---|---|---:|---:|
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi64ELi64EEEN...` | `-` | mycutlass (CUTLASS) | 432 | 72.3 |
| `_ZN7cutlassL6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi256ELi32EEE...` | `-` | mycutlass (CUTLASS) | 148 | 27.2 |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, ...` | `-` | FlashInfer | 144 | 0.2 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrf16gm...` | `-` | FlashInfer | 288 | 0.1 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __half, __half, int>(__half*, f...` | `-` | FlashInfer | 144 | 0.0 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | `aten::copy_` | PyTorch native | 144 | 0.0 |
| `void (anonymous namespace)::act_and_mul_kernel<__half, ((anonymous namespace)::ActivationKind)0, true, fals...` | `sglang::_run_activation_inplace` | sgl-kernel | 144 | 0.0 |
| `void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __half, long, 16u>((anonymous namespace)::F...` | `sglang::apply_rope_inplace` | sgl-kernel | 144 | 0.0 |
| `void (anonymous namespace)::fused_qknorm_warp<128l, true, __half>((anonymous namespace)::QKNormParams)` | `sglang::fused_inplace_qknorm` | sgl-kernel | 144 | 0.0 |
| `void (anonymous namespace)::store_kvcache<2048l, 4, true, long>((anonymous namespace)::StoreKVCacheParams)` | `sglang::store_cache` | sgl-kernel | 144 | 0.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned i...` | `aten::argmax` | PyTorch native | 4 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 4 | 0.0 |
| `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::Half, long, unsigned int, 2, 2, -2>(at::...` | `aten::index_select` | PyTorch native | 4 | 0.0 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrf16gmemalign128o2560256...` | `-` | FlashInfer | 4 | 0.0 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::...` | `aten::copy_` | PyTorch native | 8 | 0.0 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel...` | `aten::_index_put_impl_` | PyTorch native | 4 | 0.0 |
| `create_flashinfer_kv_indices_triton` | `-` | FlashInfer | 4 | 0.0 |
| `void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long...` | `aten::cumsum` | PyTorch native | 4 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*,...` | `aten::add_` | PyTorch native | 4 | 0.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` | `aten::add_` | PyTorch native | 4 | 0.0 |

