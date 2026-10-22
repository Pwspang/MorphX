# MorphX H100 Port Plan

Goal: run the MorphX artifact (`./run.sh`) on an NVIDIA H100 instead of the
original A100 (sm_80, 108 SMs) evaluation machine, **with SGLang as the LLM
engine** instead of the authors' unpublished vLLM fork (section 9).

**Decision (2026-10-22): SGLang-first.** The authors' modified vLLM
(`0.1.dev24+g232ce74.d20251027`, built from `/home/rtx/gpu/vllm`) is not
public: commit `232ce74` is not in upstream vLLM, and `VLLM_PREFILL_THREAD`
appears only in MorphX's own scripts. Its PD prefill thread has to be
re-implemented whichever engine we use. SGLang (PyTorch 2.9.1 + CUDA 13.0,
FlashInfer 0.6.8) also avoids forcing PyTorch 2.6 / old vLLM onto CUDA 13.
Go/no-go: the SGLang spike (§9 step 1). Fallback: stock vLLM with the V0
engine + CUDA 12.8 + our own prefill thread (§6.5).

Branch: `h100-port` on `Pwspang/MorphX`.

## Parallel work tracks

The sections below are grouped by topic. This section groups them into
tracks that can run **at the same time**, by different people or Claude
sessions, each on its own branch off `h100-port`. Each track owns a separate
set of files, so merges don't conflict.

### Tracks

| Track | Work (sections) | Owns these files | Needs a GPU? | Blocked by |
|---|---|---|---|---|
| **X. External requests** | Request Llama-2/Llama-3 access on HuggingFace; ask hpcc support about GDRCopy; optional: ask the authors for their vLLM fork (only useful as a reference for the prefill thread) | none | no | nothing, **start day 1** |
| **A. Script config** | §1 paths/env vars in Python + shell, §2 `TORCH_CUDA_ARCH_LIST`, §3 SM-count lists and split hints, PBS template (§8) | `run.sh`, `scripts/ae/*.py,*.sh`, `scripts/colocate/`, `scripts/overhead/`, `scripts/test-pd-online.py`, `scripts/ae/env-h100.sh`, `scripts/ae/morphx-hpcc.pbs` | no | nothing |
| **B. Build config** | §1 CMake paths, §2 sm_90, CUDA 13 version macros, CCCL from the toolkit | `microbench/CMakeLists.txt`, `ggnn/CMakeLists.txt`, `runtime/CMakeLists.txt`, `smsched-pass/CMakeLists.txt`, `TGS/hijack/` | compile-only (login node) | nothing |
| **C. Runtime on CUDA 13** | §6.1 regenerate `unmodified.c` with `parse.py`, fix `cuGetProcAddress` hooks; §3 runtime SM counts (`base.cpp`, `stub_raw.cpp`); §5 option C `memcpy` backend (done, Track C2) | `runtime/src/`, `include/common.h` | short GPU tests | nothing (hostmapped backend already done) |
| **D. LLVM + pass** | §6.2–6.3 build clang/libc++ with NVPTX for CUDA 13; build `smsched-pass` | `$HOME/morphx/llvm` on hpcc; `smsched-pass/` only if the pass needs a newer-LLVM fix | no (long CPU build) | nothing |
| **E. Models and data** | §6.8 download models + ShareGPT into `MORPHX_HF_CACHE` | data dir on hpcc | no | X (Llama access) |
| **F. Baselines** | §6.7 NVBit `mem_trace` for sm_90, TGS socket path, `neutrino`, `nanoflow` env | `nvbit-tutorial/tools/`, `TGS/` (coordinate with B) | short GPU tests | nothing |
| **G. Python stack (SGLang)** | §9 step 1: PyTorch 2.9.1 + FlashInfer 0.6.8 + `sgl-kernel` built with clang + the pass for sm_90 on CUDA 13 | conda env `smsched` on hpcc | build: no; check: short GPU | D |
| **H. SGLang spike (go/no-go)** | §9 step 1: one request through `ModelRunner` in-process with the runtime preloaded; measure the share of GPU time in pass-built kernels and whether split hints restrict SMs | `scripts/sglang/` (new) | short GPU | G (a first look is possible with stock wheels before G finishes) |
| **J. SGLang drivers** | §9 steps 2–6: `LLMBackend` interface, Fig 15 → Fig 10–12 → Fig 13–14 (prefill thread), per-backend kernel patterns | `scripts/colocate/`, `scripts/overhead/`, `scripts/test-pd-online.py`, `scripts/sglang/`, `runtime/src/cuda-llm.cpp` patterns | short GPU per step | H = go; driver code (`LLMBackend`, `ModelRunner` loop) can start before |
| **I. Integration and runs** | §7 bring-up order, §8 jobs D–F, split-hint re-tuning, full run | none (merges A–C, J) | yes, long | A, B, C, D, then G + J for Fig 10–15 |

### Dependency graph

```text
  X (external) ──► E (models)
  A (script config) ─────┐
  B (build config) ──────┤
  C (runtime CUDA 13) ───┤
  D (LLVM + pass) ───────┼──► I1: Fig 16 (Part 4)
  F (baselines) ─────────┘
  D ──► G (PyTorch/FlashInfer/sgl-kernel + pass) ──► H (spike: GO / NO-GO)
                                                       │GO            │NO-GO
                                                       ▼              ▼
  J0 (LLMBackend, ModelRunner loop; no GPU) ──► J: Fig 15 ──► Fig 10–12 ──► Fig 13–14 ──► full run
                                                              fallback: stock vLLM V0 + CUDA 12.8 (§6.5)
```

- **Day 1, all at once:** X, A, B, C, D, F (A, B, C done and merged).
- **First milestone (I1, Fig 16):** needs A + B + C + D only, not the Python
  stack, so it can be reached while G is still building.
- **Critical path:** D (LLVM) → G (pass-built PyTorch/FlashInfer/sgl-kernel)
  → H (go/no-go) → J (Fig 10–12, then the prefill thread for Fig 13–14).
- **J0 runs alongside G:** the `LLMBackend` interface and the in-process
  `ModelRunner` batching loop can be written and tested with stock SGLang
  wheels (no pass, no MorphX control) before G is done.

### Sharing hpcc

The free GPU queue allows **one running job per user**, so GPU jobs from
different tracks queue up behind each other. Keep GPU jobs short and put long
CPU work somewhere else:

| Work | Queue | Limits | Notes |
|---|---|---|---|
| LLVM, PyTorch, FlashInfer, `sgl-kernel` builds (D, G) | `gpu_free`, `ncpus=48`, build on node-local `/tmp` | ≤ 6 h, blocks GPU tests meanwhile | **`gpu_datamover` is not usable from this account** (jobs held after 21 failed starts, exit −10). Use `ccache` so a build cut off at the time limit picks up where it stopped |
| Model and dataset downloads (E) | login node, `nohup` | network-bound | Done this way for the non-gated models |
| Compile-only checks (B, C) | login node | short, light jobs only | Like the `unmodified.c` check already done |
| GPU tests (C, F, G, H) and runs (I) | `gpu_free` | 1 GPU job at a time, ≤ 6 h | Chain jobs with `qsub -W depend=afterany:...`; combine small tests from several tracks into one job |

### Merging

- Each track works on `h100-port-<track>` (e.g. `h100-port-a-script-config`;
  git can't create `h100-port/...` while `h100-port` exists) and merges into `h100-port`.
- Merge order: A, B, C as soon as they're ready (they only add config and
  defaults to the A100 values, so nothing else breaks); D/G/H produce
  environments, not code, so they write their build steps into
  `docs/h100-port-plan.md` §6 / §9 when done.
- Shared names to agree on before starting: the `MORPHX_*` env vars in §1
  (A, B and C all read them) and `scripts/ae/env-h100.sh` (owned by A).

## 0. Decisions to settle first

| Question | Why it matters | Default assumed below |
|---|---|---|
| H100 SXM (132 SMs) or PCIe (114 SMs)? | Every SM count and split hint scales with it | SXM, 132 SMs (same as the H200 used for smoke tests, see section 8) |
| CUDA toolkit version | Hardcoded `cuda-12.4` paths and `__CUDACC_VER_MINOR__=4` | **CUDA 13.0** (driver ≥ 580; hpcc has 580.126.20). See section 6 for the knock-on effects |
| LLVM/clang version for the pass | Must emit CUDA for sm_90 **and accept the CUDA 13 headers** | Recent clang (20+; verify CUDA 13 support) |
| Is GDRCopy (`gdrdrv` kernel module + `libgdrapi`) installed? | The runtime and patched workloads link `-lgdrapi` | **Not on hpcc.** Use the non-GDR backend in section 5 |
| Exclusive GPU + permission to run `nvidia-cuda-mps-control`? | MPS baseline in Figures 10–12 | **Works on hpcc** as a normal user (tested, section 8) |
| Where do models/datasets live? | Hardcoded `/huggingface-cache/hub/...` | Set via `HF_HUB_CACHE`-style env var |

## 1. Make machine-specific values configurable (no behavior change)

Introduce env vars with A100 defaults first, so the original machine still
works, then set H100 values in one place (e.g. `scripts/ae/env-h100.sh`).

| Variable | Replaces |
|---|---|
| `MORPHX_PYTHON` (existing `PYTHON`) | `/home/rtx/miniconda3/envs/smsched/bin/python` |
| `MORPHX_LLVM_ROOT` | `/home/rtx/llvm-project/install`, `/mnt/data/rtx/llvm-project/...` |
| `MORPHX_CUDA_ROOT` | `/usr/local/cuda-12.4` (hpcc: `module load cuda/13.0`, then `$CUDA_HOME`) |
| `MORPHX_SYNC_BACKEND` | `gdr` (default) or `hostmapped` / `memcpy` (section 5) |
| `MORPHX_CCCL_ROOT` | `/home/rtx/gpu/cccl` |
| `MORPHX_CUTLASS_ROOT` | `/home/rtx/gpu/cutlass` |
| `MORPHX_CUDA_ARCH` | `sm_80` / `80` / `"8.0"` |
| `MORPHX_NUM_SMS` | literal `108` |
| `MORPHX_HF_CACHE` | `/huggingface-cache/hub` |
| `MORPHX_NANOFLOW_LIB` | `/home/rtx/miniconda3/envs/nanoflow/lib` |

### Files to edit

**Top-level and AE drivers**
- [ ] `run.sh:5` – default `PYTHON_BIN`
- [ ] `scripts/ae/run_colocate_part1.sh:5`, `run_pd_colocate_part2.sh:7`, `run_overhead_model_part4.sh:6` – default `PYTHON_BIN`
- [ ] `scripts/ae/run_all.py:76-77` – candidate Python paths
- [ ] `scripts/ae/run_all.py:331-332` – `LLVM_DIR` candidates for building `smsched-pass`
- [ ] `scripts/ae/run_colocate_part1.py:123, 435-442` – conda Python, LLVM lib dirs, nanoflow lib
- [ ] `scripts/ae/run_pd_colocate_part2.py:131, 454-461` – same
- [ ] `scripts/ae/run_overhead_model_part4.py:113, 220-221, 274-275` – same

**Model and dataset paths** (`/huggingface-cache/hub/...`)
- [ ] `scripts/colocate/run.py:76-84` – model table + ShareGPT dataset
- [ ] `scripts/test-pd-online.py:45-51` – model table (includes `glm-9b` = GLM-Z1-9B-0414 used by `configs/pd/*.json`)
- [ ] `scripts/overhead/single_request.py:60-64` – model table
- [ ] `scripts/ae/run_colocate_part1.py:81-89`, `scripts/ae/run_pd_colocate_part2.py:90-91` – model/dataset existence checks
- Pinned snapshot hashes may not match a fresh download; resolve by repo name instead of hash, or download those exact revisions.

**Build files**
- [ ] `runtime/CMakeLists.txt:2-3` – hardcoded `/usr/bin/gcc-12`, `/usr/bin/g++-12` (make overridable; don't force)
- [ ] `runtime/CMakeLists.txt:104-105, 114-115` – `cuda-12.4` include/lib dirs → `CUDAToolkit` paths
- [ ] `microbench/CMakeLists.txt:6` – `SMSCHED_LLVM_ROOT` default
- [ ] `microbench/CMakeLists.txt:19-25` – CUDA and CCCL include dirs
- [ ] `microbench/CMakeLists.txt:36-37` – `__CUDACC_VER_MAJOR__/MINOR__` → derive from the toolkit
- [ ] `microbench/CMakeLists.txt:49-50, 61, 67` – CUDA lib dir, `CUTLASS_ROOT`, `CUDA_TOOLKIT_ROOT_DIR`
- [ ] `ggnn/CMakeLists.txt:4-8, 18-21` – CUDA root, clang paths, CCCL, `python3.10` include/lib (match the env's Python)

**TGS baseline**
- [ ] `TGS/hijack/src/high_priority_hijack_call.c:104`, `low_priority_hijack_call.c:218` – socket path `/home/rtx/gpu/sm-sched/TGS/gsharing/...`. Make it come from an env var or a compile-time define; keep it under 108 chars (the `SOCKET_PATH` buffer size, which is a Unix socket limit, not an SM count).

## 2. Retarget to sm_90

- [ ] `microbench/CMakeLists.txt:26` – `--cuda-gpu-arch=sm_80` → `sm_90`
- [ ] `microbench/CMakeLists.txt` – `CMAKE_CUDA_ARCHITECTURES "80"` → `"90"` (currently `FORCE`d; remove the `FORCE`)
- [ ] `ggnn/CMakeLists.txt:18` – `--cuda-gpu-arch=sm_80` → `sm_90`
- [ ] `scripts/colocate/run.py:39` – `TORCH_CUDA_ARCH_LIST = "8.0"` → `"9.0"`
- [ ] `scripts/test-pd-online.py:19` – same
- [ ] `scripts/overhead/single_request.py:35` – same
- [ ] `nvbit-tutorial/tools/mem_trace/Makefile` – `ARCH?=all` is fine; NVBit 1.7.6 supports Hopper. No change expected.
- Keep `cutlass::arch::Sm80` in `microbench/src/my_cutlass_gemm.cu:41` and `microbench/src/generator.py:124`. Sm80 tensor-op kernels run on H100, and switching to Sm90 (WGMMA/TMA) kernels would change what Figure 16 measures.

## 3. Replace hardcoded SM counts

| Location | Current | H100 SXM | Notes |
|---|---|---|---|
| `runtime/src/base.cpp:40` | `nrSM = 108` | query `multiProcessorCount` | Same pattern as `cuda.cpp:248` |
| `runtime/src/base.cpp:56` | `occup * 108` | `occup * nrSM` | |
| `runtime/src/stub_raw.cpp:74, 84` | `108` | query at runtime | Only if `stub_raw` is used |
| `microbench/src/green_ctx_bench.cu:147` | `total_sms = 108` | query at runtime | |
| `microbench/profile.sh:2` | `{8..108..8}` | `{8..132..8}` | Not called by `run.sh` |
| `runtime/src/cuda.cpp:911` | default split `76` | ~93 | 76/108 × 132; re-tune |
| `runtime/src/cuda-llm.cpp:746` | default split `60` | ~73 | 60/108 × 132; re-tune |
| `scripts/ae/run_pd_colocate_part2.py:48+` | `split-76`, `split-88` rows | ~93, ~108 | Row names encode the split; re-tune per row |
| `scripts/ae/run_colocate_part1.py` (`--smsched-split-hint` selectors) | 76 etc. | scaled | Check every override |
| `scripts/ae/run_overhead_model_part4.py:46-50` | sweeps start at 108 | start at 132, e.g. `(132, 120, 108, 96, …)` | Figure 16 x-axis |
| `scripts/ae/run_overhead_model_part4.py:484` | `SMSCHED_SPLIT_HINT "108"` | `MORPHX_NUM_SMS` | Means "all SMs" |
| `plot_figures.py` / `scripts/analysis/*` | axis limits or labels assuming 108 | check | Grep after the above changes |

Derive the H100 values from `MORPHX_NUM_SMS` instead of writing in new literals.

## 4. Check `%smid` numbering on H100 (correctness risk)

`smsched-pass/SMSched.cpp:174` adds `mov.u32 %smid` and keeps a block only if
`minSM <= smid < maxSM`, with `maxSM` up to `multiProcessorCount`. On Hopper,
`%smid` may not be numbered 0..N-1 without gaps (`%nsmid` can be larger than the
SM count). If there are gaps, some SMs are never used and splits come out uneven.

- [x] Probe kernel written and run on the hpcc H200 (job 91099, 2026-10-22):
      **`%nsmid`=132 and the IDs are 0..131 with no gaps.** No remapping
      needed on H200 (and expected the same on H100 SXM). Re-run the probe once
      on the real H100 box to confirm.
- [x] Under MPS, `%smid` and `multiProcessorCount` are rescaled per client
      (30% → 38 SMs, IDs 0..37; 50% → 66 SMs). This only matters if MorphX's
      runtime were run inside an MPS client, which the AE scripts don't do.
- [ ] Still check how `fetched/finished/minSM/maxSM` in the runtime assume
      `smid < nrSM`.

## 5. Replace GDRCopy (`libgdrapi`) where it isn't installed

**Status on hpcc:** no `/dev/gdrdrv`, no `gdrdrv` kernel module, no
`libgdrapi` on the H200 compute node (job 91099).

### What GDRCopy is used for

All of it is in one class, `GdrPool` in `include/common.h:46-160`, used by
`runtime/src/cuda.cpp`, `cuda-llm.cpp` (and the unused variants
`stub.cpp`, `cuda-lithos.cpp`, …). `-lgdrapi` is also linked by
`microbench/CMakeLists.txt:49-50` and `ggnn/CMakeLists.txt:20-21`.

It allocates a 1 MB pool of 8-byte control words **in GPU memory** and uses
GDRCopy to map that memory into the CPU's address space. This gives two fast paths:

- **GPU side (hot path):** every block of a pass-instrumented kernel does an
  atomic sub on `agents`, volatile loads of `minSM`/`maxSM`, an atomic add on
  `fetched` (block id) and an atomic add on `finished`
  (`smsched-pass/SMSched.cpp:311-412`). These are plain device-memory
  atomics, which are fast.
- **CPU side (scheduler):** the runtime reads `fetched` to track progress and
  writes `minSM`/`maxSM`/`agents` to resize a running kernel's SM range,
  each in about 1 µs, without going through a CUDA stream.

### Options

| Option | How | GPU hot path | CPU get/set | Effort | Fidelity |
|---|---|---|---|---|---|
| **A. Install GDRCopy** | HPC admins load `gdrdrv` (needs root); `libgdrapi` can be built in `$HOME` | unchanged | ~1 µs | Support ticket | Same as paper |
| **B. `hostmapped`** | Pool in pinned host memory: `cudaHostAlloc(..., cudaHostAllocMapped \| cudaHostAllocPortable)` + `cudaHostGetDevicePointer`. `GdrEntry{d, h}` maps directly; `get`/`set` become volatile loads/stores plus a memory fence | every per-block atomic/load crosses PCIe (~1–2 µs, contended atomics serialize) | ~100 ns | Small: rewrite `GdrPool` only (~40 lines), drop `-lgdrapi` | Correct, but adds overhead to every block. Inflates Fig 15 and may shift co-location results |
| **C. `memcpy`** | Pool stays in device memory. CPU `get`/`set` use `cuMemcpyDtoHAsync`/`HtoDAsync` on a dedicated non-blocking, high-priority stream with pinned staging buffers. Must call the **real** driver entry points (not the hooked ones) to avoid re-entering the interception layer | unchanged (same as GDR) | ~5–10 µs, uses a copy engine | Medium (~100–150 lines) | Close to GDR; only the scheduler's reaction time slows down |

**Recommendation**

1. Implement **B** first behind `MORPHX_SYNC_BACKEND=hostmapped` (compile-time
   `-DMORPHX_SYNC_BACKEND=...`; `gdr` stays the default). This unblocks the hpcc
   smoke tests quickly.
2. Before relying on B, run a short probe job. Check
   `cudaDevAttrHostNativeAtomicSupported`, check that GPU-side `atomicAdd` on
   mapped host memory gives the right count under contention, and measure
   per-block latency compared with device memory. On x86/PCIe these atomics
   are atomic among GPU threads but not with respect to the CPU. That's
   enough here, because the CPU only reads `fetched`/`finished` and only
   writes `minSM`/`maxSM`/`agents`.
3. If B's overhead noticeably changes Fig 15 / Fig 10–12, implement **C**.
4. In parallel, ask hpcc support whether they can install GDRCopy (option A).
   Final numbers should come from A or C, not B.

Checklist:
- [x] `hostmapped` backend in `include/common.h` (`#ifdef MORPHX_SYNC_BACKEND_HOSTMAPPED`, same `GdrPool` API)
- [x] `MORPHX_SYNC_BACKEND=gdr|hostmapped` CMake option (also read from the env var) in
      `runtime/`, `microbench/`, `ggnn/`; `-lgdrapi` is only linked for `gdr`
- [x] Probe job for mapped-memory atomics and latency: `scripts/ae/sync_backend_test.cu`
- [x] `memcpy` backend (option C): `MORPHX_SYNC_BACKEND=memcpy` → `-DMORPHX_SYNC_BACKEND_MEMCPY`;
      default in `scripts/ae/env-h100.sh` (results below)
- [x] `launch_fence(stream)` before every patched launch in `cuda.cpp` (incl. reclaim) and
      `cuda-llm.cpp`; no-op for `gdr`/`hostmapped`. Other runtime variants compile but are
      not fenced (not used by the AE)
- [x] Rerun the Part 3 (Fig 15) overhead smoke: `memcpy` before/after Track C3 (below);
      `hostmapped` not rerun (orders of magnitude slower, see above)
- [x] Batch the agent thread's reads (one `get_many` per loop instead of one `get` per kernel)
- [x] Track C3: one-copy launch init, launch-path caches, scheduler-thread pinning inside
      the job's cpuset, opt-in `MORPHX_AGENTS=grid` (below)
- [ ] Optional: pre-initialized launch slots (no copy and no fence on the launch path; see
      "Remaining overhead" below)

### `hostmapped` results on the hpcc H200 (job 91113, 2026-10-22)

PCIe Gen5, `hostNativeAtomicSupported=0`, `pageableMemoryAccess=0`.

| Test | Result |
|---|---|
| Counter correctness: 8448 blocks contending on `agents`/`fetched`/`finished` | PASS: exact counts |
| CPU sees `finished` increase while the kernel runs | PASS: 1054 distinct values seen |
| CPU write → running kernel sees it → kernel writes back → CPU sees it | PASS: **2.1 µs** round trip |
| Kernel reads `maxSM` written by the CPU and skips SMs ≥ maxSM | PASS: 264 of 528 blocks skipped at maxSM=66 |
| Per-block cost of the protocol vs device memory | **+4.7 µs per block, serialized**: 16 896 blocks take 80 ms vs 0.1 ms |

Conclusion: `hostmapped` is **functionally correct**, so it can be used for
smoke tests and for checking the scheduling logic. But every block's
atomics on the same few addresses serialize over PCIe, which caps a kernel at
about 210k blocks/s. Kernels with many short blocks (GEMM tiles, attention)
will slow down a lot, so **don't use `hostmapped` for performance numbers**.
For those, implement option C (atomics stay in device memory) or get GDRCopy
installed.

### `memcpy` backend design and results (H200, jobs 91340–91360, 2026-10-22)

Pool in device memory (`cudaMalloc`), so the per-block protocol is plain device
atomics. The CPU side uses a private non-blocking, highest-priority control
stream:

| Operation | How | Host cost |
|---|---|---|
| `set(entry, v)` (launch-time init, `writeSM`, reclaim) | 8-byte `cuMemcpyHtoDAsync` from a pinned ring slot; enqueued in host order, so the last writer wins as with GDRCopy | 2.2 µs |
| `get(entry)` (agent thread, profiler polling) | 8-byte `cuMemcpyDtoHAsync` into a pinned slot, then sync of the control stream only | 7.5–8 µs |
| `launch_fence(stream)` before a patched launch | event recorded on the control stream + `cuStreamWaitEvent` on the launch stream (no host sync); `MORPHX_MEMCPY_LAUNCH_SYNC=1` syncs on the host instead | 2 µs (14 µs with host sync) |

Copies only, no memsets: a memset may run as a kernel and couldn't start while
persistent agents fill every SM. `scripts/ae/copy_probe.cu` confirms 8-byte
D2H/H2D copies finish in ~10 µs with every SM busy.

| Test | `hostmapped` | `memcpy` |
|---|---|---|
| `sync_backend_test`: counters, live progress, host→running kernel, maxSM read, launch ordering | all PASS | all PASS |
| Host → running kernel → host round trip | 1.9 µs | 9.5 µs |
| Per-block protocol cost vs device memory (16 896 blocks) | +4.9 µs/block (83 ms vs 0.1 ms) | **+0.00 µs/block** |
| `runtime/test/run_hook_test.sh` (cuda + cuda-llm, LAZY/EAGER) | PASS | PASS |
| `runtime/test/run_pass_test.sh` (pass-built kernel, `fix_SM` range on profiling launch) | PASS | PASS |
| Fig 16 bench, CUTLASS GEMM M=16, N=K=65536 (no runtime: 2.63–2.65 ms) | 5.4–6.2 ms | **2.61–2.67 ms** (also with `SMSCHED_PROFILE=0` / launch sync) |
| Fig 16 part-4 runner, all 4 GEMM series (job 91360) | rc 0, CUTLASS 6.08 / 3.70 / 14.8 / 154 ms | rc 0, CUTLASS **2.65 / 2.46 / 9.68 / 137 ms** (cuBLAS 2.12 / 2.16 / 5.26 / 86.3 ms) |
| SGLang Qwen3-4B fp16 + mycutlass, 8 req, in 512 / out 32 (no runtime: prefill 74.2 ms, decode 5.29 ms/step, TTFT 96.6 ms, TBT 19.1 ms) | prefill 1095 ms, decode 228 ms/step, TTFT 857 ms, TBT 251 ms | prefill **77.4 ms**, decode **6.10 ms/step** (+15%), TTFT 34.1 ms, TBT 21.0 ms |
| SGLang greedy tokens vs torch under the runtime | 192/192 | **192/192** identical |

When running the part-4 runner with a hook library other than
`runtime/build/libcuda.so`, set `NEUTRINO_HOOK_DRIVER` to the same library in
`--smsched-model-env`; otherwise `libpreload.so` loads the other build too
and the process segfaults at exit.

### Runtime overhead with `memcpy` (Track C3, H200 jobs 91424–91475, 2026-10-22)

What changed (`include/common.h`, `runtime/src/cuda.cpp`, `runtime/src/cuda-llm.cpp`):

| Change | Before (C2) | After (C3) |
|---|---|---|
| Launch-time control words | 3 separate pool entries, 3 `set()` copies + `launch_fence()` | one 32-byte `LaunchSlot` (agents, minSM\|maxSM, fetched\|finished) written by **one** 24-byte copy (`launch_init`). `MORPHX_MEMCPY_INIT=control` (default): on the control stream + event wait; `=stream`: on the launch stream itself, no event and no lock |
| Copy source | pinned ring slot | pinned **host mirror** of the pool; every `set()` updates the mirror first, so an init copy that runs late still carries the newest `minSM`/`maxSM` (a `writeSM` issued before the kernel starts wins in both modes) |
| Reused slot, `stream` mode | — | fetched\|finished zeroed on free; a block is reused only after that write is known to have landed, so reads never see the previous launch's counters |
| Agent-thread reads | one D2H + sync per active kernel per loop | one `get_many` per loop (n=1: 8.1 µs, n=2: 10.8 µs, n=4: 15.2 µs vs 8.1 µs each) |
| Per-launch driver queries | parameter sizes, occupancy, `MAX_DYNAMIC_SHARED_SIZE` attribute every launch | cached per patched function (`MORPHX_LAUNCH_CACHE=0` restores) |
| Scheduler-thread pinning | CPU 0, 1, …: fails under PBS (`pthread_setaffinity_np failed`), thread runs unpinned | last CPU of the process's allowed set (`MORPHX_AGENT_CPU=<n>` / `-1`) |
| `cuda-llm.cpp` agents word | `cudaMallocAsync` + pageable `cudaMemcpyAsync` per launch | part of the `LaunchSlot` (also under `gdr`) |
| Profiled launch in `cuda-llm.cpp` | `KernelInfo` and its entries leaked | freed after the profile thread joins |
| `MORPHX_AGENTS=grid` (opt-in, `cuda.cpp`) | always `occup × nrSM` agents | `max(gridSize, nrSM)` agents for small grids, only while one stream is active and the kernel gets the whole GPU; such kernels are never shrunk later (fewer agents don't cover every SM range by pigeonhole, which is why the original launches `occup × nrSM`). Co-located launches are unchanged |

`MORPHX_LAUNCH_STATS=1` prints host time before the real launch and the agent
thread's batched-read time at exit. In SGLang decode a patched launch spends
**6.6 µs** in the runtime before `cuLaunchKernel`.

Tests (all on the H200, control and stream init, with and without
`MORPHX_AGENTS=grid`): `sync_backend_test` (new: 8 `launch_init` on a busy stream
gives every launch exactly its own words, freed slots are zeroed in stream mode;
9 a late `set()` wins; 10 `get_many` cost) PASS;
`run_hook_test.sh` 5/5 PASS; `run_pass_test.sh` 4/4 PASS (profiling launch with
`fix_SM(0,66)` stays on SMs 0–65); SGLang greedy tokens 192/192 identical.

Results (before = C2 runtime at `220b0cf`; Fig 15 / SGLang numbers from the
same job 91475, two runs each):

| Benchmark | no runtime | before (C2) | after (C3) | after + `MORPHX_AGENTS=grid` |
|---|---|---|---|---|
| Launch test, 64-block kernel (`runtime/test/launch_overhead_test.cu`): GPU time alone / launch+sync latency / host µs per launch in a burst | 6.1 / 8.3 / 2.3 µs | 26.6 / 28.0 / 19.4 µs† | 17.7–20.9 / 21.8–24.1 / 10.1–11.6 µs | 16.2 / 22.1 / 10.5 µs |
| Launch test, 1056-block kernel (~50 µs) | 53 / 56 / 24 µs | 89 / 83 / 31 µs† | 76–77 / 75–82 / 32 µs | 74 / 76 / 31 µs |
| Fig 16 bench GEMM M=16, N=K=65536 (CUTLASS) | 2.627 ms | 2.660 ms | 2.627 ms (stream: 2.667) | — |
| Fig 15 GEMM (MorphX / base) | 1.000 | 1.044, 1.048 | 1.047, 1.051 | 1.038, 1.011 |
| Fig 15 GEMV | 1.000 | 1.759, 1.644 | 1.598, 1.529 | **1.443, 1.432** |
| Fig 15 LLM (Qwen3-8B, in 4096 / out 3) | 1.000 | 1.104, 1.043 | 1.040, 1.050 | 1.038, 1.068 |
| Fig 15 DNN (cuDNN, not patched) | 1.000 | 0.981, 0.993 | 1.000, 0.990 | 0.998, 1.001 |
| SGLang Qwen3-4B + mycutlass: decode GPU time per step | 5.36, 5.28 ms | 8.52*, 6.10 ms | 6.02, 6.29 ms | **5.59, 5.64 ms** |
| SGLang: TBT mean / TTFT mean | 17.9–18.1 / 32.4–32.5 ms | 20.1–22.4 / 35.1–42.2 ms | 19.0–19.3 / 35.1–36.6 ms | 19.1–19.5 / 33.9–34.4 ms |
| Fig 11(a) `llm-1-mm-1`, 60 s: LLM mean latency (stream: 4209 ms, TTFT 30.1) | — | 4311 ms (+2.4%), TTFT 33.4 | 4107 ms (−2.4%), TTFT 32.9 | 4002 ms (−4.9%), TTFT 32.2 |

\* one run with an outlier; the other 6.10 ms. † job 91424, means; the other
launch-test numbers are medians (jobs 91440, 91449). Fig 11 single runs vary by
~10% between jobs (stream latency was 3782 ms in job 91429), so read that row
as "no longer worse than stream", not as a measured gain. Fig 15 LLM varies by
±3% run to run.

The SM split does happen in co-location (job 91429, 30 s of `llm-1-mm-1` with
`SMSCHED_DEBUG`): while the mm GEMM (compute-intensive) runs, `schedule_1`
moves the LLM's GEMMs to SMs `[93,132)` and the CI GEMM gets `[0,93)`. At this
point the mm side is busy ~2% of the time, so only 23 of 16 877 patched
launches were split.

Remaining overhead:
- **GPU side, small grids:** a patched kernel launches `occup × nrSM` agents and
  every agent does an atomic on `agents` and on `fetched` (same two words). In
  SGLang decode that is +0.8 ms GPU per step (5.3 → 6.1 ms); `MORPHX_AGENTS=grid`
  cuts it to +0.3 ms but only applies while one stream is active.
- **Launch latency on an idle stream:** the kernel can't start before its
  control words land (control copy + event, or the in-stream copy): ~10 µs.
  This is what keeps GEMV at ~1.45–1.6× and the 64-block test kernel at ~18 µs
  vs 6 µs. Hidden when the stream is already busy (SGLang decode, Fig 16).
  Removing it needs pre-initialized slots that the launch path takes without a
  copy or fence; a slot could only be re-initialized after every agent of its
  previous launch has started (`agents == 0`) plus a delay, because late agents
  still touch `fetched` (the existing FIFO pool relies on its ~1M-launch reuse
  distance for the same reason).
- **Host side:** ~6.6 µs per patched launch before the real launch (stream
  registration lookup, cached queries, argument copies for reclaim, slot init),
  vs ~2 µs for a plain launch.
- `MORPHX_MEMCPY_INIT=stream` vs `control`: no consistent difference in any
  benchmark; `control` stays the default.

## 6. Toolchain and software stack on the H100 host

Target: **CUDA 13.0** (`module load cuda/13.0` on hpcc; driver 580.126.20
reports CUDA 13.0, so 13.1 is not supported by that driver yet).

1. [ ] CUDA 13.0 toolkit. CUDA 13 changes that affect this repo:
       - The repo pins `cuda-12.4` paths and `-D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=4`
         (`microbench/CMakeLists.txt:36-37`, `ggnn/CMakeLists.txt:18`) → 13 / 0.
       - CCCL (Thrust/CUB/libcudacxx) is 3.x in CUDA 13 with API removals. Use
         the toolkit's bundled CCCL instead of `/home/rtx/gpu/cccl`, and fix any
         compile errors in `microbench/` and `ggnn/`.
       - **Confirmed on hpcc:** `runtime/src/unmodified.c` (generated by
         `runtime/src/parse.py` from the CUDA 12.4 `cuda.h`) does not compile
         against CUDA 13.0: `cuDeviceGetUuid_v2` is redefined, and
         `cuMemAdvise_v2`, `cuMemPrefetchAsync_v2`, `cuGraph*_v2` and
         `cuStreamUpdateCaptureDependencies_v2` have conflicting declarations.
         Regenerate it with `parse.py` against `$CUDA_HOME/include/cuda.h`
         (13.0) and fix up the hand-written hooks. `include/common.h` itself
         compiles fine.
       - The interception layer hooks `cuGetProcAddress`/`_v2`
         (`runtime/src/cuda.cpp`, `cuda-llm.cpp`, `cuda_hook_exports.map`).
         CUDA 13 runtimes ask for different versioned driver symbols
         (`cudaVersion=13000`). Check that every hooked symbol is still
         intercepted (run with `SMSCHED_DEBUG=1` and confirm kernels are rewritten).
       - Deprecated `cudaDeviceProp` fields were removed in 13.0. The runtime
         only reads `multiProcessorCount` (`cuda.cpp:247`, `cuda-llm.cpp:232`),
         which is still there, but recompile everything to check.
       - sm_90 is still supported (CUDA 13 only dropped sm_50–sm_72).
2. [ ] Build LLVM/clang with the NVPTX target and libc++ (the builds use
       `-stdlib=libc++`). It must support `--cuda-gpu-arch=sm_90` **and** the
       CUDA 13.0 headers. Clang releases trail new CUDA versions, so build a
       recent release (20+) and confirm with a test `.cu`. If clang rejects
       CUDA 13, try `--cuda-path=$CUDA_HOME -Wno-unknown-cuda-version` before
       anything else.
3. [ ] Build `smsched-pass` against it:
       `cmake -S smsched-pass -B smsched-pass/build -DLLVM_DIR=$MORPHX_LLVM_ROOT/lib/cmake/llvm`.
4. [ ] Build `runtime/` (both the `cuda.cpp` and `cuda-llm.cpp` variants).
5. [ ] Create the `smsched` conda env and rebuild the Python stack from source
       with `TORCH_CUDA_ARCH_LIST=9.0`, compiled with the patched clang and pass.
       **Primary (SGLang-first, §9):** PyTorch 2.9.1, FlashInfer 0.6.8.post1
       (AOT, no `flashinfer_cubin`), `sgl-kernel` and SGLang matching
       `~/sglang-main`, on CUDA 13.0. Python 3.12 matches the cluster's
       `python/3.12.13`; GGNN finds Python from the active env (Track B).
       **Fallback if the SGLang spike says no-go:** stock vLLM with the V0
       engine (the scripts force `VLLM_USE_V1=0` and drive `LLMEngine.step()`;
       roughly the 0.8.x series, which pairs with PyTorch 2.6) on a CUDA 12.8
       toolkit from conda, plus our own prefill thread for Fig 13–14. The
       notes below describe the original vLLM stack and apply to that fallback.
       - **Biggest risk with CUDA 13:** the original env uses
         `torch==2.6.0a0`, `flashinfer-python==0.2.2.post1` and a **forked vLLM**
         (`0.1.dev24+g232ce74`, built from `/home/rtx/gpu/vllm`). PyTorch 2.6
         predates CUDA 13 (official CUDA 13 support starts at PyTorch 2.9), so
         it probably won't build against 13.0 unchanged. Choices:
         (a) upgrade to PyTorch ≥ 2.9 + matching vLLM/FlashInfer and re-apply
         the MorphX vLLM changes (e.g. `VLLM_PREFILL_THREAD`), or
         (b) backport CUDA 13 build fixes to torch 2.6.
       - **The forked vLLM source is not public** (commit `232ce74` is not in
         upstream vLLM). This is why the plan moved to SGLang-first.
       - Fallback if CUDA 13 blocks the stack: driver 580 also runs CUDA 12.x
         toolkits (e.g. `conda install -c nvidia cuda-toolkit=12.8`), keeping
         the original versions.
       - Risk: those versions' Hopper-only kernels (FlashAttention-3,
         WGMMA/TMA paths in FlashInfer/vLLM) may not compile with clang's CUDA
         frontend. Fallback: disable sm_90a-only kernels and use the sm_80 code
         paths. This also keeps kernels comparable to the A100 results.
6. [ ] Build GGNN (`ggnn/`) with the pass.
7. [ ] Install baselines: `neutrino` CLI (Figure 15; optional, reports N/A if missing),
       NVBit `mem_trace` tool, TGS hijack libs, `nanoflow` env if used.
8. [ ] Download models: Llama-2-7b-chat-hf, Meta-Llama-3-8B-Instruct (both gated),
       Qwen3-4B, Qwen3-8B, GLM-Z1-9B-0414, gpt2-large; ShareGPT_Vicuna_unfiltered dataset.

## 7. Bring-up order and validation

| Step | Command / check | Pass criterion |
|---|---|---|
| 1 | `./run.sh analyse-reference` | Tables/figures regenerate (no GPU needed) |
| 2 | `%smid` probe (section 4) | IDs understood; fix applied if needed |
| 3 | Build pass + runtime; `./run.sh smoke --no-build` | All four parts' smoke points run |
| 4 | Part 4 only (Figure 16, microbench GEMM) | Model-accuracy table generated; needs no PyTorch/vLLM |
| 5 | Part 3 (Figure 15 overhead) | `base` and `smsched` columns present |
| 6 | Re-tune split hints: sweep `SMSCHED_SPLIT_HINT` for each workload | Pick the best per row; record in part1/part2 scripts |
| 7 | `./run.sh short` | Fig 10/12(a), 13(a), 15, 16 produced |
| 8 | `./run.sh` (full, >20 h; run under `tmux` or a batch job) | All figures produced |

## 8. Smoke testing on `hpcc` (H200, free queue)

The H200 is also Hopper: sm_90, 132 SMs, the same as H100 SXM. It has more
HBM (141 GB) and higher bandwidth. Every change in sections 1–5 can be
tested there unchanged, so use it for bring-up before the H100 runs.

### Cluster facts (checked from the login node `hpc-gaas-hn2`)

| Item | Value | Impact on the plan |
|---|---|---|
| Scheduler | PBS Pro (`qsub`/`qstat`), not Slurm | Jobs below are PBS scripts |
| Queue | `-q gpu_free` (routes to `gpu_free_normal`) | You have access through `hpc_fs_dmitrii.ustiugov_group` |
| Project code | `-P fs_ccds_dmitrii.ustiugov` | As in your existing job scripts |
| Limits | ≤ 4 GPUs, **≤ 6 h walltime**, **1 running job per user** | Full `./run.sh` (>20 h) must be split across jobs using `./run.sh resume` |
| Resources | Request `select=1:ncpus=12:ngpus=1:mem=120gb` | All AE experiments use only GPU 0 |
| OS / compilers | Rocky Linux 9.7; modules `gnu/gcc-12.3`, `cuda/13.0`, `cuda/13.1`, `anaconda/2025` | gcc-12 matches `runtime/CMakeLists.txt`; use `cuda/13.0` (matches driver 580) |
| LLVM/clang | Not available as a module | Build from source into `$HOME` (in a job, not on the login node) |
| GDRCopy | **Missing** on login and GPU nodes (no `/dev/gdrdrv`, no module, no `libgdrapi`) | Use section 5 backend `hostmapped`/`memcpy` |
| MPS | **Works** as a normal user (job 91099) | MPS baseline can run on hpcc |
| GPU | NVIDIA H200, 143 GB, 132 SMs, compute mode Default | |
| `CUDA_VISIBLE_DEVICES` | PBS may set GPU UUIDs (`GPU-xxxx`) | Convert to index `0` before calling `run.sh` (see script) |
| Storage | `/home` NFS, ~816 GB free | Fine for the env, LLVM build and models (~80 GB) |

### Cluster-specific risks

- **CUDA version.** We target the cluster's `cuda/13.0` module (driver
  580.126.20 = CUDA 13.0; don't use `cuda/13.1`). See section 6 for the CUDA 13
  porting items. The SGLang stack (§9) targets CUDA 13 natively.
- **GDRCopy.** Confirmed missing on the compute node. Use the
  `hostmapped`/`memcpy` backend from section 5, or ask HPC support to install it.
- **MPS.** Confirmed working (job 91099): a user-level daemon with
  `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` in `/tmp` starts, three
  concurrent clients connect to one MPS server, and
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is enforced (30% → 38 SMs, 50% → 66 SMs).
  The AE scripts already use per-run pipe dirs under `/tmp`
  (`run_colocate_part1.py:545`), so no change is needed.
- **Shared node.** Other jobs share the host, so timings are noisy. Use hpcc
  to check correctness, not to produce final numbers.

### Smoke test stages

Run them in order; each one must pass before the next.

| Job | Walltime | What it does | Pass criterion |
|---|---|---|---|
| A. `probe` ✅ | 0:30 | `nvidia-smi`, driver/CUDA version, `ls /dev/gdrdrv`, `lsmod \| grep gdrdrv`, try a user-level MPS daemon, run the `%smid` probe (section 4) | **Done, job 91099 on `hpc-gaas-g25`:** H200, 132 SMs, driver 580.126.20 / CUDA 13.0, nvcc 13.0.48; gdrdrv missing; MPS works; smid 0..131 with no gaps |
| A2. `probe-hostmapped` ✅ | 0:20 | `scripts/ae/sync_backend_test.cu` (section 5) | **Done, job 91113:** all correctness checks pass; +4.7 µs/block serialized overhead |
| B. `build-llvm` | 6:00 (`ncpus=48`) | Build LLVM/clang + libc++ with NVPTX into `$HOME/morphx/llvm` | `clang++ --cuda-gpu-arch=sm_90` compiles a test `.cu` |
| C. `build-stack` | 6:00, chained with `resume` | Build PyTorch 2.9.1, FlashInfer 0.6.8 (AOT) and `sgl-kernel` with the pass for `TORCH_CUDA_ARCH_LIST=9.0` (§9 step 1) | `import torch, flashinfer, sglang` works; Qwen3-4B generates text through `ModelRunner` |
| D. `smoke-part4` | 1:00 | Build pass + runtime + microbench; Part 4 only (Fig 16) | Fig 16 table produced |
| E. `smoke` | 2:00 | `./run.sh smoke` | All four parts' smoke points finish with `status=ok` |
| F. `short` | 6:00 ×N | `./run.sh short`, then `./run.sh resume` in chained jobs | Fig 10/12(a), 13(a), 15, 16 produced |

Chain jobs so the next one starts when the previous finishes (only one runs at
a time anyway):

```bash
J1=$(qsub -v MODE=short morphx-hpcc.pbs)
J2=$(qsub -W depend=afterany:$J1 -v MODE=resume morphx-hpcc.pbs)
J3=$(qsub -W depend=afterany:$J2 -v MODE=resume morphx-hpcc.pbs)
```

`run.sh resume` restarts at run granularity, so a job killed at the 6 h limit
only loses the run that was in progress.

### PBS job template (`scripts/ae/morphx-hpcc.pbs`, to be added)

```bash
#!/bin/bash
#PBS -N morphx-smoke
#PBS -P fs_ccds_dmitrii.ustiugov
#PBS -q gpu_free
#PBS -l select=1:ncpus=12:ngpus=1:mem=120gb
#PBS -l walltime=06:00:00
#PBS -j oe

set -euo pipefail
MODE="${MODE:-smoke}"            # smoke | short | resume | analyse | probe

module purge
module load gnu/gcc-12.3 cuda/13.0 anaconda/2025
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smsched

source "$HOME/MorphX/scripts/ae/env-h100.sh"   # MORPHX_* vars (section 1)
export MORPHX_NUM_SMS=132
export MORPHX_SYNC_BACKEND=hostmapped   # no GDRCopy on hpcc (section 5)

# PBS may hand out GPU UUIDs; the AE scripts expect index 0.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi

cd "$HOME/MorphX"
echo "host=$(hostname) job=${PBS_JOBID} mode=${MODE}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [[ "$MODE" == probe ]]; then
    ls -l /dev/gdrdrv || echo "gdrdrv: MISSING"
    lsmod | grep gdrdrv || true
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-$PBS_JOBID" CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$PBS_JOBID"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d && echo "MPS: OK" && echo quit | nvidia-cuda-mps-control || echo "MPS: FAILED"
    nvcc -arch=sm_90 -o /tmp/smid_probe scripts/ae/smid_probe.cu && /tmp/smid_probe   # section 4 probe
    exit 0
fi

./run.sh "$MODE" > "run-${PBS_JOBID}.log" 2> "run-${PBS_JOBID}.err"
```

### Getting the code there

```bash
ssh hpcc
git clone https://github.com/Pwspang/MorphX.git ~/MorphX
cd ~/MorphX && git checkout h100-port
qsub -v MODE=probe scripts/ae/morphx-hpcc.pbs
```

### 8.1 Baselines on hpcc (Track F)

Setup is all user-level (no admin requests). `scripts/ae/env-h100.sh` now puts
`$MORPHX_CUDA_ROOT/bin` (ncu) and `~/.local/bin` (CMake 4.4, patchelf) on `PATH`.
Validation job: `qsub scripts/baselines/f_baselines.pbs`.

| Baseline | Used by | Setup | Status |
|---|---|---|---|
| **ncu** | Fig 15 | Nsight Compute 2025.3.0 from `cuda/13.0`. `RmProfilingAdminOnly=1` on the login node, but counter collection worked on the GPU node (Track J2 job 91384 profiled with "1 pass") | Works |
| **NVBit** | Fig 15 | Vendored NVBit 1.7.6 (CUDA 13.0 headers per its release notes) + `tools/mem_trace` builds with nvcc 13.0 (`make -C nvbit-tutorial/tools/mem_trace`, also done by `run_all.py`). 1.7.7+ (CUDA 13.1 headers) and 1.8.x (CMake build) exist but aren't needed for CUDA 13.0 | Builds; GPU check in validation job |
| **neutrino** | Fig 15 | The authors' `neutrino==0.1.0` is **Neutrino** (OSDI'25, github.com/open-neutrino/neutrino, commit `4a82cd2`); PyPI's `neutrino` is an unrelated Redis package. `scripts/baselines/neutrino_setup/install_neutrino.sh` clones it, swaps in a CUDA 13 `parse.py` (same `_v2` macro fixes as `runtime/src/parse.py`), patches `build.py` for GPU-less login nodes and `$CUDA_HOME`, points the real driver at `/usr/lib64/libcuda.so.1` (the login node has driver 580.65.06, GPU nodes 580.126.20) and installs editable into `~/morphx/envs/sglang-stock` | Installed; GPU check in validation job |
| **TGS** | Fig 10–12 | `pip install --user patchelf` (0.19.1), then `TGS/hijack/build.sh` builds `high/low-priority-lib` with gcc 12.3 / CUDA 13 headers. The libs are LD_PRELOADed into the LLM / background processes (`--use-tgs --use-multiprocess`); the real driver is found as `libcuda.so.<version from /proc/driver/nvidia/version>`. Rate-watcher sockets go to `$MORPHX_TGS_SOCKET_DIR`. The hijack doesn't export `cuLaunchKernelEx`, so attribute launches bypass TGS throttling | Builds; GPU check in validation job |
| **orion** | Fig 10–12 | Nothing to build: `scripts/colocate/run.py --use-orion` only forces multithread mode with a high-priority LLM stream and low-priority background streams (`runtime/src/cuda-orion.cpp` is a separate experimental runtime variant not used by the AE) | Should work; GPU check in validation job |
| **timeslice** | Fig 10–12 | Nothing to build: LLM and background work in separate processes without MPS, so the driver time-slices them (default compute mode, no root needed) | Should work; GPU check in validation job |

## 9. SGLang backend (primary LLM engine)

**SGLang-first** (decision 2026-10-22, see top of this document). SGLang
replaces the authors' vLLM fork as the LLM engine. The vLLM code paths stay in
the scripts (selected with `MORPHX_LLM_BACKEND=vllm|sglang`, default `sglang`
on the H100/H200 setup) so the fallback in §6.5 remains possible.
Target versions: as in `~/sglang-main` on hpcc: PyTorch 2.9.1 (CUDA 13.0
wheels), FlashInfer 0.6.8.post1, matching `sgl-kernel`.
Estimated effort: **3–6 weeks** after the spike.

### Why it isn't a drop-in swap

| # | Issue | Where in the repo | What has to change |
|---|---|---|---|
| 1 | **Process model.** The MorphX runtime splits SMs between the CUDA streams of *one process*. The co-location driver runs vLLM in-process (`LLMEngine`, eager) in one thread and GGNN/GEMM in another. SGLang's `Engine` always runs the model in a separate scheduler subprocess, so the LLM and the co-located task would end up in different processes | `scripts/colocate/run.py:300-312`, thread workers in the same file | Drive SGLang's `ModelRunner` directly in-process (as `bench_one_batch` does) with a small batching loop, **or** run the co-located task inside the scheduler process. Prefer `ModelRunner` |
| 2 | **PD co-location depends on the vLLM fork.** Fig 13–14 use `VLLM_PREFILL_THREAD`: prefill and decode on separate streams in one process. SGLang's PD disaggregation is multi-process/multi-GPU | `scripts/ae/run_pd_colocate_part2.py:639-650`, `scripts/test-pd-online.py` | Add a prefill thread with its own CUDA stream to the `ModelRunner` loop from #1 (separate prefill and decode batches, shared KV cache pool, one lock around the KV allocator). The fork's source is not public, so this is designed from scratch |
| 3 | **Kernels must be built with the MorphX pass.** Only pass-built kernels can be resized. SGLang uses `sgl-kernel` (CUTLASS 3.x / FA3, built with nvcc), many **Triton** kernels and JIT kernels (DeepGEMM). FlashInfer 0.6 ships **precompiled cubins** (`flashinfer_cubin`) and **JIT-compiles** the rest at run time. The LLVM pass touches none of these, so they'd run without MorphX control | build stack (section 6) | Build `sgl-kernel` with clang + the pass for sm_90; build FlashInfer **AOT** with clang + the pass (no `flashinfer_cubin`, JIT disabled), or point its JIT compiler at clang + the pass; use `--attention-backend flashinfer`; use dense models so GEMMs go through PyTorch like in vLLM; disable Triton/DeepGEMM paths where SGLang allows it; list which kernels are left unpatched |
| 4 | **Runtime matches kernels by name.** `cuda-llm.cpp` decides how to split SMs from vLLM/FlashInfer kernel names (`BatchPrefillWithPagedKVCacheKernel`, `BatchDecode…`, `cutlassL6Kernel`, mangled CUTLASS names) | `runtime/src/cuda-llm.cpp:643-655, 1021-1031` | Profile SGLang with `SMSCHED_DEBUG=1` / nsys, then move the patterns into a per-backend list (e.g. `SMSCHED_TARGET_PATTERNS` default per backend) instead of hardcoded strings |
| 5 | **CUDA graphs.** SGLang captures decode in CUDA graphs by default. Graph replay skips `cuLaunchKernel` interception | engine launch args | Always pass `--disable-cuda-graph` (the equivalent of vLLM's `enforce_eager=True`); also disable the overlap scheduler at first |
| 6 | **Experiment drivers use the vLLM API.** `EngineArgs`/`LLMEngine.step()`, `SamplingParams`, `TokensPrompt`, `RequestOutput` metrics (`time_in_queue`, TTFT/TBT) | `scripts/colocate/run.py` (1255 lines), `scripts/test-pd-online.py`, `scripts/overhead/single_request.py` | Put a thin `LLMBackend` interface in front (`init`, `add_request`, `step`, `collect_metrics`), with `vllm` and `sglang` implementations; keep the Poisson arrival and timing code shared |
| 7 | **Environment.** `VLLM_ATTENTION_BACKEND`, `VLLM_USE_V1`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN` env vars; `without_child_ld_preload()` assumes the model runs in-process | same scripts | Backend-specific env setup; make sure `LD_PRELOAD` of the MorphX runtime reaches whichever process launches the kernels |

### Per experiment

| Experiment | SGLang work | Effort |
|---|---|---|
| Fig 16 (GEMM model) | None, no LLM | — |
| Fig 15 (single-request overhead) | `single_request.py` on `ModelRunner`, CUDA graphs off | Small–medium |
| Fig 10–12 (co-location) | #1 in-process runner, #4 kernel patterns, new driver | Large |
| Fig 13–14 (PD co-location) | #2 prefill thread/stream in SGLang | Largest |
| Stack build | `sgl-kernel` with the pass instead of the vLLM fork (#3) | Similar to vLLM, maybe harder because of Hopper-only CUTLASS 3 / FA3 code |

### Steps

1. [ ] **Spike on hpcc = go/no-go (about 1–2 days once LLVM is built):**
       build PyTorch 2.9.1, FlashInfer 0.6.8 (AOT) and `sgl-kernel` with
       clang + the pass for sm_90 on CUDA 13.0; load Qwen3-4B through
       `ModelRunner` in-process with the MorphX runtime preloaded
       (`MORPHX_SYNC_BACKEND=hostmapped`, `--disable-cuda-graph`,
       `--attention-backend flashinfer`).
       - Measure: share of GPU time (nsys) in pass-built (`_pk`) kernels vs
         Triton / JIT / cuBLAS / precompiled kernels, per prefill and decode step.
       - Check: with `SMSCHED_SPLIT_HINT` set, patched kernels only run on
         the allowed SMs (`smid_probe`-style check via `SMSCHED_DEBUG`).
       - **GO** if the main GEMM and attention kernels are controlled (rule of
         thumb: ≥ 70% of GPU time in controlled kernels). **NO-GO** → stock
         vLLM fallback (§6.5).
       - Early look before the pass-built stack exists: stock SGLang wheels +
         runtime hook to list kernel names and the time split.
2. [x] Add the `LLMBackend` interface; port `single_request.py` → Fig 15 with SGLang
       (Track J0/J2). `MORPHX_LLM_BACKEND=sglang` + `MORPHX_LLM_GEMM=mycutlass`
       (selected by `env-h100.sh`; the vLLM path is unchanged and stays the default).
       The GEMM/GEMV workloads also go through mycutlass: the reference A100 Fig 15
       shows MorphX overhead on GEMM (1.44×) and GEMV (1.34×), so the authors'
       `torch.matmul` ran pass-built kernels, not cuBLAS. DNN (ResNet-50, cuDNN)
       stays on torch, as in the reference (1.01×). GGNN waits for Track K.
3. [ ] Profile kernel names, make the `cuda-llm.cpp` patterns configurable per backend.
4. [~] In-process co-location driver on `ModelRunner` → Fig 10–12 with SGLang (Track J2):
       `scripts/colocate/run.py` uses the backend switch; its GEMM worker uses
       mycutlass. Fig 11 (LLM + GEMM) runs; Fig 10/12 need GGNN (Track K).
       `configs/multitask/*.json` (incl. A.json with two LLMs) are not used by any
       AE script — `run.py` only ever starts one LLM — so two SGLang runners per
       process are not needed for the AE.

#### Track J2 results on hpcc H200 (job 91384, memcpy backend, sglang + mycutlass, Llama → Qwen3 via `MORPHX_MODEL_ALIASES`)

Fig 15 (ms; LLM = Qwen3-8B fp16, 4096 in / 3 out; A100 reference in brackets):

| Task | base | MorphX | MorphX rel. | ncu rel. |
|---|---|---|---|---|
| DNN | 33.09 | 33.08 | 1.000 (1.010) | N/A |
| GEMM 8192³ | 2.368 | 2.484 | 1.049 (1.441) | 395 (200) |
| GEMV 8192×16×8192 | 0.087 | 0.149 | 1.713 (1.342) | 8156 (5553) |
| LLM | 177.4 | 192.7 | 1.087 (1.027) | 955 (103) |

GGNN, NVBit, neutrino: N/A (not built / not installed). mycutlass vs cuBLAS on
the raw shapes: 8192³ 2.32 vs 1.44 ms; GEMV 0.050 vs 0.048 ms; same max error.

Fig 11(a) point `llm-1-mm-1` (60 s instead of 180 s, Qwen3-8B, ShareGPT 2048/2048):

| Scheduler | Served | Avg TTFT (ms) | Avg latency (ms) | Avg TBT (ms) | Matmuls |
|---|---|---|---|---|---|
| stream | 48 | 28.9 | 3787 | 15.5 | 490 |
| MorphX | 48 | 33.6 | 4345 | 17.7 | 490 |
| MPS-50 | 48 | 46.9 | 4395 | — | 490 |
| A100 ref. stream / MorphX / MPS-50 (180 s, Llama-3-8B) | 156 | 46.6 / 45.9 / 61.4 | 5266 / 4910 / 5367 | — | 1700 |

At this lightest point the GEMM side is rate-limited (≈2 % of the GPU), so there
is little contention for MorphX to remove; its +15 % TBT matches the memcpy
backend's per-launch host cost measured in §5 (decode step +15 %).
5. [~] Prefill thread/stream in SGLang → Fig 13–14 with SGLang (Track J3). Runs end to
       end; MorphX doesn't split SMs yet on H200 (runtime change needed, below).
       - **Prefill thread** (`scripts/sglang/runner.py`, `VLLM_PREFILL_THREAD=1` or
         `MORPHX_PREFILL_THREAD=1`): a worker thread prefills on its own CUDA stream
         with its own attention backend (FlashInfer wrappers share one workspace per
         backend, so a second backend gets `init_new_workspace=True`) while the caller
         decodes on its stream. Finished prefills join the decode batch at the next
         `step()`; `get_prefill_result(isBlock=True)` blocks for one when nothing is
         decoding (the fork's API, used by `test-pd-online.py`). KV/req-slot
         bookkeeping (`_kv_section()`) is serialized with a lock and ordered across
         the two streams with events (the allocator's free list is a GPU tensor both
         threads slice and concatenate); merges sync the decode stream before the
         prefill-stream tensors are dropped.
       - **Chunked prefill** (`--enable-chunked-prefill`): vLLM V0-style mixed
         batches — every running request decodes and the rest of
         `max_num_batched_tokens` is filled with prompt chunks (the continuing
         chunked request first) in one forward, via `ScheduleBatch.mix_with_running`.
       - `test-pd-online.py` goes through `scripts/sglang/backend.py` (vLLM path
         unchanged) and warms up one request before the timed run
         (`MORPHX_LLM_WARMUP=0` to skip), so SGLang's JIT compiles and the MorphX
         runtime's first-launch profiling are not in the measurement.
       - **Correctness** (`scripts/sglang/check_pd_modes.py`, Qwen3-4B fp16 +
         mycutlass, 12 × 1024 in / 32 out, greedy): bookkeeping exact in every mode
         (all KV returned, reservations 0). Token-exact vs running each request
         alone: prefill thread 12/12 (also under the MorphX runtime), serial 10/12,
         chunked 11/12. Serial vs serial with reversed submission order is 12/12;
         the remaining differences come from fp16 numerics of different decode
         batch shapes (mycutlass picks tiles by shape), not from scheduling.
       - **What the MorphX LLM runtime (`cuda-llm.cpp`, `SMSCHED_LLM_TRICK_MODE=0`
         as in `run_all.py`) does with it:** only pass-built kernels are scheduled
         (here: the mycutlass GEMMs; FlashInfer/sgl-kernel are unpatched). A kernel
         profiled with CI ≥ 0.9 gets `[0, split)` at launch, and while it runs,
         `schedule_2` puts the other streams' kernels on `[split, nrSM)`. Kernels
         with a wave time < 50 µs bypass the runtime. Kernel names containing
         `float32` are dropped without launching (`cuda-llm.cpp:~950`); greedy
         tokens under the runtime matched those without it (Qwen3-4B), so nothing
         needed was dropped here, but it is a trap for other models.
       - **Finding: no PD split happens on H200.** The prefill GEMMs profile as
         memory-bound: CI 0.40 (128×128 tile, grid 768: 112 µs/wave at 132 SMs →
         72 µs at 33 SMs) and 0.41 (64×256, grid 594); decode GEMMs 0.13–0.51. So
         nothing gets `[0, split)` and `schedule_2` never restricts the decode
         stream; MorphX only adds overhead. Needed in the runtime (Track C3 or
         later): a stream-role hint from the app (e.g. the prefill thread marks its
         stream so its patched kernels get `[0, split)` and the other streams
         `[split, nrSM)` while a prefill kernel is active), or a CI threshold
         recalibrated for H200 + these Sm80-style CUTLASS kernels. Also `fix_SM` /
         `suggest_SM` are C++-mangled, so `libapi.so`'s `dlsym("fix_SM")` can't find
         them (see the hook-test note above).

#### Track J3 results on hpcc H200 (memcpy backend, sglang + mycutlass, `llama-3-8b` → Qwen3-8B fp16, 60 s instead of 180 s)

Fig 13 row `in-2048-out-256-rps-3` (job 91442; split hint 76 → 93):

| Variant | Served | TTFT avg (ms) | Latency avg (ms) | TBT avg (ms) | avg (s/token) |
|---|---|---|---|---|---|
| baseline | 148 | 74.5 | 5811 | 22.5 | 0.023 |
| stream (prefill thread) | 148 | 79.2 | 5492 | 21.2 | 0.021 |
| chunked | 148 | 87.7 | 5678 | 21.9 | 0.022 |
| MorphX (prefill thread + runtime) | 143 | 82.0 | 7178 | 27.8 | 0.028 |
| A100 ref. baseline / MorphX / stream / chunked (180 s, Llama-3-8B) | 413 / 445 / 437 / 416 | 257 / 532 / 295 / 237 | 18008 / 12427 / 13933 / 16745 | 69.6 / 46.6 / 53.5 / 64.7 | 0.070 / 0.049 / 0.054 / 0.065 |

On the H200 this row is light (TTFT ≈ 75 ms vs 257 ms on the A100), the
prefill thread already helps a little (stream −5.5 % latency vs baseline), and
MorphX is +24 % worse than baseline because it adds per-launch overhead without
splitting SMs (see the finding above).

Row `in-2048-out-256-rps-5` (job 91476), plus an experiment with a local,
uncommitted runtime build that forces CI = 1 for grids ≥ 500 blocks (the
prefill GEMMs) so the PD split actually happens:

| Variant | Served | TTFT avg (ms) | Latency avg (ms) | TBT avg (ms) |
|---|---|---|---|---|
| baseline | 242 | 74.1 | 6633 | 25.7 |
| stream | 242 | 81.1 | 6449 | 25.0 |
| MorphX | 236 | 89.7 | 8278 | 32.1 |
| MorphX, forced CI (split 93: prefill `[0,93)`, decode `[93,132)` while a prefill GEMM runs) | 232 | 105.6 | 10296 | 40.0 |
| rps-3, MorphX forced CI | 145 | 109.8 | 6711 | 25.9 |

With the split forced on, the runtime does what the paper describes (1180
prefill-stream launches limited to `[0,93)`, decode-stream kernels moved to
`[93,132)` while prefill runs), but at these loads it makes decode slower, not
faster: 39 SMs are too few for decode, and the H200 has spare capacity at the
A100 row loads. Before Fig 13–14 can show a MorphX gain on the H200 the split
hint needs re-tuning (or rows with more load), on top of the runtime changes
above. All J3 numbers use the runtime from before Track C3 (`runtime/` as of
6a2746d, memcpy backend); C3 (ae2f76e) cut the per-launch cost, so the MorphX
rows should be re-measured on it.
6. [ ] Add `--llm-backend` to `run.sh` / `scripts/ae/run_all.py`; keep logs separate
       (`logs/<timestamp>-sglang/`).

### Expected outcome

- A recent SGLang release supports CUDA 13 and Hopper. This avoids
  forward-porting PyTorch 2.6 and the old vLLM fork, which was the biggest
  risk in section 6.
- SGLang results are **not directly comparable** to the paper's vLLM numbers.
  Compare MorphX against the baselines (stream / MPS / chunked prefill) within
  SGLang.

## 10. Expected differences from the paper

- Absolute throughput/latency will differ (H100 vs A100 SM count, memory bandwidth, L2 size).
- The best split hints are different on H100; the scaled values above are only starting points.
- `logs/reference/` stays as the A100 baseline; keep H100 logs separate
  (e.g. `logs/h100-<timestamp>/`) so you can compare.
