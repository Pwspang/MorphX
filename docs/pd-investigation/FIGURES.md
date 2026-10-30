# Figures, testbed and the configuration behind each result

All figures are copied into [figures/](figures/) so this page renders on its own. The originals
and their data are under `logs/` (git-ignored). What was tried and why is in
[README.md](README.md).

## Figure index

"Runner" is the SGLang runner's prefill/decode stall. It was fixed on 2026-10-28 (see
*Fig 13/14 after the runner stall fix*). Figures marked *before the fix* were measured with the
stall in place: for Fig 13/14 they are superseded. Fig 10–12 were checked and are unaffected (no prefill thread there).

| Figure | What it shows | Runner |
|---|---|---|
| [figure13_fa3](figures/figure13_fa3.png), [figure14_fa3](figures/figure14_fa3.png) | **default setup (sm90 + FA3)**: baseline, stream, stream + high-priority decode, MorphX at 50 µs and at the tuned 35 µs | **fixed (current)** |
| [Fig 13 after fix](figures/Figure13-after-fix.png), [Fig 14 after fix](figures/Figure14-after-fix.png) | baseline / stream / MorphX, H200, Sm80 + pass FA2 (legacy setup) | fixed |
| [pd_runner_fix](figures/pd_runner_fix.png) | decode-step breakdown and TBT before / after the fix | before and after |
| [Fig 10](figures/Figure10.png), [Fig 11](figures/Figure11.png), [Fig 12](figures/Figure12.png) | LLM + GGNN / matmul co-location, H200 | before the fix, **unaffected** (stall check job 196057) |
| [Fig 13, 3× rates](figures/Figure13-x3-rates.png), [Fig 14, 3× rates](figures/Figure14-x3-rates.png) | AE runs, H200, stock FlashInfer | before the fix (superseded) |
| [Fig 13, paper rates](figures/Figure13-paper-rates.png), [Fig 14, paper rates](figures/Figure14-paper-rates.png) | AE runs at the paper's rates, H200 | before the fix (superseded) |
| [fig13_sm90_sweep](figures/fig13_sm90_sweep.png) | sm90 GEMMs, stream vs MorphX against rate | before the fix |
| [fig13_setups](figures/fig13_setups.png) | MorphX / stream per kernel setup | before the fix |
| [pd_decode_priority](figures/pd_decode_priority.png) | decode waiting and high-priority decode | before the fix |
| [a100_fig13](figures/a100_fig13.png), [a100_vs_paper](figures/a100_vs_paper.png), [a100_step_bench](figures/a100_step_bench.png) | A100 control | before the fix (no rerun planned) |

## Testbed

### Machines

| | H200 testbed (`ssh nscc2b`, NSCC ASPIRE2B) | A100 testbed (`ssh nscc`, NSCC ASPIRE2A) |
|---|---|---|
| GPU | NVIDIA H200, 141 GB, 132 SMs, one GPU per job | NVIDIA A100-SXM4-40GB, 108 SMs, one GPU per job |
| CPU | AMD EPYC Genoa | AMD EPYC Milan |
| CPUs / memory per job | 24 CPUs / 276 GB (AE runs: Fig 10–14 and MPS-60); 16 CPUs / 120 GB (all other H200 jobs) | one GPU's share of a node |
| OS | RHEL 9.6 (glibc 2.34) | RHEL 8.10 (glibc 2.28) |
| Driver / CUDA | 580.126.09, CUDA 13.0 (`module load cuda/13.0`) | 570.124.06 (CUDA 12.8) plus the CUDA 13.0 compat driver, CUDA 13.0.2 toolkit in user space (nvcc V13.0.88) |
| MorphX sync backend | **GDRCopy** (`MORPHX_SYNC_BACKEND=gdr`) | **memcpy** (no `gdrdrv` module on nscc) |
| Scheduler | PBS, project 12004493, nodes vary per job (job IDs below) | PBS, project personal-wpang013 |
| Checkouts | AE runs: `~/scratch/morphx/MorphX`; follow-ups: `~/scratch/morphx/trackFI` (branch `h100-port-fi-sm90` + local patches) | `~/scratch/morphx-a100/MorphX` (branch `h100-port-a100-nscc`) |

### Software (same versions on both machines)

| Component | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.9.1+cu130 |
| SGLang | built from source, `0.0.0+sglang.main.20260429` |
| sglang-kernel | 0.4.1.post1+cu130 |
| FlashInfer | 0.6.8.post1 (JIT kernels via ninja) |
| transformers / triton | 5.6.0 / 3.5.1 |
| clang (MorphX pass) | LLVM 23.1.4, built per machine |
| Model | Qwen3-8B (36 layers, hidden 4096, FFN 12288, 32 heads / 8 KV heads), aliased as `llama-3-8b`; fp16 unless stated |

### Terms used in the configuration tables

- **Variants (Fig 13/14):**
  - **baseline:** single stream, no prefill thread (continuous batching).
  - **stream:** a prefill thread on its own CUDA stream (`VLLM_PREFILL_THREAD=1`).
  - **MorphX:** stream, plus the MorphX runtime preloaded (`LD_PRELOAD` hook).
  - **chunked:** chunked prefill in mixed batches.
- **Serving runner:** in-process SGLang `ModelRunner` (`scripts/sglang/runner.py`, driven by
  `scripts/test-pd-online.py`):
  - up to 128 sequences, context = input + output length, 90% of GPU memory;
  - eager launches (no CUDA graphs), warm-up before the timed run.
- **PD workload:** Poisson arrivals and random-token prompts. Every request has the stated
  input length and generates exactly the stated output length.
- **PD metrics:**
  - avg / P99 request latency divided by output tokens (the paper's metric);
  - TTFT;
  - served fraction (requests finished within the run ÷ generated).
- **GEMMs:** "mycutlass", CUTLASS GEMMs built through the MorphX pass, used by *every*
  variant.
  - **Sm80:** CUTLASS 2 kernels, 28 tile configs.
  - **sm90:** CUTLASS 3 WGMMA/TMA kernels, non-persistent, 1×1 cluster.
- **Attention:**
  - **stock FlashInfer:** SGLang's default; FA3 prefill on Hopper.
  - **pass FA2:** FlashInfer pinned to FA2 and built through the MorphX pass, so MorphX can
    manage it (`MORPHX_FLASHINFER_PASS=1`).
- **Split hint:** SMs given to compute-intensive (CI) kernels. The paper's A100 hints are 76
  or 88; scaled to 132 SMs they become 93 or 108.
- **CI score:** default threshold 0.9, or `occ` (an occupancy-based score).
- **Per-grid profiling** (`MORPHX_PROFILE_PER_GRID=1`): profiles per kernel and grid size,
  not per kernel name. Without it, decode attention is classified on a small warm-up launch
  and bypassed forever.

## Fig 10–12: LLM co-located with a background task (H200) — before the runner fix

Measured before the runner fix. Stream, MorphX, Orion, Orion-RT and LithOS run the LLM and the background task as two threads in one process; MPS, Timeslice and TGS use separate processes.

**Stall check done (job 196057, 2026-10-28).** nsys traces of the *stream* scheduler at
llm-4-ggnn-8 and llm-4-mm-12, 30 s each, from the AE checkout with the pre-fix runner as used for
these figures. Analysis: `scripts/analysis/nsys_thread_stalls.py`.
- **GGNN:** no cross-thread stall. The only slow calls are one-time module loads at warm-up.
- **Matmul:** the LLM thread's slow calls are its own token copies (`cudaMemcpyAsync`, 193
  calls, 7.0 s). They wait for the GPU while the matmul worker keeps it busy, which is
  ordinary contention. Quick calls are not stuck behind the other thread (13 slow launches,
  0.36 s in total).
- **Why no stall:** Fig 10–12 run the LLM without the prefill thread, so the per-step event
  creation that stalled Fig 13/14 never happens.
- **The published Fig 10–12 results stand.**
- Data: `nscc2b:~/scratch/morphx/MorphX/logs/20261028-fig1012-stall/trace-196057/`.

| Setting | Value |
|---|---|
| Testbed | H200, GDRCopy, AE checkout |
| LLM | Qwen3-8B fp16, Sm80 mycutlass, stock FlashInfer; ShareGPT prompts, Poisson, 1 or 4 req/s |
| Background | GGNN (1M base vectors, 10k queries per request), rates 2× the paper's; or 8192² matmul, rates 3× the paper's |
| Schedulers | MPS-30, MPS-50, MPS-60, Orion, Orion-RT, MorphX, Stream, TGS, Timeslice, LithOS |
| MorphX | split hint 93, default CI score. Fig 10/12: default whitelist (`ggnn5query`). Fig 11: `SMSCHED_TARGET_PATTERNS=half_t SMSCHED_ONLY_WHITELIST=1 MORPHX_TARGET_MM_STREAM=1` (only the matmul worker's stream managed) |
| Duration | 180 s per point (MPS-60: 60 s) |
| Jobs | 186837/186838 (GGNN), 186839/186840 (matmul), 187805/187806 (Fig 11 MorphX column), 192549/192550/192551/192555 (MPS-60) |
| Data | `logs/20261027-mps60/{tables,figures,figures-nolithos}` |
| Plot notes | autoscaled x axis; Fig 10 without LithOS (its ~900 ms latencies squash the axis) |

**Fig 10:** ANNS throughput vs LLM P99 per-token latency.
- Upper left is better.
- MorphX reaches high background throughput at low latency.
- MPS-60 has the lowest latency but caps throughput at ~1.3 ×10⁵ op/s.

![Fig 10](figures/Figure10.png)

**Fig 11:** matmul throughput vs LLM P99.

![Fig 11](figures/Figure11.png)

**Fig 12:** LLM P99 against ANNS request rate.
- Stream and Timeslice blow up.
- MorphX and the MPS variants stay flat.

![Fig 12](figures/Figure12.png)

## Fig 13/14: prefill/decode co-location, AE runs (H200) — before the runner fix (superseded)

Fig 13 and Fig 14 differ only in their request-rate lists. Both run Qwen3-8B here; the paper
used Qwen2.5-7B for Fig 13 and Llama-3-8B for Fig 14.

| Setting | 3× rates | Paper's A100 rates |
|---|---|---|
| Testbed | H200, GDRCopy, AE checkout | same |
| Kernels | Sm80 mycutlass fp16, stock FlashInfer | same |
| Variants | baseline, stream, MorphX, chunked | same |
| MorphX | split hint 93, default CI score, per-name profiling | same |
| Rates (req/s) | Fig 13: 3–15 (2048/256), 3–9 (4096/*); Fig 14: 3–9, 1.2–6 | Fig 13: 1–5 (2048/256), 1–3 (4096/*) |
| Duration | 180 s | 60 s |
| Jobs | 186841 (Fig 13), 186842 (Fig 14) | 190734, 190735 |
| Data | `logs/20261026-nscc2b/partial/` | `logs/20261027-pd-paper-rates/` |

At 3× rates, MorphX is behind Stream at every point:

![Fig 13, 3x rates](figures/Figure13-x3-rates.png)

![Fig 14, 3x rates](figures/Figure14-x3-rates.png)

At the paper's rates the H200 barely queues. Every variant is within a few ms; MorphX is
5–15% behind.

![Fig 13, paper rates](figures/Figure13-paper-rates.png)

![Fig 14, paper rates](figures/Figure14-paper-rates.png)

## Fig 13/14 with the default setup (sm90 GEMMs + pass-built FA3) — current results

| Setting | Value |
|---|---|
| Testbed | H200 (nscc2b), GDRCopy, checkout `~/scratch/morphx/trackFA3` (branch `fa3-pass`, 73280a9: fixed runner, FA3 through the pass) |
| Kernels | sm90 mycutlass fp16 (CUTLASS 3, 12 pass-built kernels) + pass-built FA3 ragged prefill (non-persistent, one block per tile); paged prefill and decode stay FA2, as in stock SGLang |
| Series | baseline; stream; stream with decode at priority −5 (`MORPHX_DECODE_STREAM_PRIORITY=-5`); MorphX with the default bypass (50 µs, first wave); MorphX with the tuned bypass (35 µs, whole-kernel estimate: `MORPHX_SHORT_BYPASS_TOTAL=1 MORPHX_SHORT_KERNEL_US=35`) |
| MorphX | per-grid profiling, default per-row split (93 / 108), default CI score |
| Rates | 3× the paper's; 60 s per point |
| Jobs | baseline / stream / MorphX 50: 196338, 196340, 196342, 196344, 196346, 196348. High-priority stream: 196339–196349 (odd). MorphX 35: 196396–196401. |
| Data | `logs/20261028-fig1314-fa3/{default50,stream-hipri,tuned35}/` |

**How the 35 µs threshold was chosen** (job 196378; `scripts/analysis/kernel_time_census.py` on nsys traces of stream):
- Pass-built (manageable) kernels are about 90% of GPU time.
- Whole-kernel duration is the right measure. The old first-wave check rated the FA3 prefill at about 16 µs, so it was always bypassed.
- 80% of GPU time is in pass-built launches of at least 36 µs (in-2048-out-256, 9 req/s) or 49 µs (in-4096-out-128, 6 req/s).

**Check at 35 µs** (job 196395):
- MorphX's estimated managed share: 87% (4096/128) and 73% (2048/256), against 79% and 69% at 50 µs whole-kernel. With the old first-wave check it was about 6%.
- Tokens are identical (384/384).
- Step cost against no runtime: decode 8.06 → 9.07 ms (+12.5%), prefill 220.5 → 232.8 ms (+6%), and 256.8 ms with split 108 (+16%).

**What the figures show:**
- **MorphX at 35 µs (about 80% managed) is behind stream at 28 of 30 points**, by 4–40% (one tie; its only win is Fig 13 in-2048-out-256 at 15 req/s, +14%).
- **MorphX at 50 µs (first wave) tracks stream within a few %.** It manages almost nothing.
- **Stream with high-priority decode is the best series at almost every loaded point:** up to 47% faster than stream (Fig 13 in-4096-out-256 at 7.5 req/s), and it serves more requests (0.70 vs 0.59).
- **Why:** with Hopper-native kernels, prefill kernels are short (the largest prefill GEMM is about 1.5 ms vs about 2 ms+ with Sm80; most are under 0.5 ms). Decode's wait behind prefill is therefore small enough for simple stream priority to remove it. MorphX's per-launch cost on many short managed kernels outweighs what its SM split adds.
- **This is the opposite of the Sm80 + FA2 result after the fix,** where MorphX won by up to 49% at 4096-token inputs. The difference comes from the kernels, not the GPU.

![Fig 13, default setup](figures/figure13_fa3.png)

![Fig 14, default setup](figures/figure14_fa3.png)

### Reference: stock SGLang's own FA3 attention (2026-10-29)

Without a backend setting, SGLang on Hopper uses its `fa3` attention backend: FlashAttention-3 from the prebuilt `sglang-kernel` wheel, for prefill **and** decode. MorphX can't manage those kernels, since they never pass through the pass. This run asks whether FA3 decode would have made the stream and baseline references faster.

| Setting | Value |
|---|---|
| Change | `MORPHX_SGLANG_ATTENTION=fa3` (new in `scripts/sglang/runner.py`): SGLang `attention_backend="fa3"` (sglang-kernel 0.4.1.post1). Everything else as above: sm90 mycutlass GEMMs, fixed runner, no CUDA graphs |
| Series | baseline, stream |
| Jobs | 200788 (F13 2048/256), 200832 (F13 4096/128), 200770 (F13 4096/256), 200774 (F14 2048/256), 200833 (F14 4096/128), 200834 (F14 4096/256). Partial: 200642 (5 runs, killed by a late `qdel`) |
| Data | `logs/20261029-fig1314-sglfa3/` |

**Result: SGLang's FA3 is no faster than the FlashInfer default.**

TBT relative to the FlashInfer default (pass-built FA3 ragged prefill + FA2 paged prefill and decode), at the 26 points not overloaded (TTFT < 1 s):

| | Median | Range |
|---|---|---|
| Baseline | +1.6% | −12% to +12% |
| Stream | +4.3% | 0% to +12% |

A positive value means SGLang's FA3 is slower.
- Stream's gain over the baseline and its TTFT are unchanged.
- The ranking is unchanged: stream with high-priority decode is still best by far.
- The comparison leaves out Fig 13 2048/256 (job 200788, gpu036). That node ran about 25% slower at light load than gpu046 on the same points: TBT 11.9 vs 9.3 ms at 3 req/s, with the same TTFT. So it reflects the node, not the kernels.
- On gpu046 (partial job 200642) the same points are within 3–4% of FlashInfer.

Implication: making MorphX manage SGLang's FA3 (rebuilding `sglang-kernel` through the pass) would not change the Fig 13/14 picture. The FlashInfer setup is not a handicap to the references.

## Fig 13/14 after the runner stall fix, Sm80 + FA2 (H200) — legacy setup

| Setting | Value |
|---|---|
| Testbed | H200, GDRCopy, trackFI checkout with the runner stall fix (branch `runner-stall-fix`, 4804b97) |
| Kernels | Sm80 mycutlass fp16 + pass FA2 |
| Variants | baseline, stream, MorphX (per-grid profiling, default per-row split 93 / 108, default CI score) |
| Rates | 3× the paper's (Fig 13: 3–15 / 3–9; Fig 14: 3–9 / 1.2–6) |
| Duration | 60 s per point; the three variants of a row ran back to back in one job |
| Jobs | 195358, 195359, 195375 (Fig 13); 195441, 195377, 195378 (Fig 14). Job 195376 hung on gpu109 and was replaced by 195441. |
| Script | `scripts/baselines/n_fig1314_rerun.pbs` |
| Data | `logs/20261028-fig1314-fix/` (per-job runs; merged `tables/`; `figures/` drawn with `plot_figures.py`) |

How to read it:
- With 4096-token inputs MorphX is below Stream and the baseline from moderate load up, as in
  the paper.
- With 2048-token inputs it is still above Stream.
- MorphX's TTFT is much higher. It wins on the paper's metric by speeding decode at the
  expense of prefill.

![Fig 13 after fix](figures/Figure13-after-fix.png)

![Fig 14 after fix](figures/Figure14-after-fix.png)

### `pd_runner_fix`: what the fix changed

| Setting | Value |
|---|---|
| Testbed | H200, GDRCopy |
| Workload | in-4096-out-128, 6 req/s, 30 s per run, stream only, under `nsys profile -t cuda` |
| Kernels | Sm80 mycutlass + stock FlashInfer |
| Runs | stream, and stream with decode at priority −5, each before and after the fix |
| Jobs | before: 194603 (gpu083); after: 195305 (fixed runner, branch `runner-stall-fix`) |
| Data | `logs/20261028-pd-blocking/trace-194603/`; `nscc2b:~/scratch/morphx/trackFI/logs/20261028-pd-blocking/verify-fix/` |

How to read it:
- **Before the fix, host gaps (green) dominate.** Raising decode's priority only turns
  waiting into host gaps.
- **After the fix, host gaps nearly vanish.** Under stream, decode waits behind prefill on the
  GPU (orange). That is the paper's bottleneck.
- **Prioritising decode then cuts TBT from 62.1 to 28.3 ms.**

![runner fix](figures/pd_runner_fix.png)

## Follow-up Fig 13 experiments (H200) — before the runner fix

Drawn by `scripts/analysis/plot_pd_followups.py`; originals in `logs/20261028-figures/`.

### `fig13_sm90_sweep`: rate sweep with Hopper-native GEMMs

| Setting | Value |
|---|---|
| Testbed | H200, GDRCopy, trackFI checkout, MorphX runtime `runtime/build-llm` |
| Kernels | **sm90 mycutlass** fp16 (`microbench/build-model-sm90`) + **pass FA2** |
| Variants | stream; MorphX with per-grid profiling, split 108, default CI score (`SMSCHED_LLM_TRICK_MODE=0`) |
| Rates (req/s) | 4096/128: 5, 6, 7.5; 4096/256: 5, 6, 7.5; 2048/256: 6, 9, 12, 15, 18. The sweep stops once both variants saturate. |
| Duration | 60 s per point; stream and MorphX back to back on the same node |
| Jobs (node) | 192442 (gpu092), 192556 (gpu182), 192558 (gpu092), 192658 (gpu096), 192659 (gpu124), 192661 (gpu092), 192663 (gpu080). Excluded: 192557 (gpu131, erratic). |
| Scripts | `scripts/baselines/n_fig13_sweep_sm90.pbs`, `n_fig13_fi_sm90.pbs` (branch `h100-port-fi-sm90`) |
| Data | `logs/20261027-fi-pass/sweep-*`, `fig13-sm90-192442` |

How to read it:
- Each dot is one run.
- Hollow dots served under 50% of requests (overloaded or bistable) and are kept off the
  lines.
- MorphX is slightly above Stream at every stable rate.

![sm90 sweep](figures/fig13_sm90_sweep.png)

### `fig13_setups`: MorphX / Stream at the standard points, per kernel setup

Points: in-4096-out-128 at 6 req/s, in-4096-out-256 at 6 req/s, in-2048-out-256 at 9 req/s.
Split 108, default CI score, 60 s. Each setup's stream and MorphX ran in the same job.

| Setup (bar) | GEMMs | dtype | Attention | Per-grid | Jobs (node) | Data |
|---|---|---|---|---|---|---|
| Sm80 fp16 stock FI | Sm80 | fp16 | stock FlashInfer | no | 190793, 190846, 190847 | `logs/20261026-nscc2b/diag-fig13-*` |
| Sm80 bf16 stock FI | Sm80 | bf16 | stock FlashInfer | no | 192360 (gpu088) | `logs/20261027-bf16/fig13-bfloat16-192360` |
| Sm80 fp16 pass FA2 | Sm80 | fp16 | pass FA2 | no | 192386 (gpu092) | `logs/20261027-fi-pass/fig13-192386` |
| … + grid | Sm80 | fp16 | pass FA2 | yes | 192386 (gpu092) | same |
| sm90 fp16 pass FA2 | sm90 | fp16 | pass FA2 | no | 192442 (gpu092) | `logs/20261027-fi-pass/fig13-sm90-192442` |
| … + grid | sm90 | fp16 | pass FA2 | yes | 192442 (gpu092) | same |

How to read it:
- Above 1.0 means MorphX is slower.
- All 18 bars are above 1.0.
- The gap shrinks with sm90 because MorphX manages almost nothing there.

![setups](figures/fig13_setups.png)

### `pd_decode_priority`: decode waiting vs CPU-bound decode

| Setting | Value |
|---|---|
| Testbed | H200 (gpu083), GDRCopy, trackFI checkout |
| Workload | in-4096-out-128, 6 req/s, 30 s per run, every run under `nsys profile -t cuda` |
| Kernels | Sm80 runs: Sm80 mycutlass + stock FlashInfer. sm90 runs: sm90 mycutlass + pass FA2. |
| Runs | stream; stream with decode at priority −5 (`MORPHX_DECODE_STREAM_PRIORITY=-5`); MorphX no split (`MORPHX_CI_THRESHOLD=2`, Sm80 only); MorphX split 108 (sm90 with per-grid) |
| Job | 194603 (`scripts/baselines/n_pd_blocking_trace.pbs`, branch `pd-blocking-analysis`) |
| Analysis | `scripts/analysis/nsys_pd_blocking.py` on the nsys SQLite exports. nsys records no kernels under the MorphX hook, so the MorphX bars use the run logs only. |
| Data | `logs/20261028-pd-blocking/trace-194603/` |

How to read it:
- **Left:** mean decode step split into decode kernels, waiting behind prefill, and host gaps
  (stream empty, waiting on the CPU). High priority removes the waiting, and host gaps grow by
  about as much.
- **Right:** time between tokens relative to stream.

![decode priority](figures/pd_decode_priority.png)

Earlier blocking traces, not plotted:
- in-4096-out-128 at 18 req/s, job 192350;
- in-4096-out-128 at 6 req/s, job 194398.

Both are in `logs/20261028-pd-blocking/`.

## A100 control (the paper's GPU) — before the runner fix

| Setting | Value |
|---|---|
| Testbed | A100-40GB, **memcpy** sync backend, CUDA 13 via the compat driver |
| Kernels | Sm80 mycutlass fp16 (built for sm_80) + pass FA2 (sm_80). `MORPHX_MYCUTLASS_DROP_W=1` keeps only the transposed weights, so the KV cache fits in 40 GB. |
| Variants | stream, MorphX, single-stream baseline (26036337); repeat of stream and MorphX, plus MorphX with the `occ` CI score (26036911) |
| MorphX | paper's A100 split hints: 76 (2048-token inputs) and 88 (4096); `MORPHX_NUM_SMS=108`; per-grid profiling |
| Rates (req/s) | the paper's: 2048/256: 3, 4, 5; 4096/128 and 4096/256: 2, 2.5, 3 (`MORPHX_PD_RPS_SCALE=1`) |
| Duration | 60 s per point (the paper: 180 s) |
| Paper numbers | from the artifact's `logs/reference` (Llama-3-8B on their vLLM fork, 180 s) |
| Scripts | `scripts/ae/nscc-a100/fig13_a100.pbs`, `summarize.py`; env `scripts/ae/env-nscc-a100.sh` |
| Data | `logs/20261028-a100-nscc/`; summary tables `summary-sweep-26036337.md`, `summary-sweep2-26036911.md` |

### `a100_fig13`: latency against request rate

- Both MorphX variants are above stream at every rate.
- Stream and single stream are close (−7% to +6%). At this lighter load the prefill thread
  buys little. The paper's Stream was 0–23% ahead of its baseline.

![A100 Fig 13](figures/a100_fig13.png)

### `a100_vs_paper`: MorphX / stream, ours vs the paper

- Ours is 1.18–1.60 on average latency; the paper's is 0.78–0.97.
- At the same rates our runs sit at lighter load: stream TTFT 0.2–5.6 s vs the paper's
  0.7–29 s.

![A100 vs paper](figures/a100_vs_paper.png)

### `a100_step_bench`: runtime cost with nothing co-running

| Setting | Value |
|---|---|
| Job | 26036993 (`PARTS=bench`) |
| Workload | one stream, no co-location. Prefill step = 4 × 2048-token prompts; decode step = batch 16, context 2048+. Median of 8 prefill and 63 decode steps. |
| Runs | no runtime; MorphX runtime, no split (`MORPHX_CI_THRESHOLD=2`); MorphX runtime, split 76 |

| | Decode step | Prefill step |
|---|---|---|
| No runtime | 21.4 ms | 540 ms |
| Runtime, no split | 27.2 ms (+27%) | 568 ms (+5%) |
| Runtime, split 76 | 28.9 ms (+35%) | 717 ms (+33%) |

![A100 step bench](figures/a100_step_bench.png)
