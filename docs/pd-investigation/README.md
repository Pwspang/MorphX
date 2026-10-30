# MorphX port: what was tried, and why MorphX loses to Stream in Fig 13/14

Status: 2026-10-28. The figures and the runs behind each one are described in
[FIGURES.md](FIGURES.md). The detailed argument, with every number and job ID, is in
[../HOPPER_PD_ANALYSIS.md](../HOPPER_PD_ANALYSIS.md). Day-to-day notes are in
[../HANDOFF.md](../HANDOFF.md).

## Summary

- **Fig 10–12 reproduce the paper's shape.** These are the LLM co-located with a background
  task: ANNS (GGNN) or matmul.
  - MorphX keeps LLM latency low while the background task keeps most of its throughput. With
    GGNN at saturation: 14.2 vs Stream's 53.3 ms/token, keeping 65% of Stream's throughput.
  - MPS(6:4) gives the lowest LLM P99 but the lowest background throughput. No method
    dominates.
- **Fig 13/14 don't reproduce.** These are prefill/decode co-location (PD) in one LLM.
  - MorphX is behind Stream at every stable point we ran: on the H200 with every kernel setup
    we tried, and on the A100, the paper's GPU.
  - The paper reports MorphX up to 52% ahead of Stream.
- **Leading explanation: the port's serving stack, not the GPU.**
  - MorphX adds a fixed cost to every managed launch. On the A100 with nothing co-running, the
    runtime makes a decode step 27% slower.
  - Our SGLang runner stalls decode during every prefill.
    - The decode thread blocks in `cudaEventCreate` (~75 ms each time) while the prefill
      thread sits in a blocking device-to-host copy (`tolist()`) that waits for the whole
      prefill.
    - This is a runner bug, found 2026-10-28 (see *Correction* in
      ../HOPPER_PD_ANALYSIS.md). It was first misread as GIL/CPU load.
    - Removing the GPU-side wait doesn't speed decode up.
    - It caps how much prefill/decode GPU overlap Stream and MorphX can get. Stream still beats
      single-stream by about the paper's margin (H200 median ~5%, paper ~8%), likely from CPU-side
      overlap.
  - So there is little for MorphX to win back, and its cost is pure loss.

## Setup common to all runs

- **Model:** Qwen3-8B stands in for Llama-3-8B (gated) and for Qwen2.5-7B (the paper's Fig 13
  model).
- **Serving:** our in-process SGLang runner, with a prefill thread on its own CUDA stream for
  Stream and MorphX.
- **Linear layers:** "mycutlass", CUTLASS GEMMs built through the MorphX pass, for every variant.
  The paper likewise swapped cuBLAS for CUTLASS.
- **H200 (nscc2b, 132 SMs, CUDA 13, GDRCopy sync backend):** request rates scaled up (GGNN ×2,
  matmul ×3, PD ×3), because at the paper's rates the H200 barely queues.
- **A100-40GB (nscc, ASPIRE2A):** the paper's rates, memcpy sync backend (no GDRCopy there).

## What was tried, in order

| # | Experiment | Where / jobs | Result |
|---|---|---|---|
| 1 | Full AE re-run on the H200: Fig 10–16, Sm80 fp16 GEMMs, stock FlashInfer | nscc2b, 186837–186863 | Fig 10–12 match the paper's shape. Fig 13/14: MorphX 6–17% behind Stream at low load. |
| 2 | Fig 11: only the matmul worker's stream managed (`MORPHX_TARGET_MM_STREAM=1`) | 187805/187806 | Fixed a whitelist issue. MorphX 15.4 vs Stream 63.0 ms/token at saturation (was 68.7). |
| 3 | GDRCopy vs memcpy sync backend | 187714 | GDRCopy is cheaper: decode +6.5% vs +29% over no runtime. |
| 4 | Fig 13/14 at the paper's A100 rates on the H200 | 190734/190735 | No queueing. MorphX 5–15% behind on all 30 points, pure overhead. |
| 5 | Forcing a split: `occ` CI score, split 108 / 66 | 190335/190336 | Never beats Stream. |
| 6 | Split sweep at 4096/128 6 req/s: 66 → 120 SMs, and no split | 190793, 190815 | Wider is better; no split is best (0.094 vs Stream 0.067 s/token, +40%). |
| 7 | MPS(6:4) added to Fig 10–12 | 192549/50/51/55 | Lowest LLM P99, lowest background throughput. |
| 8 | bf16 mycutlass | 192360 | No change (+19% to +116%). |
| 9 | Pass-built FlashInfer FA2, so attention is MorphX-controllable; per-grid profiling, so decode attention is managed | 192232–192386 | 85% of GPU time managed, as in the paper. Per-grid helps 0–8%, but MorphX stays 10–65% behind. |
| 10 | Hopper-native sm90 CUTLASS 3 GEMMs (WGMMA/TMA) through the pass | Track S, 192442 | 2.7–3.4× faster overall. MorphX only 9–18% behind, but because it mostly steps aside: most sm90 GEMMs fall under the 50 µs short-kernel cutoff. |
| 11 | Rate sweep to saturation with the sm90 GEMMs | 192556–192663 | MorphX behind at every stable rate. Its one "win" (4096/256, 7.5 req/s) is in the collapsed regime, where both served under 30%. |
| 12 | nsys: how long decode waits behind prefill under Stream | 192350, 194398 | The wait is real: 24–37% of a decode step at 6 req/s, 61% at 18 req/s. Equal-priority streams make decode wait for a whole prefill kernel, not one wave. |
| 13 | Decode on a high-priority stream; MorphX TBT and counters on one node | 194603 | Priority removes the GPU wait, but TBT doesn't improve: the decode thread then stalls in `cudaEventCreate` behind the prefill thread's blocking copy (runner bug). With sm90, MorphX only confines prefill GEMMs and never touches decode. |
| 14 | **A100 control**, the paper's GPU | nscc 26036337, 26036911, 26036993 | MorphX behind at all 9 points (18–60% avg); the paper has it 3–22% ahead. The runtime alone costs decode +27% and prefill +5%; split 76 costs prefill +33%. |
| 15 | **Runner stall fix**: reuse events, no blocking syncs between the prefill and decode threads | 195305 (check), 195358–195363 (Fig 13/14 rerun) | Stall gone; high-priority decode now cuts TBT 55%. **Rerun: with 4096-token inputs MorphX beats Stream at 14/20 points (up to 49%) and the baseline at 16/20 (up to 59%), growing with load, as in the paper. With 2048-token inputs it's 2–32% behind Stream at 8/10 points.** MorphX wins by trading TTFT (3.6–11.5 s) for faster decode. |
| 16 | **FA3 through the MorphX pass**, made the default with sm90 GEMMs | hpcc (agent), nscc2b 196219 | Pass-built FA3 is bitwise identical to stock FA3, with and without MorphX. Needed a non-persistent FlashInfer planner and a barrier fix for re-entry. |
| 17 | **Short-kernel threshold tuned** to manage ~80% of GPU time | 196378 (census), 196395 (check) | Whole-kernel judging at 35 µs: 73–87% managed (vs ~6% with the old first-wave rule); decode step +12.5%. |
| 18 | **Fig 13/14 with the default setup** (sm90 + FA3): baseline, stream, stream + high-priority decode, MorphX at 50 µs and 35 µs | 196338–196349, 196396–196401 | **MorphX at 35 µs is behind stream at 28/30 points (4–40%). Stream with high-priority decode is best: up to 47% faster than stream.** With Hopper-native kernels, simple decode priority beats MorphX. |
| 19 | **Stock SGLang FA3 attention as reference** (sgl-kernel FA3 for prefill and decode; baseline and stream only) | 200770, 200774, 200788, 200832–200834 | **No faster than the FlashInfer default:** TBT median +1.6% (baseline) and +4.3% (stream), i.e. slightly slower. FlashInfer isn't handicapping the references, so moving MorphX to SGLang's FA3 isn't worth the rebuild. |

## How the explanation evolved

1. **First hypothesis: Hopper-specific.**
   - Hopper's kernels are short, and its best kernels (FA3, CUTLASS 3) are already
     self-scheduling, so MorphX can't control them.
   - Partly true for the sm90 runs (#10, #13), but it doesn't explain the Sm80 runs (#9),
     where MorphX controlled as much as in the paper and still lost.
2. **Second: little head-of-line blocking left.** Wrong. The nsys analysis (#12) found a lot
   of decode waiting behind prefill.
3. **Third: the waiting is real but can't be recovered.** The decode step looked CPU-bound (#13), so
   GPU-side scheduling has nothing to convert.
4. **A100 control (#14): MorphX loses there too.** The cause is the port and its serving stack,
   not the GPU.

## Open questions and next steps

**Runner stall: fixed 2026-10-28** (branch `runner-stall-fix`, commit 4804b97). Details and
verification are in ../HOPPER_PD_ANALYSIS.md, under *Correction and fix*.
- The runner now reuses its `_kv_section` events and replaces blocking `tolist()` /
  `synchronize()` calls with a pinned copy plus event polling.
- Verified by job 195305: the decode-thread stall is gone. Decode on a high-priority stream
  now cuts TBT 55% (62.1 → 28.3 ms), so the paper's GPU-level bottleneck shows on the H200.
- Every Fig 13/14 result above was measured before this fix.
- H200 rerun done (jobs 195358, 195359, 195375, 195377, 195378, 195441); see the table row 15 and FIGURES.md. No further A100 runs are planned
  (decision 2026-10-28); the A100 control stays as a pre-fix record.

**Decision (2026-10-28, revised): the default setup becomes sm90 GEMMs + pass-built FA3.**
- This replaces the earlier same-day "Sm80 only" decision. Sm80 + FA2 stays available as a
  legacy option.
- Done: FA3 through the pass was built on branch `fa3-pass` and the defaults flipped (cf1af6d).
  `fa3-pass` was merged into `h100-port-s-sm90` on 2026-10-30, which is now the branch for
  native (sm90 + FA3) MorphX.
- Attention follows stock SGLang's FlashInfer choices: FA3 for ragged prefill, FA2 for paged
  prefill and decode.
- Short-kernel threshold: under the paper's 50 µs first-wave rule most sm90 kernels are
  bypassed (~6% of GPU time managed). Since 2026-10-30 the LLM runtime (`cuda-llm.cpp`)
  defaults on sm90+ to 35 µs judged on whole-kernel time, which manages 73–87%. In Fig 13/14
  this made MorphX slower than the 50 µs rule (row 18). `MORPHX_SHORT_BYPASS_TOTAL=0
  MORPHX_SHORT_KERNEL_US=50` restores the paper's rule.

- **Overload test (Sm80, H200 and A100):** run stream vs MorphX at loads matching the paper's (stream TTFT in seconds, 50–60% of requests served), 3 repeats × 180 s, ~1.5–2 GPU-hours per GPU. The paper's wins are all in that regime; our runs so far were lighter.

- **How much of the A100 loss is the memcpy backend?**
  - Plan: rerun a few H200 Fig 13 points with MorphX on memcpy vs GDRCopy; one short job.
  - GDRCopy can't be installed on nscc (it needs a kernel module).
- **Would a non-CPU-bound decode path change the picture?** For example, a single-thread
  decode loop. CUDA graphs would hide kernels from MorphX, so they need care.
- **Model effect:** Qwen2.5-7B (the paper's Fig 13 model) on the H200; ~20 min of GPU time.
- **Remaining differences from the paper:**
  - their modified vLLM 0.8.1 vs our SGLang runner;
  - 60 s vs 180 s runs;
  - Llama-3-8B bf16 vs Qwen3-8B fp16.

## Where things are

- **Branches** (`h100-port-s-sm90` and `fa3-pass` are pushed; the rest are local):
  - `h100-port`, `h100-port-s-sm90`, `track-s-phase2` (sm90 library);
  - `h100-port-bf16-mycutlass`, `h100-port-flashinfer-pass`, `h100-port-fi-sm90`;
  - `pd-blocking-analysis` (nsys analysis, priority and trace jobs);
  - `h100-port-a100-nscc` (A100 setup).
- **Results:** `logs/` in this checkout (git-ignored); see [FIGURES.md](FIGURES.md) for the
  folders.
- **Plots:** `scripts/analysis/plot_pd_followups.py` redraws the six follow-up figures.
  `plot_figures.py` draws the AE figures.
