# Why Stream beats MorphX for prefill/decode co-location on Hopper

Status: 2026-10-28. Measured on the H200 (nscc2b, 132 SMs, CUDA 13) and, as a control, on
the A100-40GB (nscc). The A100 control shows MorphX also loses there (next section), so the
title's "on Hopper" no longer holds as a cause. Read the reasons below as H200 measurements. Detailed results and job notes are in `docs/HANDOFF.md`. The
sm90 and FlashInfer sections are on branch `h100-port-fi-sm90`.

## Correction and fix (2026-10-28): the decode "host gaps" were a runner stall, not CPU load

Earlier sections call the decode step "CPU-bound (GIL, ~470 launches per step)". The traces
show a more specific cause.

During each prefill, the decode thread is stuck inside `cudaEventCreateWithFlags` for about
75 ms on average. In every case (144/144, 142/142, 149/149 across three traces), the prefill
thread is meanwhile inside `cudaMemcpyAsync`: the blocking device-to-pageable-host copy behind
`next_token_ids.tolist()`, which waits for the whole prefill forward. The decode thread can't
create its event until that copy returns.

Where the event comes from: `_kv_section()` in `scripts/sglang/runner.py` creates a new
`torch.cuda.Event()` and records it every section, so the CUDA event is created lazily on
every decode step.

Decode-thread time in CUDA calls longer than 20 ms, in-4096-out-128 at 6 req/s, ~30 s window:

| Trace | stuck in `cudaEventCreate` | waiting in its own `tolist()` copy |
|---|---|---|
| Stream, job 194603 | 11.0 s | 7.9 s |
| Stream with decode high priority, job 194603 | **18.0 s** | 0.1 s |
| Stream, job 194398 | 11.2 s | 8.0 s |

What this explains:
- **High priority.** It removed decode's GPU wait, and its copy wait with it. The thread then
  hit the event-creation stall for longer, so "host gaps" grew by what the blocking lost.
- **Stream overlaps less than it should.** Decode is paused for most of each prefill's GPU
  time, which caps what Stream (and MorphX) can gain from overlapping prefill and decode on the
  GPU. Stream still beats single-stream by a similar margin to the paper (see below), probably
  from overlapping CPU work.

This is a bug in our runner, not a property of the GPU or of the paper's setup. It applies on
the A100 too, and it likely masks both Stream's and MorphX's benefit.

**Fixed 2026-10-28** (branch `runner-stall-fix`, commit 4804b97, `scripts/sglang/runner.py`):
- `_kv_section()` reuses one CUDA event per stream. A later `wait_event` sees the latest
  record, so reuse only orders more work, never less.
- `_to_list()` replaces `next_token_ids.tolist()` in prefill, decode and chunked steps: a
  non-blocking copy into a reused pinned buffer, then `_wait_stream()`.
- `_wait_stream()` replaces the blocking `stream.synchronize()` after merging prefills. It
  polls a reused event with 50 µs sleeps, so no thread sits in a blocking driver call and the
  GIL is released while waiting.
- Token values are unchanged; only how the CPU waits changed.

Patched in place on nscc2b `~/scratch/morphx/trackFI` (original kept as
`runner.py.orig-a5cd00a`). Not applied to the A100 checkout: no further A100 runs are planned (decision 2026-10-28). Analysis script:
`scripts/analysis/nsys_host_stalls.py`.

Verification, job 195305: same config as 194603 (in-4096-out-128, 6 req/s, Sm80 + stock
FlashInfer, 30 s, under nsys). Waiting and host gaps are ms per decode step.

| | before: stream | before: high priority | after: stream | after: high priority |
|---|---|---|---|---|
| TBT, ms | 70.2 | 73.2 | 62.1 | **28.3** |
| TTFT, ms | 163 | 176 | 547 | 1,772 |
| served | 95/159 | 95/159 | 103/159 | 124/159 |
| decode waiting behind prefill | 22.0 | 3.1 | 39.1 | 7.1 |
| decode host gaps | 31.3 | 52.7 | 5.2 | 1.6 |
| decode thread in `cudaEventCreate` (>20 ms calls) | 11.0 s | 18.0 s | 0 | 0 |

- **The stall is gone.**
- **With it gone, the paper's bottleneck shows:** under Stream, decode now waits 39 of 63 ms
  per step behind prefill kernels.
- **Prioritising decode now pays:** TBT −55%, at the cost of TTFT (prefill is starved). This is
  the trade-off MorphX's SM split is meant to manage.
- **Every Fig 13/14 result before this fix, on both GPUs, was measured with the stall**, which
  hid most of the GPU-level prefill/decode interference. Those results are superseded for the
  Stream vs MorphX question.
- **Rerun on the H200:** jobs 195358–195363 (Fig 13/14; baseline, stream, MorphX; Sm80 + pass
  FA2; per-grid profiling; default per-row split; 60 s), `n_fig1314_rerun.pbs`, output in
  `nscc2b:~/scratch/morphx/trackFI/logs/20261028-fig1314-fix/`.

**H200 rerun after the fix** (jobs 195358, 195359, 195375, 195377, 195378, 195441; one job per
table and length, all three variants in the same job; 60 s per point).
- Setup: Sm80 mycutlass fp16, pass-built FA2, MorphX with per-grid profiling and the default
  per-row split (93 / 108).
- Data: `logs/20261028-fig1314-fix/` (merged tables in `tables/`, AE-style plots in
  `figures/`).

MorphX avg latency per token vs Stream (+ = MorphX faster) / vs baseline:

| Config | Fig 13 rates (req/s) | MorphX vs Stream | MorphX vs baseline |
|---|---|---|---|
| in-2048-out-256 | 3, 6, 9, 12, 15 | −8, −11, −24, +12, −2% | 0, −5, −8, +14, −21% |
| in-4096-out-128 | 3, 4.5, 6, 7.5, 9 | −5, +12, +23, +23, +19% | 0, +24, +59, +34, +24% |
| in-4096-out-256 | 3, 4.5, 6, 7.5, 9 | 0, +3, +36, +30, +30% | +5, +12, +36, +6, +17% |

| Config | Fig 14 rates (req/s) | MorphX vs Stream | MorphX vs baseline |
|---|---|---|---|
| in-2048-out-256 | 3, 4.5, 6, 7.5, 9 | −15, 0, −6, −8, −32% | −7, 0, 0, 0, −7% |
| in-4096-out-128 | 1.2, 2.4, 3.6, 4.8, 6 | −8, 0, +13, +14, +4% | 0, 0, +13, +33, +53% |
| in-4096-out-256 | 1.2, 2.4, 3.6, 4.8, 6 | −14, 0, +9, +19, +49% | −14, +6, +12, +36, +50% |

- **4096-token inputs now match the paper's shape.** MorphX beats Stream at 14 of 20 points
  (by up to 49%) and beats the baseline at 16 of 20, with the gap growing with load: "MorphX
  becomes increasingly favorable and dominates the high-contention region" (paper §5.3).
- **2048-token inputs do not.** MorphX is 2–32% behind Stream at 8 of 10 points (one tie, one +12% win).
- **Low load:** MorphX is 5–15% behind (its per-launch overhead).
- **How MorphX wins: it trades TTFT for decode speed.**
  - TTFT rises to 3.6–11.5 s, vs 0.8–7.6 s for Stream, while per-token decode time falls.
  - Example, 4096/128 at 7.5 req/s, decode ≈ (latency × 128 − TTFT) / 127:
    - MorphX: TTFT 8.8 s, decode ~31 ms/token.
    - Stream: TTFT 4.8 s, decode ~93 ms/token.
  - The paper's metric (request latency per output token) favours that trade.
- **Requests served are similar or higher for MorphX at the 4096/256 winning points**
  (206–216 vs 159–164 of 317–490). Two winning points serve fewer than Stream: Fig 13 4096/128
  at 6 req/s (260 vs 281 of 317) and Fig 14 4096/128 at 6 req/s (240 vs 287 of 317). Treat
  those two with care.
- **Caveat:** single runs. A Stream variant with decode on a high-priority stream (0
  per-launch cost) should be added as a control, since it also trades TTFT for decode speed.

**Default setup, sm90 GEMMs + pass-built FA3** (2026-10-28, jobs 196338–196401; details in
docs/pd-investigation/FIGURES.md):
- MorphX tuned to manage ~80% of GPU time (35 µs whole-kernel bypass) is behind stream at
  28/30 points (4–40%).
- Stream with decode on a high-priority stream is up to 47% faster than stream, with no
  per-launch cost.
- So with Hopper-native kernels, plain stream priority captures the decode-behind-prefill gain,
  and MorphX's per-launch cost on short kernels outweighs its SM split.
- With Sm80 + FA2 (longer prefill kernels) MorphX won by up to 49%. That run had no
  high-priority stream control; it would show how much of that win is just prioritising decode.

**Fig 10–12 are not affected** (stall check, job 196057): they run the LLM without the
prefill thread, and nsys shows no quick CUDA calls stuck behind the background worker. Details
are in docs/pd-investigation/FIGURES.md.

## A100 control (2026-10-28): the gap is not Hopper-specific

Our port loses on the A100 too, so the thesis below can't be stated as a Hopper effect. The
A100 is the paper's GPU: nscc ASPIRE2A, A100-40GB.

- **What ran:** jobs 26036337 and repeat 26036911 (agrees within ~5%), branch
  `h100-port-a100-nscc`.
  - Qwen3-8B fp16 on our SGLang runner, Sm80 mycutlass + pass-built FA2, built for sm_80.
  - Paper request rates (`MORPHX_PD_RPS_SCALE=1`); the A100 split hints, 76 SMs for
    2048-token inputs and 88 for 4096; per-grid profiling; 60 s per point.
  - **memcpy sync backend:** nscc has no GDRCopy.
- **Result:** MorphX loses to Stream at all 9 Fig 13 points, by 18–60% avg and 23–61% p99. The
  paper reports MorphX ahead by 3–22% avg at the same points.
  - The `occ` CI score is no better: 18–56% behind.
  - Results: `logs/20261028-a100-nscc/summary-sweep-26036337.md`, with the paper's reference
    numbers alongside.
- **Overhead alone matches the loss.** Step-time benchmark, job 26036993: one stream, no
  co-location.

  | | prefill step | decode step |
  |---|---|---|
  | no runtime | 540 ms | 21.4 ms |
  | runtime, no split | 568 ms (+5%) | 27.2 ms (+27%) |
  | runtime, split 76 | 717 ms (+33%) | 28.9 ms (+35%) |

  That works out to about 20 µs per managed launch with memcpy.
- **Stream vs single-stream (corrected 2026-10-28).** On the A100 our Stream is within −7% to
  +6% of single-stream. The paper's reference Fig 13 table has Stream 0–23% ahead of its
  baseline (median ~8%); an earlier "13–25%" here was wrong. On the H200 AE runs (job 186841,
  3× rates) ours is −9% to +43% (median ~5%), so the margins are comparable. The A100 gap is
  smaller probably because our A100 runs were lighter-loaded.
  - This matches the H200 priority result (job 194603): our decode step is CPU-bound, with two
    Python threads sharing the GIL, ~470 launches per step and no CUDA graphs.
  - So GPU-side overlap, whether from Stream or MorphX, has little to win in this runner.
  - MorphX's per-launch cost is pure loss.
- **Load differs from the paper.** At the same rates our A100 setup is 2–3× faster than the
  paper's. Stream TTFT is 0.2–5.6 s vs the paper's 0.7–29 s, so our points sit at lighter
  load. Part of this is the 60 s vs 180 s runs.

What this changes:
- The leading explanation is now the port and its serving stack, not the GPU:
  - MorphX's per-launch overhead;
  - a host-bound runner in which prefill/decode overlap doesn't pay.
- Hopper-specific factors (Reasons 1 and 4) can still add to the H200 gap, but they are not
  what makes MorphX lose.

Open items:
- The memcpy backend inflates the A100 overhead. On the H200, gdr measured 9–13 µs per
  launch, which would still leave roughly +15–20% per decode step.
- Qwen3-8B fp16 on SGLang, not Llama-3-8B bf16 on their vLLM fork.
- The paper's own Fig 15 shows similar per-kernel overhead (GEMM 1.44×, GEMV 1.34×), yet
  MorphX wins there. That points at their runner being GPU-bound where ours is CPU-bound.
  Not verified.

## Thesis (H200, written before the A100 control)

For prefill/decode (PD) co-location on the H200, MorphX's cost is at least as large as what
it can win back, and MorphX doesn't convert the room it has. This differs from the paper
(A100, §5.3, Fig 13/14), where MorphX cuts average latency by up to 52% and P99 by up to 58%
versus Stream.

Revised 2026-10-28 after the nsys blocking analysis (Reason 2). Stream does leave room:
- Decode loses 24–37% of each decode step to prefill at 6 req/s, and about 61% at 18 req/s.
- At 6 req/s with the same kernels, the most MorphX could win back (24.6 ms/step) is about
  equal to the overhead it was measured to add (+27 ms/token).
- At 18 req/s there is far more to win, but MorphX still loses by 15–17%.

This is specific to PD co-location. In Fig 10–12 (LLM plus a background task), MorphX
reproduces the paper's shape. At saturation it reaches 14.2 vs Stream's 53.3 ms/token with
GGNN, and 15.4 vs 63.0 with matmul (Fig 11a). There the background task is pure throughput
and can give up SMs. In PD both sides are the same latency-critical LLM.

## The dilemma

| Kernel set | Can MorphX manage it? | Result |
|---|---|---|
| A100-style kernels (Sm80 CUTLASS 2 mycutlass, FA2) | Yes: 78–85% of GPU time is managed (paper: 85%) | MorphX 10–80% behind Stream. The whole system is 2.7–3.4× slower than with sm90 GEMMs. |
| Hopper-native kernels (sm90 CUTLASS 3 mycutlass; FA3 can't be passed) | Barely: 1–3 GEMM kernels pass the 50 µs short-kernel cutoff, and decode attention is managed 0 of 73,332 times at 4096/128 | MorphX 9–18% behind Stream, closer only because it mostly steps aside |

With the A100-style kernels MorphX controlled as much GPU time as in the paper and still lost.
So the loss is not just that the kernels are out of its reach.

## Reasons

### 1. Cost: a fixed overhead on every managed launch

Measured:
- No-split runs (`MORPHX_CI_THRESHOLD=2`, nothing confined, overhead only) at two Fig 13
  points:
  - in-4096-out-128, 6 req/s: 0.094 vs Stream 0.067 s/token (+40%), job 190815.
  - in-2048-out-256, 9 req/s: 0.053 vs 0.041 (+29%), job 190846.
- At the paper's A100 request rates the H200 doesn't queue. MorphX is 5–15% behind
  on all 30 points (jobs 190734/190735).
- A managed launch adds about 9–13 µs end to end. The host side is only ~0.9 µs with GDRCopy
  (job 187714), so the rest is on the GPU: the persistent agent blocks, fetching each block
  through an atomic, the SM-range checks and first-launch profiling.
- A decode step makes ~145 GEMM launches.

Why the H200 makes it worse: the cost per launch stays the same while the kernels get ~3× shorter.

Open question: the per-launch cost doesn't add up to the measured overhead. 145 managed GEMM
launches × 9–13 µs is 1.3–1.9 ms per decode step. The measured cost of the no-split run at
4096/128 6 req/s is +27 ms/token, about 15× more. Most of MorphX's cost therefore comes from
somewhere else. Untested hypothesis: host-side work. Under Stream, decode already spends 48%
(sm90) to 44% (Sm80) of each step in host gaps, with its stream empty and waiting on the CPU.
The hook sits on every one of ~470 launches per step, and the agent thread competes with two
Python threads for the CPU.

Data to add:
- An overhead breakdown on small, medium and large kernels: hook only, transformed kernel
  without the scheduler, then full MorphX.
- Overhead as a share of kernel time on the H200, next to the paper's Fig 15 (A100: GEMM 1.44×,
  GEMV 1.34×).

### 2. Benefit: Stream leaves room, but at sane load no more than MorphX costs

Measured: an nsys analysis of Stream-only Fig 13 runs, in-4096-out-128, with two traces per
setup. Script: `scripts/analysis/nsys_pd_blocking.py`, branch `pd-blocking-analysis`.

How the numbers are built:
- Decode is stream 13 and prefill is stream 29, identified by marker kernels and launching
  thread.
- **Blocked** is the time a decode kernel was ready but waited while a prefill kernel ran.
- **Slowdown** is the extra decode kernel time while overlapping prefill, against the same
  kernel running alone.
- **Ceiling** = blocked + slowdown: the most MorphX could win back.
- All numbers are ms per decode step. Each step's parts add up to its wall time.

| trace | mean step | blocked | slowdown | **ceiling** | host gaps | launches blocked | MorphX vs Stream at this point |
|---|---|---|---|---|---|---|---|
| 18 req/s, Sm80, stock FI (job 192350) | 169 | 86.2 | 16.6 | **102.8 (61%)** | 44.4 | 64% | +15% TBT, same job |
| 18 req/s, Sm80, pass FA2 (192350) | 177 | 94.3 | 17.4 | **111.8 (63%)** | 43.9 | 67% | +17% TBT, same job |
| 6 req/s, Sm80, stock FI (194398) | 66.6 | 20.8 | 3.8 | **24.6 (37%)** | 29.2 | 20% | +40% no-split (190815), +27 ms/token |
| 6 req/s, sm90, pass FA2 (194398) | 32.6 | 7.1 | 0.8 | **7.9 (24%)** | 15.7 | 10% | +9% (192442), about +3 ms/token |

Findings:
- **The "about one wave" mechanism is wrong.** With two equal-priority streams (both
  priority 0), the hardware hands out all remaining blocks of the running prefill kernel
  before any decode block. Blocked decode kernels start at about 96% of the way through a
  1.98 ms Sm80 prefill GEMM. When decode is blocked, it waits nearly a whole prefill kernel.
- **The interference is one-sided.** Prefill loses only 2–8% of its step to decode.
- **At 6 req/s, Sm80, the comparison is like for like:** same kernels, same rate, and the
  Stream TBT matches 190815. The ceiling (24.6 ms/step) is about equal to MorphX's measured
  overhead-only cost (+27 ms/token). Recovering all of it would only break even.
- **With sm90 GEMMs the ceiling shrinks to 7.9 ms/step (24%).**
- **At 18 req/s there is far more to win (61–63%), and MorphX still loses 15–17%.** There
  the loss comes from Reasons 1 and 3 (overhead, and the split slowing decode), not from a
  lack of headroom.

Caveats:
- The ceiling is an upper bound: it assumes decode gets SMs at once, at its alone speed, at
  no cost to prefill.
- 18 req/s is about 3× prefill capacity and bistable.
- Sm80 GEMMs make prefill kernels long, which inflates blocking.
- Host gaps are CPU-side, likely Python GIL contention between the two threads. MorphX
  can't recover them.
- Single runs, one node each.

Data to add:
- **Stream with decode on a high-priority stream.** Block dispatch would then favour decode
  at zero per-launch cost. It could capture much of this ceiling and is the bar MorphX should
  have to beat.
- **nsys traces of MorphX at the same 6 req/s points**, no-split and default split. They
  would show where the +27 ms/token goes: host gaps, kernel slowdown or confinement.

**Follow-up, job 194603** (one node, gpu083). Settings: in-4096-out-128, 6 req/s, 30 s,
every run under nsys. Results: `logs/20261028-pd-blocking/trace-194603/`.

TBT and TTFT come from the run logs, in ms. Blocked and host gaps are ms per decode step.

| run | TBT | TTFT | served | decode blocked | decode host gaps | MorphX launches (managed / CI-limited / moved / full GPU) |
|---|---|---|---|---|---|---|
| sm80 stream | 70.2 | 163 | 95 | 22.0 | 31.3 | |
| sm80 stream, decode high priority | 73.2 | 176 | 95 | 3.1 | 52.7 | |
| sm80 MorphX, no split | 76.0 (+8%) | 176 | 95 | n/a | n/a | 11,246 / 0 / 0 / 11,246 |
| sm80 MorphX, split 108 | 109.6 (+56%) | 197 | 78 | n/a | n/a | 10,580 / 5,687 / 833 / 4,060 |
| sm90 stream | 39.5 | 135 | 138 | 7.2 | 15.3 | |
| sm90 stream, decode high priority | 39.8 | 161 | 140 | 0.6 | 21.7 | |
| sm90 MorphX + per-grid, split 108 | 43.5 (+10%) | 176 | 133 | n/a | n/a | 5,759 / 5,759 / 0 / 0 |

- **High priority removes the blocking, but decode gets no faster.**
  - The decode stream at priority -5 cuts blocked time from 22.0 to 3.1 ms/step (Sm80) and
    from 7.2 to 0.6 (sm90).
  - TBT doesn't improve (+4% / +1%), and TTFT gets worse (+8% / +19%).
  - The time removed reappears almost entirely as host gaps (+21 / +6 ms/step).
  - So in this runner the decode step is CPU-bound. The two Python threads share the GIL,
    there are no CUDA graphs, and there are ~470 launches per step. The GPU waiting behind
    prefill overlaps CPU work that has to happen anyway.
  - The ceiling measured above is therefore not recoverable by any GPU-side scheduler,
    MorphX included. Reason 2 holds, for a different reason than first stated.
  - Caveat: this may be specific to the runner. The paper's vLLM fork may be less
    host-bound. With CUDA graphs, though, MorphX could not intercept the graph's kernels at all.
- **With sm90 GEMMs, MorphX manages only prefill GEMMs, and only to confine them.**
  - All 5,759 managed launches were CI-limited to [0, 108). That is about 1% of the ~510k
    launches in the window.
  - No decode kernel was managed, nothing was moved, and TTFT rose 30%.
  - Its +10% therefore combines hook overhead on every launch with the slower, confined
    prefill. It is not overhead alone. A sm90 no-split run would separate the two.
- **Sm80: in this job the split costs much more than the overhead.** No split is +8%;
  split 108 is +56%. This differs from 190815 (no split +40%, 60 s, main-checkout
  `runtime/build`, no nsys). Both are single runs. nsys adds CPU cost to every variant and
  may compress the overhead-only gap.
- **nsys can't trace MorphX runs.** The hooked driver hides the kernels from CUPTI: one
  kernel is recorded in 30 s, and nsys reports event-order errors. The MorphX rows above use
  the run logs and MorphX's own counters. Finding where MorphX's cost goes needs another
  tool, such as the runtime's per-kernel counters or CUDA events in the runner.
### 3. The split is zero-sum: prefill and decode don't complement each other

Measured:
- Fig 13 split sweep (in-4096-out-128, 6 req/s, job 190793): latency improves steadily as the
  split widens.

  | split (SMs) | 66 | 80 | 93 | 108 | 120 | no split | Stream |
  |---|---|---|---|---|---|---|---|
  | avg s/token | 0.223 | 0.207 | 0.190 | 0.149 | 0.103 | 0.094 | 0.067 |

- No split was the best per-token latency at all three configs tested (jobs 190815,
  190846, 190847).
- Forcing a split with the `occ` score never beat Stream (jobs 190335/190336).
- Decode GEMMs slow down when they lose SMs (hpcc Track J4).

Data to add:
- **Pairwise co-run microbenchmark.** Take a real prefill kernel and a real decode kernel
  (GEMM and paged attention). Time each alone, together under Stream, and under MorphX across
  a sweep of split points. The paper's premise is that some split beats Stream; this tests it
  directly.
- **SM-scaling curves.** Time each key kernel on 8 to 132 SMs, using MorphX's confinement. If
  decode kernels scale nearly linearly with SM count, each SM they give up costs them speed,
  so the split can't help.

### 4. Hopper's best kernels already schedule themselves, and the pass can't handle them

Measured:
- FA3 and CUTLASS 3 kernels launch one block per SM and hand out work themselves (FA3:
  `BatchPrefillPersistentTileScheduler`). They use clusters, TMA, warp specialisation and
  `setmaxnreg`, which clash with the pass's agent transform and with confining kernels to an
  SM range.
- Getting the sm90 GEMMs through the pass needed:
  - non-persistent kernels with single-block (1×1×1) clusters;
  - barriers around the block-ID hand-off in dynamic smem word 0;
  - `__grid_constant__` and `NDEBUG`.
- Pinning attention to FA2 alone costs +33% for Stream at 4096/128: the prefill attention
  kernel takes 657 µs vs 321 µs with FA3 (job 192442).

On Hopper, MorphX's portability (any kernel you can compile) clashes with the vendor's own
work: the fastest kernels are the ones it can't manage.

Data to add: a census of GPU time on the stock Hopper stack (cuBLAS + FA3). Report the share
that is closed-source or can't go through the pass, against the paper's 85%.

## Ruling out other explanations

1. **Run on an A100 (the most important control).** If this port matches the paper there,
   the gap is due to Hopper. If not, it points to the port.
2. **Check the reclaim path.** Count how often the high-priority reclaim copy (paper §4,
   "End-to-end example") fires during a Fig 13 run. If it never does, the port has a gap.
3. **Repeat runs.** Report means and confidence intervals over ≥3 runs per point. Single
   runs have been skewed by bad nodes (gpu131 gave erratic results).
4. **Mark the unstable regime.** Report the served fraction next to latency. Near saturation
   the runner is bistable, so drop or flag those points. Example: MorphX "wins" in-4096-out-256 at
   7.5 req/s by 38%, but at 6 req/s both variants collapse (served fraction 0.03, job 192661).
5. **Compare Stream's strength.** Compute Stream's gain over the single-stream baseline on
   the H200 and compare it with the paper's Stream-vs-vLLM gain. If the paper's Stream was
   relatively weaker, part of its margin came from that.
6. **Name the setup differences.** Qwen3-8B instead of Qwen2.5-7B / Llama-3-8B, and an
   SGLang-based runner instead of a modified vLLM 0.8.1.

## Planned next steps

| # | Experiment | Cost | Supports |
|---|---|---|---|
| 1 | ~~Decode queueing delay from nsys traces of Stream~~ **done 2026-10-28**, see Reason 2 | 1 short job (194398) | Reason 2 |
| 1a | ~~Stream with decode on a high-priority stream~~ **done (194603)**: no gain, decode is CPU-bound | 1 short job | Reason 2 |
| 1b | MorphX at the 6 req/s points: **TBT and counters done (194603)**; nsys traces impossible under the hook | 1 short job | Reasons 1, 2 |
| 1c | sm90 MorphX with no split (overhead only vs confinement) | ~2 min GPU | Reasons 1, 3 |
| 2 | Pairwise co-run and SM-scaling microbenchmarks | ~1 h on hpcc | Reason 3 |
| 3 | Per-launch overhead breakdown | short microbenchmark runs | Reason 1 |
| 4 | Repeated Fig 13 points with confidence intervals and served fraction | a few nscc2b jobs | all |
| 5 | Stock Hopper stack GPU-time census | one nsys run | Reason 4 |
| 6 | ~~A100 control run~~ **done**: MorphX loses on the A100 too (see the A100 control section) | 1.7 GPU-h on nscc | rules out the hardware as the cause |
| 7 | gdr-equivalent overhead on the A100, or a CUDA-graph / single-thread decode runner, to test the host-bound explanation | TBD | A100 control |

## Data locations (local, `logs/`)

| Results | Path |
|---|---|
| No-split, split sweep, `occ` splits, profile log | `20261026-nscc2b/diag-fig13-*` |
| Paper request rates | `20261027-pd-paper-rates/` |
| Pass-built FA2 census and nsys traces | `20261027-fi-pass/check-192309`, `check-192350` |
| Fig 13, pass FA2 + per-grid | `20261027-fi-pass/fig13-192386` |
| Fig 13, sm90 GEMMs | `20261027-fi-pass/fig13-sm90-192442` |
| sm90 rate sweep | `20261027-fi-pass/sweep-*` |
| bf16 mycutlass | `20261027-bf16/` |
| Fig 10–12 with MPS-60 | `20261027-mps60/` |
| nsys decode-blocking analysis, 18 req/s census | `20261028-pd-blocking/` (`report-census-18rps.txt`) |
| nsys decode-blocking analysis, 6 req/s Sm80 and sm90 | `20261028-pd-blocking/trace-194398/` (`report-6rps.txt`) |
| High-priority decode and MorphX runs, 6 req/s, one node | `20261028-pd-blocking/trace-194603/` |
| A100 control (Fig 13 sweeps, step-time benchmark, smoke) | `20261028-a100-nscc/` |
| A100 plots: `a100_fig13`, `a100_vs_paper`, `a100_step_bench` | `20261028-figures/` |
| Plots of the above: sm90 rate sweep, MorphX/stream per setup, decode priority | `20261028-figures/` (`scripts/analysis/plot_pd_followups.py`) |

`.pt` and `.sqlite` intermediates are only on nscc2b, under
`~/scratch/morphx/trackFI/logs/20261027-fi-pass/`.
