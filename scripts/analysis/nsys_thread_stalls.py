#!/usr/bin/env python3
"""Cross-thread CUDA-call stalls in one process, from an nsys SQLite export.

For the two threads that launch the most kernels (e.g. the LLM thread and the GGNN/matmul
worker in Fig 10-12 "stream"), report each thread's time inside CUDA API calls longer than
--min-ms, by call name, and which call the *other* thread was in during each long call.
A call that should be quick (cudaEventCreate, cudaLaunchKernel, cudaMalloc...) running long
while the other thread sits in a blocking sync (cudaDeviceSynchronize, cudaStreamSynchronize,
a pageable cudaMemcpyAsync) is the driver-serialization stall fixed in runner.py on 2026-10-28.

Usage: nsys_thread_stalls.py <trace.sqlite>... [--min-ms 20]
"""

import argparse
import collections
import sqlite3

LLM_HINTS = ("flashinfer", "Prefill", "Decode", "rms", "rotary", "silu", "Sm80", "cutlass")
SYNC_CALLS = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize", "cudaMemcpyAsync", "cudaMemcpy")


def analyse(path, min_ns):
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    launches = collections.Counter()
    kernel_names = collections.defaultdict(collections.Counter)
    for tid, kname in db.execute(
            "SELECT r.globalTid, k.shortName FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId"):
        launches[tid] += 1
        kernel_names[tid][names[kname]] += 1
    top = [t for t, _ in launches.most_common(2)]
    if len(top) < 2:
        print(f"{path}: fewer than two kernel-launching threads ({len(top)})")
        return

    def label(t):
        hits = sum(c for n, c in kernel_names[t].items() if any(h.lower() in n.lower() for h in LLM_HINTS))
        return "LLM" if hits > 0.3 * launches[t] else "worker"

    calls = collections.defaultdict(list)
    for s, e, t, n in db.execute("SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        if t in top:
            calls[t].append((s, e, names[n].split("_v")[0]))
    for t in top:
        calls[t].sort()
    lo = min(c[0][0] for c in calls.values())
    hi = max(c[-1][1] for c in calls.values())
    print(f"== {path}  window {(hi - lo) / 1e9:.1f} s")
    for t, other in ((top[0], top[1]), (top[1], top[0])):
        print(f"  thread {label(t)} ({launches[t]} launches; top kernels: "
              f"{', '.join(n[:40] for n, _ in kernel_names[t].most_common(2))})")
        longs = [(s, e, n) for s, e, n in calls[t] if e - s > min_ns]
        tot = collections.Counter()
        cnt = collections.Counter()
        during = collections.defaultdict(collections.Counter)
        for s, e, n in longs:
            tot[n] += (e - s) / 1e9
            cnt[n] += 1
            best = max(((min(e, oe) - max(s, os_), on) for os_, oe, on in calls[other] if oe > s and os_ < e),
                       default=(0, "-"))
            during[n][best[1] if best[0] > 0.5 * (e - s) else "(other thread not in a CUDA call)"] += 1
        if not longs:
            print("    no CUDA calls longer than the threshold")
        for n, secs in tot.most_common():
            what = ", ".join(f"{k} x{v}" for k, v in during[n].most_common(3))
            flag = "" if n in SYNC_CALLS else "   <-- non-sync call stalled"
            print(f"    {n:28s} {cnt[n]:5d} calls  {secs:7.2f} s   other thread: {what}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--min-ms", type=float, default=20.0)
    args = ap.parse_args()
    for p in args.traces:
        analyse(p, args.min_ms * 1e6)


if __name__ == "__main__":
    main()
