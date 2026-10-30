"""How much decode work waits behind prefill under Stream (and vice versa), from nsys traces.

Input: the SQLite export of an nsys trace of the Fig 13 *Stream* variant (scripts/sglang/runner.py
with the prefill thread: prefill on its own CUDA stream from a worker thread, decode on the
caller's stream). Plain sqlite3, no numpy.

For each kernel on the target stream (decode, then prefill as a cross-check):

  ready  = max(end of its launch API call,
               end of the previous op (kernel/memcpy/memset) on the same stream,
               completion of any cudaStreamWaitEvent issued on the stream since that op)
  delay  = start - ready                                     (clipped at 0)
  blocked = part of [ready, start] during which a kernel on the other stream was running
  other   = delay - blocked                                  (launch latency etc.)

Event completion = end of the last op enqueued on the recording stream before the
cudaEventRecord call (CUPTI_ACTIVITY_KIND_CUDA_EVENT matched on eventId + eventSyncId).

Co-run slowdown: a kernel "co-runs" if its [start, end] overlaps any kernel on the other stream,
and runs "alone" if the overlap is zero. "Same kernel" = same demangled name, grid and block,
and the same preceding kernel on the stream (o_proj and down_proj share name and grid but follow
attention and act_and_mul). Baseline = median duration of the alone instances of that kernel in
the same step (every layer of a step has the same batch and KV lengths, which matters for
attention), else over the whole trace (>= --min-alone samples). slowdown = duration - baseline,
summed over co-run instances that have a baseline; time of co-run kernels without a baseline is
extrapolated with the class-wide ratio (both reported).

Step wall time is decomposed over all ops on the stream: kernel time + copy time + queue delay
(blocked by the other stream / other) + host gaps (stream empty, waiting for the CPU to launch)
+ event waits (cross-stream cudaStreamWaitEvent) - same-stream overlap (PDL).

Steps: one forward per occurrence of a once-per-forward marker kernel on the stream
(decode: clamp_position_kernel, prefill: compute_position_kernel; no NVTX in these traces).
A step runs from one marker's start to the next marker's start. Only steps inside the window
where both streams are active are counted.

Ceiling per decode step = blocked delay + co-run slowdown: the decode time MorphX could at most
win back by keeping decode off prefill's SMs (assuming that costs decode nothing).

  python scripts/analysis/nsys_pd_blocking.py trace.sqlite [...] --out DIR
"""

import argparse
import bisect
import collections
import csv
import json
import os
import sqlite3
import statistics
import sys

DECODE_MARKER = "clamp_position_kernel"
PREFILL_MARKER = "compute_position_kernel"
PCTS = (50, 90, 99)


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def kclass(name):
    n = name.lower()
    if "cutlass" in n and "gemm" in n or n.startswith("nvjet") or "gemm" in n:
        return "GEMM"
    if "flashinfer::" in n and ("prefill" in n or "decode" in n or "mergestates" in n):
        return "attention"
    return "other"


class Trace:
    def __init__(self, path):
        self.path = path
        db = sqlite3.connect(path)
        self.strings = dict(db.execute("SELECT id, value FROM StringIds"))
        self.api = {c: (s, e, tid, n) for c, s, e, tid, n in db.execute(
            "SELECT correlationId, start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
        self.kernels = collections.defaultdict(list)  # stream -> [dict]
        for row in db.execute("SELECT start, end, streamId, correlationId, shortName, demangledName, gridX, gridY, gridZ, "
                              "blockX, blockY, blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            s, e, sid, corr, sn, dn, gx, gy, gz, bx, by, bz = row
            self.kernels[sid].append(dict(start=s, end=e, corr=corr, short=self.strings[sn], name=self.strings[dn],
                                          grid=(gx, gy, gz), block=(bx, by, bz), kind="kernel"))
        self.ops = collections.defaultdict(list)  # stream -> kernels + memcpy + memset, by start
        for sid, ks in self.kernels.items():
            ks.sort(key=lambda k: k["start"])
            self.ops[sid].extend(ks)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table not in tables:  # nsys omits activity tables with no rows
                continue
            for s, e, sid, corr in db.execute(f"SELECT start, end, streamId, correlationId FROM {table}"):
                self.ops[sid].append(dict(start=s, end=e, corr=corr, kind=kind))
        for sid in self.ops:
            self.ops[sid].sort(key=lambda o: o["start"])
        # event records: (eventId, eventSyncId) -> (recording stream, record correlationId)
        self.records = {(ev, sync): (sid, corr) for sid, corr, ev, sync in db.execute(
            "SELECT streamId, correlationId, eventId, eventSyncId FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT")
            } if "CUPTI_ACTIVITY_KIND_CUDA_EVENT" in tables else {}
        # stream waits: waiting stream -> [(correlationId, eventId, eventSyncId)]
        self.waits = collections.defaultdict(list)
        for sid, corr, ev, sync in db.execute(
                "SELECT streamId, correlationId, eventId, eventSyncId FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
                "WHERE syncType = 2 AND eventId != 4294967295") if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" in tables else ():
            self.waits[sid].append((corr, ev, sync))
        for sid in self.waits:
            self.waits[sid].sort()
        self.thread_names = dict(db.execute(
            "SELECT t.globalTid, s.value FROM ThreadNames t JOIN StringIds s ON s.id = t.nameId")
            ) if "ThreadNames" in tables else {}
        db.close()

    # ------------------------------------------------------------------ helpers
    def busy_union(self, sid, lo=None, hi=None):
        """Merged [start, end) intervals of kernels on `sid` (PDL lets neighbours overlap)."""
        out = []
        for k in self.kernels[sid]:
            s, e = k["start"], k["end"]
            if out and s <= out[-1][1]:
                if e > out[-1][1]:
                    out[-1][1] = e
            else:
                out.append([s, e])
        starts = [iv[0] for iv in out]
        ends = [iv[1] for iv in out]
        cum = [0]
        for s, e in out:
            cum.append(cum[-1] + e - s)
        return starts, ends, cum

    @staticmethod
    def overlap(union, a, b):
        """Length of [a, b) covered by the merged intervals in `union`."""
        if b <= a:
            return 0
        starts, ends, cum = union
        i = bisect.bisect_right(ends, a)          # first interval ending after a
        j = bisect.bisect_left(starts, b)         # intervals [i, j) start before b
        if j <= i:
            return 0
        tot = cum[j] - cum[i]
        tot -= max(0, a - starts[i])              # clip the first one at a
        tot -= max(0, ends[j - 1] - b)            # clip the last one at b
        return max(0, tot)

    def event_completion(self, ev, sync):
        rec = self.records.get((ev, sync))
        if rec is None:
            return None
        sid, rcorr = rec
        ops = self.ops_by_corr.get(sid)
        if not ops:
            return None
        corrs, ends = ops
        i = bisect.bisect_left(corrs, rcorr)
        return ends[i - 1] if i > 0 else None

    def prepare(self):
        # per stream: ops sorted by correlationId, with running max of end (in-order stream)
        self.ops_by_corr = {}
        for sid, ops in self.ops.items():
            byc = sorted(ops, key=lambda o: o["corr"])
            corrs, ends, m = [], [], 0
            for o in byc:
                m = max(m, o["end"])
                corrs.append(o["corr"])
                ends.append(m)
            self.ops_by_corr[sid] = (corrs, ends)


def identify_streams(tr, decode_marker, prefill_marker, decode_sid=None, prefill_sid=None):
    has = collections.defaultdict(set)
    for sid, ks in tr.kernels.items():
        for k in ks:
            if k["short"] in (decode_marker, prefill_marker):
                has[k["short"]].add(sid)
    if decode_sid is None:
        c = has[decode_marker]
        if len(c) != 1:
            sys.exit(f"{tr.path}: decode marker {decode_marker} on streams {sorted(c)}; pass --decode-stream")
        decode_sid = c.pop()
    if prefill_sid is None:
        c = has[prefill_marker] - {decode_sid}
        if len(c) != 1:
            sys.exit(f"{tr.path}: prefill marker {prefill_marker} on streams {sorted(c)}; pass --prefill-stream")
        prefill_sid = c.pop()
    return decode_sid, prefill_sid


def stream_summary(tr, sid, window):
    ks = tr.kernels[sid]
    threads = collections.Counter(tr.api[k["corr"]][2] for k in ks if k["corr"] in tr.api)
    names = collections.defaultdict(lambda: [0, 0])
    for k in ks:
        names[k["short"]][0] += 1
        names[k["short"]][1] += k["end"] - k["start"]
    lo, hi = window
    union = tr.busy_union(sid)
    by_class = collections.defaultdict(lambda: [0, 0])
    for k in ks:
        c = kclass(k["name"])
        by_class[c][0] += 1
        by_class[c][1] += k["end"] - k["start"]
    return dict(
        stream=sid,
        kernels=len(ks),
        kernel_time_s=sum(k["end"] - k["start"] for k in ks) / 1e9,
        busy_frac_in_window=Trace.overlap(union, lo, hi) / (hi - lo),
        launch_threads={str(t): dict(launches=n, name=tr.thread_names.get(t, "?")) for t, n in threads.items()},
        by_class={c: dict(launches=n, time_s=t / 1e9) for c, (n, t) in by_class.items()},
        top_kernels=[dict(name=n, launches=c, time_ms=t / 1e6)
                     for n, (c, t) in sorted(names.items(), key=lambda x: -x[1][1])[:8]],
    )


def analyze_stream(tr, sid, other_sid, marker, window, min_alone):
    """Per-kernel delay / co-run accounting for stream `sid` against stream `other_sid`."""
    other = tr.busy_union(other_sid)
    waits = tr.waits.get(sid, [])
    wait_corrs = [w[0] for w in waits]
    lo, hi = window

    recs = []
    prev_end, prev_corr, prev_kernel = None, None, None
    for op in tr.ops[sid]:
        api = tr.api.get(op["corr"])
        launch_end = api[1] if api else op["start"]
        dep = None
        if prev_corr is not None:
            i = bisect.bisect_right(wait_corrs, prev_corr)
            j = bisect.bisect_left(wait_corrs, op["corr"])
            for _, ev, sync in waits[i:j]:
                t = tr.event_completion(ev, sync)
                if t is not None and (dep is None or t > dep):
                    dep = t
        ready = max([launch_end] + ([prev_end] if prev_end is not None else []) + ([dep] if dep is not None else []))
        delay = max(0, op["start"] - ready)
        # gap between the previous op's end and this op's start that is not queueing: why it was not ready
        not_ready = max(0, min(ready, op["start"]) - prev_end) if prev_end is not None else 0
        dep_bound = dep is not None and dep > launch_end
        dur = op["end"] - op["start"]
        r = dict(kind=op["kind"], start=op["start"], end=op["end"], dur=dur, delay=delay,
                 host_gap=0 if dep_bound else not_ready, dep_wait=not_ready if dep_bound else 0)
        if op["kind"] == "kernel":
            blocked = Trace.overlap(other, ready, op["start"]) if delay else 0
            ov = Trace.overlap(other, op["start"], op["end"])
            # name + grid + block, plus the previous kernel's name: o_proj and down_proj share
            # name and grid but follow attention and act_and_mul respectively
            r.update(short=op["short"], name=op["name"], cls=kclass(op["name"]),
                     key=(op["name"], op["grid"], op["block"], prev_kernel),
                     blocked=blocked, other=delay - blocked, corun_frac=ov / dur if dur else 0.0)
            prev_kernel = op["short"]
        else:
            r.update(short="", name=op["kind"], cls=op["kind"], key=None, blocked=0, other=delay, corun_frac=0.0)
        recs.append(r)
        prev_end = op["end"] if prev_end is None else max(prev_end, op["end"])
        prev_corr = op["corr"]

    # steps
    marks = [r["start"] for r in recs if r["short"] == marker]
    steps = []
    for a, b in zip(marks, marks[1:]):
        if a >= lo and b <= hi:
            steps.append([a, b])
    step_starts = [s[0] for s in steps]
    in_steps = []
    for r in recs:
        i = bisect.bisect_right(step_starts, r["start"]) - 1
        if i >= 0 and r["start"] < steps[i][1]:
            r["step"] = i
            in_steps.append(r)

    # alone baselines: first the median of alone instances of the same kernel in the same step
    # (same batch and KV lengths, so this also controls attention), else over the whole trace
    alone = collections.defaultdict(list)
    alone_step = collections.defaultdict(list)
    for r in recs:
        if r["key"] is not None and r["corun_frac"] == 0.0:
            alone[r["key"]].append(r["dur"])
            if "step" in r:
                alone_step[(r["key"], r["step"])].append(r["dur"])
    base = {k: statistics.median(v) for k, v in alone.items() if len(v) >= min_alone}
    base_step = {k: statistics.median(v) for k, v in alone_step.items()}
    for r in recs:
        r["base"], r["base_src"] = None, None
        if r["key"] is None:
            pass
        elif "step" in r and (r["key"], r["step"]) in base_step:
            r["base"], r["base_src"] = base_step[(r["key"], r["step"])], "step"
        elif r["key"] in base:
            r["base"], r["base_src"] = base[r["key"]], "global"
        r["slow"] = (r["dur"] - r["base"]) if (r["corun_frac"] > 0 and r["base"] is not None) else 0
    return recs, in_steps, steps, base


def summarize(all_recs, steps, label):
    n_steps = len(steps)
    out = dict(label=label, steps=n_steps)
    if not n_steps:
        return out, []
    recs = [r for r in all_recs if r["kind"] == "kernel"]

    # co-run slowdown per class; kernels with no alone baseline are extrapolated with the class ratio
    cls = collections.defaultdict(collections.Counter)
    for r in recs:
        c = cls[r["cls"]]
        c["n"] += 1
        c["time"] += r["dur"]
        if r["corun_frac"] > 0:
            c["corun_n"] += 1
            c["corun_time"] += r["dur"]
            if r["base"] is not None:
                c["covered_time"] += r["dur"]
                c["covered_base"] += r["base"]
                c["slow"] += r["slow"]
                c["covered_step"] += r["dur"] if r["base_src"] == "step" else 0
    slow_cls = {}
    for name, c in sorted(cls.items()):
        ratio = c["slow"] / c["covered_base"] if c["covered_base"] else 0.0
        uncovered = c["corun_time"] - c["covered_time"]
        # an uncovered kernel's dur = base * (1 + ratio), so its slowdown = dur * ratio / (1 + ratio)
        extra = uncovered * ratio / (1 + ratio) if ratio > -1 else 0
        slow_cls[name] = dict(
            launches_per_step=c["n"] / n_steps, kernel_time_ms_per_step=c["time"] / n_steps / 1e6,
            corun_launch_frac=c["corun_n"] / c["n"] if c["n"] else 0,
            corun_time_frac=c["corun_time"] / c["time"] if c["time"] else 0,
            baseline_coverage=c["covered_time"] / c["corun_time"] if c["corun_time"] else 1.0,
            baseline_from_same_step=c["covered_step"] / c["covered_time"] if c["covered_time"] else 0.0,
            corun_slowdown_ratio=ratio,
            slowdown_ms_per_step_covered=c["slow"] / n_steps / 1e6,
            slowdown_ms_per_step_extrapolated=(c["slow"] + extra) / n_steps / 1e6,
        )
    out["slowdown_by_class"] = slow_cls

    # per-step decomposition over all ops (kernels, memcpy, memset) on the stream
    per_step = [collections.Counter() for _ in steps]
    for r in all_recs:
        if "step" not in r:
            continue
        s = per_step[r["step"]]
        k = r["kind"] == "kernel"
        s["kernels"] += k
        s["kernel_busy"] += r["dur"] if k else 0
        s["copy_busy"] += 0 if k else r["dur"]
        s["blocked"] += r["blocked"]
        s["other"] += r["other"] if k else 0
        s["copy_delay"] += 0 if k else r["delay"]
        s["host_gap"] += r["host_gap"]
        s["dep_wait"] += r["dep_wait"]
        s["slow"] += r["slow"] if k else 0
        if k:
            s["gpu_end"] = max(s["gpu_end"], r["end"])
    rows = []
    for (a, b), s in zip(steps, per_step):
        wall = b - a
        acc = (s["kernel_busy"] + s["copy_busy"] + s["blocked"] + s["other"] + s["copy_delay"] + s["host_gap"]
               + s["dep_wait"])
        rows.append(dict(step_start_ns=a, wall_ms=wall / 1e6, gpu_span_ms=(s["gpu_end"] - a) / 1e6,
                         kernels=s["kernels"], kernel_busy_ms=s["kernel_busy"] / 1e6, copy_busy_ms=s["copy_busy"] / 1e6,
                         blocked_ms=s["blocked"] / 1e6, other_delay_ms=s["other"] / 1e6,
                         copy_delay_ms=s["copy_delay"] / 1e6, host_gap_ms=s["host_gap"] / 1e6,
                         dep_wait_ms=s["dep_wait"] / 1e6, same_stream_overlap_ms=(acc - wall) / 1e6,
                         corun_slowdown_ms=s["slow"] / 1e6))

    keys = [k for k in rows[0] if k not in ("step_start_ns", "kernels")]
    out["per_step_mean_ms"] = {k: statistics.mean(r[k] for r in rows) for k in keys}
    extrap = sum(v["slowdown_ms_per_step_extrapolated"] for v in slow_cls.values())
    out["per_step_mean_ms"]["corun_slowdown_extrapolated_ms"] = extrap
    out["per_step_median_ms"] = {k: statistics.median(r[k] for r in rows) for k in keys}
    out["kernels_per_step"] = statistics.mean(r["kernels"] for r in rows)
    wall = out["per_step_mean_ms"]["wall_ms"]
    blocked = out["per_step_mean_ms"]["blocked_ms"]
    out["ceiling_ms_per_step"] = blocked + extrap
    out["ceiling_frac_of_step_wall"] = (blocked + extrap) / wall
    out["ceiling_covered_only_ms_per_step"] = blocked + out["per_step_mean_ms"]["corun_slowdown_ms"]

    # per-kernel delay distributions (us)
    dl = [r["delay"] / 1e3 for r in recs]
    bl = [r["blocked"] / 1e3 for r in recs]
    nz = [r["blocked"] / 1e3 for r in recs if r["blocked"] > 0]
    out["delay_us"] = {f"p{p}": pct(dl, p) for p in PCTS}
    out["delay_us"]["mean"] = statistics.mean(dl)
    out["delay_us"]["max"] = max(dl)
    out["blocked_us"] = {f"p{p}": pct(bl, p) for p in PCTS}
    out["blocked_us"]["mean"] = statistics.mean(bl)
    out["blocked_launch_frac"] = len(nz) / len(recs)
    out["blocked_us_when_blocked"] = {f"p{p}": pct(nz, p) for p in PCTS}
    by_cls = collections.defaultdict(lambda: [0, 0, 0])
    for r in recs:
        by_cls[r["cls"]][0] += r["blocked"]
        by_cls[r["cls"]][1] += r["other"]
        by_cls[r["cls"]][2] += 1
    out["delay_by_class_ms_per_step"] = {c: dict(blocked=b / n_steps / 1e6, other=o / n_steps / 1e6,
                                                 launches_per_step=n / n_steps)
                                         for c, (b, o, n) in sorted(by_cls.items())}
    return out, rows


def top_kernels(recs, n_steps, k=12):
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        if r["kind"] != "kernel":
            continue
        key = (r["short"][:60], r["key"][1], r["key"][2], r["key"][3])
        a = agg[key]
        a["n"] += 1
        a["dur"] += r["dur"]
        a["blocked"] += r["blocked"]
        a["slow"] += r["slow"]
        a["corun"] += r["corun_frac"] > 0
        if r["base"] is not None:
            a["base"] = r["base"]
            a["based"] += 1
    rows = []
    for (name, grid, block, prev), a in agg.items():
        rows.append(dict(kernel=name, after=(prev or "")[:40], grid="x".join(map(str, grid)),
                         block="x".join(map(str, block)),
                         cls=kclass(name), launches=a["n"], mean_us=a["dur"] / a["n"] / 1e3,
                         alone_median_us=a["base"] / 1e3 if a["based"] else None,
                         corun_frac=a["corun"] / a["n"], blocked_ms_per_step=a["blocked"] / n_steps / 1e6,
                         slowdown_ms_per_step=a["slow"] / n_steps / 1e6))
    rows.sort(key=lambda r: -(r["blocked_ms_per_step"] + r["slowdown_ms_per_step"]))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode-stream", type=int)
    ap.add_argument("--prefill-stream", type=int)
    ap.add_argument("--decode-marker", default=DECODE_MARKER)
    ap.add_argument("--prefill-marker", default=PREFILL_MARKER)
    ap.add_argument("--min-alone", type=int, default=3, help="alone samples needed for a baseline")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for path in args.sqlite:
        tag = os.path.splitext(os.path.basename(path))[0]
        print(f"== {tag}", file=sys.stderr)
        tr = Trace(path)
        tr.prepare()
        dsid, psid = identify_streams(tr, args.decode_marker, args.prefill_marker,
                                      args.decode_stream, args.prefill_stream)
        lo = max(tr.kernels[dsid][0]["start"], tr.kernels[psid][0]["start"])
        hi = min(max(k["end"] for k in tr.kernels[dsid]), max(k["end"] for k in tr.kernels[psid]))
        window = (lo, hi)
        res = dict(trace=path, window_s=(hi - lo) / 1e9, decode_stream=dsid, prefill_stream=psid,
                   stream_id_method=f"decode = stream running the once-per-decode-forward kernel "
                                    f"{args.decode_marker}; prefill = stream running the once-per-extend-forward "
                                    f"kernel {args.prefill_marker}",
                   streams={"decode": stream_summary(tr, dsid, window), "prefill": stream_summary(tr, psid, window)})
        res["window_busy"] = dict(decode=res["streams"]["decode"]["busy_frac_in_window"],
                                  prefill=res["streams"]["prefill"]["busy_frac_in_window"])
        for role, sid, osid, marker in (("decode", dsid, psid, args.decode_marker),
                                        ("prefill", psid, dsid, args.prefill_marker)):
            recs, in_steps, steps, base = analyze_stream(tr, sid, osid, marker, window, args.min_alone)
            summ, rows = summarize(in_steps, steps, role)
            res[role] = summ
            write_csv(os.path.join(args.out, f"{tag}-{role}-steps.csv"), rows)
            write_csv(os.path.join(args.out, f"{tag}-{role}-kernels.csv"), top_kernels(in_steps, max(1, len(steps)), 10**6))
        with open(os.path.join(args.out, f"{tag}-summary.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
        print_report(tag, res)


def print_report(tag, res):
    print(f"\n## {tag}  (window {res['window_s']:.1f} s; decode stream {res['decode_stream']}, "
          f"prefill stream {res['prefill_stream']}; busy decode {res['window_busy']['decode']:.0%}, "
          f"prefill {res['window_busy']['prefill']:.0%})")
    for role in ("decode", "prefill"):
        s = res[role]
        if not s.get("steps"):
            print(f"{role}: no steps")
            continue
        m = s["per_step_mean_ms"]
        print(f"\n### {role} ({s['steps']} steps, {s['kernels_per_step']:.0f} kernels/step)")
        print(f"step wall {m['wall_ms']:.2f} ms (median {s['per_step_median_ms']['wall_ms']:.2f}) = kernels "
              f"{m['kernel_busy_ms']:.2f} + copies {m['copy_busy_ms']:.2f} + blocked {m['blocked_ms']:.2f} "
              f"+ other delay {m['other_delay_ms']:.2f} + copy delay {m['copy_delay_ms']:.2f} "
              f"+ host gaps {m['host_gap_ms']:.2f} + event waits {m['dep_wait_ms']:.2f} "
              f"- same-stream overlap {m['same_stream_overlap_ms']:.2f}")
        print(f"per kernel delay us: " + ", ".join(f"{k} {v:.1f}" for k, v in s["delay_us"].items()))
        print(f"per kernel blocked us: " + ", ".join(f"{k} {v:.1f}" for k, v in s["blocked_us"].items())
              + f"; {s['blocked_launch_frac']:.0%} of launches blocked, when blocked: "
              + ", ".join(f"{k} {v:.1f}" for k, v in s["blocked_us_when_blocked"].items()))
        for c, d in sorted(s["delay_by_class_ms_per_step"].items()):
            print(f"  delay {c:9s} blocked {d['blocked']:.2f} other {d['other']:.2f} ms/step "
                  f"({d['launches_per_step']:.0f} launches/step)")
        for c, d in sorted(s["slowdown_by_class"].items()):
            print(f"  co-run {c:9s} kernel {d['kernel_time_ms_per_step']:.2f} ms/step, co-run {d['corun_time_frac']:.0%} "
                  f"of time, baseline coverage {d['baseline_coverage']:.0%} "
                  f"(same step {d['baseline_from_same_step']:.0%}), slowdown x{1 + d['corun_slowdown_ratio']:.2f}, "
                  f"{d['slowdown_ms_per_step_covered']:.2f} (extrap {d['slowdown_ms_per_step_extrapolated']:.2f}) ms/step")
        print(f"CEILING {s['ceiling_ms_per_step']:.2f} ms/step = {s['ceiling_frac_of_step_wall']:.1%} of step wall "
              f"(covered only {s['ceiling_covered_only_ms_per_step']:.2f})")


if __name__ == "__main__":
    main()
