"""Smoke test for the in-process SGLang runner (stock wheels, no MorphX pass).

1. Poisson workload: N requests, fixed input/output length, through
   backend.SglangEngine on a dedicated CUDA stream; prints throughput, TTFT, TBT.
2. Stream check: every GPU kernel of a profiled prefill + decode must run on the
   caller's stream (identified with a marker kernel launched on it).
3. Kernel census: GPU time per kernel group for one prefill batch and several
   decode steps, written as Markdown (--census-out).

Usage (on a GPU node):
  python scripts/sglang/smoke_runner.py --model <snapshot dir> [--census-out docs/sglang-kernel-census.md]
"""

import argparse
import collections
import json
import os
import statistics
import sys
import tempfile
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend as llm_backend  # noqa: E402

GROUPS = ("FlashInfer", "sgl-kernel", "Triton", "cuBLAS/cuBLASLt", "PyTorch native", "other")


def classify(op_name, kernel_name):
    op = op_name or ""
    k = kernel_name or ""
    lk = k.lower()
    if op.startswith("sgl_kernel::") or "sgl_kernel" in lk or "sglang" in lk:
        return "sgl-kernel"
    if "flashinfer" in lk or op.startswith("flashinfer"):
        return "FlashInfer"
    if any(s in lk for s in ("nvjet", "cublas", "gemm", "xmma", "gemv", "splitk")) and "cutlass" not in lk:
        return "cuBLAS/cuBLASLt"
    if "at::native" in k or "at_cuda_detail" in k or "elementwise_kernel" in lk or "reduce_kernel" in lk:
        return "PyTorch native"
    if op.startswith("aten::") and any(s in lk for s in ("cutlass", "sm90", "sm80")):
        return "cuBLAS/cuBLASLt"
    # Triton kernels carry their plain Python function name (no C++ signature/namespace).
    if "(" not in k and "::" not in k and not k.startswith("void"):
        return "Triton"
    if op.startswith("aten::"):
        return "PyTorch native"
    return "other"


def kernel_rows(prof):
    """(innermost op name, kernel name, duration us) for every device kernel."""
    rows = []
    for evt in prof.events():
        for k in getattr(evt, "kernels", []) or []:
            dur = getattr(k, "duration", None)
            if dur is None:
                continue
            rows.append((evt.name, k.name, float(dur)))
    return rows


def census_table(rows):
    by_group = collections.Counter()
    by_kernel = collections.Counter()
    kernel_group = {}
    count = collections.Counter()
    for op, name, dur in rows:
        g = classify(op, name)
        by_group[g] += dur
        by_kernel[name] += dur
        kernel_group[name] = g
        count[name] += 1
    total = sum(by_group.values()) or 1.0
    return by_group, by_kernel, kernel_group, count, total


def trace_streams(prof):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)
    os.unlink(path)
    streams = collections.Counter()
    for e in trace.get("traceEvents", []):
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        streams[(e.get("cat"), (e.get("args") or {}).get("stream"))] += 1
    return streams


def run_poisson(engine, args, rng):
    gaps = rng.exponential(1.0 / args.rate, args.num_requests)
    arrivals = np.cumsum(gaps) - gaps[0]
    vocab = getattr(engine.get_tokenizer(), "vocab_size", 32000)
    prompts = [[(i * 7 + j) % (vocab - 1) + 1 for j in range(args.input_len)] for i in range(args.num_requests)]
    finished = []
    start = time.time()
    i = 0
    steps = 0
    while i < args.num_requests or engine.has_unfinished_requests():
        now = time.time() - start
        while i < args.num_requests and arrivals[i] <= now:
            engine.add_request(
                str(i),
                engine_backend.make_prompt(prompts[i]),
                engine_backend.make_sampling_params(args.output_len, args.output_len),
                arrival_time=start + arrivals[i],
            )
            i += 1
        if engine.has_unfinished_requests():
            for ro in engine.step():
                if ro.finished:
                    finished.append(ro)
            steps += 1
        else:
            time.sleep(min(0.001, max(0.0, arrivals[i] - now)))
    wall = time.time() - start
    return finished, wall, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-requests", type=int, default=20)
    ap.add_argument("--rate", type=float, default=4.0, help="Poisson arrival rate (req/s)")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--census-batch", type=int, default=8)
    ap.add_argument("--census-decode-steps", type=int, default=16)
    ap.add_argument("--census-out", default="")
    ap.add_argument("--in-thread", action="store_true", help="run everything in a non-main thread (as colocate/run.py does)")
    args = ap.parse_args()

    if args.in_thread:
        result = {}
        t = threading.Thread(target=lambda: result.update(rc=_main(args)))
        t.start()
        t.join()
        sys.exit(result.get("rc", 1))
    sys.exit(_main(args))


def _main(args):
    global engine_backend
    engine_backend = llm_backend.get_backend("sglang")
    rng = np.random.default_rng(0)
    stream = torch.cuda.Stream()
    print(f"[smoke] backend={engine_backend.name} thread={threading.current_thread().name} pid={os.getpid()} "
          f"stream=0x{stream.cuda_stream:x}", flush=True)
    t0 = time.time()
    max_model_len = args.input_len + args.output_len
    engine = engine_backend.create_engine(
        args.model,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len * args.census_batch,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.mem_fraction,
        stream=stream,
    )
    runner = engine.runner
    print(f"[smoke] engine ready in {time.time() - t0:.1f}s; kv_capacity={runner.kv_capacity} tokens", flush=True)

    # warmup: one request end to end
    engine.add_request("warmup", engine_backend.make_prompt([1] * args.input_len), engine_backend.make_sampling_params(8, 8))
    while engine.has_unfinished_requests():
        engine.step()

    # 1. Poisson run
    finished, wall, steps = run_poisson(engine, args, rng)
    ttft = [(r.metrics.first_token_time - r.metrics.arrival_time) * 1e3 for r in finished]
    lat = [(r.metrics.finished_time - r.metrics.arrival_time) * 1e3 for r in finished]
    tbt = []
    for r in finished:
        tt = r.metrics.token_times
        tbt += [(b - a) * 1e3 for a, b in zip(tt, tt[1:])]
    ntok = sum(len(r.outputs[0].token_ids) for r in finished)
    ok = len(finished) == args.num_requests and all(len(r.outputs[0].token_ids) == args.output_len for r in finished)
    print(f"[poisson] requests={len(finished)}/{args.num_requests} rate={args.rate}/s in={args.input_len} out={args.output_len} "
          f"wall={wall:.2f}s steps={steps} (prefill={runner.num_prefill_steps} decode={runner.num_decode_steps})", flush=True)
    print(f"[poisson] output_tokens={ntok} throughput={ntok / wall:.1f} tok/s "
          f"({(ntok + args.input_len * len(finished)) / wall:.1f} tok/s incl. prompt)", flush=True)
    print(f"[poisson] TTFT ms: mean={statistics.mean(ttft):.1f} p50={statistics.median(ttft):.1f} max={max(ttft):.1f}", flush=True)
    print(f"[poisson] TBT  ms: mean={statistics.mean(tbt):.2f} p50={statistics.median(tbt):.2f} p99={np.percentile(tbt, 99):.2f}", flush=True)
    print(f"[poisson] latency ms: mean={statistics.mean(lat):.1f} max={max(lat):.1f}  exact_lengths={ok}", flush=True)
    print(f"[poisson] kv_reserved_after={runner.reserved_tokens} (expect 0) running_empty={not engine.has_unfinished_requests()}", flush=True)

    # 2 + 3. profile one prefill batch and several decode steps
    from torch.profiler import ProfilerActivity, profile

    for i in range(args.census_batch):
        engine.add_request(f"c{i}", engine_backend.make_prompt([(i + j) % 1000 + 1 for j in range(args.input_len)]),
                           engine_backend.make_sampling_params(args.output_len, args.output_len))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_prefill:
        with torch.cuda.stream(stream):
            torch.ones(1, device="cuda").fill_(7)  # marker kernel on the caller's stream
        out = engine.step()
        torch.cuda.synchronize()
    assert runner.num_prefill_steps > 0 and len(out) == args.census_batch, f"expected one prefill of {args.census_batch}, got {len(out)}"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_decode:
        with torch.cuda.stream(stream):
            torch.ones(1, device="cuda").fill_(7)
        for _ in range(args.census_decode_steps):
            engine.step()
        torch.cuda.synchronize()
    while engine.has_unfinished_requests():
        engine.step()

    stream_ok = True
    for label, prof in (("prefill", prof_prefill), ("decode", prof_decode)):
        streams = trace_streams(prof)
        kernel_streams = sorted({s for (cat, s) in streams if cat == "kernel"}, key=str)
        print(f"[stream] {label}: GPU activity per (category, stream): {dict(streams)}", flush=True)
        if len(kernel_streams) != 1:
            stream_ok = False
    print(f"[stream] all kernels on one stream (the caller's, incl. marker): {stream_ok}", flush=True)

    lines = []
    for label, prof, nsteps in (("prefill", prof_prefill, 1), ("decode", prof_decode, args.census_decode_steps)):
        rows = kernel_rows(prof)
        by_group, by_kernel, kernel_group, count, total = census_table(rows)
        print(f"[census] {label}: {len(rows)} kernels, {total / 1e3:.2f} ms GPU over {nsteps} step(s)", flush=True)
        for g in GROUPS:
            print(f"[census]   {g:16s} {by_group[g] / total * 100:5.1f}%  {by_group[g] / 1e3:8.3f} ms", flush=True)
        lines.append(f"### {label.capitalize()} ({'1 batch' if nsteps == 1 else f'{nsteps} steps'}, "
                     f"batch {args.census_batch}, input {args.input_len}, {len(rows)} kernel launches, {total / 1e3:.2f} ms GPU)\n")
        lines.append("| Group | % GPU time | ms |")
        lines.append("|---|---:|---:|")
        for g in GROUPS:
            lines.append(f"| {g} | {by_group[g] / total * 100:.1f} | {by_group[g] / 1e3:.3f} |")
        lines.append("")
        lines.append("Top kernels:\n")
        lines.append("| Kernel | Group | Launches | % GPU time |")
        lines.append("|---|---|---:|---:|")
        for name, dur in by_kernel.most_common(15):
            short = name if len(name) <= 110 else name[:107] + "..."
            short = short.replace("|", "\\|")
            lines.append(f"| `{short}` | {kernel_group[name]} | {count[name]} | {dur / total * 100:.1f} |")
        lines.append("")
    if args.census_out:
        with open(args.census_out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[census] written to {args.census_out}", flush=True)
    else:
        print("\n".join(lines))
    print("SMOKE: " + ("PASS" if ok and stream_ok else "FAIL"), flush=True)
    return 0 if ok and stream_ok else 1


engine_backend = None

if __name__ == "__main__":
    main()
