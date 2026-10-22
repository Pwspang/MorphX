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

GROUPS = ("mycutlass (CUTLASS)", "FlashInfer", "sgl-kernel", "Triton", "cuBLAS/cuBLASLt", "PyTorch native", "other")


def classify(op_name, kernel_name):
    op = op_name or ""
    k = kernel_name or ""
    lk = k.lower()
    if k.startswith("void cutlass::Kernel") or "cutlass::Kernel<" in k:
        return "mycutlass (CUTLASS)"
    if "at::native" in k or "at_cuda_detail" in k:
        return "PyTorch native"
    # sgl-kernel torch ops, plus sglang's own jit_kernel (tvm-ffi, nvcc at run time; anonymous namespace)
    if op.startswith("sgl_kernel::") or "sgl_kernel" in lk or "sglang" in lk or "anonymous namespace" in k:
        return "sgl-kernel"
    if "flashinfer" in lk or op.startswith("flashinfer"):
        return "FlashInfer"
    if any(s in lk for s in ("nvjet", "cublas", "gemm", "xmma", "gemv", "splitk")) and "cutlass" not in lk:
        return "cuBLAS/cuBLASLt"
    if "at::native" in k or "at_cuda_detail" in k or "elementwise_kernel" in lk or "reduce_kernel" in lk:
        return "PyTorch native"
    if op.startswith("aten::") and any(s in lk for s in ("cutlass", "sm90", "sm80")):
        return "cuBLAS/cuBLASLt"
    if k.startswith("kernel_cutlass_kernel") and "flashinfer" in lk:
        return "FlashInfer"
    # Triton kernels carry their plain Python function name (no C++ signature/namespace).
    if "(" not in k and "::" not in k and not k.startswith("void"):
        return "Triton"
    if op.startswith("aten::"):
        return "PyTorch native"
    return "other"


def trace_kernels(prof):
    """Parse the profiler's chrome trace.

    Returns (rows, streams). rows are dicts with phase (innermost
    record_function range), op (innermost torch op), name, dur (us) and stream
    for every GPU kernel, including ones launched outside any torch op
    (tvm-ffi / FlashInfer / CuTe DSL / Triton launches, which
    FunctionEvent.kernels misses); the CPU side is found through the launch's
    correlation id. streams counts GPU activity per (phase, category, CUPTI stream id).
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as f:
        events = json.load(f).get("traceEvents", [])
    os.unlink(path)

    ops_by_tid = collections.defaultdict(list)
    ranges_by_tid = collections.defaultdict(list)
    launches = {}
    gpu = []
    for e in events:
        cat = e.get("cat")
        if cat == "cpu_op":
            ops_by_tid[e.get("tid")].append(e)
        elif cat == "user_annotation":
            ranges_by_tid[e.get("tid")].append(e)
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launches[corr] = e
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(e)

    def innermost(spans_by_tid):
        """correlation id -> (innermost span name, names of all enclosing spans) for its launch."""
        found = {}
        by_tid = collections.defaultdict(list)
        for corr, e in launches.items():
            by_tid[e.get("tid")].append((e["ts"], corr))
        for tid, items in by_tid.items():
            spans = sorted(spans_by_tid.get(tid, []), key=lambda o: (o["ts"], -o.get("dur", 0)))
            items.sort()
            stack, j = [], 0
            for ts, corr in items:
                while j < len(spans) and spans[j]["ts"] <= ts:
                    stack.append(spans[j])
                    j += 1
                while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) < ts:
                    stack.pop()
                enclosing = [o["name"] for o in stack if o["ts"] + o.get("dur", 0) >= ts]
                found[corr] = (enclosing[-1] if enclosing else "", enclosing)
        return found

    launch_op = innermost(ops_by_tid)
    # SGLang adds its own record_function ranges (step[...]); phases are ours
    launch_phase = innermost({t: [r for r in rs if r["name"].startswith("morphx_")] for t, rs in ranges_by_tid.items()})
    rows = []
    streams = collections.Counter()
    for k in gpu:
        a = k.get("args") or {}
        corr = a.get("correlation")
        phase = launch_phase.get(corr, ("", []))[0]
        streams[(phase, k.get("cat"), a.get("stream"))] += 1
        if k.get("cat") == "kernel":
            op, op_stack = launch_op.get(corr, ("", []))
            rows.append(dict(phase=phase, op=op, op_stack=op_stack, name=k.get("name", ""),
                             dur=float(k.get("dur", 0.0)), stream=a.get("stream"),
                             tid=(launches.get(corr) or {}).get("tid")))
    return rows, streams


def build_origin(name):
    """How the kernel's binary is produced, i.e. whether the MorphX pass can instrument it."""
    lk = name.lower()
    if "nvjet" in lk or "cublas" in lk:
        return "cuBLAS/cuBLASLt (closed source)"
    if "cutlass::Kernel<" in name:
        return "CUTLASS 2.x from source (libmycutlass, pass-built)"
    if name.startswith("kernel_cutlass_kernel"):
        return "CuTe DSL (Python -> MLIR JIT)"
    if "(" not in name and "::" not in name and not name.startswith("void"):
        return "Triton JIT"
    return "CUDA C++ source (rebuildable with clang + pass)"


def census_table(rows):
    by_group = collections.Counter()
    by_kernel = collections.Counter()
    kernel_group = {}
    count = collections.Counter()
    for r in rows:
        op, name, dur = r["op"], r["name"], r["dur"]
        g = classify(op, name)
        by_group[g] += dur
        by_kernel[name] += dur
        kernel_group[name] = g
        count[name] += 1
    total = sum(by_group.values()) or 1.0
    return by_group, by_kernel, kernel_group, count, total


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
    ap.add_argument("--gemm", default=os.environ.get("MORPHX_LLM_GEMM", "torch"), choices=("torch", "mycutlass"))
    ap.add_argument("--dtype", default="auto", help="model dtype (mycutlass needs float16; use float16 for torch too to compare)")
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
        gemm=args.gemm,
        dtype=args.dtype,
    )
    runner = engine.runner
    print(f"[smoke] engine ready in {time.time() - t0:.1f}s; kv_capacity={runner.kv_capacity} tokens "
          f"gemm={runner.gemm} dtype={runner.model_config.dtype}", flush=True)

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

    # 2 + 3. profile one prefill batch and several decode steps in one session
    from torch.profiler import ProfilerActivity, profile, record_function

    # The first profiler sessions after CUPTI init come back empty or without
    # durations/correlations (seen on H200 + torch 2.9.1); burn two on real steps.
    for w in range(2):
        engine.add_request(f"pw{w}", engine_backend.make_prompt([2] * args.input_len), engine_backend.make_sampling_params(4, 4))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            while engine.has_unfinished_requests():
                engine.step()
            torch.cuda.synchronize()
    for i in range(args.census_batch):
        engine.add_request(f"c{i}", engine_backend.make_prompt([(i + j) % 1000 + 1 for j in range(args.input_len)]),
                           engine_backend.make_sampling_params(args.output_len, args.output_len))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("morphx_marker"), torch.cuda.stream(stream):
            torch.ones(1, device="cuda").fill_(7)  # marker kernel on the caller's stream
        with record_function("morphx_prefill"):
            out = engine.step()
        with record_function("morphx_decode"):
            for _ in range(args.census_decode_steps):
                engine.step()
        torch.cuda.synchronize()
    assert len(out) == args.census_batch, f"expected one prefill of {args.census_batch}, got {len(out)}"
    while engine.has_unfinished_requests():
        engine.step()

    rows, streams = trace_kernels(prof)
    marker_streams = {r["stream"] for r in rows if r["phase"] == "morphx_marker"}
    print(f"[stream] caller's stream (CUPTI id) from the marker kernel: {sorted(marker_streams, key=str)}", flush=True)
    print(f"[stream] GPU activity per (phase, category, stream): {dict(streams)}", flush=True)
    off = [r for r in rows if r["phase"] in ("morphx_prefill", "morphx_decode") and r["stream"] not in marker_streams]
    stream_ok = len(marker_streams) == 1 and not off
    main_tids = collections.Counter(r["tid"] for r in rows if r["stream"] in marker_streams).most_common(1)
    for key, n in collections.Counter(
        (r["phase"], r["name"][:100], " > ".join(r["op_stack"][-4:]) or "-", r["tid"]) for r in off
    ).most_common(20):
        print(f"[stream]   off-stream kernel x{n}: phase={key[0]} launch_tid={key[3]} (LLM tid {main_tids}) "
              f"ops={key[2]}: {key[1]}", flush=True)
    print(f"[stream] all LLM kernels on the caller's stream: {stream_ok}", flush=True)

    lines = []
    for label, nsteps in (("prefill", 1), ("decode", args.census_decode_steps)):
        phase_rows = [r for r in rows if r["phase"] == "morphx_" + label]
        by_group, by_kernel, kernel_group, count, total = census_table(phase_rows)
        by_origin = collections.Counter()
        for r in phase_rows:
            by_origin[build_origin(r["name"])] += r["dur"]
        print(f"[census] {label}: {len(phase_rows)} kernels, {total / 1e3:.2f} ms GPU over {nsteps} step(s)", flush=True)
        for g in GROUPS:
            print(f"[census]   {g:16s} {by_group[g] / total * 100:5.1f}%  {by_group[g] / 1e3:8.3f} ms", flush=True)
        for o, d in by_origin.most_common():
            print(f"[census]   origin {o:48s} {d / total * 100:5.1f}%", flush=True)
        lines.append(f"### {label.capitalize()} ({'1 batch' if nsteps == 1 else f'{nsteps} steps'}, "
                     f"batch {args.census_batch}, input {args.input_len}, {len(phase_rows)} kernel launches, {total / 1e3:.2f} ms GPU)\n")
        lines.append("| Group | % GPU time | ms |")
        lines.append("|---|---:|---:|")
        for g in GROUPS:
            lines.append(f"| {g} | {by_group[g] / total * 100:.1f} | {by_group[g] / 1e3:.3f} |")
        lines.append("")
        lines.append("| How the kernel binary is built | % GPU time |")
        lines.append("|---|---:|")
        for o, d in by_origin.most_common():
            lines.append(f"| {o} | {d / total * 100:.1f} |")
        lines.append("")
        lines.append("Top kernels:\n")
        op_of = {}
        for r in phase_rows:
            op_of.setdefault(r["name"], r["op"])
        lines.append("| Kernel | Launched from (torch op) | Group | Launches | % GPU time |")
        lines.append("|---|---|---|---:|---:|")
        for name, dur in by_kernel.most_common(20):
            short = name if len(name) <= 110 else name[:107] + "..."
            short = short.replace("|", "\\|")
            op = (op_of.get(name) or "-").replace("|", "\\|")
            lines.append(f"| `{short}` | `{op}` | {kernel_group[name]} | {count[name]} | {dur / total * 100:.1f} |")
        lines.append("")
    if args.census_out:
        with open(args.census_out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[census] written to {args.census_out}", flush=True)
    else:
        print("\n".join(lines))
    if runner.gemm == "mycutlass":
        import mycutlass_gemm

        print(f"[smoke] mycutlass GEMM calls/fallbacks: {mycutlass_gemm.stats()}", flush=True)
    print("SMOKE: " + ("PASS" if ok and stream_ok else "FAIL"), flush=True)
    return 0 if ok and stream_ok else 1


engine_backend = None

if __name__ == "__main__":
    main()
