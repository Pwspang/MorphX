"""Microbenchmark FlashInfer's ragged prefill attention (FA3 vs FA2), Qwen3-8B shapes, fp16.

Times BatchPrefillWithRaggedKVCacheWrapper.run (causal), the kernel SGLang's
prefill uses, for a few batch shapes. The build is chosen by the environment,
as in check_flashinfer_pass.py (one process per configuration):

  stock nvcc FA3:      python fi_fa3_bench.py --backend fa3
  stock nvcc FA2:      python fi_fa3_bench.py --backend fa2
  pass-built FA2:      MORPHX_FLASHINFER_PASS=1 MORPHX_LEGACY_SM80_FA2=1 python fi_fa3_bench.py --backend fa2
  pass-built FA3 (non-persistent):
                       MORPHX_FLASHINFER_PASS=1 MORPHX_FLASHINFER_FA3=1 python fi_fa3_bench.py --backend fa3
  ... persistent:      + MORPHX_FA3_PERSISTENT=1

Under the MorphX runtime (runtime/build, cuda.cpp) with
SMSCHED_TARGET_PATTERNS=<kernel> SMSCHED_ONLY_WHITELIST=1 SMSCHED_SPLIT_HINT=<n>,
--two-streams first runs the kernel once on a second stream: the runtime then
has two registered streams and confines every launch of the target kernel to
SMs [0, n) (Track S's confinement check). Without it the kernel is managed but
gets the whole GPU.

Each shape's output is checked against torch SDPA (fp16) and printed with the
mean time per launch (CUDA events over --iters back-to-back launches).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashinfer_pass  # noqa: E402

HQ, HKV, D = 32, 8, 128  # Qwen3-8B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fa2", "fa3"], required=True)
    ap.add_argument("--shapes", default="4x2048,1x8192,16x512,2x4096,1x1024",
                    help="comma list of <requests>x<tokens>, causal ragged prefill without prefix")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--two-streams", action="store_true", help="register a second stream first (confinement)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", help="append one JSON line per shape")
    args = ap.parse_args()

    flashinfer_pass.setup_env()
    flashinfer_pass.patch_backends()
    import torch
    import torch.nn.functional as F
    import flashinfer

    dev = "cuda"
    cfg = flashinfer_pass.config_name() + ("-runtime" if "libcuda.so" in os.environ.get("LD_PRELOAD", "") else "")
    print(f"[bench] config={cfg} backend={args.backend} two_streams={args.two_streams} "
          f"split={os.environ.get('SMSCHED_SPLIT_HINT', '-')} target={os.environ.get('SMSCHED_TARGET_PATTERNS', '-')}",
          flush=True)
    stream = torch.cuda.Stream()
    other = torch.cuda.Stream()
    ws = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    rc = 0
    with torch.cuda.stream(stream):
        for shape in args.shapes.split(","):
            nreq, ntok = (int(x) for x in shape.split("x"))
            lens = [ntok] * nreq
            qo = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
            n = sum(lens)
            q = (torch.randn(n, HQ, D, generator=g, device=dev) * 0.5).half()
            k = (torch.randn(n, HKV, D, generator=g, device=dev) * 0.5).half()
            v = (torch.randn(n, HKV, D, generator=g, device=dev) * 0.5).half()
            w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=args.backend)
            t0 = time.perf_counter()
            w.plan(qo, qo, HQ, HKV, D, causal=True, q_data_type=torch.float16)
            plan_ms = (time.perf_counter() - t0) * 1e3
            if args.two_streams:
                other.wait_stream(stream)
                with torch.cuda.stream(other):
                    w.run(q, k, v)
                other.synchronize()
            o = w.run(q, k, v)  # warm-up (+ JIT, + runtime profiling launch)
            torch.cuda.synchronize()
            # correctness vs torch SDPA, per request (GQA)
            err = 0.0
            for a, b in zip(qo[:-1].tolist(), qo[1:].tolist()):
                ref = F.scaled_dot_product_attention(q[a:b].transpose(0, 1), k[a:b].transpose(0, 1),
                                                     v[a:b].transpose(0, 1), is_causal=True, enable_gqa=True)
                err = max(err, (o[a:b].float() - ref.transpose(0, 1).float()).abs().max().item())
            for _ in range(3):
                w.run(q, k, v)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            e0.record(stream)
            for _ in range(args.iters):
                w.run(q, k, v)
            e1.record(stream)
            torch.cuda.synchronize()
            wall_us = (time.perf_counter() - t0) * 1e6 / args.iters
            us = e0.elapsed_time(e1) * 1e3 / args.iters
            ok = err < 2e-2
            rc |= 0 if ok else 1
            backend_used = getattr(w, "_backend", "?")
            print(f"[bench] {shape:>8} {backend_used}: {us:9.1f} us/launch (wall {wall_us:9.1f}) plan {plan_ms:.2f} ms "
                  f"max_err_vs_sdpa={err:.4f} {'OK' if ok else 'BAD'}", flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps({"tag": args.tag, "config": cfg, "backend": args.backend, "used": backend_used,
                                        "shape": shape, "us": us, "wall_us": wall_us, "err": err,
                                        "two_streams": args.two_streams,
                                        "split": os.environ.get("SMSCHED_SPLIT_HINT")}) + "\n")
            del q, k, v, o, w
    flashinfer_pass.runtime_report()
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
