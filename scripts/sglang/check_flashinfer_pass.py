"""Greedy-token check for pass-built FlashInfer (MORPHX_FLASHINFER_PASS=1).

The FlashInfer build is fixed per process (its JIT paths are set at import), so
each configuration runs in its own process and dumps tokens + prefill logits;
--compare then diffs two dumps:

  python scripts/sglang/check_flashinfer_pass.py --model M --out stock.pt
  MORPHX_FLASHINFER_PASS=1 python scripts/sglang/check_flashinfer_pass.py --model M --out pass.pt
  python scripts/sglang/check_flashinfer_pass.py --compare stock.pt pass.pt

Prompts: six short ones (batched decode) plus two ~2-3k-token ones, so the paged
decode kernel splits the KV and runs the merge-states kernel. Also compiles
FlashInfer's cascade module (merge_state), so one run pre-builds every
FlashInfer JIT module the Fig 13/14 drivers use.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [
    "The capital of France is",
    "Explain why the sky is blue in two sentences.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "List three prime numbers greater than 100:",
    "Translate to German: The weather is nice today, so we will go for a walk.",
    "In 1969, the first humans to land on the Moon were",
]
LONG = ("Here is a list of facts about the solar system. " * 300, "Summarize the following log.\n" + "".join(
    f"[{i:04d}] worker-{i % 7} finished task {i * 13 % 101} in {i % 17} ms\n" for i in range(150)))


def compare(a_path, b_path):
    import torch

    a, b = torch.load(a_path), torch.load(b_path)
    print(f"[cmp] {a_path} ({a['config']}) vs {b_path} ({b['config']})")
    for key in ("logits_short", "logits_long"):
        if key in a and key in b:
            d = (a[key].float() - b[key].float()).abs()
            agree = (a[key].argmax(-1) == b[key].argmax(-1)).float().mean().item() * 100
            print(f"[cmp] {key} {tuple(a[key].shape)}: max_abs_diff={d.max().item():.4f} mean={d.mean().item():.5f} "
                  f"argmax_agree={agree:.0f}%")
    total = same = exact = 0
    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        prefix = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), len(x))
        same += sum(u == v for u, v in zip(x, y))
        total += len(x)
        exact += x == y
        if x != y:
            print(f"[cmp] prompt {i}: identical_prefix={prefix}/{len(x)}")
    print(f"[cmp] greedy tokens: {same}/{total} position-wise equal, {exact}/{len(a['tokens'])} sequences identical"
          + ("  IDENTICAL" if exact == len(a["tokens"]) else "  DIFFERENT"))
    return 0 if exact == len(a["tokens"]) else 1


def bench(runner, torch, reps=3, nreq=16, in_len=2048, out_len=64):
    import statistics

    pre, dec = [], []
    for rep in range(reps):
        for i in range(nreq):
            ids = [(i * 7919 + j * 31 + rep) % 30000 + 100 for j in range(in_len)]
            runner.add_request(f"b{rep}-{i}", ids, max_tokens=out_len, min_tokens=out_len)
        while runner.has_unfinished_requests():
            np_before = runner.num_prefill_steps
            torch.cuda.synchronize()
            t = time.perf_counter()
            runner.step()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) * 1e3
            if rep == 0:
                continue  # warm-up round
            (pre if runner.num_prefill_steps != np_before else dec).append(dt)
    print(f"[bench] prefill step (4 x {in_len} tokens): median {statistics.median(pre):.2f} ms "
          f"(n={len(pre)}, min {min(pre):.2f})", flush=True)
    full = dec[: len(dec) // 2]  # first half: all {nreq} requests still decoding
    print(f"[bench] decode step (batch {nreq}, ctx {in_len}+): median {statistics.median(full):.3f} ms "
          f"(n={len(full)}, min {min(full):.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--bench", action="store_true",
                    help="also time prefill/decode steps: 16 requests x 2048 tokens (4 prefill steps of 8192 "
                         "tokens), then decode at batch 16")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)

    import torch

    import flashinfer_pass
    import mycutlass_gemm
    from runner import SglangRunner  # sets up flashinfer_pass before sglang imports flashinfer

    from flashinfer.jit import env as jit_env

    cfg = ("pass" if flashinfer_pass.enabled() else "stock") + ("-fa2" if flashinfer_pass.fa2_only() else "-auto") + \
        ("-runtime" if "libcuda.so" in os.environ.get("LD_PRELOAD", "") else "")
    print(f"[fi] config={cfg} jit_dir={jit_env.FLASHINFER_JIT_DIR} nvcc={os.environ.get('FLASHINFER_NVCC', '-')}", flush=True)
    t0 = time.time()
    stream = torch.cuda.Stream()
    runner = SglangRunner(args.model, max_model_len=4096, max_num_seqs=16, max_num_batched_tokens=8192,
                          mem_fraction_static=0.7, stream=stream, gemm=os.environ.get("MORPHX_LLM_GEMM", "mycutlass"))
    print(f"[fi] runner ready in {time.time() - t0:.1f}s dtype={runner.model_config.dtype}", flush=True)
    tok = runner.get_tokenizer()

    def generate(prompts, tag):
        logits = []
        mycutlass_gemm.capture_logits(logits)
        for i, p in enumerate(prompts):
            runner.add_request(f"{tag}{i}", tok.encode(p), max_tokens=args.max_tokens, min_tokens=args.max_tokens)
        done = {}
        t = time.time()
        while runner.has_unfinished_requests():
            for ro in runner.step():
                if ro.finished:
                    done[ro.request_id] = list(ro.outputs[0].token_ids)
        torch.cuda.synchronize()
        mycutlass_gemm.capture_logits(None)
        print(f"[fi] {tag}: {len(prompts)} prompts x {args.max_tokens} tokens in {time.time() - t:.1f}s", flush=True)
        return logits[0].cpu() if logits else None, [done[f"{tag}{i}"] for i in range(len(prompts))]

    t1 = time.time()
    ls, ts = generate(PROMPTS, "s")  # first call includes the JIT build when the cache is cold
    print(f"[fi] first generation (incl. any JIT build) {time.time() - t1:.1f}s", flush=True)
    ll, tl = generate(list(LONG), "l")

    # Pre-build the cascade module (merge_state) too; prefill-with-prefix paths use it.
    from flashinfer.cascade import merge_state

    with torch.cuda.stream(stream):
        v = torch.randn(4, 8, 128, dtype=torch.float16, device="cuda")
        s = torch.randn(4, 8, dtype=torch.float32, device="cuda")
        merge_state(v, s, v, s)
    torch.cuda.synchronize()
    if args.bench:
        bench(runner, torch)
    for i, t in enumerate(ts + tl):
        print(f"[fi] prompt {i}: {tok.decode(t)[:70]!r}", flush=True)
    if args.out:
        out = {"config": cfg, "tokens": ts + tl}
        if ls is not None:
            out["logits_short"] = ls
        if ll is not None:
            out["logits_long"] = ll
        torch.save(out, args.out)
        print(f"[fi] wrote {args.out}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # SGLang's background threads can hang interpreter shutdown


if __name__ == "__main__":
    sys.exit(main())
