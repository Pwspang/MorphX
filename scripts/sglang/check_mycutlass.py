"""Correctness of MORPHX_LLM_GEMM=mycutlass against the torch (cuBLAS) path.

One fp16 model load; mycutlass_gemm.set_enabled() switches the linear layers
between torch and mycutlass so both paths see identical weights.
  1. per-layer: every distinct (N, K) weight shape vs F.linear, several T
  2. end to end: prefill logits (max/mean abs diff, argmax agreement) and
     greedy tokens for a few prompts

Usage: MYCUTLASS_LIB_DIR=... python scripts/sglang/check_mycutlass.py --model <snapshot>
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mycutlass_gemm  # noqa: E402
from runner import SglangRunner  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Explain why the sky is blue in two sentences.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "List three prime numbers greater than 100:",
    "Translate to German: The weather is nice today, so we will go for a walk.",
    "In 1969, the first humans to land on the Moon were",
]


def generate(runner, prompts, max_tokens):
    logits = []
    mycutlass_gemm.capture_logits(logits)
    for i, ids in enumerate(prompts):
        runner.add_request(f"r{i}", ids, max_tokens=max_tokens, min_tokens=max_tokens)
    finished = {}
    while runner.has_unfinished_requests():
        for ro in runner.step():
            if ro.finished:
                finished[ro.request_id] = list(ro.outputs[0].token_ids)
    mycutlass_gemm.capture_logits(None)
    torch.cuda.synchronize()
    return logits[0], [finished[f"r{i}"] for i in range(len(prompts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    stream = torch.cuda.Stream()
    runner = SglangRunner(args.model, max_model_len=1024, max_num_seqs=16, max_num_batched_tokens=4096,
                          mem_fraction_static=0.7, stream=stream, gemm="mycutlass")
    print(f"[check] model dtype={runner.model_config.dtype} prepared W^T for {runner.num_prepared_weights} weights", flush=True)

    # 1. per-layer
    shapes = {}
    for name, m in runner.model_runner.model.named_modules():
        w = getattr(m, "weight", None)
        if hasattr(m, "_morphx_wt") and isinstance(w, torch.Tensor):
            shapes.setdefault(tuple(w.shape), (name, m))
    worst = 0.0
    with torch.cuda.stream(stream):
        for (n, k), (name, m) in sorted(shapes.items()):
            for t in (1, 7, 64, 512):
                x = torch.randn(t, k, dtype=torch.float16, device="cuda") * 0.5
                ref = F.linear(x.float(), m.weight.float())
                got = mycutlass_gemm.linear(m, x, m.weight, tag="check").float()
                tor = F.linear(x, m.weight).float()
                err = (got - ref).abs().max().item()
                err_t = (tor - ref).abs().max().item()
                worst = max(worst, err / max(err_t, 1e-6))
                print(f"[layer] {name:40s} N={n:6d} K={k:5d} T={t:3d} max|mycutlass-fp32ref|={err:.4f} "
                      f"max|torch-fp32ref|={err_t:.4f}", flush=True)
    stream.synchronize()
    print(f"[layer] worst error ratio mycutlass/torch vs fp32 reference: {worst:.2f}", flush=True)

    # 2. end to end
    tok = runner.get_tokenizer()
    prompts = [tok.encode(p) for p in PROMPTS]
    results = {}
    for mode in ("torch", "mycutlass"):
        mycutlass_gemm.set_enabled(mode == "mycutlass")
        before = mycutlass_gemm.stats()
        results[mode] = generate(runner, prompts, args.max_tokens)
        after = mycutlass_gemm.stats()
        delta = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}
        print(f"[e2e] {mode}: GEMM calls during generation: {delta}", flush=True)
    lt, tt = results["torch"]
    lm, tm = results["mycutlass"]
    diff = (lt - lm).abs()
    print(f"[e2e] prefill logits {tuple(lt.shape)}: max_abs_diff={diff.max().item():.4f} "
          f"mean_abs_diff={diff.mean().item():.5f} logit_range=[{lt.min().item():.1f},{lt.max().item():.1f}] "
          f"argmax_agree={(lt.argmax(-1) == lm.argmax(-1)).float().mean().item() * 100:.0f}%", flush=True)
    total = same = exact = 0
    for i, (a, b) in enumerate(zip(tt, tm)):
        prefix = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
        same += sum(x == y for x, y in zip(a, b))
        total += len(a)
        exact += a == b
        print(f"[e2e] prompt {i}: identical_prefix={prefix}/{len(a)} torch={tok.decode(a)[:60]!r} "
              f"mycutlass={tok.decode(b)[:60]!r}", flush=True)
    print(f"[e2e] greedy tokens: {same}/{total} position-wise equal ({same / total * 100:.1f}%), "
          f"{exact}/{len(tt)} sequences identical", flush=True)
    print(f"[e2e] fallback/usage counters: {mycutlass_gemm.stats()}", flush=True)


if __name__ == "__main__":
    main()
