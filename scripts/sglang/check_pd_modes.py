"""Check the SGLang runner's PD scheduling modes against serial mode.

Runs the same greedy requests (all submitted at once) through one mode per
process and saves the generated tokens; compare the files with --compare.

  python scripts/sglang/check_pd_modes.py --model <dir> --mode serial  --out serial.json
  python scripts/sglang/check_pd_modes.py --model <dir> --mode thread  --out thread.json
  python scripts/sglang/check_pd_modes.py --model <dir> --mode chunked --out chunked.json
  python scripts/sglang/check_pd_modes.py --compare serial.json thread.json chunked.json

thread: prefill worker thread on its own stream (VLLM_PREFILL_THREAD semantics).
chunked: mixed decode + prompt-chunk batches; --chunk-budget below the prompt
length forces requests to be split across steps.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(args):
    from runner import SglangRunner

    stream = torch.cuda.Stream()
    chunked = args.mode == "chunked"
    budget = args.chunk_budget if chunked else args.input_len + args.output_len
    runner = SglangRunner(
        args.model,
        max_model_len=args.input_len + args.output_len,
        max_num_seqs=64,
        max_num_batched_tokens=budget,
        mem_fraction_static=0.85,
        stream=stream,
        prefill_thread=args.mode == "thread",
        chunked_prefill=chunked,
    )
    vocab = runner.model_config.vocab_size
    free_before = runner.model_runner.token_to_kv_pool_allocator.available_size()
    if args.warmup:
        with torch.cuda.stream(stream):
            runner.warmup(args.input_len, 4)

    # Varied prompts so a scheduling bug can't hide behind identical requests.
    gen = torch.Generator().manual_seed(1234)
    prompts = [torch.randint(10, vocab - 10, (args.input_len,), generator=gen).tolist() for _ in range(args.num_requests)]
    t0 = time.time()
    with torch.cuda.stream(stream):
        for i, ids in enumerate(prompts):
            runner.add_request(f"r{i}", ids, max_tokens=args.output_len, min_tokens=args.output_len)
        tokens, steps, blocked = {}, 0, 0
        while runner.has_unfinished_requests():
            outs = runner.step()
            steps += 1
            if not outs:
                outs = runner.get_prefill_result(isBlock=True)
                blocked += 1
            for ro in outs:
                if ro.finished:
                    tokens[ro.request_id] = list(ro.outputs[0].token_ids)
        runner.shutdown()
    wall = time.time() - t0
    free_after = runner.model_runner.token_to_kv_pool_allocator.available_size()
    print(f"[{args.mode}] {len(tokens)}/{args.num_requests} finished in {wall:.2f}s, steps={steps} "
          f"blocked_waits={blocked} prefill_steps={runner.num_prefill_steps} decode_steps={runner.num_decode_steps} "
          f"kv_free {free_before}->{free_after} reserved={runner.reserved_tokens}", flush=True)
    ok = len(tokens) == args.num_requests and free_after == free_before and runner.reserved_tokens == 0
    ok = ok and all(len(t) == args.output_len for t in tokens.values())
    with open(args.out, "w") as f:
        json.dump({"mode": args.mode, "tokens": tokens, "wall": wall}, f)
    print(f"[{args.mode}] bookkeeping {'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    ref = runs[0]
    status = 0
    for other in runs[1:]:
        same_seq = same_tok = total = 0
        first_div = []
        for rid, a in ref["tokens"].items():
            b = other["tokens"].get(rid, [])
            total += len(a)
            prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            same_tok += prefix
            same_seq += a == b
            if a != b:
                first_div.append(prefix)
        print(f"[compare] {ref['mode']} vs {other['mode']}: identical sequences {same_seq}/{len(ref['tokens'])}, "
              f"matching prefix tokens {same_tok}/{total}, first divergence at {sorted(first_div)[:8]}", flush=True)
        if same_seq != len(ref["tokens"]):
            status = 1
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--mode", choices=("serial", "thread", "chunked"))
    p.add_argument("--out")
    p.add_argument("--num-requests", type=int, default=12)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--chunk-budget", type=int, default=600)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--compare", nargs="+")
    args = p.parse_args()
    if args.compare:
        sys.exit(compare(args.compare))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
