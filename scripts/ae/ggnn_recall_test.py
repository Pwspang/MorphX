"""Check the pass-built GGNN Python module: recall@k against brute force.

Runs a small random dataset through ggnn.GGNN (build + query on a side
stream, as scripts/overhead/single_request.py does) and compares the result
with an exact torch top-k. Run it with and without the MorphX runtime
preloaded; recall must not change.

    python scripts/ae/ggnn_recall_test.py --base 100000 --query 1000
"""
import argparse
import time

import torch

import ggnn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, default=100_000)
    parser.add_argument("--query", type=int, default=1_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.64, 1.0, 2.0])
    parser.add_argument("--max-iter", type=int, default=2000)
    # GGNN's brute-force query launches on the default stream, which the MorphX
    # runtime rejects ("Default stream is not supported"); skip it under the runtime.
    parser.add_argument("--skip-bf", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        base = torch.rand((args.base, args.dim), dtype=torch.float32, device=device)
        query = torch.rand((args.query, args.dim), dtype=torch.float32, device=device)
        index = ggnn.GGNN()
        index.set_base(base)
        index.set_return_results_on_gpu(True)
        t0 = time.perf_counter()
        index.build(k_build=24, tau_build=0.5, measure=ggnn.DistanceMeasure.Euclidean)
        torch.cuda.synchronize()
        build_s = time.perf_counter() - t0
        latencies = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            indices, dists = index.query(query, args.k, 0.64, 400, ggnn.DistanceMeasure.Euclidean)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    torch.cuda.synchronize()

    exact = torch.cdist(query, base).topk(args.k, largest=False).indices

    def recall_of(ids) -> float:
        got = torch.as_tensor(ids, device=device).long()
        hits = sum(len(set(exact[i].tolist()) & set(got[i].tolist())) for i in range(args.query))
        return hits / (args.query * args.k)

    recall = recall_of(indices)
    print(f"[ggnn] base={args.base} query={args.query} dim={args.dim} k={args.k} "
          f"build={build_s:.2f}s query_ms(min/median)={min(latencies):.3f}/{sorted(latencies)[len(latencies) // 2]:.3f} "
          f"recall@{args.k}(tau=0.64)={recall:.4f}")

    # Uniform random 128-d data is hard for graph search: with the workloads'
    # max_iterations=400 recall is ~0.64, so that alone is not a correctness
    # check. GGNN's brute-force kernel must match torch exactly, and graph
    # recall with a longer search (--max-iter) must be high.
    with torch.cuda.stream(stream):
        bf_ids = None if args.skip_bf else index.bf_query(query, args.k, ggnn.DistanceMeasure.Euclidean)[0]
        sweep = {}
        for tau in args.taus:
            ids, _ = index.query(query, args.k, tau, args.max_iter, ggnn.DistanceMeasure.Euclidean)
            sweep[tau] = recall_of(ids)
    torch.cuda.synchronize()
    bf_recall = 1.0 if bf_ids is None else recall_of(bf_ids)
    print(f"[ggnn] bf_query recall@{args.k}={'skipped' if bf_ids is None else f'{bf_recall:.4f}'}; graph recall by tau: "
          + " ".join(f"{tau}:{r:.4f}" for tau, r in sweep.items()))
    ok = bf_recall >= 0.99 and sweep[max(args.taus)] >= args.min_recall
    print(f"[ggnn] RESULT: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
