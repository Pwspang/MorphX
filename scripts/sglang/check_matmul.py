#!/usr/bin/env python3
"""mycutlass_gemm.matmul vs torch.matmul for the Fig 11/15 GEMM/GEMV shapes."""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mycutlass_gemm  # noqa: E402


def bench(fn, a, b, iters=10):
    fn(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        c = fn(a, b)
    torch.cuda.synchronize()
    return c, (time.perf_counter() - start) * 1000.0 / iters


def main():
    torch.manual_seed(0)
    for m, n, k in ((8192, 8192, 8192), (8192, 16, 8192), (16, 8192, 8192), (1000, 8192, 4096)):
        a = torch.randn((m, k), dtype=torch.float16, device="cuda") / 16
        b = torch.randn((k, n), dtype=torch.float16, device="cuda") / 16
        ref = a.float() @ b.float()
        c_t, t_t = bench(torch.matmul, a, b)
        c_m, t_m = bench(mycutlass_gemm.matmul, a, b)
        err_t = (c_t.float() - ref).abs().max().item()
        err_m = (c_m.float() - ref).abs().max().item()
        ok = err_m <= max(2 * err_t, 1e-2)
        print(f"[matmul] M={m} N={n} K={k}: torch {t_t:.3f} ms err {err_t:.4f} | mycutlass {t_m:.3f} ms err {err_m:.4f} "
              f"-> {'OK' if ok else 'MISMATCH'}")
    print(f"[matmul] stats {mycutlass_gemm.stats()}")


if __name__ == "__main__":
    main()
