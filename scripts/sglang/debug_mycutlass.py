"""Standalone libmycutlass check: one GEMM shape per subprocess (a fault is sticky).

  python scripts/sglang/debug_mycutlass.py            # all shapes
  python scripts/sglang/debug_mycutlass.py N K T mode # one case (mode: linear | hgemm)
"""

import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "microbench", "python"))

SHAPES = [(256, 256), (2560, 2560), (6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728), (151936, 2560)]
TOKENS = [1, 7, 64, 512]


def one(n, k, t, mode):
    from mycutlass import hgemm
    from mycutlass.binding import myCutlassHgemm

    torch.manual_seed(0)
    w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.05
    x = torch.randn(t, k, dtype=torch.float16, device="cuda")
    ref = x.float() @ w.float().t()
    if mode == "hgemm":  # the package's documented row-major wrapper: (T,K) @ (K,N)
        y = hgemm(x, w.t().contiguous())
    else:  # runner layout: column-major y^T[N,T] = W[N,K] @ x^T
        wt = w.t().contiguous()
        y = torch.empty(t, n, dtype=torch.float16, device="cuda")
        myCutlassHgemm(n, t, k, wt.data_ptr(), n, x.data_ptr(), x.stride(0), y.data_ptr(), n, 1.0, 0.0,
                       torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    err = (y.float() - ref).abs().max().item()
    print(f"OK N={n} K={k} T={t} mode={mode} max_abs_err={err:.4f} ref_absmax={ref.abs().max().item():.2f}", flush=True)


def main():
    if len(sys.argv) == 5:
        one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        return
    for mode in ("hgemm", "linear"):
        for n, k in SHAPES:
            for t in TOKENS:
                r = subprocess.run([sys.executable, __file__, str(n), str(k), str(t), mode],
                                   capture_output=True, text=True, timeout=300)
                out = (r.stdout + r.stderr).strip().splitlines()
                line = next((l for l in out if l.startswith("OK")), None)
                if line:
                    print(line, flush=True)
                else:
                    err = next((l for l in reversed(out) if "rror" in l or "terminate" in l), out[-1] if out else "")
                    print(f"FAIL N={n} K={k} T={t} mode={mode} rc={r.returncode}: {err[:200]}", flush=True)


if __name__ == "__main__":
    main()
