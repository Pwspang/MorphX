"""Which libmycutlass did this process load, which kernels ran, and how accurate are they?

  MYCUTLASS_LIB_DIR=<dir> python scripts/sglang/mycutlass_libcheck.py --out a.pt
  python scripts/sglang/mycutlass_libcheck.py --compare a.pt b.pt

Prints the resolved library path (from the process's own /proc/self/maps), its
size and sha1, every CUDA kernel the GEMMs launched (torch.profiler / CUPTI), and
per-shape max |err| vs an fp32 reference, on the Qwen3-8B linear-layer shapes
(y[T, N] = x[T, K] @ W[N, K]^T, the call mycutlass_gemm.linear makes). --compare
reports whether two libraries' outputs are bitwise identical, shape by shape.
"""

import argparse
import hashlib
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "microbench", "python"))

SHAPES = [(4096, 4096), (6144, 4096), (24576, 4096), (4096, 12288), (151936, 4096)]  # (N, K): o, qkv, gate_up, down, lm_head
TOKENS = [1, 16, 128, 2048, 8192]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = torch.load(args.compare[0]), torch.load(args.compare[1])
        print(f"[lib-cmp] {a['lib']}\n[lib-cmp]   vs {b['lib']}")
        same_all = True
        for key in a["y"]:
            d = (a["y"][key].float() - b["y"][key].float())
            same = bool((a["y"][key] == b["y"][key]).all())
            same_all &= same
            print(f"[lib-cmp] N,K,T={key}: bitwise_equal={same} max_abs_diff={d.abs().max().item():.5f} "
                  f"frac_elems_differ={(d != 0).float().mean().item():.4f}")
        print("[lib-cmp] " + ("ALL BITWISE IDENTICAL" if same_all else "OUTPUTS DIFFER"))
        return

    from mycutlass.binding import myCutlassHgemm  # loads the library

    libs = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libmycutlass" in line})
    for path in libs:
        data = open(path, "rb").read()
        print(f"[lib] loaded {path} size={len(data)} sha1={hashlib.sha1(data).hexdigest()[:16]}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    out = {"lib": ",".join(libs), "y": {}}
    from torch.profiler import ProfilerActivity, profile

    stream = torch.cuda.current_stream().cuda_stream
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for n, k in SHAPES:
            w = (torch.randn(n, k, generator=g, device="cuda") * 0.05).half()
            wt = w.t().contiguous()
            for t in TOKENS:
                if n * t > 151936 * 2048:
                    continue
                x = (torch.randn(t, k, generator=g, device="cuda") * 0.5).half()
                y = torch.empty(t, n, dtype=torch.float16, device="cuda")
                myCutlassHgemm(n, t, k, wt.data_ptr(), n, x.data_ptr(), x.stride(0), y.data_ptr(), n, 1.0, 0.0, stream)
                ref = x.float() @ w.float().t()
                ref16 = (x @ w.t()).float()  # cuBLAS fp16, for scale
                torch.cuda.synchronize()
                err = (y.float() - ref).abs().max().item()
                err_cublas = (ref16 - ref).abs().max().item()
                print(f"[lib] N={n:6d} K={k:5d} T={t:4d}: max|mycutlass-fp32|={err:.4f} max|cublas-fp32|={err_cublas:.4f}",
                      flush=True)
                out["y"][f"{n},{k},{t}"] = y.cpu()
    names = {}
    for e in prof.key_averages():
        if e.device_type is not None and "DeviceType.CUDA" in str(e.device_type):
            names[e.key] = e.count
    for name, cnt in sorted(names.items(), key=lambda kv: -kv[1]):
        if "gemm" in name.lower() or "cutlass" in name.lower() or "Kernel" in name:
            print(f"[lib] kernel x{cnt}: {name[:180]}", flush=True)
    if args.out:
        torch.save(out, args.out)


if __name__ == "__main__":
    main()
