"""Model-free check of the FlashInfer kernels SGLang uses (Qwen3-8B shapes, fp16).

Builds (JIT) and runs, the way SGLang's flashinfer backend calls them:
  ragged prefill (causal), paged prefill/extend with a prefix, paged decode with
  tensor cores (one query per request, long KV: split-KV + merge-states), merge_state.
Each output is compared with an fp32 torch reference and saved, so two configs
(stock vs MORPHX_FLASHINFER_PASS=1, with or without the MorphX runtime) can be
diffed with --compare. If a JIT build fails, every module's build.ninja is re-run
with `ninja -k 0` and the distinct compiler errors are listed (all at once).

  MORPHX_FLASHINFER_PASS=1 python scripts/sglang/fi_kernel_test.py --out pass.pt
  python scripts/sglang/fi_kernel_test.py --compare stock.pt pass.pt
"""

import argparse
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashinfer_pass  # noqa: E402

HQ, HKV, D = 32, 8, 128  # Qwen3-8B


def ninja_errors():
    from flashinfer.jit import env as jit_env

    errs = {}
    for nf in sorted(glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "*" / "build.ninja"))):
        d = os.path.dirname(nf)
        r = subprocess.run(["ninja", "-k", "0", "-C", d, "-f", nf], capture_output=True, text=True)
        print(f"[fi-test] ninja -k 0 {os.path.basename(d)[:90]}: rc={r.returncode}", flush=True)
        for line in (r.stdout + r.stderr).splitlines():
            m = re.search(r"([^\s/]+\.(?:cuh|cu|h|hpp|inc)):(\d+):\d+: error: (.*)", line)
            if m:
                errs.setdefault((m.group(1), m.group(2), m.group(3)), 0)
                errs[(m.group(1), m.group(2), m.group(3))] += 1
            elif "error:" in line and "too many errors" not in line:
                errs.setdefault(("?", "?", line.strip()[:200]), 0)
    print(f"[fi-test] {len(errs)} distinct compile errors:", flush=True)
    for (f, l, msg), n in sorted(errs.items()):
        print(f"[fi-err] {f}:{l}: {msg}  (x{n})", flush=True)


def ref_attn(q, k, v, causal, q_offset=0):
    import torch

    # q [Lq, HQ, D], k/v [Lk, HKV, D] -> [Lq, HQ, D]; GQA; causal with q at positions q_offset..
    g = HQ // HKV
    k = k.float().repeat_interleave(g, dim=1)
    v = v.float().repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), k) / D ** 0.5
    if causal:
        lq, lk = q.shape[0], k.shape[0]
        mask = torch.arange(lk, device=q.device)[None, :] > (torch.arange(lq, device=q.device)[:, None] + q_offset)
        s = s.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v)


def run_tests(stream):
    import torch
    import flashinfer
    from flashinfer.cascade import merge_state

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    rnd = lambda *s: (torch.randn(*s, generator=g, device=dev) * 0.5).half()  # noqa: E731
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    out, errs = {}, {}

    # 1. ragged prefill, causal: 3 requests (SGLang's extend without prefix)
    lens = [700, 1500, 64]
    qo = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    q, k, v = rnd(sum(lens), HQ, D), rnd(sum(lens), HKV, D), rnd(sum(lens), HKV, D)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
    w.plan(qo, qo, HQ, HKV, D, causal=True, q_data_type=torch.float16)
    print(f"[fi-test] ragged prefill backend: {getattr(w, '_backend', '?')}", flush=True)
    o = w.run(q, k, v)
    ref = torch.cat([ref_attn(q[a:b], k[a:b], v[a:b], True) for a, b in zip(qo[:-1].tolist(), qo[1:].tolist())])
    out["ragged"], errs["ragged"] = o, (o.float() - ref).abs().max().item()

    # paged KV cache, page size 1 (SGLang's token-granular pool)
    def paged(kv_lens):
        n = sum(kv_lens)
        kc, vc = rnd(n + 16, HKV, D), rnd(n + 16, HKV, D)
        perm = torch.randperm(n + 16, generator=g, device=dev)[:n].int()
        ind = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32, device=dev)
        last = torch.ones(len(kv_lens), dtype=torch.int32, device=dev)
        return kc, vc, perm, ind, last

    # 2. paged prefill with a prefix (extend: 2 requests, prefix 1000/300, new 200/50)
    kv_lens, q_lens = [1200, 350], [200, 50]
    kc, vc, perm, ind, last = paged(kv_lens)
    qo = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev)
    q = rnd(sum(q_lens), HQ, D)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(qo, ind, perm, last, HQ, HKV, D, 1, causal=True, q_data_type=torch.float16)
    o = w.run(q, (kc.unsqueeze(1), vc.unsqueeze(1)))
    refs = []
    for i in range(2):
        idx = perm[ind[i]:ind[i + 1]].long()
        refs.append(ref_attn(q[qo[i]:qo[i + 1]], kc[idx], vc[idx], True, q_offset=kv_lens[i] - q_lens[i]))
    out["paged_extend"], errs["paged_extend"] = o, (o.float() - torch.cat(refs)).abs().max().item()

    # 3. decode with tensor cores (SGLang: BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)),
    #    few long requests -> split-KV + PersistentVariableLengthMergeStates
    kv_lens = [4000, 3000, 2500, 100]
    kc, vc, perm, ind, last = paged(kv_lens)
    q = rnd(len(kv_lens), HQ, D)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
    w.plan(ind, perm, last, HQ, HKV, D, 1, q_data_type=torch.float16)
    o = w.run(q, (kc.unsqueeze(1), vc.unsqueeze(1)))
    refs = []
    for i in range(len(kv_lens)):
        idx = perm[ind[i]:ind[i + 1]].long()
        refs.append(ref_attn(q[i:i + 1], kc[idx], vc[idx], False))
    out["decode"], errs["decode"] = o, (o.float() - torch.cat(refs)).abs().max().item()

    # 4. merge_state (cascade module)
    va, vb = rnd(64, HQ, D), rnd(64, HQ, D)
    sa, sb = torch.randn(64, HQ, device=dev, generator=g), torch.randn(64, HQ, device=dev, generator=g)
    vm, _ = merge_state(va, sa, vb, sb)
    wa = torch.sigmoid((sa - sb) * 0.6931471805599453)[..., None]  # lse in log2 units
    out["merge"], errs["merge"] = vm, (vm.float() - (wa * va.float() + (1 - wa) * vb.float())).abs().max().item()
    torch.cuda.synchronize()
    return out, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    import torch

    if args.compare:
        a, b = torch.load(args.compare[0]), torch.load(args.compare[1])
        same = True
        for key in a["out"]:
            d = (a["out"][key].float() - b["out"][key].float()).abs().max().item()
            same &= d == 0
            print(f"[fi-cmp] {key}: max_abs_diff={d:.6f} ({a['cfg']} vs {b['cfg']})")
        print("[fi-cmp] " + ("BITWISE IDENTICAL" if same else "DIFFERENT"))
        return 0

    flashinfer_pass.setup_env()
    flashinfer_pass.patch_backends()
    from flashinfer.jit import env as jit_env

    cfg = flashinfer_pass.config_name() + \
        ("-runtime" if "libcuda.so" in os.environ.get("LD_PRELOAD", "") else "")
    print(f"[fi-test] config={cfg} jit_dir={jit_env.FLASHINFER_JIT_DIR}", flush=True)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            out, errs = run_tests(stream)
    except Exception as e:  # noqa: BLE001
        print(f"[fi-test] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        if "Ninja build failed" in str(e):
            ninja_errors()
        sys.stdout.flush()
        os._exit(1)
    ok = True
    for key, e in errs.items():
        good = e < 2e-2
        ok &= good
        print(f"[fi-test] {key}: max_abs_err_vs_fp32_ref={e:.5f} {'OK' if good else 'BAD'}", flush=True)
    if args.out:
        torch.save({"cfg": cfg, "out": {k: v.cpu() for k, v in out.items()}}, args.out)
    print("[fi-test] " + ("PASS" if ok else "FAIL"), flush=True)
    flashinfer_pass.runtime_report()
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
