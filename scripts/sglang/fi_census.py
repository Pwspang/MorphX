"""Fig 13 kernel census: which kernels MorphX can control and how much GPU time they cover.

Joins two runs of the same Fig 13 point (scripts/sglang/fi_pass_hpcc.pbs, part census):
  stream-<fi>.sqlite  nsys trace of the stream variant (no runtime): GPU time per kernel
  morphx-<fi>.err     MorphX run with SMSCHED_PROFILE_LOG=1: one profile line per
                      pass-built kernel (first launch): first-wave time, CI
for fi in (stock, pass). A kernel is pass-built if the runtime profiled it (it
has a `_pk` clone); it is managed if its first wave is >= 50 us (shorter ones
take the runtime's short-kernel bypass).

  python scripts/sglang/fi_census.py <census dir>
"""

import collections
import os
import re
import sqlite3
import subprocess
import sys

PROFILE_RE = re.compile(r"\[MorphX-profile\] name=(\S+) stream=\S+ grid=(\S+) occup=(\d+) waves=(\S+) "
                        r"ci_legacy=(\S+) ci_occ=(\S+) ci_raw=(\S+) ci=(\S+) ci_kernel=(\d)")
SHORT_US = 50.0


def demangle(names):
    try:
        out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
        return dict(zip(names, out.splitlines()))
    except (OSError, subprocess.CalledProcessError):
        return {n: n for n in names}


def kernel_times(path):
    db = sqlite3.connect(path)
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "mangledName" if "mangledName" in cols else "demangledName"
    rows = db.execute(f"SELECT s.value, SUM(k.end - k.start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                      f"JOIN StringIds s ON s.id = k.{name_col} GROUP BY s.value").fetchall()
    return {name: (ns / 1e3, n) for name, ns, n in rows}  # us, launches


def profiles(path):
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = PROFILE_RE.search(line)
            if m:
                name, grid, occup, waves, cil, cio, cir, ci, cik = m.groups()
                first = float(waves.split(",")[0].split(":")[1])
                out[name] = dict(grid=grid, occup=int(occup), first_us=first, waves=waves, ci=float(ci),
                                 ci_kernel=int(cik))
    return out


def origin(name, demangled):
    d = demangled.lower()
    if name.startswith("_ZN7cutlass") or "cutlass::Kernel<" in demangled:
        return "mycutlass"
    if name.startswith("kernel_cutlass_kernel"):
        return "FlashInfer CuTe DSL"
    if "flashinfer" in d:
        return "FlashInfer"
    if "at::native" in d or "at_cuda_detail" in d:
        return "PyTorch"
    if "anonymous namespace" in d or "sglang" in d or "sgl_kernel" in d:
        return "SGLang jit/sgl-kernel"
    return "other"


def short(demangled, n=70):
    s = re.sub(r"^void ", "", demangled)
    s = re.sub(r"<.*", "<...>", s) if len(s) > n else s
    return s[:n]


def report(d, fi):
    sq, err = os.path.join(d, f"stream-{fi}.sqlite"), os.path.join(d, f"morphx-{fi}.err")
    if not (os.path.exists(sq) and os.path.exists(err)):
        print(f"## {fi}: missing {sq} or {err}\n")
        return
    times = kernel_times(sq)
    prof = profiles(err)
    dem = demangle(list(times))
    total = sum(t for t, _ in times.values())
    # Profile names are the runtime's kernel name truncated to 120 chars.
    match = {}
    for name in times:
        for pname in prof:
            if name.startswith(pname) or name == pname:
                match[name] = pname
                break
    by_origin = collections.Counter()
    cov = collections.Counter()
    rows = []
    for name, (us, n) in times.items():
        o = origin(name, dem[name])
        by_origin[o] += us
        p = prof.get(match.get(name, ""), None)
        if p:
            cov["pass-built"] += us
            if p["first_us"] >= SHORT_US:
                cov["managed (first wave >= 50 us)"] += us
                if p["ci_kernel"]:
                    cov["CI (confined to [0, split))"] += us
        rows.append((us, n, name, o, p))
    rows.sort(reverse=True)
    built = [r for r in rows if r[4]]
    print(f"## FlashInfer {fi}: {len(times)} kernel names, {total / 1e6:.2f} s GPU kernel time; "
          f"{len(prof)} pass-built kernels profiled by the runtime, {len(built)} matched in the trace, "
          f"{sum(1 for r in built if r[4]['first_us'] >= SHORT_US)} with first wave >= 50 us\n")
    print("| GPU time by origin | % |\n|---|---:|")
    for o, us in by_origin.most_common():
        print(f"| {o} | {us / total * 100:.1f} |")
    for k in ("pass-built", "managed (first wave >= 50 us)", "CI (confined to [0, split))"):
        print(f"| **{k}** | **{cov[k] / total * 100:.1f}** |")
    print("\n| Kernel | Origin | Launches | % GPU | Pass-built | First wave us | CI | >= 50 us |")
    print("|---|---|---:|---:|---|---:|---:|---|")
    for us, n, name, o, p in rows:
        if us / total < 0.002 and not p:
            continue
        pb = "yes" if p else "no"
        fw = f"{p['first_us']:.1f}" if p else ""
        ci = f"{p['ci']:.2f}" if p else ""
        ok = ("yes" if p["first_us"] >= SHORT_US else "no") if p else ""
        print(f"| `{short(dem[name])}` | {o} | {n} | {us / total * 100:.1f} | {pb} | {fw} | {ci} | {ok} |")
    unmatched = [p for p in prof if p not in match.values()]
    if unmatched:
        print(f"\nProfiled but not in the stream trace ({len(unmatched)}): "
              + ", ".join(f"`{demangle([u])[u][:60]}` ({prof[u]['first_us']:.0f} us)" for u in unmatched))
    print()


def main():
    d = sys.argv[1]
    for fi in ("stock", "pass"):
        report(d, fi)


if __name__ == "__main__":
    main()
