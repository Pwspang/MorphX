#!/usr/bin/env python3
"""GPU-time census of an nsys trace: how much kernel time sits above a duration threshold.

For choosing MorphX's short-kernel threshold (MORPHX_SHORT_KERNEL_US): the paper reports
that 5 of 26 kernels cover 85% of GPU time. Reads the CUPTI kernel table of one or more
`nsys export -t sqlite` files and prints, per threshold T:
  share of total kernel time in launches with duration >= T, share of launches, and the
  number of distinct kernels (name + grid) with at least one launch >= T;
the T at which launches >= T cover --target (default 0.8) of GPU time; and the kernels
(name + grid, by median duration) just above and below that T.

  python -I scripts/analysis/kernel_time_census.py a.sqlite [b.sqlite ...] [--target 0.8]
"""

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict


def short(name, n=70):
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= n else name[: n - 3] + "..."


def load(path, skip_before=None):
    """Kernel launches (us, short name, demangled name, grid); with skip_before, drop every
    launch before the first one whose name contains it (model loading, warm-up)."""
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = []
    for start, end, sn, dn, gx, gy, gz in db.execute(
            "SELECT start, end, shortName, demangledName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "ORDER BY start"):
        rows.append(((end - start) / 1e3, strings.get(sn, "?"), strings.get(dn, "?"), gx * gy * gz))
    if skip_before:
        first = next((i for i, r in enumerate(rows) if skip_before in r[1] or skip_before in r[2]), 0)
        print(f"skipped {first} launches before the first '{skip_before}'")
        rows = rows[first:]
    return rows


def census(rows, target, thresholds):
    durs = sorted((d for d, *_ in rows), reverse=True)
    total = sum(durs)
    n = len(durs)
    # T for the target share: walk launches from the longest down
    acc, t_star = 0.0, 0.0
    for d in durs:
        acc += d
        if acc >= target * total:
            t_star = d
            break
    keys = defaultdict(list)
    for d, sn, dn, grid in rows:
        keys[(sn, grid)].append(d)

    def at(t):
        sel = [d for d in durs if d >= t]
        kinds = sum(1 for v in keys.values() if max(v) >= t)
        names = len({k[0] for k, v in keys.items() if max(v) >= t})
        return sum(sel) / total, len(sel) / n, kinds, names

    print(f"launches {n}, total kernel time {total / 1e3:.1f} ms, distinct kernels (name+grid) {len(keys)}, "
          f"names {len({k[0] for k in keys})}")
    print(f"T for {target:.0%} of GPU time: {t_star:.1f} us")
    print(f"| T (us) | GPU time >= T | launches >= T | kernels (name+grid) >= T | names >= T |")
    print("|---:|---:|---:|---:|---:|")
    for t in sorted(set(thresholds + [round(t_star, 1)]), reverse=True):
        g, l, k, nm = at(t)
        mark = " (target)" if abs(t - round(t_star, 1)) < 1e-9 else ""
        print(f"| {t:g}{mark} | {g:.1%} | {l:.1%} | {k} | {nm} |")
    # kernels around T, by median duration
    med = sorted(((statistics.median(v), len(v), sum(v), k) for k, v in keys.items()), key=lambda x: x[0])
    below = [m for m in med if m[0] < t_star][-6:]
    above = [m for m in med if m[0] >= t_star][:6]
    print(f"kernels just below T (median us, launches, share of GPU time, grid, name):")
    for m, c, s, (sn, grid) in below:
        print(f"   {m:8.1f} {c:6d} {s / total:6.1%} grid={grid:<6d} {short(sn)}")
    print(f"kernels just above T:")
    for m, c, s, (sn, grid) in above:
        print(f"   {m:8.1f} {c:6d} {s / total:6.1%} grid={grid:<6d} {short(sn)}")
    print("top kernels by GPU time:")
    for m, c, s, (sn, grid) in sorted(med, key=lambda x: -x[2])[:12]:
        print(f"   {s / total:6.1%} median {m:8.1f} us x{c:<6d} grid={grid:<6d} {short(sn)}")
    return t_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--thresholds", default="100,50,30,20,15,10,5,2")
    ap.add_argument("--skip-before", help="drop launches before the first kernel whose name contains this")
    args = ap.parse_args()
    ths = [float(x) for x in args.thresholds.split(",")]
    allrows = []
    for p in args.sqlite:
        print(f"== {p}")
        rows = load(p, args.skip_before)
        allrows += rows
        census(rows, args.target, ths)
    if len(args.sqlite) > 1:
        print("== all traces together")
        census(allrows, args.target, ths)


if __name__ == "__main__":
    main()
