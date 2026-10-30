#!/usr/bin/env python3
"""Plot the H200 Fig 13 follow-up runs that the AE plotter doesn't cover.

  fig13_sm90_sweep      stream vs MorphX + per-grid against request rate, sm90 mycutlass + pass FA2
  fig13_setups          MorphX / stream latency ratio at the standard points, per kernel/attention setup
  pd_decode_priority    decode-step breakdown (nsys) and TBT with decode on a high-priority stream
  a100_fig13            A100 control (nscc): stream / MorphX / MorphX occ / single stream vs request rate
  a100_vs_paper         MorphX / stream latency ratio on the A100, ours vs the paper's reference logs
  a100_step_bench       A100 single-stream prefill and decode step time with and without the runtime
  pd_runner_fix         decode-step breakdown and TBT before / after the runner stall fix
  figure13_fa3, figure14_fa3  Fig 13/14 with the default setup: baseline, stream, stream + high-priority
                        decode, MorphX at 50 us (first wave) and at 35 us (whole kernel)

Numbers come from the job logs under logs/2026102{7,8}-*; the job IDs are in the comments.
Usage: plot_pd_followups.py [--out logs/20261028-figures]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"

# Reference categorical palette, fixed order (dataviz skill, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})

# ---------------------------------------------------------------- sm90 sweep
# (rps, avg, p99, served fraction, job). Stream vs MorphX + MORPHX_PROFILE_PER_GRID=1, sm90
# mycutlass + pass-built FA2, split 108, 60 s. 192442 = n_fig13_fi_sm90.pbs (gpu092); the
# rest are n_fig13_sweep_sm90.pbs. 192557 (4096/128 on gpu131, erratic) is left out.
SWEEP = {
    "in-4096-out-128": {
        "stream": [(5, .022, .041, .99, 192658), (5, .022, .040, .99, 192663),
                   (6, .032, .056, .93, 192658), (6, .033, .058, .93, 192663), (6, .034, .060, .93, 192442),
                   (7.5, .110, .137, .73, 192658), (7.5, .115, .142, .72, 192663)],
        "MorphX": [(5, .023, .042, .99, 192658), (5, .035, .099, .99, 192663),
                   (6, .035, .061, .93, 192658), (6, .039, .067, .93, 192663), (6, .037, .064, .93, 192442),
                   (7.5, .123, .148, .71, 192658), (7.5, .123, .149, .71, 192663)],
    },
    "in-4096-out-256": {
        "stream": [(5, .027, .038, .94, 192558), (6, .054, .067, .79, 192442), (6, .040, .216, .03, 192661),
                   (7.5, .177, .204, .13, 192558)],
        "MorphX": [(5, .029, .040, .93, 192558), (6, .063, .082, .78, 192442), (6, .041, .211, .03, 192661),
                   (7.5, .109, .161, .30, 192558)],
    },
    "in-2048-out-256": {
        "stream": [(6, .012, .025, .93, 192556), (9, .017, .024, .91, 192442), (12, .038, .047, .77, 192556),
                   (15, .056, .078, .63, 192659), (18, .066, .104, .53, 192556)],
        "MorphX": [(6, .012, .015, .93, 192556), (9, .020, .028, .90, 192442), (12, .044, .051, .76, 192556),
                   (15, .059, .080, .60, 192659), (18, .070, .101, .50, 192556)],
    },
}
COLLAPSED = 0.5  # served fraction below this: overloaded / bistable, drawn hollow and off the line


def plot_sweep(out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
    for col, (cfg, series) in enumerate(SWEEP.items()):
        for row, (idx, label) in enumerate(((1, "Avg latency per output token (s)"),
                                            (2, "P99 latency per output token (s)"))):
            ax = axes[row, col]
            for color, (name, pts) in zip(SERIES, series.items()):
                ok = [p for p in pts if p[3] >= COLLAPSED]
                bad = [p for p in pts if p[3] < COLLAPSED]
                rates = sorted({p[0] for p in ok})
                mean = [sum(p[idx] for p in ok if p[0] == r) / sum(1 for p in ok if p[0] == r) for r in rates]
                ax.plot(rates, mean, color=color, lw=2, label=name, zorder=2)
                ax.scatter([p[0] for p in ok], [p[idx] for p in ok], s=28, color=color,
                           edgecolor=SURFACE, linewidth=1.5, zorder=3)
                if bad:
                    ax.scatter([p[0] for p in bad], [p[idx] for p in bad], s=34, facecolor=SURFACE,
                               edgecolor=color, linewidth=1.6, zorder=3,
                               label=f"{name}, served < 50%" if col == 0 and row == 0 else None)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(cfg)
            else:
                ax.set_xlabel("Request rate (req/s)")
            if col == 0:
                ax.set_ylabel(label)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Fig 13 on the H200, sm90 GEMMs + pass-built FA2: stream vs MorphX (per-grid, split 108)",
                 fontsize=10)
    fig.text(0.5, -0.02, "Each dot is one 60 s run (jobs 192442, 192556, 192558, 192658, 192659, 192661, 192663; "
             "Qwen3-8B fp16). Lines join the mean of runs that served at least 50% of requests.",
             ha="center", fontsize=8, color=INK2)
    save(fig, out / "fig13_sm90_sweep")


# ---------------------------------------------------------------- setups
# (stream avg, MorphX avg) s/token at the standard points, split 108, default CI, 60 s.
CONFIGS = ["in-4096-out-128\n6 req/s", "in-4096-out-256\n6 req/s", "in-2048-out-256\n9 req/s"]
SETUPS = [
    ("Sm80\nfp16\nstock FI", [(.067, .149), (.099, .119), (.041, .053)]),   # 190793, 190847, 190846
    ("Sm80\nbf16\nstock FI", [(.075, .162), (.100, .119), (.043, .054)]),   # 192360
    ("Sm80\nfp16\npass FA2", [(.117, .209), (.108, .128), (.048, .067)]),   # 192386
    ("Sm80\nfp16\npass FA2\n+ grid", [(.117, .193), (.108, .120), (.048, .067)]),  # 192386
    ("sm90\nfp16\npass FA2", [(.034, .037), (.054, .061), (.017, .018)]),   # 192442
    ("sm90\nfp16\npass FA2\n+ grid", [(.034, .037), (.054, .063), (.017, .020)]),  # 192442
]


def plot_setups(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, constrained_layout=True)
    x = range(len(SETUPS))
    for c, ax in enumerate(axes):
        ratios = [pts[c][1] / pts[c][0] for _, pts in SETUPS]
        ax.bar(x, ratios, width=0.6, color=SERIES[0], zorder=2)
        ax.axhline(1.0, color=INK, lw=1, zorder=3)
        for i, r in enumerate(ratios):
            ax.text(i, r + 0.03, f"+{(r - 1) * 100:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(list(x), [s for s, _ in SETUPS], fontsize=7)
        ax.set_title(CONFIGS[c].replace("\n", ", "))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MorphX / stream, avg latency per token\n(stream = 1.0)")
    axes[0].set_ylim(0, 2.4)
    fig.suptitle("Fig 13 on the H200: MorphX vs stream across kernel and attention setups (split 108)", fontsize=10)
    fig.text(0.5, -0.04, "Each setup's stream and MorphX ran in the same job; jobs: 190793/190846/190847, 192360, "
             "192386, 192442 (one node each, 60 s, Qwen3-8B). FI = FlashInfer (stock: FA3 prefill); grid = per-grid profiling.",
             ha="center", fontsize=8, color=INK2)
    save(fig, out / "fig13_setups")


# ---------------------------------------------------------------- decode priority (job 194603)
PRIO_DIR = LOGS / "20261028-pd-blocking" / "trace-194603" / "analysis"
TBT = [  # ms/token from the run logs, in-4096-out-128 6 req/s 30 s under nsys, gpu083
    ("Sm80", [("stream", 70.2), ("decode\nhigh prio", 73.2), ("MorphX\nno split", 76.0),
              ("MorphX\nsplit 108", 109.6)]),
    ("sm90", [("stream", 39.5), ("decode\nhigh prio", 39.8), ("MorphX\n+ grid\nsplit 108", 43.5)]),
]


def plot_priority(out: Path) -> None:
    runs = [("Sm80 stream", "sm80"), ("Sm80 decode\nhigh priority", "sm80-hipri"),
            ("sm90 stream", "sm90"), ("sm90 decode\nhigh priority", "sm90-hipri")]
    parts = [("Decode kernels", "kernel_busy_ms"), ("Waiting behind prefill", "blocked_ms"),
             ("Host gaps (waiting on CPU)", "host_gap_ms")]
    rows = []
    for label, tag in runs:
        m = json.loads((PRIO_DIR / f"stream-{tag}-summary.json").read_text())["decode"]["per_step_mean_ms"]
        vals = [m[k] for _, k in parts]
        rows.append((label, vals, m["wall_ms"] - sum(vals)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[1.1, 1], constrained_layout=True)
    y = list(range(len(rows)))[::-1]
    for yi, (label, vals, rest) in zip(y, rows):
        left = 0.0
        for color, v in zip(SERIES, vals + [rest]):
            a1.barh(yi, v, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
            left += v
        a1.text(left + 1, yi, f"{left:.1f} ms", va="center", fontsize=8, color=INK2)
    a1.set_yticks(y, [r[0] for r in rows])
    a1.set_xlabel("Mean decode step (ms), nsys")
    a1.set_xlim(0, 85)
    a1.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES]
    a1.legend(handles, [p for p, _ in parts] + ["Other (copies, launch latency)"], loc="lower right", fontsize=7.5)
    a1.set_title("Decode step breakdown: priority removes the wait, CPU time takes its place")

    labels, vals = [], []
    for kern, items in TBT:
        base = items[0][1]
        for name, v in items:
            labels.append(f"{kern}\n{name}")
            vals.append(v / base)
    a2.bar(range(len(vals)), vals, width=0.6, color=SERIES[0], zorder=2)
    a2.axhline(1.0, color=INK, lw=1, zorder=3)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.02, f"{(v - 1) * 100:+.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
    a2.set_xticks(range(len(vals)), labels, fontsize=7)
    a2.set_ylabel("TBT relative to stream (same GEMMs)")
    a2.set_ylim(0, 1.75)
    a2.grid(axis="x", visible=False)
    a2.set_title("Time between tokens")
    fig.suptitle("in-4096-out-128 at 6 req/s, 30 s, H200 (job 194603, one node, all runs under nsys)", fontsize=10)
    save(fig, out / "pd_decode_priority")


# ---------------------------------------------------------------- A100 control (nscc)
A100_DIR = LOGS / "20261028-a100-nscc"
A100_SUMMARIES = {"26036337": A100_DIR / "summary-sweep-26036337.md",
                  "26036911": A100_DIR / "summary-sweep2-26036911.md"}
CELL = re.compile(r"([\d.]+) / ([\d.]+) / ([\d.]+) \((\d+)/(\d+)\)")


def read_a100():
    """{config: {column: [(rps, avg, p99, ttft, served frac, job)]}} from the summary tables."""
    data = {}
    for job, path in A100_SUMMARIES.items():
        cfg = None
        header = []
        for line in path.read_text().splitlines():
            if line.startswith("### in-"):
                cfg = line[4:].strip()
                header = []
            elif cfg and line.startswith("| rps"):
                header = [h.strip() for h in line.strip("|").split("|")]
            elif cfg and header and line.startswith("| ") and not line.startswith("|---"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                rps = float(cells[0])
                for name, cell in zip(header[1:], cells[1:]):
                    m = CELL.fullmatch(cell)
                    if not m:
                        continue
                    if name.startswith("paper") and job != "26036337":
                        continue  # same reference numbers in both summaries
                    avg, p99, ttft, served, gen = (float(g) for g in m.groups())
                    data.setdefault(cfg, {}).setdefault(name, []).append((rps, avg, p99, ttft, served / gen, job))
    return data


A100_SERIES = [("stream", "stream"), ("MorphX", "MorphX"), ("MorphX occ", "MorphX, occ CI score"),
               ("baseline", "single stream (no prefill thread)")]


def plot_a100(out: Path) -> None:
    data = read_a100()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
    for col, (cfg, cols) in enumerate(data.items()):
        for row, (idx, label) in enumerate(((1, "Avg latency per output token (s)"),
                                            (2, "P99 latency per output token (s)"))):
            ax = axes[row, col]
            for color, (key, name) in zip(SERIES, A100_SERIES):
                pts = cols.get(key, [])
                if not pts:
                    continue
                rates = sorted({p[0] for p in pts})
                mean = [sum(p[idx] for p in pts if p[0] == r) / sum(1 for p in pts if p[0] == r) for r in rates]
                ax.plot(rates, mean, color=color, lw=2, label=name, zorder=2)
                ax.scatter([p[0] for p in pts], [p[idx] for p in pts], s=26, color=color,
                           edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.set_ylim(bottom=0)
            ax.set_xticks(sorted({p[0] for p in cols["stream"]}))
            if row == 0:
                ax.set_title(cfg)
            else:
                ax.set_xlabel("Request rate (req/s), paper's A100 rates")
            if col == 0:
                ax.set_ylabel(label)
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Fig 13 on the A100-40GB (nscc): our port, Qwen3-8B fp16, Sm80 GEMMs + pass-built FA2, memcpy backend",
                 fontsize=10)
    fig.text(0.5, -0.06, "60 s per point. Jobs 26036337 (stream, MorphX, single stream) and 26036911 (repeat of stream "
             "and MorphX, plus MorphX occ);\neach dot is one run, lines join the mean. Split 76 (2048 in) / 88 (4096 in), "
             "per-grid profiling.", ha="center", fontsize=7.5, color=INK2)
    save(fig, out / "a100_fig13")

    # MorphX / stream ratio: ours (both runs) vs the paper's reference logs (Llama-3-8B, vLLM, 180 s)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey="row", constrained_layout=True)
    for col, (cfg, cols) in enumerate(data.items()):
        for row, (idx, label) in enumerate(((1, "MorphX / stream, avg latency"), (2, "MorphX / stream, P99 latency"))):
            ax = axes[row, col]
            for key_s, key_m, color, name, ls, fill in (
                    ("stream", "MorphX", SERIES[0], "our port, A100", "-", True),
                    ("paper stream", "paper MorphX", INK2, "paper, A100", "--", False)):
                st = {(p[0], p[5]): p[idx] for p in cols[key_s]}
                pts = [(p[0], p[idx] / st[(p[0], p[5])]) for p in cols[key_m] if (p[0], p[5]) in st]
                rates = sorted({r for r, _ in pts})
                mean = [sum(v for r2, v in pts if r2 == r) / sum(1 for r2, _ in pts if r2 == r) for r in rates]
                ax.plot(rates, mean, color=color, lw=2, ls=ls, label=name, zorder=2)
                ax.scatter([r for r, _ in pts], [v for _, v in pts], s=28, zorder=3,
                           color=color if fill else SURFACE, edgecolor=color if not fill else SURFACE,
                           linewidth=1.5)
            ax.axhline(1.0, color=INK, lw=1, zorder=1)
            ax.set_xticks(sorted({p[0] for p in cols["stream"]}))
            if row == 0:
                ax.set_title(cfg)
            else:
                ax.set_xlabel("Request rate (req/s)")
            if col == 0:
                ax.set_ylabel(label + "\n(below 1 = MorphX faster)")
    for ax in axes.flat:
        ax.set_ylim(0.6, 1.8)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("A100: MorphX relative to stream, our port vs the paper", fontsize=10)
    fig.text(0.5, -0.02, "Ours: Qwen3-8B fp16 on SGLang, 60 s, memcpy backend (jobs 26036337, 26036911). "
             "Paper: logs/reference, Llama-3-8B on their vLLM fork, 180 s. Same request rates; "
             "our runs sit at lighter load (stream TTFT 0.2-5.6 s vs 0.7-29 s).",
             ha="center", fontsize=7.5, color=INK2)
    save(fig, out / "a100_vs_paper")


def plot_a100_bench(out: Path) -> None:
    # job 26036993: one stream, no co-location; median step times from its job log
    steps = {"prefill step\n(4 x 2048 tokens)": [540.0, 568.1, 716.7],
             "decode step\n(batch 16, ctx 2048+)": [21.39, 27.16, 28.85]}
    names = ["no runtime", "MorphX runtime,\nno split", "MorphX runtime,\nsplit 76"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for ax, (title, vals) in zip(axes, steps.items()):
        ax.bar(range(3), vals, width=0.6, color=SERIES[0], zorder=2)
        for i, v in enumerate(vals):
            tag = f"{v:.1f} ms" if i == 0 else f"{v:.1f} ms ({(v / vals[0] - 1) * 100:+.0f}%)"
            ax.text(i, v * 1.02, tag, ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(3), names, fontsize=8)
        ax.set_ylabel("Median step time (ms)")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(title.replace("\n", " "))
        ax.grid(axis="x", visible=False)
    fig.suptitle("A100: cost of the MorphX runtime with nothing co-running (job 26036993, memcpy backend)",
                 fontsize=10)
    save(fig, out / "a100_step_bench")


# ---------------------------------------------------------------- runner stall fix (before / after)
# Decode-step breakdown (ms per decode step, nsys) and TBT, in-4096-out-128 6 req/s 30 s, Sm80
# mycutlass + stock FlashInfer. Before: job 194603 (gpu083). After: job 195305 (the fixed runner,
# branch runner-stall-fix); numbers from its job log (nsys_pd_blocking.py summary).
FIX_RUNS = [  # label, kernels, waiting behind prefill, host gaps, wall, TBT ms/token
    ("before fix\nstream", 16.50, 21.98, 31.31, 70.14, 70.2),
    ("before fix\ndecode high prio", 16.22, 3.13, 52.73, 72.38, 73.2),
    ("after fix\nstream", 18.75, 39.06, 5.16, 63.28, 62.1),
    ("after fix\ndecode high prio", 18.89, 7.07, 1.64, 27.60, 28.3),
]


def plot_fix(out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[1.1, 1], constrained_layout=True)
    parts = ["Decode kernels", "Waiting behind prefill", "Host gaps (waiting on CPU)", "Other (copies, launch latency)"]
    y = list(range(len(FIX_RUNS)))[::-1]
    for yi, (label, k, b, h, wall, _) in zip(y, FIX_RUNS):
        left = 0.0
        for color, v in zip(SERIES, (k, b, h, max(0.0, wall - k - b - h))):
            a1.barh(yi, v, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
            left += v
        a1.text(left + 1, yi, f"{wall:.1f} ms", va="center", fontsize=8, color=INK2)
    a1.set_yticks(y, [r[0] for r in FIX_RUNS])
    a1.set_xlabel("Mean decode step (ms), nsys")
    a1.set_xlim(0, 85)
    a1.grid(axis="y", visible=False)
    a1.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES], parts, loc="lower right", fontsize=7.5)
    a1.set_title("Decode step: the fix removes the host stall")

    tbt = [r[5] for r in FIX_RUNS]
    a2.bar(range(len(tbt)), tbt, width=0.6, color=SERIES[0], zorder=2)
    for i, v in enumerate(tbt):
        a2.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    a2.set_xticks(range(len(tbt)), [r[0] for r in FIX_RUNS], fontsize=7.5)
    a2.set_ylabel("Time between tokens (ms)")
    a2.set_ylim(0, 85)
    a2.grid(axis="x", visible=False)
    a2.set_title("After the fix, prioritising decode cuts TBT 55%")
    fig.suptitle("Runner stall fix: in-4096-out-128 at 6 req/s, Sm80 GEMMs, stream only, under nsys "
                 "(before: job 194603, after: job 195305)", fontsize=10)
    save(fig, out / "pd_runner_fix")


# ---------------------------------------------------------------- Fig 13/14, default setup (sm90 + pass FA3)
# H200, fixed runner, 60 s per point. default50: baseline / stream / MorphX (first-wave 50 us bypass,
# jobs 196338-196348 even); tuned35: MorphX with whole-kernel bypass at 35 us (196396-196401);
# stream-hipri: stream with decode at priority -5 (196339-196349 odd).
FA3_DIR = LOGS / "20261028-fig1314-fa3"
FA3_SERIES = [  # (label, set dir, table column)
    ("baseline", "default50", "baseline"),
    ("stream", "default50", "stream"),
    ("stream, decode high priority", "stream-hipri", "stream"),
    ("MorphX (50 us, first wave)", "default50", "MorphX"),
    ("MorphX (35 us, whole kernel)", "tuned35", "MorphX"),
]
FA3_LENS = ["in-2048-out-256", "in-4096-out-128", "in-4096-out-256"]


def read_fa3(fig):
    """{len: {label: [(rps, avg, p99)]}} from each job's run/tables/<fig>.tsv."""
    import csv
    out = {}
    for ln in FA3_LENS:
        out[ln] = {}
        for label, sub, col in FA3_SERIES:
            pts = []
            for t in (FA3_DIR / sub / f"{fig}-{ln}" / "run" / "tables").glob("*.tsv"):
                for r in csv.DictReader(t.open(), delimiter="\t"):
                    if r.get(f"{col}(avg)"):
                        pts.append((float(r["rps"]), float(r[f"{col}(avg)"]), float(r[f"{col}(p99)"])))
            out[ln][label] = sorted(pts)
    return out


def plot_fig1314_fa3(out: Path) -> None:
    for fig in ("Figure13", "Figure14"):
        data = read_fa3(fig)
        if not any(v for d in data.values() for v in d.values()):
            continue
        f, axes = plt.subplots(2, 3, figsize=(11, 5.8), constrained_layout=True)
        for col, ln in enumerate(FA3_LENS):
            for row, (idx, ylab) in enumerate(((1, "Avg latency per output token (s)"),
                                               (2, "P99 latency per output token (s)"))):
                ax = axes[row, col]
                for color, (label, _, _) in zip(SERIES + ["#e87ba4"], FA3_SERIES):
                    pts = data[ln].get(label) or []
                    if pts:
                        ax.plot([p[0] for p in pts], [p[idx] for p in pts], color=color, lw=2,
                                marker="o", ms=4, label=label)
                ax.set_ylim(bottom=0)
                if row == 0:
                    ax.set_title(ln)
                else:
                    ax.set_xlabel("Request rate (req/s)")
                if col == 0:
                    ax.set_ylabel(ylab)
        axes[0, 0].legend(loc="upper left", fontsize=7)
        f.suptitle(f"{fig} on the H200, default setup: sm90 GEMMs + pass-built FA3, fixed runner, 60 s per point",
                   fontsize=10)
        save(f, out / f"{fig.lower()}_fa3")


def save(fig, stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=LOGS / "20261028-figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_sweep(args.out)
    plot_setups(args.out)
    plot_priority(args.out)
    plot_a100(args.out)
    plot_a100_bench(args.out)
    plot_fix(args.out)
    plot_fig1314_fa3(args.out)


if __name__ == "__main__":
    main()
