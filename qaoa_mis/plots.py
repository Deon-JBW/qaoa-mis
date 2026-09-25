"""Figures for the README from results/sweep.csv.

    python -m qaoa_mis.plots
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Depth is ordinal, so one blue ramp, light -> dark, rather than unrelated hues.
DEPTH_COLORS = {1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 4: "#0d366b"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_by(rows, metric):
    """{p: (penalties, means)} averaged over graphs."""
    acc = defaultdict(list)
    for r in rows:
        acc[(int(r["p"]), float(r["penalty"]))].append(float(r[metric]))
    out = defaultdict(lambda: ([], []))
    for (p, pen), vals in sorted(acc.items()):
        out[p][0].append(pen)
        out[p][1].append(np.mean(vals))
    return dict(out)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel("penalty weight λ", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def panel(ax, series, baseline, baseline_label, end_labels):
    for p, (x, y) in series.items():
        ax.plot(x, y, color=DEPTH_COLORS[p], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"p = {p}")
        if end_labels:
            ax.annotate(f"p={p}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=MUTED, fontsize=9)
    ax.axhline(baseline, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(baseline_label, (ax.get_xlim()[0], baseline), xytext=(4, 4), textcoords="offset points",
                color=MUTED, fontsize=9)


def main(csv_path=Path("results/sweep.csv"), out_dir=Path("figures")):
    rows = load(csv_path)
    out_dir.mkdir(exist_ok=True)
    uniform = np.mean([float(r["uniform_p_feasible"]) for r in rows])
    greedy = np.mean([float(r["greedy_ratio"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    # Left-panel lines converge at high penalty, so end labels would collide;
    # the legend sits in that panel's empty lower-right corner instead.
    panel(axes[0], mean_by(rows, "p_feasible"), uniform, "uniform random sampling", end_labels=False)
    style(axes[0], "Probability a sample is a valid independent set", "P(feasible)")
    axes[0].legend(frameon=False, labelcolor=MUTED, loc="lower right", title="circuit depth",
                   title_fontsize=9)
    panel(axes[1], mean_by(rows, "approx_ratio"), greedy, "classical greedy (best of 20)", end_labels=True)
    style(axes[1], "Expected solution quality", "approximation ratio")
    fig.tight_layout()
    fig.savefig(out_dir / "penalty_sweep.png", dpi=160)
    print(f"wrote {out_dir / 'penalty_sweep.png'}")


if __name__ == "__main__":
    main()
