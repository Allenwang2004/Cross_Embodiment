#!/usr/bin/env python3
"""plot_lso_estimator.py -- the LSO gradient-estimate ablation (lso_search.py --antithetic / --rank): best cost so far
against iteration, the paper's estimate vs mirrored pairs vs per-step ranks vs both, one panel per clip.
Writes outputs/lso/estimator_ablation.png.
usage: uv run python scripts/plot_lso_estimator.py"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CLIPS = [("Male2Walking__B13_-__Walk_turn_right_90", "Walk, turn right 90°"), ("Female1Gestures__D6-_CartWheel", "Cartwheel")]
VARIANTS = [("", "Paper (leave-one-out mean)", "#2a78d6"), ("_pair", "Antithetic pairs", "#eb6834"), ("_rank", "Per-step ranks", "#1baf7a"),
            ("_pairrank", "Pairs + ranks", "#eda100")]                     # dataviz categorical 1-4
INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
for ax, (clip, title) in zip(axes, CLIPS):
    ax.set_facecolor(SURF)
    ends = []
    for suf, name, c in VARIANTS:
        d = REPO / f"outputs/lso/{clip}_global_it64{suf}"
        rows = list(csv.DictReader(open(d / "curve.csv")))
        start = json.loads((d / "summary.json").read_text())["results"]["B(g_t) (LSO start)"]["cost"]
        y = [start] + [float(r["best_so_far"]) for r in rows]
        ax.plot(range(len(y)), y, color=c, lw=2, label=name)
        ends.append((len(y) - 1, y[-1]))
    if max(e for _, e in ends) - min(e for _, e in ends) < 0.02:      # too close to label one by one
        ax.text(ends[0][0] + 1, min(e for _, e in ends), f"all four\n{min(e for _, e in ends):.3f}–{max(e for _, e in ends):.3f}",
                color=MUTED, fontsize=9, va="bottom")
    else:
        for x, e in ends:
            ax.text(x + 1, e, f"{e:.3f}", color=MUTED, fontsize=9, va="center")
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.set_xlabel("iteration (128 rollouts each)", fontsize=10, color=INK)
    ax.set_xlim(0, 72)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)
axes[0].set_ylabel("best cost so far (lower is better)", fontsize=10, color=INK)
axes[0].legend(frameon=False, fontsize=10, loc="upper right")
fig.suptitle("LSO with the single-z ES gradient estimate: antithetic pairs, ranks", fontsize=14, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.905, "child body, 1 seed each, 64 iterations; every run starts from B(g_t) per frame", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(REPO / "outputs/lso/estimator_ablation.png", dpi=200)
print("-> outputs/lso/estimator_ablation.png")
