#!/usr/bin/env python3
"""plot_es_sampling.py -- single_z_search.py --sampling antithetic vs independent (leave-one-out mean), 16 rollouts per
step, 3 seeds each: best cost so far against step, thin lines per seed, thick the seed mean, one panel per clip.
Colours match plot_lso_estimator.py (blue = leave-one-out mean, orange = antithetic pairs).
Writes outputs/es_sampling_test/es_sampling.png.
usage: uv run python scripts/plot_es_sampling.py"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent; D = REPO / "outputs/es_sampling_test"
CLIPS = [("move-ego-0-2_2", "Walk (move-ego-0-2_2)"), ("rotate-x--5-0.8_4", "Roll (rotate-x--5-0.8_4)")]
METHODS = [("antithetic", "Antithetic pairs (current)", "#eb6834"), ("independent", "Leave-one-out mean", "#2a78d6")]
SEEDS = (0, 1, 2)
INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
for ax, (clip, title) in zip(axes, CLIPS):
    ax.set_facecolor(SURF)
    for m, name, c in METHODS:
        Y = []
        for s in SEEDS:
            d = D / f"{clip}_{m}_s{s}"
            z0 = json.loads((d / "summary.json").read_text())["origin_z"]["cost"]
            Y.append([z0] + [float(r["best_so_far"]) for r in csv.DictReader(open(d / "curve.csv"))])
        Y = np.array(Y); x = np.arange(Y.shape[1])
        for y in Y:
            ax.plot(x, y, color=c, lw=1, alpha=0.35)
        ax.plot(x, Y.mean(0), color=c, lw=2.5, label=name)
        ax.text(x[-1] + 1, Y.mean(0)[-1], f"{Y.mean(0)[-1]:.3f}", color=MUTED, fontsize=9, va="center")
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.set_xlabel("step (16 rollouts each)", fontsize=10, color=INK)
    ax.set_xlim(0, 72)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)
axes[0].set_ylabel("best cost so far (lower is better)", fontsize=10, color=INK)
axes[0].legend(frameon=False, fontsize=10, loc="upper right")
fig.suptitle("Single-z ES: antithetic pairs vs leave-one-out mean, same 16 rollouts per step", fontsize=14, color=INK,
             x=0.01, ha="left")
fig.text(0.01, 0.905, "child body, both with centred ranks, 64 steps; thin lines = 3 seeds, thick = their mean; "
         "step 0 = z0", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(D / "es_sampling.png", dpi=200)
print("->", D / "es_sampling.png")
