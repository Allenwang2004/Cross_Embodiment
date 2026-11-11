#!/usr/bin/env python3
"""plot_k_scale_bars.py -- a body with every joint's k scaled 0.8 / 1 / 1.2: median cost over the 8 clips before search
(z0) and after (the searched z), with the drop between the two medians.
  child (scripts/scale_body_gains.py): outputs/k_scale/compare_k0.8_k1_k1.2.json -> outputs/k_scale/median_cost_bars.png
  --body g1 (assets/robots/g1/gains_perjoint_k{0.8,1.2}.json, k x 1 = outputs/user_survey_search/g1_ridge):
      outputs/k_scale/compare_g1_k0.8_k1_k1.2.json -> outputs/k_scale/median_cost_bars_g1.png
usage: uv run python scripts/plot_k_scale_bars.py [--body child|g1]"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from crossenbodiment.palette import INK as _INK, Palette   # noqa: E402  (refactor step 10: this figure's own colours, VARIANTS "k_scale")
INK, MUTED, GRID, SURF = (_INK["dataviz"][k] for k in ("ink", "muted", "grid", "surface"))
C_Z0, C_BEST = Palette.hex("frozen", "k_scale"), Palette.hex("loes", "k_scale")   # dataviz categorical 2 / 1 (red-ish z0, blue searched, as in the videos)

ap = argparse.ArgumentParser(); ap.add_argument("--body", default="child", choices=["child", "g1"]); body = ap.parse_args().body
sfx = "" if body == "child" else f"_{body}"
d = json.loads((REPO / f"outputs/k_scale/compare{sfx}_k0.8_k1_k1.2.json").read_text())
keys, names = ["x0.8", "x1", "x1.2"], ["0.8k", "k", "1.2k"]
z0 = [float(np.median([d[c][k]["z0"] for c in d])) for k in keys]
best = [float(np.median([d[c][k]["best"] for c in d])) for k in keys]

fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor=SURF)
ax.set_facecolor(SURF)
x, w = np.arange(3), 0.34
b0 = ax.bar(x - w / 2 - 0.01, z0, w, color=C_Z0, label="z0 (before search)", zorder=3)
b1 = ax.bar(x + w / 2 + 0.01, best, w, color=C_BEST, label="searched z", zorder=3)
u = max(z0) / 0.885                            # label offsets scaled to the axis (0.885 = the child's tallest bar)
for i in range(3):
    ax.text(x[i] - w / 2 - 0.01, z0[i] + 0.015 * u, f"{z0[i]:.3f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.text(x[i] + w / 2 + 0.01, best[i] + 0.015 * u, f"{best[i]:.3f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    drop = 1 - best[i] / z0[i]
    ax.annotate(f"−{100 * drop:.0f}%", xy=(x[i], max(z0[i], best[i]) + 0.09 * u), ha="center", fontsize=13, color=INK, fontweight="bold")
ax.set_xticks(x, names, fontsize=13, color=INK)
ax.set_ylabel("median cost over 8 clips (lower is better)", fontsize=10, color=INK)
ax.set_ylim(0, max(z0) * 1.28)
ax.set_title(f"{'Child' if body == 'child' else 'G1'}, motor gain k scaled: median cost before and after search", fontsize=12, color=INK, loc="left", pad=14)
ax.grid(axis="y", color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED); ax.tick_params(axis="y", colors=MUTED, length=0); ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK)
fig.tight_layout()
out = REPO / f"outputs/k_scale/median_cost_bars{sfx}.png"
fig.savefig(out, facecolor=SURF)
print(out.relative_to(REPO), [round(v, 3) for v in z0], [round(v, 3) for v in best])
