#!/usr/bin/env python3
"""child_52_sd.py -- Comparison A (outputs/child_52_curves/per_run.npz): mean +- 1 standard deviation over the 260 runs of each method at every generation, plus where that spread comes from: between motions (the 52 motion means),
within a motion between seeds (the motions' mean seed sd), and the sd of the paired difference ES - other over motions. Writes outputs/child_52_curves/{sd.json, convergence_metrics_sd.png}. usage: uv run python scripts/child_52_sd.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent; CUR = REPO / "outputs/child_52_curves"; D = np.load(CUR / "per_run.npz"); KEYS = list(D["keys"]); G = (4, 8, 16, 32, 64)
ALGOS = [("es", "LOES (ours)", "#2a78d6"), ("cem", "CEM", "#eb6834"), ("random", "Random search", "#1baf7a")]; INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"
rep = dict(z0={k: dict(mean=float(D["z0"][:, i].mean()), sd=float(D["z0"][:, i].std(ddof=1))) for i, k in enumerate(KEYS)}, methods={}, paired={})
for a, _, _ in ALGOS:
    B = D[f"best_{a}"]; rep["methods"][a] = {}
    for g in G:
        rep["methods"][a][str(g * 16)] = {k: dict(mean=float(B[:, :, g - 1, i].mean()), sd=float(B[:, :, g - 1, i].std(ddof=1)), sd_between_motions=float(B[:, :, g - 1, i].mean(1).std(ddof=1)), sd_between_seeds=float(B[:, :, g - 1, i].std(1, ddof=1).mean()), min=float(B[:, :, g - 1, i].min()), max=float(B[:, :, g - 1, i].max())) for i, k in enumerate(KEYS)}
for o in ("random", "cem"):
    rep["paired"][f"es_minus_{o}"] = {str(g * 16): {k: dict(mean=float((D["best_es"][:, :, g - 1, i].mean(1) - D[f"best_{o}"][:, :, g - 1, i].mean(1)).mean()), sd=float((D["best_es"][:, :, g - 1, i].mean(1) - D[f"best_{o}"][:, :, g - 1, i].mean(1)).std(ddof=1))) for i, k in enumerate(KEYS)} for g in G}
(CUR / "sd.json").write_text(json.dumps(rep, indent=1))
x = (np.arange(64) + 1) * 16; fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=170, facecolor=SURF)
for b, (k, lab) in zip(ax, (("mpjpe", "MPJPE (cm)"), ("mmpjpe", "MMPJPE (cm)"), ("mpjae", "MPJAE (m/s²)"))):
    i = KEYS.index(k)
    for a, name, col in ALGOS:
        Y = D[f"best_{a}"][:, :, :, i].reshape(260, 64); m, s = Y.mean(0), Y.std(0, ddof=1); b.fill_between(x, m - s, m + s, color=col, alpha=0.12, lw=0, zorder=2); b.plot(x, m, color=col, lw=2, zorder=3, label=name)
    z = rep["z0"][k]["mean"]; b.axhline(z, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); b.text(x[-1], z, f"z0 {z:.2f}", color=MUTED, fontsize=9, va="bottom", ha="right")
    b.set_ylim(0, None); b.set_xlabel("simulated rollouts", fontsize=10, color=MUTED); b.set_ylabel(lab + " (lower is better)", fontsize=10, color=MUTED); b.set_facecolor(SURF); b.grid(color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=0, labelsize=9)
    for sp in ("top", "right", "left", "bottom"):
        b.spines[sp].set_visible(False)
ax[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
fig.suptitle("Child, 52 motions: best latent so far, mean ± 1 standard deviation over 260 runs (lower edge cut at 0)", x=0.01, ha="left", fontsize=12, color=INK); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(CUR / "convergence_metrics_sd.png", facecolor=SURF)
K = ("ratio", "fall_s", "mpjpe", "mmpjpe", "mpjae")
print("z0", {k: (round(rep["z0"][k]["mean"], 3), round(rep["z0"][k]["sd"], 3)) for k in K})
for a, _, _ in ALGOS:
    for n in ("256", "1024"):
        print(a, n, {k: (round(rep["methods"][a][n][k]["mean"], 3), round(rep["methods"][a][n][k]["sd"], 3), "mot", round(rep["methods"][a][n][k]["sd_between_motions"], 3), "seed", round(rep["methods"][a][n][k]["sd_between_seeds"], 3), "max", round(rep["methods"][a][n][k]["max"], 2)) for k in K})
for o, x_ in rep["paired"].items():
    print(o, {n: {k: (round(x_[n][k]["mean"], 3), round(x_[n][k]["sd"], 3)) for k in ("ratio", "mpjpe", "mmpjpe", "mpjae")} for n in ("256", "1024")})
