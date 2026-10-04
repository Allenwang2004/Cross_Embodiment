#!/usr/bin/env python3
"""plot_cost_profile.py -- per-step loss (1 - cos(B(s_t), B(g_t))) over the clip,
z0 vs the searched latent, from cost_profile.py. Time is the fraction of the clip
(headstand clips are 120 frames, jump 150, the rest 300)."""
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
P = np.load(REPO / "outputs/cost_profile/profiles.npz", allow_pickle=True)
prof = {tuple(k.split("|")): P[f"c{i}"] for i, k in enumerate(P["keys"])}
grid = np.linspace(0, 1, 101)
def resample(c):
    x = np.linspace(0, 1, len(c)); return np.interp(grid, x, 1 - c)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True); fig.patch.set_facecolor(SURF)
for x, g in zip(ax, ("headstand", "walking", "8 motions")):
    for lab, col in (("z0", "#8a8983"), ("best", "#2a78d6")):
        L = np.stack([resample(c) for (gg, s, l), c in prof.items() if gg == g and l == lab])
        for row in L: x.plot(grid, row, color=col, alpha=.18, lw=.8)
        x.plot(grid, np.median(L, 0), color=col if lab == "best" else INK2, lw=2.4,
               label=("z0" if lab == "z0" else "searched latent") + f" (cost {L.mean():.2f})")
        thirds = [L[:, :34].mean(), L[:, 34:67].mean(), L[:, 67:].mean()]
        print(f"{g:10s} {lab:4s} n={len(L):2d} cost {L.mean():.3f} | by third {thirds[0]:.3f} {thirds[1]:.3f} {thirds[2]:.3f}"
              f" | first-third share {L[:, :34].sum() / L.sum():.0%}")
    x.set_title(g + (" (10 trials)" if g != "8 motions" else " (path end, child)"), fontsize=11)
    x.set_xlabel("fraction of the clip", fontsize=10); x.legend(frameon=False, fontsize=9, loc="upper left")
    x.set_facecolor(SURF); x.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
ax[0].set_ylabel("per-step loss  1 - cos(B(s_t), B(g_t))", fontsize=10)
fig.tight_layout(); out = REPO / "outputs/cost_profile/profile.png"
fig.savefig(out, dpi=140, facecolor=SURF); print(f"-> {out}")
