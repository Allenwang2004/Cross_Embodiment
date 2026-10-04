#!/usr/bin/env python3
"""plot_lowdim.py -- the many good latents of one (body, motion) fill a LOW-dimensional region.

Data from walk_z_plateau.py (child body; walking move-ego-0-2_4 and rotation
rotate-y--5-0.8_0): ~2000 latents that each score within the plateau threshold,
harvested from the 20 seeds' ES traces, their PCA basis at the spherical mean
(basis.npz), and rollouts walking out from that mean along groups of principal
directions (validate.npz: 8 random directions per group, 10..70 deg).

  left   cumulative share of the good latents' spread, by principal direction,
         against an even spread over all 255 tangent directions
  right  how far one can walk from the centre along each group of principal
         directions before the cost leaves the plateau (crosses thr), against a
         random direction
"""
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
CLIPS = {"move-ego-0-2_4": ("walking", "#2a78d6"), "rotate-y--5-0.8_0": ("rotation", "#eb6834")}


def radius(costs, angles, c0, thr, cap):
    """angle at which one direction's cost first exceeds thr (linear interpolation)."""
    a = np.r_[0.0, angles]; c = np.r_[c0, costs]
    for i in range(1, len(a)):
        if c[i] > thr:
            return a[i - 1] + (thr - c[i - 1]) / max(c[i] - c[i - 1], 1e-9) * (a[i] - a[i - 1])
    return cap


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6)); fig.patch.set_facecolor(SURF)
RR = []
for clip, (name, col) in CLIPS.items():
    d = REPO / f"outputs/single_z_seeds_s005_5k/plateau_{clip}"
    B = np.load(d / "basis.npz"); V = np.load(d / "validate.npz")
    lam = B["sv"] ** 2; cum = np.cumsum(lam) / lam.sum(); n = int((lam > 1e-12).sum())
    k80 = int(np.searchsorted(cum, 0.8) + 1)
    pr = lam.sum() ** 2 / (lam ** 2).sum()
    ax[0].plot(np.arange(1, n + 1), cum[:n], color=col, lw=2.2, label=f"{name}: {int(B['n_points'])} good latents")
    ax[0].scatter([k80], [0.8], color=col, s=40, zorder=5)
    ax[0].annotate(f"{k80} directions\nhold 80%", (k80, 0.8), textcoords="offset points",
                   xytext=(12, -34 if name == "walking" else -64), fontsize=9.5, color=col)
    ang = V["angles"]; thr = float(B["thr"]); c0 = float(V["mean_cost"]); cap = float(ang[-1])
    bands = [k for k in V.files if k.startswith("c") and "-" in k]
    xs, rs, lo, hi = [], [], [], []
    for k in bands:
        a_, b_ = map(int, k[1:].split("-"))
        r = np.array([radius(V[k][j], ang, c0, thr, cap) for j in range(V[k].shape[0])])
        xs.append((a_ + b_) / 2); rs.append(np.median(r)); lo.append(np.percentile(r, 25)); hi.append(np.percentile(r, 75))
    xs, rs = np.array(xs), np.array(rs)
    ax[1].plot(xs, rs, color=col, lw=2.2, marker="o", ms=4, label=name)
    ax[1].fill_between(xs, lo, hi, color=col, alpha=.12, lw=0)
    rr = np.median([radius(V["random"][j], ang, c0, thr, cap) for j in range(V["random"].shape[0])])
    RR.append(rr)
    print(f"{clip}: {int(B['n_points'])} points, thr {thr:.4f} | 80/90/95% of spread in "
          f"{k80}/{int(np.searchsorted(cum, .9) + 1)}/{int(np.searchsorted(cum, .95) + 1)} directions | participation ratio {pr:.1f} of {n}"
          f" | walkable: top-32 median {np.median(rs[xs <= 32]):.0f} deg, beyond 96 {np.median(rs[xs > 96]):.0f} deg, random {rr:.0f} deg")
ax[0].plot([1, 255], [1 / 255, 1], color=INK3, ls=(0, (4, 3)), lw=1.2)
rr = float(np.mean(RR))
ax[1].axhline(rr, color=INK3, ls=(0, (4, 3)), lw=1.3)
ax[1].text(252, rr + 0.8, f"a random direction: {rr:.0f}°", ha="right", va="bottom", fontsize=9.5, color=INK2)
ax[0].text(150, 0.52, "if spread evenly over\nall 255 directions", fontsize=9, color=INK2, ha="left")
ax[0].set_xlim(0, 256); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("principal direction (sorted by spread)", fontsize=10)
ax[0].set_ylabel("cumulative share of the spread", fontsize=10)
ax[0].set_title("where the good latents spread out", fontsize=11.5)
ax[0].legend(frameon=False, fontsize=9.5, loc="lower right")
ax[1].set_xlim(0, 256); ax[1].set_ylim(0, 75)
ax[1].set_xlabel("principal direction (groups of 8–32, sorted by spread)", fontsize=10)
ax[1].set_ylabel("how far one can walk and stay good (deg)", fontsize=10)
ax[1].set_title("walking out from the centre of the good region", fontsize=11.5)
ax[1].legend(frameon=False, fontsize=9.5, loc="upper right")
for x in ax:
    x.set_facecolor(SURF); x.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
fig.tight_layout()
out = REPO / "outputs/morph_study/lowdim.png"
fig.savefig(out, dpi=140, facecolor=SURF); print(f"-> {out}")
