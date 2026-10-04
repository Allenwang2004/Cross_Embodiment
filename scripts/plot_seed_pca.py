#!/usr/bin/env python3
"""plot_seed_pca.py -- repeated-seed step: does ONE start (the clip's z0) lead to one
adapted latent, or to many?

Walking (move-ego-0-2_4) on the child body, 20 seeds, the two two-stage versions:
  global  bfm stage (4992 evals), then 2048 evals on bfm + 1.0*heading + 0.1*root-xy
          (outputs/single_z_seeds_twostage)
  align   bfm stage (4992 evals), then 2048 evals on L_align (outputs/single_z_seeds_twostage_align)

Per version, PCA is fit on the 21 unit latents (z0 + the 20 final solutions) and the
first two components are drawn: z0 as a circle, each seed's final latent as a triangle,
an arrow from z0 to each. Writes outputs/seeds_compare/pca_walk_twostage_<version>.png.
"""
import itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
VERSIONS = {"global": ("single_z_seeds_twostage", "two-stage, bfm + heading + root position"),
            "align": ("single_z_seeds_twostage_align", "two-stage, L_align")}


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / np.linalg.norm(v)


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
z0 = unit(np.load(REPO / "data/origin_z/move-ego-0-2/move-ego-0-2_4.npy"))
for v, (d, title) in VERSIONS.items():
    Z = [unit(np.load(REPO / "outputs" / d / f"move-ego-0-2_4_s{k}" / "best_z.npy")) for k in range(20)]
    P = np.stack([z0] + Z); mu = P.mean(0)
    _, sv, Vt = np.linalg.svd(P - mu, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum(); Y = (P - mu) @ Vt[:2].T
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
    from_z0 = [ang(z, z0) for z in Z]
    pair = [ang(Z[i], Z[j]) for i, j in itertools.combinations(range(20), 2)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2)); fig.patch.set_facecolor(SURF)
    for k in range(20):
        ax.annotate("", xy=Y[k + 1], xytext=Y[0], arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1.0, alpha=.55))
        ax.scatter(*Y[k + 1], s=80, color="#2a78d6", marker="^", edgecolor=INK, lw=.6, zorder=3)
        ax.annotate(f"s{k}", Y[k + 1], textcoords="offset points", xytext=(6, 3), fontsize=8, color=INK2)
    ax.scatter(*Y[0], s=110, color=INK, marker="o", zorder=4)
    ax.annotate("z0", Y[0], textcoords="offset points", xytext=(8, -12), fontsize=9.5, color=INK)
    ax.scatter([], [], s=70, marker="o", color=INK, label="start: the clip's z0 (same for every seed)")
    ax.scatter([], [], s=70, marker="^", color="#2a78d6", label="final latent of each seed")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.set_title(f"walking (move-ego-0-2_4) on the child, 20 seeds, {title}\n"
                 f"from z0 {np.median(from_z0):.0f} deg (median), between seeds {np.median(pair):.0f} deg "
                 f"[{min(pair):.0f}-{max(pair):.0f}]", fontsize=10)
    ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); out = REPO / "outputs/seeds_compare" / f"pca_walk_twostage_{v}.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(f"{v}: PC1 {ev[0]:.1%} PC2 {ev[1]:.1%} | from z0 median {np.median(from_z0):.1f} [{min(from_z0):.1f}-{max(from_z0):.1f}] | "
          f"between seeds median {np.median(pair):.1f} [{min(pair):.1f}-{max(pair):.1f}] -> {out}")
