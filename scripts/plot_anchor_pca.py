#!/usr/bin/env python3
"""plot_anchor_pca.py -- PCA of the 16 latents (8 starts + 8 searched) for each anchor-penalty
setting, drawn exactly like analyze_latent_transfer.py's PCA (fit on the 16 radius-16 latents of
that setting alone). Walking, each start tracking its own adult rollout, two-stage L_align with
+ lambda (|z - start| / 16)^2. Writes outputs/latent_transfer_anchor/pca_lam<lambda>.png.
The no-penalty version is outputs/latent_transfer_own/pca_move-ego-0-2_4_align.png.
"""
import itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
ANG = [0, 0.5, 1, 2, 5, 10, 20, 30]
STEM = "move-ego-0-2_4"


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
S = [ld(REPO / f"outputs/latent_transfer/{STEM}/starts/{n}.npy") for n in N]
for lam in ("0.03", "0.1", "0.3"):
    Z = [ld(REPO / f"outputs/latent_transfer_anchor/{STEM}/lam{lam}/align/{n}/best_z.npy") for n in N]
    P = np.stack(S + Z); mu = P.mean(0)
    _, sv, Vt = np.linalg.svd(P - mu, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum(); Y = (P - mu) @ Vt[:2].T
    before = np.mean([np.linalg.norm(S[i] - S[j]) for i, j in itertools.combinations(range(8), 2)])
    after = np.mean([np.linalg.norm(Z[i] - Z[j]) for i, j in itertools.combinations(range(8), 2)])
    moved = np.mean([np.linalg.norm(Z[i] - S[i]) for i in range(8)])
    fig, ax = plt.subplots(figsize=(7.2, 6.2)); fig.patch.set_facecolor(SURF)
    col = cm.viridis(np.linspace(0, .9, 8))
    for i in range(8):
        b, e = Y[i], Y[i + 8]
        ax.annotate("", xy=e, xytext=b, arrowprops=dict(arrowstyle="->", color=col[i], lw=1.2, alpha=.8))
        ax.scatter(*b, s=70, color=col[i], marker="o", edgecolor=INK, lw=.6, zorder=3)
        ax.scatter(*e, s=90, color=col[i], marker="^", edgecolor=INK, lw=.6, zorder=3)
        ax.annotate(f"{ANG[i]:g}°", e, textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK2)
    ax.scatter([], [], s=60, marker="o", color=INK3, label="start (adult latent: z0 rotated by the angle shown)")
    ax.scatter([], [], s=70, marker="^", color=INK3, label="after search on the child")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.set_title(f"walking, own target, two-stage L_align, penalty lambda = {lam}\n"
                 f"pairwise distance {before:.2f} -> {after:.2f}, moved {moved:.2f} from own start (PCA of these 16)",
                 fontsize=10)
    ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); out = REPO / f"outputs/latent_transfer_anchor/pca_lam{lam}.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(f"lambda {lam}: PC1 {ev[0]:.0%} PC2 {ev[1]:.0%} | pairwise {before:.2f} -> {after:.2f} | moved {moved:.2f} -> {out}")
