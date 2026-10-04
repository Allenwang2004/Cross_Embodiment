#!/usr/bin/env python3
"""plot_lowdim_trend.py -- correction difference against start distance, one panel per search dimension k.

Same data as analyze_lowdim_search.py (8 walking starts, own targets, two-stage L_align, no penalty;
k = 8 / 16 / 32 / 64 / 128 subspace of the b500 correction PCA, 256 = unconstrained). Every panel: the 28 start
pairs (dots), the 7 pairs with the z0 start joined by a line (start distance 0.14 -> 8.28), the size of
one correction, and what a smooth (learnable) map would give: a difference that shrinks with the inputs'.
Writes outputs/lowdim_search/lowdim_trend.png.
"""
import itertools
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
STEM = "move-ego-0-2_4"
RUNS = {8: f"outputs/lowdim_search/{STEM}/k8/align", 16: f"outputs/lowdim_search/{STEM}/k16/align",
        32: f"outputs/lowdim_search/{STEM}/k32/align", 64: f"outputs/lowdim_search/{STEM}/k64/align",
        128: f"outputs/lowdim_search/{STEM}/k128/align", 256: f"outputs/latent_transfer_own/{STEM}/align"}
COL = {8: "#e87ba4", 16: "#008300", 32: "#2a78d6", 64: "#1baf7a", 128: "#eda100", 256: "#eb6834"}


def ld(p):
    v = np.load(REPO / p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def main():
    S = np.stack([ld(f"outputs/latent_transfer/{STEM}/starts/{n}.npy") for n in N])
    P = list(itertools.combinations(range(8), 2))
    xs = np.array([np.linalg.norm(S[i] - S[j]) for i, j in P])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 10), sharey=True); fig.patch.set_facecolor(SURF)
    axs = axs.ravel()
    for ax, (k, root) in zip(axs, RUNS.items()):
        d = np.stack([ld(f"{root}/{n}/best_z.npy") for n in N]) - S
        y = np.array([np.linalg.norm(d[i] - d[j]) for i, j in P])
        size = np.linalg.norm(d, axis=1).mean()
        xx = np.logspace(-1, 1, 50)
        ax.plot(xx, xx, color=INK3, ls=(0, (4, 3)), lw=1.2, label="smooth (learnable) map: shrinks with the inputs")
        ax.axhline(size, color=INK2, ls=":", lw=1.1, label=f"size of one correction ({size:.1f})")
        ax.scatter(xs, y, s=26, color=COL[k], alpha=.45, lw=0, label="all 28 pairs")
        j = np.array([P.index((0, i)) for i in range(1, 8)])
        ax.plot(xs[j], y[j], "-o", color=COL[k], mec=SURF, mew=1, ms=7, lw=1.8, label="each start vs the z0 start")
        near, far = y[xs < 1].mean(), y[xs > 4].mean()
        ax.set_title(("unconstrained (k = 256)" if k == 256 else f"k = {k}")
                     + f"\nstarts < 1 apart: {near:.1f}   |   > 4 apart: {far:.1f}", fontsize=10.5, color=INK)
        ax.set_xscale("log"); ax.set_xlim(0.1, 10); ax.set_ylim(0, 25)
        ax.set_xlabel("start distance |s_i − s_j|  (log scale)", fontsize=9.5, color=INK2)
        ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)
    for a in (axs[0], axs[3]):
        a.set_ylabel("correction difference |d_i − d_j|", fontsize=9.5, color=INK2)
    axs[0].legend(frameon=False, fontsize=8.2, loc="upper left", labelcolor=INK)
    fig.suptitle("walking, 8 nearby starts, each searched in a k-dim subspace: a learnable correction would rise from 0 "
                 "along the dashed line",
                 fontsize=10.5, color=INK)
    fig.tight_layout(); out = REPO / "outputs/lowdim_search/lowdim_trend.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
