#!/usr/bin/env python3
"""plot_correction_orthogonality.py -- what a PCA picture of corrections hides.

A correction is d = z* - z_start (radius-16 latents). Two sets:
  8 starts   walking (move-ego-0-2_4): z0 and 7 rotations of it (0.5..30 deg), each searched on the
             child for its OWN adult rollout, two-stage L_align, no penalty
             (outputs/latent_transfer_own/move-ego-0-2_4/align)
  b500       the supervised dataset (outputs/b500_targets/sup_dataset): one correction per clip,
             z_align - z0, 488 clips
Left: for the same pairs, cos(d_i, d_j) measured in the PCA plane (what a PCA arrow plot shows,
PCA fitted on starts + results as in the PCA figures) against the cos in the full 256-d space.
Right: the 256-d cos of b500 corrections against the cos of random directions in 256-d.
Writes outputs/b500_targets/correction_orthogonality.png.
"""
import itertools
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
STEM = "move-ego-0-2_4"


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def unit(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def plane(start, end):
    P = np.concatenate([start, end]); mu = P.mean(0)
    return np.linalg.svd(P - mu, full_matrices=False)[2][:2]


def main():
    # 8 starts
    S = np.stack([ld(REPO / f"outputs/latent_transfer/{STEM}/starts/{n}.npy") for n in N])
    Z = np.stack([ld(REPO / f"outputs/latent_transfer_own/{STEM}/align/{n}/best_z.npy") for n in N])
    D8 = Z - S; V8 = plane(S, Z)
    iu8 = np.triu_indices(8, 1)
    c8, c8p = (unit(D8) @ unit(D8).T)[iu8], (unit(D8 @ V8.T) @ unit(D8 @ V8.T).T)[iu8]
    # b500
    X = np.load(REPO / "outputs/b500_targets/sup_dataset/targets.npz")
    z0, za, cat = X["z0"].astype(np.float64), X["z_align"].astype(np.float64), X["category"]
    D = za - z0; V = plane(z0, za); n = len(D)
    U, Up = unit(D), unit(D @ V.T)
    iu = np.triu_indices(n, 1); same = (cat[:, None] == cat[None])[iu]
    C, Cp = (U @ U.T)[iu], (Up @ Up.T)[iu]
    nn = np.argsort(np.linalg.norm(z0[:, None] - z0[None], axis=-1), axis=1)[:, 1]
    c_nn = np.array([U[i] @ U[nn[i]] for i in range(n)])
    rng = np.random.default_rng(0)
    R = unit(rng.standard_normal((40000, 256)))
    c_rand = np.sum(R[:20000] * R[20000:], axis=1)
    keep_in_plane = float(np.mean(np.linalg.norm(D @ V.T, axis=1) ** 2 / np.linalg.norm(D, axis=1) ** 2))
    keep8 = float(np.mean(np.linalg.norm(D8 @ V8.T, axis=1) ** 2 / np.linalg.norm(D8, axis=1) ** 2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 6.0), gridspec_kw=dict(width_ratios=[1, 1.08]))
    fig.patch.set_facecolor(SURF)
    # left: PCA-plane cos vs 256-d cos
    sel = rng.choice(np.nonzero(same)[0], size=min(4000, same.sum()), replace=False)
    a1.plot([-1, 1], [-1, 1], color=INK3, ls=(0, (4, 3)), lw=1.2, zorder=1)
    a1.text(0.62, 0.74, "PCA plane = 256-d", rotation=38, fontsize=8.5, color=INK2, ha="center")
    a1.axhline(0, color=INK3, lw=.8, zorder=1)
    a1.scatter(Cp[sel], C[sel], s=7, color=BLUE, alpha=.35, lw=0, zorder=2,
               label=f"b500: two clips of the same category ({same.sum()} pairs, 4000 shown)")
    a1.scatter(c8p, c8, s=58, color=ORANGE, edgecolor=SURF, lw=1.2, zorder=4,
               label="8 similar walking starts (28 pairs)")
    a1.annotate(f"8 starts: PCA plane {c8p.mean():+.2f} on average,\nreally {c8.mean():+.2f} "
                "(partly shared, not parallel)", (c8p.mean(), c8.mean()),
                xytext=(-0.98, 0.72), fontsize=8.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=.9))
    a1.annotate(f"b500: PCA plane anywhere from -1 to +1,\nreally {C[same].mean():+.2f} on average",
                (-0.55, 0.02), xytext=(-0.98, -0.62), fontsize=8.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=.9))
    a1.set_xlim(-1.03, 1.03); a1.set_ylim(-1.03, 1.03)
    a1.set_xlabel("cos between two corrections, in the PCA plane (what the PCA figure shows)", fontsize=9.5, color=INK2)
    a1.set_ylabel("cos between the same two corrections, in 256-d", fontsize=9.5, color=INK2)
    a1.set_title(f"the PCA plane holds {keep8:.0%} of an 8-start correction and {keep_in_plane:.1%} of a b500 one,\n"
                 "so the arrows' directions there say little about the real ones", fontsize=10, color=INK)
    a1.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    # right: 256-d cos distributions
    bins = np.linspace(-0.4, 0.7, 45)
    a2.hist(c_rand, bins=bins, density=True, histtype="stepfilled", color=INK3, alpha=.28,
            label=f"two random directions in 256-d (sd {c_rand.std():.3f})")
    a2.hist(C[same], bins=bins, density=True, histtype="step", color=BLUE, lw=2,
            label=f"b500, same category: mean {C[same].mean():+.3f}")
    a2.hist(c_nn, bins=bins, density=True, histtype="step", color=AQUA, lw=2,
            label=f"b500, z0 nearest neighbours: mean {c_nn.mean():+.3f}")
    ymax = a2.get_ylim()[1]
    for c in c8:
        a2.plot([c, c], [0, ymax * .1], color=ORANGE, lw=1.6)
    a2.plot([], [], color=ORANGE, lw=1.6, label=f"8 similar walking starts: {c8.min():+.2f}..{c8.max():+.2f}")
    a2.axvline(1.0, color=INK3)
    a2.set_xlim(-0.4, 0.7)
    a2.set_xlabel("cos between two corrections, in 256-d   (1 = same direction, 0 = perpendicular)",
                  fontsize=9.5, color=INK2)
    a2.set_ylabel("density", fontsize=9.5, color=INK2)
    a2.set_title("corrections of different clips are close to perpendicular, near what random directions give:\n"
                 "a map from z0 to the correction has almost nothing shared to learn", fontsize=10, color=INK)
    a2.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    for ax in (a1, a2):
        ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)
    fig.tight_layout(); out = REPO / "outputs/b500_targets/correction_orthogonality.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(f"8 starts: cos 256-d {c8.mean():+.2f}, PCA plane {c8p.mean():+.2f}, in-plane share {keep8:.0%}")
    print(f"b500: same-cat cos 256-d {C[same].mean():+.3f}, PCA plane {Cp[same].mean():+.2f}, nn {c_nn.mean():+.3f}, "
          f"random sd {c_rand.std():.3f}, in-plane share {keep_in_plane:.1%}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
