#!/usr/bin/env python3
"""plot_pair_distances.py -- did latents that started close stay close after the search?

Walking (move-ego-0-2_4), every start tracking its own adult rollout on the child body,
two-stage L_align (outputs/latent_transfer_own/move-ego-0-2_4/align). One dot per pair of
the 8 starts (28 pairs): x = Euclidean distance between the two starts, y = Euclidean
distance between their searched latents. No projection, so every distance is the real one.
Latents are on the radius-16 sphere: two random latents are ~22.6 apart, opposite ones 32.
"""
import itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
L = ["z0", "+0.5", "+1", "+2", "+5", "+10", "+20", "+30"]


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


S = [ld(REPO / f"outputs/latent_transfer/move-ego-0-2_4/starts/{n}.npy") for n in N]
Z = [ld(REPO / f"outputs/latent_transfer_own/move-ego-0-2_4/align/{n}/best_z.npy") for n in N]
P = list(itertools.combinations(range(8), 2))
x = np.array([np.linalg.norm(S[i] - S[j]) for i, j in P])
y = np.array([np.linalg.norm(Z[i] - Z[j]) for i, j in P])
rng = np.random.default_rng(0); R = rng.standard_normal((4000, 256)); R = 16 * R / np.linalg.norm(R, axis=1, keepdims=True)
rand = float(np.linalg.norm(R[:2000] - R[2000:], axis=1).mean())

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.4, 6.2)); fig.patch.set_facecolor(SURF)
xl, yl = 9, 24
ax.plot([0, xl], [0, xl], color=INK3, ls=(0, (4, 3)), lw=1.2)
ax.text(6.6, 5.6, "after = before", ha="left", va="top", fontsize=9, color=INK2)
ax.axhline(rand, color="#e34948", ls=(0, (4, 3)), lw=1.2)
ax.text(xl - 0.1, rand + 0.3, f"two random latents: {rand:.1f}", ha="right", va="bottom", fontsize=9, color="#e34948")
z0p = np.array([i == 0 for i, _ in P])
ax.scatter(x[~z0p], y[~z0p], s=46, color="#2a78d6", edgecolor=INK, lw=.5, zorder=3, label="other pairs")
ax.scatter(x[z0p], y[z0p], s=52, color="#eb6834", edgecolor=INK, lw=.5, zorder=4, label="pairs with z0 (z0 vs +0.5 ... +30)")
for (i, j), xi, yi in zip(P, x, y):
    if i == 0 and j in (1, 7):
        ax.annotate(f"z0 vs {L[j]}: {xi:.2f} -> {yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(8, -14) if j == 1 else (-150, -16), fontsize=8.5, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_xlim(0, xl); ax.set_ylim(0, yl)
ax.set_xlabel("distance between the two starts (adult latents)", fontsize=10)
ax.set_ylabel("distance between their latents after the search on the child", fontsize=10)
ax.set_title("walking (move-ego-0-2_4), each start tracks its own adult rollout, two-stage L_align\n"
             f"28 pairs of the 8 starts | before {x.mean():.2f} on average -> after {y.mean():.2f}", fontsize=10)
ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); out = REPO / "outputs/latent_transfer_own/pair_distances_move-ego-0-2_4_align.png"
fig.savefig(out, dpi=150, facecolor=SURF)
print(f"before {x.min():.2f}-{x.max():.2f} (mean {x.mean():.2f}) | after {y.min():.2f}-{y.max():.2f} (mean {y.mean():.2f}) | random {rand:.2f} -> {out}")
