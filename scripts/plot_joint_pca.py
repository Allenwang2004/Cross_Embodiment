#!/usr/bin/env python3
"""plot_joint_pca.py -- headstand and walking on ONE PCA: the 8 starts and 8 searched latents of
each clip (every start tracking its own adult rollout on the child, two-stage L_align;
outputs/latent_transfer_own/<stem>/align), 32 latents in all, PCA fit on all 32 (radius-16 vectors).
Writes outputs/latent_transfer_own/pca_joint_align.png.
"""
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
L = ["0", "0.5", "1", "2", "5", "10", "20", "30"]
CLIPS = {"headstand_3": ("headstand", "#eb6834"), "move-ego-0-2_4": ("walking", "#2a78d6")}


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


pts, meta = [], []
for stem in CLIPS:
    for kind, sub in (("start", None), ("after", "align")):
        for n, l in zip(N, L):
            p = (REPO / f"outputs/latent_transfer/{stem}/starts/{n}.npy" if kind == "start"
                 else REPO / f"outputs/latent_transfer_own/{stem}/align/{n}/best_z.npy")
            pts.append(ld(p)); meta.append((stem, kind, l))
P = np.stack(pts); mu = P.mean(0)
_, sv, Vt = np.linalg.svd(P - mu, full_matrices=False)
ev = sv ** 2 / (sv ** 2).sum(); Y = (P - mu) @ Vt[:2].T
idx = {m: i for i, m in enumerate(meta)}

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6.6)); fig.patch.set_facecolor(SURF)
for stem, (name, col) in CLIPS.items():
    for l in L:
        b, e = Y[idx[(stem, "start", l)]], Y[idx[(stem, "after", l)]]
        ax.annotate("", xy=e, xytext=b, arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=.55))
        ax.scatter(*e, s=80, color=col, marker="^", edgecolor=INK, lw=.5, zorder=3)
        ax.annotate(l, e, textcoords="offset points", xytext=(5, 3), fontsize=7.5, color=INK2)
    st = np.stack([Y[idx[(stem, "start", l)]] for l in L])
    ax.scatter(st[:, 0], st[:, 1], s=60, color=col, marker="o", edgecolor=INK, lw=.5, zorder=4)
    ax.scatter([], [], s=55, color=col, marker="o", edgecolor=INK, lw=.5, label=f"{name}: starts (adult latents)")
    ax.scatter([], [], s=65, color=col, marker="^", edgecolor=INK, lw=.5, label=f"{name}: after search on the child")
ax.legend(frameon=False, fontsize=8.5, loc="best")
ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
ax.set_title("headstand and walking, each start tracks its own adult rollout, two-stage L_align\n"
             "32 latents (8 starts + 8 searched, per clip), PCA fit on all 32; labels = start rotation (deg)", fontsize=10)
ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); out = REPO / "outputs/latent_transfer_own/pca_joint_align.png"
fig.savefig(out, dpi=150, facecolor=SURF)
print(f"PC1 {ev[0]:.1%} PC2 {ev[1]:.1%} (together {ev[0] + ev[1]:.1%}) -> {out}")
E = lambda a, b: float(np.linalg.norm(a - b))
for stem, (name, _) in CLIPS.items():
    s = [P[idx[(stem, "start", l)]] for l in L]; z = [P[idx[(stem, "after", l)]] for l in L]
    ps = [E(s[i], s[j]) for i in range(8) for j in range(i + 1, 8)]; pz = [E(z[i], z[j]) for i in range(8) for j in range(i + 1, 8)]
    print(f"{name:9s}: pairwise distance starts {np.mean(ps):.2f} -> after {np.mean(pz):.2f}")
sh = [P[idx[("headstand_3", "start", l)]] for l in L]; sw = [P[idx[("move-ego-0-2_4", "start", l)]] for l in L]
zh = [P[idx[("headstand_3", "after", l)]] for l in L]; zw = [P[idx[("move-ego-0-2_4", "after", l)]] for l in L]
print(f"headstand vs walking: starts {np.mean([E(a, b) for a in sh for b in sw]):.2f} apart on average, after search {np.mean([E(a, b) for a in zh for b in zw]):.2f}")
