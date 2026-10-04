#!/usr/bin/env python3
"""plot_path_angles.py -- how far apart are the best latents of two bodies on a path?

The 2-D projection (trajectories_m2c.png) hides this: continuation's drift lies
along its two principal directions, independent search's scatter lies in the other
254, so independent solutions LOOK clustered. Angles in the full 256-D space:

  left, middle  angle between the solutions of body i and body j on the
                adult -> child path, averaged over the 8 motions, for
                continuation and independent search
  right         the same angle against how many path steps apart the two bodies
                are, all four paths
"""
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE = "#2a78d6", "#eb6834"
PATHS = {"m2c": "adult → child", "m2s": "adult → short limbs", "m2g": "adult → giant", "m2k": "adult → short, stocky"}


def unit(v):
    return v / np.linalg.norm(v)


def angles(prefix, method):
    """(clips, bodies, bodies) angle matrix in degrees, and the body list."""
    root = REPO / "outputs/continuation" / prefix
    bodies = sorted({p.name.split("__")[1] for p in (root / "cont").iterdir()}, key=lambda b: int(b.rsplit("_t", 1)[1]))
    stems = sorted({p.name.split("__")[0] for p in (root / "cont").iterdir()})
    out = []
    for s in stems:
        fs = [root / method / f"{s}__{b}" / "best_z.npy" for b in bodies]
        if not all(f.exists() for f in fs):
            continue
        Z = np.stack([unit(np.load(f).reshape(-1).astype(np.float64)) for f in fs])
        out.append(np.degrees(np.arccos(np.clip(Z @ Z.T, -1, 1))))
    return np.stack(out), bodies


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig = plt.figure(figsize=(14, 4.6)); fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.32, 1.35], wspace=0.3)
A = {m: angles("m2c", m) for m in ("cont", "indep")}
vmax = max(A[m][0].mean(0).max() for m in A)
for k, (m, title) in enumerate((("cont", "continuation"), ("indep", "independent search"))):
    x = fig.add_subplot(gs[0, k]); M, bodies = A[m]
    im = x.imshow(M.mean(0), cmap="viridis", vmin=0, vmax=vmax)
    ts = [f"{int(b.rsplit('_t', 1)[1]) / 1000:g}" for b in bodies]
    x.set_xticks(range(len(bodies))); x.set_xticklabels(ts, fontsize=7.5, rotation=45, ha="right")
    x.set_yticks(range(len(bodies))); x.set_yticklabels(ts, fontsize=7.5)
    x.set_xlabel("body t (0 = adult, 1 = child)", fontsize=9)
    if k == 0:
        x.set_ylabel("body t", fontsize=9)
    x.set_title(title, fontsize=11)
    for i in range(len(bodies) - 1):          # write the one-step values next to the diagonal
        x.text(i + 1, i, f"{M.mean(0)[i, i + 1]:.0f}", ha="center", va="center", fontsize=7, color="white")
cax = fig.add_subplot(gs[0, 2]); fig.colorbar(im, cax=cax).set_label("angle (deg)", fontsize=9)
x = fig.add_subplot(gs[0, 4])
for prefix in PATHS:
    for m, col, ls in (("cont", BLUE, "-"), ("indep", ORANGE, "--")):
        M, bodies = angles(prefix, m); n = len(bodies)
        sep = np.arange(1, n)
        v = [np.mean([M[:, i, i + d].mean() for i in range(n - d)]) for d in sep]
        x.plot(sep, v, color=col, ls=ls, lw=1.6, alpha=.75, marker="o", ms=3.5)
x.plot([], [], color=BLUE, lw=1.8, marker="o", ms=4, label="continuation")
x.plot([], [], color=ORANGE, lw=1.8, ls="--", marker="o", ms=4, label="independent search")
x.legend(frameon=False, fontsize=9, loc="lower right")
x.set_xlabel("how many path steps apart the two bodies are", fontsize=9)
x.set_ylabel("angle between their solutions (deg)", fontsize=9)
x.set_title("all four paths, 8 motions each", fontsize=11)
x.set_ylim(0, None); x.set_facecolor(SURF); x.grid(alpha=.22, color=INK3, lw=.7)
for sp in ("top", "right"): x.spines[sp].set_visible(False)
fig.suptitle("adult → child path, averaged over 8 motions: angle between the best latents of two bodies (256-D)",
             fontsize=10.5, color=INK2, x=0.33, y=1.0)
out = REPO / "outputs/morph_study/path_angles.png"
fig.savefig(out, dpi=140, facecolor=SURF, bbox_inches="tight"); print(f"-> {out}")
for prefix in PATHS:
    for m in ("cont", "indep"):
        M, bodies = angles(prefix, m); n = len(bodies)
        one = np.mean([M[:, i, i + 1].mean() for i in range(n - 1)])
        far = M[:, 0, n - 1].mean()
        print(f"{prefix} {m:5s}: one step {one:5.1f} deg | first-to-last {far:5.1f} deg")
