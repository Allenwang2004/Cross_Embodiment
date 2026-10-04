#!/usr/bin/env python3
"""plot_nonunique.py -- the best latent for one (body, motion) is not unique.

20 searches per clip on the child body, all from the same z0 with the same
budget; only the random seed differs (outputs/single_z_seeds_s005_5k). Left:
every search's cost / z0 against how far it moved from z0. Right: the angles
between the solutions, pairwise. If there were one best latent the right panel
would pile up near 0 deg.
"""
import glob, json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
COL = {"move-ego-0-2_4": "#2a78d6", "rotate-y--5-0.8_0": "#eb6834"}
NAME = {"move-ego-0-2_4": "walking (move-ego-0-2_4)", "rotate-y--5-0.8_0": "rotation (rotate-y_0)"}


def unit(v):
    return v / np.linalg.norm(v)


def deg(c):
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4)); fig.patch.set_facecolor(SURF)
for clip, col in COL.items():
    dirs = sorted(glob.glob(str(REPO / f"outputs/single_z_seeds_s005_5k/{clip}_bfm_s*")))
    S = [json.loads(Path(d, "summary.json").read_text()) for d in dirs]
    Z = np.stack([unit(np.load(Path(d, "best_z.npy")).reshape(-1).astype(np.float64)) for d in dirs])
    task = clip.rsplit("_", 1)[0]
    z0 = unit(np.load(REPO / f"data/origin_z/{task}/{clip}.npy").reshape(-1).astype(np.float64))
    ratio = np.array([s["best"]["cost"] / s["origin_z"]["cost"] for s in S])
    from_z0 = deg(Z @ z0)
    iu = np.triu_indices(len(Z), 1)
    pair = deg((Z @ Z.T)[iu])
    # the directions each search moved away from z0, in z0's tangent plane
    T = np.stack([unit(z - (z @ z0) * z0) for z in Z])
    tcos = (T @ T.T)[iu]
    print(f"{clip}: n={len(Z)} cost/z0 {ratio.min():.3f}-{ratio.max():.3f} | from z0 {from_z0.min():.0f}-{from_z0.max():.0f} deg"
          f" | pairwise {pair.min():.0f}-{pair.max():.0f} deg (median {np.median(pair):.0f})"
          f" | direction-from-z0 cos mean {tcos.mean():+.2f}")
    ax[0].scatter(from_z0, ratio, s=34, color=col, alpha=.8, edgecolor=SURF, lw=.6, label=NAME[clip])
    ax[1].hist(pair, bins=np.arange(0, 92, 3), color=col, alpha=.6, label=NAME[clip])
ax[0].scatter([0], [1], s=70, color=INK, marker="D", zorder=5)
ax[0].annotate("z0", (0, 1), textcoords="offset points", xytext=(8, -4), fontsize=9, color=INK)
ax[0].axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.1)
ax[0].set_xlim(-4, 80); ax[0].set_ylim(0, 1.08)
ax[0].set_xlabel("angle between the solution and z0 (deg)")
ax[0].set_ylabel("cost / z0 cost")
ax[0].set_title("20 searches each: same body, motion, start and budget", fontsize=11)
ax[0].legend(frameon=False, fontsize=9, loc="center right")
ax[1].axvline(90, color=INK2, ls=(0, (4, 3)), lw=1.1)
ax[1].text(89, ax[1].get_ylim()[1] * 0.95, "unrelated\ndirections", ha="right", va="top", fontsize=8.5, color=INK2)
ax[1].set_xlim(0, 92)
ax[1].set_xlabel("angle between two of the solutions (deg)")
ax[1].set_ylabel("pairs")
ax[1].set_title("…yet the solutions are far apart", fontsize=11)
for x in ax:
    x.set_facecolor(SURF); x.grid(alpha=.22, color=INK3, lw=.7)
    for s in ("top", "right"): x.spines[s].set_visible(False)
fig.tight_layout()
out = REPO / "outputs/morph_study/nonunique.png"
fig.savefig(out, dpi=140, facecolor=SURF); print(f"-> {out}")
