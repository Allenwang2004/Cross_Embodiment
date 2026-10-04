#!/usr/bin/env python3
"""analyze_segment_z.py -- does giving headstand a short latent sequence help?

Reads outputs/segment_z/K{1,2,4}/headstand_*/ (segment_z_search.py). All three
use the same code, budget and seed, so each trial is compared paired across K.
The headline number is the REPLAYED cost of the best knots (mean of 3 re-scores),
not the best sample's own cost, which is biased low by the luckiest draw.
Also: how fast each K gets there (median best-so-far vs rollouts).
"""
import csv, json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
COL = {1: "#8a8983", 2: "#2a78d6", 4: "#eb6834"}
TR = [0, 2, 3, 4, 9, 15, 20, 23, 30, 34]
S, C = {}, {}
for K in (1, 2, 4):
    for k in TR:
        d = REPO / f"outputs/segment_z/K{K}/headstand_{k}"
        if (d / "summary.json").exists():
            S[(K, k)] = json.loads((d / "summary.json").read_text())
            C[(K, k)] = np.array([[float(r["evals"]), float(r["best_so_far"])] for r in csv.DictReader(open(d / "curve.csv"))])
ks = [k for k in TR if all((K, k) in S for K in (1, 2, 4))]
print(f"{len(ks)} trials complete for K = 1, 2, 4\n")
print(f"{'trial':14s} " + " ".join(f"{'K=' + str(K) + ' best/replay':>20s}" for K in (1, 2, 4)))
for k in ks:
    print(f"headstand_{k:<4d} " + " ".join(f"{S[(K, k)]['best_cost']:9.3f} / {S[(K, k)]['replay_mean']:.3f}" for K in (1, 2, 4)))
for K in (1, 2, 4):
    r = np.array([S[(K, k)]["replay_mean"] for k in ks]); b = np.array([S[(K, k)]["best_cost"] for k in ks])
    z0 = np.array([S[(K, k)]["origin_z_cost"] for k in ks])
    ang = np.array([np.mean(S[(K, k)]["knot_deg_from_z0"]) for k in ks])
    line = f"K={K}: replayed cost mean {r.mean():.3f} median {np.median(r):.3f} | best sample {b.mean():.3f} | z0 {z0.mean():.3f} | knots {ang.mean():.0f} deg from z0"
    if K > 1:
        r1 = np.array([S[(1, k)]["replay_mean"] for k in ks])
        line += f" | beats K=1 on {(r < r1).sum()}/{len(ks)} trials, mean diff {np.mean(r - r1):+.3f}"
    print(line)
for K in (1, 2, 4):
    for thr in (0.15, 0.10):
        hit = [C[(K, k)][np.argmax(C[(K, k)][:, 1] <= thr), 0] if (C[(K, k)][:, 1] <= thr).any() else np.nan for k in ks]
        print(f"K={K}: rollouts to best-so-far <= {thr}: median {np.nanmedian(hit):.0f}  (reached in {np.sum(~np.isnan(hit))}/{len(ks)})")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4)); fig.patch.set_facecolor(SURF)
for K in (1, 2, 4):
    ev = C[(K, ks[0])][:, 0]
    M = np.stack([np.interp(ev, C[(K, k)][:, 0], C[(K, k)][:, 1]) for k in ks])
    ax[0].plot(ev, np.median(M, 0), color=COL[K], lw=2.2, label=f"{K} knot" + ("s" if K > 1 else " (one latent)"))
    ax[0].fill_between(ev, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=COL[K], alpha=.12, lw=0)
    ax[1].scatter([K + np.random.default_rng(K).uniform(-.08, .08) for _ in ks], [S[(K, k)]["replay_mean"] for k in ks],
                  color=COL[K], s=26, zorder=3)
for k in ks:
    ax[1].plot([1, 2, 4], [S[(K, k)]["replay_mean"] for K in (1, 2, 4)], color=INK3, lw=.7, alpha=.6)
ax[0].set_xlabel("rollouts"); ax[0].set_ylabel("best cost so far (median, IQR)"); ax[0].set_xscale("log")
ax[0].set_title("headstand, 10 trials: search progress", fontsize=11); ax[0].legend(frameon=False)
ax[1].set_xticks([1, 2, 4]); ax[1].set_xlabel("number of knot latents over the clip")
ax[1].set_ylabel("replayed cost of the best latent(s)"); ax[1].set_title("final result, per trial", fontsize=11)
for x in ax:
    x.set_facecolor(SURF); x.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
fig.tight_layout(); out = REPO / "outputs/segment_z/segment_z.png"
fig.savefig(out, dpi=140, facecolor=SURF); print(f"-> {out}")
