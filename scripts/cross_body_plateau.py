#!/usr/bin/env python3
"""cross_body_plateau.py -- does one body's low-cost plateau carry over to
the other bodies?

Takes the plateau samples scripts/walk_z_plateau.py collected for one
(source body, clip) and rolls a subset of them out on EVERY body with that
body's own retargeted reference and obs scaling, scoring with the same bfm
loss. Per body it reports the cost of z0 (the naive transfer), of the source
plateau's spherical mean, and the distribution over the plateau points.

  plateau points cheap on body X  -> the plateau is shared; an adapter has
                                     little to do beyond leaving z0
  plateau points expensive on X   -> the plateau is body-specific; the
                                     adapter must learn how it moves with beta
  wide spread on X                -> only part of the source plateau survives;
                                     which part tells which directions matter

Usage:
    uv run scripts/cross_body_plateau.py --root outputs/single_z_seeds_s005_5k --clip move-ego-0-2_4 --n 96

Writes <root>/plateau_<clip>/cross_body.csv, cross_body.png and
cross_body_costs.npz (per-body cost of every evaluated z).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from single_z_search import device_arg, project_z
from walk_z_plateau import Scorer, load_seed_bests

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="outputs/single_z_seeds_s005_5k")
    p.add_argument("--clip", required=True, help="stem, e.g. move-ego-0-2_4")
    p.add_argument("--n", type=int, default=96, help="plateau samples to roll out per body (plus the seed bests, the mean and z0)")
    p.add_argument("--bodies", nargs="*", default=None, help="default: every dir under assets/robots_torque")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=device_arg, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--envs", type=int, default=16)
    args = p.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    out = root / f"plateau_{args.clip}"
    rng = np.random.default_rng(args.seed)

    tags, Zbest, s0 = load_seed_bests(root, args.clip)
    src = s0["body"]
    task, stem = s0["clip"].split("/")
    samp = np.load(out / "samples.npz")
    basis = np.load(out / "basis.npz")
    thr = float(samp["thr"])
    r = float(basis["radius"])
    Zp = samp["z"][samp["accepted"]]
    pick = rng.choice(len(Zp), size=min(args.n, len(Zp)), replace=False)
    z0 = project_z(np.load(REPO_ROOT / "data" / "origin_z" / task / f"{stem}.npy").reshape(-1).astype(np.float64))
    groups = [("z0", z0[None]), ("mean", (r * basis["mean"])[None]), ("best", Zbest), ("plateau", Zp[pick])]
    Z = np.concatenate([g for _, g in groups])
    labels = np.concatenate([[n] * len(g) for n, g in groups])
    bodies = args.bodies or sorted(d.name for d in (REPO_ROOT / "assets/robots_torque").iterdir() if (d / "robot_torque_full.xml").exists())
    bodies = [src] + [b for b in bodies if b != src]
    print(f"{args.clip}: {len(Z)} z's from {src}'s plateau (thr {thr}) x {len(bodies)} bodies")

    costs = {}
    for b in bodies:
        ref = REPO_ROOT / "data" / b / "retargeting_motion" / task / f"{stem}.npz"
        if not ref.exists():
            print(f"  {b}: no retargeted reference, skipped"); continue
        sb = dict(s0, body=b, xml=str(REPO_ROOT / "assets/robots_torque" / b / "robot_torque_full.xml"))
        sc = Scorer(sb, args.device, args.metamotivo, args.envs)
        c = sc(Z); sc.close()
        costs[b] = c
        pl = c[labels == "plateau"]; bs = c[labels == "best"]
        print(f"  {b:13s} z0 {c[0]:.3f}  mean {c[1]:.3f}  bests {bs.mean():.3f}  plateau {pl.mean():.3f} "
              f"[{pl.min():.3f}-{pl.max():.3f}] sd {pl.std():.3f}  frac<=thr {np.mean(pl <= thr):.2f}  "
              f"best-of-plateau {pl.min():.3f} vs z0 {c[0]:.3f}", flush=True)

    bodies = list(costs)
    np.savez(out / "cross_body_costs.npz", z=Z, labels=labels, bodies=bodies, thr=thr, src=src,
             **{f"cost_{b}": costs[b] for b in bodies})
    with open(out / "cross_body.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["body", "z0", "src_mean", "bests_mean", "plateau_mean", "plateau_min", "plateau_max", "plateau_sd", "frac_le_thr"])
        for b in bodies:
            c = costs[b]; pl = c[labels == "plateau"]
            w.writerow([b, f"{c[0]:.4f}", f"{c[1]:.4f}", f"{c[labels == 'best'].mean():.4f}", f"{pl.mean():.4f}",
                        f"{pl.min():.4f}", f"{pl.max():.4f}", f"{pl.std():.4f}", f"{np.mean(pl <= thr):.3f}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    x = np.arange(len(bodies))
    for i, b in enumerate(bodies):
        c = costs[b]; pl = c[labels == "plateau"]
        ax[0].scatter(np.full(len(pl), i) + rng.uniform(-.18, .18, len(pl)), pl, s=8, alpha=.5, c="C0",
                      label="source plateau points" if i == 0 else None)
        ax[0].scatter([i], [c[1]], marker="D", s=50, c="C1", zorder=3, label="source plateau mean" if i == 0 else None)
        ax[0].scatter([i], [c[0]], marker="x", s=70, c="C3", zorder=3, label="z0 (naive transfer)" if i == 0 else None)
    ax[0].axhline(thr, c="k", ls=":", lw=.8, label=f"{src} thr {thr}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(bodies, rotation=35, ha="right")
    ax[0].set_ylabel("bfm cost on that body"); ax[0].grid(alpha=.3, axis="y"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"{args.clip}: {src}'s plateau rolled out on every body")
    # how much of z0's gap does the source plateau close, per body
    gain = [(costs[b][0] - costs[b][labels == "plateau"].mean()) / max(costs[b][0], 1e-6) for b in bodies]
    ax[1].barh(x, gain, color=["C2" if g > 0 else "C3" for g in gain])
    ax[1].set_yticks(x); ax[1].set_yticklabels(bodies); ax[1].invert_yaxis(); ax[1].axvline(0, c="k", lw=.8)
    ax[1].set_xlabel("(z0 cost - plateau mean cost) / z0 cost"); ax[1].set_title("improvement over z0 from using the source plateau")
    ax[1].grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(out / "cross_body.png", dpi=130)
    print(f"-> {out / 'cross_body.csv'}\n-> {out / 'cross_body.png'}")


if __name__ == "__main__":
    main()
