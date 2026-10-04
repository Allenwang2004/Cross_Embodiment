#!/usr/bin/env python3
"""plot_seed_consistency.py -- does the single-z search land on the same z twice?

Reads the runs scripts/single_z_search.py wrote for one clip under several
seeds (outputs/single_z_seeds/<clip>_<objective>_s<seed>/) and answers two
different questions on one sheet:

  determinism   a run repeated with the SAME seed (tag s0rep) must reproduce
                seed 0 bit for bit -- common random numbers, deterministic
                actor. If it does not, nothing below means anything.
  uniqueness    runs with DIFFERENT seeds: do they find the same z, or
                different z's of the same cost? Judged in three ways --
                pairwise cosine of the best z's (a heat map), each run's
                best-so-far curve, and a PCA scatter of the best z's around
                z0 with the ES iterate's path drawn in.

The cosine matrix is the number to read. Two z's on the sqrt(256) sphere with
cosine 0.9 are 0.45 rad apart; the ES step sigma here is 0.25 of |z|, so
anything under ~0.9 is further apart than one search step and is a different
solution, not the same one found twice.

Usage:
    uv run scripts/plot_seed_consistency.py --root outputs/single_z_seeds
    uv run scripts/plot_seed_consistency.py --root outputs/single_z_seeds --clip crawl-0.4-0-d_0

Writes <root>/seed_consistency_<clip>.png and <root>/seed_consistency.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_runs(root, clip):
    runs = {}
    for d in sorted(root.iterdir()):
        m = re.match(rf"^{re.escape(clip)}_(\w+)_(s\d+(?:rep)?)$", d.name)
        if not m or not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        curve = list(csv.DictReader(open(d / "curve.csv")))
        runs[m.group(2)] = dict(dir=d, objective=m.group(1), summary=s,
                                best_z=np.load(d / "best_z.npy").reshape(-1).astype(np.float64),
                                trace=np.load(d / "z_trace.npz"),
                                evals=np.array([int(r["evals"]) for r in curve]),
                                best_so_far=np.array([float(r["best_so_far"]) for r in curve]))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="outputs/single_z_seeds")
    ap.add_argument("--clip", nargs="*", default=None, help="default: every clip found under --root")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    clips = args.clip or sorted({re.sub(r"_\w+_s\d+(rep)?$", "", d.name)
                                 for d in root.iterdir() if (d / "summary.json").exists()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for clip in clips:
        runs = load_runs(root, clip)
        if not runs:
            print(f"{clip}: no runs"); continue
        seeds = sorted(k for k in runs if not k.endswith("rep"))
        s0 = runs[seeds[0]]["summary"]
        task, stem = s0["clip"].split("/")
        z0 = np.load(REPO_ROOT / "data" / "origin_z" / task / f"{stem}.npy").reshape(-1).astype(np.float64)
        objective = runs[seeds[0]]["objective"]

        # --- determinism -----------------------------------------------------
        det = None
        if "s0rep" in runs and "s0" in runs:
            a, b = runs["s0"], runs["s0rep"]
            det = dict(max_abs_dz=float(np.abs(a["best_z"] - b["best_z"]).max()),
                       d_cost=abs(a["summary"]["best"]["cost"] - b["summary"]["best"]["cost"]),
                       max_abs_trace=float(np.abs(a["trace"]["z_mean"] - b["trace"]["z_mean"]).max()))

        # --- uniqueness ------------------------------------------------------
        Z = np.stack([runs[k]["best_z"] for k in seeds])
        C = np.array([[cosine(a, b) for b in Z] for a in Z])
        cos0 = np.array([cosine(z, z0) for z in Z])
        costs = np.array([runs[k]["summary"]["best"]["cost"] for k in seeds])
        gens = np.array([runs[k]["summary"]["best"]["gen"] for k in seeds])
        off = C[~np.eye(len(seeds), dtype=bool)]
        print(f"\n{clip}  ({objective}, {len(seeds)} seeds, {s0['evals']} evals each)")
        print(f"  best cost per seed : " + "  ".join(f"{k} {c:.4f}" for k, c in zip(seeds, costs))
              + f"   (z0 {s0['origin_z']['cost']:.4f})")
        print(f"  found at gen       : " + "  ".join(f"{k} {g}" for k, g in zip(seeds, gens)))
        print(f"  cos(best_z, z0)    : " + "  ".join(f"{k} {c:.3f}" for k, c in zip(seeds, cos0)))
        print(f"  pairwise cos(best_z_i, best_z_j): min {off.min():.3f}  median {np.median(off):.3f}  max {off.max():.3f}")
        if det:
            print(f"  same-seed repeat   : max|dz| {det['max_abs_dz']:.2e}  |dcost| {det['d_cost']:.2e}  "
                  f"max|d z_mean trace| {det['max_abs_trace']:.2e}")
        for i, k in enumerate(seeds):
            rows.append(dict(clip=clip, objective=objective, seed=k, best_cost=costs[i], best_gen=int(gens[i]),
                             cos_z0=cos0[i], cos_other_min=float(np.delete(C[i], i).min()),
                             cos_other_mean=float(np.delete(C[i], i).mean()),
                             same_seed_max_abs_dz=det["max_abs_dz"] if (det and k == "s0") else ""))

        # --- figure ----------------------------------------------------------
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
        im = ax[0].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax[0].set_xticks(range(len(seeds))); ax[0].set_xticklabels(seeds)
        ax[0].set_yticks(range(len(seeds))); ax[0].set_yticklabels(seeds)
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                ax[0].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8,
                           color="white" if abs(C[i, j]) > 0.6 else "black")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[0].set_title("cos(best_z_i, best_z_j)")

        for k in seeds:
            r = runs[k]
            ax[1].plot(r["evals"], r["best_so_far"], label=f"{k}  best {r['summary']['best']['cost']:.3f}")
        ax[1].axhline(s0["origin_z"]["cost"], color="k", ls=":", lw=1, label="z0")
        ax[1].set_xscale("log"); ax[1].set_xlabel("evals"); ax[1].set_ylabel("best-so-far cost")
        ax[1].set_title("search curve per seed"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

        x = np.arange(len(seeds))
        ax[2].bar(x - 0.2, costs, 0.4, label="best cost")
        ax[2].bar(x + 0.2, cos0, 0.4, label="cos(best_z, z0)")
        ax[2].set_xticks(x); ax[2].set_xticklabels(seeds); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
        ax[2].set_title("per seed")

        # PCA of {z0, every seed's mean-z path, best z's}: where did each run go?
        paths = [runs[k]["trace"]["z_mean"].astype(np.float64) for k in seeds]
        allz = np.concatenate([z0[None], Z] + paths)
        mu = allz.mean(0)
        U, S, Vt = np.linalg.svd(allz - mu, full_matrices=False)
        P = lambda z: (z - mu) @ Vt[:2].T
        for k, path in zip(seeds, paths):
            pp = P(path); ax[3].plot(pp[:, 0], pp[:, 1], lw=0.6, alpha=0.5)
            pb = P(runs[k]["best_z"]); ax[3].plot(pb[0], pb[1], "*", ms=12, color=ax[3].lines[-1].get_color(), label=k)
        p0 = P(z0); ax[3].plot(p0[0], p0[1], "ko", ms=8, label="z0")
        var = S[:2] ** 2 / (S ** 2).sum()
        ax[3].set_title(f"PCA of z (paths thin, best *)  {100*var[0]:.0f}% / {100*var[1]:.0f}%")
        ax[3].legend(fontsize=7); ax[3].grid(alpha=.3); ax[3].set_aspect("equal", adjustable="datalim")

        det_txt = (f"   same-seed repeat: max|dz| = {det['max_abs_dz']:.1e}" if det else "")
        fig.suptitle(f"{clip}  --  {objective} search, {len(seeds)} seeds x {s0['evals']} evals   "
                     f"pairwise cos min {off.min():.2f} / median {np.median(off):.2f}{det_txt}")
        fig.tight_layout()
        out = root / f"seed_consistency_{clip}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  -> {out}")

    if rows:
        with open(root / "seed_consistency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"-> {root / 'seed_consistency.csv'}")


if __name__ == "__main__":
    main()
