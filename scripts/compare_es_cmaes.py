#!/usr/bin/env python3
"""compare_es_cmaes.py -- antithetic ES + Adam vs CMA-ES on the same single-z
search: how fast each reaches the plateau and how straight it gets there.

Reads the curve.csv / z_trace.npz / summary.json that single_z_search.py
writes, for any number of runs per algorithm, and reports per run

  evals to best <= thr      first eval budget at which the best sample so far
                            is on the plateau
  evals to mean <= thr      same for the iterate (mean z), from the eval-every
                            rollouts of the mean
  path / net                cumulative angle the mean travelled over the net
                            angle from z0 to where it ended; 1.0 = a straight
                            great circle, larger = wandering

Usage:
    uv run scripts/compare_es_cmaes.py --thr 0.275 \
        --es outputs/single_z_seeds_s005_5k/move-ego-0-2_4_bfm_s0 outputs/single_z_seeds_s005_5k/move-ego-0-2_4_bfm_s1 \
        --cmaes outputs/es_vs_cmaes/move-ego-0-2_4_cmaes_s0 ... --out outputs/es_vs_cmaes/compare_move-ego-0-2_4.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ang(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


def load(d):
    d = Path(d)
    rows = list(csv.DictReader(open(d / "curve.csv")))
    evals = np.array([int(r["evals"]) for r in rows])
    best = np.array([float(r["best_so_far"]) for r in rows])
    mean_ev = np.array([int(r["evals"]) for r in rows if r["mean_z_cost"]])
    mean_c = np.array([float(r["mean_z_cost"]) for r in rows if r["mean_z_cost"]])
    tr = np.load(d / "z_trace.npz")["z_mean"].astype(np.float64)
    net = np.array([ang(tr[0], z) for z in tr[1:]])
    path = np.cumsum([ang(tr[i], tr[i + 1]) for i in range(len(tr) - 1)])
    s = json.loads((d / "summary.json").read_text())
    return dict(name=d.name, evals=evals, best=best, mean_ev=mean_ev, mean_c=mean_c, net=net, path=path,
                floor=s["best"]["cost"], minutes=s.get("minutes", np.nan))


def first_below(x, y, thr):
    i = np.where(y <= thr)[0]
    return int(x[i[0]]) if len(i) else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--es", nargs="+", required=True)
    p.add_argument("--cmaes", nargs="+", required=True)
    p.add_argument("--thr", type=float, required=True, help="plateau threshold, e.g. the walk_z_plateau thr")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = {"ES": [load(d) for d in args.es], "CMA-ES": [load(d) for d in args.cmaes]}

    print(f"{'run':40s} {'best':>7s} {'evals->best<=thr':>17s} {'evals->mean<=thr':>17s} {'path':>6s} {'net':>5s} {'path/net':>8s} {'min':>5s}")
    for algo, rs in runs.items():
        for r in rs:
            eb, em = first_below(r["evals"], r["best"], args.thr), first_below(r["mean_ev"], r["mean_c"], args.thr)
            print(f"{r['name']:40s} {r['floor']:7.4f} {str(eb):>17s} {str(em):>17s} {r['path'][-1]:6.0f} {r['net'][-1]:5.0f} "
                  f"{r['path'][-1] / max(r['net'][-1], 1e-6):8.2f} {r['minutes']:5.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    col = {"ES": "C0", "CMA-ES": "C3"}
    for algo, rs in runs.items():
        for i, r in enumerate(rs):
            lab = algo if i == 0 else None
            ax[0].plot(r["evals"], r["best"], c=col[algo], lw=1.2, alpha=.8, label=lab)
            ax[1].plot(r["mean_ev"], r["mean_c"], c=col[algo], marker="o", ms=3, lw=1, alpha=.8, label=lab)
            ax[2].plot(r["evals"], r["net"], c=col[algo], lw=1.2, alpha=.8, label=lab)
            ax[3].plot(r["net"], r["path"], c=col[algo], lw=1.2, alpha=.8, label=lab)
    for a in ax[:2]:
        a.axhline(args.thr, c="k", ls=":", lw=.8, label=f"thr {args.thr}")
        a.set_xlabel("evals (rollouts)"); a.grid(alpha=.3)
    ax[0].set_ylabel("cost"); ax[0].set_title("best sample so far")
    ax[1].set_title("cost of the mean z (rolled out every 25 gens)")
    ax[2].set_xlabel("evals"); ax[2].set_ylabel("deg"); ax[2].set_title("angle of the mean from z0"); ax[2].grid(alpha=.3)
    lim = max(r["net"][-1] for rs in runs.values() for r in rs) * 1.05
    ax[3].plot([0, lim], [0, lim], c="gray", ls="--", lw=.8, label="straight line")
    ax[3].set_xlabel("net angle from z0 (deg)"); ax[3].set_ylabel("path length travelled (deg)")
    ax[3].set_title("directness: path vs net displacement"); ax[3].grid(alpha=.3)
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("antithetic ES + Adam  vs  CMA-ES  --  same clip, body, budget and population")
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
