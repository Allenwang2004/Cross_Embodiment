#!/usr/bin/env python3
"""plot_z_scree.py -- how many independent directions do the seeds' best z's span?

N seeds of scripts/single_z_search.py give N best z's on the sqrt(256) sphere,
all of about the same cost. If the set of good z's is a d-dimensional region,
the N directions span only d dimensions once N > d: the SVD of the N x 256
matrix of unit directions has d large singular values and N - d near zero.
This draws that scree curve, and the same curve for N random points on the
sphere (which span N dimensions with singular values all about sqrt(N/256)...
i.e. no drop), so "has it dropped" has a reference.

Also reports the participation ratio (sum s^2)^2 / sum s^4 -- an effective
dimension that does not need a threshold -- and, for each seed, how much of
its direction is explained by the OTHER seeds (leave-one-out): near 1 means
that seed brought nothing new, near 0 means every seed is a fresh direction.

Usage:
    uv run scripts/plot_z_scree.py --root outputs/single_z_seeds_s005_5k --clip move-ego-0-2_4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="outputs/single_z_seeds_s005_5k")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seed", type=int, default=0, help="rng for the random reference cloud")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    runs = []
    for d in sorted(root.iterdir()):
        m = re.match(rf"^{re.escape(args.clip)}_(\w+?)_s(\d+)$", d.name)
        if m and (d / "best_z.npy").exists():
            runs.append((int(m.group(2)), d))
    runs.sort()
    tags = [f"s{s}" for s, _ in runs]
    Z = np.stack([np.load(d / "best_z.npy").reshape(-1).astype(np.float64) for _, d in runs])
    costs = [json.loads((d / "summary.json").read_text())["best"]["cost"] for _, d in runs]
    U = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    N, D = U.shape

    s = np.linalg.svd(U, compute_uv=False)
    # centred version: directions AROUND the shared mean, i.e. the spread of
    # the plateau itself without the common "this motion" component
    Uc = U - U.mean(0)
    sc = np.linalg.svd(Uc, compute_uv=False)
    rng = np.random.default_rng(args.seed)
    R = rng.standard_normal((N, D)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    sr = np.linalg.svd(R, compute_uv=False)
    src = np.linalg.svd(R - R.mean(0), compute_uv=False)

    pr = lambda v: float((v ** 2).sum() ** 2 / (v ** 4).sum())
    loo = []
    for i in range(N):
        Q, _ = np.linalg.qr(np.delete(U, i, 0).T)
        loo.append(float(np.linalg.norm(Q @ (Q.T @ U[i]))))

    print(f"{args.clip}: {N} seeds, cost {min(costs):.4f}-{max(costs):.4f}")
    print("singular values (raw)    :", " ".join(f"{v:.2f}" for v in s))
    print("singular values (centred):", " ".join(f"{v:.2f}" for v in sc))
    print("random cloud   (raw)     :", " ".join(f"{v:.2f}" for v in sr))
    print("random cloud   (centred) :", " ".join(f"{v:.2f}" for v in src))
    print(f"participation ratio: raw {pr(s):.1f}   centred {pr(sc):.1f}   "
          f"(random cloud: raw {pr(sr):.1f}, centred {pr(src):.1f}; max possible {N})")
    print("leave-one-out: fraction of each seed's direction inside span(others):")
    print("  " + "  ".join(f"{t} {v:.2f}" for t, v in zip(tags, loo)))
    print(f"  mean {np.mean(loo):.3f}   (random cloud would give ~{np.sqrt((N - 1) / D):.2f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    k = np.arange(1, N + 1)
    ax[0].plot(k, s, "o-", label="best z's (raw)")
    ax[0].plot(k, sr, "s--", color="gray", label="random points on the sphere")
    ax[0].set_title("singular values, raw directions"); ax[0].set_xlabel("index"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(k, sc, "o-", label="best z's (centred)")
    ax[1].plot(k, src, "s--", color="gray", label="random points (centred)")
    ax[1].set_title("singular values, centred (spread of the plateau)"); ax[1].set_xlabel("index"); ax[1].grid(alpha=.3); ax[1].legend()
    ax[2].bar(k, loo); ax[2].axhline(np.sqrt((N - 1) / D), color="gray", ls="--", label="random-cloud level")
    ax[2].set_ylim(0, 1); ax[2].set_title("leave-one-out: |proj of seed i onto span(others)|")
    ax[2].set_xlabel("seed"); ax[2].legend(); ax[2].grid(alpha=.3, axis="y")
    fig.suptitle(f"{args.clip}: {N} seeds  --  participation ratio raw {pr(s):.1f} / centred {pr(sc):.1f}  "
                 f"(random: {pr(sr):.1f} / {pr(src):.1f})")
    fig.tight_layout()
    out = root / f"z_scree_{args.clip}.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
