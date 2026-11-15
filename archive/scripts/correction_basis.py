#!/usr/bin/env python3
"""correction_basis.py -- the k-dim correction basis of a batch of single_z_search runs.

Each finished clip under --runs gives a correction d = 16 z_best/|z_best| - 16 z0/|z0| (Euclidean, |z| = 16).
The basis is the top-k right singular vectors of the stacked corrections (uncentred SVD, as for
outputs/lowdim_search/basis_corrPCA_train_exact.npy, the old exact-observation one). Printed:
  - how many clips, |d| (median), mean pairwise cos between corrections
  - the share of the corrections' energy in the top 1 / 4 / 8 / 16 / 32 directions, against a null of random
    directions with the same norms (if the corrections share directions, the real curve sits above it)
  - per category, the share of each category's energy the k-dim basis captures
  - principal cosines between this basis and --compare bases
Writes <runs>/basis_k<k>.npy (k, 256) and <runs>/basis_full.npy (all right singular vectors, by energy).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def unit16(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return 16 * v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/c540_mse_a03")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", default="train", help="train | test | all (4th column of <runs>/clips.txt)")
    ap.add_argument("--compare", nargs="*", default=["outputs/lowdim_search/basis_corrPCA_train_exact.npy",
                                                     "outputs/fast_search_c60/basis_k8_leaveout.npy"])
    a = ap.parse_args()
    runs = REPO / a.runs
    D, cats = [], []
    for l in open(runs / "clips.txt"):
        t, k, cat, split = l.split()
        d = runs / f"{t}_{k}"
        if (a.split != "all" and split != a.split) or not (d / "summary.json").exists():
            continue
        D.append(unit16(np.load(d / "best_z.npy")) - unit16(np.load(REPO / f"data/origin_z/{t}/{t}_{k}.npy")))
        cats.append(cat)
    D = np.stack(D); n = len(D)
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(20):
        R = rng.standard_normal(D.shape); R *= (np.linalg.norm(D, axis=1) / np.linalg.norm(R, axis=1))[:, None]
        s = np.linalg.svd(R, compute_uv=False); null.append(np.cumsum(s ** 2) / np.sum(s ** 2))
    null = np.mean(null, 0)
    nz = np.linalg.norm(D, axis=1) > 1e-9                      # a clip whose search never beat z0 has d = 0
    U = D[nz] / np.linalg.norm(D[nz], axis=1, keepdims=True); C = U @ U.T
    print(f"{a.runs}: {n} clips ({a.split}), |z_best - z0| median {np.median(np.linalg.norm(D, axis=1)):.2f}, "
          f"mean pairwise cos between corrections {C[np.triu_indices(len(U), 1)].mean():+.3f} ({n - len(U)} clips with d = 0)")
    print("energy in the top 1 / 4 / 8 / 16 / 32 directions:  real "
          + " / ".join(f"{energy[j - 1]:.0%}" for j in (1, 4, 8, 16, 32) if j <= len(energy))
          + "   random null " + " / ".join(f"{null[j - 1]:.0%}" for j in (1, 4, 8, 16, 32) if j <= len(null)))
    B = Vt[: a.k]
    by = defaultdict(list)
    for d, c in zip(D, cats):
        by[c].append(d)
    print(f"share of each category's correction energy inside the {a.k}-dim basis: " + "  ".join(
        f"{c} {np.sum((np.stack(v) @ B.T) ** 2) / np.sum(np.stack(v) ** 2):.0%} (n={len(v)})" for c, v in sorted(by.items())))
    for p in a.compare:
        Q = np.load(REPO / p)[: a.k]
        print(f"principal cosines vs {p}: {np.linalg.svd(B @ Q.T, compute_uv=False).round(2)}")
    np.save(runs / f"basis_k{a.k}.npy", B); np.save(runs / "basis_full.npy", Vt)
    print(f"wrote {runs / f'basis_k{a.k}.npy'}")


if __name__ == "__main__":
    main()
