#!/usr/bin/env python3
"""grid_zero_shot.py -- predict the latent of a real body from the leg x arm grid.

Library: continuation latents on the 4 diagonal paths, the 16 axis paths and the
leg x arm grid (run_grid_continuation.sh). The grid lines through the adult are
the leg and arm axis paths, so with the 64 grid bodies the lattice
{leg values} x {arm values} is complete (9 x 9, other parameters at 1).

Every predictor works on corrections (tangent vectors at the clip's z0, which is
the same on every body) and walks the predicted correction out from z0 -- the
same object the correction-transfer study moves around. Per clip, only that
clip's latents are used.

  grid:bilinear  bilinear interpolation of the 4 lattice corrections around the
                 real body's (leg, arm): neighbours on both sides in both limbs
  grid:nn        the nearest lattice body in (leg, arm)
  limb:kernel    Gaussian-weighted mean of every library body's correction,
                 distance on (leg, arm) only; bandwidth by leave-one-body-out
  all:kernel     the same over all 8 body parameters (Euclidean), i.e. the old
                 predictor on the bigger library, to separate "more bodies" from
                 "the right distance"

Also writes a warm-start plan (run_bnn.sh format) from the nearest lattice body.

usage: uv run scripts/grid_zero_shot.py --out outputs/grid_study/jobs_zero_shot.npz \
           --plan outputs/grid_study/plan_warm.tsv
"""
import argparse, functools, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.append(str(REPO))
from fit_beta_map import Library, log_map, exp_map, unit
from fit_beta_map import z0_of as _z0_of
from model.dataset import load_beta

z0_of = functools.lru_cache(maxsize=None)(_z0_of)
_CORR = {}

LEGS = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]
ARMS = [0.55, 0.6625, 0.775, 0.8875, 1.0, 1.1, 1.2, 1.3, 1.4]
REAL = ("athletic child elderly giant long_limbed pear_shaped petite short_limbed short_stocky tall_slim teen "
        "x_leg050 x_leg055_adulttorso x_leg060_longtorso x_leg062_heavy x_leg070 x_leg075_heavy x_leg140 x_leg150_thin").split()
SAME_XML = {"child", "giant", "short_limbed", "short_stocky"}      # identical to a library body, not held out


def lattice(L):
    """(i, j) -> library body at (LEGS[i], ARMS[j]) with every other parameter at 1."""
    out = {}
    for b in sorted(L.bodies):
        be = L.beta[b]
        if np.max(np.abs(be[2:] - 1)) > 1e-6:
            continue
        i = [k for k, v in enumerate(LEGS) if abs(v - be[0]) < 1e-4]
        j = [k for k, v in enumerate(ARMS) if abs(v - be[1]) < 1e-4]
        if i and j and (i[0], j[0]) not in out:
            out[(i[0], j[0])] = b
    return out


def correction(L, stem, b):
    if (stem, b) not in _CORR:
        _CORR[(stem, b)] = log_map(z0_of(stem), L.z[(stem, b)])
    return _CORR[(stem, b)]


def bilinear(L, lat, stem, leg, arm):
    i = int(np.clip(np.searchsorted(LEGS, leg) - 1, 0, len(LEGS) - 2))
    j = int(np.clip(np.searchsorted(ARMS, arm) - 1, 0, len(ARMS) - 2))
    u = np.clip((leg - LEGS[i]) / (LEGS[i + 1] - LEGS[i]), 0, 1)
    v = np.clip((arm - ARMS[j]) / (ARMS[j + 1] - ARMS[j]), 0, 1)
    V = ((1 - u) * (1 - v) * correction(L, stem, lat[(i, j)]) + u * (1 - v) * correction(L, stem, lat[(i + 1, j)])
         + (1 - u) * v * correction(L, stem, lat[(i, j + 1)]) + u * v * correction(L, stem, lat[(i + 1, j + 1)]))
    return exp_map(z0_of(stem), V)


def nearest(lat, leg, arm):
    return min(lat.items(), key=lambda kv: (LEGS[kv[0][0]] - leg) ** 2 + (ARMS[kv[0][1]] - arm) ** 2)[1]


def kernel(L, stem, beta, h, dims, exclude=()):
    bs = [b for b in L.bodies if (stem, b) in L.z and b not in exclude]
    d = np.array([np.linalg.norm((L.beta[b] - beta)[dims]) for b in bs])
    w = np.exp(-d ** 2 / (2 * h * h)) + 1e-12
    V = np.stack([correction(L, stem, b) for b in bs])
    return exp_map(z0_of(stem), (w[:, None] * V).sum(0) / w.sum())


def lobo(L, dims, grid):
    best = None
    for h in grid:
        e = [np.degrees(np.arccos(np.clip(unit(kernel(L, s, L.beta[b], h, dims, exclude=(b,))) @ unit(L.z[(s, b)]), -1, 1)))
             for b in L.bodies for s in L.stems if (s, b) in L.z]
        print(f"    h={h:<6} LOBO median angle {np.median(e):6.2f} deg")
        if best is None or np.median(e) < best[1]:
            best = (h, float(np.median(e)))
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--plan", required=True)
    ap.add_argument("--clips", default="outputs/continuation_clips.txt")
    a = ap.parse_args()
    pre = ["m2c", "m2s", "m2g", "m2k"] + [f"ax{i}{s}" for i in range(8) for s in "mp"] + ["grid"]
    L = Library(pre, "cont")
    lat = lattice(L)
    print(f"library: {len(L.bodies)} bodies x {len(L.stems)} clips; lattice {len(lat)}/{len(LEGS) * len(ARMS)} points")
    missing = [(LEGS[i], ARMS[j]) for i in range(len(LEGS)) for j in range(len(ARMS)) if (i, j) not in lat]
    if missing:
        raise SystemExit(f"lattice incomplete, missing {missing[:5]} ...")
    limb, full = np.array([0, 1]), np.arange(8)
    print("  limb-distance kernel:"); h_limb = lobo(L, limb, [0.03, 0.05, 0.08, 0.12, 0.2])
    print("  8-D kernel:"); h_full = lobo(L, full, [0.03, 0.05, 0.08, 0.12, 0.2])
    print(f"  chosen h: limb {h_limb}, 8-D {h_full}")
    clips = [l.strip() for l in open(REPO / a.clips) if l.strip()]
    C, B, Lb, Z, plan = [], [], [], [], []
    for c in clips:
        stem = c.split("/")[-1]
        for tb in REAL:
            beta = load_beta(REPO / f"assets/robots/{tb}/parameter.json")
            nb = nearest(lat, beta[0], beta[1])
            preds = {"grid:bilinear": bilinear(L, lat, stem, beta[0], beta[1]),
                     "grid:nn": L.z[(stem, nb)],
                     "limb:kernel": kernel(L, stem, beta, h_limb, limb),
                     "all:kernel": kernel(L, stem, beta, h_full, full)}
            for k, z in preds.items():
                C.append(c); B.append(tb); Lb.append(k); Z.append(unit(z) * 16)
            if tb not in SAME_XML:
                dl = float(np.hypot(L.beta[nb][0] - beta[0], L.beta[nb][1] - beta[1]))
                plan.append((c, tb, nb, dl, str(REPO / f"outputs/continuation/{_path_of(nb)}/cont/{stem}__{nb}/best_z.npy")))
    Path(REPO / a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(REPO / a.out, clip=np.array(C), body=np.array(B), label=np.array(Lb), z=np.stack(Z))
    with open(REPO / a.plan, "w") as f:
        for r in plan:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"-> {a.out}: {len(C)} jobs | -> {a.plan}: {len(plan)} warm starts")
    for tb in REAL:
        if tb in SAME_XML: continue
        beta = load_beta(REPO / f"assets/robots/{tb}/parameter.json"); nb = nearest(lat, beta[0], beta[1])
        print(f"  {tb:22s} leg {beta[0]:.2f} arm {beta[1]:.2f} -> nearest lattice {nb:12s} "
              f"limb dist {np.hypot(L.beta[nb][0] - beta[0], L.beta[nb][1] - beta[1]):.3f}")


def _path_of(body):
    if body.startswith("gl"):
        return "grid"
    return body.rsplit("_t", 1)[0]


if __name__ == "__main__":
    main()
