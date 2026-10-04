#!/usr/bin/env python3
"""build_interp_jobs.py -- can z*(beta) be INTERPOLATED, or only looked up?

Leave one body of a morph path out and predict the clip's latent there from the
other bodies' searched solutions:
  nn      the nearest other body's z* (ties -> the one nearer the adult)
  slerp   the spherical midpoint of the two neighbours' z* (interior bodies only)
  kernel  a beta-distance-weighted spherical mean over ALL other bodies,
          bandwidth h (Gaussian in ||dbeta||)
Averaging two good latents can land between their basins -- the knife-edge
failure seen on headstand -- so this is the test of whether a CONTINUOUS
beta -> latent map is even possible, before anything is trained.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
from model.dataset import load_beta
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True); ap.add_argument("--method", default="cont")
ap.add_argument("--h", type=float, nargs="+", default=[0.05, 0.1, 0.2])
ap.add_argument("--out", required=True)
a = ap.parse_args()
root = REPO / "outputs/continuation" / a.prefix / a.method
bodies = sorted({p.name.split("__")[1] for p in root.iterdir()}, key=lambda b: int(b.rsplit("_t", 1)[1]))
stems = sorted({p.name.split("__")[0] for p in root.iterdir()})
beta = {b: load_beta(REPO / f"assets/robots/{b}/parameter.json") for b in bodies}
def unit(v): return v / np.linalg.norm(v)
def slerp(a_, b_, t=.5):
    a_, b_ = unit(a_), unit(b_); om = np.arccos(np.clip(a_ @ b_, -1, 1))
    if om < 1e-6: return a_
    return (np.sin((1 - t) * om) * a_ + np.sin(t * om) * b_) / np.sin(om)
def sph_mean(Z, w):
    m = unit((w[:, None] * np.stack([unit(z) for z in Z])).sum(0))
    for _ in range(20):                     # a few Karcher iterations on the sphere
        v = []
        for z in Z:
            z = unit(z); c = np.clip(m @ z, -1, 1); th = np.arccos(c)
            v.append(np.zeros_like(m) if th < 1e-8 else th / np.sin(th) * (z - c * m))
        step = (w[:, None] * np.stack(v)).sum(0) / w.sum()
        n = np.linalg.norm(step)
        if n < 1e-7: break
        m = np.cos(n) * m + np.sin(n) * step / n
    return m
C, B, L, Z = [], [], [], []
for s in stems:
    task = s.rsplit("_", 1)[0]
    zs = {b: np.load(root / f"{s}__{b}" / "best_z.npy").reshape(-1).astype(np.float64) for b in bodies
          if (root / f"{s}__{b}" / "best_z.npy").exists()}
    if len(zs) < len(bodies): continue
    for k, bt in enumerate(bodies):
        others = [b for b in bodies if b != bt]
        d = {b: float(np.linalg.norm(beta[b] - beta[bt])) for b in others}
        nn = min(others, key=lambda b: (d[b], int(b.rsplit("_t", 1)[1])))
        jobs = [("nn", zs[nn])]
        if 0 < k < len(bodies) - 1:
            jobs.append(("slerp", slerp(zs[bodies[k - 1]], zs[bodies[k + 1]])))
        for h in a.h:
            w = np.array([np.exp(-d[b] ** 2 / (2 * h * h)) for b in others])
            jobs.append((f"kernel{h}", sph_mean([zs[b] for b in others], w)))
        jobs.append(("truth", zs[bt]))            # its own search result, for reference
        for lab, z in jobs:
            C.append(f"{task}/{s}"); B.append(bt); L.append(f"{a.method}:{lab}"); Z.append(unit(z) * 16)
np.savez(REPO / a.out, clip=np.array(C), body=np.array(B), label=np.array(L), z=np.stack(Z))
print(f"{len(C)} jobs")
