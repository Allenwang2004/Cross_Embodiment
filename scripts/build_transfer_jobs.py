#!/usr/bin/env python3
"""build_transfer_jobs.py -- every searched z* on a morph path, rolled out on
every body of that path.

z*_m(t_i) is the best latent method m (cont / indep) found for a clip on body
t_i. Rolling it out on body t_j answers: how far in body space does a latent
adapted to one body still work? If quality decays smoothly with |t_i - t_j|,
there is morphological structure to amortise. The same measurement across
CLIPS on a fixed body was ~2x worse than z0 at every distance, so the contrast
is the point.
"""
import argparse, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--methods", nargs="+", default=["cont", "indep"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
root = REPO / "outputs/continuation" / a.prefix
bodies = sorted({p.name.split("__")[1] for p in (root / "cont").iterdir()},
                key=lambda b: int(b.rsplit("_t", 1)[1]))
clips = sorted({p.name.split("__")[0] for p in (root / "cont").iterdir()})
C, B, L, Z = [], [], [], []
for c in clips:
    task = c.rsplit("_", 1)[0]
    for m in a.methods:
        for bi in bodies:
            f = root / m / f"{c}__{bi}" / "best_z.npy"
            if not f.exists():
                continue
            z = np.load(f).reshape(-1)
            for bj in bodies:
                C.append(f"{task}/{c}"); B.append(bj); L.append(f"{m}@{bi}"); Z.append(z)
np.savez(a.out, clip=np.array(C), body=np.array(B), label=np.array(L), z=np.stack(Z))
print(f"{len(C)} jobs: {len(clips)} clips x {len(a.methods)} methods x {len(bodies)} sources x {len(bodies)} targets")
