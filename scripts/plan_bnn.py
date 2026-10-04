#!/usr/bin/env python3
"""plan_bnn.py -- the morphology-nearest-neighbour warm start, planned.

A library of already-adapted latents: z*(clip, body) for every body on the morph
paths (their continuation solutions). For a NEW body B -- one not on any path --
start the latent search from the library solution of the body nearest to B in
beta, and compare with starting from z0. Nothing is learned: the only thing
used is the distance between body-parameter vectors, so whatever speed-up
appears is attributable to morphological proximity alone.

Writes a job list for scripts/run_bnn.sh: clip, test body, nearest library
body, beta distance, z_start path.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
from model.dataset import load_beta

ap = argparse.ArgumentParser()
ap.add_argument("--prefixes", nargs="+", default=["m2c", "m2s", "m2g", "m2k"])
ap.add_argument("--clips", default="outputs/continuation_clips.txt")
ap.add_argument("--test-bodies", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
clips = [l.strip() for l in open(REPO / a.clips) if l.strip()]
lib = {}                                  # (stem, body) -> path of best_z
for p in a.prefixes:
    for f in glob.glob(str(REPO / f"outputs/continuation/{p}/cont/*__*/best_z.npy")):
        stem, body = Path(f).parent.name.split("__")
        lib[(stem, body)] = f
lib_bodies = sorted({b for _, b in lib})
B = {b: load_beta(REPO / f"assets/robots/{b}/parameter.json") for b in lib_bodies + a.test_bodies}
rows = []
for c in clips:
    stem = c.split("/")[-1]
    have = [b for b in lib_bodies if (stem, b) in lib]
    if not have:
        continue
    for tb in a.test_bodies:
        d = {b: float(np.linalg.norm(B[tb] - B[b])) for b in have}
        nb = min(d, key=d.get)
        rows.append((c, tb, nb, d[nb], lib[(stem, nb)]))
with open(REPO / a.out, "w") as f:
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(f"{len(rows)} jobs ({len(clips)} clips x {len(a.test_bodies)} test bodies), "
      f"library: {len(lib_bodies)} bodies from {', '.join(a.prefixes)}")
for tb in a.test_bodies:
    r = next(x for x in rows if x[1] == tb)
    print(f"  {tb:22s} nearest library body {r[2]:12s} beta dist {r[3]:.3f}")
