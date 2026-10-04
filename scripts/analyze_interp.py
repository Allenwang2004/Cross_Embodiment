#!/usr/bin/env python3
"""analyze_interp.py -- leave-one-body-out prediction of z*(beta), scored by rollout."""
import argparse, csv, collections, statistics as st
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); a = ap.parse_args()
rows = list(csv.DictReader(open(REPO / a.csv)))
z0 = {(r["clip"], r["body"]): float(r["cost"]) for r in rows if r["label"] == "z0"}
R = collections.defaultdict(dict)
for r in rows:
    if r["label"] == "z0": continue
    R[(r["clip"], r["body"])][r["label"]] = float(r["cost"]) / z0[(r["clip"], r["body"])]
labs = sorted({l for v in R.values() for l in v})
def t(b): return int(b.rsplit("_t", 1)[1])
interior = [k for k in R if 0 < t(k[1]) < 1000]
print(f"held-out cells: {len(R)} (interior {len(interior)})\n")
print(f"{'predictor':22s} {'median':>7s} {'mean':>7s} {'<1':>6s}   {'interior median':>15s}")
for l in labs:
    v = [R[k][l] for k in R if l in R[k]]
    vi = [R[k][l] for k in interior if l in R[k]]
    print(f"{l:22s} {st.median(v):7.3f} {st.mean(v):7.3f} {sum(x<1 for x in v)/len(v):6.0%}   {st.median(vi):15.3f}")
# head to head: does interpolation beat the nearest neighbour on the SAME cell?
for m in ("cont", "indep"):
    for l in (f"{m}:slerp", f"{m}:kernel0.05", f"{m}:kernel0.1", f"{m}:kernel0.2"):
        pairs = [(R[k][l], R[k][f"{m}:nn"]) for k in R if l in R[k] and f"{m}:nn" in R[k]]
        if pairs:
            w = sum(x < y for x, y in pairs)
            print(f"  {l:18s} beats {m}:nn on {w}/{len(pairs)} cells")
