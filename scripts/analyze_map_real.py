#!/usr/bin/env python3
"""analyze_map_real.py -- zero-shot beta->latent predictors on the real bodies.
Excludes bodies whose XML is identical to a library body (not held out)."""
import argparse, csv, collections, statistics as st, json, sys, filecmp
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--plan", required=True)
ap.add_argument("--truth-dirs", nargs="+", default=["outputs/bnn/all_lib", "outputs/bnn/axis_lib"])
a = ap.parse_args()
rows = list(csv.DictReader(open(REPO / a.csv)))
z0 = {(r["clip"], r["body"]): float(r["cost"]) for r in rows if r["label"] == "z0"}
R = collections.defaultdict(dict)
for r in rows:
    if r["label"] != "z0": R[(r["clip"], r["body"])][r["label"]] = float(r["cost"]) / z0[(r["clip"], r["body"])]
dist, nb = {}, {}
for l in open(REPO / a.plan):
    c, b, n, d, _ = l.rstrip("\n").split("\t"); dist[(c, b)] = float(d); nb[b] = n
same = {b for b, n in nb.items() if filecmp.cmp(REPO / f"assets/robots/{b}/robot.xml", REPO / f"assets/robots/{n}/robot.xml", shallow=False)}
T = {}
for (c, b) in R:
    s = c.split("/")[-1]; v = []
    for d in a.truth_dirs:
        for m in ("cold", "warm"):
            p = REPO / d / m / f"{s}__{b}" / "summary.json"
            if p.exists():
                S = json.loads(p.read_text()); v.append(S["best"]["cost"] / S["origin_z"]["cost"])
    if v: T[(c, b)] = min(v)
labs = sorted({l for v in R.values() for l in v})
def summ(keys, title):
    print(f"\n{title}  ({len(keys)} cells)")
    print(f"  {'predictor':14s} {'median':>7s} {'<1':>5s} {'gain recovered':>15s}")
    for l in labs:
        v = [R[k][l] for k in keys if l in R[k]]
        g = [(1 - R[k][l]) / (1 - T[k]) for k in keys if l in R[k] and k in T and T[k] < 0.95]
        print(f"  {l:14s} {st.median(v):7.3f} {sum(x < 1 for x in v) / len(v):5.0%} {st.median(g):14.0%}")
    print(f"  {'search (truth)':14s} {st.median([T[k] for k in keys if k in T]):7.3f}")
print("excluded (XML identical to a library body):", " ".join(sorted(same)))
K = [k for k in R if k[1] not in same]
summ(K, "held-out real bodies")
summ([k for k in K if dist[k] <= 0.2], "near: nearest-library dist <= 0.2")
summ([k for k in K if dist[k] > 0.2], "far: dist > 0.2")
