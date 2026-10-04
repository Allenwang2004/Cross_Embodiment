#!/usr/bin/env python3
"""analyze_bnn.py -- does the morphologically nearest known body's latent speed up
adaptation to a NEW body, and how does the effect depend on the distance?"""
import argparse, csv, json, collections, filecmp, statistics as st
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True); ap.add_argument("--dir", required=True)
ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--out", default=None)
# a test body whose XML is byte-identical to its nearest library body is not held out
ap.add_argument("--held-out-only", action="store_true")
a = ap.parse_args()

def load(d):
    s = json.loads((d / "summary.json").read_text())
    r = list(csv.DictReader(open(d / "curve.csv")))
    ev = np.array([int(x["evals"]) for x in r]); bs = np.array([float(x["best_so_far"]) for x in r])
    return s, ev, bs

def first(ev, bs, t):
    i = np.where(bs <= t)[0]
    return int(ev[i[0]]) if len(i) else None

D = REPO / a.dir
rows = [l.rstrip("\n").split("\t") for l in open(REPO / a.plan) if l.strip()]
res = []
if a.held_out_only:
    same = {b for _, b, n, _, _ in rows if filecmp.cmp(REPO / f"assets/robots/{b}/robot.xml",
                                                     REPO / f"assets/robots/{n}/robot.xml", shallow=False)}
    print("excluded (XML identical to a library body):", " ".join(sorted(same)))
    rows = [r for r in rows if r[1] not in same]
for clip, body, nb, dist, _ in rows:
    stem = clip.split("/")[-1]
    dc, dw = D / "cold" / f"{stem}__{body}", D / "warm" / f"{stem}__{body}"
    if not ((dc / "summary.json").exists() and (dw / "summary.json").exists()):
        continue
    sc, ec, bc = load(dc); sw, ew, bw = load(dw)
    o = sc["origin_z"]["cost"]
    budget = int(max(ec[-1], ew[-1]))
    fc, fw = first(ec, bc, a.thr * o), first(ew, bw, a.thr * o)
    res.append(dict(stem=stem, body=body, nb=nb, dist=float(dist), z0=o,
                    prior=bw[0] / o,                       # the warm start's own cost (zero-shot)
                    cold=sc["best"]["cost"] / o, warm=sw["best"]["cost"] / o,
                    fc=fc, fw=fw, budget=budget))
print(f"{len(res)} (clip, body) pairs complete\n")
by = collections.defaultdict(list)
for r in res: by[r["body"]].append(r)
print(f"{'test body':14s} {'dist':>5s} {'nearest lib':>11s} | {'prior':>6s} | {'cold':>6s} {'warm':>6s} | "
      f"{f'->{a.thr}x cold':>12s} {'warm':>6s} {'speedup':>8s}")
agg = []
for b, L in sorted(by.items(), key=lambda kv: kv[1][0]["dist"]):
    fcs = [r["fc"] if r["fc"] else r["budget"] for r in L]    # censored at the budget
    fws = [r["fw"] if r["fw"] else r["budget"] for r in L]
    cens = sum(1 for r in L if r["fc"] is None)
    sp = [c / w for c, w in zip(fcs, fws)]
    agg.append((L[0]["dist"], st.median(sp), b))
    print(f"{b:14s} {L[0]['dist']:5.3f} {L[0]['nb']:>11s} | {st.median([r['prior'] for r in L]):6.3f} | "
          f"{st.median([r['cold'] for r in L]):6.3f} {st.median([r['warm'] for r in L]):6.3f} | "
          f"{st.median(fcs):>12.0f} {st.median(fws):6.0f} {st.median(sp):7.1f}x"
          + (f"  ({cens} cold censored at budget)" if cens else ""))
allsp = []
for r in res:
    c = r["fc"] if r["fc"] else r["budget"]; w = r["fw"] if r["fw"] else r["budget"]
    allsp.append(c / w)
print(f"\nall pairs: median speedup {st.median(allsp):.1f}x, mean prior cost {st.mean([r['prior'] for r in res]):.3f} x z0 "
      f"(zero rollouts), warm beats cold at the end in {sum(r['warm']<r['cold'] for r in res)}/{len(res)}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6)); fig.patch.set_facecolor(SURF)
for r in res:
    c = r["fc"] if r["fc"] else r["budget"]; w = r["fw"] if r["fw"] else r["budget"]
    ax[0].scatter(r["dist"], c / w, s=28, color="#2a78d6", alpha=.55, edgecolor=SURF, lw=.6)
    ax[1].scatter(r["dist"], r["prior"], s=28, color="#1baf7a", alpha=.55, edgecolor=SURF, lw=.6)
for d, m, b in agg:
    ax[0].scatter(d, m, s=90, color="#0b0b0b", marker="_", lw=3)
    ax[0].annotate(b, (d, m), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK2)
ax[0].axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.1); ax[0].set_yscale("log")
ax[0].set_xlabel("beta distance to the nearest body with a known latent")
ax[0].set_ylabel(f"speed-up in rollouts to reach {a.thr} x z0 cost (log)")
ax[0].set_title("warm start from the nearest known body vs from z0")
ax[1].axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.1)
ax[1].set_xlabel("beta distance to the nearest body with a known latent")
ax[1].set_ylabel("cost of that body's latent, used as is / z0 cost")
ax[1].set_title("zero rollouts: the nearest body's latent as the answer")
for x in ax:
    x.set_facecolor(SURF); x.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
fig.tight_layout()
out = Path(a.out or (REPO / "outputs/morph_study" / f"bnn_{Path(a.dir).name}.png"))
fig.savefig(out, dpi=140, facecolor=SURF); print(f"-> {out}")
