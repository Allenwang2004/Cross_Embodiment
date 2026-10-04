#!/usr/bin/env python3
"""plot_real_bodies.py -- the held-out real bodies, one row each, named uniformly.

Same data as analyze_bnn.py (warm start from the nearest library body vs from
z0), drawn as a dot plot so the body names sit on the y axis instead of on top of
each other: rows sorted by beta distance to the nearest library body, split into
near (<= 0.2) and far. Left: after the same search budget, the warm start's final
cost over the cold start's (< 1 = the warm start ended better). Right: the nearest
body's latent used as is, cost / z0.

Bodies whose XML is byte-identical to their nearest library body are not held out
and are dropped, as in analyze_map_real.py.
"""
import argparse, csv, filecmp, json, statistics as st, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
from model.dataset import load_beta

SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, AQUA = "#2a78d6", "#1baf7a"
# one naming scheme for every body: a plain description, then its leg length
DESC = {
    "teen": "teen", "petite": "petite", "elderly": "elderly", "athletic": "athletic",
    "pear_shaped": "pear-shaped", "tall_slim": "tall, slim", "long_limbed": "long-limbed",
    "x_leg050": "short-legged", "x_leg055_adulttorso": "short-legged, adult torso",
    "x_leg060_longtorso": "short-legged, long torso", "x_leg062_heavy": "short-legged, heavy",
    "x_leg070": "short-legged", "x_leg075_heavy": "short-legged, heavy",
    "x_leg140": "long-legged", "x_leg150_thin": "long-legged, thin",
}

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="outputs/morph_study/plan_bnn_axis.tsv")
ap.add_argument("--dir", default="outputs/bnn/axis_lib")
ap.add_argument("--thr", type=float, default=0.7)
ap.add_argument("--near", type=float, default=0.2)
ap.add_argument("--out", default="outputs/morph_study/real_bodies.png")
a = ap.parse_args()


def load(d):
    s = json.loads((d / "summary.json").read_text())
    r = list(csv.DictReader(open(d / "curve.csv")))
    return s, np.array([int(x["evals"]) for x in r]), np.array([float(x["best_so_far"]) for x in r])


def first(ev, bs, t):
    i = np.where(bs <= t)[0]
    return int(ev[i[0]]) if len(i) else None


def name(b):
    return f"{DESC.get(b, b)} · legs {load_beta(REPO / f'assets/robots/{b}/parameter.json')[0]:.2f}×"


rows = [l.rstrip("\n").split("\t") for l in open(REPO / a.plan) if l.strip()]
same = {b for _, b, n, _, _ in rows if filecmp.cmp(REPO / f"assets/robots/{b}/robot.xml",
                                                   REPO / f"assets/robots/{n}/robot.xml", shallow=False)}
D = REPO / a.dir
cells = {}
for clip, body, nb, dist, _ in rows:
    if body in same:
        continue
    stem = clip.split("/")[-1]
    dc, dw = D / "cold" / f"{stem}__{body}", D / "warm" / f"{stem}__{body}"
    if not ((dc / "summary.json").exists() and (dw / "summary.json").exists()):
        continue
    sc, ec, bc = load(dc); sw, ew, bw = load(dw)
    o = sc["origin_z"]["cost"]; budget = int(max(ec[-1], ew[-1]))
    fc, fw = first(ec, bc, a.thr * o), first(ew, bw, a.thr * o)
    cells.setdefault(body, {"dist": float(dist), "sp": [], "prior": []})
    cells[body]["sp"].append(sw["best"]["cost"] / sc["best"]["cost"])
    cells[body]["prior"].append(bw[0] / o)
print("excluded (XML identical to a library body):", " ".join(sorted(same)))
order = sorted(cells, key=lambda b: cells[b]["dist"])
print(f"{len(order)} held-out bodies, {sum(len(c['sp']) for c in cells.values())} cells")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n = len(order)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 0.36 * n + 1.6), sharey=True)
fig.patch.set_facecolor(SURF)
y = {b: n - 1 - i for i, b in enumerate(order)}
rng = np.random.default_rng(0)
for b in order:
    c = cells[b]; jit = rng.uniform(-0.18, 0.18, len(c["sp"]))
    ax[0].scatter(c["sp"], y[b] + jit, s=16, color=BLUE, alpha=.5, edgecolor="none")
    ax[0].scatter(st.median(c["sp"]), y[b], s=150, color=INK, marker="|", lw=2.6)
    ax[1].scatter(c["prior"], y[b] + jit, s=16, color=AQUA, alpha=.55, edgecolor="none")
    ax[1].scatter(st.median(c["prior"]), y[b], s=150, color=INK, marker="|", lw=2.6)
ax[0].set_yticks([y[b] for b in order])
ax[0].set_yticklabels([f"{name(b)}   d={cells[b]['dist']:.2f}" for b in order], fontsize=9, color=INK)
split = sum(cells[b]["dist"] <= a.near for b in order)
for x in ax:
    x.axhline(n - split - 0.5, color=INK3, lw=0.9)
    x.axvline(1, color=INK2, ls=(0, (4, 3)), lw=1.1)
    x.set_facecolor(SURF); x.grid(axis="x", alpha=.22, color=INK3, lw=.7)
    for s in ("top", "right"): x.spines[s].set_visible(False)
    x.set_ylim(-0.7, n - 0.3)
ax[0].text(1.02, n - 0.6, f"near (d ≤ {a.near})", transform=ax[0].get_yaxis_transform(),
           fontsize=8.5, color=INK2, va="top")
ax[0].text(1.02, n - split - 0.9, f"far (d > {a.near})", transform=ax[0].get_yaxis_transform(),
           fontsize=8.5, color=INK2, va="top")
ax[0].set_xscale("log")
from matplotlib.ticker import NullFormatter
ax[0].set_xticks([0.25, 0.5, 1, 2, 4]); ax[0].set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax[0].xaxis.set_minor_formatter(NullFormatter())
ax[0].set_xlabel("final cost, warm start / cold start (log; <1 = warm start ended better)")
ax[0].set_title("after 1,024 rollouts: search from the nearest library body vs from z0", fontsize=11)
ax[1].set_xlabel("cost / z0 cost (<1 = better than z0)")
ax[1].set_title("nearest library body's latent used as is (no search)", fontsize=11)
fig.text(0.01, 0.005, "d = beta distance to the nearest library body · dots = clips · bar = median",
         fontsize=8, color=INK3)
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = REPO / a.out
fig.savefig(out, dpi=140, facecolor=SURF); print(f"-> {out}")
