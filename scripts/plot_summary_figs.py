#!/usr/bin/env python3
"""Two summary figures for the research narrative:
  summary_transfer.png   cross-body transfer on all four morph paths vs cross-clip
  summary_sensitivity.png  per-dimension sensitivity from the 16 axis paths"""
import csv, collections, statistics as st, json, glob, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parent.parent; sys.path.append(str(REPO))
from model.dataset import load_beta
OUT = REPO / "docs/figures/journey"; OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
COL = {"m2c": "#2a78d6", "m2s": "#1baf7a", "m2g": "#eda100", "m2k": "#e87ba4"}
NAME = {"m2c": "shrink all (-> child)", "m2s": "shorter limbs", "m2g": "grow all (-> giant)", "m2k": "thicker (-> stocky)"}

def style(ax):
    ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK3)

# ---------- 1. transfer
fig, ax = plt.subplots(figsize=(8.6, 5.0)); fig.patch.set_facecolor(SURF)
for p in ("m2c", "m2s", "m2g", "m2k"):
    rows = list(csv.DictReader(open(REPO / f"outputs/morph_study/transfer_{p}.csv")))
    bodies = sorted({r["body"] for r in rows}, key=lambda b: int(b.rsplit("_t", 1)[1]))
    idx = {b: i for i, b in enumerate(bodies)}
    step = float(np.linalg.norm(load_beta(REPO / f"assets/robots/{bodies[-1]}/parameter.json")
                                - load_beta(REPO / f"assets/robots/{bodies[0]}/parameter.json"))) / (len(bodies) - 1)
    z0 = {(r["clip"], r["body"]): float(r["cost"]) for r in rows if r["label"] == "z0"}
    by = collections.defaultdict(list)
    for r in rows:
        if not r["label"].startswith("cont@"): continue
        src = r["label"].split("@")[1]
        by[abs(idx[src] - idx[r["body"]])].append(float(r["cost"]) / z0[(r["clip"], r["body"])])
    D = sorted(by)
    ax.plot([d * step for d in D], [st.median(by[d]) for d in D], color=COL[p], lw=2.3, marker="o", ms=5, label=NAME[p])
ax.axhline(1.0, color=INK2, ls=(0, (4, 3)), lw=1.1)
ax.text(0.02, 1.03, "no adaptation (z0)", color=INK2, fontsize=9, va="bottom")
ax.set_ylim(0, 1.15)
ax.set_xlabel("morphological distance between source and target body  ||delta beta||", fontsize=10)
ax.set_ylabel("cost / target body's z0 cost   (median, 8 clips)", fontsize=10)
ax.set_title("A latent adapted for one body, used on another body:\nit carries over to nearby bodies and fades to z0's level with distance", fontsize=11.5, color=INK, loc="left")
ax.legend(fontsize=9, loc="lower right"); style(ax)
fig.tight_layout(); fig.savefig(OUT / "summary_transfer.png", dpi=150, facecolor=SURF)

# ---------- 2. sensitivity
names = ["leg length", "arm length", "torso length", "head size", "leg girth", "arm girth", "torso girth", "head girth"]
tg = {}
for l in open(REPO / "outputs/morph_study/axis_targets.tsv"):
    p, v = l.rstrip("\n").split("\t"); tg[p] = [float(x) for x in v.split()]
vals = {}
for d in range(8):
    for s in "mp":
        p = f"ax{d}{s}"; r = []; post = []
        for f in glob.glob(str(REPO / f"outputs/continuation/{p}/cont/*__{p}_t1000/summary.json")):
            stem = Path(f).parent.name.split("__")[0]
            S = json.load(open(f)); A = json.load(open(REPO / f"outputs/continuation/m2c/cont/{stem}__m2c_t000/summary.json"))
            r.append(S["origin_z"]["cost"] / A["origin_z"]["cost"]); post.append(S["best"]["cost"] / A["origin_z"]["cost"])
        vals[p] = (st.median(r), st.median(post))
fig, ax = plt.subplots(figsize=(9.6, 4.8)); fig.patch.set_facecolor(SURF)
x = np.arange(8); w = .38
m = [vals[f"ax{d}m"][0] for d in range(8)]; pp = [vals[f"ax{d}p"][0] for d in range(8)]
b1 = ax.bar(x - w / 2, m, w, color="#2a78d6", label="dimension at its SMALLEST real-body value")
b2 = ax.bar(x + w / 2, pp, w, color="#eb6834", label="dimension at its LARGEST real-body value")
for i in range(8):
    ax.text(x[i] - w / 2, m[i] + .04, f"{tg[f'ax{i}m'][i]:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.text(x[i] + w / 2, pp[i] + .04, f"{tg[f'ax{i}p'][i]:.2f}", ha="center", fontsize=7.5, color=INK2)
ax.axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.1)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("z0 cost on the changed body / on the adult\n(median, 8 clips)", fontsize=10)
ax.set_title("Only one dimension changed at a time: the latent is sensitive to LIMB LENGTH,\nand shrinking hurts far more than growing (numbers = value the dimension was set to)", fontsize=11, color=INK, loc="left")
ax.legend(fontsize=9); style(ax)
fig.tight_layout(); fig.savefig(OUT / "summary_sensitivity.png", dpi=150, facecolor=SURF)
print("ok")
