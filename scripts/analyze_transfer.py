#!/usr/bin/env python3
"""analyze_transfer.py -- how far in body space does an adapted latent reach?

transfer_<prefix>.csv holds, for every clip, every searched z* (label
'<method>@<source body>') rolled out on every body of the path, plus each
cell's own z0. R[i, j] = cost(z*(t_i) on t_j) / cost(z0 on t_j): below 1 means
the latent adapted to body i still beats doing nothing on body j.

The contrast that matters is with the CLIP axis. On a fixed body, giving one
clip's searched z to another clip was measured at a median of 1.99x z0 (worse
than nothing at almost any pair). If the BODY axis instead shows transfer that
decays smoothly with morphological distance, morphology -- not motion -- is the
axis along which adaptation can be shared.
"""
import argparse, csv, collections, statistics as st
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
C_IND, C_CONT, C_CLIP = "#eb6834", "#2a78d6", "#e34948"

ap = argparse.ArgumentParser(); ap.add_argument("--prefix", required=True); a = ap.parse_args()
rows = list(csv.DictReader(open(REPO / f"outputs/morph_study/transfer_{a.prefix}.csv")))
bodies = sorted({r["body"] for r in rows}, key=lambda b: int(b.rsplit("_t", 1)[1]))
idx = {b: i for i, b in enumerate(bodies)}
n = len(bodies)
z0c = {(r["clip"], r["body"]): float(r["cost"]) for r in rows if r["label"] == "z0"}
M = collections.defaultdict(lambda: np.full((n, n), np.nan))
for r in rows:
    if r["label"] == "z0": continue
    m, src = r["label"].split("@")
    M[(r["clip"], m)][idx[src], idx[r["body"]]] = float(r["cost"]) / z0c[(r["clip"], r["body"])]
clips = sorted({c for c, _ in M})
print(f"{len(clips)} clips, {n} bodies\n")
by_d = {m: collections.defaultdict(list) for m in ("cont", "indep")}
beat = {m: collections.defaultdict(lambda: [0, 0]) for m in ("cont", "indep")}
for (c, m), R in M.items():
    for i in range(n):
        for j in range(n):
            if np.isnan(R[i, j]): continue
            d = abs(i - j)
            by_d[m][d].append(R[i, j]); beat[m][d][0] += R[i, j] < 1; beat[m][d][1] += 1
print(f"{'|i-j| (path steps)':>18s} {'beta dist':>9s} | {'cont median':>11s} {'<1':>6s} | {'indep median':>12s} {'<1':>6s}")
import sys; sys.path.append(str(REPO))
from model.dataset import load_beta
_b0 = load_beta(REPO / f"assets/robots/{bodies[0]}/parameter.json")
_b1 = load_beta(REPO / f"assets/robots/{bodies[-1]}/parameter.json")
step = float(np.linalg.norm(_b1 - _b0)) / (n - 1)     # beta length of one path step
for d in range(n):
    c, i = by_d["cont"][d], by_d["indep"][d]
    print(f"{d:18d} {d*step:9.3f} | {st.median(c):11.3f} {beat['cont'][d][0]/beat['cont'][d][1]:6.0%} | "
          f"{st.median(i):12.3f} {beat['indep'][d][0]/beat['indep'][d][1]:6.0%}")
print(f"\nfor reference -- the CLIP axis on a fixed body (headstand, 10 clips): "
      f"off-diagonal median 1.987x z0, 21/90 = 23% beat z0")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
fig.patch.set_facecolor(SURF)
D = np.arange(n)
for m, col, lab in (("cont", C_CONT, "continuation z*"), ("indep", C_IND, "independent z*")):
    med = [st.median(by_d[m][d]) for d in D]
    lo = [np.percentile(by_d[m][d], 25) for d in D]; hi = [np.percentile(by_d[m][d], 75) for d in D]
    ax[0].plot(D * step, med, color=col, lw=2.4, marker="o", ms=6, label=lab)
    ax[0].fill_between(D * step, lo, hi, color=col, alpha=.14)
ax[0].axhline(1.987, color=C_CLIP, ls=(0, (5, 3)), lw=1.8, label="another CLIP's z* (fixed body)")
ax[0].axhline(1.0, color=INK2, ls=(0, (4, 3)), lw=1.1)
ax[0].text(ax[0].get_xlim()[1] if False else D[-1]*step, 1.02, " z0", color=INK2, fontsize=9, va="bottom", ha="right")
ax[0].set_xlabel("morphological distance between source and target body (||delta beta||)")
ax[0].set_ylabel("cost / target body's z0 cost  (median, band = IQR)")
ax[0].set_title("a latent adapted to one body, used on another"); ax[0].legend(fontsize=9)
for k, m in enumerate(("cont", "indep")):
    Ravg = np.nanmedian(np.stack([M[(c, m)] for c in clips]), axis=0)
    im = ax[1 + k].imshow(np.clip(Ravg, 0, 2), cmap="RdYlGn_r", vmin=0, vmax=2)
    ax[1 + k].set_xticks(range(n)); ax[1 + k].set_xticklabels([f"{int(b.rsplit('_t',1)[1])/1000:.2f}" for b in bodies], fontsize=7, rotation=60)
    ax[1 + k].set_yticks(range(n)); ax[1 + k].set_yticklabels([f"{int(b.rsplit('_t',1)[1])/1000:.2f}" for b in bodies], fontsize=7)
    ax[1 + k].set_xlabel("rolled out on body t"); ax[1 + k].set_ylabel("z* searched on body t")
    ax[1 + k].set_title(f"{'continuation' if m=='cont' else 'independent'} (median over clips)", fontsize=10.5)
    for i in range(n):
        for j in range(n):
            ax[1 + k].text(j, i, f"{Ravg[i,j]:.2f}", ha="center", va="center", fontsize=6.2,
                           color="#000" if .35 < Ravg[i, j] < 1.5 else "#fff")
fig.colorbar(im, ax=ax[2], fraction=.046, label="cost / z0 cost")
for x in (ax[0],):
    x.set_facecolor(SURF); x.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
fig.suptitle(f"cross-body transfer along {a.prefix} ({len(clips)} clips)", fontsize=12.5, color=INK, y=.99)
fig.tight_layout(rect=(0, 0, 1, .93))
out = REPO / f"outputs/morph_study/transfer_{a.prefix}.png"; fig.savefig(out, dpi=140, facecolor=SURF)
print(f"-> {out}")
