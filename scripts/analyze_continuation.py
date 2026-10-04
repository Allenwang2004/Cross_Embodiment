#!/usr/bin/env python3
"""analyze_continuation.py -- is the best latent a CURVE in body space?

Reads outputs/continuation/<prefix>/{cont,indep}/<clip>__<body>/ and asks, per
clip and pooled:

  cost      z0 (the naive transfer), independent search, continuation --
            how fast does naive transfer degrade with morphological distance,
            and does following the path cost anything in quality?
  step      the angle between z* on neighbouring bodies. A smooth function of
            beta has small steps that shrink with the beta step; arbitrary
            picks from a large low-cost set jump by tens of degrees whatever
            the beta step.
  evals     rollouts to reach half of z0's cost -- does starting from the
            neighbouring body's answer buy search?
"""
from __future__ import annotations
import argparse, csv, json, glob, os, collections, statistics as st
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
C_Z0, C_IND, C_CONT = "#b8b7b0", "#eb6834", "#2a78d6"


def unit(v): return v / np.linalg.norm(v)
def ang(a, b): return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def load(d):
    s = json.loads((d / "summary.json").read_text())
    rows = list(csv.DictReader(open(d / "curve.csv")))
    ev = np.array([int(r["evals"]) for r in rows]); bs = np.array([float(r["best_so_far"]) for r in rows])
    return dict(best=s["best"]["cost"], z0c=s["origin_z"]["cost"],
                z=np.load(d / "best_z.npy").reshape(-1).astype(np.float64), ev=ev, bs=bs)


def evals_to(ev, bs, thr):
    i = np.where(bs <= thr)[0]
    return int(ev[i[0]]) if len(i) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = REPO_ROOT / "outputs/continuation" / a.prefix
    # numeric order on t: a string sort would put <prefix>_t1000 second
    bodies = sorted({p.name.split("__")[1] for p in (root / "cont").iterdir()},
                    key=lambda b: int(b.rsplit("_t", 1)[1]))
    ts = [int(b.rsplit("_t", 1)[1]) / 1000 for b in bodies]
    clips = sorted({p.name.split("__")[0] for p in (root / "cont").iterdir()})
    R = {}
    for c in clips:
        ok = all((root / m / f"{c}__{b}" / "summary.json").exists() for m in ("cont", "indep") for b in bodies)
        if not ok:
            continue
        R[c] = {m: [load(root / m / f"{c}__{b}") for b in bodies] for m in ("cont", "indep")}
    print(f"{len(R)} complete clips, {len(bodies)} bodies on the path: " + " ".join(f"{t:.3f}" for t in ts))
    if not R:
        return
    z0 = {c: np.load(REPO_ROOT / "data/origin_z" / c.rsplit("_", 1)[0] / f"{c}.npy").reshape(-1)
          for c in R}

    print(f"\n{'clip':34s} {'metric':22s} " + " ".join(f"{t:>6.3f}" for t in ts))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for c, M in R.items():
        z0c = [M["indep"][k]["z0c"] for k in range(len(bodies))]
        ind = [M["indep"][k]["best"] / z0c[k] for k in range(len(bodies))]
        con = [M["cont"][k]["best"] / z0c[k] for k in range(len(bodies))]
        base = z0c[0]
        dz0 = [z0c[k] / base for k in range(len(bodies))]
        s_ind = [ang(M["indep"][k]["z"], M["indep"][k + 1]["z"]) for k in range(len(bodies) - 1)]
        s_con = [ang(M["cont"][k]["z"], M["cont"][k + 1]["z"]) for k in range(len(bodies) - 1)]
        e_ind = [evals_to(M["indep"][k]["ev"], M["indep"][k]["bs"], 0.5 * z0c[k]) for k in range(len(bodies))]
        e_con = [evals_to(M["cont"][k]["ev"], M["cont"][k]["bs"], 0.5 * z0c[k]) for k in range(len(bodies))]
        for k in range(len(bodies)):
            agg["z0 cost (x t=0)"][k].append(dz0[k]); agg["indep best / z0"][k].append(ind[k])
            agg["cont best / z0"][k].append(con[k])
            agg["deg z*_ind from z0"][k].append(ang(M["indep"][k]["z"], z0[c]))
            agg["deg z*_cont from z0"][k].append(ang(M["cont"][k]["z"], z0[c]))
            agg["deg z*_cont vs z*_ind"][k].append(ang(M["cont"][k]["z"], M["indep"][k]["z"]))
            if e_ind[k]: agg["evals->0.5 indep"][k].append(e_ind[k])
            if e_con[k]: agg["evals->0.5 cont"][k].append(e_con[k])
        for k in range(len(bodies) - 1):
            agg["step deg indep"][k].append(s_ind[k]); agg["step deg cont"][k].append(s_con[k])
        print(f"{c:34s} {'z0 cost':22s} " + " ".join(f"{x:6.3f}" for x in z0c))
        print(f"{'':34s} {'indep best / z0':22s} " + " ".join(f"{x:6.2f}" for x in ind))
        print(f"{'':34s} {'cont  best / z0':22s} " + " ".join(f"{x:6.2f}" for x in con))
        print(f"{'':34s} {'step deg ind|cont':22s} " + " ".join(f"{x:3.0f}|{y:<2.0f}" for x, y in zip(s_ind, s_con)))

    print(f"\n==== pooled over {len(R)} clips (median) ====")
    print(f"{'metric':24s} " + " ".join(f"{t:>6.3f}" for t in ts))
    for m in ("z0 cost (x t=0)", "indep best / z0", "cont best / z0", "deg z*_ind from z0",
              "deg z*_cont from z0", "deg z*_cont vs z*_ind", "step deg indep", "step deg cont",
              "evals->0.5 indep", "evals->0.5 cont"):
        vals = [(st.median(agg[m][k]) if agg[m].get(k) else float("nan")) for k in range(len(bodies))]
        print(f"{m:24s} " + " ".join(f"{v:6.2f}" if v == v else f"{'-':>6s}" for v in vals))

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6)); fig.patch.set_facecolor(SURF)
    T = np.array(ts)
    def med(m, n): return np.array([st.median(agg[m][k]) if agg[m].get(k) else np.nan for k in range(n)])
    def band(m, n):
        lo = np.array([np.percentile(agg[m][k], 25) if agg[m].get(k) else np.nan for k in range(n)])
        hi = np.array([np.percentile(agg[m][k], 75) if agg[m].get(k) else np.nan for k in range(n)])
        return lo, hi
    n = len(bodies)
    for m, col, lab in (("indep best / z0", C_IND, "independent search"), ("cont best / z0", C_CONT, "continuation")):
        ax[0].plot(T, med(m, n), color=col, lw=2.2, marker="o", ms=5, label=lab)
        lo, hi = band(m, n); ax[0].fill_between(T, lo, hi, color=col, alpha=.15)
    ax[0].axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.1)
    ax[0].set_ylabel("best cost / that body's z0 cost"); ax[0].set_title("quality of the latent found")
    ax[1].plot(T, med("z0 cost (x t=0)", n), color=INK2, lw=2.2, marker="o", ms=5)
    lo, hi = band("z0 cost (x t=0)", n); ax[1].fill_between(T, lo, hi, color=INK2, alpha=.15)
    ax[1].set_ylabel("z0 cost / z0 cost on t=0"); ax[1].set_title("naive transfer degrades along the path")
    Tm = (T[:-1] + T[1:]) / 2
    for m, col, lab in (("step deg indep", C_IND, "independent"), ("step deg cont", C_CONT, "continuation")):
        ax[2].plot(Tm, med(m, n - 1), color=col, lw=2.2, marker="o", ms=5, label=lab)
        lo, hi = band(m, n - 1); ax[2].fill_between(Tm, lo, hi, color=col, alpha=.15)
    ax[2].set_ylabel("deg between z* on neighbouring bodies"); ax[2].set_title("is z*(beta) a curve?")
    for m, col, lab in (("evals->0.5 indep", C_IND, "from z0"), ("evals->0.5 cont", C_CONT, "from neighbour's z*")):
        ax[3].plot(T, med(m, n), color=col, lw=2.2, marker="o", ms=5, label=lab)
    ax[3].set_ylabel("rollouts to reach 0.5 x z0 cost"); ax[3].set_title("search cost per body")
    for x in ax:
        x.set_facecolor(SURF); x.set_xlabel("t along the path (0 = adult)"); x.grid(alpha=.22, color=INK3, lw=.7)
        for sp in ("top", "right"): x.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): x.spines[sp].set_color(INK3)
        if x.get_legend_handles_labels()[0]: x.legend(fontsize=9)
    fig.suptitle(f"latent along a morphology path ({a.prefix}, {len(R)} clips; median, band = IQR)",
                 fontsize=12.5, color=INK, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .93))
    out = Path(a.out or (REPO_ROOT / "outputs/morph_study" / f"continuation_{a.prefix}.png"))
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140, facecolor=SURF)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
