#!/usr/bin/env python3
"""analyze_correction_transfer.py -- does a latent correction carry over to a new
body, or to a new motion? Scores from build_correction_transfer_jobs.py.

Every number is cost / cost of the TARGET's own z0 on the target body, so 1.0 is
"no better than not correcting". Two readings per transfer:
  full   the source's correction taken at its own angle (alpha = 1)
  best   the best of alpha in {0.25, 0.5, 1} -- the most generous reading
"""
import argparse, collections, csv, statistics as st
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, AQUA, ORANGE, RED = "#2a78d6", "#1baf7a", "#eb6834", "#e34948"

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="outputs/correction_transfer")
ap.add_argument("--fig", default="outputs/correction_transfer/correction_transfer.png")
a = ap.parse_args()
D = REPO / a.dir


def load(name):
    p = D / f"scores_{name}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    z0 = {(r["clip"], r["body"]): float(r["cost"]) for r in rows if r["label"] == "z0"}
    return [dict(r, ratio=float(r["cost"]) / z0[(r["clip"], r["body"])]) for r in rows if r["label"] != "z0"]


def family(stem):
    return stem.split("-")[0].split("_")[0]


RAND = {}          # (set, method, key) -> ratio of a random direction at the same angle


def load_rand(name, key):
    rows = load(f"{name}_rand")
    for r in rows or []:
        p = r["label"].split("|")
        RAND[(name, p[1], key(r, p))] = r["ratio"]


def transfers(rows, key, name=None):
    """group alpha rows into one transfer each: key(row) -> (group id); returns list of dicts."""
    T = collections.defaultdict(dict)
    for r in rows:
        parts = r["label"].split("|")
        if parts[0] == "own" or len(parts) < 4:
            continue
        T[key(r, parts)][parts[3]] = r["ratio"]
    out = []
    for k, v in T.items():
        al = {x: v[x] for x in v if x.startswith("a")}
        m = rows[0]["label"].split("|")[1] if rows else None
        out.append(dict(key=k, full=v.get("a1.0"), best=min(al.values()) if al else None, abs=v.get("abs"),
                        rand=RAND.get((name, m, k))))
    return out


def line(tag, X, extra=""):
    f = [x["full"] for x in X]; b = [x["best"] for x in X]
    ab = [x["abs"] for x in X if x["abs"] is not None]
    s = (f"{tag:34s} n={len(X):4d} | full step median {st.median(f):.3f} (<1: {np.mean(np.array(f) < 1):4.0%})"
         f" | best step median {st.median(b):.3f} (<1: {np.mean(np.array(b) < 1):4.0%})")
    if ab:
        s += f" | whole latent (old test) {st.median(ab):.3f}"
    rd = [(x["full"], x["rand"]) for x in X if x["rand"] is not None]
    if rd:
        s += (f" | random direction, same angle {st.median([r for _, r in rd]):.3f}"
              f" (correction beats it {sum(f < r for f, r in rd)}/{len(rd)})")
    print(s + extra)
    return f, b


groups = {}
# ---- motion axis --------------------------------------------------------------
for name, methods in (("motion", ("cont", "indep")), ("headstand", ("floor",)), ("walk", ("floor",))):
    rows = load(name)
    if rows is None:
        print(f"{name}: no scores yet"); continue
    print(f"\n== {name} ==")
    for m in methods:
        R = [r for r in rows if r["label"].split("|")[1] == m]
        own = [r["ratio"] for r in R if r["label"] == f"own|{m}"]
        kf = lambda r, p: (r["clip"].split("/")[1], p[2])
        load_rand(name, kf)
        X = transfers(R, kf, name)
        print(f"  own search (ceiling) median {st.median(own):.3f}")
        f, b = line(f"  {name} / {m}", X)
        groups[(name, m)] = (f, b, [x["rand"] for x in X if x["rand"] is not None])
        if name == "motion":
            same = [x for x in X if family(x["key"][0]) == family(x["key"][1])]
            diff = [x for x in X if family(x["key"][0]) != family(x["key"][1])]
            line(f"    same family ({len(same)} pairs)", same)
            fd, bd = line(f"    different family", diff)
            groups[("motion-diff", m)] = (fd, bd, [x["rand"] for x in diff if x["rand"] is not None])
# ---- body axis ----------------------------------------------------------------
rows = load("body")
if rows is not None:
    print("\n== body (one path step, both directions) ==")
    for m in ("cont", "indep"):
        R = [r for r in rows if r["label"].split("|")[1] == m]
        kf = lambda r, p: (r["clip"], p[2], r["body"])
        load_rand("body", kf)
        X = transfers(R, kf, "body")
        f, b = line(f"  body / {m} (all 4 paths)", X)
        groups[("body", m)] = (f, b, [x["rand"] for x in X if x["rand"] is not None])
        for pre in ("m2c", "m2s", "m2g", "m2k"):
            line(f"    {pre}", [x for x in X if x["key"][2].startswith(pre)])

# ---- figure -------------------------------------------------------------------
GRAY = "#b8b7b0"
show = [(("body", "cont"), "neighbouring\nbody", AQUA),
        (("headstand", "floor"), "another trial of\nthe same headstand", ORANGE),
        (("motion-diff", "cont"), "a different\nkind of motion", RED)]
show = [x for x in show if x[0] in groups]
if show:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, x = plt.subplots(figsize=(10, 4.6)); fig.patch.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    for i, (g, lab, col) in enumerate(show):
        f, _, rd = groups[g]
        for off, v, c, name in ((-0.18, np.array(f), col, "correction"), (0.18, np.array(rd), GRAY, "random")):
            if not len(v):
                continue
            x.scatter(i + off + rng.uniform(-.1, .1, len(v)), v, s=13, color=c, alpha=.45, edgecolor="none")
            x.scatter(i + off, np.median(v), s=300, color=INK, marker="_", lw=3, zorder=5)
            x.annotate(f"{np.median(v):.2f}", (i + off, np.median(v)), textcoords="offset points",
                       xytext=(15, -4), fontsize=11, color=INK)
    x.scatter([], [], s=30, color=INK2, label="the source's correction, moved to the target's z0")
    x.scatter([], [], s=30, color=GRAY, label="a random direction, same angle (control)")
    x.legend(frameon=False, fontsize=10, loc="upper right")
    x.axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.1)
    x.text(1.47, .95, "no better than z0", ha="center", va="top", fontsize=10, color=INK2)
    x.set_xticks(range(len(show))); x.set_xticklabels([s_[1] for s_ in show], fontsize=11)
    x.set_yscale("log"); x.set_ylabel("cost / the target's own z0 cost (log)", fontsize=11)
    x.tick_params(axis="y", labelsize=10)
    x.set_title("a latent correction moved to a new body vs to a new motion", fontsize=12)
    x.set_facecolor(SURF); x.grid(axis="y", alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPO / a.fig, dpi=140, facecolor=SURF); print(f"\n-> {a.fig}")
