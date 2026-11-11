#!/usr/bin/env python3
"""child_52_report.py -- Comparison A on the child (outputs/child_52): z0, ES, CEM and random search on 52 motions, 5 search seeds, 64 generations, reference-velocity start. Metrics (scripts/eval_metrics.py) on each
search's evaluation rollout of its best z. Cost ratio = a run's cost / its motion's z0 cost. Per motion the seeds are summarised by their median, then the motions by their median; pairwise tests are Wilcoxon signed-rank on the 52 motion medians.
Writes outputs/child_52/{report.json, per_motion.csv, convergence.png}. usage: uv run python scripts/child_52_report.py"""
import csv, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import eval_metrics as E                                                  # noqa: E402
OUT = REPO / "outputs/child_52"; ALGOS = [("es", "ES (ours)", "#2a78d6"), ("cem", "CEM", "#eb6834"), ("random", "Random search", "#1baf7a")]
INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"; KEYS = ("fall_s", "mpjpe", "mmpjpe", "mpjae"); N = (16, 64, 128, 256, 512, 1024)


def group(task):
    if task.startswith("move-ego-low"): return "low walk"
    if "raisearms" in task: return "walk + arms" if task.startswith("move") else "raise arms"
    if task.startswith("move-ego"): return "walk / run"
    if task.startswith("crawl"): return "crawl"
    if task.startswith("rotate"): return "rotate"
    return "other (crouch, headstand, jump, lie, sit, split)"


def main():
    clips = [l.strip() for l in open(OUT / "clips.txt")]; body = E.Body(E.XML["child"], True); R = {}; curves = {a: [] for a, _, _ in ALGOS}; diff = []
    for c in clips:
        task, stem = c.split("/"); ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)
        z = next(csv.DictReader(open(OUT / f"{stem}_z0/eval.csv"))); m = E.metrics(body, np.load(OUT / f"{stem}_z0/eval_qpos.npz")["qpos"][0].astype(np.float64), ref); z0c = float(z["cost"])
        r = dict(task=task, stem=stem, group=group(task), z0=dict(cost=z0c, success=m["success"], **{k: m[k] for k in KEYS}))
        for a, _, _ in ALGOS:
            S = []; C = []
            for d in sorted(glob.glob(str(OUT / f"{stem}_{a}_s*"))):
                s = json.load(open(f"{d}/summary.json")); e = next(csv.DictReader(open(f"{d}/eval/eval.csv"))); m = E.metrics(body, np.load(f"{d}/eval/eval_qpos.npz")["qpos"][0].astype(np.float64), ref)
                C.append([float(x["best_so_far"]) / z0c for x in csv.DictReader(open(f"{d}/curve.csv"))]); diff.append(abs(float(e["cost"]) - s["best"]["cost"]))
                S.append(dict(seed=s["seed"], cost=float(e["cost"]), ratio=float(e["cost"]) / z0c, bfm=float(e["bfm"]), success=m["success"], fail=(m["first_fail"] or {}).get("by"), best_gen=s["best"]["gen"], **{k: m[k] for k in KEYS}))
            assert len(S) == 5, (stem, a); curves[a].append(C)
            r[a] = dict(seeds=S, success=sum(x["success"] for x in S), **{k: float(np.median([x[k] for x in S])) for k in KEYS + ("ratio", "cost")}, seed_spread=float(np.ptp([x["ratio"] for x in S])))
        R[stem] = r
    T = list(R.values()); rep = dict(n_motions=len(T), max_search_vs_reroll_cost=float(max(diff)), share_reroll_differs=float(np.mean(np.array(diff) > 1e-3)), methods={}, pairs={}, groups={})
    z = [x["z0"] for x in T]; rep["methods"]["z0"] = dict(success=sum(x["success"] for x in z), n=len(z), clips_fall=sum(x["fall_s"] >= 0.1 for x in z), fall_total_s=float(sum(x["fall_s"] for x in z)), **{k: float(np.median([x[k] for x in z])) for k in ("mpjpe", "mmpjpe", "mpjae")}, cost=float(np.median([x["cost"] for x in z])))
    fig, ax = plt.subplots(figsize=(8.6, 5), dpi=180, facecolor=SURF); ax.set_facecolor(SURF)
    for a, name, col in ALGOS:
        C = np.median(np.array(curves[a]), 1); x = (np.arange(C.shape[1]) + 1) * 16; med = np.median(C, 0)         # motions x gens
        ax.fill_between(x, np.percentile(C, 25, axis=0), np.percentile(C, 75, axis=0), color=col, alpha=0.13, lw=0, zorder=2); ax.plot(x, med, color=col, lw=2, zorder=3, label=name)
        all_s = [s for r in T for s in r[a]["seeds"]]
        rep["methods"][a] = dict(ratio_median={str(n): float(med[n // 16 - 1]) for n in N}, ratio_mean={str(n): float(C[:, n // 16 - 1].mean()) for n in N}, ratio_q25_q75=[float(np.percentile(C[:, -1], 25)), float(np.percentile(C[:, -1], 75))], success=sum(s["success"] for s in all_s), n=len(all_s), clips_fall=sum(s["fall_s"] >= 0.1 for s in all_s), fall_total_s=float(sum(s["fall_s"] for s in all_s)), seed_spread_median=float(np.median([r[a]["seed_spread"] for r in T])),
                                 fail_position=sum(s["fail"] == "position" for s in all_s), fail_rotation=sum(s["fail"] == "rotation" for s in all_s), **{k: float(np.median([r[a][k] for r in T])) for k in ("mpjpe", "mmpjpe", "mpjae")}, cost=float(np.median([r[a]["cost"] for r in T])))
    for p, q in (("es", "random"), ("es", "cem"), ("random", "cem")):
        rep["pairs"][f"{p}_vs_{q}"] = {}
        for m in ("ratio", "mpjpe", "mmpjpe", "mpjae"):
            a_, b_ = np.array([r[p][m] for r in T]), np.array([r[q][m] for r in T]); rep["pairs"][f"{p}_vs_{q}"][m] = dict(lower=int((a_ < b_).sum()), p=float(wilcoxon(a_, b_).pvalue), median_diff=float(np.median(a_ - b_)))
        C1, C2 = np.median(np.array(curves[p]), 1), np.median(np.array(curves[q]), 1); rep["pairs"][f"{p}_vs_{q}"]["budget"] = {str(n): dict(lower=int((C1[:, n // 16 - 1] < C2[:, n // 16 - 1]).sum()), p=float(wilcoxon(C1[:, n // 16 - 1], C2[:, n // 16 - 1]).pvalue)) for n in (64, 128, 256, 512)}
    for g in sorted({r["group"] for r in T}):
        G = [r for r in T if r["group"] == g]; rep["groups"][g] = dict(n=len(G), z0_mpjpe=float(np.median([r["z0"]["mpjpe"] for r in G])), **{a: dict(ratio=float(np.median([r[a]["ratio"] for r in G])), mpjpe=float(np.median([r[a]["mpjpe"] for r in G])), success=sum(r[a]["success"] for r in G)) for a, _, _ in ALGOS})
    ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); ax.text(40, 1, "z0", color=MUTED, fontsize=9, va="bottom"); ax.set_xlabel("simulated rollouts", fontsize=10, color=MUTED); ax.set_ylabel("best cost so far / z0 cost (lower is better)", fontsize=10, color=MUTED); ax.set_ylim(0, 1.08)
    ax.set_title("Child, 52 motions: median and quartiles over motions (each motion: median of 5 seeds)", fontsize=11, color=INK, loc="left", pad=12); ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right", bbox_to_anchor=(1.0, 0.92)); fig.tight_layout(); fig.savefig(OUT / "convergence.png", facecolor=SURF)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    with open(OUT / "per_motion.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "group", "z0_cost"] + [f"{a}_{k}" for a, _, _ in ALGOS for k in ("ratio", "success", "mpjpe", "mmpjpe", "mpjae", "fall_s")])
        for r in T:
            w.writerow([r["stem"], r["group"], round(r["z0"]["cost"], 4)] + [round(r[a][k], 4) for a, _, _ in ALGOS for k in ("ratio", "success", "mpjpe", "mmpjpe", "mpjae", "fall_s")])
    print(json.dumps(rep, indent=0)[:9000])


if __name__ == "__main__":
    main()
