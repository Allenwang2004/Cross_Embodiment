#!/usr/bin/env python3
"""child_new_test_report.py -- outputs/child_new_test: ES, CEM and random search vs z0 on 3 motions, 5 search seeds, 64 generations, started with the reference velocity. Metrics (scripts/eval_metrics.py)
on each search's evaluation rollout of its best z (eval/eval_qpos.npz, a separate rollout from the one the search scored). Writes outputs/child_new_test/{report.json, convergence.png}. usage: uv run python scripts/child_new_test_report.py"""
import csv, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import eval_metrics as E                                                  # noqa: E402
OUT = REPO / "outputs/child_new_test"; MOTIONS = [("move-ego-0-2", "move-ego-0-2_2", "walk"), ("move-ego-0-4", "move-ego-0-4_8", "run"), ("headstand", "headstand_4", "headstand")]
ALGOS = [("es", "ES (ours)", "#2a78d6"), ("cem", "CEM", "#eb6834"), ("random", "Random search", "#1baf7a")]
INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"; KEYS = ("fall_s", "mpjpe", "mmpjpe", "mpjae")


def row_csv(p):
    r = next(csv.DictReader(open(p))); return float(r["cost"]), float(r["bfm"])


def main():
    body = E.Body(E.XML["child"], True); rep = {}; fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), dpi=170, facecolor=SURF)
    for b, (task, stem, lab) in zip(ax, MOTIONS):
        ref = np.load(REPO / f"data/child/retargeting_motion/{task}/{stem}.npz")["qpos"].astype(np.float64); r = dict()
        q = np.load(OUT / f"{stem}_z0/eval_qpos.npz")["qpos"][0].astype(np.float64); c, f = row_csv(OUT / f"{stem}_z0/eval.csv"); r["z0"] = dict(cost=c, bfm=f, **{k: v for k, v in E.metrics(body, q, ref).items() if k in KEYS + ("success",)})
        for a, name, col in ALGOS:
            C = []; S = []
            for d in sorted(glob.glob(str(OUT / f"{stem}_{a}_s*"))):
                s = json.load(open(f"{d}/summary.json")); C.append([float(x["best_so_far"]) for x in csv.DictReader(open(f"{d}/curve.csv"))])
                q = np.load(f"{d}/eval/eval_qpos.npz")["qpos"][0].astype(np.float64); c, fb = row_csv(f"{d}/eval/eval.csv"); m = E.metrics(body, q, ref)
                S.append(dict(seed=s["seed"], search_cost=s["best"]["cost"], eval_cost=c, eval_bfm=fb, best_gen=s["best"]["gen"], minutes=s["minutes"], first_fail=m["first_fail"], success=m["success"], **{k: m[k] for k in KEYS}))
            C = np.array(C); x = (np.arange(C.shape[1]) + 1) * 16; med = np.median(C, 0)
            b.fill_between(x, C.min(0), C.max(0), color=col, alpha=0.14, lw=0, zorder=2); b.plot(x, med, color=col, lw=2, zorder=3, label=name)
            r[a] = dict(seeds=S, n=len(S), curve_median={str(n): float(med[n // 16 - 1]) for n in (64, 128, 256, 512, 1024)}, search_cost_range=[float(C[:, -1].min()), float(C[:, -1].max())], success=sum(x_["success"] for x_ in S), **{k: float(np.median([x_[k] for x_ in S])) for k in KEYS + ("eval_cost",)})
        rep[stem] = r; z0 = r["z0"]["cost"]; b.axhline(z0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); b.text(40, z0, f"z0 {z0:.3f}", color=MUTED, fontsize=9, va="bottom")
        b.set_title(lab, fontsize=12, color=INK, loc="left"); b.set_xlabel("simulated rollouts", fontsize=10, color=MUTED); b.set_ylim(0, None); b.set_facecolor(SURF); b.grid(color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=0, labelsize=9)
        for sp in ("top", "right", "left", "bottom"):
            b.spines[sp].set_visible(False)
    ax[0].set_ylabel("best cost so far (lower is better)", fontsize=10, color=MUTED); ax[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right")
    fig.suptitle("Child, reference-velocity start: best cost so far, median and range of 5 seeds", x=0.01, ha="left", fontsize=13, color=INK); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "convergence.png", facecolor=SURF)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    for stem, r in rep.items():
        z = r["z0"]; print(f"== {stem}  z0: cost {z['cost']:.3f} bfm {z['bfm']:.3f} success {int(z['success'])} fall {z['fall_s']:.1f} MPJPE {z['mpjpe']:.2f} MMPJPE {z['mmpjpe']:.2f} MPJAE {z['mpjae']:.2f}")
        for a, name, _ in ALGOS:
            x = r[a]; print(f"  {a:6s} n {x['n']} search cost med {x['curve_median']['1024']:.3f} range {x['search_cost_range'][0]:.3f}-{x['search_cost_range'][1]:.3f} | re-roll cost {x['eval_cost']:.3f} | at 64/128/256/512 {[round(x['curve_median'][n], 3) for n in ('64', '128', '256', '512')]} | success {x['success']}/{x['n']} fall {x['fall_s']:.2f} MPJPE {x['mpjpe']:.2f} MMPJPE {x['mmpjpe']:.2f} MPJAE {x['mpjae']:.2f} | best gen {[s['best_gen'] for s in x['seeds']]} fail {[(s['first_fail']['by'][:3], round(s['first_fail']['t_s'], 1)) for s in x['seeds'] if s['first_fail']]}")


def pooled():
    """all motions together: each run's cost / its motion's z0 cost, so the motions weigh the same. Writes outputs/child_new_test/{pooled.json, convergence_pooled.png}."""
    rep = json.load(open(OUT / "report.json")); P = {}; fig, ax = plt.subplots(figsize=(8.6, 5), dpi=180, facecolor=SURF); ax.set_facecolor(SURF)
    for a, name, col in ALGOS:
        C = []
        for _, stem, _l in MOTIONS:
            z0 = rep[stem]["z0"]["cost"]
            for d in sorted(glob.glob(str(OUT / f"{stem}_{a}_s*"))):
                C.append([float(x["best_so_far"]) / z0 for x in csv.DictReader(open(f"{d}/curve.csv"))])
        C = np.array(C); x = (np.arange(C.shape[1]) + 1) * 16; med, mean = np.median(C, 0), C.mean(0)
        ax.fill_between(x, np.percentile(C, 25, axis=0), np.percentile(C, 75, axis=0), color=col, alpha=0.14, lw=0, zorder=2); ax.plot(x, med, color=col, lw=2, zorder=3, label=name)
        S = [s for _, stem, _l in MOTIONS for s in rep[stem][a]["seeds"]]
        P[a] = dict(n=len(S), ratio_median={str(n): float(med[n // 16 - 1]) for n in (64, 128, 256, 512, 1024)}, ratio_mean={str(n): float(mean[n // 16 - 1]) for n in (64, 128, 256, 512, 1024)}, final_ratio_q25_q75=[float(np.percentile(C[:, -1], q)) for q in (25, 75)],
                    success=sum(s["success"] for s in S), falls=sum(s["fall_s"] >= 0.1 for s in S), final_ratios=C[:, -1].tolist(), **{k: float(np.median([s[k] for s in S])) for k in KEYS}, **{k + "_mean": float(np.mean([s[k] for s in S])) for k in KEYS})
    z = [rep[stem]["z0"] for _, stem, _l in MOTIONS]; P["z0"] = dict(n=3, success=sum(x["success"] for x in z), falls=sum(x["fall_s"] >= 0.1 for x in z), **{k: float(np.median([x[k] for x in z])) for k in KEYS}, **{k + "_mean": float(np.mean([x[k] for x in z])) for k in KEYS})
    for n1, n2 in (("es", "random"), ("es", "cem"), ("random", "cem")):
        a1, a2 = np.array(P[n1]["final_ratios"]), np.array(P[n2]["final_ratios"]); P[f"{n1}_lower_than_{n2}"] = int((a1 < a2).sum())
    ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); ax.text(40, 1, "z0", color=MUTED, fontsize=9, va="bottom"); ax.set_xlabel("simulated rollouts", fontsize=10, color=MUTED); ax.set_ylabel("best cost so far / z0 cost (lower is better)", fontsize=10, color=MUTED); ax.set_ylim(0, 1.08)
    ax.set_title("Child, 3 motions together: median and quartiles of 15 runs per method", fontsize=12, color=INK, loc="left", pad=12); ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right"); fig.tight_layout(); fig.savefig(OUT / "convergence_pooled.png", facecolor=SURF)
    (OUT / "pooled.json").write_text(json.dumps(P, indent=1))
    for a in ("z0", "es", "random", "cem"):
        x = P[a]; print(a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k != "final_ratios"})
    print({k: v for k, v in P.items() if "lower" in k})


if __name__ == "__main__":
    main(); pooled()
