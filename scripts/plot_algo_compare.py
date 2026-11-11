#!/usr/bin/env python3
"""plot_algo_compare.py -- ES vs CEM vs random search (scripts/single_z_search.py --algo) on one clip: best cost so far against rollouts, median and range over the seeds of
outputs/algo_compare/<stem>_<algo>_s<seed>/. Writes outputs/algo_compare/<stem>.{png,json}. usage: uv run python scripts/plot_algo_compare.py [stem]"""
import csv, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.geometry import sphere16   # noqa: E402  16 z / |z| (refactor step 3)
import eval_metrics as E                                                  # noqa: E402
INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"
ALGOS = [("es", "ES (ours)", "#2a78d6"), ("cem", "CEM", "#eb6834"), ("random", "Random search", "#1baf7a")]


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "move-ego-0-2_2"; out = REPO / "outputs/algo_compare"; body = E.Body(E.XML["child"], True); res = {}
    fig, ax = plt.subplots(figsize=(8.6, 5), dpi=180, facecolor=SURF); ax.set_facecolor(SURF)
    for a, label, col in ALGOS:
        runs = sorted(glob.glob(str(out / f"{stem}_{a}_s*"))); C = []; rows = []
        for r in runs:
            cur = list(csv.DictReader(open(f"{r}/curve.csv"))); C.append([float(x["best_so_far"]) for x in cur]); s = json.load(open(f"{r}/summary.json"))
            task = s["clip"].split("/")[0]; ref = np.load(REPO / f"data/child/retargeting_motion/{task}/{stem}.npz")["qpos"].astype(np.float64); m = E.metrics(body, np.load(f"{r}/best.npz")["qpos"].astype(np.float64), ref)
            z = np.load(f"{r}/best_z.npy").reshape(-1); z0 = np.load(REPO / f"data/origin_z/{s['clip']}.npy").reshape(-1); z0 = sphere16(z0)
            rows.append(dict(seed=s["seed"], z0_cost=s["origin_z"]["cost"], best_cost=s["best"]["cost"], bfm=s["best"]["bfm"], head=s["best"]["head"], pos=s["best"]["pos"], best_gen=s["best"]["gen"], dist_z0=float(np.linalg.norm(z - z0)), success=m["success"], mpjpe=m["mpjpe"], mpjae=m["mpjae"], fall_s=m["fall_s"]))
        C = np.array(C); x = (np.arange(C.shape[1]) + 1) * 16; med = np.median(C, 0)
        ax.fill_between(x, C.min(0), C.max(0), color=col, alpha=0.14, lw=0, zorder=2); ax.plot(x, med, color=col, lw=2, zorder=3, label=label)
        ax.text(x[-1] + 10, med[-1] + {"es": -0.018, "random": 0.018}.get(a, 0), label, color=INK, fontsize=9, va="center")
        res[a] = dict(curve_median={str(n): float(med[n // 16 - 1]) for n in (16, 32, 64, 128, 256, 512, 1024)}, curve_min_max={str(n): [float(C[:, n // 16 - 1].min()), float(C[:, n // 16 - 1].max())] for n in (16, 256, 1024)}, seeds=rows)
    z0c = rows[0]["z0_cost"]; ax.axhline(z0c, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); ax.text(30, z0c, f"z0 {z0c:.3f}", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel("simulated rollouts", fontsize=10, color=MUTED); ax.set_ylabel("best cost so far (lower is better)", fontsize=10, color=MUTED); ax.set_xlim(0, 1200); ax.set_ylim(0, None)
    ax.set_title(f"Child, {stem}: best cost so far, median and range of 3 seeds", fontsize=12, color=INK, loc="left", pad=12)
    ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right", bbox_to_anchor=(1.0, 0.93)); fig.tight_layout(); fig.savefig(out / f"{stem}.png", facecolor=SURF)
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1))
    for a, r in res.items():
        print(a, {k: round(v, 3) for k, v in r["curve_median"].items()}, "| range 1024", np.round(r["curve_min_max"]["1024"], 3))
        for x in r["seeds"]:
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})


if __name__ == "__main__":
    main()
