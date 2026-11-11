#!/usr/bin/env python3
"""rollout_budget_analyze.py -- does the G1 latent search converge slower with fewer rollouts per step?

Reads outputs/g1/rollout_budget/<stem>_r{16,8,4}_s{0,1}/curve.csv (robot_z_search.py, bfm, ik 1.0, the same
1024-rollout budget: 64 / 128 / 256 steps, --eval-every 0 so the best is always a rolled-out sample) and compares the
best cost so far, as a ratio to the clip's z0 cost at 16 rollouts per step (the z0 cost depends on the batch size), after the same number of rollouts, the same number of steps, and
the same estimated wall-clock time (per-step times measured alone on GPU 2 for a 300-frame clip, scaled by clip
length; --step-seconds).
Writes outputs/g1/rollout_budget/{summary.json, convergence.png}.
usage: uv run python scripts/rollout_budget_analyze.py
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/g1/rollout_budget"
NS = (16, 8, 4)
COLORS = {16: "#4C78A8", 8: "#F58518", 4: "#54A24B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-seconds", default="16:9.77,8:6.36,4:4.49",
                    help="wall seconds of one step of a 300-frame clip, measured alone on GPU 2")
    a = ap.parse_args()
    sec = {int(k): float(v) for k, v in (x.split(":") for x in a.step_seconds.split(","))}
    clips = REPO.joinpath("outputs/robots_clips.txt").read_text().split()
    runs = {}                                       # (stem, seed) -> {n: (ratio after each rollout count, T)}
    for c in clips:
        stem = c.split("/")[1]
        T = len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"])
        for s in (0, 1):
            row = {}
            if not all((OUT / f"{stem}_r{n}_s{s}/summary.json").exists() for n in NS):
                continue
            # the z0 cost itself depends on the batch size (the GPU actor's output differs by ~1e-5 between batches of
            # 4, 8 and 16, which the physics amplifies), so every setting is divided by the same number: the 16-rollout z0 cost
            ref = json.load(open(OUT / f"{stem}_r16_s{s}/summary.json"))["origin_z"]["cost"]
            for n in NS:
                d = OUT / f"{stem}_r{n}_s{s}"
                c0 = json.load(open(d / "summary.json"))["origin_z"]["cost"]
                gb = np.array([float(r["gen_best"]) for r in csv.DictReader(open(d / "curve.csv"))])
                row[n] = (np.minimum.accumulate(np.minimum(gb, c0)) / ref, T)    # best so far (own z0 included) / 16-rollout z0
            if len(row) == len(NS):
                runs[(stem, s)] = row
    print(f"{len(runs)} (clip, seed) pairs complete")
    from scipy.stats import wilcoxon

    def at_rollouts(n, k):
        return np.array([r[n][0][k // n - 1] for r in runs.values()])

    def at_steps(n, k):
        return np.array([r[n][0][k - 1] for r in runs.values()])

    summ = dict(pairs=len(runs), by_rollouts={}, by_steps={}, by_minutes={}, step_seconds_300=sec)
    print("\nbest cost / z0 cost at 16 rollouts, median over (clip, seed) [paired Wilcoxon p vs 16 per step]")
    print("rollouts used   " + "".join(f"{n:>10d}/step" for n in NS))
    for k in (64, 128, 256, 512, 1024):
        v = {n: at_rollouts(n, k) for n in NS}
        p = {n: float(wilcoxon(v[n], v[16]).pvalue) if n != 16 else None for n in NS}
        summ["by_rollouts"][k] = {n: dict(median=float(np.median(v[n])), p_vs_16=p[n],
                                          better_than_16=float(np.mean(v[n] < v[16]))) for n in NS}
        print(f"{k:13d}   " + "".join(f"{np.median(v[n]):9.3f}" + (f" ({p[n]:.0e})" if p[n] is not None else "      ")
                                     for n in NS))
    print("steps taken     " + "".join(f"{n:>10d}/step" for n in NS))
    for k in (8, 16, 32, 64):
        v = {n: at_steps(n, k) for n in NS}
        summ["by_steps"][k] = {n: float(np.median(v[n])) for n in NS}
        print(f"{k:13d}   " + "".join(f"{np.median(v[n]):9.3f}      " for n in NS))
    grid = np.linspace(0, 12, 241)                       # minutes, for a 300-frame clip equivalent

    def curve_minutes(n):
        out = []
        for r in runs.values():
            ratio, T = r[n]
            t = (np.arange(1, len(ratio) + 1)) * sec[n] * T / 300 / 60
            out.append(np.array([ratio[t <= g][-1] if (t <= g).any() else 1.0 for g in grid]))
        return np.array(out)

    M = {n: curve_minutes(n) for n in NS}
    for g in (2, 4, 8):
        i = int(np.argmin(np.abs(grid - g)))
        summ["by_minutes"][g] = {n: float(np.median(M[n][:, i])) for n in NS}
    print("minutes (300-frame equiv.)" + "".join(f"  {n}/step" for n in NS))
    for g, v in summ["by_minutes"].items():
        print(f"{g:13d}   " + "".join(f"{v[n]:9.3f}" for n in NS))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(22, 6.2), dpi=120)
    for n in NS:
        lab = f"{n} rollouts per step"
        Y = np.array([r[n][0] for r in runs.values()])                    # (pairs, steps)
        x = np.arange(1, Y.shape[1] + 1) * n
        for ax, xx, YY in ((axs[0], x, Y), (axs[1], np.arange(1, Y.shape[1] + 1), Y), (axs[2], grid, M[n])):
            med, lo, hi = np.median(YY, 0), np.percentile(YY, 25, 0), np.percentile(YY, 75, 0)
            ax.plot(xx, med, color=COLORS[n], lw=2.5, label=lab); ax.fill_between(xx, lo, hi, color=COLORS[n], alpha=.15)
    axs[0].set_xlabel("rollouts used", fontsize=13); axs[1].set_xlabel("steps taken", fontsize=13)
    axs[2].set_xlabel("minutes (300-frame clip, one search alone on GPU 2)", fontsize=13)
    axs[1].set_xscale("log", base=2)
    for ax, t in zip(axs, ("Same number of rollouts", "Same number of steps", "Same wall-clock time")):
        ax.set_title(t, fontsize=15); ax.grid(alpha=.3); ax.set_ylabel("best cost / z0 cost at 16 rollouts (median, IQR band)", fontsize=12)
    axs[0].legend(fontsize=12)
    fig.suptitle(f"G1 latent search, {len(runs) // 2} clips x 2 seeds", fontsize=15)
    fig.tight_layout(); fig.savefig(OUT / "convergence.png"); print("->", OUT / "convergence.png")


if __name__ == "__main__":
    main()
