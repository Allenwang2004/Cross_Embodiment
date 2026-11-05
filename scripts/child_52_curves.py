#!/usr/bin/env python3
"""child_52_curves.py -- MPJPE, MMPJPE and MPJAE against rollouts for Comparison A (outputs/child_52), so convergence is read on metrics no method optimises. Each run's best z so far changes only at some generations, so
  prepare  writes outputs/child_52_curves/<stem>/{z.npy, index.json}: the distinct best z so far of all of the motion's runs (z_trace.npz)
  (run)    outputs/train_logs/queue: scripts/single_z_search.py --eval-z z.npy --eval-save-qpos on each motion, the same settings as the searches
  analyse  metrics of each rollout, checked against the search's own best_so_far, then per generation: each motion's median of 5 seeds, the median over motions; Wilcoxon signed-rank on the 52 motion medians. Writes report.json, per_generation.npz, convergence_metrics.png.
usage: uv run python scripts/child_52_curves.py prepare | analyse"""
import csv, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import eval_metrics as E                                                  # noqa: E402
SRC = REPO / "outputs/child_52"; OUT = REPO / "outputs/child_52_curves"; ALGOS = [("es", "LOES (ours)", "#2a78d6"), ("cem", "CEM", "#eb6834"), ("random", "Random search", "#1baf7a")]
M = [("mpjpe", "MPJPE (cm)"), ("mmpjpe", "MMPJPE (cm)"), ("mpjae", "MPJAE (m/s²)")]; INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"; GENS = 64


def prepare():
    for c in (l.strip() for l in open(SRC / "clips.txt")):
        stem = c.split("/")[1]; d = OUT / stem; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
        for r in sorted(glob.glob(str(SRC / f"{stem}_*_s*"))):
            b = np.load(f"{r}/z_trace.npz")["z_best"]; assert len(b) == GENS + 1, r; m = []          # index 0: before the first generation
            for g in range(1, GENS + 1):
                if not m or np.abs(b[g] - b[g - 1]).max() > 0:
                    Z.append(b[g]); m.append(len(Z) - 1)
                else:
                    m.append(m[-1])
            idx[Path(r).name] = m
        np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, map=idx))); print(stem, len(Z), flush=True)


def analyse():
    body = E.Body(E.XML["child"], True); clips = [l.strip() for l in open(SRC / "clips.txt")]; P = {a: {k: [] for k, _ in M} for a, _, _ in ALGOS}; Z0 = {k: [] for k, _ in M}; worst = []
    for c in clips:
        task, stem = c.split("/"); ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); d = OUT / stem; ix = json.load(open(d / "index.json"))["map"]
        Q = np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64); cost = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]); V = [E.tracking_metrics(body, q, ref) for q in Q]
        z0 = E.tracking_metrics(body, np.load(SRC / f"{stem}_z0/eval_qpos.npz")["qpos"][0].astype(np.float64), ref)
        for k, _ in M:
            Z0[k].append(z0[k])
        for a, _, _ in ALGOS:
            runs = sorted(r for r in ix if r.rsplit("_", 2)[1] == a); assert len(runs) == 5, (stem, a)
            for k, _ in M:
                P[a][k].append(np.median([[V[i][k] for i in ix[r]] for r in runs], 0))
            for r in runs:
                bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(SRC / r / "curve.csv"))]); mp = np.array(ix[r]); worst.append(float(np.abs(cost[mp] - bs).max()))
        print(stem, flush=True)
    A = {a: {k: np.array(v) for k, v in P[a].items()} for a in P}; x = (np.arange(GENS) + 1) * 16; rep = dict(n=len(clips), max_rollout_vs_search_cost=max(worst), share_runs_over_1e3=float(np.mean(np.array(worst) > 1e-3)), z0={k: float(np.median(v)) for k, v in Z0.items()}, median={}, pairs={})
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=170, facecolor=SURF)
    for b, (k, lab) in zip(ax, M):
        for a, name, col in ALGOS:
            C = A[a][k]; med = np.median(C, 0); b.fill_between(x, np.percentile(C, 25, axis=0), np.percentile(C, 75, axis=0), color=col, alpha=0.1, lw=0, zorder=2); b.plot(x, med, color=col, lw=2, zorder=3, label=name)
            rep["median"].setdefault(a, {})[k] = {str(n): float(med[n // 16 - 1]) for n in (16, 64, 128, 256, 512, 1024)}
        z = float(np.median(Z0[k])); b.axhline(z, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); b.text(x[-1], z, f"z0 {z:.2f}", color=MUTED, fontsize=9, va="bottom", ha="right")
        b.set_xlabel("simulated rollouts", fontsize=10, color=MUTED); b.set_ylabel(lab + " (lower is better)", fontsize=10, color=MUTED); b.set_facecolor(SURF); b.grid(color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=0, labelsize=9)
        for sp in ("top", "right", "left", "bottom"):
            b.spines[sp].set_visible(False)
    ax[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right", bbox_to_anchor=(1, 0.88))
    for p_, q_ in (("es", "random"), ("es", "cem"), ("random", "cem")):
        rep["pairs"][f"{p_}_vs_{q_}"] = {k: {str(n): dict(lower=int((A[p_][k][:, n // 16 - 1] < A[q_][k][:, n // 16 - 1]).sum()), p=float(wilcoxon(A[p_][k][:, n // 16 - 1], A[q_][k][:, n // 16 - 1]).pvalue)) for n in (64, 128, 256, 512, 1024)} for k, _ in M}
    fig.suptitle("Child, 52 motions: best latent so far (median and quartiles over motions; each motion: median of 5 seeds)", x=0.01, ha="left", fontsize=12, color=INK); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "convergence_metrics.png", facecolor=SURF)
    np.savez_compressed(OUT / "per_generation.npz", **{f"{a}_{k}": A[a][k] for a in A for k, _ in M}); (OUT / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))


if __name__ == "__main__":
    {"prepare": prepare, "analyse": analyse}[sys.argv[1]]()
