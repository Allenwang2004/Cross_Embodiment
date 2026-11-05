#!/usr/bin/env python3
"""child_52_means.py -- Comparison A (outputs/child_52) summarised by means instead of medians: every run's metrics for the best z so far at every generation (outputs/child_52_curves) and the mean z at 4, 8, 16, 32, 64 generations
(outputs/child_52_mean_z) are stored in outputs/child_52_curves/per_run.npz; a method's number is the mean over its 260 runs (= over the 52 motions' means), +- the standard error over motions; pairwise tests are Wilcoxon signed-rank on the 52 motion means.
Writes outputs/child_52_curves/{per_run.npz, means.json, convergence_metrics_mean.png}. usage: uv run python scripts/child_52_means.py"""
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import eval_metrics as E                                                  # noqa: E402
SRC = REPO / "outputs/child_52"; CUR = REPO / "outputs/child_52_curves"; MZ = REPO / "outputs/child_52_mean_z"; ALGOS = [("es", "LOES (ours)", "#2a78d6"), ("cem", "CEM", "#eb6834"), ("random", "Random search", "#1baf7a")]
KEYS = ("ratio", "fall_s", "success", "mpjpe", "mmpjpe", "mpjae"); G = (4, 8, 16, 32, 64); INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"


def full(body, q, qr, ratio):
    T = min(len(q), len(qr)); pos = np.linalg.norm(q[:T, :2] - qr[:T, :2], axis=1); rot = np.degrees(2 * np.arccos(np.abs((q[:T, 3:7] * qr[:T, 3:7]).sum(1)).clip(0, 1)))
    return dict(E.tracking_metrics(body, q, qr), ratio=ratio, fall_s=float((q[:T, 2] < 0.5 * qr[:T, 2]).sum() / 30), success=float(not ((pos > 1.5) | (rot > 45)).any()))


def main():
    body = E.Body(E.XML["child"], True); clips = [l.strip() for l in open(SRC / "clips.txt")]; nk = len(KEYS)
    B = {a: np.zeros((len(clips), 5, 64, nk)) for a, _, _ in ALGOS}; C = {a: np.zeros((len(clips), 5, len(G), nk)) for a, _, _ in ALGOS}; Z0 = np.zeros((len(clips), nk))
    for ci, c in enumerate(clips):
        stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); z0c = float(next(csv.DictReader(open(SRC / f"{stem}_z0/eval.csv")))["cost"])
        m = full(body, np.load(SRC / f"{stem}_z0/eval_qpos.npz")["qpos"][0].astype(np.float64), ref, 1.0); Z0[ci] = [m[k] for k in KEYS]
        for src, arr, is_best in ((CUR, B, True), (MZ, C, False)):
            d = src / stem; ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64); cost = [float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]
            V = [None] * len(Q)
            for r, idx in ix.items():
                a, s = r.rsplit("_", 2)[1:]; s = int(s[1:])
                for j, i in enumerate(idx):
                    if V[i] is None:
                        v = full(body, Q[i], ref, cost[i] / z0c); V[i] = [v[k] for k in KEYS]
                    arr[a][ci, s, j] = V[i]
        print(stem, flush=True)
    np.savez_compressed(CUR / "per_run.npz", clips=np.array(clips), keys=np.array(KEYS), gens_mean_z=np.array(G), z0=Z0, **{f"best_{a}": B[a] for a in B}, **{f"mean_{a}": C[a] for a in C})
    rep = dict(z0={k: float(Z0[:, i].mean()) for i, k in enumerate(KEYS)}, z0_se={k: float(Z0[:, i].std(ddof=1) / np.sqrt(len(clips))) for i, k in enumerate(KEYS)}, best={}, center={}, pairs={})
    for a, _, _ in ALGOS:
        rep["best"][a] = {str(g * 16): {k: dict(mean=float(B[a][:, :, g - 1, i].mean()), se=float(B[a][:, :, g - 1, i].mean(1).std(ddof=1) / np.sqrt(len(clips))), seed_sd=float(B[a][:, :, g - 1, i].mean(0).std(ddof=1))) for i, k in enumerate(KEYS)} for g in G}
        rep["center"][a] = {str(g * 16): {k: float(C[a][:, :, j, i].mean()) for i, k in enumerate(KEYS)} for j, g in enumerate(G)}
    for p, q in (("es", "random"), ("es", "cem"), ("random", "cem")):
        rep["pairs"][f"{p}_vs_{q}"] = {k: {str(g * 16): dict(lower=int((B[p][:, :, g - 1, i].mean(1) < B[q][:, :, g - 1, i].mean(1)).sum()), p=float(wilcoxon(B[p][:, :, g - 1, i].mean(1), B[q][:, :, g - 1, i].mean(1)).pvalue)) for g in G} for i, k in enumerate(KEYS) if k in ("ratio", "mpjpe", "mmpjpe", "mpjae", "fall_s")}
    rep["pairs"]["es_center_vs_random_best"] = {k: {str(g * 16): dict(lower=int((C["es"][:, :, j, i].mean(1) < B["random"][:, :, g - 1, i].mean(1)).sum()), p=float(wilcoxon(C["es"][:, :, j, i].mean(1), B["random"][:, :, g - 1, i].mean(1)).pvalue)) for j, g in enumerate(G)} for i, k in enumerate(KEYS) if k in ("ratio", "mpjpe", "mmpjpe", "mpjae")}
    (CUR / "means.json").write_text(json.dumps(rep, indent=1))
    x = (np.arange(64) + 1) * 16; fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=170, facecolor=SURF)
    for b, (k, lab) in zip(ax, (("mpjpe", "MPJPE (cm)"), ("mmpjpe", "MMPJPE (cm)"), ("mpjae", "MPJAE (m/s²)"))):
        i = KEYS.index(k)
        for a, name, col in ALGOS:
            Y = B[a][:, :, :, i].mean(1); mean = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(len(clips)); b.fill_between(x, mean - se, mean + se, color=col, alpha=0.16, lw=0, zorder=2); b.plot(x, mean, color=col, lw=2, zorder=3, label=name)
        z = rep["z0"][k]; b.axhline(z, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1); b.text(x[-1], z, f"z0 {z:.2f}", color=MUTED, fontsize=9, va="bottom", ha="right")
        b.set_xlabel("simulated rollouts", fontsize=10, color=MUTED); b.set_ylabel(lab + " (lower is better)", fontsize=10, color=MUTED); b.set_facecolor(SURF); b.grid(color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=0, labelsize=9)
        for sp in ("top", "right", "left", "bottom"):
            b.spines[sp].set_visible(False)
    ax[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right", bbox_to_anchor=(1, 0.86))
    fig.suptitle("Child, 52 motions: best latent so far, mean over 260 runs (band: standard error over motions)", x=0.01, ha="left", fontsize=12, color=INK); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(CUR / "convergence_metrics_mean.png", facecolor=SURF); print("JSON"+json.dumps(rep))


if __name__ == "__main__":
    main()
