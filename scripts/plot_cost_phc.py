#!/usr/bin/env python3
"""plot_cost_phc.py -- figure for the paper's "Analysis of the Representation-Based Objective" (docs/latex/representation_analysis.tex):
as our tracking cost decreases, PHC's tracking reward increases.

Data: outputs/cost_corr/per_latent.npz (scripts/cost_correlation.py): for each of the 52 child task motions, every latent the
single-latent searches visited (25,782 in all, a median of 470 per motion), re-rolled; per latent the tracking cost (BFM term +
1.0 heading + 0.1 position, no anchor; key <stem>__track) and PHC's tracking reward (1 - model/phc_reward.py's cost; key <stem>__phc).
(a) Within each motion, the latents are split by their cost into 10 equal-count groups (decile 1 = the lowest cost); a motion's curve is
    the mean PHC reward of each group minus the motion's mean PHC reward ("Delta PHC reward"; motions with different reward levels
    then share one axis).
    Grey: the 52 motions; blue: their mean, band +- one standard error over motions.
(b) Per motion, the Spearman correlation between the tracking cost and the PHC reward (outputs/cost_corr/results.json); the dashed
    line is the median. Significant (p < 0.05) and negative in 51 of the 52 motions.
Paper style as the other figures: STIX, 5.5 in wide, 10 pt.
Out: docs/figures/cost_phc_correlation.{pdf,png}. usage: uv run python scripts/plot_cost_phc.py
Inputs are read through crossembodiment.paths.read_path ($CE_OUTPUTS_ROOT), the figure goes to $CE_FIGURES_ROOT (default docs/figures)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossembodiment import paths                                         # noqa: E402
SRC = paths.read_path("outputs/cost_corr")
INK, MUTED, GRID, BLUE, GREY = "#22252a", "#52514e", "#e6e6e3", "#2a78d6", "#b9b8b4"
NB = 10


def deciles(cost, rew):
    """mean reward of each cost decile (equal-count groups after sorting by cost), minus the motion's mean reward"""
    o = np.argsort(cost, kind="stable"); groups = np.array_split(o, NB)
    return np.array([rew[g].mean() for g in groups]) - rew.mean()


def main():
    D = np.load(SRC / "per_latent.npz"); R = json.load(open(SRC / "results.json"))
    stems = sorted({k.split("__")[0] for k in D.files})
    curves = np.stack([deciles(D[f"{s}__track"], D[f"{s}__phc"]) for s in stems])
    rho = np.array([R["per_motion"][s]["track_phc"]["spearman"] for s in stems]); p = np.array([R["per_motion"][s]["track_phc"]["spearman_p"] for s in stems])
    n_lat = sum(len(D[f"{s}__track"]) for s in stems)
    assert len(stems) == 52 and abs(np.median(rho) - R["summary"]["track"]["phc"]["spearman_median"]) < 1e-12
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.25), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.2, top=0.88, wspace=0.3)
    x = np.arange(1, NB + 1); ax = axs[0]
    for c in curves:
        ax.plot(x, c, color=GREY, lw=0.6, alpha=0.55, zorder=2)
    mu, se = curves.mean(0), curves.std(0, ddof=1) / np.sqrt(len(curves))
    ax.fill_between(x, mu - se, mu + se, color=BLUE, alpha=0.2, lw=0, zorder=3); ax.plot(x, mu, color=BLUE, lw=2.0, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xticks([1, 2, 4, 6, 8, 10]); ax.set_xlim(0.7, NB + 0.3)
    ax.set_xlabel("Cost decile within a motion (1 = lowest)", labelpad=1); ax.set_ylabel(r"$\Delta$ PHC reward")
    ax.set_title("(a) Lower cost, higher reward", fontsize=10, color=INK, loc="left")
    ax = axs[1]; bins = np.linspace(-1, 0.2, 25)
    ax.hist(rho, bins=bins, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
    ax.axvline(np.median(rho), color=INK, lw=1.0, ls="--", zorder=4)
    ax.text(np.median(rho) + 0.05, ax.get_ylim()[1] * 0.95, f"median ${np.median(rho):.2f}$", color=INK, fontsize=9, va="top")
    ax.text(0.97, 0.55, f"{int(((rho < 0) & (p < 0.05)).sum())} of {len(rho)} motions\n" + r"$\rho<0$, $p<0.05$", transform=ax.transAxes, ha="right",
            va="center", fontsize=9, color=INK)
    ax.set_xlim(-1.02, 0.2); ax.set_xlabel(r"Spearman $\rho$ (cost, PHC reward)", labelpad=1); ax.set_ylabel("Motions")
    ax.set_title("(b) Correlation per motion", fontsize=10, color=INK, loc="left")
    for a in axs:
        a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    OUT = paths.figures_root()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cost_phc_correlation.{ext}", dpi=300 if ext == "png" else None)
    print(f"{len(stems)} motions, {n_lat} latents | decile 1 vs 10 mean difference {mu[0] - mu[-1]:+.3f} reward | median rho {np.median(rho):.3f}, "
          f"IQR [{np.percentile(rho, 25):.2f}, {np.percentile(rho, 75):.2f}] | monotonic mean curve: {bool(np.all(np.diff(mu) < 0))}")


if __name__ == "__main__":
    main()
