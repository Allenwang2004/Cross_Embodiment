#!/usr/bin/env python3
"""plot_obs_canon.py -- figure for the paper's "Effect of Geometry-Aware Canonicalization" (docs/latex/observation_canonicalization.tex,
docs/latex/fig_obs_canon.tex): the frozen task latent z0 on the child's 52 task motions under three ways of feeding the child's state
to the adult-trained policy -- raw child observations, per-body length scaling (each body's length features / that body's
child-to-adult length ratio, model/obs_scale.py) and our canonicalizer Phi_beta (model/exact_obs.py). No latent search.

Data: outputs/obs_canon_metrics/results.json (scripts/obs_canon_metrics.py; one rollout per motion and mapping).
(a) PHC success rate (world-frame mean body error never above 0.5 m adult-equivalent), share of the 52 motions.
(b) PHC tracking reward, mean over motions, error bar +- one standard error.
(c) MPJPE per motion (cm, log axis): one grey line per motion across the three mappings, the black bar the mean.
(d) Representation check: 1 - cosine similarity between the frozen backward embedding of the child's reference observed through the
    mapping and that of the adult's own motion (mean over frames), one dot per motion, log axis; the black bar the median.
Paper style as the other figures: STIX, 5.5 in wide, 10 pt.
Out: docs/figures/obs_canonicalization.{pdf,png}. usage: uv run python scripts/plot_obs_canon.py"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent; SRC = REPO / "outputs/obs_canon_metrics/results.json"; OUT = REPO / "docs/figures"
INK, MUTED, GRID, LINE = "#22252a", "#52514e", "#e6e6e3", "#c9c8c3"
MAPS = (("raw", "Raw observations", "#9a9994"), ("perbody", "Per-body length scaling", "#eb6834"), ("exact", r"$\Phi_\beta$ (ours)", "#2a78d6"))


def main():
    R = json.loads(SRC.read_text()); P = R["per_motion"]; st = sorted(P["exact"]); assert len(st) == 52
    v = {k: {m: np.array([P[k][s][m] for s in st], dtype=float) for m in ("success", "phc", "mpjpe")} for k, _, _ in MAPS}
    gap = {k: 1 - np.array([R["cosine"][s][k] for s in st]) for k, _, _ in MAPS}
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
    # default (user, 11/12): two panels, PHC reward and the backward-embedding gap; --full: the four-panel version (success, reward,
    # MPJPE, embedding gap) as obs_canonicalization_4panel.*
    FULL = "--full" in sys.argv
    if FULL:
        fig, axs = plt.subplots(1, 4, figsize=(5.5, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.12, 1.12]))
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.05, top=0.76, wspace=0.42)
    else:
        # 11/13 (user): lower than before (was 5.5 x 2.3 in) so that it is no taller than the paper's convergence figure
        fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.8))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.70, wspace=0.32)
        axs = [None, axs[0], None, axs[1]]
    x = np.arange(3); cols = [c for _, _, c in MAPS]
    title = lambda ax, t: ax.set_title(t, fontsize=10, color=INK, loc="left", pad=4)

    ax = axs[0]; y = [100 * v[k]["success"].mean() for k, _, _ in MAPS]
    if ax is not None:
      ax.bar(x, y, width=0.7, color=cols, zorder=3)
      for i, yi in enumerate(y):
          ax.text(i, yi + 1.2, f"{yi:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
      ax.set_ylim(0, 45); ax.set_yticks([0, 10, 20, 30, 40]); title(ax, "(a) Success (%)")

    ax = axs[1]; y = [v[k]["phc"].mean() for k, _, _ in MAPS]; e = [v[k]["phc"].std(ddof=1) / np.sqrt(len(st)) for k, _, _ in MAPS]
    ax.bar(x, y, width=0.7, color=cols, zorder=3)
    ax.errorbar(x, y, yerr=e, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
    for i, (yi, ei) in enumerate(zip(y, e)):
        ax.text(i, yi + ei + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, 0.5); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); title(ax, "(b) PHC reward" if FULL else "(a) PHC reward")

    ax = axs[2]; Y = np.stack([v[k]["mpjpe"] for k, _, _ in MAPS], 1); nb = int((Y[:, 2] < Y[:, 1]).sum())
    for row in (Y if ax is not None else []):
        ax.plot(x, row, color=LINE, lw=0.5, alpha=0.8, zorder=2)
    if ax is not None:
        for i, (k, _, c) in enumerate(MAPS):
            ax.scatter(np.full(len(st), i), Y[:, i], s=6, color=c, lw=0, alpha=0.8, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [Y[:, i].mean()] * 2, color=INK, lw=1.6, zorder=4, solid_capstyle="butt")
        ax.set_yscale("log"); ax.set_ylim(1.0, 60); ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); title(ax, "(c) MPJPE (cm)")

    ax = axs[3]; rng = np.random.default_rng(0)
    for i, (k, _, c) in enumerate(MAPS):
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(st)), gap[k], s=6, color=c, lw=0, alpha=0.8, zorder=3)
        ax.plot([i - 0.32, i + 0.32], [np.median(gap[k])] * 2, color=INK, lw=1.6, zorder=4, solid_capstyle="butt")
    ax.set_yscale("log"); ax.set_ylim(3e-5, 2); ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    title(ax, r"(d) 1 $-$ cosine" if FULL else r"(b) Policy embedding gap, 1 $-$ cosine")

    for a in [a for a in axs if a is not None]:
        a.set_xticks([]); a.set_xlim(-0.6, 2.6)
        a.yaxis.grid(True, color=GRID, lw=0.6, which="major"); a.set_axisbelow(True); a.tick_params(axis="y", labelsize=9, pad=1.5)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    hs = [matplotlib.patches.Patch(color=c, label=l) for _, l, c in MAPS]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, fontsize=10, handlelength=1.0, handleheight=0.8, columnspacing=1.6,
               bbox_to_anchor=(0.5, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"obs_canonicalization{'_4panel' if FULL else ''}.{ext}", dpi=300 if ext == "png" else None)
    print("success", [f"{100 * v[k]['success'].mean():.1f}" for k, _, _ in MAPS], "| phc", [f"{v[k]['phc'].mean():.3f}" for k, _, _ in MAPS],
          "| mpjpe", [f"{v[k]['mpjpe'].mean():.2f}" for k, _, _ in MAPS], f"| Phi < per-body MPJPE on {nb}/52",
          "| median 1-cos", [f"{np.median(gap[k]):.2g}" for k, _, _ in MAPS])


if __name__ == "__main__":
    main()
