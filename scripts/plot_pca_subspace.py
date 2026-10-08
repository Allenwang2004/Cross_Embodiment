#!/usr/bin/env python3
"""plot_pca_subspace.py -- the dimensionality-reduction result as one figure (both costs).

Reads the two pca_subspace_test.py runs (MSE: outputs/pca_subspace vs outputs/c540_mse_a03; cosine + heading:
outputs/pca_subspace_bfmg vs outputs/c540_bfmglobal_a03) on the same 60 held-out clips.
  top     median % of the 64-generation full-space gain vs generation: PCA-8, full 256, random-8 (IQR wash)
  bottom  % reached after 8 generations (128 rollouts) vs search dimension: PCA k = 4..32, random k = 8, 32, full
Writes outputs/pca_subspace_figure.{png,pdf} and prints the table behind it.
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = [("Tracking MSE", "outputs/pca_subspace", "outputs/c540_mse_a03"),
        ("Cosine + heading + root-xy", "outputs/pca_subspace_bfmg", "outputs/c540_bfmglobal_a03")]
ARMS = ["pca4", "pca8", "pca16", "pca32", "rand8", "rand32", "full"]
G, AT = 16, 8
# reference palette slots 1-2 (validated all-pairs, light); the random control is a recessive neutral
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
PCA, FULL, RAND = "#2a78d6", "#eb6834", "#8d8c88"


def curves(out, src):
    out, src = REPO / out, REPO / src
    F = {a: [] for a in ARMS}
    for t, k, cat, sp, s in (l.split() for l in open(out / "clips.txt") if l.strip()):
        st = f"{t}_{k}"; ref = json.load(open(src / st / "summary.json"))
        c0, cr = ref["origin_z"]["cost"], ref["best"]["cost"]
        if c0 - cr < 1e-9:
            continue
        for a in ARMS:
            b = np.minimum.accumulate([min(float(r["best_so_far"]), c0) for r in csv.DictReader(open(out / a / st / "curve.csv"))])
            F[a].append(100 * np.concatenate([[0.0], (c0 - b[:G]) / (c0 - cr)]))
    return {a: np.array(v) for a, v in F.items()}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def main():
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2})
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2), gridspec_kw=dict(height_ratios=[1.25, 1]))
    fig.patch.set_facecolor(SURF)
    g = np.arange(G + 1)
    for j, (name, out, src) in enumerate(RUNS):
        F = curves(out, src); n = len(F["full"])
        # --- top: convergence ---------------------------------------------------------------------
        ax = axs[0, j]; style(ax)
        ax.axhline(80, color=INK2, lw=1, alpha=.35)
        ax.text(0.3, 81, "80%", color=INK2, fontsize=8, va="bottom")
        ends = {}
        for a, c, ls, lab in (("pca8", PCA, "-", "PCA 8-d"), ("full", FULL, "-", "full 256-d"), ("rand8", RAND, (0, (4, 3)), "random 8-d")):
            m = np.median(F[a], 0)
            if a != "rand8":
                ax.fill_between(g, np.percentile(F[a], 25, 0), np.percentile(F[a], 75, 0), color=c, alpha=.10, lw=0)
            ax.plot(g, m, color=c, lw=2, ls=ls, solid_capstyle="round", label=lab)
            ax.plot(g[-1], m[-1], "o", ms=7, color=c, mec=SURF, mew=2)
            ends[a] = m[-1]
        # end labels, nudged apart so close finishes stay readable
        order = sorted(ends, key=ends.get, reverse=True); ys = []
        for a in order:
            y = ends[a] if not ys else min(ends[a], ys[-1] - 5)
            ys.append(y)
            ax.annotate(f"{ends[a]:.0f}%", (g[-1], ends[a]), xytext=(G + 0.45, y), textcoords="data", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, G + 1.6); ax.set_ylim(0, 105); ax.set_xticks([0, 2, 4, 8, 12, 16])
        ax.set_xlabel("generation (16 rollouts each)")
        if j == 0:
            ax.set_ylabel("% of the 64-generation full-space gain\n(median over clips; band = IQR)")
            ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK, handlelength=2.6)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=INK)
        # --- bottom: budget-matched comparison across dimensions -----------------------------------
        ax = axs[1, j]; style(ax)
        at = {a: np.median(F[a][:, AT]) for a in ARMS}
        ks = [4, 8, 16, 32]
        ax.plot(ks, [at[f"pca{k}"] for k in ks], "-o", color=PCA, lw=2, ms=7, mec=SURF, mew=2, label="PCA k-d")
        ax.plot([8, 32], [at["rand8"], at["rand32"]], ls=(0, (4, 3)), marker="s", color=RAND, lw=2, ms=7, mec=SURF, mew=2, label="random k-d")
        ax.plot([256], [at["full"]], "D", color=FULL, ms=8, mec=SURF, mew=2, label="full 256-d")
        for k in ks:
            ax.annotate(f"{at[f'pca{k}']:.0f}", (k, at[f"pca{k}"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        for k in (8, 32):
            ax.annotate(f"{at[f'rand{k}']:.0f}", (k, at[f"rand{k}"]), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        ax.annotate(f"{at['full']:.0f}", (256, at["full"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 256]); ax.set_xticklabels(["4", "8", "16", "32", "256\n(full)"])
        lo = min(at.values()); ax.set_ylim(max(0, lo - 15), min(100, max(at.values()) + 10))
        ax.set_xlabel("search dimension k")
        if j == 0:
            ax.set_ylabel(f"% of gain after {AT} generations\n({16 * AT} rollouts, median)")
            ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK, handlelength=2.6)
        d_full, d_rand = F["pca8"][:, AT] - F["full"][:, AT], F["pca8"][:, AT] - F["rand8"][:, AT]
        ax.text(0.98, 0.04, f"PCA-8 ahead of full on {np.sum(d_full > 0)}/{n} clips (p = {wilcoxon(d_full).pvalue:.0e})\n"
                f"PCA-8 ahead of random-8 on {np.sum(d_rand > 0)}/{n} clips (p = {wilcoxon(d_rand).pvalue:.0e})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
        # --- table ---------------------------------------------------------------------------------
        reach = lambda c, x: np.where((c >= x).any(1), (c >= x).argmax(1), G + 1)
        print(f"\n{name} ({n} clips): median % after 1/2/4/8/16 gens | mean gens to 80% / 90% (not reached = {G + 1}) | reached 90% within {G}")
        for a in ARMS:
            print(f"  {a:6s} " + " / ".join(f"{np.median(F[a][:, x]):3.0f}" for x in (1, 2, 4, 8, 16))
                  + f" | {reach(F[a], 80).mean():4.1f} / {reach(F[a], 90).mean():4.1f} | {np.sum(reach(F[a], 90) <= G)}/{n}")
    fig.suptitle("ES on the principal directions of past corrections reaches a given gain in fewer generations",
                 x=0.07, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.07, 0.935, "60 held-out clips (25 from 5 unseen motions, 35 unseen clips of seen motions), child body, exact observations, seed 0.\n"
             "Basis = PCA of 455 other clips' corrections z* − z0. 100% = the clip's own 64-generation full-space search.", fontsize=9, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    for ext in ("png", "pdf"):
        fig.savefig(REPO / f"outputs/pca_subspace_figure.{ext}", dpi=160, facecolor=SURF)
    print(f"\nwrote outputs/pca_subspace_figure.png / .pdf")


if __name__ == "__main__":
    main()
