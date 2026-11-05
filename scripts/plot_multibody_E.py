#!/usr/bin/env python3
"""plot_multibody_E.py -- figure for analysis E (8-dim shared-basis search on 8 unseen bodies x 16 clips).
Left: median % of the full-space search's 64-generation gain reached by generation g (log2 x), per basis.
Right: % at generation 4 per test body. Writes outputs/multibody_corr/analysis/figure_E.{png,pdf}."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from multibody_corr import E_ARMS, E_TEST, OUT, cell_dir, clips   # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {"full": dict(c=INK, ls="-", lw=2, label="full 256-d"),
         "multi_same": dict(c="#2a78d6", ls="-", lw=2, label="multi-body 8-d (clip seen on other bodies)"),
         "multi_loco": dict(c="#eb6834", ls="-", lw=2, label="multi-body 8-d (new body + new clip)"),
         "child524": dict(c="#1baf7a", ls="-", lw=2, label="child-only 8-d (524 other clips)"),
         "rand8": dict(c=MUTED, ls=(0, (4, 3)), lw=2, label="random 8-d")}
BODY = {"child": "child", "giant": "giant", "short_stocky": "short stocky", "gl0500a0550": "leg .5 · arm .55",
        "gl1500a1400": "leg 1.5 · arm 1.4", "gl0500a1400": "leg .5 · arm 1.4", "ax2m_t1000": "torso .75",
        "ax4m_t1000": "leg girth .7"}


def curve(d):
    b = {int(r["gen"]): float(r["best_so_far"]) for r in csv.DictReader(open(d / "curve.csv"))}
    return np.minimum.accumulate([b[i] for i in range(max(b) + 1)])


def main():
    P = {a: [] for a in STYLE}; PB = {(a, b): [] for a in STYLE for b in E_TEST}
    for c in clips():
        st = c.split("/")[1]
        for b in E_TEST:
            fd = cell_dir(b, c, 0); c0 = json.load(open(fd / "summary.json"))["origin_z"]["cost"]
            full = np.minimum(curve(fd), c0); gain = c0 - full[-1]
            if gain < 0.02 * c0:
                continue
            for a in STYLE:
                d = fd if a == "full" else OUT / "esearch" / a / b / f"{st}_s0"
                pct = 100 * (c0 - np.minimum(curve(d), c0)) / gain
                P[a].append(pct[:64]); PB[(a, b)].append(pct[3])
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    g = np.arange(1, 65)
    ends = []
    for a, s in STYLE.items():
        m = np.median(np.array(P[a]), 0)
        ax[0].plot(g, m, color=s["c"], ls=s["ls"], lw=s["lw"], label=s["label"])
        ends.append((m[-1], a))
    ax[0].set_xscale("log", base=2); ax[0].set_xticks([1, 2, 4, 8, 16, 32, 64]); ax[0].set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax[0].set_xlim(1, 64); ax[0].set_ylim(0, 105)
    ax[0].set_xlabel("generation (16 rollouts each)"); ax[0].set_ylabel("% of full-space 64-gen gain (median, 125 cells)")
    ax[0].grid(True, color=GRID, lw=0.8); ax[0].set_axisbelow(True)
    for sp in ("top", "right"):
        ax[0].spines[sp].set_visible(False); ax[1].spines[sp].set_visible(False)
    ax[0].legend(loc="lower right", frameon=False, fontsize=9)
    ax[0].set_title("Shared 8-d directions keep full quality and beat random 8-d", loc="left", color=INK, fontsize=11)
    # right: per-body dot plot at generation 4
    y = np.arange(len(E_TEST))[::-1]
    for k, (a, s) in enumerate(STYLE.items()):
        x = [np.median(PB[(a, b)]) for b in E_TEST]
        ax[1].scatter(x, y + (k - 2) * 0.12, s=36, color=s["c"], marker="o" if a != "rand8" else "D",
                      edgecolors=SURF, linewidths=1.5, zorder=3)
    ax[1].set_yticks(y); ax[1].set_yticklabels([BODY[b] for b in E_TEST])
    ax[1].set_xlim(30, 100); ax[1].set_xlabel("% of full-space 64-gen gain at generation 4 (median, 16 clips)")
    ax[1].grid(True, axis="x", color=GRID, lw=0.8); ax[1].set_axisbelow(True)
    ax[1].set_title("Early speed by unseen test body (colours as left; ◆ random 8-d)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = OUT / "analysis" / "figure_E"
    fig.savefig(str(out) + ".png", dpi=150, facecolor=SURF); fig.savefig(str(out) + ".pdf", facecolor=SURF)
    print(f"-> {out}.png")


if __name__ == "__main__":
    main()
