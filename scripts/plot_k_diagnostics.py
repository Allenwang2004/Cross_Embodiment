#!/usr/bin/env python3
"""One figure per body answering "can this body's actuator k be trusted?".

torque_ratio_across_motions.py already draws the motion-vs-motion correlation
heatmap. This puts that next to the three things you actually read before
shipping a robots_torque_full.xml, on one sheet and on one shared joint axis:

  A  k per actuator, measured vs the geometric prediction (log scale -- k spans
     two decades on a half-height body, and a linear axis would flatten the
     whole leg chain into the baseline).
  B  measured / geometric, with the +-5% and +-10% bands from
     docs/new_body.md Step 5. Joints outside +-10% are named: those are where
     the geometric fallback would be wrong if R^2 ever forced one.
  C  R^2 of all 54 x 69 per-motion fits. The Step 5 question is not "is the
     median high" (it always is) but how much mass sits below --r2-min 0.9,
     because those fits are the ones that fall back.
  D  the correlation heatmap: is k a body property or a motion property.

B and C answer different questions and can disagree -- a joint can fit at
R^2 = 1.000 and still sit 28% off the geometric prediction, which is the
signature of a correct measurement against a prediction that does not hold at
that scale, not of a bad fit. Reading only one of the two panels hides that.

Usage:
  uv run scripts/plot_k_diagnostics.py --body s1_b1
  uv run scripts/plot_k_diagnostics.py --body s1_b1 --matrix-root outputs/torque_ratio_across_motions/gravity
  uv run scripts/plot_k_diagnostics.py --body s1_b1 s1_b2 s1_b3 --out-dir outputs/k_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# same palette as torque_ratio_across_motions.py, so a body's figures read as
# one set no matter which script drew them
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
MEASURED, PREDICTED, FLAG = "#1f5f8b", "#c3c2b7", "#b4472e"

CHAINS = [("left leg", 0, 12), ("right leg", 12, 24), ("torso", 24, 36),
          ("head", 36, 39), ("left arm", 39, 54), ("right arm", 54, 69)]


def read_matrix(p):
    rows = list(csv.reader(open(p)))
    return rows[0][1:], [r[0] for r in rows[1:]], np.array(
        [[float(x) if x else np.nan for x in r[1:]] for r in rows[1:]])


def read_agg(p):
    r = list(csv.DictReader(open(p)))
    return ([x["actuator"] for x in r],
            np.array([float(x["k_aggregate"]) for x in r]),
            np.array([float(x["k_predicted_subtree"]) for x in r]),
            np.array([int(x["n_motions"]) for x in r]))


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=7, length=3, color=AXIS)


def chain_bands(ax, n):
    """Alternating tints behind the six kinematic chains, plus their names."""
    for i, (name, lo, hi) in enumerate(CHAINS):
        if i % 2:
            ax.axvspan(lo - .5, hi - .5, color=GRID, alpha=.45, lw=0, zorder=0)
        ax.text((lo + hi - 1) / 2, 1.015, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xlim(-.5, n - .5)


def draw(body, mdir, out_png):
    names, km, kp, nm = read_agg(mdir / "_ratios_aggregate.csv")
    _, _, R2 = read_matrix(mdir / "_r2_matrix.csv")
    motions, _, C = read_matrix(mdir / "_corr.csv")
    n = len(names)
    x = np.arange(n)
    q = km / kp
    fallback = nm == 0

    fig = plt.figure(figsize=(15, 11.4), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.25],
                          hspace=.42, wspace=.16, left=.062, right=.985, top=.905, bottom=.055)

    fig.text(.062, .962, f"{body} — actuator k diagnostics", fontsize=15, color=INK, weight="bold")
    med_r2 = float(np.median(R2[np.isfinite(R2)]))
    fig.text(.062, .935,
             f"{len(motions)} motions × {n} joints   ·   k median {np.median(km):.4f}   ·   "
             f"measured/geometric median {np.median(q[~fallback]):.4f}   ·   "
             f"R² median {med_r2:.4f}   ·   {int(fallback.sum())} joints fell back to the prediction",
             fontsize=9.5, color=INK_2)

    # --- A: k per actuator ------------------------------------------------
    ax = fig.add_subplot(gs[0, :]); style(ax); chain_bands(ax, n)
    ax.vlines(x, np.minimum(km, kp), np.maximum(km, kp), color=AXIS, lw=.9, zorder=2)
    ax.scatter(x, kp, s=26, facecolor="white", edgecolor=PREDICTED, lw=1.3, zorder=3,
               label="geometric prediction  (subtree mass × lever)")
    ax.scatter(x, km, s=22, color=MEASURED, zorder=4, label="measured  (fitted over 54 motions)")
    ax.set_yscale("log")
    ax.set_ylabel("k  (torque ratio vs adult)", fontsize=9, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=5.4, color=INK_2)
    ax.yaxis.grid(True, color=GRID, lw=.7); ax.set_axisbelow(True)
    leg = ax.legend(fontsize=8.5, labelcolor=INK_2, loc="upper left", ncol=1,
                    facecolor=SURFACE, edgecolor=GRID, framealpha=.94)
    leg.get_frame().set_linewidth(.8)

    # --- B: measured / geometric -----------------------------------------
    ax = fig.add_subplot(gs[1, :]); style(ax); chain_bands(ax, n)
    ax.axhspan(.90, 1.10, color=GRID, alpha=.55, lw=0, zorder=0)
    ax.axhspan(.95, 1.05, color=GRID, alpha=.9, lw=0, zorder=0)
    ax.axhline(1, color=AXIS, lw=1, zorder=1)
    out = np.abs(q - 1) > .10
    ax.scatter(x[~out], q[~out], s=22, color=MEASURED, zorder=4)
    ax.scatter(x[out], q[out], s=34, color=FLAG, zorder=5)
    for i in np.where(out)[0]:
        ax.annotate(f"{names[i]}  {q[i]:.2f}", (i, q[i]), textcoords="offset points",
                    xytext=(0, 9 if q[i] > 1 else -15), ha="center", fontsize=6.6, color=FLAG)
    ax.set_ylabel("measured / geometric", fontsize=9, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=5.4, color=INK_2)
    pad = max(.13, np.abs(q - 1).max() * 1.25)
    ax.set_ylim(1 - pad, 1 + pad)
    ax.text(n - .8, 1.105, "±10%", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.text(n - .8, 1.055, "±5%", fontsize=7, color=MUTED, ha="right", va="bottom")

    # --- C: R² of every per-motion fit -----------------------------------
    ax = fig.add_subplot(gs[2, 0]); style(ax)
    v = R2[np.isfinite(R2)]
    lo = min(.5, float(np.percentile(v, .5)))
    bins = np.linspace(lo, 1.0, 60)
    ax.hist(np.clip(v, lo, 1.0), bins=bins, color=MEASURED, alpha=.85, lw=0)
    ax.axvline(.9, color=FLAG, lw=1.2, ls="--")
    below = float((v < .9).mean())
    ax.text(.9 - (1 - lo) * .02, ax.get_ylim()[1] * .85,
            f"--r2-min 0.9\n{100*below:.1f}% of fits below ",
            fontsize=8, color=FLAG, va="top", ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("R² of the per-motion, per-joint k fit", fontsize=9, color=INK_2)
    ax.set_ylabel("fits (log)", fontsize=9, color=INK_2)
    ax.yaxis.grid(True, color=GRID, lw=.7); ax.set_axisbelow(True)
    ax.set_title(f"C · every fit  (n = {v.size})", fontsize=9.5, color=INK, loc="left", pad=8)

    # --- D: motion x motion correlation ----------------------------------
    ax = fig.add_subplot(gs[2, 1]); style(ax)
    off = C[~np.eye(len(C), dtype=bool)]
    cmap = LinearSegmentedColormap.from_list("m", ["#f7f6f2", "#a8c4d6", MEASURED, "#123c5a"])
    im = ax.imshow(C, cmap=cmap, vmin=max(0, off.min()), vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK_2, labelsize=7, length=2)
    ax.set_title("D · k-vector agreement between motions", fontsize=9.5, color=INK, loc="left", pad=8)
    ax.set_xlabel(f"{len(motions)} motions, each a 69-dim k vector   ·   off-diagonal median "
                  f"{np.median(off):.3f}, min {off.min():.3f}", fontsize=8.5, color=INK_2)

    fig.text(.062, .018,
             "A/B share the joint axis. A joint can sit at R² = 1.000 in C and still miss the "
             "geometric prediction in B — that is a correct measurement against a prediction that "
             "does not hold at this scale, not a bad fit.",
             fontsize=8, color=MUTED)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--body", nargs="+", required=True)
    ap.add_argument("--matrix-root", default="outputs/torque_ratio_across_motions/gravity_advance")
    ap.add_argument("--out-dir", default="outputs/k_diagnostics")
    args = ap.parse_args()
    for b in args.body:
        mdir = ROOT / args.matrix_root / b
        assert (mdir / "_ratios_aggregate.csv").is_file(), (
            f"{mdir}/_ratios_aggregate.csv missing -- run torque_aggregate_motion_k.py first")
        draw(b, mdir, ROOT / args.out_dir / f"{b}.png")


if __name__ == "__main__":
    main()
