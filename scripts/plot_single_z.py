#!/usr/bin/env python3
"""plot_single_z.py -- figures for scripts/single_z_search.py's three objectives.

Reads only what the searches wrote (summary.json, curve.csv, best.npz,
origin_z.npz) plus the reference clip, and re-scores every trajectory here, so
no number in the figures is transcribed by hand.

ONE QUESTION PER FILE, and no in-image title -- the caption belongs wherever
the figure gets used, and combining panels only shrinks them past legibility:

  curve_align.png   HOW each search went, one file per objective, since
  curve_phys.png    L_align, L_phys and their sum are three different
  curve_both.png    quantities and side-by-side heights would mean nothing.
                    The best z is starred on the curve, labelled with its
                    cosine to origin_z.
  rank.png          each z scored on the sum -- the only shared axis.
  plane.png         where each objective lands in (L_align, L_phys).
  travel.png        what the rollout physically does.
  latent.png        cos(z, origin_z) over the search: did it actually move,
                    or is the best z a nudge on the one it started from?

A note on the retargeted reference. It is drawn as a landmark, NOT a lower
bound: it is a kinematic playback with no physics behind it, and it carries
real smooth/com_support cost of its own, so a physical rollout can and does
score BELOW it on L_phys.

Usage (from project root, after the three searches have finished). --dir is
either the parent of the three run dirs or the run prefix itself, and the
figures are written into whichever you pass -- use the prefix form when you
have more than one clip, since the figure names are fixed:
    uv run scripts/plot_single_z.py
    uv run scripts/plot_single_z.py --dir outputs/single_z --clip move-ego-0-2/move-ego-0-2_4
    uv run scripts/plot_single_z.py --dir outputs/single_z/rotate-y--5-0.8_0 \
        --clip rotate-y--5-0.8/rotate-y--5-0.8_0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK_2 = "#52514e"
INK_MUTED = "#8a8985"; GRID = "#e4e3df"
# palette slots 1-4, in order -- the ordering is the CVD-safety mechanism
CAT = {"align": "#2a78d6", "phys": "#eb6834", "both": "#1baf7a",
       "origin_z": "#eda100", "reference": "#8a8985"}
OBJ = ["align", "phys", "both"]
WHAT = {"align": "L_align", "phys": "L_phys", "both": "L_align + L_phys"}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)


def project_z(z):
    """Onto the sphere of radius sqrt(dim), metamotivo's project_z. Needed
    because averaging per-frame latents leaves the sphere, and a cosine against
    an off-sphere mean is not the cosine against a usable z."""
    a = np.atleast_2d(np.asarray(z, dtype=np.float64))
    r = np.sqrt(a.shape[-1])
    return (a * (r / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12))
            ).reshape(np.shape(z))


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# cos between two independent uniform directions on S^255 has sd 1/sqrt(256).
# Every cosine here is meaningless without it: 0.05 in 256 dimensions is not
# "slightly aligned", it is inside the noise floor of two unrelated vectors.
CHANCE_SD = 1.0 / np.sqrt(256)


def diagnostics(qpos, fps):
    """Loss-independent facts about what the body did, so the figure can say
    what a cost number means. up_z = 2*(qy*qz + qw*qx) -- this asset's rest
    quaternion maps local Y to world +Z, see losses._fall_penalty."""
    qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    up = 2.0 * (qy * qz + qw * qx)
    return {
        "travel": float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
        "upright_frac": float((up > 0.8).mean()),
        "seconds": len(qpos) / fps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/single_z")
    p.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    p.add_argument("--body", default="child")
    p.add_argument("--xml", default=None,
                   help="default: whatever the summaries were produced on")
    p.add_argument("--phys-weights", default=None,
                   help="default: whatever the summaries used")
    args = p.parse_args()

    from model.simple.config import ESConfig
    from model.simple.train import compute_batch_cost

    root = Path(args.dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    task, stem = args.clip.split("/")

    # --dir takes either shape, because both are natural to type:
    #   the PARENT of the three run dirs   outputs/single_z
    #   the run PREFIX itself              outputs/single_z/<stem>_0
    # i.e. the same string the searches got as --out minus the _<objective>.
    # The prefix form is the useful one for a second clip: the three figure
    # names are fixed, so pointing every clip at the same parent overwrites
    # the previous clip's figures.
    layouts = [{o: root / f"{stem}_{o}" for o in OBJ},
               {o: root.parent / f"{root.name}_{o}" for o in OBJ}]
    for dirs in layouts:
        if all((d / "summary.json").exists() for d in dirs.values()):
            break
    else:
        raise SystemExit(
            "no summary.json under either reading of --dir:\n  "
            + "\n  ".join(str(d) for lay in layouts for d in lay.values())
            + "\nrun scripts/single_z_search.py for each objective first")
    root.mkdir(parents=True, exist_ok=True)

    S = {o: json.loads((dirs[o] / "summary.json").read_text()) for o in OBJ}
    C = {o: list(csv.DictReader(open(dirs[o] / "curve.csv"))) for o in OBJ}

    # Every search must have run the same experiment, or the panels are not one
    # figure -- check rather than assume, since the three are launched separately.
    for k in ("clip", "xml", "phys_weights", "steps", "init", "obs_scale"):
        vals = {S[o].get(k) for o in OBJ}
        if len(vals) > 1:
            raise SystemExit(f"the three runs disagree on {k}: {vals}")

    xml = Path(args.xml) if args.xml else Path(S["align"]["xml"])
    fk = mujoco.MjModel.from_xml_path(str(xml))
    ref = np.load(REPO_ROOT / "data" / args.body / "retargeting_motion"
                  / task / f"{stem}.npz")["qpos"]

    cfg = ESConfig()
    cfg.phys_weights = args.phys_weights or S["align"]["phys_weights"]
    cfg.phys_fall_ref = S["align"]["phys_fall_ref"]
    cfg.lambda_align = cfg.lambda_phys = 1.0        # scoring, not searching

    def score(qpos):
        _, a, p = compute_batch_cost(fk, cfg, qpos[None], [ref])
        return float(a[0]), float(p[0])

    # --- re-score every trajectory from its saved qpos -----------------------
    pts = {}
    for o in OBJ:
        q = np.load(dirs[o] / "best.npz")["qpos"].astype(np.float64)
        a, ph = score(q)
        pts[o] = dict(align=a, phys=ph, **diagnostics(q, cfg.control_fps))
    q0p = dirs["align"] / "origin_z.npz"
    if q0p.exists():
        q0 = np.load(q0p)["qpos"].astype(np.float64)
        a, ph = score(q0)
        pts["origin_z"] = dict(align=a, phys=ph, **diagnostics(q0, cfg.control_fps))
    else:                                   # older runs did not save it
        s = S["both"]["origin_z"]
        pts["origin_z"] = dict(align=s["align"], phys=s["phys"],
                               travel=float("nan"), upright_frac=float("nan"),
                               seconds=float("nan"))
        print(f"note: {q0p} missing -- origin_z's rollout diagnostics unavailable")
    steps = S["align"]["steps"]
    ra, rp = score(ref[:steps].astype(np.float64))
    pts["reference"] = dict(align=ra, phys=rp,
                            **diagnostics(ref[:steps], cfg.control_fps))
    for k, v in pts.items():
        v["sum"] = v["align"] + v["phys"]

    # --- how far each z sits from z0, the latent every rollout starts at ------
    zvec = {"origin_z": project_z(np.load(REPO_ROOT / "data" / "origin_z" / task
                                          / f"{stem}.npy").reshape(-1))}
    for o in OBJ:
        zvec[o] = np.load(dirs[o] / "best_z.npy").reshape(-1).astype(np.float64)
    for k, z in zvec.items():
        pts[k]["cos_z0"] = cos(z, zvec["origin_z"])

    # "step" = one Adam update on z, which the ES literature and the search's
    # own logs call a generation -- same thing, and the figures use the
    # optimiser word. One step is 2*pairs rollouts, fixed for the whole run, so
    # steps and evaluations are one clock at a fixed ratio. The x axis is in
    # evaluations because that is what costs wall-clock and what stays
    # comparable when --pairs changes; the ratio is on the axis so the step
    # numbers in the search log and in summary.json convert by eye.
    per_step = 2 * S["align"]["pairs"]
    xlab = f"rollout evaluations   ({per_step} per step)"

    col = lambda o, k: np.array([float(r[k]) for r in C[o]])
    ev = lambda o: np.array([int(r["evals"]) for r in C[o]])

    # One question per FILE. Nothing is combined: a figure that answers two
    # questions gets read as answering one, and the panels shrink to where the
    # numbers stop being legible. No in-image title either -- the caption lives
    # wherever the figure is used, and the filename says which figure it is.

    def finish(fig, name):
        fig.savefig(root / name, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {root / name}")

    def at_best(o):
        """(evals, cost, cos_z0) of the best z.

        summary.json and curve.csv both spell the step "gen" -- that is the
        on-disk name and is left alone; only what the reader sees changes.

        cos comes from summary.json, NOT from curve.csv's cos_z0 at that
        step: the best z is a perturbed SAMPLE and cos_z0 tracks the ES mean,
        so the two differ by roughly sigma.
        """
        g = S[o]["best"]["gen"]
        row = min(C[o], key=lambda r: abs(int(r["gen"]) - g))
        return int(row["evals"]), S[o]["best"]["cost"], S[o]["cos_best_z0"]

    # ============ 1. curve_<objective>.png -- how each search went ===========
    # One file per objective: L_align, L_phys and their sum are three different
    # quantities, so putting them side by side invites a comparison of heights
    # that means nothing.
    for o in OBJ:
        fig = plt.figure(figsize=(7.2, 4.4), facecolor=SURFACE)
        ax = fig.add_subplot(111); style(ax)
        x = ev(o)
        ax.fill_between(x, col(o, "gen_best"), col(o, "gen_mean"), color=CAT[o],
                        alpha=0.12, linewidth=0, zorder=2,
                        label="each step's samples")
        # The ES ITERATE, evaluated every --eval-every steps. It is what
        # rollout_z_trace.py --which mean plays back, and unlike best_so_far it
        # is NOT monotone -- best_so_far is a running minimum and cannot go up,
        # so a figure showing only that one makes the trace's wobble look like a
        # contradiction rather than the thing it actually is.
        mz = np.array([(int(r["evals"]), float(r["mean_z_cost"]))
                       for r in C[o] if r["mean_z_cost"]])
        if len(mz):
            ax.plot(mz[:, 0], mz[:, 1], color=CAT[o], linewidth=1.3, linestyle="--",
                    zorder=4, label="the ES iterate")
        ax.plot(x, col(o, "best_so_far"), color=CAT[o], linewidth=2.4, zorder=5,
                label="best sample so far")

        z0c = S[o]["origin_z"]["cost"]
        ax.scatter([x[0]], [z0c], s=52, color=CAT[o], zorder=6,
                   edgecolor=SURFACE, linewidth=1.8)
        ax.annotate(f"origin_z  {z0c:.2f}", (x[0], z0c), textcoords="offset points",
                    xytext=(8, 2), color=INK_2, fontsize=9)

        # WHERE the best z was found, marked on the curve itself, carrying the
        # one number that says whether it is a different z at all: its cosine to
        # origin_z. A low cost reached at cos ~ 0 is a different direction in
        # latent space; the same cost reached at cos ~ 1 would be a nudge.
        bx, by, bc = at_best(o)
        ax.scatter([bx], [by], s=190, marker="*", color=CAT[o], zorder=7,
                   edgecolor=SURFACE, linewidth=1.6)
        refc = S[o]["reference_floor"]["cost"]
        if refc > 0:
            ax.plot([x[0], x[-1]], [refc, refc], color=INK_MUTED, linewidth=1.4,
                    linestyle=":", zorder=3)
            ax.annotate(f"reference  {refc:.3f}", (x[0], refc),
                        textcoords="offset points", xytext=(4, -6), ha="left",
                        va="top", color=INK_MUTED, fontsize=9)
        # x is ALWAYS log: the search is essentially over by ~300 evaluations
        # out of 10000, so on a linear axis 97% of the width is a flat line.
        ax.set_xscale("log"); ax.set_xlim(10, 1.45 * x[-1])

        # y is log only when the numbers need it. L_phys falls ~3000x, which no
        # linear axis can show; L_align falls 2x and L_align+L_phys 9x, and
        # forcing those onto a log axis buys nothing while costing readable
        # ticks (6x10^-1 instead of 0.6) and a distorted sense of the drop.
        lo = min(col(o, "best_so_far").min(), by)
        hi = max(z0c, col(o, "gen_mean").max())
        if hi / max(lo, 1e-12) > 30:
            ax.set_yscale("log")
            ylo, yhi = ax.get_ylim()
            ylo, yhi = ylo * 0.5, yhi * 1.7          # headroom, multiplicative
        else:
            ylo, yhi = 0.0, hi * 1.28                # ... and additive
        ax.set_ylim(ylo, yhi)

        # The star's numbers go in a FIXED corner with a leader line, not in an
        # offset next to the star. The curve is monotone decreasing on log-log,
        # so the star is always bottom-right and every offset direction from it
        # runs into either the curve, the samples band or the axis -- which is
        # whack-a-mole. Two corners are free by the same monotonicity: below-left
        # of the curve (legend) and above-right of it (this).
        ax.annotate(f"best z  {by:.3f}\ncos(z, origin_z) = {bc:+.3f}",
                    xy=(bx, by), xycoords="data",
                    xytext=(0.985, 0.985), textcoords="axes fraction",
                    ha="right", va="top", color=CAT[o], fontsize=9.5,
                    fontweight="bold", zorder=8,
                    arrowprops=dict(arrowstyle="-", color=CAT[o], linewidth=0.9,
                                    alpha=0.45, shrinkA=2, shrinkB=6))
        ax.set_xlabel(xlab, color=INK_2, fontsize=10)
        ax.set_ylabel(WHAT[o], color=CAT[o], fontsize=11, fontweight="bold")
        leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left",
                        handlelength=1.6, borderaxespad=0.4)
        for t in leg.get_texts():
            t.set_color(INK_2)
        finish(fig, f"curve_{o}.png")

    # ============ 2. rank.png -- which z won on the only shared axis =========
    fig = plt.figure(figsize=(5.2, 4.6), facecolor=SURFACE)
    ax = fig.add_subplot(111); style(ax)
    order = sorted(pts, key=lambda k: -pts[k]["sum"])
    xs = np.arange(len(order))
    ax.bar(xs, [pts[k]["sum"] for k in order], 0.6,
           color=[CAT[k] for k in order], edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, k in zip(xs, order):
        ax.text(xi, pts[k]["sum"], f"{pts[k]['sum']:.2f}", ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=9, fontweight="bold", rotation=20, ha="right")
    ax.set_ylim(0, max(pts[k]["sum"] for k in order) * 1.2)
    ax.set_ylabel("L_align + L_phys", color=INK_2, fontsize=10)
    finish(fig, "rank.png")

    # ============ 3. plane.png -- the align/phys trade-off ==================
    fig = plt.figure(figsize=(6.4, 5.0), facecolor=SURFACE)
    ax = fig.add_subplot(111); style(ax)
    plotted = [k for k in ("origin_z", *OBJ) if pts[k]["align"] > 0]
    xlo = min(pts[k]["align"] for k in plotted) * 0.45
    xhi = max(pts[k]["align"] for k in plotted) * 2.2
    ylo = min([pts[k]["phys"] for k in plotted] + [pts["reference"]["phys"]]) * 0.55
    yhi = max(pts[k]["phys"] for k in plotted) * 1.8
    for c in (0.25, 0.5, 1.0, 2.0, 5.0):          # iso-cost: L_align + L_phys = c
        xsv = np.geomspace(xlo, c * 0.999, 240)
        ax.plot(xsv, c - xsv, color=GRID, linewidth=1.4, zorder=1)
        # only label a line whose label lands INSIDE the axes: bbox_inches
        # "tight" grows the saved figure around stray text, so a label parked
        # off-axis silently adds a band of blank canvas
        lx, ly = c * 0.55, c * 0.45
        if xlo < lx < xhi and ylo < ly < yhi:
            ax.text(lx, ly, f"sum {c:g}", color=INK_MUTED, fontsize=8,
                    rotation=-38, ha="center", va="center", zorder=1)
    ax.axhline(pts["reference"]["phys"], color=INK_MUTED, linewidth=1.6,
               linestyle=":", zorder=2)
    ax.text(0.985, 0.965, "retargeted reference", transform=ax.transAxes,
            ha="right", va="top", color=INK_MUTED, fontsize=8.5)
    for k in plotted:
        ax.scatter([pts[k]["align"]], [pts[k]["phys"]], s=140, color=CAT[k],
                   zorder=5, edgecolor=SURFACE, linewidth=2.5)
        ax.annotate(k, (pts[k]["align"], pts[k]["phys"]), textcoords="offset points",
                    xytext=(11, 7), color=CAT[k], fontsize=10, fontweight="bold",
                    zorder=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("L_align  (reproduces the reference ->)", color=INK_2, fontsize=10)
    ax.set_ylabel("L_phys  (physically clean ->)", color=INK_2, fontsize=10)
    finish(fig, "plane.png")

    # ============ 4. travel.png -- what the rollout actually does ============
    fig = plt.figure(figsize=(6.4, 4.2), facecolor=SURFACE)
    ax = fig.add_subplot(111); style(ax)
    bars = [k for k in ("reference", *OBJ, "origin_z")
            if not np.isnan(pts[k]["travel"])]
    bars.sort(key=lambda k: pts[k]["travel"])
    y = np.arange(len(bars))
    vals = [pts[k]["travel"] for k in bars]
    ax.barh(y, vals, 0.6, color=[CAT[k] for k in bars], edgecolor=SURFACE,
            linewidth=2, zorder=3)
    hi = max(vals) * 1.45
    for yi, k in zip(y, bars):
        v, u = pts[k]["travel"], 100 * pts[k]["upright_frac"]
        if v > 0.15 * hi:                      # long bars label inside
            ax.text(v - hi * 0.02, yi, f"{v:.1f} m", ha="right", va="center",
                    fontsize=10, color="#ffffff", fontweight="bold", zorder=4)
        else:
            ax.text(v + hi * 0.02, yi, f"{v:.1f} m", va="center", fontsize=10,
                    color=INK, fontweight="bold")
        ax.text(0.985, (yi + 0.5) / len(bars), f"{u:.0f}% upright",
                transform=ax.transAxes, ha="right", va="center", fontsize=9,
                color=INK_2 if u > 90 else "#e34948")
    ax.set_yticks(y); ax.set_yticklabels(bars, fontsize=10, fontweight="bold")
    ax.set_xlim(0, hi)
    ax.set_xlabel(f"distance travelled in {pts['reference']['seconds']:.0f} s (m)",
                  color=INK_2, fontsize=10)
    finish(fig, "travel.png")

    # ============ 5. latent.png -- did the search actually leave origin_z? ===
    # Only origin_z. It is the one latent every rollout in the repo starts
    # from, so "how far did we move" is the question with consequences (an
    # adapter initialised near identity has to cover this distance); the other
    # reference latents answer a different question and were crowding this one.
    fig = plt.figure(figsize=(6.8, 4.6), facecolor=SURFACE)
    ax = fig.add_subplot(111); style(ax)
    ax.axhspan(-2 * CHANCE_SD, 2 * CHANCE_SD, color=GRID, zorder=1)
    ax.axhline(0, color=INK_MUTED, linewidth=1.5, zorder=3)
    finals = sorted(OBJ, key=lambda o: -S[o]["cos_best_z0"])
    for j, o in enumerate(finals):
        x = ev(o)
        ax.plot(x, col(o, "cos_z0"), color=CAT[o], linewidth=2.0, zorder=4)
        bx, _, bc = at_best(o)
        ax.scatter([bx], [bc], s=190, marker="*", color=CAT[o], zorder=6,
                   edgecolor=SURFACE, linewidth=1.6)
        y = S[o]["cos_best_z0"]
        ax.scatter([x[-1]], [y], s=52, color=CAT[o], zorder=5,
                   edgecolor=SURFACE, linewidth=1.8)
        # the three finals can land within 0.01 of each other -- fan the labels
        ax.text(x[-1] * 1.15, y + 0.06 * (1 - j), f"{o}  {y:+.3f}", color=CAT[o],
                fontsize=9.5, fontweight="bold", va="center")
    ax.text(0.015, 0.965, f"grey band = +-2 sd of chance ({2 * CHANCE_SD:.2f}) in "
            "256 dims;  inside it = no better than an unrelated direction\n"
            "star = the step the best z came from",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=INK_MUTED, zorder=7)
    ax.set_xscale("log"); ax.set_xlim(10, 2.6 * ev(OBJ[0])[-1]); ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(xlab, color=INK_2, fontsize=10)
    ax.set_ylabel("cos(z, origin_z)", color=INK_2, fontsize=10)
    finish(fig, "latent.png")

    print(f"\n{'':11s}{'L_align':>9s}{'L_phys':>9s}{'sum':>8s}{'travel_m':>10s}{'upright':>9s}")
    for k in sorted(pts, key=lambda k: pts[k]["sum"]):
        v = pts[k]
        print(f"{k:11s}{v['align']:9.4f}{v['phys']:9.4f}{v['sum']:8.4f}"
              f"{v['travel']:10.2f}{100 * v['upright_frac']:8.0f}%")

    print(f"\ncos(best z, origin_z)   (chance is 0 +- {CHANCE_SD:.3f} in 256 dims)")
    for o in OBJ:
        v = S[o]["cos_best_z0"]
        near = "  <- inside chance: a different direction, not a nudge" \
            if abs(v) <= 2 * CHANCE_SD else ""
        print(f"  best[{o}]{'':<6s}{v:+8.4f}{near}")


if __name__ == "__main__":
    main()
