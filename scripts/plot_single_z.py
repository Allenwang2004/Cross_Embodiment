#!/usr/bin/env python3
"""plot_single_z.py -- figures for scripts/single_z_search.py's three objectives.

Reads only what the searches wrote (summary.json, curve.csv, best.npz,
origin_z.npz) plus the reference clip, and re-scores every trajectory here, so
no number in the figures is transcribed by hand.

Two figures, and the split between them is the point:

  curves.png    HOW the search went. One panel per objective, each on its OWN
                axis -- the three costs are not the same quantity (L_align,
                L_phys, and their sum), so a curve that sits lower in one panel
                than another says nothing. The one place the three CAN be
                ranked gets its own panel: every z scored on the sum.

  plane.png     WHERE each objective ended up, in the (L_align, L_phys) plane,
                and what the rollout physically does.

A note on the retargeted reference. It is drawn on both figures, but it is NOT
a lower bound: it is a kinematic playback with no physics behind it, and it
carries real smooth/com_support cost of its own, so a physical rollout can and
does score BELOW it on L_phys. It is a landmark, not a floor.

Usage (from project root, after the three searches have finished):
    uv run scripts/plot_single_z.py
    uv run scripts/plot_single_z.py --dir outputs/single_z --clip move-ego-0-2/move-ego-0-2_4
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
    dirs = {o: root / f"{stem}_{o}" for o in OBJ}
    missing = [str(d) for d in dirs.values() if not (d / "summary.json").exists()]
    if missing:
        raise SystemExit("no summary.json in:\n  " + "\n  ".join(missing)
                         + "\nrun scripts/single_z_search.py for each objective first")

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

    init_note = ("rollout starts from the reference's frame 0"
                 if S["align"].get("init") == "reference"
                 else "humenv Default standing reset")
    head = (f"{stem} on {args.body}: ES directly on z, "
            f"{S['align']['evals']} rollouts per objective  "
            f"(L_phys weights={cfg.phys_weights}, {init_note})")

    col = lambda o, k: np.array([float(r[k]) for r in C[o]])
    ev = lambda o: np.array([int(r["evals"]) for r in C[o]])

    # ======================= figure 1: the curves ============================
    fig = plt.figure(figsize=(14, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26)

    for i, o in enumerate(OBJ):
        ax = fig.add_subplot(gs[0, i]); style(ax)
        x = ev(o)
        ax.fill_between(x, col(o, "gen_best"), col(o, "gen_mean"), color=CAT[o],
                        alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, col(o, "best_so_far"), color=CAT[o], linewidth=2.4, zorder=5)
        z0c = S[o]["origin_z"]["cost"]
        ax.scatter([x[0]], [z0c], s=52, color=CAT[o], zorder=6,
                   edgecolor=SURFACE, linewidth=1.8)
        ax.annotate("origin_z", (x[0], z0c), textcoords="offset points",
                    xytext=(8, 2), color=INK_2, fontsize=8.5)
        refc = S[o]["reference_floor"]["cost"]
        if refc > 0:
            ax.plot([x[0], x[-1]], [refc, refc], color=INK_MUTED, linewidth=1.4,
                    linestyle=":", zorder=4)
            ax.annotate("reference", (x[-1], refc), textcoords="offset points",
                        xytext=(-4, 4), ha="right", color=INK_MUTED, fontsize=8.5)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(10, 1.3 * x[-1])
        ax.set_xlabel("rollout evaluations", color=INK_2, fontsize=10)
        ax.set_ylabel(WHAT[o], color=INK_2, fontsize=10)
        ax.set_title(f"({i + 1}) minimise {WHAT[o]}", color=CAT[o], fontsize=12,
                     fontweight="bold", loc="left", pad=20)
        ax.text(0, 1.02, f"{z0c:.2f} -> {S[o]['best']['cost']:.3f}"
                + (f"   (reference {refc:.3f})" if refc > 0 else "   (reference 0)"),
                transform=ax.transAxes, fontsize=9, color=INK_MUTED)

    ax = fig.add_subplot(gs[1, 0:2]); style(ax)
    order = sorted(pts, key=lambda k: -pts[k]["sum"])
    xs = np.arange(len(order))
    ax.bar(xs, [pts[k]["sum"] for k in order], 0.6,
           color=[CAT[k] for k in order], edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, k in zip(xs, order):
        ax.text(xi, pts[k]["sum"], f"{pts[k]['sum']:.2f}", ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(pts[k]["sum"] for k in order) * 1.2)
    ax.set_ylabel("L_align + L_phys", color=INK_2, fontsize=10)
    ax.set_title("the comparable question: what does each z score on the SUM?",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=20)
    ax.text(0, 1.02, "the only axis on which the three searches can be ranked "
            "against each other", transform=ax.transAxes, fontsize=9, color=INK_MUTED)

    ax = fig.add_subplot(gs[1, 2]); style(ax)
    finals = sorted(OBJ, key=lambda o: -S[o]["cos_best_z0"])
    for j, o in enumerate(finals):
        x = ev(o)
        ax.plot(x, col(o, "cos_z0"), color=CAT[o], linewidth=2.0, zorder=4)
        y = S[o]["cos_best_z0"]
        ax.scatter([x[-1]], [y], s=52, color=CAT[o], zorder=5,
                   edgecolor=SURFACE, linewidth=1.8)
        # the three finals can land within 0.01 of each other -- fan the labels
        ax.text(x[-1] * 1.15, y + 0.06 * (1 - j), o, color=CAT[o], fontsize=9.5,
                fontweight="bold", va="center")
    ax.axhline(0, color=INK_MUTED, linewidth=1.5, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(10, 2.6 * ev(OBJ[0])[-1]); ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel("rollout evaluations", color=INK_2, fontsize=10)
    ax.set_ylabel("cos(z, origin_z)", color=INK_2, fontsize=10)
    ax.set_title("how far z drifts from origin_z", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=20)
    ax.text(0, 1.02, "1 = unchanged, 0 = orthogonal", transform=ax.transAxes,
            fontsize=9, color=INK_MUTED)

    fig.suptitle(head + "   (band = best..mean of each generation's samples)",
                 color=INK, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.975)
    fig.savefig(root / "curves.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {root / 'curves.png'}")

    # ======================= figure 2: the plane =============================
    fig = plt.figure(figsize=(13, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.26)

    ax = fig.add_subplot(gs[0, 0]); style(ax)
    plotted = [k for k in ("origin_z", *OBJ) if pts[k]["align"] > 0]
    xlo = min(pts[k]["align"] for k in plotted) * 0.45
    xhi = max(pts[k]["align"] for k in plotted) * 2.2
    ylo = min([pts[k]["phys"] for k in plotted] + [pts["reference"]["phys"]]) * 0.55
    yhi = max(pts[k]["phys"] for k in plotted) * 1.8
    for c in (0.25, 0.5, 1.0, 2.0, 5.0):          # iso-cost: L_align + L_phys = c
        xsv = np.geomspace(xlo, c * 0.999, 240)
        ax.plot(xsv, c - xsv, color=GRID, linewidth=1.4, zorder=1)
        ax.text(c * 0.55, c * 0.45, f"sum {c:g}", color=INK_MUTED, fontsize=8,
                rotation=-38, ha="center", va="center", zorder=1)
    ax.axhline(pts["reference"]["phys"], color=INK_MUTED, linewidth=1.6,
               linestyle=":", zorder=2)
    ax.text(0.985, 0.965, f"retargeted reference, L_phys {pts['reference']['phys']:.3f}"
            "  (L_align 0)", transform=ax.transAxes, ha="right", va="top",
            color=INK_MUTED, fontsize=8.5)
    for k in plotted:
        ax.scatter([pts[k]["align"]], [pts[k]["phys"]], s=150, color=CAT[k],
                   zorder=5, edgecolor=SURFACE, linewidth=2.5)
        ax.annotate(k, (pts[k]["align"], pts[k]["phys"]), textcoords="offset points",
                    xytext=(11, 9), color=CAT[k], fontsize=10.5,
                    fontweight="bold", zorder=6)
        ax.annotate(f"sum {pts[k]['sum']:.2f}", (pts[k]["align"], pts[k]["phys"]),
                    textcoords="offset points", xytext=(11, -4), color=INK_2,
                    fontsize=8.5, zorder=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("L_align  (reproduces the reference ->)", color=INK_2, fontsize=10)
    ax.set_ylabel("L_phys  (physically clean ->)", color=INK_2, fontsize=10)
    ax.set_title("where each objective lands", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=20)
    ax.text(0, 1.02, "better is toward the top right; the dotted line is a "
            "landmark, not a floor", transform=ax.transAxes, fontsize=9,
            color=INK_MUTED)

    ax = fig.add_subplot(gs[0, 1]); style(ax)
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
    ax.set_title("what the rollout actually does", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=20)
    ax.text(0, 1.02, f"the task is {task}", transform=ax.transAxes, fontsize=9,
            color=INK_MUTED)

    fig.suptitle(head, color=INK, fontsize=13, fontweight="bold",
                 x=0.012, ha="left", y=1.02)
    fig.savefig(root / "plane.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {root / 'plane.png'}")

    print(f"\n{'':11s}{'L_align':>9s}{'L_phys':>9s}{'sum':>8s}{'travel_m':>10s}{'upright':>9s}")
    for k in sorted(pts, key=lambda k: pts[k]["sum"]):
        v = pts[k]
        print(f"{k:11s}{v['align']:9.4f}{v['phys']:9.4f}{v['sum']:8.4f}"
              f"{v['travel']:10.2f}{100 * v['upright_frac']:8.0f}%")


if __name__ == "__main__":
    main()
