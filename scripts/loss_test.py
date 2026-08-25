#!/usr/bin/env python3
"""loss_test.py — does the design ladder actually lower the cost?

The claim being tested
----------------------
The adult's own z, rolled out on the child, should get better as the two
mismatches are removed one at a time:

    A  raw actuators, raw obs      the adult's z on a body it was never trained
                                   for, driven by the adult's own actuator
                                   strength -- nothing corrected
    B  + actuator adjustment       assets/robot_torque/child/robot_torque_full.xml,
                                   the measured isometric law from
                                   torque_aggregate_motion_k.py --joint-dynamics
    C  + obs canonicalisation      the actor's view rescaled to adult proportions
                                   (model/obs_scale.py, mode "auto")

cost(A) > cost(B) > cost(C) is the design's prediction. If it does not hold the
correction is not doing what it is supposed to, and no amount of adapter
training on top of it is going to fix that -- which is why this is worth
measuring before, not after.

D (raw actuators + obs canonicalisation) is also run, because A/B/C alone cannot
say WHICH correction earns the improvement. With the fourth cell the 2x2 is
complete and the two effects can be read separately, plus their interaction.

Why the comparison is paired, and why that matters
--------------------------------------------------
humenv's Default init resets every env slot to a bit-identical state, and the
actions here are the frozen actor's mean with no sampling anywhere. So for one
clip, the ONLY thing that differs between two conditions is the thing under
test. That makes every clip its own control, and the honest statistic is the
per-clip win rate, not the difference of two means: clip-to-clip cost spread is
large (sd/mean ~0.7 even after the loss fixes) and a mean difference smaller
than that spread says nothing on its own. Both are reported; read the win rate.

What is held fixed
------------------
  * z is `data/origin_z/<task>/<task>_<trial>.npy` -- the ADULT's
    reward-inferred latent, untouched. No adapter, no z map. That is the point:
    this measures the body-side corrections in isolation.
  * the reference is `data/child/retargeting_motion/...`, the same one training
    scores against.
  * all four conditions use geometrically IDENTICAL bodies -- verified here, not
    assumed (--skip-geom-check to bypass). robot_torque_full.xml changes
    actuator gain/bias/forcerange, armature, damping and stiffness; it does not
    move a single body or geom. If it did, the reference would mean something
    different in different cells and the ladder would be meaningless.
  * the cost is model/losses.py through model/simple/train.py:compute_batch_cost,
    i.e. the same lambda_align * L_align + lambda_phys * L_phys that train_es.py
    minimises and evaluate.py reports.

Usage (from project root):
    uv run scripts/loss_test.py
    uv run scripts/loss_test.py --clips 80 --tasks upright
    uv run scripts/loss_test.py --torque-variant robot_torque --n-envs 16

Writes <out>/per_clip.csv, <out>/summary.csv and <out>/loss_test.png.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# (label, actuator-adjusted?, obs-canonicalised?). Order is the ladder A -> B -> C,
# with D last because it is the diagnostic cell, not part of the claim.
CONDITIONS = [
    ("A raw / raw obs", False, False),
    ("B torque / raw obs", True, False),
    ("C torque / scaled obs", True, True),
    ("D raw / scaled obs", False, True),
]


def check_same_geometry(xml_a: Path, xml_b: Path) -> None:
    a = mujoco.MjModel.from_xml_path(str(xml_a))
    b = mujoco.MjModel.from_xml_path(str(xml_b))
    for field in ("body_pos", "body_quat", "body_mass", "geom_size", "geom_pos"):
        va, vb = getattr(a, field), getattr(b, field)
        if va.shape != vb.shape or not np.allclose(va, vb, atol=1e-12):
            raise SystemExit(
                f"{xml_b.name} differs from {xml_a.name} in {field}: the two bodies are "
                f"not geometrically identical, so scoring them against the same "
                f"retargeted reference compares two different things. Regenerate the "
                f"torque variant, or pass --skip-geom-check if you know why.")
    gain = b.actuator_gainprm[:, 0] / a.actuator_gainprm[:, 0]
    print(f"geometry identical; actuator gain x{np.median(gain):.4f} "
          f"({gain.min():.3f}..{gain.max():.3f}), armature x"
          f"{np.median(b.dof_armature / np.maximum(a.dof_armature, 1e-12)):.3f}")


@torch.no_grad()
def rollout(model, env, z_env, steps, device, obs_mul):
    """Deterministic rollout, one z per slot. Returns qpos (n_envs, T, nq)."""
    obs, _ = env.reset()
    hist = []
    for _ in range(steps):
        proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.as_tensor(proprio, dtype=torch.float32, device=device)
        mu = model._actor(model._normalize(obs_t), z_env, model.cfg.actor_std).mean
        obs, _, _, _, info = env.step(mu.cpu().numpy())
        hist.append(info["qpos"].copy())
    return np.stack(hist, axis=1)


def free_diagnostics(qpos: np.ndarray) -> dict:
    """Physical sanity read straight off qpos, touching nothing in losses.py.

    The point is to have a verdict on "did this rollout stay on its feet" that
    cannot inherit any defect of the term being audited. If L_phys and these
    disagree about which condition is more stable, L_phys is the one that is
    wrong.
    """
    h = qpos[:, 2]
    qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    up = 2.0 * (qy * qz + qw * qx)          # +1 upright, -1 inverted
    return {
        "final_h_ratio": float(h[-1] / max(h[0], 1e-9)),
        "min_h_ratio": float(h.min() / max(h[0], 1e-9)),
        "upright_frac": float((up > 0.8).mean()),
        "mean_up": float(up.mean()),
        "collapsed": float(h[-1] < 0.5 * h[0]),
    }


def collect_clips(dataset_dir: Path, body: str, task_filter, n_clips, seed):
    """[(task, trial, z0, qpos_ref)] -- only clips that have BOTH a z and a
    retargeted reference on this body."""
    z_root = dataset_dir / "data" / "origin_z"
    ref_root = dataset_dir / "data" / body / "retargeting_motion"
    if not ref_root.is_dir():
        raise SystemExit(f"{ref_root} not found")

    pairs = []
    for task_dir in sorted(ref_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter is not None and task_dir.name not in task_filter:
            continue
        for ref in sorted(task_dir.glob("*.npz")):
            z_path = z_root / task_dir.name / f"{ref.stem}.npy"
            if z_path.exists():
                pairs.append((task_dir.name, ref.stem, z_path, ref))
    if not pairs:
        raise SystemExit("no clips with both an origin_z and a retargeted reference")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n_clips] if n_clips else pairs
    pairs.sort()

    out = []
    for task, stem, z_path, ref in pairs:
        out.append((task, stem,
                    np.load(z_path).reshape(-1).astype(np.float32),
                    np.load(ref)["qpos"]))
    return out



# ---------------------------------------------------------------------------
# Plotting
#
# Palette is the dataviz reference instance, used unchanged: categorical slots
# 1-2 (blue/orange) for the L_align/L_phys stack, and the documented diverging
# pair blue<->red for "improved / worsened". No hue was re-stepped, so the
# published validation of that ordering applies as-is (node was not available
# here to re-run scripts/validate_palette.js).
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"      # categorical slot 1 -- L_align
SERIES_2 = "#eb6834"      # categorical slot 2 -- L_phys
BETTER = "#2a78d6"        # diverging cool pole
WORSE = "#e34948"         # diverging warm pole

# Longest-prefix-first, so "move-ego-low" wins over "move-ego". Grouping the 54
# tasks into families is what makes the per-task panel readable at all; the
# ungrouped numbers stay in per_clip.csv.
FAMILIES = ["move-ego-0-2-raisearms", "move-ego-low", "move-ego", "raisearms",
            "crawl", "rotate-x", "rotate-y", "rotate-z", "lieonground",
            "sitonground", "crouch", "headstand", "jump", "split"]


def task_family(task: str) -> str:
    for f in FAMILIES:
        if task.startswith(f):
            return f
    return task


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)


def plot_results(clips, rows, labels, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(lab, j=0):
        return np.array([rows[(lab, i)][j] for i in range(len(clips))])

    short = [l.split()[0] for l in labels]
    fig = plt.figure(figsize=(13, 12), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.42, wspace=0.22)

    # -- panel 1: where the cost sits, per condition (part-to-whole) ----------
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    a_vals = [col(l, 1).mean() for l in labels]
    p_vals = [col(l, 2).mean() for l in labels]
    x = np.arange(len(labels))
    # 2px surface gap between stacked segments
    ax.bar(x, a_vals, 0.62, color=SERIES_1, edgecolor=SURFACE, linewidth=2,
           label="L_align", zorder=3)
    ax.bar(x, p_vals, 0.62, bottom=a_vals, color=SERIES_2, edgecolor=SURFACE,
           linewidth=2, label="L_phys", zorder=3)
    for xi, (av, pv) in enumerate(zip(a_vals, p_vals)):
        ax.text(xi, av + pv, f"{av + pv:.2f}", ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=8.5)
    ax.set_ylim(0, max(av + pv for av, pv in zip(a_vals, p_vals)) * 1.16)
    ax.set_ylabel("mean cost", color=INK_2, fontsize=10)
    ax.set_title("cost decomposition", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=26)
    # lifted clear of the plot area -- the direct labels sit on top of the bars
    leg = ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
                    bbox_to_anchor=(0, 1.0, 1, 0.1), borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # -- panel 2: the pairing, clip by clip ----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    a, c = col(labels[0]), col(labels[2])
    imp = c < a
    lo = max(min(a.min(), c.min()) * 0.8, 1e-3)
    hi = max(a.max(), c.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=2, zorder=2)
    ax.scatter(a[imp], c[imp], s=22, color=BETTER, alpha=0.75,
               edgecolor=SURFACE, linewidth=0.6, zorder=3,
               label=f"C better  ({imp.sum()}/{len(a)})")
    ax.scatter(a[~imp], c[~imp], s=22, color=WORSE, alpha=0.75,
               edgecolor=SURFACE, linewidth=0.6, zorder=3,
               label=f"C worse  ({(~imp).sum()}/{len(a)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"cost, {short[0]} (nothing corrected)", color=INK_2, fontsize=10)
    ax.set_ylabel(f"cost, {short[2]} (both)", color=INK_2, fontsize=10)
    ax.set_title("same clip, same init, one thing changed", color=INK,
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.text(0.97, 0.06, "below the line = corrected version wins",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK_MUTED)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_2)

    # -- panel 3: how the three ladder steps distribute over clips -----------
    ax = fig.add_subplot(gs[1, :])
    _style(ax)
    ax.axhline(0, color=INK_MUTED, linewidth=2, zorder=2)
    steps = [(labels[1], f"{short[0]}->{short[1]}  actuator only", "-"),
             (labels[3], f"{short[0]}->{short[3]}  obs only", "--"),
             (labels[2], f"{short[0]}->{short[2]}  both", "-")]
    for lab, name, ls in steps:
        d = np.sort(col(lab) - a)
        xs = np.linspace(0, 100, len(d))
        colour = BETTER if lab == labels[2] else INK_MUTED
        lw = 2.4 if lab == labels[2] else 1.6
        ax.plot(xs, d, color=colour, linewidth=lw, linestyle=ls, zorder=3, label=name)
    ax.set_xlabel("clips, sorted by change (%)", color=INK_2, fontsize=10)
    ax.set_ylabel("cost change vs A", color=INK_2, fontsize=10)
    ax.set_title("per-clip change: how much of the clip set each correction helps",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_yscale("symlog", linthresh=0.1)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_2)

    # -- panel 4: which task families the correction reaches -----------------
    ax = fig.add_subplot(gs[2, :])
    _style(ax)
    fam = {}
    for i, (task, *_rest) in enumerate(clips):
        fam.setdefault(task_family(task), []).append(c[i] - a[i])
    order = sorted(fam, key=lambda k: np.mean(fam[k]))
    vals = [float(np.mean(fam[k])) for k in order]
    cols = [BETTER if v < 0 else WORSE for v in vals]
    ax.bar(np.arange(len(order)), vals, 0.62, color=cols, edgecolor=SURFACE,
           linewidth=2, zorder=3)
    ax.axhline(0, color=INK_MUTED, linewidth=2, zorder=4)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{k}\n(n={len(fam[k])})" for k in order],
                       fontsize=8.5, rotation=30, ha="right")
    ax.set_ylabel("mean cost change, A -> C", color=INK_2, fontsize=10)
    ax.set_title("by task family: blue = both corrections help, red = they hurt",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)

    fig.suptitle(f"adult z on {ARGS_BODY[0]}: does correcting the body lower the cost?  "
                 f"({len(clips)} clips, paired)",
                 color=INK, fontsize=14, fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")


ARGS_BODY = ["child"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="datasets/crossenbodiment-10bodies")
    p.add_argument("--body", default="child")
    p.add_argument("--raw-xml", default=None,
                   help="default assets/robots/<body>/robot.xml -- adult actuators")
    p.add_argument("--torque-xml", default=None,
                   help="default assets/robot_torque/<body>/<variant>.xml")
    p.add_argument("--torque-variant", default="robot_torque_full",
                   choices=["robot_torque", "robot_torque_move_only", "robot_torque_full"],
                   help="robot_torque_full is the --joint-dynamics one: it rewrites "
                        "armature/damping/stiffness too, not just the actuators")
    p.add_argument("--obs-scale-ref", default="assets/robots/adult/robot.xml")
    p.add_argument("--tasks", default=None,
                   help="a group written by scripts/split_tasks_by_fall.py "
                        "(ground|upright) or a path to a task list")
    p.add_argument("--clips", type=int, default=60, help="0 = every clip")
    p.add_argument("--n-envs", type=int, default=20, help="env slots per batch")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--skip-geom-check", action="store_true")
    p.add_argument("--out", default="outputs/loss_test")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--audit", action="store_true",
                   help="also record every L_phys sub-term and a set of loss-independent\n                         stability diagnostics -> <out>/audit.csv")
    args = p.parse_args()

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel

    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    from model.simple.train import compute_batch_cost

    dataset_dir = REPO_ROOT / args.dataset
    raw_xml = (Path(args.raw_xml) if args.raw_xml
               else REPO_ROOT / "assets" / "robots" / args.body / "robot.xml")
    torque_xml = (Path(args.torque_xml) if args.torque_xml
                  else REPO_ROOT / "assets" / "robot_torque" / args.body /
                  f"{args.torque_variant}.xml")
    for x in (raw_xml, torque_xml):
        if not x.exists():
            raise SystemExit(f"{x} not found")
    if not args.skip_geom_check:
        check_same_geometry(raw_xml, torque_xml)

    task_filter = None
    if args.tasks:
        path = Path(args.tasks)
        if not path.exists():
            path = dataset_dir / "splits" / f"{args.tasks}_tasks.txt"
        if not path.exists():
            raise SystemExit(f"task list {args.tasks} not found")
        task_filter = {l.strip() for l in path.read_text().splitlines() if l.strip()}

    clips = collect_clips(dataset_dir, args.body, task_filter, args.clips, args.seed)
    print(f"{len(clips)} clips on {args.body}"
          + (f" (tasks: {args.tasks})" if args.tasks else ""))

    cfg = ESConfig(device=args.device)
    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device)
    model.eval()

    # One env per XML (slot count fixed), one obs multiplier per (xml, mode).
    n_envs = min(args.n_envs, len(clips))
    envs, fks = {}, {}
    for key, xml in (("raw", raw_xml), ("torque", torque_xml)):
        envs[key], _ = make_humenv(num_envs=n_envs, vectorization_mode="sync",
                                   task=None, xml=str(xml), state_init="Default")
        fks[key] = mujoco.MjModel.from_xml_path(str(xml))
    obs_muls = {
        (key, scaled): (build_obs_multiplier(
            xml, REPO_ROOT / args.obs_scale_ref, mode="auto",
            parts=cfg.obs_scale_parts, verbose=False) if scaled else None)
        for key, xml in (("raw", raw_xml), ("torque", torque_xml))
        for scaled in (False, True)
    }

    from model import losses

    # rows[(condition, clip_index)] = (cost, L_align, L_phys)
    rows = {}
    audit = {} if args.audit else None
    for label, use_torque, scale_obs in CONDITIONS:
        key = "torque" if use_torque else "raw"
        env, fk, obs_mul = envs[key], fks[key], obs_muls[(key, scale_obs)]
        costs = []
        for start in range(0, len(clips), n_envs):
            batch = clips[start:start + n_envs]
            z = torch.as_tensor(np.stack([c[2] for c in batch]),
                                dtype=torch.float32, device=args.device)
            if len(batch) < n_envs:                       # pad the last batch
                pad = z[-1:].expand(n_envs - len(batch), -1)
                z = torch.cat([z, pad])
            qpos = rollout(model, env, z, args.steps, args.device, obs_mul)
            refs = [c[3] for c in batch] + [batch[-1][3]] * (n_envs - len(batch))
            c, a, l = compute_batch_cost(fk, cfg, qpos, refs)
            costs.extend(zip(c[:len(batch)], a[:len(batch)], l[:len(batch)]))
            if audit is not None:
                for i in range(len(batch)):
                    _, pt = losses.physics_penalty(fk, qpos[i])
                    audit[(label, start + i)] = {**pt, **free_diagnostics(qpos[i])}
        for i, v in enumerate(costs):
            rows[(label, i)] = v
        arr = np.array([v[0] for v in costs])
        print(f"  {label:24s} cost mean {arr.mean():8.4f}  median {np.median(arr):8.4f}")

    for e in envs.values():
        e.close()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [c[0] for c in CONDITIONS]

    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "trial"] + [f"{m}_{lab.split()[0]}"
                                        for lab in labels
                                        for m in ("cost", "L_align", "L_phys")])
        for i, (task, stem, _, _) in enumerate(clips):
            row = [task, stem]
            for lab in labels:
                row += [f"{v:.6f}" for v in rows[(lab, i)]]
            w.writerow(row)

    def col(lab, j=0):
        return np.array([rows[(lab, i)][j] for i in range(len(clips))])

    print(f"\n=== {len(clips)} clips, paired (identical init, deterministic actions) ===")
    print(f"{'condition':24s} {'cost':>9s} {'median':>9s} {'L_align':>9s} {'L_phys':>9s}")
    for lab in labels:
        c, a, l = col(lab, 0), col(lab, 1), col(lab, 2)
        print(f"{lab:24s} {c.mean():9.4f} {np.median(c):9.4f} {a.mean():9.4f} {l.mean():9.4f}")

    print(f"\n=== the ladder: each step against the one before it ===")
    base = col(labels[0])
    for prev, cur in zip(labels, labels[1:3]):
        a, b = col(prev), col(cur)
        wins = int((b < a).sum())
        print(f"  {prev.split()[0]} -> {cur.split()[0]}: mean {a.mean():.4f} -> {b.mean():.4f} "
              f"({100 * (b.mean() - a.mean()) / a.mean():+.1f}%), "
              f"better on {wins}/{len(a)} clips ({100 * wins / len(a):.0f}%)")
    c = col(labels[2])
    wins = int((c < base).sum())
    print(f"  A -> C overall: mean {base.mean():.4f} -> {c.mean():.4f} "
          f"({100 * (c.mean() - base.mean()) / base.mean():+.1f}%), "
          f"better on {wins}/{len(base)} clips ({100 * wins / len(base):.0f}%)")
    ladder_holds = col(labels[0]).mean() > col(labels[1]).mean() > col(labels[2]).mean()
    print(f"  cost(A) > cost(B) > cost(C): {'HOLDS' if ladder_holds else 'DOES NOT HOLD'}")

    print(f"\n=== 2x2: which correction earns it ===")
    a_, b_, c_, d_ = (col(l) for l in labels)
    print(f"  actuator alone (A->B)          {100 * (b_.mean() - a_.mean()) / a_.mean():+7.1f}%"
          f"   won {int((b_ < a_).sum())}/{len(a_)}")
    print(f"  obs alone      (A->D)          {100 * (d_.mean() - a_.mean()) / a_.mean():+7.1f}%"
          f"   won {int((d_ < a_).sum())}/{len(a_)}")
    print(f"  both           (A->C)          {100 * (c_.mean() - a_.mean()) / a_.mean():+7.1f}%"
          f"   won {int((c_ < a_).sum())}/{len(a_)}")
    add = (b_.mean() - a_.mean()) + (d_.mean() - a_.mean())
    print(f"  interaction: both {c_.mean() - a_.mean():+.4f} vs additive {add:+.4f} "
          f"-> {'super' if c_.mean() - a_.mean() < add else 'sub'}-additive")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "actuator_adjusted", "obs_scaled",
                    "cost_mean", "cost_median", "cost_sd", "L_align_mean", "L_phys_mean",
                    "wins_vs_A", "n_clips"])
        for (lab, use_t, sc) in CONDITIONS:
            c_i = col(lab)
            w.writerow([lab, int(use_t), int(sc), f"{c_i.mean():.6f}",
                        f"{np.median(c_i):.6f}", f"{c_i.std():.6f}",
                        f"{col(lab, 1).mean():.6f}", f"{col(lab, 2).mean():.6f}",
                        int((c_i < base).sum()), len(clips)])
    print(f"\n-> {out_dir / 'per_clip.csv'}\n-> {out_dir / 'summary.csv'}")

    if audit:
        keys = sorted(next(iter(audit.values())))
        with open(out_dir / "audit.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "trial", "condition"] + keys)
            for (lab, i), v in sorted(audit.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                w.writerow([clips[i][0], clips[i][1], lab.split()[0]]
                           + [f"{v[k]:.6f}" for k in keys])
        print(f"-> {out_dir / 'audit.csv'}")

    if not args.no_plot:
        ARGS_BODY[0] = args.body
        plot_results(clips, rows, labels, out_dir / "loss_test.png")


if __name__ == "__main__":
    main()
