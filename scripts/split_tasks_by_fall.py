#!/usr/bin/env python3
"""split_tasks_by_fall.py — partition the task set by whether P.fall fires on the
REFERENCE motion, so the fall term stops being a per-task constant.

What this is for
----------------
After the d_root fixes, P.fall is 66.6% of the cost (measured on athletic, 54
tasks) and it is the term that does not respond to z: model/simple/train_zmap.py
recorded sd/mean 0.05 for it. That is not because falling is unimportant -- it is
because "fall" is being scored against an absolute upright prior, while a large
part of the task set is SUPPOSED to be on the ground. headstand, crawl,
lieonground and sitonground all score high on a reference clip that is doing
exactly what it was asked to do.

So the number decomposes into a per-task offset plus the part that actually
depends on the rollout, and the offset is much larger than the signal. Antithetic
ES cancels the offset inside a pair (F+ and F- are the same clip), but it still
sets the scale of every reported cost, poisons a PPO baseline pooled over tasks,
and makes cross-task comparison in evaluate.py meaningless.

Splitting is the cheap half of the fix. Train within one group and the offset is
roughly common to every clip in the batch, so what moves is the part that
responds. The expensive half -- making fall reference-relative so both groups can
be trained together -- is a change to what the loss MEANS, and is worth doing
only once a within-group baseline exists to compare it against.

The cut is measured, not chosen
-------------------------------
P.fall is evaluated on `data/<body>/retargeting_motion` -- the reference itself,
never a rollout -- for every (task, body, trial), then averaged per task. The
sorted values are strongly bimodal, and the largest ratio gap in the whole
distribution is the one this splits on:

    move-ego-low-180-2   0.1469   <- last of the ground group
    jump-2               0.0225   <- first of the upright group      x6.5

--threshold 0.05 sits inside that gap with a factor of ~3 of margin on each
side, so the partition is not sensitive to it. The next largest gap is x2.6,
which is what "the cut is obvious" looks like numerically.

What the two groups are
-----------------------
    ground   24 tasks   fall 0.147 .. 3.358   headstand, crawl-*, lieonground-*,
                                              sitonground, crouch-0, split-*,
                                              rotate-x/z-*, move-ego-low-*
    upright  30 tasks   fall 0.0002 .. 0.0225 move-ego-* (non-low), raisearms-*,
                                              move-ego-*-raisearms-*, jump-2,
                                              rotate-y-*

Note the two halves are NOT the same experiment. `upright` is the group where
fall is genuinely near zero on a good rollout, so any fall the policy incurs is
real information -- that is the group to train on first. `ground` is the group
where fall is a large constant that the policy cannot remove, so training there
measures how much of the remaining variance L_align can still drive; expect a
much worse-conditioned objective, and read the per-term breakdown rather than
the total.

Also worth knowing before reading results: rotate-z--5-0.8 has sd 1.24 ACROSS
BODIES against a mean of 1.08, i.e. some bodies keep their feet on that task and
some do not. That one is a genuine per-body difference, not an offset, and it
will not average out.

Usage (from project root):
    uv run scripts/split_tasks_by_fall.py
    uv run scripts/split_tasks_by_fall.py --threshold 0.05
    uv run scripts/split_tasks_by_fall.py --dataset datasets/crossenbodiment-10bodies

Writes <dataset>/splits/ground_tasks.txt, <dataset>/splits/upright_tasks.txt and
<out>/per_task.csv. Consume them with train_es.py --tasks ground|upright.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def fall_parts(qpos: np.ndarray) -> tuple[float, float]:
    """losses._fall_penalty's two halves, kept separate.

    Duplicated rather than imported so this stays a measurement of the reference
    clips even if the fall term is later made reference-relative -- at which
    point this script is what says how big the offset it removed actually was.
    """
    pelvis_z = qpos[:, 2]
    height_ratio = np.clip(pelvis_z / qpos[0, 2], 0.0, 1.0)
    height = float(np.mean((1.0 - height_ratio) ** 2))

    qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    up_z = 2.0 * (qy * qz + qw * qx)      # local +Y is anatomical up, see losses.py
    tilt = float(np.mean((1.0 - up_z) ** 2))
    return height, tilt


def survey(dataset_dir: Path) -> dict:
    """{task: {"fall": mean, "height":, "tilt":, "sd": across bodies, "n":}}"""
    data_dir = dataset_dir / "data"
    bodies = sorted(d.name for d in data_dir.iterdir()
                    if (d / "retargeting_motion").is_dir())
    if not bodies:
        raise SystemExit(f"no <body>/retargeting_motion under {data_dir}")
    print(f"{len(bodies)} bodies: {' '.join(bodies)}")

    per = defaultdict(list)
    for body in bodies:
        base = data_dir / body / "retargeting_motion"
        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue
            for f in sorted(glob.glob(str(task_dir / "*.npz"))):
                h, t = fall_parts(np.load(f)["qpos"])
                per[task_dir.name].append((h + t, h, t))

    out = {}
    for task, vals in per.items():
        a = np.array(vals)
        out[task] = {"fall": float(a[:, 0].mean()), "height": float(a[:, 1].mean()),
                     "tilt": float(a[:, 2].mean()), "sd": float(a[:, 0].std()),
                     "n": len(vals)}
    return out


def report_gap(stats: dict, threshold: float) -> None:
    """Print where the chosen threshold sits relative to the real gaps, so a
    threshold that has drifted into a dense region announces itself."""
    s = np.sort([v["fall"] for v in stats.values()])[::-1]
    gaps = sorted(((s[i] / max(s[i + 1], 1e-12), s[i], s[i + 1], i + 1)
                   for i in range(len(s) - 1)), key=lambda g: -g[0])
    print("\nlargest ratio gaps in the sorted fall values:")
    for ratio, hi, lo, rank in gaps[:4]:
        mark = "  <- the cut" if lo < threshold <= hi else ""
        print(f"   after rank {rank:2d}: {hi:.4f} -> {lo:.4f}  (x{ratio:.1f}){mark}")
    above = [v["fall"] for v in stats.values() if v["fall"] >= threshold]
    below = [v["fall"] for v in stats.values() if v["fall"] < threshold]
    if above and below:
        margin = min(above) / max(below)
        print(f"   threshold {threshold:g}: nearest values {max(below):.4f} and "
              f"{min(above):.4f}, x{margin:.1f} apart")
        if margin < 2.0:
            print("   WARNING: under 2x of margin -- the partition is sensitive to "
                  "--threshold, re-read the table before trusting it")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="datasets/crossenbodiment-10bodies")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="mean P.fall on the reference above which a task is "
                        "'ground'. The measured distribution has a 6.5x gap "
                        "between 0.0225 and 0.1469, so anything in that range "
                        "gives the same partition")
    p.add_argument("--out", default="outputs/split_tasks_by_fall")
    p.add_argument("--dry-run", action="store_true",
                   help="print the partition but do not write the split files")
    args = p.parse_args()

    dataset_dir = REPO_ROOT / args.dataset
    stats = survey(dataset_dir)

    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["fall"])
    ground = [t for t, v in ranked if v["fall"] >= args.threshold]
    upright = [t for t, v in ranked if v["fall"] < args.threshold]

    print(f"\n{'task':32s} {'group':>8s} {'fall':>8s} {'height':>8s} {'tilt':>8s} "
          f"{'sd(bodies)':>11s}")
    for task, v in ranked:
        grp = "ground" if v["fall"] >= args.threshold else "upright"
        print(f"{task:32s} {grp:>8s} {v['fall']:8.4f} {v['height']:8.4f} "
              f"{v['tilt']:8.4f} {v['sd']:11.4f}")

    report_gap(stats, args.threshold)

    print(f"\nground  {len(ground):3d} tasks  fall "
          f"{min(stats[t]['fall'] for t in ground):.4f} .. "
          f"{max(stats[t]['fall'] for t in ground):.4f}")
    print(f"upright {len(upright):3d} tasks  fall "
          f"{min(stats[t]['fall'] for t in upright):.4f} .. "
          f"{max(stats[t]['fall'] for t in upright):.4f}")

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group", "fall", "height", "tilt", "sd_across_bodies", "n_clips"])
        for task, v in ranked:
            w.writerow([task, "ground" if v["fall"] >= args.threshold else "upright",
                        f"{v['fall']:.6f}", f"{v['height']:.6f}", f"{v['tilt']:.6f}",
                        f"{v['sd']:.6f}", v["n"]])
    print(f"\n-> {out_dir / 'per_task.csv'}")

    if args.dry_run:
        print("--dry-run: split files not written")
        return
    splits = dataset_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    for name, tasks in (("ground_tasks.txt", ground), ("upright_tasks.txt", upright)):
        (splits / name).write_text("".join(f"{t}\n" for t in sorted(tasks)))
        print(f"-> {splits / name}")

    test_tasks_path = splits / "test_tasks.txt"
    if test_tasks_path.exists():
        held = {l.strip() for l in test_tasks_path.read_text().splitlines() if l.strip()}
        # The held-out-task axis is orthogonal to this one and still applies
        # inside each group, so say how many of each group it removes -- a group
        # whose held-out share is tiny cannot report a meaningful task gap.
        for name, tasks in (("ground", ground), ("upright", upright)):
            n = len(held & set(tasks))
            print(f"   {name}: {n}/{len(tasks)} are in test_tasks.txt")


if __name__ == "__main__":
    main()
