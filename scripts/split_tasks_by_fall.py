#!/usr/bin/env python3
"""spilt_tasks.py — partition the task set by whether P.fall fires on the
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

A subset of upright: move
--------------------------
NOT a third member of the partition. The partition is still the two-way fall
cut, `upright` still holds all 30 of its tasks, and `move` is a selectable
SUBSET of it -- so every number recorded against `upright` before this section
existed is still a number about the same task set, and `--category upright`
still means what it meant. Carving move out instead would silently redefine
`upright` as its 13 stationary tasks and make those runs incomparable.

What the subset is for: the fall cut makes the fall OFFSET common within a
group. It does not make
L_align's composition common. Inside `upright` the task set is still two things:
clips whose root translates across the floor, where d_root's heading and
velocity terms carry the objective, and clips whose root stays put, where those
terms are ~0 by construction and d_pose/d_ee are the whole signal. Those are
also the two cases where beta enters differently -- leg length and girth change
a stride, arm scale changes a reach -- so an adapter trained on the pooled group
is being asked to explain both mechanisms from one gradient. `move` is the half
where the root travels, so training on it alone isolates the stride mechanism.

The membership test is the name prefix
`move-ego` AND a measured floor on net root displacement, because the name alone
gets one task wrong:

    move-ego-0-4                40.04 m   <- the group
    ...
    move-ego-0-2-raisearms-l-l   4.95 m   <- last real mover
    move-ego-0-0                 0.06 m   <- speed 0. It is named move and it
                                             does not move; it belongs with
                                             raisearms, and --move-min-disp
                                             (default 1.0) is what puts it there.

The x88 gap between 4.95 and 0.06 is what makes that floor insensitive: any
value between about 0.3 and 4 gives the same partition. Measured over every
(task, body, trial) reference clip, as |root_xy(T) - root_xy(0)|, the same way
the fall cut is measured -- a name-based group is exactly the kind of thing that
goes quietly wrong when the task set changes.

move-ego-low-* is NOT in `move`: the fall cut already put it in `ground`, and
taking it back out would rebuild inside `move` the offset this whole script
exists to remove. Locomotion on the ground stays with the ground group.

What the groups are
--------------------
The partition, by P.fall on the reference:

    ground   24 tasks   fall 0.147 .. 3.358   headstand, crawl-*, lieonground-*,
                                              sitonground, crouch-0, split-*,
                                              rotate-x/z-*, move-ego-low-*
    upright  30 tasks   fall 0.0002 .. 0.0225 move-ego-* (non-low), raisearms-*,
                                              move-ego-*-raisearms-*, jump-2,
                                              rotate-y-*

and one subset, by root displacement on the reference:

    move     17 tasks   subset of upright     move-ego-* (non-low) and
                        disp 4.95 .. 40.04    move-ego-*-raisearms-*, minus
                                              move-ego-0-0 (speed 0)

`upright \ move` is the other 13 -- raisearms-*, jump-2, rotate-y-*,
move-ego-0-0 -- which no split file names, because nothing has asked for it yet.

Note the groups are NOT the same experiment. `upright` is the group where
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
    uv run scripts/spilt_tasks.py
    uv run scripts/spilt_tasks.py --threshold 0.05
    uv run scripts/spilt_tasks.py --dataset datasets/crossenbodiment-10bodies

    # the pre-move two-way partition: no move group, upright keeps its 30 tasks
    uv run scripts/spilt_tasks.py --move-prefix ""

    # move by name only, i.e. keep the standing move-ego-0-0 in it
    uv run scripts/spilt_tasks.py --move-min-disp 0

Writes <dataset>/splits/{ground,upright,move}_tasks.txt and <out>/per_task.csv.
Consume them with train_es.py --category ground|upright|move.
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


def root_displacement(qpos: np.ndarray) -> float:
    """Net horizontal travel of the root over the clip, in metres.

    NET, not path length: rotate-y-*-0.8 spins in place and its path length is
    2.6 m of wander that returns to where it started, which would read as
    locomotion. |p(T) - p(0)| calls that 0.2 and keeps it out of `move`.
    """
    return float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))


def survey(dataset_dir: Path) -> dict:
    """{task: {"fall":, "height":, "tilt":, "disp":, "sd": across bodies, "n":}}"""
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
                qpos = np.load(f)["qpos"]
                h, t = fall_parts(qpos)
                per[task_dir.name].append((h + t, h, t, root_displacement(qpos)))

    out = {}
    for task, vals in per.items():
        a = np.array(vals)
        out[task] = {"fall": float(a[:, 0].mean()), "height": float(a[:, 1].mean()),
                     "tilt": float(a[:, 2].mean()), "disp": float(a[:, 3].mean()),
                     "sd": float(a[:, 0].std()), "n": len(vals)}
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
    p.add_argument("--move-prefix", default="move-ego",
                   help="tasks in the upright group whose name starts with this "
                        "AND that clear --move-min-disp become the `move` group. "
                        "Empty string disables the second cut entirely and "
                        "restores the two-way ground/upright partition")
    p.add_argument("--move-min-disp", type=float, default=1.0,
                   help="metres of NET root displacement on the reference below "
                        "which a --move-prefix task is not really locomotion and "
                        "stays in upright. The measured values jump from 0.06 "
                        "(move-ego-0-0, speed zero) to 4.95, so anything in "
                        "0.3..4 gives the same partition")
    p.add_argument("--out", default="outputs/spilt_tasks")
    p.add_argument("--dry-run", action="store_true",
                   help="print the partition but do not write the split files")
    args = p.parse_args()

    dataset_dir = REPO_ROOT / args.dataset
    stats = survey(dataset_dir)

    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["fall"])
    ground = [t for t, v in ranked if v["fall"] >= args.threshold]
    upright = [t for t, v in ranked if v["fall"] < args.threshold]

    # A SUBSET of upright, not a third group: upright keeps all its tasks so it
    # stays the same task set it always was. Only ever drawn from the low-fall
    # side, so move-ego-low-* stays in ground where its fall offset belongs.
    named = ([t for t in upright if t.startswith(args.move_prefix)]
             if args.move_prefix else [])
    move = [t for t in named if stats[t]["disp"] >= args.move_min_disp]
    demoted = [t for t in named if stats[t]["disp"] < args.move_min_disp]
    in_move = set(move)

    print(f"\n{'task':32s} {'group':>8s} {'move':>5s} {'fall':>8s} {'height':>8s} "
          f"{'tilt':>8s} {'disp':>8s} {'sd(bodies)':>11s}")
    for task, v in ranked:
        grp = "ground" if v["fall"] >= args.threshold else "upright"
        print(f"{task:32s} {grp:>8s} {'yes' if task in in_move else '.':>5s} "
              f"{v['fall']:8.4f} {v['height']:8.4f} "
              f"{v['tilt']:8.4f} {v['disp']:8.3f} {v['sd']:11.4f}")

    report_gap(stats, args.threshold)

    if named:
        print(f"\nmove subset: '{args.move_prefix}*' within upright, net root "
              f"displacement >= {args.move_min_disp:g} m")
        if move and demoted:
            print(f"   nearest values {max(stats[t]['disp'] for t in demoted):.3f} and "
                  f"{min(stats[t]['disp'] for t in move):.3f}, "
                  f"x{min(stats[t]['disp'] for t in move) / max(max(stats[t]['disp'] for t in demoted), 1e-12):.0f} apart")
        # Named move-ego but measurably standing still. Printed by name because
        # it is the one thing a reader would otherwise assume went the other way.
        for t in demoted:
            print(f"   excluded from move: {t} (disp {stats[t]['disp']:.3f} m -- "
                  f"it does not move); it stays in upright either way")

    for name, tasks, note in (("ground", ground, ""), ("upright", upright, ""),
                              ("move", move, "  (subset of upright)")):
        if not tasks:
            print(f"\n{name:7s} {0:3d} tasks")
            continue
        print(f"\n{name:7s} {len(tasks):3d} tasks  fall "
              f"{min(stats[t]['fall'] for t in tasks):.4f} .. "
              f"{max(stats[t]['fall'] for t in tasks):.4f}  disp "
              f"{min(stats[t]['disp'] for t in tasks):.2f} .. "
              f"{max(stats[t]['disp'] for t in tasks):.2f} m{note}")

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        # `group` is the PARTITION (ground|upright), unchanged from before the
        # move subset existed; `move` is a separate flag because a task can be
        # both upright and move.
        w.writerow(["task", "group", "move", "fall", "height", "tilt", "disp",
                    "sd_across_bodies", "n_clips"])
        for task, v in ranked:
            w.writerow([task, "ground" if v["fall"] >= args.threshold else "upright",
                        1 if task in in_move else 0,
                        f"{v['fall']:.6f}", f"{v['height']:.6f}", f"{v['tilt']:.6f}",
                        f"{v['disp']:.6f}", f"{v['sd']:.6f}", v["n"]])
    print(f"\n-> {out_dir / 'per_task.csv'}")

    if args.dry_run:
        print("--dry-run: split files not written")
        return
    splits = dataset_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    for name, tasks in (("ground_tasks.txt", ground), ("upright_tasks.txt", upright),
                        ("move_tasks.txt", move)):
        if not tasks and name == "move_tasks.txt":
            # --move-prefix "" asked for the two-way partition; leaving a stale
            # move_tasks.txt behind would let --category move silently train on
            # whatever the last run wrote.
            (splits / name).unlink(missing_ok=True)
            continue
        (splits / name).write_text("".join(f"{t}\n" for t in sorted(tasks)))
        print(f"-> {splits / name}")

    test_tasks_path = splits / "test_tasks.txt"
    if test_tasks_path.exists():
        held = {l.strip() for l in test_tasks_path.read_text().splitlines() if l.strip()}
        # The held-out-task axis is orthogonal to this one and still applies
        # inside each group, so say how many of each group it removes -- a group
        # whose held-out share is tiny cannot report a meaningful task gap.
        for name, tasks in (("ground", ground), ("upright", upright), ("move", move)):
            if tasks:
                n = len(held & set(tasks))
                print(f"   {name}: {n}/{len(tasks)} are in test_tasks.txt")


if __name__ == "__main__":
    main()
