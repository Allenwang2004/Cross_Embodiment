#!/usr/bin/env python3
"""write_balanced_clips.py -- a task-mean-filtered clip list plus the
task -> motion-category map that makes category-balanced sampling possible.

Why this exists
---------------
scripts/write_clip_list.py filters PER CLIP, which leaves tasks with 1 clip
next to tasks with 10. Sampling clips uniformly from that list then samples
tasks in proportion to how many of their clips survived, and sampling tasks
uniformly makes the 1-clip tasks repeat ten times as often as they should.
Filtering per TASK keeps every surviving task at its full 10 trials, so both
levels are clean.

It does not, on its own, fix the real imbalance: of the 54 tasks, most are
move-ego variants, so 20 of the 30 tasks that survive a 0.3 task-mean cut are
move-ego and a task-uniform sampler still spends 67% of its budget walking.
The category file is what train_es.py --clip-balance category uses to spend it
evenly over motion FAMILIES instead.

Categories come from the task-name prefix, which is how the task names are
built (move-ego-*, rotate-*, crawl-*, ...). The mapping is written out rather
than recomputed downstream so that a run's grouping is a file you can read,
not a rule buried in two places.

Usage:
    uv run scripts/write_balanced_clips.py --csv outputs/initial_cost/child/z0_cost.csv \
        --body child --min-task-cost 0.3 --out-dir datasets/crossenbodiment-child-torque/splits \
        --tag child_balanced
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Prefix -> category. Order matters: the first prefix that matches wins, and
# move-ego must be tested before the bare motion names it contains
# (move-ego-0-2-raisearms-h-l is a walking clip, not a raisearms one).
PREFIXES = [
    ("move-ego", "move"),
    ("rotate", "rotate"),
    ("crawl", "crawl"),
    ("raisearms", "raisearms"),
    ("headstand", "headstand"),
    ("jump", "jump"),
    ("split", "split"),
    ("lieonground", "lieonground"),
    ("sitonground", "sitonground"),
    ("crouch", "crouch"),
]


def category(task: str) -> str:
    for pre, cat in PREFIXES:
        if task.startswith(pre):
            return cat
    return "other"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default="outputs/initial_cost/child/z0_cost.csv",
                   help="z0_cost.csv from scripts/rank_initial_cost.py")
    p.add_argument("--body", default="child", help="which column to threshold ('mean' also works)")
    p.add_argument("--min-task-cost", type=float, default=0.3,
                   help="drop a task when the MEAN z0 cost over its clips is at or below this; "
                        "a task is kept whole or not at all")
    p.add_argument("--extra-tasks", nargs="*", default=[],
                   help="tasks to keep regardless of --min-task-cost. rotate-y sits just under a "
                        "0.3 task mean (0.281/0.284) but is a motion family the rebalanced set "
                        "needs, so it is forced in by name rather than by loosening the cut for "
                        "everything")
    p.add_argument("--all-trials", action="store_true",
                   help="take every trial that exists under data/origin_z/<task>/, not only the "
                        "ones the cost CSV was measured on. Top-up runs add trials _10.. after the "
                        "CSV was written, and those are the clips the rebalance is made of")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="datasets/crossenbodiment-child-torque/splits")
    p.add_argument("--tag", default="child_balanced")
    args = p.parse_args()

    src = Path(args.csv)
    rows = list(csv.DictReader(open(src if src.is_absolute() else REPO_ROOT / src)))
    if args.body not in rows[0]:
        raise SystemExit(f"column '{args.body}' not in {args.csv}: {list(rows[0])}")

    by_task = collections.defaultdict(list)
    for r in rows:
        task, stem = r["task"], r["clip"]
        trial = stem[len(task) + 1:] if stem.startswith(task + "_") else stem.rsplit("_", 1)[1]
        by_task[task].append((trial, float(r[args.body])))

    kept = {t: v for t, v in by_task.items()
            if sum(c for _, c in v) / len(v) > args.min_task_cost}
    for t in args.extra_tasks:
        if t not in by_task:
            raise SystemExit(f"--extra-tasks: '{t}' is not in {args.csv}")
        kept.setdefault(t, by_task[t])
    if args.all_trials:
        zroot = Path(args.data_dir)
        zroot = (zroot if zroot.is_absolute() else REPO_ROOT / zroot) / "origin_z"
        for t in list(kept):
            on_disk = sorted((f.stem[len(t) + 1:] for f in (zroot / t).glob(f"{t}_*.npy")),
                             key=int)
            if not on_disk:
                raise SystemExit(f"{zroot / t} has no z0 for '{t}'")
            known = dict(kept[t])
            kept[t] = [(tr, known.get(tr, float("nan"))) for tr in on_disk]
    if not kept:
        raise SystemExit(f"no task has mean {args.body} z0 above {args.min_task_cost}")

    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_path = out_dir / f"{args.tag}_clips.txt"
    cats_path = out_dir / f"{args.tag}_categories.txt"

    lines, cat_lines = [], []
    for t in sorted(kept):
        for trial, _ in sorted(kept[t], key=lambda x: int(x[0])):
            lines.append(f"{t} {trial}")
        cat_lines.append(f"{t} {category(t)}")
    clips_path.write_text("".join(l + "\n" for l in lines))
    cats_path.write_text("".join(l + "\n" for l in cat_lines))

    per_cat = collections.defaultdict(lambda: [0, 0])
    for t in kept:
        per_cat[category(t)][0] += 1
        per_cat[category(t)][1] += len(kept[t])
    dropped = sorted(set(by_task) - set(kept))
    print(f"{len(kept)} of {len(by_task)} tasks kept (mean {args.body} z0 > {args.min_task_cost}), "
          f"{len(lines)} clips")
    print(f"  dropped ({len(dropped)}): {' '.join(dropped)}")
    print(f"  {len(per_cat)} categories -- a category-balanced sampler gives each "
          f"{100 / len(per_cat):.0f}% of the budget:")
    for c, (nt, nc) in sorted(per_cat.items(), key=lambda kv: -kv[1][0]):
        share = 100 * nc / len(lines)
        print(f"    {c:12s} {nt:2d} tasks {nc:3d} clips   uniform-clip share {share:5.1f}% -> "
              f"balanced {100 / len(per_cat):.1f}%")
    print(f"-> {clips_path}\n-> {cats_path}")


if __name__ == "__main__":
    main()
