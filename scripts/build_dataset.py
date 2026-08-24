#!/usr/bin/env python3
"""Build the (clip x body) manifest model/simple/train.py trains on.

Writes datasets/<name>/manifest.jsonl plus the two split axes. Nothing is
copied: the manifest points at data/ and assets/robots/ through two symlinks
placed in the dataset directory, so the 2 GB of retargeted motion and per-frame
z produced by docs/new_body.md stays in exactly one place on disk.

    datasets/<name>/
        manifest.jsonl              one row per (clip, body)
        data    -> ../../data           symlink
        robots  -> ../../assets/robots  symlink
        splits/train_bodies.txt     the ONLY split axis (from each parameter.json's "split")
        splits/test_bodies.txt
        splits/tasks.txt            every task in the manifest, for reference

One split axis, not two
-----------------------
Held out means held-out BODIES. Every task is trained on, on every training
body. The generalization question this path is asking is "does a beta the
adapter never saw produce a usable z", and splitting tasks as well would answer
a different question with a third of the data and make the four-quadrant report
harder to read for no gain -- the 54 tasks are the same 54 on both sides of the
body split, so a task-axis holdout measures the frozen actor's own coverage
rather than anything the adapter did.

Two things this fixes about the old datasets/crossenbodiment-1-datasets
-----------------------------------------------------------------------
1. ONE body. Every one of its 1530 rows is `child`, so beta was a constant and
   the LatentAdapter had nothing to condition on. This builds the full cross
   product over the bodies that have been through docs/new_body.md.

2. The task balance was 1000:10. `move-ego--90-2` carried 1000 of the 1530 rows
   while the other 53 tasks had 10 each -- 65% of every batch was one task. The
   990 rows blamed elsewhere for "no retargeted_motion" are exactly that task's
   extra trials, since only 10 trials per task were ever retargeted. This
   builder uses the 10-trial core (54 x 10 = 540 clips), which is both balanced
   AND fully covered: every row has a live qpos_ref, so the D term in
   model/simple/train.py's objective is no longer identically zero.

The source of truth for which clips exist is data/origin_z (the adult's
reward-inferred z0, one (1, 256) per clip). A body is only accepted if its
retargeting_motion and infer_retargeting_z cover that exact set.

Usage (from project root):
    uv run scripts/build_dataset.py
    uv run scripts/build_dataset.py --name crossenbodiment-10bodies
    uv run scripts/build_dataset.py --bodies teen petite giant
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# `child` is excluded by default: it is the single body the legacy baseline
# (outputs/{baseline,eval}/report.json) was measured on, and keeping it out of
# the training cross product leaves those numbers meaning what they meant.
DEFAULT_EXCLUDE = ("adult", "child")


def discover_bodies(robots_dir: Path, exclude) -> list:
    out = []
    for d in sorted(robots_dir.iterdir()):
        if not d.is_dir() or d.name in exclude:
            continue
        if not (d / "parameter.json").exists():
            continue
        out.append(d.name)
    return out


def body_split(robots_dir: Path, body: str) -> str:
    p = json.loads((robots_dir / body / "parameter.json").read_text())
    split = p.get("split")
    if split not in ("train", "test"):
        raise SystemExit(
            f"assets/robots/{body}/parameter.json has split={split!r}; expected "
            f"'train' or 'test'. scripts/scale_robot.py wipes this field -- rerun "
            f"scripts/write_body_splits.py."
        )
    return split


def clip_index(origin_z: Path) -> list:
    """[(task, stem)] for every adult z0 on disk, sorted."""
    return sorted((p.parent.name, p.stem) for p in origin_z.rglob("*.npy"))


def check_body(data_dir: Path, body: str, clips) -> None:
    want = set(clips)
    for sub, ext in (("retargeting_motion", ".npz"), ("infer_retargeting_z", ".npy")):
        d = data_dir / body / sub
        if not d.exists():
            raise SystemExit(f"{d} missing -- run docs/new_body.md Steps 2-3 for {body}")
        have = {(p.parent.name, p.stem) for p in d.rglob("*" + ext)}
        if not want <= have:
            miss = sorted(want - have)[:5]
            raise SystemExit(
                f"{d} covers {len(have & want)}/{len(want)} clips; missing e.g. {miss}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="crossenbodiment-10bodies",
                    help="directory name under datasets/")
    ap.add_argument("--bodies", nargs="*", default=None,
                    help="explicit body list (default: every body under "
                         "assets/robots/ except adult and child)")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing dataset directory")
    args = ap.parse_args()

    robots = ROOT / "assets/robots"
    data = ROOT / "data"
    out = ROOT / "datasets" / args.name

    bodies = args.bodies or discover_bodies(robots, set(args.exclude))
    if not bodies:
        raise SystemExit("no bodies selected")

    clips = clip_index(data / "origin_z")
    if not clips:
        raise SystemExit(f"no z0 under {data / 'origin_z'}")

    splits = {b: body_split(robots, b) for b in bodies}
    for b in bodies:
        check_body(data, b, clips)

    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists -- pass --force to rebuild it")
        shutil.rmtree(out)
    (out / "splits").mkdir(parents=True)

    # Symlinks, not copies: the manifest's paths resolve through these.
    (out / "data").symlink_to(Path("../..") / "data")
    (out / "robots").symlink_to(Path("../../assets") / "robots")

    (out / "splits" / "tasks.txt").write_text(
        "".join(f"{t}\n" for t in sorted({t for t, _ in clips})))
    for s in ("train", "test"):
        (out / "splits" / f"{s}_bodies.txt").write_text(
            "".join(f"{b}\n" for b in bodies if splits[b] == s))

    rows = []
    for body in bodies:
        for task, stem in clips:
            rows.append({
                "id": len(rows),
                "reward_name": task,
                "trial": int(stem.rsplit("_", 1)[1]),
                "morphology_label": body,
                "body_split": splits[body],
                "origin_z": f"data/origin_z/{task}/{stem}.npy",
                "morphology": f"robots/{body}/parameter.json",
                "target_xml": f"robots/{body}/robot.xml",
                "retargeted_motion": f"data/{body}/retargeting_motion/{task}/{stem}.npz",
                "retarget_z": f"data/{body}/infer_retargeting_z/{task}/{stem}.npy",
            })
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    tr = [b for b in bodies if splits[b] == "train"]
    te = [b for b in bodies if splits[b] == "test"]
    print(f"wrote {out}")
    print(f"  {len(rows)} rows = {len(clips)} clips x {len(bodies)} bodies")
    print(f"  train bodies ({len(tr)}): {' '.join(tr)}")
    print(f"  test  bodies ({len(te)}): {' '.join(te)}")
    print(f"  tasks: {len({t for t, _ in clips})} (all trained on -- no task split), "
          f"trials/task {len(clips) // max(1, len({t for t, _ in clips}))}")
    print(f"  every row has retargeted_motion -> the D term is live")


if __name__ == "__main__":
    main()
