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
        splits/train_bodies.txt     body axis (from each parameter.json's "split")
        splits/test_bodies.txt
        splits/train_tasks.txt      task axis (49 / 5 by default, seeded)
        splits/test_tasks.txt
        splits/tasks.txt            every task in the manifest, for reference

Two split axes
--------------
BODY: splits/{train,test}_bodies.txt, from each parameter.json's "split".
TASK: splits/{train,test}_tasks.txt, 49 / 5 by default.

The task axis exists for model/simple/train_zmap.py, whose claim is "hand me an
adult motion I have never seen, and I will give you the latent that performs it
on this body". Testing that needs motions the map never fit. The split is at the
TASK level, not the clip level, so the ten correlated trials of one task cannot
straddle it -- same rationale as scripts/split_tasks.py.

The rollout-based paths (train.py, train_es.py) do not read the task split: for
them the generalization question is about beta, and holding out tasks as well
would cost a tenth of the data to answer a question about the frozen actor's
own coverage rather than about the adapter.

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
   AND fully covered: every row has a live qpos_ref, so the L_align term in
   model/simple/train.py's objective is no longer identically zero.

The source of truth for which clips exist is data/origin_z (the adult's
reward-inferred z0, one (1, 256) per clip). A body is only accepted if its
retargeting_motion and infer_retargeting_z cover that exact set.

Two DIFFERENT adult latents are referenced and they are not interchangeable:

    origin_z         (1, 256)  reward-inferred, ONE per clip -- "what task is
                               this", the conditioning input the adapter has
                               always taken
    infer_origin_z   (T, 256)  tracking-inferred PER FRAME from the adult
                               performing the clip on the ADULT skeleton

The per-frame pair (infer_origin_z, <body>/infer_retargeting_z) is a supervised
correspondence: the same motion, same frame, seen as a latent by the adult and
by the target body. That is what model/simple/train_zmap.py regresses.

Usage (from project root):
    uv run scripts/build_dataset.py
    uv run scripts/build_dataset.py --name crossenbodiment-10bodies
    uv run scripts/build_dataset.py --bodies teen petite giant
"""

import argparse
import json
import shutil

import numpy as np
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


def check_adult_frames(data_dir: Path, clips, required: bool = True) -> None:
    """data/infer_origin_z is body-independent, so it is checked once.

    NOT needed to train: model/dataset.py's `z0` is row["origin_z"], the single
    reward-inferred vector, and nothing loads row["infer_origin_z"] -- that is
    the PER-FRAME z on the adult skeleton, which scripts/fit_cross_body_z_map.py
    consumes. A manifest built without it is fully usable for train_es.py and
    only unusable for the z-map fit, so this is a warning under
    --no-adult-z rather than a hard stop.
    """
    d = data_dir / "infer_origin_z"
    have = {(p.parent.name, p.stem) for p in d.rglob("*.npy")}
    missing = set(clips) - have
    if not missing:
        return
    msg = (f"{d} covers {len(have & set(clips))}/{len(clips)} clips; missing e.g. "
           f"{sorted(missing)[:5]}. Regenerate with scripts/batch_infer_z.py "
           f"--input_dir data/origin_motion --xml assets/robots/adult/robot.xml")
    if required:
        raise SystemExit(msg)
    print(f"WARNING: {msg}\n  --no-adult-z given: the manifest still trains (z0 comes from "
          f"origin_z), but fit_cross_body_z_map.py cannot use it")


def check_body(data_dir: Path, body: str, clips, require_infer_z: bool = True) -> None:
    """model/dataset.py reads origin_z, retargeted_motion and morphology only --
    infer_retargeting_z is the PER-FRAME z on that body, consumed by the z-map
    fit alone. Bodies generated after that pipeline stopped being run (the
    x_leg_* family) have no such directory, and requiring it would exclude
    exactly the bodies that widen beta's coverage most."""
    want = set(clips)
    subs = [("retargeting_motion", ".npz")]
    if require_infer_z:
        subs.append(("infer_retargeting_z", ".npy"))
    for sub, ext in subs:
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
    ap.add_argument("--n-test-tasks", type=int, default=5,
                    help="tasks held out for model/simple/train_zmap.py (of 54)")
    ap.add_argument("--task-split-seed", type=int, default=0)
    ap.add_argument("--no-adult-z", action="store_true",
                    help="do not require data/infer_origin_z for every clip. Training never reads "
                         "it (z0 is origin_z); only the cross-body z-map fit does")
    ap.add_argument("--torque-xml", action="store_true",
                    help="point target_xml at assets/robots_torque/<body>/robot_torque_full.xml "
                         "(actuators, armature, damping and stiffness rescaled to the body's own "
                         "measured torque demand) instead of assets/robots/<body>/robot.xml, "
                         "which keeps adult's actuators")
    ap.add_argument("--no-infer-z", action="store_true",
                   help="do not require data/<body>/infer_retargeting_z. train_es never reads it "
                        "(see model/dataset.py); only fit_cross_body_z_map.py does")
    ap.add_argument("--intersect-clips", action="store_true",
                   help="keep only the clips every selected body has retargeted, instead of "
                        "stopping on the first body that is missing one. data/origin_z is the "
                        "union over bodies, so a body that never got the top-up trials is "
                        "behind it by construction")
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
    check_adult_frames(data, clips, required=not args.no_adult_z)
    if args.intersect_clips:
        # The manifest is a (clip x body) cross product, so a clip only belongs
        # in it if EVERY selected body has it retargeted. data/origin_z grows
        # whenever a body gets top-up rollouts (child has 720, the rest 540),
        # and without this the build stops on the first body that is behind
        # instead of building the rectangle that actually exists.
        keep = set(clips)
        for b in bodies:
            subs = [("retargeting_motion", ".npz")]
            if not args.no_infer_z:
                subs.append(("infer_retargeting_z", ".npy"))
            for sub, ext in subs:
                d = data / b / sub
                keep &= {(x.parent.name, x.stem) for x in d.rglob("*" + ext)}
        dropped = len(clips) - len(keep)
        if not keep:
            raise SystemExit("no clip is present for every selected body")
        print(f"--intersect-clips: {len(keep)} of {len(clips)} clips are present for all "
              f"{len(bodies)} bodies ({dropped} dropped)")
        clips = sorted(keep)
    for b in bodies:
        check_body(data, b, clips, require_infer_z=not args.no_infer_z)

    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists -- pass --force to rebuild it")
        shutil.rmtree(out)
    (out / "splits").mkdir(parents=True)

    # Symlinks, not copies: the manifest's paths resolve through these.
    (out / "data").symlink_to(Path("../..") / "data")
    (out / "robots").symlink_to(Path("../../assets") / "robots")
    # The torque-matched bodies live in a separate tree; --torque-xml points
    # target_xml at them. Without it the manifest hands training the geometry
    # with ADULT actuators, which on child is 4.7x the torque its own mass and
    # limb lengths call for -- a different body from the one every
    # single_z_search / plateau number was measured on.
    (out / "robots_torque").symlink_to(Path("../../assets") / "robots_torque")

    tasks = sorted({t for t, _ in clips})
    (out / "splits" / "tasks.txt").write_text("".join(f"{t}\n" for t in tasks))
    # Seeded, so the held-out motions are the same set every rebuild -- a split
    # that moves silently makes every "unseen task" number incomparable.
    rng = np.random.default_rng(args.task_split_seed)
    test_tasks = sorted(rng.choice(tasks, args.n_test_tasks, replace=False).tolist())
    train_tasks = [t for t in tasks if t not in set(test_tasks)]
    (out / "splits" / "train_tasks.txt").write_text("".join(f"{t}\n" for t in train_tasks))
    (out / "splits" / "test_tasks.txt").write_text("".join(f"{t}\n" for t in test_tasks))
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
                "task_split": "test" if task in set(test_tasks) else "train",
                "origin_z": f"data/origin_z/{task}/{stem}.npy",
                "infer_origin_z": f"data/infer_origin_z/{task}/{stem}.npy",
                "morphology": f"robots/{body}/parameter.json",
                "target_xml": (f"robots_torque/{body}/robot_torque_full.xml"
                               if args.torque_xml else f"robots/{body}/robot.xml"),
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
    print(f"  tasks: {len(train_tasks)} train / {len(test_tasks)} test, "
          f"trials/task {len(clips) // max(1, len(tasks))}")
    print(f"  held-out tasks: {' '.join(test_tasks)}")
    print(f"  every row has retargeted_motion -> the L_align term is live")


if __name__ == "__main__":
    main()
