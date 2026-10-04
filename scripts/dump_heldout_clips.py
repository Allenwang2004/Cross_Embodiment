#!/usr/bin/env python3
"""dump_heldout_clips.py -- reproduce a checkpoint's held-out clip split.

train_es.py picks the held-out clips with random.Random(cfg.eval_seed).sample()
over the clip list, and never writes the result anywhere except the eval log,
which prints only the first eight. Anything that wants to score or render
exactly those clips afterwards has to rebuild the split, and it has to rebuild
it the SAME way -- the sample depends on the order and the contents of the
list it is drawn from, so reconstructing it "by hand" from the clip-list file
would silently give a different 27 clips.

This loads the checkpoint's own pickled cfg and replays train()'s selection
verbatim.

Usage:
    uv run scripts/dump_heldout_clips.py --checkpoint outputs/simple_es/child_only/lr3e-4/update_00500.pt \
        --out outputs/simple_es/child_only/lr3e-4/heldout_clips.txt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import torch

from model.dataset import CrossEmbodimentDataset, load_task_list

REPO_ROOT = Path(__file__).resolve().parent.parent


def split_from_cfg(cfg):
    """the (train_clips, heldout_clips) that train_es.train() would compute."""
    from model.simple.train_es import clip_index

    dataset_dir = REPO_ROOT / cfg.dataset_dir
    task_filter = None
    if getattr(cfg, "task_group", "all") != "all":
        task_filter = load_task_list(dataset_dir / "splits" / f"{cfg.task_group}_tasks.txt")
    dataset = CrossEmbodimentDataset(dataset_dir, task_filter=task_filter)
    by_body = dataset.indices_by_body()
    if getattr(cfg, "train_bodies", None):
        bodies = list(cfg.train_bodies)
    else:
        p = dataset_dir / "splits" / "train_bodies.txt"
        bodies = ([b for b in load_task_list(p) if b in by_body] if p.exists() else list(by_body))

    clips = clip_index(dataset)
    train_clips = sorted(k for k, m in clips.items() if all(b in m for b in bodies))

    if getattr(cfg, "clip_list", ""):
        want = set()
        for line in load_task_list(REPO_ROOT / cfg.clip_list):
            parts = line.replace(",", " ").split()
            want.add((parts[0], parts[1]) if len(parts) >= 2 else tuple(line.rsplit("_", 1)))
        keep = [c for c in train_clips if (c[0], str(c[1])) in want or c in want]
        train_clips = sorted(keep)

    heldout = []
    if getattr(cfg, "heldout_clip_frac", 0) > 0:
        n_held = max(1, round(cfg.heldout_clip_frac * len(train_clips)))
        rng_h = random.Random(cfg.eval_seed)
        # MUST mirror train_es.train() exactly, including the stratified branch:
        # a uniform sample of the same size from the same list is a DIFFERENT 50
        # clips, and rendering those would report a model on clips it trained on.
        clip_cat = {}
        if getattr(cfg, "clip_categories", ""):
            for line in load_task_list(REPO_ROOT / cfg.clip_categories):
                t, c = line.split()
                clip_cat[t] = c
        if clip_cat:
            groups = {}
            for c in train_clips:
                groups.setdefault(clip_cat[c[0]], []).append(c)
            base, extra = divmod(n_held, len(groups))
            order = sorted(groups, key=lambda g: -len(groups[g]))
            heldout = []
            for i, g in enumerate(order):
                k = min(base + (1 if i < extra else 0), len(groups[g]))
                heldout += rng_h.sample(groups[g], k)
            heldout = sorted(heldout)
        else:
            heldout = sorted(rng_h.sample(train_clips, n_held))
        train_clips = sorted(set(train_clips) - set(heldout))
    return train_clips, heldout


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None, help="write the held-out clips here, '<task> <trial>' per line")
    p.add_argument("--which", default="heldout", choices=["heldout", "train"])
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    train_clips, heldout = split_from_cfg(cfg)
    print(f"{args.checkpoint} (update {ck.get('update')})")
    print(f"  dataset {cfg.dataset_dir}, clip_list {cfg.clip_list or '-'}, "
          f"heldout_frac {getattr(cfg, 'heldout_clip_frac', 0)}, eval_seed {cfg.eval_seed}")
    print(f"  {len(train_clips)} train clips, {len(heldout)} held out")
    sel = heldout if args.which == "heldout" else train_clips
    print("  " + "  ".join(f"{t}_{k}" for t, k in sel[:8]) + (" ..." if len(sel) > 8 else ""))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{t} {k}\n" for t, k in sel))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
