#!/usr/bin/env python3
"""render_seed_bests.py -- the reference next to every seed's best rollout.

For one clip, takes the runs scripts/single_z_search.py wrote under different
seeds (outputs/single_z_seeds*/<clip>_<objective>_s<seed>/) and plays their
best.npz -- the qpos the search itself recorded for its best candidate, not a
re-rollout -- side by side with the retargeting reference, on one timeline.
Each panel is captioned with the seed, its cost and cos(best_z, z0), so the
question "do z's that B() scores alike also LOOK alike?" can be answered by
eye.

Writes <root>/seed_bests_<clip>.mp4 and a contact sheet
<root>/seed_bests_<clip>.png (three moments of the clip, panels across).

Usage:
    uv run scripts/render_seed_bests.py --root outputs/single_z_seeds_s005
    uv run scripts/render_seed_bests.py --root outputs/single_z_seeds_s005 --clip move-ego-0-2_4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from rollout_z_trace import label

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="outputs/single_z_seeds_s005")
    ap.add_argument("--clip", nargs="*", default=None, help="stem(s); default: every clip under --root")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--no-z0", action="store_true", help="omit the z0 baseline panel")
    ap.add_argument("--camera", default="front_side")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    runs = {}
    for d in sorted(root.iterdir()):
        m = re.match(r"^(.+)_(\w+?)_(s\d+)$", d.name)
        if m and (d / "best.npz").exists() and not d.name.endswith("rep"):
            runs.setdefault(m.group(1), []).append((m.group(3), d))
    clips = args.clip or sorted(runs)

    for clip in clips:
        seeds = runs.get(clip)
        if not seeds:
            print(f"{clip}: no runs"); continue
        s0 = json.loads((seeds[0][1] / "summary.json").read_text())
        task, stem = s0["clip"].split("/")
        xml = Path(s0["xml"])
        ref = np.load(REPO_ROOT / "data" / s0["body"] / "retargeting_motion" / task / f"{stem}.npz")["qpos"]
        z0 = np.load(REPO_ROOT / "data" / "origin_z" / task / f"{stem}.npy").reshape(-1)

        fk = mujoco.MjModel.from_xml_path(str(xml))
        renderer = mujoco.Renderer(fk, height=args.size, width=args.size)
        data = mujoco.MjData(fk)
        T = max(len(ref), max(len(np.load(d / "best.npz")["qpos"]) for _, d in seeds))

        def play(q, text, sub, ended=None):
            frames = []
            for t in range(T):
                data.qpos[:] = q[min(t, len(q) - 1)]
                mujoco.mj_forward(fk, data)
                renderer.update_scene(data, camera=args.camera)
                frames.append(label(renderer.render().copy(), text,
                                    ended if (ended and t >= len(q)) else sub))
            return frames

        panels = [play(ref, "retargeting motion", f"reference, {len(ref)} frames",
                       ended="reference ended")]
        # z0's own rollout, which single_z_search recorded under the same init
        # and env as the search. Without it the video answers "do the seeds look
        # alike" but not "did the search find anything", which is the question
        # when the clip is new and it is the FLOOR that is in doubt.
        oz = seeds[0][1] / "origin_z.npz"
        if oz.exists() and not args.no_z0:
            s0 = json.loads((seeds[0][1] / "summary.json").read_text())
            panels.append(play(np.load(oz)["qpos"], "z0 (no search)",
                               f"{s0['objective']} {s0['origin_z']['cost']:.3f}"))
        for tag, d in seeds:
            s = json.loads((d / "summary.json").read_text())
            bz = np.load(d / "best_z.npy").reshape(-1)
            cos = float(bz @ z0 / (np.linalg.norm(bz) * np.linalg.norm(z0)))
            panels.append(play(np.load(d / "best.npz")["qpos"], f"{tag}  best z (gen {s['best']['gen']})",
                               f"{s['objective']} {s['best']['cost']:.3f}   cos(z, z0) {cos:.3f}"))
        renderer.close()

        cols = min(args.cols, len(panels))
        rows = int(np.ceil(len(panels) / cols))
        blank = np.full_like(panels[0][0], 252)
        grid = []
        for t in range(T):
            band = []
            for r in range(rows):
                band.append(np.concatenate([panels[r * cols + c][t] if r * cols + c < len(panels) else blank
                                            for c in range(cols)], axis=1))
            grid.append(np.concatenate(band, axis=0))
        out = root / f"seed_bests_{clip}.mp4"
        imageio.mimsave(out, grid, fps=args.fps)

        # contact sheet: three moments, all panels across
        ts = [int(len(ref) * f) for f in (0.25, 0.5, 0.85)]
        sheet = np.concatenate([np.concatenate([p[t] for p in panels], axis=1) for t in ts], axis=0)
        Image.fromarray(sheet).save(root / f"seed_bests_{clip}.png")
        print(f"{clip}: {len(seeds)} seeds -> {out}\n{' ' * len(clip)}  -> {root / f'seed_bests_{clip}.png'}")


if __name__ == "__main__":
    main()
