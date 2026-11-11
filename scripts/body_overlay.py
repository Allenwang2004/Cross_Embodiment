#!/usr/bin/env python3
"""body_overlay.py -- scripts/robot_overlay.py for the humanoid (humenv / SMPL) bodies: several trajectories of one clip
on one body in ONE scene, so their differences show at a glance.

The first trajectory is a translucent grey ghost (e.g. the body's retargeted reference), the others solid robots in
their own colours (red, blue, green, orange), all at their own world positions, so drift shows as a body leaving the
ghost. The camera follows the ghost's pelvis at robot_overlay.py's angle. Extra bodies are added with mjv_addGeoms
(moving geoms only, so the floor is drawn once); every body geom is drawn flat in its colour (texture off).
--clean: no title / legend band, just the scene (e.g. survey videos). 1920 x 1080, 30 fps, plus <out>.png (three moments).
The scene, painting, legend and writer are crossenbodiment.video's (VideoRenderer.overlay; OOP refactor step 10), the colours
crossenbodiment.palette.OVERLAY, the camera crossenbodiment.camera.ViewSpec. Trajectory paths are read through
crossenbodiment.paths.read_path ("outputs/..." under $CE_OUTPUTS_ROOT).
usage: MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/body_overlay.py --xml assets/robots_torque/child/robot_torque_full.xml \
           --traj "reference=data/child/retargeting_motion/<task>/<stem>.npz" --traj "z0=<run>/origin_z.npz" \
           --traj "searched z=<run>/best.npz" --out <out>.mp4 [--clean] [--distance 3.1 --lookat-z 0.5]
"""
import argparse, os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
from crossenbodiment import paths                                      # noqa: E402
from crossenbodiment.camera import ViewSpec                            # noqa: E402
from crossenbodiment.palette import OVERLAY as COLORS                  # noqa: E402
from crossenbodiment.video import OVERLAY_SIZE, VideoRenderer, legend, moments, write   # noqa: E402

W, H = OVERLAY_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--traj", action="append", required=True, help='"label=path.npz" (qpos); the first is the ghost')
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean", action="store_true", help="no title / legend band")
    ap.add_argument("--title", default="")
    ap.add_argument("--distance", type=float, default=3.1)
    ap.add_argument("--lookat-z", type=float, default=0.5)
    ap.add_argument("--root", default="Pelvis")
    a = ap.parse_args()
    runs = [(t.split("=", 1)[0], paths.read_path(t.split("=", 1)[1])) for t in a.traj]
    trajs = [np.load(p)["qpos"].astype(np.float64) for _, p in runs]
    T = min(len(q) for q in trajs)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(REPO / a.xml))
    items = [(n, COLORS[i]) for i, (n, _) in enumerate(runs)]
    frames = VideoRenderer.overlay(m, trajs, ViewSpec(a.distance, -12, 140, a.lookat_z), a.root, COLORS, (W, H),
                                   decorate=None if a.clean else (lambda img, t: legend(img, a.title or out.stem, items)))
    write(out, frames, fps=30, quality=8, macro_block_size=8)
    Image.fromarray(np.concatenate([frames[t] for t in moments(T)], axis=1)).save(str(out) + ".png")
    print(f"-> {out} ({T} frames, {W}x{H})")


if __name__ == "__main__":
    main()
