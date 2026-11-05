#!/usr/bin/env python3
"""body_overlay.py -- scripts/robot_overlay.py for the humanoid (humenv / SMPL) bodies: several trajectories of one clip
on one body in ONE scene, so their differences show at a glance.

The first trajectory is a translucent grey ghost (e.g. the body's retargeted reference), the others solid robots in
their own colours (red, blue, green, orange), all at their own world positions, so drift shows as a body leaving the
ghost. The camera follows the ghost's pelvis at robot_overlay.py's angle. Extra bodies are added with mjv_addGeoms
(moving geoms only, so the floor is drawn once); every body geom is drawn flat in its colour (texture off).
--clean: no title / legend band, just the scene (e.g. survey videos). 1920 x 1080, 30 fps, plus <out>.png (three moments).
usage: MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/body_overlay.py --xml assets/robots_torque/child/robot_torque_full.xml \
           --traj "reference=data/child/retargeting_motion/<task>/<stem>.npz" --traj "z0=<run>/origin_z.npz" \
           --traj "searched z=<run>/best.npz" --out <out>.mp4 [--clean] [--distance 3.1 --lookat-z 0.5]
"""
import argparse, os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import imageio
import mujoco
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "scripts"))
from robot_overlay import COLORS, W, H, legend   # noqa: E402


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
    runs = [(t.split("=", 1)[0], REPO / t.split("=", 1)[1]) for t in a.traj]
    trajs = [np.load(p)["qpos"].astype(np.float64) for _, p in runs]
    T = min(len(q) for q in trajs)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(REPO / a.xml))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    d0 = mujoco.MjData(m); ds = [mujoco.MjData(m) for _ in trajs[1:]]
    r = mujoco.Renderer(m, height=H, width=W)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = a.distance, -12, 140
    root = m.body(a.root).id

    def paint(scn, lo, hi, rgba):
        for i in range(lo, hi):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] != 0:
                g.rgba[:] = rgba; g.matid = -1

    items = [(n, COLORS[i]) for i, (n, _) in enumerate(runs)]
    frames = []
    for t in range(T):
        d0.qpos[:] = trajs[0][t]; mujoco.mj_forward(m, d0)      # forward, not kinematics: the lights track the body
        cam.lookat[:] = [d0.xpos[root][0], d0.xpos[root][1], a.lookat_z]
        r.update_scene(d0, camera=cam); scn = r.scene
        paint(scn, 0, scn.ngeom, COLORS[0])
        for k, dk in enumerate(ds, start=1):
            dk.qpos[:] = trajs[k][t]; mujoco.mj_kinematics(m, dk)
            n0 = scn.ngeom
            mujoco.mjv_addGeoms(m, dk, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            paint(scn, n0, scn.ngeom, COLORS[k])
        img = r.render().copy()
        frames.append(img if a.clean else legend(img, a.title or out.stem, items))
    r.close()
    imageio.mimsave(out, frames, fps=30, quality=8, macro_block_size=8)
    Image.fromarray(np.concatenate([frames[int(T * f)] for f in (0.2, 0.5, 0.85)], axis=1)).save(str(out) + ".png")
    print(f"-> {out} ({T} frames, {W}x{H})")


if __name__ == "__main__":
    main()
