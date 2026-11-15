#!/usr/bin/env python3
"""replay.py -- re-render the survey videos from the shared trajectories (needs only mujoco, numpy, imageio).

Every clip folder <body>/trajectories/<clip>/ holds the trajectories (qpos per frame, 30 fps) and video.json, which
says how each video is drawn:
  side_by_side   one square panel per trajectory, left to right, each body drawn flat in one colour; a free camera
                 follows each panel's own body (centre of mass), at one distance / height shared by all panels
  overlay        all trajectories in one scene of one model: the first a translucent ghost, the others solid; the
                 camera follows the ghost's root body; "flat": false keeps the model's materials (the robots)
Model paths in video.json are relative to the folder that holds models/ (the top of the shared folder); trajectory
paths are relative to the clip folder.
usage: python replay.py <body>/trajectories/<clip> [--only side_by_side|overlay] [--out-dir DIR]
       (run from the top of the shared folder; writes <clip>_side_by_side.mp4 / <clip>_overlay.mp4 to --out-dir, default .)
"""
import argparse, json, os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio
import mujoco
import numpy as np


def paint(m, scn, lo, hi, rgba, flat=True):
    """every body geom (not the floor) in one colour; flat: material / texture off too."""
    for i in range(lo, hi):
        g = scn.geoms[i]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] != 0:
            g.rgba[:] = rgba
            if flat:
                g.matid = -1


def side_by_side(top, clip, spec):
    size, cam_spec = spec["size"], spec["camera"]
    trajs = [np.load(clip / p["qpos"])["qpos"].astype(np.float64) for p in spec["panels"]]
    T = max(len(q) for q in trajs)
    panels = []
    for p, q in zip(spec["panels"], trajs):
        m = mujoco.MjModel.from_xml_path(str(top / p["model"])); d = mujoco.MjData(m)
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, size), max(m.vis.global_.offheight, size)
        r = mujoco.Renderer(m, height=size, width=size)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = cam_spec["distance"], cam_spec["elevation"], cam_spec["azimuth"]
        frames = []
        for t in range(T):
            d.qpos[:] = q[min(t, len(q) - 1)]; mujoco.mj_forward(m, d)
            c = d.subtree_com[1]; cam.lookat[:] = [c[0], c[1], cam_spec["lookat_z"]]
            r.update_scene(d, camera=cam); paint(m, r.scene, 0, r.scene.ngeom, p["rgba"])
            frames.append(r.render().copy())
        r.close(); panels.append(frames)
    return [np.concatenate([p[t] for p in panels], axis=1) for t in range(T)]


def overlay(top, clip, spec):
    W, H, cam_spec = spec["width"], spec["height"], spec["camera"]
    trajs = [np.load(clip / l["qpos"])["qpos"].astype(np.float64) for l in spec["layers"]]
    T = min(len(q) for q in trajs)
    m = mujoco.MjModel.from_xml_path(str(top / spec["model"]))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    d0 = mujoco.MjData(m); ds = [mujoco.MjData(m) for _ in trajs[1:]]
    r = mujoco.Renderer(m, height=H, width=W)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = cam_spec["distance"], cam_spec["elevation"], cam_spec["azimuth"]
    root = m.body(spec["root"]).id
    frames = []
    for t in range(T):
        d0.qpos[:] = trajs[0][t]; mujoco.mj_forward(m, d0)
        cam.lookat[:] = [d0.xpos[root][0], d0.xpos[root][1], cam_spec["lookat_z"]]
        r.update_scene(d0, camera=cam); scn = r.scene
        paint(m, scn, 0, scn.ngeom, spec["layers"][0]["rgba"], spec.get("flat", True))
        for k, dk in enumerate(ds, start=1):              # the other trajectories' moving geoms, added to the scene
            dk.qpos[:] = trajs[k][t]; mujoco.mj_kinematics(m, dk)
            n0 = scn.ngeom
            mujoco.mjv_addGeoms(m, dk, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            paint(m, scn, n0, scn.ngeom, spec["layers"][k]["rgba"], spec.get("flat", True))
        frames.append(r.render().copy())
    r.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="<body>/trajectories/<clip>, relative to the top of the shared folder")
    ap.add_argument("--only", choices=["side_by_side", "overlay"], default=None)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--top", default=".", help="the folder that holds models/ (default: the current folder)")
    a = ap.parse_args()
    top, clip = Path(a.top), Path(a.top) / a.clip
    spec = json.loads((clip / "video.json").read_text())
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for kind, fn in (("side_by_side", side_by_side), ("overlay", overlay)):
        if a.only and kind != a.only:
            continue
        frames = fn(top, clip, spec[kind])
        path = out / f"{clip.name}_{kind}.mp4"
        # the survey videos' writer settings: render_panels (side by side) left imageio's default 16-pixel blocks, so
        # 3240 x 1080 was stored as 3248 x 1088; the overlays used 8-pixel blocks (1920 x 1080 as is)
        imageio.mimsave(path, frames, fps=spec["fps"], quality=8, macro_block_size=16 if kind == "side_by_side" else 8)
        print(f"-> {path} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


if __name__ == "__main__":
    main()
