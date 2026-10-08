#!/usr/bin/env python3
"""render_panels.py -- play recorded qpos trajectories side by side, one panel
per (body, trajectory), for presentation videos.

Nothing is re-simulated: every panel replays a qpos array that a search already
recorded (single_z_search's origin_z.npz = the rollout of z0, best.npz = the
rollout of the best z it found) or a retargeted reference, on that panel's own
body. So a panel shows exactly the rollout its caption's cost was measured on.

Spec (JSON):
  {"out": "docs/figures/journey/v2_child_move.mp4",
   "cols": 4, "size": 320, "camera": "front_side",
   "panels": [{"title": "reference", "sub": "retargeted to child",
               "xml": "assets/robots_torque/m2c_t1000/robot_torque_full.xml",
               "qpos": "data/m2c_t1000/retargeting_motion/<task>/<stem>.npz"}, ...]}

"track": {"distance", "elevation", "azimuth", "lookat_z"[, "follow": "first"]}: a free camera on each body's
centre of mass, or with "follow": "first" on the first panel's for every panel (drift stays visible).

Writes <out> (mp4) and <out>.png (contact sheet: three moments, panels across).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from rollout_z_trace import label

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    size, cam, cols = spec.get("size", 320), spec.get("camera", "front_side"), spec.get("cols", 4)
    trajs = [np.load(REPO / p["qpos"])["qpos"] for p in spec["panels"]]
    T = max(len(q) for q in trajs)
    tr = spec.get("track")
    lead = None
    if tr and tr.get("follow") == "first":
        # every panel's camera follows the FIRST panel's centre of mass (the reference), so a rollout
        # that drifts or turns away shows up as a body leaving the centre instead of being re-centred
        m0 = mujoco.MjModel.from_xml_path(str(REPO / spec["panels"][0]["xml"])); d0 = mujoco.MjData(m0)
        lead = []
        for t in range(T):
            d0.qpos[:] = trajs[0][min(t, len(trajs[0]) - 1)]; mujoco.mj_forward(m0, d0)
            lead.append(d0.subtree_com[1][:2].copy())
    panels = []
    for p, q in zip(spec["panels"], trajs):
        m = mujoco.MjModel.from_xml_path(str(REPO / p["xml"]))
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, height=size, width=size)
        # spec["track"]: a free camera that follows the body's centre of mass
        # with ONE distance / angle / look-at height shared by every panel, so
        # bodies of different heights stay whole in frame and keep their true
        # relative size. The XML's named cameras sit at a fixed 0.8 m and cut
        # the head off anything adult-sized.
        fc = None
        if tr:
            fc = mujoco.MjvCamera()
            fc.type = mujoco.mjtCamera.mjCAMERA_FREE
            fc.distance, fc.elevation, fc.azimuth = tr["distance"], tr["elevation"], tr["azimuth"]
        frames = []
        for t in range(T):
            d.qpos[:] = q[min(t, len(q) - 1)]
            mujoco.mj_forward(m, d)
            if fc is not None:
                c = lead[t] if lead is not None else d.subtree_com[1]
                fc.lookat[:] = [c[0], c[1], tr["lookat_z"]]
                r.update_scene(d, camera=fc)
            else:
                r.update_scene(d, camera=cam)
            frames.append(label(r.render().copy(), p["title"], p.get("sub", "")))
        r.close()
        panels.append(frames)
    cols = min(cols, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    blank = np.full_like(panels[0][0], 252)
    grid = []
    for t in range(T):
        band = [np.concatenate([panels[i][t] if i < len(panels) else blank
                                for i in range(r * cols, r * cols + cols)], axis=1) for r in range(rows)]
        grid.append(np.concatenate(band, axis=0))
    out = REPO / spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, grid, fps=spec.get("fps", 30.0))
    ts = [int(T * f) for f in (0.2, 0.5, 0.85)]
    sheet = np.concatenate([np.concatenate([p[t] for p in panels], axis=1) for t in ts], axis=0)
    Image.fromarray(sheet).save(str(out) + ".png")
    print(f"-> {out}\n-> {out}.png")


if __name__ == "__main__":
    main()
