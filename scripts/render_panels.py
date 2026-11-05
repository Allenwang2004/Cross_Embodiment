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

"mesh": true (spec-wide, or per panel): dress the body in the SMPL mesh deformed to its own proportions and hide
its capsules (scripts/smpl_skin.py; visual only; mesh panels are drawn without shadows, which band the skin).
"mesh_rgba" sets the skin colour (spec-wide or per panel).

"size": panel side in pixels (default 1080; keep the whole grid within ~4096 x 2160 so ordinary players open it,
e.g. 2 x 2 or 3 x 1 / 3 x 2 panels at 1080).
"label_px": title size in pixels for a TrueType label (default: size / 30 above size 480, PIL's small bitmap font
at 480 and below);
"captions": false: no title / subtitle band at all (e.g. survey videos); titles may then be omitted.
"rgba": [r, g, b, a] (per panel): draw the body flat in that colour, e.g. the overlay videos' grey / red / blue.
"quality": the mp4's imageio-ffmpeg quality, 0-10 (default 8 at size >= 1080, else 5).

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
from PIL import Image, ImageDraw, ImageFont

from rollout_z_trace import label

REPO = Path(__file__).resolve().parent.parent
FONT = Path.home() / ".local/share/fonts/NotoSansCJKtc-Regular.otf"


def label_px(frame, text, sub, px):
    """label() at a readable size for large panels: title band px * 1.6 high, subtitle at 0.8 px below it."""
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    load = (lambda n: ImageFont.truetype(str(FONT), n)) if FONT.exists() else ImageFont.load_default
    band = int(px * 1.6)
    d.rectangle([0, 0, im.width, band], fill=(252, 252, 251))
    d.text((px // 2, band // 2), text, fill=(11, 11, 11), font=load(px), anchor="lm")
    if sub:
        d.text((px // 2, band + px // 3), sub, fill=(82, 81, 78), font=load(int(px * 0.8)), anchor="lt")
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    size, cam, cols = spec.get("size", 1080), spec.get("camera", "front_side"), spec.get("cols", 4)
    if size > 480:
        spec.setdefault("label_px", size // 30)
    if size >= 1080:
        spec.setdefault("quality", 8)
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
        mesh = p.get("mesh", spec.get("mesh", False))
        if mesh:
            from smpl_skin import SKIN_RGBA, with_skin
            m = with_skin(REPO / p["xml"], rgba=p.get("mesh_rgba", spec.get("mesh_rgba", SKIN_RGBA)))
        else:
            m = mujoco.MjModel.from_xml_path(str(REPO / p["xml"]))
        d = mujoco.MjData(m)
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, size)     # the XMLs' offscreen buffer is 960 high
        m.vis.global_.offheight = max(m.vis.global_.offheight, size)
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
            if mesh:      # MuJoCo's shadow map leaves jagged self-shadow bands on the skin; draw mesh panels unshadowed
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            if p.get("rgba") is not None:      # every body geom flat in one colour (texture off), as body_overlay draws them
                for gi in range(r.scene.ngeom):
                    g = r.scene.geoms[gi]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] != 0:
                        g.rgba[:] = p["rgba"]; g.matid = -1
            img = r.render().copy()
            if not spec.get("captions", True):
                frames.append(img)
                continue
            frames.append(label_px(img, p.get("title", ""), p.get("sub", ""), spec["label_px"]) if spec.get("label_px")
                          else label(img, p.get("title", ""), p.get("sub", "")))
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
    imageio.mimsave(out, grid, fps=spec.get("fps", 30.0), **({"quality": spec["quality"]} if "quality" in spec else {}))
    ts = [int(T * f) for f in (0.2, 0.5, 0.85)]
    sheet = np.concatenate([np.concatenate([p[t] for p in panels], axis=1) for t in ts], axis=0)
    Image.fromarray(sheet).save(str(out) + ".png")
    print(f"-> {out}\n-> {out}.png")


if __name__ == "__main__":
    main()
