#!/usr/bin/env python3
"""Contact sheet of a robot directory -- every body, one camera, one scale.

The point of the sheet is comparison, so every tile is rendered with the SAME
camera at the SAME distance: a body that is half as tall draws half as tall.
(Per-tile framing would make every body fill its box and hide exactly the
difference you are looking for.) The distance is solved once from the tallest
body in the set, so the sheet stays tight whatever range it covers.

Each tile is captioned with what the render cannot show -- stature, mass, BMI --
and a body whose robot.xml no longer matches its parameter.json is captioned
STALE and outlined, because a sheet silently showing a superseded body is worse
than no sheet. Rebuild those with scale_robot_l1.py --rebuild <label>.

Usage (from project root):
  uv run scripts/render_bodies.py
  uv run scripts/render_bodies.py --cols 5 --out outputs/renders/advance.png
  uv run scripts/render_bodies.py --robots assets/robots --views front side
  uv run scripts/render_bodies.py --body s1_b1 s1_b2 --views front side
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_robot_l1 import L1_AXES, build_body, describe

ROOT = Path(__file__).resolve().parent.parent
# azimuth per view, in MuJoCo's convention: 90 looks at the body's front.
VIEWS = {"front": 90.0, "side": 180.0, "three_quarter": 135.0}
BG_LABEL, FG_LABEL, FG_STALE = (24, 26, 25), (238, 240, 238), (226, 138, 95)


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def extents(model, data):
    """Exact bounding box of the standing body, per shape -- the same
    measurement build_body uses, so the caption agrees with the generator
    instead of a bounding sphere. Returns (stature, max |x|, max |y|); the two
    lateral numbers are what a T-pose arm span needs from the camera."""
    top = -np.inf
    hx = hy = 0.0
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor":
            continue
        gtype, xpos = model.geom_type[gid], data.geom_xpos[gid]
        xmat, size = data.geom_xmat[gid].reshape(3, 3), model.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            pts = [xpos + xmat @ (np.array(s) * size) for s in itertools.product([1, -1], repeat=3)]
            z, r = max(p[2] for p in pts), 0.0
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            pts = [xpos + xmat @ np.array([0, 0, k * size[1]]) for k in (1, -1)]
            z, r = max(p[2] for p in pts) + size[0], size[0]
        else:
            pts, z, r = [xpos], xpos[2] + size[0], size[0]
        top = max(top, z)
        hx = max(hx, max(abs(p[0]) for p in pts) + r)
        hy = max(hy, max(abs(p[1]) for p in pts) + r)
    return float(top), float(hx), float(hy)


def is_stale(body_dir, mass):
    """Does robot.xml still correspond to parameter.json? Rebuilding from the
    stored (4 dp) axes reproduces the mass to rounding, so a real edit that was
    never rebuilt shows up as a mass mismatch far above that."""
    pj = body_dir / "parameter.json"
    if not pj.is_file():
        return False
    p = json.loads(pj.read_text())
    if p.get("schema") != "l1-30" or any(a not in p for a in L1_AXES):
        return False
    try:
        _, xml = build_body({a: float(p[a]) for a in L1_AXES})
        return abs(describe(xml)["mass"] / mass - 1) > 1e-3
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robots", default="assets/robots_advance")
    ap.add_argument("--body", nargs="*", help="only these labels (default: all)")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--views", nargs="+", default=["front"], choices=list(VIEWS))
    ap.add_argument("--tile", type=int, nargs=2, default=(420, 660), metavar=("W", "H"))
    ap.add_argument("--out", default="outputs/renders/robots_advance.png")
    ap.add_argument("--elevation", type=float, default=-4.0)
    args = ap.parse_args()

    rdir = ROOT / args.robots
    dirs = [d for d in sorted(rdir.iterdir(), key=lambda p: natural_key(p.name))
            if (d / "robot.xml").is_file() and (not args.body or d.name in args.body)]
    assert dirs, f"no bodies with a robot.xml under {rdir}"

    # one pass to measure, so the camera can be solved for the whole set at once
    info = []
    for d in dirs:
        m = mujoco.MjModel.from_xml_path(str(d / "robot.xml"))
        dt = mujoco.MjData(m)
        mujoco.mj_forward(m, dt)
        mass = float(m.body_mass.sum())
        h, hx, hy = extents(m, dt)
        info.append({"dir": d, "model": m, "data": dt, "mass": mass, "stature": h,
                     "half_w": max(hx, hy), "stale": is_stale(d, mass)})
    tallest = max(i["stature"] for i in info)
    widest = max(i["half_w"] for i in info)

    W, H = args.tile
    # Solve the camera distance from the set's own bounding box rather than a
    # tuned constant: the tallest body must fit the frame HEIGHT and the widest
    # T-pose arm span must fit its WIDTH, whichever binds. MuJoCo's fovy is
    # vertical, so the horizontal half-angle carries the aspect ratio. The 1.10
    # margin covers the elevation tilt, which lifts the far foot in frame.
    fovy = float(info[0]["model"].vis.global_.fovy)
    t = np.tan(np.radians(fovy) / 2)
    distance = 1.10 * max(tallest / 2 / t, widest / (t * W / H))
    lookat_z = tallest / 2
    strip = 26

    tiles = []
    for i in info:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0, 0, lookat_z]
        cam.distance, cam.elevation = distance, args.elevation
        row = []
        for v in args.views:
            cam.azimuth = VIEWS[v]
            r = mujoco.Renderer(i["model"], height=H, width=W)
            r.update_scene(i["data"], camera=cam)
            row.append(Image.fromarray(r.render()))
            r.close()
        tile = Image.new("RGB", (W * len(row), H + strip), (255, 255, 255))
        for j, im in enumerate(row):
            tile.paste(im, (j * W, 0))
        dr = ImageDraw.Draw(tile)
        # caption UNDER the body: with one camera for the whole set a short body
        # sits low in its tile, and a caption above it reads as belonging to the
        # row above.
        dr.rectangle([0, H, tile.width, tile.height], fill=BG_LABEL)
        bmi = i["mass"] / i["stature"] ** 2
        txt = (f"{i['dir'].name}   {i['stature']:.2f} m   {i['mass']:.1f} kg   BMI {bmi:.1f}"
               + ("   STALE — rebuild" if i["stale"] else ""))
        dr.text((9, H + 8), txt, fill=FG_STALE if i["stale"] else FG_LABEL)
        if i["stale"]:
            dr.rectangle([0, 0, tile.width - 1, tile.height - 1], outline=FG_STALE, width=3)
        tiles.append(tile)

    cols = min(args.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))
    for n, t in enumerate(tiles):
        sheet.paste(t, ((n % cols) * tw, (n // cols) * th))
    grid = ImageDraw.Draw(sheet)
    for c in range(1, cols):
        grid.line([(c * tw, 0), (c * tw, sheet.height)], fill=(226, 228, 226), width=1)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    stale = [i["dir"].name for i in info if i["stale"]]
    print(f"wrote {out}  ({len(tiles)} bodies, {cols}x{rows}, {sheet.width}x{sheet.height} px)")
    print(f"  stature {min(i['stature'] for i in info):.2f}-{tallest:.2f} m   "
          f"mass {min(i['mass'] for i in info):.1f}-{max(i['mass'] for i in info):.1f} kg")
    if stale:
        print(f"  WARNING: robot.xml is out of date for {len(stale)} bodies: {', '.join(stale)}")
        print(f"           uv run scripts/scale_robot_l1.py --rebuild <label>")


if __name__ == "__main__":
    main()
