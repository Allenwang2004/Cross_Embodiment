"""video.py -- VideoRenderer: recorded qpos trajectories replayed with MuJoCo into videos (OOP refactor step 10).

Nothing is re-simulated: every frame sets qpos and runs forward kinematics. Two layouts share the painting, the frame labels and the
writer settings:
  side by side   one square panel per trajectory, each on its own model with its own camera (a free camera following that body's
                 centre of mass, or the first panel's with follow "first", or a named camera of the model), laid out in a grid
                 (scripts/render_panels.py's spec; the survey's replay.py "side_by_side")
  overlay        every trajectory in ONE scene: the first drawn by the scene itself (a translucent ghost), the others added with
                 mjv_addGeoms (moving geoms only, so the floor is drawn once), the camera following the first one's root body
                 (scripts/body_overlay.py; robot_overlay.py's loop; replay.py "overlay")
Spec (render_panels.py's JSON, unchanged): out, cols, size, camera | track {distance, elevation, azimuth, lookat_z[, follow]},
panels [{title, sub, xml, qpos, rgba, mesh, mesh_rgba}], captions, label_px, quality, fps. Paths are repository-relative (read
through paths.read_path, so "outputs/..." follows $CE_OUTPUTS_ROOT) or absolute.

  paint(m, scn, lo, hi, rgba, flat=True, check=True)   the scene's body geoms lo..hi in one colour (flat: material / texture off;
                                     check: only geoms of bodies other than the world -- robot_overlay.py's added robots skip it)
  label_px(frame, text, sub, px)    render_panels.py's TrueType title band (draw.frame_label: the small bitmap one)
  legend(img, title, items, px=34)  robot_overlay.py's title + colour legend band
  write(path, frames, fps, quality, macro_block_size)   imageio's mimsave with only the settings given (render_panels: imageio's
                                     defaults, 16-pixel blocks; the overlays: quality 8, 8-pixel blocks)
  moments(T, fractions=(0.2, 0.5, 0.85))   the contact-sheet frames
  VideoRenderer(repo).side_by_side(spec) -> (grid frames, panel frames); .render_spec(spec) writes <out> and <out>.png
  VideoRenderer(repo).overlay(model, trajs, camera, root, colours, ...) -> frames
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import paths
from .draw import frame_label
from .palette import INK

FONT = Path.home() / ".local/share/fonts/NotoSansCJKtc-Regular.otf"
MOMENTS = (0.2, 0.5, 0.85)
OVERLAY_SIZE = (1920, 1080)             # robot_overlay.W, H


def _font_loader(default_takes_size=False):
    from PIL import ImageFont
    if FONT.exists():
        return lambda n: ImageFont.truetype(str(FONT), n)
    return (lambda n: ImageFont.load_default()) if default_takes_size else ImageFont.load_default


def paint(m, scn, lo, hi, rgba, flat=True, check=True):
    """every body geom (not the floor) of scene geoms lo..hi in one colour; flat: material / texture off too"""
    import mujoco
    for i in range(lo, hi):
        g = scn.geoms[i]
        if not check or (g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] != 0):
            g.rgba[:] = rgba
            if flat:
                g.matid = -1


def label_px(frame, text, sub, px):
    """render_panels.label_px: a title band px * 1.6 high with a TrueType title, the subtitle at 0.8 px below it"""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    load = _font_loader()
    band = int(px * 1.6)
    d.rectangle([0, 0, im.width, band], fill=INK["video"]["band"])
    d.text((px // 2, band // 2), text, fill=INK["video"]["title"], font=load(px), anchor="lm")
    if sub:
        d.text((px // 2, band + px // 3), sub, fill=INK["video"]["sub"], font=load(int(px * 0.8)), anchor="lt")
    return np.asarray(im)


def legend(img, title, items, px=34):
    """robot_overlay.legend: a title line and one colour square + label per item (a translucent item: a grey outlined square)"""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    load = _font_loader(default_takes_size=True)
    band = int(px * 1.7) + int(px * 1.25) * len(items)
    d.rectangle([0, 0, im.width, band], fill=INK["video"]["band"])
    d.text((px // 2, int(px * 0.85)), title, fill=INK["video"]["title"], font=load(px), anchor="lm")
    y = int(px * 1.7)
    for (lab, rgba) in items:
        c = tuple(int(255 * x) for x in rgba[:3])
        sq = int(px * 0.8)
        if rgba[3] < 1:
            d.rectangle([px // 2, y, px // 2 + sq, y + sq], outline=INK["video"]["ghost_edge"], width=3, fill=INK["video"]["ghost_fill"])
        else:
            d.rectangle([px // 2, y, px // 2 + sq, y + sq], fill=c)
        d.text((px // 2 + sq + px // 2, y + sq // 2), lab, fill=INK["video"]["legend"], font=load(int(px * 0.8)), anchor="lm")
        y += int(px * 1.25)
    return np.asarray(im)


def write(path, frames, fps=30, quality=None, macro_block_size=None):
    """imageio.v2.mimsave with only the settings given (None: imageio's default, as render_panels.py left them)"""
    import imageio.v2 as imageio
    kw = {}
    if quality is not None:
        kw["quality"] = quality
    if macro_block_size is not None:
        kw["macro_block_size"] = macro_block_size
    imageio.mimsave(path, frames, fps=fps, **kw)


def moments(T, fractions=MOMENTS):
    return [int(T * f) for f in fractions]


def enlarge_offscreen(m, w, h):
    """the XMLs' offscreen buffer (960 high) grown to the frame size"""
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)


class VideoRenderer:
    def __init__(self, repo=None, resolve=None):
        """repo: the root repository-relative spec paths are taken from (default: paths.read_path, i.e. REPO with "outputs/..." under
        $CE_OUTPUTS_ROOT); resolve: a function rel -> Path instead"""
        if resolve is None:
            resolve = paths.read_path if repo is None else (lambda p, r=Path(repo): r / p)
        self.resolve = resolve

    # ---------------------------------------------------------------- side by side (render_panels.py)
    def side_by_side(self, spec):
        """the grid frames and every panel's frames of a render_panels spec (its defaults: size 1080, camera front_side, cols 4;
        label_px size // 30 above 480 px; quality 8 at >= 1080 px)"""
        import mujoco
        size, cam, cols = spec.get("size", 1080), spec.get("camera", "front_side"), spec.get("cols", 4)
        if size > 480:
            spec.setdefault("label_px", size // 30)
        if size >= 1080:
            spec.setdefault("quality", 8)
        trajs = [np.load(self.resolve(p["qpos"]))["qpos"] for p in spec["panels"]]
        T = max(len(q) for q in trajs)
        tr = spec.get("track")
        lead = None
        if tr and tr.get("follow") == "first":
            # every panel's camera follows the FIRST panel's centre of mass (the reference), so a rollout
            # that drifts or turns away shows up as a body leaving the centre instead of being re-centred
            m0 = mujoco.MjModel.from_xml_path(str(self.resolve(spec["panels"][0]["xml"]))); d0 = mujoco.MjData(m0)
            lead = []
            for t in range(T):
                d0.qpos[:] = trajs[0][min(t, len(trajs[0]) - 1)]; mujoco.mj_forward(m0, d0)
                lead.append(d0.subtree_com[1][:2].copy())
        panels = []
        for p, q in zip(spec["panels"], trajs):
            mesh = p.get("mesh", spec.get("mesh", False))
            if mesh:
                import sys
                sys.path.insert(0, str(paths.REPO / "scripts"))
                from smpl_skin import SKIN_RGBA, with_skin
                m = with_skin(self.resolve(p["xml"]), rgba=p.get("mesh_rgba", spec.get("mesh_rgba", SKIN_RGBA)))
            else:
                m = mujoco.MjModel.from_xml_path(str(self.resolve(p["xml"])))
            d = mujoco.MjData(m)
            enlarge_offscreen(m, size, size)
            r = mujoco.Renderer(m, height=size, width=size)
            # spec["track"]: a free camera that follows the body's centre of mass with ONE distance / angle / look-at height shared by
            # every panel, so bodies of different heights stay whole in frame and keep their true relative size. The XML's named
            # cameras sit at a fixed 0.8 m and cut the head off anything adult-sized.
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
                    paint(m, r.scene, 0, r.scene.ngeom, p["rgba"])
                img = r.render().copy()
                if not spec.get("captions", True):
                    frames.append(img)
                    continue
                frames.append(label_px(img, p.get("title", ""), p.get("sub", ""), spec["label_px"]) if spec.get("label_px")
                              else frame_label(img, p.get("title", ""), p.get("sub", "")))
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
        return grid, panels

    def render_spec(self, spec, out=None):
        """side_by_side + the mp4 (<out>, default the spec's "out" under the results root) + <out>.png (three moments, panels across)"""
        from PIL import Image
        grid, panels = self.side_by_side(spec)
        out = Path(out) if out is not None else paths.write_path(spec["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        write(out, grid, fps=spec.get("fps", 30.0), quality=spec.get("quality"))
        T = len(grid)
        sheet = np.concatenate([np.concatenate([p[t] for p in panels], axis=1) for t in moments(T)], axis=0)
        Image.fromarray(sheet).save(str(out) + ".png")
        return out

    # ---------------------------------------------------------------- overlay (body_overlay.py, replay.py)
    @staticmethod
    def overlay(m, trajs, camera, root, colours, size=OVERLAY_SIZE, flat=True, decorate=None):
        """trajs[0] drawn by the scene (forward kinematics: the lights track the body), the others added with mjv_addGeoms;
        camera: a ViewSpec-like (distance, elevation, azimuth, lookat_z) following body `root`; colours: one RGBA per trajectory;
        decorate(img, t) -> img: a label / legend per frame"""
        import mujoco
        W, H = size
        enlarge_offscreen(m, W, H)
        d0 = mujoco.MjData(m); ds = [mujoco.MjData(m) for _ in trajs[1:]]
        r = mujoco.Renderer(m, height=H, width=W)
        opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = camera.distance, camera.elevation, camera.azimuth
        rid = m.body(root).id
        T = min(len(q) for q in trajs)
        frames = []
        for t in range(T):
            d0.qpos[:] = trajs[0][t]; mujoco.mj_forward(m, d0)      # forward, not kinematics: the lights track the body
            cam.lookat[:] = [d0.xpos[rid][0], d0.xpos[rid][1], camera.lookat_z]
            r.update_scene(d0, camera=cam); scn = r.scene
            paint(m, scn, 0, scn.ngeom, colours[0], flat)
            for k, dk in enumerate(ds, start=1):              # the other trajectories' moving geoms, added to the scene
                dk.qpos[:] = trajs[k][t]; mujoco.mj_kinematics(m, dk)
                n0 = scn.ngeom
                mujoco.mjv_addGeoms(m, dk, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                paint(m, scn, n0, scn.ngeom, colours[k], flat)
            img = r.render().copy()
            frames.append(decorate(img, t) if decorate is not None else img)
        r.close()
        return frames
