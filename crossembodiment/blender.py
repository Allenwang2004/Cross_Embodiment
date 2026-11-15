"""blender.py -- MuJoCo poses as Blender panels: BlenderScene (scene.json + meshes.npz for scripts/render_blender.py) and the pose
export every Blender script shares (OOP refactor step 10; the functions are scripts/export_blender_package.py's, unchanged).

  load_model(xml, payload=False)    compile an MJCF; payload: + the drawn-only weight sphere on the right hand (the teaser's)
  pose_spec(m, q, t, x, tint, heading, red_leg, tag, meshes)
                                    the visible geoms at frame t, turned by -heading about the root, the root put at (x, 0): one
                                    pose of scene.json; mesh geoms add their vertices / faces to `meshes` (a dict) as <tag>_<mesh>
  contact_frames(m, q, frames)      the nearest frames with a foot on the ground, and the lowest point per frame
  heading(q, frames)                the walking direction over the frames (10 frames either side)
  body_height(m)                    standing height of a model in its compiled pose (walk_band / walk_bodies / teaser_band_tall)
  panel(figure, w_u, h_u, feet_u, per_m, poses, white=False)   a panel sized in figure units (0.01 in) at per_m units per metre
  BlenderScene                      panels + their own mesh store (no module-global `meshes`) + top-level extras:
      .pose(...) / .strip(...) / .fall_strip(...)   the exports above, meshes into this scene
      .add_panel(name, panel)       .write(folder)  scene.json {"version": SCHEMA_VERSION, "panels": ..., extras} + meshes.npz +
                                                   a copy of scripts/render_blender.py (which ignores the version field)
      BlenderScene.load(folder)     an existing scene (to update single panels)
      .render(folder, samples, threads, panels=None, scale=None, out="renders")   the `uv run --no-project --python 3.11 --with bpy
                                    python render_blender.py -- ...` call the scripts made (Cycles on the CPU, log in render.log)
render_blender.py stays a standalone single file: it renders a scene folder without this repository.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import paths

SCHEMA_VERSION = 1
RENDER_SCRIPT = paths.REPO / "scripts/render_blender.py"
DPI = 600                                         # export_blender_package.panel: px = figure units / 100 * DPI
LOCKED_LEG = ("R_Hip", "R_Knee", "R_Ankle", "R_Toe")


_KINDS = {}


def _geom_kinds():
    if _KINDS:
        return _KINDS
    import mujoco
    G = mujoco.mjtGeom
    _KINDS.update({G.mjGEOM_SPHERE: "sphere", G.mjGEOM_CAPSULE: "capsule", G.mjGEOM_CYLINDER: "cylinder", G.mjGEOM_BOX: "box",
            G.mjGEOM_ELLIPSOID: "ellipsoid", G.mjGEOM_MESH: "mesh"})
    return _KINDS


def load_model(xml, payload=False):
    """export_blender_package.load: payload adds the drawn-only weight sphere over the right hand (as plot_teaser.py draws it)"""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(xml))
    if payload:                                 # drawn only, as in plot_teaser.py
        hand = spec.compile(); ids = [i for i in range(hand.ngeom) if hand.body(hand.geom_bodyid[i]).name == "R_Hand"]; ctr = hand.geom_pos[ids].mean(0)
        g = spec.body("R_Hand").add_geom(); g.name = "payload"; g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.pos = ctr.tolist(); g.size = [max(0.14, 1.1 * float(hand.geom_rbound[ids].max())), 0, 0]; g.contype = 0; g.conaffinity = 0; g.mass = 0
    return spec.compile()


def color_of(m, g, red_leg):
    if m.geom(g).name == "payload":
        return "amber"
    if red_leg and m.body(m.geom_bodyid[g]).name in LOCKED_LEG:
        return "red"
    return "body"


def pose_spec(m, q, t, x, tint, heading, red_leg, tag, meshes):
    """the visible geoms at frame t, the walk turned onto +x and the root put at (x, 0)."""
    import mujoco
    GEOM = _geom_kinds()
    d = mujoco.MjData(m); d.qpos[:] = q[min(t, len(q) - 1)]; mujoco.mj_forward(m, d); c, s = np.cos(-heading), np.sin(-heading); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]); root = d.subtree_com[1][:2].copy(); gs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or m.geom_group[g] >= 3 or m.geom_rgba[g, 3] == 0 or m.geom_type[g] not in GEOM:
            continue
        p = d.geom_xpos[g].copy(); p[:2] -= root; p = Rz @ p; p[0] += x; e = dict(kind=GEOM[m.geom_type[g]], size=[round(float(v), 5) for v in m.geom_size[g]], p=[round(float(v), 5) for v in p], R=[round(float(v), 6) for v in (Rz @ d.geom_xmat[g].reshape(3, 3)).ravel()], color=color_of(m, g, red_leg))
        if e["kind"] == "mesh":
            i = m.geom_dataid[g]; name = f"{tag}_{m.mesh(i).name}"; e["mesh"] = name
            if name + "__v" not in meshes:
                a, n, fa, fn = m.mesh_vertadr[i], m.mesh_vertnum[i], m.mesh_faceadr[i], m.mesh_facenum[i]; meshes[name + "__v"] = m.mesh_vert[a:a + n].astype(np.float32); meshes[name + "__f"] = m.mesh_face[fa:fa + fn].astype(np.int32)
        gs.append(e)
    return dict(x=x, tint=tint, geoms=gs)


def contact_frames(m, q, frames, tol=0.01):
    """for each frame, the nearest one with a foot on the ground: a run spends half its time in the air, and a fixed spacing can land on that every time. Also returns the lowest point per frame."""
    import mujoco
    sys.path.insert(0, str(paths.REPO))
    from model.robot_map import Lowest
    low = Lowest(m); d = mujoco.MjData(m); lo = np.empty(len(q))
    for t in range(len(q)):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(m, d); lo[t] = low(d)
    on = np.flatnonzero(lo <= tol); return [int(on[np.abs(on - f).argmin()]) for f in frames], lo


def heading(q, frames):
    a, b = max(frames[0] - 10, 0), min(frames[-1] + 10, len(q) - 1); v = q[b, :2] - q[a, :2]; return float(np.arctan2(v[1], v[0]))


def body_height(m):
    """top minus bottom of the body geoms' bounding spheres in the compiled pose (walk_band / walk_bodies / teaser_band_tall height)"""
    import mujoco
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    z = [(d.geom_xpos[g][2] + m.geom_rbound[g], d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom) if m.geom_bodyid[g] > 0]
    return max(a for a, _ in z) - min(b for _, b in z)


def panel(figure, w_u, h_u, feet_u, per_m, poses_, white=False):
    return dict(figure=figure, width_m=w_u / per_m, height_m=h_u / per_m, feet_m=feet_u / per_m, px=[round(w_u / 100 * DPI), round(h_u / 100 * DPI)], white=white, poses=poses_)


class BlenderScene:
    """the panels of one render folder and the meshes their poses use; extras: further top-level keys of scene.json (frames,
    body_scale, groups, order, info), written after "panels" in their insertion order"""

    def __init__(self, panels=None, meshes=None, **extra):
        self.panels = dict(panels or {})
        self.base_meshes = dict(meshes or {})     # loaded with the scene: kept, overwritten by a fresh export of the same mesh
        self.meshes = {}                          # this export's meshes (pose_spec adds each once)
        self.extra = dict(extra)

    # ---------------------------------------------------------------- poses (meshes go into this scene's store)
    def pose(self, m, q, t, x=0.0, tint=0.0, heading=0.0, red_leg=False, tag="human"):
        return pose_spec(m, q, t, x, tint, heading, red_leg, tag, self.meshes)

    def strip(self, xml, qpath, frames, tints, dx_m, tag, payload=False, red_leg=False, ground=False):
        """export_blender_package.strip. ground: use the nearest frames with a foot down, set exactly on the floor (at most tol away)."""
        m = load_model(xml, payload); q = np.load(qpath)["qpos"].astype(np.float64); assert q.shape[1] == m.nq, (xml, q.shape, m.nq); h = heading(q, frames); mid = (len(frames) - 1) / 2; lo = None
        if ground:
            frames, lo = contact_frames(m, q, frames); print(" ", tag, "frames", frames, "feet at cm", (lo[frames] * 100).round(1).tolist())
        out = [self.pose(m, q, t, (i - mid) * dx_m, k, h, red_leg, tag) for i, (t, k) in enumerate(zip(frames, tints))]
        if ground:
            for po, t in zip(out, frames):
                for g in po["geoms"]:
                    g["p"][2] = round(g["p"][2] - float(lo[t]), 5)
        return out

    def fall_strip(self, xml, qpath, tag, n=6, span=40, width_m=80 / 42):
        """export_blender_package.fall_strip: the fall of a z0 rollout, up to its lowest point: standing pelvis height from the clip's
        start, the first drop below 60% of it. n = 6 poses, as the other method panels (RESULT); width_m: first to last pose, kept at the
        width of the earlier 5-pose strip so the panel's layout stays."""
        m = load_model(xml); q = np.load(qpath)["qpos"].astype(np.float64); z = q[:, 2]; stand = np.median(z[10:50]); first = int(np.flatnonzero(z < 0.6 * stand)[0]); low = first + int(np.argmin(z[first:first + span])); frames = [int(f) for f in np.linspace(low - span, low, n).round()]; print(" ", tag, "fall: stands", round(float(stand), 2), "first down", first, "lowest", low, round(float(z[low]), 2), "frames", frames); h = heading(q, frames); mid = (n - 1) / 2; tints = np.linspace(0.6, 0, n)
        return [self.pose(m, q, t, (i - mid) * width_m / (n - 1), float(k), h, False, tag) for i, (t, k) in enumerate(zip(frames, tints))]

    # ---------------------------------------------------------------- panels, files, rendering
    def add_panel(self, name, p):
        self.panels[name] = p
        return p

    def to_json(self):
        return dict(version=SCHEMA_VERSION, panels=self.panels, **self.extra)

    def write(self, folder, placeholder=False, render_script=True):
        """scene.json, meshes.npz (placeholder: + a zero array "_", as the scripts without meshes wrote it) and render_blender.py"""
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        (folder / "scene.json").write_text(json.dumps(self.to_json()))
        np.savez_compressed(folder / "meshes.npz", **({"_": np.zeros(1)} if placeholder else {}), **{**self.base_meshes, **self.meshes})
        if render_script:
            shutil.copy(RENDER_SCRIPT, folder / "render_blender.py")
        return folder

    @classmethod
    def load(cls, folder):
        folder = Path(folder); s = json.loads((folder / "scene.json").read_text()); s.pop("version", None)
        panels = s.pop("panels")
        return cls(panels, dict(np.load(folder / "meshes.npz")), **s)

    @staticmethod
    def render(folder, samples, threads, panels=None, scale=None, out="renders", device="CPU", env=None, log="render.log"):
        """render_blender.py in `folder` (Blender's Python through uv; at most one Cycles render at a time on the shared box); log: the
        log file in `folder` (None: discarded, as payload_fall_candidates.py did)"""
        folder = Path(folder)
        sel = sum([["--panel", p] for p in panels], []) if panels else []
        args = ["--samples", str(samples)] + (["--scale", str(scale)] if scale is not None else []) + ["--device", device, "--threads", str(threads), "--out", out]
        subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--"] + args + sel, cwd=folder,
                       env=env if env is not None else dict(os.environ, OMP_NUM_THREADS="1"), check=True,
                       stdout=open(folder / log, "w") if log else subprocess.DEVNULL, stderr=subprocess.STDOUT if log else subprocess.DEVNULL)
