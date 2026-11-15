"""strobe.py -- strobe panels: which frames, how they fade, how far apart and which way they face (OOP refactor step 10).

StrobeLayout(frames, dx, view, azimuth, offset, align, tint_max, nudge) -> the poses of one strobe panel:
  frames      StrobeLayout.frame_range(a, b, step) (the last frame must be b), an explicit list, around_fall(...) (from 40 frames before
              z0's first mid-motion fall, the overview pages' rule) or whole_clip(...); contact-snapped frames are
              crossembodiment.blender.contact_frames (the bands)
  tints       tint_max * (1 - i / (n - 1)): light (early) to full colour (late); 0.62 for the strobes, 0.6 for the top views
  shifts      (i - (n - 1) / 2) * dx + nudge metres sideways in the image (render_blender.py's "shift"), every pose at one depth
  view        "fixed": one world view, camera at world azimuth `azimuth` (default 140, the survey videos'): heading =
              azimuth - 125 (render_blender.py's camera azimuth in the panel frame); "path": side-on to the panel's own path from the
              first to the last frame (math.atan2, + 90 deg; strobe_final_blender.py); "orbit": the strobe_figures recipes'
              orbit_auto, path heading (np.arctan2, 0 when the path is shorter than 1 mm) + offset (export_strobe_blender.py)
  align       each pose also turned about its root by (its root yaw - the reference's) at that frame
  .poses(scene, m, q, ...)    pose specs (BlenderScene.pose + "shift"), recolored (the G1 black) if asked
  .panel(poses, W, H, ...)    the panel dict (figure "strobe", width / height in metres, feet height, px, white floor, frames, view)
  strobe_size(clip, dx, ...)  the strobe panels' width / height: (n - 1) dx + 1.6 m (2.4 for a crawl) x 2.45 m (1.4 for a crawl)
recipe_panels(...)            export_strobe_blender.py's clip_panels (archived in step 10): the panels of one survey-format clip folder
                              (adult reference, z0, LOES's best) from a strobe_figures recipe (strobe: start/end/interval, camera:
                              fixed | orbit_auto + azimuth_offset); used by scripts/body_all_blender.py and payload_fall_candidates.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths
from .camera import BLENDER_AZIMUTH as AZ, SURVEY_AZIMUTH

STROBE_TINT = 0.62
TOPDOWN_TINT = 0.6


def fade_tints(n, top=STROBE_TINT):
    """top * (1 - i / (n - 1)) for i < n: how far each of n poses fades towards the floor colour (light early, full colour last)"""
    return [top * (1 - i / (n - 1)) for i in range(n)]


def stretches(below):
    """(start, end) of every run of True in `below` (child_all_blender.py's and body_all_blender.py's copies)"""
    out, t = [], 0
    while t < len(below):
        if below[t]:
            u = t
            while u < len(below) and below[u]:
                u += 1
            out.append((t, u)); t = u
        else:
            t += 1
    return out


def mid_falls(z, ref_z, start=45, minlen=9):
    """the stretches with the pelvis below half the reference's that start at frame >= start and last >= minlen frames"""
    return [(a, b) for a, b in stretches(z < 0.5 * ref_z) if a >= start and b - a >= minlen]


def around_fall(fall, n_frames, n=6, step=15, before=40):
    """(start, end, step) of n poses step frames apart from `before` frames before the fall (start, end) (kept inside the clip)"""
    a, b = fall; s = int(np.clip(a - before, 1, n_frames - 1 - step * (n - 1)))
    return (s, s + step * (n - 1), step)


def whole_clip(n_frames, n=6):
    """(start, end, step) of n poses evenly over the clip, from frame 1"""
    iv = (n_frames - 2) // (n - 1)
    return (1, 1 + iv * (n - 1), iv)


def path_heading(q, frames):
    """export_strobe_blender.py: the root path's direction from the first to the last frame (0 if it moved less than 1 mm)"""
    a, b = frames[0], frames[-1]; v = q[b, :2] - q[a, :2]
    return float(np.arctan2(v[1], v[0])) if np.hypot(*v) > 1e-3 else 0.0


def strobe_size(clip, n, dx, wider=0.0, H=None):
    """width and height (m) of a strobe panel: one frame per clip, so the adult and the G1 share a scale; a crawl lies along the floor"""
    crawl = "crawl" in clip
    W = (n - 1) * dx + (2.4 if crawl else 1.6) + wider; H = H if H is not None else (1.4 if crawl else 2.45)
    return W, H


@dataclass
class StrobeLayout:
    frames: list
    dx: float = 0.6
    view: str = "fixed"                  # fixed | path | orbit
    azimuth: float = SURVEY_AZIMUTH      # fixed: the camera's world azimuth (deg)
    offset: float = 90.0                 # orbit: path heading + offset (deg)
    align: bool = False
    tint_max: float = STROBE_TINT
    nudge: float = 0.0
    tints: list = field(init=False)

    def __post_init__(self):
        self.frames = list(self.frames); n = len(self.frames)
        self.tints = fade_tints(n, self.tint_max)

    @staticmethod
    def frame_range(a, b, step):
        frames = list(range(a, b + 1, step)); assert frames[-1] == b
        return frames

    @property
    def n(self):
        return len(self.frames)

    def shift(self, i):
        return (i - (self.n - 1) / 2) * self.dx + self.nudge

    def heading(self, q, t=None, yaw=None, ref_yaw=None):
        """the heading (rad) a pose at frame t is turned by (minus): the camera's world azimuth minus render_blender.py's 125 deg"""
        if self.view == "fixed":
            h = math.radians(self.azimuth - AZ)
        elif self.view == "path":
            v = q[self.frames[-1], :2] - q[self.frames[0], :2]; h = math.atan2(v[1], v[0]) + math.radians(90.0 - AZ)
        elif self.view == "orbit":
            h = path_heading(q, self.frames) + math.radians(self.offset - AZ)
        else:
            raise ValueError(self.view)
        if ref_yaw is not None:
            h += float(yaw[t] - ref_yaw[t])                       # turned to the reference's heading at this frame
        return h

    def poses(self, scene, m, q, red_leg=False, tag="human", recolor=None, ref_yaw=None):
        """ref_yaw: the reference's root yaw per frame (align: each pose turned to it); recolor: every geom drawn in that colour"""
        y = None
        if ref_yaw is not None:
            from .geometry import root_yaw
            y = root_yaw(q)
        out = []
        for i, (t, k) in enumerate(zip(self.frames, self.tints)):
            ps = scene.pose(m, q, t, 0.0, k, self.heading(q, t, y, ref_yaw), red_leg, tag)
            if recolor is not None:
                for g in ps["geoms"]:
                    g["color"] = recolor
            out.append(dict(ps, shift=self.shift(i)))
        return out

    def panel(self, poses, W, H, view, pxw=1500, feet=0.3, elevation=None):
        p = dict(figure="strobe", width_m=W, height_m=H, feet_m=feet, px=[pxw, round(pxw * H / W)], white=True, poses=poses,
                 frames=self.frames, view=view)
        if elevation is not None:
            p["elevation"] = elevation
        return p


# ------------------------------------------------------------------ the survey picks' recipes (export_strobe_blender.py, archived)

SURVEY_REL = "outputs/user_survey"            # read through paths.read_path ($CE_OUTPUTS_ROOT)
RECIPE_BODIES = {      # survey folder: (body, its model in the survey's models/, load kwargs, the 52-motion results folder)
    "g1_task52": ("g1", "models/g1/scene.xml", {}, "outputs/g1_52"),
    "heavy_arm_task52": ("heavy_arm", "models/heavy_arm.xml", dict(payload=True), "outputs/body_52/payload"),
    "limit_leg_task52": ("limit_leg", "models/limit_leg.xml", dict(red_leg=True), "outputs/body_52/limit_leg")}
RECIPE_FILES = (("adult", "reference_adult.npz"), ("z0", "z0.npz"), ("zstar", "best.npz"))
RECIPE_DX = {"limit_leg_task52/crawl-0.4-0-d_1": 0.8, "limit_leg_task52/crawl-0.5-2-d_5": 0.8, "heavy_arm_task52/split-0.5_6": 0.75}   # metres between poses; default 0.6
RECIPE_PX_W = 1500


def recipe_layout(cfg, dx):
    """a strobe_figures recipe (strobe: start_step / end_step / interval; camera: type fixed | orbit_auto, azimuth_offset) as a layout"""
    st, cam = cfg["strobe"], cfg["camera"]
    frames = StrobeLayout.frame_range(st["start_step"], st["end_step"], st["interval"])
    if cam["type"] == "fixed":                      # motions that stay in place: world azimuth 140 for every panel (heading 15 deg)
        return StrobeLayout(frames, dx, "fixed"), "fixed 140 deg"
    return StrobeLayout(frames, dx, "orbit", offset=cam["azimuth_offset"]), f"side, path + {cam['azimuth_offset']} deg"


def recipe_panels(scene, folder, clip, cfg_path=None, traj_dir=None, survey=None, dx_table=None):
    """export_strobe_blender.clip_panels: one panel per source (adult reference, z0, LOES's best) of a survey-format clip folder.
    cfg_path / traj_dir: another recipe and clip folder (default: outputs/strobe_task52/configs, the survey's <folder>/trajectories)."""
    import yaml
    from .blender import load_model
    S = Path(survey) if survey else paths.read_path(SURVEY_REL); body, xml, kw, _ = RECIPE_BODIES[folder]; task = f"{body}_{clip}"
    cfg = yaml.safe_load(open(cfg_path or paths.read_path("outputs/strobe_task52") / "configs" / f"{task}.yaml"))
    traj_dir = Path(traj_dir or S / folder / "trajectories" / clip)
    dx = (RECIPE_DX if dx_table is None else dx_table).get(f"{folder}/{clip}", 0.6)
    lay, view = recipe_layout(cfg, dx)
    W, H = strobe_size(clip, lay.n, dx)
    out = {}
    for label, f in RECIPE_FILES:
        m = load_model(S / "models/adult.xml") if label == "adult" else load_model(S / xml, kw.get("payload", False))
        q = np.load(traj_dir / f)["qpos"].astype(np.float64); assert q.shape[1] == m.nq
        red = kw.get("red_leg", False) and label != "adult"; tag = "g1" if (body == "g1" and label != "adult") else "human"
        out[f"{task}_{label}"] = lay.panel(lay.poses(scene, m, q, red, tag), W, H, view, pxw=RECIPE_PX_W)
    return out
