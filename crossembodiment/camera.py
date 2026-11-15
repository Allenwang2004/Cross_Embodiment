"""camera.py -- the views the videos, overlays and Blender panels use, as data (OOP refactor step 10).

ViewSpec(distance, elevation, azimuth, lookat_z=None, follow=None): one free camera, kept in the types the scripts wrote (ints stay
ints: the render specs go through json.dumps).
  .to_mjv(lookat=None)      a free mujoco.MjvCamera with distance / elevation / azimuth (and lookat, if given)
  .to_track()               the "track" dict of a render_panels / replay spec: distance, elevation, azimuth, lookat_z [, follow]
  .to_blender()             the panel keys render_blender.py reads for its camera (elevation, cam_distance); its azimuth is fixed
                            (BLENDER_AZIMUTH: the exporters turn the poses by world azimuth - 125 instead)
  .scaled(k)                distance and lookat_z times k
  ViewSpec.project(renderer, pts, W, H) / project_point(...)   world points -> pixels through a Renderer's current camera
Presets (functions where the scripts scale them):
  survey_track(height)      the survey's side-by-side camera: the child's 2.8 m / 0.55 m for 1.26 m, scaled by
                            k = max(height, adult 1.74 m) / 1.26 and rounded to 2 places (survey_body_videos, survey_robot_videos,
                            lso_videos)
  body_overlay(height)      body_overlay.py as survey_body_videos calls it: 3.1 m / 0.5 m for 1.26 m, "%.2f" of height / 1.26
  robot_overlay(size)       robot_overlay.py's track camera: 3.3 m x size, look-at 0.7 m x size (size = s / 0.8213, the G1's 1)
  BLENDER_STROBE            render_blender.py's camera: azimuth 125, elevation -8, distance 14 (the MuJoCo figures' view)
  SURVEY_AZIMUTH            140: the survey videos' world azimuth, the strobe panels' default "fixed" view
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

BLENDER_AZIMUTH = 125.0          # render_blender.py AZIMUTH (the panel frame's camera azimuth)
SURVEY_AZIMUTH = 140.0           # the survey videos' camera azimuth (world)


@dataclass(frozen=True)
class ViewSpec:
    distance: float
    elevation: float
    azimuth: float
    lookat_z: float | None = None
    follow: str | None = None            # render_panels "track": "first" = every panel follows the first panel's centre of mass

    def to_mjv(self, lookat=None):
        import mujoco
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.distance, c.elevation, c.azimuth = self.distance, self.elevation, self.azimuth
        if lookat is not None:
            c.lookat[:] = lookat
        return c

    def to_track(self):
        d = dict(distance=self.distance, elevation=self.elevation, azimuth=self.azimuth, lookat_z=self.lookat_z)
        if self.follow is not None:
            d["follow"] = self.follow
        return d

    @classmethod
    def from_track(cls, d):
        return cls(d["distance"], d["elevation"], d["azimuth"], d.get("lookat_z"), d.get("follow"))

    def to_blender(self):
        return dict(elevation=self.elevation, cam_distance=self.distance)

    def scaled(self, k):
        return replace(self, distance=self.distance * k, lookat_z=None if self.lookat_z is None else self.lookat_z * k)

    @staticmethod
    def project(renderer, pts, W, H):
        return project(renderer, pts, W, H)

    @staticmethod
    def project_point(renderer, p, W, H):
        return project_point(renderer, p, W, H)


def project(renderer, pts, W, H):
    """world points -> pixel (u, v) through the renderer's current camera (pinhole, symmetric frustum): the formula of
    crossembodiment.draw.project and robot_overlay.py's two copies (main view and --split panels)"""
    c = renderer.scene.camera[0]
    f, up = np.array(c.forward), np.array(c.up); right = np.cross(f, up)
    k = c.frustum_near / c.frustum_top
    out = []
    for p in pts:
        v = np.asarray(p) - np.array(c.pos)
        z = v @ f
        out.append((W / 2 + (v @ right) / z * k * H / 2, H / 2 - (v @ up) / z * k * H / 2))
    return out


def project_point(renderer, p, W, H):
    return project(renderer, [p], W, H)[0]


def survey_track(height, adult_h=1.74):
    """the survey side-by-side camera for a body of `height` m next to the adult (k = max(height, adult) / 1.26)"""
    k = max(height, adult_h) / 1.26
    return ViewSpec(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2))


def body_overlay(height):
    """body_overlay.py's camera as survey_body_videos.py passes it (--distance f"{3.1 * ko:.2f}" --lookat-z f"{0.5 * ko:.2f}")"""
    ko = height / 1.26
    return ViewSpec(distance=float(f"{3.1 * ko:.2f}"), elevation=-12, azimuth=140, lookat_z=float(f"{0.5 * ko:.2f}"))


BODY_OVERLAY_DEFAULT = ViewSpec(distance=3.1, elevation=-12, azimuth=140, lookat_z=0.5)   # body_overlay.py's defaults (the child)


def robot_overlay(size):
    """robot_overlay.py's track camera (framed for the G1; size = the robot's leg ratio s / 0.8213)"""
    return ViewSpec(distance=3.3 * size, elevation=-12, azimuth=140, lookat_z=0.7 * size)


BLENDER_STROBE = ViewSpec(distance=14.0, elevation=-8.0, azimuth=BLENDER_AZIMUTH)
