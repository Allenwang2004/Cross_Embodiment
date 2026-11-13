"""Standalone strobe rendering utilities for qualitative humanoid figures.

This module is deliberately decoupled from the library environment: it wraps any
gymnasium-wrapped ``humenv.env.HumEnv`` and never touches the physics engine or
the base ``env.py``. Two orthogonal concerns live here:

1.  ``StrobeRenderWrapper`` -- a ``gym.Wrapper`` that intercepts the rendering
    pass only. It applies a temporal RGBA gradient to the robot geoms before a
    frame is rendered and restores the model's default geom colors immediately
    afterwards, so no render-time mutation ever leaks into the simulation.

2.  ``CameraController`` -- a clean re-implementation of the orbit logic used in
    ``scripts/other_project/IIP_multiple.py``: a quick look-ahead rollout gives
    the root trajectory (x, y); the motion heading comes from ``np.arctan2``;
    a ``mujoco.MjvCamera`` is then placed on a circle around the lookat point
    at a configurable azimuth relative to the heading.

Ghost compositing (background difference masks, per-pose pixel shifts and the
white figure plate) lives in :func:`composite_strobe`, also configurable and
free of hardcoded paths or coordinates.

Geometry note (MuJoCo convention, verified): with azimuth ``a`` and elevation
``e`` the free camera sits at ``lookat + distance * (-cos a, -sin a, ...)`` and
looks along ``(cos a, sin a)`` in the horizontal plane. With the motion heading
``h`` defined as the direction of travel, ``azimuth = h + azimuth_offset`` puts
the camera at ``azimuth_offset = 0`` behind the path and ``90`` side-on to it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "RobotColorizer",
    "CameraController",
    "StrobeRenderWrapper",
    "StrobeStudio",
    "capture_source",
    "strobe_window",
    "strobe_steps",
    "composite_strobe",
    "configure_white_studio",
    "white_scene_option",
    "stamp_step_index",
    "motion_heading",
]


# ---------------------------------------------------------------------------
# geom bookkeeping
# ---------------------------------------------------------------------------

# Geoms named like these belong to the stage (floor/plane), never the robot.
_STAGE_GEOM_KEYWORDS = ("floor", "plane", "ground", "wall")


def _is_stage_geom(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(k in lowered for k in _STAGE_GEOM_KEYWORDS)


def robot_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    """Indices of every geom that belongs to the humanoid, not the stage."""
    ids = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if not _is_stage_geom(name):
            ids.append(i)
    return np.asarray(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# temporal coloring
# ---------------------------------------------------------------------------

def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


@dataclass
class RobotColorizer:
    """Applies a temporal RGBA gradient to robot geoms and restores defaults.

    The gradient runs ``start_color -> [middle_color] -> final_color`` over the
    captured window; ``alpha`` in [0, 1] selects the blend. Colors are written
    into ``model.geom_rgba`` only inside :meth:`apply`/:meth:`restore`, which
    the wrapper guarantees to run around a single render call.
    """

    model: mujoco.MjModel
    start_color: Sequence[float]
    final_color: Sequence[float]
    middle_color: Sequence[float] | None = None
    _start_rgba: npt.NDArray[np.float64] = field(init=False, repr=False)
    _final_rgba: npt.NDArray[np.float64] = field(init=False, repr=False)
    _middle_rgba: npt.NDArray[np.float64] | None = field(init=False, default=None, repr=False)
    _geom_ids: npt.NDArray[np.int32] = field(init=False, repr=False)
    _default_rgba: npt.NDArray[np.float64] = field(init=False, repr=False)
    _applied: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        arrays = {}
        for key, value in (("start", self.start_color), ("final", self.final_color),
                           ("middle", self.middle_color)):
            if value is None:
                arrays[key] = None
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (4,):
                raise ValueError(f"RGBA colors must have 4 components, got {arr}")
            arrays[key] = arr
        self._start_rgba = arrays["start"]
        self._final_rgba = arrays["final"]
        self._middle_rgba = arrays["middle"]
        self._geom_ids = robot_geom_ids(self.model)
        self._default_rgba = self.model.geom_rgba[self._geom_ids].copy()

    # -- gradient ----------------------------------------------------------
    def color_at(self, alpha: float) -> npt.NDArray[np.float64]:
        """RGBA blend at normalized time ``alpha`` in [0, 1]."""
        t = float(np.clip(alpha, 0.0, 1.0))
        if self._middle_rgba is None:
            return _lerp(self._start_rgba, self._final_rgba, t)
        if t < 0.5:
            return _lerp(self._start_rgba, self._middle_rgba, t / 0.5)
        return _lerp(self._middle_rgba, self._final_rgba, (t - 0.5) / 0.5)

    # -- mutation, strictly scoped ----------------------------------------
    def apply(self, alpha: float) -> None:
        self.model.geom_rgba[self._geom_ids] = self.color_at(alpha)
        self._applied = True

    def restore(self) -> None:
        if self._applied:
            self.model.geom_rgba[self._geom_ids] = self._default_rgba
            self._applied = False

    def hide(self) -> None:
        """Make every robot geom fully transparent (for background frames)."""
        self.model.geom_rgba[self._geom_ids] = (0.0, 0.0, 0.0, 0.0)
        self._applied = True

    @contextmanager
    def temporary(self, alpha: float | None):
        """Scoped color write: ``alpha=None`` renders the robot hidden."""
        self.hide() if alpha is None else self.apply(alpha)
        try:
            yield
        finally:
            self.restore()


# ---------------------------------------------------------------------------
# camera orbit
# ---------------------------------------------------------------------------

def motion_heading(traj_xy: np.ndarray, look_ahead: int = 30,
                   min_displacement: float = 1e-4) -> float:
    """Heading (rad) of the root path around index 0, via ``np.arctan2``.

    A forward look-ahead window is used: the heading at a frame is estimated
    from the displacement between the trajectory start and the point up to
    ``look_ahead`` steps later (falling back to the global end point when the
    window is short). A path that barely moves yields heading 0.
    """
    traj = np.asarray(traj_xy, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 2 or len(traj) < 2:
        return 0.0
    end = traj[min(look_ahead, len(traj) - 1)]
    diff = end - traj[0]
    if float(np.hypot(diff[0], diff[1])) < min_displacement:
        diff = traj[-1] - traj[0]
        if float(np.hypot(diff[0], diff[1])) < min_displacement:
            return 0.0
    return float(np.arctan2(diff[1], diff[0]))


def motion_headings(traj_xy: np.ndarray, look_ahead: int = 30) -> np.ndarray:
    """Per-frame heading from a rolling look-ahead window along the trajectory."""
    traj = np.asarray(traj_xy, dtype=np.float64)
    out = np.zeros(len(traj), dtype=np.float64)
    for t in range(len(traj)):
        out[t] = motion_heading(traj[max(0, t - 1):t + look_ahead + 1],
                                look_ahead=look_ahead)
    return out


@dataclass
class CameraController:
    """Places a ``mujoco.MjvCamera`` relative to a motion path.

    ``type``:
      * ``orbit_auto`` -- orbit around the path centroid at
        ``heading + azimuth_offset``, recomputed per frame from a rolling
        look-ahead window (the IIP orbit, cleaned up).
      * ``tracking``   -- free camera that follows the root, keeping the same
        distance/elevation but a fixed azimuth.
      * ``fixed``      -- free camera locked to the first frame's placement.
    """

    type: str = "orbit_auto"
    distance: float = 4.5
    elevation: float = -12.0
    azimuth_offset: float = 90.0
    lookat_height: float = 0.9
    track_body: str | None = None
    look_ahead: int = 30
    plan_window: str = "strobe"
    _planned_lookat: npt.NDArray[np.float64] | None = field(init=False, default=None, repr=False)
    _planned_azimuth: npt.NDArray[np.float64] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.type not in ("orbit_auto", "tracking", "fixed"):
            raise ValueError(f"unknown camera type {self.type!r}")
        if self.plan_window not in ("strobe", "full"):
            raise ValueError(f"unknown plan_window {self.plan_window!r} "
                             "(expected 'strobe' or 'full')")

    # -- planning ----------------------------------------------------------
    def plan_for_trajectory(self, traj_xy: np.ndarray, start: int,
                            end: int) -> None:
        """Plan from a root trajectory, honouring :attr:`plan_window`.

        ``strobe`` (default) plans around the captured window only; ``full``
        plans around the whole clip.
        """
        self.plan(traj_xy,
                  window=None if self.plan_window == "full" else (start, end))
    def plan(self, traj_xy: np.ndarray,
             window: tuple[int, int] | None = None) -> None:
        """Precompute per-frame (lookat_xy, azimuth) from a root trajectory.

        ``window`` is the strobe window ``(start, end)``, inclusive. For
        ``orbit_auto`` and ``fixed`` the centroid and heading come from that
        slice only: a long clip must not pull the camera to the middle of a
        path whose strobed part sits at one end of it (e.g. a 300-frame,
        47 m ``move-ego`` clip with ``end_step: 70`` would otherwise look at
        ground 20 m past the poses, pushing the trail out of frame).

        * ``orbit_auto`` -- ONE placement for the whole clip: the camera sits on
          a circle around the path centroid at ``global_heading +
          azimuth_offset`` and stays there, so all ghost poses share one
          viewpoint (the IIP orbit, cleaned up).
        * ``tracking`` -- per-frame placement that follows the root with a
          rolling look-ahead heading.
        * ``fixed`` -- one placement at the window start, heading 0.
        """
        traj = np.asarray(traj_xy, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(f"traj_xy must be (T, 2), got {traj.shape}")
        n = len(traj)
        lo, hi = 0, n - 1
        if window is not None:
            lo = int(np.clip(window[0], 0, n - 1))
            hi = int(np.clip(window[1], lo, n - 1))
        part = traj[lo:hi + 1]
        if self.type == "orbit_auto":
            heading = motion_heading(part, look_ahead=self.look_ahead)
            lookat, azimuth = self._placement(part.mean(axis=0), heading)
            self._planned_lookat = np.repeat(lookat[None, :2], n, axis=0)
            self._planned_azimuth = np.full(n, azimuth)
        elif self.type == "tracking":
            headings = motion_headings(traj, look_ahead=self.look_ahead)
            lookats, azimuths = [], []
            for t in range(n):
                lookat, azimuth = self._placement(traj[t], headings[t])
                lookats.append(lookat[:2])
                azimuths.append(azimuth)
            self._planned_lookat = np.asarray(lookats)
            self._planned_azimuth = np.asarray(azimuths)
        else:  # fixed
            lookat, azimuth = self._placement(traj[lo], 0.0)
            self._planned_lookat = np.repeat(lookat[None, :2], n, axis=0)
            self._planned_azimuth = np.full(n, azimuth)

    def _placement(self, center_xy: np.ndarray, heading: float):
        """(lookat_xyz, azimuth_deg) for a lookat point and a motion heading."""
        azimuth = math.degrees(heading) + self.azimuth_offset
        lookat = np.array([center_xy[0], center_xy[1], self.lookat_height],
                          dtype=np.float64)
        return lookat, azimuth

    def root_xy(self, data: mujoco.MjData, model: mujoco.MjModel) -> np.ndarray:
        if self.track_body:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.track_body)
            if bid < 0:
                raise ValueError(f"track_body {self.track_body!r} not in model")
            return np.asarray(data.xpos[bid][:2], dtype=np.float64)
        return np.asarray(data.qpos[:2], dtype=np.float64)

    # -- per-frame camera --------------------------------------------------
    def camera_at(self, step: int, data: mujoco.MjData,
                  model: mujoco.MjModel, heading: float | None = None
                  ) -> mujoco.MjvCamera:
        """The ``MjvCamera`` for frame ``step``.

        Uses the planned placement when :meth:`plan` has run; otherwise falls
        back to a placement around the current root position.
        """
        planned_lookat, planned_azimuth = self._planned_lookat, self._planned_azimuth
        if planned_lookat is not None and planned_azimuth is not None:
            t = int(np.clip(step, 0, len(planned_azimuth) - 1))
            return self._build_cam(planned_lookat[t], planned_azimuth[t])
        lookat_xy = self.root_xy(data, model)
        return self._build_cam(lookat_xy,
                               math.degrees(heading or 0.0) + self.azimuth_offset)

    def _build_cam(self, lookat_xy: np.ndarray, azimuth_deg: float) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = self.distance
        cam.elevation = self.elevation
        cam.azimuth = azimuth_deg
        cam.lookat[:] = [lookat_xy[0], lookat_xy[1], self.lookat_height]
        return cam


# ---------------------------------------------------------------------------
# gym wrapper
# ---------------------------------------------------------------------------

class StrobeRenderWrapper(gym.Wrapper):
    """Intercepts ``render()`` to color robot geoms and orbit the camera.

    Physics is never altered: the wrapper only mutates ``model.geom_rgba`` (and
    only between the moment a frame is requested and the moment it is handed
    back) and selects a camera object for the scene update. Default geom colors
    are restored immediately after every render.

    Usage::

        wrapped = StrobeRenderWrapper(env, start_color=..., final_color=...,
                                      camera={"type": "orbit_auto", ...})
        wrapped.set_window(start_step=0, end_step=119)
        frame = wrapped.render()          # colored + camera-orbited frame
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env: gym.Env,
        start_color: Sequence[float] = (0.98, 0.55, 0.25, 1.0),
        final_color: Sequence[float] = (0.16, 0.30, 0.65, 1.0),
        middle_color: Sequence[float] | None = None,
        camera: dict[str, Any] | CameraController | None = None,
        white_background: bool = True,
        shadow: bool = False,
    ) -> None:
        super().__init__(env)
        model = self.unwrapped.model
        self.colorizer = RobotColorizer(
            model, start_color=start_color, final_color=final_color,
            middle_color=middle_color,
        )
        if isinstance(camera, CameraController):
            self.camera_ctl = camera
        else:
            self.camera_ctl = CameraController(**(camera or {}))
        self._start_step = 0
        self._end_step = 100
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._scene_option = (
            white_scene_option(shadow=shadow) if white_background else mujoco.MjvOption()
        )
        if white_background:
            configure_white_studio(model)

    # -- window ------------------------------------------------------------
    def set_window(self, start_step: int, end_step: int) -> None:
        """Define the temporal window the RGBA gradient spans."""
        if end_step < start_step:
            raise ValueError("end_step must be >= start_step")
        self._start_step = int(start_step)
        self._end_step = int(end_step)

    def plan_camera(self, traj_xy: np.ndarray,
                    window: tuple[int, int] | None = None) -> None:
        """Feed the look-ahead root trajectory to the camera controller.

        ``window`` restricts the orbit planning to the strobed frames.
        """
        self.camera_ctl.plan(traj_xy, window=window)

    def current_alpha(self) -> float:
        span = max(1, self._end_step - self._start_step)
        return (self._step_count - self._start_step) / span

    def set_step(self, step: int) -> None:
        self._step_count = int(step)

    # -- interception ------------------------------------------------------
    def set_qpos(self, qpos: npt.ArrayLike) -> None:
        """Pose the robot kinematically (zero velocity), no physics step."""
        base = self.unwrapped
        base.set_physics(qpos=np.asarray(qpos, dtype=np.float64),
                         qvel=np.zeros(base.model.nv))

    def step(self, action):
        self._step_count += 1
        return self.env.step(action)

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)

    def render(self):
        """One gradient-colored frame from the current camera placement."""
        return self._render_frame(self.current_alpha())

    def render_background(self) -> npt.NDArray[np.uint8]:
        """Robot-free frame from the current camera, for difference masks."""
        return self._render_frame(None)

    def _render_frame(self, alpha: float | None) -> npt.NDArray[np.uint8]:
        """Render one frame through our OWN offscreen renderer.

        Using a private ``mujoco.Renderer`` (instead of ``env.render()``) keeps
        the library's base ``env.py`` untouched while still letting us pick the
        camera object and the scene options per frame.
        """
        base = self.unwrapped
        renderer = self._renderer
        if renderer is None:
            renderer = mujoco.Renderer(
                base.model,
                width=int(getattr(base, "render_width", 640)),
                height=int(getattr(base, "render_height", 480)),
            )
            self._renderer = renderer
            mujoco.mj_forward(base.model, base.data)
        with self.colorizer.temporary(alpha):
            cam = self.camera_ctl.camera_at(self._step_count, base.data, base.model)
            renderer.update_scene(
                base.data, camera=cam, scene_option=self._scene_option)
            return renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()


class StrobeStudio:
    """Kinematic strobe renderer over a bare MuJoCo model: no gym, no physics.

    Same rendering surface as :class:`StrobeRenderWrapper` (``set_window`` /
    ``set_step`` / ``set_qpos`` / ``render`` / ``render_background`` /
    ``plan_camera`` / ``close``), so :func:`capture_source` drives either one.
    Frames come from ``data.qpos`` + ``mj_forward`` only, which is what a
    recorded trajectory needs: nothing is integrated and no environment object
    has to exist. Use it for shared trajectory dumps whose model XML is not a
    humenv task model.
    """

    def __init__(
        self,
        xml_path: str,
        width: int = 1280,
        height: int = 720,
        start_color: Sequence[float] = (0.98, 0.55, 0.25, 1.0),
        final_color: Sequence[float] = (0.16, 0.30, 0.65, 1.0),
        middle_color: Sequence[float] | None = None,
        camera: dict[str, Any] | CameraController | None = None,
        white_background: bool = True,
        shadow: bool = False,
    ) -> None:
        self.xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.colorizer = RobotColorizer(
            self.model, start_color=start_color, final_color=final_color,
            middle_color=middle_color,
        )
        self.camera_ctl = (camera if isinstance(camera, CameraController)
                           else CameraController(**(camera or {})))
        self._scene_option = (
            white_scene_option(shadow=shadow) if white_background
            else mujoco.MjvOption())
        if white_background:
            configure_white_studio(self.model)
        self._renderer = mujoco.Renderer(self.model, width=int(width),
                                         height=int(height))
        self._start_step = 0
        self._end_step = 100
        self._step_count = 0

    def __enter__(self) -> StrobeStudio:
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    def set_window(self, start_step: int, end_step: int) -> None:
        """Define the temporal window the RGBA gradient spans."""
        if end_step < start_step:
            raise ValueError("end_step must be >= start_step")
        self._start_step = int(start_step)
        self._end_step = int(end_step)

    def set_step(self, step: int) -> None:
        self._step_count = int(step)

    def current_alpha(self) -> float:
        span = max(1, self._end_step - self._start_step)
        return (self._step_count - self._start_step) / span

    def plan_camera(self, traj_xy: np.ndarray,
                    window: tuple[int, int] | None = None) -> None:
        self.camera_ctl.plan(traj_xy, window=window)

    def set_qpos(self, qpos: npt.ArrayLike) -> None:
        row = np.asarray(qpos, dtype=np.float64)
        if row.shape[0] != self.model.nq:
            raise ValueError(
                f"{self.xml_path}: trajectory has nq={row.shape[0]} but the "
                f"model has nq={self.model.nq} -- wrong model for this dump?")
        self.data.qpos[:] = row
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> npt.NDArray[np.uint8]:
        return self._render_frame(self.current_alpha())

    def render_background(self) -> npt.NDArray[np.uint8]:
        return self._render_frame(None)

    def _render_frame(self, alpha: float | None) -> npt.NDArray[np.uint8]:
        with self.colorizer.temporary(alpha):
            cam = self.camera_ctl.camera_at(self._step_count, self.data,
                                            self.model)
            self._renderer.update_scene(self.data, camera=cam,
                                        scene_option=self._scene_option)
            return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


# ---------------------------------------------------------------------------
# studio + composite
# ---------------------------------------------------------------------------

def white_scene_option(shadow: bool = False) -> mujoco.MjvOption:
    """Scene flags for a clean studio: no skybox, fog, haze or reflections.

    ``mujoco.MjvOption`` in 3.11 has no ``flagsmask``: the ``flags`` array is
    the option's own full flag set, so writing 0 into the entries we care about
    is what takes effect (verified by rendering: the checker stage comes back
    white and the skybox disappears).
    """
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    opt.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    opt.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    if not shadow:
        opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return opt


def configure_white_studio(model: mujoco.MjModel) -> None:
    """Neutral white studio lighting and stage materials.

    Scene *flags* live in :func:`white_scene_option` (they belong to the
    renderer, not the model); what lives on the model is the light and the
    stage geometry.
    """
    vis = model.vis
    vis.rgba.fog[:] = [1.0, 1.0, 1.0, 1.0]
    vis.rgba.haze[:] = [1.0, 1.0, 1.0, 1.0]
    vis.headlight.active = 1
    vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
    vis.headlight.specular[:] = [0.15, 0.15, 0.15]
    # Stage geoms become an unlit white plate. Emission is a *material*
    # property in MuJoCo (mjModel.mat_emission), so the stage keeps a material
    # slot -- but one that no robot geom shares -- with emission 1 and
    # specular/reflectance 0. That makes the plate independent of lighting:
    # otherwise the robot occluding the headlight darkens the floor beneath it,
    # the difference mask picks that shading up, and grey ghost blobs appear
    # under every pose in the composite.
    stage = [i for i in range(model.ngeom)
             if _is_stage_geom(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))]
    robot_mats = {int(model.geom_matid[i]) for i in range(model.ngeom)} - set(stage)
    free_mats = [m for m in range(model.nmat) if m not in robot_mats]
    for slot, i in enumerate(stage):
        model.geom_rgba[i] = [1.0, 1.0, 1.0, 1.0]
        mat = int(model.geom_matid[i])
        if mat in robot_mats:
            mat = free_mats[slot] if slot < len(free_mats) else -1
        if mat < 0:
            model.geom_matid[i] = -1      # no material available: plain white
            continue
        model.geom_matid[i] = mat
        model.mat_rgba[mat] = [1.0, 1.0, 1.0, 1.0]
        model.mat_emission[mat] = 1.0
        model.mat_specular[mat] = 0.0
        model.mat_reflectance[mat] = 0.0
        model.mat_shininess[mat] = 0.0
        model.mat_texid[mat] = -1         # drop floor textures (checkerboard)


def _load_font(size: int):
    """Bold legible font, with a dependency-free fallback."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def stamp_step_index(
    frame: npt.NDArray[np.uint8],
    step: int,
    label: str | None = None,
    highlight: bool = False,
    font_size: int | None = None,
) -> npt.NDArray[np.uint8]:
    """Badge a rendered frame with its own step index (top-left corner).

    Lets you read frame numbers straight off the comparison video and then
    feed them back as ``strobe.start_step`` / ``end_step`` / ``interval``.
    ``highlight`` paints the badge green for steps the strobe window actually
    captures, so the chosen window and interval are visible in the footage.
    The badge is only applied to video frames; strobe PNGs stay clean.
    """
    arr = np.asarray(frame)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    text = f"step {step}" if label is None else f"{label}  \u00b7  step {step}"
    size = int(font_size or max(15, arr.shape[0] // 28))
    font = _load_font(size)
    pad = max(4, size // 4)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    margin, box_pad = 8, pad
    x0, y0 = margin, margin
    x1, y1 = x0 + text_w + 2 * box_pad, y0 + text_h + 2 * box_pad
    draw.rectangle([x0, y0, x1, y1], fill=(34, 120, 62) if highlight else (18, 18, 18))
    draw.text((x0 + box_pad, y0 + box_pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def strobe_window(strobe_cfg: dict, total: int) -> tuple[int, int]:
    """Inclusive (start, end) steps of the strobe window for ``total`` frames.

    ``end_step: null`` runs through the last frame of *this* source, so sources
    of different lengths (adult clips run longer than child rollouts) each get
    their full window.
    """
    start = int(strobe_cfg.get("start_step") or 0)
    end_raw = strobe_cfg.get("end_step")
    return start, (total - 1 if end_raw is None else min(int(end_raw), total - 1))


def strobe_steps(strobe_cfg: dict, total: int) -> list[int]:
    """Which steps become ghost poses: ``target_frames`` if given, else the
    interval ladder, always ending on the window's last step."""
    start, end = strobe_window(strobe_cfg, total)
    targets = strobe_cfg.get("target_frames")
    if targets:
        return [int(t) for t in targets if start <= int(t) <= end]
    interval = max(1, int(strobe_cfg.get("interval", 15)))
    steps = list(range(start, end + 1, interval))
    if not steps or steps[-1] != end:
        steps.append(end)
    return steps


def capture_source(recorder: Any, qpos: np.ndarray, strobe_cfg: dict,
                   label: str | None = None,
                   burn_in: bool = False) -> tuple[list, list, np.ndarray, list[int]]:
    """Replay one trajectory through ``recorder``; collect video + strobe frames.

    ``recorder`` is anything exposing ``set_window`` / ``set_step`` / ``set_qpos``
    / ``render`` / ``render_background`` -- :class:`StrobeRenderWrapper` for a
    live humenv model, :class:`StrobeStudio` for a bare MuJoCo XML. Returns
    ``(video_frames, strobe_frames, background, steps)``.

    Ghost poses come from the un-stamped renders: the burn-in badge is a viewing
    aid for the video only, never part of the published figure.
    """
    total = len(qpos)
    recorder.set_window(*strobe_window(strobe_cfg, total))
    steps = strobe_steps(strobe_cfg, total)

    raw_frames = []
    for t in range(total):
        recorder.set_qpos(qpos[t])
        recorder.set_step(t)
        raw_frames.append(recorder.render())

    strobe_frames = [raw_frames[t] for t in steps]
    if burn_in:
        captured = set(steps)
        video_frames = [stamp_step_index(f, t, label, t in captured)
                        for t, f in enumerate(raw_frames)]
    else:
        video_frames = raw_frames

    recorder.set_qpos(qpos[steps[0]])
    recorder.set_step(steps[0])
    background = recorder.render_background()
    return video_frames, strobe_frames, background, steps


def _shift_columns(arr: np.ndarray, shift: int) -> np.ndarray:
    """Shift ``arr`` horizontally by ``shift`` px, clipping at the plate edge.

    ``numpy.roll`` wraps, so a ghost pushed past the plate reappears on the far
    side and reads as a rendering bug. Clipping drops the part that leaves.
    """
    out = np.zeros_like(arr)
    w = arr.shape[1]
    if shift >= 0:
        if shift < w:
            out[:, shift:] = arr[:, :w - shift]
    elif -shift < w:
        out[:, :w + shift] = arr[:, -shift:]
    return out


def composite_strobe(
    frames: Iterable[np.ndarray],
    background: np.ndarray,
    threshold: float = 5.0,
    shift_px: int = 14,
    x_offset_px: int = 0,
    crop_floor: bool = True,
    crop_floor_frac: float = 0.12,
    plate_color: tuple[float, float, float] = (255.0, 255.0, 255.0),
) -> np.ndarray:
    """Overlay ghost poses on a white plate, shifted along the path.

    Each frame is differenced against a background frame (the same scene with
    the robot hidden); pixels whose summed absolute difference exceeds
    ``threshold`` form the foreground mask. Ghost ``k`` is rolled horizontally
    by ``start + k * shift_px`` and stamped onto the composite only where its
    mask is set, so later poses occlude earlier ones.

    ``x_offset_px`` slides the **whole trail** without touching the spacing.
    ``0`` (the default) is the centred layout the reference generator produced:
    the trail's midpoint sits on the camera axis, i.e. ghost ``k`` is rolled by
    ``-((n-1) * shift_px)//2 + x_offset_px + k * shift_px``. Positive moves the
    trail right, negative left; it is a displacement from centre, not a pixel
    coordinate.

    ``crop_floor`` whitens the bottom ``crop_floor_frac`` of the plate after
    stamping, dropping ground-plane clutter (checker textures, contact
    shadows) for a clean figure on white.

    A ghost that would leave the plate is clipped at the edge and reported as a
    warning: an over-wide trail never aborts a render. The default offset ``0``
    is instead clamped to the nearest feasible offset and reported.
    """
    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("composite_strobe needs at least one frame")
    bg = np.asarray(background, dtype=np.float64)
    h, w, _ = frames[0].shape
    plate = np.full((h, w, 3), plate_color, dtype=np.float64)

    n = len(frames)
    # 0 == the trail's midpoint on the camera axis (the centred layout); the
    # offset is a displacement from that centre.
    center_start = -((n - 1) * shift_px) // 2
    start_offset = center_start + int(x_offset_px)
    spans = []
    for k, frame in enumerate(frames):
        diff = np.abs(frame.astype(np.float64) - bg).sum(axis=-1)
        mask = diff > threshold
        shift = start_offset + k * shift_px
        cols = np.flatnonzero(mask.any(axis=0))
        if cols.size:
            spans.append((int(cols[0]), int(cols[-1])))
    if spans:
        # Every pose must stay on the plate: with the first pose at base, pose
        # k sits at base + k*shift_px, so base is bounded by [-leftmost,
        # width-1-rightmost-lastshift]. Reported in offset units (relative to
        # the centred layout) so the numbers are directly usable.
        lo = max(-left for left, _ in spans) - center_start
        hi = min(w - 1 - right - k * shift_px
                 for k, (_, right) in enumerate(spans)) - center_start
        offset = int(x_offset_px)
        span_px = max(right - left for left, right in spans) + 1
        trail_px = (len(spans) - 1) * shift_px
        if lo > hi:
            print(f"  warning: {len(spans)} poses at shift_px={shift_px} span "
                  f"{span_px}px + a {trail_px}px trail, wider than the {w}px "
                  f"plate (no feasible x_offset_px). Keeping "
                  f"x_offset_px={offset} and clipping the ghosts that fall off "
                  f"the edge: reduce shift_px, poses, or widen render.width")
        elif not lo <= offset <= hi:
            if offset == 0:
                # The default placement must never abort a render: clamp to the
                # nearest feasible offset and report the move.
                offset = lo if offset < lo else hi
                start_offset = center_start + offset
                print(f"  x_offset_px=0 does not fit a {span_px}px figure with a "
                      f"{trail_px}px trail on a {w}px plate: "
                      f"using x_offset_px={offset} (feasible [{lo}, {hi}])")
            else:
                print(f"  warning: x_offset_px={offset} pushes a ghost off the "
                      f"{w}px plate (feasible [{lo}, {hi}], figure spans "
                      f"{span_px}px, trail spans {trail_px}px); rendering as "
                      f"asked and clipping at the edge")
    for k, frame in enumerate(frames):
        mask = np.abs(frame.astype(np.float64) - bg).sum(axis=-1) > threshold
        shift = start_offset + k * shift_px
        shifted_mask = _shift_columns(mask, shift)
        shifted_frame = _shift_columns(frame.astype(np.float64), shift)
        plate[shifted_mask] = shifted_frame[shifted_mask]

    if crop_floor and crop_floor_frac > 0.0:
        cut = h - int(h * crop_floor_frac)
        plate[cut:, :] = plate_color
    return np.clip(plate, 0, 255).astype(np.uint8)
