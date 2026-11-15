"""Adult-equivalent observation of a child state, by reverse retargeting instead of a fixed multiplier.

The dataset's child motions are retargeted from the adult ones by copying every joint angle and the root
orientation, scaling the root position by s = 0.611 (adult -> child pelvis-height ratio) and lifting the
root so the lowest geom sits at -5 mm (scripts/qpos_retarget.py). Measured: joint angles and root
orientation identical, root x, y exactly x s. So a child state maps back to the adult skeleton exactly,
except for the root height that the ground lift overwrote:

  qpos_a = qpos_c with root x, y, z / s, then root z shifted so the adult's lowest point sits at the
           child's ground clearance / s
  qvel_a = qvel_c with the root linear velocity / s   (joint and angular velocities are scale-free)

and the observation is humenv's own, computed on the ADULT skeleton in that state. Against the adult's own
motion this reproduces body positions, rotations and angular velocities exactly; root height to ~1 mm
(median); linear velocities to 1-6% (vs 2-4% / 8-15% / 1-11% for the fixed multiplier).

The root z velocity above omits the time derivative of the height correction c = qa[2] - qc[2] / s. Option
clearance_vel=True (default False: the default numbers are unchanged) adds it, by a backward difference over
the control step: va[2] += (c_t - c_{t-1}) / dt, dt = 1/30 s, c_{t-1} the previous call's correction; the
first call after reset() adds 0. ExactObsWrapper keeps one ExactObs per env worker, clears it on reset() and,
on the first step after it, takes c_{t-1} from the state the step starts from (after set_physics: the start
pose), so a rollout's first step is differenced against its start state.
"""
import itertools

import gymnasium

import mujoco
import numpy as np

ADULT_XML = "assets/robots/adult/robot.xml"


class _LowestPoint:
    """Vectorised scripts/qpos_retarget.py:_min_geom_z (exact lowest world-z over non-floor geoms)."""

    def __init__(self, model):
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ids = np.array([g for g in range(model.ngeom) if g != floor])
        t = model.geom_type[ids]
        self.box = ids[t == mujoco.mjtGeom.mjGEOM_BOX]
        self.cap = ids[t == mujoco.mjtGeom.mjGEOM_CAPSULE]
        self.sph = ids[t == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.other = ids[~np.isin(t, [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                       mujoco.mjtGeom.mjGEOM_SPHERE])]
        self.signs = np.array(list(itertools.product([1, -1], repeat=3)), dtype=np.float64)   # (8, 3)
        self.size = model.geom_size

    def __call__(self, data):
        lo = np.inf
        if len(self.box):
            pos, mat = data.geom_xpos[self.box], data.geom_xmat[self.box].reshape(-1, 3, 3)
            corners = pos[:, None] + np.einsum("gij,gcj->gci", mat, self.signs[None] * self.size[self.box][:, None])
            lo = min(lo, corners[..., 2].min())
        if len(self.cap):
            pos, mat = data.geom_xpos[self.cap], data.geom_xmat[self.cap].reshape(-1, 3, 3)
            half, r = self.size[self.cap, 1], self.size[self.cap, 0]
            ax = mat[:, 2, 2] * half                                  # z of the capsule's axis end offset
            lo = min(lo, (pos[:, 2] - np.abs(ax) - r).min())
        if len(self.sph):
            lo = min(lo, (data.geom_xpos[self.sph, 2] - self.size[self.sph, 0]).min())
        if len(self.other):
            lo = min(lo, data.geom_xpos[self.other, 2].min())
        return lo


class ExactObs:
    """obs(qpos_c, qvel_c) -> (358,) humenv proprio of the adult-equivalent state."""

    def __init__(self, child_xml, adult_xml=ADULT_XML, scale=0.611036, clearance_vel=False, dt=1.0 / 30):
        from humenv import make_humenv
        self.mc = mujoco.MjModel.from_xml_path(str(child_xml)); self.dc = mujoco.MjData(self.mc)
        self.ma = mujoco.MjModel.from_xml_path(str(adult_xml)); self.da = mujoco.MjData(self.ma)
        self.env, _ = make_humenv(num_envs=1, task=None, xml=str(adult_xml), state_init="Default")
        self.low_c, self.low_a = _LowestPoint(self.mc), _LowestPoint(self.ma)
        # child / adult root scale of the dataset's retargeting, measured on it (x, y ratio, identical to 1e-16
        # across clips); a different body needs its own (crossembodiment/bodies.py: the root scale's names)
        self.scale = float(scale)
        self.clearance_vel, self.dt = bool(clearance_vel), float(dt)          # option: + d/dt of the height correction
        self.reset()

    @property
    def s(self):
        """read-only alias of scale (the attribute's name before the OOP refactor's step 11)"""
        return self.scale

    def reset(self):
        """forget the previous height correction (clearance_vel only): the next call adds 0"""
        self._prev, self.added = None, 0.0

    def adult_state(self, qpos_c, qvel_c):
        qa = np.array(qpos_c, dtype=np.float64); qa[:3] /= self.scale
        self.dc.qpos[:] = qpos_c; mujoco.mj_kinematics(self.mc, self.dc)
        clearance = self.low_c(self.dc) / self.scale
        self.da.qpos[:] = qa; mujoco.mj_kinematics(self.ma, self.da)
        corr = clearance - self.low_a(self.da)                    # the height correction c = qa[2] - qc[2] / s
        qa[2] += corr
        va = np.array(qvel_c, dtype=np.float64); va[:3] /= self.scale
        if self.clearance_vel:                                    # backward difference, 0 on the first call after reset()
            self.added = 0.0 if self._prev is None else (corr - self._prev) / self.dt
            va[2] += self.added; self._prev = corr
        return qa, va

    def __call__(self, qpos_c, qvel_c):
        qa, va = self.adult_state(qpos_c, qvel_c)
        self.env.unwrapped.set_physics(qpos=qa, qvel=va)
        return self.env.unwrapped.get_obs()["proprio"].astype(np.float32)


class ExactObsWrapper(gymnasium.Wrapper):
    """Adds info["exact_proprio"] = ExactObs(info["qpos"], info["qvel"]) to every step, so in a vector env each
    worker computes its own slot's observation in parallel; same numbers as calling ExactObs on the returned info."""

    def __init__(self, env, child_xml, adult_xml=ADULT_XML, scale=0.611036, clearance_vel=False):
        super().__init__(env)
        self.exact = ExactObs(child_xml, adult_xml, scale, clearance_vel=clearance_vel)

    def reset(self, **kwargs):
        self.exact.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.exact.clearance_vel and self.exact._prev is None:  # first step after reset: c of the state it starts from
            d = self.env.unwrapped.data; self.exact.adult_state(d.qpos.copy(), d.qvel.copy())
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["exact_proprio"] = self.exact(info["qpos"], info["qvel"])
        if self.exact.clearance_vel:
            info["clearance_vel_added"] = self.exact.added
        return obs, reward, terminated, truncated, info
