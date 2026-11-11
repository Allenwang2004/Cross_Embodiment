"""geometry.py -- the small pure functions the scripts each had a copy of: the latent sphere, ranks, quaternions, heading / yaw.

OOP refactor step 3 (docs/refactor/PLAN.md). One implementation per computation, written operation for operation as the copies it
replaces, so every caller gets the identical bits (tests/test_geometry.py compares each with verbatim copies of the old code).
Where two copies use different floating-point arithmetic they stay two named functions, documented below; they are never merged.

  latent sphere   project_z (metamotivo's |z| = sqrt(dim) = 16), sphere16 / sphere16_rows (16 v / |v|), unit_sphere (v / |v|, the
                  latent-sequence searches' unit sphere; their z = 16 * unit), ang_pair (degrees between two latents)
  ranks           rank_normalize (one population), rank_normalize_steps (per step: the latent-sequence searches)
  quaternions     quat_mul (Hamilton product, wxyz)
  heading / yaw   heading_humenv / heading_humenv_one (humenv's SMPL heading of the pelvis quaternion), smpl_heading (the same from
                  qpos, forward kinematics frame by frame), robot_yaw (a robot's root yaw from its rotation matrix),
                  root_yaw / facing (the yaw of the root quaternion in qpos; facing = yaw - 90 deg)

Kept apart on purpose (different arithmetic, so a different last bit):
  * project_z vs sphere16: project_z is a * (r / max(|a|, 1e-12)) with |a| = np.linalg.norm(axis=-1) (a reduction); sphere16 is
    (16 * v) / np.linalg.norm(v), and np.linalg.norm without an axis is sqrt(dot(v, v)) (BLAS). They differ by ~1 ulp.
  * sphere16 vs sphere16_rows: the 1-D norm (dot) vs the row norm (reduction), as above.
  * unit_sphere vs robot_map / g1_map's 3-vector `unit` (v / np.linalg.norm(v), 1-D norm): geometry of frames, not latents; left there.
  * heading_humenv (batched, lso_search / robot_z_search) vs heading_humenv_one (per frame, humenv's remove_base_rot, which only
    works on one quaternion; single_z_search / robot_lso_search / g1_z_search / eval_metrics): the same element-wise arithmetic
    (equal on every input tested), kept as the two call patterns the callers had.
  * losses.root_heading (model/losses.py): kinematics.quat_to_yaw of q * conj(base), a different formula from humenv's calc_heading
    (which rotates the x axis and takes arctan2 of the result); it stays in losses (it is part of tracking_mse).
  * humenv.utils.quat_mul: humenv's 8-multiplication product, a different operation order from quat_mul here.
  * root_yaw vs model/kinematics.quat_to_yaw: the same formula (2 (x y + w z) and 2 (w z + x y) round alike: IEEE addition is
    commutative); quat_to_yaw takes quaternions, root_yaw qpos rows. kinematics.py is left as it is.

Importing this module imports numpy only (mujoco and humenv are imported inside the functions that need them).
"""
from __future__ import annotations

import numpy as np

# conj of the SMPL pelvis base rotation: humenv.utils.remove_base_rot's quat_conjugate(np.array([[0.5, 0.5, 0.5, 0.5]])) (exact values)
SMPL_BASE_CONJ = np.array([[0.5, -0.5, -0.5, -0.5]])


# ------------------------------------------------------------------ the latent sphere

def project_z(z: np.ndarray) -> np.ndarray:
    """metamotivo's project_z: onto the sphere of radius sqrt(dim). Works on a
    single vector or a batch. (scripts/single_z_search.py's, imported from there by the other searches)"""
    a = np.atleast_2d(z)
    r = np.sqrt(a.shape[-1])
    out = a * (r / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12))
    return out.reshape(z.shape)


def sphere16(v):
    """16 * v / |v| of ONE latent (1-D): the analysis scripts' `16 * z / np.linalg.norm(z)` (1-D norm = sqrt(dot(v, v)));
    also = 16 * (v / |v|) bit for bit (16 = 2^4 scales exactly; mse_vs_align_test.py); not bit-equal to project_z (module docstring)"""
    return 16 * v / np.linalg.norm(v)


def sphere16_rows(V):
    """16 * v / |v| per row of a (n, d) batch (the row norm): plot_pair_distances.py's random directions"""
    return 16 * V / np.linalg.norm(V, axis=-1, keepdims=True)


def unit_sphere(x):
    """x / |x| along the last axis: the latent-sequence searches' unit-sphere latents (lso_search.py, robot_lso_search.py `unit`;
    their policy input is 16 * unit)"""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def ang_pair(a, b):
    """degrees between two latents (single_z_search.py's ang_pair / its main()'s ang: path_len_deg, net_deg, start_deg_from_z0)"""
    return float(np.degrees(np.arccos(np.clip(
        a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


# ------------------------------------------------------------------ ranks

def rank_normalize(f: np.ndarray) -> np.ndarray:
    """Centred ranks in [-0.5, 0.5] -- train_es.py:rank_normalize. Strips the
    cost's units and its tail, so one catastrophic rollout sets a direction but
    not the step size. (scripts/single_z_search.py's)"""
    order = np.argsort(np.argsort(f))
    return order / max(len(f) - 1, 1) - 0.5


def rank_normalize_steps(R: np.ndarray) -> np.ndarray:
    """Centred ranks per step: R (n, S) returns-to-go of n samples -> ranks along the samples, / (n - 1) - 0.5 (lso_search.py and
    robot_lso_search.py --rank; n = R's rows = their --n; no guard for n = 1, as there)"""
    return np.argsort(np.argsort(R, axis=0), axis=0) / (len(R) - 1) - 0.5


# ------------------------------------------------------------------ quaternions

def quat_mul(a, b):
    """Hamilton product of (..., 4) wxyz quaternions (model/losses.py _quat_mul = model/phc_reward.py qmul)"""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


# ------------------------------------------------------------------ heading and yaw

def heading_humenv(xquat):
    """humenv's SMPL heading (calc_heading after removing the SMPL base rotation) of pelvis world quaternions, batched:
    (N, 4) wxyz -> (N,) rad. lso_search.py / robot_z_search.py's one quat_mul + calc_heading over every frame."""
    import humenv.utils as hu
    return hu.calc_heading(hu.quat_mul(xquat, np.repeat(SMPL_BASE_CONJ, len(xquat), axis=0)))


def heading_humenv_one(xquat):
    """the same for ONE pelvis quaternion (4,) -> numpy float64, through humenv's own remove_base_rot (which works on one
    quaternion only): single_z_search.py, robot_lso_search.py, g1_z_search.py, eval_metrics.Body.local."""
    import humenv.utils as hu
    return hu.calc_heading(hu.remove_base_rot(np.asarray(xquat)[None].copy(), "smpl"))[0]


def smpl_heading(m, qs, d=None, batched=False):
    """(..., nq) SMPL qpos -> (...) humenv heading of the pelvis (body 1), rad: mj_kinematics frame by frame on (m, d) (d: a
    scratch MjData, new if None), then heading_humenv_one per frame (batched=False: single_z_search.root_heading_xy,
    robot_lso_search / g1_z_search heading_xy) or heading_humenv once over all frames (batched=True: lso_search heading,
    robot_z_search heading_xy)."""
    import mujoco
    d = mujoco.MjData(m) if d is None else d
    flat = qs.reshape(-1, qs.shape[-1])
    if batched:
        Q = np.empty((len(flat), 4))
        for i, qq in enumerate(flat):
            d.qpos[:] = qq; mujoco.mj_kinematics(m, d); Q[i] = d.xquat[1]
        return heading_humenv(Q).reshape(qs.shape[:-1])
    import humenv.utils as hu
    h = np.empty(len(flat))
    for i, qq in enumerate(flat):
        d.qpos[:] = qq; mujoco.mj_kinematics(m, d)
        h[i] = hu.calc_heading(hu.remove_base_rot(d.xquat[1][None].copy(), "smpl"))[0]
    return h.reshape(qs.shape[:-1])


def robot_yaw(xmat):
    """a robot root's yaw from its world rotation matrix (MjData.xmat row, 9 values or 3 x 3): arctan2(R[1, 0], R[0, 0])
    (crossenbodiment.costs.RobotPoints, once g1_points of the archived g1_z_search.py; eval_metrics.Body.local on a robot)"""
    R = np.asarray(xmat).reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def root_yaw(q):
    """(T, nq) qpos -> (T,) yaw of the root quaternion q[:, 3:7] (wxyz) about world z, rad (strobe_final_blender.py yaw)"""
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def facing(q):
    """(T, nq) qpos -> (T,) the direction an SMPL body faces, rad: root_yaw - 90 deg (the SMPL root's x axis points to its left;
    drift_success_child.py, child_topview.py)"""
    return root_yaw(q) - np.pi / 2
