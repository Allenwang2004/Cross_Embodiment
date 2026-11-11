"""metrics.py -- the evaluation metrics of saved rollouts as named pure functions, and MetricSuite (one clip's reference, many rollouts).

OOP refactor step 3 (docs/refactor/PLAN.md). Every function is written operation for operation as the copies it replaces (the
stats() / world() copies of the Task-page drivers, child_52_means.full / hand_fall, eval_metrics.metrics), which now call these;
tests/golden/metrics_golden.py and tests/test_metrics.py prove the numbers are the same. Conventions (as on the Task page):
q (T, nq) is a rollout's qpos and r the reference frames it is scored against (pass ref[1:]: the state after step t against
reference frame t + 1), both float64 and cut to the shorter; 30 frames per second.

Task-page definitions (since 11/08)
  contact_fall_time   Fall Time: seconds with a non-foot body (not *ankle* / *toe*) within 1 cm of the floor while the reference's
                      same body is more than 5 cm above it (fall_definitions.py "contact"), from eval_metrics.Body.part_floor
  phc_success         PHC's success test (Luo et al. 2023): the mean over bodies of the world-frame body position error never above
                      0.5 m adult-equivalent, i.e. threshold 0.5 x s on a body of root scale s (0.31 m on the child)
  drift               mean pelvis x, y distance to the reference: unit "body" (the body's own metres) or "adult" (/ s)
  heading_error       mean |facing - the reference's facing|, degrees (facing = root yaw - 90 deg; drift_success_child.py)
  MPJPE, MMPJPE (cm) and MPJAE (m/s^2: the per-joint ACCELERATION error, not an angle) stay in scripts/eval_metrics.py
  tracking_metrics; the suite calls it.
Legacy definitions (the Task page until 11/08; outputs/child_52*/per_run.npz, eval_metrics.metrics, kpkd means.json)
  root_success        the root never more than 1.5 m (horizontal) or 45 deg (geodesic rotation) from the reference
  height_fall         seconds with the root below half of the reference's root height
  hand_fall           seconds with a hand within 1 cm of the floor (user, 11/07): by motion (NaN where the reference's hands are
                      within 5 cm of the floor for 0.5 s or more) and by frame (only where the reference's hands are > 5 cm up)
  legacy_metrics, body_hand_fall, LEGACY_KEYS   child_52_means.full / hand_fall / KEYS (moved here in step 4)
Other: pose_similarity (mse_vs_align_test.metrics, step 4): joint-angle, local / world MPJPE and heading similarity.

MetricSuite: an ordered list of named metric functions + one clip's reference (its world body positions and floor distances,
computed once). MetricSuite.task_page(body, s) gives the Task page's (fall_s, success, mpjpe, mmpjpe, mpjae, drift):

    suite = MetricSuite.task_page(body, s).reference(ref[1:])          # Pw, Dr computed here (or pass the caller's)
    suite.values(q)  -> [fall_s, success, mpjpe, mmpjpe, mpjae, drift]   suite(q) -> the same as a dict

`body` is a scripts/eval_metrics.py Body (m, d, part_floor, local, hand_floor); metric_body(xml, smpl) is one per (xml, smpl) and
process. Importing this module imports numpy only.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

from . import paths

FPS = 30
TOUCH, NEAR = 0.01, 0.05                  # contact: within 1 cm of the floor; clear: the reference more than 5 cm above it
PHC_THRESHOLD = 0.5                       # PHC's success threshold, m adult-equivalent (x s on the body)
ROOT_POS, ROOT_ROT = 1.5, 45              # legacy root_success: m, deg


def _eval_metrics():
    """scripts/eval_metrics.py as the scripts import it (module name eval_metrics: one module object, one Body class)"""
    if "eval_metrics" not in sys.modules:
        p = str(paths.REPO / "scripts")
        if p not in sys.path:
            sys.path.insert(0, p)
    import eval_metrics
    return eval_metrics


@lru_cache(maxsize=None)
def metric_body(xml, smpl):
    """eval_metrics.Body(xml, smpl), one per (xml spelling, smpl) and process: the drivers' worker caches (`body = body or
    E.Body(...)`, _ctx / _b / _B dicts) go through this. Its MjData is scratch space (every use sets qpos first)."""
    return _eval_metrics().Body(xml, smpl)


# ------------------------------------------------------------------ positions

def world_positions(m, d, Q):
    """(T, nq) qpos -> (T, nbody - 1, 3) world positions of every body but the world (mj_kinematics frame by frame on m, d):
    the drivers' world() / xpos() copies"""
    import mujoco
    P = np.empty((len(Q), m.nbody - 1, 3))
    for t, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); P[t] = d.xpos[1:]
    return P


def mean_body_error(P, Pr):
    """(T, B, 3) positions against the reference's (T, B, 3) -> (T,) the mean over bodies of |p - p_ref| per frame"""
    return np.linalg.norm(P - Pr, axis=-1).mean(1)


# ------------------------------------------------------------------ Task-page definitions

def contact_fall_time(D, Dr, touch=TOUCH, near=NEAR, fps=FPS):
    """Fall Time (s): frames where some non-foot body is within `touch` of the floor (D <= touch) while the reference's same body
    is more than `near` above it (Dr > near), / fps. D, Dr (T, P): eval_metrics.Body.part_floor of the rollout and the
    reference, cut to the same T."""
    return float(((D <= touch) & (Dr > near)).any(1).sum() / fps)


def phc_success(err, s=1.0, threshold=PHC_THRESHOLD):
    """PHC's success test: 1.0 if the per-frame mean body error `err` (T,) never exceeds threshold x s (0.5 m adult-equivalent on
    a body of root scale s), else 0.0. (drift_success_child.py wrote it as err.max() <= 0.5 s: the same value for every
    finite, non-empty err.)"""
    return float(not (err > threshold * s).any())


def drift(q, r, unit="body", s=None):
    """mean pelvis x, y distance to the reference: unit "body" = the body's own metres (loes_noanchor, body_52_test,
    g1_small_test, loes_weight_test, lsor_mpjae_test), "adult" = / s, adult-equivalent metres (drift_success_child,
    tall_body_52, amass_baselines)"""
    x = np.linalg.norm(q[:, :2] - r[:, :2], axis=1).mean()
    if unit == "body":
        return float(x)
    if unit == "adult":
        return float(x / s)
    raise ValueError(f"drift unit {unit!r}: 'body' or 'adult'")


def heading_error(q, r):
    """mean |facing(q) - facing(r)| over frames, degrees, the difference wrapped to [-180, 180] (drift_success_child.py)"""
    from .geometry import facing
    return float(np.degrees(np.abs(np.angle(np.exp(1j * (facing(q) - facing(r)))))).mean())


# ------------------------------------------------------------------ legacy definitions

def root_fail(q, qr):
    """per frame, is the root more than 1.5 m (x, y) or 45 deg (geodesic rotation) from the reference: (bad (T,), pos (T,) m)"""
    pos = np.linalg.norm(q[:, :2] - qr[:, :2], axis=1); rot = np.degrees(2 * np.arccos(np.abs((q[:, 3:7] * qr[:, 3:7]).sum(1)).clip(0, 1)))
    return (pos > ROOT_POS) | (rot > ROOT_ROT), pos


def root_success(q, qr):
    """legacy success (bool): root_fail never triggers"""
    return not root_fail(q, qr)[0].any()


def height_fall(q, qr, fps=FPS):
    """legacy Fall Time (s): frames with the root below half of the reference's root height, / fps"""
    return float((q[:, 2] < 0.5 * qr[:, 2]).sum() / fps)


def hand_fall(h, hr, touch=TOUCH, near=NEAR, motion_s=0.5, fps=FPS):
    """legacy hand Fall Time from the hands' floor distances h, hr (T,) (eval_metrics.Body.hand_floor of rollout and reference):
    (by_motion, by_frame) seconds with a hand within `touch` of the floor; by_motion is NaN for a motion whose reference has its
    hands within `near` for `motion_s` seconds or more; by_frame counts only frames where the reference's hands are > `near` up"""
    touched = h <= touch
    hand_motion = (hr <= near).sum() / fps >= motion_s
    return (float("nan") if hand_motion else float(touched.sum() / fps)), float((touched & (hr > near)).sum() / fps)


# the legacy rollout metrics of Comparison A and its successors (scripts/child_52_means.py full / KEYS / hand_fall until refactor
# step 4; child_52_means.py itself is archived): child_52_rerun, child_52_cmaes, child_52_cem_phc, child_52_lso and kpkd_test
# import these
LEGACY_KEYS = ("ratio", "fall_s", "success", "mpjpe", "mmpjpe", "mpjae")      # child_52_means.KEYS: per_run.npz's last axis


def legacy_metrics(body, q, qr, ratio):
    """child_52_means.full: eval_metrics.tracking_metrics + height_fall (fall_s) + root_success (1.5 m / 45 deg) and the cost
    ratio passed in, as a dict (tracking_metrics' keys first, then ratio, fall_s, success); q[t] is compared with qr[t]"""
    T = min(len(q), len(qr))
    return dict(_eval_metrics().tracking_metrics(body, q, qr), ratio=ratio, fall_s=height_fall(q[:T], qr[:T]),
                success=float(root_success(q[:T], qr[:T])))


def body_hand_fall(body, q, qr):
    """child_52_means.hand_fall: hand_fall of an eval_metrics.Body's hand floor distances (user, 11/07: touch 1 cm, the reference's
    hands within 5 cm for 0.5 s or more = a motion meant to touch the floor). Returns (by_motion, by_frame); q[t] is compared with
    qr[t] (pass qr = ref[1:])."""
    T = min(len(q), len(qr))
    return hand_fall(body.hand_floor(q[:T]), body.hand_floor(qr[:T]), touch=TOUCH, near=NEAR, motion_s=0.5)


def pose_similarity(fk, q, ref, losses, kin):
    """mse_vs_align_test.metrics (until refactor step 4): the similarity measures neither L_align nor the MSE cost optimises --
    joint_deg (mean |joint angle error|, deg), mpjpe_loc (cm, body positions relative to the root in the heading frame),
    mpjpe_glob (cm, world frame), head_deg (mean |root heading error|, deg). fk: the body's MjModel; losses, kin: model.losses
    and model.kinematics (passed in, as before)."""
    T = min(len(q), len(ref)); q, ref = q[:T], ref[:T]
    names = [fk.body(i).name for i in range(1, fk.nbody)]
    pa, qa = kin.batch_forward_pose(fk, q, names)
    pb, qb = kin.batch_forward_pose(fk, ref, names)
    ha, hb = losses.root_heading(qa[kin.ROOT_BODY]), losses.root_heading(qb[kin.ROOT_BODY])

    def local(p, h):
        c, s = np.cos(h), np.sin(h)
        return np.stack([c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1], p[:, 2]], -1)

    r = kin.ROOT_BODY
    loc = np.mean([np.linalg.norm(local(pa[n] - pa[r], ha) - local(pb[n] - pb[r], hb), axis=-1).mean() for n in names])
    glob = np.mean([np.linalg.norm(pa[n] - pb[n], axis=-1).mean() for n in names])
    dq = (q[:, 7:] - ref[:, 7:] + np.pi) % (2 * np.pi) - np.pi
    return dict(joint_deg=float(np.degrees(np.abs(dq)).mean()), mpjpe_loc=100 * float(loc), mpjpe_glob=100 * float(glob),
                head_deg=float(np.degrees(np.abs(np.angle(np.exp(1j * (ha - hb))))).mean()))


# ------------------------------------------------------------------ the suite

class Rollout:
    """one rollout against a suite's reference: q, r (float64, cut to the shorter, T frames) and, computed on first use, its
    per-frame world body error (`world_error`), floor distances (`floor`) and eval_metrics.tracking_metrics (`tracking`)"""

    def __init__(self, suite, q):
        self.suite, self.body = suite, suite.body
        q = q.astype(np.float64); T = min(len(q), len(suite.ref)); self.q, self.r, self.T = q[:T], suite.ref[:T], T
        self._cache = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def world_error(self):
        return self._get("world_error", lambda: mean_body_error(world_positions(self.body.m, self.body.d, self.q), self.suite.Pw[:self.T]))

    @property
    def floor(self):
        return self._get("floor", lambda: self.body.part_floor(self.q)[1])

    @property
    def tracking(self):
        return self._get("tracking", lambda: _eval_metrics().tracking_metrics(self.body, self.q, self.r))


# name -> f(rollout): the named metrics a suite can list (a suite also takes (name, f) pairs for its own)
METRICS = {
    "fall_s": lambda x: contact_fall_time(x.floor, x.suite.Dr[:x.T]),
    "success": lambda x: phc_success(x.world_error, x.suite.s),
    "mpjpe": lambda x: x.tracking["mpjpe"],
    "mmpjpe": lambda x: x.tracking["mmpjpe"],
    "mpjae": lambda x: x.tracking["mpjae"],
    "drift": lambda x: drift(x.q, x.r, x.suite.drift_unit, x.suite.s),
    "heading": lambda x: heading_error(x.q, x.r),
}
TASK_PAGE = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "drift")   # the stats() tuple of the Task-page drivers


class MetricSuite:
    """Named metrics, in order, of rollouts against one clip's reference on one body (module docstring)."""

    def __init__(self, body, metrics=TASK_PAGE, s=1.0, drift_unit="body"):
        """metrics: names from METRICS or (name, f(rollout)) pairs; s: the body's root scale (success threshold 0.5 x s, drift
        / s with drift_unit "adult")"""
        self.body, self.s, self.drift_unit = body, s, drift_unit
        self.metrics = tuple((m, METRICS[m]) if isinstance(m, str) else tuple(m) for m in metrics)
        self.ref = self._Pw = self._Dr = None

    @classmethod
    def task_page(cls, body, s, drift_unit="body"):
        """the Task page's (fall_s, success, mpjpe, mmpjpe, mpjae, drift); drift in the body's metres as the drivers' stats()"""
        return cls(body, TASK_PAGE, s, drift_unit)

    @property
    def names(self):
        return tuple(n for n, _ in self.metrics)

    def reference(self, ref, Pw=None, Dr=None):
        """bind one clip's reference frames (float64, frames 1..: ref[1:]); its world positions Pw and floor distances Dr are
        computed on first use unless given (the drivers pass the ones they computed once per clip). Returns self."""
        self.ref, self._Pw, self._Dr = ref, Pw, Dr
        return self

    @property
    def Pw(self):
        if self._Pw is None:
            self._Pw = world_positions(self.body.m, self.body.d, self.ref)
        return self._Pw

    @property
    def Dr(self):
        if self._Dr is None:
            self._Dr = self.body.part_floor(self.ref)[1]
        return self._Dr

    def rollout(self, q):
        return Rollout(self, q)

    def __call__(self, q):
        """{name: value} in the suite's order"""
        x = self.rollout(q)
        return {n: f(x) for n, f in self.metrics}

    def values(self, q):
        """[value] in the suite's order"""
        x = self.rollout(q)
        return [f(x) for _, f in self.metrics]
