"""retarget.py -- a reference motion for a new body: Retargeter and its subclasses (OOP refactor step 9).

A Retargeter has one job: the adult's clip (qpos, 30 fps) -> the same motion as the target body should perform it, frame for
frame. It writes `<data>/<body>/retargeting_motion/<task>/<stem>.npz` (key qpos, plus fps), the layout every search reads
(crossembodiment.paths.retargeting_motion), and for AMASS clips `<data>/<body>/amass_retargeting_motion/ACCAD/<clip>.npz` (the
layout made for len1.4_wid0.8; the child's AMASS clips live in amass_data/amass_retargeting_motion/ACCAD).

  SameTopologyRetargeter   scripts/qpos_retarget.py: the body shares the adult's joints (a ScaledBody), so every hinge angle
                           and the root orientation are copied; the root position is scaled by the ratio of the two
                           skeleton.json rest pelvis heights; then, frame by frame, the pose is lifted (root z only) until no geom
                           is below `tolerance` (-5 mm; never pushed down). Same arithmetic, same np.savez call: the files are
                           byte-identical to qpos_retarget.py's (tests/golden/verify_bodies.py).
  CopyReference            the adult's own clip, file copied byte for byte (heavy_arm_body.py: the payload body is asked to do
                           exactly what the adult did; limit_joint_body.py: the lock body's reference, so both track one motion)
  LockedJointReference     lock_joint_body.py: the adult's clip with the locked joints' angles replaced by their lock angle in
                           every frame, then the same ground correction on the locked body (optional: lift=False)
Elderly's reference (slowed walk, stooped spine, half arm swing) is planned together with its joint bands, so it lives with the
builder (bodybuild/elderly.py) and only shares ground_correct_qpos.

Robots are not retargeted here: model/robot_map.py RobotMap maps SMPL qpos onto a robot's own joints (IK per paired link) and
scripts/robot_roundtrip.py writes outputs/<robot>/roundtrip/<stem>/{robot.npz | g1.npz} (crossembodiment.paths.roundtrip), the
robots' reference; that route is a different problem (another topology) and is left as it is.

Plain functions (as qpos_retarget.py had them; other scripts import them): retarget_qpos, min_geom_z (_min_geom_z),
ground_correct_qpos.
"""
from __future__ import annotations

import itertools
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .skeleton import load_root_rest_height  # noqa: E402,F401  (re-exported: qpos_retarget.py's name)

FPS = 30.0          # the robotmotion clips carry no fps of their own; it is written into every output (qpos_retarget.py --fps)
TOLERANCE = -0.005  # lowest allowed geom z (m) before the ground correction lifts a frame (MuJoCo's soft-contact penetration)


def retarget_qpos(qpos, scale):
    out = qpos.copy()
    out[:, 0:3] *= scale  # root (free joint) world position only; quat (3:7)
    # and every hinge angle (7:) are joint-local and topology-identical, so
    # they carry over unchanged.
    return out


def min_geom_z(model, data):
    """Exact lowest world-z point over all non-floor geoms for the CURRENT
    data.qpos (caller must have already run mj_kinematics). Same math as
    shape.measure_min_z -- geom_rbound is a bounding-sphere radius and isn't tight
    for box geoms (the feet here are boxes). (qpos_retarget.py _min_geom_z)"""
    lo = np.inf
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor":
            continue
        gtype = model.geom_type[gid]
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            for signs in itertools.product([1, -1], repeat=3):
                lo = min(lo, (pos + mat @ (np.array(signs) * size))[2])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c1 = (pos + mat @ np.array([0, 0, size[1]]))[2] - size[0]
            c2 = (pos + mat @ np.array([0, 0, -size[1]]))[2] - size[0]
            lo = min(lo, c1, c2)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            lo = min(lo, pos[2] - size[0])
        else:
            lo = min(lo, pos[2])
    return lo


_min_geom_z = min_geom_z


def ground_correct_qpos(target_model, qpos, tolerance=TOLERANCE):
    """Per-frame: if the target skeleton's lowest geometry point is below
    `tolerance`, lift the whole pose (root z only) so it's exactly at
    `tolerance`. Never pushes an already-clear/airborne frame down.
    Returns (qpos, frames corrected, largest shift)."""
    data = mujoco.MjData(target_model)
    out = qpos.copy()
    n_corrected = 0
    max_shift = 0.0
    for t in range(out.shape[0]):
        data.qpos[:] = out[t]
        mujoco.mj_kinematics(target_model, data)
        min_z = min_geom_z(target_model, data)
        if min_z < tolerance:
            shift = tolerance - min_z
            out[t, 2] += shift
            n_corrected += 1
            max_shift = max(max_shift, shift)
    return out, n_corrected, max_shift


def save_clip(path, qpos, fps=FPS):
    """one reference clip, written as every retargeting script writes it: np.savez(path, qpos=..., fps=...)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, qpos=qpos, fps=fps)
    return path


class Retargeter(ABC):
    """the adult's qpos (T, nq) -> the target body's (T, nq); write_clip / write_folder / write_tasks put the results where the
    searches look for them. `last` holds the statistics of the latest retarget (frames ground-corrected, largest lift)."""

    def __init__(self):
        self.last = dict(n_corrected=0, max_shift=0.0)

    @abstractmethod
    def retarget(self, qpos):
        """(T, nq) adult qpos -> (T, nq) target qpos"""

    def write_clip(self, src, dst, fps=FPS):
        q = np.load(src)["qpos"]
        return save_clip(dst, self.retarget(q), fps)

    def write_folder(self, src_dir, dst_dir, fps=FPS, log=print):
        """qpos_retarget.py's batch mode: every .npz under src_dir (rglob, sorted) -> dst_dir/<stem>.npz (flat); returns the count
        and the frames ground-corrected in total"""
        src_dir, dst_dir = Path(src_dir), Path(dst_dir)
        npz_paths = sorted(src_dir.rglob("*.npz"))
        dst_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for npz_path in npz_paths:
            self.write_clip(npz_path, dst_dir / f"{npz_path.stem}.npz", fps)
            total += self.last["n_corrected"]
        log(f"Wrote {len(npz_paths)} retargeted qpos files -> {dst_dir} ({total} frames ground-corrected total)")
        return len(npz_paths), total


class SameTopologyRetargeter(Retargeter):
    """scripts/qpos_retarget.py: root position x (target rest pelvis height / source's), joints copied, then ground_correct_qpos
    on the target body (unless ground_correct=False)."""

    def __init__(self, source_skeleton, target_skeleton, target_xml, ground_correct=True, tolerance=TOLERANCE):
        super().__init__()
        self.source_skeleton, self.target_skeleton, self.target_xml = source_skeleton, target_skeleton, target_xml
        self.scale = load_root_rest_height(target_skeleton) / load_root_rest_height(source_skeleton)
        self.ground_correct, self.tolerance = ground_correct, tolerance
        self.model = mujoco.MjModel.from_xml_path(str(target_xml)) if ground_correct else None

    def retarget(self, qpos):
        out = retarget_qpos(qpos, self.scale)
        if self.ground_correct:
            out, n, shift = ground_correct_qpos(self.model, out, self.tolerance)
            self.last = dict(n_corrected=n, max_shift=shift)
        else:
            self.last = dict(n_corrected=0, max_shift=0.0)
        return out


class CopyReference(Retargeter):
    """the source clip itself: write_clip copies the file byte for byte (heavy_arm_body.py, limit_joint_body.py)"""

    def retarget(self, qpos):
        return qpos.copy()

    def write_clip(self, src, dst, fps=FPS):
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return dst


class LockedJointReference(Retargeter):
    """lock_joint_body.py's reference: every locked joint's angle set to its lock angle (degrees) in every frame, then (lift=True)
    ground_correct_qpos on the locked body `model` (the default tolerance)."""

    def __init__(self, model, locks, lift=True):
        super().__init__()
        self.model, self.locks, self.lift = model, list(locks), lift
        self.qadr = {jn: int(model.joint(jn).qposadr[0]) for jn, _ in self.locks}

    def locked(self, qpos):
        out = qpos.copy()
        for jn, deg in self.locks:
            out[:, self.qadr[jn]] = np.radians(deg)
        return out

    def lowest(self, qpos):
        """the lowest geom point of every frame (lock_joint_body.py's report before the lift)"""
        d0 = mujoco.MjData(self.model); low = []
        for t in range(len(qpos)):
            d0.qpos[:] = qpos[t]; mujoco.mj_kinematics(self.model, d0); low.append(min_geom_z(self.model, d0))
        return np.array(low)

    def retarget(self, qpos):
        out = self.locked(qpos)
        if self.lift:
            out, n, shift = ground_correct_qpos(self.model, out)
        else:
            n, shift = 0, 0.0
        self.last = dict(n_corrected=n, max_shift=shift)
        return out
