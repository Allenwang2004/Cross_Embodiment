"""robot_map.py -- model/g1_map.py for any robot in model/robots.py: SMPL humanoid <-> robot through per-segment world
orientations.

The same construction as G1Map (see model/g1_map.py for the reasoning), read from the robot's entry:
  pairs       SMPL body <-> robot link (ROBOTS[name]["pairs"]); C_b = R_robot(nominal)^T H R_smpl(calibration pose)
  nominal     the robot joint angles whose segment directions best match the SMPL calibration pose (arms down, turned
              to face +x), from ROBOTS[name]["calib"]
  robot -> SMPL  paired segment orientations -> SMPL joint angles; the waist's pelvis-to-chest rotation split evenly over
              Torso / Spine / Chest; SMPL wrist joint 0 (the capsule rides with the forearm); SMPL hand joint = forearm ->
              robot hand if the robot has a hand link, else Motivo's mean posture like the toes, neck, head and
              collarbones (neck and head from the robot's head when it has one, Apollo: the chest-to-head rotation
              split evenly over Neck / Head); root orientation from the pelvis, x, y / s, height from the lowest
              collision point
  SMPL -> robot  orientation-only damped least squares inside the joint ranges, warm-started, re-solved from the
              nominal pose per chain when a joint ends at a limit; robot joints with no paired link below them stay where
              they start (the nominal pose)
s = robot / SMPL hip-to-ankle length in the nominal pose. RobotMap("g1") reproduces G1Map.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from model.exact_obs import _LowestPoint
from model.robots import REPO, ROBOTS, collision_geoms

import os
_IK_MOVED_TOL = float(os.environ.get("ROBOT_IK_MOVED_TOL", 0))     # TEMPORARY test of an IK stop on the clipped step
SMPL_XML = "assets/robots/adult/robot.xml"
H = Rotation.from_euler("z", 90, degrees=True).as_matrix()     # SMPL rest pose faces -y; the robots face +x
FILL_JSON = REPO / "assets/robots/g1/smpl_fill.json"
CALIB_POSE = {"L_Shoulder_z": -np.pi / 2, "R_Shoulder_z": np.pi / 2}   # arms down; every other SMPL joint 0
SMPL_SIDES = ("L", "R")

# fill="ridge": the SMPL parts no robot link reports, predicted from the ones it does (scripts/fit_smpl_fill.py, trained
# on data/origin_motion without the survey clips' tasks). Inputs: the paired parts' orientations relative to each other
# (6-D: two columns) and world z in the pelvis / chest frames; outputs: the local rotations (rotation vectors) of
# FILL_TARGETS (+ the hands on robots without a hand link)
FILL_REL = [("Pelvis", "Chest")] + [(a, f"{S}_{b}") for S in SMPL_SIDES for a, b in
            (("Chest", "Shoulder"),)] + [(f"{S}_{a}", f"{S}_{b}") for S in SMPL_SIDES for a, b in (("Shoulder", "Elbow"),)] + \
           [("Pelvis", f"{S}_Hip") for S in SMPL_SIDES] + [(f"{S}_{a}", f"{S}_{b}") for S in SMPL_SIDES for a, b in
            (("Hip", "Knee"), ("Knee", "Ankle"))]
FILL_REL_HAND = [(f"{S}_Elbow", f"{S}_Hand") for S in SMPL_SIDES]
FILL_TARGETS = ["Torso", "Spine", "Neck", "Head", "L_Thorax", "R_Thorax", "L_Wrist", "R_Wrist", "L_Toe", "R_Toe"]
FILL_RIDGE = {True: REPO / "assets/robots/smpl_fill_ridge_hand_motivo.npz",
              False: REPO / "assets/robots/smpl_fill_ridge_nohand_motivo.npz"}


def fill_feats(R, hand):
    """the robot-reportable pose. R: {SMPL body: (..., 3, 3)} world orientations of the paired parts."""
    rel = FILL_REL + (FILL_REL_HAND if hand else [])
    f = [(np.swapaxes(R[a], -1, -2) @ R[b])[..., :, :2].reshape(*R[a].shape[:-2], 6) for a, b in rel]
    f += [R["Pelvis"][..., 2, :], R["Chest"][..., 2, :]]                  # world z in the pelvis / chest frame
    return np.concatenate(f, -1)


def fill_targets(hand):
    return FILL_TARGETS + ([] if hand else ["L_Hand", "R_Hand"])


def _quat(R):
    # assume_valid: R is always a product of rotations, so scipy's orthogonality check (75% of the call) is skipped; same bits
    x, y, z, w = Rotation.from_matrix(R, assume_valid=True).as_quat()
    return np.array([w, x, y, z])


def _rotvec(R):
    return Rotation.from_matrix(R, assume_valid=True).as_rotvec()


class Lowest:
    """Exact lowest world-z over a model's collision geoms (model/robots.py collision_geoms): spheres, capsules,
    cylinders, boxes, meshes by their vertices, ellipsoids by their bounding box."""

    def __init__(self, m):
        col = collision_geoms(m); t = m.geom_type; G = mujoco.mjtGeom
        self.sph = np.array([g for g in col if t[g] == G.mjGEOM_SPHERE], int)
        self.cap = np.array([g for g in col if t[g] == G.mjGEOM_CAPSULE], int)
        self.cyl = np.array([g for g in col if t[g] == G.mjGEOM_CYLINDER], int)
        self.box = np.array([g for g in col if t[g] in (G.mjGEOM_BOX, G.mjGEOM_ELLIPSOID)], int)
        self.mesh = [(g, m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]])
                     for g in col if t[g] == G.mjGEOM_MESH]
        self.size = m.geom_size
        self.mesh_g = np.array([g for g, _ in self.mesh], int)
        self.mesh_r = np.array([np.linalg.norm(v, axis=1).max() * (1 + 1e-9) for _, v in self.mesh])     # no vertex is farther from the geom's origin

    def __call__(self, d):
        lo = np.inf
        if len(self.sph):
            lo = min(lo, (d.geom_xpos[self.sph, 2] - self.size[self.sph, 0]).min())
        if len(self.cap):
            az = np.abs(d.geom_xmat[self.cap].reshape(-1, 3, 3)[:, 2, 2])
            lo = min(lo, (d.geom_xpos[self.cap, 2] - az * self.size[self.cap, 1] - self.size[self.cap, 0]).min())
        if len(self.cyl):
            az = np.abs(d.geom_xmat[self.cyl].reshape(-1, 3, 3)[:, 2, 2])
            r, h = self.size[self.cyl, 0], self.size[self.cyl, 1]
            lo = min(lo, (d.geom_xpos[self.cyl, 2] - az * h - r * np.sqrt(np.clip(1 - az ** 2, 0, 1))).min())
        if len(self.box):
            Rz = d.geom_xmat[self.box].reshape(-1, 3, 3)[:, 2, :]                 # world z row of each frame
            lo = min(lo, (d.geom_xpos[self.box, 2] - np.abs(Rz * self.size[self.box]).sum(1)).min())
        if self.mesh:                # lowest bound first; a mesh whose bound is above the minimum so far cannot change it
            bound = d.geom_xpos[self.mesh_g, 2] - self.mesh_r
            for k in np.argsort(bound):
                if bound[k] >= lo:
                    break
                g, v = self.mesh[k]
                lo = min(lo, (d.geom_xpos[g, 2] + v @ d.geom_xmat[g].reshape(3, 3)[2]).min())
        return lo


class RobotMap:
    def __init__(self, name, smpl_xml=SMPL_XML, weights=None, fill="fixed"):
        """weights: {robot link: weight} in the orientation least squares (default 1 each). fill: how robot -> SMPL
        fills the parts no robot link reports -- "fixed" (Motivo's mean posture, the waist split evenly, the wrist at 0)
        or "ridge" (predicted from the reported parts, FILL_RIDGE)."""
        self.name, self.cfg = name, ROBOTS[name]
        cfg = self.cfg
        self.pairs = list(cfg["pairs"])
        self.mg = mujoco.MjModel.from_xml_path(str(REPO / cfg["xml"])); self.dg = mujoco.MjData(self.mg)
        self.ms = mujoco.MjModel.from_xml_path(str(REPO / smpl_xml)); self.ds = mujoco.MjData(self.ms)
        self.low_g, self.low_s = Lowest(self.mg), _LowestPoint(self.ms)
        mg, ms = self.mg, self.ms
        self.free_q = next(mg.jnt_qposadr[j] for j in range(mg.njnt) if mg.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.g_jnt = [j for j in range(mg.njnt) if mg.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        self.g_qadr = np.array([mg.jnt_qposadr[j] for j in self.g_jnt])
        self.g_dadr = np.array([mg.jnt_dofadr[j] for j in self.g_jnt])
        self.g_range = mg.jnt_range[self.g_jnt]
        self.s_jnt = list(range(1, ms.njnt))
        self.s_names = [ms.joint(j).name for j in self.s_jnt]
        self.s_qadr = np.array([ms.jnt_qposadr[j] for j in self.s_jnt])
        self.s_adr3 = {n[:-2]: self.s_qadr[[self.s_names.index(f"{n[:-2]}_{ax}") for ax in "xyz"]] for n in self.s_names if n.endswith("_x")}     # SMPL body -> its 3 qpos
        self.rel_adr = None          # robot_to_smpl_qpos: the qpos of its bodies, in order (always the same for one map)
        self.sid = {b: ms.body(b).id for b, _ in self.pairs}
        self.gid = {b: mg.body(g).id for b, g in self.pairs}
        self.anc = np.zeros((len(self.pairs) - 1, len(self.g_jnt)))
        for i, (b, _) in enumerate(self.pairs[1:]):
            chain, x = set(), self.gid[b]
            while x > 0:
                chain.add(x); x = mg.body_parentid[x]
            self.anc[i] = [mg.jnt_bodyid[j] in chain for j in self.g_jnt]
        self.link_ids = np.array([self.gid[b] for b, _ in self.pairs[1:]])
        self.g_jnt_a = np.array(self.g_jnt)
        self.j_chain = np.array([self._chain(mg.joint(j).name) for j in self.g_jnt])
        self.l_chain = np.array([self._chain(g) for _, g in self.pairs[1:]])
        self.has_hand = any(b.endswith("_Hand") for b, _ in self.pairs)
        self.has_head = any(b == "Head" for b, _ in self.pairs)       # a robot with a neck (Apollo)
        unmapped = (["L_Toe", "R_Toe"] + ([] if self.has_head else ["Neck", "Head"]) + ["L_Thorax", "R_Thorax"]
                    + ([] if self.has_hand else ["L_Hand", "R_Hand"]))
        angles = json.loads(FILL_JSON.read_text())["angles_rad"]
        self.fill = {b: np.array([angles[f"{b}_{a}"] for a in "xyz"]) for b in unmapped}
        self.fill_R = {b: Rotation.from_euler("XYZ", v).as_matrix() for b, v in self.fill.items()}
        assert fill in ("fixed", "ridge"), fill
        self.fill_mode, self.ridge = fill, (dict(np.load(FILL_RIDGE[self.has_hand])) if fill == "ridge" else None)
        self.w = np.sqrt(np.repeat([1.0 if weights is None else weights.get(g, 1.0) for _, g in self.pairs[1:]], 3))
        self.q_nom, self.calib = self._calibrate()
        self.C = {b: self._g_rot(self.q_nom, b).T @ (H @ self._s_calib_rot(b)) for b, _ in self.pairs}
        leg = lambda d, m, a, z: np.linalg.norm(d.xpos[m.body(z).id] - d.xpos[m.body(a).id])
        self._fk_g(self.q_nom); self._fk_s_calib()
        self.s = leg(self.dg, mg, *cfg["leg"]) / leg(self.ds, ms, "L_Hip", "L_Ankle")

    def _chain(self, name):
        """robot joint or link name -> its chain: <side>leg, <side>arm or waist."""
        side = next((S for S, s in self.cfg["sides"] if name.startswith(s + "_")), "")
        if any(k in name for k in ("hip", "knee", "ankle", "foot")):
            return side + "leg"
        if any(k in name for k in ("shoulder", "elbow", "wrist", "hand")):
            return side + "arm"
        return "waist"

    # ---- forward kinematics helpers
    def _fk_g(self, q, root_quat=(1, 0, 0, 0)):
        self.dg.qpos[:] = 0; self.dg.qpos[self.free_q + 3:self.free_q + 7] = root_quat; self.dg.qpos[self.g_qadr] = q
        mujoco.mj_kinematics(self.mg, self.dg)

    def _fk_s_calib(self):
        self.ds.qpos[:] = self.ms.qpos0
        for j, v in CALIB_POSE.items():
            self.ds.qpos[self.ms.joint(j).qposadr[0]] = v
        mujoco.mj_kinematics(self.ms, self.ds)

    def _g_rot(self, q, b):
        self._fk_g(q)
        return self.dg.xmat[self.gid[b]].reshape(3, 3).copy()

    def _s_calib_rot(self, b):
        self._fk_s_calib()
        return self.ds.xmat[self.sid[b]].reshape(3, 3).copy()

    # ---- calibration
    def _robot_end(self, b):
        """the body's farthest geometry point (collision or visual; meshes by vertices) from its origin, in its frame."""
        m = self.mg; best, far = None, -1.0
        mujoco.mj_kinematics(m, self.dg)
        Rb, pb = self.dg.xmat[b].reshape(3, 3), self.dg.xpos[b]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            Rg, pg = self.dg.geom_xmat[g].reshape(3, 3), self.dg.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                P = pg + m.mesh_vert[a:a + n].astype(float) @ Rg.T
            else:
                ctr, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                P = pg + (ctr + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])) @ Rg.T
            dist = np.linalg.norm(P - pb, axis=1); i = int(np.argmax(dist))
            if dist[i] > far:
                far, best = dist[i], (P[i] - pb) @ Rb                      # body frame
        return best

    def _calibrate(self):
        """robot joint angles whose segment directions best match CALIB_POSE turned to face +x."""
        ms, ds, mg, dg = self.ms, self.ds, self.mg, self.dg
        self._fk_s_calib()
        unit = lambda v: v / np.linalg.norm(v)

        def smpl_vec(spec):
            kind = spec[0]
            if kind == "seg":           # (H p_b - H p_a: the operation order of g1_map, so the G1 calibration is bit-exact)
                return unit(H @ ds.xpos[ms.body(spec[2]).id] - H @ ds.xpos[ms.body(spec[1]).id])
            if kind == "axis":
                return H @ ds.xaxis[ms.joint(spec[1]).id]
            if kind == "col":
                return H @ ds.xmat[ms.body(spec[1]).id].reshape(3, 3)[:, int(spec[2])]
            raise ValueError(spec)

        self._fk_g(np.zeros(len(self.g_jnt)))
        ends = {spec[1]: self._robot_end(mg.body(spec[1]).id) for _, _, spec in self.cfg["calib"] if spec[0] == "end"}

        def robot_vec(spec):
            kind = spec[0]
            if kind == "seg":
                return unit(dg.xpos[mg.body(spec[2]).id] - dg.xpos[mg.body(spec[1]).id])
            if kind == "axis":
                return dg.xaxis[mg.joint(spec[1]).id]
            if kind == "col":
                return dg.xmat[mg.body(spec[1]).id].reshape(3, 3)[:, int(spec[2])]
            if kind == "geomcol":
                return dg.geom_xmat[mg.geom(spec[1]).id].reshape(3, 3)[:, int(spec[2])]
            if kind == "end":
                b = mg.body(spec[1]).id
                return unit(dg.xmat[b].reshape(3, 3) @ ends[spec[1]])
            if kind == "mid_up":
                return unit((dg.xpos[mg.body(spec[1]).id] + dg.xpos[mg.body(spec[2]).id]) / 2 - dg.xpos[mg.body(spec[3]).id])
            raise ValueError(spec)

        names = [n for n, _, _ in self.cfg["calib"]]
        target = np.stack([smpl_vec(s) for _, s, _ in self.cfg["calib"]])
        is_axis = np.array(["axis" in n for n in names])

        def res(q):
            self._fk_g(q)
            G = np.stack([robot_vec(s) for _, _, s in self.cfg["calib"]])
            r = np.where(is_axis[:, None], np.cross(G, target), G - target)
            return np.concatenate([r.ravel(), 0.05 * q])

        x0 = np.zeros(len(self.g_jnt))
        for j, v in self.cfg.get("x0", {}).items():
            x0[self.g_jnt.index(mg.joint(j).id)] = v
        lo, hi = self.g_range[:, 0], self.g_range[:, 1]
        sol = least_squares(res, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
        self._fk_g(sol.x)
        err = {}
        for n, (_, s, r) in zip(names, self.cfg["calib"]):
            a, b = robot_vec(r), target[names.index(n)]
            c = abs(a @ b) if "axis" in n else a @ b
            err[n] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        return sol.x, err

    # ---- robot -> SMPL
    def smpl_rots_from_robot(self, qpos_g):
        self.dg.qpos[:] = qpos_g; mujoco.mj_kinematics(self.mg, self.dg)
        return {b: self.dg.xmat[self.gid[b]].reshape(3, 3) @ self.C[b] for b, _ in self.pairs}

    def fill_predict(self, R):
        """fill="ridge": {SMPL body: local 3x3} for fill_targets, from the paired parts' orientations R."""
        r = self.ridge
        y = (((fill_feats(R, self.has_hand) - r["mx"]) / r["sx"]) @ r["W"] + r["my"]).reshape(-1, 3)
        return dict(zip(fill_targets(self.has_hand), Rotation.from_rotvec(y).as_matrix()))

    def robot_to_smpl_qpos(self, qpos_g):
        R = self.smpl_rots_from_robot(qpos_g)
        rel = {}
        if self.ridge is None:
            third = Rotation.from_rotvec(_rotvec(R["Pelvis"].T @ R["Chest"]) / 3).as_matrix()
            rel["Torso"] = rel["Spine"] = rel["Chest"] = third
        else:                     # predicted waist split (the chest still lands on the robot's), neck, head, toes
            p = self.fill_predict(R)
            rel["Torso"], rel["Spine"] = p["Torso"], p["Spine"]
            rel["Chest"] = (p["Torso"] @ p["Spine"]).T @ (R["Pelvis"].T @ R["Chest"])
            for b in ("Neck", "Head", "L_Toe", "R_Toe"):
                rel[b] = p[b]
        if self.has_head:         # the robot's head: the chest-to-head rotation split evenly over Neck / Head
            half = Rotation.from_rotvec(_rotvec(R["Chest"].T @ R["Head"]) / 2).as_matrix()
            rel["Neck"] = rel["Head"] = half
        for S in SMPL_SIDES:
            rel[f"{S}_Hip"] = R["Pelvis"].T @ R[f"{S}_Hip"]
            rel[f"{S}_Knee"] = R[f"{S}_Hip"].T @ R[f"{S}_Knee"]
            rel[f"{S}_Ankle"] = R[f"{S}_Knee"].T @ R[f"{S}_Ankle"]
            if self.ridge is None:
                rel[f"{S}_Shoulder"] = self.fill_R[f"{S}_Thorax"].T @ R["Chest"].T @ R[f"{S}_Shoulder"]
                rel[f"{S}_Elbow"] = R[f"{S}_Shoulder"].T @ R[f"{S}_Elbow"]
                rel[f"{S}_Wrist"] = np.eye(3)
                if self.has_hand:
                    rel[f"{S}_Hand"] = R[f"{S}_Elbow"].T @ R[f"{S}_Hand"]
            else:                 # predicted clavicle and wrist; the upper arm and hand still land on the robot's
                rel[f"{S}_Thorax"] = p[f"{S}_Thorax"]
                rel[f"{S}_Shoulder"] = p[f"{S}_Thorax"].T @ R["Chest"].T @ R[f"{S}_Shoulder"]
                rel[f"{S}_Elbow"] = R[f"{S}_Shoulder"].T @ R[f"{S}_Elbow"]
                rel[f"{S}_Wrist"] = p[f"{S}_Wrist"]
                rel[f"{S}_Hand"] = (p[f"{S}_Wrist"].T @ R[f"{S}_Elbow"].T @ R[f"{S}_Hand"] if self.has_hand
                                    else p[f"{S}_Hand"])
        q = self.ms.qpos0.copy()
        q[3:7] = _quat(R["Pelvis"])
        for body, v in self.fill.items():
            q[self.s_adr3[body]] = v
        if self.rel_adr is None:
            self.rel_adr = np.concatenate([self.s_adr3[b] for b in rel])
        ang = Rotation.from_matrix(np.stack(list(rel.values())), assume_valid=True).as_euler("XYZ")
        q[self.rel_adr] = ang.ravel()
        clearance = self.low_g(self.dg)
        q[:2] = qpos_g[self.free_q:self.free_q + 2] / self.s
        q[2] = 0.0
        self.ds.qpos[:] = q; mujoco.mj_kinematics(self.ms, self.ds)
        q[2] = clearance / self.s - self.low_s(self.ds)
        return q

    # ---- SMPL -> robot
    def smpl_to_robot_joints(self, qpos_s, q_init=None, iters=50, damping=1e-4, tol=1e-7):
        self.ds.qpos[:] = qpos_s; mujoco.mj_kinematics(self.ms, self.ds)
        Rt = [self.ds.xmat[self.sid[b]].reshape(3, 3) @ self.C[b].T for b, _ in self.pairs]
        root = _quat(Rt[0])
        Rt = np.stack(Rt[1:])        # (links, 3, 3), what _link_err, _solve and _resid take
        lo, hi = self.g_range[:, 0], self.g_range[:, 1]
        q = self._solve(Rt, root, self.q_nom if q_init is None else q_init, iters, damping, tol)
        resid = self._resid(Rt, root, q)
        if q_init is not None and np.any((q <= lo + 1e-6) | (q >= hi - 1e-6)):
            q2 = self._solve(Rt, root, self.q_nom, iters, damping, tol)
            r2 = self._resid(Rt, root, q2)
            swap = False
            for c in np.unique(self.l_chain):
                if r2[self.l_chain == c].sum() < resid[self.l_chain == c].sum() - 1e-6:
                    q[self.j_chain == c] = q2[self.j_chain == c]; swap = True
            if swap:
                q = self._solve(Rt, root, q, 10, damping, tol)
                resid = self._resid(Rt, root, q)
        return q, resid

    def _link_err(self, Rt, root, q):
        self._fk_g(q, root)
        Rc = self.dg.xmat[self.link_ids].reshape(-1, 3, 3)
        return Rotation.from_matrix(Rt @ Rc.transpose(0, 2, 1), assume_valid=True).as_rotvec()

    def _solve(self, Rt, root, q0, iters, damping, tol):
        q = np.array(q0, dtype=float)
        lo, hi = self.g_range[:, 0], self.g_range[:, 1]
        D = damping * np.eye(len(q))
        for _ in range(iters):
            e = self._link_err(Rt, root, q).ravel()
            ax = self.dg.xaxis[self.g_jnt_a]
            J = (self.anc[:, None, :] * ax.T[None]).reshape(-1, len(q))
            J, e = J * self.w[:, None], e * self.w
            dq = np.linalg.solve(J.T @ J + D, J.T @ e)
            qn = np.clip(q + dq, lo, hi); moved = qn - q; q = qn
            if dq @ dq < tol or moved @ moved < _IK_MOVED_TOL:
                break
        return q

    def _resid(self, Rt, root, q):
        return np.degrees(np.linalg.norm(self._link_err(Rt, root, q), axis=1))

    def smpl_to_robot_qpos(self, qpos_s, q_init=None):
        q, resid = self.smpl_to_robot_joints(qpos_s, q_init)
        self.ds.qpos[:] = qpos_s; mujoco.mj_kinematics(self.ms, self.ds)
        clearance = self.low_s(self.ds)
        out = np.zeros(self.mg.nq)
        f = self.free_q
        out[f + 3:f + 7] = _quat(self.ds.xmat[self.sid["Pelvis"]].reshape(3, 3) @ self.C["Pelvis"].T)
        out[self.g_qadr] = q
        out[f:f + 2] = np.asarray(qpos_s[:2]) * self.s
        self.dg.qpos[:] = out; mujoco.mj_kinematics(self.mg, self.dg)
        out[f + 2] = clearance * self.s - self.low_g(self.dg)
        return out, resid
