"""g1_map.py -- Unitree G1 <-> SMPL humanoid (Meta Motivo's body), through per-segment world orientations.

Motivo only reads SMPL observations (358-d, humenv) and only writes SMPL actions (69 PD targets). To drive a G1
with it, both directions go through one shared quantity: the world orientation of a body segment.

  pairs     SMPL body <-> G1 link carrying the same segment (PAIRS): pelvis, chest, thigh, shank, foot, upper arm,
            forearm (G1: the link after the wrist roll, so it carries the forearm twist) and hand. The SMPL arm ends
            in two short pieces: the L_Wrist capsule after the wrist joint and the L_Hand box after the hand joint;
            the capsule counts as part of the forearm and the box is the hand (since 10/20: before, the capsule was
            paired with the G1 hand and the box left at the mean posture, so a hand flat on the floor -- crawl,
            handstand -- never turned the G1 wrist).
  nominal   one-off calibration in a shared posture: the SMPL rest pose with the arms lowered to hang straight down
            (CALIB_POSE; not the T-pose, which is a singularity of the G1 shoulder: with the arm horizontal its
            pitch and yaw axes coincide), turned to face +x like the G1, and the G1 joint configuration whose segment
            directions match it best (_calibrate; hinge axes compared up to sign, since a hinge's sign is a
            convention). For each pair C_b = R_g1(nominal)^T R_smpl(nominal), so in any pose R_smpl = R_g1 C_b and
            R_g1 = R_smpl C_b^T. This absorbs the different local axes (SMPL x left / y up / z forward, G1 x forward
            / y left / z up), the G1's tilted link frames and the different rest poses.

  G1 -> SMPL (g1_to_smpl_qpos): G1 link orientations -> SMPL segment orientations -> each SMPL joint's parent-child
            rotation -> its x, y, z hinge angles (exact: an SMPL joint is three hinges in that order). The G1 waist's
            pelvis-to-chest rotation is split evenly over Torso / Spine / Chest; the SMPL wrist joint is 0 (the
            capsule rigid with the forearm) and the hand joint takes the forearm -> G1 hand rotation. SMPL joints with
            no G1 counterpart (toes, neck, head, collarbones) take Motivo's mean posture (FILL_JSON; the shoulder absorbs the
            collarbone so the upper arm keeps the G1's orientation). On the walk clip move-ego-0-2_4 with the paired
            segments kept exact (measured with the earlier wrist-capsule pairing), the bfm cost of this completion vs the true SMPL pose is 0.097 (zeros: 0.136; a
            linear regression from the paired segments: 0.092; spine re-split alone: 0.009). Root: the G1 pelvis orientation, x, y / s, and the
            height at which the SMPL body's lowest point has the G1's ground clearance / s.
  SMPL -> G1 (smpl_to_g1_joints): SMPL hinge angles -> SMPL segment orientations (MuJoCo FK) -> target G1 link
            orientations -> G1 joint angles by orientation-only damped least squares inside the joint ranges. A
            G1 joint with fewer DoF than SMPL's (knee, elbow: 1; ankle: 2) ends up with the part of the rotation it
            can make. Warm-started from the previous frame, the solve can run into a joint limit and stay there (crawl:
            the left arm parked at three limits for 298 frames, 30 deg off where a fresh solve lands within 3-11);
            so whenever a joint ends at a limit it is solved again from the nominal pose and each chain (legs, waist,
            arms) keeps whichever solution fits its own links better. The same function maps PD targets (actions)
            and actual angles (a start pose).

s = G1 / SMPL hip-to-ankle length in the nominal pose.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from model.exact_obs import _LowestPoint

SMPL_XML = "assets/robots/adult/robot.xml"
G1_XML = "assets/robots/g1/g1.xml"       # the Menagerie G1 with collision on its rubber hands (see that file's header)
SIDES = (("L", "left"), ("R", "right"))
PAIRS = [("Pelvis", "pelvis"), ("Chest", "torso_link")] + [
    (f"{S}_{a}", f"{s}_{b}") for S, s in SIDES for a, b in
    (("Hip", "hip_yaw_link"), ("Knee", "knee_link"), ("Ankle", "ankle_roll_link"),
     ("Shoulder", "shoulder_yaw_link"), ("Elbow", "wrist_roll_link"), ("Hand", "wrist_yaw_link"))]
H = Rotation.from_euler("z", 90, degrees=True).as_matrix()     # SMPL rest pose faces -y (left = +x); the G1 faces +x
FILL_JSON = Path(__file__).resolve().parent.parent / "assets/robots/g1/smpl_fill.json"
UNMAPPED = ("L_Toe", "R_Toe", "Neck", "Head", "L_Thorax", "R_Thorax")
CALIB_POSE = {"L_Shoulder_z": -np.pi / 2, "R_Shoulder_z": np.pi / 2}   # arms down; every other SMPL joint 0


def _quat(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _rotvec(R):
    return Rotation.from_matrix(R).as_rotvec()


def _chain(name):
    """G1 joint or link name -> its kinematic chain: left_leg, right_leg, waist, left_arm, right_arm."""
    side = "left_" if name.startswith("left") else "right_" if name.startswith("right") else ""
    if any(k in name for k in ("hip", "knee", "ankle")):
        return side + "leg"
    if any(k in name for k in ("shoulder", "elbow", "wrist")):
        return side + "arm"
    return "waist"


class _G1Lowest:
    """Exact lowest world-z over the G1's collision geoms (foot spheres, cylinders, meshes by their vertices)."""

    def __init__(self, m):
        col = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        t = m.geom_type
        self.sph = np.array([g for g in col if t[g] == mujoco.mjtGeom.mjGEOM_SPHERE], int)
        self.cyl = np.array([g for g in col if t[g] == mujoco.mjtGeom.mjGEOM_CYLINDER], int)
        self.mesh = [(g, m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:
                                     m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]])
                     for g in col if t[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.size = m.geom_size

    def __call__(self, d):
        lo = (d.geom_xpos[self.sph, 2] - self.size[self.sph, 0]).min() if len(self.sph) else np.inf
        if len(self.cyl):
            az = np.abs(d.geom_xmat[self.cyl].reshape(-1, 3, 3)[:, 2, 2])
            r, h = self.size[self.cyl, 0], self.size[self.cyl, 1]
            lo = min(lo, (d.geom_xpos[self.cyl, 2] - az * h - r * np.sqrt(np.clip(1 - az ** 2, 0, 1))).min())
        for g, v in self.mesh:
            lo = min(lo, (d.geom_xpos[g, 2] + v @ d.geom_xmat[g].reshape(3, 3)[2]).min())
        return lo


class G1Map:
    def __init__(self, g1_xml=G1_XML, smpl_xml=SMPL_XML, weights=None):
        """weights: {G1 link: weight} in the orientation least squares (default 1 each)."""
        self.mg = mujoco.MjModel.from_xml_path(str(g1_xml)); self.dg = mujoco.MjData(self.mg)
        self.ms = mujoco.MjModel.from_xml_path(str(smpl_xml)); self.ds = mujoco.MjData(self.ms)
        self.low_g, self.low_s = _G1Lowest(self.mg), _LowestPoint(self.ms)
        mg, ms = self.mg, self.ms
        self.g_jnt = list(range(1, mg.njnt))                                  # hinges after the free joint
        self.g_qadr = np.array([mg.jnt_qposadr[j] for j in self.g_jnt])
        self.g_dadr = np.array([mg.jnt_dofadr[j] for j in self.g_jnt])
        self.g_range = mg.jnt_range[self.g_jnt]
        self.s_jnt = list(range(1, ms.njnt))
        self.s_names = [ms.joint(j).name for j in self.s_jnt]
        self.s_qadr = np.array([ms.jnt_qposadr[j] for j in self.s_jnt])
        self.sid = {b: ms.body(b).id for b, _ in PAIRS}
        self.gid = {b: mg.body(g).id for b, g in PAIRS}
        # rotational Jacobian structure: G1 hinge j moves link b iff j's body is b or an ancestor of b
        self.anc = np.zeros((len(PAIRS) - 1, len(self.g_jnt)))
        for i, (b, _) in enumerate(PAIRS[1:]):
            chain, x = set(), self.gid[b]
            while x > 0:
                chain.add(x); x = mg.body_parentid[x]
            self.anc[i] = [mg.jnt_bodyid[j] in chain for j in self.g_jnt]
        self.link_ids = [self.gid[b] for b, _ in PAIRS[1:]]
        self.j_chain = np.array([_chain(mg.joint(j).name) for j in self.g_jnt])
        self.l_chain = np.array([_chain(g) for _, g in PAIRS[1:]])
        fill = json.loads(FILL_JSON.read_text())["angles_rad"]
        self.fill = {b: np.array([fill[f"{b}_{a}"] for a in "xyz"]) for b in UNMAPPED}
        self.fill_R = {b: Rotation.from_euler("XYZ", v).as_matrix() for b, v in self.fill.items()}
        self.w = np.sqrt(np.repeat([1.0 if weights is None else weights.get(g, 1.0) for _, g in PAIRS[1:]], 3))
        self.q_nom, self.calib = self._calibrate()
        self.C = {b: self._g_rot(self.q_nom, b).T @ (H @ self._s_calib_rot(b)) for b, _ in PAIRS}
        leg = lambda d, m, a, z: np.linalg.norm(d.xpos[m.body(z).id] - d.xpos[m.body(a).id])
        self._fk_g(self.q_nom); self._fk_s_calib()
        self.s = leg(self.dg, mg, "left_hip_pitch_link", "left_ankle_pitch_link") / leg(self.ds, ms, "L_Hip", "L_Ankle")

    # ---- forward kinematics helpers
    def _fk_g(self, q, root_quat=(1, 0, 0, 0)):
        self.dg.qpos[:] = 0; self.dg.qpos[3:7] = root_quat; self.dg.qpos[self.g_qadr] = q
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
    def _calibrate(self):
        """G1 joint angles whose segment directions best match CALIB_POSE turned to face +x."""
        ms, ds, mg, dg = self.ms, self.ds, self.mg, self.dg
        self._fk_s_calib()
        sp = lambda b: H @ ds.xpos[ms.body(b).id]
        sax = lambda j: H @ ds.xaxis[ms.joint(j).id]
        scol = lambda b, k: H @ ds.xmat[ms.body(b).id].reshape(3, 3)[:, k]
        unit = lambda v: v / np.linalg.norm(v)
        smpl, g1 = [], []                     # (name, SMPL vector) and the matching G1 vector as f(dg)
        gp = lambda b: dg.xpos[mg.body(b).id]
        gax = lambda j: dg.xaxis[mg.joint(j).id]
        gcol = lambda b, k: dg.xmat[mg.body(b).id].reshape(3, 3)[:, k]
        for S, s in SIDES:
            pairs = [
                ("thigh", unit(sp(f"{S}_Knee") - sp(f"{S}_Hip")), lambda s=s: unit(gp(f"{s}_knee_link") - gp(f"{s}_hip_pitch_link"))),
                ("shank", unit(sp(f"{S}_Ankle") - sp(f"{S}_Knee")), lambda s=s: unit(gp(f"{s}_ankle_pitch_link") - gp(f"{s}_knee_link"))),
                ("knee axis", sax(f"{S}_Knee_x"), lambda s=s: gax(f"{s}_knee_joint")),
                ("foot fwd", scol(f"{S}_Ankle", 2), lambda s=s: gcol(f"{s}_ankle_roll_link", 0)),
                ("foot up", scol(f"{S}_Ankle", 1), lambda s=s: gcol(f"{s}_ankle_roll_link", 2)),
                # the G1's shoulder and elbow joint origins sit off the bone (4 cm medial, 1.6 cm forward), so its
                # upper arm and forearm directions are the hinges that run along them: shoulder yaw, wrist roll
                ("upper arm axis", unit(sp(f"{S}_Elbow") - sp(f"{S}_Shoulder")), lambda s=s: gax(f"{s}_shoulder_yaw_joint")),
                ("forearm axis", unit(sp(f"{S}_Wrist") - sp(f"{S}_Elbow")), lambda s=s: gax(f"{s}_wrist_roll_joint")),
                ("hand", unit(sp(f"{S}_Hand") - sp(f"{S}_Wrist")), lambda s=s: gcol(f"{s}_wrist_yaw_link", 0)),
                ("elbow axis", sax(f"{S}_Elbow_y"), lambda s=s: gax(f"{s}_elbow_joint")),
            ]
            for n, v, f in pairs:
                smpl.append((f"{S} {n}", v)); g1.append(f)
        mid = lambda: (gp("left_shoulder_pitch_link") + gp("right_shoulder_pitch_link")) / 2
        smpl += [("torso up", unit(sp("Neck") - sp("Pelvis"))), ("shoulder line", unit(sp("L_Shoulder") - sp("R_Shoulder")))]
        g1 += [lambda: unit(mid() - gp("pelvis")),
               lambda: unit(gp("left_shoulder_pitch_link") - gp("right_shoulder_pitch_link"))]
        target = np.stack([v for _, v in smpl])
        is_axis = np.array(["axis" in n for n, _ in smpl])

        def res(q):
            self._fk_g(q)
            G = np.stack([f() for f in g1])
            r = np.where(is_axis[:, None], np.cross(G, target), G - target)    # hinge axes: up to sign
            return np.concatenate([r.ravel(), 0.05 * q])

        x0 = np.zeros(len(self.g_jnt))
        for side in ("left", "right"):
            x0[self.g_jnt.index(mg.joint(f"{side}_elbow_joint").id)] = np.pi / 2     # G1 elbow straight at +90 deg
        lo, hi = self.g_range[:, 0], self.g_range[:, 1]
        sol = least_squares(res, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
        self._fk_g(sol.x)
        err = {n: float(np.degrees(np.arccos(np.clip(abs(f() @ v) if "axis" in n else f() @ v, -1, 1))))
               for (n, v), f in zip(smpl, g1)}
        return sol.x, err

    # ---- G1 -> SMPL
    def smpl_rots_from_g1(self, qpos_g):
        """{SMPL body: world rotation} for the paired bodies, from a G1 qpos."""
        self.dg.qpos[:] = qpos_g; mujoco.mj_kinematics(self.mg, self.dg)
        return {b: self.dg.xmat[self.gid[b]].reshape(3, 3) @ self.C[b] for b, _ in PAIRS}

    def g1_to_smpl_qpos(self, qpos_g):
        R = self.smpl_rots_from_g1(qpos_g)
        rel = {}
        third = Rotation.from_rotvec(_rotvec(R["Pelvis"].T @ R["Chest"]) / 3).as_matrix()
        rel["Torso"] = rel["Spine"] = rel["Chest"] = third
        for S, _ in SIDES:
            rel[f"{S}_Hip"] = R["Pelvis"].T @ R[f"{S}_Hip"]
            rel[f"{S}_Knee"] = R[f"{S}_Hip"].T @ R[f"{S}_Knee"]
            rel[f"{S}_Ankle"] = R[f"{S}_Knee"].T @ R[f"{S}_Ankle"]
            rel[f"{S}_Shoulder"] = self.fill_R[f"{S}_Thorax"].T @ R["Chest"].T @ R[f"{S}_Shoulder"]
            rel[f"{S}_Elbow"] = R[f"{S}_Shoulder"].T @ R[f"{S}_Elbow"]
            rel[f"{S}_Wrist"] = np.eye(3)                          # the wrist capsule rides with the forearm
            rel[f"{S}_Hand"] = R[f"{S}_Elbow"].T @ R[f"{S}_Hand"]
        q = self.ms.qpos0.copy()
        q[3:7] = _quat(R["Pelvis"])
        for body, v in self.fill.items():
            q[[self.s_qadr[self.s_names.index(f"{body}_{ax}")] for ax in "xyz"]] = v
        bodies = list(rel)
        ang = Rotation.from_matrix(np.stack([rel[b] for b in bodies])).as_euler("XYZ")   # R = Rx(a) Ry(b) Rz(c)
        q[[self.s_qadr[self.s_names.index(f"{b}_{ax}")] for b in bodies for ax in "xyz"]] = ang.ravel()
        clearance = self.low_g(self.dg)
        q[:2] = qpos_g[:2] / self.s
        q[2] = 0.0
        self.ds.qpos[:] = q; mujoco.mj_kinematics(self.ms, self.ds)
        q[2] = clearance / self.s - self.low_s(self.ds)
        return q

    # ---- SMPL -> G1
    def smpl_to_g1_joints(self, qpos_s, q_init=None, iters=50, damping=1e-4, tol=1e-7):
        """G1 hinge angles (29,) whose paired link orientations best match the SMPL pose qpos_s (only its root
        orientation and hinge angles are used), and the per-link residual (deg)."""
        self.ds.qpos[:] = qpos_s; mujoco.mj_kinematics(self.ms, self.ds)
        Rt = [self.ds.xmat[self.sid[b]].reshape(3, 3) @ self.C[b].T for b, _ in PAIRS]
        root = _quat(Rt[0])
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
        """(links, 3) world rotation vectors taking each paired G1 link to its target (one batched conversion:
        12 separate scipy calls per iteration were 80% of a rollout step)."""
        self._fk_g(q, root)
        Rc = self.dg.xmat[self.link_ids].reshape(-1, 3, 3)
        return Rotation.from_matrix(np.stack(Rt[1:]) @ Rc.transpose(0, 2, 1)).as_rotvec()

    def _solve(self, Rt, root, q0, iters, damping, tol):
        q = np.array(q0, dtype=float)
        lo, hi = self.g_range[:, 0], self.g_range[:, 1]
        for _ in range(iters):
            e = self._link_err(Rt, root, q).ravel()
            ax = self.dg.xaxis[self.g_jnt]                                        # (29, 3) world joint axes
            J = (self.anc[:, None, :] * ax.T[None]).reshape(-1, len(self.g_jnt))  # (3 * links, 29)
            J, e = J * self.w[:, None], e * self.w
            dq = np.linalg.solve(J.T @ J + damping * np.eye(len(q)), J.T @ e)
            q = np.clip(q + dq, lo, hi)
            if dq @ dq < tol:
                break
        return q

    def _resid(self, Rt, root, q):
        return np.degrees(np.linalg.norm(self._link_err(Rt, root, q), axis=1))

    def smpl_to_g1_qpos(self, qpos_s, q_init=None):
        q, resid = self.smpl_to_g1_joints(qpos_s, q_init)
        self.ds.qpos[:] = qpos_s; mujoco.mj_kinematics(self.ms, self.ds)
        clearance = self.low_s(self.ds)
        out = np.zeros(self.mg.nq)
        out[3:7] = _quat(self.ds.xmat[self.sid["Pelvis"]].reshape(3, 3) @ self.C["Pelvis"].T)
        out[self.g_qadr] = q
        out[:2] = np.asarray(qpos_s[:2]) * self.s
        self.dg.qpos[:] = out; mujoco.mj_kinematics(self.mg, self.dg)
        out[2] = clearance * self.s - self.low_g(self.dg)
        return out, resid
