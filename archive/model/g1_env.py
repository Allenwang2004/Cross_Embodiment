"""g1_env.py -- a Unitree G1 that Meta Motivo can drive: SMPL observations in, SMPL actions in, G1 physics inside.

Timing is humenv's: physics at 450 Hz, 15 substeps per action (30 Hz control), MuJoCo implicitfast.

  action    Motivo's 69 actions u in [-1, 1] are SMPL PD targets theta* = (gain u + bias) / kp (each SMPL actuator maps
            u onto its joint's range). An SMPL joint also has a passive spring k_s towards 0, so its static equilibrium is
            theta_eff = kp theta* / (kp + k_s) and its stiffness kp + k_s. The pose with every hinge at theta_eff goes
            through G1Map.smpl_to_g1_joints (orientation IK, warm-started at the previous target) and the result is
            the G1 position actuators' ctrl.
  gains     each G1 joint takes the stiffness / damping of the SMPL hinge about the same body axis (GAIN_SOURCE;
            the waist's single hinge per axis gets Torso, Spine and Chest in series, each wrist hinge the SMPL
            wrist and hand hinges in series), scaled for size. Two schemes:
            "perjoint" (default; assets/robots/g1/gains_perjoint.json from scripts/g1_gain_scaling.py): kp = k x the
            SMPL MOTOR kp, k the G1 / adult gravity-moment ratio of the joint group measured on retargeted clips; kd
            keeps the SMPL hinge's damping ratio, its passive damping included. The SMPL passive spring is not added
            to the G1 (it only shifts the target, see action). "mass" (the first version): kp x (m_G1 / m_SMPL) s
            of the SMPL motor + spring, kd x (m_G1 / m_SMPL) s^1.5 of motor + passive damping, s = G1Map.s.
            Either way the armature (0.01) and torque limits stay the G1's own (actuatorfrcrange in its MJCF).
  feasibility  every action also leaves G1Env.last_resid: the IK residual (deg) of each of the 13 paired links for that
            action's target pose (how far the commanded SMPL pose is from anything the G1 can make: missing DoF and
            joint limits), and G1Env.last_atlim: how many of the 29 G1 targets sit at a joint limit.
  obs       the G1 state after the 15th substep and after the 14th go through G1Map.g1_to_smpl_qpos; qvel is their
            finite difference over one substep; humenv's own observation function on the adult SMPL body in that
            state gives the 358-d proprio (as model/exact_obs.py does for the child).
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from model.g1_map import G1Map, SMPL_XML

G1_SCENE = "assets/robots/g1/scene.xml"
G1_XML = "assets/robots/g1/g1.xml"
GAINS_JSON = "assets/robots/g1/gains_perjoint.json"
SIM_DT, N_SUB = 1.0 / 450.0, 15
# G1 joint (without side / _joint) -> SMPL hinges whose stiffness and damping it takes (several: in series)
GAIN_SOURCE = {"hip_pitch": ["Hip_x"], "hip_roll": ["Hip_z"], "hip_yaw": ["Hip_y"], "knee": ["Knee_x"],
               "ankle_pitch": ["Ankle_x"], "ankle_roll": ["Ankle_z"],
               "waist_yaw": ["Torso_y", "Spine_y", "Chest_y"], "waist_roll": ["Torso_z", "Spine_z", "Chest_z"],
               "waist_pitch": ["Torso_x", "Spine_x", "Chest_x"],
               "shoulder_pitch": ["Shoulder_x"], "shoulder_roll": ["Shoulder_z"], "shoulder_yaw": ["Shoulder_y"],
               "elbow": ["Elbow_y"],
               # the G1 hand is paired with the SMPL hand box (model/g1_map.py), so its wrist stands for the SMPL wrist
               # and hand joints together: their hinges about the same axis in series
               "wrist_roll": ["Wrist_x", "Hand_x"], "wrist_pitch": ["Wrist_z", "Hand_z"], "wrist_yaw": ["Wrist_y", "Hand_y"]}


class G1Env:
    def __init__(self, xml=G1_SCENE, gmap=None, gains="perjoint"):
        from humenv import make_humenv
        self.gains = gains
        self.m = mujoco.MjModel.from_xml_path(str(xml)); self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = SIM_DT
        self.map = gmap or G1Map(g1_xml=G1_XML)
        self.adult, _ = make_humenv(num_envs=1, task=None, xml=SMPL_XML, state_init="Default")
        ms = self.ms = self.adult.unwrapped.model
        # SMPL actuators: u -> theta_eff, and the qpos slot each one drives
        trn = ms.actuator_trnid[:, 0]
        self.s_kp = -ms.actuator_biasprm[:, 1]
        self.s_gain, self.s_bias = ms.actuator_gainprm[:, 0], ms.actuator_biasprm[:, 0]
        self.s_spring = ms.jnt_stiffness[trn]
        self.s_qadr = ms.jnt_qposadr[trn]
        K = self.s_kp + self.s_spring
        D = -ms.actuator_biasprm[:, 2] + ms.dof_damping[ms.jnt_dofadr[trn]]
        name = {ms.actuator(i).name: i for i in range(ms.nu)}
        mass_ratio = self.m.body_mass.sum() / ms.body_mass.sum()
        self.kp_scale, self.kd_scale = mass_ratio * self.map.s, mass_ratio * self.map.s ** 1.5
        self.g_qadr = np.array([self.m.jnt_qposadr[self.m.actuator_trnid[i, 0]] for i in range(self.m.nu)])
        assert np.array_equal(self.g_qadr, self.map.g_qadr), "G1 actuator order != G1Map joint order"
        self.kp, self.kd = np.zeros(self.m.nu), np.zeros(self.m.nu)
        for i in range(self.m.nu):
            j = self.m.actuator(i).name.replace("_joint", "")
            key = j.replace("left_", "").replace("right_", "")
            side = "L_" if j.startswith("left") else "R_" if j.startswith("right") else ""
            src = [name[side + h] for h in GAIN_SOURCE[key]]
            self.kp[i] = self.kp_scale / np.sum(1 / K[src])          # springs in series
            self.kd[i] = self.kd_scale / np.sum(1 / D[src])
        if gains == "perjoint":
            g = json.loads(Path(GAINS_JSON).read_text())
            self.kp = np.array([g["kp"][self.m.actuator(i).name] for i in range(self.m.nu)])
            self.kd = np.array([g["kd"][self.m.actuator(i).name] for i in range(self.m.nu)])
        else:
            assert gains == "mass", gains
        self.m.actuator_gainprm[:, 0] = self.kp
        self.m.actuator_biasprm[:, 1] = -self.kp
        self.m.actuator_biasprm[:, 2] = -self.kd
        self.q_tgt = None

    def smpl_obs(self, qpos_s, qvel_s):
        self.adult.unwrapped.set_physics(qpos=qpos_s, qvel=qvel_s)
        return self.adult.unwrapped.get_obs()["proprio"].astype(np.float32)

    def reset(self, smpl_qpos):
        """Start from an SMPL pose (e.g. a reference's frame 0) mapped onto the G1, at rest."""
        qg, _ = self.map.smpl_to_g1_qpos(smpl_qpos)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qg; mujoco.mj_forward(self.m, self.d)
        self.q_tgt = qg[self.g_qadr].copy()
        self.d.ctrl[:] = self.q_tgt
        self.last_resid, self.last_atlim = np.zeros(len(self.map.l_chain)), 0
        qs = self.map.g1_to_smpl_qpos(self.d.qpos)
        return self.smpl_obs(qs, np.zeros(self.ms.nv)), qs

    def action_to_ctrl(self, u):
        theta = (self.s_gain * np.clip(u, -1, 1) + self.s_bias) / self.s_kp
        qs = self.ms.qpos0.copy()
        qs[self.s_qadr] = theta * self.s_kp / (self.s_kp + self.s_spring)
        self.q_tgt, self.last_resid = self.map.smpl_to_g1_joints(qs, self.q_tgt)
        lo, hi = self.map.g_range[:, 0], self.map.g_range[:, 1]
        self.last_atlim = int(np.sum((self.q_tgt <= lo + 1e-6) | (self.q_tgt >= hi - 1e-6)))
        return self.q_tgt

    def step(self, u):
        self.d.ctrl[:] = self.action_to_ctrl(u)
        mujoco.mj_step(self.m, self.d, nstep=N_SUB - 1)
        q_prev = self.d.qpos.copy()
        mujoco.mj_step(self.m, self.d)
        qs_prev, qs = self.map.g1_to_smpl_qpos(q_prev), self.map.g1_to_smpl_qpos(self.d.qpos)
        vs = np.zeros(self.ms.nv)
        mujoco.mj_differentiatePos(self.ms, vs, SIM_DT, qs_prev, qs)
        return self.smpl_obs(qs, vs), qs


def _worker(conn, gains):
    env = G1Env(gains=gains)
    while True:
        cmd, arg = conn.recv()
        if cmd == "close":
            break
        obs, qs = env.reset(arg) if cmd == "reset" else env.step(arg)
        conn.send((obs, qs, env.d.qpos.copy(), np.append(env.last_resid, env.last_atlim)))


class G1VecEnv:
    """n G1Envs in their own processes (spawned, one MuJoCo / numpy thread each); reset / step act on all at once
    and return (obs (n, 358), SMPL-equivalent qpos (n, 76), G1 qpos (n, 36), feasibility (n, 14): the 13 links' IK
    residuals (deg) of the action just applied, then the number of G1 targets at a joint limit)."""

    def __init__(self, n, gains="perjoint"):
        import multiprocessing as mp
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, gains), daemon=True); p.start()
            self.conns.append(a); self.procs.append(p)

    def _gather(self):
        o, qs, qg, fz = zip(*[c.recv() for c in self.conns])
        return np.stack(o), np.stack(qs), np.stack(qg), np.stack(fz)

    def reset(self, smpl_qpos):
        for c in self.conns:
            c.send(("reset", smpl_qpos))
        return self._gather()

    def step(self, U):
        for c, u in zip(self.conns, U):
            c.send(("step", u))
        return self._gather()

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
