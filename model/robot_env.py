"""robot_env.py -- model/g1_env.py for any robot in model/robots.py: Meta Motivo's SMPL observations and actions in,
the robot's physics inside.

Timing is humenv's (450 Hz physics, 15 substeps per action, implicitfast). Per action:
  action    Motivo's 69 SMPL PD targets, each with the SMPL passive spring folded in (theta_eff = kp theta* /
            (kp + k_s), the SMPL joint's static rest), go through RobotMap.smpl_to_robot_joints (warm-started) and
            become the robot's position targets.
  actuators every actuated joint becomes a position servo: kp, kd from the robot's per-joint gains
            (scripts/robot_gain_scaling.py: kp = k x SMPL motor kp; motor kd giving the SMPL hinge's damping ratio
            together with the robot's own passive damping). The robot keeps its own armature, friction loss, passive
            damping and torque limits (actuator force range, else joint actuator force range, else a torque motor's
            control range).
  obs       the robot state after the 15th and 14th substeps -> RobotMap.robot_to_smpl_qpos -> humenv's observation
            on the adult SMPL body, qvel by finite difference over one substep.
G1 through this class matches model/g1_env.py with gains="perjoint".
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from model.robot_map import RobotMap, SMPL_XML
from model.robots import REPO, ROBOTS

SIM_DT, N_SUB = 1.0 / 450.0, 15


def torque_limits(m):
    """per actuator: the robot's own torque limit (N m)."""
    lim = np.zeros(m.nu)
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        if m.actuator_forcelimited[i]:
            lim[i] = m.actuator_forcerange[i, 1]
        elif m.jnt_actfrclimited[j]:
            lim[i] = m.jnt_actfrcrange[j, 1]
        else:                                  # a torque motor: its control range is the torque range
            lim[i] = np.abs(m.actuator_ctrlrange[i]).max() * abs(m.actuator_gear[i, 0])
    return lim


class RobotEnv:
    def __init__(self, name, rmap=None, gains=None, fill="fixed"):
        from humenv import make_humenv
        self.name, self.cfg = name, ROBOTS[name]
        self.m = mujoco.MjModel.from_xml_path(str(REPO / self.cfg["scene"])); self.d = mujoco.MjData(self.m)
        m = self.m
        m.opt.timestep = SIM_DT
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.map = rmap or RobotMap(name, fill=fill)        # fill: robot -> SMPL, RobotMap's fill
        self.adult, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / SMPL_XML), state_init="Default")
        ms = self.ms = self.adult.unwrapped.model
        trn = ms.actuator_trnid[:, 0]
        self.s_kp = -ms.actuator_biasprm[:, 1]
        self.s_gain, self.s_bias = ms.actuator_gainprm[:, 0], ms.actuator_biasprm[:, 0]
        self.s_spring = ms.jnt_stiffness[trn]
        self.s_qadr = ms.jnt_qposadr[trn]
        # actuator i drives map joint act_j[i]
        jnt_of = {mj: k for k, mj in enumerate(self.map.g_jnt)}
        self.act_j = np.array([jnt_of[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        g = json.loads((REPO / (gains or self.cfg["gains_json"])).read_text())
        names = [m.actuator(i).name for i in range(m.nu)]
        self.kp = np.array([g["kp"][n] for n in names]); self.kd = np.array([g["kd"][n] for n in names])
        if "passive_damping" in g:              # --damping smpl gains: the SMPL-split passive damping replaces the robot's
            for i, n in enumerate(names):
                m.dof_damping[m.jnt_dofadr[m.actuator_trnid[i, 0]]] = g["passive_damping"][n]
        self.lim = torque_limits(m)
        # torque motors (H1) become position servos: control range = joint range, their control range becomes the
        # actuator force range. Position actuators keep their own ranges and the robot's way of limiting torque
        # (actuator or joint force range), so the G1 matches model/g1_env.py bit for bit
        motor = np.array([m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE for i in range(m.nu)])
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[:] = 0; m.actuator_biasprm[:] = 0
        m.actuator_gainprm[:, 0] = self.kp
        m.actuator_biasprm[:, 1] = -self.kp
        m.actuator_biasprm[:, 2] = -self.kd
        for i in np.flatnonzero(motor):
            m.actuator_forcerange[i] = [-self.lim[i], self.lim[i]]; m.actuator_forcelimited[i] = 1
            m.actuator_ctrlrange[i] = self.map.g_range[self.act_j[i]]; m.actuator_ctrllimited[i] = 1
            m.actuator_gear[i] = 0; m.actuator_gear[i, 0] = 1
        self.q_tgt = None

    def smpl_obs(self, qpos_s, qvel_s):
        self.adult.unwrapped.set_physics(qpos=qpos_s, qvel=qvel_s)
        return self.adult.unwrapped.get_obs()["proprio"].astype(np.float32)

    def reset(self, smpl_qpos, smpl_qpos_next=None):
        """Start from an SMPL pose (e.g. a reference's frame 0) mapped onto the robot, at rest; with the next frame
        (1/30 s later) also given, moving at the velocity between the two mapped poses (the reference's own)."""
        qg, _ = self.map.smpl_to_robot_qpos(smpl_qpos)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qg
        vs = np.zeros(self.ms.nv)
        if smpl_qpos_next is not None:
            qg1, _ = self.map.smpl_to_robot_qpos(smpl_qpos_next, qg[self.map.g_qadr])
            mujoco.mj_differentiatePos(self.m, self.d.qvel, 1.0 / 30, qg, qg1)
            mujoco.mj_differentiatePos(self.ms, vs, 1.0 / 30, self.map.robot_to_smpl_qpos(qg),
                                       self.map.robot_to_smpl_qpos(qg1))
        mujoco.mj_forward(self.m, self.d)
        self.q_tgt = qg[self.map.g_qadr].copy()
        self.d.ctrl[:] = self.q_tgt[self.act_j]
        self.last_resid, self.last_atlim = np.zeros(len(self.map.l_chain)), 0
        qs = self.map.robot_to_smpl_qpos(self.d.qpos)
        return self.smpl_obs(qs, vs), qs

    def action_to_ctrl(self, u):
        theta = (self.s_gain * np.clip(u, -1, 1) + self.s_bias) / self.s_kp
        qs = self.ms.qpos0.copy()
        qs[self.s_qadr] = theta * self.s_kp / (self.s_kp + self.s_spring)
        self.q_tgt, self.last_resid = self.map.smpl_to_robot_joints(qs, self.q_tgt)
        lo, hi = self.map.g_range[:, 0], self.map.g_range[:, 1]
        self.last_atlim = int(np.sum((self.q_tgt <= lo + 1e-6) | (self.q_tgt >= hi - 1e-6)))
        return self.q_tgt[self.act_j]

    def step(self, u):
        self.d.ctrl[:] = self.action_to_ctrl(u)
        mujoco.mj_step(self.m, self.d, nstep=N_SUB - 1)
        q_prev = self.d.qpos.copy()
        mujoco.mj_step(self.m, self.d)
        qs_prev, qs = self.map.robot_to_smpl_qpos(q_prev), self.map.robot_to_smpl_qpos(self.d.qpos)
        vs = np.zeros(self.ms.nv)
        mujoco.mj_differentiatePos(self.ms, vs, SIM_DT, qs_prev, qs)
        return self.smpl_obs(qs, vs), qs


def _worker(conn, name, fill, gains=None):
    env = RobotEnv(name, fill=fill, gains=gains)
    while True:
        cmd, arg = conn.recv()
        if cmd == "close":
            break
        obs, qs = (env.reset(*arg) if isinstance(arg, tuple) else env.reset(arg)) if cmd == "reset" else env.step(arg)
        conn.send((obs, qs, env.d.qpos.copy(), np.append(env.last_resid, env.last_atlim)))


class RobotVecEnv:
    """n RobotEnvs in their own processes (as model/g1_env.G1VecEnv): reset / step act on all at once and return
    (obs, SMPL-equivalent qpos, robot qpos, feasibility: per paired link the IK residual (deg), then the number of
    robot targets at a joint limit)."""

    def __init__(self, name, n, fill="fixed", gains=None):
        import multiprocessing as mp
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, name, fill, gains), daemon=True); p.start()
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
