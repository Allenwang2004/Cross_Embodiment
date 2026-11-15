"""sim.py -- the simulators a batch rollout steps: HumenvSim and RobotSim (OOP refactor step 5, docs/refactor/PLAN.md).

Both take a batch of actions per step and return one SimStep (numpy arrays over the slots):

                  proprio (actor obs source)       qpos               qvel          exact             qpos_smpl / fz
  HumenvSim       obs["proprio"]                   info["qpos"].copy() info["qvel"]  info["exact_proprio"] (ExactObsWrapper)  -
  RobotSim        the SMPL-equivalent observation  the robot's qpos    -             -                 SMPL-equivalent qpos / IK residuals

HumenvSim wraps humenv's vector env (make_humenv(num_envs=n, vectorization_mode="async", task=None, xml, state_init="Default",
wrappers=[FloorFrictionWrapper (when a floor friction is given), then the observation mapper's wrappers], max_episode_steps only
when the script passed one: single_z_search max(--steps, 300), lso_search the rolled-out steps + 1; humenv's default 300
otherwise). reset() is the scripts' sequence of calls, and which of them a script made is an argument, not a guess:
    env.reset()
    env.call("set_physics", qpos=init_qpos, qvel=init_qvel)       # one start pose for every slot (broadcast), or
    per-slot set_physics over the workers' pipes (sync env: on each sub-env)  # rank_initial_cost / train_es.set_init_qpos
    np.stack(env.call("get_obs") proprio)                          # fetch_obs: single_z_search always, child_reward_eval for a
                                                                   # raw body, rank_initial_cost / test_exact_obs; not lso_search
Without init_qpos (single_z_search --init default) the reset's own observation is the start.

RobotSim wraps model/robot_env.RobotVecEnv (n RobotEnv worker processes, "spawn"): construct it BEFORE CUDA is touched (before
MotivoPolicy.load), as robot_z_search / robot_lso_search do. reset(init_qpos, init_next): the SMPL start pose mapped onto the
robot at rest, or with the next frame moving at the velocity between the two mapped poses (--init-vel reference); the IK warm
start (RobotEnv.q_tgt) restarts from the reset pose inside the workers. The G1-only model/g1_env was archived in step 7 (the G1
goes through RobotEnv, which reproduces it with the per-joint gains: tests/test_g1_generic.py).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np


@dataclass
class SimStep:
    """one reset / step of every slot (see the module docstring for what each field is per simulator)"""
    proprio: np.ndarray | None = None
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    exact: np.ndarray | None = None
    qpos_smpl: np.ndarray | None = None
    fz: np.ndarray | None = None


class HumenvSim:
    """humenv's vector env (async workers) of one SMPL-topology body."""

    def __init__(self, env):
        self.env = env

    @classmethod
    def make(cls, xml, n, wrappers=None, floor_friction=None, max_episode_steps=None, vectorization_mode="async"):
        """make_humenv as the scripts call it (module docstring). wrappers: the mapper's (ObsMapper.wrappers()); with
        floor_friction, model/friction.FloorFrictionWrapper goes first."""
        from humenv import make_humenv
        wrappers = list(wrappers or [])
        if floor_friction is not None:
            from model.friction import FloorFrictionWrapper
            wrappers = [functools.partial(FloorFrictionWrapper, mu=floor_friction)] + wrappers
        kw = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
        env, _ = make_humenv(num_envs=n, vectorization_mode=vectorization_mode, task=None, xml=str(xml), state_init="Default",
                             wrappers=wrappers or None, **kw)
        return cls(env)

    @property
    def n(self):
        return self.env.num_envs

    def set_physics_per_slot(self, qpos_list, qvel_list):
        """every slot its own start state (AsyncVectorEnv.call broadcasts one argument; this sends each worker its own over
        the same pipe protocol, or sets each sub-env of a SyncVectorEnv) -- rank_initial_cost.set_physics_per_slot and
        model/simple/train_es.set_init_qpos, whose order of sends and receives this keeps."""
        env = self.env
        if hasattr(env, "envs"):                                  # SyncVectorEnv
            for e, q, v in zip(env.envs, qpos_list, qvel_list):
                e.unwrapped.set_physics(qpos=q, qvel=v)
        elif hasattr(env, "parent_pipes"):                        # AsyncVectorEnv
            for pipe, q, v in zip(env.parent_pipes, qpos_list, qvel_list):
                pipe.send(("_call", ("set_physics", (), {"qpos": q, "qvel": v})))
            for pipe in env.parent_pipes:
                _, ok = pipe.recv()
                assert ok, "set_physics failed in a worker"
        else:
            raise SystemExit(f"cannot set per-slot physics on {type(env).__name__}")

    def get_obs(self):
        return np.stack([o["proprio"] for o in self.env.call("get_obs")])

    def reset(self, init_qpos=None, init_qvel=None, init_next=None, per_slot=False, fetch_obs=True):
        """init_qpos (broadcast, or one per slot with per_slot) with its init_qvel (explicit: the scripts always passed one);
        without init_qpos humenv's own reset is the start. init_next is RobotSim's (not used here)."""
        assert init_next is None, "HumenvSim.reset: the start velocity is init_qvel"
        obs, _ = self.env.reset()
        if init_qpos is None:
            return SimStep(proprio=obs["proprio"])
        assert init_qvel is not None, "HumenvSim.reset: pass init_qvel (or BatchRollout(nv=...)) with init_qpos"
        if per_slot:
            self.set_physics_per_slot(init_qpos, init_qvel)
        else:
            self.env.call("set_physics", qpos=init_qpos, qvel=init_qvel)
        return SimStep(proprio=self.get_obs() if fetch_obs else None)

    def step(self, u):
        obs, _, _, _, info = self.env.step(u)
        return SimStep(proprio=obs["proprio"], qpos=info["qpos"].copy(), qvel=np.asarray(info["qvel"]),
                       exact=info.get("exact_proprio"))

    def close(self):
        self.env.close()


class RobotSim:
    """model/robot_env.RobotVecEnv: n robot worker processes, SMPL observations / actions in and out."""

    def __init__(self, robot, n, fill="fixed", gains=None, vec=None):
        if vec is None:
            from model.robot_env import RobotVecEnv
            vec = RobotVecEnv(robot, n, fill=fill, gains=gains)
        self.vec, self.robot, self._n = vec, robot, n

    @property
    def n(self):
        return self._n

    def reset(self, init_qpos, init_qvel=None, init_next=None, per_slot=False, fetch_obs=True):
        """init_qpos: the SMPL start pose; init_next: its next frame (start moving), else at rest. init_qvel / per_slot /
        fetch_obs are HumenvSim's (not used: the robot's velocity comes from init_next, the observation always)."""
        assert init_qvel is None and not per_slot, "RobotSim.reset: the start velocity comes from init_next"
        obs, qs, qg, fz = self.vec.reset((init_qpos, init_next) if init_next is not None else init_qpos)
        return SimStep(proprio=obs, qpos=qg, qpos_smpl=qs, fz=fz)

    def step(self, u):
        obs, qs, qg, fz = self.vec.step(u)
        return SimStep(proprio=obs, qpos=qg, qpos_smpl=qs, fz=fz)

    def close(self):
        self.vec.close()
