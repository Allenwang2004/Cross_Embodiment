"""rollout.py -- BatchRollout: step a fixed batch of simulated bodies with given latents (OOP refactor step 5, docs/refactor/PLAN.md).

One implementation of the loop every search and evaluation script had a copy of:

    first = sim.reset(init)                                   # HumenvSim / RobotSim (crossenbodiment/sim.py)
    obs   = mapper.start(first, init)                         # RawObs / MultiplierObs / ExactObs (crossenbodiment/obs.py)
    for t in range(steps):
        u   = policy.act(obs, z[:, t] if a sequence else z)   # MotivoPolicy (crossenbodiment/policy.py)
        out = sim.step(u)
        obs = mapper.step(out)
        record qpos / qvel / act / obs (the mapped one, what B scores) / qpos_smpl / fz / extra mappings of the new state

    batch = BatchRollout(policy, sim, mapper, n_slots, steps, record=("qpos", "obs"))
    ro = batch.run(Z, init_qpos=ref[0], init_qvel=v0)        # Z: (n, 256) one latent per slot, or (n, steps, 256) a sequence
    ro.qpos (n, steps, nq), ro.obs (n, steps, 358), ...       # np.stack(..., axis=1) of the per-step arrays, dtypes as they come

Replaced copies (each script keeps its CLI and output files): single_z_search.rollout (and score), lso_search's and
robot_lso_search's rollout closures, robot_z_search's score loop, child_reward_eval.rollout, rank_initial_cost.rollout_multi,
test_exact_obs's loop; rollout_z_trace / score_z_matrix (and batch_z_search, archived in step 6) reach it through single_z_search.rollout and
rank_initial_cost.rollout_multi. Not replaced (outside step 5's list; each is its own analysis loop): g1_z_search.py and
g1_feasibility_compare.py (the G1 line on model/g1_env, retired / merged in steps 6-7), robot_z0.py (crossenbodiment/robot_rollout.act:
one observation computed in a batch of 16 identical rows, whose batch size changes the GPU's last bits), robot_feasibility_compare.py
(16 robot slots of which slot 0 is kept, plus an adult single env stepped one row at a time) and robot_torque_limit.py (a vec
env of its own returning per-substep torques).

Bit-identity (PLAN.md rule 2): the loop draws no random numbers; the latents become float32 exactly as before
(policy.latents: torch.as_tensor(Z, float32) / of 16.0 * Z for the sequences); the actor gets torch.as_tensor(obs, float32);
recorded arrays are the ones the loops appended (humenv qpos copied, qvel copied, the action array the env was stepped with,
the mapped observation), stacked along axis 1, never cast (the callers cast as they did: single_z_search obs -> float32).
The batch compositions stay the callers' (the helpers below build them the way the scripts did):
  copies(z, n)            np.repeat(z[None], n, axis=0): the z0 / mean-z / final-mean rollouts (all slots one latent)
  padded(chunk, pad, n)   np.concatenate([chunk, np.repeat(pad[None], n - len(chunk), axis=0)]): --eval-z batches padded with z0
  with_mean(mu, samples)  np.concatenate([mu[None], samples]): LSO / TA, slot 0 = the mean sequence (n + 1 slots)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RECORDABLE = ("qpos", "qvel", "act", "obs", "qpos_smpl", "fz")


@dataclass
class Rollout:
    """one batch rollout: each recorded field (n_slots, steps, ...), `start` the simulator's SimStep after the reset (a robot's
    start qpos: start.qpos), `extra` {name: (n_slots, steps, 358)} the extra observation mappings of every new state."""
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    act: np.ndarray | None = None
    obs: np.ndarray | None = None
    qpos_smpl: np.ndarray | None = None
    fz: np.ndarray | None = None
    start: object = None
    extra: dict = field(default_factory=dict)


class BatchRollout:
    """n_slots bodies stepped together for `steps` steps.

    policy           MotivoPolicy
    sim              HumenvSim / RobotSim with n_slots slots
    mapper           ObsMapper: what the actor sees
    z_scale          None: the latents are used as given (|z| = 16); 16.0: the latent-sequence searches' unit-sphere latents
    fetch_start_obs  HumenvSim: fetch get_obs after set_physics (single_z_search always; child_reward_eval only for a raw body;
                     lso_search never) -- crossenbodiment/sim.py
    record           which per-step arrays to keep (RECORDABLE)
    extra_obs        {name: ObsMapper}: also record these mappings of every new state (test_exact_obs records the multiplier
                     and the exact observation whatever the actor saw)
    nv               the body's nv, for the zero start velocity when run() gets init_qpos without init_qvel
    """

    def __init__(self, policy, sim, mapper, n_slots, steps, z_scale=None, fetch_start_obs=True, record=("qpos",),
                 extra_obs=None, nv=None):
        bad = set(record) - set(RECORDABLE)
        assert not bad, f"BatchRollout: cannot record {sorted(bad)}"
        self.policy, self.sim, self.mapper = policy, sim, mapper
        self.n, self.steps, self.z_scale = n_slots, steps, z_scale
        self.fetch_start_obs, self.record, self.nv = fetch_start_obs, tuple(record), nv
        self.extra_obs = dict(extra_obs or {})

    # ------------------------------------------------------------------ batch compositions (the scripts' own expressions)
    @staticmethod
    def copies(z, n):
        return np.repeat(z[None], n, axis=0)

    @staticmethod
    def padded(chunk, pad, n):
        return np.concatenate([chunk, np.repeat(pad[None], n - len(chunk), axis=0)])

    @staticmethod
    def with_mean(mu, samples):
        return np.concatenate([mu[None], samples])

    # ------------------------------------------------------------------ the rollout
    def run(self, Z, init_qpos=None, init_qvel=None, init_next=None, per_slot=False):
        """Z: numpy (n, 256) / (n, steps, 256) (float64; made float32 by policy.latents) or a ready float32 tensor.
        init_qpos: the start pose (humenv: the body's qpos, broadcast to every slot, or one per slot with per_slot; robot:
        the SMPL pose); init_qvel: humenv start velocity (default zeros(nv) when init_qpos is given); init_next: a robot's
        next start frame (start moving). Returns a Rollout."""
        z = self.policy.latents(Z, self.z_scale)
        assert z.shape[0] == self.n, f"BatchRollout: {z.shape[0]} latents for {self.n} slots"
        seq = z.dim() == 3
        if init_qpos is not None and init_qvel is None and self.nv is not None:     # humenv: start at rest
            init_qvel = [np.zeros(self.nv) for _ in init_qpos] if per_slot else np.zeros(self.nv)
        first = self.sim.reset(init_qpos, init_qvel=init_qvel, init_next=init_next, per_slot=per_slot,
                               fetch_obs=self.fetch_start_obs)
        obs = self.mapper.start(first, init_qpos, init_qvel, self.n, per_slot)
        rec = {k: [] for k in self.record}
        ext = {k: [] for k in self.extra_obs}
        want_q, want_v, want_a, want_o = "qpos" in rec, "qvel" in rec, "act" in rec, "obs" in rec
        want_s, want_f = "qpos_smpl" in rec, "fz" in rec
        for t in range(self.steps):
            u = self.policy.act(obs, z[:, t] if seq else z)
            out = self.sim.step(u)
            if want_a:
                rec["act"].append(u)
            if want_q:
                rec["qpos"].append(out.qpos)
            if want_v:
                rec["qvel"].append(out.qvel.copy())
            if want_s:
                rec["qpos_smpl"].append(out.qpos_smpl)
            if want_f:
                rec["fz"].append(out.fz)
            obs = self.mapper.step(out)
            if want_o:
                rec["obs"].append(obs)
            for k, m in self.extra_obs.items():
                ext[k].append(m.step(out))
        ro = Rollout(start=first, extra={k: np.stack(v, axis=1) for k, v in ext.items()})
        for k, v in rec.items():
            setattr(ro, k, np.stack(v, axis=1))
        return ro
