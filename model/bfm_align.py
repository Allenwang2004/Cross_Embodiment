"""BFMTrack-style alignment: compare a rollout to its reference in the BFM's
OWN latent space instead of in joint space.

Rupf et al. 2026 (paper/BFMTrack.pdf, Eq. 3) score tracking per frame as

    r_t = cos( B(s_t), B(g_t) )

where B is the behavioural foundation model's backward map, s_t the simulated
state and g_t the reference state. The loss here is 1 - mean_t r_t over the
frames both trajectories have (the same truncation rule as losses._align_length).

Why this and not the joint-space L_align (model/losses.py): on the three
single_z align runs, L_align and ReActor's loss both pick the checkpoint whose
ROOT PATH matches best (cos(z, z0) ~ 0.1, pose visibly wrong), while this loss
and the alignment-tolerant metrics (DTW, EMD) pick the checkpoint the eye
picks -- see scripts/compare_align_losses.py. d_root is 75-97% of L_align on
those runs; B() has no such term because Metamotivo's proprio obs has no
global position in it at all.

What B() must be fed. B was trained on the ADULT body's observations, so a
scaled body's metre-carrying features (root height, local body positions and
velocities) have to be canonicalised with the same obs multiplier the frozen
actor is shown (model/obs_scale.py) -- rollout() in single_z_search.py already
applies it before the actor, and reference_embeddings() applies it here. On
the child, skipping it inflates every cosine (both sides are then off the
normaliser's distribution in the same direction) and flattens the ranking.

The reference has only qpos, so its obs is produced by writing each frame into
the env with finite-difference qvel; the rollout's obs comes straight out of
the simulation with the simulator's own qvel. That asymmetry is the paper's
too (a kinematic reference against a simulated state).
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_DT = 1.0 / 30.0


def obs_from_qpos(env, qpos, dt=DEFAULT_DT, obs_mul=None):
    """(T, 358) proprio observations for a qpos sequence on a single-env
    HumEnv, with qvel from finite differences (frame 0 gets qvel 0) and the
    actor's obs rescaling applied."""
    import mujoco
    mj = env.unwrapped.model
    nv = mj.nv
    obs = np.empty((len(qpos), 358), dtype=np.float32)
    qvel = np.zeros(nv)
    for t in range(len(qpos)):
        if t > 0:
            mujoco.mj_differentiatePos(mj, qvel, dt, qpos[t - 1], qpos[t])
        env.unwrapped.set_physics(qpos=qpos[t], qvel=qvel)
        o = env.unwrapped.get_obs()["proprio"]
        obs[t] = o if obs_mul is None else o * obs_mul
    return obs


@torch.no_grad()
def embed(model, obs, device):
    """B(obs): (T, 358) -> (T, 256) numpy. `obs` must already be rescaled."""
    o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    return model.backward_map(o).cpu().numpy()


def reference_embeddings(model, env, ref_qpos, device, obs_mul=None, dt=DEFAULT_DT):
    """B(g_t) for every reference frame."""
    return embed(model, obs_from_qpos(env, ref_qpos, dt, obs_mul), device)


def cos_per_frame(Bs, Bg):
    T = min(len(Bs), len(Bg))
    a, b = Bs[:T], Bg[:T]
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def bfm_align_loss(Bs, Bg):
    """1 - mean_t cos(B(s_t), B(g_t)); 0 is a perfect match in latent space."""
    return float(1.0 - cos_per_frame(Bs, Bg).mean())


def batch_bfm_align(model, obs_batch, Bg, device):
    """obs_batch: (n, T, 358) rescaled rollout observations -> (n,) losses.
    One backward_map call for the whole batch."""
    n, T, D = obs_batch.shape
    B = embed(model, obs_batch.reshape(n * T, D), device).reshape(n, T, -1)
    return np.array([bfm_align_loss(B[i], Bg) for i in range(n)], dtype=np.float32)
