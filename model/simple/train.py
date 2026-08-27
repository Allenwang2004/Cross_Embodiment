"""Single-body cross-embodiment adapter training (model.md's "單一身材訓練流程").

Pipeline per update (batched across `cfg.batch_size` parallel episodes on one
vectorized HumEnv, see "Batching" below):
    z_beta = G_theta(beta, z0)                      # LatentAdapter
    a_t    = actor(scale(obs_t), z_beta)            # frozen actor, z is the only knob
    tau_beta = env.step(a_t) for t in 0..T           # rollout on the target body

    L = lambda_align * L_align(tau_beta, tau_beta_ref)
        + lambda_z * (1 - cos(z_beta, z0)) + lambda_phys * L_phys(tau_beta)

One body per update
--------------------
The dataset is a (clip x body) cross product -- 540 clips x 10 bodies, built by
scripts/build_dataset.py over the per-body artifacts from docs/new_body.md --
and beta therefore VARIES, which is the whole point of conditioning G_theta on
it. Each update picks ONE body (cfg.body_order: "cycle" walks the 8 training
bodies in order, "random" draws i.i.d.) and every episode in that batch is that
body, so the batch shares one env, one fk model and one obs multiplier. All ten
envs are built up front: measured 1.2 s and 0.29 GB for the set, cheaper than
rebuilding on every body change.

Two things this forces:

  * The advantage baseline is PER BODY. Cost scale is a body property -- L_phys
    on `giant` and on `petite` are not the same number for the same quality of
    rollout -- so one global EMA would make the advantage mostly encode "which
    body is this update". That is exactly the confound
    model/simple/diagnose_single_task.py was written to chase across tasks, one
    axis over.

  * The body split is enforced here, not just recorded. Only the bodies in
    splits/train_bodies.txt are ever sampled; `giant` and `short_stocky` are
    held out for extrapolation (heavier and taller than anything trained on) and
    a run prints which bodies it dropped.

cfg.target_xml is no longer read by this file -- the body comes from each row.

z_beta is the only thing learned
---------------------------------
An earlier version also trained an ActionHead -- a beta-conditioned residual
added to the frozen actor's action mean -- and optimized it jointly with the
adapter. It is gone. Everything the target body needs now has to be expressed
inside the latent the frozen actor was built to be steered by, so what is
learned is a point in FB's task space rather than a per-body patch on top of its
output. Consequences worth knowing: checkpoints no longer carry an "action_head"
entry (old ones are refused by model/simple/evaluate.py rather than silently
evaluated without their head), and the exploration noise is now added directly
to the frozen actor's mean, which is still differentiable in z_beta, so the
REINFORCE path below is unchanged.

z_beta stays on the FB sphere
------------------------------
Every z FB produces has norm exactly sqrt(z_dim) = 16 -- metamotivo calls
project_z on the way out of sample_z / reward_inference / goal_inference /
tracking_inference. It does NOT call it on the way in: actor()/_actor() take z
as given, and Actor.forward simply concatenates [obs, z] into an MLP, so an
off-sphere z is consumed as-is. z0 + alpha*MLP(...) is off-sphere in general,
so cfg.adapter_project_z is on and the adapter re-normalizes.

The drift it prevents is anisotropic, which is why it went unnoticed while the
ActionHead was there to absorb it. In 256 dims a generic delta is nearly
orthogonal to z0, and an orthogonal delta moves the RADIUS only at second order
(measured at init: ||delta|| = 0.064 puts ||z_beta|| at 15.994..16.005, 0.01%
off; even a 10x larger delta only reaches 16.014). A delta with a RADIAL
component is first order: delta = -0.1*z0 lands at 14.4, a 10% error. So the
failure mode is not "the delta grew" but "the delta learned to point along z0",
which nothing in the objective discourages.

Which is also why the anchor is now 1 - cos(z_beta, z0) rather than
||z_beta - z0||^2: once the radius is fixed by projection, the Euclidean form
spends part of itself on a distance that can no longer change. Same form as
model/bilevel/ppo.py's lambda_z anchor. It is still a plain backprop term (see
below) -- the projection is differentiable and changes nothing about that.

Scaled obs
----------
The frozen actor is shown a canonicalised obs: every length-carrying feature
divided by its own body's length ratio against the adult, which is exactly
`scripts/rollout_z_on_body.py --obs-scale auto`. model/obs_scale.py holds the
one implementation and the measured ratios. Without it, the actor's obs
normalizer -- a BatchNorm carrying adult-scale running statistics -- sees the
child's 70 metre-carrying features at a systematic offset, and z_beta would have
to spend itself undoing a unit conversion before it can say anything about the
body's actual dynamics. What is left after the canonicalisation is the genuine
mismatch -- masses, inertias, actuator authority, and limb lengths the adult
actor's motions are still calibrated for -- and that is what beta has to
explain. ONLY the actor's view is rescaled: the physics, the qpos trajectory,
and everything L_align/L_phys score run on the real body. cfg.obs_scale = "none"
restores the old raw-obs behaviour for the ablation.

No R_task term
---------------
Earlier versions included -R_task (the humenv reward for the sampled task).
Removed: humenv's reward functions are written against the ORIGINAL body's
kinematics/actuator strength (e.g. move-ego's reward integrates an ego-frame
velocity computed for the source skeleton's proportions) -- exactly what the
LatentAdapter is changing. Optimizing directly against R_task on the target
body either rewards a number that no longer means "did the task" once the
body has changed, or implicitly pressures the adapter to fight the
morphology change to look more like the source body again, defeating the
point of adapting at all. L_align (functional equivalence vs the retargeted
reference motion) and L_phys (feasibility) don't have this problem -- both
are computed straight from the target skeleton's own forward kinematics, not
from a reward tuned for a different body. (R_task is still reported as a
diagnostic in model/simple/evaluate.py and model/simple/baseline.py -- just not optimized.)

Per-step credit, not one scalar per episode
-------------------------------------------
The first multi-body run (400 updates, W&B `cycle-400-8bodies`) learned nothing
measurable: cos(z_beta, z0) stayed at 1.0000 for all 400 updates, ||z_beta - z0||
went 0.064 -> 0.078 (initialization noise is 0.064), and the pre-clip gradient
norm sat at ~3e-3 against a clip of 5.0. What moved was L_align's TAIL, not its median
(median 6.60 -> 7.40, p90 18.9 -> 37.7), i.e. sampling noise on a heavy-tailed
cost, not degradation.

The cause is the one model/bilevel/ppo.py's docstring names: episode-level
REINFORCE gives ONE scalar of learning signal for 300 x 69 = 20700 sampled
dimensions. The old rollout divided the summed log-prob by exactly that 20700 to
stop the score function's own magnitude from swamping a single episode-level
advantage -- which is a real problem, but the fix left nothing behind.

So the objective is now bilevel's lower level, minus the parts that belong to
the bilevel design (AMP, behaviour cloning, the root wrench):

  * per-WINDOW reward. losses.functional_equivalence / physics_penalty only
    produce a trajectory-level scalar and must not be rewritten -- that is what
    keeps these numbers comparable with the published baseline and with
    eval_bilevel.py -- so they are applied to 30-step SLICES instead. An episode
    yields 10 costs, each computed by exactly the same unmodified code.
  * GAE(gamma=0.97, lambda=0.95) over those rewards, bootstrapping V at the
    truncation and 0 at a real termination.
  * ValueNet(obs, z_beta, beta, phase) in place of the scalar EMA baseline. Its
    docstring already said this is what it is for: conditioning on (z_beta,
    beta) lets one network be the per-(clip, body) baseline that the EMA could
    not be.
  * PPO clipped surrogate, 4 epochs x 4 minibatches, z_beta recomputed inside
    the loop so the adapter is actually in the ratio. The clip is close to free
    insurance here: the policy is a steering signal into a strong frozen prior,
    and one bad update destroys the prior.

What is NOT ported: bilevel's PairAdvantageNormalizer. It is an EMA per (clip,
body) pair, which works there because the same pair is revisited for thousands
of iterations. This loop draws 16 fresh clips out of 430 per update, so nearly
every pair would be seen `fresh` and normalized by its own single sample -- the
exact failure that class's own docstring describes. ValueNet does that job here.

Why REINFORCE, not backprop-through-the-rollout
-------------------------------------------------
model._actor() itself IS differentiable w.r.t. z (verified: model.act()/
model.actor() are wrapped in @torch.no_grad(), but calling model._actor()
and model._normalize() directly is not -- gradients flow through the frozen
actor's activations into z just fine, its *weights* are just frozen).

The blocker is MuJoCo: env.step() runs physics (contacts, integration) with
no autodiff support, so L_align/L_phys -- both computed from the resulting qpos
trajectory -- cannot be backpropagated through. We use a score-function
(REINFORCE) estimator instead: sample actions from a Gaussian around the
(differentiable) mean, accumulate log-probabilities, and after the episode
weight them by the realized cost. This only needs episode-level returns, not
a differentiable simulator.

The lambda_z anchor is the one exception: it never touches the environment,
so it's added as a normal backprop term for a much lower-variance gradient on
that part of the loss. It is evaluated over the UNIQUE rows in the batch, not
per transition -- per transition it would be silently multiplied by the episode
length. Same reasoning as model/bilevel/ppo.py's.

Exploration noise
------------------
This is also the PPO behaviour policy: the ratio is computed against the same
Normal(mu, cfg.exploration_std), so exploration_std is now the policy's sigma
and not merely a perturbation. Actions are sampled from
Normal(the frozen actor's mean, cfg.exploration_std),
NOT the frozen model's own actor_std (0.2) -- that std is tuned for
single-step inference-time behavior, but here it's injected every one of
300 steps into a MuJoCo rollout, where per-step noise compounds nonlinearly
through contacts/integration. Empirically (50 updates, batch_size=16) this
made L_align/L_phys swing by an order of magnitude update to update with no
visible trend, i.e. the exploration noise's effect on the trajectory was
larger than the effect of the adapter actually changing -- the learning
signal was buried under it. cfg.exploration_std defaults much smaller (0.05).

Batching
--------
All cfg.batch_size episodes run in ONE vectorized HumEnv (gymnasium
VectorEnv under the hood, `cfg.vectorization_mode` picks sync vs async)
stepped in lockstep: every timestep is a single batched forward pass through
the frozen actor + adapter (batch dim = cfg.batch_size) and a
single env.step() call, not a Python loop over individual episodes. L_align/
L_phys still need per-trajectory forward kinematics (functional_equivalence/
physics_penalty operate on one qpos array at a time, not vectorized), so
those stay a short per-item Python loop over the collected (B, T, nq)
qpos_beta after the rollout finishes -- cheap relative to simulation, which
is now batched. Relies on the vector env's default auto-reset (a sub-episode
ending early mid-rollout just resets that slot and keeps stepping; we don't
special-case it since cfg.steps_per_episode is short enough that this is rare
and losing part of one sub-episode to a reset boundary doesn't bias the batch).

NOTE on the L_align term: it is live again. datasets/crossenbodiment-1-datasets (the
old single-body manifest) has no retargeted_motion, so qpos_ref was None on
every row and L_align was identically zero. The default dataset is now
datasets/crossenbodiment-10bodies, where every one of the 5400 rows carries a
per-body retargeted reference produced by docs/new_body.md Step 2, so
functional_equivalence has something to compare against on all of them.

qpos_ref (retargeted_motion) is produced by scripts/qpos_retarget.py, which
retargets each origin_motion trajectory directly onto the same
robot_<label>.xml skeleton used for the live rollout (target_xml in
config.py) -- so L_align is comparing two trajectories on the same bone lengths,
as intended.

Usage (from project root, once datasets/crossenbodiment-1-datasets exists):
    uv run model/simple/train.py
    uv run model/simple/run_train.py --gpu 1
"""

import dataclasses
import os
import random
from pathlib import Path
import sys

PARENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

import wandb

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.simple.config import TrainConfig
from model.dataset import CrossEmbodimentDataset, load_task_list
from model.obs_scale import build_obs_multiplier
from model.networks import LatentAdapter, ValueNet

REPO_ROOT = Path(__file__).resolve().parents[2]


def rollout_batch(model, adapter, value_net, env, z0_t, beta_t, cfg, obs_mul=None,
                  deterministic=False):
    """z0_t/beta_t: (B, z_dim)/(B, beta_dim). Steps the vectorized env
    cfg.steps_per_episode times, all B sub-envs in lockstep, and keeps
    everything PPO needs afterwards.

    Returns (T, B) tensors -- obs, action, logp, value, done -- plus the (B, T,
    nq) qpos trajectory. logp is PER STEP and summed over action dims only; it
    is NOT averaged over the episode any more. The old code divided it by
    steps * action_dim = 300 * 69 = 20700 to keep the score function from
    swamping a single episode-level advantage, and that rescale is what made the
    measured gradient ~1e-3 with a clip of 5.0. With a per-step advantage there
    is nothing to swamp: each step's log-prob is weighted by its own window's
    advantage, not by one number for the whole rollout.

    obs_mul: (358,) multiplier from model/obs_scale.py, or None for the raw
    obs. It changes ONLY what the actor is shown -- qpos_hist, and therefore
    L_align/L_phys, come from the real body either way."""
    B, T = z0_t.shape[0], cfg.steps_per_episode
    dev = cfg.device
    with torch.no_grad():
        z_beta = adapter(beta_t, z0_t)

    obs, _ = env.reset()
    qpos_hist, obs_hist, act_hist, logp_hist, val_hist, done_hist = [], [], [], [], [], []

    for t in range(T):
        proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.tensor(proprio, dtype=torch.float32, device=dev)

        with torch.no_grad():
            # Bypass model.act()/model.actor() (both @torch.no_grad()-wrapped);
            # the PPO update re-runs this path WITH grad so the ratio sees the
            # adapter's change.
            obs_norm = model._normalize(obs_t)
            action_mean = model._actor(obs_norm, z_beta, model.cfg.actor_std).mean
            # cfg.exploration_std, not model.cfg.actor_std -- see config.py's
            # comment: this noise physically compounds over the rollout, so it
            # is kept much smaller than the frozen model's own actor_std.
            action_dist = Normal(action_mean, cfg.exploration_std)
            # Held-out scoring takes the mean action: the number has to be a
            # property of the policy, not of one draw of exploration noise.
            action = action_mean if deterministic else action_dist.sample()
            logp = action_dist.log_prob(action).sum(dim=-1)          # (B,)
            val = value_net(obs_t, z_beta, beta_t, phase_of(t, T, B, dev))

        obs_hist.append(obs_t)
        act_hist.append(action)
        logp_hist.append(logp)
        val_hist.append(val)

        action_np = action.cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action_np)
        qpos_hist.append(info["qpos"].copy())
        # A fall IS a termination and bootstraps with 0; running out of the
        # 300-step budget is a TRUNCATION and bootstraps with V. The vector env
        # auto-resets, so `terminated` is the only one that means "the value of
        # what comes next is zero".
        done_hist.append(torch.as_tensor(np.asarray(terminated, dtype=np.float32), device=dev))

    # One extra value for the bootstrap at t = T.
    proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
    with torch.no_grad():
        last_v = value_net(torch.tensor(proprio, dtype=torch.float32, device=dev),
                           z_beta, beta_t, phase_of(T, T, B, dev))

    return {
        "obs": torch.stack(obs_hist),                       # (T, B, 358)
        "action": torch.stack(act_hist),                    # (T, B, act)
        "logp": torch.stack(logp_hist),                     # (T, B)
        "value": torch.cat([torch.stack(val_hist), last_v[None]]),   # (T+1, B)
        "done": torch.stack(done_hist),                     # (T, B)
        "qpos_beta": np.stack(qpos_hist, axis=1),           # (B, T, nq)
    }


def phase_of(t, T, B, device):
    """ValueNet's (t/T, (T-t)/T). Not optional -- see its docstring: without it
    the bootstrap at the truncation boundary is systematically mis-scaled."""
    return torch.tensor([[t / T, (T - t) / T]], dtype=torch.float32,
                        device=device).expand(B, 2)


def window_rewards(fk_model, cfg, qpos_beta, qpos_refs):
    """Per-step reward (T, B) from per-WINDOW L_align and L_phys.

    This is the port of bilevel's per-step reward that keeps model/losses.py
    untouched. losses.functional_equivalence / physics_penalty only produce a
    trajectory-level scalar, and rewriting them per-frame would break the one
    thing that makes these numbers comparable with the published baseline and
    with eval_bilevel.py. Applying them to a SLICE does not: each window is
    scored by exactly the same code, so an episode yields ~10 costs instead of
    1 while every individual number keeps its old meaning.

    Each window's cost is spread evenly over its steps (cost / len), so the
    undiscounted episode return is minus the sum of window costs and the reward
    scale does not depend on the window length.

    THE REFERENCE IS OFTEN SHORTER THAN THE ROLLOUT. Retargeted clips are 120 or
    300 frames while the episode is cfg.steps_per_episode. Whole-episode scoring
    hid this: losses._align_length silently truncated both to the shorter one.
    Per window it does not -- a window past the end of the reference slices an
    EMPTY array, _align_length returns length 0, and np.mean of nothing is NaN,
    which propagates through the advantage into the adapter. (Measured: this
    NaN'd the first 300-step PPO run inside one update.) So a window keeps its L_align
    term only over the frames the reference actually covers, and falls back to
    qpos_ref=None -- functional_equivalence's own documented "no reference"
    path, which returns 0.0 -- when there is nothing left to compare against.

    Returns (reward (T, B), align_total (B,), l_phys (B,), n_bad). align_total averages
    only the windows that HAD a reference, so it stays on the same scale as the
    old whole-episode number instead of being diluted by structural zeros.
    """
    B, T = qpos_beta.shape[0], qpos_beta.shape[1]
    W = cfg.window_steps
    d_weights = {
        "root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
        "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight,
    }
    dt = 1.0 / cfg.control_fps
    reward = np.zeros((T, B), dtype=np.float32)
    align_totals = np.zeros(B, dtype=np.float32)
    l_physes = np.zeros(B, dtype=np.float32)
    n_bad = 0

    for i in range(B):
        ref = qpos_refs[i]
        n_ref_win = n_win = 0
        for a in range(0, T, W):
            b = min(a + W, T)
            if b - a < 3:      # d_velocity/_smoothness need a difference
                continue
            # Overlap with the reference only; None past its end.
            ref_w = None
            end = b
            if ref is not None and a + 3 <= len(ref):
                end = min(b, len(ref))
                ref_w = ref[a:end]
            traj = qpos_beta[i, a:end] if ref_w is not None else qpos_beta[i, a:b]

            align_w, _ = losses.functional_equivalence(fk_model, traj, ref_w, d_weights, dt)
            # same fall reference and weights as compute_batch_cost -- if the
            # per-window reward and the episode-level cost measured "falling"
            # differently, the advantage would be optimizing a third thing
            l_w, _ = losses.physics_penalty(
                fk_model, qpos_beta[i, a:b], dt=dt,
                weights=_phys_weight_table(cfg),
                qpos_ref=ref_w if getattr(cfg, "phys_fall_ref", False) else None)
            cost = cfg.lambda_align * align_w + cfg.lambda_phys * l_w
            if not np.isfinite(cost):
                # Leave the window at reward 0 (no signal) rather than pushing a
                # NaN into the advantage. Counted and logged so a systematic
                # source cannot hide.
                n_bad += 1
                continue
            reward[a:b, i] = -cost / (b - a)
            l_physes[i] += l_w
            n_win += 1
            if ref_w is not None:
                align_totals[i] += align_w
                n_ref_win += 1
        if n_win:
            l_physes[i] /= n_win
        if n_ref_win:
            align_totals[i] /= n_ref_win
    return reward, align_totals, l_physes, n_bad


def compute_gae(reward, value, done, gamma, lam):
    """reward/done (T, B), value (T+1, B) -> (advantage, return), each (T, B).

    Copied from model/bilevel/ppo.py:compute_gae together with the reason it is
    written this way: the end of the rollout is a TRUNCATION and bootstraps with
    V(s_T), while a fall IS a termination and bootstraps with 0. Conflating them
    teaches the policy that surviving to the end of an arbitrary slice is
    worthless. Kept as a copy rather than an import so model/simple/ stays a
    standalone baseline -- the same convention model/bilevel/rewards.py:63 uses
    for its copy out of model/losses.py.
    """
    T = reward.shape[0]
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(reward[0])
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = reward[t] + gamma * value[t + 1] * nonterminal - value[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + value[:T]


def ppo_update(cfg, model, adapter, value_net, optimizer, ep, z0_t, beta_t, reward):
    """One PPO update over the collected rollout. Returns scalar metrics.

    Structure follows model/bilevel/ppo.py:update_lower -- GAE on the value
    function, advantage standardized over the batch, clipped surrogate over
    ppo_epochs x ppo_minibatches, clipped value loss, and the lambda_z cosine
    anchor evaluated over the UNIQUE rows rather than per transition so it is
    not implicitly re-weighted by episode length.
    """
    dev = cfg.device
    T, B = reward.shape

    with torch.no_grad():
        adv, ret = compute_gae(reward, ep["value"], ep["done"], cfg.gamma, cfg.gae_lambda)
        # No per-(clip, body) EMA here. bilevel needs PairAdvantageNormalizer
        # because it revisits the same pair for thousands of iterations; this
        # loop draws 16 fresh clips out of 430 every update, so a per-pair EMA
        # would be `fresh` almost every time and normalize by its own sample.
        # ValueNet(obs, z_beta, beta, phase) IS the per-pair baseline -- that is
        # what its docstring says it is for -- so one batch-wide standardization
        # on top is all that is left.
        adv_raw_std = adv.std()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_mean, ret_std = ret.mean(), ret.std().clamp(min=1e-6)
        ret_n = (ret - ret_mean) / ret_std
        v_old = (ep["value"][:T] - ret_mean) / ret_std

    def flat(x):
        return x.reshape(T * B, *x.shape[2:])

    b_obs, b_act, b_logp = flat(ep["obs"]), flat(ep["action"]), flat(ep["logp"])
    b_adv, b_ret, b_vold = flat(adv), flat(ret_n), flat(v_old)
    b_beta = beta_t.unsqueeze(0).expand(T, B, -1).reshape(T * B, -1)
    b_z0 = z0_t.unsqueeze(0).expand(T, B, -1).reshape(T * B, -1)
    b_phase = torch.stack([
        (torch.arange(T, device=dev) / T).unsqueeze(1).expand(T, B).reshape(-1),
        ((T - torch.arange(T, device=dev)) / T).unsqueeze(1).expand(T, B).reshape(-1),
    ], dim=-1)

    n = T * B
    mb = n // cfg.ppo_minibatches
    stats = {"pg": 0.0, "v": 0.0, "kl": 0.0, "clipfrac": 0.0, "grad": 0.0}
    n_steps = 0

    stopped_early = 0
    for epoch in range(cfg.ppo_epochs):
        if stopped_early:
            break
        order = torch.randperm(n, device=dev)
        for k in range(cfg.ppo_minibatches):
            idx = order[k * mb:(k + 1) * mb]

            # z_beta is RECOMPUTED, not reused from collection: the adapter is
            # the policy here, so the PPO ratio has to reflect its update.
            z_beta = adapter(b_beta[idx], b_z0[idx])
            if not torch.isfinite(z_beta).all():
                raise RuntimeError(
                    "z_beta went non-finite inside the PPO loop -- the adapter "
                    "diverged. Look at grad_norm and approx_kl on the updates "
                    "before this one; the usual cause is a step outside the "
                    "trust region (see cfg.grad_clip_norm / cfg.ppo_target_kl)."
                )
            obs_norm = model._normalize(b_obs[idx])
            mean = model._actor(obs_norm, z_beta, model.cfg.actor_std).mean
            dist = Normal(mean, cfg.exploration_std)
            logp = dist.log_prob(b_act[idx]).sum(-1)

            # Clamp the LOG ratio before exp(). With exploration_std = 0.05,
            # d logp/d mu = (a - mu)/sigma^2 amplifies by 400 per dim and the
            # log-prob is summed over 69 of them, so a small mean shift moves
            # logp by hundreds and exp() overflows to inf -- measured: the
            # 300-step rollout NaN'd the adapter inside the first update before
            # this clamp existed.
            logratio = (logp - b_logp[idx]).clamp(-20.0, 20.0)
            with torch.no_grad():
                approx_kl = (-logratio).mean()
            # Check BEFORE stepping, not after. The point of a KL guard here is
            # to not take the update that leaves the trust region; discovering
            # it afterwards is discovering it too late, because the policy is a
            # steering signal into a frozen prior that one bad step destroys.
            if cfg.ppo_target_kl and approx_kl.item() > cfg.ppo_target_kl:
                stopped_early = epoch + 1
                break

            ratio = logratio.exp()
            a = b_adv[idx]
            pg_loss = -torch.min(
                ratio * a, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * a
            ).mean()

            v = value_net(b_obs[idx], z_beta.detach(), b_beta[idx], b_phase[idx])
            v_clipped = b_vold[idx] + (v - b_vold[idx]).clamp(-cfg.value_clip, cfg.value_clip)
            v_loss = 0.5 * torch.max((v - b_ret[idx]) ** 2,
                                     (v_clipped - b_ret[idx]) ** 2).mean()

            # Over the unique rows, not the T*B transitions: otherwise the
            # anchor is silently multiplied by the episode length.
            z_uni = adapter(beta_t, z0_t)
            z_cos = (F.normalize(z_uni, dim=-1) * F.normalize(z0_t, dim=-1)).sum(-1).mean()
            z_reg_loss = cfg.lambda_z * (1.0 - z_cos)

            loss = pg_loss + cfg.value_coef * v_loss + z_reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(value_net.parameters()),
                cfg.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                stats["pg"] += pg_loss.item()
                stats["v"] += v_loss.item()
                stats["kl"] += float(approx_kl)
                stats["clipfrac"] += ((ratio - 1).abs() > cfg.ppo_clip).float().mean().item()
                stats["grad"] += float(grad)
            n_steps += 1

    out = {k: v / max(n_steps, 1) for k, v in stats.items()}
    out["ppo_steps"] = n_steps
    out["kl_stopped_epoch"] = stopped_early
    with torch.no_grad():
        z_uni = adapter(beta_t, z0_t)
        out["z_cos"] = (F.normalize(z_uni, dim=-1) * F.normalize(z0_t, dim=-1)).sum(-1).mean().item()
        out["z_norm"] = z_uni.norm(dim=-1).mean().item()
        out["z_reg_loss"] = cfg.lambda_z * (1.0 - out["z_cos"])
        out["adv_std_raw"] = float(adv_raw_std)
        out["reward_mean"] = float(reward.mean())
    return out


def _phys_weight_table(cfg):
    """cfg.phys_weights -> a weights dict, or None for physics_penalty's own
    default. See losses.PHYS_WEIGHT_TABLES."""
    w = losses.PHYS_WEIGHT_TABLES.get(getattr(cfg, "phys_weights", "default"))
    return None if w is losses.PHYS_DEFAULT_WEIGHTS else w


def compute_batch_cost(fk_model, cfg, qpos_beta, qpos_refs, return_terms=False):
    """qpos_beta: (B, T, nq) numpy. qpos_refs: length-B list, entries may be
    None (see losses.functional_equivalence). L_align/L_phys use forward kinematics
    per-trajectory (not batched), looped here since it's cheap vs simulation.
    Returns per-item cost/L_align/L_phys arrays, shape (B,); with return_terms,
    a fourth element, the length-B list of L_phys's unweighted per-term dicts
    (free -- physics_penalty already returns them, and re-deriving them costs a
    second forward-kinematics pass per trajectory)."""
    B = qpos_beta.shape[0]
    costs = np.empty(B, dtype=np.float32)
    align_totals = np.empty(B, dtype=np.float32)
    l_physes = np.empty(B, dtype=np.float32)

    d_weights = {
        "root": cfg.d_root_weight,
        "ee": cfg.d_ee_weight,
        "contact": cfg.d_contact_weight,
        "pose": cfg.d_pose_weight,
        "velocity": cfg.d_velocity_weight,
    }
    dt = 1.0 / cfg.control_fps
    phys_w = _phys_weight_table(cfg)
    fall_ref = getattr(cfg, "phys_fall_ref", False)
    term_list = []
    for i in range(B):
        align_total, _ = losses.functional_equivalence(fk_model, qpos_beta[i], qpos_refs[i],
                                                       d_weights, dt)
        l_phys, terms = losses.physics_penalty(
            fk_model, qpos_beta[i], weights=phys_w, dt=dt,
            qpos_ref=qpos_refs[i] if fall_ref else None)
        costs[i] = cfg.lambda_align * align_total + cfg.lambda_phys * l_phys
        align_totals[i] = align_total
        l_physes[i] = l_phys
        term_list.append(terms)
    if return_terms:
        return costs, align_totals, l_physes, term_list
    return costs, align_totals, l_physes


def plot_loss_curve(loss_history, align_history, l_phys_history, out_path, body_history=None):
    """total_loss (pg_loss + z_reg_loss) is NOT a progress signal for
    REINFORCE -- it's advantage-weighted log-prob, and advantage is centered
    by construction (cost minus a running mean), so its expected value
    oscillates around 0 whether or not the policy is improving. The actual
    thing to watch is L_align / L_phys (the real cost being optimized), so those
    get their own panels here rather than only total_loss.

    body_history (one label per update) splits the L_align and L_phys panels into one
    line per body. Without it a multi-body run's curve is unreadable: successive
    points are different bodies with different cost scales, so the pooled series
    is a sawtooth between bodies rather than a trend within any of them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(loss_history)
    axes[0].set_ylabel("total_loss")
    axes[0].set_title("pg_loss + z_reg_loss (surrogate, NOT expected to trend down -- see L_align/L_phys below)")

    panels = [(1, align_history, "L_align (mean)", "tab:orange",
               "functional-equivalence cost vs retargeted reference -- this SHOULD trend down"),
              (2, l_phys_history, "L_phys (mean)", "tab:green",
               "physics-feasibility penalty -- this SHOULD trend down")]
    for ax_i, series, ylabel, color, title in panels:
        ax = axes[ax_i]
        if body_history:
            for b in dict.fromkeys(body_history):
                xs = [i for i, lb in enumerate(body_history) if lb == b]
                ax.plot(xs, [series[i] for i in xs], marker=".", ms=3, lw=1, label=b)
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.plot(series, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[2].set_xlabel("update")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"wrote training curves -> {out_path}")


def make_body_ctx(cfg, dataset_dir, label, xml_rel):
    """One body's fixed resources: its vectorized env, a standalone MjModel for
    the L_align/L_phys forward kinematics, and its obs canonicaliser.

    All 10 are built once up front rather than rebuilt when the body changes --
    measured 1.2 s and 0.29 GB for all ten (about 30 MB each), which is cheaper
    than paying a rebuild every update.
    """
    xml = dataset_dir / xml_rel
    env, _ = make_humenv(
        num_envs=cfg.batch_size,
        vectorization_mode=cfg.vectorization_mode,
        task=None,
        xml=str(xml),
        state_init="Default",
    )
    return {
        "label": label,
        "env": env,
        # Separate MjModel just for L_align/L_phys forward-kinematics (numpy, no grad)
        # -- same skeleton as the env's own model, loaded standalone so it works
        # whether env is a sync or async VectorEnv.
        "fk": mujoco.MjModel.from_xml_path(str(xml)),
        # The actor's view of THIS body is canonicalised to adult scale; the
        # physics and everything scored stay on the real body. The ratios come
        # from the two rest poses and do not change during training.
        "obs_mul": build_obs_multiplier(
            xml, REPO_ROOT / cfg.obs_scale_ref_xml,
            mode=cfg.obs_scale, parts=cfg.obs_scale_parts, verbose=False,
        ),
    }


def run_eval(cfg, model, adapter, value_net, ctxs, dataset, eval_idx, splits):
    """Deterministic pass over the fixed eval clips of every body.

    Returns {body: {"L_align":, "L_phys":, "cost":}} plus "train"/"test" aggregates.
    The cost is the same lambda_align * L_align + lambda_phys * L_phys the reward is
    built from, summed over the same windows, so it is directly the quantity
    training is minimizing -- pg_loss is NOT, it is an advantage-weighted
    surrogate whose expected value is ~0 whether or not the policy improved,
    and on held-out data it means even less than that.
    """
    out = {}
    adapter.eval()
    for b, idxs in eval_idx.items():
        samples = [dataset[i] for i in idxs]
        z0_t = torch.tensor(np.stack([s["z0"] for s in samples]),
                            dtype=torch.float32, device=cfg.device)
        beta_t = torch.tensor(np.stack([s["beta"] for s in samples]),
                              dtype=torch.float32, device=cfg.device)
        qpos_refs = [s["qpos_ref"] for s in samples]
        ep = rollout_batch(model, adapter, value_net, ctxs[b]["env"],
                           z0_t, beta_t, cfg, ctxs[b]["obs_mul"], deterministic=True)
        reward, align_totals, l_physes, _ = window_rewards(
            ctxs[b]["fk"], cfg, ep["qpos_beta"], qpos_refs)
        out[b] = {
            "L_align": float(align_totals.mean()),
            "L_phys": float(l_physes.mean()),
            "cost": float(cfg.lambda_align * align_totals.mean() + cfg.lambda_phys * l_physes.mean()),
        }
    adapter.train()

    for split in ("train", "test"):
        members = [b for b in out if splits.get(b) == split]
        if members:
            out[split] = {k: float(np.mean([out[b][k] for b in members]))
                          for k in ("L_align", "L_phys", "cost")}
    return out


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_dir = REPO_ROOT / cfg.dataset_dir
    # No task filter: every task is trained on. The only held-out axis is the
    # BODY one -- see scripts/build_dataset.py's "One split axis, not two".
    dataset = CrossEmbodimentDataset(dataset_dir)

    # The BODY axis is a second, independent split. Held-out bodies must never
    # be sampled here or the extrapolation claim they exist to test is void.
    by_body = dataset.indices_by_body()
    bodies_path = dataset_dir / "splits" / "train_bodies.txt"
    if bodies_path.exists():
        bodies = [b for b in load_task_list(bodies_path) if b in by_body]
        skipped = sorted(set(by_body) - set(bodies))
        if skipped:
            print(f"held out of training ({len(skipped)}): {' '.join(skipped)}")
    else:
        # Legacy single-body manifest: one body, no body split file.
        bodies = list(by_body)
    if not bodies:
        raise SystemExit(f"no trainable bodies in {dataset_dir}")
    beta_dim = len(dataset[by_body[bodies[0]][0]]["beta"])

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()  # frozen: FBModel.__init__ already sets requires_grad_(False)

    # Envs for EVERY body, not just the trained ones -- the held-out bodies need
    # one too, and they are only ever stepped inside run_eval.
    ctxs = {}
    for b in by_body:
        xml_rel = dataset[by_body[b][0]]["target_xml"]
        if xml_rel is None:  # legacy manifest has no per-row xml
            xml_rel = Path(cfg.target_xml).resolve().relative_to(REPO_ROOT)
            ctxs[b] = make_body_ctx(cfg, REPO_ROOT, b, xml_rel)
        else:
            ctxs[b] = make_body_ctx(cfg, dataset_dir, b, xml_rel)
    splits = {b: ("train" if b in bodies else "test") for b in by_body}
    print(f"{len(bodies)} training bodies x {len(dataset) // max(1, len(by_body))} clips: "
          f"{' '.join(bodies)}")

    # The eval set is FIXED: drawn once here, reused at every eval. Its own RNG,
    # so changing eval_every cannot shift which clips training samples.
    erng = random.Random(cfg.eval_seed)
    eval_idx = {b: [erng.choice(by_body[b]) for _ in range(cfg.batch_size)]
                for b in by_body}
    n_test = sum(1 for b in splits.values() if b == "test")
    print(f"eval every {cfg.eval_every} updates on {cfg.batch_size} fixed clips "
          f"x {len(by_body)} bodies ({n_test} held out), deterministic")

    adapter = LatentAdapter(
        beta_dim=beta_dim,
        z_dim=model.cfg.archi.z_dim,
        hidden_dims=cfg.adapter_hidden_dims,
        alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable,
        project=cfg.adapter_project_z,
    ).to(cfg.device)

    # Replaces the scalar EMA baseline. Conditioning on (z_beta, beta) is what
    # lets one network stand in for a per-(clip, body) baseline -- see ValueNet's
    # docstring in model/networks.py.
    obs_dim = ctxs[bodies[0]]["env"].single_observation_space["proprio"].shape[0]
    value_net = ValueNet(obs_dim=obs_dim, z_dim=model.cfg.archi.z_dim,
                         beta_dim=beta_dim, hidden_dims=cfg.value_hidden_dims).to(cfg.device)

    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(value_net.parameters()), lr=cfg.lr)

    # EMA-tracked mean/std of the cost, used to standardize the REINFORCE
    # advantage each update: (cost - mean) / std. Centering alone (a plain
    # scalar baseline) isn't enough here -- cost's own scale drifts a lot
    # update to update (batch_size=4 is a small, noisy sample), so without
    # also normalizing by std the gradient magnitude swings wildly and
    # training doesn't converge. Both are tracked as slow EMAs (not raw
    # per-batch stats) since batch_size=4 alone is too small a sample to
    # trust for either estimate.

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config={**dataclasses.asdict(cfg), "train_bodies": bodies,
                           "n_clips_per_body": len(by_body[bodies[0]])})

    loss_history = []
    align_history = []
    l_phys_history = []
    body_history = []

    eval_history = []

    def do_eval(update):
        ev = run_eval(cfg, model, adapter, value_net, ctxs, dataset, eval_idx, splits)
        eval_history.append((update, ev))
        tr, te = ev.get("train"), ev.get("test")
        tqdm.write(
            f"  [eval @ {update:04d}] "
            + (f"train cost={tr['cost']:.4f} (L_align={tr['L_align']:.4f} Lp={tr['L_phys']:.4f})  " if tr else "")
            + (f"test cost={te['cost']:.4f} (L_align={te['L_align']:.4f} Lp={te['L_phys']:.4f})  " if te else "")
            + (f"gap={te['cost'] - tr['cost']:+.4f}" if tr and te else "")
        )
        if cfg.use_wandb:
            log = {}
            for b, v in ev.items():
                pre = f"eval/{b}" if b not in ("train", "test") else f"eval/{b}_bodies"
                log.update({f"{pre}/{k}": val for k, val in v.items()})
            if tr and te:
                log["eval/gap_cost"] = te["cost"] - tr["cost"]
            wandb.log(log, step=update)
        return ev

    if cfg.eval_every and cfg.eval_at_start:
        do_eval(0)

    pbar = tqdm(range(cfg.num_updates), desc="train", disable=not cfg.progress)
    for update in pbar:
        optimizer.zero_grad()

        # ONE body per update: every episode in the batch is the same body, so
        # the whole batch shares one env, one fk model and one obs multiplier.
        label = (bodies[update % len(bodies)] if cfg.body_order == "cycle"
                 else random.choice(bodies))
        ctx = ctxs[label]

        idx = [random.choice(by_body[label]) for _ in range(cfg.batch_size)]
        samples = [dataset[i] for i in idx]
        z0_t = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=cfg.device)
        beta_t = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=cfg.device)
        qpos_refs = [s["qpos_ref"] for s in samples]

        episode = rollout_batch(model, adapter, value_net, ctx["env"],
                                z0_t, beta_t, cfg, ctx["obs_mul"])
        reward_np, align_totals, l_physes, n_bad = window_rewards(
            ctx["fk"], cfg, episode["qpos_beta"], qpos_refs)
        reward = torch.as_tensor(reward_np, device=cfg.device)

        st = ppo_update(cfg, model, adapter, value_net, optimizer,
                        episode, z0_t, beta_t, reward)
        total_loss_v = st["pg"] + cfg.value_coef * st["v"] + st["z_reg_loss"]

        loss_history.append(total_loss_v)
        align_history.append(float(align_totals.mean()))
        l_phys_history.append(float(l_physes.mean()))
        body_history.append(label)
        pbar.set_postfix(
            body=label,
            pg=f"{st['pg']:.4f}",
            L_align=f"{align_totals.mean():.4f}",
            L_phys=f"{l_physes.mean():.4f}",
        )

        if cfg.use_wandb:
            wandb.log(
                {
                    "total_loss": total_loss_v,
                    "pg_loss": st["pg"],
                    "value_loss": st["v"],
                    "z_reg_loss": st["z_reg_loss"],
                    "approx_kl": st["kl"],
                    "clip_frac": st["clipfrac"],
                    "ppo_steps": st["ppo_steps"],
                    "grad_norm": st["grad"],
                    "adv_std_raw": st["adv_std_raw"],
                    "reward_mean": st["reward_mean"],
                    "bad_windows": n_bad,
                    "L_align": float(align_totals.mean()),
                    "L_phys": float(l_physes.mean()),
                    "z_cos": st["z_cos"],
                    "z_norm": st["z_norm"],
                    # Which body this update was. The pooled series above are a
                    # sawtooth across bodies (their cost scales differ several
                    # fold), so the per-body panels are the ones to read for a
                    # trend; each only gets a point on its own updates.
                    "body_idx": bodies.index(label),
                    f"by_body/{label}/L_align": float(align_totals.mean()),
                    f"by_body/{label}/L_phys": float(l_physes.mean()),
                    f"by_body/{label}/reward": st["reward_mean"],
                },
                step=update,
            )

        if update % cfg.log_every == 0:
            tqdm.write(
                f"[{update:04d}/{cfg.num_updates}] {label:13s} "
                f"pg={st['pg']:+.4f} v={st['v']:.4f} kl={st['kl']:+.2e} "
                f"clip={st['clipfrac']:.2f} grad={st['grad']:.3f} "
                f"1-zcos={1 - st['z_cos']:.2e} "
                f"L_align={align_totals.mean():.4f} L_phys={l_physes.mean():.4f}"
            )

        if cfg.eval_every and (update + 1) % cfg.eval_every == 0:
            do_eval(update + 1)

        if (update + 1) % cfg.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"update_{update + 1:05d}.pt"
            torch.save(
                {"adapter": adapter.state_dict(), "value_net": value_net.state_dict(),
                 "update": update + 1, "cfg": cfg, "bodies": bodies},
                ckpt_path,
            )
            tqdm.write(f"saved checkpoint -> {ckpt_path}")

    if cfg.eval_every and (eval_history and eval_history[-1][0] != cfg.num_updates):
        do_eval(cfg.num_updates)

    for ctx in ctxs.values():
        ctx["env"].close()

    if eval_history:
        print("\n=== held-out evaluation ===")
        print(f"{'update':>7s} {'train cost':>11s} {'test cost':>10s} {'gap':>9s} "
              f"{'train L_align':>13s} {'test L_align':>13s}")
        for u, ev in eval_history:
            tr, te = ev.get("train"), ev.get("test")
            if not (tr and te):
                continue
            print(f"{u:7d} {tr['cost']:11.4f} {te['cost']:10.4f} "
                  f"{te['cost'] - tr['cost']:+9.4f} {tr['L_align']:13.4f} {te['L_align']:13.4f}")

    plot_loss_curve(loss_history, align_history, l_phys_history,
                    REPO_ROOT / cfg.loss_curve_path, body_history)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train(TrainConfig())
