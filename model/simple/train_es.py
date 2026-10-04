"""Evolution strategies on z_beta -- no policy gradient, no value function, no
noise in action space at all.

    z_beta   = G_theta(beta, z0)                       # the only thing learned
    eps_k    ~ N(0, I_256)
    F+_k     = cost( rollout(project(z_beta + sigma*eps_k)) )
    F-_k     = cost( rollout(project(z_beta - sigma*eps_k)) )
    g_z      = sum_k shape(F+_k - F-_k) * eps_k        # antithetic estimate
    theta   <- theta - lr * (dz_beta/dtheta)^T g_z     # exact, by autograd

Why this fits the problem better than PPO
------------------------------------------
z_beta is computed ONCE per episode and does not depend on the state, so the
learnable decision is a single 256-dim vector, not a per-step control law.
model/simple/train.py explores that by injecting Gaussian noise into 69 action
dims x 300 steps and hoping the score function routes 20700 dimensions of noise
back to the 256 that matter. ES perturbs the 256 directly.

Three things fall out of that, and they are the point:

  * NO exploration noise in the rollout. Actions are the frozen actor's mean.
    config.py's own comment says the per-step noise "compounds over 300 MuJoCo
    steps and was swamping the L_align/L_phys learning signal" -- here it is gone,
    and the only thing that differs between two rollouts is z.
  * NO per-step credit machinery. window_rewards, GAE, ValueNet, the PPO ratio,
    the KL guard and the log-ratio clamp all exist to get a usable per-step
    advantage; ES needs one scalar per rollout. The fitness is therefore the
    whole-episode cost from the UNCHANGED losses.py, which is also the number
    model/simple/evaluate.py reports -- training and evaluation measure the same
    quantity again.
  * The "some clips are just harder" confound cancels EXACTLY rather than being
    modelled. F+ and F- are the same clip on the same body, so the difference
    is within-pair. That is what ValueNet and bilevel's PairAdvantageNormalizer
    are approximations of.

Common random numbers are what make it work
--------------------------------------------
model/bilevel/upper.py:106 measured that without CRN the ES signal "is buried
under rollout noise". Here CRN is exact and free: humenv's Default init resets
every env slot to a bit-identical state (verified: max spread 0.0 across slots),
and with deterministic actions nothing else is random. So a +eps and a -eps
rollout differ ONLY by the z perturbation.

One motion, every body
-----------------------
An update holds the CLIP fixed and varies the BODY: it draws clips_per_update
clips and evaluates each of them on all 8 training bodies. The manifest stores
one origin_z per (task, trial) and every body's row for that clip points at the
same file, so the rows of one batch share z0 EXACTLY and differ only in beta and
in the per-body retargeted reference.

That is the contrast the adapter exists to explain, and it was absent before.
With one body per update beta is a CONSTANT inside the batch, so no single
update's gradient can distinguish G_theta(beta, z0) from a beta-blind
G_theta(z0) -- the beta-dependence had to be assembled ACROSS updates, through
the optimizer state, from batches that each also differed in which clips they
drew. Here it is inside one gradient: same z0, eight betas, eight references.

Slot budget
-----------
One vectorized env has cfg.batch_size slots and one skeleton, so a body's rows
have to be rolled out in that body's OWN env -- one batched rollout per body per
update. Within a body the slots are split into 2 * es_pairs antithetic rollouts
per clip, which fixes how many clips an update can see:

    clips_per_update = batch_size // (2 * es_pairs)

At the defaults (16 slots, 4 pairs, 8 bodies) that is 2 clips x 4 directions x
2 signs = 16 slots per body, 8 batched rollouts, 128 episodes per update -- 8x
the simulator cost of the one-body update this replaces. --bodies N buys that
back at the cost of the within-update beta contrast: --bodies 1 recovers the old
loop exactly, and es_pairs=8 (--pairs 8) makes clips_per_update 1, i.e. the
literal "one motion, eight bodies" batch.

Usage (from project root):
    uv run model/simple/train_es.py
    uv run model/simple/train_es.py --updates 400 --run-name es-400-8bodies
    uv run model/simple/train_es.py --sigma 0.5 --pairs 8 --no-wandb

    # one clip x 8 bodies per update (pairs 8 => clips_per_update 1), or fewer
    # bodies per update when the 8x rollout cost is the binding constraint.
    uv run model/simple/train_es.py --pairs 8 --run-name es-1clip-8bodies
    uv run model/simple/train_es.py --bodies 4 --run-name es-4bodies

    # one task group only -- see scripts/spilt_tasks.py. `upright` is the
    # 30 low-fall tasks; `move` is its 17-task locomotion SUBSET, so the two
    # overlap and `move` is the narrower experiment, not a disjoint one.
    uv run model/simple/train_es.py --category upright --run-name es-upright
    uv run model/simple/train_es.py --category move --run-name es-move

    # which of the two terms the fitness IS. The other one is still measured and
    # logged either way -- it just does not enter the cost -- so running all
    # three is how you find out whether they pull in the same direction:
    for L in both L_align L_phys; do
      uv run model/simple/train_es.py --loss $L --category upright \
          --ckpt-dir outputs/simple_es/$L --run-name es-upright-$L
    done
"""

import argparse
import collections
import dataclasses
import json
import os
import random
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel

from model.dataset import CrossEmbodimentDataset, load_task_list
from model.networks import LatentAdapter, SubspaceAdapter
from model.simple.config import ESConfig
from model.simple.train import compute_batch_cost, make_body_ctx

REPO_ROOT = Path(__file__).resolve().parents[2]

# --loss -> (lambda_align, lambda_phys), fed straight into ESConfig so the cost
# compute_batch_cost already returns IS the objective and this file never gets a
# second definition of it. Same table as scripts/single_z_search.py:OBJECTIVES,
# so the ceiling that script measures and what training minimises are the same
# three quantities.
#
# Only "both" is sensitive to the two terms' relative scale. The single-term
# settings are not: es_rank_normalize replaces the fitness with its rank inside
# each row, and any monotone rescaling of one loss leaves the ranks unchanged.
LOSS_LAMBDAS = {
    "both": (1.0, 1.0),
    "L_align": (1.0, 0.0),
    "L_phys": (0.0, 1.0),
    # BFMTrack's latent-space alignment in place of the joint-space L_align
    # (cfg.align_loss = "bfm", model/bfm_align.py). Same lambdas as L_align: the
    # align slot is the objective, L_phys is logged only.
    "bfm": (1.0, 0.0),
}


def clip_index(dataset):
    """{(reward_name, trial): {body label: row index}} -- a CLIP's column of the
    manifest's (clip x body) cross product.

    CrossEmbodimentDataset.indices_by_body() is the transpose of this, and is
    what a one-body-per-update loop needs. This is what "same motion, different
    bodies" needs. The two rows of one column share origin_z (hence z0) and
    differ in `morphology` (beta) and `retargeted_motion` (qpos_ref).
    """
    out = {}
    for i, r in enumerate(dataset.rows):
        key = (r["reward_name"], r["trial"])
        out.setdefault(key, {})[r.get("morphology_label", "child")] = i
    return out


def select_bodies(bodies, n, order, update):
    """Which bodies one update's batch covers. n >= len(bodies) is all of them,
    which is this file's point and its default; smaller n is the wall-clock
    escape hatch, and body_order then says whether the subset walks the list
    ("cycle", so N updates still give every body N*n/len(bodies) of them) or is
    drawn i.i.d. ("random")."""
    if n >= len(bodies):
        return list(bodies)
    if order == "cycle":
        start = (update * n) % len(bodies)
        return [bodies[(start + j) % len(bodies)] for j in range(n)]
    return random.sample(bodies, n)


def train_wandb_log(cfg, st):
    """The per-update numbers that can move in this configuration; st and the
    text log keep everything. Dropped: by_body/* and body_cost_spread with one
    body (a copy of cost, and 0), g_anchor_norm / g_ratio at lambda_z 0 (0),
    g_es_norm (rank-normalized, so ~constant), the surrogate's value (only its
    gradient means anything), z_norm (always 16), z_cos (dist_z0 replaces it),
    cost_std (mixes clips), and L_align / L_phys under the bfm cost, where
    neither is optimized."""
    one_body = len(st["by_body"]) == 1
    drop = {"by_body", "z_cos", "surrogate", "z_norm", "g_es_norm", "cost_std"}
    if cfg.align_loss == "bfm":
        drop |= {"L_align", "L_phys"}
    if not cfg.lambda_z:
        drop |= {"g_anchor_norm", "g_ratio"}
    if one_body:
        drop.add("body_cost_spread")
    log = {k: v for k, v in st.items() if k not in drop}
    if not one_body:
        for lab, v in st["by_body"].items():
            log.update({f"by_body/{lab}/{k}": val for k, val in v.items()})
    return log


def eval_wandb_log(history, clip_rows=None):
    """wandb dict for the newest eval in history, a list of (update, ev).

    Kept: cost and L_align of the train / test sets and the Euclidean distance
    from z0. Dropped: the per-body rows when they repeat the train / test ones
    (a single-body clip split), L_phys (not optimized), the angles (distance
    replaces them) and the train-test gap (the two curves show it).

    clip_rows = {"train": row, "test": row} names the eval rows whose clips are
    charted one by one, each as its cost / its own cost at the first eval --
    a falling mean can hide clips that got worse."""
    import wandb

    def dist(c):  # older histories only have the angle; z and z0 are both at radius 16
        return c["dist"] if "dist" in c else float(32 * np.sin(np.radians(c["angle"]) / 2))

    u, ev = history[-1]
    log = {}
    for sp in ("train", "test"):
        if sp not in ev:
            continue
        pre = f"eval/{sp}_bodies"
        log[f"{pre}/cost"], log[f"{pre}/L_align"] = ev[sp]["cost"], ev[sp]["L_align"]
        if clip_rows and sp in clip_rows:
            log[f"{pre}/dist_z0"] = float(np.mean([dist(c) for c in ev[clip_rows[sp]]["per_clip"]]))
        elif "dist_mean" in ev[sp]:
            log[f"{pre}/dist_z0"] = ev[sp]["dist_mean"]
    if not clip_rows:
        log.update({f"eval/{b}/cost": v["cost"] for b, v in ev.items() if b not in ("train", "test")})
        return log
    u0 = history[0][0]
    for sp, row in clip_rows.items():
        per = [{c["clip"]: c["cost"] for c in e[row]["per_clip"]} for _, e in history]
        names = [c["clip"] for c in ev[row]["per_clip"]]
        traj = [[h[n] / per[0][n] for h in per] for n in names]
        now = np.array([t[-1] for t in traj])
        pre, lab = f"eval/{sp}_bodies", ("held-out" if sp == "test" else "trained")
        log[f"{pre}/clips_down"] = float((now < 1).mean())    # share of clips below their start
        log[f"{pre}/ratio_median"] = float(np.median(now))
        log[f"{pre}/ratio_worst"] = float(now.max())
        log[f"clips/{sp}_curves"] = wandb.plot.line_series(
            xs=[uu for uu, _ in history], ys=traj, keys=names, xname="update",
            title=f"{lab} clips: cost / own cost at update {u0}")
        tab = wandb.Table(data=sorted(zip(names, now.tolist()), key=lambda x: x[1]),
                          columns=["clip", "ratio"])
        log[f"clips/{sp}_now"] = wandb.plot.bar(
            tab, "clip", "ratio", title=f"{lab} clips at update {u}: cost / own cost at update {u0}")
    return log


def write_eval_history(ckpt_dir, history):
    """Rewrite eval_history.json from scratch.

    Called after EVERY eval, not only at the end. These runs are ~12 hours and
    the per-clip costs in here are the only record of which motion families
    improved -- scripts/compare_runs.py reads nothing else. Writing it once at
    exit means a run that is killed at update 590, or one still in progress,
    has produced nothing readable at all.
    """
    (Path(ckpt_dir) / "eval_history.json").write_text(json.dumps(
        [{"update": u, **{b: v for b, v in ev.items()}} for u, ev in history], indent=1))


def load_z0_cost(path):
    """(task, trial) -> z0 bfm cost, from a scripts/rank_initial_cost.py CSV.

    The body column is taken by name when there is exactly one besides the
    bookkeeping columns, which is the single-body case this is for; a
    multi-body CSV would need the caller to say which column, and silently
    picking the first one would score every body against one body's z0.
    """
    import csv
    path = Path(path)
    rows = list(csv.DictReader(open(path if path.is_absolute() else REPO_ROOT / path)))
    cols = [c for c in rows[0] if c not in ("rank", "task", "clip", "mean")]
    if len(cols) != 1:
        raise SystemExit(f"{path} has {len(cols)} body columns {cols}; row_weight=headroom "
                         f"needs exactly one")
    out = {}
    for r in rows:
        t, stem = r["task"], r["clip"]
        trial = stem[len(t) + 1:] if stem.startswith(t + "_") else stem.rsplit("_", 1)[1]
        out[(t, int(trial))] = float(r[cols[0]])
    return out


def row_weights(cfg, delta, rows, state):
    """Per-row weight in (0, 1], or None when cfg.row_weight is "none".

    See config.py:row_weight for why each form exists. Both rules score a row in
    [floor, 1] and cfg.row_weight_renorm then restores the batch's mean weight to
    1, so the run differs from its baseline in how the gradient is SHARED OUT and
    not in how big it is.
    """
    if cfg.row_weight in ("none", "relative"):
        return None
    if cfg.row_weight == "conf":
        spread = delta.abs().mean(dim=1)                          # (R,)
        med = float(spread.median())
        ref = state.get("spread_ref")
        # Seeded with the first batch's median rather than 0, so the first
        # updates are not divided by a number still climbing out of zero.
        ref = med if ref is None else (1 - cfg.row_weight_ema) * ref + cfg.row_weight_ema * med
        state["spread_ref"] = ref
        w = spread / max(ref, 1e-12)
    elif cfg.row_weight == "headroom":
        z0c = state["z0_cost"]
        cur = state["row_cost"]                                   # (R,) mean cost this update
        base = torch.tensor([z0c[(s["reward_name"], int(s["trial"]))] for _, s in rows],
                            dtype=cur.dtype, device=cur.device)
        w = cur / base.clamp(min=1e-6)
    else:
        raise SystemExit(f"unknown row_weight '{cfg.row_weight}'")
    w = w.clamp(cfg.row_weight_floor, 1.0)
    if cfg.row_weight_renorm:
        # Without this the experiment is confounded. Both rules cap at 1 and cut
        # from there, so a typical batch's mean weight settles around 0.6-0.7 and
        # the run silently trains at 2/3 of the baseline's step size -- if it then
        # behaves differently there is no telling whether the redistribution did
        # it or the smaller effective lr did. Dividing by a running mean of the
        # weights keeps the batch's TOTAL gradient where the baseline puts it and
        # leaves only the redistribution, which is the thing under test. The
        # reference is an EMA for the same reason as "conf"'s: a 4-row batch mean
        # is too noisy to divide by.
        m = float(w.mean())
        ref = state.get("w_ref")
        ref = m if ref is None else (1 - cfg.row_weight_ema) * ref + cfg.row_weight_ema * m
        state["w_ref"] = ref
        w = w / max(ref, 1e-6)
    return w


def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Evenly spaced ranks in [-0.5, 0.5].

    Copied from model/bilevel/upper.py:129 with its reason: it makes the
    estimator invariant to the fitness's drifting scale. That matters more here
    than it does there -- L_align is heavy-tailed (measured p90 18.9 against a median
    of 6.6, max 136), so one catastrophic clip would otherwise set the step size
    for the whole update.
    """
    n = x.numel()
    if n < 2:
        return torch.zeros_like(x)
    order = x.argsort()
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(n, dtype=x.dtype, device=x.device)
    return ranks / (n - 1) - 0.5


def set_init_qpos(env, qpos_list, nv):
    """Put each env slot in its OWN pose. VectorEnv.call broadcasts one argument
    to every slot, so it cannot do this; a sync env exposes its sub-envs and an
    async one its pipes, and both are handled here."""
    zero = np.zeros(nv)
    if hasattr(env, "envs"):                       # SyncVectorEnv
        for e, q in zip(env.envs, qpos_list):
            e.unwrapped.set_physics(qpos=q, qvel=zero)
    elif hasattr(env, "parent_pipes"):             # AsyncVectorEnv
        for pipe, q in zip(env.parent_pipes, qpos_list):
            pipe.send(("_call", ("set_physics", (), {"qpos": q, "qvel": zero})))
        for pipe in env.parent_pipes:
            _, ok = pipe.recv()
            assert ok, "set_physics failed in a worker"
    else:
        raise SystemExit(f"cannot set per-slot physics on {type(env).__name__}")
    return {"proprio": np.stack([o["proprio"] for o in env.call("get_obs")])}


@torch.no_grad()
def rollout_z(model, env, z_env, cfg, obs_mul=None, return_obs=False,
              init_qpos=None, nv=None, exact=None):
    """Deterministic rollout, one z per env slot. Returns qpos (n_envs, T, nq),
    and with return_obs also the RESCALED proprio of the state after each step,
    (n_envs, T, 358), index-aligned with qpos -- the input the bfm align loss
    feeds to B(), the same canonicalised obs the actor is shown.

    No sampling anywhere: this is what makes the antithetic difference a clean
    measurement of z rather than of the noise draw.
    """
    obs, _ = env.reset()
    if init_qpos is not None:
        obs = set_init_qpos(env, init_qpos, nv)
    qpos_hist, obs_hist = [], []
    if exact is not None:
        # exact (model.exact_obs.ExactObs): the adult-equivalent observation of every slot's state
        if init_qpos is None:
            raise SystemExit("obs_scale 'exact' needs --init-reference")
        ex_obs = np.stack([exact(q, np.zeros(nv)) for q in init_qpos])
    for t in range(cfg.steps_per_episode):
        if exact is not None:
            proprio = ex_obs
        else:
            proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.as_tensor(proprio, dtype=torch.float32, device=cfg.device)
        # z_env (n, 256): one z per slot; (n, T, 256): a z per step (FB's tracking mode), the last held
        z_t = z_env if z_env.dim() == 2 else z_env[:, min(t, z_env.shape[1] - 1)]
        mu = model._actor(model._normalize(obs_t), z_t, model.cfg.actor_std).mean
        obs, _, _, _, info = env.step(mu.cpu().numpy())
        qpos_hist.append(info["qpos"].copy())
        if exact is not None:
            ex_obs = np.stack([exact(q, v) for q, v in zip(info["qpos"], info["qvel"])])
        if return_obs:
            obs_hist.append(ex_obs if exact is not None else
                            (obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul))
    if return_obs:
        return np.stack(qpos_hist, axis=1), np.stack(obs_hist, axis=1).astype(np.float32)
    return np.stack(qpos_hist, axis=1)


def _root_heading_xy(fk, qpos):
    """(..., T, nq) -> pelvis heading (..., T) in radians and root xy (..., T, 2); the same
    heading as scripts/single_z_search.py (humenv's calc_heading after remove_base_rot)."""
    import mujoco
    import humenv.utils as hu
    flat = qpos.reshape(-1, qpos.shape[-1]); d = mujoco.MjData(fk); h = np.empty(len(flat))
    for i, q in enumerate(flat):
        d.qpos[:] = q; mujoco.mj_kinematics(fk, d)
        h[i] = hu.calc_heading(hu.remove_base_rot(d.xquat[1][None].copy(), "smpl"))[0]
    return h.reshape(qpos.shape[:-1]), qpos[..., :2]


def score_rollouts(cfg, model, ctx, z_env, refs, ref_keys, extra=False, terms=None):
    """Roll z_env out on one body and score every slot. Returns (cost, L_align,
    L_phys), each (n_slots,).

    cfg.align_loss == "joint": cost is compute_batch_cost's lambda-weighted sum.
    cfg.align_loss == "bfm":   cost is 1 - mean_t cos(B(s_t), B(g_t)) per slot
    (model/bfm_align.py); L_align / L_phys are still the joint-space numbers so
    the log columns keep their meaning across the two settings. B(g_t) for a
    reference is computed once per (body, clip) on the body's single-slot env
    and cached in ctx["Bg"] under ref_keys[i].

    terms, a dict, receives the unweighted heading and root-xy terms per slot
    (when extra adds them), so the log can split the cost into its parts.
    """
    init = None
    if cfg.init_from_reference:
        if any(r is None for r in refs):
            raise SystemExit("init_from_reference needs qpos_ref on every row; this manifest has none")
        init = [r[0] for r in refs]
    if cfg.align_loss != "bfm":
        qpos = rollout_z(model, ctx["env"], z_env, cfg, ctx["obs_mul"], exact=ctx.get("exact"),
                         init_qpos=init, nv=ctx["fk"].nv)
        return compute_batch_cost(ctx["fk"], cfg, qpos, refs)
    from model import bfm_align
    qpos, obs = rollout_z(model, ctx["env"], z_env, cfg, ctx["obs_mul"], return_obs=True, exact=ctx.get("exact"),
                          init_qpos=init, nv=ctx["fk"].nv)
    _, la, lp = compute_batch_cost(ctx["fk"], cfg, qpos, refs)
    Bs = bfm_align.embed(model, obs.reshape(-1, obs.shape[-1]), cfg.device).reshape(obs.shape[0], obs.shape[1], -1)
    cost = np.empty(len(refs), dtype=np.float32)
    for i, (ref, key) in enumerate(zip(refs, ref_keys)):
        if key not in ctx["Bg"]:
            if ctx.get("exact") is None:
                ctx["Bg"][key] = bfm_align.reference_embeddings(model, ctx["env1"], ref, cfg.device, ctx["obs_mul"])
            else:   # the reference through the same reverse retargeting as the rollout states
                import mujoco
                vr, ro = np.zeros(ctx["fk"].nv), []
                for t in range(len(ref)):
                    if t:
                        mujoco.mj_differentiatePos(ctx["fk"], vr, bfm_align.DEFAULT_DT, ref[t - 1], ref[t])
                    ro.append(ctx["exact"](ref[t], vr))
                ctx["Bg"][key] = bfm_align.embed(model, np.stack(ro), cfg.device)
        cost[i] = bfm_align.bfm_align_loss(Bs[i], ctx["Bg"][key])
    if extra and (cfg.heading_weight or cfg.pos_weight):
        hq, xq = _root_heading_xy(ctx["fk"], qpos)
        ctx.setdefault("ref_hxy", {})
        h = np.zeros(len(refs), dtype=np.float32); p = np.zeros(len(refs), dtype=np.float32)
        for i, (ref, key) in enumerate(zip(refs, ref_keys)):
            if key not in ctx["ref_hxy"]:
                ctx["ref_hxy"][key] = _root_heading_xy(ctx["fk"], np.asarray(ref))
            hr, xr = ctx["ref_hxy"][key]
            T = min(qpos.shape[1], len(hr))
            h[i] = float(((1 - np.cos(hq[i, :T] - hr[:T])) / 2).mean())
            p[i] = float(np.linalg.norm(xq[i, :T] - xr[:T], axis=-1).mean())
        cost += cfg.heading_weight * h + cfg.pos_weight * p
        if terms is not None:
            terms["heading"], terms["pos"] = h, p
    return cost, la, lp


def clip_key(sample):
    """(reward_name, trial) -- what identifies a clip, and the cache key for its
    reference embeddings on a given body."""
    return (sample["reward_name"], sample["trial"])


def project_z(z, z_dim, on):
    """The adapter's own projection, applied again after perturbing. The actor
    was trained with norm_z=True and never re-projects what it is given (see
    TrainConfig.adapter_project_z), so a perturbed z has to be put back on the
    sphere before it is shown -- otherwise ES would be measuring the effect of
    leaving the manifold as much as the effect of the direction."""
    return (z_dim ** 0.5) * F.normalize(z, dim=-1) if on else z


def tangent_eps(z_beta, K, sigma, project_on, basis=None):
    """(R, K, D) perturbation directions, in the TANGENT space of the sphere.

    ES weights each direction by the eps it multiplied in, so the eps in the
    estimator has to be the perturbation that was actually evaluated. It is not,
    if you sample isotropically and then project: the radial component of eps is
    annihilated by project() but still appears in the sum, which biases g_z with
    a direction nothing was ever measured along.

    Projecting eps onto the tangent plane first removes that null direction
    exactly. What is left is a scalar shrink -- project(z + sigma*e_perp) has
    tangential part sigma*e_perp / sqrt(1 + (sigma|e_perp|/|z|)^2) -- plus a
    radial term that depends on |e_perp|^2 and is therefore IDENTICAL for +e and
    -e, so it is common-mode and cancels in the antithetic difference. The
    scalar shrink (measured 0.973 at sigma=0.25) is a uniform rescale that
    rank normalization discards anyway.

    Scale check worth doing whenever sigma changes: eps ~ N(0, I_256) has
    |eps| ~ sqrt(256) = 16, so |sigma*eps| = 16*sigma against |z| = 16 -- i.e.
    sigma IS the fractional perturbation of z, and 0.25 moves z by 25%, not by
    a fraction of a percent.
    """
    if basis is None:
        eps = torch.randn(z_beta.shape[0], K, z_beta.shape[-1], device=z_beta.device)
    else:
        # inside span(basis) only, scaled by sqrt(D / k) so |eps| ~ 16 as in the full space
        k = basis.shape[0]
        eps = (torch.randn(z_beta.shape[0], K, k, device=z_beta.device) @ basis) * (z_beta.shape[-1] / k) ** 0.5
    if project_on:
        zhat = F.normalize(z_beta.detach(), dim=-1).unsqueeze(1)      # (R, 1, D)
        eps = eps - (eps * zhat).sum(-1, keepdim=True) * zhat
    return eps


def update_best_buffer(buf, keys, z_row, costs):
    """{(task, trial, body): (z, cost)} -- the best CANDIDATE ever evaluated for
    each cell, kept across updates.

    Comparing a cost measured 300 updates ago against one measured now is only
    legitimate because the rollout is deterministic and the reset is
    bit-identical (see this module's CRN section): the same z on the same cell
    scores the same number forever, so the running minimum is a real minimum and
    not a record of which update got a lucky draw.

    Note what is NOT in here: z_beta itself. Training rolls out the perturbed
    candidates, never the iterate, so the buffer holds points the simulator
    actually scored -- which is the property that makes it a supervised target
    rather than a second guess.
    """
    hit = 0
    for r, k in enumerate(keys):
        j = int(costs[r].argmin())
        if k not in buf or costs[r, j] < buf[k][1]:
            buf[k] = (z_row[r, j].detach().cpu().numpy().copy(), float(costs[r, j]))
            hit += 1
    return hit


def _score_rows(cfg, model, ctxs, rows, z_row, K, terms=None):
    """roll every row's 2K candidates out on its own body -- the body-grouped
    loop shared by es_update and cmaes_update. terms (a dict) gets the (R, 2K)
    heading and root-xy terms when the cost includes them."""
    labels = [lab for lab, _ in rows]
    samples = [s for _, s in rows]
    R = len(rows)
    groups = {}
    for r, lab in enumerate(labels):
        groups.setdefault(lab, []).append(r)
    costs = np.empty((R, 2 * K), dtype=np.float32)
    align_totals = np.empty((R, 2 * K), dtype=np.float32)
    l_physes = np.empty((R, 2 * K), dtype=np.float32)
    for lab, rs in groups.items():
        n_slots = len(rs) * 2 * K
        if n_slots != cfg.batch_size:
            raise SystemExit(f"body {lab} has {len(rs)} rows x {2 * K} = {n_slots} "
                             f"rollouts but its env has {cfg.batch_size} slots")
        z_env = torch.cat([z_row[r] for r in rs])                # (batch_size, 256)
        refs_env = [samples[r]["qpos_ref"] for r in rs for _ in range(2 * K)]
        keys_env = [clip_key(samples[r]) for r in rs for _ in range(2 * K)]
        t = {}
        c, a, p = score_rollouts(cfg, model, ctxs[lab], z_env, refs_env, keys_env, extra=True, terms=t)
        if terms is not None:
            for k, v in t.items():
                terms.setdefault(k, np.zeros((R, 2 * K), dtype=np.float32))[rs] = v.reshape(len(rs), 2 * K)
        costs[rs] = c.reshape(len(rs), 2 * K)
        align_totals[rs] = a.reshape(len(rs), 2 * K)
        l_physes[rs] = p.reshape(len(rs), 2 * K)
    return groups, costs, align_totals, l_physes


def _project_np(z, z_dim, on):
    z = np.asarray(z, dtype=np.float64)
    return z * (z_dim ** 0.5 / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)) if on else z


def cmaes_update(cfg, model, adapter, optimizer, ctxs, rows, z_dim, cma_state):
    """One update where each (clip, body) cell's candidates come from its own
    CMA-ES (pycma) instead of antithetic pairs, and the adapter is trained
    toward the cell's recombined mean with an exact cosine gradient.

    cma_state: {(task, trial, body): CMAEvolutionStrategy}, created on first
    visit at the adapter's z_beta. See ESConfig.es_algo / cma_teacher for the
    two ways the mean is handled between visits. Everything the simulator sees
    is projected onto the sphere, and so is the CMA mean after every tell, so
    sigma stays a fractional perturbation exactly as in es_update.
    """
    import cma
    dev = cfg.device
    R, K = len(rows), cfg.es_pairs
    n_cand = 2 * K
    samples = [s for _, s in rows]
    z0 = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=dev)
    beta = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=dev)
    z_beta = adapter(beta, z0)                                   # (R, 256), differentiable
    zb = _project_np(z_beta.detach().cpu().numpy(), z_dim, cfg.adapter_project_z)
    keys = [(s["reward_name"], s["trial"], lab) for lab, s in rows]

    cands = np.empty((R, n_cand, z_dim))
    for r, k in enumerate(keys):
        es = cma_state.get(k)
        if es is None:
            es = cma.CMAEvolutionStrategy(zb[r], cfg.es_sigma, {
                "popsize": n_cand, "seed": cfg.seed + 1 + len(cma_state), "verbose": -9,
                "CMA_diagonal": bool(cfg.cma_diagonal)})
            cma_state[k] = es
        elif not cfg.cma_teacher:
            es.mean = zb[r].copy()
        cands[r] = _project_np(es.ask(), z_dim, cfg.adapter_project_z)
    z_row = torch.as_tensor(cands, dtype=torch.float32, device=dev)   # (R, 2K, 256)
    groups, costs, align_totals, l_physes = _score_rows(cfg, model, ctxs, rows, z_row, K)

    tgt = np.empty((R, z_dim))
    sig = np.empty(R)
    # clips are drawn with replacement, so a cell can hold several rows of one
    # update; pycma takes one tell per iteration, so those rows are told together
    by_key = {}
    for r, k in enumerate(keys):
        by_key.setdefault(k, []).append(r)
    for k, rs in by_key.items():
        es = cma_state[k]
        es.tell([c for r in rs for c in cands[r]], [float(c) for r in rs for c in costs[r]])
        es.mean = _project_np(es.mean, z_dim, cfg.adapter_project_z)
        for r in rs:
            tgt[r] = es.mean
            sig[r] = es.sigma
    tgt_t = torch.as_tensor(tgt, dtype=torch.float32, device=dev)
    tgt_cos = F.cosine_similarity(z_beta, tgt_t, dim=-1)
    pull = (1.0 - tgt_cos).mean()
    z_cos = (F.normalize(z_beta, dim=-1) * F.normalize(z0, dim=-1)).sum(-1).mean()
    anchor = cfg.lambda_z * (1.0 - z_cos)
    loss = pull + anchor
    g_es = torch.autograd.grad(pull, z_beta, retain_graph=True)[0]
    g_anchor = torch.autograd.grad(anchor, z_beta, retain_graph=True)[0]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    per_body = {lab: {"cost": float(costs[rs].mean()),
                      "L_align": float(align_totals[rs].mean()),
                      "L_phys": float(l_physes[rs].mean())}
                for lab, rs in groups.items()}
    spread = costs - costs.mean(1, keepdims=True)
    return {
        "by_body": per_body,
        "cost": float(costs.mean()), "L_align": float(align_totals.mean()),
        "L_phys": float(l_physes.mean()),
        "body_cost_spread": float(np.std([v["cost"] for v in per_body.values()])),
        # within-row spread of the candidates' cost, the CMA analogue of |F+ - F-|
        "delta_abs": float(np.abs(spread).mean()), "delta_std": float(spread.std()),
        "cost_std": float(costs.std()),
        "grad_norm": float(grad), "z_cos": z_cos.item(), "surrogate": pull.item(),
        "z_norm": z_beta.detach().norm(dim=-1).mean().item(),
        "g_es_norm": g_es.norm(dim=-1).mean().item(),
        "g_anchor_norm": g_anchor.norm(dim=-1).mean().item(),
        "g_ratio": (g_anchor.norm(dim=-1).mean() / g_es.norm(dim=-1).mean().clamp(min=1e-12)).item(),
        # how far the CMA mean is from the adapter's output, and the cells' sigma
        "tgt_deg": float(torch.rad2deg(torch.arccos(tgt_cos.detach().clamp(-1, 1))).mean()),
        "cma_sigma": float(sig.mean()),
        "tgt_cost_best": float(costs.min(1).mean()),
    }


def es_update(cfg, model, adapter, optimizer, ctxs, rows, z_dim, best_buf=None,
              row_state=None, head_sel=None):
    """One antithetic ES step over `rows`, each a (body label, sample) pair.

    Every clip appears once per body, so a row is a (clip, body) cell and the
    batch is a rectangle of them. eps is drawn PER ROW rather than shared by the
    bodies of one clip: the perturbation lives in the tangent plane of that
    row's own z_beta, and betas differ, so a shared direction would not be
    tangent for more than one of them.

    best_buf, when cfg.lambda_bc > 0, is the running per-cell best candidate
    (update_best_buffer). It adds a term that pulls z_beta toward a point the
    simulator MEASURED to be better, with an exact gradient, instead of toward
    whatever the rank-weighted zeroth-order estimate points at. Passing None, or
    leaving lambda_bc at 0, leaves every number in this function bit-identical to
    what it was before the buffer existed -- no RNG is drawn and no branch is
    taken.
    """
    dev = cfg.device
    R, K = len(rows), cfg.es_pairs
    labels = [lab for lab, _ in rows]
    samples = [s for _, s in rows]
    z0 = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=dev)
    beta = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=dev)

    H = getattr(cfg, "adapter_heads", 1)
    Kh = K // H
    if H > 1 and Kh * H != K:
        raise SystemExit(f"--heads {H} must divide --pairs {K}")
    ZB = adapter(beta, z0)                                       # (R,256) or (R,H,256)
    z_beta = ZB if H == 1 else None                              # set below for H>1

    if H == 1:
        eps = tangent_eps(z_beta, K, cfg.es_sigma, cfg.adapter_project_z, getattr(adapter, "U", None))
        with torch.no_grad():
            base = z_beta.unsqueeze(1)                           # (R, 1, 256)
            plus = project_z(base + cfg.es_sigma * eps, z_dim, cfg.adapter_project_z)
            minus = project_z(base - cfg.es_sigma * eps, z_dim, cfg.adapter_project_z)
            # Per-row slot layout: [+k0..+k3, -k0..-k3]
            z_row = torch.cat([plus, minus], dim=1)              # (R, 2K, 256)
    else:
        # Each head gets Kh of the K pairs, so the rollout budget per update is
        # unchanged and only its allocation differs. Slot layout per row:
        # [h0+ (Kh), h0- (Kh), h1+ (Kh), h1- (Kh), ...] -- _score_rows only cares
        # that there are 2K of them.
        eps_h, cand_h = [], []
        for h in range(H):
            e = tangent_eps(ZB[:, h], Kh, cfg.es_sigma, cfg.adapter_project_z)
            with torch.no_grad():
                b = ZB[:, h].detach().unsqueeze(1)
                cand_h.append(torch.cat(
                    [project_z(b + cfg.es_sigma * e, z_dim, cfg.adapter_project_z),
                     project_z(b - cfg.es_sigma * e, z_dim, cfg.adapter_project_z)], dim=1))
            eps_h.append(e)
        z_row = torch.cat(cand_h, dim=1)                         # (R, 2K, 256)

    # One batched rollout per body: an env carries one skeleton, and scoring a
    # row against another body's forward kinematics would silently measure the
    # adapter on a body it was not asked about (model/dataset.py says the same).
    terms = {}
    groups, costs, align_totals, l_physes = _score_rows(cfg, model, ctxs, rows, z_row, K, terms)
    # the candidate cost split into its parts (means over all 2K x R candidates)
    parts = {}
    if terms:
        parts = {"bfm": float((costs - cfg.heading_weight * terms["heading"]
                               - cfg.pos_weight * terms["pos"]).mean()),
                 "heading": float(terms["heading"].mean()), "pos": float(terms["pos"].mean())}
    if cfg.anchor_weight:
        # pull every candidate toward its own row's z0, inside the cost, before any ranking
        d = (z_row.detach() - z0.unsqueeze(1)).norm(dim=-1).cpu().numpy()       # (R, 2K)
        costs = costs + cfg.anchor_weight * (d / 16.0) ** 2
        parts["anchor_term"] = float((cfg.anchor_weight * (d / 16.0) ** 2).mean())

    w_stats = {}
    if H > 1:
        # Winner-take-all: for each row, only the head that produced the best
        # candidate gets a gradient. That is the whole point -- a head's update
        # is then averaged over the clips that currently prefer IT, not over
        # every clip in the batch, so heads can specialise instead of settling
        # on one compromise aim.
        fh = torch.as_tensor(costs, dtype=torch.float32, device=dev).view(R, H, 2, Kh)
        win = fh.reshape(R, H, -1).min(-1).values.argmin(-1)      # (R,)
        gz_rows = []
        for r in range(R):
            h = int(win[r])
            d = fh[r, h, 0] - fh[r, h, 1]                         # (Kh,)
            sh = rank_normalize(d) if cfg.es_rank_normalize else d / (2.0 * cfg.es_sigma)
            gz_rows.append((sh.unsqueeze(-1) * eps_h[h][r]).mean(0))
        g_z = torch.stack(gz_rows)                                # (R, 256)
        z_beta = ZB[torch.arange(R, device=dev), win]             # (R,256), differentiable
        f = fh.reshape(R, 2, K)                                   # only for the logged stats
        delta = fh[torch.arange(R, device=dev), win, 0] - fh[torch.arange(R, device=dev), win, 1]
        w_stats.update({"head_win_entropy": float(-(torch.bincount(win, minlength=H).float()
                                               / R + 1e-9).log().mul(
                       torch.bincount(win, minlength=H).float() / R).sum()),
                        "head_win_max_share": float(torch.bincount(win, minlength=H).max() / R)})
        if head_sel is not None:
            for r, (lab, sm) in enumerate(rows):
                head_sel[(sm["reward_name"], sm["trial"], lab)] = int(win[r])
    f = torch.as_tensor(costs, dtype=torch.float32, device=dev).view(R, 2, K)
    if cfg.row_weight == "relative":
        # The literal reading of "score every cell relative to its own z0".
        # Applied to the COST, before the delta, which is the only place it can
        # change anything -- and only with es_rank_normalize off, since ranks
        # are invariant to a positive per-row divisor. train() refuses the
        # combination rather than letting it run as a silent copy of baseline.
        base = torch.tensor([row_state["z0_cost"][(sm["reward_name"], int(sm["trial"]))]
                             for _, sm in rows], dtype=f.dtype, device=dev)
        f = f / base.clamp(min=1e-6).view(R, 1, 1)
    if H == 1:
        delta = f[:, 0] - f[:, 1]                                # (R, K) = F+ - F-
        if cfg.es_rank_normalize:
            # PER ROW. Ranking across rows would compare "which clip is easier"
            # instead of "which direction is better": a clip whose deltas are
            # O(5) would take every extreme rank and a clip whose deltas are
            # O(0.1) would be assigned arbitrary middle ranks. Only directions
            # within one (clip, body) are commensurable, because only they share
            # a landscape. Cost scale is a body property, so ranking across the
            # rows of one clip would rank the BODIES.
            shaped = torch.stack([rank_normalize(delta[r]) for r in range(R)])
        else:
            shaped = delta / (2.0 * cfg.es_sigma)

        # The general (category-free) answer to "move dominates". Applied to the
        # SHAPED delta, after rank normalization rather than before it: ranks are
        # invariant to any positive per-row scale, so a factor folded into the
        # cost would leave the gradient bit-identical (config.py:row_weight).
        if cfg.row_weight not in ("none", "relative"):
            row_state["row_cost"] = f.mean(dim=(1, 2))            # (R,)
            w = row_weights(cfg, delta, rows, row_state)
            shaped = shaped * w.unsqueeze(-1)
            w_stats.update({"row_w_mean": float(w.mean()), "row_w_min": float(w.min()),
                            "row_w_max": float(w.max()),
                            "row_w_floored": float((w <= (cfg.row_weight_floor
                                                          / max(row_state.get("w_ref", 1.0), 1e-6))
                                                    + 1e-9).float().mean())})
            if cfg.row_weight == "conf":
                w_stats["spread_ref"] = row_state["spread_ref"]
        g_z = (shaped.unsqueeze(-1) * eps).mean(dim=1)           # (R, 256)

    # g_z points UPHILL in cost; the surrogate below is minimized, so gradient
    # descent on theta walks z against it. (Both branches above have built it:
    # the single-head one from the shaped delta, the multi-head one from the
    # winning head's own pairs.)

    surrogate = (g_z.detach() * z_beta).sum(-1).mean()
    z_cos = (F.normalize(z_beta, dim=-1) * F.normalize(z0, dim=-1)).sum(-1).mean()
    anchor = cfg.lambda_z * (1.0 - z_cos)
    loss = surrogate + anchor

    bc = None
    bc_stats = {}
    if cfg.lambda_bc > 0 and best_buf is not None:
        keys = [(s["reward_name"], s["trial"], lab) for lab, s in rows]
        # Updated BEFORE the loss, so this update's own candidates are eligible:
        # the term is then "move toward the best thing measured so far,
        # including a moment ago", which is the greedy step the ES gradient is a
        # noisy approximation of.
        n_new = update_best_buffer(best_buf, keys, z_row, costs)
        tgt = torch.tensor(np.stack([best_buf[k][0] for k in keys]),
                           dtype=torch.float32, device=dev)
        # Cosine, not Euclidean, for the same reason as the anchor: with
        # adapter_project_z the radius is fixed, so a squared distance would
        # spend part of itself on a gap that cannot close.
        bc_cos = F.cosine_similarity(z_beta, tgt, dim=-1)
        bc = cfg.lambda_bc * (1.0 - bc_cos).mean()
        loss = loss + bc
        bc_stats = {
            "bc_deg": float(torch.rad2deg(torch.arccos(bc_cos.detach().clamp(-1, 1))).mean()),
            "bc_new": n_new / max(len(keys), 1),      # share of cells improved this update
            "buf_size": len(best_buf),
            "buf_cost": float(np.mean([best_buf[k][1] for k in keys])),
        }

    # The two terms' gradients w.r.t. z_beta, so their relative size is visible
    # rather than assumed. Rank normalization strips g_z of the cost's units, so
    # lambda_z tuned against the PPO objective carries no meaning here -- if this
    # ratio is far from O(1) the anchor is either inert or in sole charge.
    g_es = torch.autograd.grad(surrogate, z_beta, retain_graph=True)[0]
    g_anchor = torch.autograd.grad(anchor, z_beta, retain_graph=True)[0]
    if bc is not None:
        # Same diagnostic as the anchor's, and needed for the same reason: this
        # term's gradient is EXACT while the ES term's is a K-dimensional probe
        # of a 256-dimensional landscape, so a bc_ratio well under 1 can still
        # dominate once the noise averages out over updates.
        g_bc = torch.autograd.grad(bc, z_beta, retain_graph=True)[0]
        bc_stats["g_bc_norm"] = g_bc.norm(dim=-1).mean().item()
        bc_stats["bc_ratio"] = (g_bc.norm(dim=-1).mean()
                                / g_es.norm(dim=-1).mean().clamp(min=1e-12)).item()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    per_body = {lab: {"cost": float(costs[rs].mean()),
                      "L_align": float(align_totals[rs].mean()),
                      "L_phys": float(l_physes[rs].mean())}
                for lab, rs in groups.items()}

    return {
        "by_body": per_body,
        "cost": float(costs.mean()), "L_align": float(align_totals.mean()),
        "L_phys": float(l_physes.mean()),
        # Spread of the pooled cost that is pure body, at a fixed set of clips --
        # only meaningful because every body in this batch saw the same clips.
        "body_cost_spread": float(np.std([v["cost"] for v in per_body.values()])),
        # |F+ - F-| against the spread of F itself: if the perturbation moves the
        # cost by much less than the clip-to-clip spread, sigma is too small to
        # measure anything and no amount of lr will fix it.
        "delta_abs": float(delta.abs().mean()), "delta_std": float(delta.std()),
        "cost_std": float(costs.std()),
        "grad_norm": float(grad), "z_cos": z_cos.item(), "surrogate": surrogate.item(),
        "z_norm": z_beta.detach().norm(dim=-1).mean().item(),
        "g_es_norm": g_es.norm(dim=-1).mean().item(),
        "g_anchor_norm": g_anchor.norm(dim=-1).mean().item(),
        "g_ratio": (g_anchor.norm(dim=-1).mean() / g_es.norm(dim=-1).mean().clamp(min=1e-12)).item(),
        # Euclidean, radius 16: how far the adapter has taken this batch's z from z0
        "dist_z0": (z_beta.detach() - z0).norm(dim=-1).mean().item(),
        **parts, **bc_stats, **w_stats,
    }


@torch.no_grad()
def run_eval(cfg, model, adapter, ctxs, dataset, eval_idx, splits, z_dim, eval_body=None,
             head_sel=None):
    """Unperturbed, deterministic rollout of the fixed eval clips.

    eval_idx is keyed by an eval ROW, which is normally a body. Under a
    single-body clip split it is instead "<body>" and "<body>:unseen", two
    clip sets on the same body, and eval_body maps the row back to the body
    whose env and skeleton score it."""
    out = {}
    for b, idxs in eval_idx.items():
        body = (eval_body or {}).get(b, b)
        samples = [dataset[i] for i in idxs]
        z0 = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=cfg.device)
        beta = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=cfg.device)
        z_env = adapter(beta, z0)
        if z_env.dim() == 3:
            # multi-head: use the head that last won this cell during training.
            # The selection table is part of the model -- O(cells) integers --
            # so this is one forward pass, not a best-of-H rollout, and stays
            # comparable with the single-head runs.
            sel = torch.tensor([(head_sel or {}).get((s["reward_name"], s["trial"], body), 0)
                                for s in samples], device=z_env.device)
            z_env = z_env[torch.arange(len(samples), device=z_env.device), sel]
        n = len(samples)
        # the env has batch_size slots; pad with copies of the last row and
        # drop them again, so the eval set size does not depend on batch_size
        pad = cfg.batch_size - n
        z_pad = torch.cat([z_env, z_env[-1:].expand(pad, -1)]) if pad > 0 else z_env
        refs = [s["qpos_ref"] for s in samples] + [samples[-1]["qpos_ref"]] * pad
        keys = [clip_key(s) for s in samples] + [clip_key(samples[-1])] * pad
        costs, la, lp = (x[:n] for x in score_rollouts(cfg, model, ctxs[body], z_pad, refs, keys))
        # how far the adapter has moved each eval clip's z from its z0, per
        # cell -- the batch-mean 1-zcos in the update log mixes different clips
        # every step, so it cannot show whether a GIVEN clip drifts back to z0
        ang = torch.rad2deg(torch.arccos(F.cosine_similarity(z_env, z0, dim=-1).clamp(-1, 1)))
        dist = (z_env - z0).norm(dim=-1)                         # Euclidean, radius 16
        out[b] = {"cost": float(costs.mean()), "L_align": float(la.mean()), "L_phys": float(lp.mean()),
                  "angle_mean": float(ang.mean()), "angle_min": float(ang.min()),
                  "angle_max": float(ang.max()),
                  "dist_mean": float(dist.mean()), "dist_max": float(dist.max()),
                  "per_clip": [{"clip": f"{s['reward_name']}_{s['trial']}", "cost": float(c),
                                "angle": float(a), "dist": float(d)}
                               for s, c, a, d in zip(samples, costs, ang.tolist(), dist.tolist())]}
    for sp in ("train", "test"):
        mem = [b for b in out if splits.get(b) == sp]
        if mem:
            out[sp] = {k: float(np.mean([out[b][k] for b in mem]))
                       for k in ("cost", "L_align", "L_phys", "angle_mean", "angle_min", "angle_max",
                                 "dist_mean", "dist_max")}
    return out


def train(cfg: ESConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.batch_size % (2 * cfg.es_pairs):
        raise SystemExit(
            f"batch_size {cfg.batch_size} must be a multiple of 2*es_pairs "
            f"{2 * cfg.es_pairs}: every env slot carries one antithetic rollout.")
    clips_per_update = cfg.batch_size // (2 * cfg.es_pairs)

    dataset_dir = REPO_ROOT / cfg.dataset_dir
    task_filter = None
    if cfg.task_group != "all":
        path = dataset_dir / "splits" / f"{cfg.task_group}_tasks.txt"
        if not path.exists():
            raise SystemExit(f"{path} not found -- run scripts/spilt_tasks.py")
        task_filter = load_task_list(path)
        print(f"task group '{cfg.task_group}': {len(task_filter)} tasks")
    dataset = CrossEmbodimentDataset(dataset_dir, task_filter=task_filter)
    by_body = dataset.indices_by_body()
    bodies_path = dataset_dir / "splits" / "train_bodies.txt"
    if cfg.train_bodies:
        missing = [b for b in cfg.train_bodies if b not in by_body]
        if missing:
            raise SystemExit(f"--train-bodies: not in the manifest: {' '.join(missing)}")
        bodies = list(cfg.train_bodies)
    else:
        bodies = ([b for b in load_task_list(bodies_path) if b in by_body]
                  if bodies_path.exists() else list(by_body))
    held = sorted(set(by_body) - set(bodies))
    single_body = len(bodies) == 1
    if held:
        print(f"held out of training ({len(held)}): {' '.join(held)}"
              + ("  -- NOT evaluated: a single training body makes beta constant, so the "
                 "adapter is a z0 -> z MLP and the split that means anything is over clips"
                 if single_body else ""))

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()
    z_dim = model.cfg.archi.z_dim

    # A clip is (reward_name, trial); its column of the manifest is one row per
    # body. Training may only draw a clip that every SELECTED body has (a batch
    # is a full rectangle -- see es_update's slot check); eval may only draw one
    # that every body has, train and held-out alike, since it scores all of them.
    clips = clip_index(dataset)
    train_clips = sorted(k for k, m in clips.items() if all(b in m for b in bodies))
    eval_clips_pool = sorted(k for k, m in clips.items() if len(m) == len(by_body))
    if not train_clips or not eval_clips_pool:
        raise SystemExit("no clip is present for every body -- the manifest is not a "
                         "full (clip x body) cross product; rebuild it with "
                         "scripts/build_dataset.py --force")
    if len(train_clips) < len(clips):
        print(f"{len(clips) - len(train_clips)} of {len(clips)} clips dropped: "
              f"not present for every training body")
    heldout_clips = []
    if cfg.clip_list:
        path = REPO_ROOT / cfg.clip_list
        want = set()
        for line in load_task_list(path):
            parts = line.replace(",", " ").split()
            want.add((parts[0], parts[1]) if len(parts) >= 2 else tuple(line.rsplit("_", 1)))
        keep = [c for c in train_clips if (c[0], str(c[1])) in want or c in want]
        if not keep:
            raise SystemExit(f"--clip-list {path}: none of its {len(want)} clips is a trainable clip; "
                             f"a line must be '<task> <trial>' matching the manifest")
        print(f"clip list {path.name}: {len(keep)} of {len(train_clips)} trainable clips kept"
              + (f" ({len(want) - len(keep)} listed clips are not in the manifest)" if len(want) > len(keep) else ""))
        train_clips = sorted(keep)
    clip_cat = {}
    if cfg.clip_categories:
        for line in load_task_list(REPO_ROOT / cfg.clip_categories):
            t, c = line.split()
            clip_cat[t] = c
    if cfg.heldout_clip_frac > 0:
        if not single_body:
            raise SystemExit("heldout_clip_frac is for the single-body setting; with several bodies "
                             "the held-out axis is the body and this would confound the two")
        n_held = max(1, round(cfg.heldout_clip_frac * len(train_clips)))
        rng_h = random.Random(cfg.eval_seed)
        if clip_cat:
            missing = sorted({t for t, _ in train_clips} - set(clip_cat))
            if missing:
                raise SystemExit(f"--clip-categories has no category for: {' '.join(missing[:5])}")
            groups = {}
            for c in train_clips:
                groups.setdefault(clip_cat[c[0]], []).append(c)
            # equal counts per category, the remainder going to the largest ones
            # so the total still matches heldout_clip_frac
            base, extra = divmod(n_held, len(groups))
            order = sorted(groups, key=lambda g: -len(groups[g]))
            heldout_clips = []
            for i, g in enumerate(order):
                k = min(base + (1 if i < extra else 0), len(groups[g]))
                heldout_clips += rng_h.sample(groups[g], k)
            heldout_clips = sorted(heldout_clips)
            per = collections.Counter(clip_cat[t] for t, _ in heldout_clips)
            print(f"clip split: {len(train_clips) - len(heldout_clips)} train / {len(heldout_clips)} "
                  f"held out, STRATIFIED per category (seed {cfg.eval_seed}): "
                  + " ".join(f"{g}:{per[g]}" for g in order))
        else:
            heldout_clips = sorted(rng_h.sample(train_clips, n_held))
            print(f"clip split: {len(train_clips) - len(heldout_clips)} train / {len(heldout_clips)} "
                  f"held out ({cfg.heldout_clip_frac:.0%}, seed {cfg.eval_seed})")
        train_clips = sorted(set(train_clips) - set(heldout_clips))
    if cfg.n_train_clips and cfg.n_train_clips < len(train_clips):
        train_clips = sorted(random.Random(cfg.eval_seed).sample(train_clips, cfg.n_train_clips))
        print(f"TRAINING RESTRICTED to {len(train_clips)} fixed clips (--n-train-clips): "
              + " ".join(f"{t}_{k}" for t, k in train_clips))

    ctx_bodies = bodies if single_body else list(by_body)
    ctxs = {b: make_body_ctx(cfg, dataset_dir, b, dataset[by_body[b][0]]["target_xml"])
            for b in ctx_bodies}
    if cfg.align_loss == "bfm":
        # one single-slot env per body to push reference qpos through
        # (bfm_align.obs_from_qpos), and a per-(clip, body) cache of B(g_t)
        for b, ctx in ctxs.items():
            ctx["env1"], _ = make_humenv(num_envs=1, task=None,
                                         xml=str(dataset_dir / dataset[by_body[b][0]]["target_xml"]),
                                         state_init="Default")
            ctx["Bg"] = {}
    splits = {b: ("train" if b in bodies else "test") for b in ctx_bodies}
    beta_dim = len(dataset[by_body[bodies[0]][0]]["beta"])
    n_bodies = min(cfg.es_bodies_per_update or len(bodies), len(bodies))
    print(f"{len(bodies)} training bodies, {len(train_clips)} clips; per update "
          f"{n_bodies} bodies x {clips_per_update} clips x {cfg.es_pairs} antithetic "
          f"pairs = {n_bodies} rollouts of {cfg.batch_size} slots "
          f"({n_bodies * cfg.batch_size} episodes), sigma={cfg.es_sigma}")
    if cfg.align_loss == "bfm":
        print("objective: cost = 1 - mean_t cos(B(s_t), B(g_t))  (BFMTrack latent alignment, "
              "model/bfm_align.py; joint-space L_align and L_phys are computed and logged only)")
    else:
        print(f"objective: cost = {cfg.lambda_align} * L_align + {cfg.lambda_phys} * L_phys "
              f"(the zero-weighted term is still computed and logged)")

    if cfg.adapter_subspace:
        if getattr(cfg, "adapter_heads", 1) > 1:
            raise SystemExit("--subspace is single-head")
        basis = np.load(REPO_ROOT / cfg.adapter_subspace)[: cfg.adapter_subspace_dim or None]
        adapter = SubspaceAdapter(beta_dim, z_dim, basis, hidden_dims=cfg.adapter_hidden_dims,
                                  alpha=cfg.adapter_alpha, project=cfg.adapter_project_z).to(cfg.device)
        print(f"adapter: SubspaceAdapter, correction confined to {len(basis)} dims of {cfg.adapter_subspace}")
    else:
        adapter = LatentAdapter(
            beta_dim=beta_dim, z_dim=z_dim, hidden_dims=cfg.adapter_hidden_dims,
            alpha=cfg.adapter_alpha, alpha_learnable=cfg.adapter_alpha_learnable,
            project=cfg.adapter_project_z, residual=cfg.adapter_residual,
            head=cfg.adapter_head, theta_max_deg=cfg.adapter_theta_max_deg,
            n_heads=getattr(cfg, "adapter_heads", 1),
        ).to(cfg.device)
    optimizer = (torch.optim.AdamW(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                 if cfg.weight_decay > 0 else torch.optim.Adam(adapter.parameters(), lr=cfg.lr))
    if cfg.weight_decay > 0:
        print(f"optimizer: AdamW, weight_decay={cfg.weight_decay} (bounds |MLP| and so the drift from z0)")

    # Where the run STARTS, measured rather than assumed. residual=True puts
    # z_beta on top of z0 (cos ~ 1); residual=False with projection on starts it
    # at a uniformly random point of the sphere (cos ~ 0), which is a different
    # experiment and needs to be visible in the log rather than inferred from
    # the flags.
    with torch.no_grad():
        _r = [dataset[by_body[b][0]] for b in bodies]
        _z0 = torch.tensor(np.stack([x["z0"] for x in _r]), dtype=torch.float32, device=cfg.device)
        _be = torch.tensor(np.stack([x["beta"] for x in _r]), dtype=torch.float32, device=cfg.device)
        _za = adapter(_be, _z0)
        _c = F.cosine_similarity(_za[:, 0] if _za.dim() == 3 else _za, _z0, dim=-1).mean()
        _a = torch.rad2deg(torch.arccos(_c.clamp(-1, 1)))
    if cfg.adapter_head == "geodesic":
        _form = (f"geodesic: z = sqrt(d) * (cos(th) * z0_hat + sin(th) * u_hat), "
                 f"th = {cfg.adapter_theta_max_deg:g}deg * sigmoid(.), u in z0's tangent plane "
                 f"(alpha and residual unused)")
    elif cfg.adapter_residual:
        _form = f"z0 + {cfg.adapter_alpha:g} * MLP([beta, z0])"
    else:
        _form = "MLP([beta, z0])  (NO z0 residual; alpha unused)"
    print(f"adapter: " + _form
          + f", project={cfg.adapter_project_z}; "
            f"at init cos(z_beta, z0) = {_c:.4f} ({_a:.1f} deg from z0)")

    erng = random.Random(cfg.eval_seed)
    # The SAME clips for every body. eval's headline number is the train/test
    # BODY gap, and drawing each body's clips independently put the clip
    # difficulty spread -- which is heavy-tailed, L_align p90 18.9 against a
    # median of 6.6 -- straight into it, so one body's unlucky draw could move
    # the gap by more than the adapter does. Shared clips make it a body effect.
    n_eval = min(cfg.eval_clips, cfg.batch_size)
    eval_body = {}
    if heldout_clips:
        b0 = bodies[0]
        # SAMPLED, not the first n in sort order: train_clips is sorted by task,
        # so a prefix is one or two tasks and would be compared against a
        # held-out set drawn from all of them -- the gap would then be the
        # difference between two task mixes, not between seen and unseen.
        pool_seen = [c for c in train_clips if c in eval_clips_pool]
        unseen = [c for c in heldout_clips if c in eval_clips_pool]
        seen = erng.sample(pool_seen, min(n_eval, len(pool_seen), max(len(unseen), 1)))
        if not unseen:
            raise SystemExit("none of the held-out clips is present for every body, so it cannot be evaluated")
        eval_clips = seen + unseen
        eval_is_train = [True] * len(seen) + [False] * len(unseen)
        eval_idx = {b0: [clips[c][b0] for c in seen],
                    f"{b0}:unseen": [clips[c][b0] for c in unseen]}
        eval_body = {b0: b0, f"{b0}:unseen": b0}
        splits = {b0: "train", f"{b0}:unseen": "test"}
        print(f"eval: {len(seen)} trained clips (row '{b0}') vs {len(unseen)} held-out clips "
              f"(row '{b0}:unseen'); the 'gap' column is clip generalisation, not body")
    elif cfg.n_train_clips:
        # a probe run: eval on the clips being trained (convergence) plus as
        # many unseen ones as fit (generalisation), marked * / - in the log
        seen = [c for c in train_clips if c in eval_clips_pool][:n_eval]
        rest = [c for c in eval_clips_pool if c not in seen]
        eval_clips = seen + [erng.choice(rest) for _ in range(n_eval - len(seen))]
    else:
        # From the TRAINING clips, not the whole pool. eval_clips_pool is every
        # clip the manifest has for every body; with --clip-list and no held-out
        # split that includes the clips the list deliberately excluded, and the
        # eval would then report a number for a set the run is not training on.
        # With no held-out split the eval's job is convergence, not
        # generalisation, so it must look at what is being fitted.
        pool = [c for c in train_clips if c in eval_clips_pool] or eval_clips_pool
        eval_clips = [erng.choice(pool) for _ in range(n_eval)]
    if not heldout_clips:
        eval_is_train = [c in train_clips for c in eval_clips] if cfg.n_train_clips else [True] * len(eval_clips)
        eval_idx = {b: [clips[c][b] for c in eval_clips] for b in ctx_bodies}

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config={**dataclasses.asdict(cfg), "train_bodies": bodies,
                           "held_out_bodies": held, "clips_per_update": clips_per_update,
                           "bodies_per_update": n_bodies})

    history = []
    # None unless the feature is on, so the default path allocates nothing and
    # the checkpoints it writes keep their old shape.
    best_buf = {} if cfg.lambda_bc > 0 else None
    cma_state = {} if cfg.es_algo == "cmaes" else None
    clip_pools = pool_keys = None
    if cfg.clip_balance == "category":
        if not clip_cat:
            raise SystemExit("--clip-balance category needs --clip-categories")
        clip_pools = {}
        for c in train_clips:
            clip_pools.setdefault(clip_cat[c[0]], []).append(c)
        pool_keys = sorted(clip_pools)
        share = collections.Counter(clip_cat[t] for t, _ in train_clips)
        print(f"clip sampling: CATEGORY-uniform over {len(pool_keys)} families "
              f"({100 / len(pool_keys):.1f}% each), not uniform over clips; "
              + " ".join(f"{g}:{share[g]}clips" for g in pool_keys))

    head_sel = {} if getattr(cfg, "adapter_heads", 1) > 1 else None
    if head_sel is not None:
        print(f"multi-head: {cfg.adapter_heads} heads x {cfg.es_pairs // cfg.adapter_heads} pairs "
              f"each (same {2 * cfg.es_pairs} rollouts per clip), winner-take-all gradient; "
              f"eval uses each cell's last winning head")

    row_state = None
    if cfg.row_weight != "none":
        row_state = {}
        if cfg.row_weight == "relative" and cfg.es_rank_normalize:
            raise SystemExit("row_weight=relative with es_rank_normalize on is a no-op "
                             "(ranks are invariant to a per-row divisor): pass --no-rank")
        if cfg.row_weight in ("headroom", "relative"):
            if not cfg.row_weight_z0_csv:
                raise SystemExit("row_weight=headroom needs --z0-cost-csv")
            row_state["z0_cost"] = load_z0_cost(cfg.row_weight_z0_csv)
            miss = [c for c in train_clips if (c[0], int(c[1])) not in row_state["z0_cost"]]
            if miss:
                raise SystemExit(f"{len(miss)} training clips have no z0 cost in "
                                 f"{cfg.row_weight_z0_csv}, e.g. {miss[:3]}")
        if cfg.row_weight == "relative":
            print(f"row scoring: RELATIVE -- every candidate's cost divided by that cell's z0 "
                  f"cost ({cfg.row_weight_z0_csv}), rank normalization OFF, so a cell's "
                  f"landscape enters the gradient at its own scale")
        else:
            print(f"row weighting: {cfg.row_weight}, floor {cfg.row_weight_floor}, cap 1.0"
                  + (f", EMA {cfg.row_weight_ema}" if cfg.row_weight == "conf" else
                     f", z0 costs from {cfg.row_weight_z0_csv}")
                  + (", renormalized so the batch's mean weight is 1 (the baseline's step "
                 "size is preserved; only its distribution changes)"
                 if cfg.row_weight_renorm else ", NOT renormalized (effective lr drops too)"))
    if cma_state is not None:
        print(f"search: CMA-ES per (clip, body) cell, popsize {2 * cfg.es_pairs}, sigma0 {cfg.es_sigma}, "
              f"{'TEACHER (cell means persist across visits)' if cfg.cma_teacher else 're-centred on z_beta every visit'}"
              f"{', diagonal' if cfg.cma_diagonal else ''}; adapter trained toward the cell mean (cosine)")
    if best_buf is not None:
        print(f"best-point buffer ON: lambda_bc={cfg.lambda_bc} "
              f"(cosine pull toward the best candidate measured per (clip, body); "
              f"watch bc_ratio and bc_deg)")

    def do_eval(update):
        ev = run_eval(cfg, model, adapter, ctxs, dataset, eval_idx, splits, z_dim, eval_body,
                      head_sel)
        history.append((update, ev))
        write_eval_history(ckpt_dir, history)
        tr, te = ev.get("train"), ev.get("test")
        tqdm.write(f"  [eval @ {update:04d}] "
                   + (f"train cost={tr['cost']:.4f} (L_align={tr['L_align']:.4f} Lp={tr['L_phys']:.4f} "
                      f"|z-z0| {tr['dist_mean']:.2f} (max {tr['dist_max']:.2f}))  " if tr else "")
                   + (f"test cost={te['cost']:.4f} (L_align={te['L_align']:.4f} Lp={te['L_phys']:.4f} "
                      f"|z-z0| {te['dist_mean']:.2f})  " if te else "")
                   + (f"gap={te['cost'] - tr['cost']:+.4f}" if tr and te else ""))
        # per-clip, averaged over the training bodies: the same 16 clips every
        # eval, so a clip's row across evals is its own trajectory
        if heldout_clips:
            for row, tag in ((bodies[0], "trained"), (f"{bodies[0]}:unseen", "held out")):
                pc = ev[row]["per_clip"]
                tqdm.write(f"    {tag:9s} clips ({len(pc)}): cost {np.mean([x['cost'] for x in pc]):.4f}  "
                           f"|z-z0| {np.mean([x['dist'] for x in pc]):.2f}  | "
                           + "  ".join(f"{x['clip']}:{x['cost']:.2f}" for x in pc[:8]))
            tb = []
        else:
            tb = [b for b in ev if splits.get(b) == "train" and "per_clip" in ev[b]]
        if tb:
            n = len(ev[tb[0]]["per_clip"])
            cells = [(ev[tb[0]]["per_clip"][i]["clip"],
                      np.mean([ev[b]["per_clip"][i]["cost"] for b in tb]),
                      np.mean([ev[b]["per_clip"][i]["angle"] for b in tb])) for i in range(n)]
            tqdm.write("    per-clip (train bodies; * = trained on): " + "  ".join(
                f"{'*' if eval_is_train[i] else '-'}{c}:{cost:.3f}/{a:.0f}\u00b0"
                for i, (c, cost, a) in enumerate(cells)))
            if cfg.n_train_clips:
                tr_c = [cells[i][1] for i in range(n) if eval_is_train[i]]
                un_c = [cells[i][1] for i in range(n) if not eval_is_train[i]]
                tqdm.write(f"    trained clips: cost {np.mean(tr_c):.4f}   unseen clips: cost "
                           f"{np.mean(un_c):.4f}" if un_c else f"    trained clips: cost {np.mean(tr_c):.4f}")
        if cfg.use_wandb:
            clip_rows = ({"train": bodies[0], "test": f"{bodies[0]}:unseen"}
                         if heldout_clips else None)
            wandb.log(eval_wandb_log(history, clip_rows), step=update)

    if cfg.eval_every and cfg.eval_at_start:
        do_eval(0)

    pbar = tqdm(range(cfg.num_updates), desc="es", disable=not cfg.progress)
    for update in pbar:
        sel = select_bodies(bodies, n_bodies, cfg.body_order, update)
        if clip_pools is None:
            picks = [random.choice(train_clips) for _ in range(clips_per_update)]
        else:
            # Pick the FAMILY first, then a clip inside it. Uniform-over-clips
            # sampling spends the budget in proportion to how many clips a family
            # has, and the "balanced" 500 still gives move 200 of them -- so move
            # took 40% of every gradient while headstand took 8%. The four
            # row-weight runs all reweighted rows INSIDE a batch and none of them
            # moved the result; this changes which clips are in the batch at all,
            # which is the one knob they left untouched.
            picks = [random.choice(clip_pools[random.choice(pool_keys)])
                     for _ in range(clips_per_update)]
        # Body-major so es_update's per-body group is contiguous; the (clip,
        # body) rectangle it forms is what makes beta the only thing varying.
        rows = [(b, dataset[clips[c][b]]) for b in sel for c in picks]
        if cma_state is not None:
            st = cmaes_update(cfg, model, adapter, optimizer, ctxs, rows, z_dim, cma_state)
        else:
            st = es_update(cfg, model, adapter, optimizer, ctxs, rows, z_dim, best_buf,
                               row_state, head_sel)

        # cost, not L_align: under --loss L_phys the L_align column is a
        # bystander and watching it would say nothing about whether ES is working.
        clip_tag = picks[0][0] if clips_per_update == 1 else f"{len(picks)} clips"
        pbar.set_postfix(clip=clip_tag, cost=f"{st['cost']:.4f}", d=f"{st['delta_abs']:.3f}")
        if cfg.use_wandb:
            wandb.log(train_wandb_log(cfg, st), step=update)
        if update % cfg.log_every == 0:
            tqdm.write(f"[{update:04d}/{cfg.num_updates}] {clip_tag:24s} "
                       f"cost={st['cost']:.4f} sd_body={st['body_cost_spread']:.4f} "
                       f"|dF|={st['delta_abs']:.4f} "
                       f"sd(dF)={st['delta_std']:.4f} |g_es|={st['g_es_norm']:.4f} "
                       f"|g_anc|={st['g_anchor_norm']:.2e} r={st['g_ratio']:.2e} "
                       f"grad={st['grad_norm']:.3f} "
                       f"1-zcos={1 - st['z_cos']:.2e} L_align={st['L_align']:.4f} Lp={st['L_phys']:.4f}"
                       + (f" | bc {st['bc_deg']:.1f}deg r={st['bc_ratio']:.2e} "
                          f"new={st['bc_new']:.0%} buf={st['buf_size']}"
                          if "bc_deg" in st else "")
                       + (f" | cma tgt {st['tgt_deg']:.1f}deg sigma={st['cma_sigma']:.4f} "
                          f"best={st['tgt_cost_best']:.4f}"
                          if "tgt_deg" in st else "")
                       # w is what the run is FOR: if it sits at 1.0 the weighting
                       # is inert and the run is an expensive copy of its baseline.
                       # whether the heads actually specialised. Pinned at 1.00
                       # means every clip picked the same head and the run is a
                       # single-head run wearing a costume.
                       + (f" | heads top={st['head_win_max_share']:.2f} "
                          f"H={st['head_win_entropy']:.2f}"
                          if "head_win_max_share" in st else "")
                       + (f" | w {st['row_w_mean']:.2f} [{st['row_w_min']:.2f}-"
                          f"{st['row_w_max']:.2f}] floored={st['row_w_floored']:.0%}"
                          if "row_w_mean" in st else ""))

        if cfg.eval_every and (update + 1) % cfg.eval_every == 0:
            do_eval(update + 1)
        if (update + 1) % cfg.ckpt_every == 0:
            path = ckpt_dir / f"update_{update + 1:05d}.pt"
            blob = {"adapter": adapter.state_dict(), "update": update + 1,
                    "cfg": cfg, "bodies": bodies}
            if cma_state:
                # the teacher means (and sigmas) -- the same kind of labelled
                # (clip, body) -> z dataset as best_buf, cheap to keep
                blob["cma_means"] = {"|".join(map(str, k)): (np.asarray(es.mean, dtype=np.float32), float(es.sigma))
                                     for k, es in cma_state.items()}
            if best_buf:
                # The buffer IS a labelled (clip, body) -> z dataset, collected
                # for free by training. Saved so a run can be resumed without
                # losing it and so it can be exported, but only when the feature
                # is on -- ~1 MB that a default run has no reason to carry.
                blob["best_buf"] = {"|".join(map(str, k)): (v[0], v[1])
                                    for k, v in best_buf.items()}
            if head_sel:
                # The selection table is part of a multi-head model: without it
                # nothing downstream can reproduce which head a clip uses, and
                # the checkpoint scores a different model than the run reported.
                blob["head_sel"] = {"|".join(map(str, k)): v for k, v in head_sel.items()}
            torch.save(blob, path)
            tqdm.write(f"saved checkpoint -> {path}")

    for c in ctxs.values():
        c["env"].close()
    if cfg.use_wandb:
        wandb.finish()

    if history:
        write_eval_history(ckpt_dir, history)
        print("\n=== held-out evaluation ===")
        print(f"{'update':>7s} {'train cost':>11s} {'test cost':>10s} {'gap':>9s} "
              f"{'train L_align':>13s} {'test L_align':>13s}")
        for u, ev in history:
            tr, te = ev.get("train"), ev.get("test")
            if tr and te:
                print(f"{u:7d} {tr['cost']:11.4f} {te['cost']:10.4f} "
                      f"{te['cost'] - tr['cost']:+9.4f} {tr['L_align']:13.4f} {te['L_align']:13.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--loss", default="both", choices=["both", "L_align", "L_phys", "bfm"],
                   help="bfm: BFMTrack's latent-space alignment 1 - cos(B(s), B(g)) as the "
                        "whole fitness (model/bfm_align.py), the joint-space terms logged only. "
                        "Single-term like L_align, so the rank normalisation makes its scale "
                        "irrelevant; what changes is WHICH rollouts rank high -- no global-"
                        "position term, so ground motions are not dominated by root drift")
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None, help="ES perturbation scale on z")
    p.add_argument("--pairs", type=int, default=None, help="antithetic pairs per clip")
    p.add_argument("--bodies", type=int, default=None,
                   help="bodies per update (default: all training bodies). One "
                        "batched rollout each, so this multiplies the simulator "
                        "cost of an update -- lower it only if wall clock matters "
                        "more than varying beta inside one gradient")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="residual scale in z_beta = z0 + alpha * MLP. Sets how far "
                        "a given MLP output moves z: tan(angle) = alpha*|MLP|/16, "
                        "so 0.1 needs |MLP| ~ 92 for 30 deg. Ignored under "
                        "--no-residual")
    p.add_argument("--lambda-bc", type=float, default=None,
                   help="weight of the pull toward the best candidate ever "
                        "measured for each (clip, body). 0 (default) is OFF and "
                        "bit-identical to the loop without it. This gradient is "
                        "EXACT while the ES one is estimated, so it can take over "
                        "at a bc_ratio well under 1 -- try 0.1..1.0 and read the "
                        "logged bc_ratio / bc_deg before going higher")
    p.add_argument("--lambda-z", type=float, default=None,
                   help="weight of the (1 - cos(z_beta, z0)) anchor. Inert at the "
                        "default while z sits on z0 (measured g_ratio ~5e-4), but "
                        "under --no-residual z starts ~87 deg away, 1-cos is O(1), "
                        "and 10.0 puts the anchor at ~11%% of the ES gradient "
                        "PULLING BACK TO z0 -- i.e. fighting the ablation. Turn it "
                        "down (0 disables) when running --no-residual")
    p.add_argument("--es-algo", default=None, choices=["es", "cmaes"],
                   help="per-cell search that proposes the direction (ESConfig.es_algo)")
    p.add_argument("--cma-teacher", action="store_true",
                   help="cmaes: cell means persist across visits and walk on their own; "
                        "the adapter follows them (ESConfig.cma_teacher)")
    p.add_argument("--cma-diagonal", action="store_true", help="cmaes: sep-CMA (diagonal covariance)")
    p.add_argument("--no-residual", action="store_true",
                   help="z_beta = MLP([beta, z0]) with no z0 skip and no alpha. "
                        "Removes the prior that the answer is near z0 -- and with "
                        "projection on, starts z_beta at a RANDOM point of the "
                        "sphere instead of on z0, so the frozen actor begins from "
                        "a latent it has no reason to like. Check the printed "
                        "'at init cos(z_beta, z0)' before reading the curve")
    p.add_argument("--category", default=None,
                   choices=["all", "ground", "upright", "move"],
                   help="restrict training to one task list written by "
                        "scripts/spilt_tasks.py. 'ground' (24) and "
                        "'upright' (30) are the P.fall partition -- on ground, "
                        "fall is a large per-task constant the policy cannot "
                        "remove, so pooling the two makes most of the objective "
                        "an offset. 'move' (17) is a SUBSET of upright, not a "
                        "fourth disjoint group: the upright tasks whose "
                        "reference root actually travels, where beta acts "
                        "through stride rather than through reach")
    p.add_argument("--train-bodies", nargs="+", default=None,
                   help="train on these bodies only, overriding splits/train_bodies.txt. "
                        "ONE body makes beta constant: the adapter becomes a plain z0 -> z "
                        "MLP and the only split that means anything is --heldout-clip-frac")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="decoupled weight decay (switches Adam -> AdamW). Bounds the adapter's "
                        "output, and so how far z_beta can leave z0; 0 (default) is plain Adam")
    p.add_argument("--init-reference", action="store_true",
                   help="start every rollout from its clip's retargeted frame 0 (qvel 0), the "
                        "regime scripts/single_z_search.py and the z0 ranking use, instead of "
                        "humenv's standing reset (ESConfig.init_from_reference)")
    p.add_argument("--dataset-dir", default=None,
                   help="manifest to train on (ESConfig.dataset_dir). The default "
                        "crossenbodiment-10bodies EXCLUDES child and adult -- "
                        "scripts/build_dataset.py --bodies child builds one that has it")
    p.add_argument("--clip-list", default=None,
                   help="file of '<task> <trial>' lines to train on (scripts/write_clip_list.py)")
    p.add_argument("--clip-categories", default=None,
                   help="file of '<task> <category>' lines. Makes --heldout-clip-frac sample the "
                        "same number of clips from EACH category, so the test number is not "
                        "dominated by whichever category has the most clips")
    p.add_argument("--heldout-clip-frac", type=float, default=None,
                   help="single body only: fraction of the clip list kept out of training and "
                        "scored at every eval as the test row")
    p.add_argument("--n-train-clips", type=int, default=None,
                   help="train on a FIXED random subset of this many clips (0 = all). For "
                        "short probe runs, so each clip is revisited often enough to see "
                        "whether it converges")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-clips", type=int, default=None,
                   help="clips per eval row; capped at batch_size (the eval env's slots)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="env slots per body = 2*pairs*clips_per_update; must be a multiple of 2*pairs")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--clip-balance", default=None, choices=["clip", "category"],
                   help="how a batch's clips are drawn. 'clip' (the default) is uniform over "
                        "the training clips, which spends the budget in proportion to how many "
                        "clips a motion family happens to have. 'category' draws the family "
                        "first, so every family gets an equal share; needs --clip-categories")
    p.add_argument("--row-weight", default=None,
                   choices=["none", "conf", "headroom", "relative"],
                   help="per-row weight on the shaped ES delta, capped at 1 -- the general "
                        "(category-free) fix for one task family dominating the batch. "
                        "'conf' scales a row by how much its own landscape actually moved, so a "
                        "cell already on its plateau stops emitting a full-magnitude gradient in "
                        "a direction the simulator never endorsed. 'headroom' scales it by "
                        "cost/cost_z0, so the budget drifts toward cells that have not improved "
                        "yet. See config.py:row_weight")
    p.add_argument("--row-weight-floor", type=float, default=None)
    p.add_argument("--row-weight-ema", type=float, default=None)
    p.add_argument("--z0-cost-csv", default=None,
                   help="row_weight=headroom: z0_cost.csv from scripts/rank_initial_cost.py, "
                        "measured on the SAME body and the same rollout settings as training")
    p.add_argument("--head", default=None, choices=["residual", "geodesic"],
                   help="how the adapter moves z0. 'geodesic' walks a bounded great circle "
                        "(theta <= --theta-max) instead of adding an unbounded MLP output, which "
                        "makes divergence unreachable and decouples how far from which way")
    p.add_argument("--theta-max", type=float, default=None, help="geodesic head: ceiling in degrees")
    p.add_argument("--heading-weight", type=float, default=None, help="candidate cost += w * heading term")
    p.add_argument("--pos-weight", type=float, default=None, help="candidate cost += w * mean root-xy distance (m)")
    p.add_argument("--anchor-weight", type=float, default=None, help="candidate cost += w * (|z - z0| / 16)^2")
    p.add_argument("--obs-scale", default=None, choices=["auto", "none", "exact"],
                   help="what the actor and B see: raw x fixed multiplier (auto, default), raw (none), or the "
                        "adult-equivalent observation by reverse retargeting (exact, model/exact_obs.py)")
    p.add_argument("--subspace", default=None,
                   help="(k, 256) orthonormal basis .npy: confine the adapter's correction and the ES "
                        "perturbations to its first --subspace-dim rows (model.networks.SubspaceAdapter)")
    p.add_argument("--subspace-dim", type=int, default=None)
    p.add_argument("--hidden", type=int, nargs="+", default=None,
                   help="adapter MLP hidden widths (default [256, 512, 512, 256], 0.66M params)")
    p.add_argument("--heads", type=int, default=None,
                   help="number of candidate corrections the adapter emits per clip. Each head "
                        "gets pairs/heads of the antithetic pairs, so the rollout budget is "
                        "unchanged, and only the head whose candidate scored best is given a "
                        "gradient for that clip -- so a head is shaped by the clips that prefer "
                        "it rather than by the whole batch")
    p.add_argument("--no-rank", action="store_true",
                   help="use the raw (F+ - F-)/(2 sigma) estimate instead of ranks")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--project", default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args()

    cfg = ESConfig(device=a.device or f"cuda:{a.gpu}")
    cfg.lambda_align, cfg.lambda_phys = LOSS_LAMBDAS[a.loss]
    cfg.align_loss = "bfm" if a.loss == "bfm" else "joint"
    # --category names a TASK group, never a per-clip property:
    # spilt_tasks.py averages P.fall (and root displacement) over a
    # task's clips before cutting, so the dataset carries no per-clip label and
    # "move clips" can only mean "clips of a move task".
    for attr, val in (("num_updates", a.updates), ("es_sigma", a.sigma),
                      ("es_pairs", a.pairs), ("es_bodies_per_update", a.bodies),
                      ("lr", a.lr), ("adapter_alpha", a.alpha),
                      ("lambda_z", a.lambda_z), ("lambda_bc", a.lambda_bc),
                      ("n_train_clips", a.n_train_clips), ("eval_every", a.eval_every),
                      ("eval_clips", a.eval_clips), ("weight_decay", a.weight_decay),
                      ("batch_size", a.batch_size),
                      ("ckpt_dir", a.ckpt_dir), ("es_algo", a.es_algo),
                      ("train_bodies", a.train_bodies), ("clip_list", a.clip_list),
                      ("dataset_dir", a.dataset_dir),
                      ("heldout_clip_frac", a.heldout_clip_frac),
                      ("clip_categories", a.clip_categories),
                      ("task_group", a.category),
                      ("clip_balance", a.clip_balance), ("row_weight", a.row_weight), ("row_weight_floor", a.row_weight_floor),
                      ("row_weight_ema", a.row_weight_ema),
                      ("row_weight_z0_csv", a.z0_cost_csv),
                      ("adapter_head", a.head), ("adapter_theta_max_deg", a.theta_max),
                      ("adapter_heads", a.heads), ("adapter_hidden_dims", a.hidden),
                      ("adapter_subspace", a.subspace), ("adapter_subspace_dim", a.subspace_dim),
                      ("obs_scale", a.obs_scale),
                      ("heading_weight", a.heading_weight), ("pos_weight", a.pos_weight),
                      ("anchor_weight", a.anchor_weight),
                      ("wandb_project", a.project), ("wandb_run_name", a.run_name)):
        if val is not None:
            setattr(cfg, attr, val)
    if a.init_reference:
        cfg.init_from_reference = True
    if a.cma_teacher:
        cfg.cma_teacher = True
    if a.cma_diagonal:
        cfg.cma_diagonal = True
    if a.no_residual:
        cfg.adapter_residual = False
    if a.no_rank:
        cfg.es_rank_normalize = False
    if a.no_progress:
        cfg.progress = False
    if a.no_wandb:
        cfg.use_wandb = False
    train(cfg)


if __name__ == "__main__":
    main()
