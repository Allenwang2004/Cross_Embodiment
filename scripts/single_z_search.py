#!/usr/bin/env python3
"""single_z_search.py -- how low can ONE clip's cost go if z is free?

Everything else in this repo learns a MAP from (beta, z0) to z_beta and is
therefore limited by what that map can represent, by how many clips it has to
serve at once, and by how much signal a few hundred updates can carry. This
script removes all three: one clip, one body, and z itself is the parameter.
The number it produces is the CEILING -- if the best z reachable by direct
search on a single clip still does not beat the retargeted reference, no
adapter trained over 5400 rows is going to.

Three objectives, because they answer different questions and the repo's cost
is a sum of two terms that are not obviously compatible:

    align   L_align only        can z reproduce the reference trajectory?
    phys    L_phys only         can z produce a physically clean rollout at all,
                                if it is allowed to ignore the reference?
    both    L_align + L_phys    the objective train_es.py actually minimises
    bfm     1 - cos(B(s), B(g)) BFMTrack's latent-space tracking loss
                                (model/bfm_align.py) in place of L_align --
                                the joint-space L_align and L_phys are still
                                computed and logged, they just do not drive
                                the search

Running all three is the point: if `align` and `phys` pull to different places
in z-space, `both` is a compromise neither of them would choose, and that is
worth knowing before tuning lambda_align / lambda_phys any further.

Method
------
ES on z directly (rank-normalised, common random numbers, deterministic actions)
with independent perturbations and LSO's leave-one-out baseline, the default
since 11/04:

    eps_k  ~ N(0, I_256),  k = 1 .. 2*pairs
    F_k    = cost( rollout( project(z + sigma*eps_k) ) )
    s_k    = shape(F_k)
    g      = sum_k (s_k - mean_{i != k} s_i) * eps_k / (2 * pairs * sigma)
    z     <- project( adam(z, g) )

(Centred ranks sum to zero, so with the default --fitness rank the baseline only
rescales them by 2*pairs / (2*pairs - 1).) --sampling antithetic is the version
every search before 11/04 ran, model/simple/train_es.py's estimator: mirrored
pairs project(z +- sigma*eps_k), g = sum_k shape(F+_k - F-_k) * eps_k / (2 * pairs * sigma).

project() is metamotivo's own projection onto the sphere of radius sqrt(256)
that FB latents live on -- the same constraint cfg.adapter_project_z applies.
Off-sphere z is fed straight through Actor.forward, so an unconstrained search
would be optimizing over a region the frozen actor never saw.

CRN is exact and free here for the same reason as in train_es.py: every slot
starts from a bit-identical state -- humenv's Default reset, or under
--init reference the clip's own frame 0 written into all of them -- and the
actions are the actor's mean, so two rollouts in one step differ ONLY by
their z.

The reported best is the best CANDIDATE ACTUALLY EVALUATED, not the final mean
-- the mean of an ES population is not guaranteed to be better than the points
that produced it, and reporting it without a rollout would be reporting a number
nothing measured.

Usage (from project root):
    uv run scripts/single_z_search.py --clip move-ego-0-2/move-ego-0-2_4 \
        --objective both --evals 10000

    # the three objectives in parallel (each takes ~16 async env slots)
    for o in align phys both; do
      uv run scripts/single_z_search.py --clip move-ego-0-2/move-ego-0-2_4 \
          --objective $o --out outputs/single_z/$o &
    done; wait

Writes <out>/best_z.npy, <out>/curve.csv, <out>/summary.json, <out>/best.npz and
<out>/origin_z.npz (the winning and the baseline rollout's qpos), plus
<out>/z_trace.npz -- every step's mean z and best-so-far z, so the search
can be replayed visually afterwards:

    uv run scripts/rollout_z_trace.py --trace <out>/z_trace.npz --n 8

Render the winner against the reference with:

    uv run scripts/rollout_z_on_body.py --z <out>/best_z.npy \
        --xml assets/robots_torque/child/robot_torque_full.xml \
        --reference data/child/retargeting_motion/<clip>.npz \
        --obs-scale auto --out <out>/best.mp4
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import sys
import time
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# (lambda_align, lambda_phys) per objective -- fed straight into ESConfig so the
# cost compute_batch_cost returns IS the objective, with no second definition of
# it living in this file.
OBJECTIVES = {
    "align": (1.0, 0.0),
    "phys": (0.0, 1.0),
    "both": (1.0, 1.0),
    "bfm": (1.0, 0.0),      # the "align" slot is the BFM latent loss, see score()
    "mse": (1.0, 0.0),      # cost = losses.tracking_mse (pose + ee + root + heading MSE); L_align logged only
    "dtw": (1.0, 0.0),      # cost = 1 - time-aligned B similarity (model/dtw.py: slower allowed) + aligned heading
    "phc": (1.0, 0.0),      # cost = 1 - mean_t PHC tracking reward (model/phc_reward.py; BFM-Zero's trajectory adaptation)
}


def ang_pair(a, b):
    import numpy as _np
    return float(_np.degrees(_np.arccos(_np.clip(
        a @ b / (_np.linalg.norm(a) * _np.linalg.norm(b)), -1, 1))))


def project_z(z: np.ndarray) -> np.ndarray:
    """metamotivo's project_z: onto the sphere of radius sqrt(dim). Works on a
    single vector or a batch."""
    a = np.atleast_2d(z)
    r = np.sqrt(a.shape[-1])
    out = a * (r / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12))
    return out.reshape(z.shape)


def rank_normalize(f: np.ndarray) -> np.ndarray:
    """Centred ranks in [-0.5, 0.5] -- train_es.py:rank_normalize. Strips the
    cost's units and its tail, so one catastrophic rollout sets a direction but
    not the step size."""
    order = np.argsort(np.argsort(f))
    return order / max(len(f) - 1, 1) - 0.5


@torch.no_grad()
def rollout(model, env, z_env, steps, device, obs_mul, init_qpos=None, nv=None,
            return_obs=False, exact=None, act_out=None, exact_in_env=False, init_qvel=None, qvel_out=None):
    """Deterministic rollout, one z per slot. Returns qpos (n_envs, T, nq),
    and with return_obs also the RESCALED proprio of the state after each
    step, (n_envs, T, 358), index-aligned with qpos -- what B() scores.
    Same body as scripts/loss_test.py:rollout, kept separate rather than
    imported so this script does not drag in loss_test's argument surface.

    init_qpos overrides humenv's Default reset with a pose -- here the
    reference's own first frame, so the rollout and the trajectory it is scored
    against start from the SAME place. Without it frame 0 is humenv's standing
    T-pose and the first slice of d_pose/d_ee is an offset no z can remove: the
    policy is being charged for an initial condition it did not choose. qvel is
    zeroed rather than finite-differenced from the reference, matching
    rollout_z_on_body.py:--init-from-reference so the two agree.

    exact (model.exact_obs.ExactObs): the actor and B see the ADULT-EQUIVALENT
    observation of each slot's state (reverse retargeting) instead of raw x obs_mul.
    exact_in_env: the env's workers already compute it (model.exact_obs.ExactObsWrapper) and return it as
    info["exact_proprio"] -- every slot in parallel instead of one after another here, same numbers;
    exact is then only used for the start pose.

    init_qvel: the start velocity instead of 0 (--init-vel-scale).

    act_out (list): if given, each step's actions (n_envs, n_act) are appended to it.
    qvel_out (list): if given, each step's qvel (n_envs, nv) after the step is appended to it (--eval-save-qvel).
    """
    obs, _ = env.reset()
    if init_qpos is not None:
        env.call("set_physics", qpos=init_qpos, qvel=np.zeros(nv) if init_qvel is None else init_qvel)
        obs = {"proprio": np.stack([o["proprio"] for o in env.call("get_obs")])}
    hist, obs_hist = [], []
    if exact is not None:
        if init_qpos is None:
            raise SystemExit("--obs-scale exact needs --init reference")
        ex_obs = np.repeat(exact(init_qpos, np.zeros(nv) if init_qvel is None else init_qvel)[None], z_env.shape[0], 0)
    for _ in range(steps):
        if exact is not None:
            proprio = ex_obs
        else:
            proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.as_tensor(proprio, dtype=torch.float32, device=device)
        mu = model._actor(model._normalize(obs_t), z_env, model.cfg.actor_std).mean
        if act_out is not None:
            act_out.append(mu.cpu().numpy())
        obs, _, _, _, info = env.step(mu.cpu().numpy())
        hist.append(info["qpos"].copy())
        if qvel_out is not None:
            qvel_out.append(np.asarray(info["qvel"]).copy())
        if exact is not None:
            ex_obs = (info["exact_proprio"].copy() if exact_in_env else
                      np.stack([exact(q, v) for q, v in zip(info["qpos"], info["qvel"])]))
        if return_obs:
            obs_hist.append(ex_obs if exact is not None else
                            (obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul))
    if return_obs:
        return np.stack(hist, axis=1), np.stack(obs_hist, axis=1).astype(np.float32)
    return np.stack(hist, axis=1)


def device_arg(s: str) -> str:
    """Accept a bare GPU index ("1") as well as a full torch device string.

    torch.device rejects "1" with 'Invalid device string', and argparse has
    already accepted it by then, so the failure surfaces 90 seconds later
    inside FBcprModel.to() -- after the env and the model download.
    """
    return f"cuda:{s}" if s.isdigit() else s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4",
                   help="<task>/<stem>, as under data/origin_z/")
    p.add_argument("--body", default="child")
    p.add_argument("--objective", default="both", choices=sorted(OBJECTIVES))
    p.add_argument("--evals", type=int, default=10000,
                   help="total rollouts of the search. NOT steps: the "
                        "budget is what costs wall-clock, and steps = "
                        "evals // (2*pairs)")
    p.add_argument("--fitness", default="rank", choices=["rank", "raw"],
                   help="what the ES gradient weights the noise by: rank = centred ranks of the 2*pairs costs "
                        "(the default everywhere), raw = the costs themselves (the baseline or the antithetic difference "
                        "removes their common level; Adam removes the overall scale)")
    p.add_argument("--pairs", type=int, default=8,
                   help="half the rollouts per step (one step = one Adam update "
                        "on z): 2*pairs env slots, 2*pairs perturbations (--sampling)")
    p.add_argument("--bfm-zero-vel", action="store_true",
                   help="objective bfm: B sees the observations with their velocity part (obs[214:], local body "
                        "velocities and angular velocities) set to 0, rollout and reference alike")
    p.add_argument("--fall-weight", type=float, default=0.0,
                   help="objective dtw: + this x the share of the aligned rollout frames with the head below --fall-head "
                        "(a fall the alignment would otherwise absorb by passing slowly over a few reference frames)")
    p.add_argument("--fall-head", type=float, default=0.8, help="head height (m) below which a frame counts as fallen")
    p.add_argument("--mse-no-xy", action="store_true",
                   help="objective mse: tracking_mse's root term compares the pelvis height only, not where it is")
    p.add_argument("--floor-friction", type=float, default=None,
                   help="sliding friction of the floor, which then wins every contact (model/friction.py); default: the model's own (feet 1.0, the rest 0.7)")
    p.add_argument("--sampling", default="independent", choices=["antithetic", "independent"],
                   help="independent = 2*pairs independent eps, each weighted by its score minus the mean of the other "
                        "2*pairs - 1 (LSO's leave-one-out baseline), the default since 11/04; antithetic = eps and -eps, "
                        "weighted by each pair's difference (every search before 11/04). Same rollouts per step either way")
    p.add_argument("--sigma", type=float, default=0.25,
                   help="|sigma*eps| / |z| = sigma, i.e. the FRACTIONAL "
                        "perturbation of z -- see ESConfig.es_sigma")
    p.add_argument("--lr", type=float, default=0.5,
                   help="Adam step on z. |z| = 16, so this is an absolute "
                        "displacement, not a fraction")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--init", default="reference", choices=["reference", "default"],
                   help="'reference' starts the rollout from the retargeted clip's "
                        "own first frame (qvel 0), the same thing "
                        "rollout_z_on_body.py --init-from-reference does. 'default' "
                        "is humenv's standing reset, which every other rollout path "
                        "in the repo uses -- pick that only to stay comparable with "
                        "them, since it charges L_align for a frame-0 mismatch no z "
                        "can fix")
    p.add_argument("--xml", default=None,
                   help="default assets/robots_torque/<body>/robot_torque_full.xml "
                        "-- condition C of scripts/loss_test.py, the best of the "
                        "four and the one training uses")
    p.add_argument("--obs-scale", default="auto", type=lambda s: s if s in ("auto", "none", "exact") or float(s) > 0 else None,
                   help="auto: raw obs x fixed per-feature multiplier (per-body length ratios); none: raw; exact: the "
                        "adult-equivalent observation by reverse retargeting (model/exact_obs.py), for the actor and for B; "
                        "a positive number r: every length feature / r (ONE uniform ratio, model/obs_scale.py's ablation)")
    p.add_argument("--obs-scale-ref", default="assets/robots/adult/robot.xml")
    p.add_argument("--phys-weights", default="balanced",
                   choices=["balanced", "equal", "default"],
                   help="balanced, because this script SUMS L_align and L_phys: "
                        "under 'default' L_phys is ~1e4 and 'both' would be "
                        "L_phys with a rounding error attached")
    p.add_argument("--discount", type=float, default=1.0,
                   help="per-frame weight decay gamma inside every L_align "
                        "sub-term (losses._discounted_mean). 1.0 = the plain "
                        "mean. Lower it when the objective is dominated by "
                        "accumulated drift: at 30 Hz over 300 frames, 0.995 "
                        "leaves the last frame at 22%% of the first's weight, "
                        "0.99 at 5%%")
    p.add_argument("--no-fall-ref", action="store_true", default=True)
    p.add_argument("--cem-elites", type=int, default=4, help="--algo cem: how many of the 2*pairs candidates form the next distribution")
    p.add_argument("--cem-sigma-min", type=float, default=None, help="--algo cem: floor of the per-dimension std (default --sigma / 5): 4 elites in 256 dimensions shrink it about 13%% a step otherwise")
    p.add_argument("--algo", default="es", choices=["es", "cmaes", "cem", "random"],
                   help="es: ES (--sampling) + Adam on z (the default everywhere else). "
                        "cmaes: pycma CMA-ES with popsize 2*pairs and sigma0 = --sigma, "
                        "candidates projected onto the sphere; --lr is unused. "
                        "cem: cross-entropy method, candidates N(mean, std^2) on the sphere, then mean and per-dimension std "
                        "of the --cem-elites best; starts at z with std --sigma. "
                        "random: candidates = the best z so far + sigma * noise on the sphere, it moves only when one beats it. "
                        "--lr, --sampling and --fitness are unused for cem and random")
    p.add_argument("--data-root", default="data",
                   help="where origin_z/, origin_motion/ and <body>/retargeting_motion/ live (default data/; "
                        "e.g. outputs/diffusion_adapter/data for synthetic tasks)")
    p.add_argument("--eval-z", default=None,
                   help="no search: score every latent in this (N, 256) .npy with the run's exact cost (padded to "
                        "batches of 2 * pairs rollouts) and write <out>/eval.csv (index, cost, bfm, head, pos, "
                        "anchor_dist) plus <out>/eval_base.json (the z0 terms), then exit")
    p.add_argument("--eval-save-qvel", action="store_true",
                   help="with --eval-z: also write <out>/eval_qvel.npz (qvel (N, T, nv) after each step; for rewards that need velocities)")
    p.add_argument("--eval-save-qpos", action="store_true",
                   help="with --eval-z: also write <out>/eval_qpos.npz (qpos (N, T, nq)) of every scored latent")
    p.add_argument("--eval-save-act", action="store_true",
                   help="with --eval-z: also write <out>/eval_act.npz (the actor's mean actions (N, T, n_act))")
    p.add_argument("--z-start", default=None,
                   help="start the search from this z instead of from the clip's z0 (a .npy). "
                        "The reported origin_z baseline still uses the real z0, so the two "
                        "conditions stay comparable: this changes WHERE the search begins, "
                        "not what it is measured against. Used to test whether a "
                        "beta-conditioned prior buys search time.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heading-weight", type=float, default=0.0,
                   help="add w * mean_t (1 - cos(heading_s - heading_g)) / 2. The bfm cost is blind to "
                        "global heading and root x,y (humenv's proprio is expressed in the root's heading "
                        "frame and drops x,y): 20 walking solutions of equal cost ended -112..+22 deg "
                        "and 0.2..13.4 m off the reference")
    p.add_argument("--ref", default=None,
                   help="reference motion .npz (qpos) to track instead of data/<body>/retargeting_motion/<task>/<stem>.npz; "
                        "z0 still comes from --clip")
    p.add_argument("--init-vel-scale", type=float, default=None,
                   help="--init reference: start moving at this x the reference's velocity (frames 0 -> 1, 30 Hz) "
                        "instead of at rest, e.g. 0.7 = the reference played at 0.7x speed")
    p.add_argument("--frame-offset", type=int, default=1,
                   help="the state after step t is scored against reference frame t + this. 1: the rollout starts at frame 0, so "
                        "after step t it should be at frame t + 1, as in scripts/lso_search.py. 0 reproduces every run before this "
                        "flag existed, which scored each state one frame late")
    p.add_argument("--init-ref", default=None,
                   help="with --init reference: take frame 0 from this .npz instead of --ref (e.g. track the adult's "
                        "own walk while starting from the body's feasible pose)")
    p.add_argument("--anchor", default=None,
                   help="latent .npy the --anchor-weight term pulls toward (e.g. the original start of a two-stage search)")
    p.add_argument("--anchor-weight", type=float, default=0.0,
                   help="add w * (|z - anchor| / 16)^2 to the cost: among equally good latents, prefer the one nearest "
                        "the anchor (the latent is non-identifiable, so an unregularised search drifts ~16 away)")
    p.add_argument("--best-from-start", action="store_true",
                   help="with --z-start, start best-so-far at the START latent instead of at z0. Without it a "
                        "warm-started search that never beats z0's own cost reports z0 as its answer, which "
                        "is wrong whenever the question is where the search from THAT start ended up")
    p.add_argument("--pos-weight", type=float, default=0.0,
                   help="add w * mean_t |root_xy_s - root_xy_g| (metres, see --pos-unit)")
    p.add_argument("--pos-unit", default="adult", choices=["adult", "body"],
                   help="adult (default since 11/04, as robot_z_search.py): adult-equivalent metres, the body's metres / its "
                        "root scale s (--obs-scale exact only; other obs modes have no s and stay in body metres). "
                        "body: the body's own metres, every search before 11/04")
    p.add_argument("--subspace", default=None,
                   help=".npy of orthonormal rows (K, 256): move only within start + span(first --subspace-dim rows), "
                        "plus the radial re-projection. es only. Perturbations and Adam steps are scaled by "
                        "sqrt(256 / k) so their Euclidean size matches the full-space search")
    p.add_argument("--subspace-dim", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=25,
                   help="also roll out the current mean z this often, so the "
                        "curve shows the iterate and not only its samples")
    p.add_argument("--device", type=device_arg,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel

    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    from model.simple.train import compute_batch_cost
    from model import losses

    task, stem = args.clip.split("/")
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else REPO_ROOT / args.data_root
    z0_path = data_root / "origin_z" / task / f"{stem}.npy"
    ref_path = (Path(args.ref) if args.ref else
                data_root / args.body / "retargeting_motion" / task / f"{stem}.npz")
    for q in (z0_path, ref_path):
        if not q.exists():
            raise SystemExit(f"{q} not found")
    xml = Path(args.xml) if args.xml else (
        REPO_ROOT / "assets" / "robots_torque" / args.body / "robot_torque_full.xml")
    if not xml.exists():
        raise SystemExit(f"{xml} not found")
    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "outputs" / "single_z" / f"{stem}_{args.objective}")
    out_dir.mkdir(parents=True, exist_ok=True)

    z0 = project_z(np.load(z0_path).reshape(-1).astype(np.float64))
    ref = np.load(ref_path)["qpos"]
    goal = ref[args.frame_offset:]                 # goal[t]: what the state after step t is scored against
    lam_a, lam_p = OBJECTIVES[args.objective]

    cfg = ESConfig(device=args.device)
    cfg.lambda_align, cfg.lambda_phys = lam_a, lam_p
    cfg.phys_weights = args.phys_weights
    cfg.phys_fall_ref = not args.no_fall_ref
    cfg.align_discount = args.discount

    n_envs = 2 * args.pairs
    n_gens = max(args.evals // n_envs, 1)
    print(f"clip {args.clip} on {args.body} ({xml.name})")
    if args.objective == "mse":
        print("objective mse: cost = losses.tracking_mse (pose + ee + root + heading, weights 1)"
              + (f" + {args.anchor_weight} * (|z - anchor| / 16)^2" if args.anchor_weight else ""))
    elif args.objective == "bfm":
        print("objective bfm: cost = 1 - mean_t cos(B(s_t), B(g_t))  (BFMTrack Eq. 3, "
              "B fed the same rescaled obs as the actor); L_align / L_phys logged only")
    elif args.objective == "phc":
        print("objective phc: cost = 1 - mean_t PHC tracking reward (Luo et al. 2023 Eq. 2 as PHC's code computes it: "
              "body positions, rotations, velocities, angular velocities; model/phc_reward.py); L_align / L_phys logged only")
    else:
        print(f"objective {args.objective}: cost = {lam_a} * L_align + {lam_p} * L_phys "
              f"(L_phys weights={cfg.phys_weights}, L_align discount={cfg.align_discount})")
    print(f"{n_gens} steps x {n_envs} rollouts = {n_gens * n_envs} evals, "
          f"algo {args.algo}, sigma {args.sigma}, lr {args.lr}, reference {len(ref)} frames")
    print(f"init: {args.init}" + (" (rollout starts from the reference's frame 0)"
                                  if args.init == "reference" else
                                  " (humenv standing reset)"))

    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device)
    model.eval()
    kw, wrappers = None, None
    if args.obs_scale == "exact":
        from model.exact_obs import ExactObs, ExactObsWrapper
        kw = {}
        if args.body != "child":
            # every body's retarget copies the joints and scales root x, y by its own s (constant to 1e-16),
            # so read s off this clip against the adult's; the child keeps ExactObs's measured default
            a = np.load(data_root / "origin_motion" / task / f"{stem}.npz")["qpos"][:, :2]
            n = min(len(a), len(ref)); m = np.abs(a[:n]) > 1e-3
            kw["scale"] = float(np.median(ref[:n, :2][m] / a[:n][m]))
            print(f"exact obs: root scale {kw['scale']:.6f} (from the reference vs the adult motion)")
        # each env worker computes its own slot's obs right after its physics step (was ~46% of a step when
        # the main process did all slots in turn)
        wrappers = [functools.partial(ExactObsWrapper, child_xml=str(xml),
                                      adult_xml=str(REPO_ROOT / args.obs_scale_ref), **kw)]
    if args.floor_friction is not None:       # a slippery / sticky floor in every worker (model/friction.py)
        from model.friction import FloorFrictionWrapper
        wrappers = [functools.partial(FloorFrictionWrapper, mu=args.floor_friction)] + (wrappers or [])
        print(f"floor friction {args.floor_friction} (the floor wins every contact)")
    env, _ = make_humenv(num_envs=n_envs, vectorization_mode="async", task=None, max_episode_steps=max(args.steps, 300),
                        xml=str(xml), state_init="Default", wrappers=wrappers)
    fk = mujoco.MjModel.from_xml_path(str(xml))
    exact = None
    if kw is not None:
        exact, obs_mul = ExactObs(xml, REPO_ROOT / args.obs_scale_ref, **kw), None
    else:
        obs_mul = build_obs_multiplier(xml, REPO_ROOT / args.obs_scale_ref,
                                       mode=args.obs_scale, parts=cfg.obs_scale_parts,
                                       verbose=False)

    init_qpos = ((np.load(args.init_ref)["qpos"][0] if args.init_ref else ref[0])
                 if args.init == "reference" else None)
    init_qvel = None
    if args.init_vel_scale is not None and init_qpos is not None:
        src = np.load(args.init_ref)["qpos"] if args.init_ref else ref
        init_qvel = np.zeros(fk.nv); mujoco.mj_differentiatePos(fk, init_qvel, 1.0 / 30, src[0], src[1])
        init_qvel *= args.init_vel_scale
        print(f"start velocity: {args.init_vel_scale:g} x the reference's (pelvis {np.linalg.norm(init_qvel[:2]):.2f} m/s)")

    Bg = None
    VEL = slice(214, 358)                    # humenv proprio: local_body_vel + local_body_ang_vel
    if args.objective in ("bfm", "dtw"):
        from model import bfm_align
        env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
        if exact is None:
            ro = bfm_align.obs_from_qpos(env1, ref, bfm_align.DEFAULT_DT, obs_mul)
            if args.bfm_zero_vel:
                ro[:, VEL] = 0
            Bg = bfm_align.embed(model, ro, args.device)
        else:
            # the reference through the SAME reverse retargeting as the rollout states
            vr, ro = np.zeros(fk.nv), []
            for t in range(len(ref)):
                if t:
                    mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, ref[t - 1], ref[t])
                ro.append(exact(ref[t], vr))
            ro = np.stack(ro)
            if args.bfm_zero_vel:
                ro[:, VEL] = 0
            Bg = bfm_align.embed(model, ro, args.device)

    if Bg is not None:
        Bg = Bg[args.frame_offset:]
    glob_w = (args.heading_weight, args.pos_weight)
    pos_div = exact.s if args.pos_unit == "adult" and exact is not None else 1.0
    if args.pos_weight:
        print(f"position term: metres / {pos_div:.6f} ({args.pos_unit if exact is not None else 'body, no s without --obs-scale exact'})")
    phc = None
    if args.objective == "phc":
        from model.phc_reward import PHCCost
        phc = PHCCost(fk, ref[max(args.frame_offset - 1, 0)], goal, scale=pos_div)
        print(f"phc: lengths / {pos_div:.6f} ({args.pos_unit if exact is not None else 'body'} metres), {fk.nbody - 1} bodies")
    anchor = None
    if args.anchor_weight:
        if not args.anchor:
            raise SystemExit("--anchor-weight needs --anchor")
        anchor = project_z(np.load(args.anchor).reshape(-1).astype(np.float64))
    extra = {}

    def root_heading_xy(qs):
        """(..., T, nq) qpos -> heading (..., T) in radians and root xy (..., T, 2)."""
        import humenv.utils as hu
        flat = qs.reshape(-1, qs.shape[-1]); dk = mujoco.MjData(fk); h = np.empty(len(flat))
        for i, qq in enumerate(flat):
            dk.qpos[:] = qq; mujoco.mj_kinematics(fk, dk)
            h[i] = hu.calc_heading(hu.remove_base_rot(dk.xquat[1][None].copy(), "smpl"))[0]
        return h.reshape(qs.shape[:-1]), qs[..., :2]

    ref_h, ref_xy = root_heading_xy(goal)

    def tmse(qi):
        return losses.tracking_mse(fk, qi, goal, root_xy=not args.mse_no_xy)[0]

    DTW_KEYS = ("dtw", "done", "pace", "head_al", "fallen")

    def dext(i, with_path=True):
        """the dtw numbers of slot i of the last score(), for the best / base records"""
        d = {k: float(extra[k][i]) for k in ("dtw", "done", "pace", "head_al", "fallen") if k in extra}
        if with_path and "path" in extra:
            d["path"] = extra["path"][i].copy()
        return d

    def global_terms(q):
        T = min(q.shape[1], len(goal))
        h, xy = root_heading_xy(q[:, :T])
        head = ((1 - np.cos(h - ref_h[None, :T])) / 2).mean(1)
        pos = np.linalg.norm(xy - ref_xy[None, :T], axis=-1).mean(1) / pos_div
        return head, pos

    def score(zs):
        """zs: (n, 256) -> (cost, L_align, L_phys, bfm, qpos). One batched rollout.
        cost is the objective; L_align / L_phys are always the joint-space
        terms, so a bfm run's summary stays comparable with the others."""
        zt = torch.as_tensor(zs, dtype=torch.float32, device=args.device)
        acts, vels = [], ([] if args.eval_save_qvel else None)
        q, o = rollout(model, env, zt, args.steps, args.device, obs_mul, init_qpos=init_qpos, nv=fk.nv,
                       return_obs=True, exact=exact, act_out=acts, exact_in_env=exact is not None, init_qvel=init_qvel, qvel_out=vels)
        extra["act"] = np.stack(acts, axis=1)
        if vels is not None:
            extra["qvel"] = np.stack(vels, axis=1)
        if Bg is not None or phc is not None:
            # bfm / phc only log L_align / L_phys / tracking_mse: logged() computes them for the slots kept
            cost = align = phys = extra["mse"] = np.full(len(zs), np.nan)
        else:
            cost, align, phys = compute_batch_cost(fk, cfg, q, [goal] * len(zs))
            extra["mse"] = np.array([tmse(qi) for qi in q])
        bfm = np.full(len(zs), np.nan, dtype=np.float32)
        if Bg is not None:
            if args.bfm_zero_vel:
                o = np.array(o, copy=True); o[..., VEL] = 0
            if args.objective == "dtw":
                from model import dtw
                n_, T_, D_ = o.shape
                Bs = bfm_align.embed(model, o.reshape(n_ * T_, D_), args.device).reshape(n_, T_, -1)
                bfm = np.array([bfm_align.bfm_align_loss(Bs[k], Bg) for k in range(n_)], dtype=np.float32)
                sc, je, ie, path = dtw.align_slow(dtw.cosine_matrix(Bs, Bg))
                extra["dtw"], extra["done"], extra["pace"], extra["path"] = 1 - sc, (je + 1) / len(Bg), (je + 1) / (ie + 1), path
            else:
                bfm = bfm_align.batch_bfm_align(model, o, Bg, args.device)
            cost = bfm
        if args.objective == "mse":
            cost = extra["mse"].astype(np.float32)
        if phc is not None:                            # rollouts start at init_qpos (or wherever humenv put frame 0)
            cost = phc(q, init_qpos if init_qpos is not None else q[0, 0])
        extra["head"], extra["pos"] = global_terms(q)
        if args.objective == "dtw":                    # heading along the alignment; pos stays frame by frame
            h_all, _ = root_heading_xy(q); P = extra["path"]
            extra["head_al"] = np.array([((1 - np.cos(h_all[k][P[k] >= 0] - ref_h[P[k][P[k] >= 0]])) / 2).mean()
                                         for k in range(len(zs))])
            dk, hid = mujoco.MjData(fk), fk.body("Head").id
            hz = np.empty(q.shape[:2])
            for k in range(q.shape[0]):
                for i in range(q.shape[1]):
                    dk.qpos[:] = q[k, i]; mujoco.mj_kinematics(fk, dk); hz[k, i] = dk.xpos[hid][2]
            extra["fallen"] = np.array([(hz[k][P[k] >= 0] < args.fall_head).mean() for k in range(len(zs))])
            cost = (extra["dtw"] + glob_w[0] * extra["head_al"] + glob_w[1] * extra["pos"]
                    + args.fall_weight * extra["fallen"])
        elif any(glob_w):
            cost = cost + glob_w[0] * extra["head"] + glob_w[1] * extra["pos"]
        extra["anchor_dist"] = (np.linalg.norm(np.asarray(zs) - anchor, axis=1) if anchor is not None
                                else np.full(len(zs), np.nan))
        if anchor is not None:
            cost = cost + args.anchor_weight * (extra["anchor_dist"] / 16.0) ** 2
        return cost, align, phys, bfm, q

    def logged(i, q, align, phys):
        """(L_align, L_phys, tracking_mse) of slot i of the last score() -- computed here for that one rollout
        when the objective did not need them for every slot"""
        if Bg is None and phc is None:
            return float(align[i]), float(phys[i]), float(extra["mse"][i])
        _, a1, p1 = compute_batch_cost(fk, cfg, q[i][None], [goal])
        return float(a1[0]), float(p1[0]), float(tmse(q[i]))

    # --- where we start, and the floor we are aiming at ----------------------
    c0, a0, p0, f0, q0 = score(np.repeat(z0[None], n_envs, axis=0))
    la, lp, lm = logged(0, q0, a0, p0)
    base = dict(cost=float(c0[0]), align=la, phys=lp, bfm=float(f0[0]),
                head=float(extra["head"][0]), pos=float(extra["pos"][0]), mse=lm, **dext(0, False))
    base_path = extra["path"][0].copy() if "path" in extra else None
    _, ra, rp = compute_batch_cost(fk, cfg, goal[None, :args.steps], [goal])
    floor = dict(cost=float(lam_a * ra[0] + lam_p * rp[0]),
                 align=float(ra[0]), phys=float(rp[0]), bfm=0.0 if Bg is not None else float("nan"))
    if Bg is not None:
        # the kinematic reference against itself, through obs_from_qpos on both
        # sides -- exactly 0 by construction, so the floor is 0
        floor["cost"] = 0.0
    if phc is not None:                                # the reference against itself, from its own frame 0: 0
        floor["cost"] = float(phc(goal[None, :args.steps], ref[max(args.frame_offset - 1, 0)])[0])
    print(f"  origin_z : cost {base['cost']:.4f}  L_align {base['align']:.4f}  "
          f"L_phys {base['phys']:.4f}")
    print(f"  reference: cost {floor['cost']:.4f}  L_align {floor['align']:.4f}  "
          f"L_phys {floor['phys']:.4f}   <- kinematic floor")

    if args.eval_z:
        Z = np.load(args.eval_z).reshape(-1, z0.size).astype(np.float64)
        rows, qs, acts, vels = [], [], [], []
        for i in range(0, len(Z), n_envs):
            chunk = Z[i:i + n_envs]
            pad = np.concatenate([chunk, np.repeat(z0[None], n_envs - len(chunk), axis=0)])
            c, _, _, f, qq = score(pad)
            qs.append(np.asarray(qq)[:len(chunk)]); acts.append(extra["act"][:len(chunk)])
            if args.eval_save_qvel:
                vels.append(extra["qvel"][:len(chunk)])
            for k in range(len(chunk)):
                rows.append((i + k, float(c[k]), float(f[k]), float(extra["head"][k]), float(extra["pos"][k]),
                             float(extra["anchor_dist"][k]))
                            + tuple(float(extra[x][k]) for x in DTW_KEYS if x in extra))
        with open(out_dir / "eval.csv", "w") as fh:
            fh.write("index,cost,bfm,head,pos,anchor_dist" + "".join("," + x for x in DTW_KEYS if x in extra) + "\n")
            fh.writelines(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in r) + "\n" for r in rows)
        (out_dir / "eval_base.json").write_text(json.dumps(base))
        if args.eval_save_qpos:
            np.savez_compressed(out_dir / "eval_qpos.npz", qpos=np.concatenate(qs).astype(np.float32))
        if args.eval_save_act:
            np.savez_compressed(out_dir / "eval_act.npz", act=np.concatenate(acts).astype(np.float32))
        if args.eval_save_qvel:
            np.savez_compressed(out_dir / "eval_qvel.npz", qvel=np.concatenate(vels).astype(np.float32))
        print(f"eval: {len(rows)} latents scored -> {out_dir / 'eval.csv'}")
        return

    # --- ES ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    z_start = z0.copy()
    if args.z_start:
        z_start = project_z(np.load(args.z_start).reshape(-1).astype(np.float64))
        cs, ca, cp, cf, cq = score(np.repeat(z_start[None], n_envs, axis=0))
        la, lp, lm = logged(0, cq, ca, cp)
        start_best = dict(cost=float(cs[0]), align=la, phys=lp, bfm=float(cf[0]),
                          head=float(extra["head"][0]), pos=float(extra["pos"][0]), mse=lm, **dext(0),
                          z=z_start.copy(), qpos=cq[0].copy(), gen=-1, source="z_start")
        print(f"  warm start: cost {float(cs[0]):.4f} "
              f"({float(cs[0]) / max(base['cost'], 1e-9):.3f} x origin_z), "
              f"{ang_pair(z_start, z0):.1f} deg from z0")
    z = z_start.copy()
    U = None
    if args.subspace:
        if args.algo != "es":
            raise SystemExit("--subspace is for --algo es")
        U = np.load(args.subspace).astype(np.float64)
        U = U[: (args.subspace_dim or len(U))]
        assert np.allclose(U @ U.T, np.eye(len(U)), atol=1e-6), "--subspace rows must be orthonormal"
        sub_scale = np.sqrt(z.size / len(U))
        print(f"subspace: {len(U)} of {z.size} dims ({args.subspace}), step scale {sub_scale:.2f}")
    # Adam lives in the search coordinates: z itself, or the subspace coefficients (Adam's per-coordinate
    # scaling would otherwise push a 256-d step out of the subspace)
    m = np.zeros(z.size if U is None else len(U))
    v = np.zeros_like(m)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    cem_sigma = np.full(z.size, float(args.sigma)); cem_sigma_min = args.cem_sigma_min if args.cem_sigma_min is not None else args.sigma / 5
    cma_es = None
    if args.algo == "cmaes":
        import cma
        cma_es = cma.CMAEvolutionStrategy(z_start.copy(), args.sigma,
                                          {"popsize": n_envs, "seed": args.seed + 1, "verbose": -9})
    path_len = 0.0   # cumulative angle the mean has travelled, deg; vs the net angle = directness

    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))

    best = dict(cost=base["cost"], align=base["align"], phys=base["phys"], bfm=base["bfm"],
                head=base["head"], pos=base["pos"], mse=base["mse"],
                z=z0.copy(), qpos=q0[0].copy(), gen=-1, source="origin_z")
    if args.z_start and args.best_from_start:
        best = start_best
    curve = []
    # Every step's iterate and its best-so-far, so any point in the search
    # can be rolled out afterwards without re-running it (256 floats a row --
    # 625 steps is under a megabyte, so this is not worth a flag).
    trace_mean, trace_best = [z_start.copy()], [z_start.copy()]
    t0 = time.time()
    for gen in range(n_gens):
        if args.algo == "cem":
            cand = project_z(z + cem_sigma * rng.standard_normal((n_envs, z.size)))
        elif args.algo == "random":
            cand = project_z(best["z"] + args.sigma * rng.standard_normal((n_envs, z.size)))
        elif cma_es is not None:
            cand = project_z(np.asarray(cma_es.ask(), dtype=np.float64))
        else:
            n_eps = args.pairs if args.sampling == "antithetic" else 2 * args.pairs
            if U is None:
                eps = rng.standard_normal((n_eps, z.size))
            else:
                xi = sub_scale * rng.standard_normal((n_eps, len(U)))
                eps = xi @ U
            if args.sampling == "antithetic":
                cand = project_z(np.concatenate([z + args.sigma * eps,
                                                 z - args.sigma * eps], axis=0))
            else:
                cand = project_z(z + args.sigma * eps)
        cost, align, phys, bfm, qpos = score(cand)

        i = int(np.argmin(cost))
        if cost[i] < best["cost"]:
            la, lp, lm = logged(i, qpos, align, phys)
            best = dict(cost=float(cost[i]), align=la,
                        phys=lp, bfm=float(bfm[i]), z=cand[i].copy(),
                        qpos=qpos[i].copy(), gen=gen, source="sample",
                        head=float(extra["head"][i]), pos=float(extra["pos"][i]), mse=lm, **dext(i))

        z_prev = z
        if args.algo == "cem":
            elite = cand[np.argsort(cost)[:args.cem_elites]]
            z = project_z(elite.mean(0)); cem_sigma = np.maximum(elite.std(0), cem_sigma_min)
        elif args.algo == "random":
            z = best["z"].copy()
        elif cma_es is not None:
            # tell the projected candidates (what was actually scored); keep the
            # mean on the sphere too so sigma stays a fractional perturbation
            cma_es.tell(list(cand), list(map(float, cost)))
            cma_es.mean = project_z(np.asarray(cma_es.mean, dtype=np.float64))
            z = cma_es.mean.copy()
        else:
            s = rank_normalize(cost) if args.fitness == "rank" else np.asarray(cost, dtype=np.float64)
            if args.sampling == "antithetic":
                g = ((s[:args.pairs] - s[args.pairs:])[:, None] * (eps if U is None else xi)).sum(0) \
                    / (2 * args.pairs * args.sigma)
            else:                                   # leave-one-out baseline, as lso_search.py
                adv = s - (s.sum() - s) / (len(s) - 1)
                g = (adv[:, None] * (eps if U is None else xi)).sum(0) / (2 * args.pairs * args.sigma)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** (gen + 1))
            vh = v / (1 - b2 ** (gen + 1))
            step = args.lr * mh / (np.sqrt(vh) + eps_adam)
            z = project_z(z - (step if U is None else sub_scale * step @ U))
        path_len += ang(z_prev, z)

        row = dict(gen=gen, evals=(gen + 1) * n_envs,
                   gen_best=float(cost.min()), gen_mean=float(cost.mean()),
                   best_so_far=best["cost"], mean_z_cost="",
                   z_start=(args.z_start or "origin_z"),
                   start_deg_from_z0=ang_pair(z_start, z0),
                   cos_z0=float(np.dot(z, z0) / (np.linalg.norm(z) * np.linalg.norm(z0))),
                   path_len_deg=round(path_len, 2), net_deg=round(ang(z0, z), 2),
                   sigma=float(cma_es.sigma) if cma_es is not None else float(cem_sigma.mean()) if args.algo == "cem" else args.sigma)
        # The iterate itself, not just its samples -- one extra rollout, and it
        # is the only way to see whether the mean is tracking the population.
        if args.eval_every and (gen % args.eval_every == 0 or gen == n_gens - 1):
            mc, ma, mp, mf, mq = score(np.repeat(z[None], n_envs, axis=0))
            row["mean_z_cost"] = float(mc[0])
            if mc[0] < best["cost"]:
                la, lp, lm = logged(0, mq, ma, mp)
                best = dict(cost=float(mc[0]), align=la, phys=lp,
                            bfm=float(mf[0]), z=z.copy(), qpos=mq[0].copy(), gen=gen,
                            source="mean_z", head=float(extra["head"][0]), pos=float(extra["pos"][0]), mse=lm, **dext(0))
            el = time.time() - t0
            print(f"  gen {gen:5d}/{n_gens}  best {best['cost']:9.4f}  "
                  f"gen_best {cost.min():9.4f}  mean_z {mc[0]:9.4f}  "
                  f"cos(z,z0) {row['cos_z0']:.3f}  path {path_len:5.0f} deg / net {row['net_deg']:4.0f} deg"
                  + (f"  sigma {cma_es.sigma:.4f}" if cma_es is not None else "")
                  + (f"  | best dtw {best['dtw']:.4f} done {best['done']:.2f} pace {best['pace']:.2f} heading_al {best['head_al']:.4f}"
                     f" fallen {best['fallen']:.3f}"
                     if "pace" in best else "") + f"  [{el / 60:.1f} min]", flush=True)
        curve.append(row)
        trace_mean.append(z.copy())
        trace_best.append(best["z"].copy())

    env.close()
    if Bg is not None:
        env1.close()

    # --- what came out -------------------------------------------------------
    np.save(out_dir / "best_z.npy", best["z"].astype(np.float32).reshape(1, -1))
    np.savez_compressed(out_dir / "best.npz", qpos=best["qpos"].astype(np.float32),
                        fps=cfg.control_fps, **({"dtw_path": best["path"]} if "path" in best else {}))
    # The origin_z rollout under the SAME init and env as the search. Saved so a
    # plot does not have to re-derive the baseline from a separate render, which
    # would not reproduce it: a rollout that falls is chaotic enough that a
    # different device or a different env wrapper lands somewhere else entirely
    # (measured: L_align 1.569 in the async search vs 1.290 re-rendered).
    np.savez_compressed(out_dir / "origin_z.npz", qpos=q0[0].astype(np.float32),
                        fps=cfg.control_fps, **({"dtw_path": base_path} if base_path is not None else {}))
    # gen -1 is z0 itself, so row i is "after i steps" and row 0 is the
    # starting point -- scripts/rollout_z_trace.py indexes it that way.
    np.savez_compressed(
        out_dir / "z_trace.npz",
        gen=np.arange(-1, len(trace_mean) - 1, dtype=np.int32),
        z_mean=np.stack(trace_mean).astype(np.float32),
        z_best=np.stack(trace_best).astype(np.float32))
    with open(out_dir / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)

    summary = {
        "clip": args.clip, "body": args.body, "xml": str(xml), "ref": str(ref_path),
        "objective": args.objective, "bfm_zero_vel": args.bfm_zero_vel,
        "fall_weight": args.fall_weight, "fall_head": args.fall_head, "init_vel_scale": args.init_vel_scale, "mse_no_xy": args.mse_no_xy,
        "lambda_align": lam_a, "lambda_phys": lam_p,
        "phys_weights": cfg.phys_weights, "phys_fall_ref": cfg.phys_fall_ref,
        # "generations" is the on-disk name for what the logs and figures now
        # call steps -- kept so older summary.json files stay readable
        "evals": n_gens * n_envs, "generations": n_gens, "pairs": args.pairs,
        "sigma": args.sigma, "lr": args.lr, "steps": args.steps, "algo": args.algo, "fitness": args.fitness, "sampling": args.sampling, "floor_friction": args.floor_friction,
        "path_len_deg": round(path_len, 1), "net_deg": round(ang(z0, z), 1),
        "obs_scale": args.obs_scale, "seed": args.seed, "init": args.init, "init_ref": args.init_ref, "frame_offset": args.frame_offset,
        "align_discount": args.discount, "heading_weight": args.heading_weight, "pos_weight": args.pos_weight,
        "pos_unit": args.pos_unit if exact is not None else "body", "pos_div": pos_div,
        "anchor": args.anchor, "anchor_weight": args.anchor_weight,
        "subspace": args.subspace, "subspace_dim": None if U is None else len(U),
        "origin_z": base, "reference_floor": floor,
        "best": {k: best[k] for k in ("cost", "align", "phys", "bfm", "gen", "source", "head", "pos", "mse",
                                      "dtw", "done", "pace", "head_al", "fallen") if k in best},
        "improvement_vs_origin_z": (base["cost"] - best["cost"]) / max(abs(base["cost"]), 1e-12),
        "cos_best_z0": float(np.dot(best["z"], z0)
                             / (np.linalg.norm(best["z"]) * np.linalg.norm(z0))),
        "minutes": (time.time() - t0) / 60,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.objective}: {n_gens * n_envs} evals in "
          f"{summary['minutes']:.1f} min ===")
    print(f"{'':12s}{'cost':>10s}{'L_align':>10s}{'L_phys':>10s}")
    print(f"{'origin_z':12s}{base['cost']:10.4f}{base['align']:10.4f}{base['phys']:10.4f}")
    print(f"{'best z':12s}{best['cost']:10.4f}{best['align']:10.4f}{best['phys']:10.4f}"
          f"   ({100 * summary['improvement_vs_origin_z']:+.1f}%, from {best['source']} "
          f"at gen {best['gen']})")
    print(f"{'reference':12s}{floor['cost']:10.4f}{floor['align']:10.4f}{floor['phys']:10.4f}"
          f"   <- kinematic floor")
    print(f"cos(best z, z0) = {summary['cos_best_z0']:.4f}")
    if "pace" in best:
        for n_, b_ in (("origin_z", base), ("best z", best)):
            print(f"{n_:9s} dtw {b_['dtw']:.4f}  done {b_['done']:.2f}  pace {b_['pace']:.2f}  heading_al {b_['head_al']:.4f}  "
                  f"fallen {b_['fallen']:.3f}  "
                  f"(frame-by-frame bfm {b_['bfm']:.4f})")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
