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
import os
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402,F401  (imported first, as before)
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# OOP refactor step 6: the search lives in the package -- the cost terms (crossenbodiment/costs.py), the problem
# (crossenbodiment/problems.py HumanLatentProblem: body, reference, observation mapper, simulator, policy, cost), the
# optimisers (crossenbodiment/optim.py: ES, CEM, CMAES, RandomSearch), the loop (crossenbodiment/search.py SearchLoop) and the
# output files (crossenbodiment/records.py RunRecorder). OBJECTIVES ((lambda_align, lambda_phys) per --objective) moved
# with it and is re-exported here.
from crossenbodiment.problems import OBJECTIVES  # noqa: E402,F401

# ang_pair, project_z (metamotivo's |z| = 16 projection) and rank_normalize live in crossenbodiment/geometry.py (refactor step 3);
# imported here so `from single_z_search import project_z, rank_normalize, ang_pair` keeps working
from crossenbodiment.geometry import ang_pair, project_z, rank_normalize, smpl_heading  # noqa: E402,F401


def rollout(model, env, z_env, steps, device, obs_mul, init_qpos=None, nv=None,
            return_obs=False, exact=None, act_out=None, exact_in_env=False, init_qvel=None, qvel_out=None):
    """Deterministic rollout, one z per slot. Returns qpos (n_envs, T, nq),
    and with return_obs also the RESCALED proprio of the state after each
    step, (n_envs, T, 358), index-aligned with qpos -- what B() scores.

    Since OOP refactor step 5 a thin wrapper of crossenbodiment.rollout.BatchRollout (the loop every rollout script shares),
    kept with its signature for rollout_z_trace.py (and the archived scripts that import it); main() goes through
    crossenbodiment/problems.py HumanLatentProblem (refactor step 6).

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
    from crossenbodiment.obs import ExactObs, multiplier_or_raw
    from crossenbodiment.policy import MotivoPolicy
    from crossenbodiment.rollout import BatchRollout
    from crossenbodiment.sim import HumenvSim
    mapper = ExactObs.wrap(exact, in_env=exact_in_env) if exact is not None else multiplier_or_raw(obs_mul)
    record = ("qpos",) + (("obs",) if return_obs else ()) + (("act",) if act_out is not None else ()) \
        + (("qvel",) if qvel_out is not None else ())
    batch = BatchRollout(MotivoPolicy(model, device), HumenvSim(env), mapper, z_env.shape[0], steps, record=record, nv=nv)
    ro = batch.run(z_env, init_qpos=init_qpos, init_qvel=init_qvel)
    if act_out is not None:
        act_out.extend(np.moveaxis(ro.act, 1, 0))
    if qvel_out is not None:
        qvel_out.extend(np.moveaxis(ro.qvel, 1, 0))
    if return_obs:
        return ro.qpos, ro.obs.astype(np.float32)
    return ro.qpos


def device_arg(s: str) -> str:
    """Accept a bare GPU index ("1") as well as a full torch device string.

    torch.device rejects "1" with 'Invalid device string', and argparse has
    already accepted it by then, so the failure surfaces 90 seconds later
    inside FBcprModel.to() -- after the env and the model download.
    """
    return f"cuda:{s}" if s.isdigit() else s


def build_parser():
    """the CLI (flags, defaults and help as they were)"""
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
    return p


def main():
    """parse -> the problem (body, reference, observation mapper, simulator, policy, cost: crossenbodiment/problems.py) ->
    the optimiser (crossenbodiment/optim.py) -> SearchLoop (crossenbodiment/search.py) -> RunRecorder (records.py)"""
    args = build_parser().parse_args()
    from crossenbodiment.problems import HumanLatentProblem
    from crossenbodiment.records import RunRecorder
    from crossenbodiment.search import SearchLoop

    P = HumanLatentProblem(args)                       # prints the run's settings as before
    P.baseline()                                       # z0's rollout (origin_z) and the kinematic floor
    rec = RunRecorder(P.out_dir)
    if args.eval_z:                                    # no search: score the given latents, write eval.csv, exit
        P.eval_latents(args.eval_z, rec)
        return
    opt, best, z_start = P.start()                     # --z-start, --subspace, --algo; the best-so-far start record
    # every step's iterate and best-so-far are traced (z_trace.npz, gen -1 = the start latent)
    loop = SearchLoop(opt, P.score, P.record, P.n_envs, P.n_gens, best, eval_every=args.eval_every, row=P.curve_row,
                      report=P.report, trace_mean=[z_start.copy()], trace_best=[z_start.copy()])
    loop.run()
    P.close()
    P.save(rec, loop)


if __name__ == "__main__":
    main()
