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
Antithetic ES on z directly -- the same estimator as model/simple/train_es.py
(rank-normalised, common random numbers, deterministic actions), minus the
adapter and its autograd chain, because here there is no theta to route the
gradient into:

    eps_k  ~ N(0, I_256)
    F+-_k  = cost( rollout( project(z +- sigma*eps_k) ) )
    g      = sum_k shape(F+_k - F-_k) * eps_k / (2 * pairs * sigma)
    z     <- project( adam(z, g) )

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
            return_obs=False, exact=None):
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
    """
    obs, _ = env.reset()
    if init_qpos is not None:
        env.call("set_physics", qpos=init_qpos, qvel=np.zeros(nv))
        obs = {"proprio": np.stack([o["proprio"] for o in env.call("get_obs")])}
    hist, obs_hist = [], []
    if exact is not None:
        if init_qpos is None:
            raise SystemExit("--obs-scale exact needs --init reference")
        ex_obs = np.repeat(exact(init_qpos, np.zeros(nv))[None], z_env.shape[0], 0)
    for _ in range(steps):
        if exact is not None:
            proprio = ex_obs
        else:
            proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.as_tensor(proprio, dtype=torch.float32, device=device)
        mu = model._actor(model._normalize(obs_t), z_env, model.cfg.actor_std).mean
        obs, _, _, _, info = env.step(mu.cpu().numpy())
        hist.append(info["qpos"].copy())
        if exact is not None:
            ex_obs = np.stack([exact(q, v) for q, v in zip(info["qpos"], info["qvel"])])
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
    p.add_argument("--pairs", type=int, default=8,
                   help="antithetic pairs per step (one step = one Adam update "
                        "on z); 2*pairs env slots")
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
    p.add_argument("--obs-scale", default="auto", choices=["auto", "none", "exact"],
                   help="auto: raw obs x fixed per-feature multiplier; none: raw; exact: the adult-equivalent "
                        "observation by reverse retargeting (model/exact_obs.py), for the actor and for B")
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
    p.add_argument("--algo", default="es", choices=["es", "cmaes"],
                   help="es: antithetic ES + Adam on z (the default everywhere else). "
                        "cmaes: pycma CMA-ES with popsize 2*pairs and sigma0 = --sigma, "
                        "candidates projected onto the sphere; --lr is unused")
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
                   help="add w * mean_t |root_xy_s - root_xy_g| (metres)")
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
    z0_path = REPO_ROOT / "data" / "origin_z" / task / f"{stem}.npy"
    ref_path = (Path(args.ref) if args.ref else
                REPO_ROOT / "data" / args.body / "retargeting_motion" / task / f"{stem}.npz")
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
    env, _ = make_humenv(num_envs=n_envs, vectorization_mode="async", task=None,
                        xml=str(xml), state_init="Default")
    fk = mujoco.MjModel.from_xml_path(str(xml))
    exact = None
    if args.obs_scale == "exact":
        from model.exact_obs import ExactObs
        kw = {}
        if args.body != "child":
            # every body's retarget copies the joints and scales root x, y by its own s (constant to 1e-16),
            # so read s off this clip against the adult's; the child keeps ExactObs's measured default
            a = np.load(REPO_ROOT / "data" / "origin_motion" / task / f"{stem}.npz")["qpos"][:, :2]
            n = min(len(a), len(ref)); m = np.abs(a[:n]) > 1e-3
            kw["scale"] = float(np.median(ref[:n, :2][m] / a[:n][m]))
            print(f"exact obs: root scale {kw['scale']:.6f} (from the reference vs the adult motion)")
        exact, obs_mul = ExactObs(xml, REPO_ROOT / args.obs_scale_ref, **kw), None
    else:
        obs_mul = build_obs_multiplier(xml, REPO_ROOT / args.obs_scale_ref,
                                       mode=args.obs_scale, parts=cfg.obs_scale_parts,
                                       verbose=False)

    init_qpos = ref[0] if args.init == "reference" else None

    Bg = None
    if args.objective == "bfm":
        from model import bfm_align
        env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
        if exact is None:
            Bg = bfm_align.reference_embeddings(model, env1, ref, args.device, obs_mul)
        else:
            # the reference through the SAME reverse retargeting as the rollout states
            vr, ro = np.zeros(fk.nv), []
            for t in range(len(ref)):
                if t:
                    mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, ref[t - 1], ref[t])
                ro.append(exact(ref[t], vr))
            Bg = bfm_align.embed(model, np.stack(ro), args.device)

    glob_w = (args.heading_weight, args.pos_weight)
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

    ref_h, ref_xy = root_heading_xy(ref)

    def global_terms(q):
        T = min(q.shape[1], len(ref))
        h, xy = root_heading_xy(q[:, :T])
        head = ((1 - np.cos(h - ref_h[None, :T])) / 2).mean(1)
        pos = np.linalg.norm(xy - ref_xy[None, :T], axis=-1).mean(1)
        return head, pos

    def score(zs):
        """zs: (n, 256) -> (cost, L_align, L_phys, bfm, qpos). One batched rollout.
        cost is the objective; L_align / L_phys are always the joint-space
        terms, so a bfm run's summary stays comparable with the others."""
        zt = torch.as_tensor(zs, dtype=torch.float32, device=args.device)
        q, o = rollout(model, env, zt, args.steps, args.device, obs_mul,
                       init_qpos=init_qpos, nv=fk.nv, return_obs=True, exact=exact)
        cost, align, phys = compute_batch_cost(fk, cfg, q, [ref] * len(zs))
        bfm = np.full(len(zs), np.nan, dtype=np.float32)
        if Bg is not None:
            bfm = bfm_align.batch_bfm_align(model, o, Bg, args.device)
            cost = bfm
        extra["mse"] = np.array([losses.tracking_mse(fk, qi, ref)[0] for qi in q])
        if args.objective == "mse":
            cost = extra["mse"].astype(np.float32)
        extra["head"], extra["pos"] = global_terms(q)
        if any(glob_w):
            cost = cost + glob_w[0] * extra["head"] + glob_w[1] * extra["pos"]
        extra["anchor_dist"] = (np.linalg.norm(np.asarray(zs) - anchor, axis=1) if anchor is not None
                                else np.full(len(zs), np.nan))
        if anchor is not None:
            cost = cost + args.anchor_weight * (extra["anchor_dist"] / 16.0) ** 2
        return cost, align, phys, bfm, q

    # --- where we start, and the floor we are aiming at ----------------------
    c0, a0, p0, f0, q0 = score(np.repeat(z0[None], n_envs, axis=0))
    base = dict(cost=float(c0[0]), align=float(a0[0]), phys=float(p0[0]), bfm=float(f0[0]),
                head=float(extra["head"][0]), pos=float(extra["pos"][0]), mse=float(extra["mse"][0]))
    _, ra, rp = compute_batch_cost(fk, cfg, ref[None, :args.steps], [ref])
    floor = dict(cost=float(lam_a * ra[0] + lam_p * rp[0]),
                 align=float(ra[0]), phys=float(rp[0]), bfm=0.0 if Bg is not None else float("nan"))
    if Bg is not None:
        # the kinematic reference against itself, through obs_from_qpos on both
        # sides -- exactly 0 by construction, so the floor is 0
        floor["cost"] = 0.0
    print(f"  origin_z : cost {base['cost']:.4f}  L_align {base['align']:.4f}  "
          f"L_phys {base['phys']:.4f}")
    print(f"  reference: cost {floor['cost']:.4f}  L_align {floor['align']:.4f}  "
          f"L_phys {floor['phys']:.4f}   <- kinematic floor")

    # --- ES ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    z_start = z0.copy()
    if args.z_start:
        z_start = project_z(np.load(args.z_start).reshape(-1).astype(np.float64))
        cs, ca, cp, cf, cq = score(np.repeat(z_start[None], n_envs, axis=0))
        start_best = dict(cost=float(cs[0]), align=float(ca[0]), phys=float(cp[0]), bfm=float(cf[0]),
                          head=float(extra["head"][0]), pos=float(extra["pos"][0]), mse=float(extra["mse"][0]),
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
        if cma_es is not None:
            cand = project_z(np.asarray(cma_es.ask(), dtype=np.float64))
        else:
            if U is None:
                eps = rng.standard_normal((args.pairs, z.size))
            else:
                xi = sub_scale * rng.standard_normal((args.pairs, len(U)))
                eps = xi @ U
            cand = project_z(np.concatenate([z + args.sigma * eps,
                                             z - args.sigma * eps], axis=0))
        cost, align, phys, bfm, qpos = score(cand)

        i = int(np.argmin(cost))
        if cost[i] < best["cost"]:
            best = dict(cost=float(cost[i]), align=float(align[i]),
                        phys=float(phys[i]), bfm=float(bfm[i]), z=cand[i].copy(),
                        qpos=qpos[i].copy(), gen=gen, source="sample",
                        head=float(extra["head"][i]), pos=float(extra["pos"][i]), mse=float(extra["mse"][i]))

        z_prev = z
        if cma_es is not None:
            # tell the projected candidates (what was actually scored); keep the
            # mean on the sphere too so sigma stays a fractional perturbation
            cma_es.tell(list(cand), list(map(float, cost)))
            cma_es.mean = project_z(np.asarray(cma_es.mean, dtype=np.float64))
            z = cma_es.mean.copy()
        else:
            s = rank_normalize(cost)
            g = ((s[:args.pairs] - s[args.pairs:])[:, None] * (eps if U is None else xi)).sum(0) \
                / (2 * args.pairs * args.sigma)
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
                   sigma=float(cma_es.sigma) if cma_es is not None else args.sigma)
        # The iterate itself, not just its samples -- one extra rollout, and it
        # is the only way to see whether the mean is tracking the population.
        if args.eval_every and (gen % args.eval_every == 0 or gen == n_gens - 1):
            mc, ma, mp, mf, mq = score(np.repeat(z[None], n_envs, axis=0))
            row["mean_z_cost"] = float(mc[0])
            if mc[0] < best["cost"]:
                best = dict(cost=float(mc[0]), align=float(ma[0]), phys=float(mp[0]),
                            bfm=float(mf[0]), z=z.copy(), qpos=mq[0].copy(), gen=gen,
                            source="mean_z", head=float(extra["head"][0]), pos=float(extra["pos"][0]), mse=float(extra["mse"][0]))
            el = time.time() - t0
            print(f"  gen {gen:5d}/{n_gens}  best {best['cost']:9.4f}  "
                  f"gen_best {cost.min():9.4f}  mean_z {mc[0]:9.4f}  "
                  f"cos(z,z0) {row['cos_z0']:.3f}  path {path_len:5.0f} deg / net {row['net_deg']:4.0f} deg"
                  + (f"  sigma {cma_es.sigma:.4f}" if cma_es is not None else "") + f"  [{el / 60:.1f} min]", flush=True)
        curve.append(row)
        trace_mean.append(z.copy())
        trace_best.append(best["z"].copy())

    env.close()
    if Bg is not None:
        env1.close()

    # --- what came out -------------------------------------------------------
    np.save(out_dir / "best_z.npy", best["z"].astype(np.float32).reshape(1, -1))
    np.savez_compressed(out_dir / "best.npz", qpos=best["qpos"].astype(np.float32),
                        fps=cfg.control_fps)
    # The origin_z rollout under the SAME init and env as the search. Saved so a
    # plot does not have to re-derive the baseline from a separate render, which
    # would not reproduce it: a rollout that falls is chaotic enough that a
    # different device or a different env wrapper lands somewhere else entirely
    # (measured: L_align 1.569 in the async search vs 1.290 re-rendered).
    np.savez_compressed(out_dir / "origin_z.npz", qpos=q0[0].astype(np.float32),
                        fps=cfg.control_fps)
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
        "objective": args.objective, "lambda_align": lam_a, "lambda_phys": lam_p,
        "phys_weights": cfg.phys_weights, "phys_fall_ref": cfg.phys_fall_ref,
        # "generations" is the on-disk name for what the logs and figures now
        # call steps -- kept so older summary.json files stay readable
        "evals": n_gens * n_envs, "generations": n_gens, "pairs": args.pairs,
        "sigma": args.sigma, "lr": args.lr, "steps": args.steps, "algo": args.algo,
        "path_len_deg": round(path_len, 1), "net_deg": round(ang(z0, z), 1),
        "obs_scale": args.obs_scale, "seed": args.seed, "init": args.init,
        "align_discount": args.discount, "heading_weight": args.heading_weight, "pos_weight": args.pos_weight,
        "anchor": args.anchor, "anchor_weight": args.anchor_weight,
        "subspace": args.subspace, "subspace_dim": None if U is None else len(U),
        "origin_z": base, "reference_floor": floor,
        "best": {k: best[k] for k in ("cost", "align", "phys", "bfm", "gen", "source", "head", "pos", "mse") if k in best},
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
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
