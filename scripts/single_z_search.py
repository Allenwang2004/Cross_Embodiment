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
        --xml assets/robot_torque/child/robot_torque_full.xml \
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
}


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
def rollout(model, env, z_env, steps, device, obs_mul, init_qpos=None, nv=None):
    """Deterministic rollout, one z per slot. Returns qpos (n_envs, T, nq).
    Same body as scripts/loss_test.py:rollout, kept separate rather than
    imported so this script does not drag in loss_test's argument surface.

    init_qpos overrides humenv's Default reset with a pose -- here the
    reference's own first frame, so the rollout and the trajectory it is scored
    against start from the SAME place. Without it frame 0 is humenv's standing
    T-pose and the first slice of d_pose/d_ee is an offset no z can remove: the
    policy is being charged for an initial condition it did not choose. qvel is
    zeroed rather than finite-differenced from the reference, matching
    rollout_z_on_body.py:--init-from-reference so the two agree.
    """
    obs, _ = env.reset()
    if init_qpos is not None:
        env.call("set_physics", qpos=init_qpos, qvel=np.zeros(nv))
        obs = {"proprio": np.stack([o["proprio"] for o in env.call("get_obs")])}
    hist = []
    for _ in range(steps):
        proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.as_tensor(proprio, dtype=torch.float32, device=device)
        mu = model._actor(model._normalize(obs_t), z_env, model.cfg.actor_std).mean
        obs, _, _, _, info = env.step(mu.cpu().numpy())
        hist.append(info["qpos"].copy())
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
                   help="default assets/robot_torque/<body>/robot_torque_full.xml "
                        "-- condition C of scripts/loss_test.py, the best of the "
                        "four and the one training uses")
    p.add_argument("--obs-scale", default="auto", choices=["auto", "none"])
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
    p.add_argument("--seed", type=int, default=0)
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

    task, stem = args.clip.split("/")
    z0_path = REPO_ROOT / "data" / "origin_z" / task / f"{stem}.npy"
    ref_path = REPO_ROOT / "data" / args.body / "retargeting_motion" / task / f"{stem}.npz"
    for q in (z0_path, ref_path):
        if not q.exists():
            raise SystemExit(f"{q} not found")
    xml = Path(args.xml) if args.xml else (
        REPO_ROOT / "assets" / "robot_torque" / args.body / "robot_torque_full.xml")
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
    print(f"objective {args.objective}: cost = {lam_a} * L_align + {lam_p} * L_phys "
          f"(L_phys weights={cfg.phys_weights}, L_align discount={cfg.align_discount})")
    print(f"{n_gens} steps x {n_envs} rollouts = {n_gens * n_envs} evals, "
          f"sigma {args.sigma}, lr {args.lr}, reference {len(ref)} frames")
    print(f"init: {args.init}" + (" (rollout starts from the reference's frame 0)"
                                  if args.init == "reference" else
                                  " (humenv standing reset)"))

    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device)
    model.eval()
    env, _ = make_humenv(num_envs=n_envs, vectorization_mode="async", task=None,
                        xml=str(xml), state_init="Default")
    fk = mujoco.MjModel.from_xml_path(str(xml))
    obs_mul = build_obs_multiplier(xml, REPO_ROOT / args.obs_scale_ref,
                                   mode=args.obs_scale, parts=cfg.obs_scale_parts,
                                   verbose=False)

    init_qpos = ref[0] if args.init == "reference" else None

    def score(zs):
        """zs: (n, 256) -> (cost, L_align, L_phys) each (n,). One batched rollout."""
        zt = torch.as_tensor(zs, dtype=torch.float32, device=args.device)
        q = rollout(model, env, zt, args.steps, args.device, obs_mul,
                    init_qpos=init_qpos, nv=fk.nv)
        return compute_batch_cost(fk, cfg, q, [ref] * len(zs)) + (q,)

    # --- where we start, and the floor we are aiming at ----------------------
    c0, a0, p0, q0 = score(np.repeat(z0[None], n_envs, axis=0))
    base = dict(cost=float(c0[0]), align=float(a0[0]), phys=float(p0[0]))
    _, ra, rp = compute_batch_cost(fk, cfg, ref[None, :args.steps], [ref])
    floor = dict(cost=float(lam_a * ra[0] + lam_p * rp[0]),
                 align=float(ra[0]), phys=float(rp[0]))
    print(f"  origin_z : cost {base['cost']:.4f}  L_align {base['align']:.4f}  "
          f"L_phys {base['phys']:.4f}")
    print(f"  reference: cost {floor['cost']:.4f}  L_align {floor['align']:.4f}  "
          f"L_phys {floor['phys']:.4f}   <- kinematic floor")

    # --- ES ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    z = z0.copy()
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8

    best = dict(cost=base["cost"], align=base["align"], phys=base["phys"],
                z=z0.copy(), qpos=q0[0].copy(), gen=-1, source="origin_z")
    curve = []
    # Every step's iterate and its best-so-far, so any point in the search
    # can be rolled out afterwards without re-running it (256 floats a row --
    # 625 steps is under a megabyte, so this is not worth a flag).
    trace_mean, trace_best = [z0.copy()], [z0.copy()]
    t0 = time.time()
    for gen in range(n_gens):
        eps = rng.standard_normal((args.pairs, z.size))
        cand = project_z(np.concatenate([z + args.sigma * eps,
                                         z - args.sigma * eps], axis=0))
        cost, align, phys, qpos = score(cand)

        i = int(np.argmin(cost))
        if cost[i] < best["cost"]:
            best = dict(cost=float(cost[i]), align=float(align[i]),
                        phys=float(phys[i]), z=cand[i].copy(),
                        qpos=qpos[i].copy(), gen=gen, source="sample")

        s = rank_normalize(cost)
        g = ((s[:args.pairs] - s[args.pairs:])[:, None] * eps).sum(0) \
            / (2 * args.pairs * args.sigma)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (gen + 1))
        vh = v / (1 - b2 ** (gen + 1))
        z = project_z(z - args.lr * mh / (np.sqrt(vh) + eps_adam))

        row = dict(gen=gen, evals=(gen + 1) * n_envs,
                   gen_best=float(cost.min()), gen_mean=float(cost.mean()),
                   best_so_far=best["cost"], mean_z_cost="",
                   cos_z0=float(np.dot(z, z0) / (np.linalg.norm(z) * np.linalg.norm(z0))))
        # The iterate itself, not just its samples -- one extra rollout, and it
        # is the only way to see whether the mean is tracking the population.
        if args.eval_every and (gen % args.eval_every == 0 or gen == n_gens - 1):
            mc, ma, mp, mq = score(np.repeat(z[None], n_envs, axis=0))
            row["mean_z_cost"] = float(mc[0])
            if mc[0] < best["cost"]:
                best = dict(cost=float(mc[0]), align=float(ma[0]), phys=float(mp[0]),
                            z=z.copy(), qpos=mq[0].copy(), gen=gen, source="mean_z")
            el = time.time() - t0
            print(f"  gen {gen:5d}/{n_gens}  best {best['cost']:9.4f}  "
                  f"gen_best {cost.min():9.4f}  mean_z {mc[0]:9.4f}  "
                  f"cos(z,z0) {row['cos_z0']:.3f}  [{el / 60:.1f} min]")
        curve.append(row)
        trace_mean.append(z.copy())
        trace_best.append(best["z"].copy())

    env.close()

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
        "clip": args.clip, "body": args.body, "xml": str(xml),
        "objective": args.objective, "lambda_align": lam_a, "lambda_phys": lam_p,
        "phys_weights": cfg.phys_weights, "phys_fall_ref": cfg.phys_fall_ref,
        # "generations" is the on-disk name for what the logs and figures now
        # call steps -- kept so older summary.json files stay readable
        "evals": n_gens * n_envs, "generations": n_gens, "pairs": args.pairs,
        "sigma": args.sigma, "lr": args.lr, "steps": args.steps,
        "obs_scale": args.obs_scale, "seed": args.seed, "init": args.init,
        "align_discount": args.discount,
        "origin_z": base, "reference_floor": floor,
        "best": {k: best[k] for k in ("cost", "align", "phys", "gen", "source")},
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
