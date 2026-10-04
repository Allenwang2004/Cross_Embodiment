#!/usr/bin/env python3
"""compare_align_losses.py -- score one search's checkpoints under other
papers' alignment losses, next to our own L_align.

The question this answers: when L_align says checkpoint A is better than B
but the eye says the opposite, is that our L_align being wrong about
similarity, or is "looks most similar" simply not what any tracking loss
measures? Both papers in paper/ drive an optimisation with a loss of their
own, so plugging ours, theirs, and their evaluation metrics into the SAME
rollouts is the direct test. Everything is computed on the qpos that
scripts/rollout_z_trace.py recorded (trace_<which>_qpos.npz), so each row is
exactly one panel of the video it wrote, not a re-rollout.

Losses (lower is better everywhere; cosine rewards are reported as 1 - cos):

  ours          L_align = d_root + d_ee + d_contact + d_pose + d_velocity
                (model/losses.py, weights from ESConfig), plus each term.

  reactor       ReActor (Muller et al. 2026), Eq. 15-17 with the Table 1
                weights: per body pair, squared position error (global frame)
                and squared geodesic orientation error, plus root xy / height /
                orientation / linear and angular velocity terms. Source and
                target are the same skeleton here, so every body is its own
                pair and the retargeting map g_t is the identity.

  bfm_cos       BFMTrack (Rupf et al. 2026), Eq. 3: the tracking reward is the
                cosine between the backward embeddings B(s_t) of the simulated
                state and B(g_t) of the reference state, in Metamotivo's own
                latent space. Reported as 1 - mean_t cos. B needs an
                observation, so each qpos is pushed through the target body's
                HumEnv with finite-difference qvel, and then through the SAME
                obs rescaling the search showed the actor (model/obs_scale.py,
                --obs-scale, default = the run's own setting): B was trained on
                adult-scale observations, so a child's metre-carrying features
                have to be canonicalised before B can read them. --obs-scale
                none gives the raw-obs variant.
  bfm_cos8      the same with B averaged over the next 8 frames, which is what
                Metamotivo's tracking_inference does before projecting.

  Evaluation metrics from BFMTrack Sec. 7.1, all in the local path frame
  (origin at the root, x = root heading, z up):
  mpjpe         mean per-joint position error [m]
  mmpjpe        mean over joints of the max-over-time position error [m]
  mpjae         mean per-joint linear-acceleration error [m/s^2]
  dtw           L1 joint-angle distance under optimal time warping, per
                warping-path step [rad]
  emd           earth mover's distance between the two sets of poses (joint
                positions + 6D orientations in the path frame), via an
                assignment on equal-size sets

Usage (after rollout_z_trace.py has written trace_mean_qpos.npz):
    uv run scripts/compare_align_losses.py --run outputs/single_z/move-ego-0-2_4_align
    uv run scripts/compare_align_losses.py --run ... --picks 48,144   # evals the eye prefers

Writes <run>/align_compare.csv and <run>/align_compare.png.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from model import kinematics as kin
from model import losses

REPO_ROOT = Path(__file__).resolve().parent.parent
DT = losses.DEFAULT_DT

# ReActor Table 1 (motion-tracking rows). Rigid-body terms are per pair,
# summed over pairs, as in Eq. 17.
REACTOR_W = dict(root_xy=2.0, root_z=10.0, root_R=2.0, root_v=0.5, root_w=0.5,
                 body_x=2.0, body_R=2.0)


# --------------------------------------------------------------- rotations
def quat_to_mat(q):
    """(..., 4) wxyz -> (..., 3, 3)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def log_so3(R):
    """(..., 3, 3) -> (..., 3) rotation vector (Log map)."""
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.where(th > 1e-6, th / (2 * np.sin(np.clip(th, 1e-6, np.pi))), 0.5)
    return v * s[..., None]


def geodesic_sq(Ra, Rb):
    """|| Log(Ra^T Rb) ||^2, per element of the batch."""
    rel = np.einsum("...ji,...jk->...ik", Ra, Rb)
    return (log_so3(rel) ** 2).sum(-1)


def ang_vel(R, dt):
    """body-frame angular velocity from a rotation sequence, (T-1, 3)."""
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    return log_so3(rel) / dt


# --------------------------------------------------------------- frames
def fk(model, qpos, bodies):
    pos, quat = kin.batch_forward_pose(model, qpos, bodies)
    P = np.stack([pos[b] for b in bodies], 1)            # (T, B, 3)
    R = quat_to_mat(np.stack([quat[b] for b in bodies], 1))  # (T, B, 3, 3)
    return P, R


def to_path_frame(P, R, root_idx):
    """BFMTrack's local path frame: origin at the root, x-axis = root heading
    (yaw), z up. Returns positions and rotations expressed in it."""
    root_p = P[:, root_idx]
    yaw = kin.quat_to_yaw(mat_to_quat(R[:, root_idx]))
    c, s = np.cos(yaw), np.sin(yaw)
    Rw = np.zeros((len(yaw), 3, 3))              # world->path
    Rw[:, 0, 0], Rw[:, 0, 1] = c, s
    Rw[:, 1, 0], Rw[:, 1, 1] = -s, c
    Rw[:, 2, 2] = 1
    Pl = np.einsum("tij,tbj->tbi", Rw, P - root_p[:, None])
    Rl = np.einsum("tij,tbjk->tbik", Rw, R)
    return Pl, Rl


def mat_to_quat(R):
    """(..., 3, 3) -> (..., 4) wxyz (only used for the yaw, so any sign is fine)."""
    out = np.zeros(R.shape[:-2] + (4,))
    flat = R.reshape(-1, 3, 3)
    q = np.zeros((len(flat), 4))
    for i, m in enumerate(flat):
        mujoco.mju_mat2Quat(q[i], m.reshape(9))
    return q.reshape(out.shape)


# --------------------------------------------------------------- losses
def reactor_loss(P_s, R_s, P_g, R_g, root_idx, dt=DT):
    """Eq. 15-17 + Table 1; mean over time of the per-frame weighted sum."""
    T = min(len(P_s), len(P_g))
    P_s, R_s, P_g, R_g = P_s[:T], R_s[:T], P_g[:T], R_g[:T]
    others = [i for i in range(P_s.shape[1]) if i != root_idx]
    r = root_idx
    v_s, v_g = np.diff(P_s[:, r], axis=0) / dt, np.diff(P_g[:, r], axis=0) / dt
    w_s, w_g = ang_vel(R_s[:, r], dt), ang_vel(R_g[:, r], dt)
    terms = dict(
        root_xy=float(((P_s[:, r, :2] - P_g[:, r, :2]) ** 2).sum(-1).mean()),
        root_z=float(((P_s[:, r, 2] - P_g[:, r, 2]) ** 2).mean()),
        root_R=float(geodesic_sq(R_s[:, r], R_g[:, r]).mean()),
        root_v=float(((v_s - v_g) ** 2).sum(-1).mean()),
        root_w=float(((w_s - w_g) ** 2).sum(-1).mean()),
        body_x=float(((P_s[:, others] - P_g[:, others]) ** 2).sum(-1).sum(-1).mean()),
        body_R=float(geodesic_sq(R_s[:, others], R_g[:, others]).sum(-1).mean()),
    )
    total = sum(REACTOR_W[k] * v for k, v in terms.items())
    return total, terms


def bfm_metrics(P_s, R_s, P_g, R_g, root_idx, dt=DT):
    """BFMTrack Sec. 7.1 in the local path frame."""
    T = min(len(P_s), len(P_g))
    Ps, Rs = to_path_frame(P_s[:T], R_s[:T], root_idx)
    Pg, Rg = to_path_frame(P_g[:T], R_g[:T], root_idx)
    err = np.linalg.norm(Ps - Pg, axis=-1)                 # (T, B)
    acc = lambda P: np.diff(P, n=2, axis=0) / dt ** 2
    out = dict(mpjpe=float(err.mean()),
               mmpjpe=float(err.max(0).mean()),
               mpjae=float(np.linalg.norm(acc(Ps) - acc(Pg), axis=-1).mean()))
    # EMD between the two pose SETS (alignment-tolerant): equal weights, so it
    # is an assignment problem on the pose-feature distance matrix.
    feat = lambda P, R: np.concatenate([P.reshape(T, -1), R[..., :, :2].reshape(T, -1)], 1)
    fs, fg = feat(Ps, Rs), feat(Pg, Rg)
    D = np.linalg.norm(fs[:, None] - fg[None], axis=-1)
    i, j = linear_sum_assignment(D)
    out["emd"] = float(D[i, j].mean())
    return out


def dtw_l1(qa, qb):
    """L1 on joint angles under optimal time warping, normalised by the
    warping path length (BFMTrack Sec. 7.1)."""
    A, B = qa[:, 7:], qb[:, 7:]
    n, m = len(A), len(B)
    D = np.abs(A[:, None] - B[None]).sum(-1)
    C = np.full((n + 1, m + 1), np.inf); C[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    # path length by backtracking
    i, j, L = n, m, 0
    while i > 0 and j > 0:
        L += 1
        k = int(np.argmin([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return float(C[n, m] / max(L, 1))


def backward_embeddings(model_bfm, env, qpos, device, obs_mul=None):
    """B(s_t) for every frame, with qvel from finite differences so the
    velocity part of the observation is the trajectory's own, not zero, and
    the obs rescaled exactly as single_z_search.rollout rescales it for the actor."""
    mj = env.unwrapped.model
    nv = mj.nv
    obs = []
    for t in range(len(qpos)):
        qvel = np.zeros(nv)
        if t > 0:
            mujoco.mj_differentiatePos(mj, qvel, DT, qpos[t - 1], qpos[t])
        env.unwrapped.set_physics(qpos=qpos[t], qvel=qvel)
        o = env.unwrapped.get_obs()["proprio"].copy()
        obs.append(o if obs_mul is None else o * obs_mul)
    obs = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device)
    with torch.no_grad():
        return model_bfm.backward_map(obs).cpu().numpy()


def cos_rows(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="a single_z run directory")
    p.add_argument("--which", default="mean")
    p.add_argument("--picks", default="", help="comma-separated evals the eye prefers, "
                                              "marked on the figure")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--no-bfm", action="store_true", help="skip the Metamotivo B() scores")
    p.add_argument("--obs-scale", default=None,
                   help="obs rescaling before B(): 'auto' / 'none' / a ratio; "
                        "default: what the run's summary.json says the search used")
    p.add_argument("--obs-scale-ref", default="assets/robots/adult/robot.xml")
    p.add_argument("--tag", default="", help="suffix for the output files (e.g. _rawobs)")
    args = p.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = REPO_ROOT / run
    d = np.load(run / f"trace_{args.which}_qpos.npz")
    gens, qpos, ref, xml = d["gen"], d["qpos"].astype(np.float64), d["ref"].astype(np.float64), str(d["xml"])
    cos_z0 = d["cos_z0"]
    import json
    summary = json.loads((run / "summary.json").read_text())
    evals = [int((g + 1) * 2 * summary["pairs"]) for g in gens]

    model = mujoco.MjModel.from_xml_path(xml)
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    root_idx = bodies.index(kin.ROOT_BODY)
    Pg, Rg = fk(model, ref, bodies)

    from model.simple.config import ESConfig
    cfg = ESConfig(device="cpu")
    d_w = {"root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
           "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight}

    Bg = None
    if not args.no_bfm:
        from humenv import make_humenv
        from metamotivo.fb_cpr.huggingface import FBcprModel
        bfm = FBcprModel.from_pretrained(args.metamotivo).to(args.device); bfm.eval()
        env, _ = make_humenv(num_envs=1, task=None, xml=xml, state_init="Default")
        from model.obs_scale import build_obs_multiplier
        obs_scale = args.obs_scale or summary.get("obs_scale", "auto")
        obs_mul = build_obs_multiplier(xml, REPO_ROOT / args.obs_scale_ref, mode=obs_scale,
                                       parts=cfg.obs_scale_parts, verbose=True)
        print(f"B() sees obs rescaled with obs_scale={obs_scale}")
        Bg = backward_embeddings(bfm, env, ref, args.device, obs_mul)

    def avg8(z):
        out = z.copy()
        for t in range(len(z)):
            out[t] = z[t:t + 8].mean(0)
        return out

    rows = []
    for i, g in enumerate(gens):
        q = qpos[i]
        ours, terms = losses.functional_equivalence(model, q, ref, d_w, DT)
        Ps, Rs = fk(model, q, bodies)
        rea, rterms = reactor_loss(Ps, Rs, Pg, Rg, root_idx)
        met = bfm_metrics(Ps, Rs, Pg, Rg, root_idx)
        row = dict(gen=int(g), evals=evals[i], cos_z0=float(cos_z0[i]), ours=float(ours),
                   **{f"ours_{k}": float(d_w[k] * v) for k, v in terms.items()},
                   reactor=float(rea), **{f"reactor_{k}": float(REACTOR_W[k] * v) for k, v in rterms.items()},
                   **met, dtw=dtw_l1(q, ref))
        if Bg is not None:
            Bs = backward_embeddings(bfm, env, q, args.device, obs_mul)
            T = min(len(Bs), len(Bg))
            row["bfm_cos"] = float(1 - cos_rows(Bs[:T], Bg[:T]).mean())
            row["bfm_cos8"] = float(1 - cos_rows(avg8(Bs)[:T], avg8(Bg)[:T]).mean())
        rows.append(row)
        print(f"  gen {int(g):4d} done", flush=True)
    if Bg is not None:
        env.close()

    with open(run / f"align_compare{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # --- table: every loss, and which checkpoint each one would pick ----------
    main_keys = ["ours", "reactor"] + (["bfm_cos", "bfm_cos8"] if Bg is not None else []) \
                + ["mpjpe", "mmpjpe", "mpjae", "dtw", "emd"]
    picks = [int(x) for x in args.picks.split(",") if x]
    print(f"\n{summary['clip']} on {Path(xml).name}  ({summary['objective']} run)")
    print(f"{'evals':>7s}{'cos_z0':>8s}" + "".join(f"{k:>10s}" for k in main_keys))
    # gen -1 (0 evals) is z0, the point the search started from. It IS a
    # candidate: the search's own best starts from it too (single_z_search
    # reports source=origin_z when nothing beat it), so a loss that prefers z0
    # over everything the search produced should say so with the star.
    best = {k: int(np.argmin([r[k] for r in rows])) for k in main_keys}
    for i, r in enumerate(rows):
        mark = " <- eye" if r["evals"] in picks else ("   (origin z)" if r["gen"] < 0 else "")
        print(f"{r['evals']:7d}{r['cos_z0']:8.3f}"
              + "".join(f"{r[k]:9.4f}{'*' if best[k] == i else ' '}" for k in main_keys) + mark)
    print("  * = that loss's own best checkpoint (0 evals = the origin z, a candidate like any other)")
    print("\nours, by term (weighted):")
    tk = [k for k in rows[0] if k.startswith("ours_")]
    print(f"{'evals':>7s}" + "".join(f"{k[5:]:>10s}" for k in tk))
    for r in rows:
        print(f"{r['evals']:7d}" + "".join(f"{r[k]:10.4f}" for k in tk))
    print("\nreactor, by term (weighted):")
    tk = [k for k in rows[0] if k.startswith("reactor_")]
    print(f"{'evals':>7s}" + "".join(f"{k[8:]:>10s}" for k in tk))
    for r in rows:
        print(f"{r['evals']:7d}" + "".join(f"{r[k]:10.4f}" for k in tk))

    # --- figure: each loss on its own scale, so the SHAPE is comparable ---------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.array([r["evals"] for r in rows], dtype=float) + 1
    groups = [("optimised losses", ["ours", "reactor"] + (["bfm_cos", "bfm_cos8"] if Bg is not None else [])),
              ("BFMTrack eval metrics", ["mpjpe", "mmpjpe", "mpjae"]),
              ("alignment-tolerant", ["dtw", "emd"])]
    for ax, (title, keys) in zip(axes, groups):
        for k in keys:
            y = np.array([r[k] for r in rows]); y = y / y.max()
            ax.plot(x, y, marker="o", label=k)
            ax.plot(x[best[k]], y[best[k]], marker="*", ms=14, color=ax.lines[-1].get_color())
        for pv in picks:
            ax.axvline(pv + 1, color="k", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("evals + 1   (1 = origin z)")
        ax.set_ylabel("loss / max"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f"{summary['clip']} ({summary['objective']} run): * = each loss's best; "
                 f"dotted = the eye's pick{'s' if len(picks) > 1 else ''} {picks}")
    fig.tight_layout()
    fig.savefig(run / f"align_compare{args.tag}.png", dpi=130)
    print(f"\n-> {run / f'align_compare{args.tag}.csv'}\n-> {run / f'align_compare{args.tag}.png'}")


if __name__ == "__main__":
    main()
