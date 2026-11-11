#!/usr/bin/env python3
"""controlled_pairs.py -- step 1 of the latent-vs-behaviour study: pick source
latents at controlled distances from a clip's z0 and check, on the ORIGINAL body
(the adult, m2c_t000), how alike their behaviours are.

Sources for one clip (all on the radius-16 sphere):
  z0            the clip's own reward-inferred latent ("identical" pair: z0 vs z0,
                rolled out twice in different batch slots, must give distance 0)
  rot<a>_<j>    z0 rotated by a in {2, 5, 10, 20, 30} deg along 3 fixed random
                tangent directions j (very close .. moderately close)
  trial_<k>     the z0 of another trial of the same task (naturally "moderately
                close": same reward, different inference sample)

Every source is rolled out on the adult from the SAME start (the clip's adult
reference frame 0), 300 steps, mean actions, in batches of 16 like the searches.
Behaviour distance to the z0 rollout:
  joint     mean |dq| over the 69 joint coordinates, deg
  ee        mean distance of hands, toes and head relative to the pelvis, m
  heading   mean |d yaw| of the pelvis, deg
  root_xy   mean distance between the pelvis xy positions, m
  bspace    1 - mean_t cos(B(s_a), B(s_b))   (the model's own behaviour space)
  l_align   L_align with the z0 rollout as the reference
and the bfm cost of each source against the clip's adult reference.

Writes outputs/controlled_pairs/<stem>/adult.npz (z, qpos, B) and adult_metrics.csv.

usage: CUDA_VISIBLE_DEVICES=1 uv run scripts/controlled_pairs.py --clip move-ego-0-2/move-ego-0-2_4
"""
import argparse, csv, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.append(str(REPO))
import numpy as np
import torch

ANGLES = (2, 5, 10, 20, 30)
TRIALS = {"move-ego-0-2": list(range(10)), "headstand": [0, 2, 3, 4, 9, 15, 20, 23, 30, 34]}


def unit(v):
    return v / np.linalg.norm(v)


def sources(task, stem, angles=ANGLES, n_dirs=3, trials=True):
    z0 = unit(np.load(REPO / "data/origin_z" / task / f"{stem}.npy").reshape(-1).astype(np.float64))
    rng = np.random.default_rng(0)
    S = [("z0", "identical", z0), ("z0_repeat", "identical", z0)]
    dirs = []
    for _ in range(n_dirs):
        r = rng.standard_normal(z0.size); r -= (r @ z0) * z0; dirs.append(unit(r))
    for a in angles:
        for j, r in enumerate(dirs):
            t = np.radians(a)
            S.append((f"rot{a:g}_{j}", "very close" if a <= 5 else ("close" if a <= 10 else "moderate"),
                      np.cos(t) * z0 + np.sin(t) * r))
    k_self = int(stem.rsplit("_", 1)[1])
    for k in (TRIALS.get(task, []) if trials else []):
        if k != k_self:
            z = unit(np.load(REPO / "data/origin_z" / task / f"{task}_{k}.npy").reshape(-1).astype(np.float64))
            S.append((f"trial_{k}", "other trial", z))
    return [(n, kind, 16.0 * z) for n, kind, z in S]


def behaviour_distances(fk, qa, qb, Ba, Bb, losses, kin, W, dt):
    T = min(len(qa), len(qb))
    qa, qb = qa[:T], qb[:T]
    joint = float(np.degrees(np.abs(qa[:, 7:] - qb[:, 7:])).mean())
    bodies = kin.EE_BODIES + [kin.ROOT_BODY]
    pa, _ = kin.batch_forward_pose(fk, qa, bodies); pb, _ = kin.batch_forward_pose(fk, qb, bodies)
    ee = float(np.mean([np.linalg.norm((pa[b] - pa[kin.ROOT_BODY]) - (pb[b] - pb[kin.ROOT_BODY]), axis=-1).mean()
                        for b in kin.EE_BODIES]))
    _, ra = kin.batch_forward_pose(fk, qa, [kin.ROOT_BODY]); _, rb = kin.batch_forward_pose(fk, qb, [kin.ROOT_BODY])
    ya, yb = kin.quat_to_yaw(ra[kin.ROOT_BODY]), kin.quat_to_yaw(rb[kin.ROOT_BODY])
    heading = float(np.degrees(np.abs((ya - yb + np.pi) % (2 * np.pi) - np.pi)).mean())
    root_xy = float(np.linalg.norm(qa[:, :2] - qb[:, :2], axis=-1).mean())
    ca = (Ba[:T] * Bb[:T]).sum(-1) / (np.linalg.norm(Ba[:T], axis=-1) * np.linalg.norm(Bb[:T], axis=-1) + 1e-12)
    bspace = float(1 - ca.mean())
    la, _ = losses.functional_equivalence(fk, qa, qb, W, dt)
    return dict(joint=joint, ee=ee, heading=heading, root_xy=root_xy, bspace=bspace, l_align=float(la))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--body", default="m2c_t000", help="the original (adult) body")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--angles", type=float, nargs="+", default=list(ANGLES))
    ap.add_argument("--dirs", type=int, default=3, help="random tangent directions per angle")
    ap.add_argument("--no-trials", action="store_true", help="skip the other trials' z0")
    a = ap.parse_args()
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    import mujoco
    from model import bfm_align, losses
    from model import kinematics as kin
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    from single_z_search import rollout
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = ESConfig(device=dev)
    W = {"root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
         "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight}
    task, stem = a.clip.split("/")
    xml = REPO / "assets/robots_torque" / a.body / "robot_torque_full.xml"
    ref = np.load(REPO / "data" / a.body / "retargeting_motion" / task / f"{stem}.npz")["qpos"]
    fk = mujoco.MjModel.from_xml_path(str(xml))
    obs_mul = build_obs_multiplier(xml, REPO / "assets/robots/adult/robot.xml", mode="auto",
                                   parts=cfg.obs_scale_parts, verbose=False)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev); model.eval()
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    Bg = bfm_align.reference_embeddings(model, env1, ref, dev, obs_mul); env1.close()
    n = 16
    env, _ = make_humenv(num_envs=n, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")
    S = sources(task, stem, a.angles, a.dirs, not a.no_trials)
    # z0_repeat is placed in the LAST batch so the identical pair is two different batches and slots
    order = [i for i, s in enumerate(S) if s[0] != "z0_repeat"] + [i for i, s in enumerate(S) if s[0] == "z0_repeat"]
    Q, B = [None] * len(S), [None] * len(S)
    for b0 in range(0, len(order), n):
        idx = order[b0:b0 + n]; pad = idx + [idx[-1]] * (n - len(idx))
        zt = torch.as_tensor(np.stack([S[i][2] for i in pad]), dtype=torch.float32, device=dev)
        q, o = rollout(model, env, zt, a.steps, dev, obs_mul, init_qpos=ref[0], nv=fk.nv, return_obs=True)
        Bo = bfm_align.embed(model, o.reshape(-1, o.shape[-1]), dev).reshape(n, a.steps, -1)
        for slot, i in enumerate(idx):
            Q[i], B[i] = q[slot], Bo[slot]
    env.close()
    out = REPO / "outputs/controlled_pairs" / stem; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "adult.npz", names=np.array([s[0] for s in S]), kinds=np.array([s[1] for s in S]),
             z=np.stack([s[2] for s in S]), qpos=np.stack(Q), B=np.stack(B).astype(np.float32))
    i0 = 0
    z0h = unit(S[i0][2])
    rows = []
    for i, (name, kind, z) in enumerate(S):
        ang = float(np.degrees(np.arccos(np.clip(unit(z) @ z0h, -1, 1))))
        T = min(len(B[i]), len(Bg))
        cost = float(1 - ((B[i][:T] * Bg[:T]).sum(-1) / (np.linalg.norm(B[i][:T], axis=-1) * np.linalg.norm(Bg[:T], axis=-1))).mean())
        d = behaviour_distances(fk, Q[i], Q[i0], B[i], B[i0], losses, kin, W, 1.0 / cfg.control_fps)
        rows.append(dict(name=name, kind=kind, latent_deg=ang, cost_vs_ref=cost, **d))
    with open(out / "adult_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'source':12s} {'kind':12s} {'deg':>5s} {'cost':>6s} | {'joint':>6s} {'ee(m)':>6s} {'head':>6s} {'xy(m)':>6s} {'bspace':>7s} {'L_align':>8s}")
    for r in rows:
        print(f"{r['name']:12s} {r['kind']:12s} {r['latent_deg']:5.1f} {r['cost_vs_ref']:6.3f} | {r['joint']:6.2f} {r['ee']:6.3f} "
              f"{r['heading']:6.1f} {r['root_xy']:6.2f} {r['bspace']:7.3f} {r['l_align']:8.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
