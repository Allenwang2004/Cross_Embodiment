#!/usr/bin/env python3
"""robot_lso_search.py -- scripts/lso_search.py (BFMTrack's latent sequence optimisation: one latent per control step) on
a robot of model/robots.py, driven through model/robot_env.RobotVecEnv exactly as scripts/robot_z_search.py drives it
(SMPL-equivalent observations in, Motivo's SMPL action mapped onto the robot out).

Reference: the ADULT's SMPL clip (--ref, e.g. amass_data/amass_origin_motion_grounded/ACCAD/<clip>.npz). Goal of step t
(state s_t -> s_t+1) = B of frame t+1 of the adult's clip (bfm_align.obs_from_qpos at 30 Hz, as robot_z_search); every
rollout starts from frame 0 mapped onto the robot, at rest. Per-step reward
  r_t = cos(B(o_t+1), B(g_t+1)) - heading_weight * (1 - cos(heading - ref heading)) / 2 - pos_weight * |xy - ref xy|
        - ik_weight * ik_t
heading / xy on the robot's SMPL-equivalent root against the adult's clip (as robot_z_search), ik_t = mean IK residual
(rad) over the paired links of the action of step t (RobotEnv's last_resid: how far Motivo's command is from any pose
the robot can make). Cost reported = 1 - mean per-step reward, so with ik_weight 1 it is robot_z_search's cost without
the anchor term. The optimiser is lso_search's: mu_t starts at unit(B(g_t+1)), pink noise over time, discounted
return-to-go (--gamma), per-step ranks (--rank), Adam, mu_t back on the sphere, slot 0 of every batch = the mean
sequence. Baseline rolled out first: Motivo's zero-shot tracking ER with window 8.
Writes <out>/{curve.csv, summary.json, mu_trace.npz, rollouts.npz}: rollouts.npz holds per sequence (ER8, LSO start,
LSO final mean, LSO best) the robot qpos q_* (S x nq), its SMPL-equivalent qs_* and the per-step IK numbers fz_*.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/robot_lso_search.py --robot g1 \
           --ref amass_data/amass_origin_motion_grounded/ACCAD/<clip>.npz --out outputs/g1/lso/<clip> \
           [--n 128 --iters 64 --rank --heading-weight 1 --pos-weight 0.1 --ik-weight 1 --fill ridge]"""
import argparse, csv, json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from lso_search import colored_noise                                # noqa: E402
from model import bfm_align                                         # noqa: E402
from model.robot_env import RobotVecEnv                             # noqa: E402
from model.robot_map import RobotMap                                # noqa: E402
from model.robots import ROBOTS                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--ref", required=True, help="the adult's SMPL clip (.npz with qpos)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.0125)
    ap.add_argument("--lr", type=float, default=0.00625)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.8, 0.99))
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--beta", type=float, default=1.0, help="noise colour: 0 white, 1 pink, 2 red")
    ap.add_argument("--rank", action="store_true", help="per-step centred ranks of the returns-to-go (lso_search --rank)")
    ap.add_argument("--heading-weight", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=0.1)
    ap.add_argument("--ik-weight", type=float, default=1.0)
    ap.add_argument("--fill", default="ridge", choices=["fixed", "ridge"])
    ap.add_argument("--gains", default=None, help="gains json (default: ROBOTS[robot]['gains_json'])")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ref = np.load(REPO / a.ref)["qpos"].astype(np.float64); T = len(ref); S = T - 1
    n_envs = a.n + 1
    vec = RobotVecEnv(a.robot, n_envs, fill=a.fill, gains=a.gains)       # spawn the workers before CUDA is touched
    rm = RobotMap(a.robot, fill=a.fill); n_link = len(rm.pairs) - 1
    from humenv import make_humenv
    import humenv.utils as hu
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device); model.eval()
    adult, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
    fk = adult.unwrapped.model; dk = mujoco.MjData(fk)
    Bg = bfm_align.embed(model, bfm_align.obs_from_qpos(adult, ref), a.device)[1:]      # (S, 256): goal of step t = frame t+1
    Bg_t = torch.as_tensor(Bg, device=a.device)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)

    def heading_xy(qs):
        flat = qs.reshape(-1, qs.shape[-1]); h = np.empty(len(flat))
        for i, qq in enumerate(flat):
            dk.qpos[:] = qq; mujoco.mj_kinematics(fk, dk)
            h[i] = hu.calc_heading(hu.remove_base_rot(dk.xquat[1][None].copy(), "smpl"))[0]
        return h.reshape(qs.shape[:-1]), qs[..., :2]
    ref_h, ref_xy = heading_xy(ref[1:])

    @torch.no_grad()
    def rollout(Z):
        """Z: (n_envs, S, 256) on the unit sphere -> per-step reward (n_envs, S); keeps the trajectories in rollout.keep"""
        zt = torch.as_tensor(16.0 * Z, dtype=torch.float32, device=a.device)
        obs, _, _, _ = vec.reset(ref[0])
        O, QS, QG, FZ = [], [], [], []
        for t in range(S):
            o = torch.as_tensor(obs, dtype=torch.float32, device=a.device)
            u = model._actor(model._normalize(o), zt[:, t], model.cfg.actor_std).mean.cpu().numpy()
            obs, qs, qg, fz = vec.step(u)
            O.append(obs); QS.append(qs); QG.append(qg); FZ.append(fz)
        O, QS, QG, FZ = (np.stack(x, axis=1) for x in (O, QS, QG, FZ))
        Bs = torch.as_tensor(bfm_align.embed(model, O.astype(np.float32).reshape(n_envs * S, -1), a.device),
                             device=a.device).reshape(n_envs, S, -1)
        cos = torch.nn.functional.cosine_similarity(Bs, Bg_t[None], dim=-1).cpu().numpy()
        h, xy = heading_xy(QS)
        hd = (1 - np.cos(h - ref_h[None])) / 2; ps = np.linalg.norm(xy - ref_xy[None], axis=-1)
        ik = np.radians(FZ[:, :, :n_link].mean(-1))                      # (n_envs, S) rad
        r = cos - a.heading_weight * hd - a.pos_weight * ps - a.ik_weight * ik
        rollout.parts = dict(bfm=1 - cos.mean(1), head=hd.mean(1), pos=ps.mean(1), ik_deg=np.degrees(ik.mean(1)),
                             atlim=FZ[:, :, n_link].mean(1) / len(rm.g_jnt), pelvis_min=QG[:, :, 2].min(1))
        rollout.keep = dict(qg=QG, qs=QS, fz=FZ)
        return r

    def terms(r, i):
        return dict(cost=float(1 - r[i].sum() / S), **{k: float(v[i]) for k, v in rollout.parts.items()})

    def line(k, v):
        return (f"  {k:28s} cost {v['cost']:.4f} (bfm {v['bfm']:.4f}, heading {v['head']:.3f}, pos {v['pos']:.2f} m, "
                f"ik {v['ik_deg']:.1f} deg, at limit {100 * v['atlim']:.1f}%, pelvis min {v['pelvis_min']:.2f} m)")

    keep = {}

    def store(tag, i):
        for k, v in rollout.keep.items():
            keep[f"{'q' if k == 'qg' else k}_{tag}"] = v[i].astype(np.float32)

    # ---- zero-shot references: B(g_t) (the LSO start) and Motivo's ER window 8 ------------------------------------
    mu = unit(Bg.astype(np.float64))
    er8 = unit(np.stack([Bg[t:t + 8].mean(0) for t in range(S)]).astype(np.float64))
    Z = np.repeat(mu[None], n_envs, 0); Z[1] = er8
    t0 = time.time(); r = rollout(Z)
    res = {"B(g_t) (LSO start)": terms(r, 0), "ER L=8 (Motivo zero-shot)": terms(r, 1)}
    store("Bg", 0); store("ER8", 1)
    print(f"{ROBOTS[a.robot]['name']} LSO on {a.ref}: {T} frames, {S} steps, {a.n} sequences per iteration, {a.iters} iterations "
          f"({time.time() - t0:.0f} s per batch)")
    for k, v in res.items():
        print(line(k, v), flush=True)

    # ---- LSO (lso_search.py's optimiser) ---------------------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    m, v = np.zeros_like(mu), np.zeros_like(mu); b1, b2 = a.betas
    disc = a.gamma ** np.arange(S)
    curve, trace = [], [mu.astype(np.float32)]
    best = dict(cost=res["B(g_t) (LSO start)"]["cost"], z=mu.copy(), it=-1, src="start")
    for it in range(a.iters):
        eps = colored_noise(a.beta, (a.n, 256, S), rng).transpose(0, 2, 1)       # (n, S, 256), coloured over time
        Z = np.concatenate([mu[None], unit(mu[None] + a.sigma * eps)])
        r = rollout(Z)
        costs = 1 - r.sum(1) / S
        R = np.stack([np.convolve(x[::-1], disc)[:S][::-1] for x in r[1:]])       # discounted return-to-go
        if a.rank:
            R = np.argsort(np.argsort(R, axis=0), axis=0) / (a.n - 1) - 0.5
        A = R - (R.sum(0, keepdims=True) - R) / (a.n - 1)
        g = (A[:, :, None] * eps).mean(0) / a.sigma
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mu = unit(mu + a.lr * (m / (1 - b1 ** (it + 1))) / (np.sqrt(v / (1 - b2 ** (it + 1))) + 1e-8))
        i = int(np.argmin(costs))
        if costs[i] < best["cost"]:
            best = dict(cost=float(costs[i]), z=Z[i].copy(), it=it, src="mean" if i == 0 else "sample")
        p = rollout.parts
        row = dict(iter=it, mean_seq_cost=float(costs[0]), mean_seq_bfm=float(p["bfm"][0]), mean_seq_ik_deg=float(p["ik_deg"][0]),
                   best_sample_cost=float(costs[1:].min()), sample_mean_cost=float(costs[1:].mean()), best_so_far=best["cost"],
                   mu_move=float(np.linalg.norm(16 * (mu - trace[-1]), axis=1).mean()), sec=time.time() - t0)
        curve.append(row); trace.append(mu.astype(np.float32))
        print(f"  iter {it:3d}  mean sequence cost {costs[0]:.4f} (bfm {p['bfm'][0]:.4f}, ik {p['ik_deg'][0]:.1f} deg)  "
              f"samples best {costs[1:].min():.4f} mean {costs[1:].mean():.4f}  |d mu_t| {row['mu_move']:.3f}  [{row['sec'] / 60:.1f} min]", flush=True)
        with open(out / "curve.csv", "w", newline="") as f:                       # written every iteration: a stop keeps the curve
            w = csv.DictWriter(f, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
    r = rollout(np.repeat(mu[None], n_envs, 0)); res["LSO final mean"] = terms(r, 0); store("LSO", 0)
    r = rollout(np.repeat(best["z"][None], n_envs, 0))
    res["LSO best (re-rolled)"] = dict(**terms(r, 0), it=best["it"], src=best["src"]); store("LSO_best", 0)
    vec.close()
    np.savez_compressed(out / "mu_trace.npz", mu=np.stack(trace), best=best["z"].astype(np.float32))
    np.savez_compressed(out / "rollouts.npz", ref=ref.astype(np.float32), **keep)
    (out / "summary.json").write_text(json.dumps(dict(robot=a.robot, ref=a.ref, frames=T, n=a.n, iters=a.iters, sigma=a.sigma,
                                                      lr=a.lr, gamma=a.gamma, beta=a.beta, rank=a.rank, seed=a.seed,
                                                      heading_weight=a.heading_weight, pos_weight=a.pos_weight, ik_weight=a.ik_weight,
                                                      fill=a.fill, gains=a.gains or ROBOTS[a.robot].get("gains_json"), results=res,
                                                      minutes=(time.time() - t0) / 60), indent=1))
    print("\n" + "\n".join(line(k, v) for k, v in res.items()))
    print(f"-> {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
