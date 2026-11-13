#!/usr/bin/env python3
"""robot_lso_search.py -- scripts/lso_search.py (BFMTrack's latent sequence optimisation: one latent per control step) on
a robot of model/robots.py, driven through model/robot_env.RobotVecEnv exactly as scripts/robot_z_search.py drives it
(SMPL-equivalent observations in, Motivo's SMPL action mapped onto the robot out).

Reference: the ADULT's SMPL clip (--ref, e.g. amass_data/amass_origin_motion_grounded/ACCAD/<clip>.npz). Goal of step t
(state s_t -> s_t+1) = B of frame t+1 of the adult's clip (bfm_align.obs_from_qpos at 30 Hz, as robot_z_search); every
rollout starts from frame 0 mapped onto the robot, at rest or (--init-vel reference) moving at the reference's velocity
(frames 0 -> 1 mapped onto the robot, as robot_z_search --init-vel reference). Per-step reward
  r_t = cos(B(o_t+1), B(g_t+1)) - heading_weight * (1 - cos(heading - ref heading)) / 2 - pos_weight * |xy - ref xy|
        - ik_weight * ik_t
heading / xy on the robot's SMPL-equivalent root against the adult's clip (as robot_z_search), ik_t = mean IK residual
(rad) over the paired links of the action of step t (RobotEnv's last_resid: how far Motivo's command is from any pose
the robot can make). Cost reported = 1 - mean per-step reward, so with ik_weight 1 it is robot_z_search's cost without
the anchor term. --objective phc (11/09): the cos term is replaced by PHC's tracking reward per step (model/phc_reward.py,
BFM-Zero's tracking reward, as lso_search.py --objective phc) on the robot's own bodies against the robot target (--target,
default model/robots.target_path: the clip mapped onto the robot), lengths / the map's root scale s; the state before the
first step is the rollout's start pose on one side and target frame 0 on the other.
The optimiser is lso_search's: mu_t starts at unit(B(g_t+1)), pink noise over time, discounted return-to-go (--gamma),
per-step ranks (--rank), Adam, mu_t back on the sphere, slot 0 of every batch = the mean sequence. --method ta (11/09):
lso_search.py's BFM-Zero trajectory adaptation, same code: from Motivo's ER L = 8 sequence, white noise
--sigma x exp(-(M - i) / (beta1 M) - (H - h) / (beta2 H)), softmax average of the samples by whole-rollout cost
(z-scored, temperature --ta-temp). Baseline rolled out first: Motivo's zero-shot tracking ER with window 8.
Best so far: the lowest-cost SAMPLE (slot 0, the mean, is logged only), as lso_search.py; --best-from all lets the mean
win too, as every run of this script before 11/09.
Writes <out>/{curve.csv, summary.json, mu_trace.npz, rollouts.npz}: rollouts.npz holds per sequence (ER8, LSO start,
LSO final mean, LSO best) the robot qpos q_* (S x nq), its SMPL-equivalent qs_* and the per-step IK numbers fz_*, and,
as lso_search.py, the robot qpos of the best sequence so far after k iterations (q_best_it<k>) and of the mean sequence
after k updates (q_mean_it<k>) for k in --budgets.
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
from model.robots import ROBOTS, target_path                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--ref", required=True, help="the adult's SMPL clip (.npz with qpos)")
    ap.add_argument("--target", default=None, help="--objective phc: the robot target (default model/robots.target_path of --ref's stem)")
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
    ap.add_argument("--objective", choices=("bfm", "phc"), default="bfm")
    ap.add_argument("--method", choices=("lso", "ta"), default="lso")
    ap.add_argument("--ta-beta1", type=float, default=0.85, help="--method ta: iteration annealing (BFM-Zero)")
    ap.add_argument("--ta-beta2", type=float, default=0.9, help="--method ta: horizon annealing (BFM-Zero)")
    ap.add_argument("--ta-temp", type=float, default=0.05, help="--method ta: softmax temperature (DIAL-MPC's example config)")
    ap.add_argument("--init-vel", default="rest", choices=["rest", "reference"])
    ap.add_argument("--best-from", default="samples", choices=["samples", "all"],
                    help="samples: the mean sequence (slot 0) never becomes the best, as lso_search.py; all: it can (runs before 11/09)")
    ap.add_argument("--trace", choices=("all", "budgets"), default="all",
                    help="mu_trace.npz: the mean sequence after every iteration, or only after 0 and --budgets iterations")
    ap.add_argument("--budgets", type=int, nargs="*", default=[4, 8, 16, 32, 64])
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
    phc = None
    if a.objective == "phc":
        from model.phc_reward import PHCCost
        tgt = Path(a.target) if a.target else target_path(a.robot, Path(a.ref).stem)
        gref = np.load(tgt if tgt.is_absolute() else REPO / tgt)["qpos"].astype(np.float64)
        assert len(gref) == T, (len(gref), T)
        mg = mujoco.MjModel.from_xml_path(str(REPO / ROBOTS[a.robot]["xml"]))
        phc = PHCCost(mg, gref[0], gref[1:], scale=rm.s)
        print(f"phc: {mg.nbody - 1} robot bodies against {tgt}, lengths / s = {rm.s:.6f}", flush=True)

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
        obs, _, qg0, _ = vec.reset((ref[0], ref[1]) if a.init_vel == "reference" else ref[0])
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
        main = cos
        if phc is not None:                                               # every slot starts from the same pose (checked)
            assert np.abs(qg0 - qg0[:1]).max() == 0
            main = np.stack([phc.parts(QG[k], qg0[0])[0] for k in range(n_envs)])
        r = main - a.heading_weight * hd - a.pos_weight * ps - a.ik_weight * ik
        rollout.parts = dict(bfm=1 - cos.mean(1), head=hd.mean(1), pos=ps.mean(1), ik_deg=np.degrees(ik.mean(1)),
                             atlim=FZ[:, :, n_link].mean(1) / len(rm.g_jnt), pelvis_min=QG[:, :, 2].min(1),
                             **({"phc": 1 - main.mean(1)} if phc is not None else {}))
        rollout.keep = dict(qg=QG, qs=QS, fz=FZ)
        return r

    def terms(r, i):
        return dict(cost=float(1 - r[i].sum() / S), **{k: float(v[i]) for k, v in rollout.parts.items()})

    def line(k, v):
        return (f"  {k:28s} cost {v['cost']:.4f} (bfm {v['bfm']:.4f}" + (f", phc {v['phc']:.4f}" if "phc" in v else "")
                + f", heading {v['head']:.3f}, pos {v['pos']:.2f} m, "
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
    q_start = {"B(g_t) (LSO start)": rollout.keep["qg"][0].copy(), "ER L=8 (Motivo zero-shot)": rollout.keep["qg"][1].copy()}
    store("Bg", 0); store("ER8", 1)
    print(f"{ROBOTS[a.robot]['name']} {a.method.upper()} on {a.ref}: {T} frames, {S} steps, {a.n} sequences per iteration, {a.iters} iterations "
          f"({time.time() - t0:.0f} s per batch)")
    for k, v in res.items():
        print(line(k, v), flush=True)

    # ---- LSO / trajectory adaptation (lso_search.py's optimisers) --------------------------------------------------
    rng = np.random.default_rng(a.seed)
    start_key = "B(g_t) (LSO start)" if a.method == "lso" else "ER L=8 (Motivo zero-shot)"
    if a.method == "ta":
        assert not a.rank, "--method ta: no --rank"
        mu = er8.copy()
    m, v = np.zeros_like(mu), np.zeros_like(mu); b1, b2 = a.betas
    disc = a.gamma ** np.arange(S)
    curve, trace = [], [mu.astype(np.float32)]
    best = dict(cost=res[start_key]["cost"], z=mu.copy(), it=-1, src="start", q=q_start[start_key])
    budget_cost = {}
    ta_h = np.exp(-(S - np.arange(1, S + 1)) / (a.ta_beta2 * S))                      # horizon factor, 1 at the last step
    for it in range(a.iters):
        if a.method == "ta":
            sig = a.sigma * np.exp(-(a.iters - (a.iters - it)) / (a.ta_beta1 * a.iters)) * ta_h   # i = M - it counts down from M
            eps = rng.standard_normal((a.n, S, 256))
            Z = np.concatenate([mu[None], unit(mu[None] + sig[None, :, None] * eps)])
        else:
            eps = colored_noise(a.beta, (a.n, 256, S), rng).transpose(0, 2, 1)       # (n, S, 256), coloured over time
            Z = np.concatenate([mu[None], unit(mu[None] + a.sigma * eps)])
        r = rollout(Z)
        if it in a.budgets:                                                            # slot 0: the mean after `it` updates
            keep[f"q_mean_it{it}"] = rollout.keep["qg"][0].astype(np.float32)
        costs = 1 - r.sum(1) / S
        i = int(np.argmin(costs)) if a.best_from == "all" else 1 + int(np.argmin(costs[1:]))
        if costs[i] < best["cost"]:
            best = dict(cost=float(costs[i]), z=Z[i].copy(), it=it, src="mean" if i == 0 else "sample", q=rollout.keep["qg"][i].copy())
        if it + 1 in a.budgets:
            keep[f"q_best_it{it + 1}"] = best["q"].astype(np.float32); budget_cost[it + 1] = best["cost"]
        p = rollout.parts
        if a.method == "ta":                                                           # softmax over the z-scored costs of the samples
            c = costs[1:]; zc = (c - c.mean()) / (c.std() + 1e-12); w = np.exp(-(zc - zc.min()) / a.ta_temp); w /= w.sum()
            mu = unit((w[:, None, None] * Z[1:]).sum(0))
        else:
            R = np.stack([np.convolve(x[::-1], disc)[:S][::-1] for x in r[1:]])       # discounted return-to-go
            if a.rank:
                R = np.argsort(np.argsort(R, axis=0), axis=0) / (a.n - 1) - 0.5
            A = R - (R.sum(0, keepdims=True) - R) / (a.n - 1)
            g = (A[:, :, None] * eps).mean(0) / a.sigma
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            mu = unit(mu + a.lr * (m / (1 - b1 ** (it + 1))) / (np.sqrt(v / (1 - b2 ** (it + 1))) + 1e-8))
        row = dict(iter=it, mean_seq_cost=float(costs[0]), mean_seq_bfm=float(p["bfm"][0]), mean_seq_ik_deg=float(p["ik_deg"][0]),
                   best_sample_cost=float(costs[1:].min()), sample_mean_cost=float(costs[1:].mean()), best_so_far=best["cost"],
                   mu_move=float(np.linalg.norm(16 * (mu - trace[-1]), axis=1).mean()), sec=time.time() - t0)
        curve.append(row); trace.append(mu.astype(np.float32))
        print(f"  iter {it:3d}  mean sequence cost {costs[0]:.4f} (bfm {p['bfm'][0]:.4f}, ik {p['ik_deg'][0]:.1f} deg)  "
              f"samples best {costs[1:].min():.4f} mean {costs[1:].mean():.4f}  |d mu_t| {row['mu_move']:.3f}  [{row['sec'] / 60:.1f} min]", flush=True)
        with open(out / "curve.csv", "w", newline="") as f:                       # written every iteration: a stop keeps the curve
            w_ = csv.DictWriter(f, fieldnames=list(curve[0])); w_.writeheader(); w_.writerows(curve)
    r = rollout(np.repeat(mu[None], n_envs, 0)); res["LSO final mean"] = terms(r, 0); store("LSO", 0)
    if a.iters in a.budgets:
        keep[f"q_mean_it{a.iters}"] = rollout.keep["qg"][0].astype(np.float32)
    r = rollout(np.repeat(best["z"][None], n_envs, 0))
    res["LSO best (re-rolled)"] = dict(**terms(r, 0), it=best["it"], src=best["src"]); store("LSO_best", 0)
    vec.close()
    its = list(range(len(trace))) if a.trace == "all" else [0] + [k for k in a.budgets if k < len(trace)]
    np.savez_compressed(out / "mu_trace.npz", mu=np.stack([trace[k] for k in its]), mu_iters=np.array(its), best=best["z"].astype(np.float32))
    np.savez_compressed(out / "rollouts.npz", ref=ref.astype(np.float32), **keep)
    (out / "summary.json").write_text(json.dumps(dict(robot=a.robot, ref=a.ref, frames=T, n=a.n, iters=a.iters, sigma=a.sigma,
                                                      lr=a.lr, betas=list(a.betas), gamma=a.gamma, beta=a.beta, rank=a.rank, seed=a.seed,
                                                      heading_weight=a.heading_weight, pos_weight=a.pos_weight, ik_weight=a.ik_weight,
                                                      objective=a.objective, target=str(a.target) if a.target else None, method=a.method,
                                                      ta_beta1=a.ta_beta1, ta_beta2=a.ta_beta2, ta_temp=a.ta_temp, init_vel=a.init_vel,
                                                      best_from=a.best_from, budgets=a.budgets, best_cost_at=budget_cost,
                                                      best_it=best["it"], best_src=best["src"], sample_rollouts=a.iters * a.n,
                                                      fill=a.fill, gains=a.gains or ROBOTS[a.robot].get("gains_json"), results=res,
                                                      minutes=(time.time() - t0) / 60), indent=1))
    print("\n" + "\n".join(line(k, v) for k, v in res.items()))
    print(f"-> {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
