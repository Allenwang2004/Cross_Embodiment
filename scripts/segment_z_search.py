#!/usr/bin/env python3
"""segment_z_search.py -- single_z_search with a short latent SEQUENCE.

cost_profile.py showed that on headstand the cost of z0 sits in two transition
phases (10-15% and 30-35% of the clip) rather than in a fall, and that the
per-trial search wins by flattening them. One latent has to serve every phase of
the clip; BFMTrack's point is that time-varying motion wants a time-varying
latent. This tests the smallest version of that: K knot latents spread evenly
over the reference, slerped in between (no abrupt switches, which BFMTrack notes
fall outside the policy's training distribution), after the clip the last knot.

Everything else is single_z_search.py's bfm setting: 16 rollouts per step (8
antithetic pairs, each knot perturbed by its own eps), sigma 0.05, Adam lr 0.1 on
every knot, projection of every knot to the sphere, rank-normalised costs,
reference-frame init, 300 steps scored over the reference's length, best sample
kept. K = 1 is exactly single_z_search.

At the end the best knots are re-scored 3 times (as a whole batch), because a
kept "best" is partly the luckiest draw (replay_noise.py).

usage: CUDA_VISIBLE_DEVICES=1 uv run scripts/segment_z_search.py \
           --clip headstand/headstand_0 --body child --knots 4 --out outputs/segment_z/K4/headstand_0
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.append(str(REPO))
import numpy as np
import torch
from single_z_search import project_z, rank_normalize


def slerp_knots(knots, L, T):
    """knots: (n, K, D) on the sphere -> per-step latents (T, n, D)."""
    n, K, D = knots.shape
    out = torch.empty((T, n, D), dtype=knots.dtype, device=knots.device)
    kn = torch.nn.functional.normalize(knots, dim=-1)
    r = knots.norm(dim=-1, keepdim=True)
    for t in range(T):
        if K == 1:
            out[t] = knots[:, 0]; continue
        s = min(t, L - 1) / max(L - 1, 1) * (K - 1)
        i = min(int(s), K - 2); f = s - i
        a, b = kn[:, i], kn[:, i + 1]
        c = (a * b).sum(-1, keepdim=True).clamp(-1, 1); om = torch.arccos(c)
        so = torch.sin(om)
        w_a = torch.where(so > 1e-6, torch.sin((1 - f) * om) / so, torch.full_like(so, 1 - f))
        w_b = torch.where(so > 1e-6, torch.sin(f * om) / so, torch.full_like(so, f))
        out[t] = (w_a * a + w_b * b) * r[:, i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--body", default="child")
    ap.add_argument("--knots", type=int, default=4)
    ap.add_argument("--evals", type=int, default=4992); ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05); ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    import mujoco
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    task, stem = a.clip.split("/")
    z0 = project_z(np.load(REPO / "data/origin_z" / task / f"{stem}.npy").reshape(-1).astype(np.float64))
    ref = np.load(REPO / "data" / a.body / "retargeting_motion" / task / f"{stem}.npz")["qpos"]
    L = len(ref)
    xml = REPO / "assets/robots_torque" / a.body / "robot_torque_full.xml"
    cfg = ESConfig(device=dev)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev); model.eval()
    n = 2 * a.pairs
    env, _ = make_humenv(num_envs=n, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")
    nv = mujoco.MjModel.from_xml_path(str(xml)).nv
    obs_mul = build_obs_multiplier(xml, REPO / "assets/robots/adult/robot.xml", mode="auto",
                                   parts=cfg.obs_scale_parts, verbose=False)
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    Bg = bfm_align.reference_embeddings(model, env1, ref, dev, obs_mul); env1.close()

    @torch.no_grad()
    def score(K):   # K: (n, knots, 256) numpy -> (n,) cost
        zt = slerp_knots(torch.as_tensor(K, dtype=torch.float32, device=dev), L, a.steps)
        obs, _ = env.reset()
        env.call("set_physics", qpos=ref[0], qvel=np.zeros(nv))
        obs = {"proprio": np.stack([o["proprio"] for o in env.call("get_obs")])}
        hist = []
        for t in range(a.steps):
            p = obs["proprio"] * obs_mul
            mu = model._actor(model._normalize(torch.as_tensor(p, dtype=torch.float32, device=dev)),
                              zt[t], model.cfg.actor_std).mean
            obs, _, _, _, _ = env.step(mu.cpu().numpy())
            hist.append(obs["proprio"] * obs_mul)
        return bfm_align.batch_bfm_align(model, np.stack(hist, 1).astype(np.float32), Bg, dev)

    Kn = a.knots
    knots = np.repeat(z0[None], Kn, 0)
    c0 = float(score(np.repeat(knots[None], n, 0))[0])
    best = dict(cost=c0, knots=knots.copy(), gen=-1)
    m = np.zeros_like(knots); v = np.zeros_like(knots); b1, b2 = 0.9, 0.999
    rng = np.random.default_rng(a.seed)
    gens = max(a.evals // n, 1); curve = []; t0 = time.time()
    for g in range(gens):
        eps = rng.standard_normal((a.pairs, Kn, knots.shape[1]))
        cand = np.concatenate([knots[None] + a.sigma * eps, knots[None] - a.sigma * eps], 0)
        cand = np.stack([[project_z(k) for k in c] for c in cand])
        cost = score(cand)
        i = int(np.argmin(cost))
        if cost[i] < best["cost"]:
            best = dict(cost=float(cost[i]), knots=cand[i].copy(), gen=g)
        s = rank_normalize(cost)
        grad = ((s[:a.pairs] - s[a.pairs:])[:, None, None] * eps).sum(0) / (2 * a.pairs * a.sigma)
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad * grad
        mh = m / (1 - b1 ** (g + 1)); vh = v / (1 - b2 ** (g + 1))
        knots = np.stack([project_z(k) for k in knots - a.lr * mh / (np.sqrt(vh) + 1e-8)])
        curve.append(((g + 1) * n, float(cost.min()), float(cost.mean()), best["cost"]))
        if g % 50 == 0 or g == gens - 1:
            print(f"gen {g:4d}/{gens}  best {best['cost']:.4f}  gen_mean {cost.mean():.4f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    rep = [float(score(np.repeat(best["knots"][None], n, 0))[0]) for _ in range(3)]
    env.close()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "best_knots.npy", best["knots"])
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["evals", "gen_best", "gen_mean", "best_so_far"]); w.writerows(curve)
    ang = [float(np.degrees(np.arccos(np.clip(k @ z0 / (np.linalg.norm(k) * np.linalg.norm(z0)), -1, 1)))) for k in best["knots"]]
    json.dump(dict(clip=a.clip, body=a.body, knots=Kn, evals=gens * n, origin_z_cost=c0, best_cost=best["cost"],
                   best_gen=best["gen"], replay_costs=rep, replay_mean=float(np.mean(rep)), knot_deg_from_z0=ang),
              open(out / "summary.json", "w"), indent=1)
    print(f"-> {out}: z0 {c0:.4f}  best {best['cost']:.4f}  replay {np.mean(rep):.4f}  knots from z0 {np.round(ang, 1).tolist()} deg")


if __name__ == "__main__":
    main()
