#!/usr/bin/env python3
"""rank_initial_cost.py -- the bfm cost of z0 (the naive transfer) for every
clip, per body, ranked high to low.

Same rollout as scripts/single_z_search.py --init reference --objective bfm:
each slot starts from its own reference's frame 0 and is scored with
1 - mean_t cos(B(s_t), B(g_t)) against its own retargeted reference. So the
number for (clip, body) here is exactly the "origin_z" cost a single-z search
on that cell would print, and the ranking says which cells have the most to
gain from a search.

Usage:
    uv run scripts/rank_initial_cost.py                       # child only
    uv run scripts/rank_initial_cost.py --bodies all          # every body under assets/robots_torque
    uv run scripts/rank_initial_cost.py --bodies child giant

Writes outputs/initial_cost/z0_cost.csv (one row per clip, one column per
body, sorted by the first body's cost descending) and z0_cost.png.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from single_z_search import device_arg, project_z

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_physics_per_slot(env, qpos_list, nv):
    """AsyncVectorEnv.call broadcasts one argument to every worker; this sends
    each worker its own qpos over the same pipe protocol."""
    for pipe, q in zip(env.parent_pipes, qpos_list):
        pipe.send(("_call", ("set_physics", (), {"qpos": q, "qvel": np.zeros(nv)})))
    for pipe in env.parent_pipes:
        _, ok = pipe.recv()
        assert ok


@torch.no_grad()
def rollout_multi(model, env, z_env, steps, device, obs_mul, init_qpos_list, nv):
    """like single_z_search.rollout(return_obs=True) but with a per-slot start pose."""
    env.reset()
    set_physics_per_slot(env, init_qpos_list, nv)
    obs = {"proprio": np.stack([o["proprio"] for o in env.call("get_obs")])}
    obs_hist = []
    for _ in range(steps):
        proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.as_tensor(proprio, dtype=torch.float32, device=device)
        mu = model._actor(model._normalize(obs_t), z_env, model.cfg.actor_std).mean
        obs, _, _, _, _ = env.step(mu.cpu().numpy())
        obs_hist.append(obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul)
    return np.stack(obs_hist, axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bodies", nargs="+", default=["child"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--device", type=device_arg, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--out", default="outputs/initial_cost")
    p.add_argument("--robots-dir", default="assets/robots_torque", help="<dir>/<body>/robot*_torque_full.xml")
    p.add_argument("--data-dir", default="data", help="<dir>/<body>/retargeting_motion/")
    p.add_argument("--clip-list", default=None,
                   help="'<task> <trial>' per line: measure only these clips instead of every "
                        "clip under <data-dir>/origin_z. The rebalanced set's top-up trials "
                        "(_10..) were added after the first sweep, so this is how the 180 of them "
                        "that have no z0 cost get one without re-measuring the other 540")
    args = p.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.bodies == ["all"]:
        args.bodies = sorted(d.name for d in (REPO_ROOT / args.robots_dir).iterdir()
                             if list(d.glob("robot*_torque_full.xml")))

    zroot = Path(args.data_dir)
    zroot = (zroot if zroot.is_absolute() else REPO_ROOT / zroot) / "origin_z"
    if args.clip_list:
        lp = Path(args.clip_list)
        want = []
        for line in (lp if lp.is_absolute() else REPO_ROOT / lp).read_text().split("\n"):
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                want.append((parts[0], f"{parts[0]}_{parts[1]}"))
        missing = [c for c in want if not (zroot / c[0] / f"{c[1]}.npy").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} clips in --clip-list have no z0, e.g. {missing[:3]}")
        clips = sorted(set(want))
    else:
        clips = sorted((t.name, f.stem) for t in zroot.iterdir() if t.is_dir()
                       for f in t.glob("*.npy"))
    Z0 = np.stack([project_z(np.load(zroot / t / f"{s}.npy").reshape(-1).astype(np.float64))
                   for t, s in clips])
    print(f"{len(clips)} clips x {len(args.bodies)} bodies")

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    import mujoco

    cfg = ESConfig(device=args.device)
    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device); model.eval()
    cost = {}
    for body in args.bodies:
        t0 = time.time()
        xml = next((REPO_ROOT / args.robots_dir / body).glob("robot*_torque_full.xml"))
        obs_mul = build_obs_multiplier(xml, REPO_ROOT / "assets/robots/adult/robot.xml",
                                       mode="auto", parts=cfg.obs_scale_parts, verbose=False)
        nv = mujoco.MjModel.from_xml_path(str(xml)).nv
        env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
        env, _ = make_humenv(num_envs=args.envs, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")
        c = np.full(len(clips), np.nan)
        for b0 in range(0, len(clips), args.envs):
            idx = list(range(b0, min(b0 + args.envs, len(clips))))
            refs = []
            for i in idx:
                t, s = clips[i]
                f = REPO_ROOT / args.data_dir / body / "retargeting_motion" / t / f"{s}.npz"
                refs.append(np.load(f)["qpos"] if f.exists() else None)
            keep = [k for k, r in enumerate(refs) if r is not None]
            if not keep:
                continue
            pad = [idx[k] for k in keep] + [idx[keep[-1]]] * (args.envs - len(keep))
            init = [refs[k][0] for k in keep] + [refs[keep[-1]][0]] * (args.envs - len(keep))
            zt = torch.as_tensor(Z0[pad], dtype=torch.float32, device=args.device)
            obs = rollout_multi(model, env, zt, args.steps, args.device, obs_mul, init, nv)
            for j, k in enumerate(keep):
                Bg = bfm_align.reference_embeddings(model, env1, refs[k], args.device, obs_mul)
                c[idx[k]] = bfm_align.batch_bfm_align(model, obs[j:j + 1], Bg, args.device)[0]
            print(f"  {body}: {min(b0 + args.envs, len(clips))}/{len(clips)}  running mean {np.nanmean(c[:b0 + args.envs]):.3f}", flush=True)
        env.close(); env1.close()
        cost[body] = c
        print(f"{body}: mean {np.nanmean(c):.3f}  median {np.nanmedian(c):.3f}  "
              f"max {np.nanmax(c):.3f}  min {np.nanmin(c):.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        # write after every body so a partial run is still usable
        order = np.argsort(-cost[args.bodies[0]])
        with open(out / "z0_cost.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "task", "clip"] + list(cost) + (["mean"] if len(cost) > 1 else []))
            for r, i in enumerate(order):
                row = [r + 1, clips[i][0], clips[i][1]] + [f"{cost[b][i]:.4f}" for b in cost]
                if len(cost) > 1:
                    row.append(f"{np.nanmean([cost[b][i] for b in cost]):.4f}")
                w.writerow(row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    b0name = args.bodies[0]
    order = np.argsort(-cost[b0name])
    for b in cost:
        ax[0].plot(np.arange(len(clips)), cost[b][order], lw=1, alpha=.8 if b == b0name else .35, label=b)
    ax[0].set_xlabel(f"clip rank (sorted by {b0name} z0 cost)"); ax[0].set_ylabel("bfm cost of z0")
    ax[0].set_title("initial (z0) cost per clip"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=2)
    top = order[:30]
    ax[1].barh(np.arange(len(top)), cost[b0name][top], color="C3")
    ax[1].set_yticks(np.arange(len(top))); ax[1].set_yticklabels([clips[i][1] for i in top], fontsize=7)
    ax[1].invert_yaxis(); ax[1].set_xlabel(f"z0 cost on {b0name}"); ax[1].set_title(f"30 hardest clips for {b0name}")
    ax[1].grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(out / "z0_cost.png", dpi=130)
    print(f"-> {out / 'z0_cost.csv'}\n-> {out / 'z0_cost.png'}")


if __name__ == "__main__":
    main()
