#!/usr/bin/env python3
"""geodesic_profile.py -- the cost along the path from z0 to a clip's searched
best_z. What does the road actually look like?

Why this exists: headstand needs the SHORTEST trip of any category (best_z 28.3
deg from z0, against rotate's 55.1 and move's 53.5) and the adapter covers a
normal fraction of it (9.3 deg, 33%), yet headstand is the only category that
ends up WORSE than z0 (ratio 1.24 against move's 0.57). Distance is therefore
not the explanation. The remaining candidate is the shape of the road: if the
cost rises before it falls, stopping partway is worse than not starting.

Walks the great circle
    z(t) = sqrt(d) * normalize( cos(t*th) * z0_hat + sin(t*th) * u_hat )
from t=0 (z0) to t=1.2 (20% past the target), scoring every step exactly as
training does -- bfm, reference init, the same torque XML.

Usage:
    uv run scripts/geodesic_profile.py --clips headstand:3,4,9,15,20 move-ego-0-2:0,4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT not in sys.path:
    sys.path.append(PARENT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from single_z_search import device_arg, project_z
from rank_initial_cost import rollout_multi

REPO_ROOT = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"


def slerp(z0, z1, ts):
    a = z0 / np.linalg.norm(z0)
    b = z1 / np.linalg.norm(z1)
    th = np.arccos(np.clip(a @ b, -1, 1))
    u = b - (a @ b) * a
    u = u / max(np.linalg.norm(u), 1e-12)
    return np.stack([project_z(np.cos(t * th) * a + np.sin(t * th) * u) for t in ts]), np.degrees(th)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--clips", nargs="+", required=True,
                   help="task:trial,trial,... e.g. headstand:3,4,9 move-ego-0-2:0,4")
    p.add_argument("--floor-dir", default="outputs/single_z_floor")
    p.add_argument("--alt-floor", nargs="*",
                   default=["outputs/single_z_seeds_s005_5k", "outputs/single_z", "outputs/es_vs_cmaes"],
                   help="other places a *_bfm_s* search for that clip may live")
    p.add_argument("--steps-t", type=int, default=25, help="points along the path")
    p.add_argument("--tmax", type=float, default=1.2)
    p.add_argument("--xml", default="assets/robots_torque/child/robot_torque_full.xml")
    p.add_argument("--data-dir", default="data/child/retargeting_motion")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", type=device_arg,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--out", default="outputs/z_target_analysis")
    args = p.parse_args()
    out = REPO_ROOT / args.out; out.mkdir(parents=True, exist_ok=True)

    import glob, json
    jobs = []
    for spec in args.clips:
        task, _, trials = spec.partition(":")
        for k in trials.split(","):
            stem = f"{task}_{k}"
            cand = [REPO_ROOT / args.floor_dir / f"{stem}_bfm_s0" / "best_z.npy"]
            for d in args.alt_floor:
                cand += [Path(x) for x in glob.glob(str(REPO_ROOT / d / f"{stem}_bfm_s*" / "best_z.npy"))]
            best = None
            for c in cand:
                if not Path(c).exists(): continue
                s = json.loads((Path(c).parent / "summary.json").read_text())
                if best is None or s["best"]["cost"] < best[1]:
                    best = (np.load(c).reshape(-1).astype(np.float64), s["best"]["cost"])
            if best is None:
                print(f"  skip {stem}: no bfm search found"); continue
            jobs.append((task, int(k), best[0]))
    if not jobs:
        raise SystemExit("no clip had a searched best_z")
    ts = np.linspace(0, args.tmax, args.steps_t)

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    import mujoco

    cfg = ESConfig(device=args.device)
    xml = REPO_ROOT / args.xml
    obs_mul = build_obs_multiplier(xml, REPO_ROOT / "assets/robots/adult/robot.xml",
                                   mode="auto", parts=cfg.obs_scale_parts, verbose=False)
    nv = mujoco.MjModel.from_xml_path(str(xml)).nv
    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device); model.eval()
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    env, _ = make_humenv(num_envs=len(ts), vectorization_mode="async", task=None,
                         xml=str(xml), state_init="Default")

    prof = {}
    for task, k, bz in jobs:
        z0 = project_z(np.load(REPO_ROOT / "data/origin_z" / task / f"{task}_{k}.npy")
                       .reshape(-1).astype(np.float64))
        ref = np.load(REPO_ROOT / args.data_dir / task / f"{task}_{k}.npz")["qpos"]
        Bg = bfm_align.reference_embeddings(model, env1, ref, args.device, obs_mul)
        Z, th = slerp(z0, bz, ts)
        obs = rollout_multi(model, env, torch.as_tensor(Z, dtype=torch.float32, device=args.device),
                            args.steps, args.device, obs_mul, [ref[0]] * len(ts), nv)
        c = np.array([bfm_align.batch_bfm_align(model, obs[i:i+1], Bg, args.device)[0]
                      for i in range(len(ts))])
        prof[f"{task}_{k}"] = dict(t=ts, deg=ts * th, cost=c, theta=th)
        peak = c.max() / c[0]
        print(f"{task}_{k:<3d} th={th:5.1f}deg  cost {c[0]:.3f} -> min {c.min():.3f} "
              f"(at {ts[c.argmin()]*th:5.1f}deg)   highest point on the way: "
              f"{peak:5.2f}x z0 at {ts[c.argmax()]*th:5.1f}deg", flush=True)
    env.close(); env1.close()
    np.savez(out / "geodesic_profile.npz", **{k: v["cost"] for k, v in prof.items()},
             **{k + "_deg": v["deg"] for k, v in prof.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = sorted({n.rsplit("_", 1)[0] for n in prof})
    fig, ax = plt.subplots(1, len(tasks) + 1, figsize=(5.2 * (len(tasks) + 1), 4.8))
    fig.patch.set_facecolor(SURF)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
    for a, tk in zip(ax, tasks):
        a.set_facecolor(SURF)
        for i, n in enumerate([n for n in prof if n.rsplit("_", 1)[0] == tk]):
            d = prof[n]
            a.plot(d["deg"], d["cost"] / d["cost"][0], lw=1.9, color=pal[i % len(pal)], label=n)
        a.axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.2)
        a.set_title(tk, fontsize=11.5, color=INK)
        a.set_xlabel("degrees travelled from z0", fontsize=9.5, color=INK2)
        a.legend(fontsize=8)
    ax[0].set_ylabel("cost / cost at z0", fontsize=10.5, color=INK)
    a = ax[-1]; a.set_facecolor(SURF)
    for tk, col in zip(tasks, pal):
        M = [prof[n] for n in prof if n.rsplit("_", 1)[0] == tk]
        g = np.linspace(0, 1, 40)
        Y = np.stack([np.interp(g, d["t"], d["cost"] / d["cost"][0]) for d in M])
        a.plot(g * np.mean([d["theta"] for d in M]), Y.mean(0), lw=2.6, color=col, label=tk)
        a.fill_between(g * np.mean([d["theta"] for d in M]), Y.min(0), Y.max(0),
                       color=col, alpha=.14)
    a.axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.2)
    a.set_title("mean per task (band = min..max)", fontsize=11.5, color=INK)
    a.set_xlabel("degrees travelled from z0", fontsize=9.5, color=INK2)
    a.legend(fontsize=9)
    for a in ax:
        a.grid(alpha=.22, color=INK3, lw=.7); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_color(INK3)
    fig.suptitle("cost along the great circle from z0 to that clip's searched best_z",
                 fontsize=12.5, color=INK, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .93))
    f = out / "geodesic_profile.png"
    fig.savefig(f, dpi=140, facecolor=SURF)
    print(f"-> {f}")


if __name__ == "__main__":
    main()
