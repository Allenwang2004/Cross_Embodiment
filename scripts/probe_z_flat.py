#!/usr/bin/env python3
"""probe_z_flat.py -- is the set of good z's one connected flat region, or
several separate basins?

Several seeds of scripts/single_z_search.py end at z's with the same cost
but cos ~0.4 apart. Two pictures fit that: a connected plateau (every point
between two solutions is also a solution) or separate basins (the path
between them climbs). This walks the great-circle (slerp) between every pair
of best z's on the sqrt(256) sphere, rolls each waypoint out on the same body
with the same settings the search used, and scores it with the same loss.
Three families of paths:

  pair      best_i -> best_j, every pair                (the question)
  from_z0   z0 -> best_i                                (how the search descended)
  random    best_i -> a random point the SAME angle away, tangent-random
            direction                                   (control: is the loss
                                                        flat in every direction,
                                                        or only along the paths
                                                        the seeds found?)

Everything is rolled out in one batched env, so the cost of a run is
(#waypoints / 16) rollouts of 300 steps.

Usage:
    uv run scripts/probe_z_flat.py --root outputs/single_z_seeds_s005_5k --clip move-ego-0-2_4
    uv run scripts/probe_z_flat.py --root ... --clip ... --n 9 --random 3

Writes <root>/z_flat_<clip>.png, <root>/z_flat_<clip>.csv and a contact sheet
<root>/z_flat_<clip>_midpoints.png (the midpoint of every pair path, one frame).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import numpy as np
import torch

from single_z_search import device_arg, project_z, rollout

REPO_ROOT = Path(__file__).resolve().parent.parent


def slerp(a, b, t):
    """great-circle interpolation between two points of the same norm."""
    r = np.linalg.norm(a)
    ua, ub = a / r, b / np.linalg.norm(b)
    th = np.arccos(np.clip(ua @ ub, -1, 1))
    if th < 1e-6:
        return a.copy()
    return r * (np.sin((1 - t) * th) * ua + np.sin(t * th) * ub) / np.sin(th)


def random_point_at_angle(a, angle, rng):
    """a point on a's sphere, `angle` radians from a, in a uniformly random
    tangent direction."""
    r = np.linalg.norm(a)
    ua = a / r
    v = rng.standard_normal(a.shape)
    v -= (v @ ua) * ua
    v /= np.linalg.norm(v)
    return r * (np.cos(angle) * ua + np.sin(angle) * v)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="outputs/single_z_seeds_s005_5k")
    p.add_argument("--clip", required=True, help="stem, e.g. move-ego-0-2_4")
    p.add_argument("--n", type=int, default=9, help="waypoints per path, endpoints included")
    p.add_argument("--random", type=int, default=2, help="random control paths per best z")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=device_arg, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--envs", type=int, default=16)
    args = p.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    runs = []
    for d in sorted(root.iterdir()):
        m = re.match(rf"^{re.escape(args.clip)}_(\w+?)_(s\d+)$", d.name)
        if m and (d / "best_z.npy").exists():
            runs.append((m.group(2), d))
    if len(runs) < 2:
        raise SystemExit(f"need >= 2 seed runs for {args.clip} under {root}")
    tags = [t for t, _ in runs]
    Z = np.stack([np.load(d / "best_z.npy").reshape(-1).astype(np.float64) for _, d in runs])
    s0 = json.loads((runs[0][1] / "summary.json").read_text())
    assert s0["objective"] == "bfm", "this probe scores with the bfm loss; the runs must be bfm runs"
    task, stem = s0["clip"].split("/")
    xml = Path(s0["xml"])
    z0 = project_z(np.load(REPO_ROOT / "data" / "origin_z" / task / f"{stem}.npy").reshape(-1).astype(np.float64))
    ref = np.load(REPO_ROOT / "data" / s0["body"] / "retargeting_motion" / task / f"{stem}.npz")["qpos"]

    # --- the waypoints -----------------------------------------------------
    ts = np.linspace(0, 1, args.n)
    rng = np.random.default_rng(args.seed)
    paths = []      # (family, name, angle_deg, [z...])
    for i in range(len(Z)):
        for j in range(i + 1, len(Z)):
            ang = np.degrees(np.arccos(np.clip(Z[i] @ Z[j] / (np.linalg.norm(Z[i]) * np.linalg.norm(Z[j])), -1, 1)))
            paths.append(("pair", f"{tags[i]}->{tags[j]}", ang, [slerp(Z[i], Z[j], t) for t in ts]))
    for i in range(len(Z)):
        ang = np.degrees(np.arccos(np.clip(z0 @ Z[i] / (np.linalg.norm(z0) * np.linalg.norm(Z[i])), -1, 1)))
        paths.append(("from_z0", f"z0->{tags[i]}", ang, [slerp(z0, Z[i], t) for t in ts]))
    pair_angle = np.mean([a for f, _, a, _ in paths if f == "pair"])
    for i in range(len(Z)):
        for k in range(args.random):
            b = random_point_at_angle(Z[i], np.radians(pair_angle), rng)
            paths.append(("random", f"{tags[i]}->rand{k}", pair_angle, [slerp(Z[i], b, t) for t in ts]))
    allz = np.stack([z for _, _, _, zs in paths for z in zs])
    print(f"{args.clip}: {len(Z)} best z's, {len(paths)} paths x {args.n} waypoints = {len(allz)} rollouts "
          f"(pair angle mean {pair_angle:.1f} deg)")

    # --- roll them all out, score with the search's own loss ------------------
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig

    cfg = ESConfig(device=args.device)
    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device); model.eval()
    obs_mul = build_obs_multiplier(xml, REPO_ROOT / "assets/robots/adult/robot.xml",
                                   mode=s0.get("obs_scale", "auto"), parts=cfg.obs_scale_parts, verbose=False)
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    Bg = bfm_align.reference_embeddings(model, env1, ref, args.device, obs_mul)
    env1.close()
    env, _ = make_humenv(num_envs=args.envs, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")
    fk = mujoco.MjModel.from_xml_path(str(xml))
    init_qpos = ref[0] if s0.get("init", "reference") == "reference" else None

    costs = np.empty(len(allz)); mid_qpos = {}
    for b0 in range(0, len(allz), args.envs):
        zb = allz[b0:b0 + args.envs]
        pad = np.concatenate([zb, np.repeat(zb[-1:], args.envs - len(zb), axis=0)]) if len(zb) < args.envs else zb
        zt = torch.as_tensor(pad, dtype=torch.float32, device=args.device)
        q, o = rollout(model, env, zt, s0["steps"], args.device, obs_mul, init_qpos=init_qpos, nv=fk.nv, return_obs=True)
        costs[b0:b0 + len(zb)] = bfm_align.batch_bfm_align(model, o[:len(zb)], Bg, args.device)
        for k in range(len(zb)):
            mid_qpos[b0 + k] = q[k]
        print(f"  {min(b0 + args.envs, len(allz))}/{len(allz)}", flush=True)
    env.close()

    # --- table + figure -------------------------------------------------------
    rows, k = [], 0
    for fam, name, ang, zs in paths:
        c = costs[k:k + args.n]; k += args.n
        for t, v in zip(ts, c):
            rows.append(dict(family=fam, path=name, angle_deg=round(ang, 1), t=round(float(t), 3), bfm=float(v)))
    with open(root / f"z_flat_{args.clip}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    fam_ax = {"pair": ax[0], "from_z0": ax[1], "random": ax[2]}
    fam_title = {"pair": "best_i -> best_j (slerp)", "from_z0": "z0 -> best_i",
                 "random": f"best_i -> random point {pair_angle:.0f} deg away (control)"}
    k = 0
    for fam, name, ang, zs in paths:
        c = costs[k:k + args.n]; k += args.n
        fam_ax[fam].plot(ts, c, marker="o", ms=3, lw=1, label=f"{name} ({ang:.0f} deg)")
    for fam, a in fam_ax.items():
        a.set_title(fam_title[fam]); a.set_xlabel("t along the great circle"); a.grid(alpha=.3)
        a.legend(fontsize=6, ncol=2)
    ax[0].set_ylabel("1 - mean cos(B(s), B(g))")
    pair_c = np.array([costs[i * args.n:(i + 1) * args.n] for i, (f, *_ ) in enumerate(paths) if f == "pair"])
    rand_c = np.array([costs[i * args.n:(i + 1) * args.n] for i, (f, *_ ) in enumerate(paths) if f == "random"])
    fig.suptitle(f"{args.clip}: bfm cost along great circles between the {len(Z)} seeds' best z  --  "
                 f"pair paths: endpoints {pair_c[:, [0, -1]].mean():.3f}, worst interior {pair_c[:, 1:-1].max():.3f}, "
                 f"mean interior {pair_c[:, 1:-1].mean():.3f}   |   random control at same angle: {rand_c[:, -1].mean():.3f}")
    fig.tight_layout()
    fig.savefig(root / f"z_flat_{args.clip}.png", dpi=130)

    # contact sheet: the midpoint of every pair path, one frame at 50% of the clip
    from PIL import Image
    from rollout_z_trace import label
    renderer = mujoco.Renderer(fk, height=240, width=240); data = mujoco.MjData(fk)
    tiles = []
    tmid = len(ref) // 2
    for i, (fam, name, ang, zs) in enumerate(paths):
        if fam != "pair":
            continue
        q = mid_qpos[i * args.n + args.n // 2]
        data.qpos[:] = q[min(tmid, len(q) - 1)]; mujoco.mj_forward(fk, data)
        renderer.update_scene(data, camera="front_side")
        tiles.append(label(renderer.render().copy(), f"mid {name}", f"bfm {costs[i * args.n + args.n // 2]:.3f}"))
    data.qpos[:] = ref[tmid]; mujoco.mj_forward(fk, data); renderer.update_scene(data, camera="front_side")
    tiles.insert(0, label(renderer.render().copy(), "reference", f"frame {tmid}"))
    renderer.close()
    cols = 6
    rowsn = int(np.ceil(len(tiles) / cols)); blank = np.full_like(tiles[0], 252)
    sheet = np.concatenate([np.concatenate([tiles[r * cols + c] if r * cols + c < len(tiles) else blank
                                            for c in range(cols)], 1) for r in range(rowsn)], 0)
    Image.fromarray(sheet).save(root / f"z_flat_{args.clip}_midpoints.png")

    print(f"\npair paths   : endpoints mean {pair_c[:, [0, -1]].mean():.4f}   interior mean {pair_c[:, 1:-1].mean():.4f}"
          f"   interior worst {pair_c[:, 1:-1].max():.4f}")
    print(f"random paths : start {rand_c[:, 0].mean():.4f}   end (same angle as a pair) {rand_c[:, -1].mean():.4f}"
          f"   interior worst {rand_c[:, 1:-1].max():.4f}")
    print(f"-> {root / f'z_flat_{args.clip}.png'}\n-> {root / f'z_flat_{args.clip}.csv'}\n-> {root / f'z_flat_{args.clip}_midpoints.png'}")


if __name__ == "__main__":
    main()
