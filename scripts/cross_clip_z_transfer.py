#!/usr/bin/env python3
"""cross_clip_z_transfer.py -- roll clip i's searched best_z on clip j.

The ground truth behind scripts/analyze_z_targets.py. That script measures
cosines between correction directions, which is PESSIMISTIC: the plateau study
put the low-cost set around one clip at ~32-64 dimensions, so two z far apart in
angle can both work. Only the simulator can say whether one z serves several
clips.

Every cell (i, j) is scored exactly as training scores it -- bfm cost, reference
init, the same torque XML -- and reported relative to clip j's OWN z0 cost, so
the diagonal is "what a dedicated search achieves" and everything else is "what
borrowing someone else's answer achieves".

  off-diagonal ~ diagonal -> one z serves many clips; a map exists and the
                             problem is that ES cannot find it
  off-diagonal ~ 1 or worse -> each clip needs its own z; z0 -> z is not a
                             function a shared adapter can represent

Usage:
    uv run scripts/cross_clip_z_transfer.py --task headstand --clips 0 2 3 4 9 15 20 23 30 34
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="headstand")
    p.add_argument("--clips", type=int, nargs="+",
                   default=[0, 2, 3, 4, 9, 15, 20, 23, 30, 34])
    p.add_argument("--floor-dir", default="outputs/single_z_floor")
    p.add_argument("--xml", default="assets/robots_torque/child/robot_torque_full.xml")
    p.add_argument("--data-dir", default="data/child/retargeting_motion")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", type=device_arg,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--out", default="outputs/z_target_analysis")
    args = p.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    K = args.clips
    n = len(K)

    Z = []
    for k in K:
        f = REPO_ROOT / args.floor_dir / f"{args.task}_{k}_bfm_s0" / "best_z.npy"
        if not f.exists():
            raise SystemExit(f"no searched target for {args.task}_{k}: {f}")
        Z.append(project_z(np.load(f).reshape(-1).astype(np.float64)))
    Z0 = [project_z(np.load(REPO_ROOT / "data/origin_z" / args.task /
                            f"{args.task}_{k}.npy").reshape(-1).astype(np.float64)) for k in K]
    refs = [np.load(REPO_ROOT / args.data_dir / args.task / f"{args.task}_{k}.npz")["qpos"]
            for k in K]

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
    # one env slot per target clip: every batched rollout scores the same donor z
    # against all n clips at once, so the slot layout IS the clip order.
    env, _ = make_humenv(num_envs=n, vectorization_mode="async", task=None,
                         xml=str(xml), state_init="Default")

    # Bg[j] is clip j's reference embedding -- the thing every z is scored against
    Bg = [bfm_align.reference_embeddings(model, env1, refs[j], args.device, obs_mul)
          for j in range(n)]

    # One batched rollout per DONOR z: all n target clips at once, each slot
    # starting from its own reference frame 0.
    C = np.full((n, n), np.nan)      # C[i, j] = cost of z_i on clip j
    C0 = np.full(n, np.nan)          # clip j under its own z0
    init = [r[0] for r in refs]

    def score(zrow):
        obs = rollout_multi(model, env, torch.as_tensor(zrow, dtype=torch.float32,
                                                        device=args.device),
                            args.steps, args.device, obs_mul, init, nv)
        return [bfm_align.batch_bfm_align(model, obs[j:j+1], Bg[j], args.device)[0]
                for j in range(n)]

    C0[:] = score(np.stack(Z0))
    print(f"  {'z0':16s} " + " ".join(f"{x:.3f}" for x in C0), flush=True)
    for i in range(n):
        C[i] = score(np.repeat(Z[i][None], n, 0))
        print(f"  {f'best_z[{K[i]}]':16s} " + " ".join(f"{x:.3f}" for x in C[i]), flush=True)
    env.close(); env1.close()

    R = C / C0[None, :]
    np.savez(out / f"{args.task}_transfer.npz", clips=K, cost=C, cost_z0=C0, ratio=R)
    diag = np.array([R[i, i] for i in range(n)])
    off = R[~np.eye(n, dtype=bool)]
    print(f"\nratio to each clip's own z0 cost:")
    print(f"  diagonal  (its own searched z) : mean {diag.mean():.3f}  median {np.median(diag):.3f}")
    print(f"  off-diag  (someone else's z)   : mean {off.mean():.3f}  median {np.median(off):.3f}")
    print(f"  off-diagonal cells that still beat z0: {(off < 1).sum()}/{off.size}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6),
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor(SURF)
    v = np.clip(R, 0, 2)
    im = ax[0].imshow(v, cmap="RdYlGn_r", vmin=0, vmax=2)
    ax[0].set_xticks(range(n)); ax[0].set_xticklabels(K, fontsize=9)
    ax[0].set_yticks(range(n)); ax[0].set_yticklabels([f"best_z[{k}]" for k in K], fontsize=9)
    ax[0].set_xlabel("rolled out on clip", fontsize=10, color=INK2)
    ax[0].set_ylabel("z searched for clip", fontsize=10, color=INK2)
    for i in range(n):
        for j in range(n):
            ax[0].text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=7.6,
                       color="#000000" if .35 < v[i,j] < 1.5 else "#ffffff")
        ax[0].add_patch(plt.Rectangle((i-.5, i-.5), 1, 1, fill=False, lw=2.2, ec=INK))
    fig.colorbar(im, ax=ax[0], label="cost / that clip's z0 cost", fraction=.046)
    ax[0].set_title("one clip's answer, used on another", fontsize=11.5, color=INK)

    ax[1].set_facecolor(SURF)
    ax[1].hist(off, bins=np.linspace(0, 2.2, 34), color="#eb6834", alpha=.85,
               label=f"someone else's z (n={off.size})")
    ax[1].hist(diag, bins=np.linspace(0, 2.2, 34), color="#1baf7a", alpha=.9,
               label=f"its own searched z (n={n})")
    ax[1].axvline(1.0, color=INK2, ls=(0, (4, 3)), lw=1.3)
    ax[1].text(1.02, ax[1].get_ylim()[1]*.95, " z0", color=INK2, fontsize=9.5, va="top")
    ax[1].set_xlabel("cost / that clip's z0 cost", fontsize=10, color=INK2)
    ax[1].set_ylabel("cells", fontsize=10, color=INK)
    ax[1].grid(alpha=.22, color=INK3, lw=.7); ax[1].set_axisbelow(True)
    for sp in ("top", "right"): ax[1].spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax[1].spines[sp].set_color(INK3)
    ax[1].legend(fontsize=9)
    ax[1].set_title("(c) does one z serve several clips?", fontsize=11.5, color=INK)
    fig.suptitle(f"cross-clip transfer of searched z  --  {args.task}, {n} clips, "
                 f"bfm cost, reference init", fontsize=12.5, color=INK, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .94))
    f = out / f"{args.task}_cross_clip_transfer.png"
    fig.savefig(f, dpi=140, facecolor=SURF)
    print(f"-> {f}")


if __name__ == "__main__":
    main()
