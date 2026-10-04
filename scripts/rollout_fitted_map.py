#!/usr/bin/env python3
"""rollout_fitted_map.py -- does a SUPERVISED map from z0 to the searched best_z
generalize, measured by the simulator instead of by cosine?

scripts/analyze_z_targets.py fits the real LatentAdapter to per-clip search
targets and reports cos(pred, target). That metric is pessimistic: the low-cost
plateau around a clip is ~32-64 dimensional, so a prediction can land well
inside it and still have a poor cosine to the one point the search happened to
stop at. Cosine therefore cannot answer "does the map generalize" -- only a
rollout can.

Four z per clip, all scored identically (bfm, reference init, the training XML),
all reported as cost / that clip's own z0 cost:

  z0            1.0 by construction -- the do-nothing baseline
  searched      the per-clip target itself; reproduces the search floor and
                confirms the scoring path matches the one that produced it
  fitted MLP    the supervised map's output. TRAIN clips say whether a target it
                memorized (cos 1.000) actually works when rolled out; HELD-OUT
                clips are the question this script exists for.
  ES adapter    what train_es actually learned, for the side by side

Read the held-out fitted row. Clearly below 1 -> a learnable map exists and the
bottleneck is that ES estimates each cell's gradient by probing; per-cell
measured targets (train_es.py --lambda-bc) are then the right move. At or above
1 -> even exact gradients onto perfect targets do not transfer, and the model's
input or the framing has to change instead.

Usage:
    uv run scripts/rollout_fitted_map.py --task headstand --n 40
"""

from __future__ import annotations

import argparse
import json
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
import torch.nn.functional as F

from single_z_search import device_arg, project_z
from rank_initial_cost import rollout_multi

REPO_ROOT = Path(__file__).resolve().parent.parent
HELD = {0, 10, 12, 17, 19, 23, 26, 27}      # exp5's split, reused verbatim
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
NEUT, BLUE, AQUA, ORANGE = "#b8b7b0", "#2a78d6", "#1baf7a", "#eb6834"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="headstand")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--floor-dir", default="outputs/single_z_floor")
    p.add_argument("--es-ckpt",
                   default="outputs/simple_es/child_balanced/exp5_headstand_only/update_00600.pt")
    p.add_argument("--xml", default="assets/robots_torque/child/robot_torque_full.xml")
    p.add_argument("--data-dir", default="data/child/retargeting_motion")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--device", type=device_arg,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--out", default="outputs/z_target_analysis")
    args = p.parse_args()
    out = REPO_ROOT / args.out; out.mkdir(parents=True, exist_ok=True)

    K, Z0, ZT = [], [], []
    for k in range(args.n):
        f = REPO_ROOT / args.floor_dir / f"{args.task}_{k}_bfm_s0" / "best_z.npy"
        zp = REPO_ROOT / "data/origin_z" / args.task / f"{args.task}_{k}.npy"
        if not (f.exists() and zp.exists()):
            continue
        K.append(k)
        Z0.append(project_z(np.load(zp).reshape(-1).astype(np.float64)))
        ZT.append(project_z(np.load(f).reshape(-1).astype(np.float64)))
    Z0 = np.stack(Z0); ZT = np.stack(ZT); N = len(K)
    tr = [i for i, k in enumerate(K) if k not in HELD]
    te = [i for i, k in enumerate(K) if k in HELD]
    print(f"{N} clips, {len(tr)} train / {len(te)} held out")

    # ---- supervised fit, identical to analyze_z_targets' (d) ------------------
    from model.networks import LatentAdapter
    torch.manual_seed(0)
    zin = torch.tensor(Z0, dtype=torch.float32)
    tgt = torch.tensor(ZT, dtype=torch.float32)
    bz = torch.zeros(N, 1)
    net = LatentAdapter(beta_dim=1, z_dim=256, hidden_dims=[256, 512, 512, 256],
                        alpha=1.0, project=True, residual=True)
    opt = (torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=args.weight_decay)
           if args.weight_decay > 0 else torch.optim.Adam(net.parameters(), lr=1e-3))
    for _ in range(args.epochs):
        opt.zero_grad()
        (1 - F.cosine_similarity(net(bz[tr], zin[tr]), tgt[tr], dim=-1)).mean().backward()
        opt.step()
    with torch.no_grad():
        ZF = net(bz, zin).numpy().astype(np.float64)
        ct = F.cosine_similarity(torch.tensor(ZF, dtype=torch.float32), tgt, dim=-1).numpy()
    print(f"fit cos(pred, target): train {ct[tr].mean():+.3f}  held out {ct[te].mean():+.3f}")

    # ---- what train_es actually learned ---------------------------------------
    ck = torch.load(REPO_ROOT / args.es_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    from model.dataset import CrossEmbodimentDataset
    ds = CrossEmbodimentDataset(REPO_ROOT / cfg.dataset_dir)
    beta = torch.tensor(ds[ds.indices_by_body()["child"][0]]["beta"], dtype=torch.float32)[None]
    es = LatentAdapter(beta_dim=beta.shape[1], z_dim=256, hidden_dims=cfg.adapter_hidden_dims,
                       alpha=cfg.adapter_alpha, alpha_learnable=cfg.adapter_alpha_learnable,
                       project=cfg.adapter_project_z, residual=cfg.adapter_residual,
                       head=getattr(cfg, "adapter_head", "residual"),
                       theta_max_deg=getattr(cfg, "adapter_theta_max_deg", 60.0))
    es.load_state_dict(ck["adapter"]); es.eval()
    with torch.no_grad():
        ZE = es(beta.repeat(N, 1), zin).numpy().astype(np.float64)

    # ---- roll all four out -----------------------------------------------------
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    import mujoco

    cfg_r = ESConfig(device=args.device)
    xml = REPO_ROOT / args.xml
    obs_mul = build_obs_multiplier(xml, REPO_ROOT / "assets/robots/adult/robot.xml",
                                   mode="auto", parts=cfg_r.obs_scale_parts, verbose=False)
    nv = mujoco.MjModel.from_xml_path(str(xml)).nv
    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device); model.eval()
    refs = [np.load(REPO_ROOT / args.data_dir / args.task / f"{args.task}_{k}.npz")["qpos"]
            for k in K]
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    Bg = [bfm_align.reference_embeddings(model, env1, r, args.device, obs_mul) for r in refs]
    C = args.chunk
    env, _ = make_humenv(num_envs=C, vectorization_mode="async", task=None,
                         xml=str(xml), state_init="Default")

    def score(Z):
        c = np.zeros(N)
        for a in range(0, N, C):
            idx = list(range(a, min(a + C, N)))
            pad = idx + [idx[-1]] * (C - len(idx))
            obs = rollout_multi(model, env,
                                torch.as_tensor(project_z(Z[pad]), dtype=torch.float32,
                                                device=args.device),
                                args.steps, args.device, obs_mul,
                                [refs[j][0] for j in pad], nv)
            for s, j in enumerate(idx):
                c[j] = bfm_align.batch_bfm_align(model, obs[s:s+1], Bg[j], args.device)[0]
        return c

    res = {}
    for name, Z in (("z0", Z0), ("searched", ZT), ("fitted MLP", ZF), ("ES adapter", ZE)):
        res[name] = score(Z)
        print(f"  scored {name}", flush=True)
    env.close(); env1.close()

    R = {k: v / res["z0"] for k, v in res.items()}
    print(f"\ncost / that clip's z0 cost\n")
    print(f"{'':14s} {'train (32)':>22s}   {'HELD OUT (8)':>22s}")
    print(f"{'':14s} {'mean':>7s} {'median':>7s} {'<1':>5s}   {'mean':>7s} {'median':>7s} {'<1':>5s}")
    for name in ("z0", "searched", "fitted MLP", "ES adapter"):
        r = R[name]
        cells = []
        for idxs in (tr, te):
            v = r[idxs]
            cells.append(f"{v.mean():7.3f} {np.median(v):7.3f} {int((v<1).sum()):3d}/{len(v):<2d}")
        print(f"{name:14s} {cells[0]}   {cells[1]}")

    np.savez(out / f"{args.task}_fitted_map.npz", clips=K, **{k.replace(" ", "_"): v
                                                              for k, v in res.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.patch.set_facecolor(SURF)
    names = ["searched", "fitted MLP", "ES adapter"]
    cols = {"searched": AQUA, "fitted MLP": BLUE, "ES adapter": ORANGE}
    order = sorted(range(N), key=lambda i: R["searched"][i])
    x = np.arange(N); w = .27
    for s, nm in enumerate(names):
        ax[0].bar(x + (s - 1) * w, np.clip([R[nm][i] for i in order], 0, 3),
                  w, color=cols[nm], label=nm)
    ax[0].axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.3)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{K[i]}{'*' if K[i] in HELD else ''}" for i in order], fontsize=7.2)
    ax[0].set_xlabel(f"{args.task} trial   (* = held out of the fit)", fontsize=9.5, color=INK2)
    ax[0].set_ylabel("cost / cost of z0", fontsize=10.5, color=INK)
    ax[0].legend(fontsize=9)
    ax[0].set_title("per clip", fontsize=11.5, color=INK)

    grp = [("train", tr), ("held out", te)]
    xx = np.arange(2); w2 = .26
    for s, nm in enumerate(names):
        ax[1].bar(xx + (s - 1) * w2, [np.median(R[nm][g]) for _, g in grp], w2,
                  color=cols[nm], label=nm)
    ax[1].axhline(1, color=INK2, ls=(0, (4, 3)), lw=1.3)
    ax[1].text(1.52, 1.0, " z0", color=INK2, fontsize=10, va="center")
    ax[1].set_xticks(xx); ax[1].set_xticklabels([g for g, _ in grp], fontsize=10, color=INK2)
    ax[1].set_ylabel("median cost / cost of z0", fontsize=10.5, color=INK)
    ax[1].set_title("does the supervised map generalize?", fontsize=11.5, color=INK)
    ax[1].legend(fontsize=9)
    for a in ax:
        a.set_facecolor(SURF); a.grid(axis="y", alpha=.22, color=INK3, lw=.7); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_color(INK3)
    fig.suptitle(f"A supervised map fit to per-clip searched targets, judged by ROLLOUT "
                 f"-- {args.task}, {N} clips", fontsize=12.5, color=INK, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .94))
    f = out / f"{args.task}_fitted_map_rollout.png"
    fig.savefig(f, dpi=140, facecolor=SURF)
    print(f"\n-> {f}")


if __name__ == "__main__":
    main()
