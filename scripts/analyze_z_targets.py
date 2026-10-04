#!/usr/bin/env python3
"""analyze_z_targets.py -- is the per-clip correction a FUNCTION of z0?

exp5 measured that a headstand-only adapter learns one global correction: the 40
clips' correction directions have mean pairwise cos +0.973, where independent
directions in 255-d would be ~0. That is what ES converges to, because it
averages its gradient over the batch and each clip's own component cancels.

This asks whether that is the estimator's fault or the problem's. Given a
per-clip searched best_z for every clip:

  (a) geometry   -- are the TARGET corrections mutually aligned too? If they are,
                    the single-direction solution was right and something else is
                    wrong. If they are near-orthogonal, no single direction can
                    work and ES was always going to fail here.
  (b) k-NN       -- does the nearest clip in z0 space have a similar correction?
                    Non-parametric, nothing to memorize with: a direct test of
                    "is the correction a smooth function of z0".
  (d) MLP fit    -- fit the real LatentAdapter to the targets and read HELD-OUT
                    error against those baselines. With 32 training points and
                    659k parameters the training fit proves nothing; only the
                    held-out number, and only relative to k-NN, means anything.

Caveat that (c), the rollout transfer, exists to cover: cosine-to-target is
pessimistic. The plateau study measured the low-cost set around a clip at ~32-64
dimensions, so two z that are far apart in angle can both be good. A low cosine
here is evidence of no smooth structure; it is not proof that a map cannot work.
scripts/cross_clip_z_transfer.py is the ground truth.

Usage:
    uv run scripts/analyze_z_targets.py --task headstand --n 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT not in sys.path:
    sys.path.append(PARENT)

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HELD = {0, 10, 12, 17, 19, 23, 26, 27}      # exp5's held-out split, reused verbatim

SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA, RED, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#b8b7b0"


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def tangent_dir(z0h, zt):
    """the correction direction: where z0 has to move, in z0's tangent plane."""
    d = zt - (zt @ z0h) * z0h
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.zeros_like(d)


def load(task, n, floor_dir):
    Z0, ZT, ok, ang, ratio = [], [], [], [], []
    for k in range(n):
        d = REPO_ROOT / floor_dir / f"{task}_{k}_bfm_s0"
        zp = REPO_ROOT / "data/origin_z" / task / f"{task}_{k}.npy"
        if not (d / "best_z.npy").exists() or not zp.exists():
            continue
        z0 = unit(np.load(zp).reshape(-1).astype(np.float64))
        zt = unit(np.load(d / "best_z.npy").reshape(-1).astype(np.float64))
        s = json.loads((d / "summary.json").read_text())
        Z0.append(z0); ZT.append(zt); ok.append(k)
        ang.append(np.degrees(np.arccos(np.clip(z0 @ zt, -1, 1))))
        ratio.append(s["best"]["cost"] / s["origin_z"]["cost"])
    return np.stack(Z0), np.stack(ZT), ok, np.array(ang), np.array(ratio)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="headstand")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--floor-dir", default="outputs/single_z_floor")
    p.add_argument("--adapter-ckpt",
                   default="outputs/simple_es/child_balanced/exp5_headstand_only/update_00600.pt")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--out", default="outputs/z_target_analysis")
    args = p.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    Z0, ZT, ok, ang, ratio = load(args.task, args.n, args.floor_dir)
    N = len(ok)
    print(f"{N} clips with a searched target for '{args.task}'")
    print(f"  best_z angle from z0: mean {ang.mean():.1f} deg  min {ang.min():.1f}  max {ang.max():.1f}")
    print(f"  search ratio best/z0: mean {ratio.mean():.3f}  median {np.median(ratio):.3f}  "
          f"max {ratio.max():.3f}")

    D = np.stack([tangent_dir(Z0[i], ZT[i]) for i in range(N)])
    iu = np.triu_indices(N, 1)
    Ct = (D @ D.T)[iu]

    # the adapter's own correction directions, for the side-by-side
    Ca = None
    ck_path = REPO_ROOT / args.adapter_ckpt
    if ck_path.exists():
        import torch
        from model.networks import LatentAdapter
        from model.dataset import CrossEmbodimentDataset
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        ds = CrossEmbodimentDataset(REPO_ROOT / cfg.dataset_dir)
        beta = torch.tensor(ds[ds.indices_by_body()["child"][0]]["beta"], dtype=torch.float32)[None]
        ad = LatentAdapter(beta_dim=beta.shape[1], z_dim=256, hidden_dims=cfg.adapter_hidden_dims,
                           alpha=cfg.adapter_alpha, alpha_learnable=cfg.adapter_alpha_learnable,
                           project=cfg.adapter_project_z, residual=cfg.adapter_residual,
                           head=getattr(cfg, "adapter_head", "residual"),
                           theta_max_deg=getattr(cfg, "adapter_theta_max_deg", 60.0))
        ad.load_state_dict(ck["adapter"]); ad.eval()
        with torch.no_grad():
            zb = ad(beta.repeat(N, 1), torch.tensor(Z0 * 16.0, dtype=torch.float32)).numpy()
        zb = unit(zb.astype(np.float64))
        A = np.stack([tangent_dir(Z0[i], zb[i]) for i in range(N)])
        Ca = (A @ A.T)[iu]
        # how well does the adapter's direction match the target's?
        match_ad = np.array([float(A[i] @ D[i]) for i in range(N)])
        print(f"\n(a) TARGET correction directions: mean pairwise cos {Ct.mean():+.3f} "
              f"[{Ct.min():+.3f}, {Ct.max():+.3f}]")
        print(f"    ADAPTER correction directions: mean pairwise cos {Ca.mean():+.3f} "
              f"[{Ca.min():+.3f}, {Ca.max():+.3f}]")
        print(f"    adapter direction vs its clip's target: mean cos {match_ad.mean():+.3f}")
    else:
        match_ad = None
        print(f"\n(a) TARGET correction directions: mean pairwise cos {Ct.mean():+.3f} "
              f"[{Ct.min():+.3f}, {Ct.max():+.3f}]")

    # ---- (b) k-NN in z0 space -------------------------------------------------
    tr = [i for i, k in enumerate(ok) if k not in HELD]
    te = [i for i, k in enumerate(ok) if k in HELD]
    zang = np.degrees(np.arccos(np.clip(Z0 @ Z0.T, -1, 1)))
    gmean = unit(D[tr].mean(0))

    def knn_pred(i, pool):
        j = min([q for q in pool if q != i], key=lambda q: zang[i, q])
        return D[j], j

    rows = []
    rng = np.random.default_rng(0)
    for name, idxs in (("train", tr), ("held out", te)):
        knn = np.array([float(knn_pred(i, tr)[0] @ D[i]) for i in idxs])
        glb = np.array([float(gmean @ D[i]) for i in idxs])
        rnd = np.array([float(D[rng.choice([q for q in tr if q != i])] @ D[i]) for i in idxs])
        rows.append((name, knn, glb, rnd))
        print(f"\n(b) {name} ({len(idxs)} clips) -- cos(predicted correction, true correction)")
        print(f"    nearest clip in z0 : {knn.mean():+.3f}  (median {np.median(knn):+.3f})")
        print(f"    global mean        : {glb.mean():+.3f}")
        print(f"    a random other clip: {rnd.mean():+.3f}")
    dnn = np.array([zang[i, knn_pred(i, tr)[1]] for i in range(N)])
    knn_all = np.array([float(knn_pred(i, tr)[0] @ D[i]) for i in range(N)])
    print(f"\n    corr(distance to nearest clip, cos) = {np.corrcoef(dnn, knn_all)[0,1]:+.3f}")

    # ---- (d) fit the real adapter --------------------------------------------
    import torch
    import torch.nn.functional as F
    from model.networks import LatentAdapter
    torch.manual_seed(0)
    zt = torch.tensor(Z0 * 16.0, dtype=torch.float32)
    tgt = torch.tensor(ZT, dtype=torch.float32)
    bz = torch.zeros(N, 1)
    net = LatentAdapter(beta_dim=1, z_dim=256, hidden_dims=[256, 512, 512, 256],
                        alpha=1.0, project=True, residual=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    itr, tr_c, te_c = [], [], []
    for ep in range(args.epochs + 1):
        opt.zero_grad()
        pred = net(bz[tr], zt[tr])
        loss = (1 - F.cosine_similarity(pred, tgt[tr], dim=-1)).mean()
        loss.backward(); opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                a = F.cosine_similarity(net(bz[tr], zt[tr]), tgt[tr], dim=-1).mean().item()
                b = F.cosine_similarity(net(bz[te], zt[te]), tgt[te], dim=-1).mean().item()
            itr.append(ep); tr_c.append(a); te_c.append(b)
    ident_tr = float(np.mean([Z0[i] @ ZT[i] for i in tr]))
    ident_te = float(np.mean([Z0[i] @ ZT[i] for i in te]))
    print(f"\n(d) LatentAdapter fit to the targets ({args.epochs} epochs, {len(tr)} train / {len(te)} held out)")
    print(f"    cos(pred, target)  train {tr_c[-1]:+.3f}   held out {te_c[-1]:+.3f}")
    print(f"    identity (z0)      train {ident_tr:+.3f}   held out {ident_te:+.3f}")
    print(f"    NOTE the identity baseline is high because best_z is only "
          f"{ang.mean():.0f} deg from z0 on average;")
    print(f"    what matters is whether the fit beats it, and whether it beats k-NN's direction.")

    json.dump(dict(task=args.task, n=N, clips=ok,
                   target_pair_cos=float(Ct.mean()),
                   adapter_pair_cos=(float(Ca.mean()) if Ca is not None else None),
                   knn_train=float(rows[0][1].mean()), knn_held=float(rows[1][1].mean()),
                   global_train=float(rows[0][2].mean()), global_held=float(rows[1][2].mean()),
                   random_train=float(rows[0][3].mean()), random_held=float(rows[1][3].mean()),
                   fit_train=tr_c[-1], fit_held=te_c[-1],
                   identity_train=ident_tr, identity_held=ident_te),
              open(out / f"{args.task}_summary.json", "w"), indent=1)

    # ---- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
    fig.patch.set_facecolor(SURF)
    for a in ax:
        a.set_facecolor(SURF); a.grid(alpha=.22, color=INK3, lw=.7); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_color(INK3)

    bins = np.linspace(-1, 1, 41)
    ax[0].hist(Ct, bins=bins, color=BLUE, alpha=.85, label="targets (searched)")
    if Ca is not None:
        ax[0].hist(Ca, bins=bins, color=ORANGE, alpha=.8, label="what the adapter learned")
    ax[0].axvline(0, color=INK2, ls=(0, (4, 3)), lw=1.2)
    ax[0].text(0.02, .97, " 0 = independent\n     directions", transform=ax[0].transAxes,
               fontsize=8.5, color=INK2, va="top")
    ax[0].set_xlabel("cos between two clips' correction directions", fontsize=9.5, color=INK2)
    ax[0].set_ylabel("pairs", fontsize=10, color=INK)
    ax[0].set_title("(a) are the corrections the same direction?", fontsize=11, color=INK)
    ax[0].legend(fontsize=8.5)

    ax[1].scatter(dnn, knn_all, s=34, color=BLUE, edgecolor=SURF, lw=.8)
    ax[1].axhline(0, color=INK2, ls=(0, (4, 3)), lw=1.2)
    ax[1].set_xlabel("angle to the nearest training clip's z0 (deg)", fontsize=9.5, color=INK2)
    ax[1].set_ylabel("cos(its correction, mine)", fontsize=10, color=INK)
    ax[1].set_title("(b) does a nearby z0 predict the correction?", fontsize=11, color=INK)

    labs = ["nearest\nclip", "global\nmean", "random\nclip"]
    x = np.arange(3); w = .36
    ax[2].bar(x - w/2, [rows[0][i+1].mean() for i in range(3)], w, color=NEUT, label="train")
    ax[2].bar(x + w/2, [rows[1][i+1].mean() for i in range(3)], w, color=BLUE, label="held out")
    ax[2].axhline(0, color=INK2, lw=1)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labs, fontsize=9, color=INK2)
    ax[2].set_ylabel("cos(predicted, true) correction", fontsize=10, color=INK)
    ax[2].set_title("(b) non-parametric predictors", fontsize=11, color=INK)
    ax[2].legend(fontsize=8.5)

    ax[3].plot(itr, tr_c, color=NEUT, lw=1.8, label="fit, train")
    ax[3].plot(itr, te_c, color=BLUE, lw=1.8, label="fit, held out")
    ax[3].axhline(ident_te, color=AQUA, ls=(0, (5, 3)), lw=1.6, label="identity (do nothing), held out")
    ax[3].axhline(rows[1][1].mean(), color=RED, ls=(0, (2, 2)), lw=1.6, label="nearest clip, held out")
    ax[3].set_xlabel("epoch", fontsize=9.5, color=INK2)
    ax[3].set_ylabel("cos(pred, target)", fontsize=10, color=INK)
    ax[3].set_title("(d) fitting the real adapter to the targets", fontsize=11, color=INK)
    ax[3].legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Is the per-clip correction a function of z0?   {args.task}, {N} searched targets",
                 fontsize=12.5, color=INK, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .945))
    f = out / f"{args.task}_target_structure.png"
    fig.savefig(f, dpi=140, facecolor=SURF)
    print(f"\n-> {f}")


if __name__ == "__main__":
    main()
