#!/usr/bin/env python3
"""PCA scatter of the learned cross-body z map, on the HELD-OUT bodies.

One panel per body. Three clouds in the same 2-D projection:

    adult      z_adult      where the frozen policy's latent starts
    target     z_body       where that body's own tracking inference puts it
    predicted  G(beta, z_adult)

The question the picture answers is whether the adapter moves the adult cloud
ONTO the target cloud, and the failure it is meant to expose is the one a
cosine average cannot: a map that lands in the right region on average while
collapsing the structure -- every clip mapped to the same point would still
score well against a per-frame mean.

The projection is fit ONCE, on the adult and target latents of the body in the
panel, and then applied to all three clouds. Fitting it separately per cloud
would let each pick its own axes and the overlap would be meaningless.

Usage (from project root):
    uv run scripts/zmap_pca_scatter.py
    uv run scripts/zmap_pca_scatter.py --ckpt outputs/simple_zmap/latest.pt \\
        --bodies giant short_stocky --out outputs/simple_zmap/pca_scatter.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from model.dataset import BETA_AXES, load_task_list
from model.networks import LatentAdapter


def load_body(dataset_dir: Path, rows, body, max_clips, rng):
    """(src, dst, beta) for one body, subsampled BY CLIP."""
    mine = [r for r in rows if r["morphology_label"] == body]
    if max_clips and len(mine) > max_clips:
        mine = [mine[i] for i in sorted(rng.choice(len(mine), max_clips, replace=False))]
    src, dst, clip = [], [], []
    for k, r in enumerate(mine):
        a = np.load(dataset_dir / r["infer_origin_z"]).astype(np.float32)
        b = np.load(dataset_dir / r["retarget_z"]).astype(np.float32)
        n = min(len(a), len(b))
        src.append(a[:n])
        dst.append(b[:n])
        clip.append(np.full(n, k))
    beta = json.loads((dataset_dir / mine[0]["morphology"]).read_text())
    return (np.concatenate(src), np.concatenate(dst), np.concatenate(clip),
            np.array([beta[a] for a in BETA_AXES], dtype=np.float32))


def pca_fit(X, k=2):
    """Plain PCA. Returns (mean, components (k, D))."""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    # SVD on the centred matrix -- no covariance matrix, D is 256 and N is large.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/simple_zmap/latest.pt")
    p.add_argument("--dataset", default=None, help="default: the checkpoint's own")
    p.add_argument("--bodies", nargs="*", default=None,
                   help="default: the held-out bodies the checkpoint records")
    p.add_argument("--max-clips", type=int, default=60,
                   help="clips per body to plot; frames within a clip are highly "
                        "correlated so this is the number that sets the picture")
    p.add_argument("--stride", type=int, default=3, help="plot every Nth frame")
    p.add_argument("--out", default="outputs/simple_zmap/pca_scatter.png")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ck = torch.load(ROOT / a.ckpt if not Path(a.ckpt).is_absolute() else a.ckpt,
                    map_location=a.device, weights_only=False)
    cfg = ck["cfg"]
    dataset_dir = ROOT / (a.dataset or cfg.dataset_dir)
    bodies = a.bodies or ck.get("test_bodies") or load_task_list(
        dataset_dir / "splits" / "test_bodies.txt")

    rows = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text().splitlines() if l]
    rng = np.random.default_rng(a.seed)

    adapter = LatentAdapter(
        beta_dim=len(BETA_AXES), z_dim=256, hidden_dims=cfg.adapter_hidden_dims,
        alpha=cfg.adapter_alpha, alpha_learnable=cfg.adapter_alpha_learnable,
        project=cfg.adapter_project_z).to(a.device)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()

    fig, axes = plt.subplots(1, len(bodies), figsize=(6.4 * len(bodies), 6.0), squeeze=False)
    for ax, body in zip(axes[0], bodies):
        src, dst, clip, beta = load_body(dataset_dir, rows, body, a.max_clips, rng)
        with torch.no_grad():
            pred = adapter(torch.from_numpy(beta).expand(len(src), -1).to(a.device),
                           torch.from_numpy(src).to(a.device)).cpu().numpy()

        # Fit on adult+target only: the projection must not be chosen to flatter
        # the prediction.
        mu, W = pca_fit(np.concatenate([src, dst]))
        proj = lambda X: (X - mu) @ W.T
        s, d, q = proj(src)[::a.stride], proj(dst)[::a.stride], proj(pred)[::a.stride]

        cos_id = float((F.normalize(torch.from_numpy(src), dim=-1)
                        * F.normalize(torch.from_numpy(dst), dim=-1)).sum(-1).mean())
        cos_ad = float((F.normalize(torch.from_numpy(pred), dim=-1)
                        * F.normalize(torch.from_numpy(dst), dim=-1)).sum(-1).mean())

        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.30, c="#888888", label="adult  z_adult")
        ax.scatter(d[:, 0], d[:, 1], s=4, alpha=0.30, c="#1f77b4", label="target z_body")
        ax.scatter(q[:, 0], q[:, 1], s=4, alpha=0.30, c="#d62728", label="predicted G(beta, z_adult)")
        ax.set_title(f"{body}  (held out)\ncos: identity {cos_id:.3f} -> adapter {cos_ad:.3f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=3, fontsize=9, loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        print(f"{body:14s} clips={clip.max() + 1:3d} frames={len(src):6d}  "
              f"cos identity {cos_id:.4f} -> adapter {cos_ad:.4f}")

    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
