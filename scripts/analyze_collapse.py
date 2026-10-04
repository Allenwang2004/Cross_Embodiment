#!/usr/bin/env python3
"""analyze_collapse.py -- does a trained adapter apply ONE correction to every clip?

For each checkpoint, every clip's correction direction is where the adapter moves
z0, in z0's tangent plane: tangent(z0, G(beta, z0)). If the adapter learned a
per-clip map, these point different ways (pairwise cos ~0 for 255-d directions).
If it collapsed to one global shift, they all point the same way (cos ~1).

Reported per checkpoint, for all clips and for one task (default headstand):
  pair   mean pairwise cos of the adapter's correction directions
  pc1    fraction of their spread along the top principal direction (1 = one direction)
  ang    mean angle the adapter moves z0 (deg)
and, where the checkpoint carries a best-point buffer (lambda_bc runs), the same
numbers for the buffer's targets plus the per-clip match cos(adapter, target).

Usage:
    uv run scripts/analyze_collapse.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT not in sys.path:
    sys.path.append(PARENT)

import numpy as np
import torch

from model.dataset import CrossEmbodimentDataset
from model.networks import LatentAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = [
    ("ES, headstand only", "outputs/simple_es/child_balanced/exp5_headstand_only/update_00600.pt"),
    ("ES, all tasks", "outputs/simple_es/child_balanced/lr3e-4_pairs16/update_00600.pt"),
    ("bc 0", "outputs/simple_es/child_memorize/bc0/update_01000.pt"),
    ("bc 0.1", "outputs/simple_es/child_memorize/bc0.1/update_01000.pt"),
    ("bc 0.3", "outputs/simple_es/child_memorize/bc0.3/update_01000.pt"),
    ("bc 1.0", "outputs/simple_es/child_memorize/bc1.0/update_01000.pt"),
    ("bc 3.0", "outputs/simple_es/child_memorize/bc3.0/update_01000.pt"),
    ("bc 10.0", "outputs/simple_es/child_memorize/bc10.0/update_01000.pt"),
    ("bc 1.0 @150", "outputs/simple_es/child_memorize/bc1.0/update_00150.pt"),
    ("4 heads @150", "outputs/simple_es/child_memorize/h4/update_00150.pt"),
    ("8 heads @150", "outputs/simple_es/child_memorize/h8/update_00150.pt"),
]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def tangent(z0h, z):
    d = z - (z @ z0h) * z0h
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def spread(D):
    """mean pairwise cos, and top-PC share of the (uncentred) direction spread."""
    if len(D) < 2:
        return float("nan"), float("nan")
    C = D @ D.T
    iu = np.triu_indices(len(D), 1)
    s = np.linalg.svd(D, compute_uv=False) ** 2
    return float(C[iu].mean()), float(s[0] / s.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="headstand")
    args = p.parse_args()

    ds_cache = {}
    print(f"{'run':<20} {'set':<10} {'n':>4}  {'pair':>6} {'pc1':>5} {'ang':>5}  "
          f"{'tgt pair':>8} {'tgt pc1':>7} {'tgt ang':>7}  {'match':>6}")
    for name, rel in RUNS:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"{name:<20} missing {rel}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if cfg.dataset_dir not in ds_cache:
            ds = CrossEmbodimentDataset(REPO_ROOT / cfg.dataset_dir)
            ds_cache[cfg.dataset_dir] = [ds[i] for i in ds.indices_by_body()["child"]]
        samples = ds_cache[cfg.dataset_dir]
        buf = ck.get("best_buf")
        head_sel = ck.get("head_sel") or {}
        if buf is not None:
            samples = [s for s in samples if f"{s['reward_name']}|{s['trial']}|child" in buf]
        elif name.endswith("headstand only"):
            samples = [s for s in samples if s["reward_name"].startswith(args.task)]

        H = getattr(cfg, "adapter_heads", 1)
        ad = LatentAdapter(beta_dim=len(samples[0]["beta"]), z_dim=256,
                           hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
                           alpha_learnable=cfg.adapter_alpha_learnable,
                           project=cfg.adapter_project_z, residual=cfg.adapter_residual,
                           head=getattr(cfg, "adapter_head", "residual"),
                           theta_max_deg=getattr(cfg, "adapter_theta_max_deg", 60.0),
                           n_heads=H)
        ad.load_state_dict(ck["adapter"])
        ad.eval()
        beta = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32)
        z0 = np.stack([s["z0"] for s in samples]).astype(np.float64)
        with torch.no_grad():
            out = ad(beta, torch.tensor(z0, dtype=torch.float32)).numpy().astype(np.float64)
        if H > 1:
            # the head each clip last won with; clips never selected fall back to head 0
            sel = [head_sel.get(f"{s['reward_name']}|{s['trial']}|child", 0) for s in samples]
            out = out[np.arange(len(samples)), sel]
        z0h, zh = unit(z0), unit(out)

        rows = []
        for i, s in enumerate(samples):
            a = tangent(z0h[i], zh[i])
            ang = np.degrees(np.arccos(np.clip(z0h[i] @ zh[i], -1, 1)))
            t = tang = None
            if buf is not None:
                zt = unit(np.asarray(buf[f"{s['reward_name']}|{s['trial']}|child"][0],
                                     dtype=np.float64).reshape(-1))
                t = tangent(z0h[i], zt)
                tang = np.degrees(np.arccos(np.clip(z0h[i] @ zt, -1, 1)))
            rows.append((s["reward_name"], a, ang, t, tang))

        for label, keep in (("all", lambda r: True),
                            (args.task, lambda r: r[0].startswith(args.task))):
            R = [r for r in rows if keep(r) and r[1] is not None]
            if not R:
                continue
            A = np.stack([r[1] for r in R])
            pa, pc = spread(A)
            line = (f"{name:<20} {label:<10} {len(R):>4}  {pa:+.3f} {pc:5.2f} "
                    f"{np.mean([r[2] for r in R]):5.1f}")
            RT = [r for r in R if r[3] is not None]
            if RT:
                T = np.stack([r[3] for r in RT])
                tp, tc = spread(T)
                m = np.mean([float(r[1] @ r[3]) for r in RT])
                line += (f"  {tp:+8.3f} {tc:7.2f} {np.mean([r[4] for r in RT]):7.1f}  {m:+.3f}")
            print(line)


if __name__ == "__main__":
    main()
