#!/usr/bin/env python3
"""fit_beta_map.py -- learn the body-parameter -> latent transformation from a
library of adapted latents, and predict it for bodies the library never saw.

Library: z*(clip, body) for every body on the morph paths, found by
continuation (method 'cont'; 'indep' is the ablation). Every predictor maps
(beta, clip) -> z and is evaluated ZERO-SHOT on held-out bodies by rollout
(scripts/score_z_matrix.py), so nothing here is judged by cosine.

  nn      the latent of the library body nearest in beta (a lookup, no model)
  kernel  Gaussian-weighted mean of the library bodies' CORRECTIONS, taken in z0's
          tangent space and walked out from z0:  Exp_z0( sum_b w_b Log_z0(z*_b) / sum_b w_b ).
          z0 is the same on every body for a given clip, so this averages what each
          body changed, the same object the correction-transfer study moves around.
          (Until 9/30 this was the Karcher mean of the latents themselves; the two
          differ by a median 1.4 deg on the real bodies.)
  linear  per clip, in z0's tangent space:
              Log_z0(z*) = A_c (beta - 1) + b_c          (ridge)
          then z = Exp_z0(.). "The latent correction is linear in the body
          parameters" -- the most interpretable form of the transformation.
  mlp     ONE network for all clips, z = G(beta, z0): the LatentAdapter this
          project set out to learn, now trained on consistent targets.

Hyper-parameters (kernel bandwidth, ridge strength) are picked by
leave-one-BODY-out on the library -- the test bodies are never looked at.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
from model.dataset import load_beta


def unit(v): return v / np.linalg.norm(v, axis=-1, keepdims=True)
def log_map(p, q):
    p, q = unit(p), unit(q); c = np.clip(p @ q, -1, 1); th = np.arccos(c)
    return np.zeros_like(p) if th < 1e-9 else th / np.sin(th) * (q - c * p)
def exp_map(p, v):
    p = unit(p); n = np.linalg.norm(v)
    return p if n < 1e-9 else np.cos(n) * p + np.sin(n) * v / n
def karcher(Z, w, iters=30):
    Z = unit(np.asarray(Z)); m = unit((w[:, None] * Z).sum(0))
    for _ in range(iters):
        V = np.stack([log_map(m, z) for z in Z]); step = (w[:, None] * V).sum(0) / w.sum()
        if np.linalg.norm(step) < 1e-8: break
        m = exp_map(m, step)
    return m


class Library:
    def __init__(self, prefixes, method):
        self.z = {}                               # (stem, body) -> z
        for p in prefixes:
            for f in glob.glob(str(REPO / f"outputs/continuation/{p}/{method}/*__*/best_z.npy")):
                stem, body = Path(f).parent.name.split("__")
                self.z[(stem, body)] = np.load(f).reshape(-1).astype(np.float64)
        self.bodies = sorted({b for _, b in self.z})
        self.stems = sorted({s for s, _ in self.z})
        self.beta = {b: load_beta(REPO / f"assets/robots/{b}/parameter.json") for b in self.bodies}


def z0_of(stem):
    return np.load(REPO / "data/origin_z" / stem.rsplit("_", 1)[0] / f"{stem}.npy").reshape(-1).astype(np.float64)


def pred_nn(L, stem, beta, exclude=()):
    bs = [b for b in L.bodies if (stem, b) in L.z and b not in exclude]
    b = min(bs, key=lambda b: np.linalg.norm(L.beta[b] - beta))
    return L.z[(stem, b)]


def pred_kernel(L, stem, beta, h, exclude=()):
    bs = [b for b in L.bodies if (stem, b) in L.z and b not in exclude]
    d = np.array([np.linalg.norm(L.beta[b] - beta) for b in bs])
    w = np.exp(-d ** 2 / (2 * h * h)) + 1e-12
    z0 = z0_of(stem)
    V = np.stack([log_map(z0, L.z[(stem, b)]) for b in bs])
    return exp_map(z0, (w[:, None] * V).sum(0) / w.sum())


def fit_linear(L, stem, lam, exclude=()):
    z0 = z0_of(stem)
    bs = [b for b in L.bodies if (stem, b) in L.z and b not in exclude]
    X = np.stack([np.r_[L.beta[b] - 1.0, 1.0] for b in bs])            # (n, 9)
    Y = np.stack([log_map(z0, L.z[(stem, b)]) for b in bs])              # (n, 256)
    R = lam * np.eye(X.shape[1]); R[-1, -1] = 0                          # do not shrink the bias
    W = np.linalg.solve(X.T @ X + R, X.T @ Y)                            # (9, 256)
    return z0, W


def pred_linear(L, stem, beta, lam, exclude=()):
    z0, W = fit_linear(L, stem, lam, exclude)
    return exp_map(z0, np.r_[beta - 1.0, 1.0] @ W)


def train_mlp(L, epochs=3000, exclude=(), seed=0, wd=1e-4, hidden=(256, 512, 512, 256), device="cpu"):
    import torch, torch.nn.functional as F
    from model.networks import LatentAdapter
    torch.manual_seed(seed)
    keys = [(s, b) for (s, b) in L.z if b not in exclude]
    be = torch.tensor(np.stack([L.beta[b] for _, b in keys]), dtype=torch.float32, device=device)
    z0 = torch.tensor(np.stack([unit(z0_of(s)) * 16 for s, _ in keys]), dtype=torch.float32, device=device)
    tg = torch.tensor(np.stack([unit(L.z[k]) for k in keys]), dtype=torch.float32, device=device)
    net = LatentAdapter(beta_dim=8, z_dim=256, hidden_dims=list(hidden),
                        alpha=1.0, project=True, residual=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        (1 - F.cosine_similarity(net(be, z0), tg, dim=-1)).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        c = F.cosine_similarity(net(be, z0), tg, dim=-1).clamp(-1, 1)
    print(f"  mlp {list(hidden)}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params, "
          f"library fit median {float(torch.rad2deg(torch.arccos(c)).median()):.2f} deg over {len(keys)} latents")
    return net


def pred_mlp(net, stem, beta):
    import torch
    dev = next(net.parameters()).device
    with torch.no_grad():
        return net(torch.tensor(beta[None], dtype=torch.float32, device=dev),
                   torch.tensor(unit(z0_of(stem))[None] * 16, dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)


def lobo_select(L, kind, grid):
    """leave-one-BODY-out on the library, scored by angle to the held-out body's
    own latent (a cheap proxy; the real evaluation is by rollout)."""
    best = None
    for g in grid:
        errs = []
        for b in L.bodies:
            for s in L.stems:
                if (s, b) not in L.z: continue
                p = (pred_kernel(L, s, L.beta[b], g, exclude=(b,)) if kind == "kernel"
                     else pred_linear(L, s, L.beta[b], g, exclude=(b,)))
                errs.append(np.degrees(np.arccos(np.clip(unit(p) @ unit(L.z[(s, b)]), -1, 1))))
        e = float(np.median(errs))
        print(f"    {kind} {g:<8} LOBO median angle {e:6.2f} deg")
        if best is None or e < best[1]: best = (g, e)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", nargs="+", default=["m2c", "m2s", "m2g", "m2k"])
    ap.add_argument("--method", default="cont")
    ap.add_argument("--test-bodies", nargs="+", required=True)
    ap.add_argument("--clips", default="outputs/continuation_clips.txt")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="predictors to write (default: all)")
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 512, 512, 256])
    ap.add_argument("--mlp-tag", default=None, help="label the mlp predictor <method>:mlp-<tag>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = Library(a.prefixes, a.method)
    print(f"library ({a.method}): {len(L.bodies)} bodies x {len(L.stems)} clips = {len(L.z)} latents")
    want = lambda k: a.only is None or k in a.only
    h = lobo_select(L, "kernel", [0.05, 0.1, 0.15, 0.25, 0.4]) if want("kernel") else None
    lam = lobo_select(L, "linear", [1e-3, 1e-2, 1e-1, 1.0, 10.0]) if want("linear") else None
    print(f"  chosen: kernel h={h}, linear ridge={lam}")
    net = None if (a.no_mlp or not want("mlp")) else train_mlp(L, hidden=a.mlp_hidden, device=a.device, seed=a.seed)
    clips = [l.strip() for l in open(REPO / a.clips) if l.strip()]
    C, B, Lb, Z = [], [], [], []
    for c in clips:
        stem = c.split("/")[-1]
        if stem not in L.stems: continue
        for tb in a.test_bodies:
            beta = load_beta(REPO / f"assets/robots/{tb}/parameter.json")
            preds = {}
            if want("nn"): preds["nn"] = pred_nn(L, stem, beta)
            if want("kernel"): preds["kernel"] = pred_kernel(L, stem, beta, h)
            if want("linear"): preds["linear"] = pred_linear(L, stem, beta, lam)
            if net is not None: preds["mlp" if a.mlp_tag is None else f"mlp-{a.mlp_tag}"] = pred_mlp(net, stem, beta)
            for k, z in preds.items():
                C.append(c); B.append(tb); Lb.append(f"{a.method}:{k}"); Z.append(unit(z) * 16)
    np.savez(REPO / a.out, clip=np.array(C), body=np.array(B), label=np.array(Lb), z=np.stack(Z))
    print(f"-> {a.out}: {len(C)} jobs")


if __name__ == "__main__":
    main()
