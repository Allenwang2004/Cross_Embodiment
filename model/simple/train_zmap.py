"""Supervised cross-body z map: (z_adult, beta) -> z_body. No simulator.

    z_pred = G_theta(beta, z_adult)                 # the same LatentAdapter
    loss   = mean_t [ 1 - cos(z_pred, z_body) ]

Every frame of every clip is a labelled pair. `data/infer_origin_z` is the adult
performing the motion on the ADULT skeleton, run through Metamotivo's
tracking_inference; `data/<body>/infer_retargeting_z` is the SAME motion,
retargeted, run through the same inference on THAT BODY's skeleton. Same clip,
same frame, two latents. The map between them is exactly what the adapter is
supposed to represent, and here it is available as ground truth instead of
having to be discovered through a physics rollout.

Why this is worth doing before any more rollout-based training
--------------------------------------------------------------
The rollout-based objective was measured to be 96.5% two terms, one of which
(L_align.root, 60%) is a global heading proxy and the other (P.fall, 36%) does not
respond to z at all (sd/mean 0.05). Three different estimators -- REINFORCE,
PPO, ES -- all failed to improve it, and a direct sweep of the landscape showed
why: near z0 only 24% of directions help. None of that applies here. The target
is a labelled 256-d vector, the gradient is exact, and one epoch costs seconds.

What this does and does not answer
-----------------------------------
It answers: is there a learnable function from (adult latent, body shape) to the
body's own latent, and does it generalize to a body never trained on?

It does NOT answer whether feeding z_pred to the frozen actor makes the robot
move well. tracking_inference's output is the latent that BEST EXPLAINS a motion
the body already performed kinematically; it is not certified to be a latent the
body can physically execute. Closing that gap is still a rollout question --
model/simple/evaluate.py -- and a good cosine here does not imply a good L_align.

The identity baseline is the number that matters
-------------------------------------------------
cos(z_adult, z_body) is what you get by doing nothing. The adapter is only
worth anything if it beats that, per body, on bodies it never saw. Both are
reported at every eval.

Usage (from project root):
    uv run model/simple/train_zmap.py
    uv run model/simple/train_zmap.py --epochs 40 --run-name zmap-40
    uv run model/simple/train_zmap.py --no-wandb
"""

import argparse
import dataclasses
import json
import os
import random
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb

from model.dataset import BETA_AXES, load_task_list
from model.networks import LatentAdapter
from model.simple.config import ZMapConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_pairs(cfg, dataset_dir: Path):
    """Flatten every (clip, body) into per-frame tensors.

    Returns dict (body, task_split) -> (src, dst, beta, clip_id), so the body and
    task axes can be crossed at evaluation time.
    clip_id is kept so an evaluation can subsample by clip rather than by frame
    -- frames inside one clip are highly correlated and treating them as
    independent would make any per-frame average look far more precise than it is.
    """
    rows = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text().splitlines() if l]
    out = {}
    for r in tqdm(rows, desc="load", disable=not cfg.progress):
        b = (r["morphology_label"], r.get("task_split", "train"))
        src = np.load(dataset_dir / r["infer_origin_z"]).astype(np.float32)
        dst = np.load(dataset_dir / r["retarget_z"]).astype(np.float32)
        n = min(len(src), len(dst))
        if b not in out:
            beta = json.loads((dataset_dir / r["morphology"]).read_text())
            out[b] = {"src": [], "dst": [], "clip": [],
                      "beta": np.array([beta[a] for a in BETA_AXES], dtype=np.float32)}
        out[b]["src"].append(src[:n])
        out[b]["dst"].append(dst[:n])
        out[b]["clip"].append(np.full(n, r["id"], dtype=np.int64))
    for b, d in out.items():
        d["src"] = torch.from_numpy(np.concatenate(d["src"]))
        d["dst"] = torch.from_numpy(np.concatenate(d["dst"]))
        d["clip"] = torch.from_numpy(np.concatenate(d["clip"]))
        d["beta"] = torch.from_numpy(d["beta"])
    return out


def cos_of(a, b):
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)


@torch.no_grad()
def evaluate(cfg, adapter, data, keys, max_frames=200_000):
    """Per (body, task_split): the adapter's cosine, and the identity baseline."""
    adapter.eval()
    out = {}
    for b in keys:
        d = data[b]
        n = len(d["src"])
        idx = torch.arange(n) if n <= max_frames else torch.randperm(n)[:max_frames]
        src = d["src"][idx].to(cfg.device)
        dst = d["dst"][idx].to(cfg.device)
        beta = d["beta"].to(cfg.device).expand(len(idx), -1)
        pred = adapter(beta, src)
        out[b] = {
            "cos": cos_of(pred, dst).mean().item(),
            "cos_identity": cos_of(src, dst).mean().item(),
            "mse": F.mse_loss(pred, dst).item(),
        }
        out[b]["gain"] = out[b]["cos"] - out[b]["cos_identity"]
    adapter.train()
    return out


def summarize(per_key, body_split):
    """Four quadrants: (train|test body) x (train|test task)."""
    agg = {}
    for bs in ("train", "test"):
        for ts in ("train", "test"):
            mem = [k for k in per_key if body_split.get(k[0]) == bs and k[1] == ts]
            if mem:
                agg[f"body_{bs}/task_{ts}"] = {
                    k: float(np.mean([per_key[m][k] for m in mem]))
                    for k in ("cos", "cos_identity", "gain", "mse")}
    return agg


def train(cfg: ZMapConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_dir = REPO_ROOT / cfg.dataset_dir
    data = load_pairs(cfg, dataset_dir)
    bodies_present = {k[0] for k in data}
    train_bodies = [b for b in load_task_list(dataset_dir / "splits" / "train_bodies.txt")
                    if b in bodies_present]
    test_bodies = [b for b in load_task_list(dataset_dir / "splits" / "test_bodies.txt")
                   if b in bodies_present]
    body_split = {**{b: "train" for b in train_bodies}, **{b: "test" for b in test_bodies}}
    test_tasks = load_task_list(dataset_dir / "splits" / "test_tasks.txt")

    # The training pool is the (train body, train task) quadrant ONLY. Everything
    # else is held out on one axis or both.
    pool_keys = [(b, "train") for b in train_bodies if (b, "train") in data]
    n_frames = sum(len(data[k]["src"]) for k in pool_keys)
    print(f"train bodies ({len(train_bodies)}): {' '.join(train_bodies)}")
    print(f"held-out bodies ({len(test_bodies)}): {' '.join(test_bodies)}")
    print(f"held-out tasks  ({len(test_tasks)}): {' '.join(test_tasks)}")
    print(f"{n_frames:,} training frame pairs "
          f"({n_frames // max(len(train_bodies), 1):,} per body)")

    # One flat pool over the training bodies. Bodies mix freely inside a batch --
    # there is no env here, so nothing forces one body per step, and a batch that
    # spans several betas is what makes the conditioning learnable.
    src = torch.cat([data[k]["src"] for k in pool_keys])
    dst = torch.cat([data[k]["dst"] for k in pool_keys])
    beta = torch.cat([data[k]["beta"].expand(len(data[k]["src"]), -1) for k in pool_keys])
    if cfg.preload_device:
        src, dst, beta = src.to(cfg.device), dst.to(cfg.device), beta.to(cfg.device)

    adapter = LatentAdapter(
        beta_dim=len(BETA_AXES), z_dim=src.shape[-1],
        hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable, project=cfg.adapter_project_z,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)
    steps_per_epoch = len(src) // cfg.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs * steps_per_epoch, 1), eta_min=cfg.lr_final)

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config={**dataclasses.asdict(cfg), "train_bodies": train_bodies,
                           "test_bodies": test_bodies, "n_frames": n_frames})

    def report(tag, step):
        per = evaluate(cfg, adapter, data, list(data))
        agg = summarize(per, body_split)
        line = "  ".join(
            f"{q}: cos={d['cos']:.4f} (id {d['cos_identity']:.4f}, {d['gain']:+.4f})"
            for q, d in agg.items())
        tqdm.write(f"  [{tag}] {line}")
        if cfg.use_wandb:
            log = {f"eval/{b}_{ts}/{k}": v for (b, ts), d in per.items() for k, v in d.items()}
            log.update({f"eval/{q}/{k}": v for q, d in agg.items() for k, v in d.items()})
            wandb.log(log, step=step)
        return per, agg

    report("epoch 0", 0)

    step = 0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(src), device=src.device)
        run_loss = run_cos = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{cfg.epochs}",
                    disable=not cfg.progress)
        for i in pbar:
            idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            s = src[idx].to(cfg.device, non_blocking=True)
            t = dst[idx].to(cfg.device, non_blocking=True)
            bta = beta[idx].to(cfg.device, non_blocking=True)

            pred = adapter(bta, s)
            cos = cos_of(pred, t).mean()
            # Cosine, not MSE: both endpoints live on the sphere of radius
            # sqrt(256) (tracking_inference calls project_z), so 1 - cos is the
            # distance on the manifold the targets actually occupy and MSE would
            # partly penalize a radius that is fixed by construction.
            loss = 1.0 - cos
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            sched.step()

            run_loss += loss.item()
            run_cos += cos.item()
            step += 1
            if cfg.use_wandb and step % cfg.log_every == 0:
                wandb.log({"loss": loss.item(), "cos": cos.item(),
                           "grad_norm": float(grad), "lr": sched.get_last_lr()[0]}, step=step)
            if i % 200 == 0:
                pbar.set_postfix(cos=f"{cos.item():.4f}")

        tqdm.write(f"[epoch {epoch + 1}/{cfg.epochs}] train-batch cos "
                   f"{run_cos / max(steps_per_epoch, 1):.4f}")
        report(f"epoch {epoch + 1}", step)
        torch.save({"adapter": adapter.state_dict(), "epoch": epoch + 1, "cfg": cfg,
                    "train_bodies": train_bodies, "test_bodies": test_bodies,
                    "test_tasks": test_tasks}, ckpt_dir / "latest.pt")

    per, agg = report("final", step)
    print("\n=== cosine to the target latent (higher is better) ===")
    print(f"{'body':14s} {'body':6s} {'task':6s} {'adapter':>9s} {'identity':>9s} "
          f"{'gain':>8s} {'mse':>9s}")
    for k in sorted(per, key=lambda x: (body_split[x[0]] != "train", x[1] != "train", x[0])):
        d = per[k]
        print(f"{k[0]:14s} {body_split[k[0]]:6s} {k[1]:6s} {d['cos']:9.4f} "
              f"{d['cos_identity']:9.4f} {d['gain']:+8.4f} {d['mse']:9.4f}")
    print()
    for q, a in agg.items():
        print(f"{'-- ' + q:28s} {a['cos']:9.4f} {a['cos_identity']:9.4f} "
              f"{a['gain']:+8.4f} {a['mse']:9.4f}")
    if cfg.use_wandb:
        wandb.finish()
    print(f"\ncheckpoint -> {ckpt_dir / 'latest.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--project", default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args()

    cfg = ZMapConfig(device=a.device or f"cuda:{a.gpu}")
    for attr, val in (("epochs", a.epochs), ("batch_size", a.batch_size), ("lr", a.lr),
                      ("ckpt_dir", a.ckpt_dir), ("wandb_project", a.project),
                      ("wandb_run_name", a.run_name)):
        if val is not None:
            setattr(cfg, attr, val)
    if a.no_progress:
        cfg.progress = False
    if a.no_wandb:
        cfg.use_wandb = False
    train(cfg)


if __name__ == "__main__":
    main()
