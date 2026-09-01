"""Diagnostic: is the adapter actually learning anything, or is
the L_align/L_phys noise seen in normal training (see outputs/train_logs/loss_curve.png)
coming entirely from a different confound -- a NEW random set of tasks being
sampled every update?

Normal training draws cfg.batch_size fresh random (z0, beta, qpos_ref) rows
every update. Since the 43 train tasks likely have very different intrinsic L_align
scales (a standing pose task is probably much easier to match than crawl or
headstand), the batch-mean L_align swinging an order of magnitude update to update
could just reflect "which tasks got sampled this time", with the actual
policy-quality signal buried underneath.

This script removes that confound: it repeats a SINGLE fixed dataset row
cfg.batch_size times every update. With the (clip x body) manifest that also
pins the BODY -- normal training walks 8 of them and their cost scales differ
by several times, so --task-idx now isolates both axes at once (all sub-envs run the exact same z0/beta/
qpos_ref -- the only thing that differs between sub-envs is the stochastic
action noise). If L_align trends down here, task-composition noise was the real
problem in normal training. If it still doesn't move, the issue is deeper
(e.g. reward/cost signal too weak, lr, or REINFORCE variance itself).

Usage (from project root):
    uv run model/simple/diagnose_single_task.py
    uv run model/simple/diagnose_single_task.py --task-idx 0 --num-updates 100 --gpu 0
"""

import argparse
import os
import random
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel

from model.simple.config import TrainConfig
from model.dataset import CrossEmbodimentDataset, load_task_list
from model.networks import LatentAdapter, ValueNet
from model.simple.train import (make_body_ctx, plot_loss_curve, ppo_update,
                                rollout_batch, window_rewards)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-idx", type=int, default=0, help="dataset row index to repeat every update")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", default="outputs/train_logs/diagnose_single_task.png")
    args = parser.parse_args()

    cfg = TrainConfig(device=f"cuda:{args.gpu}", batch_size=args.batch_size, num_updates=args.num_updates)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_dir = REPO_ROOT / cfg.dataset_dir
    dataset = CrossEmbodimentDataset(dataset_dir)   # no task split, same as train.py

    sample = dataset[args.task_idx]
    beta_dim = len(sample["beta"])
    print(f"repeating fixed row: {sample['reward_name']} trial {sample['trial']} "
          f"on {sample['morphology_label']} "
          f"(has qpos_ref: {sample['qpos_ref'] is not None})")

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()

    # The env/fk/obs-multiplier must be THIS ROW'S body, not cfg.target_xml --
    # the manifest is a (clip x body) cross product, so --task-idx selects a
    # body as much as it selects a clip. Same builder train.py uses, so this
    # stays a diagnostic of the same setup rather than of a near-miss.
    ctx = make_body_ctx(cfg, dataset_dir, sample["morphology_label"], sample["target_xml"])
    env, fk_model, obs_mul = ctx["env"], ctx["fk"], ctx["obs_mul"]

    adapter = LatentAdapter(
        beta_dim=beta_dim, z_dim=model.cfg.archi.z_dim,
        hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable,
        project=cfg.adapter_project_z,
        residual=getattr(cfg, "adapter_residual", True),
    ).to(cfg.device)
    obs_dim = env.single_observation_space["proprio"].shape[0]
    value_net = ValueNet(obs_dim=obs_dim, z_dim=model.cfg.archi.z_dim,
                         beta_dim=beta_dim, hidden_dims=cfg.value_hidden_dims).to(cfg.device)
    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(value_net.parameters()), lr=cfg.lr)

    z0_t = torch.tensor(sample["z0"], dtype=torch.float32, device=cfg.device).unsqueeze(0).repeat(cfg.batch_size, 1)
    beta_t = torch.tensor(sample["beta"], dtype=torch.float32, device=cfg.device).unsqueeze(0).repeat(cfg.batch_size, 1)
    qpos_refs = [sample["qpos_ref"]] * cfg.batch_size

    loss_history, align_history, l_phys_history = [], [], []

    pbar = tqdm(range(cfg.num_updates), desc="diagnose")
    for update in pbar:
        # Exactly train.py's update, on one pinned (clip, body) -- same
        # rollout_batch / window_rewards / ppo_update, so what this isolates is
        # the sampling, not a second implementation of the objective.
        episode = rollout_batch(model, adapter, value_net, env, z0_t, beta_t, cfg, obs_mul)
        reward_np, align_totals, l_physes, _ = window_rewards(
            fk_model, cfg, episode["qpos_beta"], qpos_refs)
        reward = torch.as_tensor(reward_np, device=cfg.device)
        st = ppo_update(cfg, model, adapter, value_net, optimizer,
                        episode, z0_t, beta_t, reward)

        loss_history.append(st["pg"] + cfg.value_coef * st["v"] + st["z_reg_loss"])
        align_history.append(float(align_totals.mean()))
        l_phys_history.append(float(l_physes.mean()))
        pbar.set_postfix(L_align=f"{align_totals.mean():.4f}", L_phys=f"{l_physes.mean():.4f}")

        if update % 10 == 0:
            tqdm.write(f"[{update:04d}/{cfg.num_updates}] pg={st['pg']:+.4f} "
                       f"v={st['v']:.4f} 1-zcos={1 - st['z_cos']:.2e} "
                       f"L_align={align_totals.mean():.4f} L_phys={l_physes.mean():.4f}")

    env.close()
    plot_loss_curve(loss_history, align_history, l_phys_history, REPO_ROOT / args.out)

    first10 = np.mean(align_history[:10])
    last10 = np.mean(align_history[-10:])
    print(f"\nL_align mean, first 10 updates: {first10:.4f}")
    print(f"L_align mean, last 10 updates:  {last10:.4f}")
    print(f"{'IMPROVED' if last10 < first10 else 'DID NOT IMPROVE'} ({(1 - last10/first10)*100:+.1f}%)")


if __name__ == "__main__":
    main()
