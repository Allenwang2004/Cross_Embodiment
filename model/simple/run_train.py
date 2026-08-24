"""Entry point for launching training with the default config on GPU.

Usage (from project root):
    uv run model/simple/run_train.py
    uv run model/simple/run_train.py --gpu 1      # run on cuda:1
    uv run model/simple/run_train.py --device cpu # or any explicit device string
    uv run model/simple/run_train.py --updates 400 --run-name cycle-400
    uv run model/simple/run_train.py --no-wandb   # local only, no run created
"""
import argparse
import sys
from pathlib import Path

# Running this file directly (not `-m model.simple.run_train`) puts model/simple/
# itself on sys.path, not the repo root -- so `model.simple.config`/`model.simple.train`
# absolute imports can't find the `model` package unless the repo root is added too.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.simple.config import TrainConfig
from model.simple.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index, e.g. 1 for cuda:1")
    parser.add_argument("--device", default=None,
                         help="explicit device string (e.g. 'cpu', 'cuda:2'); overrides --gpu")
    parser.add_argument("--updates", type=int, default=None,
                         help="override cfg.num_updates. One update is one body, so a "
                              "multiple of len(train_bodies) gives every body the same count")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--body-order", choices=["cycle", "random"], default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--loss-curve", default=None)
    parser.add_argument("--no-progress", action="store_true",
                         help="no tqdm bar -- use for a run redirected to a log file, "
                              "where the bar's \\r refreshes collapse onto one line")
    parser.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    parser.add_argument("--project", default=None, help="W&B project name")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    args = parser.parse_args()

    cfg = TrainConfig(device=args.device or f"cuda:{args.gpu}")
    for attr, val in (("num_updates", args.updates), ("batch_size", args.batch_size),
                      ("body_order", args.body_order), ("ckpt_dir", args.ckpt_dir),
                      ("loss_curve_path", args.loss_curve), ("wandb_project", args.project),
                      ("wandb_run_name", args.run_name)):
        if val is not None:
            setattr(cfg, attr, val)
    if args.no_progress:
        cfg.progress = False
    if args.no_wandb:
        cfg.use_wandb = False
    train(cfg)


if __name__ == "__main__":
    main()
