#!/usr/bin/env bash
# run_size_headstand.sh -- the fair capacity test: exp5 (run_budget_experiments.sh)
# with a 39x bigger adapter and nothing else changed.
#
# exp5 trained the default adapter (0.66M) by ES on headstand alone: 32 training
# clips, so every clip got ~75 updates and ~2400 rollouts, more than a dedicated
# per-clip search needs, and it still ended at 1.04 x z0 on its own training clips
# (per-clip search floor ~0.24-0.28). If the network were too small to hold 32
# different corrections, a 25.7M one should do better here; if it does not, the
# limit is the training signal, not capacity.
set -euo pipefail
cd "$(dirname "$0")/.."

DS=datasets/crossenbodiment-child-balanced
OUT=outputs/simple_es/child_balanced
LOG=outputs/train_logs/child_balanced
mkdir -p "$LOG"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup setsid uv run python -m model.simple.train_es \
  --loss bfm --dataset-dir "$DS" --train-bodies child --init-reference \
  --updates 600 --pairs 16 --batch-size 128 --sigma 0.05 \
  --lr 3e-4 --alpha 1.0 --lambda-z 0 \
  --eval-every 25 --eval-clips 64 --no-progress \
  --project crossenbodiment-simple \
  --clip-list "$DS/splits/headstand_only_clips.txt" \
  --clip-categories "$DS/splits/headstand_only_categories.txt" \
  --heldout-clip-frac 0.2 \
  --hidden 1024 2048 2048 2048 2048 2048 2048 1024 \
  --ckpt-dir "$OUT/exp5_headstand_only_widedeep" --run-name "child-bal-exp5-widedeep" \
  > "$LOG/exp5_headstand_only_widedeep.log" 2>&1 < /dev/null &
echo "exp5_headstand_only_widedeep -> pid $!  log $LOG/exp5_headstand_only_widedeep.log"
