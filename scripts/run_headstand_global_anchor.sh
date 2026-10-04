#!/usr/bin/env bash
# run_headstand_global_anchor.sh -- exp5 (headstand only, 32 train / 8 held out, pure ES, default
# adapter, 600 updates) with only the CANDIDATE cost changed:
#   bfm + 1.0 * heading + 0.1 * mean root-xy distance + 0.3 * (|z - z0| / 16)^2
# all added before ranking (model/simple/config.py: heading_weight, pos_weight, anchor_weight).
# The evaluation still reports the plain bfm cost, so the numbers compare with exp5 directly.
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
  --heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0.3 \
  --ckpt-dir "$OUT/exp5_global_anchor" --run-name "child-bal-exp5-global-anchor" \
  > "$LOG/exp5_global_anchor.log" 2>&1 < /dev/null &
echo "exp5_global_anchor -> pid $!  log $LOG/exp5_global_anchor.log"
