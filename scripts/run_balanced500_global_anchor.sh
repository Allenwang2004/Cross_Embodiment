#!/usr/bin/env bash
# run_balanced500_global_anchor.sh -- the 500-clip child run lr3e-4_pairs16 (balanced500, 10% held out
# stratified by category, pure ES, default adapter) with the CANDIDATE cost changed to
#   bfm + 1.0 * heading + 0.1 * mean root-xy distance + 0.3 * (|z - z0| / 16)^2
# (added before ranking; the evaluation still reports plain bfm), and 1000 updates instead of 600.
set -euo pipefail
cd "$(dirname "$0")/.."

DS=datasets/crossenbodiment-child-balanced
OUT=outputs/simple_es/child_balanced
LOG=outputs/train_logs/child_balanced
mkdir -p "$LOG"

CUDA_VISIBLE_DEVICES=${GPU:-3} nohup setsid uv run python -m model.simple.train_es \
  --loss bfm --dataset-dir "$DS" --train-bodies child --init-reference \
  --updates 1000 --pairs 16 --batch-size 128 --sigma 0.05 \
  --lr 3e-4 --alpha 1.0 --lambda-z 0 \
  --eval-every 25 --eval-clips 64 --no-progress \
  --project crossenbodiment-simple \
  --clip-list "$DS/splits/balanced500_clips.txt" \
  --clip-categories "$DS/splits/balanced500_categories.txt" \
  --heldout-clip-frac 0.1 \
  --heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0.3 \
  --ckpt-dir "$OUT/b500_global_anchor" --run-name "child-bal-b500-global-anchor" \
  > "$LOG/b500_global_anchor.log" 2>&1 < /dev/null &
echo "b500_global_anchor -> pid $!  log $LOG/b500_global_anchor.log"
