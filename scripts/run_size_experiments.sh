#!/usr/bin/env bash
# run_size_experiments.sh -- does a BIGGER adapter help the single-body ES batch
# training? Same setting as run_memorize_experiments.sh's bc1.0 (the best of that
# sweep: child body, all 500 balanced clips, no held-out split, best-point pull
# lambda_bc = 1.0), only the adapter's hidden widths change. 500 updates: bc1.0
# had reached 0.70 of its start cost by then (0.67 at 1000).
#
#   base      [256, 512, 512, 256]                              0.66M  (= bc1.0)
#   wide      [1024, 2048, 2048, 1024]                          8.9M
#   deep      [256, 512, 512, 512, 512, 512, 512, 256]          1.7M
#   widedeep  [1024, 2048 x 6, 1024]                            25.7M
#
# Read against bc1.0/update_00500.pt: eval cost, and scripts/analyze_collapse.py
# (do the corrections still all point one way?).
set -euo pipefail
cd "$(dirname "$0")/.."

DS=datasets/crossenbodiment-child-balanced
OUT=outputs/simple_es/child_memorize
LOG=outputs/train_logs/child_memorize
mkdir -p "$LOG"

COMMON=(--loss bfm --dataset-dir "$DS" --train-bodies child --init-reference
        --clip-list "$DS/splits/balanced500_clips.txt"
        --clip-categories "$DS/splits/balanced500_categories.txt"
        --heldout-clip-frac 0
        --updates 500 --pairs 16 --batch-size 128 --sigma 0.05
        --lr 3e-4 --alpha 1.0 --lambda-z 0 --lambda-bc 1.0
        --eval-every 25 --eval-clips 64 --no-progress
        --project crossenbodiment-simple)

launch () {
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup setsid uv run python -m model.simple.train_es \
    "${COMMON[@]}" "$@" \
    --ckpt-dir "$OUT/$tag" --run-name "child-mem-$tag" \
    > "$LOG/$tag.log" 2>&1 < /dev/null &
  echo "$tag -> pid $!  log $LOG/$tag.log"
}

launch bc1.0_wide     --hidden 1024 2048 2048 1024
launch bc1.0_deep     --hidden 256 512 512 512 512 512 512 256
launch bc1.0_widedeep --hidden 1024 2048 2048 2048 2048 2048 2048 1024
