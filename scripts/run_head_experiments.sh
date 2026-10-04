#!/usr/bin/env bash
# run_head_experiments.sh -- stronger best-point pull, and the multi-head fix.
#
# Why these four. The measured cause of "only move learns": along the path from
# z0 to a clip's searched answer, the window where the cost beats z0 is ~14 deg
# wide on headstand and ~60 deg on move. One shared aim lands inside move's
# window and outside headstand's -- headstand ends at 1.24 (WORSE than doing
# nothing) while move reaches 0.57. It is an aiming-precision problem, not a
# distance one: headstand needs the SHORTEST trip of any category (28.3 deg),
# and walking its 9.3 deg in the RIGHT direction would already give 0.85.
#
#   bc3 / bc10   the best-point buffer is the only thing that has moved the
#                stuck categories (lambda_bc 0 -> 1.0 took raisearms 0.998 ->
#                0.690 and headstand 1.335 -> 1.188) and the trend had not
#                turned over at 1.0, so push it.
#   h4 / h8      the adapter emits H candidate corrections and only the head
#                that scored best on a clip receives that clip's gradient, so a
#                head is shaped by the clips that prefer it rather than by the
#                whole batch. Each head gets pairs/H of the antithetic pairs,
#                so the rollout budget per update is unchanged.
#
# lambda_bc for the multi-head runs is raised to hold bc_ratio at the value the
# single-head lambda_bc=1.0 run had (~2.5e-3). Splitting the pairs across heads
# makes |g_es| grow (0.31 single-head, 0.76 at H=4, 1.43 at H=8), so keeping
# lambda_bc at 1.0 would quietly weaken the buffer and confound the comparison.
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
        --updates 1000 --pairs 16 --batch-size 128 --sigma 0.05
        --lr 3e-4 --alpha 1.0 --lambda-z 0
        --eval-every 25 --eval-clips 64 --no-progress
        --project crossenbodiment-simple)

launch () {
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=${GPU:-0} nohup uv run python -m model.simple.train_es \
    "${COMMON[@]}" "$@" --ckpt-dir "$OUT/$tag" --run-name "child-mem-$tag" \
    > "$LOG/$tag.log" 2>&1 &
  echo "$tag -> pid $!"
}

launch bc3.0  --lambda-bc 3.0
launch bc10.0 --lambda-bc 10.0
launch h4     --heads 4 --lambda-bc 2.5
launch h8     --heads 8 --lambda-bc 4.5
