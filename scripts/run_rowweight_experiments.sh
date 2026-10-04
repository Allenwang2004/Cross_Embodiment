#!/usr/bin/env bash
# run_rowweight_experiments.sh -- the four runs that test why only `move` learns.
#
# Baseline (already done): outputs/simple_es/child_balanced/lr3e-4_pairs16, wandb
# child-bal500-lr3e-4-pairs16. Measured on the 50 held-out clips, cost/cost_z0
# went 0.960 -> 0.460 on move and 1.179 -> 1.255 on headstand, 1.020 -> 1.038 on
# raisearms: the shared MLP is spending itself on one family and actively hurting
# two others. Every flag below is a CATEGORY-FREE attempt at that, so whatever
# wins carries over to motions that do not exist yet.
#
#   1  --row-weight conf       a row's gradient is scaled by how much its own
#                              landscape actually moved. Cells already on their
#                              plateau stop emitting full-magnitude noise.
#   2a --row-weight headroom   scaled by cost/cost_z0: the budget drifts toward
#                              cells that have not improved yet.
#   2b --row-weight relative   the literal "score each cell relative to its own
#       --no-rank              z0". Needs rank normalization off to do anything.
#   4  --head geodesic         bounded great-circle step instead of an unbounded
#                              residual; a stability check, not a loss fix.
#                              NOTE the first attempt at this head split the MLP
#                              into a unit direction and an independent sigmoid
#                              angle, and did not train at all: at theta ~ 0 the
#                              direction's gradient is proportional to sin(theta)
#                              (theta moved 1.07 -> 1.13 deg in 33 updates).
#                              model/networks.py now reads the angle off the
#                              tangent vector's LENGTH, which is full rank at z0.
#
# Everything else is bit-identical to the baseline's config, including seed,
# eval_seed and therefore the held-out 50.
set -euo pipefail
cd "$(dirname "$0")/.."

DS=datasets/crossenbodiment-child-balanced
Z0=outputs/initial_cost/child_balanced500/z0_cost.csv
OUT=outputs/simple_es/child_balanced
LOG=outputs/train_logs/child_balanced
mkdir -p "$LOG"

COMMON=(--loss bfm --dataset-dir "$DS" --train-bodies child
        --clip-list "$DS/splits/balanced500_clips.txt"
        --clip-categories "$DS/splits/balanced500_categories.txt"
        --heldout-clip-frac 0.1 --init-reference
        --updates 600 --pairs 16 --batch-size 128 --sigma 0.05
        --lr 3e-4 --alpha 1.0 --lambda-z 0
        --eval-every 25 --eval-clips 64 --no-progress
        --project crossenbodiment-simple)

launch () {  # launch <tag> <extra args...>
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=${GPU:-0} nohup uv run python -m model.simple.train_es \
    "${COMMON[@]}" "$@" \
    --ckpt-dir "$OUT/$tag" --run-name "child-bal500-$tag" \
    > "$LOG/$tag.log" 2>&1 &
  echo "$tag -> pid $!  log $LOG/$tag.log"
}

launch exp1_conf      --row-weight conf
launch exp2a_headroom --row-weight headroom --z0-cost-csv "$Z0"
launch exp2b_relative --row-weight relative --no-rank --z0-cost-csv "$Z0"
launch exp4_geodesic  --head geodesic --theta-max 60
