#!/usr/bin/env bash
# run_budget_experiments.sh -- the two runs that follow the row-weight null result.
#
# scripts/run_rowweight_experiments.sh tried four category-free ways of stopping
# `move` from dominating the gradient. All four landed inside the eval noise
# (see the table in outputs/simple_es/child_balanced/results_rowweight.png), and
# they share a blind spot: every one of them reweights the rows ALREADY IN a
# batch. None of them changes which clips get into the batch, or how many
# rollouts a single clip ever receives.
#
# What the null result left standing: a dedicated per-clip ES search reaches
# cost/cost_z0 of 0.17-0.29 on headstand with 4992 rollouts, while the shared
# adapter -- which gets ~170 rollouts per clip (600 updates x 4 clips / 450 clips
# x 32 candidates) -- sits at 1.22. And headstand_3/_4's solutions are only
# 17-18 deg from z0, inside the 22 deg the adapter already travels, so it is
# neither infeasible nor out of reach.
#
#   5  headstand only     ~32 training clips, so each gets ~2400 rollouts -- the
#                         same order as the per-clip search. Reaching ~0.3 says
#                         the problem was budget and cross-family interference;
#                         staying at ~1.2 says the correction is not a function
#                         of z0 and the model's INPUT has to change.
#   6  --clip-balance     draw the motion family first, then a clip. Uniform-
#      category           over-clips sampling gives move 200 of the 500 clips and
#                         so 40% of every gradient; this makes it 16.7%.
set -euo pipefail
cd "$(dirname "$0")/.."

DS=datasets/crossenbodiment-child-balanced
OUT=outputs/simple_es/child_balanced
LOG=outputs/train_logs/child_balanced
mkdir -p "$LOG"

COMMON=(--loss bfm --dataset-dir "$DS" --train-bodies child --init-reference
        --updates 600 --pairs 16 --batch-size 128 --sigma 0.05
        --lr 3e-4 --alpha 1.0 --lambda-z 0
        --eval-every 25 --eval-clips 64 --no-progress
        --project crossenbodiment-simple)

launch () {  # launch <tag> <extra args...>
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=${GPU:-0} nohup uv run python -m model.simple.train_es \
    "${COMMON[@]}" "$@" \
    --ckpt-dir "$OUT/$tag" --run-name "child-bal-$tag" \
    > "$LOG/$tag.log" 2>&1 &
  echo "$tag -> pid $!  log $LOG/$tag.log"
}

# 0.2 rather than 0.1: 10% of 40 clips is 4 held out, too few to read a test
# number off at all. 8/32 still leaves every training clip ~2400 rollouts.
launch exp5_headstand_only \
  --clip-list "$DS/splits/headstand_only_clips.txt" \
  --clip-categories "$DS/splits/headstand_only_categories.txt" \
  --heldout-clip-frac 0.2

launch exp6_catbalance \
  --clip-list "$DS/splits/balanced500_clips.txt" \
  --clip-categories "$DS/splits/balanced500_categories.txt" \
  --heldout-clip-frac 0.1 --clip-balance category
