#!/usr/bin/env bash
# run_memorize_experiments.sh -- can ONE adapter hold a good z for ALL 500 clips
# at once, on one body? That is the prerequisite for adding beta, and it is a
# fitting question, not a generalisation one -- so there is no held-out split.
#
# What the earlier work settled:
#   * The shared adapter learns ONE global correction direction (40 headstand
#     clips' corrections at mean pairwise cos +0.973) because ES estimates each
#     cell's gradient by probing, and averaging over a batch keeps only the
#     component every cell agrees on.
#   * CAPACITY is not the limit: fitting the same LatentAdapter to 32 searched
#     targets supervised lands 0.01 deg from each of them and reproduces the
#     search floor when rolled out (0.442 against the targets' own 0.454).
#   * Clip-to-clip generalisation is dead (held-out predictions sit further from
#     the target than z0 does) -- which is why these runs do not test it.
#
# So the open question is whether the search can FIND those per-cell points
# without precomputing them. --lambda-bc is the mechanism already in the file:
# update_best_buffer keeps the best candidate the simulator ever scored for each
# (clip, body) and adds an EXACT-gradient cosine pull toward it, which is not
# averaged away the way the probed gradient is.
#
# Yardstick: all 40 headstand clips have a measured per-clip floor in
# outputs/single_z_floor/, so "how close to the known optimum" is answerable
# per clip rather than only in aggregate.
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
    "${COMMON[@]}" "$@" \
    --ckpt-dir "$OUT/$tag" --run-name "child-mem-$tag" \
    > "$LOG/$tag.log" 2>&1 &
  echo "$tag -> pid $!  log $LOG/$tag.log"
}

# bc0 is the control: same 500 clips, same no-holdout setting, buffer off, so a
# difference cannot be attributed to the split or the clip count.
launch bc0   --lambda-bc 0
launch bc0.1 --lambda-bc 0.1
launch bc0.3 --lambda-bc 0.3
launch bc1.0 --lambda-bc 1.0
