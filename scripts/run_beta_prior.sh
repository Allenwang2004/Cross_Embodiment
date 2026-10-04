#!/usr/bin/env bash
# run_beta_prior.sh -- Phase 2: a beta-conditioned prior over the latent.
#
# The pivot: stop asking the adapter to BE the answer (measured: 8.3 deg of
# error, and headstand's usable window is a few degrees wide, so as an answer it
# is worthless) and ask it to be a STARTING POINT for the per-cell search. An
# inaccurate answer is useless; an inaccurate prior that is still closer than z0
# is search someone does not have to pay for.
#
# 42 clips x 15 training bodies = 630 cells, chosen so each cell gets ~6 visits
# at 1000 updates -- the same density as the child run that worked (500 cells,
# 8 visits). 4 bodies are held out entirely (giant, short_limbed, x_leg140,
# x_leg062_heavy: the largest, the shortest-limbed, the longest-legged and a
# heavy one) so the prior cannot pass by interpolating between near-identical
# bodies.
#
# lambda_bc 1.0 is the recipe that worked on child (ALL 0.923 -> 0.828,
# raisearms 0.801 -> 0.507). One body per update, cycled, so the cost per update
# matches the child run and the wall clock is predictable.
set -euo pipefail
cd "$(dirname "$0")/.."
DS=datasets/crossenbodiment-19bodies-torque
OUT=outputs/simple_es/beta_prior
LOG=outputs/train_logs/beta_prior
mkdir -p "$LOG"

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup uv run python -m model.simple.train_es \
  --loss bfm --dataset-dir "$DS" --init-reference \
  --clip-list "$DS/splits/beta_clips.txt" \
  --clip-categories "$DS/splits/beta_categories.txt" \
  --heldout-clip-frac 0 \
  --bodies 1 \
  --updates ${UPDATES:-1000} --pairs 16 --batch-size 128 --sigma 0.05 \
  --lr 3e-4 --alpha 1.0 --lambda-z 0 --lambda-bc ${LBC:-1.0} \
  --eval-every 25 --eval-clips 64 --no-progress \
  --project crossenbodiment-simple \
  --ckpt-dir "$OUT/${TAG:-bc1.0}" --run-name "beta-prior-${TAG:-bc1.0}" \
  > "$LOG/${TAG:-bc1.0}.log" 2>&1 &
echo "beta_prior/${TAG:-bc1.0} -> pid $!"
