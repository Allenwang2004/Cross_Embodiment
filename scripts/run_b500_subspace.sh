#!/usr/bin/env bash
# run_b500_subspace.sh <k> -- b500_global_anchor (balanced500, 10% held out stratified by category,
# pure ES, cost bfm + 1.0 * heading + 0.1 * root-xy + 0.3 * (|z - z0| / 16)^2) with ONE change:
# the adapter's correction and the ES perturbations are confined to the top-k rows of
# outputs/lowdim_search/basis_corrPCA_train.npy (uncentered PCA of the b500 corrections z_align - z0,
# TRAIN clips only). 500 updates. OBS=exact: the actor and B see the adult-equivalent observation
# (--obs-scale exact, model/exact_obs.py) instead of the fixed multiplier; checkpoints go to b500_sub<k>_exact.
# BASIS=<.npy> / TAG=<name> override the subspace basis and the run's tag.
set -euo pipefail
cd "$(dirname "$0")/.."
K=${1:?usage: run_b500_subspace.sh <k>}
OBS=${OBS:-auto}
BASIS=${BASIS:-outputs/lowdim_search/basis_corrPCA_train.npy}
DEF_TAG=sub$K; [ "$OBS" = auto ] || DEF_TAG=sub${K}_$OBS
TAG=${TAG:-$DEF_TAG}

DS=datasets/crossenbodiment-child-balanced
OUT=outputs/simple_es/child_balanced
LOG=outputs/train_logs/child_balanced
mkdir -p "$LOG"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup setsid uv run python -m model.simple.train_es \
  --loss bfm --dataset-dir "$DS" --train-bodies child --init-reference \
  --updates 500 --pairs 16 --batch-size 128 --sigma 0.05 \
  --lr 3e-4 --alpha 1.0 --lambda-z 0 \
  --eval-every 25 --eval-clips 64 --no-progress \
  --project crossenbodiment-simple \
  --clip-list "$DS/splits/balanced500_clips.txt" \
  --clip-categories "$DS/splits/balanced500_categories.txt" \
  --heldout-clip-frac 0.1 \
  --heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0.3 \
  --subspace "$BASIS" --subspace-dim "$K" --obs-scale "$OBS" \
  --ckpt-dir "$OUT/b500_$TAG" --run-name "child-bal-b500-$TAG" \
  > "$LOG/b500_$TAG.log" 2>&1 < /dev/null &
echo "b500_$TAG -> pid $!  log $LOG/b500_$TAG.log"
