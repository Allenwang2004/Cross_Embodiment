#!/usr/bin/env bash
# run_headstand_floor_search.sh -- a per-clip ES search for every headstand clip,
# so the best z found becomes a supervised TARGET.
#
# Why: exp5 showed a headstand-only adapter learns ONE global correction
# direction -- the 40 clips' corrections have mean pairwise cos +0.973, where
# independent directions in 255-d would be ~0. ES averages its gradient over the
# batch, so each clip's own component cancels and only the shared one survives.
# Before spending more runs on the estimator, find out whether a per-clip map
# EXISTS at all: search each clip alone, then fit the adapter to those targets
# supervised (model/simple/train_zmap.py). Fits -> ES is the bottleneck. Does not
# fit -> z0 -> z is not a function and the model's input has to change.
#
# Settings are copied EXACTLY from outputs/single_z_floor/headstand_{3,4,9}_bfm_s0
# so the three floors already measured stay comparable and need not be re-run.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/single_z_floor
LOG=outputs/train_logs/floor_search
mkdir -p "$LOG"
PAR=${PAR:-4}          # 4 x 16 async env workers = 64, one per core

pids=()
for k in $(seq 0 39); do
  d="$OUT/headstand_${k}_bfm_s0"
  if [ -f "$d/best_z.npy" ]; then echo "skip headstand_$k (already searched)"; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python scripts/single_z_search.py \
    --clip "headstand/headstand_$k" --body child \
    --objective bfm --init reference \
    --evals 4992 --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed 0 \
    --out "$d" > "$LOG/headstand_$k.log" 2>&1 &
  echo "launched headstand_$k (pid $!)"
done
wait
echo "=== all searches done ==="
