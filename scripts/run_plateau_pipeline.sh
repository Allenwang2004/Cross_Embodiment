#!/usr/bin/env bash
# 20 bfm seeds of single_z_search for one (body, clip) at the s005_5k settings,
# then the plateau analysis of scripts/walk_z_plateau.py -- the exact recipe
# used for child x move-ego-0-2_4.
#   usage: scripts/run_plateau_pipeline.sh <task>/<stem> [body] [gpu] [parallel]
set -euo pipefail
CLIP=$1; BODY=${2:-child}; GPU=${3:-1}; PAR=${4:-4}
STEM=${CLIP#*/}
ROOT=outputs/single_z_seeds_s005_5k
LOG=outputs/train_logs/plateau; mkdir -p $LOG
BANDS="1-8 9-16 17-24 25-32 33-40 41-48 49-56 57-64 65-72 73-80 81-88 89-96 97-112 113-128 129-144 145-160 161-192 193-224 225-255"

echo "[$(date +%T)] seeds for $CLIP on $BODY"
for s in $(seq 0 19); do
    OUT=$ROOT/${STEM}_bfm_s$s
    [ -f $OUT/best_z.npy ] && { echo "  s$s done already"; continue; }
    uv run scripts/single_z_search.py --clip $CLIP --body $BODY --objective bfm \
        --evals 5000 --pairs 8 --sigma 0.05 --lr 0.1 --eval-every 25 --init reference \
        --seed $s --device cuda:$GPU --out $OUT > $LOG/${STEM}_${BODY}_s$s.log 2>&1 &
    if (( (s + 1) % PAR == 0 )); then wait; echo "[$(date +%T)]   seeds up to s$s done"; fi
done
wait
echo "[$(date +%T)] plateau analysis"
uv run scripts/walk_z_plateau.py --clip $STEM --root $ROOT --sampler trace --every 2 --thr-rel 1.1 \
    --ks 32 --bands $BANDS --angles 10 20 30 40 50 60 70 --ndir 8 --device cuda:$GPU \
    > $LOG/${STEM}_${BODY}_plateau.log 2>&1
echo "[$(date +%T)] done -> $ROOT/plateau_$STEM/"
