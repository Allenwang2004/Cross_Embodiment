#!/usr/bin/env bash
# run_warm_start_ab.sh -- does starting the latent search from a learned prior
# buy search time?
#
# The pivot this belongs to: stop asking the adapter to BE the answer and ask it
# to be a starting point. Every negative result so far supports the switch --
# the adapter sits 8.3 deg from a good z and headstand's usable window is a few
# degrees wide, so as an answer it is worthless; but 8 degrees closer to the
# target than z0 is still 8 degrees of search someone does not have to do.
#
# A/B on the same clips, same budget, same seed, same everything except where
# the search begins:
#   cold  from the clip's z0 (what every search in this repo has done)
#   warm  from bc1.0's adapter output for that clip
# The reported origin_z baseline stays the real z0 in both, so the curves are
# directly comparable and "evals to reach ratio X" is well defined.
#
# NOTE this adapter was trained on child alone, so beta is constant in it --
# this measures whether a LEARNED PRIOR helps at all, which is the enabling
# fact. Whether BETA specifically accelerates the search needs the multi-body
# run and is the next step.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/warm_start
LOG=outputs/train_logs/warm_start
mkdir -p "$LOG"
PAR=${PAR:-3}
CLIPS=${CLIPS:-"2 3 4 9 15 20 23 30 34 37"}

for k in $CLIPS; do
  for cond in cold warm; do
    d="$OUT/headstand_${k}_${cond}"
    [ -f "$d/summary.json" ] && { echo "skip $k/$cond"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    extra=()
    [ "$cond" = warm ] && extra=(--z-start "$OUT/z/headstand_$k.npy")
    CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python scripts/single_z_search.py \
      --clip "headstand/headstand_$k" --body child \
      --objective bfm --init reference \
      --evals 2496 --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed 0 \
      "${extra[@]}" --out "$d" > "$LOG/headstand_${k}_${cond}.log" 2>&1 &
    echo "launched $k/$cond (pid $!)"
  done
done
wait
echo "=== warm-start A/B done ==="
