#!/usr/bin/env bash
# run_warm_start_ab2.sh -- the same cold/warm A/B, on clips where the prior is
# actually GOOD.
#
# The headstand A/B came back negative (warm 0.69x the speed of cold), but
# headstand is the case where the prior is worse than doing nothing: bc1.0's
# adapter scores 1.042 there. A prior below z0 cannot help by construction, so
# that run did not test the idea. On move the same adapter reaches 0.532 and its
# warm start sits 36-45 deg from z0, so this is the fair test.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/warm_start
LOG=outputs/train_logs/warm_start
mkdir -p "$LOG"
PAR=${PAR:-3}
while read -r clip; do
  [ -z "$clip" ] && continue
  stem="${clip##*/}"
  for cond in cold warm; do
    d="$OUT/${stem}_${cond}"
    [ -f "$d/summary.json" ] && { echo "skip $stem/$cond"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    extra=()
    [ "$cond" = warm ] && extra=(--z-start "$OUT/z/${stem}.npy")
    CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python scripts/single_z_search.py \
      --clip "$clip" --body child --objective bfm --init reference \
      --evals 2496 --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed 0 \
      "${extra[@]}" --out "$d" > "$LOG/${stem}_${cond}.log" 2>&1 &
    echo "launched $stem/$cond"
  done
done < "${1:-$OUT/move_clips.txt}"
wait
echo "=== move warm-start A/B done ==="
