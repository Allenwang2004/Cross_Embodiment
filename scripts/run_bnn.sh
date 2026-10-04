#!/usr/bin/env bash
# run_bnn.sh <plan.tsv> <outdir> -- cold (from z0) vs beta-nearest-neighbour warm
# start, on bodies that are on no morph path. Same budget, same seed.
set -uo pipefail
cd "$(dirname "$0")/.."
PLAN=$1; OUT=$2; LOG=outputs/train_logs/$(basename "$OUT")
mkdir -p "$OUT" "$LOG"
PAR=${PAR:-3}; EV=${EV:-1024}
run () {  # clip body out [zstart]
  [ -f "$3/summary.json" ] && return 0
  local extra=(); [ -n "${4:-}" ] && extra=(--z-start "$4")
  CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python scripts/single_z_search.py \
    --clip "$1" --body "$2" --objective bfm --init reference \
    --evals "$EV" --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed 0 \
    "${extra[@]}" --out "$3" > "$LOG/$(basename "$3").log" 2>&1
}
while IFS=$'\t' read -r clip body nb dist zs; do
  stem=${clip##*/}
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  ( run "$clip" "$body" "$OUT/cold/${stem}__$body"
    run "$clip" "$body" "$OUT/warm/${stem}__$body" "$zs" ) &
done < "$PLAN"
wait
echo "=== bnn $(basename "$OUT") done ==="
