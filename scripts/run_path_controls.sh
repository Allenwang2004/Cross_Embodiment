#!/usr/bin/env bash
# run_path_controls.sh -- is it the MORPHOLOGY PATH, or just more search?
#
# Continuation along m2c reached best/z0 = 0.33 on the target body where an
# independent 512-eval search from z0 reached 0.74. But continuation also spent
# 9 x 512 = 4608 evals getting there. Three controls on the TARGET body, each
# with the same 4608 total:
#   long     one search from z0, 4608 evals               (just a bigger budget)
#   restart  9 searches from z0 with different seeds, best (just more tries)
#   self     9 sequential searches ON THE TARGET BODY, each from the previous
#            best -- continuation with the body held fixed. This separates "the
#            path through bodies" from "restarting the optimiser 9 times"
# Continuation only means something if it beats all three.
set -uo pipefail
cd "$(dirname "$0")/.."
PREFIX=$1; CLIPS=$2; TARGET=${TARGET:-${PREFIX}_t1000}
OUT=outputs/continuation/$PREFIX/controls
LOG=outputs/train_logs/continuation/$PREFIX/controls
mkdir -p "$OUT" "$LOG"
PAR=${PAR:-3}; STEPS=${STEPS:-9}; EV=${EV:-512}

search () {  # clip body out evals seed [zstart]
  local d=$3; [ -f "$d/summary.json" ] && return 0
  local extra=(); [ -n "${6:-}" ] && extra=(--z-start "$6")
  CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python scripts/single_z_search.py \
    --clip "$1" --body "$2" --objective bfm --init reference \
    --evals "$4" --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed "$5" \
    "${extra[@]}" --out "$d" > "$LOG/$(basename "$d").log" 2>&1
}
one () {
  local clip=$1 stem=${1##*/}
  search "$clip" "$TARGET" "$OUT/long/${stem}" $((EV * STEPS)) 0
  for s in $(seq 1 "$STEPS"); do search "$clip" "$TARGET" "$OUT/restart/${stem}__s$s" "$EV" "$s"; done
  local prev=""
  for k in $(seq 1 "$STEPS"); do
    search "$clip" "$TARGET" "$OUT/self/${stem}__k$k" "$EV" 0 "$prev"
    prev="$OUT/self/${stem}__k$k/best_z.npy"
  done
  echo "[$(date +%T)] $stem controls done"
}
while read -r clip; do
  [ -z "$clip" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  one "$clip" &
done < "$CLIPS"
wait
echo "=== controls $PREFIX done ==="
