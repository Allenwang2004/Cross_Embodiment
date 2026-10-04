#!/usr/bin/env bash
# The decisive control only: one long search from z0 on the target body with the
# same total budget continuation spent (9 x 512 = 4608). The restart and
# self-continuation arms already agreed with it on the first two clips.
set -uo pipefail
cd "$(dirname "$0")/.."
PREFIX=$1; CLIPS=$2; TARGET=${TARGET:-${PREFIX}_t1000}
OUT=outputs/continuation/$PREFIX/controls/long; LOG=outputs/train_logs/continuation/$PREFIX/controls
mkdir -p "$OUT" "$LOG"; PAR=${PAR:-2}
while read -r clip; do
  [ -z "$clip" ] && continue
  d="$OUT/${clip##*/}"; [ -f "$d/summary.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python scripts/single_z_search.py --clip "$clip" --body "$TARGET" \
    --objective bfm --init reference --evals 4608 --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed 0 \
    --out "$d" > "$LOG/long_${clip##*/}.log" 2>&1 &
done < "$CLIPS"
wait; echo "=== long controls done ==="
