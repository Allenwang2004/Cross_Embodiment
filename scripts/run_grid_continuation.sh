#!/usr/bin/env bash
# run_grid_continuation.sh -- continuation latents on the leg x arm grid
# (make_grid_bodies.sh), same search settings as run_continuation.sh.
#
# Every grid row has a fixed leg length. Its arm = 1 body is already solved on the
# leg axis path (ax0m / ax0p, itself continued from the adult), so each row is
# continued from there outward in arm length, both ways:
#
#   (leg, 1) -> (leg, 0.8875) -> (leg, 0.775) -> (leg, 0.6625) -> (leg, 0.55)
#   (leg, 1) -> (leg, 1.1)    -> (leg, 1.2)   -> (leg, 1.3)    -> (leg, 1.4)
#
# 8 legs x 2 directions x 8 clips = 128 independent chains of 4 searches.
#
# usage: GPU=1 PAR=4 scripts/run_grid_continuation.sh [clip-list]
set -uo pipefail
cd "$(dirname "$0")/.."
CLIPS=${1:-outputs/continuation_clips.txt}
OUT=outputs/continuation/grid/cont
LOG=outputs/train_logs/continuation/grid
mkdir -p "$OUT" "$LOG"
EVALS=${EVALS:-512}; PAR=${PAR:-4}; GPU=${GPU:-1}
label_of () { python3 -c "print(f'gl{round($1*1000):04d}a{round($2*1000):04d}')"; }
seed_of () {   # seed_of <stem> <leg>: the solved (leg, arm=1) body on the leg axis
  local stem=$1 leg=$2
  python3 - "$stem" "$leg" <<'PY'
import sys
stem, leg = sys.argv[1], float(sys.argv[2])
pre = "ax0m" if leg < 1 else "ax0p"
t = round(abs(leg - 1) / 0.5 * 1000)
print(f"outputs/continuation/{pre}/cont/{stem}__{pre}_t{t:03d}/best_z.npy")
PY
}
chain () {   # chain <clip> <leg> <arm values...>
  local clip=$1 leg=$2; shift 2
  local stem=${clip##*/} prev; prev=$(seed_of "$stem" "$leg")
  [ -f "$prev" ] || { echo "missing seed $prev"; return; }
  for arm in "$@"; do
    local b; b=$(label_of "$leg" "$arm")
    local d="$OUT/${stem}__$b"
    if [ ! -f "$d/summary.json" ]; then
      CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/single_z_search.py \
        --clip "$clip" --body "$b" --objective bfm --init reference \
        --evals "$EVALS" --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed 0 \
        --z-start "$prev" --out "$d" > "$LOG/${stem}__$b.log" 2>&1
    fi
    prev="$d/best_z.npy"
  done
  echo "[$(date +%T)] $stem leg=$leg arms $* done"
}
for clip in $(cat "$CLIPS"); do
  for leg in 0.5 0.625 0.75 0.875 1.125 1.25 1.375 1.5; do
    for dir in down up; do
      while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
      if [ $dir = down ]; then chain "$clip" "$leg" 0.8875 0.775 0.6625 0.55 &
      else chain "$clip" "$leg" 1.1 1.2 1.3 1.4 & fi
    done
  done
done
wait
echo "=== grid continuation done ==="
