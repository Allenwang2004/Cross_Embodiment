#!/usr/bin/env bash
# run_continuation.sh -- track the best latent along a path of bodies.
#
# For each clip, two ways of finding z* on every body of the path, same budget
# per body, same seed:
#   indep  every body searched from the clip's z0 (what every search so far did)
#   cont   body k searched from body k-1's best z (homotopy continuation in beta)
#
# The question is not only which is cheaper: it is whether the z* found are a
# CURVE. Independent searches each stop at an arbitrary point of a large
# low-cost set -- on a fixed body that made the targets for different motions
# mutually orthogonal and unlearnable. Continuation picks one connected branch,
# which is the only way a smooth function of beta could exist to be learned.
#
# usage: scripts/run_continuation.sh <path-prefix> <clip-list-file>
set -uo pipefail
cd "$(dirname "$0")/.."
PREFIX=$1; CLIPS=$2
OUT=outputs/continuation/$PREFIX
LOG=outputs/train_logs/continuation/$PREFIX
mkdir -p "$OUT" "$LOG"
EVALS=${EVALS:-512}
PAR=${PAR:-4}
# numeric order on the t value: a string sort puts <prefix>_t1000 right after
# <prefix>_t000, and the continuation would then jump straight to the target
BODIES=$(ls -d assets/robots_torque/${PREFIX}_t* | xargs -n1 basename \
         | awk -F_t '{print $NF" "$0}' | sort -n | cut -d" " -f2)

search () {   # search <clip> <body> <outdir> [zstart]
  local clip=$1 body=$2 d=$3 zs=${4:-}
  [ -f "$d/summary.json" ] && return 0
  local extra=()
  [ -n "$zs" ] && extra=(--z-start "$zs")
  CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python scripts/single_z_search.py \
    --clip "$clip" --body "$body" --objective bfm --init reference \
    --evals "$EVALS" --pairs 8 --sigma 0.05 --lr 0.1 --steps 300 --seed 0 \
    "${extra[@]}" --out "$d" > "$LOG/$(basename "$d").log" 2>&1
}

one_clip () {
  local clip=$1 stem=${1##*/}
  # continuation: sequential along the path
  local prev=""
  # a path built with SKIP0 has no t=0 body: continue from the shared adult
  # solution instead of from z0
  if [ -n "${SEED_PREFIX:-}" ]; then
    local sd="outputs/continuation/$SEED_PREFIX/cont/${stem}__${SEED_PREFIX}_t000/best_z.npy"
    [ -f "$sd" ] && prev="$sd"
  fi
  for b in $BODIES; do
    local d="$OUT/cont/${stem}__$b"
    search "$clip" "$b" "$d" "$prev"
    prev="$d/best_z.npy"
  done
  # independent: every body from z0 (order irrelevant); INDEP=0 skips it
  [ "${INDEP:-1}" = 0 ] && { echo "[$(date +%T)] $stem done"; return; }
  for b in $BODIES; do
    search "$clip" "$b" "$OUT/indep/${stem}__$b"
  done
  echo "[$(date +%T)] $stem done"
}

while read -r clip; do
  [ -z "$clip" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  one_clip "$clip" &
done < "$CLIPS"
wait
echo "=== continuation $PREFIX done ==="
