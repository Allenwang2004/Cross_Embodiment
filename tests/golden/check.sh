#!/bin/bash
# tests/golden/check.sh LABEL -- the refactor's per-step check: every golden case (tests/golden/run.sh LABEL, resource-gated),
# the Task-page metrics of the fixed rollouts (tests/golden/metrics_golden.py LABEL), then tests/golden/compare.py baseline LABEL.
# Exit code: 0 only if every case ran (exit 0) and everything is identical to outputs/refactor_golden/baseline.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[ "$REPO" != "/home/allen19/crossenbodiment" ] || { echo "refusing to run in the main tree" >&2; exit 3; }
[ $# -eq 1 ] || { echo "usage: $0 LABEL" >&2; exit 3; }
LABEL=$1; cd "$REPO" || exit 3
[ "$LABEL" != baseline ] || { echo "LABEL baseline is the reference; use another label" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1
OUT=outputs/refactor_golden/$LABEL; mkdir -p "$OUT"

tests/golden/run.sh "$LABEL"; run_rc=$?
[ $run_rc == 2 ] && { echo "run.sh stopped on low memory; not comparing" >&2; exit 2; }

# metrics: one CPU process (~1-2 min, < 2 GB); the same resource check before starting it
while :; do
    a=$(free -g | awk '/^Mem:/{print $7}')
    i=$(top -bn2 -d 2 | grep "%Cpu(s)" | tail -1 | awk -F, '{for (k = 1; k <= NF; k++) if ($k ~ / id/) {split($k, x, " "); print int(x[1])}}')
    [ "$a" -ge 52 ] && [ "$i" -ge 15 ] && break
    echo "$(date '+%m-%d %H:%M:%S') metrics waiting: available $a GB, CPU idle $i %" | tee -a "$OUT/run.log"; sleep 30
done
uv run python tests/golden/metrics_golden.py "$LABEL" > "$OUT/metrics.log" 2>&1; met_rc=$?
echo "$(date '+%m-%d %H:%M:%S') metrics_golden exit $met_rc" | tee -a "$OUT/run.log"

uv run python tests/golden/compare.py baseline "$LABEL" | tee "$OUT/compare.log"; cmp_rc=${PIPESTATUS[0]}
echo "$(date '+%m-%d %H:%M:%S') check $LABEL: run.sh exit $run_rc, metrics exit $met_rc, compare exit $cmp_rc" | tee -a "$OUT/run.log"
[ $run_rc == 0 ] && [ $met_rc == 0 ] && [ $cmp_rc == 0 ]
