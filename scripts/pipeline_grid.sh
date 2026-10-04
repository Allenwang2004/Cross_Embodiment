#!/usr/bin/env bash
# pipeline_grid.sh -- leg x arm grid library: build the 64 off-axis bodies, then
# continue every clip's latent across them. Evaluation on the real bodies is
# scripts/grid_zero_shot.py once this is done.
set -uo pipefail
cd "$(dirname "$0")/.."
L=outputs/train_logs/pipeline_grid.log
echo "[$(date '+%F %T')] start" >> $L
PAR=4 bash scripts/make_grid_bodies.sh >> outputs/train_logs/make_grid_bodies.log 2>&1
echo "[$(date '+%F %T')] bodies built: $(ls -d assets/robots_torque/gl* | wc -l); disk $(df -h . | awk 'NR==2{print $4}') free" >> $L
GPU=${GPU:-1} PAR=4 bash scripts/run_grid_continuation.sh >> outputs/train_logs/continuation_grid_driver.log 2>&1
echo "[$(date '+%F %T')] continuation done: $(ls outputs/continuation/grid/cont/*/best_z.npy | wc -l) latents" >> $L
