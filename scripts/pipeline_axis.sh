#!/usr/bin/env bash
# pipeline_axis.sh -- fair coverage of body-parameter space.
# 16 axis-aligned paths from the adult: for each of the 8 beta dimensions, one
# path to the smallest and one to the largest value that dimension takes among
# the 19 real bodies, with every other dimension held at 1. Same 4 steps on every
# path, so each direction is covered over its full observed range at the same
# relative resolution.
set -uo pipefail
cd "$(dirname "$0")/.."
L=outputs/train_logs/pipeline_axis.log; log(){ echo "[$(date +%F' '%T)] $*" >> $L; }
REAL="athletic child elderly giant long_limbed pear_shaped petite short_limbed short_stocky tall_slim teen x_leg050 x_leg055_adulttorso x_leg060_longtorso x_leg062_heavy x_leg070 x_leg075_heavy x_leg140 x_leg150_thin"
log start
uv run python - > outputs/morph_study/axis_targets.tsv <<PY
import sys, numpy as np; sys.path.append('.')
from model.dataset import load_beta
M=np.stack([load_beta(f"assets/robots/{b}/parameter.json") for b in "$REAL".split()])
for d in range(8):
    for tag,v in (("m",M[:,d].min()),("p",M[:,d].max())):
        t=np.ones(8); t[d]=v
        print(f"ax{d}{tag}\t"+" ".join(f"{x:.4f}" for x in t))
PY
log "targets: $(wc -l < outputs/morph_study/axis_targets.tsv) paths"
while IFS=$'\t' read -r pre vals; do
  SKIP0=1 KEEP_TASKS=outputs/morph_study/keep_tasks.txt PAR=4 bash scripts/make_morph_path.sh $pre 4 $vals >> outputs/train_logs/make_axis_paths.log 2>&1
done < outputs/morph_study/axis_targets.tsv
log "bodies built; disk $(df -h . | tail -1 | awk '{print $4}') free"
AX=$(cut -f1 outputs/morph_study/axis_targets.tsv | tr '\n' ' ')
for p in $AX; do
  INDEP=0 SEED_PREFIX=m2c EVALS=512 PAR=4 bash scripts/run_continuation.sh $p outputs/continuation_clips.txt > outputs/train_logs/continuation_${p}_driver.log 2>&1
  log "continuation $p done"
done
ALL="m2c m2s m2g m2k $AX"
uv run python scripts/fit_beta_map.py --prefixes $ALL --method cont --test-bodies $REAL --out outputs/morph_study/jobs_map_axis.npz > outputs/morph_study/fit_map_axis.txt 2>&1
uv run python scripts/score_z_matrix.py --jobs outputs/morph_study/jobs_map_axis.npz --out outputs/morph_study/map_real_axis.csv --with-z0 --chunk 24 > outputs/train_logs/map_real_axis.log 2>&1
log "map zero-shot done"
uv run python scripts/plan_bnn.py --prefixes $ALL --test-bodies $REAL --out outputs/morph_study/plan_bnn_axis.tsv > outputs/morph_study/plan_bnn_axis.txt 2>&1
mkdir -p outputs/bnn/axis_lib; [ -e outputs/bnn/axis_lib/cold ] || ln -s ../all_lib/cold outputs/bnn/axis_lib/cold
EV=1024 PAR=4 bash scripts/run_bnn.sh outputs/morph_study/plan_bnn_axis.tsv outputs/bnn/axis_lib > outputs/train_logs/bnn_axis_driver.log 2>&1
uv run python scripts/analyze_bnn.py --plan outputs/morph_study/plan_bnn_axis.tsv --dir outputs/bnn/axis_lib > outputs/morph_study/bnn_axis.txt 2>&1
log "ALL DONE"
