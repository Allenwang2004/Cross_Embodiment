#!/usr/bin/env bash
set -uo pipefail; cd /home/allen19/crossenbodiment
L=outputs/train_logs/pipeline_morph.log; log(){ echo "[$(date +%F" "%T)] $*" >> $L; }
REAL="athletic child elderly giant long_limbed pear_shaped petite short_limbed short_stocky tall_slim teen x_leg050 x_leg055_adulttorso x_leg060_longtorso x_leg062_heavy x_leg070 x_leg075_heavy x_leg140 x_leg150_thin"
log "resume at analyses"
# 3. analyses for every path
for p in m2c m2s m2g m2k; do
  uv run python scripts/analyze_continuation.py --prefix $p > outputs/morph_study/continuation_$p.txt 2>&1
  uv run python scripts/build_transfer_jobs.py --prefix $p --out outputs/morph_study/jobs_transfer_$p.npz >/dev/null 2>&1
  [ -f outputs/morph_study/transfer_$p.csv ] || uv run python scripts/score_z_matrix.py --jobs outputs/morph_study/jobs_transfer_$p.npz --out outputs/morph_study/transfer_$p.csv --with-z0 --chunk 24 > outputs/train_logs/transfer_$p.log 2>&1
  uv run python scripts/analyze_transfer.py --prefix $p > outputs/morph_study/transfer_$p.txt 2>&1
  uv run python scripts/build_interp_jobs.py --prefix $p --method cont --out outputs/morph_study/jobs_interp_${p}_cont.npz >/dev/null 2>&1
  uv run python scripts/build_interp_jobs.py --prefix $p --method indep --out outputs/morph_study/jobs_interp_${p}_indep.npz >/dev/null 2>&1
  uv run python - <<PY
import numpy as np
a=np.load('outputs/morph_study/jobs_interp_${p}_cont.npz'); b=np.load('outputs/morph_study/jobs_interp_${p}_indep.npz')
np.savez('outputs/morph_study/jobs_interp_${p}.npz', **{k: np.concatenate([a[k], b[k]]) for k in ('clip','body','label','z')})
PY
  [ -f outputs/morph_study/interp_$p.csv ] || uv run python scripts/score_z_matrix.py --jobs outputs/morph_study/jobs_interp_$p.npz --out outputs/morph_study/interp_$p.csv --with-z0 --chunk 24 > outputs/train_logs/interp_$p.log 2>&1
  uv run python scripts/analyze_interp.py --csv outputs/morph_study/interp_$p.csv > outputs/morph_study/interp_$p.txt 2>&1
  log "analysed $p"
done
# 4. the beta -> latent map from all four paths, zero-shot on the 19 real bodies
for m in cont indep; do
  uv run python scripts/fit_beta_map.py --method $m --test-bodies $REAL --out outputs/morph_study/jobs_map_$m.npz > outputs/morph_study/fit_map_$m.txt 2>&1
done
uv run python - <<'PY'
import numpy as np
a=np.load('outputs/morph_study/jobs_map_cont.npz'); b=np.load('outputs/morph_study/jobs_map_indep.npz')
np.savez('outputs/morph_study/jobs_map.npz', **{k: np.concatenate([a[k], b[k]]) for k in ('clip','body','label','z')})
PY
uv run python scripts/score_z_matrix.py --jobs outputs/morph_study/jobs_map.npz --out outputs/morph_study/map_real.csv --with-z0 --chunk 24 > outputs/train_logs/map_real.log 2>&1
uv run python scripts/analyze_interp.py --csv outputs/morph_study/map_real.csv > outputs/morph_study/map_real.txt 2>&1
log "map zero-shot done"
# 5. warm start from the full library, on the real bodies (cold runs reused where they exist)
uv run python scripts/plan_bnn.py --test-bodies $REAL --out outputs/morph_study/plan_bnn_all.tsv > outputs/morph_study/plan_bnn_all.txt 2>&1
EV=1024 PAR=4 bash scripts/run_bnn.sh outputs/morph_study/plan_bnn_all.tsv outputs/bnn/all_lib > outputs/train_logs/bnn_all_driver.log 2>&1
uv run python scripts/analyze_bnn.py --plan outputs/morph_study/plan_bnn_all.tsv --dir outputs/bnn/all_lib > outputs/morph_study/bnn_all.txt 2>&1
log "ALL DONE"
