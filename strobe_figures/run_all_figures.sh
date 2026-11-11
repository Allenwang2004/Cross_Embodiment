#!/usr/bin/env bash
# Re-render every frozen figure recipe in configs/strobe/.
#
#   bash run_all_figures.sh                      -> outputs/figures/strobe
#   OUT_DIR=figures_check bash run_all_figures.sh
#   MUJOCO_GL=osmesa bash run_all_figures.sh    # if EGL is unavailable
#
# Recipes without a `replay:` block in their YAML were made by the rollout path
# (render_strobe.py + the frozen MetaMotivo actor) and are skipped here: they
# need the full repository, not this package. The skip list is printed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
PY="${PYTHON:-python}"
OUT="${OUT_DIR:-outputs/figures/strobe}"

shopt -s nullglob
configs=(configs/strobe/*.yaml configs/strobe/*/*.yaml)

# Frozen recipes that must NOT be replayed: their YAML was later re-pointed at a
# different clip than the published figure of the same task_name, so replaying
# them overwrites a paper figure with an unrelated one. See README "Stale
# recipes".
STALE=("configs/strobe/headstand_strobe.yaml")

ok=0
skip=0
for cfg in "${configs[@]}"; do
  for s in "${STALE[@]}"; do
    if [ "$cfg" = "$s" ]; then
      echo "SKIP $cfg  (stale recipe: task_name and replay.clip disagree with the published figure)"
      skip=$((skip + 1))
      continue 2
    fi
  done
  if ! "$PY" - "$cfg" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
raise SystemExit(0 if (cfg.get("replay") or {}).get("clip") else 1)
PYEOF
  then
    echo "SKIP $cfg  (no replay.clip -> rollout path, needs the full repo)"
    skip=$((skip + 1))
    continue
  fi
  echo "================================================================"
  echo "RUN  $cfg"
  if ! "$PY" replay_strobe.py --config "$cfg" --out-dir "$OUT"; then
    echo "FAILED $cfg"
    skip=$((skip + 1))
    continue
  fi
  ok=$((ok + 1))
done

echo "================================================================"
echo "done: $ok recipe(s) rendered into $OUT, $skip skipped/failed"
