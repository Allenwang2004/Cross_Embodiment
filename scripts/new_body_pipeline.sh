#!/usr/bin/env bash
# docs/new_body.md steps 1, 2, 4, 6 for one body, then the z0 cost ranking.
# Skips step 3 (batch_infer_z) and write_body_splits: neither is needed to
# score z0 or to run single_z_search on the body.
#   usage: scripts/new_body_pipeline.sh <label> <scale_robot axis flags...>
set -euo pipefail
LABEL=$1; shift
echo "[$(date +%T)] $LABEL: scale_robot $*"
uv run scripts/scale_robot.py --label $LABEL --no-actuator-scale "$@"
uv run scripts/export_skeleton_json.py --input assets/robots/$LABEL/robot.xml --output assets/robots/$LABEL/skeleton.json
echo "[$(date +%T)] $LABEL: retarget 54 tasks"
for d in data/origin_motion/*/; do
  m=$(basename "$d")
  uv run scripts/qpos_retarget.py --input_dir "data/origin_motion/$m" --output_dir "data/$LABEL/retargeting_motion/$m" \
    --source_skeleton_json assets/robots/adult/skeleton.json --target_skeleton_json assets/robots/$LABEL/skeleton.json \
    --target_xml assets/robots/$LABEL/robot.xml > /dev/null
done
echo "  clips: $(find data/$LABEL/retargeting_motion -name '*.npz' | wc -l)"
echo "[$(date +%T)] $LABEL: torque ratio + robots_torque_full.xml"
uv run scripts/torque_ratio_across_motions.py --origin data/origin_motion --retarget data/$LABEL/retargeting_motion \
  --adult-xml assets/robots/adult/robot.xml --child-xml assets/robots/$LABEL/robot.xml \
  --outdir outputs/torque_ratio_across_motions/gravity/$LABEL > /dev/null
mkdir -p assets/robots_torque/$LABEL
uv run scripts/torque_aggregate_motion_k.py --matrix outputs/torque_ratio_across_motions/gravity/$LABEL \
  --src assets/robots/$LABEL/robot.xml --out assets/robots_torque/$LABEL/robot_torque_full.xml --joint-dynamics \
  | grep -E "matches the law|tau=C/K|omega_n|dt\*sqrt|mirror"
echo "[$(date +%T)] $LABEL: z0 cost on 540 clips"
uv run scripts/rank_initial_cost.py --bodies $LABEL --device ${GPU:-cuda:1} --out outputs/initial_cost/$LABEL 2>&1 | grep -E "^$LABEL:"
echo "[$(date +%T)] $LABEL done"
