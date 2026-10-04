#!/usr/bin/env bash
# make_grid_bodies.sh -- bodies on a leg-length x arm-length grid, every other
# body parameter at the adult's 1.0, built by the same pipeline as the morph paths
# (make_morph_path.sh): scale, skeleton, retarget every motion, torque-calibrate on
# all of them, then keep only the study's tasks.
#
# Why: the 16 axis paths showed the latent's sensitivity to the body is almost all
# limb length (leg 0.5x -> 2.72x z0 cost, arm 0.55x -> 1.68x, everything else
# <= 1.07x except thin legs), yet real bodies change leg AND arm at once and sit
# between the single-axis paths, so zero-shot prediction on them recovered only
# ~20-30%. A grid in the two limb lengths gives every real body library bodies on
# all sides.
#
# The grid's leg and arm values are exactly the ones on the existing axis paths
# (ax0m/ax0p for legs, ax1m/ax1p for arms), so the grid lines through the adult
# are already built and solved; this builds the 64 off-axis points.
#
# usage: scripts/make_grid_bodies.sh            (labels gl<leg*1000>a<arm*1000>)
set -euo pipefail
cd "$(dirname "$0")/.."
PAR=${PAR:-4}
LEGS=(0.5 0.625 0.75 0.875 1.125 1.25 1.375 1.5)
ARMS=(0.55 0.6625 0.775 0.8875 1.1 1.2 1.3 1.4)
label_of () { python3 -c "print(f'gl{round($1*1000):04d}a{round($2*1000):04d}')"; }
build () {
  local leg=$1 arm=$2 label; label=$(label_of "$leg" "$arm")
  if [ -f "assets/robots_torque/$label/robot_torque_full.xml" ] && \
     [ -d "data/$label/retargeting_motion" ]; then
    echo "skip $label"; return
  fi
  echo "[$(date +%T)] $label  leg=$leg arm=$arm"
  uv run scripts/scale_robot.py --label "$label" --no-actuator-scale \
    --leg-scale "$leg" --arm-scale "$arm" --torso-scale 1.0 --head-scale 1.0 \
    --leg-girth 1.0 --arm-girth 1.0 --torso-girth 1.0 --head-girth 1.0 > /dev/null
  uv run scripts/export_skeleton_json.py --input "assets/robots/$label/robot.xml" \
    --output "assets/robots/$label/skeleton.json" > /dev/null
  for d in data/origin_motion/*/; do
    m=$(basename "$d")
    uv run scripts/qpos_retarget.py --input_dir "data/origin_motion/$m" \
      --output_dir "data/$label/retargeting_motion/$m" \
      --source_skeleton_json assets/robots/adult/skeleton.json \
      --target_skeleton_json "assets/robots/$label/skeleton.json" \
      --target_xml "assets/robots/$label/robot.xml" > /dev/null
  done
  uv run scripts/torque_ratio_across_motions.py --origin data/origin_motion \
    --retarget "data/$label/retargeting_motion" --adult-xml assets/robots/adult/robot.xml \
    --child-xml "assets/robots/$label/robot.xml" \
    --outdir "outputs/torque_ratio_across_motions/gravity/$label" > /dev/null
  mkdir -p "assets/robots_torque/$label"
  uv run scripts/torque_aggregate_motion_k.py --matrix "outputs/torque_ratio_across_motions/gravity/$label" \
    --src "assets/robots/$label/robot.xml" --out "assets/robots_torque/$label/robot_torque_full.xml" \
    --joint-dynamics > /dev/null
  for d in data/$label/retargeting_motion/*/; do
    grep -qx "$(basename "$d")" outputs/morph_study/keep_tasks.txt || rm -rf "$d"
  done
  echo "[$(date +%T)] $label done: $(find data/$label/retargeting_motion -name '*.npz' | wc -l) clips kept"
}
for leg in "${LEGS[@]}"; do
  for arm in "${ARMS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    build "$leg" "$arm" &
  done
done
wait
echo "=== grid bodies built ==="
