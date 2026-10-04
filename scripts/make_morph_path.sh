#!/usr/bin/env bash
# make_morph_path.sh -- a straight line in body-parameter space from the adult
# (the body the behaviour foundation model was trained on, beta = 1) to a target
# body, sampled at N+1 evenly spaced points, each built by the SAME pipeline so
# the only thing that varies along the path is beta.
#
#   beta(t) = (1 - t) * 1 + t * beta_target,   t = 0, 1/N, ..., 1
#
# Why: on a fixed body the best latent for different MOTIONS was measured to be
# mutually unrelated (pairwise cos +0.003, no transfer between clips). Whether
# the best latent for different BODIES is also unrelated -- or traces a smooth
# curve in beta -- is the question the paper's axis rests on, and it needs a
# family of bodies that differ in beta and nothing else.
#
# t = 0 is rebuilt through the pipeline too rather than borrowed from
# assets/robots/adult, so its actuators come from the same torque calibration as
# every other point on the path; otherwise the first step would mix a beta change
# with a pipeline change.
#
# usage: scripts/make_morph_path.sh <prefix> <N> <leg> <arm> <torso> <head> <legG> <armG> <torsoG> <headG>
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX=$1; N=$2; shift 2
TGT=("$@")
AX=(leg-scale arm-scale torso-scale head-scale leg-girth arm-girth torso-girth head-girth)
PAR=${PAR:-3}
build () {
  local i=$1
  local t; t=$(python3 -c "print($i/$N)")
  local label; label=$(printf "%s_t%03d" "$PREFIX" "$((i * 1000 / N))")
  local flags=()
  for j in 0 1 2 3 4 5 6 7; do
    v=$(python3 -c "print(round((1-$t)*1.0 + $t*${TGT[$j]}, 4))")
    flags+=(--"${AX[$j]}" "$v")
  done
  if [ -f "assets/robots_torque/$label/robot_torque_full.xml" ] && \
     [ "$(find data/$label/retargeting_motion -name '*.npz' 2>/dev/null | wc -l)" -ge 540 ]; then
    echo "skip $label"; return
  fi
  echo "[$(date +%T)] $label  t=$t  ${flags[*]}"
  uv run scripts/scale_robot.py --label "$label" --no-actuator-scale "${flags[@]}" > /dev/null
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
  if [ -n "${KEEP_TASKS:-}" ]; then
    # Torque calibration above used EVERY motion (so actuators are scaled the same
    # way as on every other body); after it, only the tasks the study rolls out
    # are kept -- a full retarget is ~200 MB per body and the disk is nearly full.
    for d in data/$label/retargeting_motion/*/; do
      grep -qx "$(basename "$d")" "$KEEP_TASKS" || rm -rf "$d"
    done
  fi
  echo "[$(date +%T)] $label done: $(find data/$label/retargeting_motion -name '*.npz' | wc -l) clips kept"
}
# SKIP0=1: do not rebuild t=0 -- every path from the adult starts at the same
# body, whose solutions already exist (m2c_t000)
for i in $(seq "${SKIP0:-0}" "$N"); do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  build "$i" &
done
wait
echo "=== path $PREFIX built ==="
