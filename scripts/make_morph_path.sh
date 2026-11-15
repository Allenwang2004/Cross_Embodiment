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
# OOP refactor step 9: a thin wrapper over scripts/build_body.py morph (crossembodiment/bodybuild BodyPipeline; beta(t) rounded
# to 4 decimals as before: bodybuild/beta.py lerp_beta; labels <prefix>_t<i*1000/N>). KEEP_TASKS=<file>: after the torque
# calibration (on every motion) keep only those tasks' retargets. SKIP0=1: do not rebuild t = 0.
# usage: scripts/make_morph_path.sh <prefix> <N> <leg> <arm> <torso> <head> <legG> <armG> <torsoG> <headG>
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX=$1; N=$2; shift 2
PAR=${PAR:-3}
CMD=(uv run python scripts/build_body.py morph "$PREFIX" "$N" "$@" --parallel "$PAR" --workers 1)
[ -n "${KEEP_TASKS:-}" ] && CMD+=(--keep-tasks "$KEEP_TASKS")
[ "${SKIP0:-0}" != "0" ] && CMD+=(--skip0)
echo "$ ${CMD[*]}"
"${CMD[@]}"
echo "=== path $PREFIX built ==="
