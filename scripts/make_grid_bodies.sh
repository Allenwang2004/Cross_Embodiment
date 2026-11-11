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
# OOP refactor step 9: a thin wrapper over scripts/build_body.py grid (crossenbodiment/bodybuild BodyPipeline; the grid values
# and labels are build_body.py GRID_LEGS / GRID_ARMS; finished steps are skipped, a body counts as done when its steps are).
# usage: scripts/make_grid_bodies.sh            (labels gl<leg*1000>a<arm*1000>)
set -euo pipefail
cd "$(dirname "$0")/.."
PAR=${PAR:-4}
CMD=(uv run python scripts/build_body.py grid --parallel "$PAR" --workers 1 --keep-tasks outputs/morph_study/keep_tasks.txt)
echo "$ ${CMD[*]}"
"${CMD[@]}"
echo "=== grid bodies built ==="
