#!/usr/bin/env bash
# docs/new_body.md steps 1, 2, 4, 6 for one body, then the z0 cost ranking.
# OOP refactor step 9: a thin wrapper. The body is built by scripts/build_body.py (crossenbodiment/bodybuild BodyPipeline:
# shape, skeleton, retarget every motion folder, torque ratio, aggregate k, robot_torque_full.xml; finished steps skipped);
# the z0 cost ranking below is unchanged. Skips step 3 (batch_infer_z) and write_body_splits, as before.
#   usage: scripts/new_body_pipeline.sh <label> <scale_robot axis flags...>
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=$1; shift
echo "[$(date +%T)] $ uv run python scripts/build_body.py scaled $LABEL $*"
uv run python scripts/build_body.py scaled "$LABEL" "$@"
echo "[$(date +%T)] $LABEL: z0 cost on 540 clips"
uv run scripts/rank_initial_cost.py --bodies $LABEL --device ${GPU:-cuda:1} --out outputs/initial_cost/$LABEL 2>&1 | grep -E "^$LABEL:"
echo "[$(date +%T)] $LABEL done"
