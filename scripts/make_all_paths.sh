#!/usr/bin/env bash
# Three more morphology paths from the adult, each in a different direction of
# body space, so the continuation result can be tested for generality rather
# than for "shrinking" alone:
#   m2s  limbs only        -> short_limbed  (torso and girth untouched)
#   m2g  everything bigger -> giant         (the opposite direction to child)
#   m2k  thicker           -> short_stocky  (girth up, lengths slightly down)
set -euo pipefail
cd "$(dirname "$0")/.."
PAR=${PAR:-3} bash scripts/make_morph_path.sh m2s 8 0.65 0.68 1.0 1.0 1.0 1.0 1.0 1.0
PAR=${PAR:-3} bash scripts/make_morph_path.sh m2g 8 1.25 1.25 1.2 1.05 1.15 1.15 1.15 1.05
PAR=${PAR:-3} bash scripts/make_morph_path.sh m2k 8 0.85 0.88 0.92 1.0 1.25 1.25 1.3 1.05
echo "=== all paths built ==="
