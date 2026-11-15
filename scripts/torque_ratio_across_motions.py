#!/usr/bin/env python3
"""torque_ratio_across_motions.py — is the per-joint child/adult torque ratio a
property of the BODY, or of the motion?

scripts/torque_ratio_per_joint.py answers "does a ratio k exist" for one clip.
This asks the follow-up: run that same fit on the first clip of every motion in
data/origin_motion, giving one 69-dim k vector per motion, and correlate those
vectors against each other. If k is a body property the 54 vectors are nearly
identical and the correlation matrix is uniformly ~1; wherever it is not, the
motion is changing which joints carry the load, and a single scaled MJCF cannot
serve every motion equally.

Same measurement as the per-joint script (both use joint_torques / fit_ratio of
crossembodiment/bodybuild/torque.py), so the k column of any single motion here reproduces that
script's _ratios.csv for the same clip. No per-joint figures are drawn.

OOP refactor step 9: the code is crossembodiment/bodybuild/torque.py TorqueCalibrator.ratio_matrix (the "torque_ratio" step
of scripts/build_body.py); this file is its old CLI, same flags, same files. Always pass --outdir <mode>/<body>.

Outputs (default outputs/torque_ratio_across_motions/<mode>/):
  _k_matrix.csv   motions x 69 fitted k          (plus R^2 in _r2_matrix.csv)
  _corr.csv       motions x motions correlation of those k vectors
  _corr.png       that matrix as a heatmap

Usage:
  uv run scripts/torque_ratio_across_motions.py
  uv run scripts/torque_ratio_across_motions.py --metric spearman
  uv run scripts/torque_ratio_across_motions.py --r2-min 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossembodiment.bodybuild.torque import (TorqueCalibrator, correlate, draw_corr, first_clip,  # noqa: E402,F401
                                              k_vector, rankdata)

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--origin", default=str(ROOT / "data/origin_motion"))
    p.add_argument("--retarget", default=str(ROOT / "data/retargeting_motion"))
    p.add_argument("--adult-xml", default=str(ROOT / "assets/robots/adult/robot.xml"))
    p.add_argument("--child-xml", default=str(ROOT / "assets/robots/child/robot.xml"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--mode", choices=["gravity", "quasi", "full"], default="gravity")
    p.add_argument("--metric", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--r2-min", type=float, default=0.0,
                   help="drop any joint whose fit falls below this R^2 in ANY "
                        "motion, so the correlation is over joints where k is "
                        "actually meaningful everywhere")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    outdir = Path(args.outdir or ROOT / "outputs/torque_ratio_across_motions" / args.mode)
    cal = TorqueCalibrator(args.child_xml, args.retarget, outdir, adult_xml=args.adult_xml, origin=args.origin,
                           mode=args.mode, fps=args.fps)
    cal.ratio_matrix(metric=args.metric, r2_min=args.r2_min)


if __name__ == "__main__":
    main()
