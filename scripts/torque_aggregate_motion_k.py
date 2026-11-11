#!/usr/bin/env python3
"""torque_aggregate_motion_k.py (first named aggregate_motion_k.py) — one actuator scaling from all 54 motions, after the
motions that disagree with the rest are thrown out.

The one-clip route (archive/scripts/torque_scale_actuators.py) built robots_torque.xml from ONE clip's fit, so
whatever that clip happens to load determines the whole body's actuator sizing.
scripts/torque_ratio_across_motions.py showed that is not safe: the k vectors of
the 54 motions correlate at median 0.93 but down to 0.53, and the disagreement
is concentrated in the torso (Chest_y varies 119% across motions). This script
takes that same k matrix, drops the motions that sit clearly outside the pack,
averages what remains, and writes the scaled MJCF.

Which motions get dropped, and why by this rule
------------------------------------------------
Each motion's mean correlation to the other 53 is a single number saying "does
this motion agree with the consensus". They form ONE continuous run from 0.79 to
0.95 -- there is no bimodal gap, so no clustering method would find a clean
"other group" either; the honest reading is a tail, not a second cluster. So the
cut is a robust outlier rule on that number: drop anything more than --z scaled
MADs below the median (median absolute deviation x 1.4826, so --z is in
normal-sigma units). At the default --z 3 that removes the arm-raising cluster
plus move-ego-0-0, which is exactly the tail the heatmap shows as green.

Per-joint, not just per-motion
------------------------------
A kept motion can still have one meaningless joint: a joint carrying almost no
gravity torque in that clip fits an arbitrary k at R^2~0. So a joint's average
uses only the kept motions where that joint ALSO cleared --r2-min, and a joint
with no such motion anywhere falls back to k_predicted_subtree (downstream
subtree mass x lever arm), the same positive-by-construction fallback
the one-clip route used.

Left and right get the same k here too
--------------------------------------
Averaging over motions shrinks the left/right disagreement but does not remove
it -- the 54 clips are individually one-sided and the set is not balanced -- so
the mirrored pairs are equalised after the average, by the same rule and the same
code as the one-clip route (torque.symmetrize_k). See its docstring for why the body's own
symmetry (k_predicted_subtree agrees L/R to 0.026%) makes this the correct
reading and the per-actuator spread the artefact.

OOP refactor step 9: the code is crossenbodiment/bodybuild/torque.py TorqueCalibrator (aggregate, write_xml with
write_scaled_xml, motion_choice: the "aggregate" and "physics" steps of scripts/build_body.py); this file is its old CLI,
same flags, same _ratios_aggregate.csv and XML. --joint-dynamics is what every body uses (armature x Ir, damping x
sqrt(k Ir), stiffness x k from the adult; see torque.py).

Usage:
  uv run scripts/torque_aggregate_motion_k.py
  uv run scripts/torque_aggregate_motion_k.py --z 2.5 --agg median
  uv run scripts/torque_aggregate_motion_k.py --out assets/robots/child/robots_torque.xml
  uv run scripts/torque_aggregate_motion_k.py --symmetry none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossenbodiment.bodybuild.torque import (TorqueCalibrator, keep_mask, print_symmetry_report,  # noqa: E402,F401
                                              read_matrix, symmetrize_k, write_scaled_xml)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = ROOT / "outputs/torque_ratio_across_motions/gravity"
DEFAULT_SRC = ROOT / "assets/robots/child/robot.xml"
DEFAULT_OUT = ROOT / "assets/robots_torque/robots_torque.xml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=str(DEFAULT_MATRIX),
                   help="directory written by torque_ratio_across_motions.py")
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--z", type=float, default=3.0,
                   help="drop motions this many scaled MADs below the median "
                        "mean-correlation; 0 keeps every motion")
    p.add_argument("--r2-min", type=float, default=0.9,
                   help="a motion contributes to a joint's average only if its "
                        "fit for that joint reached this R²")
    p.add_argument("--agg", choices=["mean", "median"], default="mean")
    p.add_argument("--motions", default=None,
                   help="regex restricting which motions are aggregated, e.g. "
                        "'^move-ego' for locomotion only. Applied BEFORE the "
                        "--z agreement cut, so the outlier rule then judges "
                        "each motion against the subset rather than all 54")
    p.add_argument("--symmetry", choices=["pair", "none"], default="pair",
                   help="'pair' gives each mirrored L/R actuator pair one shared "
                        "k; 'none' keeps the raw per-actuator average")
    p.add_argument("--joint-dynamics", action="store_true",
                   help="also rewrite each actuated joint's armature (adult x Ir), damping "
                        "(adult x sqrt(k Ir)) and stiffness (adult x k): the measured isometric law")
    args = p.parse_args()
    cal = TorqueCalibrator(args.src, None, args.matrix)
    cal.calibrate(args.out, z=args.z, r2_min=args.r2_min, agg=args.agg, motions=args.motions, symmetry=args.symmetry,
                  joint_dynamics=args.joint_dynamics)


if __name__ == "__main__":
    main()
