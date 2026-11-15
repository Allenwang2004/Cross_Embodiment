#!/usr/bin/env python3
"""scale_body_gains.py -- a body whose per-joint gain ratio k is multiplied by m, everything else unchanged.

The torque bodies are the adult scaled joint by joint (crossembodiment/bodybuild/torque.py TorqueCalibrator, joint dynamics on):
  actuator gainprm[0], biasprm[0], biasprm[1], forcerange  x k        (kp, its offset, the torque limit)
  actuator biasprm[2]                                      x sqrt(k Ir)  (kd)
  joint stiffness x k,  joint damping x sqrt(k Ir),  armature x Ir
Replacing k by m k under the same law therefore multiplies the first group and the joint stiffness by m, the
damping terms (biasprm[2], joint damping) by sqrt(m), and leaves the armature (Ir: inertia, not strength) alone,
so every joint keeps its damping ratio zeta = C / (2 sqrt(K I)); ctrl keeps its meaning (the servo's target angle
-(gainprm*ctrl + biasprm[0]) / biasprm[1] has the same m above and below). Only hinge joints (the actuated ones)
are touched. verify() compiles both models and checks every ratio, and that nothing else (masses, inertias,
geometry, ranges, contacts) moved.
usage: uv run python scripts/scale_body_gains.py --src assets/robots_torque/child/robot_torque_full.xml --m 1.2 \
           --out assets/robots_torque/child_k1.2/robot_torque_full.xml
--kp A --kd B instead of --m moves the two apart: kp's group (gainprm[0], biasprm[0], biasprm[1], joint stiffness) x A, the damping terms (biasprm[2], joint damping) x B,
torque limit and armature unchanged, so zeta changes by B / sqrt(A). A joint whose stiffness or damping comes from a <default> class (the adult's XML) gets its compiled value written out, scaled.
OOP refactor step 9: the code is crossembodiment/bodybuild/gains.py GainScaler (GainScaler.k(m) for --m, GainScaler(kp, kd) for
--kp --kd; scale_json does the robots' gains json); this file is its old CLI, same flags, byte-identical XML."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossembodiment.bodybuild.gains import GainScaler, fmt, scale_attr, scale_joint_attr, verify  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/robots_torque/child/robot_torque_full.xml")
    ap.add_argument("--m", type=float, default=None, help="multiply every joint's k by this")
    ap.add_argument("--kp", type=float, default=None, help="instead of --m: multiply kp alone by this (needs --kd)")
    ap.add_argument("--kd", type=float, default=None, help="with --kp: multiply the damping alone by this")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if (a.m is None) == (a.kp is None or a.kd is None):
        raise SystemExit("give --m, or --kp and --kd")
    g = GainScaler.k(a.m) if a.m is not None else GainScaler(a.kp, a.kd)
    g.scale_xml(a.src, a.out)


if __name__ == "__main__":
    main()
