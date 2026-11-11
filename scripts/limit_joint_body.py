#!/usr/bin/env python3
"""limit_joint_body.py -- a lock_joint_body.py body with its joints held by their RANGE instead of switched off.

lock_joint_body.py holds a joint with an equality constraint and sets its actuator's gear to 0 (the motor does
nothing). This builds the same body with the motors left as the adult's: no equality, gear untouched, and every
locked joint's range (MJCF range) narrowed to its lock angle +- --band deg (MuJoCo needs range[0] < range[1]), with a
stiff joint limit (solreflimit --solref). The policy still drives every joint; on a limited one the servo pushes
against the limit.

  body      assets/robots/<lock label>_lim/robot.xml (locks read from assets/robots/<lock label>/robot.xml); parameter.json:
            the adult's keys + name / kind / parent / builder / edit; skeleton.json exported from the new XML
  retarget  data/<lock label>_lim/retargeting_motion/<clip>.npz: a copy of the lock body's reference (same angles,
            same lift), so the two bodies track the identical motion
OOP refactor step 9: crossenbodiment/bodybuild LimitJoints + CopyReference (scripts/build_body.py limit-joints); this file keeps
its flags; robot.xml and the reference are byte-identical.
usage: uv run python scripts/limit_joint_body.py --lock-body adult_lockRleg --clip move-ego-0-2/move-ego-0-2_4 [--band 1]
"""
import argparse
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossenbodiment.bodybuild.builders import LimitJoints  # noqa: E402
from crossenbodiment.bodybuild.retarget import CopyReference  # noqa: E402
from crossenbodiment.bodybuild.skeleton import dump_skeleton  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-body", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--band", type=float, default=1.0, help="deg either side of the lock angle")
    ap.add_argument("--solref", type=float, default=0.005, help="joint-limit time constant (s)")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    body = LimitJoints(a.lock_body, band=a.band, solref=a.solref, name=a.label)
    print(f"scripts/limit_joint_body.py -> crossenbodiment.bodybuild.LimitJoints({a.lock_body!r}, band={a.band:g}, "
          f"solref={a.solref:g}) (= scripts/build_body.py limit-joints --lock-body {a.lock_body} --clip {a.clip})")
    body.build()
    dump_skeleton(body.xml, body.dir / "skeleton.json")
    ref_dst = REPO / f"data/{body.name}/retargeting_motion/{a.clip}.npz"
    CopyReference().write_clip(REPO / f"data/{a.lock_body}/retargeting_motion/{a.clip}.npz", ref_dst)
    # does the limit hold? settle from the reference's first frame under gravity with zero control
    off = body.limit_error(np.load(ref_dst)["qpos"][0])
    print(f"{body.name}: {len(body.locks)} joints limited to lock angle +- {a.band:g} deg (from {a.lock_body}), gears kept, "
          f"no equality -> {body.xml}")
    for jn, (lo, hi) in body.bands.items():
        print(f"  {jn:13s} [{lo:+7.2f}, {hi:+7.2f}] deg")
    print(f"  limit check, 1 s with zero control: worst joint outside its band by {off:.2f} deg")
    print(f"  reference copied -> {ref_dst}")


if __name__ == "__main__":
    main()
