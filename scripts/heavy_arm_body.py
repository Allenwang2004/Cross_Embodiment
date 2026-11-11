#!/usr/bin/env python3
"""heavy_arm_body.py -- the original adult with one arm --scale x heavier (same shape, same motors).

  body      assets/robots/<label>/robot.xml: assets/robots/adult/robot.xml with the density of the arm's four geoms
            (<side>_Shoulder upper arm, <side>_Elbow forearm, <side>_Wrist, <side>_Hand; or only --parts) x --scale, so
            their mass and inertia scale by it; parameter.json: the adult's keys + name / kind / parent / builder / edit;
            skeleton.json: exported from the new XML (its own masses; until step 9 the adult's file was copied)
  retarget  data/<label>/retargeting_motion/<clip>.npz: the adult's own motion (data/origin_motion), unchanged --
            the heavy-armed body is asked to do exactly what the adult did
OOP refactor step 9: crossenbodiment/bodybuild HeavyLimb + CopyReference (scripts/build_body.py heavy-limb does the same); this
file keeps its flags and its robot.xml is byte-identical.
usage: uv run python scripts/heavy_arm_body.py --scale 5 [--side R] [--clip move-ego-0-2/move-ego-0-2_4]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossenbodiment.bodybuild.builders import HeavyLimb  # noqa: E402
from crossenbodiment.bodybuild.retarget import CopyReference  # noqa: E402
from crossenbodiment.bodybuild.skeleton import dump_skeleton  # noqa: E402

PARTS = HeavyLimb.PARTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--side", default="R", choices=["L", "R"])
    ap.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    ap.add_argument("--parts", default=",".join(PARTS),
                    help="which of the arm's geoms get heavier, e.g. Wrist,Hand for a weight carried in the hand")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    b = HeavyLimb(a.scale, a.side, a.parts.split(","), name=a.label)
    print(f"scripts/heavy_arm_body.py -> crossenbodiment.bodybuild.HeavyLimb({a.scale:g}, {a.side!r}, {a.parts.split(',')}) "
          f"(= scripts/build_body.py heavy-limb --scale {a.scale:g} --side {a.side} --parts {a.parts} --clip {a.clip})")
    b.build()
    dump_skeleton(b.xml, b.dir / "skeleton.json")
    ref = REPO / f"data/{b.name}/retargeting_motion/{a.clip}.npz"
    CopyReference().write_clip(REPO / f"data/origin_motion/{a.clip}.npz", ref)
    arm0, arm, body0, body = b.arm_mass()
    print(f"{b.name}: {a.side} arm {arm0:.2f} -> {arm:.2f} kg (body {body0:.1f} -> {body:.1f} kg) "
          f"-> {b.xml}; reference = the adult's {a.clip}")


if __name__ == "__main__":
    main()
