#!/usr/bin/env python3
"""elderly_body.py -- the original adult turned into a stooped elderly walker by RESTRICTING a band per joint (nothing is
locked: every actuator still acts), plus a reference walk for it.

  --restrict joint   the joint's range (MJCF range) is the band; the policy may still command any target angle and the
                     servo then pushes against the limit
  --restrict action  the actuator's ctrlrange is cut so the PD target angle it can command spans exactly the band; the
                     joint range stays the adult's (each actuator maps ctrl in [-1, 1] linearly onto its joint's range:
                     target = (gain * u + bias0) / kp)
  --restrict both    both

  body      assets/robots/<label>/robot.xml: assets/robots/adult/robot.xml with
              spine  Torso_x / Spine_x / Chest_x limited to +-SPINE_BAND deg around a forward bend whose pelvis -> neck
                     lean is --stoop deg (the original walk leans ~3), the added flexion split SPINE_SPLIT (mostly lumbar,
                     so the chest turns little more than the lean and the neck can still bring the face level): it
                     cannot straighten up
              head   Neck_x / Head_x limited to the band the reference uses to keep the face level (looking ahead)
              arms   both Shoulder_x/y/z and Elbow_y limited to half the range the walk uses (plus MARGIN)
              hips   with --limit-hip-extension, both Hip_x capped at the reference's maximum (+MARGIN)
            joint limits made stiff (solreflimit --solref); parameter.json: the adult's keys + name / kind / parent / builder /
            edit; skeleton.json exported from the new XML (until step 9: the adult's files copied).
            (Bands are described below as joint ranges; with --restrict action they are target-angle ranges.)
  retarget  data/<label>/retargeting_motion/<clip>.npz: the adult's own walk slowed --slow x (its first 1/slow
            stretched to the full length: root position and joints linear, root rotation slerp), then
              spine  held at the stoop
              head   Neck_x / Head_x pitched back so the head faces the way the original head faced
              arms   centre + 0.5 x (walk angle - its median) on Shoulder_x/y/z + Elbow_y: half the swing; the centre
                     solved by IK so the arms hang from the bent trunk the way they hung from the upright one
            and ground-corrected like the other retargets. Bands come from the steady walk (after the clip's opening
            squat); the opening frames are clipped into them.
  also      outputs/elderly/<label>_young_slow.npz: the slowed walk on the unchanged adult (for side-by-side videos).
OOP refactor step 9: the code is crossembodiment/bodybuild/elderly.py (Elderly: plan, the body, reference;
scripts/build_body.py elderly); this file keeps its flags, and its robot.xml, reference and elderly.json are byte-identical.
usage: uv run python scripts/elderly_body.py --clip move-ego-0-2/move-ego-0-2_4 [--stoop 40 --slow 2]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossembodiment.bodybuild.elderly import (ARM, HEAD, MARGIN, OPENING, SPINE, SPINE_BAND, SPINE_SPLIT,  # noqa: E402,F401
                                               TRUNK, Elderly, Kin, slerp, slow_down)
from crossembodiment.bodybuild.skeleton import dump_skeleton  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--label", default="adult_elderly")
    ap.add_argument("--stoop", type=float, default=40.0, help="median pelvis -> neck forward lean (deg)")
    ap.add_argument("--slow", type=float, default=2.0, help="slow the walk down this many times")
    ap.add_argument("--solref", type=float, default=0.005, help="joint-limit time constant (s) of the restricted joints")
    ap.add_argument("--spine-band", type=float, default=SPINE_BAND, help="deg either side of the stoop the spine may move")
    ap.add_argument("--limit-hip-extension", action="store_true",
                    help="also cap both Hip_x at the reference's own maximum (+MARGIN): the pelvis cannot tilt back "
                         "(extending the hips) to undo the stoop")
    ap.add_argument("--restrict", choices=["joint", "action", "both"], default="joint",
                    help="what carries the band: the joint range, the actuator's ctrlrange (commandable target angles), "
                         "or both")
    a = ap.parse_args()
    print(f"scripts/elderly_body.py -> crossembodiment.bodybuild.Elderly({a.clip!r}, name={a.label!r}, stoop={a.stoop:g}, "
          f"slow={a.slow:g}, restrict={a.restrict!r})")
    body = Elderly(a.clip, name=a.label, stoop=a.stoop, slow=a.slow, solref=a.solref, spine_band=a.spine_band,
                   limit_hip_extension=a.limit_hip_extension, restrict=a.restrict)
    body.plan()
    body.build()
    dump_skeleton(body.xml, body.dir / "skeleton.json")
    body.reference()


if __name__ == "__main__":
    main()
