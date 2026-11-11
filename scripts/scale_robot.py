"""
Generate a scaled variant of assets/robots/adult/robot.xml with independent
per-body-group scale factors: legs / arms / torso / head, each with its own
length scale and girth (thickness) scale. Body/joint/actuator names and gear
ratios are untouched, so the pretrained Meta Motivo model still works on the
result (same obs/action size).

After scaling, the Pelvis root height is auto-corrected via one MuJoCo
forward-kinematics pass so the feet still rest at ground level (z=0) in the
default pose -- otherwise a taller/shorter skeleton would float or clip
through the floor.

Each body lives in its own folder: assets/robots/<label>/robot.xml +
assets/robots/<label>/parameter.json.

Presets:
    adult     -- identity (scale=1.0 everywhere). Doesn't touch
                 adult/robot.xml itself, just writes adult/parameter.json
                 with the identity params so every body variant has a
                 matching metadata file.
    child     -- proportionally shorter limbs/torso, slightly bigger head,
                 thinner limbs.
    elderly   -- slightly shorter stature, thinner limbs, thicker torso.
These are rough illustrative defaults -- override any axis from the CLI.

By default the actuator gain/bias/force-range are rescaled per joint by its
own load ratio (see compute_joint_loads/apply_scale below) so a smaller body
isn't driven by comically oversized torque. Pass --no-actuator-scale to skip
that entirely and copy robot.xml's actuators verbatim -- e.g. when the point
of the experiment is to isolate "can this body shape alone (same raw
actuator strength as the adult) perform the motion" from actuator retuning.

Joint armature/damping/stiffness are NOT touched here; they are inherited
from robot.xml verbatim, so a generated body carries the adult's joint
coefficients until something downstream rewrites them. That job belongs to
the torque calibration (crossenbodiment/bodybuild/torque.py TorqueCalibrator, joint dynamics on), which sets all three from
the measured isometric law (armature x Ir, damping x sqrt(k*Ir), stiffness x
k) against the reference body.

This script used to scale them by length^3 * girth^2, the joint's own segment
inertia. Two things were wrong with that. A joint carries its whole distal
SUBTREE, not just its own segment, so wherever the subtree crosses a body
group the factor is simply the wrong one -- the Neck joints, grouped under
"torso", carry the head, which the child preset grows while shrinking the
torso, and the factor came out 5.7x off. And it compounded: applying anything
downstream multiplied two unrelated factors together. Removing it costs
nothing downstream: that law writes absolute values read off the reference
body, and the only thing it reads from a generated body is the subtree
inertia, recovered as M[dof,dof] - armature -- so whatever armature was
written here cancels out, leaving the result invariant to about 1e-14
relative (float rounding on the cancellation, verified).

OOP refactor step 9: the code is crossenbodiment/bodybuild (shape.py: apply_scale / measure_min_z / ground_pelvis,
beta.py: BETA_AXES / BODY_GROUPS / PRESETS, builders.py: ScaledBody, which writes parameter.json with the old keys plus
name / kind / parent / builder / edit, and keeps an existing "split"); this file is its old CLI, same flags, same robot.xml.
The whole pipeline for a new body (shape, skeleton, retargets, torque calibration) is scripts/build_body.py.

Usage (from project root):
    uv run scripts/scale_robot.py --preset adult
    uv run scripts/scale_robot.py --preset child
    uv run scripts/scale_robot.py --preset elderly
    uv run scripts/scale_robot.py --label custom --leg-scale 1.1 --leg-girth 0.9
    uv run scripts/scale_robot.py --label custom --leg-scale 1.1 --no-actuator-scale

Writes assets/robots/<label>/robot.xml and assets/robots/<label>/parameter.json.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from crossenbodiment.bodybuild.beta import BETA_AXES, BODY_GROUPS, GROUP_OF_BODY, PRESETS  # noqa: E402,F401
from crossenbodiment.bodybuild.builders import ScaledBody  # noqa: E402
from crossenbodiment.bodybuild.shape import (_scale_floats, apply_scale, compute_joint_loads,  # noqa: E402,F401
                                             measure_min_z)

ROBOTS_DIR = REPO_ROOT / "assets" / "robots"
SOURCE_XML = ROBOTS_DIR / "adult" / "robot.xml"
AXES = BETA_AXES                     # one list (model/dataset.py BETA_AXES is the same object)


def generate(label, params, source_xml=SOURCE_XML, out_dir=ROBOTS_DIR, scale_actuators=True, edit=None):
    """<out_dir>/<label>/robot.xml + parameter.json (ScaledBody.build); returns both paths"""
    b = ScaledBody(label, params, scale_actuators=scale_actuators, robots_dir=out_dir, source_xml=source_xml, edit=edit)
    out_xml = b.build()
    return out_xml, b.parameter_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default=None)
    parser.add_argument("--label", default=None,
                        help="output folder name -> assets/robots/<label>/robot.xml+parameter.json "
                             "(defaults to --preset name)")
    parser.add_argument("--no-actuator-scale", action="store_true",
                        help="copy robot.xml's actuators verbatim instead of rescaling "
                             "gain/bias/force-range by per-joint load ratio")
    for axis in AXES:
        parser.add_argument(f"--{axis.replace('_', '-')}", type=float, default=None,
                            help=f"override this preset's {axis}")
    args = parser.parse_args()

    if args.preset is None and args.label is None:
        parser.error("pass --preset or --label")

    params = dict(PRESETS[args.preset] if args.preset else PRESETS["adult"])
    label = args.label or args.preset
    over = {}
    for axis in AXES:
        cli_val = getattr(args, axis.replace("-", "_"))
        if cli_val is not None:
            params[axis] = cli_val
            over[axis] = cli_val

    if label == "adult" and all(params[a] == 1.0 for a in AXES):
        # adult/robot.xml already exists and is untouched -- just record its
        # (identity) morphology params next to it.
        body_dir = ROBOTS_DIR / "adult"
        body_dir.mkdir(parents=True, exist_ok=True)
        out_json = body_dir / "parameter.json"
        out_json.write_text(json.dumps({"label": "adult", "scale_actuators": True, **params}, indent=2) + "\n")
        print(f"wrote {out_json} (adult/robot.xml left untouched)")
        return

    edit = dict(preset=args.preset, **({"overrides": over} if over else {})) if args.preset else dict(beta="axes")
    print(f"scripts/scale_robot.py -> crossenbodiment.bodybuild.ScaledBody({label!r}, scale_actuators={not args.no_actuator_scale})")
    generate(label, params, scale_actuators=not args.no_actuator_scale, edit=edit)


if __name__ == "__main__":
    main()
