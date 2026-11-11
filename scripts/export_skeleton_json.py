"""Dump the rest-pose (qpos=0) world-space transform of every body in a MJCF
humanoid, plus its geoms, to a plain JSON file.

Building the corresponding Blender armature from raw MJCF <body pos>/<quat>
chains would mean re-implementing MuJoCo's quaternion composition/normalization
rules by hand. Instead we let MuJoCo itself compute the authoritative rest-pose
world transform for every body (mj_forward at qpos=0), and only the resulting
flat world positions/orientations get handed to Blender. This keeps the MJCF
parsing decoupled from any assumptions about how Blender will build the rig.

OOP refactor step 9: the code is crossenbodiment/bodybuild/skeleton.py (dump_skeleton, the pipeline's "skeleton" step of
scripts/build_body.py); this is its old CLI, same flags, same file.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossenbodiment.bodybuild.skeleton import _mat_to_quat, body_parent_name, dump_skeleton  # noqa: E402,F401


def main():
    parser = argparse.ArgumentParser(description="Dump MJCF rest-pose world transforms to JSON.")
    parser.add_argument("--input", default="assets/robots/adult/robot.xml")
    parser.add_argument("--output", default="assets/robots/adult/skeleton.json")
    args = parser.parse_args()
    dump_skeleton(args.input, args.output)


if __name__ == "__main__":
    main()
