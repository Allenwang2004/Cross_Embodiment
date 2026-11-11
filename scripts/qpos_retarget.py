"""Direct qpos-space retargeting between two MJCF models that share the exact
same body/joint topology (same body names, same joint types/axes/declaration
order -- just different segment lengths/offsets, i.e. `body_pos`). This is
NOT a general-purpose retargeter: it only works because `robot.xml` and
`robot_child.xml`/`robot_elderly.xml` were built as scaled variants of the
same skeleton (confirmed via direct MuJoCo inspection -- 0 diffs in
body/joint names, types, axes, or declaration order between them; only
`body_pos` differs, non-uniformly).

Because the topology is identical, every non-root qpos value is already a
joint-local hinge angle (relative to its own parent, per that body's own
`axis=` declarations) -- copying it unchanged from source to target
skeleton reproduces the same *articulation* regardless of segment length
differences. Only the free-joint root (qpos[0:3], the pelvis's world
position) needs adjusting, since it's an absolute world-space translation
recorded for the SOURCE skeleton's proportions -- left unscaled, a smaller
skeleton's feet would float above the ground (or a larger skeleton's would
clip into it). It's rescaled uniformly by the ratio of each skeleton's own
rest-pose (qpos=0) pelvis height, which keeps the motion's overall path
shape and grounding proportional to the resulting skeleton's size.

This does NOT do proper IK-based retargeting (no per-limb correction for
foot/hand placement -- e.g. exact foot-ground contact timing can drift
slightly since only the root height is corrected, not each leg's reach).
For that, use the qpos_to_world_pose.py -> world_pose_to_fbx.py -> Unreal
Engine IK Retargeter pipeline instead, which supports true cross-topology
retargeting. For a same-topology proportion change like robot->robot_child,
this is a much cheaper approximation of what that pipeline's own IK-disabled
FK-chain retargeting already does.

The single rest-pose height ratio used for the root z scale is only exact
when the body is actually in a standing-like pose. For non-standing clips
(crawl, headstand, lieonground, split, crouch, ...) the ratio between "root
height at rest" and "root height in this specific pose" is NOT the same
across skeletons whose limb segments are scaled non-uniformly (this repo's
robot_child.xml scales legs/arms/torso/head independently -- see
assets/robots/child/parameter.json), so the naive scale alone
systematically drives the target skeleton's lowest geometry below the floor.
Measured across all 540 clips in data/retargeted_motion: naive scaling alone
put 83.7% of clips' mean lowest-point z below -0.01m (vs +0.0065m for the
unretargeted origin motion), worst on non-standing tasks (headstand -0.082,
crawl-0.4-2-d -0.056) -- see scripts/check_retarget_ground_clearance.py.

`--ground-correct` (on by default) fixes this with a per-frame vertical-only
correction on top of the naive scale: after naively retargeting, run forward
kinematics on the TARGET skeleton for each frame and find the exact lowest
point across all its geoms (box corners / capsule caps / sphere surface,
not the loose geom_rbound sphere). If that point is below `--tolerance`
(matching MuJoCo's own few-mm/cm soft-contact penetration, so corrected
frames still look/behave normally under the contact solver), shift qpos[:,2]
(root world z only -- x/y and all joint angles are untouched) up by exactly
enough to bring it to `--tolerance`. Frames that are already at or above
tolerance (e.g. genuine mid-air frames in a jump or rotate flip) are left
alone -- this only ever pulls the body UP out of the floor, never pushes a
legitimately airborne pose down.

OOP refactor step 9: the code is crossenbodiment/bodybuild/retarget.py (SameTopologyRetargeter; retarget_qpos,
min_geom_z = _min_geom_z, ground_correct_qpos are re-exported here for the scripts that import them); scripts/build_body.py
runs it on every motion folder of a new body (nested output, data/<body>/retargeting_motion/<task>/<stem>.npz). This file is the
old CLI, same flags, byte-identical files.

Run with (this repo's own venv):
    python3 scripts/qpos_retarget.py --input_npz <robot_clip.npz> \
        --output_npz <robot_child_clip.npz> \
        [--source_skeleton_json assets/robots/adult/skeleton.json] \
        [--target_skeleton_json assets/robots/child/skeleton.json] \
        [--target_xml assets/robots/child/robot.xml]

    # batch mode:
    python3 scripts/qpos_retarget.py --input_dir <robotmotion_dir> \
        --output_dir <robot_child_qpos_dir> \
        [--source_skeleton_json assets/robots/adult/skeleton.json] \
        [--target_skeleton_json assets/robots/child/skeleton.json] \
        [--target_xml assets/robots/child/robot.xml]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossenbodiment.bodybuild.retarget import (SameTopologyRetargeter, _min_geom_z, ground_correct_qpos,  # noqa: E402,F401
                                                retarget_qpos, save_clip)
from crossenbodiment.bodybuild.skeleton import load_root_rest_height  # noqa: E402,F401

import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=Path)
    parser.add_argument("--output_npz", type=Path)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--source_skeleton_json", default="assets/robots/adult/skeleton.json")
    parser.add_argument("--target_skeleton_json", default="assets/robots/child/skeleton.json")
    parser.add_argument("--target_xml", default="assets/robots/child/robot.xml",
                        help="target MJCF, used for --ground-correct's forward kinematics")
    parser.add_argument("--ground-correct", dest="ground_correct", action="store_true", default=True,
                        help="per-frame lift the pose so it never sinks below --tolerance (default: on)")
    parser.add_argument("--no-ground-correct", dest="ground_correct", action="store_false",
                        help="disable ground correction, output the pure naive-scale retarget")
    parser.add_argument("--tolerance", type=float, default=-0.005,
                        help="min allowed lowest-geom z before correction kicks in (m); "
                             "matches MuJoCo's normal soft-contact penetration range")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="robotmotion .npz files carry no fps of their own; render_qpos.py "
                             "expects one in the output, so it's passed through explicitly.")
    args = parser.parse_args()

    if bool(args.input_npz) == bool(args.input_dir):
        raise SystemExit("Pass exactly one of --input_npz/--output_npz or --input_dir/--output_dir")

    r = SameTopologyRetargeter(args.source_skeleton_json, args.target_skeleton_json, args.target_xml,
                               ground_correct=args.ground_correct, tolerance=args.tolerance)
    print(f"Root height scale ({args.source_skeleton_json} -> {args.target_skeleton_json}): {r.scale:.4f}")

    if args.input_npz:
        out = r.retarget(np.load(args.input_npz)["qpos"])
        save_clip(args.output_npz, out, args.fps)
        print(f"Wrote {out.shape} -> {args.output_npz}"
              + (f" (ground-corrected {r.last['n_corrected']}/{out.shape[0]} frames, max shift {r.last['max_shift']:.4f}m)"
                 if args.ground_correct else ""))
    else:
        npz_paths = sorted(args.input_dir.rglob("*.npz"))
        print(f"Found {len(npz_paths)} .npz files under {args.input_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        total_corrected = 0
        for i, npz_path in enumerate(npz_paths):
            r.write_clip(npz_path, args.output_dir / f"{npz_path.stem}.npz", args.fps)
            total_corrected += r.last["n_corrected"]
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(npz_paths)}")
        print(f"Wrote {len(npz_paths)} retargeted qpos files -> {args.output_dir}"
              + (f" ({total_corrected} frames ground-corrected total)" if args.ground_correct else ""))


if __name__ == "__main__":
    main()
