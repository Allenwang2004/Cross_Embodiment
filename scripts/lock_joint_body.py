#!/usr/bin/env python3
"""lock_joint_body.py -- the original adult with one joint locked, and a reference motion retargeted to it.

  body      assets/robots/<label>/robot.xml: assets/robots/adult/robot.xml (the body Meta Motivo was trained on)
            with every locked joint (--joint/--deg, or --lock "NAME=DEG,NAME=DEG,...") held by an equality
            constraint and its actuator's gear set to 0 (the policy still emits 69 actions; those no longer act). parameter.json:
            the adult's keys + name / kind / parent / builder / edit; skeleton.json exported from the new XML (until step 9: the
            adult's files copied).
  retarget  data/<label>/retargeting_motion/<clip>.npz: the adult's own motion (data/origin_motion) with the locked
            joint's angle replaced by --deg in every frame, then lifted frame by frame so no geometry is below the
            floor (scripts/qpos_retarget.py's ground correction) -- a straight-legged version of the same motion.
OOP refactor step 9: crossenbodiment/bodybuild LockJoints + LockedJointReference (scripts/build_body.py lock-joints); this file
keeps its flags, LOCK_DEG and --show-angles; robot.xml and the reference are byte-identical.
usage: uv run python scripts/lock_joint_body.py --clip move-ego-0-2/move-ego-0-2_4 [--joint R_Knee_x --deg 10]
       uv run python scripts/lock_joint_body.py --clip ... --user-lock [--no-lift] --solref 0.005   (LOCK_DEG below)
       uv run python scripts/lock_joint_body.py --clip ... --show-angles
       uv run python scripts/lock_joint_body.py --clip ... --lock "R_Hip_x=0,...,R_Knee_x=10,..." --label adult_lockRleg
"""
import argparse, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossenbodiment.bodybuild.builders import LockJoints   # noqa: E402
from crossenbodiment.bodybuild.retarget import save_clip   # noqa: E402
from crossenbodiment.bodybuild.skeleton import dump_skeleton   # noqa: E402

# ============================== YOUR CODE HERE (used with --user-lock) ==============================
# The angle (degrees) each joint is locked at. Bend the leg (hip / knee / ankle flexion axes: R_Hip_x, R_Knee_x,
# R_Ankle_x) so the locked leg is about as long as the walking leg, and the reference needs little or no lift.
# `--show-angles` prints every right-leg joint's range and its min / median / max during the clip.
LOCK_DEG = {
    # flexion axes: fitted on move-ego-0-2_4's steady walk (frames >= 30) -- no frame below the floor, and the locked
    # foot within ~1 cm of the ground where the walk had its right foot planted
    "R_Hip_x": 13, "R_Knee_x": 15, "R_Ankle_x": -13,
    # the other axes: their median during the walk
    "R_Hip_y": 1.4, "R_Hip_z": -5.1, "R_Knee_y": -6.1, "R_Knee_z": 5.6,
    "R_Ankle_y": 0.4, "R_Ankle_z": -3.7, "R_Toe_x": -7.8, "R_Toe_y": -2.3, "R_Toe_z": 2.6,
}
# =====================================================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--joint", default="R_Knee_x")
    ap.add_argument("--deg", type=float, default=10.0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--lock", default=None, help='"NAME=DEG,..." -- several joints; overrides --joint/--deg')
    ap.add_argument("--solref", type=float, default=None,
                    help="equality time constant (s); MuJoCo's default 0.02 lets a whole locked leg sag ~5 deg, "
                         "0.005 holds it (must stay >= 2 x the 1/450 s timestep)")
    ap.add_argument("--user-lock", action="store_true", help="lock the joints in LOCK_DEG (top of this file)")
    ap.add_argument("--no-lift", action="store_true",
                    help="write the reference WITHOUT the ground correction, and report how far it goes below the floor")
    ap.add_argument("--show-angles", action="store_true",
                    help="print the right-leg joints' ranges and their angles during --clip, then exit")
    a = ap.parse_args()
    if a.show_angles:
        m0 = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml"))
        q = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
        for j in [f"R_{p}_{x}" for p in ("Hip", "Knee", "Ankle", "Toe") for x in "xyz"]:
            jj = m0.joint(j); v = np.degrees(q[:, int(jj.qposadr[0])])
            print(f"{j:10s} range {np.degrees(jj.range[0]):6.0f} .. {np.degrees(jj.range[1]):4.0f} deg | "
                  f"{a.clip.split('/')[1]}: min {v.min():6.1f}  median {np.median(v):6.1f}  max {v.max():6.1f}")
        return
    if a.user_lock:
        if not LOCK_DEG:
            raise SystemExit("--user-lock: LOCK_DEG at the top of scripts/lock_joint_body.py is empty")
        locks = [(k, float(v)) for k, v in LOCK_DEG.items()]
    else:
        locks = ([(kv.split("=")[0].strip(), float(kv.split("=")[1])) for kv in a.lock.split(",")] if a.lock
                 else [(a.joint, a.deg)])
    label = a.label or (f"adult_lock{a.joint.replace('_', '')}{int(round(a.deg))}" if not a.user_lock else "adult_lockRleg_bent")
    print(f"scripts/lock_joint_body.py -> crossenbodiment.bodybuild.LockJoints({label!r}, {len(locks)} locks, solref={a.solref})")
    body = LockJoints(label, locks, solref=a.solref)
    body.build()
    dump_skeleton(body.xml, body.dir / "skeleton.json")
    # does the lock hold? settle from the motion's first frame under gravity with zero control
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
    r = body.reference(lift=not a.no_lift)
    low = r.lowest(r.locked(ref))
    print(f"  before any lift: {int(np.sum(low < -0.005))}/{len(ref)} frames below the floor, deepest {-low.min() * 100:.1f} cm")
    out = r.retarget(ref)
    n, shift = r.last["n_corrected"], r.last["max_shift"]
    p = save_clip(REPO / f"data/{label}/retargeting_motion/{a.clip}.npz", out)
    err = body.lock_error(out[0])
    print(f"{label}: {len(locks)} joints locked ({', '.join(f'{jn}={deg:g}' for jn, deg in locks)}), actuator gears 0 -> {body.xml}")
    print(f"  lock check, 1 s of passive simulation: worst joint off its lock by {err:.2f} deg")
    print(f"  reference {a.clip}: locked joints fixed; " + ("NOT lifted (--no-lift)" if a.no_lift else
          f"lifted {n}/{len(out)} frames (max {shift * 100:.1f} cm)") + f" -> {p}")


if __name__ == "__main__":
    main()
