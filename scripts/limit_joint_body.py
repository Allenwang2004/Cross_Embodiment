#!/usr/bin/env python3
"""limit_joint_body.py -- a lock_joint_body.py body with its joints held by their RANGE instead of switched off.

lock_joint_body.py holds a joint with an equality constraint and sets its actuator's gear to 0 (the motor does
nothing). This builds the same body with the motors left as the adult's: no equality, gear untouched, and every
locked joint's range (MJCF range) narrowed to its lock angle +- --band deg (MuJoCo needs range[0] < range[1]), with a
stiff joint limit (solreflimit --solref). The policy still drives every joint; on a limited one the servo pushes
against the limit.

  body      assets/robots/<lock label>_lim/robot.xml (locks read from assets/robots/<lock label>/robot.xml)
  retarget  data/<lock label>_lim/retargeting_motion/<clip>.npz: a copy of the lock body's reference (same angles,
            same lift), so the two bodies track the identical motion
usage: uv run python scripts/limit_joint_body.py --lock-body adult_lockRleg --clip move-ego-0-2/move-ego-0-2_4 [--band 1]
"""
import argparse, shutil
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-body", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--band", type=float, default=1.0, help="deg either side of the lock angle")
    ap.add_argument("--solref", type=float, default=0.005, help="joint-limit time constant (s)")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    label = a.label or f"{a.lock_body}_lim"
    src = REPO / "assets/robots/adult"
    ml = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots" / a.lock_body / "robot.xml"))
    locks = {ml.joint(int(ml.eq_obj1id[i])).name: float(ml.eq_data[i, 0]) for i in range(ml.neq)
             if ml.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT}
    m0 = mujoco.MjModel.from_xml_path(str(src / "robot.xml"))
    spec = mujoco.MjSpec.from_file(str(src / "robot.xml"))
    unit = np.degrees if spec.compiler.degree else (lambda r: r)
    J = {j.name: j for j in spec.joints}
    band = {}
    for jn, q in locks.items():
        r0 = m0.jnt_range[m0.joint(jn).id]
        lo, hi = max(q - np.radians(a.band), r0[0]), min(q + np.radians(a.band), r0[1])
        J[jn].range = [unit(lo), unit(hi)]
        J[jn].limited = mujoco.mjtLimited.mjLIMITED_TRUE
        J[jn].solref_limit = [a.solref, 1.0]
        band[jn] = (np.degrees(lo), np.degrees(hi))
    m = spec.compile()
    assert m.neq == 0 and np.allclose(m.actuator_gear, m0.actuator_gear)
    dst = REPO / "assets/robots" / label; dst.mkdir(parents=True, exist_ok=True)
    (dst / "robot.xml").write_text(spec.to_xml())
    for f in ("parameter.json", "skeleton.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    ref_src = REPO / f"data/{a.lock_body}/retargeting_motion/{a.clip}.npz"
    ref_dst = REPO / f"data/{label}/retargeting_motion/{a.clip}.npz"; ref_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ref_src, ref_dst)
    # does the limit hold? settle from the reference's first frame under gravity with zero control (the servos then
    # pull every joint toward its zero-control target, so the limited ones press on their limits)
    d = mujoco.MjData(m); d.qpos[:] = np.load(ref_dst)["qpos"][0]; mujoco.mj_forward(m, d)
    for _ in range(450):
        mujoco.mj_step(m, d)
    off = max(max(band[jn][0] - np.degrees(d.qpos[m.joint(jn).qposadr[0]]),
                  np.degrees(d.qpos[m.joint(jn).qposadr[0]]) - band[jn][1], 0.0) for jn in locks)
    print(f"{label}: {len(locks)} joints limited to lock angle +- {a.band:g} deg (from {a.lock_body}), gears kept, "
          f"no equality -> {dst / 'robot.xml'}")
    for jn, (lo, hi) in band.items():
        print(f"  {jn:13s} [{lo:+7.2f}, {hi:+7.2f}] deg")
    print(f"  limit check, 1 s with zero control: worst joint outside its band by {off:.2f} deg")
    print(f"  reference copied -> {ref_dst}")


if __name__ == "__main__":
    main()
