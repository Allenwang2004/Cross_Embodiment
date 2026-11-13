#!/usr/bin/env python3
"""scale_body_gains.py -- a body whose per-joint gain ratio k is multiplied by m, everything else unchanged.

The torque bodies are the adult scaled joint by joint (scripts/torque_scale_actuators.py --joint-dynamics):
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
torque limit and armature unchanged, so zeta changes by B / sqrt(A). A joint whose stiffness or damping comes from a <default> class (the adult's XML) gets its compiled value written out, scaled."""
import argparse, re
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def verify(src, out, m, md=None, mf=None):
    a, b = mujoco.MjModel.from_xml_path(str(src)), mujoco.MjModel.from_xml_path(str(out))
    assert (a.nu, a.njnt, a.nbody, a.ngeom) == (b.nu, b.njnt, b.nbody, b.ngeom)
    sm = np.sqrt(m) if md is None else md; mf = m if mf is None else mf
    ratio = lambda x, y: np.divide(y, x, out=np.full(x.shape, np.nan), where=x != 0)
    checks = {"gainprm[0]": (a.actuator_gainprm[:, 0], b.actuator_gainprm[:, 0], m),
              "biasprm[0]": (a.actuator_biasprm[:, 0], b.actuator_biasprm[:, 0], m),
              "biasprm[1]": (a.actuator_biasprm[:, 1], b.actuator_biasprm[:, 1], m),
              "biasprm[2]": (a.actuator_biasprm[:, 2], b.actuator_biasprm[:, 2], sm),
              "forcerange": (a.actuator_forcerange, b.actuator_forcerange, mf)}
    hinge = a.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    dof = a.jnt_dofadr[hinge]
    checks.update({"joint stiffness": (a.jnt_stiffness[hinge], b.jnt_stiffness[hinge], m),
                   "joint damping": (a.dof_damping[dof], b.dof_damping[dof], sm),
                   "armature": (a.dof_armature, b.dof_armature, 1.0)})
    for k, (x, y, want) in checks.items():
        r = ratio(x, y); ok = np.allclose(r[np.isfinite(r)], want, rtol=1e-9) and np.array_equal(x == 0, y == 0)
        print(f"  {k:16s} ratio {np.nanmin(r):.6f} .. {np.nanmax(r):.6f} (want {want:.6f})  {'ok' if ok else 'FAIL'}")
        assert ok, k
    free = ~hinge
    same = {"body_mass": (a.body_mass, b.body_mass), "body_inertia": (a.body_inertia, b.body_inertia),
            "body_pos": (a.body_pos, b.body_pos), "geom_size": (a.geom_size, b.geom_size), "geom_pos": (a.geom_pos, b.geom_pos),
            "geom_friction": (a.geom_friction, b.geom_friction), "jnt_range": (a.jnt_range, b.jnt_range),
            "actuator_ctrlrange": (a.actuator_ctrlrange, b.actuator_ctrlrange),
            "free-joint damping": (a.dof_damping[a.jnt_dofadr[free][0]:a.jnt_dofadr[free][0] + 6] if free.any() else np.zeros(1),
                                   b.dof_damping[b.jnt_dofadr[free][0]:b.jnt_dofadr[free][0] + 6] if free.any() else np.zeros(1))}
    for k, (x, y) in same.items():
        assert np.allclose(x, y, rtol=1e-12, atol=0), k
    print(f"  unchanged: {', '.join(same)}")
    # damping ratio per actuated joint: C / (2 sqrt(K I)), K = kp + joint stiffness, C = kd + joint damping, I = M_ii
    def zeta(mm):
        d = mujoco.MjData(mm); mujoco.mj_forward(mm, d); M = np.zeros((mm.nv, mm.nv)); mujoco.mj_fullM(mm, d, M)
        j = mm.actuator_trnid[:, 0]; dd = mm.jnt_dofadr[j]
        K = -mm.actuator_biasprm[:, 1] + mm.jnt_stiffness[j]; C = -mm.actuator_biasprm[:, 2] + mm.dof_damping[dd]
        return C / (2 * np.sqrt(K * M[dd, dd]))
    za, zb = zeta(a), zeta(b); want = sm / np.sqrt(m)
    print(f"  damping ratio zeta: src median {np.median(za):.3f}, new median {np.median(zb):.3f}, want x {want:.4f}, "
          f"largest miss {np.max(np.abs(zb / za / want - 1)) * 100:.2g}% (K x {m:g}, C x {sm:.4f}, the inertia unchanged)")


def fmt(x):
    return repr(float(x))


def scale_attr(tag, name, f):
    """multiply every number of attribute `name` in the element text `tag` by f (f a scalar or per-number list)"""
    m = re.search(rf'\b{name}="([^"]*)"', tag)
    if m is None:
        return tag, False
    v = [float(x) for x in m.group(1).split()]
    f = list(f) if isinstance(f, (list, tuple)) else [f] * len(v)
    return tag[:m.start(1)] + " ".join(fmt(x * g) for x, g in zip(v, f + [1.0] * len(v))) + tag[m.end(1):], True


def scale_joint_attr(tag, name, f, value):
    """the joint's own attribute scaled; if it has none (a <default> class sets it), its compiled value scaled and written out"""
    tag, ok = scale_attr(tag, name, f)
    return tag if ok else re.sub(r"\s*/?>$", lambda e: f' {name}="{fmt(value * f)}"' + e.group(0), tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/robots_torque/child/robot_torque_full.xml")
    ap.add_argument("--m", type=float, default=None, help="multiply every joint's k by this")
    ap.add_argument("--kp", type=float, default=None, help="instead of --m: multiply kp alone by this (needs --kd)")
    ap.add_argument("--kd", type=float, default=None, help="with --kp: multiply the damping alone by this")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = REPO / a.src, REPO / a.out
    if (a.m is None) == (a.kp is None or a.kd is None):
        raise SystemExit("give --m, or --kp and --kd")
    mp, sm, mf = (a.m, float(np.sqrt(a.m)), a.m) if a.m is not None else (a.kp, a.kd, 1.0)
    txt = src.read_text(); n_j = n_u = 0; m0 = mujoco.MjModel.from_xml_path(str(src))

    def joint(mt):                       # text-level, so every other number keeps its full precision
        nonlocal n_j
        tag = mt.group(0)
        name = re.search(r'\bname="([^"]*)"', tag)                         # no name: a <default> class's joint
        if name is None or m0.joint(name.group(1)).type[0] != mujoco.mjtJoint.mjJNT_HINGE:
            return tag
        j = m0.joint(name.group(1)).id
        tag = scale_joint_attr(tag, "stiffness", mp, m0.jnt_stiffness[j]); tag = scale_joint_attr(tag, "damping", sm, m0.dof_damping[m0.jnt_dofadr[j]])
        n_j += 1
        return tag

    def general(mt):
        nonlocal n_u
        tag = mt.group(0)
        tag, ok1 = scale_attr(tag, "gainprm", mp); tag, ok2 = scale_attr(tag, "biasprm", [mp, mp, sm])
        tag, ok3 = scale_attr(tag, "forcerange", mf)
        assert ok1 and ok2 and ok3, f"actuator without explicit gainprm / biasprm / forcerange: {tag}"
        n_u += 1
        return tag

    txt = re.sub(r"<joint\b[^>]*>", joint, txt)
    txt = re.sub(r"<general\b[^>]*>", general, txt)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(txt)
    print(f"{src.relative_to(REPO)} -> {out.relative_to(REPO)}: kp x {mp:g}, damping x {sm:.4f}, torque limit x {mf:g} on {n_u} actuators, {n_j} hinge joints")
    verify(src, out, mp, sm, mf)


if __name__ == "__main__":
    main()
