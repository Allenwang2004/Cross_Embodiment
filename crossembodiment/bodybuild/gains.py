"""gains.py -- GainScaler: a body with its joint gains multiplied, for both gain formats (OOP refactor step 9).

One job: kp x `kp`, kd x `kd`, torque limit x `force` on a body that already has its gains, writing a new gains file.
  human bodies (MJCF):  scale_xml(src, out) -- scripts/scale_body_gains.py, text level so every other number keeps its precision:
                        kp's group (actuator gainprm[0], biasprm[0], biasprm[1] and the hinge's joint stiffness) x kp, the
                        damping (biasprm[2] and the hinge's joint damping) x kd, forcerange x force; armature untouched. A joint
                        whose stiffness / damping comes from a <default> class gets its compiled value written out, scaled.
                        verify() then checks every ratio on the compiled models and that nothing else moved.
  robots (JSON):        scale_json(src, out) -- the per-joint gains json model/robot_env.py reads (assets/robots/<r>/gains_*.json,
                        scripts/robot_gain_scaling.py's format): its kp / kd tables scaled, a note prepended, the multipliers
                        recorded, exactly as scripts/kpkd_test.py prepare (kp / kd apart) and the k-scale test (11/03: k_group, kp
                        x m, kd x sqrt(m)) wrote them (json indent 1, no trailing newline).
Two ways to make one:
  GainScaler.k(m)          every joint's torque ratio k x m under the calibration law: kp and the torque limit x m, damping x
                           sqrt(m) (each joint keeps its damping ratio zeta); scale_body_gains.py --m
  GainScaler(kp, kd)       kp and kd apart, torque limit unchanged: zeta changes by kd / sqrt(kp); scale_body_gains.py --kp --kd,
                           kpkd_test.py
configs/bodies.yaml gain_variants lists the variants made so far (<parent>_<condition>).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import numpy as np

from .. import paths


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


def verify(src, out, m, md=None, mf=None):
    """scale_body_gains.py verify: every ratio of the scaled terms on the compiled models (kp group x m, damping x md, torque limit
    x mf, armature x 1), nothing else moved, and the damping ratio's change"""
    a, b = mujoco.MjModel.from_xml_path(str(src)), mujoco.MjModel.from_xml_path(str(out))
    assert (a.nu, a.njnt, a.nbody, a.ngeom) == (b.nu, b.njnt, b.nbody, b.ngeom)
    sm = np.sqrt(m) if md is None else md; mf = m if mf is None else mf
    ratio = lambda x, y: np.divide(y, x, out=np.full(x.shape, np.nan), where=x != 0)  # noqa: E731
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


class GainScaler:
    """kp x kp, damping x kd, torque limit x force (module docstring); `m` is set for the k-scale form (GainScaler.k)"""

    def __init__(self, kp, kd, force=1.0, m=None):
        self.kp, self.kd, self.force, self.m = kp, kd, force, m

    @classmethod
    def k(cls, m):
        """every joint's k x m: kp x m, damping x sqrt(m), torque limit x m (scale_body_gains.py --m)"""
        return cls(m, float(np.sqrt(m)), m, m=m)

    def condition(self):
        """the variant's name suffix (configs/bodies.yaml gain_variants): k<m> or kp<kp>_kd<kd>"""
        return f"k{self.m:g}" if self.m is not None else f"kp{self.kp:g}_kd{self.kd:g}"

    # ------------------------------------------------------------ human bodies: the MJCF
    def scale_xml(self, src, out, check=True, log=print):
        """scale_body_gains.py main: src -> out (text level), then verify(); src / out absolute or repository-relative"""
        src, out = paths.REPO / src, paths.REPO / out
        mp, sm, mf = self.kp, self.kd, self.force
        txt = src.read_text(); n_j = n_u = 0; m0 = mujoco.MjModel.from_xml_path(str(src))

        def joint(mt):                       # text-level, so every other number keeps its full precision
            nonlocal n_j
            tag = mt.group(0)
            name = re.search(r'\bname="([^"]*)"', tag)                         # no name: a <default> class's joint
            if name is None or m0.joint(name.group(1)).type[0] != mujoco.mjtJoint.mjJNT_HINGE:
                return tag
            j = m0.joint(name.group(1)).id
            tag = scale_joint_attr(tag, "stiffness", mp, m0.jnt_stiffness[j])
            tag = scale_joint_attr(tag, "damping", sm, m0.dof_damping[m0.jnt_dofadr[j]])
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
        rel = lambda p: p.relative_to(paths.REPO) if p.resolve().is_relative_to(paths.REPO) else p  # noqa: E731
        log(f"{rel(src)} -> {rel(out)}: kp x {mp:g}, damping x {sm:.4f}, torque limit x {mf:g} on {n_u} actuators, {n_j} hinge joints")
        if check:
            verify(src, out, mp, sm, mf)
        return out

    # ------------------------------------------------------------ robots: the gains json
    def json_note(self, src_rel, origin=None):
        """the note the existing variants carry, before the source json's own note"""
        if self.m is not None:
            return (f"{src_rel} with every joint's k multiplied by {self.m} ({origin or 'scripts in the k-scale test, 11/03'}): "
                    f"kp x {self.m}, kd x sqrt({self.m}) so each joint keeps its damping ratio; torque limits (the G1's hardware) "
                    f"and armature unchanged. ")
        return (f"{src_rel} with kp x {self.kp} and kd x {self.kd} ({origin or 'scripts/kpkd_test.py'}); torque limits and "
                f"armature unchanged. ")

    def scaled_gains(self, g, src_rel, origin=None):
        """the new gains dict from the source json's dict g (key order kept, the multipliers appended)"""
        if self.m is not None:
            m = self.m
            return dict(g, note=self.json_note(src_rel, origin) + g["note"], k_group={k: v * m for k, v in g["k_group"].items()},
                        kp={k: v * m for k, v in g["kp"].items()}, kd={k: v * self.kd for k, v in g["kd"].items()}, k_scale=m)
        a, b = self.kp, self.kd
        return dict(g, note=self.json_note(src_rel, origin) + g["note"], kp_scale=a, kd_scale=b,
                    kp={k: v * a for k, v in g["kp"].items()}, kd={k: v * b for k, v in g["kd"].items()})

    def scale_json(self, src, out, src_rel=None, origin=None):
        """src gains json -> out (json.dumps indent 1, as the variants were written); torque limits are the robot's (model/robots.py),
        so `force` must be 1 here"""
        assert self.force == 1.0 or self.m is not None, "a robot's torque limits are its hardware: scale kp / kd only"
        src, out = paths.REPO / src, paths.REPO / out
        g = json.loads(src.read_text())
        src_rel = src_rel or (str(src.relative_to(paths.REPO)) if src.resolve().is_relative_to(paths.REPO) else str(src))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.scaled_gains(g, src_rel, origin), indent=1))
        return out
