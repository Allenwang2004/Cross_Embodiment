"""builders.py -- BodyBuilder and its subclasses: one new body's MJCF + parameter.json (OOP refactor step 9).

Every builder does the same five things (BodyBuilder.build): load the parent (the adult, assets/robots/adult/robot.xml) -> apply
its edit -> ground the pelvis -> write <robots>/<name>/robot.xml -> write an honest <robots>/<name>/parameter.json. The skeleton
(skeleton.json), the references and the physics are later steps (bodybuild/pipeline.py BodyPipeline).

  ScaledBody        scripts/scale_robot.py --no-actuator-scale: the 8 body parameters beta (bodybuild/beta.py BETA_AXES), from
                    the axes, a preset, or len x wid (scripts/make_scaled_bodies.py: length x L, girth x L * W); ElementTree on the
                    adult's XML, the Pelvis grounded so the feet are flush with the floor. Joints and actuators stay the adult's
                    (TorqueCalibrator rewrites them).
  HeavyLimb         scripts/heavy_arm_body.py: one arm's geoms (or only --parts) x `scale` density (payload: R Wrist + Hand x 80)
  LockJoints        scripts/lock_joint_body.py: joints held at an angle by equality constraints, their actuators' gear 0
  LimitJoints       scripts/limit_joint_body.py: a LockJoints body's locks as narrow joint ranges instead (motors kept)
  Elderly           scripts/elderly_body.py (bodybuild/elderly.py): bands per joint (stooped spine, head, half arm swing)
  ActuatorStrength  scripts/strength_rollout.py's build: every actuator's gear (S1) or forcerange (S2) x `scale`
The adult-size variants (all but ScaledBody) edit the adult through mujoco.MjSpec and leave its geometry, so their pelvis already
stands on the floor: their ground() is a no-op. Their XML is byte-identical to the old scripts' (same MjSpec calls in the same order:
tests/test_bodybuild.py).

parameter.json, honest: the old scripts wrote scale_robot.py's flat schema for scaled bodies ({"label", "scale_actuators", the 8
axes}) and copied the adult's file verbatim for every variant (label "baseline", the adult's split "source"). The readers
(model/dataset.py load_beta: the 8 axes; scripts/build_dataset.py / write_body_splits.py: "split"; "label") keep their keys and
values, in the old order; added after them:
  name      the body's folder name (a variant's "label" stays the adult's "baseline", which old readers saw)
  kind      configs/bodies.yaml's kind: smpl_scaled | smpl_variant
  parent    the body it was made from
  builder   the class that made it
  edit      the edit's parameters (len / wid or preset for a scaled body; side, parts, scale; locks; bands; ...)
A "split" that the destination's existing parameter.json, or the repository's copy of this body's, already has is kept (scale_robot.py
wiped it on every rebuild: docs/new_body.md's old trap).
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import paths  # noqa: E402
from .beta import BETA_AXES, PRESETS, isometric_beta  # noqa: E402
from .shape import apply_scale, ground_pelvis  # noqa: E402

ROBOTS = paths.REPO / paths.ROBOTS_REL


def body_dir(name, robots_dir=None):
    """<robots_dir>/<name> if it exists there, else the repository's assets/robots/<name> (a body this build did not make)"""
    if robots_dir is not None and (Path(robots_dir) / name).exists():
        return Path(robots_dir) / name
    return ROBOTS / name


class BodyBuilder(ABC):
    """load the parent -> edit -> ground -> robot.xml + parameter.json (module docstring)"""
    kind = "smpl_variant"
    parent = "adult"

    def __init__(self, name, robots_dir=None, source_xml=None):
        self.name = name
        self.robots_dir = Path(robots_dir) if robots_dir is not None else ROBOTS
        self.source_xml = Path(source_xml) if source_xml is not None else paths.ADULT_XML
        self.model = None                                     # the compiled result (after build)

    # ------------------------------------------------------------ where
    @property
    def dir(self):
        return self.robots_dir / self.name

    @property
    def xml(self):
        return self.dir / "robot.xml"

    @property
    def parameter_json(self):
        return self.dir / "parameter.json"

    def done(self):
        return self.xml.exists() and self.parameter_json.exists()

    # ------------------------------------------------------------ the five steps
    @abstractmethod
    def load(self):
        """the parent, editable (an ElementTree or an MjSpec)"""

    @abstractmethod
    def edit(self, body):
        """apply this builder's edit to `body` in place"""

    def ground(self, body):
        """put the pelvis on the floor; default: nothing (the edit keeps the parent's geometry). Returns the shift (m)."""
        return 0.0

    @abstractmethod
    def write_xml(self, body, path):
        """write the edited body to `path`"""

    @abstractmethod
    def edit_parameters(self):
        """the edit's parameters (parameter.json "edit")"""

    def legacy_parameters(self):
        """the keys the old scripts wrote (default: the parent's parameter.json, as the variants copied it)"""
        src = self.source_xml.parent / "parameter.json"
        return json.loads(src.read_text()) if src.exists() else {}

    def parameters(self):
        p = self.legacy_parameters()
        split = self.existing_split()
        if split is not None and "split" not in p:
            p["split"] = split
        p.update(name=self.name, kind=self.kind, parent=self.parent, builder=type(self).__name__, edit=self.edit_parameters())
        return p

    def existing_split(self):
        """the "split" of an existing parameter.json of this body (the destination's, else the repository's), or None"""
        for f in (self.parameter_json, ROBOTS / self.name / "parameter.json"):
            if f.exists():
                try:
                    s = json.loads(f.read_text()).get("split")
                except ValueError:
                    continue
                if s is not None:
                    return s
        return None

    def build(self, log=print):
        """robot.xml + parameter.json of the body; returns the XML path"""
        if self.xml.exists() and self.xml.resolve() == self.source_xml.resolve():
            raise ValueError(f"refusing to overwrite the parent {self.source_xml} with a body built from it (name {self.name!r})")
        body = self.load()
        self.edit(body)
        self.dz = self.ground(body)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.write_xml(body, self.xml)
        self.parameter_json.write_text(json.dumps(self.parameters(), indent=2) + "\n")
        log(f"wrote {self.xml}")
        log(f"wrote {self.parameter_json}")
        return self.xml


class ScaledBody(BodyBuilder):
    """the adult's skeleton at another size (scale_robot.py; make_scaled_bodies.py's len x wid). beta: the 8 axes (missing ones 1.0)."""
    kind = "smpl_scaled"

    def __init__(self, name, beta=None, *, scale_actuators=False, edit=None, **kw):
        super().__init__(name, **kw)
        b = dict(PRESETS["adult"]); b.update(beta or {})
        unknown = set(b) - set(BETA_AXES)
        assert not unknown, f"unknown beta axes {sorted(unknown)} (BETA_AXES: {BETA_AXES})"
        self.beta = {a: float(b[a]) for a in BETA_AXES}
        self.scale_actuators = scale_actuators
        self._edit = dict(edit or {})

    @classmethod
    def from_len_wid(cls, L, W, name=None, **kw):
        """make_scaled_bodies.py's len<L>_wid<W>: every length x L, every girth x L * W"""
        return cls(name or f"len{L}_wid{W}", isometric_beta(L, W), edit=dict(len=L, wid=W, girth="len * wid"), **kw)

    @classmethod
    def from_preset(cls, preset, name=None, overrides=None, **kw):
        b = dict(PRESETS[preset]); b.update(overrides or {})
        return cls(name or preset, b, edit=dict(preset=preset, **({"overrides": overrides} if overrides else {})), **kw)

    def load(self):
        return ET.parse(self.source_xml)

    def edit(self, tree):
        apply_scale(tree, self.beta, source_xml=self.source_xml, scale_actuators=self.scale_actuators)

    def ground(self, tree):
        return ground_pelvis(tree)

    def write_xml(self, tree, path):
        tree.write(path)

    def edit_parameters(self):
        return dict(self._edit, pelvis_shift_m=float(self.dz))

    def legacy_parameters(self):
        """scale_robot.py's schema: label, scale_actuators, the 8 axes (BETA_AXES order)"""
        return {"label": self.name, "scale_actuators": self.scale_actuators, **self.beta}


class SpecBody(BodyBuilder):
    """an adult-size variant edited through mujoco.MjSpec: load = MjSpec.from_file(parent), the edit, compile, to_xml()"""
    kind = "smpl_variant"

    def load(self):
        return mujoco.MjSpec.from_file(str(self.source_xml))

    def check(self, model):
        """assertions on the compiled body (default none)"""

    def write_xml(self, spec, path):
        self.model = spec.compile()
        self.check(self.model)
        Path(path).write_text(spec.to_xml())


class HeavyLimb(SpecBody):
    """heavy_arm_body.py: the density of one arm's geoms (<side>_Shoulder upper arm, _Elbow forearm, _Wrist, _Hand, or only
    `parts`) x scale, so their mass and inertia scale by it; same shape, same motors"""
    PARTS = ("Shoulder", "Elbow", "Wrist", "Hand")

    def __init__(self, scale, side="R", parts=PARTS, name=None, **kw):
        parts = list(parts)
        name = name or (f"adult_heavy{side}arm_x{scale:g}" if parts == list(self.PARTS) else
                        f"adult_heavy{side}{''.join(parts)}_x{scale:g}")
        super().__init__(name, **kw)
        self.scale, self.side, self.parts = scale, side, parts

    def edit(self, spec):
        self.model0 = spec.compile()                          # the adult (heavy_arm_body.py compiled it before the edit)
        G = {g.name: g for g in spec.geoms}
        for n in [f"{self.side}_{p}" for p in self.parts]:
            G[n].density *= self.scale

    def edit_parameters(self):
        return dict(side=self.side, parts=self.parts, density_scale=self.scale)

    def arm_mass(self):
        """(arm mass before, after, body mass before, after) in kg (heavy_arm_body.py's report)"""
        names = [f"{self.side}_{p}" for p in self.PARTS]
        m0, m = self.model0, self.model
        return (sum(m0.body(n).mass[0] for n in names), sum(m.body(n).mass[0] for n in names),
                float(m0.body_mass.sum()), float(m.body_mass.sum()))


class LockJoints(SpecBody):
    """lock_joint_body.py: every (joint, degrees) of `locks` held by an equality constraint (solref: its time constant, MuJoCo's
    0.02 when None) and its actuator's gear set to 0 (the policy still emits 69 actions; those no longer act)"""

    def __init__(self, name, locks, solref=None, **kw):
        super().__init__(name, **kw)
        self.locks, self.solref = [(j, float(d)) for j, d in locks], solref

    def edit(self, spec):
        for jn, deg in self.locks:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
            eq.name1 = jn
            eq.data[:5] = [np.radians(deg), 0, 0, 0, 0]  # q = angle (polynomial in the second joint, absent)
            if self.solref:
                eq.solref = [self.solref, 1.0]
            act = [x for x in spec.actuators if x.name == jn]
            assert len(act) == 1, f"no actuator named {jn}"
            act[0].gear[0] = 0.0

    def edit_parameters(self):
        return dict(locks_deg={j: d for j, d in self.locks}, solref=self.solref, actuator_gear=0.0)

    def reference(self, lift=True):
        """the LockedJointReference of this (built) body"""
        from .retarget import LockedJointReference
        return LockedJointReference(self.model, self.locks, lift=lift)

    def lock_error(self, qpos0, steps=450):
        """settle from qpos0 under gravity with zero control for `steps` steps (1 s); the worst joint's distance from its lock (deg)"""
        m = self.model
        qadr = {jn: int(m.joint(jn).qposadr[0]) for jn, _ in self.locks}
        d = mujoco.MjData(m); d.qpos[:] = qpos0; mujoco.mj_forward(m, d)
        for _ in range(steps):
            mujoco.mj_step(m, d)
        return max(abs(np.degrees(d.qpos[qadr[jn]]) - deg) for jn, deg in self.locks)


class LimitJoints(SpecBody):
    """limit_joint_body.py: a LockJoints body's locks read back from its XML (equality constraints), each locked joint's range
    narrowed to its lock angle +- band (deg) with a stiff limit (solreflimit solref); no equality, gears kept: the policy drives
    every joint and the servo pushes against the limit"""

    def __init__(self, lock_body, band=1.0, solref=0.005, name=None, **kw):
        super().__init__(name or f"{lock_body}_lim", **kw)
        self.lock_body, self.band, self.solref = lock_body, band, solref

    def edit(self, spec):
        ml = mujoco.MjModel.from_xml_path(str(body_dir(self.lock_body, self.robots_dir) / "robot.xml"))
        self.locks = {ml.joint(int(ml.eq_obj1id[i])).name: float(ml.eq_data[i, 0]) for i in range(ml.neq)
                      if ml.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT}
        m0 = mujoco.MjModel.from_xml_path(str(self.source_xml))
        self.model0 = m0
        unit = np.degrees if spec.compiler.degree else (lambda r: r)
        J = {j.name: j for j in spec.joints}
        self.bands = {}
        for jn, q in self.locks.items():
            r0 = m0.jnt_range[m0.joint(jn).id]
            lo, hi = max(q - np.radians(self.band), r0[0]), min(q + np.radians(self.band), r0[1])
            J[jn].range = [unit(lo), unit(hi)]
            J[jn].limited = mujoco.mjtLimited.mjLIMITED_TRUE
            J[jn].solref_limit = [self.solref, 1.0]
            self.bands[jn] = (np.degrees(lo), np.degrees(hi))

    def check(self, m):
        assert m.neq == 0 and np.allclose(m.actuator_gear, self.model0.actuator_gear)

    def edit_parameters(self):
        return dict(lock_body=self.lock_body, band_deg=self.band, solref_limit=self.solref,
                    bands_deg={j: [float(lo), float(hi)] for j, (lo, hi) in self.bands.items()})

    def reference_source(self, clip, data_dir=None):
        """the lock body's reference of `clip` (copied: the two bodies track the identical motion)"""
        d = Path(data_dir) if data_dir is not None else paths.REPO / paths.DATA_REL
        p = d / self.lock_body / "retargeting_motion" / f"{clip}.npz"
        return p if p.exists() else paths.REPO / paths.retargeting_motion(self.lock_body, clip)

    def limit_error(self, qpos0, steps=450):
        """settle from qpos0 with zero control (the servos pull every joint to its zero-control target, so the limited ones press
        on their limits); the worst limited joint's distance outside its band (deg)"""
        m = self.model
        d = mujoco.MjData(m); d.qpos[:] = qpos0; mujoco.mj_forward(m, d)
        for _ in range(steps):
            mujoco.mj_step(m, d)
        return max(max(self.bands[jn][0] - np.degrees(d.qpos[m.joint(jn).qposadr[0]]),
                       np.degrees(d.qpos[m.joint(jn).qposadr[0]]) - self.bands[jn][1], 0.0) for jn in self.locks)


class ActuatorStrength(SpecBody):
    """strength_rollout.py's build: every actuator's gear x scale (what="gear", S1: every joint torque x scale, the equilibrium
    angle of a command unchanged) or its forcerange x scale (what="cap", S2: only the peak torque capped earlier)"""

    def __init__(self, name, scale, what="gear", **kw):
        assert what in ("gear", "cap"), what
        super().__init__(name, **kw)
        self.scale, self.what = scale, what

    def edit(self, spec):
        for a in spec.actuators:
            if self.what == "gear":
                a.gear[0] *= self.scale
            else:
                a.forcerange = [a.forcerange[0] * self.scale, a.forcerange[1] * self.scale]

    def edit_parameters(self):
        return dict(what=self.what, scale=self.scale)
