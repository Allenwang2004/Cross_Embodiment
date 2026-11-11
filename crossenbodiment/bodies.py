"""bodies.py -- the bodies as data: BodySpec, BodyRegistry (configs/bodies.yaml) and RobotSpec (model/robots.py ROBOTS).

OOP refactor step 2 (docs/refactor/PLAN.md). Nothing here runs a simulation or changes what a script does: the registry
reproduces the paths and root scales the current code uses (tests/test_registry.py checks each against the code and the
existing runs), so a literal can be replaced by a registry lookup without changing a bit.

    from crossenbodiment.bodies import BODIES
    BODIES["tall"].physics_xml            -> "assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml"
    BODIES.reference("child", clip)       -> "data/child/retargeting_motion/<clip>.npz"
    BODIES.scale("child")                 -> 0.611036 (ExactObs's default, the literal)
    BODIES.scale("len1.4_wid0.8", clip)   -> single_z_search.py's rule on that clip (1.364408905613045 on move-ego-0-2_1)
    BODIES.scale("payload")               -> 1.0 (lso_search.py --scale 1, body_52_test.py); with a clip: the rule (also 1.0)
    BODIES.scale("g1")                    -> 0.8212611427612194 (= RobotMap("g1").s)

The root scale s (see configs/bodies.yaml): fixed bodies return their literal; clip_ratio bodies compute, once per clip,
exactly single_z_search.py's expression (median of reference root x, y / the adult clip's, where |adult| > 1e-3), on the
task clip (data/origin_motion) or the AMASS clip (amass_data/amass_origin_motion/ACCAD) as amass_baselines.py scale() does.

One quantity, many spellings (step 11: `scale` in the code, `s` only inside a formula). The root scale is body size / adult size,
the factor between a body's metres and adult-equivalent metres; every one of these is the same number for one body and clip:
  BODIES.scale(body[, clip])            the registry (BodySpec.scale / scale_rule / scale_default; configs/bodies.yaml keys
                                        s / s_rule / s_default, kept on disk; BodySpec.s / s_rule / s_default read-only aliases)
  model/exact_obs.ExactObs(scale=)      .scale (alias .s): positions and velocities / scale before Motivo sees them; its default
                                        0.611036 is the child's (crossenbodiment/obs.py CHILD_SCALE)
  crossenbodiment/obs.ExactObs.scale    the same, through the mapper (alias .s)
  model/robot_map.RobotMap.scale        a robot's: robot / SMPL hip-to-ankle length in the nominal pose (alias .s)
  PHCCost(scale=), Position(scale=)     PHC's lengths and the position term in adult-equivalent metres (crossenbodiment/costs.py)
  problems' pos_scale                   what the searches divide positions by: the scale with --pos-unit adult, else 1.0
  summary.json "pos_div"                that value as single_z_search.py / lso_search.py write it (on-disk key, unchanged)
  lso_search.py --scale                 the flag (default: none = ExactObs's 0.611036, the child's, for ANY body; obs.lso_scale_kw)
  MetricSuite(body, keys, scale)        the metrics' success threshold (0.5 m x scale) and adult-equivalent drift (.s alias)
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import paths

CONFIG = paths.REPO / "configs" / "bodies.yaml"
KINDS = ("smpl_scaled", "smpl_variant", "gain_variant", "robot")
S_RULES = ("fixed", "clip_ratio")
YAML_KEYS = {"s": "scale", "s_rule": "scale_rule", "s_default": "scale_default"}     # configs/bodies.yaml key -> BodySpec field


@dataclass(frozen=True)
class BodySpec:
    """One body. Paths are repository-relative strings (REPO / spec.physics_xml for the absolute path)."""
    name: str
    kind: str                         # smpl_scaled | smpl_variant | gain_variant | robot
    shape_xml: str                    # geometry
    physics_xml: str                  # what is simulated (the searches' --xml; robots: the scene)
    smpl: bool                        # SMPL topology (humenv observation)
    ref_task: str | None              # reference rule on a task clip ({clip}, {stem})
    ref_amass: str | None             # reference rule on an AMASS clip ({clip}); None: none made or used
    scale_rule: str                   # fixed | clip_ratio (configs/bodies.yaml s_rule)
    scale: float | None = None        # fixed: the literal (yaml s); BODIES.scale(name, clip) is the lookup to use
    scale_default: float | None = None    # clip_ratio: the value used without a clip (yaml s_default; lso_search.py --scale)
    label: str = ""                   # the on-disk asset folder name (assets/robots[_torque]/<label>), = the --body / --robot value
    parent: str | None = None
    aliases: tuple[str, ...] = ()
    display: str = ""
    gains: str | None = None          # robots: the per-joint gains JSON (--gains)
    kp_scale: float | None = None     # gain variants: kp x, kd x (kpkd_test.py) or k x (scale_body_gains.py --m)
    kd_scale: float | None = None
    k_scale: float | None = None

    # the field names before step 11 (= the yaml keys), read-only
    s = property(lambda self: self.scale, doc="alias of scale (configs/bodies.yaml key s)")
    s_rule = property(lambda self: self.scale_rule, doc="alias of scale_rule (yaml s_rule)")
    s_default = property(lambda self: self.scale_default, doc="alias of scale_default (yaml s_default)")

    def reference(self, clip, amass=None):
        """repository-relative path of the reference this body tracks on `clip` (task clip '<task>/<stem>' or AMASS name)."""
        amass = (not paths.is_task_clip(clip)) if amass is None else amass
        rule = self.ref_amass if amass else self.ref_task
        if rule is None:
            raise KeyError(f"{self.name}: no {'AMASS' if amass else 'task'} reference")
        return rule.format(clip=clip, stem=paths.stem(clip))


@dataclass(frozen=True)
class RobotSpec:
    """model/robots.py ROBOTS[name] as frozen data (lists -> tuples, dicts -> read-only mappings); robots.py keeps working."""
    name: str                         # the key (--robot): g1, h1, apollo
    display: str                      # ROBOTS[name]["name"]
    xml: str
    scene: str
    root: str
    gains_json: str
    damping: str = "robot"            # robots.py's default when the key is absent
    floor_pairs: bool = False
    src_scene: str | None = None
    sides: tuple = ()
    pairs: tuple = ()
    calib: tuple = ()
    leg: tuple = ()
    ee: tuple = ()
    extra_collision: tuple = ()
    x0: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))
    gain_source: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))
    groups: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))
    group_keys: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))

    @classmethod
    def from_config(cls, name, cfg):
        known = {f for f in cls.__dataclass_fields__} - {"name", "display"}
        extra = set(cfg) - known - {"name"}
        assert not extra, f"RobotSpec misses ROBOTS['{name}'] keys {sorted(extra)}"
        return cls(name=name, display=cfg["name"], **{k: _frozen(v) for k, v in cfg.items() if k in known})


def _frozen(x):
    if isinstance(x, dict):
        return types.MappingProxyType({k: _frozen(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return tuple(_frozen(v) for v in x)
    return x


@lru_cache(maxsize=1)
def robot_specs():
    """{name: RobotSpec} built from model/robots.py ROBOTS (imported here, on first use: it imports mujoco)."""
    from model.robots import ROBOTS
    return {k: RobotSpec.from_config(k, v) for k, v in ROBOTS.items()}


def _fill(x, key, value):
    """replace one placeholder ({name} / {cond}) and leave the others ({clip}, {stem}) for reference()"""
    return x.replace("{" + key + "}", value) if isinstance(x, str) else x


class BodyRegistry:
    """Bodies by name or alias. BODIES[name] / BODIES.get(name) -> BodySpec; scale(); reference(); names()."""

    def __init__(self, specs):
        self._specs = {b.name: b for b in specs}
        self._alias = {}
        for b in specs:
            assert b.kind in KINDS, (b.name, b.kind)
            assert b.scale_rule in S_RULES, (b.name, b.scale_rule)
            assert (b.scale is not None) == (b.scale_rule == "fixed"), (b.name, "a fixed body needs s, a clip_ratio body none")
            for a in (b.name,) + b.aliases:
                assert a not in self._alias, f"name / alias {a!r} used twice ({self._alias.get(a)}, {b.name})"
                self._alias[a] = b.name
        self._s = {}

    @classmethod
    def load(cls, path=CONFIG):
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        specs = {}
        for name, d in cfg["bodies"].items():
            d = {YAML_KEYS.get(k, k): _fill(v, "name", name) for k, v in d.items()}
            specs[name] = BodySpec(name=name, kind=d.pop("kind"), label=d.pop("label", name), aliases=tuple(d.pop("aliases", ())), **d)
        gv = cfg.get("gain_variants", {})
        for parent, d in gv.get("parents", {}).items():
            p = specs[parent]
            for cond in d["conditions"]:
                sc = gv["conditions"][cond]
                over = {k: _fill(d[k], "cond", cond) for k in ("physics_xml", "gains") if k in d}
                name = f"{parent}_{cond}"
                specs[name] = BodySpec(
                    name=name, kind="gain_variant", shape_xml=p.shape_xml, physics_xml=over.get("physics_xml", p.physics_xml),
                    smpl=p.smpl, ref_task=p.ref_task, ref_amass=p.ref_amass, scale_rule=p.scale_rule, scale=p.scale,
                    scale_default=p.scale_default,
                    label=p.label, parent=parent, aliases=tuple(_fill(a, "cond", cond) for a in d.get("aliases", ())),
                    display=f"{p.display} ({cond.replace('_', ', ')})", gains=over.get("gains", p.gains),
                    kp_scale=sc.get("kp"), kd_scale=sc.get("kd"), k_scale=sc.get("k"))
        return cls(list(specs.values()))

    # ------------------------------------------------------------ lookup
    def get(self, name):
        if isinstance(name, BodySpec):
            return name
        try:
            return self._specs[self._alias[name]]
        except KeyError:
            raise KeyError(f"unknown body {name!r}; known: {', '.join(self._specs)}") from None

    __getitem__ = get

    def __contains__(self, name):
        return name in self._alias

    def __iter__(self):
        return iter(self._specs.values())

    def names(self, kind=None):
        return [b.name for b in self._specs.values() if kind is None or b.kind == kind]

    def canonical(self, name):
        return self.get(name).name

    def robot(self, name):
        """the RobotSpec of a robot body (a G1 gain variant gives the G1's)."""
        b = self.get(name)
        assert b.kind == "robot" or (b.kind == "gain_variant" and not b.smpl), f"{b.name} is not a robot"
        return robot_specs()[b.label]

    # ------------------------------------------------------------ references and the root scale
    def reference(self, body, clip, amass=None):
        return self.get(body).reference(clip, amass)

    @staticmethod
    def adult_reference(clip, amass=None):
        """the adult's clip the root scale is read against (single_z_search.py: data/origin_motion; amass_baselines.py: ACCAD)."""
        amass = (not paths.is_task_clip(clip)) if amass is None else amass
        return paths.amass_origin_motion(clip) if amass else paths.origin_motion(clip)

    def scale(self, body, clip=None, amass=None):
        """the root scale s of `body` (on `clip`), exactly as the current code gets it (module docstring, configs/bodies.yaml)."""
        b = self.get(body)
        if b.scale_rule == "fixed":
            return b.scale
        if clip is None:
            if b.scale_default is None:
                raise ValueError(f"{b.name}: s is read off the clip (s_rule clip_ratio); pass the clip")
            return b.scale_default
        amass = (not paths.is_task_clip(clip)) if amass is None else amass
        key = (b.reference(clip, amass), self.adult_reference(clip, amass))
        if key not in self._s:
            self._s[key] = clip_ratio(paths.REPO / key[0], paths.REPO / key[1])
        return self._s[key]


def clip_ratio(ref_path, adult_path):
    """single_z_search.py's root scale, operation for operation (its `kw["scale"]` lines; amass_baselines.py scale()):
    the median over frames of the reference's root x, y over the adult clip's, where |adult| > 1e-3."""
    a = np.load(adult_path)["qpos"][:, :2]
    ref = np.load(ref_path)["qpos"]
    n = min(len(a), len(ref)); m = np.abs(a[:n]) > 1e-3
    return float(np.median(ref[:n, :2][m] / a[:n][m]))


BODIES = BodyRegistry.load()
