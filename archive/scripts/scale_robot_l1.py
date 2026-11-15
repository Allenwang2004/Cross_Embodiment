"""L1 morphology generator: per-SEGMENT length + girth (30 axes).

scripts/scale_robot.py exposes 8 axes -- four body GROUPS (leg / arm / torso /
head), each with one length and one girth. That locks every within-group
proportion to the adult's: upper-arm/forearm ratio, foot-length/leg-length
ratio, waist/chest ratio, hip width, shoulder width. Every generated body is
the adult under a piecewise-uniform zoom.

This script drops to the next level down -- the SEGMENT. Every body in
robot.xml belongs to exactly one of 15 mirror-unique segments, and each gets
its own `<seg>_len` and `<seg>_girth`:

    pelvis  thigh  shank  foot  toe                    (root + leg)
    torso   spine  chest  neck  head                   (axial)
    clavicle  upperarm  forearm  wrist  hand           (arm)

30 numbers, and they are the complete "length and width" description of this
skeleton at segment granularity: 15 geoms, each with one longitudinal and one
transverse scale.

HOW A SEGMENT SCALES
--------------------
Each segment has a local axis u (a capsule's fromto direction; a box's long
edge). Everything expressed in that segment's frame is split into the part
parallel to u (scaled by `_len`) and the part perpendicular (scaled by
`_girth`):

    v  ->  len * (v.u)u  +  girth * (v - (v.u)u)

applied to (a) the segment's own geom -- capsule fromto, box pos and size --
and (b) the `pos` of every CHILD body, because a child's offset is spanned by
the PARENT's segment, not its own. All bodies share the Pelvis frame (no child
body carries a quat), so the parent's u is directly usable on the child's pos.

That decomposition is what makes the two axes mean the right thing without
extra parameters. `pelvis_girth` widens the pelvis box AND pushes the two hip
joints apart (hip width); `pelvis_len` only raises the Torso attachment.
`foot_len` extends the foot box forward AND moves the toe body forward, while
`foot_girth` sets its width and ankle height. Nothing else in the skeleton is
touched.

WHAT IS NOT TOUCHED
-------------------
actuator gain/bias/forcerange, joint armature/damping/stiffness, and joint
ranges are copied from adult/robot.xml VERBATIM. This is the
`--no-actuator-scale` path of scale_robot.py and it is the only path here:
torque retuning belongs to scripts/torque_aggregate_motion_k.py, which
requires its source to carry adult's actuators unchanged (see
docs/new_body.md Step 1).

Geom `density` is untouched too, so mass follows the scaled volume. Note the
torso capsules are near-SPHERES (fromto length 0.006-0.016 m vs radius
0.076-0.100 m), so torso mass goes as girth^3, not girth^2 -- `chest_girth`
is the single strongest mass knob in the model.

The Pelvis root height is re-solved after scaling so the lowest geom corner
sits exactly at z=0 in the default pose (shared with scale_robot.py).

USAGE
-----
    # build one built-in preset (or all of them)
    uv run scripts/scale_robot_l1.py --preset ape
    uv run scripts/scale_robot_l1.py --all-presets

    # edit assets/robots_advance/<label>/parameter.json by hand, then:
    uv run scripts/scale_robot_l1.py --rebuild ape
    uv run scripts/scale_robot_l1.py --rebuild-all

    # one-off from the CLI (any of the 30 axes, unset ones default to 1.0)
    uv run scripts/scale_robot_l1.py --label mine --thigh-len 1.2 --chest-girth 1.4

Writes assets/robots_advance/<label>/{robot.xml,parameter.json}.
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_robot import measure_min_z  # exact lowest geom corner, per shape

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_XML = REPO_ROOT / "assets" / "robots" / "adult" / "robot.xml"
OUT_ROOT = REPO_ROOT / "assets" / "robots_advance"

# segment -> the bodies it owns (L/R mirrored pairs share one segment) and how
# to get its longitudinal axis. "geom" = the capsule's own fromto direction;
# "x"/"y"/"z" = that coordinate axis, for the five box segments (all boxes here
# have identity quat). Frame convention, shared by every body: +x = left,
# +y = up, +z = forward.
SEGMENTS = {
    "pelvis":   (["Pelvis"],                     "y"),     # box: long axis = vertical
    "thigh":    (["L_Hip", "R_Hip"],             "geom"),
    "shank":    (["L_Knee", "R_Knee"],           "geom"),
    "foot":     (["L_Ankle", "R_Ankle"],         "z"),     # box: long axis = forward
    "toe":      (["L_Toe", "R_Toe"],             "z"),
    "torso":    (["Torso"],                      "geom"),
    "spine":    (["Spine"],                      "geom"),
    "chest":    (["Chest"],                      "geom"),
    "neck":     (["Neck"],                       "geom"),
    "head":     (["Head"],                       "y"),
    "clavicle": (["L_Thorax", "R_Thorax"],       "geom"),
    "upperarm": (["L_Shoulder", "R_Shoulder"],   "geom"),
    "forearm":  (["L_Elbow", "R_Elbow"],         "geom"),
    "wrist":    (["L_Wrist", "R_Wrist"],         "geom"),
    "hand":     (["L_Hand", "R_Hand"],           "x"),     # box: long axis = lateral (T-pose)
}
SEGMENT_OF_BODY = {b: seg for seg, (bodies, _) in SEGMENTS.items() for b in bodies}

# beta vector order. Grouped by chain so a hand-edited parameter.json reads
# proximal -> distal.
L1_AXES = [f"{seg}_{kind}" for seg in SEGMENTS for kind in ("len", "girth")]

IDENTITY = {a: 1.0 for a in L1_AXES}


# --- human growth curve -------------------------------------------------
# Proportion is not scale-invariant, so a short body must not be a shrunken
# adult. These are the two proportions that change most with age, expressed as
# a RATIO to the adult's value at stature ratio s. Anchors (stature vs adult
# 1.736 m; head height / stature; subischial leg length / stature), from
# standard growth references:
#
#     age      1yr    3yr    6yr    9yr   12yr  adult
#     s        0.43   0.55   0.67   0.77   0.86   1.00
#     head/H   0.210  0.195  0.180  0.165  0.150  0.133   -> linear in s
#     leg/H    0.370  0.415  0.440  0.460  0.475  0.480   -> ~ s**0.22
#
# HEAD_K and LEG_E are those two fits. Everything else keeps a plain power law.
# The values below are RELATIVE to the stature ratio, so they survive the
# uniform length rescale that solves for stature (ratios are preserved by it).
HEAD_K = 1.015          # head/H ratio = 1 + HEAD_K*(1-s)
LEG_E = 0.22            # leg/H ratio  = s**LEG_E
ARM_E = 0.10            # arm span/H, milder than the legs
TRUNK_E = -0.07         # trunk is RELATIVELY longer on a short body
GIRTH_E = -0.04         # and relatively slightly stockier

# segment -> exponent e such that the segment's length axis is s**(1+e)
SEG_E = dict(thigh=LEG_E, shank=LEG_E, foot=-0.05, toe=-0.05,
             upperarm=ARM_E, forearm=ARM_E, wrist=ARM_E, hand=-0.05,
             torso=TRUNK_E, spine=TRUNK_E, chest=TRUNK_E, pelvis=-0.05,
             clavicle=-0.03, neck=-0.05, head=None)   # head uses HEAD_K


def growth_axes(s):
    """The 30 axes of an 'average person of stature ratio s' -- the spine of the
    sampler and of the interactive UI. Deviations from a real body are then
    layered on top as multiplicative factors."""
    p = {}
    for seg, e in SEG_E.items():
        rel = (1.0 + HEAD_K * (1.0 - s)) if e is None else s ** e
        p[f"{seg}_len"] = s * rel
        p[f"{seg}_girth"] = s * rel * s ** GIRTH_E
    return p


def _p(**kw):
    """A preset is the identity with a few axes overridden."""
    unknown = set(kw) - set(L1_AXES)
    assert not unknown, f"unknown axes {sorted(unknown)}"
    return {**IDENTITY, **kw}


# Ten bodies chosen to spread PROPORTION, not size: every one is meant to sit
# close to the adult's stature (see the table printed by --all-presets) while
# rearranging where that height comes from. Anything that would read as "a
# giant" -- a uniform up-scale -- is deliberately absent, and the two axes that
# actually buy stature (thigh/shank _len) are traded against torso length in
# opposite directions across the set.
PRESETS = {
    # Sitting height >> leg length. Short femur/tibia, long lumbar+thoracic.
    "long_torso": _p(thigh_len=0.80, shank_len=0.80, foot_len=0.95,
                     torso_len=1.30, spine_len=1.30, chest_len=1.25, neck_len=1.10,
                     upperarm_len=0.92, forearm_len=0.92),
    # The opposite: high waist, long legs, short trunk.
    "long_leg": _p(thigh_len=1.16, shank_len=1.18,
                   torso_len=0.72, spine_len=0.72, chest_len=0.78, neck_len=0.90,
                   pelvis_len=0.85, chest_girth=0.90),
    # Long arms reaching past the knee, short legs, deep chest, narrow hips.
    "ape": _p(clavicle_len=1.25, upperarm_len=1.35, forearm_len=1.40, hand_len=1.20,
              upperarm_girth=1.25, forearm_girth=1.20,
              thigh_len=0.84, shank_len=0.80, thigh_girth=1.05,
              chest_girth=1.35, chest_len=1.10, pelvis_girth=0.88),
    # Vestigial arms, heavy legs, big feet -- the extreme of arm/leg asymmetry.
    "t_rex": _p(clavicle_len=0.85, upperarm_len=0.55, forearm_len=0.50,
                wrist_len=0.60, hand_len=0.70, hand_girth=0.75,
                upperarm_girth=0.80, forearm_girth=0.80,
                thigh_girth=1.40, shank_girth=1.35, thigh_len=1.04,
                # pelvis widened to 1.15 so the thickened thighs clear each
                # other -- at 1.0 they overlap by 1.9 cm in the default pose.
                pelvis_girth=1.15,
                foot_len=1.30, foot_girth=1.20, toe_len=1.30, toe_girth=1.15),
    # Barrel trunk on thin limbs: mass concentrated high and central.
    "barrel": _p(chest_girth=1.50, torso_girth=1.45, spine_girth=1.45, pelvis_girth=1.30,
                 chest_len=1.10, thigh_girth=0.75, shank_girth=0.72,
                 upperarm_girth=0.72, forearm_girth=0.70, clavicle_girth=0.85,
                 thigh_len=0.92, shank_len=0.92),
    # Uniformly thin, slightly elongated -- the low-inertia end of the set.
    "reed": _p(pelvis_girth=0.74, torso_girth=0.72, spine_girth=0.72, chest_girth=0.74,
               neck_girth=0.80, thigh_girth=0.70, shank_girth=0.70,
               upperarm_girth=0.68, forearm_girth=0.68, clavicle_girth=0.80,
               hand_girth=0.85, foot_girth=0.85, toe_girth=0.85,
               thigh_len=1.04, shank_len=1.04, spine_len=1.05),
    # Oversized contact surfaces and hands, small head. Support polygon ~2x.
    "bigfoot": _p(foot_len=1.45, foot_girth=1.40, toe_len=1.40, toe_girth=1.35,
                  hand_len=1.40, hand_girth=1.35, wrist_girth=1.20,
                  head_len=0.85, head_girth=0.80, neck_len=0.90,
                  thigh_len=0.94, shank_len=0.94),
    # Head-dominated: big head on a short neck. Moves the COM up and forward.
    "bobblehead": _p(head_len=1.35, head_girth=1.40, neck_len=0.60, neck_girth=1.15,
                     thigh_len=0.92, shank_len=0.92, torso_len=0.95,
                     upperarm_len=0.95, forearm_len=0.95),
    # V-taper: wide clavicles and chest over a narrow pelvis.
    "broad_shoulder": _p(clavicle_len=1.45, clavicle_girth=1.25,
                         chest_girth=1.30, chest_len=1.10,
                         upperarm_girth=1.20, forearm_girth=1.15,
                         pelvis_girth=0.78, thigh_girth=0.88, shank_girth=0.90,
                         thigh_len=0.96, shank_len=0.96),
    # True pear: wide pelvis and thighs, narrow shoulders and chest.
    "wide_hip": _p(pelvis_girth=1.50, pelvis_len=1.10,
                   thigh_girth=1.40, shank_girth=1.10,
                   clavicle_len=0.78, clavicle_girth=0.85,
                   chest_girth=0.85, torso_girth=0.95,
                   upperarm_girth=0.85, forearm_girth=0.85,
                   thigh_len=0.94, shank_len=0.94),
    # Short, thick limbs on a normal trunk -- low stature without shrinking
    # the torso, i.e. achondroplasia-like proportions.
    "stubby": _p(thigh_len=0.68, shank_len=0.66, upperarm_len=0.70, forearm_len=0.68,
                 thigh_girth=1.25, shank_girth=1.25,
                 upperarm_girth=1.20, forearm_girth=1.20,
                 foot_len=1.10, hand_len=1.05, head_girth=1.10,
                 pelvis_girth=1.10),
    # Distal-heavy legs: short femur, long tibia, small feet.
    "flamingo": _p(thigh_len=0.78, thigh_girth=1.10, shank_len=1.30, shank_girth=0.78,
                   foot_len=0.80, foot_girth=0.85, toe_len=0.85,
                   chest_girth=0.90, torso_len=0.92, spine_len=0.92),
}


def _axis_of(body, spec):
    """Unit longitudinal axis of a segment, in the shared body frame."""
    if spec != "geom":
        return np.eye(3)["xyz".index(spec)]
    for geom in body.findall("geom"):
        if geom.get("fromto") is not None:
            v = np.array([float(x) for x in geom.get("fromto").split()])
            d = v[3:] - v[:3]
            n = np.linalg.norm(d)
            assert n > 1e-6, f"degenerate fromto on {body.get('name')}"
            return d / n
    raise AssertionError(f"{body.get('name')} has no fromto geom but axis='geom'")


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def apply_l1(tree, params):
    """Scale every segment by its own (len, girth) pair. Returns a per-segment
    report of what was touched, for the caller to sanity-check."""
    root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}
    missing = set(SEGMENT_OF_BODY) - set(bodies)
    assert not missing, f"source xml is missing bodies {sorted(missing)}"
    unmapped = set(bodies) - set(SEGMENT_OF_BODY)
    assert not unmapped, f"source xml has bodies with no segment: {sorted(unmapped)}"

    # Resolve every axis BEFORE mutating anything -- a capsule's fromto is both
    # the axis source and a scaling target, and scaling it with unequal
    # len/girth rotates it slightly.
    axes = {name: _axis_of(bodies[name], SEGMENTS[SEGMENT_OF_BODY[name]][1])
            for name in bodies}

    report = {}
    for name, body in bodies.items():
        seg = SEGMENT_OF_BODY[name]
        u = axes[name]
        a, b = params[f"{seg}_len"], params[f"{seg}_girth"]

        def sc(v):
            v = np.asarray(v, dtype=float)
            par = np.dot(v, u) * u
            return a * par + b * (v - par)

        n_geom = 0
        for geom in body.findall("geom"):
            if geom.get("fromto") is not None:
                v = np.array([float(x) for x in geom.get("fromto").split()])
                geom.set("fromto", _fmt(np.concatenate([sc(v[:3]), sc(v[3:])])))
                # capsule `size` is a scalar radius -> purely transverse
                geom.set("size", _fmt(b * float(geom.get("size"))))
            else:
                if geom.get("pos") is not None:
                    geom.set("pos", _fmt(sc([float(x) for x in geom.get("pos").split()])))
                size = [float(x) for x in geom.get("size").split()]
                if len(size) == 3:                     # box half-extents, per axis
                    geom.set("size", _fmt(sc(size)))
                else:                                  # the massless "nose" capsule
                    geom.set("size", _fmt(b * np.array(size)))
            n_geom += 1

        # A child's pos is an offset spanned by THIS segment.
        n_child = 0
        for child in body.findall("body"):
            child.set("pos", _fmt(sc([float(x) for x in child.get("pos").split()])))
            n_child += 1

        report[name] = (seg, a, b, n_geom, n_child)
    return report


def describe(xml_string):
    """Stature / mass / balance summary of a finished body."""
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Exact highest point, per shape -- the mirror of measure_min_z. A
    # bounding sphere would inflate the boxy head by several cm, and stature
    # is the one number these presets are constrained on.
    top = -np.inf
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor":
            continue
        gtype, xpos = model.geom_type[gid], data.geom_xpos[gid]
        xmat, size = data.geom_xmat[gid].reshape(3, 3), model.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            z = max((xpos + xmat @ (np.array(s) * size))[2]
                    for s in itertools.product([1, -1], repeat=3))
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = max((xpos + xmat @ np.array([0, 0, sgn * size[1]]))[2]
                    for sgn in (1, -1)) + size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = xpos[2] + size[0]
        else:
            z = xpos[2]
        top = max(top, z)

    # self-collisions at rest (floor contacts excluded) -- the failure mode of
    # aggressive girth: a limb buried inside the trunk.
    floor = model.geom("floor").id
    self_con = sum(1 for i in range(data.ncon)
                   if floor not in (data.contact[i].geom1, data.contact[i].geom2))

    groups = {"leg": ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
              "arm": ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
                      "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
              "trunk": ["Pelvis", "Torso", "Spine", "Chest", "Neck"],
              "head": ["Head"]}
    return {
        "root_h": float(model.body("Pelvis").pos[2]),
        "stature": float(top),
        "mass": float(model.body_mass.sum()),
        "com_h": float(data.subtree_com[model.body("Pelvis").id][2]),
        "self_contacts": self_con,
        "mass_frac": {g: float(sum(model.body(n).mass[0] for n in names) / model.body_mass.sum())
                      for g, names in groups.items()},
    }


def build_body(params):
    """Scale + ground-correct, in memory. Returns (tree, xml_string). Split out
    of generate() so samplers can evaluate a candidate without touching disk."""
    tree = ET.parse(SOURCE_XML)
    apply_l1(tree, params)

    # Feet exactly flush with z=0 in the default pose.
    dz = -measure_min_z(ET.tostring(tree.getroot(), encoding="unicode"))
    pelvis = next(b for b in tree.getroot().iter("body") if b.get("name") == "Pelvis")
    x, y, z = (float(v) for v in pelvis.get("pos").split())
    pelvis.set("pos", f"{x:.6g} {y:.6g} {z + dz:.6g}")
    return tree, ET.tostring(tree.getroot(), encoding="unicode")


def generate(label, params, out_root=OUT_ROOT, quiet=False, extra=None):
    for axis in L1_AXES:
        assert axis in params, f"{label}: parameter.json is missing '{axis}'"
        v = params[axis]
        if not 0.25 <= v <= 2.5:
            # matches sample_l1_bodies.AXIS_LO/AXIS_HI -- outside this the
            # segment stops reading as the body part it is meant to be
            print(f"WARNING: {label}.{axis} = {v} is outside the sane range [0.25, 2.5]")

    tree, xml_string = build_body(params)
    info = describe(xml_string)

    body_dir = out_root / label
    body_dir.mkdir(parents=True, exist_ok=True)
    out_xml = body_dir / "robot.xml"
    tree.write(out_xml)
    out_json = body_dir / "parameter.json"
    out_json.write_text(json.dumps(
        {"label": label, "schema": "l1-30", "scale_actuators": False, **(extra or {}),
         **{a: round(float(params[a]), 4) for a in L1_AXES}}, indent=2) + "\n")

    if not quiet:
        print(f"wrote {out_xml}")
        print(f"wrote {out_json}")
        print(f"  stature {info['stature']:.3f} m   root {info['root_h']:.3f} m   "
              f"mass {info['mass']:.1f} kg   self-contacts {info['self_contacts']}")
    return info


def load_params(path):
    d = json.loads(Path(path).read_text())
    extra = set(d) - set(L1_AXES) - {"label", "schema", "scale_actuators", "split",
                                      "notes", "stature_ratio", "seed"}
    if extra:
        print(f"NOTE: ignoring non-axis keys in {path}: {sorted(extra)}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--all-presets", action="store_true", help="build every built-in preset")
    ap.add_argument("--rebuild", metavar="LABEL",
                    help="re-read assets/robots_advance/LABEL/parameter.json and rewrite robot.xml")
    ap.add_argument("--rebuild-all", action="store_true",
                    help="rebuild every body already in assets/robots_advance/")
    ap.add_argument("--from-json", metavar="PATH", help="build from an arbitrary parameter.json")
    ap.add_argument("--label", help="output folder name (defaults to the preset/json label)")
    for axis in L1_AXES:
        ap.add_argument(f"--{axis.replace('_', '-')}", type=float, default=None)
    args = ap.parse_args()

    jobs = []  # (label, params)
    if args.all_presets:
        jobs += [(n, dict(PRESETS[n])) for n in PRESETS]
    if args.preset:
        jobs.append((args.label or args.preset, dict(PRESETS[args.preset])))
    if args.rebuild_all:
        for d in sorted(p for p in OUT_ROOT.iterdir() if (p / "parameter.json").exists()):
            jobs.append((d.name, load_params(d / "parameter.json")))
    if args.rebuild:
        p = OUT_ROOT / args.rebuild / "parameter.json"
        assert p.exists(), f"no such body: {p}"
        jobs.append((args.rebuild, load_params(p)))
    if args.from_json:
        d = load_params(args.from_json)
        jobs.append((args.label or d.get("label") or Path(args.from_json).parent.name, d))
    if not jobs:
        if args.label is None:
            ap.error("pass --preset / --all-presets / --rebuild / --rebuild-all / --from-json, "
                     "or --label with per-axis overrides")
        jobs.append((args.label, dict(IDENTITY)))

    # CLI overrides apply on top of whatever each job started from.
    overrides = {a: getattr(args, a) for a in L1_AXES if getattr(args, a) is not None}
    if overrides and len(jobs) > 1:
        ap.error("per-axis overrides only make sense with a single body")

    rows = []
    for label, params in jobs:
        params = {**IDENTITY, **{k: v for k, v in params.items() if k in L1_AXES}, **overrides}
        rows.append((label, generate(label, params)))

    if len(rows) > 1:
        base = describe(SOURCE_XML.read_text())
        print(f"\n{'body':<16}{'stature':>9}{'vs adult':>10}{'mass':>9}{'vs adult':>10}"
              f"{'COM h':>8}{'leg%':>7}{'arm%':>7}{'trunk%':>8}{'selfc':>7}")
        print(f"{'adult (source)':<16}{base['stature']:9.3f}{1.0:10.3f}{base['mass']:9.1f}"
              f"{1.0:10.3f}{base['com_h']:8.3f}{100*base['mass_frac']['leg']:7.1f}"
              f"{100*base['mass_frac']['arm']:7.1f}{100*base['mass_frac']['trunk']:8.1f}"
              f"{base['self_contacts']:7d}")
        for label, i in sorted(rows, key=lambda r: r[1]["stature"]):
            print(f"{label:<16}{i['stature']:9.3f}{i['stature']/base['stature']:10.3f}"
                  f"{i['mass']:9.1f}{i['mass']/base['mass']:10.3f}{i['com_h']:8.3f}"
                  f"{100*i['mass_frac']['leg']:7.1f}{100*i['mass_frac']['arm']:7.1f}"
                  f"{100*i['mass_frac']['trunk']:8.1f}{i['self_contacts']:7d}")


if __name__ == "__main__":
    main()
