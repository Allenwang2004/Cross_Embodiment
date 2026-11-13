"""shape.py -- the geometry edits of a scaled body, as plain functions (moved verbatim from scripts/scale_robot.py, OOP step 9).

  apply_scale(tree, params)   ElementTree of the adult MJCF -> every grouped body's pos, geom fromto / pos x its group's length
                              factor, geom size x its group's girth (the Pelvis box follows the torso girth); joint
                              armature / damping / stiffness are NOT touched (the torque calibration rewrites them from the
                              adult: bodybuild/torque.py). With scale_actuators the old per-joint load heuristic also scales the
                              actuators (scale_robot.py's default; every body of the current pipeline is built without it).
  measure_min_z(xml_string)   the exact lowest point of the body's geoms in the default pose (box corners, capsule caps, spheres)
  ground_pelvis(tree)         lift / lower the Pelvis so that lowest point is exactly 0 (feet flush with the floor)

scale_robot.py's docstring has the history of these choices (why the joint coefficients are left alone, why the load heuristic
replaced the per-group torque scale). The arithmetic and the text formatting ("%.6g") are those of scale_robot.py, so a body
built here is byte-identical to one it built (tests/test_bodybuild.py, tests/golden/verify_bodies.py).
"""
import itertools
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import paths  # noqa: E402
from .beta import GROUP_OF_BODY  # noqa: E402

SOURCE_XML = paths.ADULT_XML


def _scale_floats(text, factor):
    return " ".join(f"{float(x) * factor:.6g}" for x in text.split())


def compute_joint_loads(xml_string, leg_joint_names):
    """Per-joint 'load' = mass supported * lever arm from that mass's center
    of mass to the joint's anchor point -- an approximation of the static
    gravity torque this joint must supply. Uses MuJoCo's own
    body_subtreemass/subtree_com (auto-derived from geom size + density, so
    it reflects the ACTUAL scaled geometry, not a hand-derived volume
    formula). Leg joints are treated specially: during stance phase a single
    leg supports close to the WHOLE body's weight, not just its own subtree,
    so their load uses max(own_subtree_mass, total_mass - own_subtree_mass)
    at the whole-body center of mass -- everything else (arms, torso, neck,
    head) just uses its own subtree's mass/COM, i.e. what's actually distal
    to that joint. (scale_robot.py's; only used with scale_actuators=True.)"""
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_id = model.body("Pelvis").id
    total_mass = model.body_subtreemass[root_id]
    root_com = data.subtree_com[root_id]

    loads = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        body_id = model.jnt_bodyid[jid]
        anchor = data.xanchor[jid]
        if name in leg_joint_names:
            own = model.body_subtreemass[body_id]
            mass = max(own, total_mass - own)
            com = root_com
        else:
            mass = model.body_subtreemass[body_id]
            com = data.subtree_com[body_id]
        lever = np.linalg.norm(com - anchor)
        loads[name] = mass * max(lever, 1e-3)
    return loads


def apply_scale(tree, params, source_xml=SOURCE_XML, scale_actuators=True):
    """scale_robot.py apply_scale: the geometry of `tree` (in place) by `params` (the 8 BETA_AXES); see the module docstring"""
    root = tree.getroot()

    n_geoms_touched = 0
    for body in root.iter("body"):
        group = GROUP_OF_BODY.get(body.get("name"))
        if group is None:
            continue  # Pelvis (root) and anything ungrouped stay untouched
        length_scale = params[f"{group}_scale"]
        girth_scale = params[f"{group}_girth"]

        body.set("pos", _scale_floats(body.get("pos"), length_scale))

        for joint in body.findall("joint"):
            # Every joint in robot.xml is at pos="0 0 0" (relative to its
            # body's own origin) -- load-bearing for the per-joint actuator
            # load scaling below, which reads each joint's anchor and only
            # rescales body/geom pos, never joint pos. Fail loudly if a
            # future skeleton edit breaks this.
            jpos = joint.get("pos")
            assert jpos is None or tuple(float(v) for v in jpos.split()) == (0.0, 0.0, 0.0), (
                f"joint {joint.get('name')!r} has non-zero pos={jpos!r} -- "
                "apply_scale doesn't rescale joint pos, only body/geom pos"
            )

        for geom in body.findall("geom"):
            if geom.get("fromto") is not None:
                geom.set("fromto", _scale_floats(geom.get("fromto"), length_scale))
            elif geom.get("pos") is not None:
                geom.set("pos", _scale_floats(geom.get("pos"), length_scale))
            if geom.get("size") is not None:
                geom.set("size", _scale_floats(geom.get("size"), girth_scale))
                n_geoms_touched += 1

    # Pelvis's own contact box has no group -- follow torso girth.
    for body in root.iter("body"):
        if body.get("name") == "Pelvis":
            for geom in body.findall("geom"):
                if geom.get("size") is not None:
                    geom.set("size", _scale_floats(geom.get("size"), params["torso_girth"]))
                    n_geoms_touched += 1

    # root.iter("geom") also walks <default><geom .../> template elements
    # (not real instantiated geoms) -- only count geoms that are actual
    # <body> descendants. "floor" (in worldbody) is deliberately excluded:
    # it should never be scaled.
    body_geom_ids = {id(g) for b in root.iter("body") for g in b.findall("geom")}
    real_geoms = [g for g in root.iter("geom") if id(g) in body_geom_ids]
    n_untouched = len(real_geoms) - n_geoms_touched
    if n_untouched:
        print(f"NOTE: {n_untouched}/{len(real_geoms)} body geoms have no `size` attr "
              f"(e.g. inherit from a <default> class) and were not scaled -- "
              f"verify this is expected.")

    joint_to_body_name = {}
    for body in root.iter("body"):
        body_name = body.get("name")
        for joint in body.findall("joint"):
            joint_to_body_name[joint.get("name")] = body_name
    leg_joint_names = {name for name, body_name in joint_to_body_name.items()
                       if GROUP_OF_BODY.get(body_name) == "leg"}

    if not scale_actuators:
        return

    # Actuators: find each <general> actuator's owning joint, then scale
    # gain/bias/force-limit by that joint's OWN load ratio (see
    # compute_joint_loads) -- the mass+lever-arm it actually has to support,
    # not just its own body group's local inertia (scale_robot.py's history).
    original_xml_string = Path(source_xml).read_text()
    loads_orig = compute_joint_loads(original_xml_string, leg_joint_names)
    scaled_geometry_xml_string = ET.tostring(root, encoding="unicode")
    loads_new = compute_joint_loads(scaled_geometry_xml_string, leg_joint_names)

    actuator_root = root.find("actuator")
    if actuator_root is not None:
        clamp_lo, clamp_hi = 0.02, 5.0
        scale_report = []
        for act in actuator_root.findall("general"):
            joint_name = act.get("joint")
            if joint_name not in loads_orig:
                continue  # no group (shouldn't happen for the 69 hinge actuators)
            raw_scale = loads_new[joint_name] / loads_orig[joint_name]
            scale = min(max(raw_scale, clamp_lo), clamp_hi)
            scale_report.append((joint_name, raw_scale, scale))
            if raw_scale != scale:
                print(f"WARNING: {joint_name} torque scale {raw_scale:.4f} clamped to {scale:.4f}")

            if act.get("gainprm") is not None:
                act.set("gainprm", _scale_floats(act.get("gainprm"), scale))
            if act.get("biasprm") is not None:
                act.set("biasprm", _scale_floats(act.get("biasprm"), scale))
            if act.get("forcerange") is not None:
                act.set("forcerange", _scale_floats(act.get("forcerange"), scale))

        scale_report.sort(key=lambda r: r[2])
        print("per-joint actuator torque scale (lowest/highest 5):")
        for name, raw, clamped in scale_report[:5]:
            print(f"  {name:16s} {clamped:.4f}" + (f"  (raw {raw:.4f})" if raw != clamped else ""))
        print("  ...")
        for name, raw, clamped in scale_report[-5:]:
            print(f"  {name:16s} {clamped:.4f}" + (f"  (raw {raw:.4f})" if raw != clamped else ""))


def measure_min_z(xml_string):
    """Exact lowest point (z) of any non-floor geom in the default pose:
    box geoms via their 8 corners, capsules via their two cap-sphere
    centers minus radius, spheres via center minus radius. (geom_rbound is
    a bounding-SPHERE radius and is not tight for box geoms -- feet here
    are boxes -- so it under/over-estimates clearance depending on the
    box's aspect ratio, which changes non-uniformly once girth and length
    are scaled independently. Exact per-shape geometry avoids that.)"""
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    min_z = np.inf
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor":
            continue
        gtype = model.geom_type[gid]
        xpos = data.geom_xpos[gid]
        xmat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            corners = [
                xpos + xmat @ (np.array(signs) * size)
                for signs in itertools.product([1, -1], repeat=3)
            ]
            z = min(c[2] for c in corners)
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c1 = xpos + xmat @ np.array([0, 0, size[1]])
            c2 = xpos + xmat @ np.array([0, 0, -size[1]])
            z = min(c1[2], c2[2]) - size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = xpos[2] - size[0]
        else:
            z = xpos[2]
        min_z = min(min_z, z)
    return min_z


def ground_pelvis(tree):
    """Feet exactly flush with the ground (gap = 0) in the default pose: the Pelvis z shifted by -measure_min_z (scale_robot.py
    generate). Returns the shift (m)."""
    xml_string = ET.tostring(tree.getroot(), encoding="unicode")
    dz = -measure_min_z(xml_string)
    for body in tree.getroot().iter("body"):
        if body.get("name") == "Pelvis":
            x, y, z = (float(v) for v in body.get("pos").split())
            body.set("pos", f"{x:.6g} {y:.6g} {z + dz:.6g}")
    return dz
