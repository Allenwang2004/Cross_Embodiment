"""robots.py -- the humanoids Meta Motivo is driven on, as data, for model/robot_map.py and model/robot_env.py.

Per robot:
  xml / scene   its MJCF (for the SMPL <-> robot map) and a scene in the SMPL adult's surroundings (sky, grey checker
                floor, tracking light, headlight) for physics and rendering; build_scene() writes the scene for the
                Menagerie robots (assets/robots/<name>/scene.xml) with the robot's own lights removed and, where the robot
                only touches the floor through explicit contact pairs (Apollo: soles only), a pair with the floor for
                every collision geom, so hands, knees and head can land on it
  pairs         SMPL body <-> robot link carrying the same segment, pelvis first; the SMPL hand box pairs with the
                robot's hand link if it has one (model/g1_map.py: the wrist capsule rides with the forearm)
  calib         segment directions matched between the SMPL calibration pose (arms down, facing +x) and the robot's
                nominal pose: ("seg", a, b) = b - a between body origins, ("axis", joint), ("col", body, k) = column k of
                the body frame, ("geomcol", geom, k) = of a geom's frame, ("end", body) = the body's farthest collision / visual point from its origin, ("mid_up",
                a, b, c) = mean(a, b) - c; a name containing "axis" is compared up to sign
  gain_source   robot joint -> SMPL hinges whose stiffness / damping it takes (several: in series)
  groups        joint group -> (robot subtree root body, robot joint at it, SMPL body) for the gravity-moment ratio k
  damping       "robot" (default): the robot keeps its passive damping; "smpl": replaced by the SMPL passive / motor split
  ee            hands, feet, head points for the target-MSE search objective: ("body", b) origin, ("site", s),
                ("geom_top", b) = the centre of b's highest-placed geom (the G1 / H1 head), ("end", b) = b's farthest point
                (H1's forearm end: it has no hand)
G1's entries reproduce model/g1_map.py and model/g1_env.py exactly.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ADULT_XML = REPO / "assets/robots/adult/robot.xml"
MEN = "assets/robots/menagerie"


def _per_side(sides, rows):
    """rows of format strings with {S} (SMPL side L / R) and {s} (robot side prefix) -> both sides."""
    out = []
    for S, s in sides:
        out += [tuple(x.format(S=S, s=s) if isinstance(x, str) else
                      tuple(y.format(S=S, s=s) for y in x) for x in row) for row in rows]
    return out


def _smpl_pairs(sides, links, hand=None):
    pairs = []
    for S, s in sides:
        for b, l in zip(("Hip", "Knee", "Ankle", "Shoulder", "Elbow"), links):
            pairs.append((f"{S}_{b}", l.format(s=s)))
        if hand:
            pairs.append((f"{S}_Hand", hand.format(s=s)))
    return pairs


def _gains(sides, legs_arms, centre):
    out = dict(centre)
    for S, s in sides:
        for j, src in legs_arms.items():
            out[j.format(s=s)] = [f"{S}_{h}" for h in src]
    return out


# ---------------------------------------------------------------- Unitree G1 (29 DoF)
G1_SIDES = (("L", "left"), ("R", "right"))
G1 = dict(
    name="Unitree G1", xml="assets/robots/g1/g1.xml", scene="assets/robots/g1/scene.xml", root="pelvis",
    sides=G1_SIDES,
    pairs=[("Pelvis", "pelvis"), ("Chest", "torso_link")] + _smpl_pairs(
        G1_SIDES, ("{s}_hip_yaw_link", "{s}_knee_link", "{s}_ankle_roll_link", "{s}_shoulder_yaw_link",
                   "{s}_wrist_roll_link"), hand="{s}_wrist_yaw_link"),
    calib=_per_side(G1_SIDES, [
        ("{S} thigh", ("seg", "{S}_Hip", "{S}_Knee"), ("seg", "{s}_hip_pitch_link", "{s}_knee_link")),
        ("{S} shank", ("seg", "{S}_Knee", "{S}_Ankle"), ("seg", "{s}_knee_link", "{s}_ankle_pitch_link")),
        ("{S} knee axis", ("axis", "{S}_Knee_x"), ("axis", "{s}_knee_joint")),
        ("{S} foot fwd", ("col", "{S}_Ankle", "2"), ("col", "{s}_ankle_roll_link", "0")),
        ("{S} foot up", ("col", "{S}_Ankle", "1"), ("col", "{s}_ankle_roll_link", "2")),
        ("{S} upper arm axis", ("seg", "{S}_Shoulder", "{S}_Elbow"), ("axis", "{s}_shoulder_yaw_joint")),
        ("{S} forearm axis", ("seg", "{S}_Elbow", "{S}_Wrist"), ("axis", "{s}_wrist_roll_joint")),
        ("{S} hand", ("seg", "{S}_Wrist", "{S}_Hand"), ("col", "{s}_wrist_yaw_link", "0")),
        ("{S} elbow axis", ("axis", "{S}_Elbow_y"), ("axis", "{s}_elbow_joint")),
    ]) + [("torso up", ("seg", "Pelvis", "Neck"), ("mid_up", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "pelvis")),
          ("shoulder line", ("seg", "R_Shoulder", "L_Shoulder"), ("seg", "right_shoulder_pitch_link", "left_shoulder_pitch_link"))],
    x0={"left_elbow_joint": np.pi / 2, "right_elbow_joint": np.pi / 2},
    leg=("left_hip_pitch_link", "left_ankle_pitch_link"),
    gain_source=_gains(G1_SIDES, {
        "{s}_hip_pitch_joint": ["Hip_x"], "{s}_hip_roll_joint": ["Hip_z"], "{s}_hip_yaw_joint": ["Hip_y"],
        "{s}_knee_joint": ["Knee_x"], "{s}_ankle_pitch_joint": ["Ankle_x"], "{s}_ankle_roll_joint": ["Ankle_z"],
        "{s}_shoulder_pitch_joint": ["Shoulder_x"], "{s}_shoulder_roll_joint": ["Shoulder_z"],
        "{s}_shoulder_yaw_joint": ["Shoulder_y"], "{s}_elbow_joint": ["Elbow_y"],
        "{s}_wrist_roll_joint": ["Wrist_x", "Hand_x"], "{s}_wrist_pitch_joint": ["Wrist_z", "Hand_z"],
        "{s}_wrist_yaw_joint": ["Wrist_y", "Hand_y"]},
        {"waist_yaw_joint": ["Torso_y", "Spine_y", "Chest_y"], "waist_roll_joint": ["Torso_z", "Spine_z", "Chest_z"],
         "waist_pitch_joint": ["Torso_x", "Spine_x", "Chest_x"]}),
    groups={"hip": ("{s}_hip_pitch_link", "{s}_hip_pitch_joint", "{S}_Hip"),
            "knee": ("{s}_knee_link", "{s}_knee_joint", "{S}_Knee"),
            "ankle": ("{s}_ankle_pitch_link", "{s}_ankle_pitch_joint", "{S}_Ankle"),
            "shoulder": ("{s}_shoulder_pitch_link", "{s}_shoulder_pitch_joint", "{S}_Shoulder"),
            "elbow": ("{s}_elbow_link", "{s}_elbow_joint", "{S}_Elbow"),
            "wrist": ("{s}_wrist_pitch_link", "{s}_wrist_pitch_joint", "{S}_Wrist"),
            "waist": ("waist_yaw_link", "waist_yaw_joint", "Torso")},
    group_keys={"hip": ("hip",), "knee": ("knee",), "ankle": ("ankle",), "shoulder": ("shoulder",), "elbow": ("elbow",),
                "wrist": ("wrist",), "waist": ("waist",)},
    gains_json="assets/robots/g1/gains_perjoint.json",
    ee=[("body", "left_wrist_yaw_link"), ("body", "right_wrist_yaw_link"), ("site", "left_foot"), ("site", "right_foot"),
        ("geom_top", "torso_link")],
)

# ---------------------------------------------------------------- Unitree H1 (19 DoF: no wrists, 1-DoF ankles, waist yaw only)
H1_SIDES = (("L", "left"), ("R", "right"))
H1 = dict(
    name="Unitree H1", xml="assets/robots/h1/scene.xml", scene="assets/robots/h1/scene.xml", root="pelvis",
    src_scene=f"{MEN}/unitree_h1/scene.xml", sides=H1_SIDES,
    pairs=[("Pelvis", "pelvis"), ("Chest", "torso_link")] + _smpl_pairs(
        H1_SIDES, ("{s}_hip_pitch_link", "{s}_knee_link", "{s}_ankle_link", "{s}_shoulder_yaw_link", "{s}_elbow_link")),
    calib=_per_side(H1_SIDES, [
        ("{S} thigh", ("seg", "{S}_Hip", "{S}_Knee"), ("seg", "{s}_hip_pitch_link", "{s}_knee_link")),
        ("{S} shank", ("seg", "{S}_Knee", "{S}_Ankle"), ("seg", "{s}_knee_link", "{s}_ankle_link")),
        ("{S} knee axis", ("axis", "{S}_Knee_x"), ("axis", "{s}_knee")),
        ("{S} foot fwd", ("col", "{S}_Ankle", "2"), ("col", "{s}_ankle_link", "0")),
        ("{S} foot up", ("col", "{S}_Ankle", "1"), ("col", "{s}_ankle_link", "2")),
        ("{S} upper arm axis", ("seg", "{S}_Shoulder", "{S}_Elbow"), ("axis", "{s}_shoulder_yaw")),
        ("{S} forearm", ("seg", "{S}_Elbow", "{S}_Wrist"), ("end", "{s}_elbow_link")),
        ("{S} elbow axis", ("axis", "{S}_Elbow_y"), ("axis", "{s}_elbow")),
    ]) + [("torso up", ("seg", "Pelvis", "Neck"), ("mid_up", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "pelvis")),
          ("shoulder line", ("seg", "R_Shoulder", "L_Shoulder"), ("seg", "right_shoulder_pitch_link", "left_shoulder_pitch_link"))],
    x0={},
    leg=("left_hip_pitch_link", "left_ankle_link"),
    gain_source=_gains(H1_SIDES, {
        "{s}_hip_yaw": ["Hip_y"], "{s}_hip_roll": ["Hip_z"], "{s}_hip_pitch": ["Hip_x"], "{s}_knee": ["Knee_x"],
        "{s}_ankle": ["Ankle_x"], "{s}_shoulder_pitch": ["Shoulder_x"], "{s}_shoulder_roll": ["Shoulder_z"],
        "{s}_shoulder_yaw": ["Shoulder_y"], "{s}_elbow": ["Elbow_y"]},
        {"torso": ["Torso_y", "Spine_y", "Chest_y"]}),
    groups={"hip": ("{s}_hip_yaw_link", "{s}_hip_yaw", "{S}_Hip"),
            "knee": ("{s}_knee_link", "{s}_knee", "{S}_Knee"),
            "ankle": ("{s}_ankle_link", "{s}_ankle", "{S}_Ankle"),
            "shoulder": ("{s}_shoulder_pitch_link", "{s}_shoulder_pitch", "{S}_Shoulder"),
            "elbow": ("{s}_elbow_link", "{s}_elbow", "{S}_Elbow"),
            "waist": ("torso_link", "torso", "Torso")},
    group_keys={"hip": ("hip",), "knee": ("knee",), "ankle": ("ankle",), "shoulder": ("shoulder",), "elbow": ("elbow",),
                "waist": ("torso",)},
    gains_json="assets/robots/h1/gains_perjoint.json",
    ee=[("end", "left_elbow_link"), ("end", "right_elbow_link"), ("body", "left_ankle_link"), ("body", "right_ankle_link"),
        ("geom_top", "torso_link")],
)

# ---------------------------------------------------------------- Apptronik Apollo (32 DoF: 3-DoF waist and neck, 3-DoF wrists)
AP_SIDES = (("L", "l"), ("R", "r"))
APOLLO = dict(
    name="Apptronik Apollo", xml="assets/robots/apollo/scene.xml", scene="assets/robots/apollo/scene.xml",
    root="base_link", src_scene=f"{MEN}/apptronik_apollo/scene.xml", sides=AP_SIDES, floor_pairs=True,
    # Menagerie's Apollo has no collision geom on the head or neck (visual mesh only), so the head sank up to 29 cm into
    # the floor in headstand / crawl (11/03): the head mesh itself (MuJoCo collides its convex hull) is added as a
    # collision-class geom, and build_scene's floor_pairs then gives it a floor pair. The body's explicit inertial keeps
    # its mass and inertia as they were.
    extra_collision=[("neck_pitch_link", "collision_head", "afh_2_1_link")],
    # Menagerie's Apollo has 92-200 N m s/rad of passive joint damping (hips, waist), 2-27x the SMPL hinges' damping
    # ratio: it fell in every walking clip. Its passive damping is replaced by the SMPL split, scaled like the child
    # bodies' (scripts/robot_gain_scaling.py --damping smpl; user, 10/20); its own: gains_perjoint_owndamp.json
    damping="smpl",
    # the head too (10/30): Apollo has a 3-DoF neck, so the IK drives it and robot -> SMPL reads the SMPL head off it
    pairs=[("Pelvis", "base_link"), ("Chest", "torso_link"), ("Head", "neck_pitch_link")] + _smpl_pairs(
        AP_SIDES, ("{s}_hip_fe_link", "{s}_knee_fe_link", "{s}_foot_link", "{s}_shoulder_fe_link", "{s}_wrist_roll_link"),
        hand="{s}_wrist_pitch_link"),
    calib=_per_side(AP_SIDES, [
        ("{S} thigh", ("seg", "{S}_Hip", "{S}_Knee"), ("seg", "{s}_hip_fe_link", "{s}_knee_fe_link")),
        ("{S} shank", ("seg", "{S}_Knee", "{S}_Ankle"), ("seg", "{s}_knee_fe_link", "{s}_foot_link")),
        ("{S} knee axis", ("axis", "{S}_Knee_x"), ("axis", "{s}_knee_fe")),
        # Apollo's foot_link frame is pitched ~15 deg at zero angles while its sole box is flat: use the sole's frame
        ("{S} foot fwd", ("col", "{S}_Ankle", "2"), ("geomcol", "collision_{s}_sole", "0")),
        ("{S} foot up", ("col", "{S}_Ankle", "1"), ("geomcol", "collision_{s}_sole", "2")),
        ("{S} upper arm axis", ("seg", "{S}_Shoulder", "{S}_Elbow"), ("axis", "{s}_shoulder_ie")),
        ("{S} forearm axis", ("seg", "{S}_Elbow", "{S}_Wrist"), ("axis", "{s}_wrist_roll")),
        ("{S} hand", ("seg", "{S}_Wrist", "{S}_Hand"), ("end", "{s}_wrist_pitch_link")),
        ("{S} elbow axis", ("axis", "{S}_Elbow_y"), ("axis", "{s}_elbow_fe")),
    ]) + [("torso up", ("seg", "Pelvis", "Neck"), ("mid_up", "l_shoulder_aa_link", "r_shoulder_aa_link", "base_link")),
          ("shoulder line", ("seg", "R_Shoulder", "L_Shoulder"), ("seg", "r_shoulder_aa_link", "l_shoulder_aa_link"))],
    x0={},
    leg=("l_hip_fe_link", "l_foot_link"),
    gain_source=_gains(AP_SIDES, {
        "{s}_hip_ie": ["Hip_y"], "{s}_hip_aa": ["Hip_z"], "{s}_hip_fe": ["Hip_x"], "{s}_knee_fe": ["Knee_x"],
        "{s}_ankle_ie": ["Ankle_z"], "{s}_ankle_pd": ["Ankle_x"], "{s}_shoulder_aa": ["Shoulder_z"],
        "{s}_shoulder_ie": ["Shoulder_y"], "{s}_shoulder_fe": ["Shoulder_x"], "{s}_elbow_fe": ["Elbow_y"],
        "{s}_wrist_roll": ["Wrist_x", "Hand_x"], "{s}_wrist_yaw": ["Wrist_y", "Hand_y"],
        "{s}_wrist_pitch": ["Wrist_z", "Hand_z"]},
        {"torso_yaw": ["Torso_y", "Spine_y", "Chest_y"], "torso_roll": ["Torso_z", "Spine_z", "Chest_z"],
         "torso_pitch": ["Torso_x", "Spine_x", "Chest_x"], "neck_yaw": ["Neck_y", "Head_y"],
         "neck_roll": ["Neck_z", "Head_z"], "neck_pitch": ["Neck_x", "Head_x"]}),
    groups={"hip": ("{s}_hip_ie_link", "{s}_hip_ie", "{S}_Hip"),
            "knee": ("{s}_knee_fe_link", "{s}_knee_fe", "{S}_Knee"),
            "ankle": ("{s}_ankle_ie_link", "{s}_ankle_ie", "{S}_Ankle"),
            "shoulder": ("{s}_shoulder_aa_link", "{s}_shoulder_aa", "{S}_Shoulder"),
            "elbow": ("{s}_elbow_fe_link", "{s}_elbow_fe", "{S}_Elbow"),
            "wrist": ("{s}_wrist_roll_link", "{s}_wrist_roll", "{S}_Wrist"),
            "waist": ("torso_roll_link", "torso_yaw", "Torso"),
            "neck": ("neck_yaw_link", "neck_yaw", "Neck")},
    group_keys={"hip": ("hip",), "knee": ("knee",), "ankle": ("ankle",), "shoulder": ("shoulder",), "elbow": ("elbow",),
                "wrist": ("wrist",), "waist": ("torso",), "neck": ("neck",)},
    gains_json="assets/robots/apollo/gains_perjoint.json",
    ee=[("body", "l_wrist_pitch_link"), ("body", "r_wrist_pitch_link"), ("body", "l_foot_link"), ("body", "r_foot_link"),
        ("body", "neck_pitch_link")],
)

ROBOTS = {"g1": G1, "h1": H1, "apollo": APOLLO}


def collision_geoms(m):
    """non-floor geoms that collide: contype / conaffinity set, or in group 3 (Apollo's collision class, which only
    touches the floor through explicit pairs)."""
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
            and (m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_group[g] == 3)]


def build_scene(name):
    """write assets/robots/<name>/scene.xml: the Menagerie scene with the robot's lights removed, the SMPL adult's sky,
    floor material, tracking light and headlight, and (floor_pairs) a floor pair for every collision geom."""
    cfg = ROBOTS[name]
    src = REPO / cfg["src_scene"]; out = REPO / cfg["scene"]; out.parent.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(src))
    for t in [t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]:
        spec.delete(t)
    for l in list(spec.lights):
        spec.delete(l)
    spec.add_texture(name="look_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.99, 0.99, 0.99], rgb2=[0.608, 0.828, 0.892], width=100, height=600)
    spec.add_texture(name="look_plane", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_NONE, markrgb=[1, 1, 1], rgb1=[0.756, 0.772, 0.76], rgb2=[0.6, 0.6, 0.6],
                     width=100, height=100)
    mat = spec.add_material(name="look_floor", reflectance=0.5, texrepeat=[1, 1], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "look_plane"
    spec.worldbody.add_light(name="tracking_light", pos=[0, 0, 7], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             cutoff=100, exponent=1, diffuse=[1, 1, 1], specular=[0.1, 0.1, 0.1],
                             mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, castshadow=True)
    floor = None
    for g in spec.worldbody.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.material, g.rgba, g.size = "look_floor", [1, 1, 1, 1], [100, 100, 0.2]
            floor = floor or g
    floor.name = floor.name or "floor"
    a = mujoco.MjModel.from_xml_path(str(ADULT_XML))
    spec.visual.headlight.active = int(a.vis.headlight.active)
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = a.vis.headlight.ambient, a.vis.headlight.diffuse
    spec.visual.headlight.specular = a.vis.headlight.specular
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 1080
    for body, gname, mesh in cfg.get("extra_collision", []):         # (body, new geom name, mesh of its visual geom)
        b = spec.body(body); vis = next(g for g in b.geoms if g.meshname == mesh)
        b.add_geom(default=spec.find_default("collision"), name=gname, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                   pos=vis.pos, quat=vis.quat)
    if cfg.get("floor_pairs"):
        paired = {(p.geomname1, p.geomname2) for p in spec.pairs} | {(p.geomname2, p.geomname1) for p in spec.pairs}
        for i, g in enumerate(spec.geoms):
            if g.group == 3 and g.type != mujoco.mjtGeom.mjGEOM_PLANE:
                g.name = g.name or f"collision_auto_{i}"
                if (g.name, floor.name) not in paired:
                    spec.add_pair(geomname1=g.name, geomname2=floor.name, condim=3)
    spec.compile()
    root = ET.fromstring(spec.to_xml())             # paths still relative to the source scene: re-anchor them
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler"); root.insert(0, comp)
    for key in ("meshdir", "texturedir"):
        comp.set(key, os.path.relpath(src.parent / (comp.get(key) or ""), out.parent))
    out.write_text(f"<!-- generated by model/robots.py build_scene('{name}') from {cfg['src_scene']} -->\n"
                   + ET.tostring(root, encoding="unicode"))
    mujoco.MjModel.from_xml_path(str(out))          # loads back
    return out


def out_dir(name):
    """where a robot's outputs go: outputs/<name> (roundtrip/, z0/ or the G1's z0_perjoint/, search/, explain/)."""
    return REPO / "outputs" / name


def target_path(name, stem):
    """the clip mapped onto the robot kinematically (scripts/robot_roundtrip.py; the G1's from g1_roundtrip.py)."""
    return out_dir(name) / "roundtrip" / stem / ("g1.npz" if name == "g1" else "robot.npz")
