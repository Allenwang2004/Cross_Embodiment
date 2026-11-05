#!/usr/bin/env python3
"""render_humanoids.py -- one still per humanoid in MuJoCo (MuJoCo Menagerie + LimX Oli), to pick the next robot.

Every robot is put in the SMPL adult's surroundings (assets/robots/adult/robot.xml: sky gradient, grey checker floor,
tracking light, headlight), replacing its own sky, floor and world lights, so all images share one background.
Each robot stands in its own keyframe (stand / home / walk_pose; qpos0 if it has none), lifted so its lowest geometry
touches the floor. Per robot one 2160 x 1080 image of two 1080 px panels:
  left   close-up, the camera framed on this robot
  right  the same camera for every robot (and for the SMPL adult, Motivo's body, rendered first as the reference),
         so sizes compare across images
plus outputs/humanoids/lineup.png: the right panels' centres side by side at one scale, adult first.
Height = top - bottom of the geometry in that pose (meshes by their vertices, primitives by their bounding boxes).
usage: MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/render_humanoids.py
"""
import json, sys
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from render_panels import FONT              # noqa: E402

M = "assets/robots/menagerie"
# name, maker, scene, azimuth that shows the front three-quarter (robots facing +x: 150; the SMPL adult faces -y: 60)
ROBOTS = [
    ("SMPL adult (Motivo's body)", "reference", "assets/robots/adult/robot.xml", 60),
    ("Unitree G1", "Unitree", f"{M}/unitree_g1/scene.xml", 150),
    ("Unitree H1", "Unitree", f"{M}/unitree_h1/scene.xml", 150),
    ("Apptronik Apollo", "Apptronik", f"{M}/apptronik_apollo/scene.xml", 150),
    ("PAL TALOS", "PAL Robotics", f"{M}/pal_talos/scene_position.xml", 150),
    ("PNDbotics Adam_lite", "PNDbotics", f"{M}/pndbotics_adam_lite/scene.xml", 150),
    ("Booster T1", "Booster Robotics", f"{M}/booster_t1/scene.xml", 150),
    ("Fourier N1", "Fourier", f"{M}/fourier_n1/scene.xml", 150),
    ("Robotis OP3", "Robotis", f"{M}/robotis_op3/scene.xml", 150),
    ("ToddlerBot 2XC", "Stanford", f"{M}/toddlerbot_2xc/scene.xml", 150),
    ("ToddlerBot 2XM", "Stanford", f"{M}/toddlerbot_2xm/scene.xml", 150),
    ("Berkeley Humanoid", "UC Berkeley", f"{M}/berkeley_humanoid/scene.xml", 150),
    ("Agility Cassie", "Agility Robotics", f"{M}/agility_cassie/scene.xml", 150),
    ("LimX Oli (HU_D04)", "LimX Dynamics", "assets/robots/limx_humanoid/HU_D04_description/xml/HU_D04_01.xml", 150),
]
S, EL, SAME_DIST, SAME_Z = 1080, -8.0, 4.6, 0.95
OUT = REPO / "outputs/humanoids"


def label(frame, text, sub, px=36):
    """title and subtitle on one light band (the Menagerie skies are dark blue, so nothing is drawn on the scene)."""
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    load = (lambda n: ImageFont.truetype(str(FONT), n)) if FONT.exists() else (lambda n: ImageFont.load_default())
    band = int(px * 2.7)
    dr.rectangle([0, 0, im.width, band], fill=(252, 252, 251))
    dr.text((px // 2, int(px * 0.85)), text, fill=(11, 11, 11), font=load(px), anchor="lm")
    dr.text((px // 2, int(px * 1.95)), sub, fill=(82, 81, 78), font=load(int(px * 0.72)), anchor="lm")
    return np.asarray(im)


ADULT = REPO / "assets/robots/adult/robot.xml"


def adult_look(xml, hide_collision=False):
    """the model with its skybox and world lights swapped for the adult scene's and its floor plane(s) given the adult's
    floor material (kept, not replaced: contact pairs may refer to them by name) (MjSpec)."""
    spec = mujoco.MjSpec.from_file(str(REPO / xml))
    for t in [t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]:
        spec.delete(t)
    for l in list(spec.worldbody.lights):
        spec.delete(l)
    spec.add_texture(name="look_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.99, 0.99, 0.99], rgb2=[0.608, 0.828, 0.892], width=100, height=600)
    spec.add_texture(name="look_plane", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_NONE, markrgb=[1, 1, 1], rgb1=[0.756, 0.772, 0.76], rgb2=[0.6, 0.6, 0.6],
                     width=100, height=100)
    mat = spec.add_material(name="look_floor", reflectance=0.5, texrepeat=[1, 1], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "look_plane"
    spec.worldbody.add_light(name="look_light", pos=[0, 0, 7], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             cutoff=100, exponent=1, diffuse=[1, 1, 1], specular=[0.1, 0.1, 0.1],
                             mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, castshadow=True)   # LimX's default class turns
                                                                                            # shadows off, which also dims it
    planes = [g for g in spec.worldbody.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    for g in planes:            # LimX's is infinite (size 0): its far floor then fogs the horizon grey
        g.material, g.rgba, g.size = "look_floor", [1, 1, 1, 1], [100, 100, 0.2]
    if hide_collision:          # collision geoms drawn in group 0 (LimX): move them to the hidden group 3
        for g in spec.geoms:
            if g.type != mujoco.mjtGeom.mjGEOM_PLANE and g.group == 0 and (g.contype or g.conaffinity):
                g.group = 3
    if not planes:
        spec.worldbody.add_geom(name="look_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100, 100, 0.2],
                                material="look_floor", rgba=[1, 1, 1, 1], condim=3)
    m = spec.compile()
    a = mujoco.MjModel.from_xml_path(str(ADULT))
    m.vis.headlight.active, m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:], m.vis.headlight.specular[:] = (
        a.vis.headlight.active, a.vis.headlight.ambient, a.vis.headlight.diffuse, a.vis.headlight.specular)
    m.vis.rgba.haze[:], m.vis.rgba.fog[:], m.vis.quality.shadowsize = a.vis.rgba.haze, a.vis.rgba.fog, a.vis.quality.shadowsize
    return m


def corners(m, d, g):
    """world points bounding geom g: mesh vertices, else its local bounding-box corners."""
    R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
        return c + m.mesh_vert[a:a + n].astype(float) @ R.T
    ctr, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    box = ctr + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return c + box @ R.T


def pose(m, d):
    keys = [m.key(i).name for i in range(m.nkey)]
    pick = next((keys.index(k) for k in ("stand", "home", "walk_pose") if k in keys), 0 if keys else None)
    if pick is None:
        mujoco.mj_resetData(m, d)
    else:
        mujoco.mj_resetDataKeyframe(m, d, pick)
    mujoco.mj_forward(m, d)
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    P = np.concatenate([corners(m, d, g) for g in geoms])
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if free:
        d.qpos[m.jnt_qposadr[free[0]] + 2] -= P[:, 2].min()
        mujoco.mj_forward(m, d)
        P = np.concatenate([corners(m, d, g) for g in geoms])
    lo, hi = P.min(0), P.max(0)
    return lo, hi, pick is not None and keys[pick] or "qpos0"


def render(m, d, lookat, dist, az):
    r = mujoco.Renderer(m, height=S, width=S)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, EL
    r.update_scene(d, camera=cam); img = r.render().copy(); r.close()
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info, same = [], []
    for i, (name, maker, xml, az) in enumerate(ROBOTS):
        m = adult_look(xml, hide_collision="limx" in xml); d = mujoco.MjData(m)
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, S); m.vis.global_.offheight = max(m.vis.global_.offheight, S)
        lo, hi, key = pose(m, d)
        h, ctr = hi[2] - lo[2], (lo + hi) / 2
        joints = sum(m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in range(m.njnt))
        mass = float(m.body_subtreemass[0])
        sub = f"{maker} | {m.nu} motors, {joints} joints | {h:.2f} m tall, {mass:.1f} kg | pose: {key}"
        close = render(m, d, [ctr[0], ctr[1], h / 2], max(h, hi[0] - lo[0], hi[1] - lo[1]) * 2.0 + 0.3, az)
        far = render(m, d, [ctr[0], ctr[1], SAME_Z], SAME_DIST, az)
        tag = f"{i:02d}_{name.split(' (')[0].replace(' ', '_').replace('/', '')}"
        a = label(close, f"{i}. {name}" if i else name, sub)
        b = label(far, "same camera for every robot", "compare sizes across the images")
        Image.fromarray(np.concatenate([a, b], 1)).save(OUT / f"{tag}.png")
        same.append((name, far))
        info.append(dict(index=i, name=name, maker=maker, xml=xml, motors=int(m.nu), joints=int(joints), height_m=round(float(h), 3),
                         mass_kg=round(mass, 1), pose=key, image=f"{tag}.png"))
        print(f"{tag}: {sub}", flush=True)
    # lineup: the middle 360 px of each same-camera panel, names underneath
    W, band = 360, 70
    font = ImageFont.truetype(str(FONT), 22) if FONT.exists() else ImageFont.load_default()
    canvas = Image.new("RGB", (W * len(same), S + band), (252, 252, 251)); dr = ImageDraw.Draw(canvas)
    for k, (name, img) in enumerate(same):
        canvas.paste(Image.fromarray(img[:, S // 2 - W // 2:S // 2 + W // 2]), (k * W, 0))
        short = name.split(" (")[0]
        for li, line in enumerate([short[:18], short[18:36]] if len(short) > 18 else [short]):
            dr.text((k * W + W // 2, S + 18 + li * 26), line, fill=(11, 11, 11), font=font, anchor="mm")
    canvas.save(OUT / "lineup.png")
    (OUT / "humanoids.json").write_text(json.dumps(info, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
