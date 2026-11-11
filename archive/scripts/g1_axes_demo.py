#!/usr/bin/env python3
"""g1_axes_demo.py -- what pitch, roll and yaw are, shown on the G1's left hip (for explaining the SMPL <-> G1 map).

Writes to outputs/g1/explain/ (English labels):
  pitch_roll_yaw.png   columns: standing / hip pitch -45 deg / hip roll +45 deg / hip yaw +45 deg; rows: front view,
                       left-side view, oblique view; the hinge that moved drawn as an arrow along its axis (positive
                       rotation by the right-hand rule), the G1 drawn see-through so an axis inside the body shows
  pitch_roll_yaw.mp4   each of the three hinges swung out to 45 deg and back, in turn
  frames.png           the two bodies side by side, both facing +x in the nominal pose of model/g1_map.py, with the left
                       hip's local frame drawn (red x, green y, blue z): SMPL L_Hip (x left, y up, z forward) vs G1
                       left_hip_pitch_link (x forward, y left, z up)
  follow.mp4 / .png    the G1's left hip does pitch -90 deg, then roll 45 deg, then yaw 45 deg; the thigh's local frame
                       (bold) moves with it, the pelvis frame (faint) stays, and the hinge turning now is a yellow line:
                       each hinge's axis is fixed in the link before it, so pitch carries the roll axis to vertical
  smpl_rotation.png    why the SMPL root carries R_x(90 deg) and what H = R_z(90 deg) adds: the SMPL with an identity
                       root rotation (lying: its data is y-up), at qpos0 (stood up, facing -y), after H (facing +x),
                       and the G1 at rest (facing +x); world axes on the floor, the pelvis's local axes, a 'front' arrow
  pairs.png            the 14 SMPL body <-> G1 link pairs of model/g1_map.py, each pair one colour and one number on
                       both bodies (nominal pose, facing the camera); bodies with no partner in light grey; a key below
  g1_to_smpl.png       one walking frame: the G1 pose and the SMPL pose G1Map.g1_to_smpl_qpos makes from it, paired
                       segments in the colours of pairs.png (their orientation is copied), SMPL joints with no G1
                       partner in grey (they take Motivo's mean posture), both facing the camera the same way
  ik_residual.png      what an IK residual is: the walk frame where the G1's left foot misses its target most (15 deg);
                       the foot's achieved toe direction (bold) vs the target's (dashed), sole normals on top of each
                       other: the whole miss is a turn about the vertical, the ankle yaw the G1 does not have
  order.png            the same two numbers (yaw 90 deg, roll 45 deg) in the two hinge orders: the G1's hip (roll
                       before yaw) and the SMPL hip (yaw before roll)
usage: MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/g1_axes_demo.py
"""
import os
import sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/g1/explain"
# px, font, cam, project, label, hcat (and FONT, S, K) live in crossenbodiment/draw.py since refactor step 4 (unchanged): robot_pairs.py
# uses them too. px reads draw.K, which this script never changes.
sys.path.insert(0, str(REPO))
from crossenbodiment.draw import FONT, K, S, cam, font, hcat, label, project, px   # noqa: E402,F401
COL = {"pitch": (0.85, 0.2, 0.2, 1), "roll": (0.2, 0.65, 0.25, 1), "yaw": (0.2, 0.4, 0.9, 1)}
XYZ = [(210, 25, 25), (20, 150, 50), (25, 70, 220)]


def save_mp4(path, frames, fps=30):
    """One size for every frame (title bands differ: pad white on top), even for libx264, no resampling."""
    H = max(f.shape[0] for f in frames); W = max(f.shape[1] for f in frames)
    H, W = H + H % 2, W + W % 2
    frames = [np.pad(f, ((H - f.shape[0], 0), (0, W - f.shape[1]), (0, 0)), constant_values=255) for f in frames]
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def render(m, d, r, c, axes=()):
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=c)
    for jid, rgba in axes:
        p, a = d.xanchor[jid], d.xaxis[jid]
        g = r.scene.geoms[r.scene.ngeom]; r.scene.ngeom += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.018, p - 0.35 * a, p + 0.35 * a)
    return r.render().copy()


def arrow2d(dr, a, b, col, w, text=None, fnt=None):
    """A 2D arrow a -> b with a white outline, optionally labelled beyond its tip."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; L = np.linalg.norm(d)
    if L < 1e-6:
        return
    d /= L; n = np.array([-d[1], d[0]]); base = b - (3 * w + 4) * d
    head = [tuple(b), tuple(base + (1.6 * w + 2) * n), tuple(base - (1.6 * w + 2) * n)]
    dr.line([tuple(a), tuple(base)], fill=(255, 255, 255), width=w + px(4))
    dr.polygon(head, fill=(255, 255, 255))
    dr.line([tuple(a), tuple(base)], fill=col, width=w)
    dr.polygon(head, fill=col)
    if text:
        q = b + px(16) * d
        tw = dr.textlength(text, font=fnt)
        x = q[0] - tw / 2 + d[0] * tw / 2            # push the words out along the arrow, clear of its head
        y = q[1] - fnt.size / 2 + d[1] * fnt.size / 2
        Wi, Hi = dr._image.size                      # keep the words inside the picture
        x = min(max(x, px(4)), Wi - tw - px(4)); y = min(max(y, px(4)), Hi - fnt.size - px(4))
        dr.text((x, y), text, fill=col, font=fnt, stroke_width=px(3), stroke_fill=(255, 255, 255))


def triad(m, d, r, c, body, length, names, W, H):
    """Render the body, then draw its local frame on top in 2D (projected through the same camera), so the axes are
    never hidden inside the body; each arrow is labelled at its tip."""
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=c)
    img = r.render().copy()
    o, R = d.xpos[body].copy(), d.xmat[body].reshape(3, 3).copy()
    t0, *tips = project(r, [o] + [o + length * R[:, k] for k in range(3)], W, H)
    im = Image.fromarray(img); dr = ImageDraw.Draw(im); f = font(22)
    for q, t, col in zip(tips, names, XYZ):
        arrow2d(dr, t0, q, col, px(5), t, f)
    dr.ellipse([t0[0] - px(5), t0[1] - px(5), t0[0] + px(5), t0[1] + px(5)], fill=(30, 30, 30))
    return np.array(im)


def see_through(m, alpha=0.45):
    floor = m.geom("floor").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0 else -1
    for g in range(m.ngeom):
        if g != floor:
            m.geom_rgba[g, 3] = alpha
    for k in range(m.nmat):
        if floor < 0 or k != m.geom_matid[floor]:
            m.mat_rgba[k, 3] = alpha


def follow_frame(m, d, r, views, axis_jid, lines, W):
    """One frame of follow.mp4: per view, the robot, the pelvis frame (faint), the thigh frame (bold), the hinge
    turning now (yellow), under a title band."""
    mujoco.mj_forward(m, d)
    thigh, pelvis = m.body("left_hip_yaw_link").id, m.body("pelvis").id
    f = font(22)
    out = []
    for vname, c in views:
        r.update_scene(d, camera=c); img = r.render().copy()
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        if axis_jid is not None:
            pa, ax = d.xanchor[axis_jid], d.xaxis[axis_jid]
            (u0, v0), (u1, v1) = project(r, [pa - 0.32 * ax, pa + 0.32 * ax], W, W)
            dr.line([(u0, v0), (u1, v1)], fill=(255, 255, 255), width=px(14))
            dr.line([(u0, v0), (u1, v1)], fill=(240, 190, 0), width=px(9))
        po, Rp = d.xpos[pelvis].copy(), d.xmat[pelvis].reshape(3, 3).copy()
        po = po + np.array([0, -0.12, 0.12])                     # beside the right hip, clear of the left leg
        p0, *pt = project(r, [po] + [po + 0.18 * Rp[:, k] for k in range(3)], W, W)
        for q, col in zip(pt, [(235, 150, 150), (150, 205, 160), (150, 170, 235)]):
            arrow2d(dr, p0, q, col, px(3))
        to, Rt = d.xpos[thigh].copy(), d.xmat[thigh].reshape(3, 3).copy()
        t0, *tt = project(r, [to] + [to + 0.26 * Rt[:, k] for k in range(3)], W, W)
        for q, col, t in zip(tt, XYZ, "xyz"):
            arrow2d(dr, t0, q, col, px(5), t, f)
        dr.ellipse([t0[0] - px(5), t0[1] - px(5), t0[0] + px(5), t0[1] + px(5)], fill=(30, 30, 30))
        out.append(label(np.array(im), [f"{lines[0]} ({vname})"] + list(lines[1:])))
    return hcat(out)


def make_follow(W=S):
    m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml")); d = mujoco.MjData(m)
    r = mujoco.Renderer(m, W, W)
    J = {k: m.joint(f"left_hip_{k}_joint") for k in ("pitch", "roll", "yaw")}
    views = [("oblique", cam(-135, (0, 0.05, 0.6), dist=2.3, el=-22)), ("left side", cam(-90, (0.1, 0.05, 0.6), dist=2.3, el=-8))]
    base = np.zeros(m.nq); base[2] = 0.80; base[3] = 1.0
    steps = [("pitch", -90, "Step 1: pitch", ["Axis (yellow) is fixed in the pelvis, pointing left-right.",
                                              "The whole thigh frame turns with it; the leg kicks up to horizontal."]),
             ("roll", 45, "Step 2: roll", ["Its axis was carried to vertical by step 1,",
                                           "so the leg sweeps sideways in the horizontal plane."]),
             ("yaw", 45, "Step 3: yaw", ["Its axis runs along the thigh (now pointing fore-aft).",
                                         "The leg twists about itself; the thigh's x and y turn."])]
    ang = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
    frames, keys = [], []

    def put():
        d.qpos[:] = base
        for k, v in ang.items():
            d.qpos[J[k].qposadr[0]] = np.radians(v)
    put()
    first = follow_frame(m, d, r, views, None, ["Start: all joints at 0°", "Bold arrows: the thigh's local frame.",
                                                "Faint arrows: the pelvis frame (fixed)."], W)
    frames += [first] * 30; keys.append(first)
    for k, target, l1, l2 in steps:
        for t in np.linspace(0, 1, 60):
            ang[k] = target * (0.5 - 0.5 * np.cos(np.pi * t)); put()
            frames.append(follow_frame(m, d, r, views, J[k].id, [f"{l1} {ang[k]:+.0f}°"] + l2, W))
        frames += [frames[-1]] * 30; keys.append(frames[-1])
    save_mp4(OUT / "follow.mp4", frames)
    Image.fromarray(np.concatenate(keys, 0)).save(OUT / "follow.png")


def make_base_rotation(W=S):
    sys.path.insert(0, str(REPO))
    from model.exact_obs import _LowestPoint
    ms = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml")); ds = mujoco.MjData(ms)
    mg = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml")); dg = mujoco.MjData(mg)
    rs, rg = mujoco.Renderer(ms, W, W), mujoco.Renderer(mg, W, W)
    low = _LowestPoint(ms)
    c = cam(-140, (0, 0, 0.45), dist=3.6, el=-28)
    fl, fb = font(20), font(24)
    world = [(210, 25, 25), (20, 150, 50), (25, 70, 220)]

    def smpl(quat):
        ds.qpos[:] = ms.qpos0; ds.qpos[:2] = 0; ds.qpos[3:7] = quat; ds.qpos[2] = 0
        mujoco.mj_kinematics(ms, ds); ds.qpos[2] = -low(ds) + 0.01
        mujoco.mj_forward(ms, ds)
        return ms, ds, rs, ms.body("Pelvis").id, 2                 # SMPL front = local +z

    def g1():
        dg.qpos[:] = 0; dg.qpos[2] = 0.80; dg.qpos[3] = 1; mujoco.mj_forward(mg, dg)
        return mg, dg, rg, mg.body("pelvis").id, 0                 # G1 front = local +x

    def panel(model, data, r, body, front, lines):
        r.update_scene(data, camera=c); img = r.render().copy()
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        # world axes: their on-screen directions at the body's feet, drawn in a fixed inset (bottom right)
        g0, *gt = project(r, [np.zeros(3)] + list(np.eye(3)), W, W)
        v = [np.subtract(q, g0) for q in gt]; sc = px(42) / max(np.linalg.norm(x) for x in v)
        x0, y0 = W - px(150), px(8)                  # top right, in the sky
        dr.rounded_rectangle([x0, y0, W - px(8), y0 + px(142)], radius=px(8), fill=(250, 250, 250),
                             outline=(190, 190, 190), width=max(1, px(1)))
        dr.text((x0 + px(8), y0 + px(4)), "world axes", fill=(90, 90, 90), font=font(14))
        A = np.array([x0 + px(72), y0 + px(88)])
        for x, col, t in zip(v, world, ("X", "Y", "Z")):
            arrow2d(dr, A, A + sc * x, col, px(3), t, fl)
        po, R = data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()
        b0, *bt = project(r, [po] + [po + 0.32 * R[:, k] for k in range(3)], W, W)
        for k, (q, col, t) in enumerate(zip(bt, XYZ, ("local x", "local y", "local z"))):
            if k == front:       # the front arrow below is this axis
                continue
            arrow2d(dr, b0, q, col, px(4), t, fl)
        f0, ft = project(r, [po, po + 0.6 * R[:, front]], W, W)
        arrow2d(dr, f0, ft, (235, 130, 0), px(7), f"front (= local {'xyz'[front]})", fl)
        dr.ellipse([b0[0] - px(5), b0[1] - px(5), b0[0] + px(5), b0[1] + px(5)], fill=(30, 30, 30))
        return label(np.array(im), lines)

    rx90 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    panels = [
        panel(*smpl([1, 0, 0, 0]), ["1) SMPL, root rotation = identity",
                                    "Its data is y-up (knee at -0.375 in y), so the body lies along world Y."]),
        panel(*smpl(rx90), ["2) SMPL at qpos0: R_x(90°)",
                            "From euler=\"90 0 0\" in robot.xml: local y -> world +Z, so it stands;",
                            "local z (front) -> world -Y, so it faces -Y."]),
        panel(*smpl([0.5, 0.5, 0.5, 0.5]), ["3) Then H = R_z(90°)",
                                            "Turn 90° about world Z: front -Y -> +X, left +X -> +Y.",
                                            "Now it faces +X, like the G1."]),
        panel(*g1(), ["4) G1 at rest (identity root)", "Faces +X; its left is +Y; its local axes = world axes."]),
    ]
    Image.fromarray(hcat(panels)).save(OUT / "smpl_rotation.png")


def make_pairs(W=int(S * 1.25)):
    sys.path.insert(0, str(REPO))
    from model.g1_map import G1Map, PAIRS, CALIB_POSE
    gm = G1Map()
    kinds = {"Pelvis": ("pelvis", (0.95, 0.55, 0.1)), "Chest": ("chest / torso", (0.95, 0.8, 0.2)),
             "Hip": ("thigh", (0.1, 0.65, 0.65)), "Knee": ("shank", (0.2, 0.7, 0.3)), "Ankle": ("foot", (0.6, 0.8, 0.1)),
             "Shoulder": ("upper arm", (0.45, 0.35, 0.85)), "Elbow": ("forearm", (0.8, 0.3, 0.75)),
             "Hand": ("hand", (0.95, 0.45, 0.55))}
    grey = (0.82, 0.82, 0.82, 1)

    def kind(b):
        return kinds[b.split("_")[-1]]
    pairs = [(i + 1, sb, gb, *kind(sb)) for i, (sb, gb) in enumerate(PAIRS)]
    ms = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml")); ds = mujoco.MjData(ms)
    mg = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml")); dg = mujoco.MjData(mg)
    for m in (ms, mg):
        fl_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for g in range(m.ngeom):
            if g != fl_:
                m.geom_matid[g] = -1; m.geom_rgba[g] = grey
    for n, sb, gb, _, col in pairs:
        for m, b in ((ms, sb), (mg, gb)):
            bid = m.body(b).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid:
                    m.geom_rgba[g] = (*col, 1)
    ds.qpos[:] = ms.qpos0; ds.qpos[:2] = 0; ds.qpos[3:7] = [0.5, 0.5, 0.5, 0.5]
    for j, v in CALIB_POSE.items():
        ds.qpos[ms.joint(j).qposadr[0]] = v
    dg.qpos[:] = 0; dg.qpos[2] = 0.80; dg.qpos[3] = 1; dg.qpos[gm.g_qadr] = gm.q_nom
    out = []
    f, fs = font(20), font(17)
    for m, d, title, which, zc, dist in ((ms, ds, "SMPL adult (Motivo's body)", 1, 0.95, 3.3),
                                         (mg, dg, "Unitree G1", 2, 0.66, 2.35)):
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, W, W); c = cam(180, (0, 0, zc), dist=dist, el=-5)
        r.update_scene(d, camera=c); img = r.render().copy()
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        for n, sb, gb, _, col in pairs:
            b = m.body(sb if which == 1 else gb).id
            (u, v), = project(r, [d.xipos[b]], W, W)
            rad = px(13)
            dr.ellipse([u - rad, v - rad, u + rad, v + rad], fill=(255, 255, 255), outline=tuple(int(255 * x * 0.7) for x in col),
                       width=px(3))
            t = str(n); tw = dr.textlength(t, font=fs)
            dr.text((u - tw / 2, v - fs.size * 0.62), t, fill=(30, 30, 30), font=fs)
        out.append(label(np.array(im), [title, "same colour and number = one pair; grey = no partner"]))
        r.close()
    top = hcat(out)
    # the key: number, colour, SMPL body <-> G1 link
    rows, colw = 7, top.shape[1] // 2
    key = Image.new("RGB", (top.shape[1], px(40) + rows * px(34) + px(58)), (255, 255, 255)); dr = ImageDraw.Draw(key)
    dr.text((px(14), px(8)), "Pairs (SMPL body <-> G1 link)", fill=(20, 20, 20), font=font(20))
    for i, (n, sb, gb, name, col) in enumerate(pairs):
        x = px(14) + (i // rows) * colw; y = px(40) + (i % rows) * px(34)
        dr.rectangle([x, y + px(4), x + px(22), y + px(26)], fill=tuple(int(255 * c_) for c_ in col))
        dr.text((x + px(32), y), f"{n:2d}  {name}:  {sb}  <->  {gb}", fill=(30, 30, 30), font=fs)
    y = px(40) + rows * px(34) + px(4)
    dr.text((px(14), y), "Grey on the SMPL (no G1 partner): Torso, Spine, Neck, Head, Thorax (collarbones), Toe, Hand (fingers).",
            fill=(110, 110, 110), font=font(15))
    dr.text((px(14), y + px(24)), "Grey on the G1: links between paired ones (hip pitch/roll, waist yaw/roll, shoulder pitch/roll, "
            "elbow, wrist pitch, ankle pitch).", fill=(110, 110, 110), font=font(15))
    Image.fromarray(np.concatenate([top, np.array(key)], 0)).save(OUT / "pairs.png")


def make_g1_to_smpl(W=int(S * 1.25), clip="move-ego-0-2_4", t=100):
    sys.path.insert(0, str(REPO))
    from model.g1_map import PAIRS
    qg = np.load(REPO / f"outputs/g1/roundtrip/{clip}/g1.npz")["qpos"][t].copy()
    qs = np.load(REPO / f"outputs/g1/roundtrip/{clip}/smpl_rt.npz")["qpos"][t].copy()
    kinds = {"Pelvis": (0.95, 0.55, 0.1), "Chest": (0.95, 0.8, 0.2), "Hip": (0.1, 0.65, 0.65), "Knee": (0.2, 0.7, 0.3),
             "Ankle": (0.6, 0.8, 0.1), "Shoulder": (0.45, 0.35, 0.85), "Elbow": (0.8, 0.3, 0.75), "Hand": (0.95, 0.45, 0.55)}
    grey = (0.8, 0.8, 0.8, 1)
    ms = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml")); ds = mujoco.MjData(ms)
    mg = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml")); dg = mujoco.MjData(mg)
    for m in (ms, mg):
        fl_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for g in range(m.ngeom):
            if g != fl_:
                m.geom_matid[g] = -1; m.geom_rgba[g] = grey
    for sb, gb in PAIRS:
        col = (*kinds[sb.split("_")[-1]], 1)
        for m, b in ((ms, sb), (mg, gb)):
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == m.body(b).id:
                    m.geom_rgba[g] = col
    out = []
    # both pelvises at the origin, same heading: the camera looks at each body from the same side relative to it
    for m, d, q, title, sub, zc, dist in (
            (mg, dg, qg, "G1 pose (walking frame %d)" % t, "the real robot state", 0.62, 2.35),
            (ms, ds, qs, "Virtual SMPL from it (G1Map.g1_to_smpl_qpos)",
             "coloured = orientation copied from the G1; grey = Motivo's mean posture", 0.9, 3.2)):
        d.qpos[:] = q; d.qpos[:2] = 0; mujoco.mj_forward(m, d)
        f = d.xmat[m.body("pelvis" if m is mg else "Pelvis").id].reshape(3, 3)[:, 0 if m is mg else 2]
        az = np.degrees(np.arctan2(f[1], f[0])) + 180 - 35     # from the front, 35 deg to the body's left
        r = mujoco.Renderer(m, W, W); r.update_scene(d, camera=cam(az, (0, 0, zc), dist=dist, el=-10))
        out.append(label(r.render().copy(), [title, sub])); r.close()
    Image.fromarray(hcat(out)).save(OUT / "g1_to_smpl.png")


def dashed(dr, a, b, col, w, dash=None):
    a, b = np.asarray(a, float), np.asarray(b, float); L = np.linalg.norm(b - a); dash = dash or px(10)
    for s0 in np.arange(0, L, 2 * dash):
        p0 = a + (b - a) * s0 / L; p1 = a + (b - a) * min(s0 + dash, L) / L
        dr.line([tuple(p0), tuple(p1)], fill=col, width=w)


def make_ik_residual(W=S, clip="move-ego-0-2_4"):
    sys.path.insert(0, str(REPO))
    from model.g1_map import G1Map
    from scipy.spatial.transform import Rotation as Rot
    M = G1Map()
    ref = np.load(REPO / f"data/origin_motion/{clip.rsplit('_', 1)[0]}/{clip}.npz")["qpos"]
    g = np.load(REPO / f"outputs/g1/roundtrip/{clip}/g1.npz")["qpos"]
    best = (0, 0, None, None)
    for t in range(len(ref)):                      # the frame where the left foot misses its target most
        M.ds.qpos[:] = ref[t]; mujoco.mj_kinematics(M.ms, M.ds)
        Rt = M.ds.xmat[M.sid["L_Ankle"]].reshape(3, 3) @ M.C["L_Ankle"].T
        M.dg.qpos[:] = g[t]; mujoco.mj_kinematics(M.mg, M.dg)
        Ra = M.dg.xmat[M.mg.body("left_ankle_roll_link").id].reshape(3, 3).copy()
        a = np.degrees(np.linalg.norm(Rot.from_matrix(Rt @ Ra.T).as_rotvec()))
        if a > best[0]:
            best = (a, t, Rt.copy(), Ra)
    ang, t, Rt, Ra = best
    toe = np.degrees(np.arccos(np.clip(Rt[:, 0] @ Ra[:, 0], -1, 1)))
    sole = np.degrees(np.arccos(np.clip(Rt[:, 2] @ Ra[:, 2], -1, 1)))
    m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml")); d = mujoco.MjData(m)
    d.qpos[:] = g[t]; mujoco.mj_forward(m, d)
    foot = d.xpos[m.body("left_ankle_roll_link").id].copy()
    r = mujoco.Renderer(m, W, W); f, fb = font(20), font(26)
    fwd = d.xmat[m.body("pelvis").id].reshape(3, 3)[:, 0]; az = np.degrees(np.arctan2(fwd[1], fwd[0]))
    # panel 1: the whole robot, the left foot circled
    r.update_scene(d, camera=cam(az + 180 - 40, (d.qpos[0], d.qpos[1], 0.55), dist=2.4, el=-15))
    im = Image.fromarray(r.render().copy()); dr = ImageDraw.Draw(im)
    (u, v), = project(r, [foot], W, W); rad = px(38)
    dr.ellipse([u - rad, v - rad, u + rad, v + rad], outline=(240, 160, 0), width=px(5))
    dr.text((u + rad + px(6), v - px(14)), "left foot", fill=(200, 120, 0), font=f, stroke_width=px(3), stroke_fill=(255, 255, 255))
    p1 = label(np.array(im), [f"Walk frame {t}: G1 after IK", "Circled: the link with the largest miss in this frame."])
    # panel 2: straight down on the foot, the target's toes pointing up the picture
    th = np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0]))
    L = 0.32
    fl_, keep = m.geom("floor").id, m.body("left_ankle_roll_link").id
    for gi in range(m.ngeom):                      # only the left foot: the shin would hide it from above
        if gi != fl_ and m.geom_bodyid[gi] != keep:
            m.geom_matid[gi] = -1; m.geom_rgba[gi, 3] = 0.0
    r.update_scene(d, camera=cam(th, foot + 0.12 * Rt[:, 0], dist=0.95, el=-89))
    im = Image.fromarray(r.render().copy()); dr = ImageDraw.Draw(im)
    p0, pa, pt = (np.array(x) for x in project(r, [foot, foot + L * Ra[:, 0], foot + L * Rt[:, 0]], W, W))
    dashed(dr, p0, pt, (255, 255, 255), px(9)); dashed(dr, p0, pt, (120, 120, 120), px(5))
    arrow2d(dr, p0, pa, (210, 25, 25), px(6))
    a0, a1 = (np.arctan2(*(q - p0)[::-1]) for q in (pa, pt))   # image angles of the two arrows
    rho = 0.55 * np.linalg.norm(pt - p0)
    arc = [tuple(p0 + rho * np.array([np.cos(x), np.sin(x)])) for x in np.linspace(a0, a1, 30)]
    dr.line(arc, fill=(240, 160, 0), width=px(4))
    mid = p0 + (rho + px(16)) * np.array([np.cos((a0 + a1) / 2), np.sin((a0 + a1) / 2)])
    side = 1 if (pa - pt)[0] > 0 else -1               # which side of the target the G1 toes fall on, in the picture
    mid = p0 + (rho + px(8)) * np.array([np.cos((a0 + a1) / 2), np.sin((a0 + a1) / 2)]) + np.array([side * px(14), 0])
    tw = dr.textlength(f"{ang:.1f}°", font=fb)
    dr.text((mid[0] if side > 0 else mid[0] - tw, mid[1] - px(16)), f"{ang:.1f}°", fill=(200, 120, 0), font=fb,
            stroke_width=px(3), stroke_fill=(255, 255, 255))
    for q, t_, col, sd in ((pt, "target", (90, 90, 90), -side), (pa, "G1 after IK", (210, 25, 25), side)):
        tw = dr.textlength(t_, font=fb); y_ = q[1] - px(40)
        dr.text((q[0] + px(10) if sd > 0 else q[0] - px(10) - tw, y_), t_, fill=col, font=fb,
                stroke_width=px(3), stroke_fill=(255, 255, 255))
    p2 = label(np.array(im), [f"Left foot from straight above: IK residual {ang:.1f}°",
                              f"Dashed: where the toes should point (from the SMPL foot). Red: where the G1's toes point.",
                              f"The toes miss by {toe:.1f}°; the sole's tilt misses by only {sole:.1f}° (the G1 ankle has no yaw)."])
    r.close()
    Image.fromarray(hcat([p1, p2])).save(OUT / "ik_residual.png")
    return ang, t, toe, sole


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml")); d = mujoco.MjData(m)
    see_through(m)                                    # the hinge axes run inside the body
    r = mujoco.Renderer(m, S, S)
    J = {k: m.joint(f"left_hip_{k}_joint") for k in ("pitch", "roll", "yaw")}
    base = np.zeros(m.nq); base[2] = 0.80; base[3] = 1.0
    views = [("front", cam(180, (0, 0.06, 0.55))), ("left side", cam(-90, (0, 0.06, 0.55))),
             ("oblique", cam(-135, (0, 0.06, 0.5), el=-20))]
    poses = [("Standing", "Standing", "All joints at 0°.", None, 0),
             ("Pitch -45° (swing fore-aft)", "Pitch", "About the left-right axis: the leg kicks forward.", "pitch", -45),
             ("Roll +45° (swing sideways)", "Roll", "About the fore-aft axis: the leg lifts out to the side.", "roll", 45),
             ("Yaw +45° (twist)", "Yaw", "About the vertical axis: the leg twists, toes turn out.", "yaw", 45)]
    rows = []
    for vname, c in views:
        row = []
        for title, _, sub, k, deg in poses:
            d.qpos[:] = base
            if k:
                d.qpos[J[k].qposadr[0]] = np.radians(deg)
            img = render(m, d, r, c, [(J[k].id, COL[k])] if k else [])
            row.append(label(img, [f"{title} ({vname})", sub]))
        rows.append(hcat(row))
    Image.fromarray(np.concatenate(rows, 0)).save(OUT / "pitch_roll_yaw.png")

    frames = []
    for _, name, sub, k, _ in poses[1:]:
        for t in np.linspace(0, 1, 75):
            deg = 45 * np.sin(np.pi * t) * (-1 if k == "pitch" else 1)
            d.qpos[:] = base; d.qpos[J[k].qposadr[0]] = np.radians(deg)
            frames.append(hcat([label(render(m, d, r, c, [(J[k].id, COL[k])]), [f"{name} {deg:+.0f}° ({vname})", sub])
                                for vname, c in views]))
    save_mp4(OUT / "pitch_roll_yaw.mp4", frames)

    # the same two numbers in two hinge orders
    ms = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml")); ds = mujoco.MjData(ms)
    rs = mujoco.Renderer(ms, S, S)
    imgs = []
    for vname, az in (("front", 180), ("left side", -90)):
        d.qpos[:] = base; d.qpos[J["roll"].qposadr[0]] = np.radians(45); d.qpos[J["yaw"].qposadr[0]] = np.radians(90)
        g1 = label(render(m, d, r, cam(az, (0, 0.06, 0.55))),
                   [f"G1 order: roll first, then yaw ({vname})", "roll 45°, yaw 90° -> the leg goes out to the side"])
        ds.qpos[:] = ms.qpos0
        ds.qpos[3:7] = [0.5, 0.5, 0.5, 0.5]       # face +x like the G1 (rest pose faces -y)
        ds.qpos[ms.joint("L_Hip_y").qposadr[0]] = np.radians(90)
        ds.qpos[ms.joint("L_Hip_z").qposadr[0]] = np.radians(45)
        mujoco.mj_forward(ms, ds)
        p = ds.xpos[ms.body("Pelvis").id]
        sm = label(render(ms, ds, rs, cam(az, (p[0], p[1] + 0.06, 0.8), dist=3.4)),
                   [f"SMPL order: yaw first, then roll ({vname})", "yaw 90°, roll 45° -> the leg goes backward"])
        imgs.append(hcat([g1, sm]))
    Image.fromarray(np.concatenate(imgs, 0)).save(OUT / "order.png")

    # the two left-hip frames, both bodies facing +x in the shared nominal pose
    sys.path.insert(0, str(REPO))
    from model.g1_map import G1Map, CALIB_POSE
    gm = G1Map()
    W = int(S * 1.2)
    mg = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml")); dg = mujoco.MjData(mg); see_through(mg)
    ms2 = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml")); ds2 = mujoco.MjData(ms2)
    for g in range(ms2.ngeom):
        if ms2.geom(g).name != "floor":
            ms2.geom_rgba[g, 3] = 0.45
    rg, rs2 = mujoco.Renderer(mg, W, W), mujoco.Renderer(ms2, W, W)
    dg.qpos[:] = 0; dg.qpos[2] = 0.80; dg.qpos[3] = 1; dg.qpos[gm.g_qadr] = gm.q_nom
    ds2.qpos[:] = ms2.qpos0; ds2.qpos[3:7] = [0.5, 0.5, 0.5, 0.5]
    for j, v in CALIB_POSE.items():
        ds2.qpos[ms2.joint(j).qposadr[0]] = v
    mujoco.mj_forward(ms2, ds2); ps = ds2.xpos[ms2.body("L_Hip").id]
    smpl = triad(ms2, ds2, rs2, cam(-135, (ps[0], ps[1], 0.75), dist=2.6, el=-18), ms2.body("L_Hip").id, 0.30,
                 ["x (left)", "y (up)", "z (forward)"], W, W)
    g1 = triad(mg, dg, rg, cam(-135, (0, 0.06, 0.62), dist=2.2, el=-18), mg.body("left_hip_pitch_link").id, 0.25,
               ["x (forward)", "y (left)", "z (up)"], W, W)
    smpl = label(smpl, ["SMPL left hip (L_Hip): local frame", "x points left, y up, z forward"])
    g1 = label(g1, ["G1 left hip (left_hip_pitch_link): local frame", "x points forward, y left, z up"])
    Image.fromarray(hcat([smpl, g1])).save(OUT / "frames.png")
    make_follow()
    make_base_rotation()
    make_pairs()
    make_g1_to_smpl()
    make_ik_residual()
    print(f"-> {OUT}/pitch_roll_yaw.png, pitch_roll_yaw.mp4, order.png, frames.png, follow.mp4, follow.png, smpl_rotation.png, pairs.png, g1_to_smpl.png, ik_residual.png")


if __name__ == "__main__":
    main()
