#!/usr/bin/env python3
"""robot_pairs.py -- the SMPL body <-> robot link pairs of model/robots.py as a picture (g1_axes_demo.make_pairs for
any robot, 1080 px panels; scripts/g1_axes_demo.py archived in OOP refactor step 7).

Left the SMPL adult in the calibration pose (arms down, turned to face +x), right the robot in its nominal pose
(RobotMap.q_nom); every paired segment in its pair's colour with the pair's number on it, everything else grey; below,
the key and what has no partner on either side: outputs/<robot>/explain/pairs.png.
--to-smpl also draws outputs/<robot>/explain/<robot>_to_smpl.png (g1_axes_demo.make_g1_to_smpl for any robot): one
walking frame of the clip mapped onto the robot (scripts/robot_roundtrip.py) and the SMPL pose
RobotMap.robot_to_smpl_qpos makes from it, paired segments coloured, the SMPL joints left at Motivo's mean posture grey
(--fill ridge: predicted from the paired ones). The frame is mapped with the current map, not read from the roundtrip files.
usage: MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/robot_pairs.py --robot h1 [--robot apollo ...] [--to-smpl]
"""
import argparse, sys
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment import draw as gad                 # noqa: E402  g1_axes_demo's px / font / cam / project / label / hcat (refactor step 4)
from model.robot_map import RobotMap, CALIB_POSE, SMPL_XML   # noqa: E402
from model.robots import ROBOTS, out_dir, target_path   # noqa: E402

W = 1080
gad.K = W / 1.25 / 420                                  # make_pairs' proportions (panel = 1.25 S) at 1080 px (crossembodiment.draw's px scale)
KINDS = {"Pelvis": ("pelvis", (0.95, 0.55, 0.1)), "Chest": ("chest / torso", (0.95, 0.8, 0.2)),
         "Hip": ("thigh", (0.1, 0.65, 0.65)), "Knee": ("shank", (0.2, 0.7, 0.3)), "Ankle": ("foot", (0.6, 0.8, 0.1)),
         "Shoulder": ("upper arm", (0.45, 0.35, 0.85)), "Elbow": ("forearm", (0.8, 0.3, 0.75)),
         "Hand": ("hand", (0.95, 0.45, 0.55)), "Head": ("head", (0.3, 0.6, 0.95))}
GREY = (0.82, 0.82, 0.82, 1)


def figure(name):
    cfg = ROBOTS[name]; rm = RobotMap(name); px, font = gad.px, gad.font
    pairs = [(i + 1, sb, gb, *KINDS[sb.split("_")[-1]]) for i, (sb, gb) in enumerate(rm.pairs)]
    ms = mujoco.MjModel.from_xml_path(str(REPO / SMPL_XML)); ds = mujoco.MjData(ms)
    mg = mujoco.MjModel.from_xml_path(str(REPO / cfg["scene"])); dg = mujoco.MjData(mg)
    for m in (ms, mg):
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, W)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0:
                m.geom_matid[g] = -1; m.geom_rgba[g] = GREY
    for _, sb, gb, _, col in pairs:
        for m, b in ((ms, sb), (mg, gb)):
            bid = m.body(b).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid:
                    m.geom_rgba[g] = (*col, 1)
    ds.qpos[:] = ms.qpos0; ds.qpos[:2] = 0; ds.qpos[3:7] = [0.5, 0.5, 0.5, 0.5]
    for j, v in CALIB_POSE.items():
        ds.qpos[ms.joint(j).qposadr[0]] = v
    f = rm.free_q
    dg.qpos[:] = 0; dg.qpos[f + 3] = 1; dg.qpos[rm.robot_qadr] = rm.q_nom; mujoco.mj_kinematics(mg, dg)
    rmap_low = rm.low_robot; dg2 = rm.d_robot; dg2.qpos[:] = dg.qpos; mujoco.mj_kinematics(rm.m_robot, dg2)
    dg.qpos[f + 2] = -rmap_low(dg2)
    out = []
    for m, d, title, which in ((ms, ds, "SMPL adult (Motivo's body)", 1), (mg, dg, cfg["name"], 2)):
        mujoco.mj_forward(m, d)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
        top = max(d.geom_xpos[g, 2] + m.geom_rbound[g] * 0.5 for g in geoms)
        r = mujoco.Renderer(m, W, W); c = gad.cam(180, (0, 0, top / 2), dist=1.95 * top, el=-5)
        r.update_scene(d, camera=c); img = r.render().copy()
        im = Image.fromarray(img); dr = ImageDraw.Draw(im); fs = font(17)
        for n, sb, gb, _, col in pairs:
            b = m.body(sb if which == 1 else gb).id
            (u, v), = gad.project(r, [d.xipos[b]], W, W)
            rad = px(13)
            dr.ellipse([u - rad, v - rad, u + rad, v + rad], fill=(255, 255, 255),
                       outline=tuple(int(255 * x * 0.7) for x in col), width=px(3))
            t = str(n); tw = dr.textlength(t, font=fs)
            dr.text((u - tw / 2, v - fs.size * 0.62), t, fill=(30, 30, 30), font=fs)
        out.append(gad.label(np.array(im), [title, "same colour and number = one pair; grey = no partner"]))
        r.close()
    top = gad.hcat(out)
    s_un = sorted({ms.body(i).name for i in range(1, ms.nbody)} - {sb for _, sb, _, _, _ in pairs},
                  key=lambda n: [ms.body(i).name for i in range(ms.nbody)].index(n))
    g_paired = {gb for _, _, gb, _, _ in pairs}
    g_un = [mg.body(i).name for i in range(1, mg.nbody)
            if mg.body(i).name not in g_paired and any(mg.geom_bodyid[g] == i for g in range(mg.ngeom))]
    rows, colw = (len(pairs) + 1) // 2, top.shape[1] // 2
    fs = font(17); fn = font(15)

    def wrap(text, width):
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1))); lines, cur = [], ""
        for w in text.split(" "):
            nxt = (cur + " " + w).strip()
            if cur and probe.textlength(nxt, font=fn) > width:
                lines.append(cur); cur = w
            else:
                cur = nxt
        return lines + [cur]
    notes = (wrap("Grey on the SMPL (no robot partner): " + ", ".join(s_un) + ".", top.shape[1] - px(28))
             + wrap(f"Grey on the {cfg['name']}: " + (", ".join(g_un) if g_un else "none") + ".", top.shape[1] - px(28)))
    key = Image.new("RGB", (top.shape[1], px(40) + rows * px(34) + px(10) + len(notes) * px(22) + px(10)), (255, 255, 255))
    dr = ImageDraw.Draw(key)
    dr.text((px(14), px(8)), f"Pairs (SMPL body <-> {cfg['name']} link)", fill=(20, 20, 20), font=font(20))
    for i, (n, sb, gb, kind, col) in enumerate(pairs):
        x = px(14) + (i // rows) * colw; y = px(40) + (i % rows) * px(34)
        dr.rectangle([x, y + px(4), x + px(22), y + px(26)], fill=tuple(int(255 * c_) for c_ in col))
        dr.text((x + px(32), y), f"{n:2d}  {kind}:  {sb}  <->  {gb}", fill=(30, 30, 30), font=fs)
    y = px(40) + rows * px(34) + px(10)
    for line in notes:
        dr.text((px(14), y), line, fill=(110, 110, 110), font=fn); y += px(22)
    path = out_dir(name) / "explain" / "pairs.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([top, np.array(key)], 0)).save(path)
    print(f"-> {path}")


def figure_to_smpl(name, clip="move-ego-0-2_4", t=100, fill="fixed"):
    cfg = ROBOTS[name]; rm = RobotMap(name, fill=fill)
    # the clip mapped onto the robot with the CURRENT map (warm-started IK up to frame t), then back to SMPL
    ref = np.load(REPO / f"data/origin_motion/{clip.rsplit('_', 1)[0]}/{clip}.npz")["qpos"]; q = None
    for k in range(t + 1):
        qg, _ = rm.smpl_to_robot_qpos(ref[k], q); q = qg[rm.robot_qadr]
    qs = rm.robot_to_smpl_qpos(qg)
    ms = mujoco.MjModel.from_xml_path(str(REPO / SMPL_XML)); ds = mujoco.MjData(ms)
    mg = mujoco.MjModel.from_xml_path(str(REPO / cfg["scene"])); dg = mujoco.MjData(mg)
    for m in (ms, mg):
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, W)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0:
                m.geom_matid[g] = -1; m.geom_rgba[g] = GREY
    for sb, gb in rm.pairs:
        col = (*KINDS[sb.split("_")[-1]][1], 1)
        for m, b in ((ms, sb), (mg, gb)):
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == m.body(b).id:
                    m.geom_rgba[g] = col
    out = []
    f = rm.free_q
    root_g, root_s = mg.body(cfg["root"]).id, ms.body("Pelvis").id
    # both pelvises at the origin, same heading: each body seen from the same side relative to itself
    for m, d, q, root, fwd, title, sub in (
            (mg, dg, qg, root_g, 0, f"{cfg['name']} pose (walking frame {t})", "the real robot state"),
            (ms, ds, qs, root_s, 2, "Virtual SMPL from it (RobotMap.robot_to_smpl_qpos)",
             f"coloured = orientation copied from the {cfg['name']}; grey = "
             + ("predicted from the coloured parts (linear regression)" if fill == "ridge" else "Motivo's mean posture"))):
        d.qpos[:] = q
        if m is mg:
            d.qpos[f:f + 2] = 0
        else:
            d.qpos[:2] = 0
        mujoco.mj_forward(m, d)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
        top = max(d.geom_xpos[g, 2] + m.geom_rbound[g] * 0.5 for g in geoms)
        fv = d.xmat[root].reshape(3, 3)[:, fwd]
        az = np.degrees(np.arctan2(fv[1], fv[0])) + 180 - 35      # from the front, 35 deg to the body's left
        r = mujoco.Renderer(m, W, W); r.update_scene(d, camera=gad.cam(az, (0, 0, 0.5 * top), dist=1.85 * top, el=-10))
        out.append(gad.label(r.render().copy(), [title, sub])); r.close()
    path = out_dir(name) / "explain" / f"{name}_to_smpl.png"; path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gad.hcat(out)).save(path)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", action="append", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--to-smpl", action="store_true")
    ap.add_argument("--fill", default="fixed", choices=["fixed", "ridge"], help="the robot -> SMPL fill for --to-smpl")
    a = ap.parse_args()
    for name in a.robot:
        figure(name)
        if a.to_smpl:
            figure_to_smpl(name, fill=a.fill)


if __name__ == "__main__":
    main()
