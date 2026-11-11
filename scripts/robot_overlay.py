#!/usr/bin/env python3
"""robot_overlay.py -- scripts/g1_overlay.py (archived in OOP refactor step 7) for any robot in model/robots.py (--robot): the
extra robots are drawn from the robot's own scene model (mjv_addGeoms adds only the moving geoms, so the floor is not repeated).

---- g1_overlay.py: several G1 trajectories of one clip in ONE scene, so their differences show at a glance.

The first trajectory is drawn as a translucent ghost (by default the G1 target: the clip mapped onto the G1
kinematically, outputs/g1/roundtrip/<stem>/g1.npz), the others as solid robots in their own colours, all at their own
world positions (drift shows as a robot leaving the ghost). The camera follows the ghost's pelvis. Extra robots are
drawn with mjv_addGeoms from assets/robots/g1/g1.xml, whose meshes match scene.xml's one for one.
Default runs for --search <dir>: target / z0 (<dir>/origin_z.npz) / the searched z (<dir>/best.npz), labelled with
bfm, fallen frames and end-point error from <dir>/compare/compare.json when it exists.
Writes <dir>/compare/overlay.mp4 (1920 x 1080) and overlay.mp4.png (three moments).
usage: MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/robot_overlay.py --robot h1 --clip jump-2/jump-2_8 --search outputs/g1/search/jump-2_8_s0
       [--traj "label=path.npz" ...]   (replaces the default runs; the first is the ghost)
"""
import argparse, json, os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
from model.robots import ROBOTS, target_path   # noqa: E402
FONT = Path.home() / ".local/share/fonts/NotoSansCJKtc-Regular.otf"
COLORS = [(0.55, 0.57, 0.62, 0.35), (0.85, 0.22, 0.20, 1.0), (0.18, 0.42, 0.85, 1.0), (0.20, 0.65, 0.30, 1.0),
          (0.95, 0.60, 0.10, 1.0)]
W, H = 1920, 1080


def legend(img, title, items, px=34):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    load = (lambda n: ImageFont.truetype(str(FONT), n)) if FONT.exists() else (lambda n: ImageFont.load_default())
    band = int(px * 1.7) + int(px * 1.25) * len(items)
    d.rectangle([0, 0, im.width, band], fill=(252, 252, 251))
    d.text((px // 2, int(px * 0.85)), title, fill=(11, 11, 11), font=load(px), anchor="lm")
    y = int(px * 1.7)
    for (label, rgba) in items:
        c = tuple(int(255 * x) for x in rgba[:3])
        sq = int(px * 0.8)
        if rgba[3] < 1:
            d.rectangle([px // 2, y, px // 2 + sq, y + sq], outline=(90, 90, 95), width=3, fill=(200, 202, 208))
        else:
            d.rectangle([px // 2, y, px // 2 + sq, y + sq], fill=c)
        d.text((px // 2 + sq + px // 2, y + sq // 2), label, fill=(60, 60, 58), font=load(int(px * 0.8)), anchor="lm")
        y += int(px * 1.25)
    return np.asarray(im)


def split_panels(a, runs, trajs, rob, cfg, cam0, size, head_geoms, floor, T, T_rec, out):
    """--split: one solid robot per 1080 px panel (target grey, then the runs' colours), each camera following its own
    robot with the shared side-on angle; the head ring as in the overlay; a title band per panel."""
    P = 1080
    rob.vis.global_.offwidth, rob.vis.global_.offheight = max(rob.vis.global_.offwidth, P), max(rob.vis.global_.offheight, P)
    r = mujoco.Renderer(rob, height=P, width=P)
    ds = [mujoco.MjData(rob) for _ in trajs]
    root = rob.body(cfg["root"]).id; ft = np.zeros(6)
    fnt = (lambda n: ImageFont.truetype(str(FONT), n)) if FONT.exists() else (lambda n: ImageFont.load_default())
    titles = [n.split(":")[0].replace(" (clip mapped kinematically, no physics)", " (kinematic)") for n, _ in runs]
    subs = [(n.split(":", 1)[1].strip() if ":" in n else "the clip mapped onto the robot, no physics") for n, _ in runs]
    head_min = [(np.inf, 0) for _ in trajs]
    frames = []
    for t in range(T):
        tiles = []
        for k, dk in enumerate(ds):
            dk.qpos[:] = trajs[k][t]; mujoco.mj_forward(rob, dk)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth, cam.elevation, cam.distance = cam0.azimuth, cam0.elevation, cam0.distance * 1.15   # square panel: pull back
            cam.lookat[:] = [dk.xpos[root][0], dk.xpos[root][1], 0.35 * size]
            r.update_scene(dk, camera=cam)
            col = (*COLORS[k][:3], 1.0)
            for i in range(r.scene.ngeom):
                g = r.scene.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and rob.geom_bodyid[g.objid] != 0:
                    g.rgba[:] = col
            img = Image.fromarray(r.render().copy()); dr = ImageDraw.Draw(img)
            if a.mark_head and head_geoms:
                gap, pt = min(((mujoco.mj_geomDistance(rob, dk, g, floor, 1.0, ft), ft[:3].copy()) for g in head_geoms),
                              key=lambda x: x[0])
                head_min[k] = min(head_min[k], (100 * gap, t))
                if k > 0 and 100 * gap < a.mark_head:
                    c = r.scene.camera[0]; fw, up = np.array(c.forward), np.array(c.up); rt = np.cross(fw, up)
                    kk = c.frustum_near / c.frustum_top; v = pt - np.array(c.pos); z = v @ fw
                    u, w = P / 2 + (v @ rt) / z * kk * P / 2, P / 2 - (v @ up) / z * kk * P / 2
                    rc = tuple(int(255 * x) for x in COLORS[k][:3])
                    dr.ellipse([u - 46, w - 46, u + 46, w + 46], outline=rc, width=7)
                    txt = f"head {100 * gap:.1f} cm from the floor"; tw = dr.textlength(txt, font=fnt(34))
                    x = min(max(u - tw / 2, 12), P - tw - 12)          # kept inside the panel
                    dr.text((x, w + 60), txt, fill=rc, font=fnt(34), anchor="lt", stroke_width=4, stroke_fill=(255, 255, 255))
            band = Image.new("RGB", (P, 112), (252, 252, 251)); bd = ImageDraw.Draw(band)
            sq = (*(int(255 * x) for x in COLORS[k][:3]),)
            bd.rectangle([20, 22, 50, 52], fill=sq)
            bd.text((64, 37), titles[k], fill=(11, 11, 11), font=fnt(36), anchor="lm")
            bd.text((20, 86), subs[k], fill=(82, 81, 78), font=fnt(26), anchor="lm")
            tile = Image.new("RGB", (P, P + 112)); tile.paste(band, (0, 0)); tile.paste(img, (0, 112))
            tiles.append(np.asarray(tile))
        row = np.concatenate(tiles, 1)
        when = (f"frame {t / a.slow:.2f} of {T_rec}, {1 / a.slow:g}x speed" if a.slow > 1 else f"step {t + 1}/{T}")
        head = Image.new("RGB", (row.shape[1], 72), (255, 255, 255))
        ImageDraw.Draw(head).text((24, 35), f"{a.clip} on the {cfg['name']}, side view  ({when})", fill=(11, 11, 11),
                                   font=fnt(36), anchor="lm")
        frames.append(np.concatenate([np.asarray(head), row], 0))
    r.close()
    imageio.mimsave(out, frames, fps=30, quality=8, macro_block_size=8)
    ts = sorted({int(T * 0.12), head_min[1][1] if len(head_min) > 1 else T // 2, int(T * 0.85)})
    Image.fromarray(np.concatenate([frames[i] for i in ts], axis=0)).save(str(out) + ".png")
    print("closest head-floor approach per panel (cm, recorded frame):", [(round(g, 1), round(f / a.slow, 2)) for g, f in head_min])
    print(f"-> {out} ({T} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--clip", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--traj", action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--camera", default="track", choices=["track", "side"],
                    help="side: exactly side-on to the ghost's heading at frame 0, 3 deg above the floor")
    ap.add_argument("--until", type=int, default=None, help="only the first N frames")
    ap.add_argument("--split", action="store_true",
                    help="each trajectory in its own 1080 px panel, side by side, solid, the camera following it")
    ap.add_argument("--slow", type=int, default=1,
                    help="slow motion: K poses per recorded frame, the extra ones interpolated (joints and root position "
                         "linearly, root orientation by slerp) for smooth playback at 30 fps = 1/K speed; display only")
    ap.add_argument("--clean", action="store_true", help="no title / legend band, just the scene (e.g. survey videos)")
    ap.add_argument("--mark-head", type=float, default=0.0,
                    help="ring the head (the robot's head ee point's collision geoms) of every solid robot whose head comes "
                         "within this many cm of the floor (0: off)")
    a = ap.parse_args()
    stem = a.clip.split("/")[1]
    sdir = REPO / a.search
    cfg = ROBOTS[a.robot]; RN = cfg["name"]
    if a.traj:
        runs = [(t.split("=", 1)[0], REPO / t.split("=", 1)[1]) for t in a.traj]
    else:
        cmp = sdir / "compare/compare.json"
        c = json.loads(cmp.read_text()) if cmp.exists() else {}
        s = json.loads((sdir / "summary.json").read_text())
        tag = f"{'bfm' if s.get('objective', 'bfm') == 'bfm' else 'target MSE'}, IK penalty {s.get('ik_weight', 0):g}"
        lab = lambda k: (f": bfm {c[k]['bfm']:.3f}, fell {c[k]['motion']['fallen']} frames, end-point off "
                         f"{c[k]['motion']['end_xy']:.2f} m") if k in c else ""
        runs = [(f"{RN} target (clip mapped kinematically, no physics)", target_path(a.robot, stem)),
                (f"{RN} with z0" + lab("z0"), sdir / "origin_z.npz"),
                (f"{RN} with the searched z ({tag})" + lab("new"), sdir / "best.npz")]
    trajs = [np.load(p)["qpos"].astype(np.float64) for _, p in runs]
    T = min(len(q) for q in trajs)
    T = min(T, a.until) if a.until else T
    trajs = [q[:T] for q in trajs]
    if a.slow > 1:
        from scipy.spatial.transform import Rotation, Slerp
        tt = np.arange((T - 1) * a.slow + 1) / a.slow

        def interp(q):
            out = np.empty((len(tt), q.shape[1]))
            for j in range(q.shape[1]):
                out[:, j] = np.interp(tt, np.arange(T), q[:, j])
            rot = Slerp(np.arange(T), Rotation.from_quat(q[:, [4, 5, 6, 3]]))(tt).as_quat()
            out[:, 3:7] = rot[:, [3, 0, 1, 2]]
            return out
        trajs = [interp(q) for q in trajs]
    T_rec, T = T, len(trajs[0])
    out = Path(a.out) if a.out else sdir / "compare/overlay.mp4"; out.parent.mkdir(parents=True, exist_ok=True)

    base = mujoco.MjModel.from_xml_path(str(REPO / cfg["scene"]))
    base.vis.global_.offwidth, base.vis.global_.offheight = max(base.vis.global_.offwidth, W), max(base.vis.global_.offheight, H)
    rob = mujoco.MjModel.from_xml_path(str(REPO / cfg["scene"]))
    assert [base.mesh(i).name for i in range(base.nmesh)] == [rob.mesh(i).name for i in range(rob.nmesh)]
    d0 = mujoco.MjData(base); ds = [mujoco.MjData(rob) for _ in trajs[1:]]
    r = mujoco.Renderer(base, height=H, width=W)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # framed for the G1 (leg ratio s = 0.821 to the SMPL adult); taller robots get the camera pulled back in proportion
    size = json.loads((target_path(a.robot, stem).parent / "summary.json").read_text())["s"] / 0.8213
    cam.distance, cam.elevation, cam.azimuth = 3.3 * size, -12, 140
    root = base.body(cfg["root"]).id
    if a.camera == "side":
        d0.qpos[:] = trajs[0][0]; mujoco.mj_forward(base, d0)
        fwd = d0.xmat[root].reshape(3, 3)[:, 0]
        cam.distance, cam.elevation, cam.azimuth = 2.3 * size, -3, np.degrees(np.arctan2(fwd[1], fwd[0])) + 90
    head_body = rob.body(cfg["ee"][-1][1]).id
    head_geoms = [g for g in range(rob.ngeom) if rob.geom_bodyid[g] == head_body and
                  (rob.geom_contype[g] or rob.geom_conaffinity[g] or rob.geom_group[g] == 3)]
    floor = rob.geom("floor").id; ft = np.zeros(6)
    head_min = [(np.inf, 0) for _ in trajs[1:]]

    def project(p):
        c = r.scene.camera[0]; f, up = np.array(c.forward), np.array(c.up); right = np.cross(f, up)
        k = c.frustum_near / c.frustum_top; v = np.asarray(p) - np.array(c.pos); z = v @ f
        return W / 2 + (v @ right) / z * k * H / 2, H / 2 - (v @ up) / z * k * H / 2
    items = [(n, COLORS[i]) for i, (n, _) in enumerate(runs)]
    if a.split:
        split_panels(a, runs, trajs, rob, cfg, cam, size, head_geoms, floor, T, T_rec, out)
        return
    frames = []
    for t in range(T):
        d0.qpos[:] = trajs[0][t]; mujoco.mj_forward(base, d0)
        cam.lookat[:] = [d0.xpos[root][0], d0.xpos[root][1], (0.35 if a.camera == "side" else 0.7) * size]
        r.update_scene(d0, camera=cam)
        scn = r.scene
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and base.geom_bodyid[g.objid] != 0:
                g.rgba[:] = COLORS[0]
        rings = []
        for k, dk in enumerate(ds, start=1):
            dk.qpos[:] = trajs[k][t]; mujoco.mj_kinematics(rob, dk)
            if a.mark_head and head_geoms:
                mujoco.mj_forward(rob, dk)
                gap, pt = min(((mujoco.mj_geomDistance(rob, dk, g, floor, 1.0, ft), ft[:3].copy()) for g in head_geoms),
                              key=lambda x: x[0])
                head_min[k - 1] = min(head_min[k - 1], (100 * gap, t))
                if 100 * gap < a.mark_head:
                    rings.append((pt, k, 100 * gap))
            n0 = scn.ngeom
            mujoco.mjv_addGeoms(rob, dk, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            for i in range(n0, scn.ngeom):
                scn.geoms[i].rgba[:] = COLORS[k]
        img = r.render().copy()
        if rings:
            im = Image.fromarray(img); dr = ImageDraw.Draw(im)
            fnt = ImageFont.truetype(str(FONT), 34) if FONT.exists() else ImageFont.load_default()
            for pt, k, gap in rings:
                u, v = project(pt); col = tuple(int(255 * x) for x in COLORS[k][:3])
                dr.ellipse([u - 46, v - 46, u + 46, v + 46], outline=col, width=7)
                dr.text((u + 56, v - 20), f"head {gap:.1f} cm from the floor", fill=col, font=fnt, anchor="lm",
                        stroke_width=4, stroke_fill=(255, 255, 255))
            img = np.asarray(im)
        view = "side view" if a.camera == "side" else "all in one scene"
        when = (f"frame {t / a.slow:.2f} of {T_rec}, {1 / a.slow:g}x speed" if a.slow > 1 else f"step {t + 1}/{T}")
        frames.append(img if a.clean else legend(img, f"{a.clip} on the {RN}, {view}  ({when})", items))
    r.close()
    imageio.mimsave(out, frames, fps=30, quality=8, macro_block_size=8)
    ts = [int(T * f) for f in (0.2, 0.5, 0.85)]
    if a.mark_head and np.isfinite(head_min[0][0]):
        ts = sorted({int(T * 0.12), head_min[0][1], int(T * 0.85)})
        print("closest head-floor approach per solid robot (cm, recorded frame):",
              [(round(g, 1), round(f / a.slow, 2)) for g, f in head_min])
    Image.fromarray(np.concatenate([frames[i] for i in ts], axis=1)).save(str(out) + ".png")
    print(f"-> {out} ({T} frames, {W}x{H})")


if __name__ == "__main__":
    main()
