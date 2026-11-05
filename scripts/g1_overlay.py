#!/usr/bin/env python3
"""g1_overlay.py -- several G1 trajectories of one clip in ONE scene, so their differences show at a glance.

The first trajectory is drawn as a translucent ghost (by default the G1 target: the clip mapped onto the G1
kinematically, outputs/g1/roundtrip/<stem>/g1.npz), the others as solid robots in their own colours, all at their own
world positions (drift shows as a robot leaving the ghost). The camera follows the ghost's pelvis. Extra robots are
drawn with mjv_addGeoms from assets/robots/g1/g1.xml, whose meshes match scene.xml's one for one.
Default runs for --search <dir>: target / z0 (<dir>/origin_z.npz) / the searched z (<dir>/best.npz), labelled with
bfm, fallen frames and end-point error from <dir>/compare/compare.json when it exists.
Writes <dir>/compare/overlay.mp4 (1920 x 1080) and overlay.mp4.png (three moments).
usage: MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/g1_overlay.py --clip jump-2/jump-2_8 --search outputs/g1/search/jump-2_8_s0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--traj", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stem = a.clip.split("/")[1]
    sdir = REPO / a.search
    if a.traj:
        runs = [(t.split("=", 1)[0], REPO / t.split("=", 1)[1]) for t in a.traj]
    else:
        cmp = sdir / "compare/compare.json"
        c = json.loads(cmp.read_text()) if cmp.exists() else {}
        s = json.loads((sdir / "summary.json").read_text())
        tag = f"{'bfm' if s.get('objective', 'bfm') == 'bfm' else 'G1-target MSE'}, IK penalty {s.get('ik_weight', 0):g}"
        lab = lambda k: (f": bfm {c[k]['bfm']:.3f}, fell {c[k]['motion']['fallen']} frames, end-point off "
                         f"{c[k]['motion']['end_xy']:.2f} m") if k in c else ""
        runs = [("G1 target (clip mapped kinematically, no physics)", REPO / f"outputs/g1/roundtrip/{stem}/g1.npz"),
                ("G1 with z0" + lab("z0"), sdir / "origin_z.npz"),
                (f"G1 with the searched z ({tag})" + lab("new"), sdir / "best.npz")]
    trajs = [np.load(p)["qpos"].astype(np.float64) for _, p in runs]
    T = min(len(q) for q in trajs)
    out = Path(a.out) if a.out else sdir / "compare/overlay.mp4"; out.parent.mkdir(parents=True, exist_ok=True)

    base = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/scene.xml"))
    base.vis.global_.offwidth, base.vis.global_.offheight = max(base.vis.global_.offwidth, W), max(base.vis.global_.offheight, H)
    rob = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/g1.xml"))
    assert [base.mesh(i).name for i in range(base.nmesh)] == [rob.mesh(i).name for i in range(rob.nmesh)]
    d0 = mujoco.MjData(base); ds = [mujoco.MjData(rob) for _ in trajs[1:]]
    r = mujoco.Renderer(base, height=H, width=W)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.3, -12, 140
    items = [(n, COLORS[i]) for i, (n, _) in enumerate(runs)]
    frames = []
    for t in range(T):
        d0.qpos[:] = trajs[0][t]; mujoco.mj_forward(base, d0)
        cam.lookat[:] = [d0.xpos[1][0], d0.xpos[1][1], 0.7]
        r.update_scene(d0, camera=cam)
        scn = r.scene
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and base.geom_bodyid[g.objid] != 0:
                g.rgba[:] = COLORS[0]
        for k, dk in enumerate(ds, start=1):
            dk.qpos[:] = trajs[k][t]; mujoco.mj_kinematics(rob, dk)
            n0 = scn.ngeom
            mujoco.mjv_addGeoms(rob, dk, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
            for i in range(n0, scn.ngeom):
                scn.geoms[i].rgba[:] = COLORS[k]
        frames.append(legend(r.render().copy(), f"{a.clip} on the G1, all in one scene  (step {t + 1}/{T})", items))
    r.close()
    imageio.mimsave(out, frames, fps=30, quality=8, macro_block_size=8)
    ts = [int(T * f) for f in (0.2, 0.5, 0.85)]
    Image.fromarray(np.concatenate([frames[i] for i in ts], axis=1)).save(str(out) + ".png")
    print(f"-> {out} ({T} frames, {W}x{H})")


if __name__ == "__main__":
    main()
