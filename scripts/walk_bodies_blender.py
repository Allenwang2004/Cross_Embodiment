#!/usr/bin/env python3
"""walk_bodies_blender.py -- the walking task (move-ego-0-2_1) on every body, one Blender strobe row per body (user, 11/11): the smallest
body at the bottom left, taller bodies stacked upwards, every row starting at the same left edge and drawn at one common scale, so a
larger body's longer walk simply reaches further right.

Rollouts: LOES's best latent after 2,048 rollouts, seed 0 (outputs/walk_bodies/<body>_es_s0/best.npz; scripts in run_queue.sh).
Poses: every STEP frames from frame 0, each at its real distance along the walk: the pelvis's displacement projected on the body's
reference start-to-end direction (sideways drift is depth, not drawn); the body turned so that direction runs left to right.
Bodies: the 8 isometrically scaled bodies (scripts/make_scaled_bodies.py) and the child, ordered by standing height.
Look: scripts/render_blender.py (Cycles), one body colour fading from light (early) to full (late), white floor, camera 8 deg above,
60 m away (nearly orthographic, so a 30 m row is not distorted).
Out: outputs/walk_bodies/blender/{scene.json, meshes.npz, render_blender.py, renders/} and outputs/walk_bodies/walk_bodies_rows.png
(rows bottom-up, a label column on the left).
usage: uv run python scripts/walk_bodies_blender.py [export] [render] [combine] [--samples 128] [--threads 32] [--step 30]"""
import json, math, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.blender import BlenderScene, body_height as height, load_model   # noqa: E402
from crossenbodiment.camera import BLENDER_AZIMUTH                        # noqa: E402
from crossenbodiment.sheet import font                                    # noqa: E402
from crossenbodiment.strobe import fade_tints                             # noqa: E402

ARG = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
SAMPLES = ARG("--samples", 128); THREADS = ARG("--threads", 32); STEP = ARG("--step", 30)
CLIP = "move-ego-0-2/move-ego-0-2_1"; W_REL = "outputs/walk_bodies"; W_RUN = paths.write_path(W_REL); OUT = W_RUN / "blender"
BODIES = ["len0.6_wid0.8", "len0.6_wid1.2", "child", "len0.8_wid0.8", "len0.8_wid1.2", "len1.2_wid0.8", "len1.2_wid1.2", "len1.4_wid0.8", "len1.4_wid1.2"]
AZ, MARGIN, IMG_W = BLENDER_AZIMUTH, 0.8, 3000                            # the renderer's camera azimuth; metres around; image width (px)


def export():
    rows = []; sc = BlenderScene()
    for b in BODIES:
        m = load_model(REPO / f"assets/robots_torque/{b}/robot_torque_full.xml")
        q = np.load(paths.read_path(f"{W_REL}/{b}_es_s0/best.npz"))["qpos"].astype(np.float64); ref = np.load(REPO / f"data/{b}/retargeting_motion/{CLIP}.npz")["qpos"]
        v = ref[-1, :2] - ref[0, :2]; a = math.atan2(v[1], v[0]); dhat = np.array([math.cos(a), math.sin(a)])
        frames = list(range(0, len(q), STEP)); dist = [(q[t, :2] - q[0, :2]) @ dhat for t in frames]
        rows.append(dict(body=b, m=m, q=q, frames=frames, dist=dist, h=height(m), heading=a + math.radians(90.0 - AZ)))
    rows.sort(key=lambda r: r["h"])
    lo = min(min(r["dist"]) for r in rows) - MARGIN; hi = max(max(r["dist"]) for r in rows) + MARGIN; W = hi - lo; pxpm = IMG_W / W
    for r in rows:
        poses = []
        for (t, x), k in zip(zip(r["frames"], r["dist"]), fade_tints(len(r["frames"]))):
            p = sc.pose(r["m"], r["q"], t, 0.0, k, r["heading"], False, "human"); p["shift"] = float(x - (lo + hi) / 2); poses.append(p)
        H = 1.25 * r["h"] + 0.3
        sc.add_panel(r["body"], dict(figure="walk", width_m=W, height_m=H, feet_m=0.18, px=[IMG_W, round(H * pxpm)], white=True, cam_distance=60.0, poses=poses))
    sc.extra.update(order=[r["body"] for r in rows],
                    info={r["body"]: dict(height_m=r["h"], mass_kg=float(r["m"].body_subtreemass[1]), walked_m=float(r["dist"][-1] - r["dist"][0]), frames=r["frames"]) for r in rows})
    sc.write(OUT, placeholder=True)
    for r in rows:
        print(f"{r['body']:14s} height {r['h']:.2f} m  walked {r['dist'][-1] - r['dist'][0]:5.1f} m along the reference direction  ({len(r['frames'])} poses)")
    print(f"scene {W:.1f} m wide, {pxpm:.0f} px/m", flush=True)


def render():
    BlenderScene.render(OUT, samples=SAMPLES, threads=THREADS, out="renders")


def combine():
    from PIL import Image, ImageDraw
    sc = json.loads(paths.own_path(f"{W_REL}/blender/scene.json").read_text()); order = sc["order"]; F = font(30); FS = font(24); R = paths.own_path(f"{W_REL}/blender/renders")
    ims = [Image.open(R / f"{b}.png").convert("RGB") for b in order]; LW = 330
    out = Image.new("RGB", (LW + IMG_W, sum(im.height for im in ims)), "white"); dr = ImageDraw.Draw(out); y = out.height
    for b, im in zip(order, ims):                                         # smallest at the bottom
        y -= im.height; out.paste(im, (LW, y)); i = sc["info"][b]
        lab = "child" if b == "child" else b.replace("len", "L ").replace("_wid", ", W ")
        dr.text((20, y + im.height // 2 - 34), lab, fill=(0, 0, 0), font=F); dr.text((20, y + im.height // 2 + 4), f"{i['height_m']:.2f} m, {i['mass_kg']:.0f} kg", fill=(90, 90, 90), font=FS)
    out.save(W_RUN / "walk_bodies_rows.png"); print("combined", out.size, flush=True)


if __name__ == "__main__":
    steps = {x for x in sys.argv[1:] if x in ("export", "render", "combine")} or {"export", "render", "combine"}
    for s_ in ("export", "render", "combine"):
        if s_ in steps:
            globals()[s_]()
