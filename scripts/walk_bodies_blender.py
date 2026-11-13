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
import json, math, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import export_blender_package as E                                        # noqa: E402
import mujoco                                                             # noqa: E402

ARG = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
SAMPLES = ARG("--samples", 128); THREADS = ARG("--threads", 32); STEP = ARG("--step", 30)
CLIP = "move-ego-0-2/move-ego-0-2_1"; W_RUN = REPO / "outputs/walk_bodies"; OUT = W_RUN / "blender"
BODIES = ["len0.6_wid0.8", "len0.6_wid1.2", "child", "len0.8_wid0.8", "len0.8_wid1.2", "len1.2_wid0.8", "len1.2_wid1.2", "len1.4_wid0.8", "len1.4_wid1.2"]
AZ, MARGIN, IMG_W = 125.0, 0.8, 3000                                      # the renderer's camera azimuth; metres around; image width (px)


def height(m):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    z = [(d.geom_xpos[g][2] + m.geom_rbound[g], d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom) if m.geom_bodyid[g] > 0]
    return max(a for a, _ in z) - min(b for _, b in z)


def export():
    rows = []
    for b in BODIES:
        m = E.load(REPO / f"assets/robots_torque/{b}/robot_torque_full.xml")
        q = np.load(W_RUN / f"{b}_es_s0/best.npz")["qpos"].astype(np.float64); ref = np.load(REPO / f"data/{b}/retargeting_motion/{CLIP}.npz")["qpos"]
        v = ref[-1, :2] - ref[0, :2]; a = math.atan2(v[1], v[0]); dhat = np.array([math.cos(a), math.sin(a)])
        frames = list(range(0, len(q), STEP)); dist = [(q[t, :2] - q[0, :2]) @ dhat for t in frames]
        rows.append(dict(body=b, m=m, q=q, frames=frames, dist=dist, h=height(m), heading=a + math.radians(90.0 - AZ)))
    rows.sort(key=lambda r: r["h"])
    lo = min(min(r["dist"]) for r in rows) - MARGIN; hi = max(max(r["dist"]) for r in rows) + MARGIN; W = hi - lo; pxpm = IMG_W / W
    P = {}
    for r in rows:
        n = len(r["frames"]); poses = []
        for i, (t, x) in enumerate(zip(r["frames"], r["dist"])):
            p = E.pose(r["m"], r["q"], t, 0.0, 0.62 * (1 - i / (n - 1)), r["heading"], False, "human"); p["shift"] = float(x - (lo + hi) / 2); poses.append(p)
        H = 1.25 * r["h"] + 0.3
        P[r["body"]] = dict(figure="walk", width_m=W, height_m=H, feet_m=0.18, px=[IMG_W, round(H * pxpm)], white=True, cam_distance=60.0, poses=poses)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "scene.json").write_text(json.dumps(dict(panels=P, order=[r["body"] for r in rows],
        info={r["body"]: dict(height_m=r["h"], mass_kg=float(r["m"].body_subtreemass[1]), walked_m=float(r["dist"][-1] - r["dist"][0]), frames=r["frames"]) for r in rows})))
    np.savez_compressed(OUT / "meshes.npz", _=np.zeros(1), **E.meshes); shutil.copy(REPO / "scripts/render_blender.py", OUT / "render_blender.py")
    for r in rows:
        print(f"{r['body']:14s} height {r['h']:.2f} m  walked {r['dist'][-1] - r['dist'][0]:5.1f} m along the reference direction  ({len(r['frames'])} poses)")
    print(f"scene {W:.1f} m wide, {pxpm:.0f} px/m", flush=True)


def render():
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", str(SAMPLES),
                    "--device", "CPU", "--threads", str(THREADS), "--out", "renders"], cwd=OUT, env=dict(os.environ, OMP_NUM_THREADS="1"), check=True,
                   stdout=open(OUT / "render.log", "w"), stderr=subprocess.STDOUT)


def combine():
    from PIL import Image, ImageDraw, ImageFont
    sc = json.loads((OUT / "scene.json").read_text()); order = sc["order"]; F = ImageFont.truetype("DejaVuSans.ttf", 30); FS = ImageFont.truetype("DejaVuSans.ttf", 24)
    ims = [Image.open(OUT / "renders" / f"{b}.png").convert("RGB") for b in order]; LW = 330
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
