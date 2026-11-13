#!/usr/bin/env python3
"""walk_band_blender.py -- the walking task on every body as one band in the style of the teaser (user, 11/11: "風格類似 teaser", not
horizontal but slanted): all bodies on one white floor, each a group of four overlapping poses (see-through ghosts 0.2, 0.42, 0.7 and a solid
pose, as the teaser's BAND_ALPHA), the gait carrying on from one body to the next (frames START + (i * 4 + k) * DT), the smallest body first.
The band runs into the scene at PHI degrees from the image plane and the camera looks down EL degrees, so it climbs from the lower left
to the upper right; the camera is far away (nearly orthographic), so the far bodies are not shrunk. Spacing scales with each body's
standing height (the teaser's 0.57 m within a group and 1.65 m between groups, for the adult's 1.81 m), so a taller body takes more room.
Rollouts: outputs/walk_bodies/<body>_es_s0/best.npz (LOES, 2,048 rollouts, seed 0); poses at the nearest frames with a foot down, set on
the floor (export_blender_package.contact_frames), each turned to walk along the band.
Out: outputs/walk_bodies/band/{scene.json, meshes.npz, render_blender.py, renders/} and outputs/walk_bodies/walk_band.png
(+ walk_band_labeled.png with the body names under the solid poses).
usage: uv run python scripts/walk_band_blender.py [export] [render] [combine] [--el 15] [--phi 35] [--samples 128] [--threads 32]"""
import json, math, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import export_blender_package as E                                        # noqa: E402
import mujoco                                                             # noqa: E402

ARG = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
EL, PHI = ARG("--el", 15.0), ARG("--phi", 35.0); SAMPLES = ARG("--samples", 128); THREADS = ARG("--threads", 32)
W_RUN = REPO / "outputs/walk_bodies"; OUT = W_RUN / "band"; AZ = 125.0; IMG_W = 4200
BODIES = ["len0.6_wid0.8", "len0.6_wid1.2", "child", "len0.8_wid0.8", "len0.8_wid1.2", "len1.2_wid0.8", "len1.2_wid1.2", "len1.4_wid0.8", "len1.4_wid1.2"]
ALPHA = (0.2, 0.42, 0.7, 1.0); START, DT = 4, 8; POSE_DX, GROUP_DX, ADULT_H = 0.57, 1.65, 1.81


def height(m):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    z = [(d.geom_xpos[g][2] + m.geom_rbound[g], d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom) if m.geom_bodyid[g] > 0]
    return max(a for a, _ in z) - min(b for _, b in z)


def label(b):
    return "Child" if b == "child" else b.replace("len", "L ").replace("_wid", ", W ")


def export():
    az, el = math.radians(AZ), math.radians(-EL)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]); right = np.array([math.sin(az), -math.cos(az), 0.0])
    up = np.array([0, 0, 1.0]) - fwd[2] * fwd; up /= np.linalg.norm(up); hz = np.array([math.cos(az), math.sin(az), 0.0])
    u = math.cos(math.radians(PHI)) * right + math.sin(math.radians(PHI)) * hz; psi = math.atan2(u[1], u[0])          # the band's direction on the floor
    bodies = []
    for b in BODIES:
        m = E.load(REPO / f"assets/robots_torque/{b}/robot_torque_full.xml"); bodies.append((b, m, np.load(W_RUN / f"{b}_es_s0/best.npz")["qpos"].astype(np.float64), height(m)))
    bodies.sort(key=lambda x: x[3]); poses, groups, s = [], [], 0.0
    for i, (b, m, q, h) in enumerate(bodies):
        sc = h / ADULT_H; want = [START + (i * len(ALPHA) + k) * DT for k in range(len(ALPHA))]
        frames, lo = E.contact_frames(m, q, want); a = E.heading(q, frames)
        if i:
            s += GROUP_DX * (sc + bodies[i - 1][3] / ADULT_H) / 2 - POSE_DX * (len(ALPHA) - 1) * bodies[i - 1][3] / ADULT_H
        for k, (t, al) in enumerate(zip(frames, ALPHA)):
            p = E.pose(m, q, t, 0.0, 0.0, a - psi, False, "human"); pos = (s + k * POSE_DX * sc) * u
            for g in p["geoms"]:
                g["p"] = [g["p"][0] + float(pos[0]), g["p"][1] + float(pos[1]), g["p"][2] - float(lo[t])]
            p["alpha"] = al; p["x"] = 0.0; poses.append(p)
        groups.append(dict(body=b, label=label(b), height_m=h, mass_kg=float(m.body_subtreemass[1]), frames=frames, solid=(s + (len(ALPHA) - 1) * POSE_DX * sc) * u))
        s += (len(ALPHA) - 1) * POSE_DX * sc
    G = np.array([g_["p"] for p in poses for g_ in p["geoms"]]); A, Bv = G @ right, G @ up
    ctr = -(A.min() + A.max()) / 2 * right - (Bv.min() + Bv.max()) / 2 / (hz @ up) * hz          # centred in the image's own axes (look at the origin)
    for p in poses:
        for g_ in p["geoms"]:
            g_["p"] = [g_["p"][0] + float(ctr[0]), g_["p"][1] + float(ctr[1]), g_["p"][2]]
    M = 0.45; W = float(np.ptp(A)) + 2 * M; H = float(np.ptp(Bv)) + 2 * M; pxpm = IMG_W / W
    for gr in groups:                                                     # label anchor: under the solid pose's feet, in image pixels
        f = gr["solid"] + ctr; gr["px"] = [float((f @ right + W / 2) * pxpm), float((H / 2 - f @ up) * pxpm)]; del gr["solid"]
    P = {"walk_band": dict(figure="walk", width_m=W, height_m=H, feet_m=H / 2, px=[IMG_W, round(H * pxpm)], white=True, elevation=-EL, cam_distance=150.0, poses=poses)}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "scene.json").write_text(json.dumps(dict(panels=P, groups=groups)))
    np.savez_compressed(OUT / "meshes.npz", _=np.zeros(1), **E.meshes); shutil.copy(REPO / "scripts/render_blender.py", OUT / "render_blender.py")
    print(f"band: {len(groups)} bodies, {W:.1f} x {H:.1f} m, el {EL}, phi {PHI}, slope in the image {math.degrees(math.atan(math.tan(math.radians(PHI)) * (hz @ up))):.1f} deg", flush=True)


def render(scale=1.0, out="renders", samples=None):
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", str(samples or SAMPLES),
                    "--scale", str(scale), "--device", "CPU", "--threads", str(THREADS), "--out", out], cwd=OUT, env=dict(os.environ, OMP_NUM_THREADS="1"), check=True,
                   stdout=open(OUT / "render.log", "w"), stderr=subprocess.STDOUT)


def combine():
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(OUT / "renders/walk_band.png").convert("RGB"); im.save(W_RUN / "walk_band.png")
    sc = json.loads((OUT / "scene.json").read_text()); F = ImageFont.truetype("DejaVuSans.ttf", 34); lab = im.copy(); dr = ImageDraw.Draw(lab)
    for g in sc["groups"]:
        x, y = g["px"]; t = g["label"]; w = dr.textlength(t, font=F); dr.text((x - w / 2, y + 26), t, fill=(40, 40, 40), font=F)
    lab.save(W_RUN / "walk_band_labeled.png"); print("saved", im.size, flush=True)


if __name__ == "__main__":
    if "preview" in sys.argv[1:]:
        export(); render(scale=0.35, out="preview", samples=24); sys.exit()
    steps = {x for x in sys.argv[1:] if x in ("export", "render", "combine")} or {"export", "render", "combine"}
    for s_ in ("export", "render", "combine"):
        if s_ in steps:
            globals()[s_]()
