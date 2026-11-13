#!/usr/bin/env python3
"""topdown_blender.py -- top-down Blender renders of one strobe example with every pose at its real place on the floor (user, 11/11:
the strobe panels remove the root's x, y and space the poses evenly, so the paths cannot be compared; first child move-ego-0-4_9).

Bodies: the child's reference (its target, black), z0 (grey), BFM-Zero's CEM (green) and ours, LOES (blue), seed 0, the strobe
figure's trajectories (outputs/strobe_task52_blender/trajectories/<example>: rollout frame t against reference frame t + 1).
Every EVERY-th frame and the last, fading from light (early) to full colour (late); a thin line on the floor = the pelvis path, every frame.
The scene is turned so the reference's start-to-end direction runs left to right; one scale and centre for every panel.
Panels: overlay (all four on one floor) and one per body (its own poses and path, plus the reference's path as a thin black line).
Camera: straight down (elevation -89.5 deg, 80 m away: nearly orthographic). Look and renderer: scripts/render_blender.py (Cycles).
Out: outputs/strobe_task52_blender/topdown/{scene.json, meshes.npz, render_blender.py, renders/} and in outputs/strobe_task52_blender/combined:
<example>_topdown_overlay.png and <example>_topdown_panels.png (2 x 2: reference | z0 / BFM-Zero | ours), both with a label strip.
--poses N (user, 11/11: 6): only the first N poses (frames 0, 20, ...), the paths and framing up to the last of them.
--frames 200,290,18 --match-width 11.22 (user, 11/11): the strobe figure's frames, paths over that window (--path-from 0: from the start;
then the bodies must be ~3x larger and crowd), bodies (and path lines, margins) drawn larger than the floor so they look as big as in the
--poses 6 render.
--ours <run> --layout row --panel-px 1500 --poses 6 --tag 2048 (user, 11/11): ours from a longer search, the four panels in one row as
the side-view strobes -> combined/<example>_topdown_2048.png.
--rotate 90 --tag wide (user, 11/11): turned so the walk runs upwards and the two groups sit side by side; files *_wide.png.
usage: uv run python scripts/topdown_blender.py [export] [render] [combine] [--example child_move-ego-0-4_9] [--poses N] [--samples 256] [--threads 32]"""
import json, math, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import export_blender_package as E                                        # noqa: E402  (pose, load, meshes)
import mujoco                                                             # noqa: E402

ARG = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
TAG = ARG("--tag", "")                                                   # a separate version: its own working folder and file names (_<tag>)
ROT = ARG("--rotate", 0.0)
OURS = ARG("--ours", "")                                                 # a search run folder: ours = its best.npz (cut to the reference) instead of the strobe trajectory
LAYOUT = ARG("--layout", "grid")                                         # grid: overlay + 2 x 2; row: the four panels in one row, as the side-view strobes
PANEL_PX = ARG("--panel-px", 3000)
UNIFORM = "--uniform-color" in sys.argv                                 # the side-view strobes' look: every body and its own path in the body colour, the reference's path black; no legend
NOREF = "--no-ref-line" in sys.argv                                     # no thin reference path in the other panels (user, 11/11); panel copies named _noref
SIDE_KEY = dict(ref="adult", z0="z0", bfmzero="cem", ours="zstar")      # --uniform-color also copies each panel to ../panels/<example>_topdown_<side-view key>.png                                       # image width of a panel (the side-view strobe panels: 1500)                                               # degrees, counter-clockwise, after turning the reference left-to-right
ST = REPO / "outputs/strobe_task52_blender"; OUT = ST / ("topdown" + (f"_{TAG}" if TAG else "")); SUF = f"_{TAG}" if TAG else ""
EXAMPLE = ARG("--example", "child_move-ego-0-4_9"); SAMPLES = ARG("--samples", 256); THREADS = ARG("--threads", 32); POSES = ARG("--poses", 0)
FRAMES = ARG("--frames", "")                                             # "a,b,step": the poses at these frames (the strobe figure's window)
MATCH_W = ARG("--match-width", 0.0)   # > 0: bodies, path lines and margins drawn this much larger than the floor, so a body looks as big as in a
                                       # scene MATCH_W metres wide at the same image width (11.22: the --poses 6 render); pelvis positions unchanged
MODEL = REPO / "outputs/user_survey/models/child.xml"
BODIES = (("reference_body", "ref", "Reference"), ("z0", "z0", "z0"), ("cem", "bfmzero", "BFM-Zero"), ("best", "ours", "Ours"))
RGB = dict(ref=(26, 26, 26), z0=(140, 139, 135), bfmzero=(27, 175, 122), ours=(42, 120, 214))     # sRGB, as the top-view plots
EVERY, AZ, PXPM, MARGIN = 20, 125.0, 90, 1.2                              # pose spacing (frames), the renderer's camera azimuth, pixels per metre


def trail(xy, color, r):
    """thin capsules on the floor along a path"""
    gs = []
    for a, b in zip(xy[:-1], xy[1:]):
        d = b - a; L = float(np.linalg.norm(d))
        if L < 1e-4:
            continue
        dx, dy = d / L; mid = (a + b) / 2
        gs.append(dict(kind="capsule", size=[r, L / 2], p=[float(mid[0]), float(mid[1]), r], R=[-dy, 0.0, dx, dx, 0.0, dy, 0.0, 1.0, 0.0], color=color))
    return gs


def export():
    d = ST / "trajectories" / EXAMPLE; Q = {k: np.load(d / f"{k}.npz")["qpos"].astype(np.float64) for k, *_ in BODIES}
    if OURS:
        Q["best"] = np.load(REPO / OURS / "best.npz")["qpos"].astype(np.float64)[:len(Q["reference_body"])]
    m = E.load(MODEL); dd = mujoco.MjData(m); n = {len(q) for q in Q.values()}; assert len(n) == 1, n; T = n.pop()
    frames = list(range(0, T, EVERY)) + ([T - 1] if (T - 1) % EVERY else [])
    if POSES:                                                             # --poses N: only the first N poses; paths, direction and framing up to the last of them
        frames = frames[:POSES]; Q = {k: q[:frames[-1] + 1] for k, q in Q.items()}
    if FRAMES:                                                            # --frames a,b,step: those poses; the paths from frame 0 to b
        fa, fb, fs = map(int, FRAMES.split(",")); frames = list(range(fa, fb + 1, fs)); assert frames[-1] == fb; Q = {k: q[:fb + 1] for k, q in Q.items()}
    P0 = ARG("--path-from", frames[0] if FRAMES else 0)                   # paths, direction and framing from this frame (with --frames: the window's first)
    tints = [0.6 * (1 - i / (len(frames) - 1)) for i in range(len(frames))]
    r_hat = np.array([math.sin(math.radians(AZ)), -math.cos(math.radians(AZ))]); u_hat = np.array([math.cos(math.radians(AZ)), math.sin(math.radians(AZ))])   # image right, image up
    ref = Q["reference_body"]; v = ref[-1, :2] - ref[P0, :2]; th = math.atan2(r_hat[1], r_hat[0]) - math.atan2(v[1], v[0]) + math.radians(ROT)
    Rz = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])          # reference direction -> the image's left-to-right
    allxy = np.concatenate([q[P0:, :2] for q in Q.values()]) @ Rz.T; a, b = allxy @ r_hat, allxy @ u_hat                # centred in the image's own axes
    ctr = (a.min() + a.max()) / 2 * r_hat + (b.min() + b.max()) / 2 * u_hat
    place = lambda xy: xy @ Rz.T - ctr
    S = float(np.ptp(a)) / (MATCH_W - 2 * MARGIN) if MATCH_W else 1.0        # W = ptp + 2 MARGIN S = MATCH_W S
    W = float(np.ptp(a)) + 2 * MARGIN * S; H = float(np.ptp(b)) + 2 * MARGIN * S
    poses, trails = {}, {}
    for k, col, _ in BODIES:
        q = Q[k]; ps = []
        for t, tint in zip(frames, tints):
            dd.qpos[:] = q[t]; mujoco.mj_forward(m, dd); root = dd.subtree_com[1][:2].copy()
            p = E.pose(m, q, t, 0.0, tint, -th, False, "human"); off = place(root[None])[0]       # E.pose: root-centred and turned by th
            for g in p["geoms"]:                                          # scaled S about the pelvis's floor point (boxes and capsules only)
                assert g["kind"] in ("box", "capsule", "sphere"), g["kind"]
                g["p"] = [S * g["p"][0] + float(off[0]), S * g["p"][1] + float(off[1]), S * g["p"][2]]; g["size"] = [S * x for x in g["size"]]; g["color"] = "body" if UNIFORM else col
            p["x"] = 0.0; ps.append(p)
        poses[k] = ps; trails[k] = place(q[P0:, :2])
    base = dict(figure="topdown", width_m=W, height_m=H, feet_m=H / 2, px=[round(PANEL_PX), round(H * PANEL_PX / W)], white=True, elevation=-89.5, cam_distance=80.0)
    P = {f"{EXAMPLE}_topdown_overlay": dict(base, poses=[x for k, *_ in BODIES for x in poses[k]] + [dict(x=0.0, tint=0.0, geoms=sum([trail(trails[k], c, 0.03 * S) for k, c, _ in BODIES], []))])}
    for k, col, _ in BODIES:
        extra = trail(trails[k], "body" if UNIFORM else col, 0.03 * S) + ([] if k == "reference_body" or NOREF else trail(trails["reference_body"], "black" if UNIFORM else "ref", 0.018 * S))
        P[f"{EXAMPLE}_topdown_{col}"] = dict(base, poses=poses[k] + [dict(x=0.0, tint=0.0, geoms=extra)])
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "scene.json").write_text(json.dumps(dict(panels=P, frames=frames, body_scale=S))); np.savez_compressed(OUT / "meshes.npz", _=np.zeros(1), **E.meshes)
    shutil.copy(REPO / "scripts/render_blender.py", OUT / "render_blender.py")
    print(f"exported {len(P)} panels, {W:.1f} x {H:.1f} m, {len(frames)} poses per body, frames {frames}, bodies x{S:.2f}", flush=True)


def render():
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", str(SAMPLES),
                    "--device", "CPU", "--threads", str(THREADS), "--out", "renders"], cwd=OUT, env=dict(os.environ, OMP_NUM_THREADS="1"), check=True,
                   stdout=open(OUT / "render.log", "w"), stderr=subprocess.STDOUT)


def combine():
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype("DejaVuSans.ttf", 34); FS = ImageFont.truetype("DejaVuSans.ttf", 28); R = OUT / "renders"; ST.joinpath("combined").mkdir(exist_ok=True)

    def strip(w, items, title):
        im = Image.new("RGB", (w, 70), "white"); dr = ImageDraw.Draw(im); x = 20             # legend only (user, 11/11: no title line)
        for lab, col in items:
            dr.rectangle([x, 22, x + 46, 46], fill=RGB[col]); dr.text((x + 58, 16), lab, fill=(0, 0, 0), font=F); x += 90 + int(dr.textlength(lab, font=F))
        return im
    sc = json.loads((OUT / "scene.json").read_text()); fr = sc["frames"]; bs = sc.get("body_scale", 1.0)
    note = f"{EXAMPLE.replace('child_', 'child ')}, top-down: every pose at its real place on the floor, frames {fr[0]}-{fr[-1]} every {EVERY} (light = early); line = pelvis path"
    if LAYOUT == "row":                                                   # as scripts/strobe_final_blender.py combine: one row, 24 px apart, legend strip on top
        ims = [Image.open(R / f"{EXAMPLE}_topdown_{c}.png").convert("RGB") for _, c, _ in BODIES]; gap = 24
        if UNIFORM:
            (ST / "panels").mkdir(exist_ok=True)
            for (_, c, _), im in zip(BODIES, ims):
                im.save(ST / "panels" / f"{EXAMPLE}_topdown{'' if TAG == '16384' else '_' + TAG.replace('16384_', '')}_{SIDE_KEY[c]}.png")   # 16384: plain; 16384_x: _x
        row = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), max(im.height for im in ims)), "white"); x = 0
        for im in ims:
            row.paste(im, (x, 0)); x += im.width + gap
        s = strip(row.width, [(lab, c) for _, c, lab in BODIES], "") if not UNIFORM else Image.new("RGB", (row.width, 0), "white")
        out = Image.new("RGB", (row.width, row.height + s.height), "white"); out.paste(s, (0, 0)); out.paste(row, (0, s.height))
        out.save(ST / "combined" / f"{EXAMPLE}_topdown{SUF}.png"); print("combined", out.size, flush=True); return
    ov = Image.open(R / f"{EXAMPLE}_topdown_overlay.png").convert("RGB"); s = strip(ov.width, [(lab, c) for _, c, lab in BODIES], note)
    out = Image.new("RGB", (ov.width, ov.height + s.height), "white"); out.paste(s, (0, 0)); out.paste(ov, (0, s.height)); out.save(ST / "combined" / f"{EXAMPLE}_topdown_overlay{SUF}.png")
    ims = [(lab, c, Image.open(R / f"{EXAMPLE}_topdown_{c}.png").convert("RGB")) for _, c, lab in BODIES]; w, h = ims[0][2].size; gap = 24; L = 56
    grid = Image.new("RGB", (2 * w + gap, 2 * (h + L) + gap), "white"); dr = ImageDraw.Draw(grid)
    for i, (lab, c, im) in enumerate(ims):
        x, y = (i % 2) * (w + gap), (i // 2) * (h + L + gap); dr.rectangle([x + 16, y + 14, x + 52, y + 40], fill=RGB[c]); dr.text((x + 64, y + 6), lab + ("" if c == "ref" else "   (thin black line: the reference's path)"), fill=(0, 0, 0), font=F); grid.paste(im, (x, y + L))
    s = strip(grid.width, [(lab, c) for _, c, lab in BODIES], note); out = Image.new("RGB", (grid.width, grid.height + s.height), "white"); out.paste(s, (0, 0)); out.paste(grid, (0, s.height))
    out.save(ST / "combined" / f"{EXAMPLE}_topdown_panels{SUF}.png"); print("combined", out.size, flush=True)


if __name__ == "__main__":
    steps = {x for x in sys.argv[1:] if x in ("export", "render", "combine")} or {"export", "render", "combine"}
    for s_ in ("export", "render", "combine"):
        if s_ in steps:
            globals()[s_]()
