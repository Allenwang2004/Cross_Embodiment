#!/usr/bin/env python3
"""teaser_band_tall.py -- the teaser band (export_blender_package.py's teaser_band / teaser_band_grey) with H1 replaced by the tallest
thin scaled body, len1.4_wid0.8 (L x 1.4, W x 0.8, 2.6 m; user, 11/11), placed right of the child and left of the G1. Everything else as
the original band: the same rollouts, frames START + (i * 4 + k) * DT, alphas, floor and camera; nothing of the original is overwritten.
  tall body   assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml, LOES rollout outputs/walk_bodies/len1.4_wid0.8_es_s0/best.npz
              (move-ego-0-2_1, 2,048 rollouts, seed 0: the walking-task search; the other bodies walk move-ego-0-2_2). Its poses are spaced
              by its height (0.57 m x 2.6 / 1.81, as walk_band_blender.py), so its four poses do not pile up; the gap between groups stays.
  G1          now the sixth group, so its frames would run past the clip's 300: it goes back one stride (G1_BACK = 18 frames, the
              autocorrelation peak of its foot signal, 0.99), as H1 went back three of its strides in the original.
  panel       the same 50.5 px per metre; wider by the tall group's extra room, taller by the tall body's extra height.
Out: outputs/teaser_band_tall/{scene.json, meshes.npz, render_blender.py, renders/}, docs/figures/blender_renders/teaser_band_tall{,_grey}.png,
docs/figures/teaser_band_tall.{pdf,svg,png} (labels over the bodies) and docs/figures/teaser_band_tall_grey.{pdf,svg,png} (grey floor, no labels).
usage: uv run python scripts/teaser_band_tall.py [export] [render] [draw] [--samples 256] [--threads 32]"""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import export_blender_package as E                                        # noqa: E402
import mujoco                                                             # noqa: E402

ARG = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
SAMPLES, THREADS = ARG("--samples", 256), ARG("--threads", 32)
OUT = REPO / "outputs/teaser_band_tall"; FIG = REPO / "docs/figures"; RENDERS = FIG / "blender_renders"
TALL = "len1.4_wid0.8"; TALL_XML = REPO / f"assets/robots_torque/{TALL}/robot_torque_full.xml"; TALL_Q = REPO / f"outputs/walk_bodies/{TALL}_es_s0/best.npz"
G1_BACK = 18; ADULT_H = 1.81; K = np.sin(np.radians(125.0))                # 125: AZIMUTH in render_blender.py (a pose's x seen sideways)


def height(m):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    z = [(d.geom_xpos[g][2] + m.geom_rbound[g], d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom) if m.geom_bodyid[g] > 0]
    return max(a for a, _ in z) - min(b for _, b in z)


def export():
    tr = lambda b, c, f: E.S / b / "trajectories" / c / f; W2, W4 = E.WALK2, E.WALK4
    h_tall = height(E.load(TALL_XML)); sc_tall = h_tall / ADULT_H
    band = [("Origin", E.ADULT, tr("child", W2, "reference_adult.npz"), "adult", {}, 1.0, 0),
            ("Payload", E.HEAVY, tr("heavy_arm", W4, "best.npz"), "heavy_arm", dict(payload=True), 1.0, 0),
            ("Restricted joint", E.LIMIT, tr("limit_leg", W2, "best.npz"), "limit_leg", dict(red_leg=True), 1.0, 0),
            ("Reshaped body", E.CHILD, tr("child", W2, "best.npz"), "child", {}, 1.0, 0),
            ("Tall body", TALL_XML, TALL_Q, "tall", {}, sc_tall, 0),
            ("G1", E.G1, tr("g1", W2, "best.npz"), "g1", {}, 1.0, G1_BACK)]
    n = len(E.BAND_ALPHA); half = [1.5 * E.BAND_POSE_DX * sc * K for *_, sc, _ in band]   # half a group's width in the image
    gap = E.BAND_DX - 2 * 1.5 * E.BAND_POSE_DX * K; c = [0.0]                            # the original gap between groups (0.25 m)
    for i in range(1, len(band)):
        c.append(c[-1] + half[i - 1] + gap + half[i])
    shifts = [x - (c[0] + c[-1]) / 2 - E.BAND_LEFT for x in c]                          # as the original (i - 2.5) * BAND_DX - BAND_LEFT
    poses, groups = [], []
    for i, (label, xml, qpath, tag, kw, sc, back) in enumerate(band):
        dx = E.BAND_POSE_DX * sc; frames = [E.BAND_START + (i * n + k) * E.BAND_DT - back for k in range(n)]
        groups.append(dict(label=label, x_m=shifts[i] + (n - 1) / 2 * dx * K))
        for po, a in zip(E.strip(xml, qpath, frames, (0.0,) * n, dx, tag, ground=True, **kw), E.BAND_ALPHA):
            po["shift"] = shifts[i]; po["alpha"] = a; poses.append(po)
    w_u, h_u, feet_u, U = E.BAND_U; w_u = round(w_u + (c[-1] - c[0] - (len(band) - 1) * E.BAND_DX) * U); h_u = round(h_u + (h_tall - ADULT_H) * U)
    P = {"teaser_band_tall": dict(E.panel("teaser", w_u, h_u, feet_u, U, poses, white=True), cam_distance=60.0, groups=groups)}
    P["teaser_band_tall_grey"] = dict(P["teaser_band_tall"], white=False)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "scene.json").write_text(json.dumps(dict(panels=P)))
    np.savez_compressed(OUT / "meshes.npz", _=np.zeros(1), **E.meshes); shutil.copy(REPO / "scripts/render_blender.py", OUT / "render_blender.py")
    print(f"tall body {h_tall:.2f} m (x{sc_tall:.2f}); panel {w_u} x {h_u} (was {E.BAND_U[0]} x {E.BAND_U[1]}); group shifts {np.round(shifts, 2).tolist()}", flush=True)


def render():
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", str(SAMPLES),
                    "--device", "CPU", "--threads", str(THREADS), "--out", "renders"], cwd=OUT, env=dict(os.environ, OMP_NUM_THREADS="1"), check=True,
                   stdout=open(OUT / "render.log", "w"), stderr=subprocess.STDOUT)
    for k in ("teaser_band_tall", "teaser_band_tall_grey"):
        shutil.copy(OUT / "renders" / f"{k}.png", RENDERS / f"{k}.png")


def draw():
    """as plot_teaser.py's draw_band (labels over the solid poses) and draw_band_grey (the render alone)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    p = json.loads((OUT / "scene.json").read_text())["panels"]["teaser_band_tall"]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"})
    W, LAB = p["px"][0] / 6, 13; IM = W * p["height_m"] / p["width_m"]; fig = plt.figure(figsize=(W / 100, (IM + LAB) / 100)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, IM + LAB); ax.axis("off")
    im = np.asarray(Image.open(RENDERS / "teaser_band_tall.png").convert("RGB")).copy(); im[im.min(2) >= 251] = 255; im[[0, -1]] = 255; im[:, [0, -1]] = 255
    ax.imshow(im, extent=(0, W, 0, IM), zorder=1, interpolation="lanczos", aspect="auto")
    for g in p["groups"]:
        ax.text(W / 2 + g["x_m"] / p["width_m"] * W, IM + LAB / 2 - 1, g["label"], fontsize=8.0, color="#22252a", ha="center", va="center", zorder=2)
    for f in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"teaser_band_tall.{f}", facecolor="white", dpi=600)
    plt.close(fig)
    im = np.asarray(Image.open(RENDERS / "teaser_band_tall_grey.png").convert("RGB")); W, H = W / 100, W / 100 * im.shape[0] / im.shape[1]
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(im, interpolation="lanczos", aspect="auto")
    for f in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"teaser_band_tall_grey.{f}", facecolor="white", dpi=600)
    print(FIG / "teaser_band_tall.png", FIG / "teaser_band_tall_grey.png")


if __name__ == "__main__":
    steps = [s for s in sys.argv[1:] if s in ("export", "render", "draw")] or ["export", "render", "draw"]
    for s in steps:
        globals()[s]()
