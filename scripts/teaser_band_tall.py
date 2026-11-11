#!/usr/bin/env python3
"""teaser_band_tall.py -- the teaser band (export_blender_package.py's teaser_band / teaser_band_grey) with H1 replaced by the tallest
thin scaled body, len1.4_wid0.8 (L x 1.4, W x 0.8, 2.6 m; user, 11/11), placed right of the child and left of the G1. Everything else as
the original band: the same rollouts, frames START + (i * 4 + k) * DT, alphas, floor and camera; nothing of the original is overwritten.
The band's settings and this one's (band_tall) are in configs/figures/teaser.yaml.
  tall body   assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml, LOES rollout outputs/walk_bodies/len1.4_wid0.8_es_s0/best.npz
              (move-ego-0-2_1, 2,048 rollouts, seed 0: the walking-task search; the other bodies walk move-ego-0-2_2). Its poses are spaced
              by its height (0.57 m x 2.6 / 1.81, as walk_band_blender.py), so its four poses do not pile up; the gap between groups stays.
  G1          now the sixth group, so its frames would run past the clip's 300: it goes back one stride (G1_BACK = 18 frames, the
              autocorrelation peak of its foot signal, 0.99), as H1 went back three of its strides in the original.
  panel       the same 50.5 px per metre; wider by the tall group's extra room, taller by the tall body's extra height.
Out: outputs/teaser_band_tall/{scene.json, meshes.npz, render_blender.py, renders/}, docs/figures/blender_renders/teaser_band_tall{,_grey}.png,
docs/figures/teaser_band_tall.{pdf,svg,png} (labels over the bodies) and docs/figures/teaser_band_tall_grey.{pdf,svg,png} (grey floor, no labels).
usage: uv run python scripts/teaser_band_tall.py [export] [render] [draw] [--samples 256] [--threads 32]"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.blender import BlenderScene, body_height as height, load_model, panel   # noqa: E402
from crossenbodiment.camera import BLENDER_AZIMUTH                        # noqa: E402
from crossenbodiment.figstyle import FigureStyle                          # noqa: E402
from crossenbodiment.palette import INK                                   # noqa: E402

ARG = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
SAMPLES, THREADS = ARG("--samples", 256), ARG("--threads", 32)
T = yaml.safe_load((paths.FIGURE_CONFIGS / "teaser.yaml").read_text()); BT = T["band_tall"]; BAND = T["band"]
OUT_REL = "outputs/teaser_band_tall"; OUT = paths.write_path(OUT_REL); RENDERS = paths.FIGURES / "blender_renders"
TALL = BT["body"]; TALL_XML = REPO / BT["xml"]; TALL_Q = paths.read_path(BT["qpos"])
G1_BACK = BT["g1_back"]; ADULT_H = BT["adult_h"]; K = np.sin(np.radians(BLENDER_AZIMUTH))   # 125: AZIMUTH in render_blender.py (a pose's x seen sideways)


def export():
    R = lambda p: paths.read_path(p.format(walk2=T["walk2"], walk4=T["walk4"])); mdl = {k: paths.read_path(v) for k, v in T["models"].items()}
    h_tall = height(load_model(TALL_XML)); sc_tall = h_tall / ADULT_H; G = {g["label"]: g for g in BAND["groups"]}
    band = [(lab, TALL_XML, TALL_Q, "tall", {}, sc_tall, 0) if lab == "Tall body" else
            (lab, mdl[G[lab]["model"]], R(G[lab]["qpos"]), G[lab]["tag"], {k: G[lab][k] for k in ("payload", "red_leg") if k in G[lab]}, 1.0, G1_BACK if lab == "G1" else 0)
            for lab in BT["groups"]]
    n = len(BAND["alpha"]); half = [1.5 * BAND["pose_dx"] * sc * K for *_, sc, _ in band]   # half a group's width in the image
    gap = BAND["dx"] - 2 * 1.5 * BAND["pose_dx"] * K; c = [0.0]                            # the original gap between groups (0.25 m)
    for i in range(1, len(band)):
        c.append(c[-1] + half[i - 1] + gap + half[i])
    shifts = [x - (c[0] + c[-1]) / 2 - BAND["left"] for x in c]                          # as the original (i - 2.5) * BAND_DX - BAND_LEFT
    s = BlenderScene(); poses, groups = [], []
    for i, (label, xml, qpath, tag, kw, sc, back) in enumerate(band):
        dx = BAND["pose_dx"] * sc; frames = [BAND["start"] + (i * n + k) * BAND["dt"] - back for k in range(n)]
        groups.append(dict(label=label, x_m=shifts[i] + (n - 1) / 2 * dx * K))
        for po, a in zip(s.strip(xml, qpath, frames, (0.0,) * n, dx, tag, ground=True, **kw), BAND["alpha"]):
            po["shift"] = shifts[i]; po["alpha"] = a; poses.append(po)
    w_u, h_u, feet_u, U = BAND["units"]; w_u = round(w_u + (c[-1] - c[0] - (len(band) - 1) * BAND["dx"]) * U); h_u = round(h_u + (h_tall - ADULT_H) * U)
    s.add_panel("teaser_band_tall", dict(panel("teaser", w_u, h_u, feet_u, U, poses, white=True), cam_distance=BAND["cam_distance"], groups=groups))
    s.add_panel("teaser_band_tall_grey", dict(s.panels["teaser_band_tall"], white=False))
    s.write(OUT, placeholder=True)
    print(f"tall body {h_tall:.2f} m (x{sc_tall:.2f}); panel {w_u} x {h_u} (was {BAND['units'][0]} x {BAND['units'][1]}); group shifts {np.round(shifts, 2).tolist()}", flush=True)


def render():
    BlenderScene.render(OUT, samples=SAMPLES, threads=THREADS, out="renders"); dst = paths.figures_root() / "blender_renders"; dst.mkdir(parents=True, exist_ok=True)
    for k in ("teaser_band_tall", "teaser_band_tall_grey"):
        shutil.copy(OUT / "renders" / f"{k}.png", dst / f"{k}.png")


def draw():
    """as plot_teaser.py's draw_band (labels over the solid poses) and draw_band_grey (the render alone)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    p = json.loads(paths.own_path(f"{OUT_REL}/scene.json").read_text())["panels"]["teaser_band_tall"]; FIG = paths.figures_root()
    FigureStyle.paper().apply()
    W, LAB = p["px"][0] / 6, 13; IM = W * p["height_m"] / p["width_m"]; fig, ax = FigureStyle.canvas(W, IM + LAB)
    im = np.asarray(Image.open(RENDERS / "teaser_band_tall.png").convert("RGB")).copy(); im[im.min(2) >= 251] = 255; im[[0, -1]] = 255; im[:, [0, -1]] = 255
    ax.imshow(im, extent=(0, W, 0, IM), zorder=1, interpolation="lanczos", aspect="auto")
    for g in p["groups"]:
        ax.text(W / 2 + g["x_m"] / p["width_m"] * W, IM + LAB / 2 - 1, g["label"], fontsize=8.0, color=INK["paper"]["ink"], ha="center", va="center", zorder=2)
    FigureStyle.save(fig, FIG / "teaser_band_tall", facecolor="white", dpi=600)
    plt.close(fig)
    im = np.asarray(Image.open(RENDERS / "teaser_band_tall_grey.png").convert("RGB")); W, H = W / 100, W / 100 * im.shape[0] / im.shape[1]
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(im, interpolation="lanczos", aspect="auto")
    FigureStyle.save(fig, FIG / "teaser_band_tall_grey", facecolor="white", dpi=600)
    print(FIG / "teaser_band_tall.png", FIG / "teaser_band_tall_grey.png")


if __name__ == "__main__":
    steps = [s for s in sys.argv[1:] if s in ("export", "render", "draw")] or ["export", "render", "draw"]
    for s in steps:
        globals()[s]()
