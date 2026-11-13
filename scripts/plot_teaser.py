#!/usr/bin/env python3
"""plot_teaser.py -- the paper's teaser (single-column page, 5.5 in wide): one adult motion, one latent space, a searched latent per physical condition. The search paths on the sphere are schematic.
Rollouts are the survey's recorded ones (outputs/user_survey), nothing is re-simulated. Payload walks move-ego-0-2_4, the others move-ego-0-2_2: same task, different clip.
The panels (conditions, titles, clips, frames, scale) are in configs/figures/teaser.yaml, the colours in crossembodiment/palette.py
(CONDITIONS, INK["paper"]), the style and canvas in crossembodiment/figstyle.py (OOP refactor step 10). Figures are written to
$CE_FIGURES_ROOT (default docs/figures); the renders and cut-outs they draw are read from docs/figures.

  render   bodies as cut-outs on one scale: docs/figures/teaser_parts/<name>_<frame>.png
  draw     docs/figures/teaser_figure.{pdf,svg,png}
  band     docs/figures/teaser_band.{pdf,svg,png}: one row, no latent sphere, from the Blender render
  band_grey  docs/figures/teaser_band_grey.{pdf,svg,png}: the same on grey, no labels

usage: MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl uv run python scripts/plot_teaser.py render | draw | band | band_grey"""
import json, sys
from functools import partial
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_bodies import BG, BODY_RGBA, place, poses                # noqa: E402
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
from crossembodiment import paths                                     # noqa: E402
from crossembodiment.figstyle import FigureStyle, box as _box, txt as _txt   # noqa: E402
from crossembodiment.palette import ACCENTS, CONDITIONS, INK, hex_rgba01   # noqa: E402

REPO = Path(__file__).resolve().parent.parent; FIG = paths.FIGURES; PARTS = FIG / "teaser_parts"; RENDERS = FIG / "blender_renders"; S = paths.read_path("outputs/user_survey")
CFG = yaml.safe_load((paths.FIGURE_CONFIGS / "teaser.yaml").read_text())
FRAMES = tuple(CFG["frames"]); INK_, MUTED, RULE = INK["paper"]["ink"], INK["paper"]["muted"], INK["paper"]["rule"]
AMBER, RED = ACCENTS["payload"]["hex"], ACCENTS["locked_leg"]["hex"]
WALK2, WALK4 = CFG["walk2"], CFG["walk4"]
PANELS = [(p["key"], p["title"], p["body"], CFG[p["clip"]], CONDITIONS[p["key"]]) for p in CFG["panels"]]
U_PER_M = CFG["units_per_m"]
STYLE = FigureStyle.paper()


def to_rgba(h):
    return hex_rgba01(h)


def cutouts(name, xml, qpos, payload=False, red_leg=False):
    import mujoco
    spec = mujoco.MjSpec.from_file(str(xml))
    if payload:                                                    # drawn only: the model already carries the mass (+38 kg on the right wrist and hand)
        hand = spec.compile(); ids = [i for i in range(hand.ngeom) if hand.body(hand.geom_bodyid[i]).name == "R_Hand"]; ctr = hand.geom_pos[ids].mean(0)     # over the hand, covering it
        g = spec.body("R_Hand").add_geom(); g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.pos = ctr.tolist(); g.size = [max(0.14, 1.1 * float(hand.geom_rbound[ids].max())), 0, 0]; g.contype = 0; g.conaffinity = 0; g.mass = 0
    m = spec.compile(); q = np.load(qpos)["qpos"].astype(np.float64); assert q.shape[1] == m.nq, (name, q.shape, m.nq)
    m.geom_rgba[m.geom_bodyid != 0] = BODY_RGBA; painted = []
    if payload:
        m.geom_rgba[m.ngeom - 1] = to_rgba(AMBER); painted.append(m.ngeom - 1)
    if red_leg:
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name in ACCENTS["locked_leg"]["bodies"]:
                m.geom_rgba[g] = to_rgba(RED); painted.append(g)
    return poses(m, q, FRAMES, PARTS, name, painted=painted)


def render():
    PARTS.mkdir(parents=True, exist_ok=True); meta = {}
    meta.update(cutouts("adult", S / "models/adult.xml", S / f"child/trajectories/{WALK2}/reference_adult.npz"))
    for key, _, body, clip, _ in PANELS:
        tr = S / body / "trajectories" / clip; xml = S / json.load(open(tr / "video.json"))["side_by_side"]["panels"][1]["model"]
        meta.update(cutouts(key, xml, tr / "best.npz", payload=key == "payload", red_leg=key == "joint")); print(key, "ok", flush=True)
    (PARTS / "meta.json").write_text(json.dumps(meta, indent=1))


def search_path(seed, bow, m=15):
    """schematic: a jagged walk from 0 to 1 along the net move, as a search takes it. (along, across), across in units of the net move."""
    r = np.random.default_rng(seed); al = np.linspace(0, 1, m + 1); al[1:-1] += r.uniform(-0.45, 0.45, m - 1) / m
    w = np.r_[0, np.cumsum(r.standard_normal(m))]; w -= al * w[-1]; w *= 0.11 / np.abs(w).max()
    return al, w + bow * np.sin(np.pi * al)


def draw():
    meta = json.loads((PARTS / "meta.json").read_text())
    STYLE.apply()
    W, H = 550, 312; fig, ax = FigureStyle.canvas(W, H); FS, SM = 9.5, 8.0
    txt = partial(_txt, ax, size=FS, color=INK_, zorder=8)
    box = partial(_box, ax, ec=INK_)

    def frame(x0, y0, x1, y1, title, ec, name, base):
        """a grey frame, its label as in ReActor, three poses of the walk getting darker, the last at the true colour."""
        bg = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=6", fc=BG, ec=ec, lw=1.0, zorder=2); ax.add_patch(bg); cx, dx = (x0 + x1) / 2, CFG["pose_dx_u"]
        png = RENDERS / f"teaser_{name}.png"
        if png.exists():                       # the Blender render, when there is one (scripts/export_blender_package.py)
            ax.imshow(np.asarray(Image.open(png).convert("RGB")), extent=(x0, x1, y0, y1), zorder=2.2, interpolation="lanczos", aspect="auto").set_clip_path(bg)
        else:
            for i, (t, k) in enumerate(zip(FRAMES, CFG["tints"])):
                place(ax, PARTS, meta, name, t, cx + (i - 1) * dx, base, U_PER_M, tint=k, z=4 + i * 0.1, clip=bg)
        txt(x0 + 8, y1 - 11, title, size=SM + 0.5, ha="left", bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc="white", ec="none", alpha=0.8))

    # ---- source
    frame(28, 176, 170, 306, "Origin", RULE, "adult", 186)

    # ---- latent space: a sphere, a start, one searched latent per condition
    cx, cy, R = 340, 234, 60
    n = 900; yy, xx = np.mgrid[1:-1:n * 1j, -1:1:n * 1j]; rr = np.hypot(xx + 0.28, yy - 0.3); shade = np.clip(0.99 - 0.17 * rr ** 2, 0, 1); al = (np.hypot(xx, yy) <= 1).astype(float)
    ax.imshow(np.dstack([shade, shade, shade * 1.005, al]).clip(0, 1), extent=(cx - R, cx + R, cy - R, cy + R), zorder=1.5, aspect="auto", interpolation="bilinear")
    ax.add_patch(Circle((cx, cy), R, fc="none", ec=INK_, lw=0.9, zorder=2))
    dash = dict(ec=RULE, lw=0.55, ls=(0, (1.5, 2)), zorder=2.1)
    ax.add_patch(Arc((cx, cy), 2 * R, 2 * R * 0.34, theta1=180, theta2=360, **dash)); ax.add_patch(Arc((cx, cy), 2 * R * 0.42, 2 * R, theta1=-90, theta2=90, **dash)); ax.add_patch(Arc((cx, cy + 28), 2 * R * 0.87, 2 * R * 0.28, theta1=180, theta2=360, **dash))
    z0 = np.array([cx - 6, cy + 30.0]); ax.add_patch(Circle(z0, 3.4, fc=INK_, ec="white", lw=0.5, zorder=7)); txt(z0[0] + 2, z0[1] + 10, r"$z_0$", size=FS + 0.5)
    ax.add_patch(FancyArrowPatch((170, 240), (z0[0] - 4, z0[1] - 0.5), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=INK_, shrinkA=0, shrinkB=0, zorder=6))
    ends = {"payload": ((cx - 40, cy - 16), 0.22), "joint": ((cx - 15, cy - 38), 0.16), "child": ((cx + 13, cy - 40), -0.16), "g1": ((cx + 40, cy - 13), -0.24)}; xs = {"payload": 73, "joint": 208, "child": 342, "g1": 477}; TOP = 150; base = 34
    for i, (key, _, _, _, c) in enumerate(PANELS):
        e, side = ends[key]; e = np.array(e, float); d = e - z0; n = np.array([d[1], -d[0]]) * np.sign(side)
        al, ac = search_path(11 + i, 0.13); pts = z0 + al[:, None] * d + ac[:, None] * n
        rgb = np.array(to_rgb(c)); shade = lambda f: tuple(rgb * (0.45 + 0.55 * f) + 0.55 * (1 - f))             # lighter at z0, the true colour at z*
        for j in range(len(pts) - 2):
            f = (j + 1) / (len(pts) - 2); ax.plot(pts[j:j + 2, 0], pts[j:j + 2, 1], color=shade(f), lw=0.95, solid_capstyle="round", zorder=5); ax.scatter(pts[j + 1, 0], pts[j + 1, 1], s=1.8, color=[shade(f)], linewidths=0, zorder=5.5)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=6.5, lw=0.95, color=c, shrinkA=0, shrinkB=2.5, zorder=5))
        ax.add_patch(Circle(e, 3.2, fc=c, ec="white", lw=0.5, zorder=7))
        ax.add_patch(FancyArrowPatch(e, (xs[key], TOP), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=c, shrinkA=3, shrinkB=0, zorder=3.5))
    txt(412, 268, "Latent space", ha="left"); txt(412, 254, r"LOES searches from $z_0$", size=SM, color=MUTED, ha="left"); txt(412, 243, r"to one $z^*$ per condition", size=SM, color=MUTED, ha="left")

    # ---- targets
    box(6, 6, 544, 162, ec=RULE, lw=0.8, z=0.5, r=7); txt(275, 15, "Target (changed physical conditions)")
    for key, title, _, _, c in PANELS:
        frame(xs[key] - 64, 25, xs[key] + 64, TOP, title, c, key, base)
    FigureStyle.save(fig, paths.figures_root() / "teaser_figure", facecolor="white", dpi=600)   # dpi sets the resolution of the bodies in the PDF and SVG too
    print(paths.figures_root() / "teaser_figure.png")


def draw_band():
    """the teaser as one band, as in ReActor's Fig. 1: the Blender render of panel teaser_band with a label over each body. docs/figures/teaser_band.{pdf,svg,png}."""
    p = json.loads(paths.read_path("outputs/blender_package/scene.json").read_text())["panels"]["teaser_band"]
    STYLE.apply()
    W, LAB = 550, 13; IM = W * p["height_m"] / p["width_m"]; fig, ax = FigureStyle.canvas(W, IM + LAB)
    im = np.asarray(Image.open(RENDERS / "teaser_band.png").convert("RGB")).copy(); im[im.min(2) >= 251] = 255; im[[0, -1]] = 255; im[:, [0, -1]] = 255          # the denoiser leaves specks on the white
    ax.imshow(im, extent=(0, W, 0, IM), zorder=1, interpolation="lanczos", aspect="auto")
    for g in p["groups"]:
        ax.text(W / 2 + g["x_m"] / p["width_m"] * W, IM + LAB / 2 - 1, g["label"], fontsize=8.0, color=INK_, ha="center", va="center", zorder=2)
    FigureStyle.save(fig, paths.figures_root() / "teaser_band", facecolor="white", dpi=600)
    print(paths.figures_root() / "teaser_band.pdf")


def draw_band_grey():
    """the band on ReActor's grey floor, no labels, nothing but the render: docs/figures/teaser_band_grey.{pdf,svg,png}."""
    im = np.asarray(Image.open(RENDERS / "teaser_band_grey.png").convert("RGB")); W, H = 5.5, 5.5 * im.shape[0] / im.shape[1]; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(im, interpolation="lanczos", aspect="auto")
    FigureStyle.save(fig, paths.figures_root() / "teaser_band_grey", facecolor="white", dpi=600)
    print(paths.figures_root() / "teaser_band_grey.pdf")


if __name__ == "__main__":
    {"render": render, "draw": draw, "band": draw_band, "band_grey": draw_band_grey}[sys.argv[1]]()
