#!/usr/bin/env python3
"""plot_method_figure.py -- the paper's method figure (single-column page, 5.5 in wide): the frozen policy and backward map never change,
only the latent z does. Replaces scripts/plot_method_diagram.py, whose figure still shows the correction basis.

  render   bodies as cut-outs on the same scale, from recorded rollouts (nothing is re-simulated): docs/figures/method_parts/<name>_<frame>.png
  draw     docs/figures/method_figure.{pdf,svg,png}
  first    the first layout with Blender bodies: docs/figures/method_figure_first.{pdf,svg,png}, and method_figure_first_no_loes without the word LOES

Change the rollouts in configs/figures/method.yaml (cutouts: the MuJoCo cut-outs; blender: the Blender panels) when the final searches
exist. Colours: crossembodiment/palette.py INK["paper"]; style, canvas and the txt / arrow helpers: crossembodiment/figstyle.py (OOP
refactor step 10). Figures are written to $CE_FIGURES_ROOT (default docs/figures); renders and cut-outs are read from docs/figures.
usage: MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl uv run python scripts/plot_method_figure.py render | draw | simple | first"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from functools import partial
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_bodies import BG, BODY_RGBA, place, poses                # noqa: E402
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
from crossembodiment import paths                                     # noqa: E402
from crossembodiment.figstyle import FigureStyle, arrow as _arrow, txt as _txt   # noqa: E402
from crossembodiment.palette import INK as _INK                       # noqa: E402

REPO = Path(__file__).resolve().parent.parent; RENDERS = paths.FIGURES / "blender_renders"; PARTS = paths.FIGURES / "method_parts"
CFG = yaml.safe_load((paths.FIGURE_CONFIGS / "method.yaml").read_text())
RUN = CFG["run"]; STAND = CFG["stand"]; RESULT = tuple(CFG["result"])
INK, MUTED, RULE, FROZEN, ACCENT, ACCENT_BG = (_INK["paper"][k] for k in ("ink", "muted", "rule", "frozen", "accent", "accent_bg"))
SOURCES = {k: dict(xml=v["xml"], qpos=v["qpos"].format(run=RUN)) for k, v in CFG["cutouts"].items()}
U_PER_M = CFG["units_per_m"]                    # figure units (0.01 in) per metre of body
STYLE = FigureStyle.paper()


def save(fig, name):
    """every format at 600 dpi (dpi sets the resolution of the bodies in the PDF and SVG too) into $CE_FIGURES_ROOT"""
    FigureStyle.save(fig, paths.figures_root() / name, facecolor="white", dpi=600)
    return paths.figures_root() / name


def render():
    import mujoco
    PARTS.mkdir(parents=True, exist_ok=True); meta = {}
    for name, s in SOURCES.items():
        m = mujoco.MjModel.from_xml_path(str(REPO / s["xml"])); q = np.load(paths.read_path(s["qpos"]))["qpos"].astype(np.float64); assert q.shape[1] == m.nq, (name, q.shape, m.nq)
        m.geom_rgba[m.geom_bodyid != 0] = BODY_RGBA; meta.update(poses(m, q, sorted({*RESULT, STAND}), PARTS, name)); print(name, "ok", flush=True)
    (PARTS / "meta.json").write_text(json.dumps(meta, indent=1))


def draw():
    meta = json.loads((PARTS / "meta.json").read_text())
    STYLE.apply()
    W, Y0, Y1 = 550, 0, 258; fig, ax = FigureStyle.canvas(W, Y1, Y0)
    FS, SM = 9.0, 8.0
    txt = partial(_txt, ax, size=FS, color=INK, zorder=6)

    def arrow(pts, color=INK, lw=0.9, ls="-", head=True):
        _arrow(ax, pts, color, lw, ls=ls, style="-|>" if head else "-")

    def frame(x0, y0, x1, y1, title, poses_, right=False, label=True):
        """a grey frame, its label as in ReActor; poses_: (name, frame, x, feet[, tint])."""
        bg = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=5", fc=BG, ec=RULE, lw=0.8, zorder=2); ax.add_patch(bg)
        png = RENDERS / ("method_" + {"Reshaped body": "fall_child", "G1": "fall_g1", "Origin reference": "adult", "Result: Reshaped body": "child", "Result: G1": "g1"}[title] + ".png")
        if png.exists():                       # the Blender render, when there is one (scripts/export_blender_package.py)
            ax.imshow(np.asarray(Image.open(png).convert("RGB")), extent=(x0, x1, y0, y1), zorder=2.2, interpolation="lanczos", aspect="auto").set_clip_path(bg)
        else:
            for i, (name, t, x, y, *k) in enumerate(poses_):
                place(ax, PARTS, meta, name, t, x, y, U_PER_M, tint=k[0] if k else 0.0, z=4 + i * 0.1, clip=bg)
        if label:
            txt(x1 - 6 if right else x0 + 6, y1 - 9.5, title, size=SM, ha="right" if right else "left", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.5", fc="white", ec="none", alpha=0.8))

    def trail(name, x0, dx, y0):
        """a motion's poses overlapping, getting darker, the last at the true colour."""
        return [(name, t, x0 + i * dx, y0, 0.62 * (1 - i / (len(RESULT) - 1))) for i, t in enumerate(RESULT)]

    def network(x0, x1, y0, y1, label, wide_right):
        n = (y1 - y0) * 0.17; pts = ([(x0, y0 + n), (x1, y0), (x1, y1), (x0, y1 - n)] if wide_right else [(x0, y0), (x1, y0 + n), (x1, y1 - n), (x0, y1)])
        ax.add_patch(Polygon(pts, closed=True, fc=FROZEN, ec=INK, lw=0.9, joinstyle="round", zorder=4))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; txt(cx, cy + 5, label, size=FS + 0.5); txt(cx, cy - 8, "frozen", size=SM, color=MUTED)

    # ---- LOES proposes z
    py = 188; ax.add_patch(FancyBboxPatch((14, py - 24), 70, 48, boxstyle="round,pad=0,rounding_size=4", fc=ACCENT_BG, ec=ACCENT, lw=1.1, zorder=4)); txt(49, py, "LOES", size=FS + 1, color=ACCENT, fontweight="bold")
    ax.add_patch(Circle((106, py), 11, fc=ACCENT, ec="none", zorder=5)); txt(106, py + 0.5, r"$z$", size=11, color="white")
    arrow([(84, py), (95, py)], color=ACCENT, lw=1.5); arrow([(117, py), (140, py)], color=ACCENT, lw=1.5)

    # ---- every z is rolled out on the new body by the frozen policy; with z0 the new bodies fall
    ax.add_patch(FancyBboxPatch((128, 114), 226, 142, boxstyle="round,pad=0,rounding_size=5", fc="none", ec=RULE, lw=0.8, ls=(0, (3, 2)), zorder=1))
    network(140, 200, py - 24, py + 24, "Policy", wide_right=False)
    ax.add_patch(FancyBboxPatch((226, 118), 118, 133, boxstyle="round,pad=0,rounding_size=5", fc="white", ec=INK, lw=0.9, zorder=1.5))
    frame(230, 186, 340, 249, "Reshaped body", [], label=False); frame(230, 120, 340, 183, "G1", [], label=False)
    arrow([(200, py), (226, py)]); txt(213, py + 6, r"$a_t$")
    ax.add_patch(Rectangle((132, 122), 76, 30, fc="white", ec=INK, lw=0.9, zorder=4)); txt(170, 139.5, r"$\Phi$", size=11); txt(170, 128.5, "scale to origin", size=SM - 0.5, color=MUTED)
    arrow([(226, 137), (208, 137)]); txt(217, 144, r"$s_t$")
    arrow([(170, 152), (170, py - 22)]); txt(177, 158, r"$o_t$")

    # ---- the rollouts are scored against the adult
    arrow([(354, py - 16), (402, py - 16)]); txt(378, py - 9, r"$o_{1:T}$")
    ax.add_patch(Rectangle((402, py - 48), 108, 64, fc="white", ec=INK, lw=0.9, zorder=4))
    txt(456, py + 6, "Cost", size=FS + 0.5, fontweight="bold"); txt(456, py - 7, "BFM similarity", size=SM); txt(456, py - 18.5, r"+ heading + position", size=SM); txt(456, py - 30, r"+ anchor", size=SM)
    frame(392, 2, 522, 102, "Origin reference", trail("adult", 412, 18, 10), label=False); arrow([(470, 102), (470, py - 48)])

    # ---- which updates z, and the search goes round again
    arrow([(426, py - 48), (426, 108), (62, 108), (62, py - 24)], color=ACCENT, lw=1.5); txt(333, 108, r"update $z$, repeat", size=SM + 0.5, color=ACCENT, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    # ---- result
    arrow([(30, py - 24), (30, 100)], color=ACCENT, lw=1.5); txt(36, 118, r"$z^*$", color=ACCENT, ha="left")
    frame(10, 2, 138, 100, "Result: Reshaped body", trail("child", 29, 18, 10), label=False); frame(146, 2, 274, 100, "Result: G1", trail("g1", 165, 18, 10), label=False)
    print(save(fig, "method_figure").with_suffix(".pdf"))


def draw_simple():
    """the short version, docs/figures/method_figure_simple.*: z goes in, the rollout is compared with the origin reference, z is updated from the comparison, the result comes out. Black but for z, its sampling and its update. Uses the Blender renders."""
    STYLE.apply()
    W, H = 562, 176; fig, ax = FigureStyle.canvas(W, H); FS, SM = 9.0, 8.0; py = 96
    txt = partial(_txt, ax, size=FS, color=INK, zorder=8)

    def arrow(pts, color=INK, lw=0.9, style="-|>"):
        _arrow(ax, pts, color, lw, style=style, head_z=6)

    def frame(x0, y0, x1, y1, title, panel, right=False):
        bg = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=5", fc=BG, ec=INK, lw=0.8, zorder=2); ax.add_patch(bg)
        im = np.asarray(Image.open(RENDERS / f"method_{panel}.png").convert("RGB")); im = im[max(im.shape[0] - round(im.shape[1] * (y1 - y0) / (x1 - x0)), 0):]        # a shorter frame cuts the empty top, not the scale
        ax.imshow(im, extent=(x0, x1, y0, y1), zorder=2.2, interpolation="lanczos", aspect="auto").set_clip_path(bg)
        if title:
            txt(x1 - 5 if right else x0 + 5, y1 - 9, title, size=SM, ha="right" if right else "left", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.5", fc="white", ec="none", alpha=0.8))

    def group(x0, y0, x1, y1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=5", fc="white", ec=INK, lw=0.9, zorder=1.5))

    # z goes in
    ax.add_patch(Circle((16, py), 11, fc=ACCENT, ec="none", zorder=5)); txt(16, py + 0.5, r"$z$", size=11, color="white")
    arrow([(27, py), (74, py)], color=ACCENT); txt(44, py + 6.5, "sample", size=SM, color=ACCENT)

    # the box: the policy on the new body, compared with the origin reference
    ax.add_patch(FancyBboxPatch((62, 20), 367, 152, boxstyle="round,pad=0,rounding_size=5", fc="none", ec=RULE, lw=0.8, ls=(0, (3, 2)), zorder=1))
    n = 8; ax.add_patch(Polygon([(74, py - 24), (130, py - 24 + n), (130, py + 24 - n), (74, py + 24)], closed=True, fc=FROZEN, ec=INK, lw=0.9, joinstyle="round", zorder=4)); txt(102, py + 5, "Policy", size=FS + 0.5); txt(102, py - 8, "frozen", size=SM, color=INK)
    group(152, 29, 270, 163); frame(156, 97, 266, 160, "", "fall_child"); frame(156, 32, 266, 95, "", "fall_g1")
    arrow([(130, py), (152, py)]); txt(141, py + 6, r"$a_t$")
    ax.add_patch(Rectangle((68, 32), 68, 30, fc="white", ec=INK, lw=0.9, zorder=4)); txt(102, 49.5, r"$\Phi$", size=11); txt(102, 38.5, "scale to origin", size=SM - 0.5, color=INK)
    arrow([(152, 47), (136, 47)]); txt(144, 54, r"$s_t$")
    arrow([(102, 62), (102, py - 22)]); txt(109, 67, r"$o_t$")
    frame(306, py - 45, 423, py + 45, "", "adult"); txt(364.5, py + 51, "Origin")
    arrow([(270, py), (306, py)], style="<|-|>"); txt(288, py + 6.5, "Cost", size=SM + 0.5)

    # what the comparison gives updates z
    ax.add_patch(Circle((288, py), 1.8, fc=ACCENT, ec="none", zorder=7)); arrow([(288, py), (288, 10), (16, 10), (16, py - 11)], color=ACCENT); txt(150, 10, "Update", size=SM + 0.5, color=ACCENT, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    # the result comes out
    arrow([(429, py), (438, py)])
    group(438, 29, 561, 163); frame(442, 97, 557, 160, "", "child"); frame(442, 32, 557, 95, "", "g1"); txt(499.5, 169.5, "Result")
    print(save(fig, "method_figure_simple").with_suffix(".pdf"))


def draw_first(loes=True):
    """the figure's first layout (commit dea2957, whose bodies were MuJoCo cut-outs), its bodies now from Blender (panels method_first_*, transparent background). loes=False leaves "LOES" off the update box."""
    scene = json.loads(paths.read_path("outputs/blender_package/scene.json").read_text())["panels"]; FRAMES = tuple(CFG["first"])
    STYLE.apply()
    W, Y0, Y1 = 550, 6, 243; fig, ax = FigureStyle.canvas(W, Y1, Y0)
    FS, SM = 9.0, 8.0
    txt = partial(_txt, ax, size=FS, color=INK, zorder=6)

    def arrow(pts, color=INK, lw=0.9, ls="-", head=True):
        _arrow(ax, pts, color, lw, ls=ls, style="-|>" if head else "-")

    def body(name, t, x, y0, z=4, clip=None):
        """the body centred on x, its lowest point at y0, at the figure's one scale. Only the solid body counts for this, not its shadow."""
        p = f"method_first_{name}_{t}"; im = np.asarray(Image.open(RENDERS / f"{p}.png")); H, W_ = im.shape[:2]; u = scene[p]["width_m"] * U_PER_M / W_; ys, xs = np.nonzero(im[..., 3] > 240); l = x - u * (xs.min() + xs.max() + 1) / 2; b = y0 - u * (H - ys.max() - 1)
        h = ax.imshow(im, extent=(l, l + u * W_, b, b + u * H), zorder=z, interpolation="lanczos", aspect="auto")
        if clip is not None:
            h.set_clip_path(clip)

    def strip(name, x0, dx, y0):
        for i, t in enumerate(FRAMES):
            body(name, t, x0 + i * dx, y0, z=4 + i * 0.01)
        ax.plot([x0 - dx * 0.45, x0 + dx * (len(FRAMES) - 0.55)], [y0 - 1, y0 - 1], color=RULE, lw=0.5, zorder=2)

    def lock(x, y, c=MUTED):
        ax.add_patch(Rectangle((x - 2.6, y - 2.6), 5.2, 3.8, fc=c, ec="none", zorder=6))
        ax.add_patch(Arc((x, y + 1.2), 3.4, 4.6, theta1=0, theta2=180, ec=c, lw=0.7, zorder=6))

    def network(x0, x1, y0, y1, label, wide_right):
        n = (y1 - y0) * 0.17; pts = ([(x0, y0 + n), (x1, y0), (x1, y1), (x0, y1 - n)] if wide_right else [(x0, y0), (x1, y0 + n), (x1, y1 - n), (x0, y1)])
        ax.add_patch(Polygon(pts, closed=True, fc=FROZEN, ec=INK, lw=0.9, joinstyle="round", zorder=4))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; txt(cx, cy + 5, label, size=12); lock(cx - 15, cy - 11.5); txt(cx - 10, cy - 12, "frozen", size=SM, color=MUTED, ha="left")

    # ---- rollout on the new body: z -> policy -> body -> map -> policy
    ax.add_patch(FancyBboxPatch((4, 118), 296, 123, boxstyle="round,pad=0,rounding_size=5", fc="none", ec=RULE, lw=0.8, ls=(0, (3, 2)), zorder=1))
    txt(11, 232, r"Rollout on the new body, $t = 1 \ldots T$", size=SM + 0.5, color=MUTED, ha="left")
    box = Rectangle((12, 127), 92, 95, fc="white", ec=INK, lw=0.9, zorder=2); ax.add_patch(box); txt(58, 212, "New body", size=FS)
    body("child", "stand", 37, 134, clip=box); body("g1", "stand", 79, 134, clip=box); ax.plot([18, 98], [133, 133], color=RULE, lw=0.5, zorder=3)   # standing still (user, 11/11)
    py = 196; network(190, 250, py - 28, py + 28, r"$\pi$", wide_right=True)
    arrow([(190, py), (104, py)]); txt(147, py + 6, r"$a_t$")
    ax.add_patch(Rectangle((119, 127), 68, 30, fc="white", ec=INK, lw=0.9, zorder=4)); txt(153, 144.5, r"$\Phi$", size=11); txt(153, 133.5, "scale to origin", size=SM - 0.5, color=MUTED)
    arrow([(104, 142), (119, 142)]); txt(111, 149, r"$s_t$")
    ax.plot([187, 200], [142, 142], color=INK, lw=0.9, zorder=3)          # one scaled observation, shared by the policy and the backward map
    ax.add_patch(Circle((200, 142), 1.8, fc=INK, ec="none", zorder=5))
    arrow([(200, 142), (225, 142), (225, py - 28)]); txt(232, 152, r"$o_t$")

    # ---- the only thing that changes
    zx = 369
    ax.add_patch(Circle((zx, py), 13, fc=ACCENT, ec="none", zorder=5)); txt(zx, py + 0.5, r"$z$", size=12, color="white")
    arrow([(zx - 13, py), (250, py)], color=ACCENT, lw=1.5)
    arrow([(zx, 240), (zx, py + 13)], color=ACCENT, lw=0.9, ls=(0, (2.5, 2))); txt(zx + 6, 230, r"$z_0$", color=ACCENT, ha="left")
    ax.add_patch(FancyBboxPatch((346, 45), 46, 40, boxstyle="round,pad=0,rounding_size=4", fc=ACCENT_BG, ec=ACCENT, lw=1.1, zorder=4))
    if loes:
        txt(zx, 71, "LOES", size=FS + 1, color=ACCENT, fontweight="bold"); txt(zx, 57, r"update $z$", size=SM, color=ACCENT)
    else:
        txt(zx, 65, r"update $z$", size=FS - 0.5, color=ACCENT)
    arrow([(zx, 85), (zx, py - 13)], color=ACCENT, lw=1.5); txt(zx + 6, 135, r"new $z$", color=ACCENT, ha="left")

    # ---- what the cost compares
    txt(70, 104, "Origin reference", size=FS); strip("adult", 28, 42, 22)
    network(174, 226, 39, 91, r"$B$", wide_right=False)
    arrow([(136, 65), (174, 65)]); arrow([(200, 142), (200, 87.5)]); txt(205, 106, r"$o_{1:T}$", ha="left")
    ax.add_patch(Rectangle((238, 33), 100, 64, fc="white", ec=INK, lw=0.9, zorder=4))
    txt(288, 77, "Cost", size=FS + 0.5, fontweight="bold"); txt(288, 63.5, "BFM similarity", size=SM); txt(288, 52, r"+ heading + position", size=SM)   # "+ anchor" removed (user, 11/11)
    arrow([(226, 65), (238, 65)]); arrow([(338, 65), (346, 65)])

    # ---- result
    ax.add_patch(FancyBboxPatch((418, 18), 128, 223, boxstyle="round,pad=0,rounding_size=5", fc="none", ec=RULE, lw=0.8, zorder=1))
    arrow([(392, 65), (418, 65)], color=ACCENT, lw=1.5); txt(405, 72, r"$z^*$", color=ACCENT)
    txt(482, 232, "Result on each body", size=FS)
    strip("child", 442, 40, 144); txt(426, 212, "Reshaped body", size=SM, color=MUTED, ha="left")
    strip("g1", 442, 40, 28); txt(426, 104, "G1", size=SM, color=MUTED, ha="left")
    name = "method_figure_first" + ("" if loes else "_no_loes")
    print(save(fig, name).with_suffix(".pdf")); plt.close(fig)


def first():
    draw_first(True); draw_first(False)


if __name__ == "__main__":
    {"render": render, "draw": draw, "simple": draw_simple, "first": first}[sys.argv[1]]()
