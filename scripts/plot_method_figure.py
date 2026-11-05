#!/usr/bin/env python3
"""plot_method_figure.py -- the paper's method figure (single-column page, 5.5 in wide): the frozen policy and backward map never change,
only the latent z does. Replaces scripts/plot_method_diagram.py, whose figure still shows the correction basis.

  render   bodies as cut-outs on the same scale, from recorded rollouts (nothing is re-simulated): docs/figures/method_parts/<name>_<frame>.png
  draw     docs/figures/method_figure.{pdf,svg,png}

Change the rollouts in SOURCES when the final searches exist.
usage: MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl uv run python scripts/plot_method_figure.py render | draw"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parent.parent; OUT = REPO / "docs/figures"; PARTS = OUT / "method_parts"
CLIP = "move-ego-0-2-raisearms-h-h/move-ego-0-2-raisearms-h-h_6"; FRAMES = (60, 165, 270); STAND = 165
INK, MUTED, RULE, FROZEN, ACCENT, ACCENT_BG = "#22252a", "#62666e", "#a4a7ad", "#e4e5e7", "#d2601a", "#fbe9dc"
ADULT_RGBA, BODY_RGBA = (0.70, 0.72, 0.75, 1), (0.33, 0.38, 0.45, 1)
SOURCES = {
    "adult": dict(xml="assets/robots/adult/robot.xml", qpos=f"data/origin_motion/{CLIP}.npz", rgba=ADULT_RGBA),
    "child": dict(xml="assets/robots_torque/child/robot_torque_full.xml", qpos="outputs/child_rollout_budget/move-ego-0-2-raisearms-h-h_6_r16_s0/best.npz", rgba=BODY_RGBA),
    "g1": dict(xml="assets/robots/g1/scene.xml", qpos="outputs/g1/search/move-ego-0-2-raisearms-h-h_6_s0/best.npz", rgba=BODY_RGBA),
}
PX, SS, DIST, LOOK_Z = 500, 3, 5.0, 0.9        # final px, supersampling, camera distance (m), look-at height (m), the same for every body: true relative size
U_PER_M = 42.0                                  # figure units (0.01 in) per metre of body


def render():
    import mujoco
    PARTS.mkdir(parents=True, exist_ok=True); meta = {}
    for name, s in SOURCES.items():
        m = mujoco.MjModel.from_xml_path(str(REPO / s["xml"])); d = mujoco.MjData(m); q = np.load(REPO / s["qpos"])["qpos"].astype(np.float64); assert q.shape[1] == m.nq, (name, q.shape, m.nq)
        size = PX * SS; m.vis.global_.offwidth = max(m.vis.global_.offwidth, size); m.vis.global_.offheight = max(m.vis.global_.offheight, size)
        r, seg = mujoco.Renderer(m, height=size, width=size), mujoco.Renderer(m, height=size, width=size); seg.enable_segmentation_rendering()
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.elevation, cam.azimuth = DIST, -8, 125
        fov = float(m.vis.global_.fovy); px_per_m = size / (2 * DIST * np.tan(np.radians(fov) / 2)) / SS
        for t in FRAMES:
            d.qpos[:] = q[min(t, len(q) - 1)]; mujoco.mj_forward(m, d); cam.lookat[:] = [d.subtree_com[1][0], d.subtree_com[1][1], LOOK_Z]
            r.update_scene(d, camera=cam); r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False; r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
            for gi in range(r.scene.ngeom):
                g = r.scene.geoms[gi]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] != 0:
                    g.rgba[:] = s["rgba"]; g.matid = -1
            rgb = r.render().copy()
            seg.update_scene(d, camera=cam); id_ = seg.render().copy(); is_geom = (id_[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (id_[..., 0] >= 0)
            mask = np.zeros(is_geom.shape, bool); mask[is_geom] = m.geom_bodyid[id_[..., 0][is_geom]] != 0      # every geom but the floor
            im = Image.fromarray(np.dstack([rgb, mask.astype(np.uint8) * 255]), "RGBA")
            premul = np.asarray(im, float); premul[..., :3] *= premul[..., 3:] / 255          # no dark fringe when shrinking
            small = np.asarray(Image.fromarray(premul[..., :3].astype(np.uint8)).resize((PX, PX), Image.LANCZOS), float); alpha = np.asarray(Image.fromarray(premul[..., 3].astype(np.uint8)).resize((PX, PX), Image.LANCZOS), float)
            rgb2 = np.where(alpha[..., None] > 0, np.clip(small * 255 / np.maximum(alpha[..., None], 1), 0, 255), 255).astype(np.uint8); out = np.dstack([rgb2, alpha.astype(np.uint8)])
            ys, xs = np.nonzero(alpha > 8); box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            Image.fromarray(out, "RGBA").crop(box).save(PARTS / f"{name}_{t}.png")
            meta[f"{name}_{t}"] = dict(w_m=(box[2] - box[0]) / px_per_m, h_m=(box[3] - box[1]) / px_per_m)
        r.close(); seg.close(); print(name, "ok", flush=True)
    (PARTS / "meta.json").write_text(json.dumps(meta, indent=1))


def draw():
    meta = json.loads((PARTS / "meta.json").read_text())
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"})
    W, Y0, Y1 = 550, 14, 243; fig = plt.figure(figsize=(W / 100, (Y1 - Y0) / 100)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Y0, Y1); ax.axis("off")
    FS, SM = 9.0, 8.0

    def txt(x, y, s, size=FS, color=INK, ha="center", va="center", **k):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=6, **k)

    def arrow(pts, color=INK, lw=0.9, ls="-", head=True):
        for a, b in zip(pts[:-2], pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=3)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=7, lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0, zorder=3))

    def body(name, t, x, y0, z=4):
        """cut-out centred on x, feet at y0, at the figure's one scale."""
        mt = meta[f"{name}_{t}"]; w, h = mt["w_m"] * U_PER_M, mt["h_m"] * U_PER_M
        ax.imshow(np.asarray(Image.open(PARTS / f"{name}_{t}.png")), extent=(x - w / 2, x + w / 2, y0, y0 + h), zorder=z, interpolation="lanczos", aspect="auto")

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
    ax.add_patch(Rectangle((12, 127), 92, 95, fc="white", ec=INK, lw=0.9, zorder=2)); txt(58, 212, "New body", size=FS)
    body("child", STAND, 37, 134); body("g1", STAND, 79, 134); ax.plot([18, 98], [133, 133], color=RULE, lw=0.5, zorder=3)
    py = 196; network(190, 250, py - 28, py + 28, r"$\pi$", wide_right=True)
    arrow([(190, py), (104, py)]); txt(147, py + 6, r"$a_t$")
    ax.add_patch(Rectangle((124, 127), 58, 30, fc="white", ec=INK, lw=0.9, zorder=4)); txt(153, 144.5, r"$\Phi$", size=11); txt(153, 133.5, "scale to adult", size=SM - 0.5, color=MUTED)
    arrow([(104, 142), (124, 142)]); txt(114, 149, r"$s_t$")
    ax.plot([182, 200], [142, 142], color=INK, lw=0.9, zorder=3)          # one scaled observation, shared by the policy and the backward map
    ax.add_patch(Circle((200, 142), 1.8, fc=INK, ec="none", zorder=5))
    arrow([(200, 142), (225, 142), (225, py - 28)]); txt(232, 152, r"$o_t$")

    # ---- the only thing that changes
    zx = 369
    ax.add_patch(Circle((zx, py), 13, fc=ACCENT, ec="none", zorder=5)); txt(zx, py + 0.5, r"$z$", size=12, color="white")
    arrow([(zx - 13, py), (250, py)], color=ACCENT, lw=1.5)
    arrow([(zx, 240), (zx, py + 13)], color=ACCENT, lw=0.9, ls=(0, (2.5, 2))); txt(zx + 6, 230, r"$z_0$", color=ACCENT, ha="left")
    ax.add_patch(FancyBboxPatch((346, 45), 46, 40, boxstyle="round,pad=0,rounding_size=4", fc=ACCENT_BG, ec=ACCENT, lw=1.1, zorder=4)); txt(zx, 71, "LOES", size=FS + 1, color=ACCENT, fontweight="bold"); txt(zx, 57, r"update $z$", size=SM, color=ACCENT)
    arrow([(zx, 85), (zx, py - 13)], color=ACCENT, lw=1.5); txt(zx + 6, 135, r"new $z$", color=ACCENT, ha="left")

    # ---- what the cost compares
    txt(68, 104, "Adult reference", size=FS); strip("adult", 24, 44, 22)
    network(174, 226, 39, 91, r"$B$", wide_right=False)
    arrow([(136, 65), (174, 65)]); arrow([(200, 142), (200, 87.5)]); txt(205, 106, r"$o_{1:T}$", ha="left")
    ax.add_patch(Rectangle((238, 33), 100, 64, fc="white", ec=INK, lw=0.9, zorder=4))
    txt(288, 87, "Cost", size=FS + 0.5, fontweight="bold"); txt(288, 74, "BFM similarity", size=SM); txt(288, 62.5, r"+ heading + position", size=SM); txt(288, 51, r"+ anchor", size=SM)
    arrow([(226, 65), (238, 65)]); arrow([(338, 65), (346, 65)])

    # ---- result
    ax.add_patch(FancyBboxPatch((418, 18), 128, 223, boxstyle="round,pad=0,rounding_size=5", fc="none", ec=RULE, lw=0.8, zorder=1))
    arrow([(392, 65), (418, 65)], color=ACCENT, lw=1.5); txt(405, 72, r"$z^*$", color=ACCENT)
    txt(482, 232, "Result on each body", size=FS)
    strip("child", 442, 40, 144); txt(426, 212, "Child", size=SM, color=MUTED, ha="left")
    strip("g1", 442, 40, 28); txt(426, 104, "G1", size=SM, color=MUTED, ha="left")
    for f, k in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        fig.savefig(OUT / f"method_figure.{f}", facecolor="white", **k)
    print(OUT / "method_figure.pdf")


if __name__ == "__main__":
    {"render": render, "draw": draw}[sys.argv[1]]()
