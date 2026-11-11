#!/usr/bin/env python3
"""g1_floating_figure.py -- where the kinematic G1 target floats: one clip, a few frames, side view, in ReActor's look.

ReActor (arXiv 2605.06593, Fig. 1 / 5) draws every body in one steel blue on a seamless light-grey floor, so a body
off the ground shows as a gap between it and its shadow; here the same palette (sampled from their Fig. 5: body
RGB 97/126/143, background 178/178/178), with the gap of the lower hand to the floor drawn as an orange dimension line.
Rows: the adult's original clip (Source) and the clip mapped onto the G1 kinematically (outputs/g1/roundtrip/<stem>/
g1.npz, the retargeted target). Below: the lower hand's height above the floor over the first frames, adult vs G1
target, the frames where the adult's hand is on the floor (< 1 cm) and the G1's is not (> 1 cm) shaded.
Writes outputs/g1/explain/<stem>_floating.png (1080 px panels).
usage: MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/g1_floating_figure.py \\
         [--clip crawl-0.4-0-d/crawl-0.4-0-d_6] [--frames 15 27 45] [--until 60]
"""
import argparse, io, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.robots import collision_geoms   # noqa: E402

S = 1080
BODY = (0.43, 0.57, 0.66, 1.0)       # with emission 0.5, renders as ReActor's steel blue (~97/126/143) under the lights below
GREY = 178 / 255                                      # ReActor background / floor
ORANGE = (232, 116, 59)                               # the floating gap
INK, INK2 = (28, 28, 30), (90, 90, 95)
FONT = Path.home() / ".local/share/fonts/NotoSansCJKtc-Regular.otf"
font = lambda n: ImageFont.truetype(str(FONT), n)
SIDES = {"adult": (("L_Wrist", "L_Hand"), ("R_Wrist", "R_Hand")), "g1": (("left_wrist",), ("right_wrist",))}


def reactor_look(xml):
    """the scene re-dressed as ReActor's: flat grey sky and floor, every body geom steel blue, one shadow-casting light."""
    spec = mujoco.MjSpec.from_file(str(REPO / xml))
    for t in [t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]:
        spec.delete(t)
    for l in list(spec.lights):
        spec.delete(l)
    spec.add_texture(name="rx_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[GREY + 0.05] * 3, rgb2=[GREY] * 3, width=64, height=384)
    # floor colour and lights tuned so the lit floor renders as the sky's grey (~185): no horizon line, as in ReActor
    spec.add_material(name="rx_floor", rgba=[1.0, 1.0, 1.0, 1], reflectance=0.18, specular=0.0, shininess=0.0)
    spec.worldbody.add_light(name="rx_key", pos=[1.5, -1.0, 4.0], dir=[-0.3, 0.2, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.5, 0.5, 0.5], specular=[0.1, 0.1, 0.1], castshadow=True,
                             mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM)
    spec.add_material(name="rx_body", rgba=list(BODY), emission=0.5, specular=0.15, shininess=0.3)
    world = {id(g) for g in spec.worldbody.geoms}
    for g in spec.geoms:                    # geom rgba back to MuJoCo's default so the material's colour applies
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.material, g.rgba = "rx_floor", [0.5, 0.5, 0.5, 1]
        elif id(g) not in world:
            g.material, g.rgba = "rx_body", [0.5, 0.5, 0.5, 1]
    m = spec.compile()
    m.vis.headlight.ambient[:] = 0.25; m.vis.headlight.diffuse[:] = 0.15; m.vis.headlight.specular[:] = 0.0
    m.vis.rgba.haze[:] = [GREY, GREY, GREY, 1]
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, S); m.vis.global_.offheight = max(m.vis.global_.offheight, S)
    m.vis.quality.shadowsize = 8192
    return m


def lower_hand(m, d, who):
    """(gap m, point on the hand, point on the floor) for whichever hand is closer to the floor."""
    floor = m.geom("floor").id; ft = np.zeros(6); best = (np.inf, None, None)
    for keys in SIDES[who]:
        for g in collision_geoms(m):
            if any(k in m.body(m.geom_bodyid[g]).name for k in keys):
                dist = mujoco.mj_geomDistance(m, d, g, floor, 1.0, ft)
                if dist < best[0]:
                    best = (dist, ft[:3].copy(), ft[3:].copy())
    return best


def project(r, p):
    c = r.scene.camera[0]; f, up = np.array(c.forward), np.array(c.up); right = np.cross(f, up)
    k = c.frustum_near / c.frustum_top; v = np.asarray(p) - np.array(c.pos); z = v @ f
    return S / 2 + (v @ right) / z * k * S / 2, S / 2 - (v @ up) / z * k * S / 2


def panel(m, q, who, title, scale):
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    root = 1
    fwd = d.xmat[root].reshape(3, 3)[:, 2 if who == "adult" else 0]
    yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
    gap, ph, pf = lower_hand(m, d, who)
    look = (d.xpos[root] + ph) / 2
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = yaw + 90, -4, 2.0 * scale
    cam.lookat[:] = [look[0], look[1], 0.32 * scale]
    r = mujoco.Renderer(m, S, S); r.update_scene(d, camera=cam); img = Image.fromarray(r.render().copy())
    dr = ImageDraw.Draw(img)
    (u0, v0), (u1, v1) = project(r, ph), project(r, pf)
    r.close()
    cm = max(100 * gap, 0.0)                            # a hand pressed a few mm into the floor counts as 0
    if cm > 1.0:                                        # floating: an orange dimension line hand -> floor
        x = u0
        for y in (v0, v1):
            dr.line([(x - 22, y), (x + 22, y)], fill=ORANGE, width=6)
        dr.line([(x, v0), (x, v1)], fill=ORANGE, width=6)
        dr.text((x + 34, (v0 + v1) / 2), f"{cm:.1f} cm", fill=ORANGE, font=font(48), anchor="lm", stroke_width=5,
                stroke_fill=(245, 245, 245))
    else:
        dr.ellipse([u0 - 14, v0 - 14, u0 + 14, v0 + 14], outline=INK, width=5)
        dr.text((u0 + 30, v0 - 40), "on the floor (0 cm)" if cm < 0.05 else f"on the floor ({cm:.1f} cm)", fill=INK, font=font(40), anchor="lm", stroke_width=5,
                stroke_fill=(235, 235, 235))
    band = Image.new("RGB", (S, 120), (250, 250, 249)); bd = ImageDraw.Draw(band)
    bd.text((24, 38), title, fill=INK, font=font(40), anchor="lm")
    bd.text((24, 88), f"lower hand {cm:.1f} cm above the floor", fill=INK2, font=font(30), anchor="lm")
    out = Image.new("RGB", (S, S + 120)); out.paste(band, (0, 0)); out.paste(img, (0, 120))
    return out, cm


def hand_heights(m, Q, who):
    d = mujoco.MjData(m); h = []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_forward(m, d); h.append(max(100 * lower_hand(m, d, who)[0], 0.0))
    return np.array(h)


def strip(ha, hg, frames, until, width_px):
    t = np.arange(until)
    fig, ax = plt.subplots(figsize=(width_px / 200, 3.6), dpi=200)
    fig.patch.set_facecolor("#FAFAF9"); ax.set_facecolor("#FAFAF9")
    gap = (ha[:until] < 1.0) & (hg[:until] > 1.0)
    for i in np.flatnonzero(gap):
        ax.axvspan(i - 0.5, i + 0.5, color="#E8743B", alpha=0.16, lw=0)
    ax.plot(t, ha[:until], color="#8C8C8C", lw=2, solid_capstyle="round", label="Source (SMPL adult)")
    ax.plot(t, hg[:until], color="#4F7389", lw=2, solid_capstyle="round", label="Retargeted G1 (kinematic target)")
    for f in frames:
        ax.axvline(f, color="#C8C8C8", lw=1)
        ax.text(f, ax.get_ylim()[1] if False else 1.02, f"frame {f}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9, color="#5A5A5F")
    if gap.any():
        mid = np.flatnonzero(gap).mean()
        ax.text(mid, 0.62, f"adult hand down, G1 hand still in the air\n({gap.sum()} frames, {gap.sum() / 30:.1f} s)",
                transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=10, color="#B4572A")
    ax.set_xlim(0, until - 1); ax.set_ylim(-2, max(ha[:until].max(), hg[:until].max()) * 1.05)
    ax.set_xlabel("frame (30 per second)", fontsize=10, color="#3A3A3E")
    ax.set_ylabel("lower hand above floor (cm)", fontsize=10, color="#3A3A3E")
    ax.grid(axis="y", color="#E3E3E3", lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C8C8C8")
    ax.tick_params(colors="#5A5A5F", labelsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    im = Image.open(buf).convert("RGB")
    return im.resize((width_px, int(im.height * width_px / im.width)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="crawl-0.4-0-d/crawl-0.4-0-d_6")
    ap.add_argument("--frames", type=int, nargs="+", default=[15, 27, 45])
    ap.add_argument("--until", type=int, default=60)
    a = ap.parse_args()
    stem = a.clip.split("/")[1]
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
    tgt = np.load(REPO / f"outputs/g1/roundtrip/{stem}/g1.npz")["qpos"]
    ma, mg = reactor_look("assets/robots/adult/robot.xml"), reactor_look("assets/robots/g1/scene.xml")
    cols = []
    for f in a.frames:
        top, _ = panel(ma, ref[f], "adult", f"Source (SMPL adult) · frame {f}", 1.0)
        bot, _ = panel(mg, tgt[f], "g1", f"Retargeted G1 (kinematic target) · frame {f}", 0.82)
        col = Image.new("RGB", (S, top.height + bot.height)); col.paste(top, (0, 0)); col.paste(bot, (0, top.height))
        cols.append(col)
    W = S * len(cols)
    grid = Image.new("RGB", (W, cols[0].height), (255, 255, 255))
    for i, c in enumerate(cols):
        grid.paste(c, (i * S, 0))
        ImageDraw.Draw(grid).line([(i * S, 0), (i * S, grid.height)], fill=(225, 225, 225), width=2)
    ha = hand_heights(ma, ref[:a.until], "adult"); hg = hand_heights(mg, tgt[:a.until], "g1")
    st = strip(ha, hg, a.frames, a.until, W)
    head = Image.new("RGB", (W, 150), (255, 255, 255)); hd = ImageDraw.Draw(head)
    hd.text((30, 52), f"{a.clip}: the retargeted G1 floats when the hands first reach the floor", fill=INK, font=font(52), anchor="lm")
    hd.text((30, 112), "the target puts only its lowest point (here a foot) on the floor; the G1's shorter arms leave its hands "
            "in the air until the torso comes down", fill=INK2, font=font(34), anchor="lm")
    out = Image.new("RGB", (W, head.height + grid.height + st.height), (255, 255, 255))
    out.paste(head, (0, 0)); out.paste(grid, (0, head.height)); out.paste(st, (0, head.height + grid.height))
    path = REPO / "outputs/g1/explain" / f"{stem}_floating.png"; path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    print(f"-> {path} {out.size}; lower hand at frames {a.frames}: adult "
          f"{[round(float(ha[f]), 1) for f in a.frames if f < a.until]} cm, G1 {[round(float(hg[f]), 1) for f in a.frames if f < a.until]} cm")


if __name__ == "__main__":
    main()
