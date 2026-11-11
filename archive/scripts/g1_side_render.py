#!/usr/bin/env python3
"""g1_side_render.py -- one frame of a clip, straight from the side, in ReActor's look, no text.

The adult's original frame and the same frame of the clip mapped onto the G1 kinematically (outputs/g1/roundtrip/
<stem>/g1.npz), each seen exactly side-on (camera at the body's left, 2 deg above the floor). ReActor's palette
(arXiv 2605.06593, sampled from their Fig. 5): bodies steel blue ~97/126/143, sky and floor one grey ~178, solid
shading (no emission) on a reflective floor and one shadow-casting light.
Writes outputs/g1/explain/<stem>_f<frame>_side_{g1,adult,both}.png (1620 x 1080 each; both side by side).
usage: MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/g1_side_render.py [--clip ...] [--frame 45]
"""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
GREY = 178 / 255
BODY = (0.50, 0.65, 0.73)          # renders as ~97/126/143 under the lights below
W, H = 1620, 1080


def reactor_look(xml):
    spec = mujoco.MjSpec.from_file(str(REPO / xml))
    for t in [t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]:
        spec.delete(t)
    for l in list(spec.lights):
        spec.delete(l)
    spec.add_texture(name="rx_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[GREY] * 3, rgb2=[GREY] * 3, width=64, height=384)
    spec.add_material(name="rx_floor", rgba=[0.72, 0.72, 0.72, 1], reflectance=0.25, specular=0.0, shininess=0.0)
    spec.add_material(name="rx_body", rgba=[*BODY, 1], specular=0.25, shininess=0.4)
    spec.worldbody.add_light(name="rx_key", pos=[1.5, -1.0, 4.0], dir=[-0.3, 0.2, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.55] * 3, specular=[0.15] * 3, castshadow=True, mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM)
    world = {id(g) for g in spec.worldbody.geoms}
    for g in spec.geoms:            # geom rgba back to MuJoCo's default so the materials' colours apply
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.material, g.rgba = "rx_floor", [0.5, 0.5, 0.5, 1]
        elif id(g) not in world:
            g.material, g.rgba = "rx_body", [0.5, 0.5, 0.5, 1]
    m = spec.compile()
    m.vis.headlight.ambient[:] = 0.45; m.vis.headlight.diffuse[:] = 0.35; m.vis.headlight.specular[:] = 0.0
    m.vis.rgba.haze[:] = [GREY, GREY, GREY, 1]
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    m.vis.quality.shadowsize = 8192
    return m


def side(m, q, fwd_col, scale):
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    P = d.geom_xpos[geoms]; ctr = (P.min(0) + P.max(0)) / 2
    fwd = d.xmat[1].reshape(3, 3)[:, fwd_col]
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0])) + 90      # looking at the body's left side
    cam.elevation, cam.distance = -2, 1.75 * scale
    cam.lookat[:] = [ctr[0], ctr[1], 0.30 * scale]
    r = mujoco.Renderer(m, H, W); r.update_scene(d, camera=cam); img = r.render().copy(); r.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="crawl-0.4-0-d/crawl-0.4-0-d_6")
    ap.add_argument("--frame", type=int, default=45)
    a = ap.parse_args()
    stem = a.clip.split("/")[1]
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"][a.frame]
    tgt = np.load(REPO / f"outputs/g1/roundtrip/{stem}/g1.npz")["qpos"][a.frame]
    out = REPO / "outputs/g1/explain"; out.mkdir(parents=True, exist_ok=True)
    adult = side(reactor_look("assets/robots/adult/robot.xml"), ref, 2, 1.0)
    g1 = side(reactor_look("assets/robots/g1/scene.xml"), tgt, 0, 0.82)
    for name, img in (("adult", adult), ("g1", g1), ("both", np.concatenate([adult, g1], 1))):
        p = out / f"{stem}_f{a.frame}_side_{name}.png"; Image.fromarray(img).save(p); print(f"-> {p}")


if __name__ == "__main__":
    main()
