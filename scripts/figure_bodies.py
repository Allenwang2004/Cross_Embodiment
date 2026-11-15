"""figure_bodies.py -- bodies for the paper's figures: matte cut-outs with a soft floor shadow, to be laid over a flat grey (BFMTrack's background, ReActor's body colour; both sampled from the papers' figures).
The two colours are crossembodiment.palette's INK["figure_bg"] / INK["figure_body"] (OOP refactor step 10)."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
from crossembodiment.palette import INK                  # noqa: E402

BG = INK["figure_bg"]; BODY_RGBA = INK["figure_body"]
PX, SS, DIST, LOOK_Z = 1400, 2, 5.0, 0.9                # about 340 px per metre: over 600 dpi at the figures' scale


def poses(m, q, frames, out, name, azimuth=125, painted=()):
    """Save <out>/<name>_<t>.png (RGBA body) and <name>_<t>_sh.png (floor shadow as alpha), same box. The camera sees the walking direction at `azimuth`. painted: geoms keeping their rgba."""
    import mujoco
    d = mujoco.MjData(m); body = m.geom_bodyid != 0; keep = np.zeros(m.ngeom, bool); keep[list(painted)] = True
    m.vis.quality.numslices, m.vis.quality.numstacks, m.vis.quality.numquads, m.vis.quality.offsamples, m.vis.quality.shadowsize = 96, 48, 12, 8, 8192
    m.vis.headlight.ambient[:] = 0.62; m.vis.headlight.diffuse[:] = 0.26; m.vis.headlight.specular[:] = 0; m.light_diffuse[:] = 0.2; m.light_specular[:] = 0; m.light_ambient[:] = 0.05
    m.geom_matid[:] = -1; m.geom_rgba[~body] = (0.75, 0.75, 0.75, 1)
    size = PX * SS; m.vis.global_.offwidth = max(m.vis.global_.offwidth, size); m.vis.global_.offheight = max(m.vis.global_.offheight, size)
    r, seg = mujoco.Renderer(m, height=size, width=size), mujoco.Renderer(m, height=size, width=size); seg.enable_segmentation_rendering()
    a, b = max(frames[0] - 10, 0), min(frames[-1] + 10, len(q) - 1); v = q[b, :2] - q[a, :2]; h = np.degrees(np.arctan2(v[1], v[0]))
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.elevation, cam.azimuth = DIST, -8, h + azimuth
    ppm = size / (2 * DIST * np.tan(np.radians(float(m.vis.global_.fovy)) / 2)); meta = {}; group = m.geom_group.copy()

    def shot(rend):
        rend.update_scene(d, camera=cam); s = rend.scene; s.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False; s.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True; s.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        for i in range(s.ngeom):
            s.geoms[i].specular = 0.0; s.geoms[i].shininess = 0.05; s.geoms[i].reflectance = 0.0
        return rend.render().copy()
    for t in frames:
        d.qpos[:] = q[min(t, len(q) - 1)]; mujoco.mj_forward(m, d); cam.lookat[:] = [d.subtree_com[1][0], d.subtree_com[1][1], LOOK_Z]
        rgb = shot(r); ids = shot(seg); isg = (ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (ids[..., 0] >= 0); mask = np.zeros(isg.shape, bool); mask[isg] = body[ids[..., 0][isg]]
        m.geom_group[body] = 5; clean = shot(r).astype(float); m.geom_group[:] = group
        sh = 1 - rgb.astype(float).mean(-1) / np.maximum(clean.mean(-1), 1); sh = np.clip(sh, 0, 1); sh[clean.mean(-1) < 5] = 0                       # how much the floor darkened
        sig = 0.03 * ppm; vis = (~mask).astype(float); sh = gaussian_filter(sh * vis, sig) / np.maximum(gaussian_filter(vis, sig), 1e-3); sh = np.clip(gaussian_filter(sh, sig * 0.5), 0, 1)     # soft, filled in under the body
        pre = (rgb.astype(float) * mask[..., None]).astype(np.uint8); small = np.asarray(Image.fromarray(pre).resize((PX, PX), Image.LANCZOS), float); al = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((PX, PX), Image.LANCZOS), float)
        out_im = np.dstack([np.where(al[..., None] > 0, np.clip(small * 255 / np.maximum(al[..., None], 1), 0, 255), 255).astype(np.uint8), al.astype(np.uint8)]); sh = np.asarray(Image.fromarray((sh * 255).astype(np.uint8)).resize((PX, PX), Image.LANCZOS))
        ys, xs = np.nonzero(al > 8); per = ppm / SS; mx, mb = int(0.45 * per), int(0.28 * per); box = (max(xs.min() - mx, 0), ys.min(), min(xs.max() + 1 + mx, PX), min(ys.max() + 1 + mb, PX))
        Image.fromarray(out_im, "RGBA").crop(box).save(out / f"{name}_{t}.png"); Image.fromarray(sh, "L").crop(box).save(out / f"{name}_{t}_sh.png")
        meta[f"{name}_{t}"] = dict(w_m=(box[2] - box[0]) / per, h_m=(box[3] - box[1]) / per, foot_m=(box[3] - 1 - ys.max()) / per, dx_m=(((xs.min() + xs.max() + 1) / 2) - (box[0] + box[2]) / 2) / per)
    r.close(); seg.close(); return meta


def place(ax, parts, meta, name, t, x, y0, scale, tint=0.0, z=4.0, clip=None, shadow=0.6):
    """the cut-out with its body centred on x and its feet at y0; the shadow just under it. tint: how far an earlier pose fades towards the background, still opaque."""
    mt = meta[f"{name}_{t}"]; w, h, f, c = mt["w_m"] * scale, mt["h_m"] * scale, mt["foot_m"] * scale, mt["dx_m"] * scale; ext = (x - c - w / 2, x - c + w / 2, y0 - f, y0 - f + h)
    sh = np.asarray(Image.open(parts / f"{name}_{t}_sh.png")).astype(float) / 255; s = np.zeros(sh.shape + (4,)); s[..., 3] = np.clip(sh * shadow * (1 - tint), 0, 1)
    im = np.asarray(Image.open(parts / f"{name}_{t}.png")).astype(float) / 255; im[..., :3] = im[..., :3] * (1 - tint) + np.array(to_rgb(BG)) * tint
    for a, zz in ((s, 2.5), (im, z)):
        i = ax.imshow(a, extent=ext, zorder=zz, interpolation="lanczos", aspect="auto")
        if clip is not None:
            i.set_clip_path(clip)
