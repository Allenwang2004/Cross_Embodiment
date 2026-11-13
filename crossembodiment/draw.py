"""draw.py -- small PIL / MuJoCo-camera drawing helpers shared by the render scripts (OOP refactor step 4; step 10 builds the
visualisation layer on them: project is crossembodiment.camera.project now, frame_label is VideoRenderer's bitmap title band).

From scripts/rollout_z_trace.py (render_panels.py and render_seed_bests.py imported it from there):
  frame_label(frame, text, sub=None)   rollout_z_trace.label: a 26 px light title band with PIL's bitmap font burnt into a video
                                       frame, an optional grey subtitle below it

From scripts/g1_axes_demo.py (robot_pairs.py imported the module as `gad`; archived in step 7), the explanation panels' helpers:
  S, K, px(v)            panel size 720 px; every pixel size v is drawn as round(v * K). K is module state: robot_pairs.py sets
                         `draw.K = W / 1.25 / 420` for its 1080 px panels, as it set g1_axes_demo.K before (g1_axes_demo.py
                         imported these back and never changed K)
  FONT, font(size)       Noto Sans CJK TC at px(size)
  cam(az, lookat, dist, el)       a free MjvCamera
  project(r, pts, W, H)  world points -> pixel (u, v) through a Renderer's current camera
  label(img, lines, size=22)      the panel under a white, wrapping title band
  hcat(panels)           panels side by side, shorter title bands padded white on top
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------------ video frames (rollout_z_trace.label)

def frame_label(frame, text, sub=None):
    """Burn the checkpoint's identity into its own panel -- a grid of otherwise
    identical humanoids is unreadable without it, and a legend outside the
    video cannot survive being clipped into a gif or a slide."""
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(252, 252, 251))
    d.text((6, 6), text, fill=(11, 11, 11))
    if sub:
        d.text((6, 30), sub, fill=(82, 81, 78))
    return np.asarray(im)


# ------------------------------------------------------------------ explanation panels (g1_axes_demo)

FONT = str(Path.home() / ".local/share/fonts/NotoSansCJKtc-Regular.otf")
S = 720                    # panel size (px); every pixel size below scales with K
K = S / 420


def px(v):
    return int(round(v * K))


def cam(az, lookat, dist=2.3, el=-8):
    import mujoco
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.azimuth, c.elevation, c.distance = az, el, dist; c.lookat[:] = lookat
    return c


def font(size):
    return ImageFont.truetype(FONT, px(size))


def project(r, pts, W, H):
    """world points -> pixel (u, v) through the renderer's current camera (pinhole, symmetric frustum): crossembodiment.camera.project
    since step 10 (the one copy of this formula)"""
    from .camera import project as _project
    return _project(r, pts, W, H)


def label(img, lines, size=22):
    """The panel under a white title band (the words never sit on the robot); lines wider than the panel wrap."""
    W = img.shape[1]; probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    rows = []
    for i, t in enumerate(lines):
        s = size if i == 0 else size - 6; f = font(s); cur = ""
        for w in t.split(" "):
            nxt = (cur + " " + w).strip()
            if cur and probe.textlength(nxt, font=f) > W - px(20):
                rows.append((cur, s)); cur = w
            else:
                cur = nxt
        rows.append((cur, s))
    band = px(10 + sum(s + 6 for _, s in rows))
    im = Image.new("RGB", (W, img.shape[0] + band), (255, 255, 255))
    im.paste(Image.fromarray(img), (0, band)); dr = ImageDraw.Draw(im)
    y = px(6)
    for t, s in rows:
        dr.text((px(10), y), t, fill=(20, 20, 20), font=font(s)); y += px(s + 6)
    dr.line([(W - 1, 0), (W - 1, im.height)], fill=(200, 200, 200), width=max(1, px(1)))
    return np.array(im)


def hcat(panels):
    """Panels side by side; a shorter title band is padded white on top so the renders line up."""
    H = max(p.shape[0] for p in panels)
    return np.concatenate([np.pad(p, ((H - p.shape[0], 0), (0, 0), (0, 0)), constant_values=255) for p in panels], 1)
