#!/usr/bin/env python3
"""fake_render.py SCRIPT [ARGS ...] -- run a render script with MuJoCo's GPU rasterisation replaced by a deterministic picture of the
scene it was given (OOP refactor step 10, tests/golden/verify_vis.py).

MuJoCo's EGL renders on this machine's GPU are not bit-reproducible: the same scene rendered twice, even in one process, differs by
one intensity level on a few thousand pixels of a 30-frame 3240 x 1080 video. To compare two render pipelines exactly, this wrapper
patches mujoco.Renderer.render: everything else runs for real (models, mj_forward, update_scene, mjv_addGeoms, the painting of the
scene's geoms, the labels and legends burnt into the frames, the grid, imageio / ffmpeg), but the returned frame is a 16 x 16 grid of
colours derived from a SHA-256 of the scene the GPU would have drawn: every geom's type, data / object ids, category, material,
size, position, orientation, colour and emission, the light / camera / frame state (camera position, forward, up, frustum) and the
render flags. Any difference in what would have been drawn changes the frame; identical scenes give identical frames and videos.
usage: python tests/golden/fake_render.py scripts/render_panels.py spec.json   (cwd and sys.argv as the script expects)
"""
import hashlib
import os
import runpy
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

GEOM_FIELDS = ("type", "dataid", "objtype", "objid", "category", "matid", "texcoord", "segid", "size", "pos", "mat", "rgba", "emission",
               "specular", "shininess", "reflectance", "transparent")
CAM_FIELDS = ("pos", "forward", "up", "frustum_center", "frustum_width", "frustum_bottom", "frustum_top", "frustum_near", "frustum_far", "orthographic")


def scene_digest(scn):
    h = hashlib.sha256()
    for i in range(scn.ngeom):
        g = scn.geoms[i]
        for f in GEOM_FIELDS:
            if hasattr(g, f):
                h.update(np.asarray(getattr(g, f), dtype=np.float64).tobytes())
    for c in scn.camera:
        for f in CAM_FIELDS:
            if hasattr(c, f):
                h.update(np.asarray(getattr(c, f), dtype=np.float64).tobytes())
    for i in range(scn.nlight):
        L = scn.lights[i]; h.update(np.concatenate([np.ravel(np.asarray(getattr(L, f), dtype=np.float64)) for f in ("pos", "dir", "diffuse")]).tobytes())
    h.update(np.asarray(scn.flags, dtype=np.float64).tobytes()); h.update(np.int64(scn.ngeom).tobytes())
    return h.digest()


def fake_render(self, *a, **k):
    d = scene_digest(self.scene); rng = np.random.default_rng(int.from_bytes(d[:8], "little"))
    H, W = self.height, self.width; cells = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    yi = (np.arange(H) * 16 // H)[:, None]; xi = (np.arange(W) * 16 // W)[None, :]
    return cells[yi, xi]


mujoco.Renderer.render = fake_render

if __name__ == "__main__":
    script = sys.argv[1]; sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
