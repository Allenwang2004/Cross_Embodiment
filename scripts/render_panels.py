#!/usr/bin/env python3
"""render_panels.py -- play recorded qpos trajectories side by side, one panel
per (body, trajectory), for presentation videos.

Nothing is re-simulated: every panel replays a qpos array that a search already
recorded (single_z_search's origin_z.npz = the rollout of z0, best.npz = the
rollout of the best z it found) or a retargeted reference, on that panel's own
body. So a panel shows exactly the rollout its caption's cost was measured on.

Spec (JSON):
  {"out": "docs/figures/journey/v2_child_move.mp4",
   "cols": 4, "size": 320, "camera": "front_side",
   "panels": [{"title": "reference", "sub": "retargeted to child",
               "xml": "assets/robots_torque/m2c_t1000/robot_torque_full.xml",
               "qpos": "data/m2c_t1000/retargeting_motion/<task>/<stem>.npz"}, ...]}

"track": {"distance", "elevation", "azimuth", "lookat_z"[, "follow": "first"]}: a free camera on each body's
centre of mass, or with "follow": "first" on the first panel's for every panel (drift stays visible).

"mesh": true (spec-wide, or per panel): dress the body in the SMPL mesh deformed to its own proportions and hide
its capsules (scripts/smpl_skin.py; visual only; mesh panels are drawn without shadows, which band the skin).
"mesh_rgba" sets the skin colour (spec-wide or per panel).

"size": panel side in pixels (default 1080; keep the whole grid within ~4096 x 2160 so ordinary players open it,
e.g. 2 x 2 or 3 x 1 / 3 x 2 panels at 1080).
"label_px": title size in pixels for a TrueType label (default: size / 30 above size 480, PIL's small bitmap font
at 480 and below);
"captions": false: no title / subtitle band at all (e.g. survey videos); titles may then be omitted.
"rgba": [r, g, b, a] (per panel): draw the body flat in that colour, e.g. the overlay videos' grey / red / blue.
"quality": the mp4's imageio-ffmpeg quality, 0-10 (default 8 at size >= 1080, else 5).

Writes <out> (mp4) and <out>.png (contact sheet: three moments, panels across).
The rendering is crossenbodiment.video.VideoRenderer.render_spec (OOP refactor step 10: the side-by-side and overlay videos share its
painting and writer settings). Spec paths are repository-relative ("outputs/..." read through $CE_OUTPUTS_ROOT, the video written
under $CE_RESULTS_ROOT; defaults outputs/) or absolute; --out writes the video elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # the repository root, appended as rollout_z_trace did
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)
from crossenbodiment.draw import frame_label as label   # noqa: E402,F401  rollout_z_trace.label (refactor step 4)
from crossenbodiment.video import FONT, VideoRenderer, label_px   # noqa: E402,F401  (render_humanoids.py imports FONT from here)

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", default=None, help="write the video here instead of the spec's out")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = VideoRenderer().render_spec(spec, out=a.out)
    print(f"-> {out}\n-> {out}.png")


if __name__ == "__main__":
    main()
