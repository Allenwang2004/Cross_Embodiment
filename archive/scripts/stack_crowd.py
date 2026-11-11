#!/usr/bin/env python3
"""stack_crowd.py -- z0 (top) and searched-z (bottom) crowd videos of one clip stacked into one video, frame by frame.
Both halves share the camera path (render_crowd.py), so they line up. Streams frames (no full load).
  python scripts/stack_crowd.py <clip stem> [...]   ->  outputs/multibody_corr/crowd/<stem>_z0_vs_best.mp4 (+ .png)"""
import sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

CROWD = Path(__file__).resolve().parent.parent / "outputs/multibody_corr/crowd"
for st in sys.argv[1:]:
    top, bot = imageio.get_reader(CROWD / f"{st}_z0.mp4"), imageio.get_reader(CROWD / f"{st}_best.mp4")
    out = CROWD / f"{st}_z0_vs_best.mp4"
    w = imageio.get_writer(out, fps=top.get_meta_data()["fps"], quality=8, macro_block_size=8)
    frames, keep = 0, {}
    n = top.count_frames()
    picks = {int(n * f) for f in (0.1, 0.5, 0.9)}
    for i, (a, b) in enumerate(zip(top, bot)):
        f = np.concatenate([a, np.full((8, a.shape[1], 3), 255, np.uint8), b], 0)
        w.append_data(f); frames += 1
        if i in picks:
            keep[i] = f
    w.close()
    Image.fromarray(np.concatenate([keep[k] for k in sorted(keep)], 1)).resize((3 * 2560 // 4, (2 * 1080 + 8) // 4)).save(str(out) + ".png")
    print(f"-> {out} ({frames} frames)")
