#!/usr/bin/env python3
"""walk_band_labels.py -- the reshaped-body walking band (outputs/walk_bodies/walk_band.png, scripts/walk_band_blender.py) with labels
large enough for a full-width figure in the paper's supplementary material (user, 11/13): two-line labels "L x / W y" (the child: "Child")
under each body's solid pose, font size --size px (64 px is ~8 pt at \\textwidth for the 4,200 px image).
-> docs/figures/reshaped_bodies_band.png.  usage: uv run python scripts/walk_band_labels.py [--size 64]"""
import json, re, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent; BAND = REPO / "outputs/walk_bodies/band"
SIZE = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 64
src = Image.open(REPO / "outputs/walk_bodies/walk_band.png").convert("RGB")
im = Image.new("RGB", (src.width, src.height + 90), (255, 255, 255)); im.paste(src, (0, 0)); dr = ImageDraw.Draw(im)   # room under the lowest label
F = ImageFont.truetype("DejaVuSans.ttf", SIZE); boxes = []
for g in json.loads((BAND / "scene.json").read_text())["groups"]:
    x, y = g["px"]; m = re.match(r"L ([\d.]+), W ([\d.]+)", g["label"])
    text = f"L {m.group(1)}\nW {m.group(2)}" if m else g["label"]
    l, t, r, b = dr.multiline_textbbox((0, 0), text, font=F, align="center"); w = r - l
    box = (x - w / 2, y + 30, x + w / 2, y + 30 + b - t); boxes.append((box, text))
    dr.multiline_text((x - w / 2, y + 30), text, fill=(40, 40, 40), font=F, align="center")
over = [(a[1], c[1]) for i, a in enumerate(boxes) for c in boxes[i + 1:] if a[0][0] < c[0][2] and c[0][0] < a[0][2] and a[0][1] < c[0][3] and c[0][1] < a[0][3]]
print("labels:", len(boxes), "| overlapping pairs:", over)
im.save(REPO / "docs/figures/reshaped_bodies_band.png"); print("saved", im.size)
