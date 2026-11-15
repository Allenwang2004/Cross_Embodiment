#!/usr/bin/env python3
"""export_strobe_blender.py -- the strobe figures of the 12 user-survey picks (outputs/strobe_task52, made with
strobe_figures/replay_strobe.py) as Blender panels in the paper figures' look (scripts/render_blender.py): one panel per clip and
source (adult reference, z0, LOES's best), the same frames as each figure's frozen recipe (outputs/strobe_task52/configs).
Writes outputs/strobe_task52_blender_picks/{scene.json, meshes.npz, render_blender.py}; render there with
  uv run --no-project --python 3.11 --with bpy python render_blender.py -- --device CPU --threads 16 --samples 256 --out renders
then `uv run python scripts/export_strobe_blender.py overview` puts each body's panels on one page (overview_<body>.jpg).
View: render_blender.py's camera sits at azimuth 125 deg in the panel frame; each pose is turned by `heading` first, so the camera's
world azimuth is 125 + heading. Recipes with camera.type fixed (motions that stay in place) used world azimuth 140 for every
panel: heading = 15 deg for all three sources. orbit_auto recipes put the camera at the source's own path heading + azimuth_offset:
heading = path heading + offset - 125. Poses are centred on their root at one depth and spaced dx apart sideways in the image (render_blender.py's "shift", as the teaser band).
Look: render_blender.py's single body colour; the payload's hand weight drawn amber and the limited leg's locked right leg red, as in
the teaser; poses fade from light (early) to full colour (late).
usage: uv run python scripts/export_strobe_blender.py [overview]"""
import json, math, shutil, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import export_blender_package as E                                       # noqa: E402  (pose, load, meshes)

SRC = REPO / "outputs/strobe_task52"; OUT = REPO / "outputs/strobe_task52_blender_picks"; S = REPO / "outputs/user_survey"      # not outputs/strobe_task52_blender: since 11/10 that holds the final figures (scripts/strobe_final_blender.py)
BODY = {"g1_task52": ("g1", S / "models/g1/scene.xml", {}, "outputs/g1_52"),
        "heavy_arm_task52": ("heavy_arm", S / "models/heavy_arm.xml", dict(payload=True), "outputs/body_52/payload"),
        "limit_leg_task52": ("limit_leg", S / "models/limit_leg.xml", dict(red_leg=True), "outputs/body_52/limit_leg")}
FILES = (("adult", "reference_adult.npz"), ("z0", "z0.npz"), ("zstar", "best.npz"))
DX = {"limit_leg_task52/crawl-0.4-0-d_1": 0.8, "limit_leg_task52/crawl-0.5-2-d_5": 0.8, "heavy_arm_task52/split-0.5_6": 0.75}   # metres between poses; default below
AZ = 125.0; PX_W = 1500


def path_heading(q, frames):
    a, b = frames[0], frames[-1]; v = q[b, :2] - q[a, :2]
    return float(np.arctan2(v[1], v[0])) if np.hypot(*v) > 1e-3 else 0.0


def clip_panels(folder, clip, cfg_path=None, traj_dir=None):
    """cfg_path / traj_dir: another recipe and clip folder (default: outputs/strobe_task52/configs, the survey's <folder>/trajectories)."""
    body, xml, kw, _ = BODY[folder]; task = f"{body}_{clip}"; cfg = yaml.safe_load(open(cfg_path or SRC / "configs" / f"{task}.yaml"))
    traj_dir = Path(traj_dir or S / folder / "trajectories" / clip)
    st = cfg["strobe"]; frames = list(range(st["start_step"], st["end_step"] + 1, st["interval"])); assert frames[-1] == st["end_step"]
    cam = cfg["camera"]; n = len(frames); tints = [0.62 * (1 - i / (n - 1)) for i in range(n)]; dx = DX.get(f"{folder}/{clip}", 0.6)
    still = cam["type"] == "fixed"; crawl = "crawl" in clip
    W = (n - 1) * dx + (2.4 if crawl else 1.6); H = 1.4 if crawl else 2.45          # one frame per clip, so the adult and the G1 share a scale
    out = {}
    for label, f in FILES:
        m = E.load(S / "models/adult.xml") if label == "adult" else E.load(xml, kw.get("payload", False))
        q = np.load(traj_dir / f)["qpos"].astype(np.float64); assert q.shape[1] == m.nq
        h = math.radians(15.0) if still else path_heading(q, frames) + math.radians(cam["azimuth_offset"] - AZ)
        red = kw.get("red_leg", False) and label != "adult"; tag = "g1" if (body == "g1" and label != "adult") else "human"
        poses = [dict(E.pose(m, q, t, 0.0, k, h, red, tag), shift=(i - (n - 1) / 2) * dx) for i, (t, k) in enumerate(zip(frames, tints))]
        out[f"{task}_{label}"] = dict(figure="strobe", width_m=W, height_m=H, feet_m=0.3, px=[PX_W, round(PX_W * H / W)], white=True,
                                      poses=poses, frames=frames, view="fixed 140 deg" if still else f"side, path + {cam['azimuth_offset']} deg")
    return out


def export():
    OUT.mkdir(parents=True, exist_ok=True); P = {}
    for folder, (body, _, _, res) in BODY.items():
        for r in json.load(open(REPO / res / "top_clips.json"))["top5"]:
            P.update(clip_panels(folder, r["clip"]))
    (OUT / "scene.json").write_text(json.dumps(dict(panels=P))); np.savez_compressed(OUT / "meshes.npz", **E.meshes)
    shutil.copy(REPO / "scripts/render_blender.py", OUT / "render_blender.py")
    for k, v in P.items():
        print(k, v["frames"], v["view"], "px", v["px"], "W m", round(v["width_m"], 2))


def overview():
    from PIL import Image, ImageDraw, ImageFont
    try:
        F = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        F = ImageFont.load_default()
    for folder, (body, _, _, res) in BODY.items():
        top = json.load(open(REPO / res / "top_clips.json"))["top5"]; Wc, L = 640, 40
        ims = [[Image.open(OUT / "renders" / f"{body}_{r['clip']}_{k}.png").convert("RGB") for k, _ in FILES] for r in top]
        ims = [[im.resize((Wc, round(Wc * im.height / im.width))) for im in row] for row in ims]; hs = [row[0].height for row in ims]
        sheet = Image.new("RGB", (3 * Wc, sum(hs) + L * (len(top) + 1)), "white"); d = ImageDraw.Draw(sheet); y = L
        for j, lab in enumerate(("reference (adult)", "z0", "LOES (best latent)")):
            d.text((j * Wc + 10, 8), lab, fill=(0, 0, 0), font=F)
        for i, (r, row) in enumerate(zip(top, ims)):
            flag = "  ⚠ z0 passes the success test" if r["z0_phc_ok"] else ""
            d.text((10, y + 6), f"#{i + 1} {r['clip']}: MPJPE z0 {r['frozen_mpjpe']:.1f} cm -> LOES {r['es_mpjpe']:.1f} cm{flag}", fill=(0, 0, 0), font=F)
            for j, im in enumerate(row):
                sheet.paste(im, (j * Wc, y + L))
            y += L + hs[i]
        sheet.save(OUT / f"overview_{body}.jpg", quality=90); print(body, sheet.size)


if __name__ == "__main__":
    overview() if "overview" in sys.argv else export()
