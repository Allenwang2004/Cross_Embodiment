#!/usr/bin/env python3
"""payload_all_blender.py -- Blender strobe panels (adult reference, z0, LOES's best) of all 52 payload task motions, for the user to
choose payload examples from (user, 11/10). Preview quality (96 samples, 0.6 x resolution); the chosen ones get a final render.

Data: the 52-motion payload run (outputs/body_52/payload, seed 0, LOES's best after 2,048 rollouts), packaged as
scripts/payload_fall_candidates.py does (survey format, survey frame alignment) -> outputs/payload_all52/trajectories/<clip>.
Frames: 6 poses. Where z0 falls (pelvis below half the reference's for >= 0.5 s): 15 frames apart from 40 frames before z0's fall,
as in payload_fall_candidates.py; otherwise evenly over the whole clip. View: one fixed world view (azimuth 140, the survey videos')
for every panel. Pages: overview_<k>.jpg, 9 motions each, ordered by z0's time below half (longest first); index.md lists all 52.
usage: uv run python scripts/payload_all_blender.py [render] [pages]   (no argument: package, export, render, pages)"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import payload_fall_candidates as P                                       # noqa: E402
import export_strobe_blender as B                                         # noqa: E402

OUT = REPO / "outputs/payload_all52"; P.OUT = OUT
CLIPS = sorted(json.load(open(REPO / "outputs/body_52/payload/results.json"))["per_motion"]["best"])
ROWS = 9


def export():
    plan, panels = {}, {}
    for c in CLIPS:
        p = P.package(c); falls = p["pelvis_below_half_s"]["z0"] >= 0.5
        if not falls:                                     # evenly over the clip
            iv = (p["n"] - 2) // (P.N - 1); p.update(start=1, end=1 + iv * (P.N - 1))
        p["window"] = "around z0's fall" if falls else "whole clip"; iv = (p["end"] - p["start"]) // (P.N - 1)
        cfg = OUT / "configs" / f"heavy_arm_{c}.yaml"; cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(yaml.safe_dump(dict(strobe=dict(start_step=p["start"], end_step=p["end"], interval=iv), camera=dict(type="fixed", azimuth_offset=140.0))))
        panels.update(B.clip_panels("heavy_arm_task52", c, cfg_path=cfg, traj_dir=OUT / "trajectories" / c)); plan[c] = p
    d = OUT / "blender"; d.mkdir(exist_ok=True)
    (d / "scene.json").write_text(json.dumps(dict(panels=panels))); np.savez_compressed(d / "meshes.npz", **B.E.meshes)
    (d / "render_blender.py").write_bytes((REPO / "scripts/render_blender.py").read_bytes()); (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    print("panels", len(panels), "clips", len(plan), "z0 falls", sum(v["window"] != "whole clip" for v in plan.values()), flush=True)


def render():
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", "96", "--scale", "0.6",
                    "--device", "CPU", "--threads", "32", "--out", "renders"], cwd=OUT / "blender", env=P.ENV, check=True, stdout=open(OUT / "blender/render.log", "w"), stderr=subprocess.STDOUT)


def pages():
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype("DejaVuSans.ttf", 20); Wc, L = 640, 36; plan = json.load(open(OUT / "plan.json"))
    order = sorted(plan, key=lambda c: (-plan[c]["pelvis_below_half_s"]["z0"], c)); lines = []
    for k in range(0, len(order), ROWS):
        rows = []
        for c in order[k:k + ROWS]:
            ims = [Image.open(OUT / "blender/renders" / f"heavy_arm_{c}_{s}.png").convert("RGB") for s in ("adult", "z0", "zstar")]
            rows.append([im.resize((Wc, round(Wc * im.height / im.width))) for im in ims])
        sheet = Image.new("RGB", (3 * Wc, sum(r[0].height for r in rows) + L * (len(rows) + 1)), "white"); dr = ImageDraw.Draw(sheet); y = L
        for j, lab in enumerate(("reference (adult)", "z0", "LOES (best latent)")):
            dr.text((j * Wc + 10, 8), lab, fill=(0, 0, 0), font=F)
        for i, (c, row) in enumerate(zip(order[k:k + ROWS], rows)):
            p = plan[c]; num = k + i + 1
            dr.text((10, y + 6), f"#{num} {c}: z0 down {p['pelvis_below_half_s']['z0']:.1f} s, LOES {p['pelvis_below_half_s']['best']:.1f} s | MPJPE {p['mpjpe_cm']['z0']:.1f} -> {p['mpjpe_cm']['best']:.1f} cm | "
                                 f"frames {p['start']}-{p['end']} ({p['window']})", fill=(0, 0, 0), font=F)
            for j, im in enumerate(row):
                sheet.paste(im, (j * Wc, y + L))
            y += L + row[0].height
            lines.append(f"| {num} | `{c}` | {p['n'] / 30:.1f} | {p['pelvis_below_half_s']['z0']:.1f} | {p['pelvis_below_half_s']['best']:.1f} | {p['contact_fall_s']['z0']:.1f} | "
                         f"{p['contact_fall_s']['best']:.1f} | {p['mpjpe_cm']['z0']:.1f} | {p['mpjpe_cm']['best']:.1f} | {'pass' if p['success']['best'] else 'fail'} | "
                         f"{p['start']}-{p['end']} | overview_{k // ROWS + 1}.jpg |")
        sheet.save(OUT / f"overview_{k // ROWS + 1}.jpg", quality=88); print("page", k // ROWS + 1, sheet.size, flush=True)
    (OUT / "index.md").write_text("# Payload, all 52 task motions: Blender strobe previews\n\nSeed 0, LOES's best after 2,048 rollouts. \"Down\": seconds with the pelvis below half "
                                  "the reference's pelvis height. Contact fall: seconds with a body part other than the feet on the floor while the reference's is not. "
                                  "Success: PHC 0.5 m (almost every payload search fails it by drifting from the path).\n\n"
                                  "| # | clip | length s | z0 down s | LOES down s | z0 contact fall s | LOES contact fall s | MPJPE z0 cm | MPJPE LOES cm | LOES success | frames | page |\n"
                                  "|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if not args:
        export(); render(); pages()
    else:
        if "render" in args:
            render()
        if "pages" in args:
            pages()
