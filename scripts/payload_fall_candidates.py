#!/usr/bin/env python3
"""payload_fall_candidates.py -- strobe figures (MuJoCo and Blender) of the payload motions where z0 falls and LOES does not, for the
user to choose new payload examples from (user, 11/10). Not added to the user survey until one is chosen.

Candidates: the 52-motion payload run (outputs/body_52/payload, seed 0, LOES's best after 2,048 rollouts); "falls" = the pelvis
below half the reference's pelvis height at the same frame for >= 0.5 s; kept: z0 falls, LOES < 0.5 s. Ordered by the drop in that time.
Steps: (1) trajectories in the survey format (outputs/user_survey's frame alignment: references from their second frame, all files
the reference's length minus one) -> outputs/payload_fall_candidates/trajectories/<clip>; (2) strobe figures with
strobe_figures/replay_strobe.py, 6 poses 15 frames apart from 40 frames before z0's fall, camera fixed at the survey view (azimuth 140)
-> strobe/, configs/; (3) the same frames in Blender (scripts/export_strobe_blender.py's panels, render_blender.py) -> blender/;
(4) overview_mujoco.jpg, overview_blender.jpg.
usage: uv run python scripts/payload_fall_candidates.py"""
import glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "outputs/payload_fall_candidates"; RUN = REPO / "outputs/body_52/payload"; PKG = REPO / "strobe_figures"
CLIPS = ["move-ego-0-2-raisearms-h-m_1", "move-ego-0-2-raisearms-m-h_1", "raisearms-h-l_1", "split-1_7", "rotate-z-5-0.8_9"]
ENV = dict(os.environ, MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="2", MUJOCO_EGL_DEVICE_ID="2", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
N, STEP, BEFORE = 6, 15, 40


def low_s(q, ref):
    return float((q[:, 2] < 0.5 * ref[:, 2]).sum() / 30)


def package(c):
    t = c.rsplit("_", 1)[0]; adult = np.load(glob.glob(str(REPO / f"data/origin_motion/{t}/{c}.npz"))[0])["qpos"][1:]; n = len(adult)
    q = dict(reference_adult=adult, reference_body=adult, z0=np.load(RUN / f"{c}_es_s0/origin_z.npz")["qpos"][:n], best=np.load(RUN / f"{c}_es_s0/best.npz")["qpos"][:n])
    d = OUT / "trajectories" / c; d.mkdir(parents=True, exist_ok=True)
    for k, v in q.items():
        np.savez_compressed(d / f"{k}.npz", qpos=v.astype(np.float32), fps=np.float32(30))
    R = json.load(open(RUN / "results.json"))["per_motion"]["best"][c]
    below = q["z0"][:, 2] < 0.5 * adult[:, 2]; tf = int(np.argmax(below))
    info = dict(pelvis_below_half_s=dict(z0=low_s(q["z0"], adult), best=low_s(q["best"], adult)), z0_falls_at_frame=tf,
                contact_fall_s=dict(z0=round(R["frozen"]["fall_s"], 2), best=round(R["es"]["fall_s"], 2)),
                mpjpe_cm=dict(z0=round(R["frozen"]["mpjpe"], 2), best=round(R["es"]["mpjpe"], 2)), success=dict(z0=bool(R["frozen"]["success"]), best=bool(R["es"]["success"])))
    tmpl = json.loads(next((REPO / "outputs/user_survey/heavy_arm/trajectories").glob("*/video.json")).read_text())
    (d / "video.json").write_text(json.dumps(dict(clip=c, body="heavy_arm", fps=30, frames={k: len(v) for k, v in q.items()}, search_run=str((RUN / f"{c}_es_s0").relative_to(REPO)),
                                                  candidate=info, side_by_side=tmpl["side_by_side"], overlay=tmpl["overlay"]), indent=1))
    s = int(np.clip(tf - BEFORE, 1, n - 1 - STEP * (N - 1))); travel = float(np.linalg.norm(adult[s + STEP * (N - 1), :2] - adult[s, :2]))
    return dict(n=n, start=s, end=s + STEP * (N - 1), travel=travel, **info)


def strobe(c, p):
    task = f"heavy_arm_{c}"; (OUT / "configs").mkdir(parents=True, exist_ok=True)
    sets = [f"replay.clip=../outputs/payload_fall_candidates/trajectories/{c}", f"task_name={task}", f"strobe.start_step={p['start']}", f"strobe.end_step={p['end']}",
            f"strobe.interval={STEP}", "camera.type=fixed", "camera.azimuth_offset=140", "strobe.shift_px=130",
            f"camera.distance={3.5 + round(p['travel'], 1)}", "render.out_dir=../outputs/payload_fall_candidates/strobe"]
    r = subprocess.run(["uv", "run", "python", "replay_strobe.py", "--config", "configs/strobe/heavy_arm/rotate-y-5-0.8_strobe.yaml"] + sum([["--set", x] for x in sets], [])
                       + ["--save-config", str(OUT / "configs" / f"{task}.yaml")], cwd=PKG, env=ENV, capture_output=True, text=True)
    print(" ", c, "strobe exit", r.returncode, [l.strip()[:120] for l in (r.stdout + r.stderr).splitlines() if "warning:" in l], flush=True)


def blender():
    import export_strobe_blender as B
    B.DX["heavy_arm_task52/split-1_7"] = 0.75; P = {}
    for c in CLIPS:
        P.update(B.clip_panels("heavy_arm_task52", c, cfg_path=OUT / "configs" / f"heavy_arm_{c}.yaml", traj_dir=OUT / "trajectories" / c))
    d = OUT / "blender"; d.mkdir(exist_ok=True)
    (d / "scene.json").write_text(json.dumps(dict(panels=P))); np.savez_compressed(d / "meshes.npz", **B.E.meshes)
    (d / "render_blender.py").write_bytes((REPO / "scripts/render_blender.py").read_bytes())
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", "128", "--device", "CPU",
                    "--threads", "32", "--out", "renders"], cwd=d, env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def overview(plan, kind):
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype("DejaVuSans.ttf", 22); Wc, L = 640, 40; rows = []
    for c in CLIPS:
        paths = [glob.glob(str(OUT / "strobe" / f"heavy_arm_{c}_fixed_{k}_strobe.png"))[0] for k in ("adult", "z0", "zstar")] if kind == "mujoco" else \
                [OUT / "blender/renders" / f"heavy_arm_{c}_{k}.png" for k in ("adult", "z0", "zstar")]
        ims = [Image.open(p).convert("RGB") for p in paths]; rows.append([im.resize((Wc, round(Wc * im.height / im.width))) for im in ims])
    sheet = Image.new("RGB", (3 * Wc, sum(r[0].height for r in rows) + L * (len(rows) + 1)), "white"); dr = ImageDraw.Draw(sheet); y = L
    for j, lab in enumerate(("reference (adult)", "z0", "LOES (best latent)")):
        dr.text((j * Wc + 10, 8), lab, fill=(0, 0, 0), font=F)
    for i, (c, row) in enumerate(zip(CLIPS, rows)):
        p = plan[c]; dr.text((10, y + 6), f"#{i + 1} {c}: down {p['pelvis_below_half_s']['z0']:.1f} s -> {p['pelvis_below_half_s']['best']:.1f} s, MPJPE {p['mpjpe_cm']['z0']:.1f} -> {p['mpjpe_cm']['best']:.1f} cm, frames {p['start']}-{p['end']}",
                             fill=(0, 0, 0), font=F)
        for j, im in enumerate(row):
            sheet.paste(im, (j * Wc, y + L))
        y += L + row[0].height
    sheet.save(OUT / f"overview_{kind}.jpg", quality=90); print("overview", kind, sheet.size)


if __name__ == "__main__":
    plan = {}
    for c in CLIPS:
        plan[c] = package(c); p = plan[c]
        print(c, f"n {p['n']}, z0 down at frame {p['z0_falls_at_frame']}, down s z0 {p['pelvis_below_half_s']['z0']:.1f} LOES {p['pelvis_below_half_s']['best']:.1f}, window {p['start']}-{p['end']}, travel {p['travel']:.2f} m", flush=True)
        strobe(c, p)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    blender()
    overview(plan, "mujoco"); overview(plan, "blender")
