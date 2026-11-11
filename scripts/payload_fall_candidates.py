#!/usr/bin/env python3
"""payload_fall_candidates.py -- strobe figures (MuJoCo and Blender) of the payload motions where z0 falls and LOES does not, for the
user to choose new payload examples from (user, 11/10). Not added to the user survey until one is chosen.

Candidates: the 52-motion payload run (outputs/body_52/payload, seed 0, LOES's best after 2,048 rollouts); "falls" = the pelvis
below half the reference's pelvis height at the same frame for >= 0.5 s; kept: z0 falls, LOES < 0.5 s. Ordered by the drop in that time.
Steps: (1) trajectories in the survey format (outputs/user_survey's frame alignment: references from their second frame, all files
the reference's length minus one; crossenbodiment.clips.ClipBundle) -> outputs/payload_fall_candidates/trajectories/<clip>; (2) strobe
figures with strobe_figures/replay_strobe.py, 6 poses 15 frames apart from 40 frames before z0's fall, camera fixed at the survey view
(azimuth 140) -> strobe/, configs/; (3) the same frames in Blender (crossenbodiment.strobe.recipe_panels, render_blender.py) -> blender/;
(4) overview_mujoco.jpg, overview_blender.jpg (crossenbodiment.sheet.ContactSheet).
usage: uv run python scripts/payload_fall_candidates.py"""
import glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.blender import BlenderScene                          # noqa: E402
from crossenbodiment.clips import ClipBundle, origin_clip                 # noqa: E402
from crossenbodiment.palette import Palette                               # noqa: E402
from crossenbodiment.sheet import ContactSheet                            # noqa: E402
from crossenbodiment.strobe import RECIPE_DX, recipe_panels               # noqa: E402

OUT = paths.write_path("outputs/payload_fall_candidates"); RUN_REL = "outputs/body_52/payload"; RUN = paths.read_path(RUN_REL); PKG = REPO / "strobe_figures"
CLIPS = ["move-ego-0-2-raisearms-h-m_1", "move-ego-0-2-raisearms-m-h_1", "raisearms-h-l_1", "split-1_7", "rotate-z-5-0.8_9"]
ENV = dict(os.environ, MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="2", MUJOCO_EGL_DEVICE_ID="2", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
N, STEP, BEFORE = 6, 15, 40


def low_s(q, ref):
    return float((q[:, 2] < 0.5 * ref[:, 2]).sum() / 30)


def package(c):
    b = ClipBundle.from_run(c, "heavy_arm", run=f"{RUN_REL}/{c}_es_s0", reference_body=origin_clip(c)); q = b.qpos; adult = q["reference_adult"]; n = len(adult)
    R = json.load(open(RUN / "results.json"))["per_motion"]["best"][c]
    below = q["z0"][:, 2] < 0.5 * adult[:, 2]; tf = int(np.argmax(below))
    info = dict(pelvis_below_half_s=dict(z0=low_s(q["z0"], adult), best=low_s(q["best"], adult)), z0_falls_at_frame=tf,
                contact_fall_s=dict(z0=round(R["frozen"]["fall_s"], 2), best=round(R["es"]["fall_s"], 2)),
                mpjpe_cm=dict(z0=round(R["frozen"]["mpjpe"], 2), best=round(R["es"]["mpjpe"], 2)), success=dict(z0=bool(R["frozen"]["success"]), best=bool(R["es"]["success"])))
    tmpl = json.loads(next((paths.read_path("outputs/user_survey") / "heavy_arm/trajectories").glob("*/video.json")).read_text())
    b.meta.update(candidate=info, side_by_side=tmpl["side_by_side"], overlay=tmpl["overlay"]); b.save(OUT / "trajectories" / c)
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
    dx = dict(RECIPE_DX, **{"heavy_arm_task52/split-1_7": 0.75}); sc = BlenderScene()
    for c in CLIPS:
        sc.panels.update(recipe_panels(sc, "heavy_arm_task52", c, cfg_path=OUT / "configs" / f"heavy_arm_{c}.yaml", traj_dir=OUT / "trajectories" / c, dx_table=dx))
    d = sc.write(OUT / "blender")
    BlenderScene.render(d, samples=128, threads=32, out="renders", env=ENV, log=None)


def overview(plan, kind):
    sheet = ContactSheet((Palette.label("adult", "pages"), Palette.label("z0", "plain"), Palette.label("loes", "pages_best")), col_width=640, label_h=40, font_size=22, quality=90)
    for i, c in enumerate(CLIPS):
        ims = [glob.glob(str(OUT / "strobe" / f"heavy_arm_{c}_fixed_{k}_strobe.png"))[0] for k in ("adult", "z0", "zstar")] if kind == "mujoco" else \
              [OUT / "blender/renders" / f"heavy_arm_{c}_{k}.png" for k in ("adult", "z0", "zstar")]
        p = plan[c]
        sheet.add_row(f"#{i + 1} {c}: down {p['pelvis_below_half_s']['z0']:.1f} s -> {p['pelvis_below_half_s']['best']:.1f} s, MPJPE {p['mpjpe_cm']['z0']:.1f} -> {p['mpjpe_cm']['best']:.1f} cm, frames {p['start']}-{p['end']}", ims)
    im = sheet.save(OUT / f"overview_{kind}.jpg"); print("overview", kind, im.size)


if __name__ == "__main__":
    plan = {}
    for c in CLIPS:
        plan[c] = package(c); p = plan[c]
        print(c, f"n {p['n']}, z0 down at frame {p['z0_falls_at_frame']}, down s z0 {p['pelvis_below_half_s']['z0']:.1f} LOES {p['pelvis_below_half_s']['best']:.1f}, window {p['start']}-{p['end']}, travel {p['travel']:.2f} m", flush=True)
        strobe(c, p)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    blender()
    overview(plan, "mujoco"); overview(plan, "blender")
