#!/usr/bin/env python3
"""body_all_blender.py -- Blender strobe panels (adult reference, z0, LOES's best) of all 52 task motions of one body, for the user to
choose examples from, with z0's falls DURING the motion picked out (user, 11/10: falls right at the start are not wanted).

Data: the 52-motion runs (outputs/body_52/<body>, seed 0, LOES's best after 2,048 rollouts), packaged in the survey format and frame
alignment -> outputs/<body>_all52/trajectories/<clip> (crossenbodiment.clips.ClipBundle). reference_body: the body's search target
(limited leg: the motion with the right leg bent; payload: the adult's motion).
Falls: the pelvis below half the target's pelvis height at the same frame. A mid-motion fall: such a stretch that starts after 1.5 s
(frame 45) and lasts at least 0.3 s (9 frames). Frames: 6 poses 15 frames apart from 40 frames before z0's first mid-motion fall;
without one, evenly over the clip (crossenbodiment.strobe). Order: z0 falls mid-motion and LOES is not down during it; z0 falls
mid-motion and LOES is down too; no mid-motion fall of z0 (by MPJPE drop). View: one fixed world view (azimuth 140, the survey
videos') for every panel (crossenbodiment.strobe.recipe_panels, export_strobe_blender.py's clip_panels before step 10).
Preview quality (96 samples, 0.6 x resolution). Pages overview_<k>.jpg, 9 motions each (crossenbodiment.sheet); index.md lists all 52.
usage: uv run python scripts/body_all_blender.py <limit_leg|payload> [export] [render] [pages]   (none: all three)"""
import json, os, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.blender import BlenderScene                          # noqa: E402
from crossenbodiment.clips import ClipBundle, task_of                     # noqa: E402
from crossenbodiment.palette import Palette                               # noqa: E402
from crossenbodiment.sheet import ContactSheet, write_index_md            # noqa: E402
from crossenbodiment.strobe import around_fall, recipe_panels, stretches, whole_clip   # noqa: E402

BODY = sys.argv[1]; assert BODY in ("limit_leg", "payload")
FOLDER = {"limit_leg": "limit_leg_task52", "payload": "heavy_arm_task52"}[BODY]; TAG = {"limit_leg": "limit_leg", "payload": "heavy_arm"}[BODY]
REF = {"limit_leg": "data/adult_lockRleg_bent_lim/retargeting_motion/{t}/{c}.npz", "payload": "data/origin_motion/{t}/{c}.npz"}[BODY]
RUN_REL = f"outputs/body_52/{BODY}"; RUN = paths.read_path(RUN_REL); OUT_REL = f"outputs/{BODY}_all52"; OUT = paths.write_path(OUT_REL)
CLIPS = sorted(json.load(open(RUN / "results.json"))["per_motion"]["best"]); N, STEP, BEFORE, START, MINLEN, ROWS = 6, 15, 40, 45, 9, 9
ENV = dict(os.environ, MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="2", MUJOCO_EGL_DEVICE_ID="2", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
COLUMNS = (Palette.label("adult", "pages"), Palette.label("frozen", "plain"), Palette.label("loes", "pages_best"))


def package(c):
    t = task_of(c); run = f"{RUN_REL}/{c}_es_s0"
    b = ClipBundle.from_run(c, TAG, run=run, reference_body=REF.format(t=t, c=c), same_length=True); q = b.qpos; n = b.n
    ref = q["reference_body"][:, 2]; below = {k: q[k][:, 2] < 0.5 * ref for k in ("z0", "best")}
    mid = {k: [(a, b_) for a, b_ in stretches(below[k]) if a >= START and b_ - a >= MINLEN] for k in below}
    early = {k: [(a, b_) for a, b_ in stretches(below[k]) if a < START] for k in below}
    R = json.load(open(RUN / "results.json"))["per_motion"]["best"][c]
    p = dict(n=n, z0_mid_falls=mid["z0"], best_mid_falls=mid["best"], z0_early_falls=early["z0"],
             mid_down_s={k: sum(b_ - a for a, b_ in mid[k]) / 30 for k in mid}, mpjpe_cm=dict(z0=round(R["frozen"]["mpjpe"], 2), best=round(R["es"]["mpjpe"], 2)),
             success=dict(z0=bool(R["frozen"]["success"]), best=bool(R["es"]["success"])), contact_fall_s=dict(z0=round(R["frozen"]["fall_s"], 2), best=round(R["es"]["fall_s"], 2)))
    if mid["z0"]:
        a, b_ = mid["z0"][0]; s, e, iv = around_fall(mid["z0"][0], n, N, STEP, BEFORE); p.update(start=s, end=e, interval=iv, window=f"z0's fall at frames {a}-{b_}")
        p["best_down_during"] = float(below["best"][a:b_].mean())
    else:
        s, e, iv = whole_clip(n, N); p.update(start=s, end=e, interval=iv, window="whole clip", best_down_during=None)
    tmpl = json.loads(next((paths.read_path("outputs/user_survey") / TAG / "trajectories").glob("*/video.json")).read_text())
    b.meta.update(falls=p, side_by_side=tmpl["side_by_side"], overlay=tmpl["overlay"]); b.save(OUT / "trajectories" / c)
    return p


def group(p):
    if p["z0_mid_falls"]:
        return 0 if p["best_down_during"] < 0.1 else 1
    return 2


def export():
    plan, sc = {}, BlenderScene()
    for c in CLIPS:
        p = package(c); plan[c] = p
        cfg = OUT / "configs" / f"{TAG}_{c}.yaml"; cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(yaml.safe_dump(dict(strobe=dict(start_step=p["start"], end_step=p["end"], interval=p["interval"]), camera=dict(type="fixed", azimuth_offset=140.0))))
        sc.panels.update(recipe_panels(sc, FOLDER, c, cfg_path=cfg, traj_dir=OUT / "trajectories" / c))
    sc.write(OUT / "blender"); (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    g = [group(p) for p in plan.values()]
    print(f"{BODY}: {len(plan)} clips, z0 falls mid-motion on {g.count(0) + g.count(1)} (LOES not down during it: {g.count(0)}), panels {len(sc.panels)}", flush=True)
    for c in sorted(plan, key=lambda c: (group(plan[c]), -plan[c]["mid_down_s"]["z0"])):
        p = plan[c]
        if group(p) < 2:
            print(f"  {c:30s} z0 mid falls {p['z0_mid_falls']} ({p['mid_down_s']['z0']:.1f} s) | LOES mid falls {p['best_mid_falls']} ({p['mid_down_s']['best']:.1f} s), down during z0's first: "
                  f"{100 * p['best_down_during']:.0f}% | z0 early falls {p['z0_early_falls']} | MPJPE {p['mpjpe_cm']['z0']:.1f} -> {p['mpjpe_cm']['best']:.1f}", flush=True)


def render():
    BlenderScene.render(OUT / "blender", samples=96, threads=32, scale=0.6, out="renders", env=ENV)


def pages():
    plan = json.load(open(paths.own_path(f"{OUT_REL}/plan.json"))); R = paths.own_path(f"{OUT_REL}/blender/renders")
    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(plan, key=lambda c: (group(plan[c]), -plan[c]["mid_down_s"]["z0"], -(plan[c]["mpjpe_cm"]["z0"] - plan[c]["mpjpe_cm"]["best"])))
    lines, GN = [], ("z0 falls mid-motion, LOES stays up", "z0 falls mid-motion, LOES down too", "no mid-motion fall of z0")
    for k in range(0, len(order), ROWS):
        sheet = ContactSheet(COLUMNS, col_width=640, label_h=36, font_size=20, quality=88)
        for i, c in enumerate(order[k:k + ROWS]):
            p = plan[c]; num = k + i + 1
            sheet.add_row(f"#{num} {c}: {GN[group(p)]} | z0 mid-motion down {p['mid_down_s']['z0']:.1f} s, LOES {p['mid_down_s']['best']:.1f} s | MPJPE {p['mpjpe_cm']['z0']:.1f} -> "
                          f"{p['mpjpe_cm']['best']:.1f} cm | frames {p['start']}-{p['end']}", [R / f"{TAG}_{c}_{s}.png" for s in ("adult", "z0", "zstar")])
            lines.append(f"| {num} | `{c}` | {GN[group(p)]} | {p['n'] / 30:.1f} | {p['z0_mid_falls']} | {p['best_mid_falls']} | {p['mid_down_s']['z0']:.1f} | {p['mid_down_s']['best']:.1f} | "
                         f"{p['z0_early_falls']} | {p['mpjpe_cm']['z0']:.1f} | {p['mpjpe_cm']['best']:.1f} | {'pass' if p['success']['z0'] else 'fail'} | {'pass' if p['success']['best'] else 'fail'} | "
                         f"{p['start']}-{p['end']} ({p['window']}) | overview_{k // ROWS + 1}.jpg |")
        im = sheet.save(OUT / f"overview_{k // ROWS + 1}.jpg"); print("page", k // ROWS + 1, im.size, flush=True)
    write_index_md(OUT / "index.md", f"{BODY}, all 52 task motions: Blender strobe previews",
                   "Seed 0, LOES's best after 2,048 rollouts. Down: the pelvis below half the target's "
                   "pelvis height at the same frame. Mid-motion fall: such a stretch starting after 1.5 s (frame 45) and lasting at least 0.3 s; early falls start "
                   "before 1.5 s. Success: PHC 0.5 m. Frames are the survey alignment (reference from its second frame).",
                   ["#", "clip", "group", "length s", "z0 mid-motion falls (frames)", "LOES mid-motion falls", "z0 mid down s", "LOES mid down s", "z0 early falls",
                    "MPJPE z0 cm", "MPJPE LOES cm", "z0 success", "LOES success", "frames shown", "page"], lines)


if __name__ == "__main__":
    steps = set(sys.argv[2:]) or {"export", "render", "pages"}
    if "export" in steps:
        export()
    if "render" in steps:
        render()
    if "pages" in steps:
        pages()
