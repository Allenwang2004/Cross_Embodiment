#!/usr/bin/env python3
"""body_all_blender.py -- Blender strobe panels (adult reference, z0, LOES's best) of all 52 task motions of one body, for the user to
choose examples from, with z0's falls DURING the motion picked out (user, 11/10: falls right at the start are not wanted).

Data: the 52-motion runs (outputs/body_52/<body>, seed 0, LOES's best after 2,048 rollouts), packaged in the survey format and frame
alignment -> outputs/<body>_all52/trajectories/<clip>. reference_body: the body's search target (limited leg: the motion with the right
leg bent; payload: the adult's motion).
Falls: the pelvis below half the target's pelvis height at the same frame. A mid-motion fall: such a stretch that starts after 1.5 s
(frame 45) and lasts at least 0.3 s (9 frames). Frames: 6 poses 15 frames apart from 40 frames before z0's first mid-motion fall;
without one, evenly over the clip. Order: z0 falls mid-motion and LOES is not down during it; z0 falls mid-motion and LOES is down
too; no mid-motion fall of z0 (by MPJPE drop). View: one fixed world view (azimuth 140, the survey videos') for every panel.
Preview quality (96 samples, 0.6 x resolution). Pages overview_<k>.jpg, 9 motions each; index.md lists all 52.
usage: uv run python scripts/body_all_blender.py <limit_leg|payload> [export] [render] [pages]   (none: all three)"""
import glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import export_strobe_blender as B                                         # noqa: E402

BODY = sys.argv[1]; assert BODY in ("limit_leg", "payload")
FOLDER = {"limit_leg": "limit_leg_task52", "payload": "heavy_arm_task52"}[BODY]; TAG = {"limit_leg": "limit_leg", "payload": "heavy_arm"}[BODY]
REF = {"limit_leg": "data/adult_lockRleg_bent_lim/retargeting_motion/{t}/{c}.npz", "payload": "data/origin_motion/{t}/{c}.npz"}[BODY]
RUN = REPO / "outputs/body_52" / BODY; OUT = REPO / f"outputs/{BODY}_all52"
CLIPS = sorted(json.load(open(RUN / "results.json"))["per_motion"]["best"]); N, STEP, BEFORE, START, MINLEN, ROWS = 6, 15, 40, 45, 9, 9
ENV = dict(os.environ, MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="2", MUJOCO_EGL_DEVICE_ID="2", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def stretches(below):
    out, t = [], 0
    while t < len(below):
        if below[t]:
            u = t
            while u < len(below) and below[u]:
                u += 1
            out.append((t, u)); t = u
        else:
            t += 1
    return out


def package(c):
    t = c.rsplit("_", 1)[0]
    q = dict(reference_adult=np.load(glob.glob(str(REPO / f"data/origin_motion/{t}/{c}.npz"))[0])["qpos"][1:], reference_body=np.load(REPO / REF.format(t=t, c=c))["qpos"][1:])
    n = len(q["reference_body"]); assert len(q["reference_adult"]) == n
    q.update(z0=np.load(RUN / f"{c}_es_s0/origin_z.npz")["qpos"][:n], best=np.load(RUN / f"{c}_es_s0/best.npz")["qpos"][:n])
    d = OUT / "trajectories" / c; d.mkdir(parents=True, exist_ok=True)
    for k, v in q.items():
        np.savez_compressed(d / f"{k}.npz", qpos=v.astype(np.float32), fps=np.float32(30))
    ref = q["reference_body"][:, 2]; below = {k: q[k][:, 2] < 0.5 * ref for k in ("z0", "best")}
    mid = {k: [(a, b) for a, b in stretches(below[k]) if a >= START and b - a >= MINLEN] for k in below}
    early = {k: [(a, b) for a, b in stretches(below[k]) if a < START] for k in below}
    R = json.load(open(RUN / "results.json"))["per_motion"]["best"][c]
    p = dict(n=n, z0_mid_falls=mid["z0"], best_mid_falls=mid["best"], z0_early_falls=early["z0"],
             mid_down_s={k: sum(b - a for a, b in mid[k]) / 30 for k in mid}, mpjpe_cm=dict(z0=round(R["frozen"]["mpjpe"], 2), best=round(R["es"]["mpjpe"], 2)),
             success=dict(z0=bool(R["frozen"]["success"]), best=bool(R["es"]["success"])), contact_fall_s=dict(z0=round(R["frozen"]["fall_s"], 2), best=round(R["es"]["fall_s"], 2)))
    if mid["z0"]:
        a, b = mid["z0"][0]; s = int(np.clip(a - BEFORE, 1, n - 1 - STEP * (N - 1))); p.update(start=s, end=s + STEP * (N - 1), interval=STEP, window=f"z0's fall at frames {a}-{b}")
        p["best_down_during"] = float(below["best"][a:b].mean())
    else:
        iv = (n - 2) // (N - 1); p.update(start=1, end=1 + iv * (N - 1), interval=iv, window="whole clip", best_down_during=None)
    tmpl = json.loads(next((REPO / "outputs/user_survey" / TAG / "trajectories").glob("*/video.json")).read_text())
    (d / "video.json").write_text(json.dumps(dict(clip=c, body=TAG, fps=30, frames={k: len(v) for k, v in q.items()}, search_run=str((RUN / f"{c}_es_s0").relative_to(REPO)),
                                                  falls=p, side_by_side=tmpl["side_by_side"], overlay=tmpl["overlay"]), indent=1))
    return p


def group(p):
    if p["z0_mid_falls"]:
        return 0 if p["best_down_during"] < 0.1 else 1
    return 2


def export():
    plan, panels = {}, {}
    for c in CLIPS:
        p = package(c); plan[c] = p
        cfg = OUT / "configs" / f"{TAG}_{c}.yaml"; cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(yaml.safe_dump(dict(strobe=dict(start_step=p["start"], end_step=p["end"], interval=p["interval"]), camera=dict(type="fixed", azimuth_offset=140.0))))
        panels.update(B.clip_panels(FOLDER, c, cfg_path=cfg, traj_dir=OUT / "trajectories" / c))
    d = OUT / "blender"; d.mkdir(exist_ok=True)
    (d / "scene.json").write_text(json.dumps(dict(panels=panels))); np.savez_compressed(d / "meshes.npz", **B.E.meshes)
    (d / "render_blender.py").write_bytes((REPO / "scripts/render_blender.py").read_bytes()); (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    g = [group(p) for p in plan.values()]
    print(f"{BODY}: {len(plan)} clips, z0 falls mid-motion on {g.count(0) + g.count(1)} (LOES not down during it: {g.count(0)}), panels {len(panels)}", flush=True)
    for c in sorted(plan, key=lambda c: (group(plan[c]), -plan[c]["mid_down_s"]["z0"])):
        p = plan[c]
        if group(p) < 2:
            print(f"  {c:30s} z0 mid falls {p['z0_mid_falls']} ({p['mid_down_s']['z0']:.1f} s) | LOES mid falls {p['best_mid_falls']} ({p['mid_down_s']['best']:.1f} s), down during z0's first: "
                  f"{100 * p['best_down_during']:.0f}% | z0 early falls {p['z0_early_falls']} | MPJPE {p['mpjpe_cm']['z0']:.1f} -> {p['mpjpe_cm']['best']:.1f}", flush=True)


def render():
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", "96", "--scale", "0.6",
                    "--device", "CPU", "--threads", "32", "--out", "renders"], cwd=OUT / "blender", env=ENV, check=True, stdout=open(OUT / "blender/render.log", "w"), stderr=subprocess.STDOUT)


def pages():
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype("DejaVuSans.ttf", 20); Wc, L = 640, 36; plan = json.load(open(OUT / "plan.json"))
    order = sorted(plan, key=lambda c: (group(plan[c]), -plan[c]["mid_down_s"]["z0"], -(plan[c]["mpjpe_cm"]["z0"] - plan[c]["mpjpe_cm"]["best"])))
    lines, GN = [], ("z0 falls mid-motion, LOES stays up", "z0 falls mid-motion, LOES down too", "no mid-motion fall of z0")
    for k in range(0, len(order), ROWS):
        rows = []
        for c in order[k:k + ROWS]:
            ims = [Image.open(OUT / "blender/renders" / f"{TAG}_{c}_{s}.png").convert("RGB") for s in ("adult", "z0", "zstar")]
            rows.append([im.resize((Wc, round(Wc * im.height / im.width))) for im in ims])
        sheet = Image.new("RGB", (3 * Wc, sum(r[0].height for r in rows) + L * (len(rows) + 1)), "white"); dr = ImageDraw.Draw(sheet); y = L
        for j, lab in enumerate(("reference (adult)", "z0", "LOES (best latent)")):
            dr.text((j * Wc + 10, 8), lab, fill=(0, 0, 0), font=F)
        for i, (c, row) in enumerate(zip(order[k:k + ROWS], rows)):
            p = plan[c]; num = k + i + 1
            dr.text((10, y + 6), f"#{num} {c}: {GN[group(p)]} | z0 mid-motion down {p['mid_down_s']['z0']:.1f} s, LOES {p['mid_down_s']['best']:.1f} s | MPJPE {p['mpjpe_cm']['z0']:.1f} -> "
                                 f"{p['mpjpe_cm']['best']:.1f} cm | frames {p['start']}-{p['end']}", fill=(0, 0, 0), font=F)
            for j, im in enumerate(row):
                sheet.paste(im, (j * Wc, y + L))
            y += L + row[0].height
            lines.append(f"| {num} | `{c}` | {GN[group(p)]} | {p['n'] / 30:.1f} | {p['z0_mid_falls']} | {p['best_mid_falls']} | {p['mid_down_s']['z0']:.1f} | {p['mid_down_s']['best']:.1f} | "
                         f"{p['z0_early_falls']} | {p['mpjpe_cm']['z0']:.1f} | {p['mpjpe_cm']['best']:.1f} | {'pass' if p['success']['z0'] else 'fail'} | {'pass' if p['success']['best'] else 'fail'} | "
                         f"{p['start']}-{p['end']} ({p['window']}) | overview_{k // ROWS + 1}.jpg |")
        sheet.save(OUT / f"overview_{k // ROWS + 1}.jpg", quality=88); print("page", k // ROWS + 1, sheet.size, flush=True)
    (OUT / "index.md").write_text(f"# {BODY}, all 52 task motions: Blender strobe previews\n\nSeed 0, LOES's best after 2,048 rollouts. Down: the pelvis below half the target's "
                                  "pelvis height at the same frame. Mid-motion fall: such a stretch starting after 1.5 s (frame 45) and lasting at least 0.3 s; early falls start "
                                  "before 1.5 s. Success: PHC 0.5 m. Frames are the survey alignment (reference from its second frame).\n\n"
                                  "| # | clip | group | length s | z0 mid-motion falls (frames) | LOES mid-motion falls | z0 mid down s | LOES mid down s | z0 early falls | "
                                  "MPJPE z0 cm | MPJPE LOES cm | z0 success | LOES success | frames shown | page |\n|" + "---|" * 15 + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    steps = set(sys.argv[2:]) or {"export", "render", "pages"}
    if "export" in steps:
        export()
    if "render" in steps:
        render()
    if "pages" in steps:
        pages()
