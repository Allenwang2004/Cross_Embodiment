#!/usr/bin/env python3
"""strobe_final_blender.py -- the strobe figures the user kept (11/10), rendered in Blender at final quality into
outputs/strobe_task52_blender: per example four panels (adult reference, z0, CEM (BFM-Zero)'s best, LOES's best) and the four side by side.

Examples, frames and views are the ones the user chose from the previews (payload, limited leg, G1; payload rotate-y-5-0.8_7 is the
original strobe folder's window on the user survey's clip, an older search and not one of the 52 task motions) plus the child's headstand_4 and
move-ego-0-4_9 (the child 52-motion run, seed 0, LOES's best after 1,024 rollouts). Runs: payload / limited leg / G1 from the 52-motion
runs of 11/09 (seed 0, LOES's best after 2,048 rollouts); child from outputs/child_52_rerun (the corrected-alignment runs of the Task page; outputs/child_52 holds the older runs, used by mistake until 11/10). Trajectories are in the survey's frame
alignment (references from their second frame; a rollout's frame t against the reference's frame t + 1), copied or packaged into
trajectories/<example>/.
View: "fixed" = one world view for every panel (azimuth 140, the survey videos'); "path" = side-on to each panel's own path
(path heading + 90 deg). split-1_7 (limited leg): z0's and LOES's poses are each turned about the root to the reference's heading at
that frame (LOES had turned about 90 deg); likewise move-ego-low-90-2_9 (the reference turns left about a full circle, z0 and LOES
turn right). The user labels both in the figure.
Look: scripts/render_blender.py (Cycles, 256 samples): one body colour (the G1 black), payload weight amber, the limited leg's locked right leg red,
poses at one depth fading from light (early) to full colour (late).
The payload and limited-leg sources (outputs/payload_all52, outputs/limit_leg_all52) were deleted on 11/10 after the export; their copies are in
outputs/strobe_task52_blender/trajectories, and scene.json + meshes.npz there re-render everything with render_blender.py.
usage: uv run python scripts/strobe_final_blender.py [export] [render] [combine]   (none: all three)
       uv run python scripts/strobe_final_blender.py update <example> ...   (re-make single examples in place)
       uv run python scripts/strobe_final_blender.py cem [<example> ...]      (add the CEM panel to examples already made)"""
import glob, json, math, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import export_blender_package as E                                        # noqa: E402  (pose, load, meshes)
sys.path.insert(1, str(REPO))
from crossenbodiment.geometry import root_yaw as yaw                      # noqa: E402  root quaternion yaw (refactor step 3)

OUT = REPO / "outputs/strobe_task52_blender"; S = REPO / "outputs/user_survey"; AZ = 125.0; THREADS = int(os.environ.get("STROBE_THREADS", 32))
MODEL = dict(payload=S / "models/heavy_arm.xml", limit_leg=S / "models/limit_leg.xml", g1=S / "models/g1/scene.xml", child=S / "models/child.xml")
EX = [  # name, body, clip, trajectory source (a survey-format clip folder, or "child_52"), frames (start, end, step), view, extra
    ("payload_move-ego-0-2-raisearms-h-m_1", "payload", "move-ego-0-2-raisearms-h-m_1", "outputs/payload_all52/trajectories", (120, 170, 10), "fixed", {}),   # z0 fully down (pelvis and head below half the reference's) at frames 118-177
    ("payload_rotate-y-5-0.8_7", "payload", "rotate-y-5-0.8_7", "outputs/user_survey/heavy_arm/trajectories", (150, 192, 7), "fixed", dict(align=True)),   # the original strobe folder's window (strobe_figures); z0 falls mid-motion (pelvis below half 185-195), LOES stays up. Not one of the 52 task motions: the older user-survey search
    ("g1_rotate-z--5-0.8_6", "g1", "rotate-z--5-0.8_6", "outputs/user_survey/g1_task52/trajectories", (30, 130, 20), "fixed", {}),
    ("g1_jump-2_3", "g1", "jump-2_3", "outputs/user_survey/g1_task52/trajectories", (2, 102, 20), "path", {}),
    ("limit_leg_move-ego-low-90-2_9", "limit_leg", "move-ego-low-90-2_9", "outputs/limit_leg_all52/trajectories", (1, 296, 59), "fixed", dict(align=True)),   # the reference turns about a full circle to the left, z0 and LOES to the right
    ("limit_leg_move-ego-0-2_2", "limit_leg", "move-ego-0-2_2", "outputs/user_survey/limit_leg/trajectories", (37, 72, 7), "path", {}),   # replaces split-1_7 (user, 11/10): the user survey's clip (older search, not one of the 52), the original strobe folder's window; z0 stoops forward, LOES upright
    ("child_headstand_4", "child", "headstand_4", "child_52", (45, 90, 9), "fixed", {}),   # LOES holds both legs up from frame 45 (its left foot comes down at 25-40); z0s left foot is down at 30-90
    ("child_move-ego-0-4_9", "child", "move-ego-0-4_9", "child_52", (200, 290, 18), "path", {}),       # z0's hands down from frame 235
    # user, 11/11: a child example where ours looks clearly closer to the reference than BFM-Zero (headstand_4 has none). Frames: the 6
    # (at least 8 apart) where BFM-Zero's error minus ours is largest, the error being the mean body distance to the child reference
    # with the centre of mass's x, y removed and the world heading kept (as this figure draws a pose); BFM-Zero 38.0 cm, ours 6.8 cm
    # on them. One world view side-on to the reference's path over those frames (heading -92 deg), so a wrong heading shows.
    ("child_move-ego-90-4_5", "child", "move-ego-90-4_5", "child_52", None, "fixed", dict(frames=[137, 150, 163, 175, 183, 191], az=-2.4)),
    # user, 11/11: headstand with seed 2 of both ours and BFM-Zero (the one seed where ours is closer on most frames: 77 of 119), frames
    # chosen as above (BFM-Zero 5.5 cm, ours 3.6 cm on them; seed 0, the figure's, has no frames where ours is closer)
    ("child_headstand_4_s2", "child", "headstand_4", "child_52_s2", None, "fixed", dict(frames=[22, 41, 56, 72, 83, 111])),
]
FILES = (("adult", "reference_adult.npz"), ("z0", "z0.npz"), ("cem", "cem.npz"), ("zstar", "best.npz"))
# CEM (BFM-Zero): BFM-Zero's tracking reward (PHC), seed 0, 2,048 rollouts (the child's: 1,024, as its LOES) (user, 11/10: add CEM to the figures).
# The child's are the 52-motion CEM runs (1,024 rollouts); the others were run for these figures (outputs/train_logs/queue/run_strobe_cem.sh).
CEM = {"payload_move-ego-0-2-raisearms-h-m_1": "outputs/body_52/payload/move-ego-0-2-raisearms-h-m_1_cem_s0",
       "payload_rotate-y-5-0.8_7": "outputs/user_survey_search/heavy_arm/rotate-y-5-0.8_7_cem_g128_s0",     # 2,048 rollouts (user), start at rest as its LOES run (8,192)
       "g1_rotate-z--5-0.8_6": "outputs/g1_52/rotate-z--5-0.8_6_cem_s0", "g1_jump-2_3": "outputs/g1_52/jump-2_3_cem_s0",
       "limit_leg_move-ego-low-90-2_9": "outputs/body_52/limit_leg/move-ego-low-90-2_9_cem_s0", "limit_leg_move-ego-0-2_2": "outputs/user_survey_search/limit_leg/move-ego-0-2_2_cem_g128_s0",
       "child_headstand_4": "outputs/child_52_cem_phc/headstand_4_cem_s0", "child_move-ego-0-4_9": "outputs/child_52_cem_phc/move-ego-0-4_9_cem_s0",
       "child_move-ego-90-4_5": "outputs/child_52_cem_phc/move-ego-90-4_5_cem_s0", "child_headstand_4_s2": "outputs/child_52_cem_phc/headstand_4_cem_s2"}


def add_cem(name):
    """trajectories/<name>/cem.npz: the CEM run's best rollout, cut to the reference's length (as z0 and best)."""
    d = OUT / "trajectories" / name; n = len(np.load(d / "reference_body.npz")["qpos"]); q = np.load(REPO / CEM[name] / "best.npz")["qpos"]
    assert len(q) >= n, (name, len(q), n); np.savez_compressed(d / "cem.npz", qpos=q[:n].astype(np.float32), fps=np.float32(30))
    v = json.loads((d / "video.json").read_text()); v.setdefault("frames", {})["cem"] = n; v["cem_run"] = CEM[name]; (d / "video.json").write_text(json.dumps(v, indent=1))


def trajectories(name, body, clip, src):
    d = OUT / "trajectories" / name; d.mkdir(parents=True, exist_ok=True)
    if src not in ("child_52", "c540") and not src.startswith(("body_52/", "child_52_s")):
        for f in ("reference_adult.npz", "reference_body.npz", "z0.npz", "best.npz", "video.json"):
            if (REPO / src / clip / f).exists():
                shutil.copy(REPO / src / clip / f, d / f)
        return d
    t = clip.rsplit("_", 1)[0]; target = {"payload": "data/origin_motion", "limit_leg": "data/adult_lockRleg_bent_lim/retargeting_motion"}.get(body, "data/child/retargeting_motion")
    q = dict(reference_adult=np.load(glob.glob(str(REPO / f"data/origin_motion/{t}/{clip}.npz"))[0])["qpos"][1:],
             reference_body=np.load(REPO / f"{target}/{t}/{clip}.npz")["qpos"][1:]); n = len(q["reference_body"])
    run = REPO / {"child_52": f"outputs/child_52_rerun/{clip}_es_s0", "c540": f"outputs/c540_bfmglobal_a03/{clip}"}.get(src, f"outputs/{src}/{clip}_es_s0")
    if src.startswith("child_52_s"):                                   # child_52_s<k>: the child 52-motion run of seed k
        run = REPO / f"outputs/child_52_rerun/{clip}_es_s{src[len('child_52_s'):]}"   # c540: the older child search behind the survey's child videos; body_52/<body>: the 52-motion runs
    q.update(z0=np.load(run / "origin_z.npz")["qpos"][:n], best=np.load(run / "best.npz")["qpos"][:n])
    for k, v in q.items():
        np.savez_compressed(d / f"{k}.npz", qpos=v.astype(np.float32), fps=np.float32(30))
    (d / "video.json").write_text(json.dumps(dict(clip=clip, body=body, fps=30, frames={k: len(v) for k, v in q.items()}, search_run=str(run.relative_to(REPO))), indent=1))
    return d


def export(only=None):
    """only: rebuild just these examples into the existing scene.json, from the trajectories already in OUT/trajectories."""
    if OUT.exists() and only is None:
        raise SystemExit(f"{OUT} exists: move it away first, or update single examples")
    P = json.load(open(OUT / "scene.json"))["panels"] if only else {}
    for name, body, clip, src, rng, view, x in EX:
        if only and name not in only:
            continue
        d = OUT / "trajectories" / name if only else trajectories(name, body, clip, src)
        if not only and name in CEM:
            add_cem(name)
        if "frames" in x:                                                 # an explicit frame list
            frames = list(x["frames"])
        else:
            a, b, st = rng; frames = list(range(a, b + 1, st)); assert frames[-1] == b
        n = len(frames)
        crawl = "crawl" in clip; dx = x.get("dx", 0.8 if crawl else 0.6)                 # a crawl lies along the floor: a lower, wider frame
        tints = [0.62 * (1 - i / (n - 1)) for i in range(n)]; W = (n - 1) * dx + (2.4 if crawl else 1.6) + x.get("wider", 0.0); H = x.get("H", 1.4 if crawl else 2.45); pxw = x.get("px", 1500)
        ya = yaw(np.load(d / "reference_adult.npz")["qpos"].astype(np.float64))
        for label, f in FILES:
            if not (d / f).exists():
                print("  no", f, "yet: panel", label, "skipped", flush=True); continue
            xml = S / "models/adult.xml" if label == "adult" else MODEL[body]; m = E.load(xml, payload=(body == "payload" and label != "adult"))
            q = np.load(d / f)["qpos"].astype(np.float64); assert q.shape[1] == m.nq, (name, label, q.shape, m.nq)
            y = yaw(q); poses = []
            for i, (t, k) in enumerate(zip(frames, tints)):
                if view == "fixed":
                    h = math.radians(x.get("az", 140.0) - AZ)                # world azimuth of the camera (default 140, the survey view)
                else:
                    v = q[frames[-1], :2] - q[frames[0], :2]; h = math.atan2(v[1], v[0]) + math.radians(90.0 - AZ)
                if x.get("align") and label != "adult":
                    h += float(y[t] - ya[t])                                 # turned to the reference's heading at this frame
                red = body == "limit_leg" and label != "adult"; tag = "g1" if (body == "g1" and label != "adult") else "human"
                ps = E.pose(m, q, t, 0.0, k, h, red, tag)
                if tag == "g1":                                            # the G1 drawn black (user, 11/10)
                    for g in ps["geoms"]:
                        g["color"] = "black"
                poses.append(dict(ps, shift=(i - (n - 1) / 2) * dx + x.get("nudge", 0.0)))
            P[f"{name}_{label}"] = dict(figure="strobe", width_m=W, height_m=H, feet_m=x.get("feet", 0.3), px=[pxw, round(pxw * H / W)], white=True, poses=poses, frames=frames, view=view)
            if "elev" in x:
                P[f"{name}_{label}"]["elevation"] = x["elev"]
        print(name, frames, view, "aligned" if x.get("align") else "", flush=True)
    meshes = dict(np.load(OUT / "meshes.npz")) if only else {}
    meshes.update(E.meshes); (OUT / "scene.json").write_text(json.dumps(dict(panels=P))); np.savez_compressed(OUT / "meshes.npz", **meshes)
    shutil.copy(REPO / "scripts/render_blender.py", OUT / "render_blender.py")


def render(only=None, labels=None):
    env = dict(os.environ, OMP_NUM_THREADS="1"); sel = sum([["--panel", f"{n}_{k}"] for n in only for k, _ in FILES if labels is None or k in labels], []) if only else []
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", "256", "--device", "CPU",
                    "--threads", str(THREADS), "--out", "panels"] + sel, cwd=OUT, env=env, check=True, stdout=open(OUT / "render.log", "w"), stderr=subprocess.STDOUT)


def combine(gap=24, only=None):
    from PIL import Image
    (OUT / "combined").mkdir(exist_ok=True)
    for name, *_ in EX:
        if only and name not in only:
            continue
        ims = [Image.open(OUT / "panels" / f"{name}_{k}.png").convert("RGB") for k, _ in FILES if (OUT / "panels" / f"{name}_{k}.png").exists()]
        out = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), max(im.height for im in ims)), "white"); x = 0
        for im in ims:
            out.paste(im, (x, 0)); x += im.width + gap
        out.save(OUT / "combined" / f"{name}.png"); print(name, out.size, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["update"]:                      # update NAME ...: re-export, render and combine only these examples
        names = sys.argv[2:]; assert names and all(n in [e[0] for e in EX] for n in names), names
        export(names); render(names); combine(only=names); sys.exit()
    if sys.argv[1:2] == ["cem"]:                         # cem [NAME ...]: add the CEM panel (cem.npz, export, render the CEM panels only, combine)
        names = sys.argv[2:] or [e[0] for e in EX]; assert all(n in CEM for n in names), names
        for n in names:
            add_cem(n)
        export(names); render(names, labels=("cem",)); combine(only=names); sys.exit()
    steps = set(sys.argv[1:]) or {"export", "render", "combine"}
    for s in ("export", "render", "combine"):
        if s in steps:
            globals()[s]()
