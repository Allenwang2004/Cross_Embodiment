#!/usr/bin/env python3
"""child_all_blender.py -- Blender strobe previews of all 52 child task motions with four panels (adult reference, z0, CEM (BFM-Zero),
LOES), for the user to choose child examples where LOES does better than CEM (user, 11/10: in the chosen child clips BFM-Zero's CEM
seemed to do better).

Runs, seed 0 (the figures' seed): LOES = outputs/child_52_rerun/<clip>_es_s0 (our cost, best after 1,024 rollouts), z0 = its origin_z rollout,
CEM (BFM-Zero) = outputs/child_52_cem_phc/<clip>_cem_s0 (BFM-Zero's tracking reward, best after 1,024). Trajectories in the survey
alignment (references from their second frame), packaged by crossembodiment.clips.ClipBundle (as scripts/strobe_final_blender.py's
"child_52" source + its CEM panel).
Frames: 6 poses; where z0 falls mid-motion (pelvis below half the child target's, starting after frame 45, at least 9 frames), 15 apart
from 40 frames before that fall, as scripts/body_all_blender.py; otherwise evenly over the clip (crossembodiment.strobe around_fall /
whole_clip). One fixed view (azimuth 140), no heading alignment. Preview quality (96 samples, 0.6 x resolution).
Order: by CEM's MPJPE minus LOES's (seed 0), largest first (where LOES beats CEM most). Metrics (seed 0 and the 5-seed means) from
outputs/child_52_rerun/per_run.npz (z0, LOES) and outputs/child_52_cem_phc/per_run.npz (CEM), q[t] against ref[t + 1].
Out: outputs/child_all52/{overview_<k>.jpg (9 motions a page), index.md, renders/, trajectories/, scene.json} (crossembodiment.sheet
ContactSheet / write_index_md for the pages).
usage: uv run python scripts/child_all_blender.py [export] [render] [pages]   (none: all three); --threads N, --labels zstar,... for render
11/10: LOES first drawn from outputs/child_52 (the older runs, scored one frame late) by mistake; re-made from outputs/child_52_rerun."""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(1, str(REPO))
import strobe_final_blender as F                                         # noqa: E402
from crossembodiment import paths                                        # noqa: E402
from crossembodiment.blender import BlenderScene                         # noqa: E402
from crossembodiment.clips import CHILD_TARGET, ClipBundle, task_of      # noqa: E402
from crossembodiment.palette import Palette                              # noqa: E402
from crossembodiment.sheet import ContactSheet, write_index_md           # noqa: E402
from crossembodiment.strobe import around_fall, mid_falls, whole_clip   # noqa: E402

OUT_REL = "outputs/child_all52"; OUT = paths.write_path(OUT_REL); N, STEP, BEFORE, START, MINLEN, ROWS = 6, 15, 40, 45, 9, 9
THREADS = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 32
LABELS = (("adult", Palette.label("adult", "pages")), ("z0", Palette.label("frozen", "plain")), ("cem", Palette.label("cem_phc", "pages")),
          ("zstar", Palette.label("loes", "pages")))


def metrics():
    a = np.load(paths.read_path("outputs/child_52_rerun/per_run.npz"), allow_pickle=True); b = np.load(paths.read_path("outputs/child_52_cem_phc/per_run.npz"), allow_pickle=True)
    assert (a["clips"] == b["clips"]).all(); K = list(a["keys"]); M = {}
    for j, c in enumerate(a["clips"]):
        st = str(c).split("/")[1]; v = lambda x: dict(zip(K, [float(y) for y in x]))
        M[st] = dict(z0=v(a["z0"][j]), loes_s0=v(a["best_es"][j, 0, -1]), cem_s0=v(b["best_cem"][j, 0, -1]),
                     loes_mean=v(a["best_es"][j, :, -1].mean(0)), cem_mean=v(b["best_cem"][j, :, -1].mean(0)))
    return M


def window(d):
    """the frames shown (from the saved float32 trajectories): around z0's first mid-motion fall, else the whole clip"""
    ref = np.load(d / "reference_body.npz")["qpos"][:, 2]; n = len(ref)
    mid = {k: mid_falls(np.load(d / f"{f}.npz")["qpos"][:, 2], ref, START, MINLEN) for k, f in (("z0", "z0"), ("cem", "cem"), ("loes", "best"))}
    if mid["z0"]:
        a, b = mid["z0"][0]; return around_fall(mid["z0"][0], n, N, STEP, BEFORE), mid, f"z0's fall at frames {a}-{b}"
    return whole_clip(n, N), mid, "whole clip"


def export():
    clips = [c.split("/")[1] for c in paths.read_clips(paths.CHILD_52_TASK)]   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    plan = {}; sc = BlenderScene(meshes={"_": np.zeros(1)}); todo = []
    for c in clips:                                                       # trajectories first: the window depends on them
        name = f"child_{c}"; d = OUT / "trajectories" / name
        ClipBundle.from_run(c, "child", run=f"outputs/child_52_rerun/{c}_es_s0", reference_body=f"{CHILD_TARGET}/{task_of(c)}/{c}.npz"
                            ).add_cem(f"outputs/child_52_cem_phc/{c}_cem_s0").save(d)
        fr, mid, why = window(d); todo.append((name, c, fr, d)); plan[c] = dict(frames=fr, why=why, mid_falls=mid)
    for name, c, fr, d in todo:                                           # into one (empty) scene, from the trajectories made above
        F.example_panels(sc, name, "child", c, fr, "fixed", {}, d)
    sc.write(OUT); (OUT / "plan.json").write_text(json.dumps(plan, indent=1)); print("exported", len(todo), "motions", flush=True)


def render():
    sel = None                                                            # --labels zstar,...: only these panels (e.g. after re-packaging LOES)
    if "--labels" in sys.argv:
        keep = sys.argv[sys.argv.index("--labels") + 1].split(","); sel = [k for k in json.load(open(OUT / "scene.json"))["panels"] if k.rsplit("_", 1)[1] in keep]
    BlenderScene.render(OUT, samples=96, threads=THREADS, panels=sel, scale=0.6, out="renders")


def pages():
    plan = json.load(open(paths.own_path(f"{OUT_REL}/plan.json"))); M = metrics(); R = paths.own_path(f"{OUT_REL}/renders")
    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(plan, key=lambda c: -(M[c]["cem_s0"]["mpjpe"] - M[c]["loes_s0"]["mpjpe"])); lines = []
    ok = lambda x: "pass" if x["success"] >= 0.5 else "fail"
    for k in range(0, len(order), ROWS):
        sheet = ContactSheet([lab for _, lab in LABELS], col_width=480, label_h=36, font_size=20, row_height="max", quality=88)
        for i, c in enumerate(order[k:k + ROWS]):
            m, p, num = M[c], plan[c], k + i + 1; a, b, st = p["frames"]
            sheet.add_row(f"#{num} {c} | MPJPE z0 {m['z0']['mpjpe']:.1f}, CEM {m['cem_s0']['mpjpe']:.1f}, LOES {m['loes_s0']['mpjpe']:.1f} cm | PHC z0 {ok(m['z0'])}, "
                          f"CEM {ok(m['cem_s0'])}, LOES {ok(m['loes_s0'])} | frames {a}-{b}", [R / f"child_{c}_{s}.png" for s, _ in LABELS])
            lines.append(f"| {num} | `{c}` | {m['z0']['mpjpe']:.2f} | {m['cem_s0']['mpjpe']:.2f} | {m['loes_s0']['mpjpe']:.2f} | {m['cem_mean']['mpjpe']:.2f} | "
                         f"{m['loes_mean']['mpjpe']:.2f} | {ok(m['z0'])} | {ok(m['cem_s0'])} | {ok(m['loes_s0'])} | {m['z0']['fall_s']:.2f} | {m['cem_s0']['fall_s']:.2f} | "
                         f"{m['loes_s0']['fall_s']:.2f} | {a}-{b} ({p['why']}) | overview_{k // ROWS + 1}.jpg |")
        im = sheet.save(OUT / f"overview_{k // ROWS + 1}.jpg"); print("page", k // ROWS + 1, im.size, flush=True)
    write_index_md(OUT / "index.md", "Child, all 52 task motions: z0, CEM (BFM-Zero), LOES",
                   "Seed 0 (the figures): LOES's and CEM's best latents after 1,024 rollouts "
                   "(outputs/child_52_rerun/<clip>_es_s0, outputs/child_52_cem_phc/<clip>_cem_s0); 5-seed means alongside. Ordered by CEM's MPJPE minus LOES's (seed 0), "
                   "largest first. PHC success: mean body error never above 0.5 m adult-equivalent. Fall: seconds with a non-foot body on the floor while the "
                   "reference's is not. Frames: survey alignment (references from their second frame).",
                   ["#", "clip", "MPJPE z0", "MPJPE CEM s0", "MPJPE LOES s0", "MPJPE CEM 5-seed", "MPJPE LOES 5-seed", "z0 PHC", "CEM PHC", "LOES PHC",
                    "fall z0 s", "fall CEM s", "fall LOES s", "frames shown", "page"], lines)


if __name__ == "__main__":
    steps = {x for x in sys.argv[1:] if x in ("export", "render", "pages")} or {"export", "render", "pages"}
    for s in ("export", "render", "pages"):
        if s in steps:
            globals()[s]()
