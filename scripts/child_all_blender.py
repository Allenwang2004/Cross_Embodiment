#!/usr/bin/env python3
"""child_all_blender.py -- Blender strobe previews of all 52 child task motions with four panels (adult reference, z0, CEM (BFM-Zero),
LOES), for the user to choose child examples where LOES does better than CEM (user, 11/10: in the chosen child clips BFM-Zero's CEM
seemed to do better).

Runs, seed 0 (the figures' seed): LOES = outputs/child_52_rerun/<clip>_es_s0 (our cost, best after 1,024 rollouts), z0 = its origin_z rollout,
CEM (BFM-Zero) = outputs/child_52_cem_phc/<clip>_cem_s0 (BFM-Zero's tracking reward, best after 1,024). Trajectories in the survey
alignment (references from their second frame), made by scripts/strobe_final_blender.py's trajectories() / add_cem().
Frames: 6 poses; where z0 falls mid-motion (pelvis below half the child target's, starting after frame 45, at least 9 frames), 15 apart
from 40 frames before that fall, as scripts/body_all_blender.py; otherwise evenly over the clip. One fixed view (azimuth 140), no heading
alignment. Preview quality (96 samples, 0.6 x resolution).
Order: by CEM's MPJPE minus LOES's (seed 0), largest first (where LOES beats CEM most). Metrics (seed 0 and the 5-seed means) from
outputs/child_52_rerun/per_run.npz (z0, LOES) and outputs/child_52_cem_phc/per_run.npz (CEM), q[t] against ref[t + 1].
Out: outputs/child_all52/{overview_<k>.jpg (9 motions a page), index.md, renders/, trajectories/, scene.json}.
usage: uv run python scripts/child_all_blender.py [export] [render] [pages]   (none: all three); --threads N, --labels zstar,... for render
11/10: LOES first drawn from outputs/child_52 (the older runs, scored one frame late) by mistake; re-made from outputs/child_52_rerun."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts"))
import strobe_final_blender as F                                         # noqa: E402

OUT = REPO / "outputs/child_all52"; N, STEP, BEFORE, START, MINLEN, ROWS = 6, 15, 40, 45, 9, 9
THREADS = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 32
LABELS = (("adult", "reference (adult)"), ("z0", "z0"), ("cem", "CEM (BFM-Zero)"), ("zstar", "LOES (ours)"))


def metrics():
    a = np.load(REPO / "outputs/child_52_rerun/per_run.npz", allow_pickle=True); b = np.load(REPO / "outputs/child_52_cem_phc/per_run.npz", allow_pickle=True)
    assert (a["clips"] == b["clips"]).all(); K = list(a["keys"]); M = {}
    for j, c in enumerate(a["clips"]):
        st = str(c).split("/")[1]; v = lambda x: dict(zip(K, [float(y) for y in x]))
        M[st] = dict(z0=v(a["z0"][j]), loes_s0=v(a["best_es"][j, 0, -1]), cem_s0=v(b["best_cem"][j, 0, -1]),
                     loes_mean=v(a["best_es"][j, :, -1].mean(0)), cem_mean=v(b["best_cem"][j, :, -1].mean(0)))
    return M


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


def window(d):
    ref = np.load(d / "reference_body.npz")["qpos"][:, 2]; n = len(ref)
    mid = {k: [(a, b) for a, b in stretches(np.load(d / f"{f}.npz")["qpos"][:, 2] < 0.5 * ref) if a >= START and b - a >= MINLEN] for k, f in (("z0", "z0"), ("cem", "cem"), ("loes", "best"))}
    if mid["z0"]:
        a, b = mid["z0"][0]; s = int(np.clip(a - BEFORE, 1, n - 1 - STEP * (N - 1))); return (s, s + STEP * (N - 1), STEP), mid, f"z0's fall at frames {a}-{b}"
    iv = (n - 2) // (N - 1); return (1, 1 + iv * (N - 1), iv), mid, "whole clip"


def export():
    clips = [l.strip().split("/")[1] for l in open(REPO / "outputs/child_52/clips.txt")]
    F.OUT = OUT; F.CEM = {f"child_{c}": f"outputs/child_52_cem_phc/{c}_cem_s0" for c in clips}; plan = {}
    F.EX = []
    for c in clips:                                                       # trajectories first: the window depends on them
        name = f"child_{c}"; d = F.trajectories(name, "child", c, "child_52"); F.add_cem(name); fr, mid, why = window(d)
        F.EX.append((name, "child", c, "child_52", fr, "fixed", {})); plan[c] = dict(frames=fr, why=why, mid_falls=mid)
    F.OUT.mkdir(exist_ok=True); (OUT / "scene.json").write_text(json.dumps(dict(panels={})))
    np.savez_compressed(OUT / "meshes.npz", _=np.zeros(1)); F.export(only=[e[0] for e in F.EX])       # into the (empty) scene, from the trajectories made above
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1)); print("exported", len(F.EX), "motions", flush=True)


def render():
    sel = []                                                              # --labels zstar,...: only these panels (e.g. after re-packaging LOES)
    if "--labels" in sys.argv:
        keep = sys.argv[sys.argv.index("--labels") + 1].split(","); sel = sum([["--panel", k] for k in json.load(open(OUT / "scene.json"))["panels"] if k.rsplit("_", 1)[1] in keep], [])
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", "96", "--scale", "0.6",
                    "--device", "CPU", "--threads", str(THREADS), "--out", "renders"] + sel, cwd=OUT, env=dict(os.environ, OMP_NUM_THREADS="1"), check=True,
                   stdout=open(OUT / "render.log", "w"), stderr=subprocess.STDOUT)


def pages():
    from PIL import Image, ImageDraw, ImageFont
    Fn = ImageFont.truetype("DejaVuSans.ttf", 20); Wc, L = 480, 36; plan = json.load(open(OUT / "plan.json")); M = metrics()
    order = sorted(plan, key=lambda c: -(M[c]["cem_s0"]["mpjpe"] - M[c]["loes_s0"]["mpjpe"])); lines = []
    for k in range(0, len(order), ROWS):
        rows = []
        for c in order[k:k + ROWS]:
            ims = [Image.open(OUT / "renders" / f"child_{c}_{s}.png").convert("RGB") for s, _ in LABELS]
            rows.append([im.resize((Wc, round(Wc * im.height / im.width))) for im in ims])
        sheet = Image.new("RGB", (len(LABELS) * Wc, sum(max(im.height for im in r) for r in rows) + L * (len(rows) + 1)), "white"); dr = ImageDraw.Draw(sheet); y = L
        for j, (_, lab) in enumerate(LABELS):
            dr.text((j * Wc + 10, 8), lab, fill=(0, 0, 0), font=Fn)
        for i, (c, row) in enumerate(zip(order[k:k + ROWS], rows)):
            m, p, num = M[c], plan[c], k + i + 1; a, b, st = p["frames"]
            ok = lambda x: "pass" if x["success"] >= 0.5 else "fail"
            dr.text((10, y + 6), f"#{num} {c} | MPJPE z0 {m['z0']['mpjpe']:.1f}, CEM {m['cem_s0']['mpjpe']:.1f}, LOES {m['loes_s0']['mpjpe']:.1f} cm | PHC z0 {ok(m['z0'])}, "
                                 f"CEM {ok(m['cem_s0'])}, LOES {ok(m['loes_s0'])} | frames {a}-{b}", fill=(0, 0, 0), font=Fn)
            for j, im in enumerate(row):
                sheet.paste(im, (j * Wc, y + L))
            y += L + max(im.height for im in row)
            lines.append(f"| {num} | `{c}` | {m['z0']['mpjpe']:.2f} | {m['cem_s0']['mpjpe']:.2f} | {m['loes_s0']['mpjpe']:.2f} | {m['cem_mean']['mpjpe']:.2f} | "
                         f"{m['loes_mean']['mpjpe']:.2f} | {ok(m['z0'])} | {ok(m['cem_s0'])} | {ok(m['loes_s0'])} | {m['z0']['fall_s']:.2f} | {m['cem_s0']['fall_s']:.2f} | "
                         f"{m['loes_s0']['fall_s']:.2f} | {a}-{b} ({p['why']}) | overview_{k // ROWS + 1}.jpg |")
        sheet.save(OUT / f"overview_{k // ROWS + 1}.jpg", quality=88); print("page", k // ROWS + 1, sheet.size, flush=True)
    (OUT / "index.md").write_text(
        "# Child, all 52 task motions: z0, CEM (BFM-Zero), LOES\n\nSeed 0 (the figures): LOES's and CEM's best latents after 1,024 rollouts "
        "(outputs/child_52_rerun/<clip>_es_s0, outputs/child_52_cem_phc/<clip>_cem_s0); 5-seed means alongside. Ordered by CEM's MPJPE minus LOES's (seed 0), "
        "largest first. PHC success: mean body error never above 0.5 m adult-equivalent. Fall: seconds with a non-foot body on the floor while the "
        "reference's is not. Frames: survey alignment (references from their second frame).\n\n"
        "| # | clip | MPJPE z0 | MPJPE CEM s0 | MPJPE LOES s0 | MPJPE CEM 5-seed | MPJPE LOES 5-seed | z0 PHC | CEM PHC | LOES PHC | fall z0 s | fall CEM s | "
        "fall LOES s | frames shown | page |\n|" + "---|" * 15 + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    steps = {x for x in sys.argv[1:] if x in ("export", "render", "pages")} or {"export", "render", "pages"}
    for s in ("export", "render", "pages"):
        if s in steps:
            globals()[s]()
