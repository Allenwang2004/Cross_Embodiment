"""Survey videos for the humanoid bodies: outputs/user_survey/<name>/side_by_side/<stem>.mp4 and .../overlay/<stem>.mp4.

side_by_side (render_panels, no captions): [the ADULT doing the original motion | the body with z0 | the body with the
searched z], drawn flat grey / red / blue -- the overlay videos' colours -- with one camera sized for the taller of the
adult and the body, so the two keep their true relative size.
overlay (body_overlay --clean): on the body, its reference as a translucent grey ghost (its retargeted motion for
child / fat / tall, the adult's original motion for heavy_arm / limit_leg), z0 red, searched z blue.
Searches: child from outputs/c540_bfmglobal_a03/<stem> (rotate-x--5-0.8_4 from its 256-step rerun); the others from
outputs/user_survey_search/<name>/<stem>{_s0, _g512_s0, _sig015_s0, ...}, whichever has the lowest best cost.
heavy_arm walks move-ego-0-2_4 (its 0-2_2 search never got the walk right) instead of move-ego-0-2_2.
Cameras: the child's (2.8 m / 0.55 m panels, 3.1 m / 0.5 m overlay) scaled by stature (child 1.26, adult 1.74, tall 2.07 m).
Contact sheets and specs go to outputs/survey_preview/.
usage: uv run python scripts/survey_body_videos.py <child|fat|tall|heavy_arm|limit_leg> [stem ...] [--only side_by_side|overlay]
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# the survey's data (clips, bodies, runs) and colours live in the package since refactor step 4 (crossenbodiment/survey.py,
# configs/survey.yaml, configs/clips/user_survey.txt, crossenbodiment/palette.py); imported back so this script and
# `from survey_body_videos import ...` keep working. REPO stays the main tree's absolute path, as before (renders and the
# run lookup use it); the package is found from this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossenbodiment import survey as _survey                                                    # noqa: E402
from crossenbodiment.palette import BLUE, GREY, RED                                              # noqa: E402,F401
from crossenbodiment.survey import ADULT, ADULT_H, BODY, CLIPS, SUFFIXES                         # noqa: E402,F401

REPO = Path("/home/allen19/crossenbodiment"); PREV = REPO / "outputs/survey_preview"
ENV = {**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2", "CUDA_VISIBLE_DEVICES": "2"}


def run_of(name, stem):
    return _survey.run_of(name, stem, root=REPO)


def render(name, stem, only):
    b = BODY[name]; task = stem.rsplit("_", 1)[0]; run = run_of(name, stem)
    root = REPO / "outputs/user_survey" / name
    sbs, ovl = root / "side_by_side" / f"{stem}.mp4", root / "overlay" / f"{stem}.mp4"
    k = max(b["h"], ADULT_H) / 1.26                                   # side by side: frame the taller of adult and body
    spec = dict(out=str(sbs.relative_to(REPO)), cols=3, size=1080, captions=False,
                track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)),
                panels=[dict(xml=ADULT, qpos=f"data/origin_motion/{task}/{stem}.npz", rgba=GREY),
                        dict(xml=b["xml"], qpos=f"{run}/origin_z.npz", rgba=RED),
                        dict(xml=b["xml"], qpos=f"{run}/best.npz", rgba=BLUE)])
    sp = PREV / f"spec_{name}_{stem}.json"; sp.write_text(json.dumps(spec, indent=1))
    ko = b["h"] / 1.26
    jobs = {"side_by_side": (["uv", "run", "python", "scripts/render_panels.py", str(sp)], sbs),
            "overlay": (["uv", "run", "python", "scripts/body_overlay.py", "--xml", b["xml"], "--clean", "--out", str(ovl),
                         "--traj", f"reference={b['ref']}/{task}/{stem}.npz", "--traj", f"z0={run}/origin_z.npz",
                         "--traj", f"search={run}/best.npz", "--distance", f"{3.1 * ko:.2f}", "--lookat-z", f"{0.5 * ko:.2f}"], ovl)}
    msgs = []
    for kind, (cmd, out) in jobs.items():
        if only and kind != only:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=ENV)
        if r.returncode:
            msgs.append(f"FAILED {out.relative_to(REPO)}: {r.stderr[-300:]}"); continue
        os.replace(str(out) + ".png", PREV / f"{name}_{out.parent.name}_{stem}.mp4.png")
        msgs.append(f"ok {out.relative_to(REPO)}  (search: {run})")
    return "\n".join(msgs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(BODY))
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--only", choices=["side_by_side", "overlay"], default=None)
    a = ap.parse_args()
    PREV.mkdir(parents=True, exist_ok=True)
    stems = a.stems or [BODY[a.name].get("clips", {}).get(s, s) for s in CLIPS]
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(lambda s: render(a.name, s, a.only), stems):
            print(msg, flush=True)
