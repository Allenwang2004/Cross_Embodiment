#!/usr/bin/env python3
"""Survey videos for the robots, as scripts/survey_body_videos.py does for the humanoid bodies:
outputs/user_survey/<robot>/side_by_side/<stem>.mp4 [the ADULT doing the original motion | the robot with z0 | the
robot with the searched z] (render_panels, no captions), drawn flat grey / red / blue -- the overlay videos' colours --
with one camera sized for the taller of the adult (1.74 m) and the robot, so the two keep their true relative size; and
outputs/user_survey/<robot>/overlay/<stem>.mp4 (robot_overlay --clean: target as a translucent grey ghost, z0 red,
searched z blue), from outputs/user_survey_search/<robot>/<stem>_s0 (robot_z_search). The overlay's target is the clip
mapped onto the robot kinematically (model.robots.target_path). Cameras: the child's (2.8 m / 0.55 m for 1.26 m) scaled
by stature; the robot's from its size factor as robot_overlay uses it (leg ratio s / 0.8213: G1 1.0, H1 1.26,
Apollo 1.34), G1 taken as 1.32 m tall. Contact sheets and specs go to outputs/survey_preview/.
usage: uv run python scripts/survey_robot_videos.py <g1|h1|apollo> [stem ...] [--only side_by_side|overlay]
       [--search outputs/user_survey_search/g1_ridge --name g1_ridge]"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent; PREV = REPO / "outputs/survey_preview"
sys.path.insert(0, str(REPO))
from model.robots import ROBOTS, target_path                                 # noqa: E402
from crossembodiment.survey import ADULT, ADULT_H, CLIPS   # noqa: E402  survey_body_videos' (refactor step 4)
from crossembodiment.palette import BLUE, GREY, RED   # noqa: E402  survey_body_videos' (refactor step 4)
from crossembodiment.camera import survey_track   # noqa: E402  the survey's side-by-side camera (refactor step 10)
ENV = {**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2", "CUDA_VISIBLE_DEVICES": "2"}


def render(r, stem, only, search=None, name=None):
    """search: the searches' folder (default outputs/user_survey_search/<robot>); name: the output folder under
    outputs/user_survey (default <robot>), e.g. g1_ridge for the ridge-fill searches."""
    task = stem.rsplit("_", 1)[0]; tgt = target_path(r, stem)
    size = json.loads((tgt.parent / "summary.json").read_text())["s"] / 0.8213
    run = f"{search or f'outputs/user_survey_search/{r}'}/{stem}_s0"; xml = ROBOTS[r]["scene"]
    root = REPO / "outputs/user_survey" / (name or r)
    sbs, ovl = root / "side_by_side" / f"{stem}.mp4", root / "overlay" / f"{stem}.mp4"
    spec = dict(out=str(sbs.relative_to(REPO)), cols=3, size=1080, captions=False,
                track=survey_track(size * 1.32, ADULT_H).to_track(),      # side by side: frame the taller of adult and robot
                panels=[dict(xml=ADULT, qpos=f"data/origin_motion/{task}/{stem}.npz", rgba=GREY),
                        dict(xml=xml, qpos=f"{run}/origin_z.npz", rgba=RED),
                        dict(xml=xml, qpos=f"{run}/best.npz", rgba=BLUE)])
    sp = PREV / f"spec_{name or r}_{stem}.json"; sp.write_text(json.dumps(spec, indent=1))
    jobs = {"side_by_side": (["uv", "run", "python", "scripts/render_panels.py", str(sp)], sbs),
            "overlay": (["uv", "run", "python", "scripts/robot_overlay.py", "--robot", r, "--clip", f"{task}/{stem}",
                         "--search", run, "--out", str(ovl), "--clean"], ovl)}
    msgs = []
    for kind, (cmd, out) in jobs.items():
        if only and kind != only:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=ENV)
        if p.returncode:
            msgs.append(f"FAILED {out.relative_to(REPO)}: {p.stderr[-300:]}"); continue
        os.replace(str(out) + ".png", PREV / f"{name or r}_{out.parent.name}_{stem}.mp4.png")
        msgs.append(f"ok {out.relative_to(REPO)}")
    return "\n".join(msgs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", choices=sorted(ROBOTS))
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--only", choices=["side_by_side", "overlay"], default=None)
    ap.add_argument("--search", default=None, help="searches folder (default outputs/user_survey_search/<robot>)")
    ap.add_argument("--name", default=None, help="output folder under outputs/user_survey (default <robot>)")
    a = ap.parse_args()
    PREV.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(lambda s: render(a.robot, s, a.only, a.search, a.name), a.stems or CLIPS):
            print(msg, flush=True)
