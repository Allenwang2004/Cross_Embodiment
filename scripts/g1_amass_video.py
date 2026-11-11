#!/usr/bin/env python3
"""g1_amass_video.py -- side-by-side video of a scripts/robot_lso_search.py run (an AMASS clip on a robot):
[the adult doing the original motion, grey | the robot with Motivo's zero-shot tracking (ER, window 8), red | the robot
with the best LSO latent sequence, blue], 1080 px panels (render_panels, flat colours as the survey videos), one camera
framing the taller of the adult and the robot (scripts/survey_robot_videos.py: robot size G1 1.0, H1 1.26, Apollo 1.34;
G1 taken as 1.32 m tall), captions with bfm and the IK error of each rollout.
Writes <out-dir>/<clip>_side_by_side.mp4 (+ .png contact sheet) and the panels' robot qpos as <out-dir>/<clip>_{ER8,LSO_best}.npz.
usage: uv run python scripts/g1_amass_video.py outputs/g1/lso/<run> [--out-dir outputs/g1_amass]"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robots import ROBOTS                                             # noqa: E402
from crossenbodiment.survey import ADULT, ADULT_H   # noqa: E402  survey_body_videos' (refactor step 4)
from crossenbodiment.palette import BLUE, GREY, RED   # noqa: E402  survey_body_videos' (refactor step 4)
SIZE = {"g1": 1.0, "h1": 1.26, "apollo": 1.34}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out-dir", default="outputs/g1_amass")
    a = ap.parse_args()
    run = REPO / a.run; s = json.loads((run / "summary.json").read_text()); res = s["results"]; r = np.load(run / "rollouts.npz")
    clip = Path(s["ref"]).stem; robot = s["robot"]; name = ROBOTS[robot]["name"].split()[-1]
    out = REPO / a.out_dir; out.mkdir(parents=True, exist_ok=True)
    for key in ("ER8", "LSO_best"):
        np.savez_compressed(out / f"{clip}_{key}.npz", qpos=r[f"q_{key}"].astype(np.float32), fps=np.float32(30))
    k = max(SIZE[robot] * 1.32, ADULT_H) / 1.26
    cap = lambda v: f"bfm {v['bfm']:.3f}, IK error {v['ik_deg']:.1f} deg"
    rel = lambda p: str(p.relative_to(REPO))
    spec = dict(out=rel(out / f"{clip}_side_by_side.mp4"), cols=3, size=1080,
                track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)),
                panels=[dict(title="adult, original motion", sub=clip, xml=ADULT, qpos=s["ref"], rgba=GREY),
                        dict(title=f"{name}, Motivo zero-shot tracking (window 8)", sub=cap(res["ER L=8 (Motivo zero-shot)"]),
                             xml=ROBOTS[robot]["scene"], qpos=rel(out / f"{clip}_ER8.npz"), rgba=RED),
                        dict(title=f"{name}, LSO best of {s['iters']} iterations", sub=cap(res["LSO best (re-rolled)"]),
                             xml=ROBOTS[robot]["scene"], qpos=rel(out / f"{clip}_LSO_best.npz"), rgba=BLUE)])
    sp = out / f"spec_{clip}.json"; sp.write_text(json.dumps(spec, indent=1))
    subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(sp)], cwd=REPO, check=True,
                   env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2", "CUDA_VISIBLE_DEVICES": "2"})
    print(f"-> {spec['out']}")


if __name__ == "__main__":
    main()
