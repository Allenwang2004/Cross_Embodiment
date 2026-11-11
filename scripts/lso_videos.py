#!/usr/bin/env python3
"""lso_videos.py -- one comparison video per lso_search.py run: [the ADULT doing the original motion | Motivo's zero-shot
tracking (ER window L=8) on the body | the best LSO sequence on the body], drawn flat grey / red / blue as the survey
videos are (scripts/survey_body_videos.py), with one camera sized for the taller of the adult (1.74 m) and the body so
the two keep their true relative size; captioned with bfm, root error and end-point error from its summary.json.
The adult's clip is the run's reference with amass_retargeting_motion -> amass_origin_motion (same frames), taken from
amass_origin_motion_grounded (floating removed, scripts/ground_amass.py) when that has the clip.
Writes <run>/compare.mp4 (+ .png) and the panels' qpos as <run>/q_*.npz.
usage: uv run python scripts/lso_videos.py outputs/lso/<run> [...] [--xml assets/robots_torque/child/robot_torque_full.xml]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossenbodiment.survey import ADULT, ADULT_H   # noqa: E402  survey_body_videos' (refactor step 4)
from crossenbodiment.palette import BLUE, GREY, RED   # noqa: E402  survey_body_videos' (refactor step 4)
from crossenbodiment.camera import survey_track   # noqa: E402  the survey's side-by-side camera (refactor step 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--xml", default="assets/robots_torque/child/robot_torque_full.xml")
    ap.add_argument("--height", type=float, default=1.26, help="the body's stature (m); child 1.26")
    a = ap.parse_args()
    track = survey_track(a.height, ADULT_H).to_track()                # the child's camera (2.8 m / 0.55 m) scaled
    for run in a.runs:
        d = REPO / run; s = json.loads((d / "summary.json").read_text()); res = s["results"]
        r = np.load(d / "rollouts.npz")
        for key in ("q_ER8", "q_LSO_best"):
            np.savez(d / f"{key}.npz", qpos=r[key].astype(np.float64), fps=30.0)
        adult = s["ref"].replace("amass_retargeting_motion", "amass_origin_motion")
        if "_grounded" not in adult:              # prefer the adult clip with the floating fixed (scripts/ground_amass.py)
            g = adult.replace("amass_origin_motion", "amass_origin_motion_grounded")
            adult = g if (REPO / g).exists() else adult
        assert len(np.load(REPO / adult)["qpos"]) == s["frames"], adult
        cap = lambda v: f"bfm {v['bfm']:.3f}, root error {v['root_rms_cm']:.0f} cm, end-point off {v['end_off_m']:.2f} m"
        rel = lambda p: str(p.relative_to(REPO))
        lso = f"LSO{' with ranks' if s.get('rank') else ''}, best of {s['iters']} iterations"
        spec = dict(out=rel(d / "compare.mp4"), cols=3, size=1080,
                    track=dict(track),
                    panels=[dict(title="adult, original motion", sub=Path(s["ref"]).stem, xml=ADULT, qpos=adult, rgba=GREY),
                            dict(title="Motivo zero-shot tracking (window L=8)", sub=cap(res["ER L=8 (Motivo zero-shot)"]),
                                 xml=a.xml, qpos=rel(d / "q_ER8.npz"), rgba=RED),
                            dict(title=lso, sub=cap(res["LSO best (re-rolled)"]),
                                 xml=a.xml, qpos=rel(d / "q_LSO_best.npz"), rgba=BLUE)])
        sp = d / "spec_compare.json"; sp.write_text(json.dumps(spec, indent=1))
        p = subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(sp)], cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2", "CUDA_VISIBLE_DEVICES": "2"})
        print(("ok " if p.returncode == 0 else "FAILED ") + rel(d / "compare.mp4") + ("" if p.returncode == 0 else p.stderr[-300:]),
              flush=True)


if __name__ == "__main__":
    main()
