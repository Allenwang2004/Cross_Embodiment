#!/usr/bin/env python3
"""package_amass_lso_survey.py -- the AMASS (ACCAD) clips searched with LSO on the child (scripts/lso_search.py --rank,
outputs/lso/<clip>_global_it256_rank or _it128_rank) or, with --body g1, on the G1 (scripts/robot_lso_search.py,
outputs/g1/lso/<clip>_it<n>_rank) as user-survey entries, in the format of outputs/user_survey (README.md there):

  outputs/user_survey/child_amass/trajectories/<clip>/
      reference_adult.npz   the adult's original motion (amass_origin_motion_grounded, else amass_origin_motion)
      reference_body.npz    the clip retargeted to the child (amass_retargeting_motion): the overlay's ghost
      zero_shot.npz         Motivo's own zero-shot tracking on the child (ER, window 8: one latent per step), red
      lso_best.npz          the best LSO latent sequence on the child, blue
      video.json            models, colours and cameras, the child survey's
  outputs/user_survey/child_amass/{side_by_side,overlay}/<clip>.mp4, drawn by outputs/user_survey/replay.py from the above

  outputs/user_survey/g1_amass/...         the same for the G1: reference_body = the adult's clip mapped onto the G1
      kinematically (RobotMap.smpl_to_robot_qpos, warm-started, as scripts/robot_roundtrip.py), cameras as the g1 survey's
      (side by side framed for the adult, overlay by the G1's size with its materials kept)

The rollouts hold the states after each of the clip's T-1 control steps; the start state (the reference's frame 0 --
on the G1, frame 0 mapped onto it -- at rest) is put in front so every trajectory has the reference's T frames. Runs without a summary.json
(still running) are skipped; packaged clips are skipped unless --force.
OOP refactor step 10: colours crossenbodiment.palette, cameras crossenbodiment.camera, the clip folder crossenbodiment.clips.ClipBundle;
runs are read through crossenbodiment.paths.read_path ($CE_OUTPUTS_ROOT), the survey folder written under $CE_RESULTS_ROOT.
usage: uv run python scripts/package_amass_lso_survey.py [clip ...] [--body child|g1] [--force]"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.camera import BODY_OVERLAY_DEFAULT, robot_overlay, survey_track   # noqa: E402
from crossenbodiment.clips import ClipBundle                              # noqa: E402
from crossenbodiment.palette import BLUE, GHOST, GREY, RED                # noqa: E402
TOP = paths.write_path("outputs/user_survey")
CLIPS = ["Male2MartialArtsStances__D9_-_warm_up_to_ready_to_warm_up", "Male2MartialArtsExtended__Extended_1",
         "Male2MartialArtsExtended__Form_1", "Male2General__A12-_Crawl_Backwards", "Male2MartialArtsStances__D9_-_victory_1",
         "MartialArtsWalksTurns__E5_-_retreat", "MartialArtsWalksTurns__E8_-_bounce", "Male2Walking__B15_-__Walk_turn_around",
         "Male2Walking__B19_-__Walk_to_pickup_box", "MartialArtsWalksTurns__E3_-_advance", "Male2Walking__B24_-__walk_to_crouch",
         "MartialArtsWalksTurns__E6_-_quick_retreat", "MartialArtsWalksTurns__E4_-_quick_advance"]
README_NOTE = """
## child_amass: AMASS clips with a latent sequence (LSO)

`child_amass/` holds longer AMASS (ACCAD) clips on the child, searched differently from the other folders: instead of
one latent for the whole clip, LSO optimises one latent per control step (BFMTrack's latent sequence optimisation,
with per-step ranks, 128 rollouts per iteration; 256 iterations for the first clips, 128 for the later ones,
which reach 99% of their improvement by about iteration 100 -- `search_run` in each video.json says which).
Its trajectory files are named for that:

- `zero_shot.npz` (red, in place of `z0`): Motivo's own zero-shot tracking on the child (one latent per step, the
  average of the next 8 frames' embeddings), the starting point that LSO improves on.
- `lso_best.npz` (blue, in place of `best`): the best latent sequence LSO found.
- `reference_adult.npz` / `reference_body.npz`: the adult's original motion / the clip retargeted to the child.

Clips here run 4-20 s; every trajectory has the reference's frame count (the start pose first).
"""
README_NOTE_G1 = """
## g1_amass: AMASS clips on the G1 with a latent sequence (LSO)

`g1_amass/` is `child_amass/` on the Unitree G1: the same LSO (scripts/robot_lso_search.py, through the human-to-robot
mapping, with the IK penalty of the g1 searches), its iteration count in `search_run`. `zero_shot.npz` (red) /
`lso_best.npz` (blue) as in `child_amass/`; `reference_body.npz` is the adult's clip mapped onto the G1 kinematically,
the overlay's ghost, as in `g1/`.
"""


def package(clip, force, body):
    OUT = TOP / f"{body}_amass"
    if body == "child":
        runs = [paths.read_path(f"outputs/lso/{clip}_global_it{n}_rank") for n in (256, 128)]     # 256 iterations, or 128 (later clips)
    else:
        runs = sorted((Path(p) for p in glob.glob(str(paths.read_path(f"outputs/{body}/lso") / f"{clip}_it*_rank"))),
                      key=lambda p: -int(p.name.rsplit("_it", 1)[1].split("_")[0]))
    run = next((r for r in runs if (r / "summary.json").exists()), None)
    if run is None:
        return f"skip {clip}: not finished"
    d = OUT / "trajectories" / clip
    if (OUT / "overlay" / f"{clip}.mp4").exists() and not force:
        return f"skip {clip}: already packaged"
    d.mkdir(parents=True, exist_ok=True)
    s = json.loads((run / "summary.json").read_text()); r = np.load(run / "rollouts.npz")
    g = REPO / f"amass_data/amass_origin_motion_grounded/ACCAD/{clip}.npz"
    adult = np.load(g if g.exists() else REPO / f"amass_data/amass_origin_motion/ACCAD/{clip}.npz")["qpos"]
    if body == "child":
        ref = np.load(REPO / f"amass_data/amass_retargeting_motion/ACCAD/{clip}.npz")["qpos"]; start = ref[:1]
        mdl = "models/child.xml"
        sbs_cam = survey_track(1.26).to_track()                                       # the child survey's cameras (3.87 m / 0.76 m)
        ovl = dict(model=mdl, root="Pelvis", flat=True, camera=BODY_OVERLAY_DEFAULT.to_track())
    else:
        sys.path.insert(0, str(REPO))
        from model.robot_map import RobotMap
        from model.robots import ROBOTS
        sa = np.load(REPO / s["ref"])["qpos"].astype(np.float64)       # the adult's clip the search ran on
        assert np.allclose(sa, adult), clip
        M = RobotMap(body, fill=s["fill"]); ref = np.zeros((len(sa), M.m_robot.nq)); q = None
        for t in range(len(sa)):                                         # as scripts/robot_roundtrip.py
            ref[t], _ = M.smpl_to_robot_qpos(sa[t], q); q = ref[t][M.robot_qadr]
        start = M.smpl_to_robot_qpos(sa[0])[0][None]                     # RobotEnv.reset's start pose
        size = M.scale / 0.8213
        mdl = f"models/{body}/scene.xml"
        sbs_cam = survey_track(size * 1.32, 1.74).to_track()
        ovl = dict(model=mdl, root=ROBOTS[body]["root"], flat=False, camera=robot_overlay(size).to_track())
    assert len(adult) == len(ref) == s["frames"] == len(r["q_LSO_best"]) + 1, clip
    b = ClipBundle(clip, body, dict(reference_adult=np.asarray(adult), reference_body=np.asarray(ref), zero_shot=np.concatenate([start, r["q_ER8"]]),
                                    lso_best=np.concatenate([start, r["q_LSO_best"]])))
    b.save_npz(d); n = {k: len(v) for k, v in b.qpos.items()}
    res = s["results"]
    ovl.update(width=1920, height=1080, layers=[dict(qpos="reference_body.npz", rgba=GHOST), dict(qpos="zero_shot.npz", rgba=RED),
                                               dict(qpos="lso_best.npz", rgba=BLUE)])
    b.meta.update(search_run=f"outputs/{run.relative_to(paths.outputs_root())}",
                  bfm=dict(zero_shot=res["ER L=8 (Motivo zero-shot)"]["bfm"], lso_best=res["LSO best (re-rolled)"]["bfm"]),
                  side_by_side=dict(size=1080, camera=sbs_cam,
                                    panels=[dict(model="models/adult.xml", qpos="reference_adult.npz", rgba=GREY),
                                            dict(model=mdl, qpos="zero_shot.npz", rgba=RED),
                                            dict(model=mdl, qpos="lso_best.npz", rgba=BLUE)]),
                  overlay=ovl)
    b.save_json(d); spec = b.video_json()
    env = {**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2", "CUDA_VISIBLE_DEVICES": "2"}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        p = subprocess.run(["uv", "run", "--project", str(REPO), "python", "replay.py", f"{body}_amass/trajectories/{clip}",
                            "--out-dir", tmp], cwd=TOP, env=env, capture_output=True, text=True)
        if p.returncode:
            return f"FAILED {clip}: {p.stderr[-400:]}"
        for kind in ("side_by_side", "overlay"):
            (OUT / kind).mkdir(parents=True, exist_ok=True)
            shutil.move(f"{tmp}/{clip}_{kind}.mp4", OUT / kind / f"{clip}.mp4")
    return f"ok {clip}: {n['reference_body']} frames, bfm zero-shot {spec['bfm']['zero_shot']:.3f} -> LSO {spec['bfm']['lso_best']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--body", default="child", choices=["child", "g1"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (TOP / f"{a.body}_amass").mkdir(parents=True, exist_ok=True)
    readme = TOP / "README.md"
    for head, note in (("## child_amass", README_NOTE), ("## g1_amass", README_NOTE_G1)):
        t = readme.read_text()
        if (head == "## child_amass" or a.body == "g1") and head not in t:
            readme.write_text(t.rstrip("\n") + "\n" + note)
    for c in a.clips or CLIPS:
        print(package(c, a.force, a.body), flush=True)


if __name__ == "__main__":
    main()
