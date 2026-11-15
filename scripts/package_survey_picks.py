#!/usr/bin/env python3
"""package_survey_picks.py -- add the motions where LOES helps most (scripts/select_clips.py's top 5 per body, user 11/10)
to outputs/user_survey, in the survey's format, in new folders that leave the earlier survey videos alone:

  g1_task52/, heavy_arm_task52/ (the payload body), limit_leg_task52/
    trajectories/<clip>/  reference_adult.npz, reference_body.npz, z0.npz, best.npz, video.json
    side_by_side/<clip>.mp4, overlay/<clip>.mp4      (rendered by the survey's replay.py from those files)

Runs: the 52-motion task runs of 11/09 (seed 0, 2,048 rollouts, reference-velocity start): z0 = the LOES run's z0 rollout,
best = its best latent's rollout after 2,048 rollouts (outputs/g1_52/<clip>_es_s0, outputs/body_52/<body>/<clip>_es_s0).
reference_body: the G1 target (the clip mapped onto the G1), the adult's motion (payload), the motion with the right leg
bent (limited leg) -- each body's search target. Frames are aligned as the evaluation aligns them (rollout frame t against
the reference's frame t + 1): the references start at their second frame and every file has the reference's frame count
minus one (crossembodiment.clips.ClipBundle.from_run, OOP refactor step 10). Models, colours and cameras: those of the body's earlier
survey folder (g1/, heavy_arm/, limit_leg/).
usage: uv run python scripts/package_survey_picks.py [--no-render]
"""
import hashlib, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.robots import target_path                                  # noqa: E402
from crossembodiment.clips import ClipBundle                          # noqa: E402
from crossembodiment import paths                                     # noqa: E402

TOP = paths.write_path("outputs/user_survey")
BODIES = dict(  # folder: (earlier survey folder, results dir, run dir, reference_body, the run's model, the survey's copy)
    g1_task52=("g1", "outputs/g1_52", "outputs/g1_52/{c}_es_s0", None, "assets/robots/g1/g1.xml", "models/g1/g1.xml"),
    heavy_arm_task52=("heavy_arm", "outputs/body_52/payload", "outputs/body_52/payload/{c}_es_s0", "data/origin_motion/{t}/{c}.npz",
                      "assets/robots/adult_heavyRWristHand_x80/robot.xml", "models/heavy_arm.xml"),
    limit_leg_task52=("limit_leg", "outputs/body_52/limit_leg", "outputs/body_52/limit_leg/{c}_es_s0",
                      "data/adult_lockRleg_bent_lim/retargeting_motion/{t}/{c}.npz", "assets/robots/adult_lockRleg_bent_lim/robot.xml",
                      "models/limit_leg.xml"))


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def package(folder):
    old, res, run_t, ref_t, xml, copy = BODIES[folder]
    assert md5(REPO / xml) == md5(TOP / copy), (folder, "the survey's model copy differs from the run's model")
    template = json.loads(next((TOP / old / "trajectories").glob("*/video.json")).read_text())
    top = json.loads((paths.read_path(res) / "top_clips.json").read_text())
    out = []
    for rank, r in enumerate(top["top5"], 1):
        c = r["clip"]; t = c.rsplit("_", 1)[0]; run = run_t.format(c=c)
        b = ClipBundle.from_run(c, folder, run=run, reference_adult=f"data/origin_motion/{t}/{c}.npz", same_length=True, at_least=True,
                                reference_body=str(target_path("g1", c).relative_to(REPO)) if ref_t is None else ref_t.format(t=t, c=c),
                                selection=dict(rank=rank, rule=top["rule"], z0_passes_success=bool(r["z0_phc_ok"]),
                                               mpjpe_cm=dict(z0=round(r["frozen_mpjpe"], 2), best=round(r["es_mpjpe"], 2)),
                                               success=dict(z0=bool(r["frozen_success"]), best=bool(r["es_success"])),
                                               fall_time_s=dict(z0=round(r["frozen_fall_s"], 2), best=round(r["es_fall_s"], 2))),
                                side_by_side=template["side_by_side"], overlay=template["overlay"])
        b.save(TOP / folder / "trajectories" / c)
        out.append((c, b.n, bool(r["z0_phc_ok"])))
    return out


def render(folder, clip):
    tmp = TOP / folder / "_render"
    subprocess.run([sys.executable, str(TOP / "replay.py"), f"{folder}/trajectories/{clip}", "--out-dir", str(tmp), "--top", str(TOP)],
                   check=True, cwd=TOP)
    for kind in ("side_by_side", "overlay"):
        (TOP / folder / kind).mkdir(exist_ok=True)
        (tmp / f"{clip}_{kind}.mp4").replace(TOP / folder / kind / f"{clip}.mp4")
    tmp.rmdir()


if __name__ == "__main__":
    for folder in BODIES:
        done = package(folder)
        print(folder, [(c, n, "z0 passes" if f else "") for c, n, f in done], flush=True)
        if "--no-render" not in sys.argv:
            for c, _, _ in done:
                render(folder, c)
