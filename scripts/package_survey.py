#!/usr/bin/env python3
"""package_survey.py -- make outputs/user_survey self-contained, so anyone can re-render its videos with replay.py:

  <body>/trajectories/<clip>/  reference_adult.npz  the adult's original motion (side-by-side, left panel)
                               reference_body.npz   the overlay's ghost: the body's own reference (child / fat / tall),
                                                    the adult's motion (heavy_arm / limit_leg), the clip mapped onto
                                                    the G1 kinematically (g1)
                               z0.npz, best.npz     the rollouts of z0 and of the searched z (the survey's runs:
                                                    survey_body_videos.run_of; g1: the ridge-fill searches)
                               video.json           how both videos are drawn (models, colours, cameras)
  models/                      every XML the videos use; the G1's with its meshes at their relative paths
  replay.py, README.md
Every .npz holds qpos (frames x nq) and fps (30). Cameras and colours are the ones the survey videos were made with
(outputs/survey_preview/spec_*.json, scripts/body_overlay.py, scripts/robot_overlay.py): crossembodiment.palette (GHOST / RED / BLUE),
crossembodiment.camera (robot_overlay, body_overlay); the clip folders are crossembodiment.clips.ClipBundle's (OOP refactor step 10).
usage: uv run python scripts/package_survey.py
"""
import json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robots import ROBOTS, target_path                     # noqa: E402
from crossembodiment.survey import ADULT, BODY, CLIPS, run_of   # noqa: E402  survey_body_videos' (refactor step 4)
from crossembodiment.camera import body_overlay, robot_overlay   # noqa: E402  (refactor step 10)
from crossembodiment.clips import ClipBundle                     # noqa: E402
from crossembodiment import paths                                # noqa: E402
from crossembodiment.palette import BLUE, GHOST, RED             # noqa: E402

TOP = paths.write_path("outputs/user_survey"); PREV = paths.read_path("outputs/survey_preview")
G1_SEARCH = "outputs/user_survey_search/g1_ridge"


def models():
    md = TOP / "models"; md.mkdir(parents=True, exist_ok=True)
    out = {"adult": "models/adult.xml"}
    shutil.copy(REPO / ADULT, md / "adult.xml")
    for b, cfg in BODY.items():
        shutil.copy(REPO / cfg["xml"], md / f"{b}.xml"); out[b] = f"models/{b}.xml"
    g = REPO / ROBOTS["g1"]["scene"]; (md / "g1").mkdir(exist_ok=True)
    shutil.copy(g, md / "g1/scene.xml"); shutil.copy(g.parent / "g1.xml", md / "g1/g1.xml")
    t = (g.parent / "g1.xml").read_text()
    meshdir = re.search(r'meshdir="([^"]+)"', t).group(1)
    for f in sorted(set(re.findall(r'\bfile="([^"]+)"', t))):
        dst = (md / "g1" / meshdir / f).resolve(); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy((g.parent / meshdir / f).resolve(), dst)
    out["g1"] = "models/g1/scene.xml"
    return out


def clip(body, stem, mdl):
    task = stem.rsplit("_", 1)[0]
    d = TOP / body / "trajectories" / stem; d.mkdir(parents=True, exist_ok=True)
    if body == "g1":
        run, ref_body = f"{G1_SEARCH}/{stem}_s0", str(target_path("g1", stem).relative_to(REPO))
        spec = json.loads((PREV / f"spec_g1_ridge_{stem}.json").read_text())
    else:
        run, ref_body = run_of(body, stem), f"{BODY[body]['ref']}/{task}/{stem}.npz"
        spec = json.loads((PREV / f"spec_{body}_{stem}.json").read_text())
    adult = f"data/origin_motion/{task}/{stem}.npz"
    # the side-by-side spec must be the one the video was drawn from
    assert [p["qpos"] for p in spec["panels"]] == [adult, f"{run}/origin_z.npz", f"{run}/best.npz"], (body, stem)
    b = ClipBundle.from_files(stem, body, dict(reference_adult=adult, reference_body=ref_body, z0=f"{run}/origin_z.npz", best=f"{run}/best.npz"))
    n = {k: len(v) for k, v in b.qpos.items()}
    tr = spec["track"]
    sbs = dict(size=spec["size"], camera=dict(distance=tr["distance"], elevation=tr["elevation"], azimuth=tr["azimuth"],
                                              lookat_z=tr["lookat_z"]),
               panels=[dict(model=mdl["adult"], qpos="reference_adult.npz", rgba=spec["panels"][0]["rgba"]),
                       dict(model=mdl[body], qpos="z0.npz", rgba=spec["panels"][1]["rgba"]),
                       dict(model=mdl[body], qpos="best.npz", rgba=spec["panels"][2]["rgba"])])
    if body == "g1":         # scripts/robot_overlay.py (track camera): framed by the G1's size, materials kept
        size = json.loads((target_path("g1", stem).parent / "summary.json").read_text())["s"] / 0.8213
        ovl = dict(model=mdl["g1"], root=ROBOTS["g1"]["root"], flat=False, camera=robot_overlay(size).to_track())
    else:                    # scripts/body_overlay.py as survey_body_videos.py calls it
        ovl = dict(model=mdl[body], root="Pelvis", flat=True, camera=body_overlay(BODY[body]["h"]).to_track())
    ovl.update(width=1920, height=1080, layers=[dict(qpos="reference_body.npz", rgba=GHOST), dict(qpos="z0.npz", rgba=RED),
                                               dict(qpos="best.npz", rgba=BLUE)])
    b.meta.update(search_run=run, side_by_side=sbs, overlay=ovl); b.save(d)
    return n


def main():
    mdl = models()
    shutil.copy(REPO / "scripts/survey_replay.py", TOP / "replay.py")
    for body in ("child", "fat", "tall", "heavy_arm", "limit_leg", "g1"):
        stems = CLIPS if body == "g1" else [BODY[body].get("clips", {}).get(s, s) for s in CLIPS]
        for stem in stems:
            n = clip(body, stem, mdl)
        print(f"{body}: {len(stems)} clips packaged (last: {stem} {n})", flush=True)


if __name__ == "__main__":
    main()
