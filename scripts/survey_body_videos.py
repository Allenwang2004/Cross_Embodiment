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
import argparse, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/home/allen19/crossenbodiment"); PREV = REPO / "outputs/survey_preview"
CLIPS = ["move-ego-0-2_2", "move-ego-0-4_8", "crawl-0.4-2-d_8", "rotate-x-5-0.8_9", "rotate-x--5-0.8_4", "rotate-y-5-0.8_7",
         "rotate-z--5-0.8_6", "jump-2_6", "jump-2_7", "headstand_4", "crouch-0_8"]
ADULT, ADULT_H = "assets/robots/adult/robot.xml", 1.74
BODY = {"child": dict(xml="assets/robots_torque/child/robot_torque_full.xml", ref="data/child/retargeting_motion", h=1.26),
        "fat": dict(xml="assets/robots_torque/fat/robot_torque_full.xml", ref="data/fat/retargeting_motion", h=1.74),
        "tall": dict(xml="assets/robots_torque/tall/robot_torque_full.xml", ref="data/tall/retargeting_motion", h=2.07),
        "heavy_arm": dict(xml="assets/robots/adult_heavyRWristHand_x80/robot.xml", ref="data/origin_motion", h=1.74,
                          clips={"move-ego-0-2_2": "move-ego-0-2_4"}),
        "limit_leg": dict(xml="assets/robots/adult_lockRleg_bent_lim/robot.xml", ref="data/origin_motion", h=1.74)}
GREY, RED, BLUE = [0.55, 0.57, 0.62, 1.0], [0.85, 0.22, 0.20, 1.0], [0.18, 0.42, 0.85, 1.0]   # robot_overlay.COLORS, grey opaque
SUFFIXES = ("_s0", "_g512_s0", "_sig015_s0", "_sig015_cont_s1", "_sig015_cont2_s2", "_sig015_s1", "_sig015_s2", "_sig015_s3")
ENV = {**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2", "CUDA_VISIBLE_DEVICES": "2"}


def run_of(name, stem):
    if name == "child":
        return {"rotate-x--5-0.8_4": "outputs/user_survey_search/rotate-x--5-0.8_4_g256_s0"}.get(stem, f"outputs/c540_bfmglobal_a03/{stem}")
    runs = [f"outputs/user_survey_search/{name}/{stem}{suf}" for suf in SUFFIXES
            if (REPO / f"outputs/user_survey_search/{name}/{stem}{suf}/summary.json").exists()]
    return min(runs, key=lambda d: json.loads((REPO / d / "summary.json").read_text())["best"]["cost"])


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
