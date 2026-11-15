#!/usr/bin/env python3
"""dtw_eval.py -- a single_z_search.py --objective dtw run (model/dtw.py: the rollout may be slower than its
reference): how much slower, how far it got, whether it fell, from the trajectories the search saved.

Per z0 and the best z: dtw (1 - aligned B similarity), done (share of the reference reached), pace (reference frames
per rollout frame up to the end of the alignment), the frame-by-frame bfm over the reference's length, the time it
took to finish the matched part and the distance walked by then, the lowest head height (over the matched part and
over the whole rollout: a fall shows as a head near the floor).
Video (2 x 2, 1080 px panels) <run>/../../<stem>/dtw_<tag>.mp4 next to the other slippery videos: [the adult's
original motion, grey | the adult played along the best z's alignment (its pace), light grey | z0, red | the best z,
blue], captioned with the numbers. Numbers in dtw_<tag>.json.
usage: uv run python scripts/dtw_eval.py outputs/slippery/search/<stem>_mu<mu>_dtw[_...]_s0 [...]"""
import json, os, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from survey_body_videos import ADULT, BLUE, GREY, RED       # noqa: E402
LIGHT = [0.80, 0.81, 0.84, 1]


def main():
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT)); da = mujoco.MjData(ma); head = ma.body("Head").id
    for run in map(Path, sys.argv[1:]):
        run = run if run.is_absolute() else REPO / run
        s = json.loads((run / "summary.json").read_text()); task, stem = s["clip"].split("/")
        ref = np.load(REPO / f"data/origin_motion/{s['clip']}.npz")["qpos"]
        tag = run.name.removeprefix(f"{stem}_").removesuffix(f"_s{s['seed']}")
        out = REPO / "outputs/slippery" / stem; out.mkdir(parents=True, exist_ok=True)
        res, aligned = {}, None
        ie = None
        for key, f, rec in (("z0", "origin_z.npz", s["origin_z"]), ("best", "best.npz", s["best"])):
            z = np.load(run / f); q, path = z["qpos"].astype(np.float64), z["dtw_path"]
            ie = int(np.flatnonzero(path >= 0).max())
            hz = []
            for qq in q:
                da.qpos[:] = qq; mujoco.mj_kinematics(ma, da); hz.append(da.xpos[head][2])
            hz = np.array(hz)
            res[key] = dict(dtw=rec["dtw"], done=rec["done"], pace=rec["pace"], heading_al=rec["head_al"], bfm_frame_by_frame=rec["bfm"],
                            fallen_share=float((hz[:ie + 1] < s.get("fall_head", 0.8)).mean()),
                            matched_seconds=(ie + 1) / 30, distance_m=float(np.linalg.norm(q[ie, :2] - q[0, :2])),
                            head_lowest_matched_m=float(hz[:ie + 1].min()), head_lowest_all_m=float(hz.min()))
            if key == "best":
                aligned = ref[path[:ie + 1]].copy()
                np.savez_compressed(out / f"dtw_{tag}_adult_aligned.npz", qpos=aligned.astype(np.float32), fps=30)
            print(f"{tag:22s} {key:4s} dtw {rec['dtw']:.3f} | done {rec['done']:.2f} pace {rec['pace']:.2f} | heading_al {rec['head_al']:.3f} | "
                  f"frame-by-frame bfm {rec['bfm']:.3f} | {res[key]['distance_m']:.1f} m in {res[key]['matched_seconds']:.1f} s | "
                  f"head lowest {res[key]['head_lowest_matched_m']:.2f} m (whole rollout {res[key]['head_lowest_all_m']:.2f}), "
                  f"fallen {100 * res[key]['fallen_share']:.0f}% of the matched frames", flush=True)
        ref_dist = float(np.linalg.norm(ref[-1, :2] - ref[0, :2]))
        (out / f"dtw_{tag}.json").write_text(json.dumps(dict(run=str(run.relative_to(REPO)), floor_friction=s["floor_friction"],
                                                            adult_distance_m=ref_dist, adult_seconds=len(ref) / 30, results=res), indent=1))
        b, z0 = res["best"], res["z0"]
        sub = lambda r: (f"pace {r['pace']:.2f}x, done {100 * r['done']:.0f}%, aligned dtw {r['dtw']:.3f}, "
                         f"{r['distance_m']:.1f} m in {r['matched_seconds']:.1f} s, head lowest {r['head_lowest_matched_m']:.2f} m")
        k = 1.74 / 1.26
        spec = dict(out=str((out / f"dtw_{tag}.mp4").relative_to(REPO)), cols=2, size=1080,
                    track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)),
                    panels=[dict(title="adult, original motion (normal floor)", sub=f"{ref_dist:.1f} m in {len(ref) / 30:.0f} s",
                                 xml=ADULT, qpos=f"data/origin_motion/{s['clip']}.npz", rgba=GREY),
                            dict(title="adult, played at the searched z's pace", sub=f"pace {b['pace']:.2f}x (along the alignment)",
                                 xml=ADULT, qpos=str((out / f"dtw_{tag}_adult_aligned.npz").relative_to(REPO)), rgba=LIGHT),
                            dict(title=f"z0, floor friction {s['floor_friction']:g}", sub=sub(z0), xml=ADULT,
                                 qpos=str((run / "origin_z.npz").relative_to(REPO)), rgba=RED),
                            dict(title=f"searched z ({tag}), floor friction {s['floor_friction']:g}", sub=sub(b), xml=ADULT,
                                 qpos=str((run / "best.npz").relative_to(REPO)), rgba=BLUE)])
        sp = out / f"spec_dtw_{tag}.json"; sp.write_text(json.dumps(spec, indent=1))
        subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(sp)], cwd=REPO, check=True,
                       env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2"})


if __name__ == "__main__":
    main()
