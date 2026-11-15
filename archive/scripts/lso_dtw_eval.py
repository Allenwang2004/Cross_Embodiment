#!/usr/bin/env python3
"""lso_dtw_eval.py -- an lso_search.py --dtw run on a slippery floor: how much slower than the adult, how far, whether
it fell, from the sequences the run rolled out again at its end (rollouts.npz: q_LSO / q_LSO_best and their
alignments path_LSO / path_LSO_best).

Per sequence: dtw / done / pace / fallen from summary.json, the walking speed once under way (pelvis, from 3 s to the
end of the matched part) against the adult's over the same part of its clip, the time to finish the matched part and
the distance by then, the lowest head height (matched part and whole rollout).
Video (2 x 2, 1080 px panels; the LSO panel stops where its alignment ends: what it does after the motion is done
is not optimised) outputs/slippery/<stem>/lso_<run>.mp4: [the adult's original motion, grey | the adult
played along the sequence's alignment, light grey | z0 on the same floor (a single_z_search run's origin_z.npz), red |
the LSO sequence, blue]. Numbers in lso_<run>.json.
A run without --dtw (frame by frame) is read with the identity alignment: the matched part is the whole rollout, pace 1.
The fallen share is recomputed here from the head height (< the run's --fall-head) for every run.
usage: uv run python scripts/lso_dtw_eval.py outputs/slippery/lso/<run> [--z0 outputs/slippery/search/<run>/origin_z.npz]
       [--no-video]"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from survey_body_videos import ADULT, BLUE, GREY, RED       # noqa: E402
LIGHT = [0.80, 0.81, 0.84, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--z0", default="outputs/slippery/search/move-ego-0-2_2_mu0.2_dtw_fall2_lr03_s0/origin_z.npz")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    run = REPO / a.run; s = json.loads((run / "summary.json").read_text()); R = np.load(run / "rollouts.npz")
    ref = np.load(REPO / s["ref"])["qpos"]; stem = Path(s["ref"]).stem
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT)); da = mujoco.MjData(ma); head = ma.body("Head").id
    out = REPO / "outputs/slippery" / stem; out.mkdir(parents=True, exist_ok=True)
    adult_v = np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1) * 30
    res = {}
    for key, qk, pk, rk in (("final", "q_LSO", "path_LSO", "LSO final mean"), ("best", "q_LSO_best", "path_LSO_best", "LSO best (re-rolled)")):
        q = R[qk].astype(np.float64)
        path = R[pk] if pk in R else np.arange(len(q))          # frame by frame: rollout frame i is reference frame i+1
        ie = int(np.flatnonzero(path >= 0).max()); je = int(path[ie])
        hz = []
        for qq in q:
            da.qpos[:] = qq; mujoco.mj_kinematics(ma, da); hz.append(da.xpos[head][2])
        hz = np.array(hz); r = s["results"][rk]
        t3 = min(90, ie - 1)
        v = np.linalg.norm(np.diff(q[t3:ie + 1, :2], axis=0), axis=1).mean() * 30
        va = adult_v[path[t3]:je + 1].mean()            # the adult over the same stretch of its clip
        res[key] = dict(cost=r["cost"], dtw=r.get("dtw", float("nan")), done=r.get("done", 1.0), pace=r.get("pace", 1.0),
                        fallen=float((hz[:ie + 1] < s.get("fall_head", 0.8)).mean()), heading=r.get("head", float("nan")),
                        bfm_frame_by_frame=r["bfm"], mse_obj=r.get("mse_obj", float("nan")),
                        speed_under_way=float(v), adult_speed_same_part=float(va), speed_ratio=float(v / va),
                        matched_seconds=(ie + 1) / 30, distance_m=float(np.linalg.norm(q[ie, :2] - q[0, :2])),
                        head_lowest_matched_m=float(hz[:ie + 1].min()), head_lowest_all_m=float(hz.min()))
        np.savez_compressed(out / f"lso_{run.name}_{key}_adult_aligned.npz", qpos=ref[np.maximum(path[:ie + 1], 0) + 1].astype(np.float32), fps=30)
        np.savez_compressed(out / f"lso_{run.name}_{key}.npz", qpos=q[:ie + 1].astype(np.float32), fps=30)   # the matched part
        x = res[key]
        print(f"{run.name} {key:5s} cost {x['cost']:.3f} | dtw {x['dtw']:.3f} done {x['done']:.2f} pace {x['pace']:.2f} fallen {x['fallen']:.3f} "
              f"heading {x['heading']:.3f} | under way {x['speed_under_way']:.2f} m/s vs adult {x['adult_speed_same_part']:.2f} "
              f"({x['speed_ratio']:.2f}x) | {x['distance_m']:.1f} m in {x['matched_seconds']:.1f} s | head lowest {x['head_lowest_matched_m']:.2f} m "
              f"(whole rollout {x['head_lowest_all_m']:.2f})", flush=True)
    (out / f"lso_{run.name}.json").write_text(json.dumps(dict(run=str(run.relative_to(REPO)), results=res), indent=1))
    key = min(res, key=lambda k: res[k]["cost"]); b = res[key]
    if a.no_video:
        return
    k = 1.74 / 1.26
    spec = dict(out=str((out / f"lso_{run.name}.mp4").relative_to(REPO)), cols=2, size=1080,
                track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)),
                panels=[dict(title="adult, original motion (normal floor)", sub=f"{np.linalg.norm(ref[-1, :2] - ref[0, :2]):.1f} m in {len(ref) / 30:.0f} s",
                             xml=ADULT, qpos=s["ref"], rgba=GREY),
                        dict(title="adult, played along the LSO alignment", sub=f"pace {b['pace']:.2f}x", xml=ADULT,
                             qpos=str((out / f"lso_{run.name}_{key}_adult_aligned.npz").relative_to(REPO)), rgba=LIGHT),
                        dict(title=f"z0, floor friction {s['floor_friction']:g}", sub="one latent for the whole clip", xml=ADULT,
                             qpos=a.z0, rgba=RED),
                        dict(title=f"LSO latent sequence ({key}), floor friction {s['floor_friction']:g}",
                             sub=(f"{b['speed_under_way']:.2f} m/s under way ({b['speed_ratio']:.2f}x the adult), aligned dtw {b['dtw']:.3f}, "
                                  f"head lowest {b['head_lowest_matched_m']:.2f} m"),
                             xml=ADULT, qpos=str((out / f"lso_{run.name}_{key}.npz").relative_to(REPO)), rgba=BLUE)])
    sp = out / f"spec_lso_{run.name}.json"; sp.write_text(json.dumps(spec, indent=1))
    subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(sp)], cwd=REPO, check=True,
                   env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2"})


if __name__ == "__main__":
    main()
