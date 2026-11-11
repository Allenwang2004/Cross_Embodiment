#!/usr/bin/env python3
"""robot_long_rollout.py -- a robot_z_search.py result rolled out for longer than its clip: z0 and the searched z, each
held for --steps control steps (the clip's own length is where the search's cost stopped), from the clip's frame 0
mapped onto the robot at rest, exactly as the search rolled them out (model/robot_env.RobotVecEnv, same fill and gains).
Logged per second: the pelvis height and, where the robot has one (Apollo's collision_head), the head's lowest point.
Video as the survey's side by side (scripts/survey_robot_videos.py: no captions, the adult grey -- it holds its last
frame once its clip ends -- z0 red, searched z blue, one camera for the taller of the adult and the robot).
Writes <out-dir>/<stem>_<steps>f_{z0,best}.npz, <stem>_<steps>f.json and <stem>_<steps>f_side_by_side.mp4 (+ .png).
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/robot_long_rollout.py --robot apollo --clip headstand/headstand_4 \
           --search outputs/user_survey_search/apollo_ridge_head_zerodamp/headstand_4_s0 --steps 300 \
           [--gains ...] [--fill ridge] [--out-dir outputs/<robot>/long_rollout]"""
import argparse, json, os, subprocess, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from single_z_search import project_z                                      # noqa: E402
from model.robot_env import RobotVecEnv                                    # noqa: E402
from model.robots import ROBOTS, out_dir, target_path                       # noqa: E402
from survey_body_videos import ADULT, ADULT_H, BLUE, GREY, RED              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--clip", required=True, help="task/stem")
    ap.add_argument("--search", required=True, help="the robot_z_search output dir (best_z.npy, summary.json)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fill", default=None, help="default: the search's")
    ap.add_argument("--gains", default=None, help="default: the search's")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    task, stem = a.clip.split("/"); srch = REPO / a.search; ss = json.loads((srch / "summary.json").read_text())
    fill, gains = a.fill or ss.get("fill", "fixed"), a.gains or ss.get("gains")
    out = Path(a.out_dir) if a.out_dir else out_dir(a.robot) / "long_rollout"; out.mkdir(parents=True, exist_ok=True)
    vec = RobotVecEnv(a.robot, 2, fill=fill, gains=gains)                  # spawn the workers before CUDA is touched
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device); model.eval()
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
    Z = np.stack([project_z(np.load(REPO / f"data/origin_z/{a.clip}.npy").reshape(-1).astype(np.float64)),
                  np.load(srch / "best_z.npy").reshape(-1).astype(np.float64)])
    zt = torch.as_tensor(Z, dtype=torch.float32, device=a.device)
    obs, _, _, _ = vec.reset(ref[0]); QG = []
    with torch.no_grad():
        for _ in range(a.steps):
            u = model._actor(model._normalize(torch.as_tensor(obs, dtype=torch.float32, device=a.device)), zt,
                             model.cfg.actor_std).mean.cpu().numpy()
            obs, _, qg, _ = vec.step(u); QG.append(qg)
    vec.close()
    QG = np.stack(QG, 1)                                                    # (2, steps, nq)
    m = mujoco.MjModel.from_xml_path(str(REPO / ROBOTS[a.robot]["scene"])); d = mujoco.MjData(m)
    head = None
    try:
        g = m.geom("collision_head").id; mid = m.geom_dataid[g]
        head = (g, m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]][::20])
    except KeyError:
        pass
    saved = np.load(srch / "best.npz")["qpos"]; n0 = min(len(saved), a.steps)
    rep = dict(robot=a.robot, clip=a.clip, search=a.search, steps=a.steps, clip_frames=len(ref), fill=fill, gains=gains,
               best_first_frames_vs_search_max_abs_qpos_diff=float(np.abs(QG[1, :n0] - saved[:n0]).max()), per_second={})
    for i, key in enumerate(("z0", "best")):
        np.savez_compressed(out / f"{stem}_{a.steps}f_{key}.npz", qpos=QG[i].astype(np.float32), fps=np.float32(30))
        rows = []
        for t0 in range(0, a.steps, 30):
            seg = QG[i, t0:t0 + 30]; hz = []
            if head:
                for q in seg:
                    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
                    hz.append((d.geom_xpos[head[0]] + head[1] @ d.geom_xmat[head[0]].reshape(3, 3).T)[:, 2].min())
            rows.append(dict(second=t0 // 30, pelvis_z=[round(float(seg[:, 2].min()), 3), round(float(seg[:, 2].max()), 3)],
                             head_lowest=round(float(min(hz)), 3) if hz else None))
        rep["per_second"][key] = rows
    (out / f"{stem}_{a.steps}f.json").write_text(json.dumps(rep, indent=1))
    size = json.loads((target_path(a.robot, stem).parent / "summary.json").read_text())["s"] / 0.8213
    k = max(size * 1.32, ADULT_H) / 1.26
    rel = lambda p: str(p.relative_to(REPO))
    spec = dict(out=rel(out / f"{stem}_{a.steps}f_side_by_side.mp4"), cols=3, size=1080, captions=False,
                track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)),
                panels=[dict(xml=ADULT, qpos=f"data/origin_motion/{a.clip}.npz", rgba=GREY),
                        dict(xml=ROBOTS[a.robot]["scene"], qpos=rel(out / f"{stem}_{a.steps}f_z0.npz"), rgba=RED),
                        dict(xml=ROBOTS[a.robot]["scene"], qpos=rel(out / f"{stem}_{a.steps}f_best.npz"), rgba=BLUE)])
    sp = out / f"spec_{stem}_{a.steps}f.json"; sp.write_text(json.dumps(spec, indent=1))
    subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(sp)], cwd=REPO, check=True,
                   env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2", "CUDA_VISIBLE_DEVICES": "2"})
    for key in ("z0", "best"):
        print(key, " | ".join(f"{r['second']}s pelvis {r['pelvis_z'][0]:.2f}-{r['pelvis_z'][1]:.2f}"
                              + (f" head {r['head_lowest']:+.2f}" if r["head_lowest"] is not None else "") for r in rep["per_second"][key]))
    print(f"first {n0} frames of the searched z vs the search's own rollout: max |qpos diff| {rep['best_first_frames_vs_search_max_abs_qpos_diff']:.2e}")
    print(f"-> {spec['out']}")


if __name__ == "__main__":
    main()
