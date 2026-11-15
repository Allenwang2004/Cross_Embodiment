#!/usr/bin/env python3
"""robot_z0.py -- scripts/g1_z0_rollout.py (archived in OOP refactor step 7) for any robot in model/robots.py
(model/robot_env.py), many clips at once.

Per clip one deterministic rollout of the clip's z0 (re-projected onto Motivo's sphere and the actor batched by 16, as
g1_z_search did), from frame 0 mapped onto the robot at rest, for as many steps as the clip has:
  bfm     1 - mean_t cos(B(obs_t), B(g_t)) against the adult's original clip; read against the clip's kinematic floor
  motion  against the clip mapped onto the robot kinematically: fallen frames, travel, end-point distance, pelvis height
Writes <out_dir>/z0/<stem>/{rollout.npz, summary.json} (the G1: outputs/g1/z0_perjoint/<stem>, g1_z0_rollout's folder) and,
with --render, a 1080 px video (adult original / robot mapped kinematically / robot under z0).
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/robot_z0.py --robot h1 \\
         [--clips-file outputs/robots_clips.txt | --clip task/stem ...] [--render]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model import bfm_align                                    # noqa: E402
from model.robot_env import RobotEnv                           # noqa: E402
from model.robots import ROBOTS, out_dir, target_path          # noqa: E402
from crossembodiment.geometry import project_z   # noqa: E402  single_z_search re-exported it (refactor step 4)
from crossembodiment.robot_rollout import act, motion   # noqa: E402  g1_z0_rollout's (refactor step 4)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--clip", action="append", default=[])
    ap.add_argument("--clips-file", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gains", default=None, help="gains json (default ROBOTS[robot]['gains_json'])")
    ap.add_argument("--tag", default=None, help="output folder under the robot's outputs (default z0)")
    a = ap.parse_args()
    cfg = ROBOTS[a.robot]
    clips = a.clip + ([c for c in (REPO / a.clips_file).read_text().split() if c] if a.clips_file else [])
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device)
    env = RobotEnv(a.robot, gains=a.gains)
    print(f"{cfg['name']} gains (kp/kd): " + ", ".join(f"{env.m.actuator(i).name} {env.kp[i]:.0f}/{env.kd[i]:.1f}"
                                                     for i in range(env.m.nu)), flush=True)
    for clip in clips:
        stem = clip.split("/")[1]
        out = out_dir(a.robot) / (a.tag or ("z0_perjoint" if a.robot == "g1" else "z0")) / stem
        out.mkdir(parents=True, exist_ok=True)
        ref = np.load(REPO / f"data/origin_motion/{clip}.npz")["qpos"]
        z0 = project_z(np.load(REPO / f"data/origin_z/{clip}.npy").reshape(-1).astype(np.float64))[None]
        zt = torch.as_tensor(z0, dtype=torch.float32, device=a.device)
        Bg = bfm_align.embed(model, bfm_align.obs_from_qpos(env.adult, ref), a.device)
        obs, _ = env.reset(ref[0]); Q, QS, O, FZ = [], [], [], []
        t0 = time.time()
        for _ in range(len(ref)):
            obs, qs = env.step(act(model, obs, zt, a.device))
            Q.append(env.d.qpos.copy()); QS.append(qs); O.append(obs); FZ.append(np.append(env.last_resid, env.last_atlim))
        Q, QS, O, FZ = np.array(Q), np.array(QS), np.array(O, np.float32), np.array(FZ)
        bfm = bfm_align.bfm_align_loss(bfm_align.embed(model, O, a.device), Bg)
        np.savez_compressed(out / "rollout.npz", qpos=Q, qpos_smpl=QS, obs=O, fz=FZ, fps=30)
        tgt = target_path(a.robot, stem); gref = np.load(tgt)["qpos"]
        floor = json.loads((tgt.parent / "summary.json").read_text())["bfm_floor"]
        mo = motion(Q, gref)
        summ = dict(robot=a.robot, clip=clip, z="z0", gains=a.gains or cfg["gains_json"], bfm=bfm, bfm_floor=floor, motion=mo,
                    ik_cmd_deg=float(FZ[:, :-1].mean()), seconds=time.time() - t0,
                    kp={env.m.actuator(i).name: float(env.kp[i]) for i in range(env.m.nu)},
                    kd={env.m.actuator(i).name: float(env.kd[i]) for i in range(env.m.nu)})
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        print(f"{stem:30s} bfm {bfm:.3f} (floor {floor:.3f}) | fallen {mo['fallen']:3d} | end-point off {mo['end_xy']:.2f} m | "
              f"pelvis {mo['pelvis']:.2f} m (target {mo['ref_pelvis']:.2f})", flush=True)
        if a.render:
            rel = lambda p: str(p.relative_to(REPO))
            spec = {"out": rel(out / f"{stem}_{a.robot}_z0.mp4"), "cols": 3, "size": 1080,
                    "track": {"distance": 3.6, "elevation": -12, "azimuth": 140, "lookat_z": 0.8},
                    "panels": [
                        {"title": "adult (SMPL), original", "sub": clip, "xml": "assets/robots/adult/robot.xml",
                         "qpos": f"data/origin_motion/{clip}.npz"},
                        {"title": f"{cfg['name']}, clip mapped kinematically", "sub": f"the target, no physics; floor {floor:.3f}",
                         "xml": cfg["scene"], "qpos": rel(tgt)},
                        {"title": f"{cfg['name']} under Motivo, z0", "sub": f"bfm {bfm:.3f}, fallen {mo['fallen']}, "
                         f"end-point off {mo['end_xy']:.1f} m", "xml": cfg["scene"], "qpos": rel(out / "rollout.npz")}]}
            (out / "spec.json").write_text(json.dumps(spec, indent=1))
            subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(out / "spec.json")], cwd=REPO,
                           check=True, capture_output=True)


if __name__ == "__main__":
    main()
