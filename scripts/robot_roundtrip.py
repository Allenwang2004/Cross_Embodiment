#!/usr/bin/env python3
"""robot_roundtrip.py -- scripts/g1_roundtrip.py for any robot in model/robots.py (model/robot_map.py), many clips at once.

Per clip: SMPL qpos -> robot qpos (smpl_to_robot_qpos, warm-started) -> back to SMPL (robot_to_smpl_qpos), and
  IK residual   per paired robot link (deg, mean / max over frames)
  limits        share of frames each robot joint sits at an end of its range
  bfm floor     1 - mean_t cos(B(round trip), B(original)): the best bfm any controller on the robot could reach
Writes <out_dir>/roundtrip/<stem>/{robot.npz, smpl_rt.npz, summary.json} (model/robots.py out_dir, target_path) and,
with --render, a 1080 px video (adult original / robot mapped / adult after the round trip).
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/robot_roundtrip.py --robot h1 \\
         [--clips-file outputs/g1/search_clips.txt | --clip task/stem ...] [--render]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.robot_map import RobotMap, SMPL_XML          # noqa: E402
from model.robots import ROBOTS, out_dir, target_path    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--clip", action="append", default=[])
    ap.add_argument("--clips-file", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    clips = a.clip + ([c for c in (REPO / a.clips_file).read_text().split() if c] if a.clips_file else [])
    M = RobotMap(a.robot); cfg = ROBOTS[a.robot]
    print(f"{cfg['name']}: s (robot / SMPL leg length) = {M.s:.3f}")
    print("calibration residual (deg): " + ", ".join(f"{k} {v:.1f}" for k, v in M.calib.items()))
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    env, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / SMPL_XML), state_init="Default")
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device)
    links = [g for _, g in M.pairs[1:]]
    for clip in clips:
        stem = clip.split("/")[1]
        tgt = target_path(a.robot, stem); out = tgt.parent; out.mkdir(parents=True, exist_ok=True)
        ref = np.load(REPO / f"data/origin_motion/{clip}.npz")["qpos"]
        T = len(ref); qg = np.zeros((T, M.mg.nq)); rt = np.zeros_like(ref); resid = np.zeros((T, len(links)))
        t0 = time.time(); q = None
        for t in range(T):
            qg[t], resid[t] = M.smpl_to_robot_qpos(ref[t], q); q = qg[t][M.g_qadr]
        ms_ik = 1e3 * (time.time() - t0) / T
        for t in range(T):
            rt[t] = M.robot_to_smpl_qpos(qg[t])
        np.savez_compressed(tgt, qpos=qg, fps=30)
        np.savez_compressed(out / "smpl_rt.npz", qpos=rt, fps=30)
        lo, hi = M.g_range[:, 0], M.g_range[:, 1]
        jq = qg[:, M.g_qadr]
        at = (np.abs(jq - lo) < 1e-4) | (np.abs(jq - hi) < 1e-4)
        lim = {M.mg.joint(j).name: float(at[:, k].mean()) for k, j in enumerate(M.g_jnt) if at[:, k].any()}
        B0 = bfm_align.embed(model, bfm_align.obs_from_qpos(env, ref), a.device)
        B1 = bfm_align.embed(model, bfm_align.obs_from_qpos(env, rt), a.device)
        floor = bfm_align.bfm_align_loss(B1, B0)
        summ = dict(robot=a.robot, clip=clip, s=M.s, calib_deg=M.calib,
                    q_nom_deg={M.mg.joint(j).name: float(np.degrees(x)) for j, x in zip(M.g_jnt, M.q_nom)},
                    ik_resid_deg={g: [float(resid[:, i].mean()), float(resid[:, i].max())] for i, g in enumerate(links)},
                    at_limit=lim, bfm_floor=floor, ms_per_frame_ik=ms_ik)
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        print(f"{stem:30s} floor {floor:.3f} | IK residual mean {resid.mean():5.1f} deg (worst link "
              f"{links[int(resid.mean(0).argmax())]} {resid.mean(0).max():.1f}) | {ms_ik:.1f} ms/frame", flush=True)
        if a.render:
            rel = lambda p: str(p.relative_to(REPO))
            spec = {"out": rel(out / f"{stem}_{a.robot}_kinematic.mp4"), "cols": 3, "size": 1080,
                    "track": {"distance": 3.6, "elevation": -12, "azimuth": 140, "lookat_z": 0.8},
                    "panels": [
                        {"title": "adult (SMPL), original", "sub": clip, "xml": SMPL_XML, "qpos": f"data/origin_motion/{clip}.npz"},
                        {"title": f"{cfg['name']}, mapped", "sub": "kinematic, no physics", "xml": cfg["scene"], "qpos": rel(tgt)},
                        {"title": "adult after round trip", "sub": f"SMPL -> robot -> SMPL, bfm floor {floor:.3f}",
                         "xml": SMPL_XML, "qpos": rel(out / "smpl_rt.npz")}]}
            (out / "spec.json").write_text(json.dumps(spec, indent=1))
            subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(out / "spec.json")], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
