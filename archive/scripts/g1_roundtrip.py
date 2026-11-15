#!/usr/bin/env python3
"""g1_roundtrip.py -- how much of an SMPL motion survives the trip through the Unitree G1 (model/g1_map.py)?

Kinematic only (no physics, no policy). For one reference clip (the adult's own motion, data/origin_motion):
  SMPL qpos -> G1 qpos (smpl_to_g1_qpos, warm-started frame to frame) -> back to SMPL (g1_to_smpl_qpos)
and reports
  calibration   residual of the nominal-pose match per segment direction (deg)
  IK residual   per paired G1 link, how far its orientation stays from the SMPL segment's (deg): what the G1's
                joints cannot reproduce
  round trip    per SMPL body, orientation error of the round-tripped pose vs the original (deg); hands, feet and
                head position error relative to the pelvis (cm, on the adult skeleton)
  limits        share of frames each G1 joint sits at an end of its range
  bfm floor     1 - mean_t cos(B(round trip), B(original)) with Motivo's backward map, obs from qpos at 30 Hz
                (bfm_align.obs_from_qpos): the best bfm cost any controller on the G1 could reach through this map
Writes outputs/g1/roundtrip/<stem>/{g1.npz, smpl_rt.npz, summary.json} and, with --render, a side-by-side video
(adult original / G1 / adult after the round trip) via render_panels.py.
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/g1_roundtrip.py [--clip ...] [--render]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.g1_map import G1Map, PAIRS   # noqa: E402

G1_SCENE = "assets/robots/g1/scene.xml"          # the G1 in the adult's surroundings, so the panels match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    stem = a.clip.split("/")[1]
    out = REPO / "outputs/g1/roundtrip" / stem; out.mkdir(parents=True, exist_ok=True)
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
    M = G1Map()
    print(f"s (G1 / SMPL leg length) = {M.s:.3f}")
    print("calibration residual (deg): " + ", ".join(f"{k} {v:.1f}" for k, v in M.calib.items()))

    T = len(ref); qg = np.zeros((T, M.mg.nq)); rt = np.zeros_like(ref); resid = np.zeros((T, len(PAIRS) - 1))
    t0 = time.time(); q = None
    for t in range(T):
        qg[t], resid[t] = M.smpl_to_g1_qpos(ref[t], q)
        q = qg[t][M.g_qadr]
    t_ik = (time.time() - t0) / T
    t0 = time.time()
    for t in range(T):
        rt[t] = M.g1_to_smpl_qpos(qg[t])
    t_back = (time.time() - t0) / T
    np.savez_compressed(out / "g1.npz", qpos=qg, fps=30)
    np.savez_compressed(out / "smpl_rt.npz", qpos=rt, fps=30)

    links = [g for _, g in PAIRS[1:]]
    print(f"\nIK residual per G1 link (deg, mean / max over frames)   [{1e3 * t_ik:.1f} ms/frame; back {1e3 * t_back:.1f} ms]")
    for i, g in enumerate(links):
        print(f"  {g:26s} {resid[:, i].mean():6.2f} / {resid[:, i].max():6.2f}")

    ms, ds = M.ms, mujoco.MjData(M.ms)
    def fk(qs):
        R, P = [], []
        for x in qs:
            ds.qpos[:] = x; mujoco.mj_kinematics(ms, ds)
            R.append(ds.xmat[1:].reshape(-1, 3, 3).copy()); P.append(ds.xpos[1:].copy())
        return np.array(R), np.array(P)
    R0, P0 = fk(ref); R1, P1 = fk(rt)
    names = [ms.body(i).name for i in range(1, ms.nbody)]
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(
        np.einsum("tbji,tbjk->tbik", R0, R1).reshape(-1, 3, 3)).as_rotvec(), axis=1)).reshape(T, -1)
    print("\nround trip: SMPL body orientation error (deg, mean / max)")
    for i, n in enumerate(names):
        print(f"  {n:11s} {ang[:, i].mean():6.2f} / {ang[:, i].max():6.2f}")
    pos = {}
    for n in ("L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"):
        i = names.index(n)
        e = np.linalg.norm((P1[:, i] - P1[:, 0]) - (P0[:, i] - P0[:, 0]), axis=1) * 100
        pos[n] = (float(e.mean()), float(e.max()))
    print("round trip: position error relative to the pelvis (cm, mean / max): "
          + ", ".join(f"{k} {v[0]:.1f} / {v[1]:.1f}" for k, v in pos.items()))
    print(f"root height: original mean {ref[:, 2].mean():.3f} m, round trip {rt[:, 2].mean():.3f} m, "
          f"G1 pelvis {qg[:, 2].mean():.3f} m")

    lo, hi = M.g_range[:, 0], M.g_range[:, 1]
    jq = qg[:, M.g_qadr]
    at = (np.abs(jq - lo) < 1e-4) | (np.abs(jq - hi) < 1e-4)
    lim = {M.mg.joint(j).name: float(at[:, k].mean()) for k, j in enumerate(M.g_jnt) if at[:, k].any()}
    print("G1 joints at a range end (share of frames): " + (", ".join(f"{k} {v:.0%}" for k, v in lim.items()) or "none"))

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    env, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device)
    B0 = bfm_align.embed(model, bfm_align.obs_from_qpos(env, ref), a.device)
    B1 = bfm_align.embed(model, bfm_align.obs_from_qpos(env, rt), a.device)
    floor = bfm_align.bfm_align_loss(B1, B0)
    print(f"\nbfm floor (round trip vs original): {floor:.4f}")

    summ = dict(clip=a.clip, s=M.s, calib_deg=M.calib, q_nom_deg={M.mg.joint(j).name: float(np.degrees(x))
                for j, x in zip(M.g_jnt, M.q_nom)},
                ik_resid_deg={g: [float(resid[:, i].mean()), float(resid[:, i].max())] for i, g in enumerate(links)},
                rt_orient_deg={n: [float(ang[:, i].mean()), float(ang[:, i].max())] for i, n in enumerate(names)},
                rt_pos_cm=pos, at_limit=lim, bfm_floor=floor, ms_per_frame_ik=1e3 * t_ik)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))

    if a.render:
        rel = lambda p: str(p.relative_to(REPO))
        spec = {"out": rel(out / f"{stem}_g1_kinematic.mp4"), "cols": 3, "size": 1080,
                "track": {"distance": 3.6, "elevation": -12, "azimuth": 140, "lookat_z": 0.8},
                "panels": [
                    {"title": "adult (SMPL), original", "sub": a.clip, "xml": "assets/robots/adult/robot.xml",
                     "qpos": f"data/origin_motion/{a.clip}.npz"},
                    {"title": "Unitree G1, mapped", "sub": "kinematic, no physics", "xml": G1_SCENE,
                     "qpos": rel(out / "g1.npz")},
                    {"title": "adult after round trip", "sub": f"SMPL -> G1 -> SMPL, bfm floor {floor:.3f}",
                     "xml": "assets/robots/adult/robot.xml", "qpos": rel(out / "smpl_rt.npz")}]}
        (out / "spec.json").write_text(json.dumps(spec, indent=1))
        subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(out / "spec.json")], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
