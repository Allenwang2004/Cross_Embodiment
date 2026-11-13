#!/usr/bin/env python3
"""g1_z0_rollout.py -- Meta Motivo, unchanged, driving the Unitree G1 with one latent (model/g1_env.py).

One deterministic rollout (the actor's mean action) of --z (default z0, the latent that produced the clip on the
adult) on the G1, starting from the clip's frame 0 mapped onto the G1 at rest, for as many steps as the clip has.
Reports
  bfm       1 - mean_t cos(B(obs_t), B(g_t)): the G1's SMPL-equivalent observations against the adult's original
            clip (obs from qpos at 30 Hz, bfm_align.obs_from_qpos), the cost single_z_search optimises.
            Read against the clip's kinematic floor of the G1 map (scripts/g1_roundtrip.py; 0.176 on the walk).
  motion    against the clip mapped onto the G1 kinematically (outputs/g1/roundtrip/<stem>/g1.npz): fallen frames
            (as strength_search_summary.py), travel, end-point distance, mean pelvis height.
--check-adult runs the same loop on the adult itself (humenv, same start) as a check of the loop: its bfm should be
near 0. Writes outputs/g1/z0_perjoint/<stem>/{rollout.npz, summary.json} (--gains mass: outputs/g1/z0/<stem>/) and,
with --render, a video (adult original / G1 mapped kinematically / G1 under Motivo).
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/g1_z0_rollout.py [--clip ...] [--render]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.g1_env import G1Env, G1_SCENE   # noqa: E402
from model import bfm_align             # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment.geometry import project_z   # noqa: E402  single_z_search re-exported it (refactor step 4)


# act, g1_up_z and motion live in crossembodiment/robot_rollout.py (refactor step 4, unchanged); imported here so this
# script and `from g1_z0_rollout import act, motion` keep working
from crossembodiment.robot_rollout import act, g1_up_z, motion   # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    ap.add_argument("--z", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--check-adult", action="store_true")
    ap.add_argument("--gains", default="perjoint", choices=["perjoint", "mass"], help="G1 PD gains, see model/g1_env.py")
    a = ap.parse_args()
    task, stem = a.clip.split("/")
    out = REPO / ("outputs/g1/z0" if a.gains == "mass" else f"outputs/g1/z0_{a.gains}") / stem
    out.mkdir(parents=True, exist_ok=True)
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device)
    # z0 re-projected onto Motivo's sphere exactly as g1_z_search does (the stored z0 differs from that in the 7th digit)
    z = np.load(a.z).reshape(1, -1) if a.z else project_z(np.load(REPO / f"data/origin_z/{a.clip}.npy").reshape(-1).astype(np.float64))[None]
    zt = torch.as_tensor(z, dtype=torch.float32, device=a.device)

    env = G1Env(gains=a.gains)
    print(f"G1 gains ({a.gains}), kp/kd:")
    print("  " + ", ".join(f"{env.m.actuator(i).name.replace('_joint', '')} {env.kp[i]:.0f}/{env.kd[i]:.1f}"
                            for i in range(env.m.nu) if not env.m.actuator(i).name.startswith("right")))
    Bg = bfm_align.embed(model, bfm_align.obs_from_qpos(env.adult, ref), a.device)

    if a.check_adult:
        ad = env.adult.unwrapped
        ad.set_physics(qpos=ref[0], qvel=np.zeros(ad.model.nv)); o = ad.get_obs()["proprio"]; obs_a = []
        for _ in range(len(ref)):
            o = ad.step(act(model, o, zt, a.device))[0]["proprio"]; obs_a.append(o)
        print(f"check: same loop on the adult, bfm {bfm_align.bfm_align_loss(bfm_align.embed(model, np.array(obs_a, np.float32), a.device), Bg):.4f}")

    obs, _ = env.reset(ref[0])
    qg, qs, us, os_ = [], [], [], []
    t0 = time.time()
    for t in range(len(ref)):
        u = act(model, obs, zt, a.device)
        obs, q_s = env.step(u)
        qg.append(env.d.qpos.copy()); qs.append(q_s); us.append(u); os_.append(obs)
    sec = time.time() - t0
    qg, qs, us, os_ = map(np.array, (qg, qs, us, os_))
    bfm = bfm_align.bfm_align_loss(bfm_align.embed(model, os_, a.device), Bg)
    np.savez_compressed(out / "rollout.npz", qpos=qg, qpos_smpl=qs, act=us, obs=os_, fps=30)

    kin = REPO / f"outputs/g1/roundtrip/{stem}/g1.npz"
    if not kin.exists():
        subprocess.run(["uv", "run", "python", "scripts/g1_roundtrip.py", "--clip", a.clip], cwd=REPO, check=True)
    gref = np.load(kin)["qpos"]
    floor = json.loads((kin.parent / "summary.json").read_text())["bfm_floor"]
    mo = motion(qg, gref)
    print(f"\nG1 under Motivo, {'z0' if a.z is None else a.z}: {len(ref)} steps in {sec:.1f} s")
    print(f"  bfm {bfm:.4f}   (kinematic floor of the map on this clip {floor:.3f})")
    print(f"  fallen frames {mo['fallen']}" + (f" (first at step {mo['first_fall']})" if mo['first_fall'] is not None else "")
          + f" | travel {mo['travel']:.1f} m (mapped clip {mo['ref_travel']:.1f} m) | end-point off {mo['end_xy']:.2f} m"
          f" | pelvis {mo['pelvis']:.3f} m (mapped clip {mo['ref_pelvis']:.3f} m)")
    summ = dict(clip=a.clip, z=a.z or "z0", gains=a.gains, bfm=bfm, bfm_floor=floor, motion=mo, seconds=sec,
                kp={env.m.actuator(i).name: float(env.kp[i]) for i in range(env.m.nu)},
                kd={env.m.actuator(i).name: float(env.kd[i]) for i in range(env.m.nu)})
    (out / "summary.json").write_text(json.dumps(summ, indent=1))

    if a.render:
        rel = lambda p: str(p.relative_to(REPO))
        spec = {"out": rel(out / f"{stem}_g1_z0.mp4"), "cols": 3, "size": 1080,
                "track": {"distance": 3.6, "elevation": -12, "azimuth": 140, "lookat_z": 0.8},
                "panels": [
                    {"title": "adult (SMPL), original", "sub": a.clip, "xml": "assets/robots/adult/robot.xml",
                     "qpos": f"data/origin_motion/{a.clip}.npz"},
                    {"title": "G1, clip mapped kinematically", "sub": "the target, no physics", "xml": G1_SCENE,
                     "qpos": rel(kin)},
                    {"title": "G1 under Motivo, z0", "sub": f"bfm {bfm:.3f}, fallen {mo['fallen']} frames",
                     "xml": G1_SCENE, "qpos": rel(out / "rollout.npz")}]}
        (out / "spec.json").write_text(json.dumps(spec, indent=1))
        subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(out / "spec.json")], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
