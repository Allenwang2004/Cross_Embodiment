#!/usr/bin/env python3
"""child_reference_backward_on_adult.py -- the backward z of the CHILD's retargeted motion, run on the ADULT.

For one clip, per-frame z by Metamotivo tracking_inference on the child's reference
(data/child/retargeting_motion), with B fed one of:
  raw          the child's own proprio, unscaled
  multiplier   raw x the fixed per-feature multiplier (model/obs_scale.py)
  exact        the adult-equivalent observation (model/exact_obs.py)
and, as controls, the adult's own backward z (the adult's motion on the adult) and z0. All of these observations
carry finite-difference velocities; the files in data/*/infer_* were made with qvel = 0 (scripts/batch_infer_z.py)
and are run as their own rows.
Each z drives the frozen policy on the ADULT body (raw adult obs) from the adult-equivalent of the child's
first frame; step t uses the z inferred from frame t + 1 (Metamotivo's tracking convention).
Writes outputs/child_ref_backward_on_adult/<clip>/{*.npz, metrics.json, panels.json}.
"""
import argparse, json, sys
from pathlib import Path
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
ADULT = REPO / "assets/robots/adult/robot.xml"
CHILD = REPO / "assets/robots_torque/child/robot_torque_full.xml"
DT = 1 / 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    args = ap.parse_args()
    task, stem = args.clip.split("/")
    out = REPO / f"outputs/child_ref_backward_on_adult/{stem}"; out.mkdir(parents=True, exist_ok=True)
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align, losses
    from model.exact_obs import ExactObs
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    cfg = ESConfig(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = {"root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
         "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight}
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    X = ExactObs(CHILD, ADULT)
    fa = mujoco.MjModel.from_xml_path(str(ADULT)); fc = mujoco.MjModel.from_xml_path(str(CHILD))
    env_a, _ = make_humenv(num_envs=1, task=None, xml=str(ADULT), state_init="Default")
    env_c, _ = make_humenv(num_envs=1, task=None, xml=str(CHILD), state_init="Default")
    mul = build_obs_multiplier(CHILD, ADULT, mode="auto", parts=cfg.obs_scale_parts, verbose=False)
    ref_c = np.load(REPO / f"data/child/retargeting_motion/{task}/{stem}.npz")["qpos"].astype(np.float64)
    adult_motion = np.load(REPO / f"data/origin_motion/{task}/{stem}.npz")["qpos"]
    z0 = np.load(REPO / f"data/origin_z/{task}/{stem}.npy").reshape(-1)

    def ti(obs):
        with torch.no_grad():
            return model.tracking_inference(next_obs=torch.as_tensor(np.asarray(obs, dtype=np.float32), device=dev)).cpu().numpy()

    def roll_adult(zs, init_q):
        env_a.reset(); env_a.unwrapped.set_physics(qpos=init_q, qvel=np.zeros(fa.nv))
        obs = env_a.unwrapped.get_obs()["proprio"]; Q = []
        for t in range(300):
            z = torch.as_tensor(zs[min(t, len(zs) - 1)][None], dtype=torch.float32, device=dev)
            o = torch.as_tensor(obs[None], dtype=torch.float32, device=dev)
            a = model._actor(model._normalize(o), z, model.cfg.actor_std).mean[0].detach().cpu().numpy()
            obs, *_ = env_a.step(a); obs = obs["proprio"]; Q.append(env_a.unwrapped.data.qpos.copy())
        return np.array(Q)

    def roll_child(zs):
        """zs per step -> child qpos (300, nq); the policy sees the adult-equivalent observation (exact)."""
        env_c.reset(); env_c.unwrapped.set_physics(qpos=ref_c[0], qvel=np.zeros(fc.nv))
        q, v, Q = ref_c[0], np.zeros(fc.nv), []
        for t in range(300):
            z = torch.as_tensor(zs[min(t, len(zs) - 1)][None], dtype=torch.float32, device=dev)
            o = torch.as_tensor(X(q, v)[None], dtype=torch.float32, device=dev)
            a = model._actor(model._normalize(o), z, model.cfg.actor_std).mean[0].detach().cpu().numpy()
            env_c.step(a); q, v = env_c.unwrapped.data.qpos.copy(), env_c.unwrapped.data.qvel.copy(); Q.append(q)
        return np.array(Q)

    def cmp(q, target, fk):
        n = min(len(q), len(target))
        return dict(L_align=float(losses.functional_equivalence(fk, q[:n], target[:n], W, DT)[0]),
                    joint_deg=float(np.degrees(np.abs(q[:n, 7:] - target[:n, 7:])).mean()),
                    end_xy_m=float(np.linalg.norm(q[n - 1, :2] - target[n - 1, :2])))

    # the child reference's observations three ways
    raw = bfm_align.obs_from_qpos(env_c, ref_c)
    v = np.zeros(fc.nv); exact = []
    for t in range(len(ref_c)):
        if t:
            mujoco.mj_differentiatePos(fc, v, DT, ref_c[t - 1], ref_c[t])
        exact.append(X(ref_c[t], v))
    stored_child = np.load(REPO / f"data/child/infer_retargeting_z/{task}/{stem}.npy")
    Z = {"raw": ti(raw[1:]), "multiplier": ti((raw * mul)[1:]), "exact": ti(np.array(exact)[1:]),
         "adult own backward z": ti(bfm_align.obs_from_qpos(env_a, adult_motion)[1:]),
         "stored child z (qvel=0)": stored_child[1:],
         "stored adult z (qvel=0)": np.load(REPO / f"data/infer_origin_z/{task}/{stem}.npy")[1:],
         "z0": z0[None]}
    n = min(len(stored_child) - 1, len(Z["raw"]))
    print(f"check: recomputed raw (with velocities) vs stored data/child/infer_retargeting_z (qvel=0), mean cos "
          f"{np.mean(np.sum(stored_child[1:n + 1] * Z['raw'][:n], 1) / 256):+.3f}")
    adult_own = Z["adult own backward z"]
    init_adult, _ = X.adult_state(ref_c[0], np.zeros(fc.nv))
    metrics, files = {}, {}
    for name, zs in Z.items():
        q = roll_adult(zs, init_adult)
        tag = name.replace(" ", "_")
        np.savez(out / f"adult_{tag}.npz", qpos=q.astype(np.float32), fps=30); files[name] = out / f"adult_{tag}.npz"
        m = cmp(q, adult_motion, fa)
        m["cos_to_adult_own_backward_z"] = float(np.mean(np.sum(zs[:min(len(zs), len(adult_own))] * adult_own[:min(len(zs), len(adult_own))], 1) / 256)) if len(zs) > 1 else float(np.mean(adult_own @ z0) / 256)
        metrics[name] = m
    qc = roll_child(Z["exact"])
    np.savez(out / "child_exact.npz", qpos=qc.astype(np.float32), fps=30); files["child exact"] = out / "child_exact.npz"
    metrics["exact z on the CHILD (vs child reference)"] = dict(cmp(qc, ref_c, fc), cos_to_adult_own_backward_z=1.0)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    print(f"{'z fed to the policy on the ADULT':28s} | vs adult's original motion: L_align  joint diff  end position | cos to adult's own backward z")
    for k, m in metrics.items():
        print(f"{k:28s} | {m['L_align']:27.3f}  {m['joint_deg']:8.1f} deg  {m['end_xy_m']:6.2f} m      | {m['cos_to_adult_own_backward_z']:+.3f}")
    rel = lambda p: str(Path(p).relative_to(REPO))
    panels = {"out": rel(out / "child_ref_backward_on_adult.mp4"), "cols": 3, "size": 320, "camera": "front_side",
              "track": {"distance": 4.2, "elevation": -12, "azimuth": 135, "lookat_z": 0.75}, "panels": [
        {"title": "adult original motion", "sub": "the target", "xml": rel(ADULT), "qpos": rel(REPO / f"data/origin_motion/{task}/{stem}.npz")},
        {"title": "child reference", "sub": "retargeted, on the child", "xml": rel(CHILD), "qpos": rel(REPO / f"data/child/retargeting_motion/{task}/{stem}.npz")},
        {"title": "adult, adult's own backward z", "sub": "control", "xml": rel(ADULT), "qpos": rel(files["adult own backward z"])},
        {"title": "adult, child-ref backward z (raw obs)", "sub": "", "xml": rel(ADULT), "qpos": rel(files["raw"])},
        {"title": "CHILD, child-ref backward z (exact)", "sub": "same z, on the child", "xml": rel(CHILD), "qpos": rel(files["child exact"])},
        {"title": "adult, child-ref backward z (exact)", "sub": "", "xml": rel(ADULT), "qpos": rel(files["exact"])}]}
    json.dump(panels, open(out / "panels.json", "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
