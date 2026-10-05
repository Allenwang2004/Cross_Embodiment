#!/usr/bin/env python3
"""child_to_adult_backward.py -- what the child DID, turned back into an adult latent and run on the adult.

For one clip, take recorded child rollouts (single_z_search's origin_z.npz = z0's rollout, best.npz = the
searched z's), and for each:
  1. every child state -> the adult-equivalent state (model/exact_obs.py: joint angles kept, root / 0.611,
     ground clearance kept) -> the adult's observation;
  2. Metamotivo tracking_inference on those observations -> a per-frame z;
  3. that z sequence drives the frozen policy on the ADULT body (raw obs), starting from the adult-equivalent
     of the child's start pose.
The adult's rollout is compared with what the child did (in adult units) and with the adult's original motion;
z0 on the adult is the control. Writes outputs/child_to_adult/<clip>/{*.npz, metrics.json, panels.json}.
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
    ap.add_argument("--search", default=None, help="search dir with origin_z.npz / best.npz (default b500_targets/<stem>/align)")
    args = ap.parse_args()
    task, stem = args.clip.split("/")
    src = Path(args.search) if args.search else REPO / f"outputs/b500_targets/{stem}/align"
    out = REPO / f"outputs/child_to_adult/{stem}"; out.mkdir(parents=True, exist_ok=True)
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import losses
    from model.exact_obs import ExactObs
    from model.simple.config import ESConfig
    cfg = ESConfig(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = {"root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
         "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight}
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    X = ExactObs(CHILD, ADULT)
    fa = mujoco.MjModel.from_xml_path(str(ADULT)); fc = mujoco.MjModel.from_xml_path(str(CHILD))
    env, _ = make_humenv(num_envs=1, task=None, xml=str(ADULT), state_init="Default")
    ref_c = np.load(REPO / f"data/child/retargeting_motion/{task}/{stem}.npz")["qpos"]
    adult_motion = np.load(REPO / f"data/origin_motion/{task}/{stem}.npz")["qpos"]
    z0 = np.load(REPO / f"data/origin_z/{task}/{stem}.npy").reshape(-1)

    def to_adult(qc):
        """child qpos sequence -> adult-equivalent qpos and observation sequences."""
        qa, ob = [], []
        v = np.zeros(fc.nv)
        for t in range(len(qc)):
            if t:
                mujoco.mj_differentiatePos(fc, v, DT, qc[t - 1], qc[t])
            a, va = X.adult_state(qc[t], v)
            qa.append(a); ob.append(X(qc[t], v))
        return np.array(qa), np.array(ob, dtype=np.float32)

    def roll_adult(zs, init_q):
        """zs (T, 256) per step (last held) -> adult qpos (300, nq); raw adult obs."""
        env.reset(); env.unwrapped.set_physics(qpos=init_q, qvel=np.zeros(fa.nv))
        obs = env.unwrapped.get_obs()["proprio"]; Q = []
        for t in range(300):
            z = torch.as_tensor(zs[min(t, len(zs) - 1)][None], dtype=torch.float32, device=dev)
            o = torch.as_tensor(obs[None], dtype=torch.float32, device=dev)
            a = model._actor(model._normalize(o), z, model.cfg.actor_std).mean[0].detach().cpu().numpy()
            obs, *_ = env.step(a); obs = obs["proprio"]; Q.append(env.unwrapped.data.qpos.copy())
        return np.array(Q)

    def joint_deg(a, b):
        n = min(len(a), len(b)); return float(np.degrees(np.abs(a[:n, 7:] - b[:n, 7:])).mean())

    def l_align(a, b, fk):
        n = min(len(a), len(b)); return float(losses.functional_equivalence(fk, a[:n], b[:n], W, DT)[0])

    init_adult, _ = X.adult_state(ref_c[0], np.zeros(fc.nv))
    metrics = {}
    q_z0_adult = roll_adult(z0[None], init_adult)
    np.savez(out / "adult_z0.npz", qpos=q_z0_adult.astype(np.float32), fps=30)
    metrics["z0 on adult"] = dict(vs_adult_motion_L_align=l_align(q_z0_adult, adult_motion, fa),
                                  vs_adult_motion_joint_deg=joint_deg(q_z0_adult, adult_motion))
    for name, f in (("child z0 rollout", "origin_z"), ("child searched rollout", "best")):
        qc = np.load(src / f"{f}.npz")["qpos"].astype(np.float64)
        qa_eq, ob = to_adult(qc)
        with torch.no_grad():
            zs = model.tracking_inference(next_obs=torch.as_tensor(ob, device=dev)).cpu().numpy()
        qa = roll_adult(zs, init_adult)
        tag = f.replace("origin_z", "z0")
        np.savez(out / f"adult_from_child_{tag}.npz", qpos=qa.astype(np.float32), fps=30)
        np.savez(out / f"child_{tag}_as_adult.npz", qpos=qa_eq.astype(np.float32), fps=30)
        metrics[name] = dict(
            child_vs_child_reference_L_align=l_align(qc, ref_c, fc),
            adult_vs_what_the_child_did_L_align=l_align(qa, qa_eq, fa),
            adult_vs_what_the_child_did_joint_deg=joint_deg(qa, qa_eq),
            adult_vs_adult_motion_L_align=l_align(qa, adult_motion, fa),
            adult_vs_adult_motion_joint_deg=joint_deg(qa, adult_motion),
            child_did_vs_adult_motion_joint_deg=joint_deg(qa_eq, adult_motion),
            backward_z_cos_to_z0=float(np.mean(zs @ z0 / 256)))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    for k, v in metrics.items():
        print(k); [print(f"   {kk:42s} {vv:.3f}") for kk, vv in v.items()]
    rel = lambda p: str(Path(p).relative_to(REPO))
    panels = {"out": rel(out / "child_to_adult.mp4"), "cols": 3, "size": 320, "camera": "front_side", "panels": [
        {"title": "adult original motion", "sub": "the target", "xml": rel(ADULT), "qpos": rel(REPO / f"data/origin_motion/{task}/{stem}.npz")},
        {"title": "child, z0", "sub": "rollout on the child", "xml": rel(CHILD), "qpos": rel(src / "origin_z.npz")},
        {"title": "child, searched z", "sub": "rollout on the child", "xml": rel(CHILD), "qpos": rel(src / "best.npz")},
        {"title": "adult, z0", "sub": "control", "xml": rel(ADULT), "qpos": rel(out / "adult_z0.npz")},
        {"title": "adult, backward z of child z0", "sub": "child's motion re-run on adult", "xml": rel(ADULT), "qpos": rel(out / "adult_from_child_z0.npz")},
        {"title": "adult, backward z of child searched", "sub": "child's motion re-run on adult", "xml": rel(ADULT), "qpos": rel(out / "adult_from_child_best.npz")}]}
    json.dump(panels, open(out / "panels.json", "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
