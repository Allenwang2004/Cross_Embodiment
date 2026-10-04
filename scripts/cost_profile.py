#!/usr/bin/env python3
"""cost_profile.py -- WHERE in time does the bfm cost come from?

The search objective is cost(z) = 1 - mean_t cos(B(s_t), B(g_t)) over 300 steps
with one z for the whole clip. Before changing it (discounting, early termination,
a latent sequence), look at the per-step similarity of rollouts we already have:

  - if a bad rollout is good at first and then collapses for good (a fall), most
    of its cost is the tail after the failure, and a time-weighted objective /
    early termination targets the real problem;
  - if the similarity is mediocre all the way through, the single latent is
    tracking every phase only partly, which points at a latent sequence instead.

Groups (child-sized body): 10 headstand trials and 10 walking trials (z0 vs the
4992-rollout per-trial search), and the 8 path motions on m2c_t1000 (z0 vs
continuation). Every rollout is re-run exactly as score_z_matrix.py scores it.

usage: CUDA_VISIBLE_DEVICES=1 uv run scripts/cost_profile.py
"""
import collections, os, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.append(str(REPO))
import numpy as np
import torch
from single_z_search import project_z
from rank_initial_cost import rollout_multi

HEAD = [0, 2, 3, 4, 9, 15, 20, 23, 30, 34]
PATH8 = [l.strip().split("/")[1] for l in open(REPO / "outputs/continuation_clips.txt") if l.strip()]


def jobs():
    J = []
    for k in HEAD:
        s = f"headstand_{k}"; d = REPO / f"outputs/single_z_floor/{s}_bfm_s0"
        J += [("headstand", s, "child", "z0", None), ("headstand", s, "child", "best", d / "best_z.npy")]
    for k in range(10):
        s = f"move-ego-0-2_{k}"; d = REPO / f"outputs/single_z_floor/{s}_bfm_s0"
        J += [("walking", s, "child", "z0", None), ("walking", s, "child", "best", d / "best_z.npy")]
    for s in PATH8:
        d = REPO / f"outputs/continuation/m2c/cont/{s}__m2c_t1000"
        J += [("8 motions", s, "m2c_t1000", "z0", None), ("8 motions", s, "m2c_t1000", "best", d / "best_z.npy")]
    return J


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    import mujoco
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = ESConfig(device=dev)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev); model.eval()
    by_body = collections.defaultdict(list)
    for j in jobs():
        by_body[j[2]].append(j)
    out = {}
    t0 = time.time()
    for body, bj in by_body.items():
        xml = REPO / "assets/robots_torque" / body / "robot_torque_full.xml"
        obs_mul = build_obs_multiplier(xml, REPO / "assets/robots/adult/robot.xml", mode="auto",
                                       parts=cfg.obs_scale_parts, verbose=False)
        nv = mujoco.MjModel.from_xml_path(str(xml)).nv
        env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
        chunk = 20
        env, _ = make_humenv(num_envs=chunk, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")
        refs, Bg = {}, {}
        for _, s, _, _, _ in bj:
            if s in refs: continue
            t = s.rsplit("_", 1)[0]
            refs[s] = np.load(REPO / "data" / body / "retargeting_motion" / t / f"{s}.npz")["qpos"]
            Bg[s] = bfm_align.reference_embeddings(model, env1, refs[s], dev, obs_mul)
        for a in range(0, len(bj), chunk):
            part = bj[a:a + chunk]; pad = part + [part[-1]] * (chunk - len(part))
            Z = []
            for g, s, b, lab, f in pad:
                t = s.rsplit("_", 1)[0]
                z = np.load(REPO / "data/origin_z" / t / f"{s}.npy") if f is None else np.load(f)
                Z.append(project_z(z.reshape(-1).astype(np.float64)))
            obs = rollout_multi(model, env, torch.as_tensor(np.stack(Z), dtype=torch.float32, device=dev),
                                300, dev, obs_mul, [refs[j[1]][0] for j in pad], nv)
            for i, (g, s, b, lab, f) in enumerate(part):
                Bs = bfm_align.embed(model, obs[i], dev)
                out[(g, s, lab)] = bfm_align.cos_per_frame(Bs, Bg[s])
        env.close(); env1.close()
        print(f"  {body}: {len(bj)} rollouts ({(time.time() - t0) / 60:.1f} min)", flush=True)
    keys = sorted(out)
    od = REPO / "outputs/cost_profile"; od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "profiles.npz", keys=np.array(["|".join(k) for k in keys], dtype=object),
             **{f"c{i}": out[k] for i, k in enumerate(keys)})
    print(f"-> {od / 'profiles.npz'}")


if __name__ == "__main__":
    main()
