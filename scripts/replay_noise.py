#!/usr/bin/env python3
"""replay_noise.py -- is the bfm cost of a FIXED latent reproducible?

cost_profile.py found replayed headstand costs up to 0.16 away from what the
search recorded for the very same z (walking within ~0.03). The search assumes a
rollout is a deterministic function of z (identical start states, mean actions),
so candidates can be compared exactly and the best one kept. If the cost instead
depends on how the rollout is batched (GPU float differences amplified by
chaotic contact dynamics), every comparison carries noise and "best so far" is
partly the luckiest draw.

Each latent (z0 and the searched best, 10 headstand + 10 walking trials, child
body) is rolled out R times, each time as a different slot of a batch of a
different size, filled with the other latents. Reports the spread of its cost.

usage: CUDA_VISIBLE_DEVICES=1 uv run scripts/replay_noise.py
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.append(str(REPO))
import numpy as np
import torch
from single_z_search import project_z
from rank_initial_cost import rollout_multi

HEAD = [0, 2, 3, 4, 9, 15, 20, 23, 30, 34]


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    import mujoco
    dev = "cuda:0"
    cfg = ESConfig(device=dev)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev); model.eval()
    body = "child"
    xml = REPO / "assets/robots_torque" / body / "robot_torque_full.xml"
    obs_mul = build_obs_multiplier(xml, REPO / "assets/robots/adult/robot.xml", mode="auto",
                                   parts=cfg.obs_scale_parts, verbose=False)
    nv = mujoco.MjModel.from_xml_path(str(xml)).nv
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    items = []
    for task, ks in (("headstand", HEAD), ("move-ego-0-2", range(10))):
        for k in ks:
            s = f"{task}_{k}"
            ref = np.load(REPO / "data" / body / "retargeting_motion" / task / f"{s}.npz")["qpos"]
            Bg = bfm_align.reference_embeddings(model, env1, ref, dev, obs_mul)
            z0 = np.load(REPO / "data/origin_z" / task / f"{s}.npy").reshape(-1)
            zb = np.load(REPO / f"outputs/single_z_floor/{s}_bfm_s0/best_z.npy").reshape(-1)
            for lab, z in (("z0", z0), ("best", zb)):
                items.append(dict(task=task, clip=s, lab=lab, z=project_z(z.astype(np.float64)), ref=ref, Bg=Bg, costs=[]))
    env1.close()
    rng = np.random.default_rng(0)
    for n in (2, 4, 16, 20, 32):
        env, _ = make_humenv(num_envs=n, vectorization_mode="async", task=None,
                             xml=str(xml), state_init="Default")
        order = rng.permutation(len(items))
        for a in range(0, len(order), n):
            idx = list(order[a:a + n]); idx += [idx[-1]] * (n - len(idx))
            obs = rollout_multi(model, env, torch.as_tensor(np.stack([items[i]["z"] for i in idx]), dtype=torch.float32, device=dev),
                                300, dev, obs_mul, [items[i]["ref"][0] for i in idx], nv)
            done = set()
            for slot, i in enumerate(idx):
                if i in done: continue
                done.add(i)
                items[i]["costs"].append(float(bfm_align.batch_bfm_align(model, obs[slot:slot + 1], items[i]["Bg"], dev)[0]))
        env.close()
        print(f"  batch size {n} done", flush=True)
    for task in ("headstand", "move-ego-0-2"):
        for lab in ("z0", "best"):
            S = [np.array(it["costs"]) for it in items if it["task"] == task and it["lab"] == lab]
            spread = np.array([c.max() - c.min() for c in S]); sd = np.array([c.std() for c in S]); mu = np.array([c.mean() for c in S])
            print(f"{task:13s} {lab:4s}: mean cost {mu.mean():.3f} | across 5 replays: sd median {np.median(sd):.3f} max {sd.max():.3f} | "
                  f"max-min median {np.median(spread):.3f} max {spread.max():.3f}")
    np.save(REPO / "outputs/cost_profile/replay_noise.npy",
            np.array([(it["task"], it["clip"], it["lab"], *it["costs"]) for it in items], dtype=object))


if __name__ == "__main__":
    main()
