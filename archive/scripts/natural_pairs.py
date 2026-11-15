#!/usr/bin/env python3
"""natural_pairs.py -- step 1 of the latent-vs-behaviour study, on REAL clips.

Pairs of different clips from the 500-clip training list whose reward-inferred z0
are close (2.5..26 deg), plus far same-task pairs as a control. The closest pairs
are near-identical tasks with the same trial index (crawl at 0.4 vs 0.5 height,
raised-arm variants), which suggests trial k reuses one inference sample.

On the ORIGINAL body (the adult, m2c_t000):
  own       each z0 from its own reference frame 0, cost against its own reference
            (does each clip work on the adult at all)
  pair      z_A and z_B from the SAME start, once from A's frame 0 and once from B's,
            behaviour distances (controlled_pairs.behaviour_distances) averaged
            over the two starts

Writes outputs/natural_pairs/adult.npz (rollouts) and adult_pairs.csv.

usage: CUDA_VISIBLE_DEVICES=1 uv run scripts/natural_pairs.py
"""
import csv, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.append(str(REPO))
import numpy as np
import torch
from controlled_pairs import behaviour_distances

PAIRS = [  # (group, clip A, clip B)
    ("<5", "crawl-0.4-2-d_14", "crawl-0.5-2-d_14"),
    ("<5", "crawl-0.4-2-d_3", "crawl-0.5-2-d_3"),
    ("<5", "move-ego-0-2-raisearms-h-m_5", "move-ego-0-2-raisearms-m-h_5"),
    ("5-10", "move-ego-0-2-raisearms-h-m_8", "move-ego-0-2-raisearms-m-h_8"),
    ("5-10", "move-ego-0-2-raisearms-h-h_4", "move-ego-0-2-raisearms-m-h_4"),
    ("5-10", "crawl-0.4-2-d_11", "crawl-0.5-2-d_11"),
    ("10-15", "crawl-0.4-2-d_4", "crawl-0.5-2-d_4"),
    ("10-15", "headstand_19", "headstand_28"),
    ("15-20", "move-ego-0-2_1", "move-ego-0-2-raisearms-m-h_1"),
    ("15-20", "headstand_8", "headstand_32"),
    ("15-20", "rotate-z-5-0.8_8", "rotate-z-5-0.8_9"),
    ("20-25", "move-ego-0-2_4", "move-ego-0-2-raisearms-h-h_4"),
    ("20-25", "headstand_18", "headstand_22"),
    ("20-25", "rotate-x--5-0.8_10", "rotate-x--5-0.8_12"),
    ("25-30", "jump-2_23", "jump-2_38"),
    ("far", "move-ego-0-2_4", "move-ego-0-2_9"),
    ("far", "headstand_3", "headstand_34"),
    ("far", "crawl-0.4-2-d_14", "crawl-0.4-2-d_9"),
]


def task_of(c):
    return c.rsplit("_", 1)[0]


def unit(v):
    return v / np.linalg.norm(v)


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    import mujoco
    from model import bfm_align, losses
    from model import kinematics as kin
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    from single_z_search import rollout
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = ESConfig(device=dev)
    W = {"root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
         "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight}
    body = "m2c_t000"
    xml = REPO / "assets/robots_torque" / body / "robot_torque_full.xml"
    fk = mujoco.MjModel.from_xml_path(str(xml))
    obs_mul = build_obs_multiplier(xml, REPO / "assets/robots/adult/robot.xml", mode="auto",
                                   parts=cfg.obs_scale_parts, verbose=False)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev); model.eval()
    clips = sorted({c for _, a, b in PAIRS for c in (a, b)})
    Z = {c: 16.0 * unit(np.load(REPO / "data/origin_z" / task_of(c) / f"{c}.npy").reshape(-1).astype(np.float64)) for c in clips}
    ref = {c: np.load(REPO / "data" / body / "retargeting_motion" / task_of(c) / f"{c}.npz")["qpos"] for c in clips}
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    Bg = {c: bfm_align.reference_embeddings(model, env1, ref[c], dev, obs_mul) for c in clips}
    env1.close()
    n = 16
    env, _ = make_humenv(num_envs=n, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")

    # every (latent, start) rollout needed: each clip from its own start, and both
    # members of a pair from both starts. One start per batch (rollout() sets one pose).
    need = {}
    for c in clips:
        need.setdefault(c, set()).add(c)
    for _, a, b in PAIRS:
        for s in (a, b):
            need.setdefault(s, set()).update((a, b))
    R = {}   # (latent clip, start clip) -> (qpos, B)
    for start, lat in need.items():
        lat = sorted(lat)
        for b0 in range(0, len(lat), n):
            part = lat[b0:b0 + n]; pad = part + [part[-1]] * (n - len(part))
            zt = torch.as_tensor(np.stack([Z[c] for c in pad]), dtype=torch.float32, device=dev)
            q, o = rollout(model, env, zt, 300, dev, obs_mul, init_qpos=ref[start][0], nv=fk.nv, return_obs=True)
            Bo = bfm_align.embed(model, o.reshape(-1, o.shape[-1]), dev).reshape(n, 300, -1)
            for slot, c in enumerate(part):
                R[(c, start)] = (q[slot], Bo[slot])
    env.close()

    def cost(c):
        q, B = R[(c, c)]; T = min(len(B), len(Bg[c]))
        return float(1 - ((B[:T] * Bg[c][:T]).sum(-1) / (np.linalg.norm(B[:T], axis=-1) * np.linalg.norm(Bg[c][:T], axis=-1))).mean())

    rows = []
    for g, a, b in PAIRS:
        ang = float(np.degrees(np.arccos(np.clip(unit(Z[a]) @ unit(Z[b]), -1, 1))))
        ds = [behaviour_distances(fk, R[(a, s)][0], R[(b, s)][0], R[(a, s)][1], R[(b, s)][1], losses, kin, W, 1 / cfg.control_fps)
              for s in (a, b)]
        d = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
        rows.append(dict(group=g, clip_a=a, clip_b=b, latent_deg=ang, cost_a=cost(a), cost_b=cost(b), **d))
    out = REPO / "outputs/natural_pairs"; out.mkdir(parents=True, exist_ok=True)
    keys = sorted(R)
    np.savez(out / "adult.npz", keys=np.array([f"{c}|{s}" for c, s in keys]),
             qpos=np.stack([R[k][0] for k in keys]), B=np.stack([R[k][1] for k in keys]).astype(np.float32),
             clips=np.array(clips), z=np.stack([Z[c] for c in clips]))
    with open(out / "adult_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'grp':6s} {'clip A':30s} {'clip B':30s} {'deg':>5s} {'costA':>6s} {'costB':>6s} | {'joint':>6s} {'ee(m)':>6s} {'head':>6s} {'xy(m)':>6s} {'bspace':>7s} {'L_align':>8s}")
    for r in rows:
        print(f"{r['group']:6s} {r['clip_a'][-30:]:30s} {r['clip_b'][-30:]:30s} {r['latent_deg']:5.1f} {r['cost_a']:6.3f} {r['cost_b']:6.3f} | "
              f"{r['joint']:6.2f} {r['ee']:6.3f} {r['heading']:6.1f} {r['root_xy']:6.2f} {r['bspace']:7.3f} {r['l_align']:8.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
