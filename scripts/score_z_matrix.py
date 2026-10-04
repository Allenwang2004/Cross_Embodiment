#!/usr/bin/env python3
"""score_z_matrix.py -- roll out arbitrary latents on arbitrary (clip, body)
cells and score them exactly as training does.

The workhorse behind every transfer matrix and baseline in the morphology
study: "z found for body A, rolled out on body B", "the BFM's own tracking
inference on the retargeted motion", "the adapter's prediction", ... are all
the same operation -- a set of (clip, body, label, z) jobs -- and they must be
scored identically or the comparisons mean nothing:

  * the body's own torque-calibrated XML (assets/robots_torque/<body>/)
  * reference init: every slot starts from ITS clip's retargeted frame 0
  * bfm cost = 1 - mean_t cos(B(s_t), B(g_t)) against that body's retargeted
    reference, with obs rescaled to the adult's units (model/obs_scale.py)

Jobs are grouped by body (an env carries one skeleton) and rolled out in
batches of --chunk slots.

Input: a .npz with arrays  clip (str), body (str), label (str), z (N, 256)
Output: <out>.csv with clip, body, label, cost   (and cost_z0 when --with-z0)

Usage:
    uv run scripts/score_z_matrix.py --jobs jobs.npz --out scores.csv --with-z0
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT not in sys.path:
    sys.path.append(PARENT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from single_z_search import device_arg, project_z
from rank_initial_cost import rollout_multi

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--with-z0", action="store_true",
                   help="also score each (clip, body)'s own z0 once, as label 'z0'")
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--robots-dir", default="assets/robots_torque")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", type=device_arg,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    args = p.parse_args()

    J = np.load(args.jobs, allow_pickle=True)
    jobs = [(str(c), str(b), str(l), np.asarray(z, dtype=np.float64))
            for c, b, l, z in zip(J["clip"], J["body"], J["label"], J["z"])]
    if args.with_z0:
        seen = set()
        for c, b, _, _ in list(jobs):
            if (c, b) in seen:
                continue
            seen.add((c, b))
            t, s = c.split("/")
            z0 = np.load(REPO_ROOT / args.data_dir / "origin_z" / t / f"{s}.npy").reshape(-1)
            jobs.append((c, b, "z0", z0.astype(np.float64)))
    by_body = collections.defaultdict(list)
    for j in jobs:
        by_body[j[1]].append(j)
    print(f"{len(jobs)} jobs over {len(by_body)} bodies")

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    import mujoco

    cfg = ESConfig(device=args.device)
    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device); model.eval()
    rows = []
    t0 = time.time()
    for body, bj in by_body.items():
        xml = REPO_ROOT / args.robots_dir / body / "robot_torque_full.xml"
        obs_mul = build_obs_multiplier(xml, REPO_ROOT / "assets/robots/adult/robot.xml",
                                       mode="auto", parts=cfg.obs_scale_parts, verbose=False)
        nv = mujoco.MjModel.from_xml_path(str(xml)).nv
        env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
        env, _ = make_humenv(num_envs=args.chunk, vectorization_mode="async", task=None,
                             xml=str(xml), state_init="Default")
        refs, Bg = {}, {}
        for c in {j[0] for j in bj}:
            t, s = c.split("/")
            refs[c] = np.load(REPO_ROOT / args.data_dir / body / "retargeting_motion" / t
                              / f"{s}.npz")["qpos"]
            Bg[c] = bfm_align.reference_embeddings(model, env1, refs[c], args.device, obs_mul)
        for a in range(0, len(bj), args.chunk):
            part = bj[a:a + args.chunk]
            pad = part + [part[-1]] * (args.chunk - len(part))
            Z = np.stack([project_z(j[3].reshape(-1)) for j in pad])
            obs = rollout_multi(model, env, torch.as_tensor(Z, dtype=torch.float32,
                                                            device=args.device),
                                args.steps, args.device, obs_mul, [refs[j[0]][0] for j in pad], nv)
            for i, j in enumerate(part):
                c = float(bfm_align.batch_bfm_align(model, obs[i:i + 1], Bg[j[0]], args.device)[0])
                rows.append((j[0], body, j[2], c))
        env.close(); env1.close()
        print(f"  {body}: {len(bj)} jobs  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "body", "label", "cost"])
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
