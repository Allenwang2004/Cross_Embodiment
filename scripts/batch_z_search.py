#!/usr/bin/env python3
"""batch_z_search.py -- one antithetic ES search per (clip, body), so the best z
found becomes a supervised TARGET instead of a one-off ceiling number.

Why this exists
---------------
model/simple/train_es.py cannot leave the basin around z0. Measured: the cost
rises ~9% per degree away from z0 and ~110% by 10 degrees, while the adapter
moves 0.007 deg/update -- so z_beta sits at the bottom of a steep well, and lr
and adapter_alpha only set how far it rattles around inside it (raising alpha
10x moved z_cos and left the loss unchanged, which is that prediction).

scripts/single_z_search.py escapes because Adam at lr=0.5 acts on z DIRECTLY:
its first step is 26 degrees, it is at 44 degrees by generation 2, and 99% of
its total gain is done by generation 9 -- 160 evals. It does not descend the
well, it steps over it, and it keeps the best candidate it ever evaluated.

So: run that search per (clip, body), keep the best z, and the result is a
labelled dataset. A map fit to it is solving a supervised regression, with an
exact gradient, instead of chasing a zeroth-order estimate that the well
flattens. model/simple/train_zmap.py already trains this exact LatentAdapter
supervised; it just points at retargeting-derived targets rather than searched
ones.

What this answers, and it is falsifiable
-----------------------------------------
Whether the map EXISTS. Fit train_zmap to these targets and read the held-out
regression error:

  * fits    -> there is learnable structure; use the fitted adapter as ES's
               starting point and let ES do the local refinement it is good at.
  * does not fit -> the best z of different (clip, body) cells share nothing a
               map can express, and no amount of sigma / lr / alpha tuning on
               train_es will help. The setup has to change instead.

Cheaper than one more training run, and it can come back negative.

Matched to train_es, NOT to single_z_search
--------------------------------------------
The targets are for model/simple/train_es.py's adapter, so every setting that
defines the objective is taken from ESConfig and the rollout is the one
train_es performs:

  * <dataset>/robots/<body>/robot.xml, the manifest's target_xml -- NOT
    assets/robots_torque/child/, which is what single_z_search defaults to and
    which is a body the 10-body manifest does not even contain.
  * humenv's Default reset. single_z_search's --init reference scores z0 better
    (measured L_align 0.045 vs 0.052) but the landscape has the same shape, and
    a target searched under an init training never uses is a target for a
    different problem.
  * cost = ESConfig's lambda_align/lambda_phys, via the unchanged
    model/losses.py, so the number this minimises is the number train_es
    minimises.

--loss L_phys is refused. Its optimum is to stand still -- single_z_search's
own phys runs reached cost 0.0007..0.0038 at 91..95 degrees from z0, which is
that degenerate solution, and regressing a map onto it would teach the adapter
to stop moving.

Usage (from project root):
    uv run scripts/batch_z_search.py                       # 17 move tasks x 8 bodies
    uv run scripts/batch_z_search.py --trials 3 --evals 400
    uv run scripts/batch_z_search.py --tasks-file datasets/.../splits/upright_tasks.txt

Writes <out>/targets.npz (z0, beta, best_z, costs) and <out>/index.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

from scripts.single_z_search import project_z, rank_normalize, rollout, device_arg

LOSS_LAMBDAS = {"both": (1.0, 1.0), "L_align": (1.0, 0.0)}


def search_one(model, env, fk, obs_mul, cfg, z0, qpos_ref, gens, pairs, sigma, lr, rng):
    """Antithetic ES on z for one clip. Returns (best_z, best_cost, origin_cost).

    The same estimator as single_z_search.main's loop -- rank-normalised
    antithetic differences, Adam on z, re-projected every step -- kept here
    rather than imported because that one is welded into its own argparse and
    per-run file output.
    """
    from model.simple.train import compute_batch_cost

    n = 2 * pairs
    c0, _, _ = compute_batch_cost(
        fk, cfg, rollout(model, env, torch.as_tensor(
            np.repeat(z0[None], n, 0), dtype=torch.float32, device=cfg.device),
            cfg.steps_per_episode, cfg.device, obs_mul), [qpos_ref] * n)
    origin = float(c0[0])
    best_z, best = z0.copy(), origin

    z = z0.copy()
    m = np.zeros_like(z); v = np.zeros_like(z)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    for gen in range(gens):
        eps = rng.standard_normal((pairs, z.size))
        cand = project_z(np.concatenate([z + sigma * eps, z - sigma * eps], axis=0))
        q = rollout(model, env, torch.as_tensor(cand, dtype=torch.float32, device=cfg.device),
                    cfg.steps_per_episode, cfg.device, obs_mul)
        cost, _, _ = compute_batch_cost(fk, cfg, q, [qpos_ref] * n)
        i = int(np.argmin(cost))
        if cost[i] < best:
            best, best_z = float(cost[i]), cand[i].copy()
        s = rank_normalize(cost)
        g = ((s[:pairs] - s[pairs:])[:, None] * eps).sum(0) / (2 * pairs * sigma)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        z = project_z(z - lr * (m / (1 - b1 ** (gen + 1)))
                      / (np.sqrt(v / (1 - b2 ** (gen + 1))) + eps_adam))
    return best_z, best, origin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="datasets/crossenbodiment-10bodies")
    p.add_argument("--tasks-file", default=None,
                   help="default <dataset>/splits/move_tasks.txt")
    p.add_argument("--trials", type=int, default=1,
                   help="trials per task. 1 x 17 tasks x 8 bodies = 136 searches")
    p.add_argument("--bodies", default=None,
                   help="comma-separated; default the train_bodies.txt split")
    p.add_argument("--evals", type=int, default=200,
                   help="rollouts per search. single_z_search reached 99%% of its "
                        "gain by eval 160, so 200 is the measured knee, not a guess")
    p.add_argument("--pairs", type=int, default=8, help="2*pairs env slots")
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.5,
                   help="Adam on z DIRECTLY. |z|=16 and Adam's per-coordinate step "
                        "is ~lr, so 0.5 is a ~26 degree first step -- that is the "
                        "point, it is what clears the well")
    p.add_argument("--loss", default="L_align", choices=sorted(LOSS_LAMBDAS))
    p.add_argument("--vectorization", default="async", choices=["async", "sync"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=device_arg,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="outputs/batch_z_search")
    a = p.parse_args()

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.dataset import CrossEmbodimentDataset, load_task_list
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig

    ds_dir = REPO_ROOT / a.dataset
    tasks_path = Path(a.tasks_file) if a.tasks_file else ds_dir / "splits" / "move_tasks.txt"
    if not tasks_path.exists():
        raise SystemExit(f"{tasks_path} not found -- run scripts/spilt_tasks.py")
    tasks = load_task_list(tasks_path)
    dataset = CrossEmbodimentDataset(ds_dir, task_filter=tasks)

    cfg = ESConfig(device=a.device)
    cfg.lambda_align, cfg.lambda_phys = LOSS_LAMBDAS[a.loss]
    cfg.batch_size = 2 * a.pairs

    by_body = dataset.indices_by_body()
    if a.bodies:
        bodies = [b for b in a.bodies.split(",") if b in by_body]
    else:
        bp = ds_dir / "splits" / "train_bodies.txt"
        bodies = ([b for b in load_task_list(bp) if b in by_body] if bp.exists()
                  else sorted(by_body))

    # (task, trial) cells, so every body searches the SAME clips -- the map is
    # fit across bodies at fixed motion, which is what varying beta means.
    cells = {}
    for i, r in enumerate(dataset.rows):
        if r["trial"] < a.trials:
            cells.setdefault((r["reward_name"], r["trial"]), {})[r["morphology_label"]] = i
    clips = sorted(k for k, v in cells.items() if all(b in v for b in bodies))
    gens = max(a.evals // (2 * a.pairs), 1)
    total = len(clips) * len(bodies)
    print(f"{len(tasks)} tasks x {a.trials} trials -> {len(clips)} clips x {len(bodies)} bodies "
          f"= {total} searches, {gens} gens x {2 * a.pairs} = {gens * 2 * a.pairs} evals each")
    print(f"objective: {a.loss} (lambda_align={cfg.lambda_align}, lambda_phys={cfg.lambda_phys}), "
          f"sigma={a.sigma}, lr={a.lr}, {cfg.steps_per_episode}-step rollouts")

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(a.device)
    model.eval()

    out_dir = REPO_ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, Z0, BE, BZ = [], [], [], []
    t0 = time.time()
    done = 0
    for body in bodies:
        xml = ds_dir / dataset[by_body[body][0]]["target_xml"]
        env, _ = make_humenv(num_envs=2 * a.pairs, vectorization_mode=a.vectorization,
                             task=None, xml=str(xml), state_init="Default")
        fk = mujoco.MjModel.from_xml_path(str(xml))
        obs_mul = build_obs_multiplier(xml, REPO_ROOT / cfg.obs_scale_ref_xml,
                                       mode=cfg.obs_scale, parts=cfg.obs_scale_parts,
                                       verbose=False)
        for clip in clips:
            s = dataset[cells[clip][body]]
            # zlib.crc32, not hash(): Python randomises str hashing per process
            # unless PYTHONHASHSEED is set, so hash() would give a different
            # per-cell stream on every run and --seed would not name anything.
            cell = f"{clip[0]}|{clip[1]}|{body}".encode()
            rng = np.random.default_rng(a.seed * 10**6 + zlib.crc32(cell))
            bz, bc, oc = search_one(model, env, fk, obs_mul, cfg,
                                    s["z0"].astype(np.float64), s["qpos_ref"],
                                    gens, a.pairs, a.sigma, a.lr, rng)
            cos = float(np.dot(bz, s["z0"]) / (np.linalg.norm(bz) * np.linalg.norm(s["z0"])))
            rows.append(dict(body=body, task=clip[0], trial=clip[1], origin_cost=oc,
                             best_cost=bc, improvement=(bc - oc) / oc if oc else 0.0,
                             deg_from_z0=float(np.degrees(np.arccos(np.clip(cos, -1, 1))))))
            Z0.append(s["z0"]); BE.append(s["beta"]); BZ.append(bz.astype(np.float32))
            done += 1
            el = time.time() - t0
            print(f"[{done:4d}/{total}] {body:12s} {clip[0]:28s} t{clip[1]} "
                  f"{oc:8.4f} -> {bc:8.4f} ({rows[-1]['improvement']*100:+6.1f}%) "
                  f"{rows[-1]['deg_from_z0']:5.1f}deg  [{el/60:.1f}m, "
                  f"eta {el/done*(total-done)/60:.0f}m]", flush=True)
        env.close()

    np.savez_compressed(out_dir / "targets.npz",
                        z0=np.stack(Z0).astype(np.float32),
                        beta=np.stack(BE).astype(np.float32),
                        best_z=np.stack(BZ),
                        origin_cost=np.array([r["origin_cost"] for r in rows], dtype=np.float32),
                        best_cost=np.array([r["best_cost"] for r in rows], dtype=np.float32),
                        body=np.array([r["body"] for r in rows]),
                        task=np.array([r["task"] for r in rows]),
                        trial=np.array([r["trial"] for r in rows]))
    with open(out_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out_dir / "args.json").write_text(json.dumps(vars(a), default=str, indent=1))

    imp = np.array([r["improvement"] for r in rows])
    deg = np.array([r["deg_from_z0"] for r in rows])
    print(f"\n{len(rows)} searches in {(time.time()-t0)/60:.1f} min")
    print(f"improvement: median {np.median(imp)*100:+.1f}%  "
          f"[p10 {np.percentile(imp,10)*100:+.1f}%, p90 {np.percentile(imp,90)*100:+.1f}%]  "
          f"{(imp < 0).sum()}/{len(imp)} beat z0")
    print(f"best_z from z0: median {np.median(deg):.1f} deg  "
          f"[{deg.min():.1f}, {deg.max():.1f}]")
    print(f"-> {out_dir/'targets.npz'}\n-> {out_dir/'index.csv'}")


if __name__ == "__main__":
    main()
