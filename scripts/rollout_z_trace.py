#!/usr/bin/env python3
"""rollout_z_trace.py -- watch a single_z_search run learn.

Takes the z_trace.npz that scripts/single_z_search.py writes (every
generation's iterate and its best-so-far) and turns a selection of those
checkpoints into one video: the same clip, driven by the z the search held
after 0, 10, 40, ... generations, all playing side by side on the same timeline.

Two things make this cheap rather than a re-run of the search:

  * every selected checkpoint is rolled out in ONE batched vectorized env, one
    z per slot, so N checkpoints cost the same wall clock as one rollout;
  * the video is a kinematic playback of the qpos that rollout recorded
    (mj_forward, no stepping), so nothing is simulated twice and each panel is
    exactly the trajectory that produced its numbers.

--which mean is the ES iterate, which is what "the model after k steps" means.
--which best is the best sample seen so far, which is what the search would
hand you if stopped at k -- it is monotone by construction, so it looks like
smooth progress even when the iterate is thrashing. Both are in the trace;
mean is the honest one to show as training progress.

Usage (from project root):
    uv run scripts/rollout_z_trace.py \
        --trace outputs/single_z/move-ego-0-2_4_both/z_trace.npz --n 8

    uv run scripts/rollout_z_trace.py --trace ... --gens 0,5,10,25,50,100,300,624

Writes <out>/trace.mp4 (the grid) and <out>/trace.csv (each checkpoint's
L_align / L_phys / distance travelled / upright fraction).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent


def label(frame, text, sub=None):
    """Burn the checkpoint's identity into its own panel -- a grid of otherwise
    identical humanoids is unreadable without it, and a legend outside the
    video cannot survive being clipped into a gif or a slide."""
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(252, 252, 251))
    d.text((6, 6), text, fill=(11, 11, 11))
    if sub:
        d.text((6, 30), sub, fill=(82, 81, 78))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="a z_trace.npz")
    p.add_argument("--which", default="mean", choices=["mean", "best"],
                   help="'mean' = the ES iterate after k generations (training "
                        "progress). 'best' = best sample so far, monotone by "
                        "construction")
    p.add_argument("--n", type=int, default=8,
                   help="how many checkpoints, log-spaced over the run (early "
                        "generations are where everything happens)")
    p.add_argument("--gens", default=None,
                   help="explicit comma-separated generations, overrides --n")
    p.add_argument("--clip", default=None, help="default: from the run's summary.json")
    p.add_argument("--body", default="child")
    p.add_argument("--xml", default=None, help="default: from the run's summary.json")
    p.add_argument("--init", default=None, choices=["reference", "default"],
                   help="default: from the run's summary.json -- match the search "
                        "or the rollouts are not the ones it optimised")
    p.add_argument("--obs-scale", default=None)
    p.add_argument("--obs-scale-ref", default="assets/robots/adult/robot.xml")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--size", type=int, default=320, help="panel pixels")
    p.add_argument("--camera", default="front_side")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--out", default=None, help="default: next to the trace")
    args = p.parse_args()

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel

    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    from model.simple.train import compute_batch_cost
    from single_z_search import rollout

    trace_path = Path(args.trace)
    if not trace_path.is_absolute():
        trace_path = REPO_ROOT / trace_path
    run_dir = trace_path.parent
    summary = json.loads((run_dir / "summary.json").read_text())

    # Everything defaults to what the search actually ran with. Overriding any
    # of it silently would produce a video of a different experiment.
    clip = args.clip or summary["clip"]
    xml = Path(args.xml) if args.xml else Path(summary["xml"])
    init_mode = args.init or summary.get("init", "default")
    obs_scale = args.obs_scale or summary.get("obs_scale", "auto")
    steps = args.steps or summary["steps"]
    task, stem = clip.split("/")
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = np.load(trace_path)
    gens, zs = tr["gen"], tr[f"z_{args.which}"]
    if args.gens:
        want = [int(g) for g in args.gens.split(",")]
        idx = [int(np.argmin(np.abs(gens - g))) for g in want]
    else:
        # log-spaced, because the interesting part is the first few hundred
        # evaluations -- linear spacing spends most panels on a converged run
        lo, hi = 1, len(gens) - 1
        idx = [0] + sorted({int(round(v)) for v in
                            np.geomspace(lo, hi, max(args.n - 1, 1))})
    idx = sorted(set(idx))
    sel_gen, sel_z = gens[idx], zs[idx].astype(np.float64)
    print(f"{clip} on {xml.name}, {args.which} z at generations "
          f"{list(map(int, sel_gen))}  ({len(idx)} rollouts, init={init_mode})")

    ref = np.load(REPO_ROOT / "data" / args.body / "retargeting_motion"
                  / task / f"{stem}.npz")["qpos"]

    cfg = ESConfig(device=args.device)
    cfg.phys_weights = summary["phys_weights"]
    cfg.phys_fall_ref = summary["phys_fall_ref"]
    cfg.lambda_align = cfg.lambda_phys = 1.0

    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device)
    model.eval()
    n = len(idx)
    env, _ = make_humenv(num_envs=n, vectorization_mode="async", task=None,
                         xml=str(xml), state_init="Default")
    fk = mujoco.MjModel.from_xml_path(str(xml))
    obs_mul = build_obs_multiplier(xml, REPO_ROOT / args.obs_scale_ref,
                                   mode=obs_scale, parts=cfg.obs_scale_parts,
                                   verbose=False)

    zt = torch.as_tensor(sel_z, dtype=torch.float32, device=args.device)
    qpos = rollout(model, env, zt, steps, args.device, obs_mul,
                   init_qpos=ref[0] if init_mode == "reference" else None, nv=fk.nv)
    env.close()

    _, aligns, physes = compute_batch_cost(fk, cfg, qpos, [ref] * n)

    rows = []
    for i, g in enumerate(sel_gen):
        q = qpos[i]
        qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        up = 2.0 * (qy * qz + qw * qx)
        rows.append(dict(gen=int(g), evals=int((g + 1) * 2 * summary["pairs"]),
                         L_align=float(aligns[i]), L_phys=float(physes[i]),
                         sum=float(aligns[i] + physes[i]),
                         travel=float(np.linalg.norm(q[-1, :2] - q[0, :2])),
                         upright_frac=float((up > 0.8).mean())))
    with open(out_dir / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'gen':>6s}{'evals':>8s}{'L_align':>9s}{'L_phys':>9s}{'sum':>8s}"
          f"{'travel':>9s}{'upright':>9s}")
    for r in rows:
        print(f"{r['gen']:6d}{r['evals']:8d}{r['L_align']:9.4f}{r['L_phys']:9.4f}"
              f"{r['sum']:8.4f}{r['travel']:9.2f}{100 * r['upright_frac']:8.0f}%")

    # --- kinematic playback of what was just simulated -----------------------
    renderer = mujoco.Renderer(fk, height=args.size, width=args.size)
    data = mujoco.MjData(fk)
    panels = []
    for i, g in enumerate(sel_gen):
        frames = []
        for t in range(steps):
            data.qpos[:] = qpos[i, t]
            mujoco.mj_forward(fk, data)
            renderer.update_scene(data, camera=args.camera)
            frames.append(label(renderer.render().copy(),
                                f"gen {int(g)}  ({rows[i]['evals']} evals)",
                                f"sum {rows[i]['sum']:.3f}"))
        panels.append(frames)
    # the reference itself, as the last panel -- the thing being chased
    ref_frames = []
    for t in range(steps):
        data.qpos[:] = ref[min(t, len(ref) - 1)]
        mujoco.mj_forward(fk, data)
        renderer.update_scene(data, camera=args.camera)
        ref_frames.append(label(renderer.render().copy(), "reference",
                                "the retargeted clip"))
    panels.append(ref_frames)
    renderer.close()

    cols = min(args.cols, len(panels))
    rowsn = int(np.ceil(len(panels) / cols))
    blank = np.full_like(panels[0][0], 252)
    grid = []
    for t in range(steps):
        band = []
        for r in range(rowsn):
            row = [panels[r * cols + c][t] if r * cols + c < len(panels) else blank
                   for c in range(cols)]
            band.append(np.concatenate(row, axis=1))
        grid.append(np.concatenate(band, axis=0))
    out_mp4 = out_dir / f"trace_{args.which}.mp4"
    imageio.mimsave(out_mp4, grid, fps=args.fps)
    print(f"\n-> {out_mp4}\n-> {out_dir / 'trace.csv'}")


if __name__ == "__main__":
    main()
