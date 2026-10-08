#!/usr/bin/env python3
"""rollout_z_trace.py -- watch a single_z_search run learn.

Takes the z_trace.npz that scripts/single_z_search.py writes (every
step's iterate and its best-so-far) and turns a selection of those
checkpoints into one video: the same clip, driven by the z the search held
after 0, 10, 40, ... steps, all playing side by side on the same timeline.

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

The first panel is always the retargeting motion itself (the reference the
search is aligned to), so every checkpoint is read against the target rather
than against each other. Each panel's caption shows the objective the run
actually minimised -- L_align for an --objective align run, L_phys for phys,
their sum for both -- and cos(z, z0), how far that checkpoint's z has drifted
from the clip's original z. The gen -1 panel is z0 itself, and is labelled so.

Writes <out>/trace_<which>.mp4 (the grid), <out>/trace.csv (each checkpoint's
L_align / L_phys / objective / cos(z, z0) / distance travelled / upright fraction)
and <out>/trace_<which>_qpos.npz (the rolled-out qpos of every panel, so other
scores -- scripts/compare_align_losses.py -- are computed on exactly the
trajectories in the video rather than on a re-rollout that may land elsewhere).
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


def logspaced(count, n):
    """Exactly min(n, count) distinct indices into range(count), always
    including 0, spaced logarithmically.

    Log-spaced because the interesting part is the first few hundred
    evaluations; linear spacing spends most of the panels on an already
    converged run. Rounding collides at the dense end, which is why the gaps
    are then filled from the low end rather than the result being handed back
    short -- asking for 8 panels and getting 6 is a silent surprise.
    """
    n = min(n, count)
    out = {0}
    if n > 1:
        for v in np.geomspace(1, count - 1, n - 1):
            out.add(int(round(v)))
    i = 1
    while len(out) < n and i < count:
        out.add(i)
        i += 1
    return sorted(out)


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
    from single_z_search import device_arg

    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="a z_trace.npz")
    p.add_argument("--which", default="mean", choices=["mean", "best"],
                   help="'mean' = the ES iterate after k steps (training "
                        "progress). 'best' = best sample so far, monotone by "
                        "construction")
    p.add_argument("--n", type=int, default=8,
                   help="how many checkpoints, log-spaced over the run (early "
                        "steps are where everything happens)")
    p.add_argument("--gens", default=None,
                   help="explicit comma-separated step numbers, overrides --n. "
                        "Spelled --gens because --steps on this script is "
                        "the rollout LENGTH, which is a different number")
    p.add_argument("--clip", default=None, help="default: from the run's summary.json")
    p.add_argument("--body", default="child")
    p.add_argument("--xml", default=None, help="default: from the run's summary.json")
    p.add_argument("--init", default=None, choices=["reference", "default"],
                   help="default: from the run's summary.json -- match the search "
                        "or the rollouts are not the ones it optimised")
    p.add_argument("--obs-scale", default=None)
    p.add_argument("--obs-scale-ref", default="assets/robots/adult/robot.xml")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cols", type=int, default=3,
                   help="3 makes the default (reference + 8 checkpoints) a 3x3 grid")
    p.add_argument("--size", type=int, default=320, help="panel pixels")
    p.add_argument("--camera", default="front_side")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", type=device_arg,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
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
    if not xml.exists() and "assets/robot_torque/" in str(xml):
        moved = Path(str(xml).replace("assets/robot_torque/", "assets/robots_torque/"))
        if moved.exists():                     # the directory was renamed after these runs
            print(f"NOTE: {xml} is gone; using the renamed {moved}")
            xml = moved
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
        idx = logspaced(len(gens), args.n)
    idx = sorted(set(idx))
    sel_gen, sel_z = gens[idx], zs[idx].astype(np.float64)
    print(f"{clip} on {xml.name}, {args.which} z at steps "
          f"{list(map(int, sel_gen))}  ({len(idx)} rollouts, init={init_mode})")

    ref = np.load(REPO_ROOT / "data" / args.body / "retargeting_motion"
                  / task / f"{stem}.npz")["qpos"]

    # z0 for the cosine column: the clip's original z, the same file the search
    # started from. The trace's gen -1 row is z0 too; prefer the source of truth.
    z0_path = REPO_ROOT / "data" / "origin_z" / task / f"{stem}.npy"
    z0 = np.load(z0_path).reshape(-1) if z0_path.exists() else zs[0].astype(np.float64)
    cos_z0 = sel_z @ z0 / (np.linalg.norm(sel_z, axis=1) * np.linalg.norm(z0))

    # the objective this run minimised, named the way the run was launched
    objective = summary.get("objective", "both")
    lam_a, lam_p = summary.get("lambda_align", 1.0), summary.get("lambda_phys", 1.0)
    obj_name = {"align": "L_align", "phys": "L_phys", "bfm": "1-cos(B)", "mse": "MSE"}.get(objective, "sum")

    cfg = ESConfig(device=args.device)
    cfg.phys_weights = summary["phys_weights"]
    cfg.phys_fall_ref = summary["phys_fall_ref"]
    cfg.lambda_align = cfg.lambda_phys = 1.0

    model = FBcprModel.from_pretrained(args.metamotivo).to(args.device)
    model.eval()
    n = len(idx)
    # roll out in the search's own batch size (2 x pairs, padded with the last z): the actor's GPU rounding
    # depends on the batch size, and on a chaotic clip that alone moves the rollout (headstand_9's z0: MSE
    # 0.905 in the search's 16-env batch, 1.350 in a 3-env one)
    n_env = max(n, 2 * summary["pairs"])
    env, _ = make_humenv(num_envs=n_env, vectorization_mode="async", task=None,
                         xml=str(xml), state_init="Default")
    fk = mujoco.MjModel.from_xml_path(str(xml))
    exact = None
    if obs_scale == "exact":
        # the same adult-equivalent observation the search used (single_z_search.py)
        from model.exact_obs import ExactObs
        kw = {}
        if args.body != "child":
            a = np.load(REPO_ROOT / "data" / "origin_motion" / task / f"{stem}.npz")["qpos"][:, :2]
            m_ = min(len(a), len(ref)); msk = np.abs(a[:m_]) > 1e-3
            kw["scale"] = float(np.median(ref[:m_, :2][msk] / a[:m_][msk]))
        exact, obs_mul = ExactObs(xml, REPO_ROOT / args.obs_scale_ref, **kw), None
    else:
        obs_mul = build_obs_multiplier(xml, REPO_ROOT / args.obs_scale_ref,
                                       mode=obs_scale, parts=cfg.obs_scale_parts,
                                       verbose=False)

    zpad = np.concatenate([sel_z, np.repeat(sel_z[-1:], n_env - n, 0)])
    zt = torch.as_tensor(zpad, dtype=torch.float32, device=args.device)
    qpos = rollout(model, env, zt, steps, args.device, obs_mul,
                   init_qpos=ref[0] if init_mode == "reference" else None, nv=fk.nv, exact=exact)[:n]
    env.close()

    # the new cost and the similarity measures neither cost optimises directly (mse_vs_align_test.py)
    from model import losses, kinematics as kin
    from mse_vs_align_test import metrics as sim_metrics
    mses = [losses.tracking_mse(fk, qpos[i].astype(np.float64), ref)[0] for i in range(n)]
    sims = [sim_metrics(fk, qpos[i].astype(np.float64), ref.astype(np.float64), losses, kin) for i in range(n)]
    aw = summary.get("anchor_weight") or 0.0
    anc = np.load(REPO_ROOT / summary["anchor"]).reshape(-1) if aw else None
    if anc is not None:
        anc = 16 * anc / np.linalg.norm(anc)

    _, aligns, physes = compute_batch_cost(fk, cfg, qpos, [ref] * n)
    bfm_cost = None
    if objective == "bfm":
        # score the panels the way the search scored them: B() on the same
        # rescaled obs, reference through obs_from_qpos
        from model import bfm_align
        env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
        Bg = bfm_align.reference_embeddings(model, env1, ref, args.device, obs_mul)
        bfm_cost = [bfm_align.bfm_align_loss(
            bfm_align.embed(model, bfm_align.obs_from_qpos(env1, qpos[i], obs_mul=obs_mul), args.device), Bg)
            for i in range(n)]
        env1.close()
    np.savez_compressed(out_dir / f"trace_{args.which}_qpos.npz",
                        gen=sel_gen.astype(np.int32), z=sel_z.astype(np.float32),
                        qpos=qpos.astype(np.float32), ref=ref.astype(np.float32),
                        cos_z0=cos_z0.astype(np.float32), xml=str(xml), clip=clip)

    rows = []
    for i, g in enumerate(sel_gen):
        q = qpos[i]
        qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        up = 2.0 * (qy * qz + qw * qx)
        rows.append(dict(gen=int(g), evals=int((g + 1) * 2 * summary["pairs"]),
                         L_align=float(aligns[i]), L_phys=float(physes[i]),
                         objective=obj_name,
                         cost=float(bfm_cost[i]) if bfm_cost is not None
                              else float(mses[i]) if objective == "mse"
                              else float(lam_a * aligns[i] + lam_p * physes[i]),
                         mse=float(mses[i]), **sims[i],
                         anchor_term=float(aw * (np.linalg.norm(16 * sel_z[i] / np.linalg.norm(sel_z[i]) - anc) / 16) ** 2)
                         if anc is not None else 0.0,
                         cos_z0=float(cos_z0[i]),
                         travel=float(np.linalg.norm(q[-1, :2] - q[0, :2])),
                         upright_frac=float((up > 0.8).mean())))
    with open(out_dir / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"objective {objective}: cost = " + ("losses.tracking_mse" if objective == "mse" else
          f"{lam_a:g} * L_align + {lam_p:g} * L_phys") + f"  ({obj_name})")
    print(f"{'gen':>6s}{'evals':>8s}{'L_align':>9s}{'MSE':>8s}{'cost':>9s}{'MPJPE':>8s}{'local':>7s}{'head':>7s}"
          f"{'joint':>7s}{'cos_z0':>8s}{'upright':>9s}   (MPJPE / local in cm, head / joint in deg)")
    for r in rows:
        print(f"{r['gen']:6d}{r['evals']:8d}{r['L_align']:9.4f}{r['mse']:8.4f}{r['cost']:9.4f}{r['mpjpe_glob']:8.1f}"
              f"{r['mpjpe_loc']:7.1f}{r['head_deg']:7.1f}{r['joint_deg']:7.1f}{r['cos_z0']:8.3f}{100 * r['upright_frac']:8.0f}%")

    # --- kinematic playback of what was just simulated -----------------------
    renderer = mujoco.Renderer(fk, height=args.size, width=args.size)
    data = mujoco.MjData(fk)

    def play(q, text, sub, ended=None):
        frames = []
        for t in range(steps):
            data.qpos[:] = q[min(t, len(q) - 1)]      # a shorter reference holds its last frame
            mujoco.mj_forward(fk, data)
            renderer.update_scene(data, camera=args.camera)
            frames.append(label(renderer.render().copy(), text,
                                ended if (ended and t >= len(q)) else sub))
        return frames

    # panel 0: the target itself, so the checkpoints are read against it. The
    # loss only scores frames both trajectories have (losses._align_length), so
    # once a shorter reference runs out its panel says so instead of pretending.
    panels = [play(ref, "retargeting motion", f"reference, {len(ref)} frames  (L_align 0)",
                   ended=f"reference ended -- L_align scores frames 0-{len(ref) - 1} only")]
    for i, g in enumerate(sel_gen):
        head = "origin z  (gen -1)" if g < 0 else f"gen {int(g)}  ({rows[i]['evals']} evals)"
        panels.append(play(qpos[i], head,
                           f"{obj_name} {rows[i]['cost']:.3f}  MPJPE {rows[i]['mpjpe_glob']:.0f} cm  "
                           f"head {rows[i]['head_deg']:.0f} deg"))
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
    print(f"\n-> {out_mp4}\n-> {out_dir / 'trace.csv'}\n-> {out_dir / f'trace_{args.which}_qpos.npz'}")


if __name__ == "__main__":
    main()
