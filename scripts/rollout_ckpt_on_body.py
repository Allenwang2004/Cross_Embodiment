#!/usr/bin/env python3
"""rollout_ckpt_on_body.py -- roll a trained LatentAdapter checkpoint out on
bodies it was never trained on, scored against the z0 baseline in the same batch.

What this is for
----------------
model/simple/evaluate.py already scores a checkpoint on every body, but it rolls
out from humenv's Default standing reset -- the init training uses. That is the
right default for a number comparable with the training curve, and the wrong one
for asking "can this adapter drive an unseen body through this motion", because
the first frames of L_align are then charged for a starting pose no z chose.
--init-from-reference puts the rollout on the retargeted clip's own frame 0
instead, the same thing scripts/rollout_z_on_body.py's flag of that name does.

Every setting that defines the policy comes from the PICKLED cfg, never from a
flag: adapter width, alpha, project, residual, and the obs canonicalisation. An
evaluation that disagreed with its training run on any of those would be scoring
a policy that never existed. `residual` in particular changes the forward pass
without changing a single tensor shape, so a --no-residual checkpoint loads
cleanly into a residual adapter and silently becomes a different model.

The z0 baseline is not optional
--------------------------------
Every clip is rolled out TWICE in the same batched call, once with
z_beta = G_theta(beta, z0) and once with z0 itself, from the same initial state.
Cost on an unseen body is meaningless on its own -- `giant`'s z0 L_align is a
measured 1.34 median against `athletic`'s 0.165, so a large number can mean the
body is hard rather than the adapter bad. The gap between the two columns is
what the adapter is responsible for, and CRN makes it exact: both rollouts are
deterministic from a bit-identical start and differ ONLY by z.

Comparing runs
--------------
--checkpoint takes a LIST, and each one becomes its own panel of the same video
of the same clip, beside z0 and the reference. That is the whole point: a config
change (alpha, residual, task group) shows up as two figures moving differently
on one screen, which the cost columns alone will not tell you.

Two things are deliberately NOT per-checkpoint:

  * the objective. Every panel is scored with the FIRST checkpoint's
    lambda_align / lambda_phys, and a differing one is called out at startup.
    Scoring each rollout under its own training weights would put different
    quantities in one column and call the difference a result.
  * the clip and the initial state. All panels of one video are the same clip on
    the same body from the same starting pose, so the only thing that differs
    between them is z.

Everything that IS part of the policy stays per-checkpoint, including the obs
canonicalisation: two checkpoints trained under different obs_scale settings are
shown different observations inside the same batched rollout, via a per-slot
multiplier. Broadcasting one vector would quietly evaluate one of them on the
other's inputs.

--video-dir needs GPU memory of its own: mujoco.Renderer opens an EGL context on
the same device, and on a shared box that is enough to turn a run that scores
fine into a CUDA OOM inside the adapter's forward pass. Point --device at a
quieter GPU, or drop --video-dir and render later from --save-qpos.

Usage (from project root):
    uv run scripts/rollout_ckpt_on_body.py \
        --checkpoint outputs/simple_es/L_align_upright/update_00200.pt \
        --init-from-reference

    # what a config change did, on the same motion, side by side
    uv run scripts/rollout_ckpt_on_body.py --init-from-reference \
        --checkpoint outputs/simple_es/L_align_upright/update_00200.pt \
                     outputs/simple_es/move-L_align-single-clip-alpha_1/update_00200.pt \
        --labels alpha0.1 alpha1 \
        --video-dir outputs/ckpt_rollout/alpha

    # a body the checkpoints DID train on, as a control
    uv run scripts/rollout_ckpt_on_body.py --checkpoint <ckpt> \
        --init-from-reference --bodies athletic petite

    # keep the trajectories to render or plot later
    uv run scripts/rollout_ckpt_on_body.py --checkpoint <ckpt> \
        --init-from-reference --save-qpos outputs/ckpt_rollout/qpos

Defaults to splits/test_bodies.txt (the bodies held out of training) and
splits/move_tasks.txt, trial 0 of each task. Writes <out>/per_clip.csv and
prints a per-body summary.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent


def _caption_font(size=20):
    """PIL's built-in bitmap font is ~11px and unreadable beside a 416px panel.
    matplotlib ships DejaVuSans and is already a dependency, so borrow it rather
    than add one; fall back to the bitmap font if that ever moves."""
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.__file__).parent / "mpl-data"
                                      / "fonts" / "ttf" / "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def render_panels(model, data, renderer, seqs, labels, camera, n_frames, font=None):
    """Kinematic playback of several qpos sequences into one side-by-side clip.

    mj_forward only, no stepping -- the trajectories were already produced by
    the simulator, and re-stepping them would drift. Same math as
    scripts/rollout_video.py:render and rollout_z_on_body.py:render_reference;
    inlined for the same reason they inline each other's.

    Sequences of different lengths are held on their last frame rather than
    truncated to the shortest, so a reference that runs out does not silently
    shorten the rollout it is being compared against.
    """
    frames = []
    for t in range(n_frames):
        panels = []
        for seq in seqs:
            data.qpos[:] = seq[min(t, len(seq) - 1)]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            panels.append(renderer.render().copy())
        img = Image.fromarray(np.concatenate(panels, axis=1))
        d = ImageDraw.Draw(img)
        w = panels[0].shape[1]
        for i, lab in enumerate(labels):
            # Drawn twice, offset by a pixel, so the caption stays legible over
            # both the pale sky and the figure. `lab` may carry a newline: the
            # score line spells out which terms the number is made of, because
            # "0.587" alone does not say whether L_phys is in it.
            d.multiline_text((i * w + 13, 11), lab, fill=(255, 255, 255),
                             font=font, spacing=4)
            d.multiline_text((i * w + 12, 10), lab, fill=(20, 20, 20),
                             font=font, spacing=4)
        frames.append(np.asarray(img))
    return frames


@torch.no_grad()
def rollout(model, env, z_env, steps, device, obs_mul, init_qpos=None, return_obs=False):
    """Deterministic rollout, one z per slot. init_qpos: length-n_slots list of
    per-slot qpos (or None entries), so slots carrying different clips can each
    start on their own reference.

    Set per sub-env rather than through env.call: VectorEnv.call broadcasts ONE
    argument to every slot, which is exactly wrong when the slots are different
    clips. Requires the sync vectorization mode -- async has no .envs.
    """
    obs, _ = env.reset()
    if init_qpos is not None:
        for e, q in zip(env.envs, init_qpos):
            if q is not None:
                e.unwrapped.set_physics(qpos=q, qvel=np.zeros(e.unwrapped.model.nv))
        obs = {"proprio": np.stack([e.unwrapped.get_obs()["proprio"] for e in env.envs])}
    hist, obs_hist = [], []
    for _ in range(steps):
        p = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        o = torch.as_tensor(p, dtype=torch.float32, device=device)
        mu = model._actor(model._normalize(o), z_env, model.cfg.actor_std).mean
        obs, _, _, _, info = env.step(mu.cpu().numpy())
        hist.append(info["qpos"].copy())
        if return_obs:
            obs_hist.append(obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul)
    if return_obs:
        return np.stack(hist, axis=1), np.stack(obs_hist, axis=1).astype(np.float32)
    return np.stack(hist, axis=1)


def load_ckpt(path, device):
    """(name, cfg, state_dict). The name is the checkpoint's DIRECTORY, not the
    file: runs are one directory each and every file in one is the same
    experiment at a different update, so `move_alpha1` reads better on a video
    panel than `update_00200`."""
    q = Path(path)
    q = q if q.is_absolute() else REPO_ROOT / q
    ck = torch.load(q, map_location=device, weights_only=False)
    if "action_head" in ck:
        raise SystemExit(f"{path} carries an action_head; that path is gone (see "
                         "model/simple/train.py) and evaluating without it would "
                         "score a policy that never existed")
    return f"{q.parent.name}@{ck.get('update','?')}", ck["cfg"], ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, nargs="+",
                   help="one or more .pt files, rendered as one panel each in "
                        "the order given. Two checkpoints of the same clip is "
                        "how you see what a config change actually did")
    p.add_argument("--labels", nargs="*", default=None,
                   help="panel captions, one per --checkpoint; default is "
                        "<run directory>@<update>")
    p.add_argument("--dataset", default="datasets/crossenbodiment-10bodies")
    p.add_argument("--bodies", nargs="*", default=None,
                   help="default: splits/test_bodies.txt, i.e. the bodies held "
                        "out of training")
    p.add_argument("--tasks-file", default=None,
                   help="default splits/test_tasks.txt, the held-out TASK split. "
                        "That axis is independent of the body split and of the "
                        "P.fall/move grouping, so some of its tasks may be inside "
                        "a checkpoint's own training group -- the run marks "
                        "which, per task, rather than letting the word 'test' "
                        "carry an assumption it does not support")
    p.add_argument("--trials", type=int, default=1, help="trials per task")
    p.add_argument("--score-loss", default=None, choices=["bfm", "joint"],
                   help="objective for the cost column. Default: whatever the FIRST checkpoint "
                        "was trained with (cfg.align_loss). 'bfm' is 1 - mean_t cos(B(s), B(g)), "
                        "the BFMTrack latent alignment; 'joint' is lambda_align*L_align + "
                        "lambda_phys*L_phys. A checkpoint trained on bfm and scored on joint is "
                        "being judged by a different quantity than it optimised -- the two "
                        "disagree per clip, so this defaults to matching the training run")
    p.add_argument("--clip-list", default=None,
                   help="file of '<task> <trial>' lines naming the EXACT clips to roll out, "
                        "overriding --tasks-file/--trials. scripts/dump_heldout_clips.py writes "
                        "one for a checkpoint's own held-out split -- that split is a seeded "
                        "sample of specific (task, trial) pairs scattered over many tasks, which "
                        "'every task in a file, trials 0..N-1' cannot express")
    p.add_argument("--init-from-reference", action="store_true",
                   help="start the rollout from the retargeted clip's frame 0 "
                        "(qvel 0) instead of humenv's Default standing reset. "
                        "Training uses Default, so this scores a DIFFERENT "
                        "quantity than the training curve -- it removes the "
                        "frame-0 pose mismatch that no z can fix")
    p.add_argument("--steps", type=int, default=None,
                   help="default: the first checkpoint cfg's steps_per_episode")
    p.add_argument("--clips-per-batch", type=int, default=4,
                   help="env slots are (n_checkpoints + 1) x this")
    p.add_argument("--save-qpos", default=None,
                   help="directory for <body>__<task>_t<trial>.npz (one qpos "
                        "array per checkpoint, plus z0 and the reference)")
    p.add_argument("--video-dir", default=None,
                   help="render <body>__<task>_t<trial>.mp4: one panel per "
                        "checkpoint, then z0's rollout, then the retargeted "
                        "reference, all on this body's own MJCF")
    p.add_argument("--camera", default="front_side", choices=["front_side", "side", "back"])
    p.add_argument("--width", type=int, default=416,
                   help="per panel. Keep (n_panels * w) and h divisible by 16, "
                        "which is what ffmpeg wants -- other values get silently "
                        "resized")
    p.add_argument("--height", type=int, default=416)
    p.add_argument("--out", default="outputs/ckpt_rollout")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.dataset import CrossEmbodimentDataset, load_task_list
    from model.networks import LatentAdapter
    from model.obs_scale import build_obs_multiplier
    from model.simple.train import compute_batch_cost

    loaded = [load_ckpt(c, a.device) for c in a.checkpoint]
    names = a.labels if a.labels else [n for n, _, _ in loaded]
    if len(names) != len(loaded):
        raise SystemExit(f"{len(names)} --labels for {len(loaded)} --checkpoint")
    M = len(loaded)
    # ONE scoring objective for every panel, taken from the first checkpoint.
    # Scoring each rollout under its own cfg's lambdas would put different
    # quantities in the same column and call the difference a result.
    score_cfg = loaded[0][1]
    for nm, cfg_i, _ in loaded[1:]:
        if (cfg_i.lambda_align, cfg_i.lambda_phys) != (score_cfg.lambda_align,
                                                       score_cfg.lambda_phys):
            print(f"NOTE: {nm} trained on "
                  f"{cfg_i.lambda_align:g}*L_align + {cfg_i.lambda_phys:g}*L_phys; "
                  f"scoring every panel with {names[0]}'s "
                  f"{score_cfg.lambda_align:g}/{score_cfg.lambda_phys:g} instead")

    ds_dir = REPO_ROOT / a.dataset
    want_pairs = None
    if a.clip_list:
        cl = Path(a.clip_list)
        if not cl.is_absolute():
            cl = REPO_ROOT / cl
        want_pairs = set()
        for line in load_task_list(cl):
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise SystemExit(f"{cl}: '{line}' is not '<task> <trial>'")
            want_pairs.add((parts[0], int(parts[1])))
        wanted = sorted({t for t, _ in want_pairs})
    else:
        tasks_path = Path(a.tasks_file) if a.tasks_file else ds_dir / "splits" / "test_tasks.txt"
        if not tasks_path.exists():
            raise SystemExit(f"{tasks_path} not found -- see scripts/split_tasks.py "
                             "and scripts/split_tasks_by_fall.py")
        wanted = load_task_list(tasks_path)
    dataset = CrossEmbodimentDataset(ds_dir, task_filter=wanted)

    def trained_tasks_of(cfg_i):
        tg = getattr(cfg_i, "task_group", "all")
        if tg == "all":
            return set(wanted), tg
        tp = ds_dir / "splits" / f"{tg}_tasks.txt"
        return (set(load_task_list(tp)) if tp.exists() else set()), tg

    by_body = dataset.indices_by_body()
    if a.bodies:
        bodies = [b for b in a.bodies if b in by_body]
        missing = set(a.bodies) - set(bodies)
        if missing:
            raise SystemExit(f"not in the manifest: {' '.join(sorted(missing))}")
    else:
        bp = ds_dir / "splits" / "test_bodies.txt"
        bodies = [b for b in load_task_list(bp) if b in by_body] if bp.exists() else []
        if not bodies:
            raise SystemExit(f"{bp} named no body present in the manifest; pass --bodies")

    if want_pairs is not None:
        rows_by_body = {b: [i for i in by_body[b]
                            if (dataset.rows[i]["reward_name"], int(dataset.rows[i]["trial"])) in want_pairs]
                        for b in bodies}
        for b, rs in rows_by_body.items():
            if len(rs) != len(want_pairs):
                have = {(dataset.rows[i]["reward_name"], int(dataset.rows[i]["trial"])) for i in rs}
                miss = sorted(want_pairs - have)[:5]
                raise SystemExit(f"{b}: the manifest has {len(rs)} of the {len(want_pairs)} listed "
                                 f"clips; missing e.g. {miss}")
    else:
        rows_by_body = {b: [i for i in by_body[b] if dataset.rows[i]["trial"] < a.trials]
                        for b in bodies}
    steps = a.steps or score_cfg.steps_per_episode

    print(f"{M} checkpoint(s), {len(bodies)} bodies, "
          + (f"{len(want_pairs)} listed clips" if want_pairs is not None
             else f"{len(wanted)} tasks x {a.trials} trial(s)")
          + f", {steps} steps")
    for (nm, cfg_i, ck), lab in zip(loaded, names):
        tset, tg = trained_tasks_of(cfg_i)
        tb = list(ck.get("bodies", []))
        print(f"  [{lab}] {nm}")
        print(f"      adapter: " + ("z0 + %g * MLP" % cfg_i.adapter_alpha
                                    if getattr(cfg_i, "adapter_residual", True)
                                    else "MLP (no residual)")
              + f", project={cfg_i.adapter_project_z}, obs_scale={cfg_i.obs_scale}, "
                f"lambda_z={cfg_i.lambda_z:g}, lambda_bc={getattr(cfg_i,'lambda_bc',0.0):g}")
        src = Path(a.clip_list).name if a.clip_list else tasks_path.name
        print(f"      trained: task group '{tg}' "
              f"({sum(t in tset for t in wanted)}/{len(wanted)} tasks of {src}), "
              f"bodies {' '.join(tb) or '?'}")
        print(f"      unseen here: bodies "
              f"{' '.join(b for b in bodies if b not in tb) or 'none'} | tasks "
              f"{' '.join(t for t in wanted if t not in tset) or 'none'}")
    score_loss = a.score_loss or ("bfm" if getattr(score_cfg, "align_loss", "joint") == "bfm" else "joint")
    if score_loss == "bfm":
        print("  scoring all panels with bfm: cost = 1 - mean_t cos(B(s_t), B(g_t))  "
              "(L_align / L_phys still logged, in joint space)")
    else:
        print(f"  scoring all panels with {score_cfg.lambda_align:g} * L_align + "
              f"{score_cfg.lambda_phys:g} * L_phys")
    print(f"  init: " + ("reference frame 0 (qvel 0)" if a.init_from_reference
                         else "humenv Default standing reset"))

    model = FBcprModel.from_pretrained(score_cfg.metamotivo_repo).to(a.device)
    model.eval()
    z_dim = model.cfg.archi.z_dim
    beta_dim = len(dataset[rows_by_body[bodies[0]][0]]["beta"])
    adapters = []
    for nm, cfg_i, ck in loaded:
        ad = LatentAdapter(
            beta_dim=beta_dim, z_dim=z_dim, hidden_dims=cfg_i.adapter_hidden_dims,
            alpha=cfg_i.adapter_alpha, alpha_learnable=cfg_i.adapter_alpha_learnable,
            project=getattr(cfg_i, "adapter_project_z", True),
            residual=getattr(cfg_i, "adapter_residual", True),
            # The geodesic head has one extra output unit, so a checkpoint
            # trained with it will not even load into the residual form -- the
            # head has to come from the checkpoint's own cfg, exactly as
            # project/residual already do.
            head=getattr(cfg_i, "adapter_head", "residual"),
            theta_max_deg=getattr(cfg_i, "adapter_theta_max_deg", 60.0),
        ).to(a.device)
        ad.load_state_dict(ck["adapter"])
        ad.eval()
        adapters.append(ad)

    out_dir = REPO_ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    qdir = Path(a.save_qpos) if a.save_qpos else None
    if qdir:
        qdir = qdir if qdir.is_absolute() else REPO_ROOT / qdir
        qdir.mkdir(parents=True, exist_ok=True)
    vdir = Path(a.video_dir) if a.video_dir else None
    if vdir:
        vdir = vdir if vdir.is_absolute() else REPO_ROOT / vdir
        vdir.mkdir(parents=True, exist_ok=True)

    C = a.clips_per_batch
    SLOTS = (M + 1) * C          # one slot per checkpoint per clip, plus z0
    records = []
    for body in bodies:
        idxs = rows_by_body[body]
        xml = ds_dir / dataset[idxs[0]]["target_xml"]
        env, _ = make_humenv(num_envs=SLOTS, vectorization_mode="sync", task=None,
                             xml=str(xml), state_init="Default")
        fk = mujoco.MjModel.from_xml_path(str(xml))
        obs_dim = env.single_observation_space["proprio"].shape[0]
        # One multiplier PER CHECKPOINT, assembled into a per-slot matrix: the
        # obs canonicalisation is part of the policy, so two checkpoints trained
        # under different obs_scale settings have to be shown different obs even
        # inside one batched rollout. Broadcasting one vector would quietly
        # evaluate one of them on the other's inputs.
        muls = []
        for _, cfg_i, _ in loaded:
            m = build_obs_multiplier(xml, REPO_ROOT / cfg_i.obs_scale_ref_xml,
                                     mode=cfg_i.obs_scale, parts=cfg_i.obs_scale_parts,
                                     verbose=False)
            muls.append(np.ones(obs_dim, dtype=np.float32) if m is None
                        else np.asarray(m, dtype=np.float32))
        z0_mul = build_obs_multiplier(xml, REPO_ROOT / score_cfg.obs_scale_ref_xml,
                                      mode=score_cfg.obs_scale,
                                      parts=score_cfg.obs_scale_parts, verbose=False)
        z0_mul = (np.ones(obs_dim, dtype=np.float32) if z0_mul is None
                  else np.asarray(z0_mul, dtype=np.float32))

        env1 = Bg_cache = None
        if score_loss == "bfm":
            from model import bfm_align
            env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
            Bg_cache = {}

        rend = rdata = font = None
        if vdir:
            rdata = mujoco.MjData(fk)
            rend = mujoco.Renderer(fk, height=a.height, width=a.width)
            font = _caption_font(max(13, a.height // 26))

        for s in range(0, len(idxs), C):
            chunk = [dataset[i] for i in idxs[s:s + C]]
            k = len(chunk)
            z0 = torch.tensor(np.stack([c["z0"] for c in chunk]),
                              dtype=torch.float32, device=a.device)
            beta = torch.tensor(np.stack([c["beta"] for c in chunk]),
                                dtype=torch.float32, device=a.device)
            with torch.no_grad():
                zs = [ad(beta, z0) for ad in adapters] + [z0]
            # Slot layout [ckpt0 x k | ckpt1 x k | ... | z0 x k | padding],
            # padded to the env's width so a short final chunk still steps a
            # full action array.
            pad = SLOTS - (M + 1) * k
            z_env = torch.cat(zs + ([z0[:1].expand(pad, -1)] if pad else []))
            mul = np.concatenate([np.repeat(m[None], k, 0) for m in muls]
                                 + [np.repeat(z0_mul[None], k, 0)]
                                 + ([np.repeat(z0_mul[None], pad, 0)] if pad else []))
            init = None
            if a.init_from_reference:
                init = [c["qpos_ref"][0] for c in chunk] * (M + 1) + [None] * pad
            if score_loss == "bfm":
                q, obs_hist = rollout(model, env, z_env, steps, a.device, mul, init, return_obs=True)
            else:
                q = rollout(model, env, z_env, steps, a.device, mul, init)
            refs = [c["qpos_ref"] for c in chunk] * (M + 1) + [chunk[0]["qpos_ref"]] * pad
            cost, la, lp = compute_batch_cost(fk, score_cfg, q, refs)
            if score_loss == "bfm":
                # B(g) per clip, cached: the reference is the same for every panel
                # of one clip, so it is embedded once and reused across the M+1 slots.
                for c in chunk:
                    key = (c["reward_name"], c["trial"])
                    if key not in Bg_cache:
                        Bg_cache[key] = bfm_align.reference_embeddings(
                            model, env1, c["qpos_ref"], a.device, z0_mul)
                cost = np.empty(len(refs), dtype=np.float32)
                for slot in range(len(refs)):
                    c = chunk[slot % k] if slot < (M + 1) * k else chunk[0]
                    Bg = Bg_cache[(c["reward_name"], c["trial"])]
                    cost[slot] = bfm_align.batch_bfm_align(
                        model, obs_hist[slot:slot + 1], Bg, a.device)[0]

            for j, c in enumerate(chunk):
                z0_slot = M * k + j
                rec = dict(body=body, task=c["reward_name"], trial=c["trial"],
                           cost_z0=float(cost[z0_slot]),
                           L_align_z0=float(la[z0_slot]), L_phys_z0=float(lp[z0_slot]))
                for m_i, lab in enumerate(names):
                    o = m_i * k + j
                    rec[f"cost[{lab}]"] = float(cost[o])
                    rec[f"L_align[{lab}]"] = float(la[o])
                    rec[f"L_phys[{lab}]"] = float(lp[o])
                    rec[f"gain[{lab}]"] = ((cost[o] - cost[z0_slot]) / cost[z0_slot]
                                           if cost[z0_slot] else 0.0)
                    tset, _ = trained_tasks_of(loaded[m_i][1])
                    rec[f"task_trained_on[{lab}]"] = int(c["reward_name"] in tset)
                records.append(rec)
                cols = "  ".join(f"{lab} {rec[f'cost[{lab}]']:7.4f} "
                                 f"({rec[f'gain[{lab}]']*100:+6.1f}%)" for lab in names)
                print(f"  {body:13s} {c['reward_name']:26s} t{c['trial']}  "
                      f"z0 {rec['cost_z0']:7.4f} | {cols}", flush=True)

                stem = f"{body}__{c['reward_name']}_t{c['trial']}"
                if qdir:
                    np.savez_compressed(
                        qdir / f"{stem}.npz",
                        **{f"qpos[{lab}]": q[m_i * k + j].astype(np.float32)
                           for m_i, lab in enumerate(names)},
                        qpos_z0=q[z0_slot].astype(np.float32),
                        qpos_ref=c["qpos_ref"].astype(np.float32),
                        fps=score_cfg.control_fps)
                if vdir:
                    def cap(name, ck_, lak, lpk):
                        # the caption must name the objective the cost column IS,
                        # not the joint-space one it used to always be: a bfm run
                        # captioned "1*La + 0*Lp" reads as a number it is not
                        if score_loss == "bfm":
                            return (f"{name}\n bfm {rec[ck_]:.3f}   "
                                    f"(La {rec[lak]:.2f}  Lp {rec[lpk]:.2f})")
                        return (f"{name}\n cost {rec[ck_]:.3f} = "
                                f"{score_cfg.lambda_align:g}*La {rec[lak]:.3f} + "
                                f"{score_cfg.lambda_phys:g}*Lp {rec[lpk]:.3f}")
                    labs = [cap(lab, f"cost[{lab}]", f"L_align[{lab}]", f"L_phys[{lab}]")
                            for lab in names]
                    labs.append(cap("z0 (no adapter)", "cost_z0", "L_align_z0", "L_phys_z0"))
                    labs.append(f"reference\n {body} / {c['reward_name']}")
                    frames = render_panels(
                        fk, rdata, rend,
                        [q[m_i * k + j] for m_i in range(M)] + [q[z0_slot], c["qpos_ref"]],
                        labs, a.camera, steps, font)
                    imageio.mimsave(vdir / f"{stem}.mp4", frames, fps=score_cfg.control_fps)
                    print(f"      -> {vdir.name}/{stem}.mp4", flush=True)
        if rend is not None:
            rend.close()
        env.close()
        if env1 is not None:
            env1.close()

    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

    print(f"\n{'body / checkpoint':44s} {'n':>3s} {'median gain':>12s} {'p10':>8s} "
          f"{'p90':>8s} {'improved':>10s}")
    for body in bodies:
        for lab in names:
            allr = [x for x in records if x["body"] == body]
            mixed = len({x[f"task_trained_on[{lab}]"] for x in allr}) > 1
            for split, sel in ((" [trained tasks]", 1), (" [unseen tasks]", 0), ("", None)):
                # The pooled row only says something the split rows do not when
                # both splits are actually present; otherwise it repeats one of
                # them verbatim.
                if sel is None and not mixed:
                    continue
                r = [x for x in allr if sel is None or x[f"task_trained_on[{lab}]"] == sel]
                if not r:
                    continue
                g = np.array([x[f"gain[{lab}]"] for x in r]) * 100
                print(f"{body + ' / ' + lab + split:44s} {len(r):3d} "
                      f"{np.median(g):11.1f}% {np.percentile(g,10):7.1f}% "
                      f"{np.percentile(g,90):7.1f}% {(g < 0).sum():>6d}/{len(r):<3d}")
    print("(gain is per-clip (checkpoint - z0)/z0 under one shared objective; "
          "negative means it beat z0)")
    print(f"-> {out_dir/'per_clip.csv'}" + (f"\n-> {qdir}" if qdir else "")
          + (f"\n-> {vdir}  ({len(records)} mp4)" if vdir else ""))


if __name__ == "__main__":
    main()
