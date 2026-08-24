"""Evaluate a trained LatentAdapter checkpoint over every task in its dataset,
on every body.

Unlike train.py (which samples actions for REINFORCE), this rolls out
deterministically (the frozen actor's mean action, no exploration noise) since
eval should be reproducible.

The actor is shown the same canonicalised obs the checkpoint was trained with --
the settings come from the pickled cfg, not from a flag, so an evaluation cannot
silently disagree with its training run. Checkpoints written before the
ActionHead was removed are refused rather than evaluated without their head,
which would score a policy that never existed.

One split axis: the body
-------------------------
There is no task holdout. Every task is trained on, on every training body, so
the only thing "held out" means here is a BODY the adapter never saw --
splits/test_bodies.txt, i.e. `giant` and `short_stocky`, picked to sit outside
the training range rather than inside it. The report's train/test rows are that
split and nothing else.

Each row is rolled out on ITS OWN body's MJCF with that body's obs multiplier;
rolling every row on one global cfg.target_xml would score the adapter against a
body it was not asked about. `--tasks-file` still exists for restricting the run
to a subset when a full pass is too slow, but it is a sampling convenience, not
a generalization measurement.

For each task: R_task (task reward), D (vs the retargeted reference motion,
skipped if a row has no retargeted_motion), L_phys -- reported per-task and
aggregated, plus optional per-task comparison videos (target-body rollout
side by side with its retargeted reference).

Usage (from project root):
    uv run model/simple/evaluate.py --checkpoint model/checkpoints/update_00200.pt
    uv run model/simple/evaluate.py --checkpoint ... --render-videos --out-dir outputs/eval
    uv run model/simple/evaluate.py --checkpoint ... --bodies giant short_stocky
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

# Running this file directly (not `-m model.simple.evaluate`) puts model/simple/
# itself on sys.path, not the repo root -- see the same fix in run_train.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from humenv import make_humenv
from humenv.env import make_from_name
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.simple.config import TrainConfig
from model.dataset import BETA_AXES, CrossEmbodimentDataset, load_beta, load_task_list
from model.obs_scale import build_obs_multiplier
from model.networks import LatentAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_body_ctx(cfg, dataset_dir, xml_rel, device):
    """One body's eval env + obs multiplier. `xml_rel` is the manifest's
    target_xml, resolved through the dataset directory's `robots` symlink."""
    xml = dataset_dir / xml_rel
    env, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
    return {
        "env": env,
        # Same canonicalisation the checkpoint was trained under -- read off the
        # pickled cfg so eval and training cannot drift apart.
        "obs_mul": build_obs_multiplier(
            xml,
            REPO_ROOT / getattr(cfg, "obs_scale_ref_xml", "assets/robots/adult/robot.xml"),
            mode=getattr(cfg, "obs_scale", "auto"),
            parts=getattr(cfg, "obs_scale_parts", "length"),
            verbose=False,
        ),
    }


def rollout_deterministic(model, adapter, env, reward_fn, z0_t, beta_t, cfg,
                          obs_mul=None, record_video=False):
    """obs_mul: (358,) multiplier from model/obs_scale.py, or None for the raw
    obs -- the actor's view only. R_task, D and L_phys are all scored from the
    real body's qpos/qvel, so they stay comparable with baseline.py."""
    z_beta = adapter(beta_t, z0_t)

    obs, _ = env.reset()
    qpos_hist = []
    frames = [] if record_video else None
    r_task = 0.0

    with torch.no_grad():
        for t in range(cfg.steps_per_episode):
            proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
            obs_t = torch.tensor(proprio, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            obs_norm = model._normalize(obs_t)
            dist = model._actor(obs_norm, z_beta, model.cfg.actor_std)
            action_np = dist.mean.cpu().numpy().ravel()

            obs, _, terminated, truncated, info = env.step(action_np)
            qpos_hist.append(info["qpos"].copy())
            if record_video:
                frames.append(env.render())

            r_task += reward_fn(env.unwrapped.model, qpos=info["qpos"], qvel=info["qvel"], ctrl=action_np)

            if terminated or truncated:
                obs, _ = env.reset()

    return {
        "qpos_beta": np.stack(qpos_hist),
        "r_task": r_task,
        "frames": frames,
    }


def evaluate(checkpoint_path, tasks_file=None, trials_per_task=None, bodies=None,
             out_dir="outputs/eval", render_videos=False, device="cuda:0"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "action_head" in ckpt:
        raise SystemExit(
            f"{checkpoint_path} was trained with the ActionHead, which this "
            f"script no longer runs -- evaluating it adapter-only would score a "
            f"policy that never existed. Check out a revision from before the "
            f"head was removed, or retrain with the current model/simple/train.py."
        )
    cfg: TrainConfig = ckpt["cfg"]
    cfg.device = device

    dataset_dir = REPO_ROOT / cfg.dataset_dir
    # Default is EVERY task in the manifest. There is no task holdout to honour;
    # --tasks-file is only for cutting a full pass down when it is too slow.
    task_list = load_task_list(tasks_file) if tasks_file else None
    dataset = CrossEmbodimentDataset(dataset_dir, task_filter=task_list)
    beta_dim = len(BETA_AXES)

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(device)
    model.eval()

    adapter = LatentAdapter(
        beta_dim=beta_dim, z_dim=model.cfg.archi.z_dim,
        hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable,
        # from the pickled cfg, like obs_scale: projection is part of the
        # forward pass, so evaluating with a different setting than the
        # checkpoint trained under would score a different policy.
        project=getattr(cfg, "adapter_project_z", True),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    by_body = dataset.indices_by_body()
    if bodies:
        missing = sorted(set(bodies) - set(by_body))
        if missing:
            raise SystemExit(f"--bodies not in {dataset_dir}: {' '.join(missing)}")
        by_body = {b: by_body[b] for b in bodies}
    trained_on = set(ckpt.get("bodies") or [])

    ctxs = {}
    for b, idxs in by_body.items():
        xml_rel = dataset[idxs[0]]["target_xml"]
        if xml_rel is None:  # legacy single-body manifest
            ctxs[b] = make_body_ctx(cfg, REPO_ROOT, cfg.target_xml, device)
        else:
            ctxs[b] = make_body_ctx(cfg, dataset_dir, xml_rel, device)
    held = sorted(set(by_body) - trained_on) if trained_on else []
    print(f"{len(by_body)} bodies: {' '.join(by_body)}"
          + (f"   (never trained on: {' '.join(held)})" if held else ""))

    d_weights = {
        "root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
        "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight,
    }

    out_dir = Path(out_dir)
    video_dir = out_dir / "video"
    if render_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    per_row = []
    rendered = set()

    for body, idxs in by_body.items():
        ctx = ctxs[body]
        env = ctx["env"]
        trials_seen = defaultdict(int)  # the cap is PER BODY, not shared

        for idx in idxs:
            sample = dataset[idx]
            reward_name = sample["reward_name"]
            if trials_per_task is not None and trials_seen[reward_name] >= trials_per_task:
                continue
            trials_seen[reward_name] += 1

            reward_fn = make_from_name(reward_name)
            z0_t = torch.tensor(sample["z0"], dtype=torch.float32, device=device).unsqueeze(0)
            beta_t = torch.tensor(sample["beta"], dtype=torch.float32, device=device).unsqueeze(0)

            record_video = render_videos and (body, reward_name) not in rendered
            episode = rollout_deterministic(model, adapter, env, reward_fn, z0_t, beta_t,
                                            cfg, obs_mul=ctx["obs_mul"],
                                            record_video=record_video)

            d_total, d_terms = losses.functional_equivalence(
                env.unwrapped.model, episode["qpos_beta"], sample["qpos_ref"], d_weights
            )
            l_phys, _ = losses.physics_penalty(env.unwrapped.model, episode["qpos_beta"])

            per_row.append({
                "body": body, "body_split": "train" if body in trained_on else "test",
                "reward_name": reward_name, "trial": sample["trial"],
                "r_task": episode["r_task"], "d_total": d_total, "l_phys": l_phys, **d_terms,
            })

            if record_video:
                import imageio
                (video_dir / body).mkdir(parents=True, exist_ok=True)
                imageio.mimsave(video_dir / body / f"{reward_name}.mp4", episode["frames"], fps=30)
                rendered.add((body, reward_name))

            print(f"[{body} {reward_name} trial {sample['trial']}] "
                  f"r_task={episode['r_task']:.4f} D={d_total:.4f} L_phys={l_phys:.4f}")

    for ctx in ctxs.values():
        ctx["env"].close()

    per_task = defaultdict(list)
    for row in per_row:
        per_task[row["reward_name"]].append(row)

    summary = {}
    for reward_name, rows in per_task.items():
        summary[reward_name] = {
            "n_trials": len(rows),
            "r_task_mean": float(np.mean([r["r_task"] for r in rows])),
            "r_task_std": float(np.std([r["r_task"] for r in rows])),
            "d_total_mean": float(np.mean([r["d_total"] for r in rows])),
            "l_phys_mean": float(np.mean([r["l_phys"] for r in rows])),
        }

    def agg(rows):
        return {
            "n_rows": len(rows),
            "r_task_mean": float(np.mean([r["r_task"] for r in rows])),
            "d_total_mean": float(np.mean([r["d_total"] for r in rows])),
            "l_phys_mean": float(np.mean([r["l_phys"] for r in rows])),
        }

    per_body_rows = defaultdict(list)
    per_split_rows = defaultdict(list)
    for row in per_row:
        per_body_rows[row["body"]].append(row)
        per_split_rows[row["body_split"]].append(row)
    per_body = {b: agg(rows) for b, rows in per_body_rows.items()}
    # The other half of the quadrant: `--tasks-file` picks the task axis, this
    # splits the body axis, so one run reports two of the four cells.
    per_body_split = {k: agg(rows) for k, rows in per_split_rows.items()}

    overall = {
        "n_tasks": len(per_task),
        "n_bodies": len(per_body),
        "n_rows": len(per_row),
        "r_task_mean": float(np.mean([r["r_task"] for r in per_row])) if per_row else None,
        "d_total_mean": float(np.mean([r["d_total"] for r in per_row])) if per_row else None,
        "l_phys_mean": float(np.mean([r["l_phys"] for r in per_row])) if per_row else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(
        {"checkpoint": str(checkpoint_path),
         "tasks_file": str(tasks_file) if tasks_file else "all",
         "trained_on_bodies": sorted(trained_on),
         "overall": overall, "per_body_split": per_body_split,
         "per_body": per_body, "per_task": summary, "per_row": per_row},
        indent=2,
    ))

    print("\n=== summary ===")
    print(f"{overall['n_tasks']} tasks x {overall['n_bodies']} bodies, "
          f"{overall['n_rows']} rows   "
          f"(tasks: {Path(tasks_file).name if tasks_file else 'all'})")
    print(f"{'body':14s} {'split':6s} {'rows':>5s} {'r_task':>9s} {'D':>9s} {'L_phys':>9s}")
    for b in sorted(per_body, key=lambda x: per_body_rows[x][0]["body_split"]):
        a = per_body[b]
        print(f"{b:14s} {per_body_rows[b][0]['body_split']:6s} {a['n_rows']:5d} "
              f"{a['r_task_mean']:9.4f} {a['d_total_mean']:9.4f} {a['l_phys_mean']:9.4f}")
    for k in ("train", "test"):
        if k in per_body_split:
            a = per_body_split[k]
            print(f"{'-- ' + k + ' bodies':21s} {a['n_rows']:5d} "
                  f"{a['r_task_mean']:9.4f} {a['d_total_mean']:9.4f} {a['l_phys_mean']:9.4f}")
    print(f"full report -> {report_path}")

    return overall, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks-file", default=None,
                         help="restrict to the tasks listed in this file "
                              "(default: every task in the checkpoint's dataset). "
                              "A speed knob, not a holdout -- there is no task split")
    parser.add_argument("--trials-per-task", type=int, default=None,
                         help="cap trials evaluated per task (default: all "
                              "available in the dataset)")
    parser.add_argument("--bodies", nargs="*", default=None,
                        help="restrict to these bodies (default: every body in "
                             "the checkpoint's dataset, train and held-out alike)")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--render-videos", action="store_true",
                         help="save one comparison video per task (first trial only)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        tasks_file=args.tasks_file,
        trials_per_task=args.trials_per_task,
        bodies=args.bodies,
        out_dir=args.out_dir,
        render_videos=args.render_videos,
        device=args.device,
    )


if __name__ == "__main__":
    main()
