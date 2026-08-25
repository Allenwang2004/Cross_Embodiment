#!/usr/bin/env python3
"""The end-to-end test: an adult motion the map has never seen, run on a body.

For each (body, held-out task):

    z_adult[t]  = tracking_inference of the ADULT performing the clip   (T, 256)
    z_body[t]   = G_theta(beta, z_adult[t])                             the map
    tau         = rollout(frozen actor, z_body[t]) on THAT BODY's MJCF

and the same rollout with z_adult[t] fed straight through, which is the control.
Without that control the video proves nothing: the frozen actor already does
something reasonable on a body it was not built for, and the question is
whether the map makes it better.

Video is three panels, left to right:

    reference   the retargeted motion, played back kinematically (no physics)
    mapped      the physical rollout under G(beta, z_adult)
    raw         the physical rollout under z_adult, unmapped

--obs-scale defaults to `auto`
-------------------------------
The frozen actor's obs normalizer is a BatchNorm holding ADULT running
statistics, so raw obs from a 0.62-scale body arrives at a systematic offset --
that is what model/obs_scale.py exists to remove, and what every other rollout in
this project does (model/simple/{train,evaluate,train_es}.py, and
scripts/rollout_z_on_body.py --obs-scale auto). Not canonicalising here would
make these L_align / L_phys numbers incomparable with all of them.

An earlier version defaulted to `none`, arguing that the target latents came out
of scripts/batch_infer_z.py on raw obs so the rollout should match. That
argument is weak -- the backward map B and the actor pi are different functions,
and nothing requires the obs they see to agree -- and it was wrong in practice:
measured over the same 40 held-out rollouts, `none` gave mapped L_align 26.85
against raw L_align 10.53, i.e. the map made things twice as bad. Run with --obs-scale none to
reproduce that ablation.

Usage (from project root):
    uv run scripts/zmap_rollout.py
    uv run scripts/zmap_rollout.py --bodies teen petite --trials 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.dataset import BETA_AXES, load_task_list
from model.networks import LatentAdapter
from model.obs_scale import build_obs_multiplier

ALIGN_WEIGHTS = {"root": 1.0, "ee": 1.0, "contact": 1.0, "pose": 1.0, "velocity": 1.0}


def render_reference(xml, qpos_seq, w, h, camera):
    """Kinematic playback: mj_forward only, no stepping."""
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=h, width=w)
    out = []
    for t in range(len(qpos_seq)):
        d.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camera)
        out.append(r.render().copy())
    r.close()
    return out


@torch.no_grad()
def rollout(model, env, z_seq, steps, device, obs_mul, init_qpos, render):
    """Per-frame z, deterministic actions. Returns (qpos, frames, diverged_at)."""
    env.reset()
    if init_qpos is not None:
        env.unwrapped.set_physics(qpos=init_qpos, qvel=np.zeros(env.unwrapped.model.nv))
    obs = env.unwrapped.get_obs()

    qpos_hist, frames, diverged = [], [], None
    for t in range(steps):
        proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        o = torch.as_tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0)
        z = z_seq[min(t, len(z_seq) - 1)].unsqueeze(0)
        a = model.act(o, z, mean=True).cpu().numpy().ravel()
        try:
            obs, _, term, trunc, info = env.step(a)
        except ValueError as e:
            # humenv raises on mjWARN_BADQACC. A body the actor cannot stabilise
            # diverges rather than merely falling, and that IS the result --
            # keep the frames up to that point.
            print(f"    DIVERGED at t={t}: {e}")
            diverged = t
            break
        qpos_hist.append(info["qpos"].copy())
        if render:
            frames.append(env.render())
        if term or trunc:
            break
    return (np.stack(qpos_hist) if qpos_hist else None), frames, diverged


def pad_to(frames, n, shape):
    """Freeze the last frame so three panels of different length still stack."""
    if not frames:
        return [np.zeros(shape, dtype=np.uint8)] * n
    return frames + [frames[-1]] * (n - len(frames))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/simple_zmap/latest.pt")
    p.add_argument("--out-dir", default="outputs/z_map_rollout")
    p.add_argument("--bodies", nargs="*", default=None,
                   help="default: the checkpoint's TRAINING bodies, which with the "
                        "default --tasks gives the body_train/task_test quadrant. "
                        "Pass the held-out bodies to get body_test/task_test, the "
                        "quadrant where neither the shape nor the motion was seen.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="default: the checkpoint's held-out tasks")
    p.add_argument("--trials", type=int, default=1, help="clips per task")
    p.add_argument("--steps", type=int, default=None, help="default: the clip's length")
    p.add_argument("--obs-scale", default="auto")
    p.add_argument("--obs-scale-parts", default="length")
    p.add_argument("--no-video", action="store_true", help="metrics only, much faster")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--camera", default="front_side")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    ck = torch.load(ROOT / a.ckpt, map_location=a.device, weights_only=False)
    cfg = ck["cfg"]
    dataset_dir = ROOT / cfg.dataset_dir
    bodies = a.bodies or ck["train_bodies"]
    tasks = a.tasks or ck.get("test_tasks") or load_task_list(
        dataset_dir / "splits" / "test_tasks.txt")
    out_dir = ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(bodies)} bodies x {len(tasks)} held-out tasks x {a.trials} trial(s)")
    print(f"tasks: {' '.join(tasks)}")

    rows = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text().splitlines() if l]
    by = {}
    for r in rows:
        by.setdefault((r["morphology_label"], r["reward_name"]), []).append(r)

    model = FBcprModel.from_pretrained(getattr(cfg, "metamotivo_repo",
                                               "facebook/metamotivo-M-1")).to(a.device)
    model.eval()
    adapter = LatentAdapter(
        beta_dim=len(BETA_AXES), z_dim=model.cfg.archi.z_dim,
        hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable,
        project=cfg.adapter_project_z).to(a.device)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()

    results = []
    for body in bodies:
        xml = dataset_dir / f"robots/{body}/robot.xml"
        env, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default",
                             render_width=a.width, render_height=a.height, camera=a.camera)
        fk = mujoco.MjModel.from_xml_path(str(xml))
        obs_mul = build_obs_multiplier(xml, ROOT / getattr(cfg, "obs_scale_ref_xml",
                                                           "assets/robots/adult/robot.xml"),
                                       mode=a.obs_scale, parts=a.obs_scale_parts, verbose=False)
        beta = json.loads((dataset_dir / f"robots/{body}/parameter.json").read_text())
        beta_t = torch.tensor([[beta[k] for k in BETA_AXES]], dtype=torch.float32, device=a.device)

        for task in tasks:
            for r in sorted(by.get((body, task), []), key=lambda x: x["trial"])[:a.trials]:
                stem = f"{task}_{r['trial']}"
                z_adult = torch.as_tensor(np.load(dataset_dir / r["infer_origin_z"]),
                                          dtype=torch.float32, device=a.device)
                with torch.no_grad():
                    z_mapped = adapter(beta_t.expand(len(z_adult), -1), z_adult)
                ref = np.load(dataset_dir / r["retargeted_motion"])["qpos"]
                steps = a.steps or len(ref)

                row = {"body": body, "task": task, "trial": r["trial"]}
                panels = {}
                for name, zs in (("mapped", z_mapped), ("raw", z_adult)):
                    qpos, frames, div = rollout(model, env, zs, steps, a.device, obs_mul,
                                                ref[0], not a.no_video)
                    panels[name] = frames
                    if qpos is None:
                        row.update({f"{name}_Lalign": float("nan"), f"{name}_Lphys": float("nan"),
                                    f"{name}_diverged": div, f"{name}_frames": 0})
                        continue
                    d, _ = losses.functional_equivalence(fk, qpos, ref, ALIGN_WEIGHTS)
                    lp, _ = losses.physics_penalty(fk, qpos)
                    row.update({f"{name}_Lalign": d, f"{name}_Lphys": lp,
                                f"{name}_diverged": div, f"{name}_frames": len(qpos)})
                    np.savez(out_dir / f"{body}__{stem}__{name}.npz", qpos=qpos)

                if not a.no_video:
                    refv = render_reference(xml, ref[:steps], a.width, a.height, a.camera)
                    n = max(len(refv), len(panels["mapped"]), len(panels["raw"]))
                    shape = (a.height, a.width, 3)
                    vid = [np.concatenate([f, g, h], axis=1) for f, g, h in
                           zip(pad_to(refv, n, shape), pad_to(panels["mapped"], n, shape),
                               pad_to(panels["raw"], n, shape))]
                    imageio.mimsave(out_dir / f"{body}__{stem}.mp4", vid, fps=a.fps)

                results.append(row)
                print(f"  {body:14s} {stem:22s} "
                      f"mapped L_align={row['mapped_Lalign']:7.3f} Lp={row['mapped_Lphys']:6.3f} | "
                      f"raw L_align={row['raw_Lalign']:7.3f} Lp={row['raw_Lphys']:6.3f}")
        env.close()

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    def col(k):
        v = np.array([r[k] for r in results], dtype=float)
        return v[np.isfinite(v)]

    print(f"\n=== {len(results)} rollouts on HELD-OUT motions ===")
    print(f"{'':10s} {'L_align mean':>13s} {'L_align median':>15s} {'L_phys mean':>12s} {'diverged':>9s}")
    for name in ("mapped", "raw"):
        d, l = col(f"{name}_Lalign"), col(f"{name}_Lphys")
        nd = sum(1 for r in results if r[f"{name}_diverged"] is not None)
        print(f"{name:10s} {d.mean():13.3f} {np.median(d):15.3f} {l.mean():12.3f} "
              f"{nd:6d}/{len(results)}")
    md, rd = col("mapped_Lalign"), col("raw_Lalign")
    if len(md) == len(rd) and len(md):
        wins = int((md < rd).sum())
        print(f"\nmapped beats raw on L_align in {wins}/{len(md)} rollouts "
              f"({100 * wins / len(md):.0f}%)")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
