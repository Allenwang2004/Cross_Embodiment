#!/usr/bin/env python3
"""slippery_velstart.py -- does starting in motion help on a slippery floor? The searches start every rollout from the
clip's first frame AT REST, while the reference is already moving at that frame, so the body has to accelerate at once;
on a slippery floor that is where it slips. Here the start velocity is the reference's own, by finite difference of
its frames 0 and 1 (30 Hz). No new search: z0 from rest (as before), z0 from the reference velocity, and the z an
earlier search found (single_z_search --floor-friction, cost with heading and position) from the reference velocity.
Per floor friction, the 3-panel video outputs/slippery/<stem>/search_<name>.mp4 (name = mu<mu>[_<tag>], replacing the
search's own): [the adult's original motion | z0 from the reference velocity | the searched z from the reference
velocity], captioned with task return, bfm against the original motion, travel and the lowest head height; numbers
(z0 from rest too) in velstart_<name>.json.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/slippery_velstart.py [--clip move-ego-0-2/move-ego-0-2_2] [--mu 0.3 0.2] [--tag nopos]
"""
import argparse, functools, json, os, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.friction import FloorFrictionWrapper                    # noqa: E402
from child_reward_eval import reward_of, rollout                   # noqa: E402
from survey_body_videos import ADULT, BLUE, GREY, RED              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_2")
    ap.add_argument("--mu", type=float, nargs="+", default=[0.3, 0.2])
    ap.add_argument("--tag", default="", help="which search: <stem>_mu<mu>[_<tag>]_s0, e.g. nopos")
    a = ap.parse_args()
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    task, stem = a.clip.split("/")
    out = REPO / "outputs/slippery" / stem; out.mkdir(parents=True, exist_ok=True)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT)); da = mujoco.MjData(ma); head = ma.body("Head").id
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]; steps = len(ref)
    v0 = np.zeros(ma.nv); mujoco.mj_differentiatePos(ma, v0, 1 / 30, ref[0], ref[1])
    print(f"reference start velocity: pelvis {np.linalg.norm(v0[:3]):.2f} m/s")
    z0 = np.load(REPO / f"data/origin_z/{a.clip}.npy").reshape(-1)
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / ADULT), state_init="Default")
    emb = lambda Q: bfm_align.embed(model, bfm_align.obs_from_qpos(env1, Q), "cuda")
    B0 = emb(ref); rew = reward_of(task)
    for mu in a.mu:
        name = f"mu{mu:g}" + (f"_{a.tag}" if a.tag else "")               # mu0.3, mu0.3_nopos
        run = REPO / f"outputs/slippery/search/{stem}_{name}_s0"
        zb = np.load(run / "best_z.npy").reshape(-1)
        env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(REPO / ADULT), state_init="Default",
                             wrappers=[functools.partial(FloorFrictionWrapper, mu=mu)])
        res, panels = {}, [dict(title="adult, original motion (normal floor)", sub=stem, xml=ADULT,
                                qpos=f"data/origin_motion/{a.clip}.npz", rgba=GREY)]
        for key, z, vel, title, col in (("z0_rest", z0, None, "z0, start at rest", RED),
                                        ("z0_moving", z0, v0, "z0, start at the reference velocity", RED),
                                        ("searched_moving", zb, v0, "searched z, start at the reference velocity", BLUE)):
            Q, V, A = rollout(model, env, [z] * 16, ref[0], steps, ma.nv, qvel0=vel)
            q, v, u = Q[0], V[0], A[0]
            hz = []
            for qq in q:
                da.qpos[:] = qq; mujoco.mj_kinematics(ma, da); hz.append(da.xpos[head][2])
            r = dict(return_=float(sum(rew(ma, qq, vv, aa) for qq, vv, aa in zip(q, v, u))), bfm=bfm_align.bfm_align_loss(emb(q), B0),
                     travel_m=float(np.linalg.norm(q[-1, :2] - q[0, :2])), end_off_m=float(np.linalg.norm(q[-1, :2] - ref[-1, :2])),
                     head_min_m=float(min(hz)))
            res[key] = r
            np.savez_compressed(out / f"velstart_{name}_{key}.npz", qpos=q.astype(np.float32), fps=30)
            print(f"friction {mu:g} {key:16s} return {r['return_']:6.1f}/{steps} | bfm {r['bfm']:.3f} | travel {r['travel_m']:.1f} m | "
                  f"end-point off {r['end_off_m']:.2f} m | head lowest {r['head_min_m']:.2f} m", flush=True)
            if key == "z0_rest":                       # numbers only: the video shows the moving starts
                continue
            panels.append(dict(title=f"{title}, friction {mu:g}",
                               sub=f"return {r['return_']:.0f}/{steps}, bfm {r['bfm']:.3f}, travel {r['travel_m']:.1f} m, head lowest {r['head_min_m']:.2f} m",
                               xml=ADULT, qpos=str((out / f"velstart_{name}_{key}.npz").relative_to(REPO)), rgba=col))
        env.close()
        (out / f"velstart_{name}.json").write_text(json.dumps(dict(clip=a.clip, floor_friction=mu, search_run=str(run.relative_to(REPO)),
                                                                     start_velocity="reference frames 0-1", results=res), indent=1))
        k = 1.74 / 1.26
        spec = dict(out=str((out / f"search_{name}.mp4").relative_to(REPO)), cols=3, size=1080,
                    track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)), panels=panels)
        sp = out / f"spec_search_{name}.json"; sp.write_text(json.dumps(spec, indent=1))
        subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(sp)], cwd=REPO, check=True,
                       env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2"})


if __name__ == "__main__":
    main()
