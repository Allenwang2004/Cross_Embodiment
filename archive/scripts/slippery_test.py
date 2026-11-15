#!/usr/bin/env python3
"""slippery_test.py -- the adult body, its clip's z0, on floors of different friction (model/friction.py: the floor
wins every contact; "original" = the unchanged model, where the feet's 1.0 decides foot contacts). No search.

Each condition: z0 rolled out from the clip's first frame at rest, 16 copies in one batch as the searches roll z0,
for the clip's length. Reported per condition: humenv's own task return (max = steps), bfm against the adult's original
motion (1 - mean_t cos(B(obs), B(obs_original))), the distance the pelvis travelled, how far it ends from where the
original motion ends, the lowest head height (a fall shows as the head near the floor).
Writes outputs/slippery/<clip>/{mu_*.npz, summary.json, friction.mp4 (+ .png)}: 3 x 2 panels of 1080 px.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/slippery_test.py [--clip move-ego-0-2/move-ego-0-2_2] [--mu 0.5 0.3 ...]
"""
import argparse, functools, json, os, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.friction import FloorFrictionWrapper                  # noqa: E402
from child_reward_eval import reward_of, rollout                 # noqa: E402

ADULT = "assets/robots/adult/robot.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_2")
    ap.add_argument("--mu", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1, 0.05])
    a = ap.parse_args()
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    task, stem = a.clip.split("/")
    out = REPO / "outputs/slippery" / stem; out.mkdir(parents=True, exist_ok=True)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT)); da = mujoco.MjData(ma); head = ma.body("Head").id
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]; steps = len(ref)
    z0 = np.load(REPO / f"data/origin_z/{a.clip}.npy").reshape(-1)
    rew = reward_of(task)
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / ADULT), state_init="Default")
    emb = lambda Q: bfm_align.embed(model, bfm_align.obs_from_qpos(env1, Q), "cuda")
    B0 = emb(ref)
    res, panels = {}, []
    for mu in [None] + list(a.mu):
        tag = "original" if mu is None else f"{mu:g}"
        wr = [] if mu is None else [functools.partial(FloorFrictionWrapper, mu=mu)]
        env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(REPO / ADULT), state_init="Default",
                             wrappers=wr)
        Q, V, A = rollout(model, env, [z0] * 16, ref[0], steps, ma.nv)
        env.close()
        q, v, u = Q[0], V[0], A[0]
        ret = float(sum(rew(ma, qq, vv, aa) for qq, vv, aa in zip(q, v, u)))
        bfm = bfm_align.bfm_align_loss(emb(q), B0)
        hz = []
        for qq in q:
            da.qpos[:] = qq; mujoco.mj_kinematics(ma, da); hz.append(da.xpos[head][2])
        r = dict(return_=ret, max_return=steps, bfm=bfm, travel_m=float(np.linalg.norm(q[-1, :2] - q[0, :2])),
                 ref_travel_m=float(np.linalg.norm(ref[-1, :2] - ref[0, :2])), end_off_m=float(np.linalg.norm(q[-1, :2] - ref[-1, :2])),
                 head_min_m=float(min(hz)))
        res[tag] = r
        np.savez_compressed(out / f"mu_{tag}.npz", qpos=q.astype(np.float32), fps=30)
        print(f"friction {tag:8s} return {ret:6.1f}/{steps} | bfm {bfm:.3f} | travel {r['travel_m']:.2f} m (original motion "
              f"{r['ref_travel_m']:.2f}) | end-point off {r['end_off_m']:.2f} m | head lowest {r['head_min_m']:.2f} m", flush=True)
        panels.append(dict(title=("floor friction: original (feet 1.0)" if mu is None else f"floor friction {mu:g}"),
                           sub=f"return {ret:.0f}/{steps}, bfm {bfm:.3f}, travel {r['travel_m']:.1f} m, head lowest {r['head_min_m']:.2f} m",
                           xml=ADULT, qpos=str((out / f"mu_{tag}.npz").relative_to(REPO))))
    (out / "summary.json").write_text(json.dumps(dict(clip=a.clip, z="z0", body="adult", results=res), indent=1))
    k = 1.74 / 1.26                                                      # the survey's adult framing
    spec = dict(out=str((out / "friction.mp4").relative_to(REPO)), cols=3, size=1080,
                track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)), panels=panels)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(out / "spec.json")], cwd=REPO, check=True,
                   env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2"})


if __name__ == "__main__":
    main()
