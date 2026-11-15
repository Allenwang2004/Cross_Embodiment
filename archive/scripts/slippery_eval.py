#!/usr/bin/env python3
"""slippery_eval.py -- a single_z_search.py --floor-friction run on the adult: z0 and the searched z on that floor.

From the qpos the search saved (origin_z.npz, best.npz): bfm against the adult's original motion (normal floor), the
distance the pelvis travelled, the end-point offset, the lowest head height. The task return needs velocities and
actions, which the search does not save: z0 and the searched z are rolled out again (16 copies in one batch, on the
same floor) and the deviation of each re-roll from the saved qpos is reported with it. z0 on the normal floor is
rolled out too, as the reference value.
Video (outputs/slippery/<stem>/search_<tag>.mp4, tag = the run's name without clip and seed, e.g. mu0.3): [the adult's original motion, grey | z0 on the floor, red | the
searched z on the floor, blue], the survey's colours and adult framing, captioned with the numbers.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/slippery_eval.py outputs/slippery/search/<stem>_mu<mu>_s0
"""
import functools, json, os, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.friction import FloorFrictionWrapper                    # noqa: E402
from child_reward_eval import reward_of, rollout                   # noqa: E402
from survey_body_videos import ADULT, BLUE, GREY, RED              # noqa: E402


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    run = REPO / sys.argv[1]; s = json.loads((run / "summary.json").read_text())
    mu = s["floor_friction"]; task, stem = s["clip"].split("/")
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT)); da = mujoco.MjData(ma); head = ma.body("Head").id
    ref = np.load(REPO / f"data/origin_motion/{s['clip']}.npz")["qpos"]
    env1, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / ADULT), state_init="Default")
    emb = lambda Q: bfm_align.embed(model, bfm_align.obs_from_qpos(env1, Q), "cuda")
    B0 = emb(ref); rew = reward_of(task)
    env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(REPO / ADULT), state_init="Default",
                         wrappers=[functools.partial(FloorFrictionWrapper, mu=mu)])
    zs = {"z0": np.load(REPO / f"data/origin_z/{s['clip']}.npy").reshape(-1), "best": np.load(run / "best_z.npy").reshape(-1)}
    saved = {"z0": "origin_z.npz", "best": "best.npz"}
    res, panels = {}, [dict(title="adult, original motion (normal floor)", sub=stem, xml=ADULT,
                            qpos=f"data/origin_motion/{s['clip']}.npz", rgba=GREY)]
    for k, col in (("z0", RED), ("best", BLUE)):
        q = np.load(run / saved[k])["qpos"].astype(np.float64)
        hz = []
        for qq in q:
            da.qpos[:] = qq; mujoco.mj_kinematics(ma, da); hz.append(da.xpos[head][2])
        Q, V, A = rollout(model, env, [zs[k]] * 16, ref[0], s["steps"], ma.nv)
        ret = float(sum(rew(ma, qq, vv, aa) for qq, vv, aa in zip(Q[0], V[0], A[0])))
        n = min(len(q), len(ref))                                # never past the reference's last frame
        r = dict(bfm=bfm_align.bfm_align_loss(emb(q[:n]), B0[:n]), travel_m=float(np.linalg.norm(q[n - 1, :2] - q[0, :2])),
                 end_off_m=float(np.linalg.norm(q[n - 1, :2] - ref[n - 1, :2])), head_min_m=float(min(hz[:n])),
                 return_=ret, return_reroll_vs_saved_max_dq=float(np.abs(Q[0] - q).max()))
        res[k] = r
        print(f"{k:4s} on friction {mu}: return {ret:.1f}/{s['steps']} | bfm {r['bfm']:.3f} | travel {r['travel_m']:.2f} m | "
              f"end-point off {r['end_off_m']:.2f} m | head lowest {r['head_min_m']:.2f} m | re-roll vs saved {r['return_reroll_vs_saved_max_dq']:.1e}",
              flush=True)
        panels.append(dict(title=f"{'z0' if k == 'z0' else 'searched z'} on floor friction {mu:g}",
                           sub=f"return {ret:.0f}/{s['steps']}, bfm {r['bfm']:.3f}, travel {r['travel_m']:.1f} m, head lowest {r['head_min_m']:.2f} m",
                           xml=ADULT, qpos=str((run / saved[k]).relative_to(REPO)), rgba=col))
    env.close()
    # the reference value: z0 on the normal floor, the same protocol
    env0, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(REPO / ADULT), state_init="Default")
    Q, V, A = rollout(model, env0, [zs["z0"]] * 16, ref[0], s["steps"], ma.nv); env0.close()
    res["z0_normal_floor"] = dict(return_=float(sum(rew(ma, qq, vv, aa) for qq, vv, aa in zip(Q[0], V[0], A[0]))),
                                  bfm=bfm_align.bfm_align_loss(emb(Q[0]), B0[:len(Q[0])]))
    print(f"z0 on the normal floor: return {res['z0_normal_floor']['return_']:.1f}/{s['steps']} | bfm {res['z0_normal_floor']['bfm']:.3f}", flush=True)
    out = REPO / "outputs/slippery" / stem; out.mkdir(parents=True, exist_ok=True)
    tag = run.name.removeprefix(f"{stem}_").removesuffix(f"_s{s['seed']}")   # mu0.3, mu0.3_noroot, ...
    (out / f"search_{tag}.json").write_text(json.dumps(dict(run=str(run.relative_to(REPO)), floor_friction=mu, results=res), indent=1))
    k = 1.74 / 1.26
    spec = dict(out=str((out / f"search_{tag}.mp4").relative_to(REPO)), cols=3, size=1080,
                track=dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)), panels=panels)
    (out / f"spec_search_{tag}.json").write_text(json.dumps(spec, indent=1))
    subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(out / f"spec_search_{tag}.json")], cwd=REPO,
                   check=True, env={**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "2"})


if __name__ == "__main__":
    main()
