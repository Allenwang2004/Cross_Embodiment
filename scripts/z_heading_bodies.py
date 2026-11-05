#!/usr/bin/env python3
"""z_heading_bodies.py -- why does one z turn differently on different bodies? One z (the adult's z0), deterministic actor, 300 steps
from the body's own reference frame 0 at rest, as the searches roll out (adult-equivalent observation, model.exact_obs), on bodies going
from the adult to the child (the m2c morph path) plus fat and tall: slot 0 as is, 15 slots with N(0, sigma^2) on each joint angle.
A turn that follows the body smoothly and barely depends on the start is the body's own systematic turning; one that scatters is chaos.
Heading is taken from the root quaternion (humenv.utils.calc_heading, as the search cost), unwrapped, so turns are not wrapped to 180 deg.
Writes outputs/z_heading/bodies.{json,png}.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/z_heading_bodies.py [--sigma 0.01]"""
import argparse, functools, json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.exact_obs import ExactObs, ExactObsWrapper                    # noqa: E402
from single_z_search import project_z                                    # noqa: E402

ADULT = REPO / "assets/robots/adult/robot.xml"
CLIPS = [("move-ego-0-2", "move-ego-0-2_2", "walk 2 m/s"), ("move-ego-0-4", "move-ego-0-4_8", "run 4 m/s")]
BODIES = ["adult", "m2c_t250", "m2c_t500", "m2c_t750", "child", "fat", "tall"]
INK, MUTED, GRID, SURF, BLUE, GREY = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb", "#2a78d6", "#a8a7a0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from humenv import make_humenv
    import humenv.utils as hu
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda"); model.eval()
    rng = np.random.default_rng(a.seed); res = {}

    def heading(Q):
        return np.stack([np.unwrap([hu.calc_heading(hu.remove_base_rot(np.asarray(q[3:7], dtype=np.float64)[None], "smpl"))[0] for q in x]) for x in Q])

    for task, stem, label in CLIPS:
        adult_ref = np.load(REPO / f"data/origin_motion/{task}/{stem}.npz")["qpos"]
        z = project_z(np.load(REPO / f"data/origin_z/{task}/{stem}.npy").reshape(-1).astype(np.float64)); res[stem] = dict(label=label, bodies={})
        zt = torch.as_tensor(np.repeat(z[None], 16, 0), dtype=torch.float32, device="cuda")
        for body in BODIES:
            if body == "adult":
                xml, ref, kw = ADULT, adult_ref, dict(scale=1.0)
            else:
                xml = REPO / f"assets/robots_torque/{body}/robot_torque_full.xml"; ref = np.load(REPO / f"data/{body}/retargeting_motion/{task}/{stem}.npz")["qpos"]
                n = min(len(ref), len(adult_ref)); m = np.abs(adult_ref[:n, :2]) > 1e-3; kw = dict(scale=float(np.median(ref[:n, :2][m] / adult_ref[:n, :2][m])))
            env, _ = make_humenv(num_envs=16, vectorization_mode="sync", task=None, xml=str(xml), state_init="Default",
                                 wrappers=[functools.partial(ExactObsWrapper, child_xml=str(xml), adult_xml=str(ADULT), **kw)])
            ex = ExactObs(xml, ADULT, **kw); nv = env.envs[0].unwrapped.model.nv
            st = np.repeat(ref[0][None].astype(np.float64), 16, 0); st[1:, 7:] += a.sigma * rng.standard_normal((15, st.shape[1] - 7))
            env.reset()
            for e, q in zip(env.envs, st):
                e.unwrapped.set_physics(qpos=q, qvel=np.zeros(nv))
            obs = np.stack([ex(q, np.zeros(nv)) for q in st]); Q = []
            with torch.no_grad():
                for _ in range(a.steps):
                    u = model._actor(model._normalize(torch.as_tensor(obs, dtype=torch.float32, device="cuda")), zt, model.cfg.actor_std).mean.cpu().numpy()
                    _, _, _, _, info = env.step(u); obs = info["exact_proprio"].copy(); Q.append(info["qpos"].copy())
            env.close()
            Q = np.stack(Q, 1); H = np.degrees(heading(Q)); net = H[:, -1] - H[:, 0]
            s = kw["scale"]; fell = (Q[:, :, 2] < 0.5 * np.median(ref[:, 2])).any(1)
            res[stem]["bodies"][body] = dict(scale=s, net_deg=net.tolist(), base_net=float(net[0]), spread_median_deg=float(np.median(np.abs(net[1:] - net[0]))), spread_max_deg=float(np.abs(net[1:] - net[0]).max()),
                                            n_fell=int(fell.sum()), turn_by_s=[float(H[0, min(t, a.steps - 1)] - H[0, 0]) for t in (30, 60, 150, 299)], travel_m=float(np.linalg.norm(Q[0, -1, :2] - Q[0, 0, :2])))
            print(f"{label:11s} {body:9s} s {s:.3f} | net turn {net[0]:8.1f} deg | 15 starts: median |diff| {res[stem]['bodies'][body]['spread_median_deg']:7.1f} max {res[stem]['bodies'][body]['spread_max_deg']:7.1f} | fell {int(fell.sum())}/16 | at 1,2,5,10 s {np.round(res[stem]['bodies'][body]['turn_by_s'], 0)}", flush=True)
            (REPO / "outputs/z_heading/bodies.json").write_text(json.dumps(res, indent=1))
    path = ["adult", "m2c_t250", "m2c_t500", "m2c_t750", "child"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), dpi=170, facecolor=SURF)
    for b, (stem, r) in zip(ax, res.items()):
        xs = [r["bodies"][p]["scale"] for p in path]; base = [r["bodies"][p]["base_net"] for p in path]
        for x, p in zip(xs, path):
            v = np.array(r["bodies"][p]["net_deg"][1:])
            b.scatter(np.full(15, x), v, s=18, color=GREY, alpha=0.75, zorder=2, label="15 slightly different starts" if p == "adult" else None)
        b.plot(xs, base, color=BLUE, lw=2, marker="o", ms=6, zorder=3, label="original start")
        b.invert_xaxis(); b.set_xlabel("body size relative to the adult (adult 1.0 to child 0.61)", fontsize=10, color=MUTED); b.set_title(r["label"], fontsize=12, color=INK, loc="left")
        b.set_facecolor(SURF); b.grid(color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=0, labelsize=9)
        for sp in ("top", "right", "left", "bottom"):
            b.spines[sp].set_visible(False)
    ax[0].set_ylabel("total turn over 10 s (deg)", fontsize=10, color=INK); ax[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Same z on bodies from the adult to the child", x=0.01, ha="left", fontsize=13, color=INK); fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(REPO / "outputs/z_heading/bodies.png", facecolor=SURF); print("-> outputs/z_heading/bodies.png")


if __name__ == "__main__":
    main()
