#!/usr/bin/env python3
"""z_heading_spread.py -- does Motivo care which way the body faces? One z (the clip's z0), the adult's own body, deterministic
actor, many rollouts of 300 steps (10 s), all from the clip's frame 0 at rest:
  same    16 identical starts: the baseline, how much numerics alone change a rollout
  noise   slot 0 as is, the other 15 with N(0, sigma^2) added to each joint angle (root untouched; default 0.01 rad = 0.6 deg)
  yaw     the start turned about world z by 0 / 90 / 180 / -90 deg: if the policy has no notion of world heading, each rollout
          is the first one turned by that angle
Against slot 0: heading difference (the search cost's heading: humenv.utils.calc_heading on the root), pelvis xy distance,
joint-angle RMS, bfm (1 - mean cos of B(obs), both rollouts' own observations), and whether the pelvis ever fell below half of slot 0's.
Writes outputs/z_heading/{summary.json, <clip>.npz, heading_spread.png}.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/z_heading_spread.py [--sigma 0.01] [--clips task/stem ...]"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.exact_obs import ADULT_XML                                    # noqa: E402
from model import bfm_align                                              # noqa: E402
from single_z_search import project_z                                    # noqa: E402

OUT = REPO / "outputs/z_heading"
CLIPS = [("move-ego-0-0/move-ego-0-0_0", "stand"), ("move-ego-0-2/move-ego-0-2_2", "walk 2 m/s"), ("move-ego-0-4/move-ego-0-4_8", "run 4 m/s"),
         ("raisearms-l-l/raisearms-l-l_8", "raise arms"), ("crouch-0/crouch-0_8", "crouch"), ("headstand/headstand_4", "headstand")]
INK, MUTED, GRID, SURF, BLUE, GREY = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb", "#2a78d6", "#a8a7a0"


def wrap_deg(x):
    return (np.degrees(x) + 180) % 360 - 180


@torch.no_grad()
def rollout(model, env, z, starts, steps):
    """starts: (n, nq), one per slot; the rest of the slots (sync env) unused is an error."""
    env.reset(); nv = env.envs[0].unwrapped.model.nv
    for e, q in zip(env.envs, starts):
        e.unwrapped.set_physics(qpos=q, qvel=np.zeros(nv))
    obs = np.stack([e.unwrapped.get_obs()["proprio"] for e in env.envs])
    zt = torch.as_tensor(np.repeat(z[None], len(starts), 0), dtype=torch.float32, device="cuda")
    Q, O = [], []
    for _ in range(steps):
        a = model._actor(model._normalize(torch.as_tensor(obs, dtype=torch.float32, device="cuda")), zt, model.cfg.actor_std).mean.cpu().numpy()
        o, _, _, _, info = env.step(a)
        obs = o["proprio"]; Q.append(info["qpos"].copy()); O.append(obs.copy())
    return np.stack(Q, 1), np.stack(O, 1)                                # (n, T, nq), (n, T, 358)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", nargs="*", default=None)
    a = ap.parse_args()
    clips = [(c, c.split("/")[1]) for c in a.clips] if a.clips else CLIPS
    OUT.mkdir(parents=True, exist_ok=True)
    from humenv import make_humenv
    import humenv.utils as hu
    from metamotivo.fb_cpr.huggingface import FBcprModel
    xml = str(REPO / ADULT_XML) if not Path(ADULT_XML).is_absolute() else str(ADULT_XML)
    env, _ = make_humenv(num_envs=16, vectorization_mode="sync", task=None, xml=xml, state_init="Default")
    fk = mujoco.MjModel.from_xml_path(xml); dk = mujoco.MjData(fk)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda"); model.eval()
    rng = np.random.default_rng(a.seed)

    def heading(Q):
        h = np.empty(Q.shape[:2])
        for i in range(Q.shape[0]):
            for t in range(Q.shape[1]):
                dk.qpos[:] = Q[i, t]; mujoco.mj_kinematics(fk, dk)
                h[i, t] = hu.calc_heading(hu.remove_base_rot(dk.xquat[1][None].copy(), "smpl"))[0]
        return h

    summary, panels = {}, []
    for clip, label in clips:
        stem = clip.split("/")[1]; q0 = np.load(REPO / f"data/origin_motion/{clip}.npz")["qpos"][0].astype(np.float64)
        z = project_z(np.load(REPO / f"data/origin_z/{clip}.npy").reshape(-1).astype(np.float64))
        # same
        Qs, _ = rollout(model, env, z, np.repeat(q0[None], 16, 0), a.steps)
        same_max = float(np.abs(Qs - Qs[0]).max())
        # noise
        st = np.repeat(q0[None], 16, 0); st[1:, 7:] += a.sigma * rng.standard_normal((15, q0.size - 7))
        sn = st.copy(); Qn, On = rollout(model, env, z, sn, a.steps); Hn = heading(Qn)
        dh = wrap_deg(Hn[1:] - Hn[0]); xy = np.linalg.norm(Qn[1:, :, :2] - Qn[0, :, :2], axis=-1)
        jr = np.degrees(np.sqrt(((Qn[1:, :, 7:] - Qn[0, :, 7:]) ** 2).mean((1, 2))))
        B = bfm_align.embed(model, On.reshape(-1, On.shape[-1]), "cuda").reshape(16, a.steps, -1)
        B = B / np.linalg.norm(B, axis=-1, keepdims=True)
        bfm = 1 - (B[1:] * B[0]).sum(-1).mean(1)
        fell = (Qn[1:, :, 2] < 0.5 * Qn[0, :, 2]).any(1)
        # yaw
        yaws = np.radians([0, 90, 180, -90]); st = np.repeat(q0[None], 16, 0)
        for i in range(16):
            y = yaws[i % 4]; qz = np.array([np.cos(y / 2), 0, 0, np.sin(y / 2)]); r = np.empty(4); mujoco.mju_mulQuat(r, qz, q0[3:7]); st[i, 3:7] = r
        Qy, _ = rollout(model, env, z, st, a.steps); Hy = heading(Qy)
        yaw_err = max(float(np.abs(wrap_deg(Hy[i] - Hy[0] - yaws[i])).max()) for i in range(1, 4))
        def rot(xy_, y):
            c, s = np.cos(-y), np.sin(-y); d = xy_ - q0[:2]
            return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], -1)
        yaw_xy_err = max(float(np.linalg.norm(rot(Qy[i, :, :2], yaws[i]) - (Qy[0, :, :2] - q0[:2]), axis=-1).max()) for i in range(1, 4))
        ix = {s: min(int(s * 30), a.steps) - 1 for s in (1, 5, 10)}
        ok = ~fell
        r = dict(label=label, sigma=a.sigma, same_max_abs_qpos_diff=same_max, n_fell=int(fell.sum()),
                 base_travel=float(np.linalg.norm(Qn[0, -1, :2] - q0[:2])), base_pelvis_end=float(Qn[0, -1, 2]),
                 heading_abs_deg={f"{s}s": dict(median=float(np.median(np.abs(dh[:, t]))), max=float(np.abs(dh[:, t]).max())) for s, t in ix.items()},
                 heading_abs_deg_not_fallen_10s=float(np.median(np.abs(dh[ok, -1]))) if ok.any() else None,
                 xy_m_10s=dict(median=float(np.median(xy[:, -1])), max=float(xy[:, -1].max())), joint_rms_deg_median=float(np.median(jr)),
                 joint_rms_deg_not_fallen=float(np.median(jr[ok])) if ok.any() else None,
                 bfm_to_base_median=float(np.median(bfm)), bfm_to_base_not_fallen=float(np.median(bfm[ok])) if ok.any() else None,
                 yaw_start_heading_err_deg=yaw_err, yaw_start_xy_err_m=yaw_xy_err)
        summary[stem] = r; panels.append((label, Hn, Qn))
        np.savez_compressed(OUT / f"{stem}.npz", qpos_noise=Qn.astype(np.float32), heading_noise=Hn.astype(np.float32), qpos_yaw=Qy.astype(np.float32),
                            starts_noise=sn.astype(np.float32), fps=np.float32(30))
        print(f"{label:12s} same diff {same_max:.1e} | fell {r['n_fell']}/15 | heading |diff| median @1s {r['heading_abs_deg']['1s']['median']:.1f} @5s {r['heading_abs_deg']['5s']['median']:.1f} @10s {r['heading_abs_deg']['10s']['median']:.1f} (max {r['heading_abs_deg']['10s']['max']:.1f}) "
              f"| xy @10s median {r['xy_m_10s']['median']:.2f} m | joint rms {r['joint_rms_deg_median']:.1f} deg | bfm {r['bfm_to_base_median']:.3f} | yaw start: heading err {yaw_err:.3f} deg, xy err {yaw_xy_err:.3f} m", flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    env.close()
    t = np.arange(a.steps) / 30
    fig, ax = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 6.6), dpi=170, facecolor=SURF, gridspec_kw=dict(height_ratios=[1, 1.15]), squeeze=False)
    for j, (label, H, Q) in enumerate(panels):
        d = wrap_deg(H - H[0]); up, lo = ax[0, j], ax[1, j]
        for i in range(1, 16):
            up.plot(t, d[i], color=GREY, lw=0.9, alpha=0.8, zorder=2, label="15 slightly different starts" if i == 1 else None)
            lo.plot(Q[i, :, 0] - Q[0, 0, 0], Q[i, :, 1] - Q[0, 0, 1], color=GREY, lw=0.9, alpha=0.8, zorder=2)
        up.plot(t, d[0], color=BLUE, lw=2, zorder=3, label="original start")
        lo.plot(Q[0, :, 0] - Q[0, 0, 0], Q[0, :, 1] - Q[0, 0, 1], color=BLUE, lw=2, zorder=3)
        lo.scatter([0], [0], s=30, color=INK, zorder=4)
        up.set_ylim(-185, 185); up.set_yticks([-180, -90, 0, 90, 180]); up.set_xlabel("time (s)", fontsize=9, color=MUTED); up.set_title(label, fontsize=11, color=INK, loc="left")
        lo.set_aspect("equal", adjustable="datalim"); lo.set_xlabel("x (m)", fontsize=9, color=MUTED)
        for b in (up, lo):
            b.set_facecolor(SURF); b.grid(color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=0, labelsize=8)
            for s in ("top", "right", "left", "bottom"):
                b.spines[s].set_visible(False)
    ax[0, 0].set_ylabel("heading vs. original start (deg)", fontsize=9, color=INK); ax[1, 0].set_ylabel("top view: y (m)", fontsize=9, color=INK)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=9, ncol=2, labelcolor=INK)
    fig.suptitle(f"Same z, adult body, joint angles of the start perturbed by {a.sigma} rad", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "heading_spread.png", facecolor=SURF)
    print("->", (OUT / "heading_spread.png").relative_to(REPO))


if __name__ == "__main__":
    main()
