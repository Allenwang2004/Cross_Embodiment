#!/usr/bin/env python3
"""plot_bfm_blind_paths.py -- different walking latents, same BFM cosine, very different paths. The 20 bfm-only searches of
outputs/single_z_seeds_s005_5k (child, move-ego-0-2_4, 20 seeds, cost = 1 - mean cos(B(rollout), B(reference)), no heading or
position term; from 09/18, obs_scale auto, not the later exact observation): each seed's best latent, its rollout's mean BFM cosine to the reference (1 - bfm), its pelvis path, and how
far it ended from the reference (position, heading). Writes outputs/z_heading/bfm_blind_paths.{png,json}.
usage: uv run python scripts/plot_bfm_blind_paths.py"""
import glob, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import humenv.utils as hu

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "outputs/single_z_seeds_s005_5k"
REF = REPO / "data/child/retargeting_motion/move-ego-0-2/move-ego-0-2_4.npz"
XML = REPO / "assets/robots_torque/child/robot_torque_full.xml"
INK, MUTED, GRID, SURF, BLUE, GREY = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb", "#2a78d6", "#a8a7a0"

m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)


def heading(Q):
    h = []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        h.append(hu.calc_heading(hu.remove_base_rot(d.xquat[1][None].copy(), "smpl"))[0])
    return np.array(h)


ref = np.load(REF)["qpos"]; href = heading(ref); rows = []; Z = []; paths = []
for f in sorted(glob.glob(str(RUNS / "move-ego-0-2_4_bfm_s*/summary.json"))):
    run = Path(f).parent; s = json.load(open(f)); q = np.load(run / "best.npz")["qpos"]; T = min(len(q), len(ref))
    dh = (np.degrees(heading(q[:T]) - href[:T]) + 180) % 360 - 180
    rows.append(dict(run=run.name, seed=s.get("seed"), cos=1 - s["best"]["bfm"], end_pos_m=float(np.linalg.norm(q[T - 1, :2] - ref[T - 1, :2])),
                     mean_pos_m=float(np.linalg.norm(q[:T, :2] - ref[:T, :2], axis=-1).mean()), end_heading_deg=float(dh[-5:].mean()),
                     z0_cos=1 - s["origin_z"]["bfm"], travel_m=float(np.linalg.norm(q[T - 1, :2] - q[0, :2]))))
    Z.append(np.load(run / "best_z.npy").reshape(-1)); paths.append(q[:T, :2])
Z = np.stack(Z); iu = np.triu_indices(len(Z), 1); dz = np.linalg.norm(Z[iu[0]] - Z[iu[1]], axis=1)
cos = np.array([r["cos"] for r in rows]); pos = np.array([r["end_pos_m"] for r in rows]); hd = np.array([r["end_heading_deg"] for r in rows])
summ = dict(n=len(rows), cos=dict(min=float(cos.min()), max=float(cos.max()), std=float(cos.std())), end_pos_m=dict(min=float(pos.min()), max=float(pos.max()), median=float(np.median(pos))),
            end_heading_deg=dict(min=float(hd.min()), max=float(hd.max())), z_distance=dict(min=float(dz.min()), median=float(np.median(dz)), max=float(dz.max())),
            ref_travel_m=float(np.linalg.norm(ref[-1, :2] - ref[0, :2])), z0_cos=rows[0]["z0_cos"], rows=rows)
(REPO / "outputs/z_heading").mkdir(exist_ok=True, parents=True)
(REPO / "outputs/z_heading/bfm_blind_paths.json").write_text(json.dumps(summ, indent=1))
print({k: v for k, v in summ.items() if k != "rows"})

fig, ax = plt.subplots(1, 3, figsize=(14.5, 5), dpi=170, facecolor=SURF, gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
o = ref[0, :2]
for i, p in enumerate(paths):
    ax[0].plot(p[:, 0] - o[0], p[:, 1] - o[1], color=BLUE, lw=1.1, alpha=0.55, zorder=2, label="20 searched latents" if i == 0 else None)
    ax[0].scatter(p[-1, 0] - o[0], p[-1, 1] - o[1], s=14, color=BLUE, zorder=3)
ax[0].plot(ref[:, 0] - o[0], ref[:, 1] - o[1], color=INK, lw=2.4, zorder=4, label="reference")
ax[0].scatter([0], [0], s=36, color=INK, zorder=5)
ax[0].set_aspect("equal", adjustable="datalim"); ax[0].set_xlabel("x (m)", fontsize=10, color=MUTED); ax[0].set_ylabel("y (m)", fontsize=10, color=MUTED)
ax[0].set_title("Their pelvis paths differ widely", fontsize=12, color=INK, loc="left", pad=10)
ax[0].legend(frameon=False, fontsize=9, loc="best", labelcolor=INK)
for b, x, xl, tt in ((ax[1], pos, "distance from the reference at the end (m)", "Their BFM cosine barely differs"), (ax[2], hd, "heading off the reference at the end (deg)", "...across headings too")):
    b.scatter(x, cos, s=46, color=BLUE, edgecolor=SURF, linewidth=1.2, zorder=3)
    b.set_ylim(0, 1); b.set_xlabel(xl, fontsize=10, color=MUTED); b.set_title(tt, fontsize=12, color=INK, loc="left", pad=10)
ax[1].set_ylabel("mean BFM cosine to the reference", fontsize=10, color=MUTED)
for b in ax:
    b.set_facecolor(SURF); b.grid(color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=0, labelsize=9)
    for s in ("top", "right", "left", "bottom"):
        b.spines[s].set_visible(False)
fig.suptitle("Child walking (move-ego-0-2_4): 20 latents from BFM-only searches, earlier observation scaling (obs_scale auto)", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(REPO / "outputs/z_heading/bfm_blind_paths.png", facecolor=SURF)
print("-> outputs/z_heading/bfm_blind_paths.png")
