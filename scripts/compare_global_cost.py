#!/usr/bin/env python3
"""compare_global_cost.py -- what adding heading and root-position terms to the
search cost changes, on walking (move-ego-0-2_4, child body, 20 seeds each).

  bfm only   outputs/single_z_seeds_s005_5k/move-ego-0-2_4_bfm_s*   (cost blind to
             heading and root x,y: humenv's proprio is heading-relative, drops x,y)
  + global   outputs/single_z_seeds_global/move-ego-0-2_4_s*        (bfm + 1.0 * heading
             term + 0.1 * mean root-xy distance)
  two-stage  outputs/single_z_seeds_twostage/move-ego-0-2_4_s*      (the bfm-only best of each
             seed, then 2048 more evals on the global cost at lr 0.03)

For each set: the bfm part of the cost (body posture tracking), heading error and
end position against the reference, and how unique the solutions are (pairwise
latent angle) and how alike their motions are (joint-angle difference).
"""
import csv, itertools, json, sys
from pathlib import Path
import numpy as np
import mujoco
import humenv.utils as hu
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
from model import losses
from model.simple.config import ESConfig
_cfg = ESConfig()
W = {"root": _cfg.d_root_weight, "ee": _cfg.d_ee_weight, "contact": _cfg.d_contact_weight,
     "pose": _cfg.d_pose_weight, "velocity": _cfg.d_velocity_weight}
# the L_align stage logs no bfm; it is re-scored by score_z_matrix.py into this csv
BFM_RESCORED = {r["label"].replace("align_", ""): float(r["cost"])
                for r in csv.DictReader(open(REPO / "outputs/seeds_compare/bfm_align_stage.csv"))} \
    if (REPO / "outputs/seeds_compare/bfm_align_stage.csv").exists() else {}
m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots_torque/child/robot_torque_full.xml")); d = mujoco.MjData(m)
ref = np.load(REPO / "data/child/retargeting_motion/move-ego-0-2/move-ego-0-2_4.npz")["qpos"]


def heading(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    return float(np.degrees(hu.calc_heading(hu.remove_base_rot(d.xquat[1][None].copy(), "smpl")))[0])


def load(pattern):
    R = []
    for dd in sorted((REPO / "outputs").glob(pattern)):
        if not (dd / "summary.json").exists():
            continue
        S = json.loads((dd / "summary.json").read_text()); Q = np.load(dd / "best.npz")["qpos"]
        T = min(len(Q), len(ref))
        dh = np.array([((heading(Q[t]) - heading(ref[t]) + 180) % 360) - 180 for t in range(0, T, 5)])
        z = np.load(dd / "best_z.npy").reshape(-1)
        bfm = S["best"]["bfm"]
        if bfm is None or not np.isfinite(bfm):
            bfm = BFM_RESCORED.get(dd.name.rsplit("_", 1)[-1], np.nan)
        la, _ = losses.functional_equivalence(m, Q[:T], ref[:T], W, 1.0 / _cfg.control_fps)
        R.append(dict(bfm=bfm, align=float(la), head=np.abs(dh).mean(), end_h=dh[-1],
                      end_pos=float(np.linalg.norm(Q[T - 1, :2] - ref[T - 1, :2])), z=z / np.linalg.norm(z), q=Q[:T]))
    return R


for name, pat in (("bfm only", "single_z_seeds_s005_5k/move-ego-0-2_4_bfm_s*"), ("+ global", "single_z_seeds_global/move-ego-0-2_4_s*"),
                  ("two-stage", "single_z_seeds_twostage/move-ego-0-2_4_s*"),
                  ("2-st align", "single_z_seeds_twostage_align/move-ego-0-2_4_s*")):
    R = load(pat)
    if not R:
        print(f"{name}: no results yet"); continue
    Z = np.stack([r["z"] for r in R]); pairs = list(itertools.combinations(range(len(R)), 2))
    ang = [np.degrees(np.arccos(np.clip(Z[i] @ Z[j], -1, 1))) for i, j in pairs]
    jd = [float(np.degrees(np.abs(R[i]["q"][:, 7:] - R[j]["q"][:, 7:])).mean()) for i, j in pairs]
    f = lambda k: np.array([r[k] for r in R])
    print(f"{name:10s} n={len(R):2d} | L_align {np.median(f('align')):.3f} [{f('align').min():.3f}-{f('align').max():.3f}] | bfm {f('bfm').mean():.3f} [{f('bfm').min():.3f}-{f('bfm').max():.3f}] | heading err median {np.median(f('head')):.0f} deg "
          f"[{f('head').min():.0f}-{f('head').max():.0f}] | end heading {f('end_h').min():+.0f}..{f('end_h').max():+.0f} | end pos median {np.median(f('end_pos')):.1f} m "
          f"[{f('end_pos').min():.1f}-{f('end_pos').max():.1f}] | latent pairwise {np.median(ang):.0f} deg [{min(ang):.0f}-{max(ang):.0f}] | joint diff {np.mean(jd):.1f} deg")
