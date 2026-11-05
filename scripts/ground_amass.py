#!/usr/bin/env python3
"""ground_amass.py -- remove the slow floating of the AMASS clips: stretches of several seconds where the whole body hovers
5-17 cm above the floor (mocap height drift). The existing grounding only lifts frames that sink into the floor
(qpos_retarget.ground_correct_qpos; the adult clips also got one constant floor_offset per clip), so a clip can touch the
floor in some frames and float in others.

Per clip, on the body it is played on:
  h(t)   exact lowest point of the body (model.exact_obs._LowestPoint, the same as everywhere else)
  g(t)   the floor drift: a morphological opening of h with a +-window (running min, then running max of that), lightly
         smoothed. The opening never rises above h and ignores anything off the floor for less than 2 * window
         (a jump's or a run's flight), so airborne phases keep their height above the floor
  qpos[:, 2] -= g, then frames still below the floor are lifted to the usual -0.5 cm (ground_correct_qpos)
Root x / y and every joint angle are untouched. Writes the same .npz keys (qpos replaced) plus ground_shift (the per-frame
change of root z, m).
usage: uv run python scripts/ground_amass.py --src amass_data/amass_origin_motion/ACCAD \
           --dst amass_data/amass_origin_motion_grounded/ACCAD --xml assets/robots/adult/robot.xml [--window 0.5]
"""
import argparse, glob, sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.exact_obs import _LowestPoint      # noqa: E402
from qpos_retarget import ground_correct_qpos  # noqa: E402


def lowest(m, d, low, Q):
    h = np.empty(len(Q))
    for t, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); h[t] = low(d)
    return h


def ground(m, Q, fps, window=0.5, smooth=0.1, tol=-0.005):
    """-> grounded qpos, lowest point before, lowest point after."""
    d, low = mujoco.MjData(m), _LowestPoint(m)
    h = lowest(m, d, low, Q)
    size = 2 * int(round(window * fps)) + 1
    g = maximum_filter1d(minimum_filter1d(h, size, mode="nearest"), size, mode="nearest")   # opening: <= h
    g = gaussian_filter1d(g, smooth * fps, mode="nearest") if smooth > 0 else g
    out = Q.copy(); out[:, 2] -= g
    out, _, _ = ground_correct_qpos(m, out, tol)
    return out, h, lowest(m, d, low, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--window", type=float, default=0.5, help="half-width (s): off the floor shorter than 2x this is kept")
    ap.add_argument("--smooth", type=float, default=0.1, help="Gaussian sigma (s) on the floor drift")
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(REPO / a.xml))
    dst = REPO / a.dst; dst.mkdir(parents=True, exist_ok=True)
    stats = []
    for f in sorted(glob.glob(str(REPO / a.src / "*.npz"))):
        z = dict(np.load(f)); fps = float(z.get("fps", 30.0))
        out, h0, h1 = ground(m, z["qpos"], fps, a.window, a.smooth)
        z["ground_shift"] = out[:, 2] - z["qpos"][:, 2]; z["qpos"] = out
        np.savez(dst / Path(f).name, **z)
        stats.append(((h0 > 0.03).mean(), (h1 > 0.03).mean(), h0.max(), h1.max(), h1.min(), np.abs(z["ground_shift"]).max()))
    s = np.array(stats)
    print(f"{len(s)} clips -> {a.dst} | frames > 3 cm above the floor: {s[:, 0].mean() * 100:.0f}% -> {s[:, 1].mean() * 100:.0f}% | "
          f"lowest point after: min {s[:, 4].min() * 100:.1f} cm | largest root shift {s[:, 5].max() * 100:.1f} cm")


if __name__ == "__main__":
    main()
