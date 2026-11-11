#!/usr/bin/env python3
"""scan_adult_reward.py -- does the adult's own clip actually earn its humenv reward? For every clip of data/origin_action
(z0's rollout on the adult: the qpos, qvel and action of each control step, recorded after env.step), the task's humenv
reward at each step, so what env.step returned when the clip was made. A clip whose adult scores near zero cannot use
the reward as a metric: the motion to reproduce is not what the reward asks for.
Writes outputs/reward/adult_clip_scan.csv (task, stem, frames, return, mean_reward = return / frames, in_c540: the clip is
in the 540 of outputs/c540_bfmglobal_g128) and prints per-task numbers. CPU only.
usage: uv run python scripts/scan_adult_reward.py"""
import csv, glob, os, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.exact_obs import ADULT_XML                                   # noqa: E402
from child_reward_eval import reward_of                                 # noqa: E402


def main():
    m = mujoco.MjModel.from_xml_path(str(ADULT_XML)); d = mujoco.MjData(m)
    c540 = {p.name for p in (REPO / "outputs/c540_bfmglobal_g128").iterdir() if p.is_dir()}
    rows = []
    for f in sorted(glob.glob(str(REPO / "data/origin_action/*/*.npz"))):
        task, stem = os.path.basename(os.path.dirname(f)), Path(f).stem
        z = np.load(f); rw = reward_of(task); T = len(z["qpos"]); ret = 0.0
        for q, v, a in zip(z["qpos"], z["qvel"], z["action"]):
            d.qpos[:] = q; d.qvel[:] = v; d.ctrl[:] = a
            mujoco.mj_forward(m, d)
            ret += float(rw.compute(m, d))
        rows.append(dict(task=task, stem=stem, frames=T, ret=ret, mean_reward=ret / T, in_c540=stem in c540))
    out = REPO / "outputs/reward/adult_clip_scan.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {out.relative_to(REPO)}: {len(rows)} clips, {sum(r['in_c540'] for r in rows)} in the 540")
    for t in sorted({r["task"] for r in rows}):
        v = np.array([r["mean_reward"] for r in rows if r["task"] == t and r["in_c540"]])
        print(f"{t:30s} n {len(v):3d}  mean reward min {v.min():.2f} median {np.median(v):.2f} max {v.max():.2f}  >=0.5: {(v >= 0.5).sum():2d}  >=0.8: {(v >= 0.8).sum():2d}")


if __name__ == "__main__":
    main()
