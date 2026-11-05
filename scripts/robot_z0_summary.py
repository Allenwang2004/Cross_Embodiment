#!/usr/bin/env python3
"""robot_z0_summary.py -- z0 on several robots side by side, per clip: kinematic floor, bfm under z0, fallen frames,
end-point distance (scripts/robot_z0.py; the G1's from g1_z0_rollout.py, outputs/g1/z0_perjoint).
Writes outputs/z0_compare_robots.json.
usage: uv run python scripts/robot_z0_summary.py [--robot g1 --robot h1 --robot apollo]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.robots import ROBOTS, out_dir   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", action="append", default=None, choices=sorted(ROBOTS))
    robots = ap.parse_args().robot or ["g1", "h1", "apollo"]
    rows = {}
    for r in robots:
        zdir = out_dir(r) / ("z0_perjoint" if r == "g1" else "z0")
        for f in sorted(zdir.glob("*/summary.json")):
            stem = f.parent.name; z = json.loads(f.read_text())
            rt = out_dir(r) / "roundtrip" / stem / "summary.json"
            if not rt.exists():
                continue
            rows.setdefault(stem, {})[r] = dict(floor=json.loads(rt.read_text())["bfm_floor"], bfm=z["bfm"],
                                                fallen=z["motion"]["fallen"], end=z["motion"]["end_xy"])
    stems = [s for s in rows if all(r in rows[s] for r in robots)]
    head = "".join(f" | {ROBOTS[r]['name']:^30s}" for r in robots)
    print(f"{'clip':30s}{head}\n{'':30s}" + " | floor  bfm z0  fallen  end m" * len(robots))
    for s in stems:
        print(f"{s:30s}" + "".join(f" | {rows[s][r]['floor']:5.3f}  {rows[s][r]['bfm']:5.3f}  {rows[s][r]['fallen']:6d}  "
                                   f"{rows[s][r]['end']:5.2f}" for r in robots))
    print(f"\n{len(stems)} clips on every robot. medians:")
    for r in robots:
        v = [rows[s][r] for s in stems]
        print(f"  {ROBOTS[r]['name']:18s} floor {np.median([x['floor'] for x in v]):.3f}, bfm z0 {np.median([x['bfm'] for x in v]):.3f}; "
              f"clips with > 10 fallen frames {sum(x['fallen'] > 10 for x in v)} / {len(v)}")
    (REPO / "outputs/z0_compare_robots.json").write_text(json.dumps({s: rows[s] for s in stems}, indent=1))


if __name__ == "__main__":
    main()
