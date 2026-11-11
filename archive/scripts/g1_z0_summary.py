#!/usr/bin/env python3
"""g1_z0_summary.py -- z0 on the Unitree G1 across clips (no search): one table, and optionally one video per clip.

Per clip, from outputs/g1/roundtrip/<stem>/summary.json (scripts/g1_roundtrip.py) and the z0 rollouts of
scripts/g1_z0_rollout.py (--gains perjoint: outputs/g1/z0_perjoint/<stem>/, --gains mass: outputs/g1/z0/<stem>/):
  floor     bfm of the clip after SMPL -> G1 -> SMPL, kinematic: the best any G1 controller could score through the map
  ik        mean IK residual over the paired G1 links (deg)
  bfm z0    Motivo with the clip's own z0 driving the G1 (and, for --gains perjoint, the same with the mass gains)
  fallen, travel (G1 / clip mapped onto the G1), end-point distance, mean pelvis height (G1 / mapped clip)
Writes <z0 dir>/summary.json and, with --render, <z0 dir>/videos/<stem>.mp4 per clip, 1080 px panels: adult original,
G1 mapped kinematically, G1 under z0 with the mass gains (perjoint only) and with --gains.
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/g1_z0_summary.py [--gains perjoint] [--render]
"""
import argparse, json, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
G1 = "assets/robots/g1/scene.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="perjoint", choices=["perjoint", "mass"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--jobs", type=int, default=6, help="videos rendered at once")
    a = ap.parse_args()
    zdir = REPO / ("outputs/g1/z0" if a.gains == "mass" else f"outputs/g1/z0_{a.gains}")
    base = REPO / "outputs/g1/z0"
    rows = {}
    for f in sorted(zdir.glob("*/summary.json")):
        stem = f.parent.name
        rt = REPO / f"outputs/g1/roundtrip/{stem}/summary.json"
        if not rt.exists():
            continue
        z, r = json.loads(f.read_text()), json.loads(rt.read_text())
        rows[stem] = dict(clip=z["clip"], floor=r["bfm_floor"], ik=float(np.mean([v[0] for v in r["ik_resid_deg"].values()])),
                          at_limit=r["at_limit"], bfm=z["bfm"], **z["motion"])
        old = base / stem / "summary.json"
        if a.gains != "mass" and old.exists():
            o = json.loads(old.read_text())
            rows[stem].update(bfm_mass=o["bfm"], fallen_mass=o["motion"]["fallen"], end_xy_mass=o["motion"]["end_xy"])
    cmp = a.gains != "mass" and all("bfm_mass" in x for x in rows.values())
    print(f"gains {a.gains}" + (" (bfm / fallen / end-point with the mass gains in brackets)" if cmp else ""))
    print(f"{'clip':30s} {'floor':>6s} {'ik deg':>6s} {'bfm z0':>15s} {'fallen':>10s} {'travel m (mapped)':>18s} "
          f"{'end-pt m':>13s} {'pelvis m (mapped)':>18s}")
    for s, x in rows.items():
        b = f"{x['bfm']:.3f} ({x['bfm_mass']:.3f})" if cmp else f"{x['bfm']:.3f}"
        fl = f"{x['fallen']} ({x['fallen_mass']})" if cmp else f"{x['fallen']}"
        e = f"{x['end_xy']:.2f} ({x['end_xy_mass']:.2f})" if cmp else f"{x['end_xy']:.2f}"
        print(f"{s:30s} {x['floor']:6.3f} {x['ik']:6.1f} {b:>15s} {fl:>10s} {x['travel']:8.1f} ({x['ref_travel']:5.1f})  "
              f"{e:>13s} {x['pelvis']:8.3f} ({x['ref_pelvis']:.3f})")
    v = list(rows.values())
    print(f"\nmedian: floor {np.median([x['floor'] for x in v]):.3f}, bfm z0 {np.median([x['bfm'] for x in v]):.3f}"
          + (f" (mass gains {np.median([x['bfm_mass'] for x in v]):.3f}; lower on "
             f"{sum(x['bfm'] < x['bfm_mass'] for x in v)} / {len(v)} clips)" if cmp else "")
          + f"; clips with > 10 fallen frames: {sum(x['fallen'] > 10 for x in v)} / {len(v)}"
          + (f" (mass gains {sum(x['fallen_mass'] > 10 for x in v)})" if cmp else ""))
    (zdir / "summary.json").write_text(json.dumps(rows, indent=1))

    if a.render:
        vdir = zdir / "videos"; vdir.mkdir(exist_ok=True)
        rel = lambda p: str(p.relative_to(REPO))
        specs = []
        for s, x in rows.items():
            panels = [{"title": "adult (SMPL), original", "sub": x["clip"], "xml": "assets/robots/adult/robot.xml",
                       "qpos": f"data/origin_motion/{x['clip']}.npz"},
                      {"title": "G1, clip mapped kinematically", "sub": f"the target, no physics; floor {x['floor']:.3f}",
                       "xml": G1, "qpos": f"outputs/g1/roundtrip/{s}/g1.npz"}]
            if cmp:
                panels.append({"title": "G1 under Motivo z0, old gains (mass)",
                               "sub": f"bfm {x['bfm_mass']:.3f}, fallen {x['fallen_mass']}, end-point off {x['end_xy_mass']:.1f} m",
                               "xml": G1, "qpos": rel(base / s / "rollout.npz")})
            panels.append({"title": f"G1 under Motivo z0, {'per-joint' if a.gains == 'perjoint' else a.gains} gains",
                           "sub": f"bfm {x['bfm']:.3f}, fallen {x['fallen']}, end-point off {x['end_xy']:.1f} m",
                           "xml": G1, "qpos": rel(zdir / s / "rollout.npz")})
            spec = {"out": rel(vdir / f"{s}.mp4"), "cols": 2 if len(panels) == 4 else 3, "size": 1080,
                    "track": {"distance": 3.6, "elevation": -12, "azimuth": 140, "lookat_z": 0.7}, "panels": panels}
            (vdir / f"{s}.json").write_text(json.dumps(spec, indent=1)); specs.append(vdir / f"{s}.json")
        run = lambda f: subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(f)], cwd=REPO, check=True,
                                       capture_output=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(run, specs))
        print(f"-> {rel(vdir)}/<clip>.mp4 ({len(specs)} videos)")


if __name__ == "__main__":
    main()
