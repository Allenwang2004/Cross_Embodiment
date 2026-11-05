#!/usr/bin/env python3
"""heavy_arm_body.py -- the original adult with one arm --scale x heavier (same shape, same motors).

  body      assets/robots/<label>/robot.xml: assets/robots/adult/robot.xml with the density of the arm's four geoms
            (<side>_Shoulder upper arm, <side>_Elbow forearm, <side>_Wrist, <side>_Hand; or only --parts) x --scale, so
            their mass and inertia scale by it; parameter.json / skeleton.json copied from the adult (same geometry)
  retarget  data/<label>/retargeting_motion/<clip>.npz: the adult's own motion (data/origin_motion), unchanged --
            the heavy-armed body is asked to do exactly what the adult did
usage: uv run python scripts/heavy_arm_body.py --scale 5 [--side R] [--clip move-ego-0-2/move-ego-0-2_4]
"""
import argparse, shutil
from pathlib import Path
import mujoco

REPO = Path(__file__).resolve().parent.parent
PARTS = ("Shoulder", "Elbow", "Wrist", "Hand")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--side", default="R", choices=["L", "R"])
    ap.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    ap.add_argument("--parts", default=",".join(PARTS),
                    help="which of the arm's geoms get heavier, e.g. Wrist,Hand for a weight carried in the hand")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    parts = a.parts.split(",")
    label = a.label or (f"adult_heavy{a.side}arm_x{a.scale:g}" if parts == list(PARTS) else
                        f"adult_heavy{a.side}{''.join(parts)}_x{a.scale:g}")
    src = REPO / "assets/robots/adult"
    spec = mujoco.MjSpec.from_file(str(src / "robot.xml"))
    m0 = spec.compile()
    names = [f"{a.side}_{p}" for p in PARTS]
    G = {g.name: g for g in spec.geoms}
    for n in [f"{a.side}_{p}" for p in parts]:
        G[n].density *= a.scale
    m = spec.compile()
    dst = REPO / "assets/robots" / label; dst.mkdir(parents=True, exist_ok=True)
    (dst / "robot.xml").write_text(spec.to_xml())
    for f in ("parameter.json", "skeleton.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    ref = REPO / f"data/{label}/retargeting_motion/{a.clip}.npz"; ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / f"data/origin_motion/{a.clip}.npz", ref)
    arm0 = sum(m0.body(n).mass[0] for n in names); arm = sum(m.body(n).mass[0] for n in names)
    print(f"{label}: {a.side} arm {arm0:.2f} -> {arm:.2f} kg (body {m0.body_mass.sum():.1f} -> {m.body_mass.sum():.1f} kg) "
          f"-> {dst / 'robot.xml'}; reference = the adult's {a.clip}")


if __name__ == "__main__":
    main()
