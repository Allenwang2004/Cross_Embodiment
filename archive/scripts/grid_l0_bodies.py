"""Fill assets/robots/ (the 8-axis, scale_robot.py bodies) to a stature x build grid.

The 30-axis set in assets/robots_advance/ is laid out as s<i>_b<j>: stature
strata down, build across. This does the same for the ORIGINAL 8-axis schema
(leg / arm / torso / head, each with a length scale and a girth), so the two
sets can be compared cell for cell, and so the Morphology Bench can show them
the same way (`morphology_bench.py --schema l0`).

Stature is walked along the same human growth curve scale_robot_l1 uses,
collapsed from 15 segments to the 4 groups this schema has:

    leg_scale   = s ** (1 + LEG_E)          legs are RELATIVELY longer on a tall body
    arm_scale   = s ** (1 + ARM_E)
    torso_scale = s ** (1 + TRUNK_E)        trunk relatively longer on a short one
    head_scale  = s * (1 + HEAD_K * (1-s))  head/stature falls from 1/5 to 1/8 with age
    <group>_girth = <group>_scale * s ** GIRTH_E * build

so a 0.50x body is a four-year-old's proportions, not a shrunken adult. The
length scales are then rescaled uniformly until the built body measures the
target stature (the 8-axis scaler moves every child body by its OWN group's
scale, so stature is not a closed-form function of the axes).

Build multiplies the four girths, with per-group exponents: arms build**1,
torso build**0.9, legs build**0.6, head build**0.4 (a heavy person's head is
not 25% wider). The leg and torso exponents are where they are because this
schema has no hip-width axis (hip spacing follows leg_scale) and its trunk
capsules are near-spheres: at build 1.25 with full-exponent girths the thighs
and the clavicle/spine intersect in the default pose. With these exponents
every cell is free of self-contact at rest.

The grid is 4 strata x 5 builds = 20 cells. The one cell that reproduces the
adult itself (1.00x stature, build 1.00) is skipped -- adult/robot.xml is the
source and already exists -- which leaves 19 new bodies, taking assets/robots/
from 11 to 30 (adult not counted, it is the source).

Existing bodies (child, tall_slim, ...) are never touched. `split` is written
afterwards by scripts/write_body_splits.py (new bodies come out "unused" until
BilevelConfig names them).

    uv run scripts/grid_l0_bodies.py --dry-run
    uv run scripts/grid_l0_bodies.py
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_robot import AXES as L0_AXES, ROBOTS_DIR, SOURCE_XML, apply_scale, measure_min_z
from scale_robot_l1 import ARM_E, GIRTH_E, HEAD_K, LEG_E, TRUNK_E, describe

STRATA = [0.50, 0.62, 0.78, 1.00]                 # stature ratio vs adult
BUILDS = [0.75, 0.88, 1.00, 1.12, 1.25]           # girth multiplier
SKIP = {(4, 3)}                                   # (stratum, build) == the adult itself
LEN_AXES = [a for a in L0_AXES if a.endswith("_scale")]


def growth_axes_l0(s, build=1.0):
    """The 8 axes of an average person of stature ratio s, times a build."""
    g = s ** GIRTH_E
    p = {"leg_scale": s ** (1 + LEG_E), "arm_scale": s ** (1 + ARM_E),
         "torso_scale": s ** (1 + TRUNK_E), "head_scale": s * (1 + HEAD_K * (1 - s))}
    p["leg_girth"] = p["leg_scale"] * g * build ** 0.6
    p["arm_girth"] = p["arm_scale"] * g * build
    p["torso_girth"] = p["torso_scale"] * g * build ** 0.9
    p["head_girth"] = p["head_scale"] * g * build ** 0.4
    return p


def build_body(params):
    """Scale + ground-correct in memory, actuators untouched -- exactly what
    scale_robot.generate(..., scale_actuators=False) writes, without the write."""
    tree = ET.parse(SOURCE_XML)
    apply_scale(tree, params, scale_actuators=False)
    dz = -measure_min_z(ET.tostring(tree.getroot(), encoding="unicode"))
    pelvis = next(b for b in tree.getroot().iter("body") if b.get("name") == "Pelvis")
    x, y, z = (float(v) for v in pelvis.get("pos").split())
    pelvis.set("pos", f"{x:.6g} {y:.6g} {z + dz:.6g}")
    return tree, ET.tostring(tree.getroot(), encoding="unicode")


def fit_stature(p, target_h, tol=0.003, iters=6):
    p = dict(p)
    for _ in range(iters):
        _, xml = build_body(p)
        h = describe(xml)["stature"]
        if abs(h / target_h - 1) < tol:
            break
        for a in LEN_AXES:
            p[a] *= (target_h / h) ** 0.95
    return p


def write_body(label, params, extra=None, out_root=ROBOTS_DIR):
    """Same files scale_robot.generate writes, plus skeleton.json, keeping any
    non-axis keys (split, notes) the caller passes."""
    tree, xml = build_body(params)
    d = out_root / label
    d.mkdir(parents=True, exist_ok=True)
    tree.write(d / "robot.xml")
    (d / "parameter.json").write_text(json.dumps(
        {"label": label, "scale_actuators": False, **(extra or {}),
         **{a: round(float(params[a]), 4) for a in L0_AXES}}, indent=2) + "\n")
    return describe(xml)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite s<i>_b<j> bodies that already exist")
    args = ap.parse_args()

    adult_h = describe(SOURCE_XML.read_text())["stature"]
    print(f"{'body':<8}{'s':>6}{'build':>7}{'stature':>9}{'mass':>8}{'BMI':>7}{'selfc':>7}")
    for i, s in enumerate(STRATA, 1):
        for j, b in enumerate(BUILDS, 1):
            if (i, j) in SKIP:
                continue
            label = f"s{i}_b{j}"
            p = fit_stature(growth_axes_l0(s, b), s * adult_h)
            _, xml = build_body(p)
            info = describe(xml)
            bmi = info["mass"] / info["stature"] ** 2
            print(f"{label:<8}{s:6.2f}{b:7.2f}{info['stature']:9.3f}{info['mass']:8.1f}{bmi:7.1f}"
                  f"{info['self_contacts']:7d}")
            if args.dry_run:
                continue
            if (ROBOTS_DIR / label).exists() and not args.force:
                print(f"  exists, skipped (--force to overwrite): {ROBOTS_DIR / label}")
                continue
            write_body(label, p, extra={
                "notes": f"stature x build grid: {s:.2f}x adult stature on the growth curve, "
                         f"build {b:.2f}; {info['stature']:.2f} m, BMI {bmi:.1f}",
                "stature_ratio": round(info["stature"] / adult_h, 4)})
    if args.dry_run:
        print("(dry run -- nothing written)")
    else:
        print("now run: uv run scripts/write_body_splits.py --robots assets/robots")


if __name__ == "__main__":
    main()
