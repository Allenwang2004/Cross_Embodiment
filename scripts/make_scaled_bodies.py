#!/usr/bin/env python3
"""make_scaled_bodies.py -- the 8 uniformly scaled bodies (user, 11/11: replace the earlier shape designs, child kept): the whole adult
scaled isometrically by L and then made wider or narrower by W: every segment length (leg, arm, torso, head) x L and every girth x L*W
(scale_robot.py's girth multiplies geom sizes absolutely, so girth = W alone would leave a short body with adult-sized head, feet and
limbs; user chose L*W on 11/11), L in {1.4, 1.2, 0.8, 0.6}, W in {1.2, 0.8} -> assets/robots/len<L>_wid<W>.

Per body, the steps of docs/new_body.md (all existing scripts):
  1 scale_robot.py --no-actuator-scale (shape only; joints and actuators stay the adult's)  -> assets/robots/<body>/{robot.xml, parameter.json}
  2 export_skeleton_json.py                                                                 -> assets/robots/<body>/skeleton.json
  3 qpos_retarget.py, one call per motion folder (nested output)                             -> data/<body>/retargeting_motion (540 clips)
  4 torque_ratio_across_motions.py (gravity torque ratio per joint and motion)                -> outputs/torque_ratio_across_motions/gravity/<body>
  6 torque_aggregate_motion_k.py --joint-dynamics (actuators, armature, damping, stiffness)  -> assets/robots_torque/<body>/robot_torque_full.xml
  5 the check: measured k vs the geometric prediction (the k of a body is a property of the body, not of the motion)
Not done: step 3 of the runbook (per-frame z, data/<body>/infer_retargeting_z; not used by the current searches) and write_body_splits.py (bilevel only).
usage: uv run python scripts/make_scaled_bodies.py [all | <body> ...]   (log per body: outputs/scaled_bodies/<body>.log)"""
import csv, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; LOG = REPO / "outputs/scaled_bodies"
BODIES = {f"len{L}_wid{W}": (L, W) for L in (1.4, 1.2, 0.8, 0.6) for W in (1.2, 0.8)}
PY = ["uv", "run", "python"]


class Failed(Exception):
    pass


def run(cmd, log, tolerate=None):
    """tolerate: a substring of the only '!!' line allowed (the step's file is written before its self-check)"""
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n"); f.flush()
        r = subprocess.run(list(map(str, cmd)), cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        bad = [l for l in log.read_text().splitlines() if l.strip().startswith("!!")]
        if tolerate and bad and all(tolerate in l for l in bad):
            return "warning: " + bad[-1].strip()
        raise Failed(f"{log.stem}: failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return ""


def make(body):
    L, W = BODIES[body]; log = LOG / f"{body}.log"; log.write_text("")
    run(PY + ["scripts/scale_robot.py", "--label", body, "--no-actuator-scale"] + sum([[f"--{g}-scale", L] for g in ("leg", "arm", "torso", "head")], [])
        + sum([[f"--{g}-girth", round(L * W, 6)] for g in ("leg", "arm", "torso", "head")], []), log)
    A = REPO / "assets/robots" / body
    run(PY + ["scripts/export_skeleton_json.py", "--input", A / "robot.xml", "--output", A / "skeleton.json"], log)
    for m in sorted(p.name for p in (REPO / "data/origin_motion").iterdir() if p.is_dir()):
        run(PY + ["scripts/qpos_retarget.py", "--input_dir", f"data/origin_motion/{m}", "--output_dir", f"data/{body}/retargeting_motion/{m}",
                  "--source_skeleton_json", "assets/robots/adult/skeleton.json", "--target_skeleton_json", A / "skeleton.json", "--target_xml", A / "robot.xml"], log)
    n_dir = sum(1 for p in (REPO / f"data/{body}/retargeting_motion").iterdir() if p.is_dir()); n_npz = len(list((REPO / f"data/{body}/retargeting_motion").rglob("*.npz")))
    K = REPO / "outputs/torque_ratio_across_motions/gravity" / body
    run(PY + ["scripts/torque_ratio_across_motions.py", "--origin", "data/origin_motion", "--retarget", f"data/{body}/retargeting_motion",
              "--adult-xml", "assets/robots/adult/robot.xml", "--child-xml", A / "robot.xml", "--outdir", K], log)
    out = REPO / "assets/robots_torque" / body / "robot_torque_full.xml"; out.parent.mkdir(parents=True, exist_ok=True)
    warn = run(PY + ["scripts/torque_aggregate_motion_k.py", "--matrix", K, "--src", A / "robot.xml", "--out", out, "--joint-dynamics"], log, tolerate="zeta off by")
    assert out.exists(), out
    r = list(csv.DictReader(open(K / "_ratios_aggregate.csv"))); km = np.array([float(x["k_aggregate"]) for x in r]); kp = np.array([float(x["k_predicted_subtree"]) for x in r])
    nm = np.array([int(x["n_motions"]) for x in r]); real = (nm > 0) & (~np.isclose(km, kp, rtol=1e-9)); q = km[real] / kp[real] if real.any() else np.array([1.0])
    msg = (f"{body}: L {L} W {W} | retargeted {n_dir} motion folders, {n_npz} clips | k median {np.median(km):.4f} | "
           f"independent fits {int(real.sum())}/{len(r)}, measured/predicted median {np.median(q):.4f} range {q.min():.3f}-{q.max():.3f}" + (f" | {warn}" if warn else ""))
    with open(log, "a") as f:
        f.write(msg + "\n")
    return msg


if __name__ == "__main__":
    LOG.mkdir(parents=True, exist_ok=True); names = list(BODIES) if sys.argv[1:] in ([], ["all"]) else sys.argv[1:]
    def safe(b):
        try:
            return make(b)
        except Failed as e:
            return f"FAILED {e}"
    with Pool(len(names)) as p:
        for m in p.imap(safe, names):
            print(m, flush=True)
