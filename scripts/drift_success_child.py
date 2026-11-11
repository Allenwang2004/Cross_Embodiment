#!/usr/bin/env python3
"""drift_success_child.py -- local joint error vs global drift on the child (user, 11/11, for the paper's representation analysis: BFMTrack
ignores global position and heading, drifts off the path, yet has the lowest local joint error; our heading and position terms prevent
the drift and raise the success rate).

52 task motions x 5 seeds, the best latent / sequence after 1,024 rollouts: LOES (outputs/child_52_rerun, re-rolled latents), LSO =
BFMTrack, LSOR (lsor_w005: LSO + ranks + 0.05 x our heading and position terms; "LSOR (our method)"), Trajectory Adaptation
(outputs/child_52_lso, rollouts.npz q_best_it64), native tracking (q_ER8) and z0 (one rollout per motion).
Per rollout, q[t] against ref[t + 1] (as the Task page):
  MPJPE    eval_metrics.tracking_metrics (root-relative, each body in its own heading frame: blind to drift and heading)
  drift    mean pelvis x, y distance to the reference (m, adult-equivalent: the child's / 0.611)
  heading  mean |facing - the reference's facing| (deg; facing = root yaw - 90 deg)
  success  PHC's test: the world-frame mean body position error never above 0.5 m adult-equivalent (0.31 m on the child)
Out: outputs/drift_success_child.json and a printed table.
usage: uv run python scripts/drift_success_child.py"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.bodies import BODIES
from crossenbodiment.geometry import facing as _facing
from crossenbodiment.metrics import MetricSuite, metric_body
RR, LS = REPO / "outputs/child_52_rerun", REPO / "outputs/child_52_lso"; S = BODIES.scale("child"); SEEDS = range(5)
# --loes-dir outputs/noanchor/child (user 11/11): LOES from the runs without the anchor term -> outputs/drift_success_child_noanchor.json
RL = REPO / sys.argv[sys.argv.index("--loes-dir") + 1] if "--loes-dir" in sys.argv else None; SUFFIX = "_noanchor" if "--loes-dir" in sys.argv else ""   # None: RR, read at call time
SEQ = dict(lso="q_best_it64", lsor_w005="q_best_it64", ta="q_best_it64")
_b = {}


def facing(q):
    """root yaw - 90 deg (crossenbodiment.geometry.facing)"""
    return _facing(q)


def one(c):
    import eval_metrics as E
    if not _b:
        _b["body"] = metric_body(E.XML["child"], True)
    body = _b["body"]; st = c.split("/")[1]
    ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    # {mpjpe, drift (adult-equivalent m), heading (deg), success (PHC, 0.5 m x S)} per rollout (crossenbodiment.metrics); the world
    # positions are the Body's own model's (formerly a second MjModel of the same XML: the same numbers)
    stats = MetricSuite(body, ("mpjpe", "drift", "heading", "success"), S, drift_unit="adult").reference(ref)
    out = {k: [] for k in ("z0", "native", "es", *SEQ)}
    out["z0"].append(stats(np.load(RR / f"{st}_es_s0/origin_z.npz")["qpos"]))
    out["native"].append(stats(np.load(LS / f"{st}_lso_s0/rollouts.npz")["q_ER8"]))
    L_ = RL or RR; ix = json.load(open(L_ / "rollouts" / st / "index.json"))["map"]; Q = np.load(L_ / "rollouts" / st / "eval_qpos.npz")["qpos"]
    for s in SEEDS:
        out["es"].append(stats(Q[ix[f"{st}_es_s{s}"]["best"][-1]]))
        for k, key in SEQ.items():
            out[k].append(stats(np.load(LS / f"{st}_{k}_s{s}/rollouts.npz")[key]))
    return st, out


def main():
    clips = [l.strip() for l in open(REPO / "outputs/child_52/clips.txt")]
    with Pool(12) as p:
        res = p.map(one, clips, chunksize=1)
    names = dict(z0="Frozen z0", native="Native tracking", lso="LSO (BFMTrack)", lsor_w005="LSOR (ours)", ta="Trajectory Adaptation", es="LOES (ours)")
    rep = {}
    print(f"{'method':24s} {'MPJPE cm':>9s} {'drift m':>8s} {'heading deg':>12s} {'success':>8s}")
    for k in ("z0", "native", "lso", "lsor_w005", "ta", "es"):
        v = {m: float(np.mean([np.mean([x[m] for x in o[k]]) for _, o in res])) for m in ("mpjpe", "drift", "heading", "success")}; rep[k] = v
        print(f"{names[k]:24s} {v['mpjpe']:9.2f} {v['drift']:8.2f} {v['heading']:12.1f} {100 * v['success']:7.1f}%")
    per = {st: {k: {m: float(np.mean([x[m] for x in o[k]])) for m in ("mpjpe", "drift", "heading", "success")} for k in o} for st, o in res}
    lso_lower = sum(per[st]["lso"]["mpjpe"] < per[st]["es"]["mpjpe"] for st in per); es_less_drift = sum(per[st]["es"]["drift"] < per[st]["lso"]["drift"] for st in per)
    print(f"motions where LSO's MPJPE < LOES's: {lso_lower}/52; LOES drifts less than LSO: {es_less_drift}/52")
    (REPO / f"outputs/drift_success_child{SUFFIX}.json").write_text(json.dumps(dict(mean=rep, per_motion=per, lso_lower_mpjpe=lso_lower, loes_less_drift=es_less_drift), indent=1))


if __name__ == "__main__":
    main()
