#!/usr/bin/env python3
"""fall_definitions.py -- Fall Time under several definitions, for every method on the 52 child task motions (best and centre after
1,024 rollouts), to choose a definition on validity grounds and to show how much the comparison depends on it.
  height     seconds with the root below half of the reference's root height (the Task page's definition until 11/08)
  hand_mot   seconds with a hand (wrist or hand) within 1 cm of the floor; NaN for a motion whose reference has its hands within
             5 cm of the floor for 0.5 s or more (the user's rule of 11/07, on the kpkd page)
  hand_frm   seconds with a hand within 1 cm of the floor while the reference's hands are more than 5 cm above it
  contact    seconds with any non-foot body (not ankle or toe) within 1 cm of the floor while the same body of the reference is more
             than 5 cm above it (chosen before comparing methods: reference-relative, frame by frame, all motions kept; non-foot
             contact as the fall signal follows DeepMimic's early termination on torso or head contact)
  head       seconds with the head below half of the reference's head height
Each rollout's state after step t is compared with reference frame t + 1. Writes outputs/fall_definitions/task_motions.json.
usage: uv run python scripts/fall_definitions.py"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.metrics import contact_fall_time, height_fall, metric_body   # noqa: E402
from crossenbodiment import paths   # noqa: E402
from crossenbodiment.experiment import output_arg, parse_flag, workers   # noqa: E402
from crossenbodiment.methods import RESULT_KEYS, experiment_keys, key_in   # noqa: E402
from crossenbodiment.reroll import Reroller   # noqa: E402
from crossenbodiment.results import write_json   # noqa: E402
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), the result written under paths.results_root()
# ($CE_RESULTS_ROOT); both REPO/outputs by default. --loes-dir is read in main (not at import).
OUT = paths.result("fall_definitions"); RERUN = paths.output("child_52_rerun"); CEMP = paths.output("child_52_cem_phc"); LSO = paths.output("child_52_lso")
# --loes-dir outputs/noanchor/child (user 11/11): LOES ("es") from LOES without the anchor term, everything else unchanged;
# the result goes to task_motions_noanchor.json next to the anchored one
LOESD = None       # None: RERUN, read at call time
# run-directory keys (crossenbodiment/methods.py RUNS) and this file's method keys (RESULT_KEYS "outputs/fall_definitions"): in child_52_rerun
# "es" = LOES with the anchor, "cem" = CEM with our cost (cem_ours); in child_52_cem_phc "cem" = BFM-Zero's CEM, written as "cem_phc";
# in child_52_lso "lsor" = LSOR with BFMTrack's cost (lsor_bfm)
RR, LK = experiment_keys("outputs/child_52_rerun"), experiment_keys("outputs/child_52_lso")
LOES, CEMP_KEY = RR["loes_anchor"], key_in("outputs/child_52_cem_phc", "cem_phc")
CEMP_RESULT = next(k for k, m in RESULT_KEYS["outputs/fall_definitions"].items() if m == "cem_phc")      # "cem_phc"
RERUN_ALGOS = (RR["random"], RR["cem_ours"], RR["cmaes"])
SEQ_KEYS = (LK["lso"], LK["lsor_bfm"], LK["ta"], LK["lsor_ours"], LK["lsor_w005"])
SUFFIX = ""
DEFS = ("height", "hand_mot", "hand_frm", "contact", "head"); TOUCH, NEAR = 0.01, 0.05
body = None


def clips():
    from crossenbodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    return read_clips(CHILD_52_TASK)


def falls(q, ref, R):
    """q (T, nq) rollout, ref (T', nq) = reference frames 1.., R the reference's precomputed (names, part distances, head z)"""
    names, Dr, hr = R; T = min(len(q), len(ref)); q = q[:T].astype(np.float64); Dr = Dr[:T]; hr = hr[:T]
    _, D = body.part_floor(q); hand = np.array([("wrist" in n.lower() or "hand" in n.lower()) for n in names])
    d = mujoco.MjData(body.m); hid = body.m.body("Head").id; hz = np.empty(T)
    for t in range(T):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(body.m, d); hz[t] = d.xpos[hid][2]
    touch = D <= TOUCH; ht = touch[:, hand].any(1)
    hand_motion = (Dr[:, hand].min(1) <= NEAR).sum() / 30 >= 0.5
    # height (legacy) and contact (the Task page's Fall Time): crossenbodiment.metrics
    return dict(height=height_fall(q, ref[:T]), hand_mot=float("nan") if hand_motion else float(ht.sum() / 30),
                hand_frm=float((ht & (Dr[:, hand].min(1) > NEAR)).sum() / 30), contact=contact_fall_time(D, Dr, TOUCH, NEAR),
                head=float((hz < 0.5 * hr).sum() / 30))


def clip(c):
    global body
    import eval_metrics as E
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]
    ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    names, Dr = body.part_floor(ref); d = mujoco.MjData(body.m); hid = body.m.body("Head").id; hr = np.empty(len(ref))
    for t in range(len(ref)):
        d.qpos[:] = ref[t]; mujoco.mj_kinematics(body.m, d); hr[t] = d.xpos[hid][2]
    R = (names, Dr, hr); res = {"frozen": falls(np.load(RERUN / f"{stem}_{LOES}_s0/origin_z.npz")["qpos"], ref, R)}
    self_check = falls(ref, ref, R)
    for src, algos in ((LOESD or RERUN, (LOES,)), (RERUN, RERUN_ALGOS), (CEMP, (CEMP_KEY,))):            # LOES from --loes-dir (default RERUN)
        ix = Reroller.read(src / f"rollouts/{stem}"); Q = np.load(src / f"rollouts/{stem}/eval_qpos.npz")["qpos"]
        for a in algos:
            key = CEMP_RESULT if src == CEMP else a
            res[key + "_best"] = [falls(Q[ix.best_last(f"{stem}_{a}_s{s}")], ref, R) for s in range(5)]
            res[key + "_centre"] = [falls(Q[ix.centre_last(f"{stem}_{a}_s{s}")], ref, R) for s in range(5)]
    res["native"] = falls(np.load(LSO / f"{stem}_{LK['lso']}_s0/rollouts.npz")["q_ER8"], ref, R)
    for k in SEQ_KEYS:
        if all((LSO / f"{stem}_{k}_s{s}/rollouts.npz").exists() for s in range(5)):
            Rs = [np.load(LSO / f"{stem}_{k}_s{s}/rollouts.npz") for s in range(5)]
            res[k + "_best"] = [falls(x["q_best_it64"], ref, R) for x in Rs]; res[k + "_centre"] = [falls(x["q_mean_it64"], ref, R) for x in Rs]
    return stem, res, self_check


def main(argv=()):
    global LOESD, SUFFIX
    if "--loes-dir" in argv:                                           # LOES ("es") from --loes-dir -> task_motions_noanchor.json
        LOESD, SUFFIX = output_arg(parse_flag(argv, "--loes-dir")), "_noanchor"
    cl = clips()
    with Pool(workers(24)) as p:
        out = p.map(clip, cl)
    print("reference against itself, largest value per definition:", {k: max(o[2][k] for o in out if not np.isnan(o[2][k])) for k in DEFS})
    per = {o[0]: o[1] for o in out}; methods = sorted({m for o in out for m in o[1]}, key=lambda m: list(out[0][1]).index(m) if m in out[0][1] else 99)
    summ = {}
    for m in methods:
        if not all(m in per[s] for s in per):
            continue
        rows = [per[s][m] if isinstance(per[s][m], list) else [per[s][m]] for s in per]
        summ[m] = {k: float(np.nanmean([np.nanmean([r[k] for r in rr]) if not all(np.isnan(r[k]) for r in rr) else np.nan for rr in rows])) for k in DEFS}
        summ[m]["motions_with_fall"] = {k: int(sum(np.nanmean([r[k] for r in rr]) > 0 for rr in rows if not all(np.isnan(r[k]) for r in rr))) for k in DEFS}
    write_json(OUT / f"task_motions{SUFFIX}.json", dict(per_clip=per, mean=summ))
    print(f"{'method':16s}" + "".join(f"{k:>10s}" for k in DEFS))
    for m, v in summ.items():
        print(f"{m:16s}" + "".join(f"{v[k]:10.3f}" for k in DEFS))


if __name__ == "__main__":
    main(sys.argv)
