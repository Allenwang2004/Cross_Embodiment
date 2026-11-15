#!/usr/bin/env python3
"""success_definitions.py -- Success Rate under several definitions, for every method on the 52 child task motions (best and
centre after 1,024 rollouts). A clip succeeds if the failure test never triggers; each rollout's state after step t is compared
with reference frame t + 1.
  root       the Task page's definition until 11/08: root more than 1.5 m (horizontal) or 45 deg (rotation) from the reference
  global     chosen before comparing methods: the mean over bodies of the world-frame body position error exceeds 0.5 m
             (adult-equivalent metres, i.e. 0.5 x the body's root scale s on the child); the success test of PHC (Luo et al. 2023), as in
             UHC: joints on average more than 0.5 m from the reference at any point; it counts drift and a collapsed pose alike
  global_raw the same with 0.5 m of the body's own metres
  local      the same 0.5 m (adult-equivalent) on the root-relative, heading-aligned positions of MPJPE: pose only, no drift
Writes outputs/success_definitions/task_motions.json. usage: uv run python scripts/success_definitions.py"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment.metrics import mean_body_error, metric_body, phc_success, root_success, world_positions   # noqa: E402
from crossembodiment import paths   # noqa: E402
from crossembodiment.experiment import output_arg, parse_flag, workers   # noqa: E402
from crossembodiment.methods import RESULT_KEYS, experiment_keys, key_in   # noqa: E402
from crossembodiment.reroll import Reroller   # noqa: E402
from crossembodiment.results import write_json   # noqa: E402
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), the result written under paths.results_root()
# ($CE_RESULTS_ROOT); both REPO/outputs by default. --loes-dir is read in main (not at import).
OUT = paths.result("success_definitions"); RERUN = paths.output("child_52_rerun"); CEMP = paths.output("child_52_cem_phc"); LSO = paths.output("child_52_lso")
# --loes-dir outputs/noanchor/child (user 11/11): LOES ("es") from LOES without the anchor term, everything else unchanged;
# the result goes to task_motions_noanchor.json next to the anchored one
LOESD = None       # None: RERUN, read at call time
# run-directory keys (crossembodiment/methods.py RUNS) and this file's method keys (RESULT_KEYS "outputs/success_definitions"): in child_52_rerun
# "es" = LOES with the anchor, "cem" = CEM with our cost (cem_ours); in child_52_cem_phc "cem" = BFM-Zero's CEM, written as "cem_phc";
# in child_52_lso "lsor" = LSOR with BFMTrack's cost (lsor_bfm)
RR, LK = experiment_keys("outputs/child_52_rerun"), experiment_keys("outputs/child_52_lso")
LOES, CEMP_KEY = RR["loes_anchor"], key_in("outputs/child_52_cem_phc", "cem_phc")
CEMP_RESULT = next(k for k, m in RESULT_KEYS["outputs/success_definitions"].items() if m == "cem_phc")      # "cem_phc"
RERUN_ALGOS = (RR["random"], RR["cmaes"])
SEQ_KEYS = (LK["lso"], LK["lsor_ours"], LK["ta"], LK["lsor_w005"])
SUFFIX = ""
DEFS = ("root", "global", "global_raw", "local"); THR = 0.5
body = None; S = None


def clips():
    from crossembodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    return read_clips(CHILD_52_TASK)


def world(Q):
    """world positions of every body of the module's `body` (crossembodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def tests(q, ref, R):
    """q (T, nq) rollout, ref the reference frames 1.., R = (world positions, local positions) of the reference"""
    Pw_r, Pl_r = R; T = min(len(q), len(ref)); q = q[:T].astype(np.float64); r = ref[:T]
    gw = mean_body_error(world(q), Pw_r[:T]); (pl, _) = body.local(q); gl = mean_body_error(pl, Pl_r[:T])
    # crossembodiment.metrics: the legacy root_success; phc_success on world positions (s, or 1 = raw metres) and on local ones
    return dict(root=float(root_success(q, r)), global_=phc_success(gw, S, THR), global_raw=phc_success(gw, 1.0, THR),
                local=phc_success(gl, S, THR))


def clip(c):
    global body, S
    import eval_metrics as E
    from model.exact_obs import ExactObs
    body = body or metric_body(E.XML["child"], True); S = S or ExactObs(REPO / E.XML["child"], REPO / "assets/robots/adult/robot.xml").scale
    stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    R = (world(ref), body.local(ref)[0]); res = {"frozen": tests(np.load(RERUN / f"{stem}_{LOES}_s0/origin_z.npz")["qpos"], ref, R)}
    self_check = tests(ref, ref, R)
    for src, algos in ((LOESD or RERUN, (LOES,)), (RERUN, RERUN_ALGOS), (CEMP, (CEMP_KEY,))):            # LOES from --loes-dir (default RERUN)
        ix = Reroller.read(src / f"rollouts/{stem}"); Q = np.load(src / f"rollouts/{stem}/eval_qpos.npz")["qpos"]
        for a in algos:
            key = CEMP_RESULT if src == CEMP else a
            res[key + "_best"] = [tests(Q[ix.best_last(f"{stem}_{a}_s{s}")], ref, R) for s in range(5)]
            res[key + "_centre"] = [tests(Q[ix.centre_last(f"{stem}_{a}_s{s}")], ref, R) for s in range(5)]
    res["native"] = tests(np.load(LSO / f"{stem}_{LK['lso']}_s0/rollouts.npz")["q_ER8"], ref, R)
    for k in SEQ_KEYS:
        Rs = [np.load(LSO / f"{stem}_{k}_s{s}/rollouts.npz") for s in range(5)]
        res[k + "_best"] = [tests(x["q_best_it64"], ref, R) for x in Rs]; res[k + "_centre"] = [tests(x["q_mean_it64"], ref, R) for x in Rs]
    return stem, res, self_check


def main(argv=()):
    global LOESD, SUFFIX
    if "--loes-dir" in argv:                                           # LOES ("es") from --loes-dir -> task_motions_noanchor.json
        LOESD, SUFFIX = output_arg(parse_flag(argv, "--loes-dir")), "_noanchor"
    from scipy.stats import wilcoxon
    cl = clips()
    with Pool(workers(24)) as p:
        out = p.map(clip, cl)
    print("reference against itself, all succeed:", all(all(v == 1.0 for v in o[2].values()) for o in out), "| child root scale s:", S if S else "(in workers)")
    per = {o[0]: o[1] for o in out}; keys = {"root": "root", "global": "global_", "global_raw": "global_raw", "local": "local"}
    motion = lambda m, k: np.array([np.mean([x[k] for x in per[s][m]]) if isinstance(per[s][m], list) else per[s][m][k] for s in per])
    methods = list(out[0][1]); summ = {}
    for m in methods:
        summ[m] = {d: float(motion(m, k).mean()) for d, k in keys.items()}
        if m != "es_best":
            summ[m]["vs_loes_best"] = {}
            for d, k in keys.items():
                L, X = motion("es_best", k), motion(m, k); diff = L - X
                summ[m]["vs_loes_best"][d] = dict(loes_higher=int((diff > 0).sum()), loes_lower=int((diff < 0).sum()), p=float(wilcoxon(L, X).pvalue) if (diff != 0).any() else 1.0)
    write_json(OUT / f"task_motions{SUFFIX}.json", dict(per_clip=per, mean=summ))
    print(f"{'method':16s}" + "".join(f"{d:>11s}" for d in keys))
    for m, v in summ.items():
        print(f"{m:16s}" + "".join(f"{100 * v[d]:10.1f}%" for d in keys))
    print("LOES best against each method's best, per motion (LOES higher / lower, Wilcoxon p):")
    for m, v in summ.items():
        if m.endswith("_best") or m in ("frozen", "native"):
            if "vs_loes_best" in v:
                print(f"  {m:16s}" + "  ".join(f"{d}: {x['loes_higher']}/{x['loes_lower']} p {x['p']:.2g}" for d, x in v["vs_loes_best"].items()))


if __name__ == "__main__":
    main(sys.argv)
