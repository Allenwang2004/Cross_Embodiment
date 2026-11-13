#!/usr/bin/env python3
"""success_thresholds.py -- PHC's success test (Luo et al. 2023) at looser thresholds for the single-latent methods on the 52 task
motions (outputs/child_52/clips.txt), child and tall body (len1.4_wid0.8, kp x 0.6, kd x 1). Per frame t: the mean over bodies of the
world-frame body position error between the rollout's state after step t and reference frame t + 1, divided by the body's root
scale s (adult-equivalent metres). A rollout succeeds if that error never exceeds h, for h = 0.5 (the current definition), 0.75, 1.0 m;
"frames within h" = the share of a rollout's frames with error <= h. Best latent so far after 1,024 rollouts; means over the 5 seeds,
then over the motions; Frozen (z0): one rollout per motion.
Same rollouts and test as the existing analyses (no new rollouts):
  child  scripts/success_definitions.py --loes-dir outputs/noanchor/child, its "global" (THR = 0.5): z0 outputs/child_52_rerun/<stem>_es_s0/
         origin_z.npz; Random, CMA-ES outputs/child_52_rerun/rollouts; CEM (PHC objective) outputs/child_52_cem_phc/rollouts; LOES (no
         anchor) outputs/noanchor/child/rollouts; index map[run]["best"][-1]; s = ExactObs(child, adult).s
  tall   scripts/tall_kp06_analysis.py analyse (loes_noanchor.stats, success = not (gw > 0.5 x s).any()): outputs/tall_kp06_52/rollouts/
         <stem>_<algo>/eval_qpos.npz, rows map[run]["best"][-1] - rows[algo][0]; z0 = <stem>_es_s0/origin_z.npz; s = 1.364408905613045
Check: h = 0.5 reproduces, per run, outputs/success_definitions/task_motions_noanchor.json ("global_") and outputs/tall_kp06_52/results.json
("success", budget 1,024); exit code 3 if not (the files are still written).
-> outputs/success_thresholds/task_motions.{json,md}. usage: uv run python scripts/success_thresholds.py [--procs 16]"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.metrics import mean_body_error, world_positions      # noqa: E402
from crossenbodiment.results import write_json                            # noqa: E402
# OOP refactor (ported from the main tree, 11/13): the runs and the stored results are read under paths.outputs_root() ($CE_OUTPUTS_ROOT,
# default outputs/), the tables written under $CE_RESULTS_ROOT (paths.write_path, default outputs/); the clip list is configs/clips' copy.
# The docstring is unchanged (task_motions.md copies it).
OUT_REL = "outputs/success_thresholds"; THRS = (0.5, 0.75, 1.0); SEEDS = range(5)
METHODS = ("frozen", "random", "cem", "cmaes", "es")
NAME = dict(frozen="Frozen (z0)", random="Random Search", cem="CEM (BFM-Zero, PHC objective)", cmaes="CMA-ES", es="LOES (no anchor)")
CLIPS = paths.read_clips(paths.CHILD_52_TASK)         # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
C_RERUN, C_CEM, C_LOES = (paths.output(x) for x in ("child_52_rerun", "child_52_cem_phc", "noanchor/child"))
C_XML = "assets/robots_torque/child/robot_torque_full.xml"; C_REF = "data/child/retargeting_motion/{c}.npz"
T_D = paths.output("tall_kp06_52"); T_XML = "assets/robots_torque/len1.4_wid0.8_kp0.6_kd1.0/robot_torque_full.xml"
T_REF = "data/len1.4_wid0.8/retargeting_motion/{c}.npz"; T_S = 1.364408905613045
PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
_B = {}


def world(body, Q):
    """(T, nbody - 1, 3) world positions of an eval_metrics.Body's bodies (crossenbodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def score(body, q, Pw, S, n_ref):
    """per threshold h: (success, share of frames within h); plus the largest adult-equivalent error and the rollout's frame count"""
    q = np.asarray(q).astype(np.float64); T = min(len(q), n_ref); gw = mean_body_error(world(body, q[:T]), Pw[:T])
    return dict(success={str(h): float(not (gw > h * S).any()) for h in THRS}, within={str(h): float((gw <= h * S).mean()) for h in THRS},
                max_err=float(gw.max() / S), T=int(T), T_ref=int(n_ref))


def child_job(c):
    from crossenbodiment import eval_metrics as E             # scripts/eval_metrics.py re-exports this module (refactor step 8)
    from model.exact_obs import ExactObs
    if "child" not in _B:
        _B["child"] = (E.Body(C_XML, True), ExactObs(REPO / C_XML, REPO / "assets/robots/adult/robot.xml").scale)
    body, S = _B["child"]; st = c.split("/")[1]
    ref = np.load(REPO / C_REF.format(c=c))["qpos"].astype(np.float64)[1:]; Pw = world(body, ref); f = lambda q: score(body, q, Pw, S, len(ref))
    R = {"frozen": [f(np.load(C_RERUN / f"{st}_es_s0/origin_z.npz")["qpos"])]}
    for src, algos in ((C_LOES, ("es",)), (C_RERUN, ("random", "cmaes")), (C_CEM, ("cem",))):
        ix = json.load(open(src / f"rollouts/{st}/index.json"))["map"]; Q = np.load(src / f"rollouts/{st}/eval_qpos.npz")["qpos"]
        for a in algos:
            R[a] = [f(Q[ix[f"{st}_{a}_s{s}"]["best"][-1]]) for s in SEEDS]
    return st, R, float(S)


def tall_job(c):
    from crossenbodiment import eval_metrics as E
    if "tall" not in _B:
        _B["tall"] = E.Body(T_XML, True)
    body = _B["tall"]; st = c.split("/")[1]
    ref = np.load(REPO / T_REF.format(c=c))["qpos"].astype(np.float64)[1:]; Pw = world(body, ref); f = lambda q: score(body, q, Pw, T_S, len(ref))
    ix = json.load(open(T_D / "rollouts" / st / "index.json")); R = {"frozen": [f(np.load(T_D / f"{st}_es_s0/origin_z.npz")["qpos"])]}
    for a in ("random", "cem", "cmaes", "es"):
        lo = ix["rows"][a][0]; Q = np.load(T_D / "rollouts" / f"{st}_{a}" / "eval_qpos.npz")["qpos"]
        R[a] = [f(Q[ix["map"][f"{st}_{a}_s{s}"]["best"][-1] - lo]) for s in SEEDS]
    return st, R, T_S


def reference_success(b):
    """the existing analyses' stored h = 0.5 success, per motion -> per method -> per run"""
    if b == "child":
        P = json.load(open(paths.read_path("outputs/success_definitions/task_motions_noanchor.json")))["per_clip"]
        key = dict(frozen="frozen", random="random_best", cem="cem_phc_best", cmaes="cmaes_best", es="es_best")
        return {st: {m: [x["global_"] for x in (v[key[m]] if isinstance(v[key[m]], list) else [v[key[m]]])] for m in METHODS} for st, v in P.items()}
    r = json.load(open(T_D / "results.json")); k = r["keys"].index("success"); j = r["budgets"].index(1024); P = r["per_motion"]
    return {st: {m: ([v["frozen"][0][k]] if m == "frozen" else [x[j][k] for x in v[m]["best"]]) for m in METHODS} for st, v in P.items()}


def summarise(b, out):
    per = {st: R for st, R, _ in out}; S = out[0][2]; ref = reference_success(b)
    mismatch = [(st, m, i, ref[st][m][i], per[st][m][i]["success"]["0.5"]) for st in per for m in METHODS for i in range(len(per[st][m]))
                if ref[st][m][i] != per[st][m][i]["success"]["0.5"]]
    tab = {}
    for m in METHODS:
        mot = lambda q, h: np.array([np.mean([x[q][str(h)] for x in per[st][m]]) for st in per])
        tab[m] = dict(success={str(h): float(mot("success", h).mean()) for h in THRS}, within={str(h): float(mot("within", h).mean()) for h in THRS},
                      count={str(h): int(sum(x["success"][str(h)] for st in per for x in per[st][m])) for h in THRS},
                      runs=sum(len(per[st][m]) for st in per))
    stored = {m: float(np.mean([np.mean(ref[st][m]) for st in ref])) for m in METHODS}
    short = [(st, m, i, x["T"], x["T_ref"]) for st in per for m in METHODS for i, x in enumerate(per[st][m]) if x["T"] < x["T_ref"]]
    return dict(s=S, motions=len(per), table=tab, check=dict(stored_success_05=stored, runs_differing_at_05=len(mismatch), examples=mismatch[:10],
                                                           rollouts_shorter_than_reference=len(short), short_examples=short[:10]), per_motion=per)


def md(rep):
    L = ["# PHC success at looser thresholds: single-latent methods on the 52 task motions", "",
         "Best latent so far after 1,024 rollouts. Means over the 5 seeds, then over the 52 motions. Frozen (z0): one rollout per motion.",
         "Success: the mean body position error (world frame, adult-equivalent metres) never exceeds h. "
         "Frames within: the share of frames with error at most h."]
    for b, title in (("child", "Child"), ("tall", "Tall body (len1.4_wid0.8, kp x 0.6)")):
        r = rep[b]; T = r["table"]
        L += ["", f"## {title}", "", f"Root scale s = {r['s']:.6g}; h = 0.5 m is {0.5 * r['s']:.3f} m on this body.", "",
              "| Method | Success 0.5 | Success 0.75 | Success 1.0 | Frames within 0.5 | Frames within 0.75 | Frames within 1.0 |", "|---|---|---|---|---|---|---|"]
        L += [f"| {NAME[m]} | " + " | ".join(f"{100 * T[m]['success'][str(h)]:.1f}% ({T[m]['count'][str(h)]}/{T[m]['runs']})" for h in THRS) + " | "
              + " | ".join(f"{100 * T[m]['within'][str(h)]:.1f}%" for h in THRS) + " |" for m in METHODS]
        ch = r["check"]
        L += ["", f"Check, h = 0.5 against the stored success ({'success_definitions task_motions_noanchor.json' if b == 'child' else 'tall_kp06_52/results.json'}): "
              f"{ch['runs_differing_at_05']} runs differ; stored means " + ", ".join(f"{NAME[m]} {100 * v:.1f}%" for m, v in ch["stored_success_05"].items()) + ".",
              f"Rollouts shorter than the reference: {ch['rollouts_shorter_than_reference']}."]
    return "\n".join(L) + "\n\n## Method details\n\n```\n" + __doc__.strip() + "\n```\n"


def main():
    rep = {}
    with Pool(PROCS) as p:
        for b, job in (("child", child_job), ("tall", tall_job)):
            rep[b] = summarise(b, p.map(job, CLIPS, chunksize=1))
    OUT = paths.write_path(OUT_REL); OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "task_motions.json", dict(thresholds=THRS, methods=METHODS, names=NAME, **rep)); (OUT / "task_motions.md").write_text(md(rep))
    for b in rep:
        print(b, "s", rep[b]["s"], json.dumps(rep[b]["check"]))
        for m, v in rep[b]["table"].items():
            print(f"  {m:8s}", " ".join(f"{100 * v['success'][str(h)]:5.1f}% ({v['count'][str(h)]}/{v['runs']})" for h in THRS), "| within",
                  " ".join(f"{100 * v['within'][str(h)]:5.1f}%" for h in THRS))
    if any(rep[b]["check"]["runs_differing_at_05"] for b in rep):
        print("CHECK FAILED: h = 0.5 differs from the stored success"); sys.exit(3)


if __name__ == "__main__":
    main()
