#!/usr/bin/env python3
"""g1_small_test.py -- the Task page's methods on the Unitree G1, small sample before the 52 task motions (user, 11/09):
3 task motions (headstand_4, rotate-x--5-0.8_4: their G1 targets existed; move-ego-0-2_1, a slow walk, target made
11/09 by scripts/robot_roundtrip.py), seeds 0-2, every method with the child's Task-page settings carried over:
  single latent (scripts/robot_z_search.py, 64 x 16 rollouts, start z0, sigma 0.05, reference-velocity start, ridge fill,
                 per-joint gains)
    es      LOES: our cost = BFM term + 1.0 heading + 0.1 position + 0.3 anchor + 1.0 IK residual (the G1 cost of kpkd)
    random  random search, our cost
    cmaes   CMA-ES, our cost
    cem     CEM (BFM-Zero), BFM-Zero's tracking reward (PHC) on the G1's bodies against the G1 target, nothing else
  latent sequence (scripts/robot_lso_search.py, 64 x 16 rollouts + the logged mean, reference-velocity start)
    lso     LSO (BFMTrack): BFMTrack's values, the BFM term alone
    lsor    LSOR: + per-step ranks, 0.05 heading + 0.005 position (x 0.05 as on the child) + 1.0 IK (user, 11/09)
    ta      Trajectory Adaptation (BFM-Zero): from ER L = 8, PHC reward alone
    native  Motivo's ER L = 8 rollout every latent-sequence run makes first
  frozen z0: every single-latent run's own origin_z rollout.
  commands  "<out dir>|<command>" per search, into outputs/g1_small/<stem>_<method>_s<seed>
  prepare   outputs/g1_small/rollouts/<stem>_<algo>/{z.npy, index.json}: each single-latent run's best latent so far and its
            centre after 4, 8, 16, 32, 64 generations (64 to 1,024 rollouts)
  evalcmds  one --eval-z command per motion and single-latent method, with that method's own cost
  analyse   per rollout, against the G1 target (q[t] vs target[t + 1]): MPJPE, MMPJPE, MPJAE (eval_metrics.tracking_metrics
            on the G1), Fall Time (non-foot contact, scripts/fall_definitions.py 'contact'), PHC success (world-frame mean
            body error never above 0.5 m adult-equivalent = 0.41 m on the G1), root drift (mean pelvis xy distance, G1 m).
            Checks: settings, frozen identical across runs, native identical across runs, re-rolled latents reproduce
            the search's best cost so far, latent-sequence best re-roll = saved best. Writes outputs/g1_small/results.json.
  --set 52  the same on the 52 task motions (outputs/child_52/clips.txt; G1 targets of the 47 missing made 11/09 by
            scripts/robot_roundtrip.py), seed 0 only, 128 x 16 = 2,048 rollouts (user, 11/09: 4,096 first, cut to 2,048 after 16 min), budgets 64 to 2,048:
            outputs/g1_52/. Its LOES runs are also checked against the 1,024-rollout runs of the same motion and seed
            (g1_small, kpkd): nothing in the search depends on the total budget, so the first 64 generations must agree.
  --methods es,lso,lsor,ta  only these methods (es and lso always: frozen and native come from their runs)
usage: uv run python scripts/g1_small_test.py [--set small | 52] [--methods a,b,...] commands | prepare | evalcmds | analyse"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.paths import CHILD_52_TASK, read_clips   # noqa: E402  configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
from crossenbodiment.bodies import BODIES
from crossenbodiment.metrics import MetricSuite, metric_body, world_positions
from crossenbodiment import paths
from crossenbodiment.experiment import ClipSet, Experiment, parse_flag, workers
from crossenbodiment.methods import experiment_keys
from crossenbodiment.reroll import EvalRollouts, Reroller, read_best_so_far
from crossenbodiment.results import Checks, ResultsTable, write_json
# refactor step 8: --set / --methods are read in main (configure(), not at import: importing gives --set small with every method);
# runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under paths.results_root() ($CE_RESULTS_ROOT), both
# REPO/outputs by default; prepare writes the schema-2 index (crossenbodiment/reroll.py), analyse reads any.
CONFS = {"small": lambda: dict(out="outputs/g1_small", clips=("headstand/headstand_4", "rotate-x--5-0.8/rotate-x--5-0.8_4", "move-ego-0-2/move-ego-0-2_1"),
                               seeds=(0, 1, 2), G=(4, 8, 16, 32, 64)),
         "52": lambda: dict(out="outputs/g1_52", clips=tuple(read_clips(CHILD_52_TASK)), seeds=(0,), G=(4, 8, 16, 32, 64, 128))}
S = BODIES.scale("g1")              # RobotMap("g1").scale
KEYS = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "drift")
GAINS = "assets/robots/g1/gains_perjoint.json"
OURS = "--heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0.3 --ik-weight 1.0"
# the run-directory key of each method (crossenbodiment/methods.py RUNS: outputs/g1_52 and outputs/g1_small use the same keys; they are
# also results.json's method keys and the --methods values): LOES with the anchor "es", BFM-Zero's CEM "cem", LSOR (w005) "lsor"
K = experiment_keys("outputs/g1_52")
LOES, RANDOM, CMAES, CEM, LSO, LSOR, TA = (K[m] for m in ("loes_anchor", "random", "cmaes", "cem_phc", "lso", "lsor_w005", "ta"))
ALL_SINGLE = {LOES: f"--algo es {OURS}", RANDOM: f"--algo random {OURS}", CMAES: f"--algo cmaes {OURS}",
              CEM: "--algo cem --objective phc --heading-weight 0 --pos-weight 0 --anchor-weight 0 --ik-weight 0"}
ALL_SEQ = {LSO: "--heading-weight 0 --pos-weight 0 --ik-weight 0", LSOR: "--rank --heading-weight 0.05 --pos-weight 0.005 --ik-weight 1.0",
           TA: "--method ta --objective phc --heading-weight 0 --pos-weight 0 --ik-weight 0"}
W_SINGLE = {LOES: (1.0, 0.1, 0.3, 1.0, "bfm", "es"), RANDOM: (1.0, 0.1, 0.3, 1.0, "bfm", "random"), CMAES: (1.0, 0.1, 0.3, 1.0, "bfm", "cmaes"),
            CEM: (0.0, 0.0, 0.0, 0.0, "phc", "cem")}
W_SEQ = {LSO: (0.0, 0.0, 0.0, False, "bfm", "lso"), LSOR: (0.05, 0.005, 1.0, True, "bfm", "lso"), TA: (0.0, 0.0, 0.0, False, "phc", "ta")}
NAMES = {"frozen": "Frozen (z0)", RANDOM: "Random Search", CEM: "CEM (BFM-Zero)", CMAES: "CMA-ES", "native": "Native BFM Tracking Inference",
         LSO: "LSO (BFMTrack)", LSOR: "LSOR (our method)", TA: "Trajectory Adaptation (BFM-Zero)", LOES: "LOES (our method)"}
body = None


def configure(set_="small", methods=None):
    """--set small | 52 and --methods a,b,... (e.g. es,lso,lsor,ta; user, 11/09: the 52 with these four first): set the module's
    SET, CONF, OUT_REL, OUT (read), RES (results), ROLL, CLIPS, SEEDS, G, GENS, EVALS, SINGLE, SEQ, ORDER, EXP and REROLL"""
    global SET, CONF, OUT_REL, OUT, RES, ROLL, CLIPS, SEEDS, G, GENS, EVALS, SINGLE, SEQ, ORDER, EXP, REROLL
    SET = set_; CONF = CONFS[SET]()
    OUT_REL = CONF["out"]; OUT = paths.output(OUT_REL.split("/", 1)[1]); RES = paths.result(OUT_REL.split("/", 1)[1]); ROLL = OUT / "rollouts"
    CLIPS = CONF["clips"]; SEEDS = CONF["seeds"]; G = CONF["G"]
    GENS = G[-1]; EVALS = 16 * GENS                       # 16 rollouts per generation / iteration
    SINGLE, SEQ = dict(ALL_SINGLE), dict(ALL_SEQ)
    if methods is not None:
        M = set(methods)
        assert {LOES, LSO} <= M <= set(SINGLE) | set(SEQ), "frozen z0 comes from the LOES runs and native from the LSO runs"
        SINGLE = {k: v for k, v in SINGLE.items() if k in M}; SEQ = {k: v for k, v in SEQ.items() if k in M}
    ORDER = tuple(k for k in ("frozen", RANDOM, CEM, CMAES, "native", LSO, LSOR, TA, LOES) if k in ("frozen", "native") or k in SINGLE or k in SEQ)
    EXP = Experiment(OUT_REL, clips=ClipSet(CLIPS), methods=(*SINGLE, *SEQ), seeds=SEEDS, budgets=G, order=("seed", "clip", "method"))
    REROLL = Reroller(GENS, G, best="budgets")


configure()


def zcmd(c, k):
    return (f"uv run python scripts/robot_z_search.py --robot g1 --clip {c} --fill ridge --gains {GAINS} --evals {EVALS} --pairs 8 --sigma 0.05 "
            f"--lr 0.03 --sampling independent --init-vel reference --eval-every 0 {SINGLE[k]}")


def lcmd(c, k):
    return (f"uv run python scripts/robot_lso_search.py --robot g1 --ref data/origin_motion/{c}.npz --fill ridge --gains {GAINS} --n 16 --iters {GENS} "
            f"--init-vel reference --trace budgets {SEQ[k]}" + ("" if SET == "small" else " --budgets " + " ".join(map(str, G))))


def stem(c):
    return c.split("/")[1]


def commands():
    """the 52: longest clips first (300, 150, 120 frames), every method of a motion together, so the queue ends on short runs"""
    order = EXP.clips if SET == "small" else EXP.clips.longest_first()
    EXP.print_commands(lambda r: f"{zcmd(r.clip, r.method) if r.method in SINGLE else lcmd(r.clip, r.method)} --seed {r.seed} --out {r.dir}", clip=order)


def prepare():
    for c in CLIPS:
        for k in SINGLE:
            REROLL.prepare(RES / "rollouts" / f"{stem(c)}_{k}", [(f"{stem(c)}_{k}_s{s}", OUT / f"{stem(c)}_{k}_s{s}") for s in SEEDS], clip=c, algo=k)
    print("prepared")


def evalcmds():
    for c in CLIPS:
        for k in SINGLE:
            d = EXP.rollouts_dir(f"{stem(c)}_{k}"); print(EXP.eval_line(d, zcmd(c, k)))


def world(Q):
    """world positions of every body of the module's `body` (crossenbodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def stats(q, ref, Pw, Dr):
    """[fall_s, success, mpjpe, mmpjpe, mpjae, drift] of the module's `body` (the G1; MetricSuite.task_page): success threshold
    0.5 m x S (the G1's s), drift in the G1's metres"""
    return MetricSuite.task_page(body, S).reference(ref, Pw, Dr).values(q)


def settings_bad(k, S_, single):
    if single:
        h, p, an, ik, obj, algo = W_SINGLE[k]
        want = dict(evals=EVALS, pairs=8, sigma=0.05, lr=0.03, sampling="independent", init_vel="reference", frame_offset=1, fill="ridge", gains=GAINS,
                    heading_weight=h, pos_weight=p, anchor_weight=an, ik_weight=ik, objective=obj, algo=algo)
    else:
        h, p, ik, rank, obj, method = W_SEQ[k]
        want = dict(n=16, iters=GENS, budgets=list(G), sigma=0.0125, lr=0.00625, betas=[0.8, 0.99], gamma=0.97, beta=1.0, init_vel="reference", fill="ridge", gains=GAINS,
                    heading_weight=h, pos_weight=p, ik_weight=ik, rank=rank, objective=obj, method=method, best_from="samples")
    return [x for x, v in want.items() if S_.get(x) != v]


def clip(c):
    global body
    import eval_metrics as E
    body = body or metric_body(E.XML["g1"], False); st = stem(c)
    ref = np.load(REPO / f"outputs/g1/roundtrip/{st}/g1.npz")["qpos"].astype(np.float64)[1:]; Pw = world(ref); _, Dr = body.part_floor(ref)
    R = {}; chk = Checks(("frozen_diff", "native_diff", "reroll_cost", "reroll_best_qpos", "seq_reroll"), ("bad",))
    z0 = np.load(OUT / f"{st}_{LOES}_s0/origin_z.npz")["qpos"]; R["frozen"] = np.array([stats(z0, ref, Pw, Dr)])
    for k in SINGLE:
        ev = EvalRollouts(ROLL / f"{st}_{k}", G, GENS); ix = ev.index; Q = ev.qpos
        cost = np.array(ev.cost); B = np.zeros((len(SEEDS), len(G), len(KEYS))); C = np.zeros_like(B)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(OUT / r / "summary.json")); chk.add("bad", [(r, x) for x in settings_bad(k, S_, True)])
            chk.diff("frozen_diff", np.load(OUT / r / "origin_z.npz")["qpos"], z0)
            chk.high("reroll_cost", ix.check_vs_curve(r, cost, read_best_so_far(OUT / r)))
            chk.diff("reroll_best_qpos", Q[ix.best_last(r)], np.load(OUT / r / "best.npz")["qpos"])
            bi, ci = ix.best_budgets(r), ix.centre(r)
            for j in range(len(G)):
                B[i, j] = stats(Q[bi[j]], ref, Pw, Dr); C[i, j] = stats(Q[ci[j]], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = B, C
    nat = np.load(OUT / f"{st}_{LSO}_s0/rollouts.npz")["q_ER8"]; R["native"] = np.array([stats(nat, ref, Pw, Dr)])
    for k in SEQ:
        B = np.zeros((len(SEEDS), len(G), len(KEYS))); C = np.zeros_like(B)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(OUT / r / "summary.json")); chk.add("bad", [(r, x) for x in settings_bad(k, S_, False)])
            X = np.load(OUT / r / "rollouts.npz"); chk.diff("native_diff", X["q_ER8"], nat)
            chk.diff("seq_reroll", X["q_LSO_best"], X[f"q_best_it{GENS}"])
            for j, g in enumerate(G):
                B[i, j] = stats(X[f"q_best_it{g}"], ref, Pw, Dr); C[i, j] = stats(X[f"q_mean_it{g}"], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = B, C
    return st, R, chk


def analyse():
    with Pool(workers(min(len(CLIPS), 16))) as p:
        out = p.map(clip, CLIPS)
    chk = {x: Checks.worst([o[2] for o in out], x) for x in ("frozen_diff", "native_diff", "reroll_cost", "reroll_best_qpos", "seq_reroll")}
    bad = [b for o in out for b in o[2]["bad"]]
    print(f"settings not as intended: {len(bad)} {bad[:6]} | frozen across runs, largest qpos difference: {chk['frozen_diff']} | native across runs: {chk['native_diff']}")
    print(f"single-latent re-rolls vs the search's best cost so far, largest difference: {chk['reroll_cost']:.3g} | best latent re-rolled vs saved best rollout, "
          f"largest qpos difference: {chk['reroll_best_qpos']:.3g} | latent-sequence best re-rolled vs saved: {chk['seq_reroll']:.3g}")
    others = [paths.output("kpkd/g1_kp1_kd1")] + ([paths.output("g1_small")] if SET != "small" else [])
    kp = [(st, s, o) for st in (stem(c) for c in CLIPS) for s in SEEDS for o in others if (o / f"{st}_s{s}/z_trace.npz").exists() or (o / f"{st}_{LOES}_s{s}/z_trace.npz").exists()]
    def zb(o, st, s):
        f = o / f"{st}_s{s}/z_trace.npz"; return np.load(f if f.exists() else o / f"{st}_{LOES}_s{s}/z_trace.npz")["z_best"]
    kpd = max((float(np.abs(np.load(OUT / f"{st}_{LOES}_s{s}/z_trace.npz")["z_best"][:len(zb(o, st, s))] - zb(o, st, s)[:GENS + 1]).max()) for st, s, o in kp), default=float("nan"))
    print(f"LOES against the same motion and seed in {', '.join(o.name for o in others)} ({len(kp)} runs), largest best-latent difference over the generations both have: {kpd}")
    chk["loes_vs_kpkd"] = kpd
    res = {o[0]: o[1] for o in out}; rep = dict(checks=dict(chk, bad=bad), per_motion={}, mean={})

    T = ResultsTable(KEYS, list(res), G, SEEDS)                          # frozen / native: one rollout per motion
    for k in res[next(iter(res))]:
        T.add(k, np.stack([res[st][k][0] if k in ("frozen", "native") else res[st][k] for st in res]))
    key = lambda k, w: k if k in ("frozen", "native") else f"{k}_{w}"
    for w in ("best", "centre"):
        rep["mean"][w] = {k: T.by_budget(key(k, w)) for k in ORDER}
        pm = {k: T.per_motion(key(k, w), -1) for k in ORDER}
        rep["per_motion"][w] = {st: {k: pm[k][st] for k in ORDER} for st in res}
    write_json(RES / "results.json", rep)
    shown = [b for b in dict.fromkeys((1024, EVALS)) if b // 16 in G]
    for b in shown:
        for w, lab in (("best", "Best latent so far"), ("centre", "Centre latent")):
            print(f"\n{lab}, after {b:,} rollouts: {len(CLIPS)} motions x {len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''} (frozen, native: one rollout per motion)")
            print(f"{'method':34s}{'Fall (s)':>9s}{'Success':>9s}{'MPJPE':>8s}{'MMPJPE':>8s}{'MPJAE':>8s}{'drift m':>9s}")
            for k in ORDER:
                v = rep["mean"][w][k][str(b)]
                print(f"{NAMES[k]:34s}{v['fall_s']:9.3f}{100 * v['success']:8.1f}%{v['mpjpe']:8.2f}{v['mmpjpe']:8.2f}{v['mpjae']:8.2f}{v['drift']:9.2f}")
    for w in ("best", "centre"):
        for m in ("success", "mpjpe", "mmpjpe", "mpjae"):
            print(f"\n{w}, {m} by rollout budget ({', '.join(str(g * 16) for g in G)})")
            for k in ORDER:
                print(f"  {NAMES[k]:34s}" + "".join(f"{(100 if m == 'success' else 1) * rep['mean'][w][k][str(g * 16)][m]:8.2f}" for g in G))
    n = len(SEEDS)
    print(f"\nper motion, best after {EVALS:,}: success (of {n} seed{'s' if n > 1 else ''}) | MPJPE | drift")
    for st in res:
        print(f"  {st:24s}" + " | ".join(f"{k}: {n * rep['per_motion']['best'][st][k]['success']:.0f}/{n} {rep['per_motion']['best'][st][k]['mpjpe']:.2f} "
                                         f"{rep['per_motion']['best'][st][k]['drift']:.2f}" for k in ORDER if k not in ("frozen", "native")))
        print(f"  {'':24s}frozen: success {rep['per_motion']['best'][st]['frozen']['success']:.0f} MPJPE {rep['per_motion']['best'][st]['frozen']['mpjpe']:.2f} | "
              f"native: success {rep['per_motion']['best'][st]['native']['success']:.0f} MPJPE {rep['per_motion']['best'][st]['native']['mpjpe']:.2f}")

def main(argv):
    m = parse_flag(argv, "--methods")
    configure(parse_flag(argv, "--set", "small"), m.split(",") if m is not None else None)
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[[x for x in argv[1:] if x in ("commands", "prepare", "evalcmds", "analyse")][0]]()


if __name__ == "__main__":
    main(sys.argv)
