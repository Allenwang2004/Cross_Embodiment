#!/usr/bin/env python3
"""body_52_test.py -- LOES, LSO, LSOR and TA on the 52 task motions (outputs/child_52/clips.txt) for the payload and the limited leg
(user, 11/09: "the same as the G1", after the G1 run of scripts/g1_small_test.py --set 52): seed 0, 128 x 16 = 2,048 rollouts.
  payload    assets/robots/adult_heavyRWristHand_x80 (right hand x 80), tracks the adult's own motion (data/origin_motion)
  limit_leg  assets/robots/adult_lockRleg_bent_lim (right leg held bent by its joint ranges), tracks the bent-leg reference
             (data/adult_lockRleg_bent_lim/retargeting_motion; the 48 missing of the 52 built 11/09 exactly as
             scripts/lock_joint_body.py --user-lock builds them -- the 12 existing ones rebuild to 0.0)
Both bodies are the adult's size: root scale s = 1 (the reference's x, y over the adult's, 1.000000 on every clip), so
adult-equivalent metres are the body's own and PHC's 0.5 m is 0.5 m.
  es    LOES, scripts/single_z_search.py with the settings of scripts/kpkd_test.py (the same bodies' LOES runs) at 2,048 rollouts:
        our cost = BFM term + 1.0 heading + 0.1 position + 0.3 anchor (z0), reference-velocity start, exact obs (s read off the clip)
  lso   LSO (BFMTrack), scripts/lso_search.py with scripts/child_52_lso.py's settings (BFMTrack's values, the BFM term alone),
        --scale 1 (lso_search.py's default is the child's 0.611), reference-velocity start
  lsor  + per-step ranks, 0.05 heading + 0.005 position (as on the child, lsor_w005)
  ta    Trajectory Adaptation (BFM-Zero): --method ta --objective phc, from ER L = 8
  native  Motivo's ER L = 8 rollout of the LSO run; frozen z0: the LOES run's origin_z rollout
  commands  "<out dir>|<command>" per search, into outputs/body_52/<body>/<stem>_<method>_s0 (the two bodies interleaved, longest clips first)
  prepare   <body>/rollouts/<stem>/{z.npy, index.json}: LOES's best latent so far and its centre after 4 ... 128 generations
  evalcmds  one single_z_search.py --eval-z per body and motion
  cem   CEM (BFM-Zero), added 11/10 (user): scripts/single_z_search.py --algo cem --objective phc, BFM-Zero's tracking reward alone (as
        outputs/child_52_cem_phc), 2,048 rollouts; cemcommands, then prepare / evalcmds --algos cem (latents in rollouts/<stem>_cem)
  analyse   per rollout, q[t] against ref[t + 1], on the body itself: Fall Time (non-foot contact), PHC success (world-frame mean
            body error never above 0.5 m), MPJPE, MMPJPE, MPJAE, root drift (mean pelvis xy distance, m). Checks: settings, re-rolled
            LOES latents reproduce the search's best cost so far and its saved best rollout, native identical across a motion's
            runs, latent-sequence best re-roll = saved best, LOES's first 64 generations = the kpkd runs of the same body, motion, seed.
            Writes outputs/body_52/<body>/results.json (the layout of scripts/g1_small_test.py's) and prints the tables.
usage: uv run python scripts/body_52_test.py commands | prepare | evalcmds | analyse [--body payload|limit_leg]"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment import paths   # noqa: E402
from crossenbodiment.experiment import ClipSet, Experiment, parse_flag, workers   # noqa: E402
from crossenbodiment.metrics import MetricSuite, metric_body, world_positions   # noqa: E402
from crossenbodiment.paths import CHILD_52_TASK, read_clips   # noqa: E402  configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
from crossenbodiment.reroll import EvalRollouts, Reroller, read_best_so_far   # noqa: E402
from crossenbodiment.results import Checks, ResultsTable, write_json   # noqa: E402
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under paths.results_root()
# ($CE_RESULTS_ROOT); both REPO/outputs by default. --algos / --body are read in main (not at import); prepare writes the
# schema-2 index (crossenbodiment/reroll.py), analyse reads any.
OUT = paths.output("body_52"); RES = paths.result("body_52"); CLIPS = tuple(read_clips(CHILD_52_TASK))
SEEDS = (0,); G = (4, 8, 16, 32, 64, 128); GENS = G[-1]; EVALS = 16 * GENS; KEYS = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "drift")
BODIES = dict(payload=dict(label="adult_heavyRWristHand_x80", xml="assets/robots/adult_heavyRWristHand_x80/robot.xml", ref="data/origin_motion/{c}.npz",
                           kpkd="heavy_arm_kp1_kd1"),
              limit_leg=dict(label="adult_lockRleg_bent_lim", xml="assets/robots/adult_lockRleg_bent_lim/robot.xml",
                             ref="data/adult_lockRleg_bent_lim/retargeting_motion/{c}.npz", kpkd="limit_leg_kp1_kd1"))
SEQ = dict(lso="", lsor="--rank --heading-weight 0.05 --pos-weight 0.005", ta="--method ta --objective phc")
W_SEQ = dict(lso=dict(method="lso", objective="bfm", rank=False, heading_weight=0.0, pos_weight=0.0),
             lsor=dict(method="lso", objective="bfm", rank=True, heading_weight=0.05, pos_weight=0.005),
             ta=dict(method="ta", objective="phc", rank=False, heading_weight=0.0, pos_weight=0.0))
NAMES = dict(frozen="Frozen (z0)", random="Random Search", cem="CEM (BFM-Zero)", cmaes="CMA-ES", native="Native BFM Tracking Inference", lso="LSO (BFMTrack)", lsor="LSOR (our method)",
             ta="Trajectory Adaptation (BFM-Zero)", es="LOES (our method)")
ORDER = ("frozen", "random", "cem", "cmaes", "native", "lso", "lsor", "ta", "es")   # cem added 11/10, random / cmaes 11/10 (algocommands --algos ...);
SINGLE = ("es", "random", "cem", "cmaes")                                # analyse keeps the methods whose latents were re-rolled for every motion
ALGOS = ["es", "cem"]                                                    # --algos a,b (main)
EXP = Experiment("outputs/body_52", clips=ClipSet(CLIPS), methods=("es", *SEQ), seeds=SEEDS, bodies=tuple(BODIES), budgets=G,
                 name="{body}/{stem}_{method}_s{seed}", order=("seed", "clip", "body", "method"))
REROLL = Reroller(GENS, G, best="budgets")                              # the best and the centre latent at each budget
body = None


def stem(c):
    return c.split("/")[1]


def zcmd(b, c, k="es"):
    """LOES (es) and, with our cost and LOES's settings, random search and CMA-ES (--algo random / cmaes; user, 11/10)"""
    B = BODIES[b]
    return (f"uv run python scripts/single_z_search.py --clip {c} --body {B['label']} --xml {B['xml']} --ref {B['ref'].format(c=c)} --objective bfm "
            f"--evals {EVALS} --pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy "
            f"--anchor-weight 0.3 --eval-every 0 --init-vel-scale 1" + ("" if k == "es" else f" --algo {k}"))


def lcmd(b, c, k):
    B = BODIES[b]
    return (f"uv run python scripts/lso_search.py --ref {B['ref'].format(c=c)} --body {B['label']} --xml {B['xml']} --scale 1 --n 16 --iters {GENS} "
            f"--init-vel-scale 1 --trace budgets --budgets {' '.join(map(str, G))} {SEQ[k]}").replace("  ", " ")


def ccmd(b, c):
    """CEM (BFM-Zero): BFM-Zero's tracking reward (PHC) alone (no heading, position or anchor), as outputs/child_52_cem_phc (user, 11/10)."""
    B = BODIES[b]
    return (f"uv run python scripts/single_z_search.py --clip {c} --body {B['label']} --xml {B['xml']} --ref {B['ref'].format(c=c)} --objective phc --algo cem "
            f"--pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --eval-every 0 --evals {EVALS} --init-vel-scale 1")


def longest_first():
    return ClipSet(CLIPS).longest_first()


def cemcommands():
    EXP.print_commands(lambda r: f"{ccmd(r.body, r.clip)} --seed {r.seed} --out {r.dir}", clip=longest_first(), method=("cem",))


def algocommands(algos=None):
    """--algos random,cmaes (or cem): those single-latent searches, longest clips first, the two bodies interleaved"""
    EXP.print_commands(lambda r: f"{ccmd(r.body, r.clip) if r.method == 'cem' else zcmd(r.body, r.clip, r.method)} --seed {r.seed} --out {r.dir}",
                       clip=longest_first(), method=algos or ALGOS)


def commands():
    def cmd(r):
        if r.method == "es":
            return f"{zcmd(r.body, r.clip)} --seed {r.seed} --out {r.dir}"
        return " ".join(f"{lcmd(r.body, r.clip, r.method)} --seed {r.seed} --out {r.dir}".split())
    EXP.print_commands(cmd, clip=longest_first())


def rname(c, k):
    return stem(c) + ("" if k == "es" else f"_{k}")                     # LOES's latents in rollouts/<stem>, CEM's in rollouts/<stem>_cem


def rdir(b, c, k):
    return OUT / b / "rollouts" / rname(c, k)


def prepare(algos=None):
    algos = algos or ALGOS
    for b in BODIES:
        for c in CLIPS:
            for k in algos:
                REROLL.prepare(RES / b / "rollouts" / rname(c, k), [(f"{stem(c)}_{k}_s{s}", OUT / b / f"{stem(c)}_{k}_s{s}") for s in SEEDS],
                               body=b, clip=c)
    print("prepared", algos)


def evalcmds(algos=None):
    for b in BODIES:
        for c in CLIPS:
            for k in algos or ALGOS:            # each latent rolled out and scored with its own search's cost
                d = EXP.rollouts_dir(rname(c, k), body=b); print(EXP.eval_line(d, ccmd(b, c) if k == 'cem' else zcmd(b, c, k)))


def world(Q):
    """world positions of every body of the module's `body` (crossenbodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def stats(q, ref, Pw, Dr):
    """[fall_s, success, mpjpe, mmpjpe, mpjae, drift] of the module's `body` (the Task page's metrics, MetricSuite.task_page):
    success threshold 0.5 m (s = 1: the payload and the limited leg are adult-sized), drift in metres"""
    return MetricSuite.task_page(body, 1.0).reference(ref, Pw, Dr).values(q)


def settings_bad(b, c, k, S_, s):
    B = BODIES[b]
    if k == "cem":
        want = dict(evals=EVALS, pairs=8, sigma=0.05, lr=0.03, sampling="independent", fitness="rank", init="reference", frame_offset=1, obs_scale="exact",
                    pos_div=1.0, pos_unit="adult", heading_weight=0.0, pos_weight=0.0, anchor=None, anchor_weight=0.0, init_vel_scale=1.0,
                    objective="phc", algo="cem", xml=B["xml"], ref=B["ref"].format(c=c), body=B["label"], seed=s)
    elif k in ("es", "random", "cmaes"):
        want = dict(evals=EVALS, pairs=8, sigma=0.05, lr=0.03, sampling="independent", fitness="rank", init="reference", frame_offset=1, obs_scale="exact",
                    pos_div=1.0, pos_unit="adult", heading_weight=1.0, pos_weight=0.1, anchor=f"data/origin_z/{c}.npy", anchor_weight=0.3, init_vel_scale=1.0,
                    objective="bfm", algo=k, xml=B["xml"], ref=B["ref"].format(c=c), body=B["label"], seed=s)
    else:
        want = dict(n=16, iters=GENS, budgets=list(G), sigma=0.0125, lr=0.00625, betas=[0.8, 0.99], gamma=0.97, beta=1.0, init_vel_scale=1.0, antithetic=False,
                    pos_unit="adult", pos_div=1.0, sample_rollouts=EVALS, ref=B["ref"].format(c=c), body=B["label"], seed=s, **W_SEQ[k])
    return [x for x, v in want.items() if S_.get(x) != v]


def clip(job):
    global body
    import eval_metrics as E
    b, c = job; B = BODIES[b]; D = OUT / b; st = stem(c)
    if body is None or body.xml != B["xml"]:
        body = metric_body(str(REPO / B["xml"]), True); body.xml = B["xml"]
    ref = np.load(REPO / B["ref"].format(c=c))["qpos"].astype(np.float64)[1:]; Pw = world(ref); _, Dr = body.part_floor(ref)
    R = {}; chk = Checks(("native_diff", "reroll_cost", "reroll_best_qpos", "seq_reroll", "cem_z0"), ("bad",))
    R["frozen"] = np.array([stats(np.load(D / f"{st}_es_s0/origin_z.npz")["qpos"], ref, Pw, Dr)])
    for k in [k for k in SINGLE if (rdir(b, c, k) / "eval_qpos.npz").exists()]:   # the single-latent searches: best and centre latents re-rolled
        ev = EvalRollouts(rdir(b, c, k), G, GENS); ix = ev.index; Q = ev.qpos
        cost = np.array(ev.cost); Bm = np.zeros((len(SEEDS), len(G), len(KEYS))); Cm = np.zeros_like(Bm)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(D / r / "summary.json")); chk.add("bad", [(b, r, x) for x in settings_bad(b, c, k, S_, s)])
            chk.high("reroll_cost", ix.check_vs_curve(r, cost, read_best_so_far(D / r)))
            chk.diff("reroll_best_qpos", Q[ix.best_last(r)], np.load(D / r / "best.npz")["qpos"][:Q.shape[1]])
            bi, ci = ix.best_budgets(r), ix.centre(r)
            for j in range(len(G)):
                Bm[i, j] = stats(Q[bi[j]], ref, Pw, Dr); Cm[i, j] = stats(Q[ci[j]], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = Bm, Cm
    z0 = np.load(D / f"{st}_es_s0/origin_z.npz")["qpos"]                # every single-latent search starts from the same z0 rollout
    chk["cem_z0"] = max(float(np.abs(np.load(D / f"{st}_{k}_s0/origin_z.npz")["qpos"] - z0).max()) for k in SINGLE if f"{k}_best" in R and k != "es")
    nat = np.load(D / f"{st}_lso_s0/rollouts.npz")["q_ER8"]; R["native"] = np.array([stats(nat, ref, Pw, Dr)])
    for k in SEQ:
        Bm = np.zeros((len(SEEDS), len(G), len(KEYS))); Cm = np.zeros_like(Bm)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(D / r / "summary.json")); chk.add("bad", [(b, r, x) for x in settings_bad(b, c, k, S_, s)])
            X = np.load(D / r / "rollouts.npz"); chk.diff("native_diff", X["q_ER8"], nat)
            chk.diff("seq_reroll", X["q_LSO_best"], X[f"q_best_it{GENS}"])
            for j, g in enumerate(G):
                Bm[i, j] = stats(X[f"q_best_it{g}"], ref, Pw, Dr); Cm[i, j] = stats(X[f"q_mean_it{g}"], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = Bm, Cm
    kp = paths.output(f"kpkd/{B['kpkd']}/{st}_s0/z_trace.npz")
    chk["loes_vs_kpkd"] = (float(np.abs(np.load(D / f"{st}_es_s0/z_trace.npz")["z_best"][:65] - np.load(kp)["z_best"]).max()) if kp.exists() else None)
    return b, st, R, chk


def analyse(bodies=None):
    bodies = bodies or list(BODIES)
    with Pool(workers(16)) as p:
        out = p.map(clip, [(b, c) for b in bodies for c in CLIPS], chunksize=1)
    for b in bodies:
        o = [x for x in out if x[0] == b]; bad = [y for x in o for y in x[3]["bad"]]
        chk = {k: Checks.worst([x[3] for x in o], k) for k in ("native_diff", "reroll_cost", "reroll_best_qpos", "seq_reroll", "cem_z0")}
        kp = [x[3]["loes_vs_kpkd"] for x in o if x[3]["loes_vs_kpkd"] is not None]; chk["loes_vs_kpkd"] = (max(kp) if kp else None, len(kp))
        print(f"\n===== {b} ({BODIES[b]['label']}) =====")
        print(f"settings not as intended: {len(bad)} {bad[:6]} | native across runs, largest qpos difference: {chk['native_diff']}")
        print(f"z0 rollout of the other single-latent searches vs LOES's, largest qpos difference: {chk['cem_z0']}")
        print(f"LOES and CEM re-rolls vs the search's best cost so far, largest difference: {chk['reroll_cost']:.3g} | best latent re-rolled vs saved best rollout: "
              f"{chk['reroll_best_qpos']:.3g} | latent-sequence best re-rolled vs saved: {chk['seq_reroll']:.3g}")
        print(f"LOES against outputs/kpkd/{BODIES[b]['kpkd']} (same motion, seed 0; {kp and len(kp)} runs), largest best-latent difference over their 64 generations: {chk['loes_vs_kpkd'][0]}")
        res = {x[1]: x[2] for x in o}; rep = dict(checks=dict(chk, bad=bad), per_motion={}, mean={})
        ORDER = [k for k in globals()["ORDER"] if k in ("frozen", "native") or all(f"{k}_best" in res[st] for st in res)]

        T = table(res)
        key = lambda k, w: k if k in ("frozen", "native") else f"{k}_{w}"
        for w in ("best", "centre"):
            rep["mean"][w] = {k: T.by_budget(key(k, w)) for k in ORDER}
            rep["per_motion"][w] = per_motion(T, ORDER, key, w, -1)
        rep["per_motion_1024"] = {w: per_motion(T, ORDER, key, w, G.index(64)) for w in ("best", "centre")}
        write_json(RES / b / "results.json", rep)
        for n_ in (1024, EVALS):
            for w, lab in (("best", "Best latent so far"), ("centre", "Centre latent")):
                print(f"\n{lab}, after {n_:,} rollouts: {len(res)} motions x 1 seed (frozen, native: one rollout per motion)")
                print(f"{'method':34s}{'Fall (s)':>9s}{'Success':>9s}{'MPJPE':>8s}{'MMPJPE':>8s}{'MPJAE':>8s}{'drift m':>9s}")
                for k in ORDER:
                    v = rep["mean"][w][k][str(n_)]
                    print(f"{NAMES[k]:34s}{v['fall_s']:9.3f}{100 * v['success']:8.1f}%{v['mpjpe']:8.2f}{v['mmpjpe']:8.2f}{v['mpjae']:8.2f}{v['drift']:9.2f}")
        for w in ("best", "centre"):
            for m in ("success", "mpjpe", "mmpjpe", "mpjae"):
                print(f"\n{w}, {m} by rollout budget ({', '.join(str(g * 16) for g in G)})")
                for k in ORDER:
                    print(f"  {NAMES[k]:34s}" + "".join(f"{(100 if m == 'success' else 1) * rep['mean'][w][k][str(g * 16)][m]:8.2f}" for g in G))


def table(res):
    """{motion: R} (clip()'s results) -> a ResultsTable: frozen / native (one rollout per motion) and <method>_best / _centre"""
    T = ResultsTable(KEYS, list(res), G, SEEDS)
    for k in [k for k in next(iter(res.values())) if all(k in res[st] for st in res)]:   # the methods every motion has
        T.add(k, np.stack([res[st][k][0] if k in ("frozen", "native") else res[st][k] for st in res]))
    return T


def per_motion(T, order, key, w, j):
    """{motion: {method: {metric: mean over seeds}}} at budget index j (motion-major, as results.json lays it out)"""
    pm = {k: T.per_motion(key(k, w), j) for k in order}
    return {st: {k: pm[k][st] for k in order} for st in T.motions}


def main(argv):
    global ALGOS
    ALGOS = parse_flag(argv, "--algos", "es,cem").split(",")
    A = {"commands": commands, "cemcommands": cemcommands, "algocommands": algocommands, "prepare": prepare, "evalcmds": evalcmds,
         "analyse": lambda: analyse([argv[argv.index("--body") + 1]] if "--body" in argv else None)}
    A[[x for x in argv[1:] if x in A][0]]()


if __name__ == "__main__":
    main(sys.argv)
