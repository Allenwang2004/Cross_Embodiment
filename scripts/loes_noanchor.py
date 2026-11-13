#!/usr/bin/env python3
"""loes_noanchor.py -- LOES without the anchor term (user, 11/11: the method figure dropped "+ anchor"; re-run our method first, the
baselines later). Every LOES search of the Task page again with --anchor-weight 0 and nothing else changed:
  child        52 task motions x 5 seeds, 1,024 rollouts   (outputs/child_52_rerun's settings)       -> outputs/noanchor/child/<stem>_es_s<seed>
  payload      52 x seed 0, 2,048 rollouts                 (scripts/body_52_test.py zcmd)             -> outputs/noanchor/payload/<stem>_es_s0
  limit_leg    52 x seed 0, 2,048 rollouts                 (scripts/body_52_test.py zcmd)             -> outputs/noanchor/limit_leg/<stem>_es_s0
  g1           52 x seed 0, 2,048 rollouts                 (scripts/g1_small_test.py --set 52 zcmd)   -> outputs/noanchor/g1/<stem>_es_s0
The commands are the original builders' with "--anchor-weight 0.3" replaced by "--anchor-weight 0" (checked: exactly one each).
--anchor stays given, so anchor_dist is still logged. LSOR has no anchor term (scripts/lso_search.py), so it is not re-run.
  commands   "<out dir>|<command>" for every search (child first, then the bodies and the G1, longest clips first)
  prepare    <body>/rollouts/<stem>/{z.npy, index.json}: each run's best latent so far and its centre after G generations; on the
             child the best latent after every generation 1-64 (repeats shared), as outputs/child_52_rerun, for the convergence plot
  evalcmds   "<rollouts dir>|<command>": those latents re-rolled with the search's own settings (--eval-z --eval-save-qpos)
  analyse    the Task page's metrics per rollout, q[t] against ref[t + 1]: Fall Time (non-foot contact while the reference's is
             not), PHC success (world-frame mean body error never above 0.5 m adult-equivalent), MPJPE, MMPJPE, MPJAE, root drift
             (the body's own metres, every body; divide the child's by 0.611 for adult-equivalent); means over motions (child: over the 5 seeds first). Next to the anchored
             LOES (child: re-computed here from outputs/child_52_rerun with the same code, which must give the Task page's numbers;
             bodies, G1: their results.json) and the baselines. Checks: settings equal the anchored runs' but anchor_weight (and the
             xml path's spelling); the z0 rollout is the anchored runs'; re-rolls reproduce each search's best cost so far.
             -> outputs/noanchor/results.json, outputs/noanchor/README.md
usage: uv run python scripts/loes_noanchor.py commands | prepare | evalcmds | analyse [--bodies child,payload,limit_leg,g1]"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts")); OUT = "outputs/noanchor"
from crossembodiment.bodies import BODIES
from crossembodiment.paths import CHILD_XML_REL
from crossembodiment.metrics import MetricSuite, metric_body, world_positions
from crossembodiment.paths import CHILD_52_TASK, read_clips   # noqa: E402  configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
from crossembodiment import paths
from crossembodiment.experiment import ClipSet, Experiment, parse_flag, workers
from crossembodiment.methods import key_in
from crossembodiment.reroll import EvalRollouts, Reroller, read_best_so_far
from crossembodiment.results import Checks, ReportWriter, ResultsTable, write_json
# refactor step 8: --bodies is read in main (not at import); runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results
# written under paths.results_root() ($CE_RESULTS_ROOT), both REPO/outputs by default; prepare writes the schema-2 index
# (crossembodiment/reroll.py), analyse reads any. body_52_test / g1_small_test are imported for their builders (g1: configure("52")).
CLIPS = read_clips(CHILD_52_TASK)
ORDER = sorted(CLIPS, key=lambda c: -len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]))
# run-directory keys (crossembodiment/methods.py RUNS): LOES without the anchor is "es" here; the anchored LOES it is compared with is
# "es" too in child_52_rerun, g1_52 and body_52/<body> (and their results.json method key)
LOES = key_in("outputs/noanchor/<body>", "loes")
ANCHORED = key_in("outputs/child_52_rerun", "loes_anchor")
assert ANCHORED == key_in("outputs/g1_52", "loes_anchor") == key_in("outputs/body_52/<body>", "loes_anchor")
EXP = Experiment(OUT, clips=ClipSet(ORDER), methods=(LOES,), name="{body}/{stem}_{method}_s{seed}")


def child_cmd(c, s):
    return (f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {CHILD_XML_REL} --objective bfm "
            f"--evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy "
            f"--anchor-weight 0.3 --eval-every 0 --init-vel-scale 1 --seed {s}")


def noanchor(cmd):
    assert cmd.count("--anchor-weight 0.3") == 1, cmd
    return cmd.replace("--anchor-weight 0.3", "--anchor-weight 0")


def builders():
    """body_52_test (zcmd, BODIES) and g1_small_test configured for --set 52 (zcmd with 2,048 rollouts)"""
    import body_52_test as B
    import g1_small_test as G1
    G1.configure("52")
    return B, G1


def commands():
    B, G1 = builders()
    EXP.print_commands(lambda r: f"{noanchor(child_cmd(r.clip, r.seed))} --out {r.dir}", order=("seed", "clip"), seed=range(5), body=("child",))
    for c in ORDER:
        EXP.print_commands(lambda r: f"{noanchor(B.zcmd(r.body, r.clip, B.LOES))} --seed 0 --out {r.dir}", order=("body",), clip=(c,),
                           body=("payload", "limit_leg"))
        EXP.print_commands(lambda r: f"{noanchor(G1.zcmd(r.clip, G1.LOES))} --seed 0 --out {r.dir}", order=(), clip=(c,), body=("g1",))


REP = Path(__file__).resolve().parent.parent
IN, RESD = paths.output("noanchor"), paths.result("noanchor")      # the runs (read) and the analysis (written)
CONF = dict(child=dict(seeds=range(5), G=(4, 8, 16, 32, 64), s=BODIES.scale("child")), payload=dict(seeds=(0,), G=(4, 8, 16, 32, 64, 128), s=BODIES.scale("payload")),
            limit_leg=dict(seeds=(0,), G=(4, 8, 16, 32, 64, 128), s=BODIES.scale("limit_leg")), g1=dict(seeds=(0,), G=(4, 8, 16, 32, 64, 128), s=BODIES.scale("g1")))
KEYS = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "drift")
BODIES = list(CONF)                                                   # --bodies a,b (main)


def st_(c):
    return c.split("/")[1]


def done(b, c):
    return all(EXP.done(EXP.run(c, LOES, s, b)) for s in CONF[b]["seeds"])


def reroller(b):
    """child: every generation's distinct best (as outputs/child_52_rerun, for the convergence plot); the others: the budgets"""
    G = CONF[b]["G"]
    return Reroller(G[-1], G, best="every" if b == "child" else "budgets")


def prepare():
    n = 0
    for b in BODIES:
        for c in CLIPS:
            if not done(b, c):
                continue
            reroller(b).prepare(RESD / b / "rollouts" / st_(c), [(f"{st_(c)}_{LOES}_s{s}", IN / b / f"{st_(c)}_{LOES}_s{s}") for s in CONF[b]["seeds"]],
                                body=b, clip=c); n += 1
    print("prepared", n)


def evalcmds():
    B, G1 = builders()
    for b in BODIES:
        for c in CLIPS:
            d = EXP.rollouts_dir(st_(c), body=b)
            if not (IN / b / "rollouts" / st_(c) / "z.npy").exists():
                continue
            base = {"child": lambda: child_cmd(c, 0), "g1": lambda: G1.zcmd(c, G1.LOES)}.get(b, lambda: B.zcmd(b, c, B.LOES))()
            print(EXP.eval_line(d, noanchor(base), seed=None if b == "child" else 0))


_B = {}


def body_of(b):
    import eval_metrics as E
    if b not in _B:
        import body_52_test as Bt
        _B[b] = (metric_body(E.XML["child"], True) if b == "child" else metric_body(E.XML["g1"], False) if b == "g1" else metric_body(str(REP / Bt.BODIES[b]["xml"]), True))
    return _B[b]


def ref_of(b, c):
    import body_52_test as Bt
    p = {"child": f"data/child/retargeting_motion/{c}.npz", "g1": f"outputs/g1/roundtrip/{st_(c)}/g1.npz"}.get(b) or Bt.BODIES[b]["ref"].format(c=c)
    return np.load(REP / p)["qpos"].astype(np.float64)[1:]


def stats(body, q, ref, Pw, Dr, s):
    """body_52_test.py / g1_small_test.py's stats (the Task page's metrics), with the success threshold 0.5 m x the root scale s:
    [fall_s, success, mpjpe, mmpjpe, mpjae, drift (the body's metres)] (crossembodiment.metrics.MetricSuite.task_page)"""
    return MetricSuite.task_page(body, s).reference(ref, Pw, Dr).values(q)


def old_dir_of(b):
    return {"child": paths.output("child_52_rerun"), "g1": paths.output("g1_52")}.get(b, paths.output("body_52") / b)


def clip_job(job):
    import json
    b, c = job; body = body_of(b); ref = ref_of(b, c); s_ = CONF[b]["s"]; st = st_(c); G = CONF[b]["G"]
    Pw = world_positions(body.m, body.d, ref)
    _, Dr = body.part_floor(ref); f = lambda q: stats(body, q, ref, Pw, Dr, s_)
    old_dir = old_dir_of(b)
    ev = EvalRollouts(IN / b / "rollouts" / st, G, G[-1]); ix = ev.index; Q = ev.qpos
    cost = np.array(ev.cost); chk = Checks(("reroll_cost", "z0_vs_anchored"), ("bad",))
    R = dict(best=np.zeros((len(CONF[b]["seeds"]), len(G), len(KEYS))), centre=np.zeros((len(CONF[b]["seeds"]), len(G), len(KEYS))))
    for i, s in enumerate(CONF[b]["seeds"]):
        r, r_old = f"{st}_{LOES}_s{s}", f"{st}_{ANCHORED}_s{s}"
        new = json.load(open(IN / b / r / "summary.json")); old = json.load(open(old_dir / r_old / "summary.json"))
        skip = {"minutes", "best", "improvement_vs_origin_z", "cos_best_z0", "path_len_deg", "net_deg", "origin_z", "anchor_weight", "xml", "out", "summary", "elapsed_min", "best_from"}
        chk["bad"] += [(r, k, old.get(k), new.get(k)) for k in set(old) | set(new) if k not in skip and old.get(k) != new.get(k)]
        if new.get("anchor_weight") not in (0, 0.0):
            chk["bad"].append((r, "anchor_weight", new.get("anchor_weight")))
        chk.high("reroll_cost", ix.check_vs_curve(r, cost, read_best_so_far(IN / b / r)))   # every re-rolled generation
        bi = ix.best_budgets(r)                                                              # the best latent at each budget of G
        zn = np.load(IN / b / r / "origin_z.npz")["qpos"]; zo = np.load(old_dir / r_old / "origin_z.npz")["qpos"]
        chk.diff("z0_vs_anchored", zn, zo)
        ci = ix.centre(r)
        for j in range(len(G)):
            R["best"][i, j] = f(Q[bi[j]]); R["centre"][i, j] = f(Q[ci[j]])
    R["frozen"] = np.array(f(np.load(IN / b / f"{st}_{LOES}_s0/origin_z.npz")["qpos"]))
    if b == "child":                                                   # the anchored LOES, the same code (must give the Task page's numbers)
        oev = EvalRollouts(old_dir / "rollouts" / st, G, G[-1]); oix = oev.index; OQ = oev.qpos
        R["anchored_best"] = np.array([[f(OQ[oix.best_at(f"{st}_{ANCHORED}_s{s}", g)]) for g in G] for s in CONF[b]["seeds"]])
        R["anchored_centre"] = np.array([[f(OQ[oix.centre(f"{st}_{ANCHORED}_s{s}")[j]]) for j in range(len(G))] for s in CONF[b]["seeds"]])
    return b, st, R, chk


def analyse():
    import json
    from multiprocessing import Pool
    jobs = [(b, c) for b in BODIES for c in CLIPS if (IN / b / "rollouts" / st_(c) / "eval_qpos.npz").exists()]
    with Pool(workers(16)) as p:
        res = p.map(clip_job, jobs, chunksize=1)
    rep, L = {}, []
    for b in BODIES:
        o = [x for x in res if x[0] == b]
        if not o:
            continue
        chk = dict(reroll_cost=Checks.worst([x[3] for x in o], "reroll_cost"), z0_vs_anchored=Checks.worst([x[3] for x in o], "z0_vs_anchored"),
                   bad=[y for x in o for y in x[3]["bad"]][:10], motions=len(o))
        G = CONF[b]["G"]; T = ResultsTable(KEYS, [x[1] for x in o], G, CONF[b]["seeds"])
        for key in ("best", "centre", "frozen") + (("anchored_best", "anchored_centre") if b == "child" else ()):
            T.add(key, np.stack([x[2][key] for x in o]))
        mean = lambda key: T.by_budget(key)
        rep[b] = dict(checks=chk, noanchor=dict(best=mean("best"), centre=mean("centre")), frozen=dict(zip(KEYS, np.mean([x[2]["frozen"] for x in o], 0).tolist())))
        if b == "child":
            rep[b]["anchored"] = dict(best=mean("anchored_best"), centre=mean("anchored_centre"))
        else:
            old = json.load(open(paths.output("g1_52" if b == "g1" else f"body_52/{b}") / "results.json"))["mean"]
            rep[b]["anchored"] = {w: old[w][ANCHORED] for w in ("best", "centre")}; rep[b]["baselines"] = {w: {k: v for k, v in old[w].items() if k != ANCHORED} for w in ("best", "centre")}
        L.append(f"\n## {b} ({chk['motions']} motions)\nchecks: settings differing beyond anchor_weight: {len(chk['bad'])} {chk['bad'][:3]} | re-roll vs best cost so far: {chk['reroll_cost']:.2g} | z0 rollout vs the anchored runs': {chk['z0_vs_anchored']}")
        for n_ in sorted({1024, 16 * G[-1]}):
            for w in ("best", "centre"):
                rows = [("LOES, anchor 0.3 (Task page)", rep[b]["anchored"][w][str(n_)]), ("LOES, no anchor", rep[b]["noanchor"][w][str(n_)])]
                rows += [(k, v[str(n_)]) for k, v in rep[b].get("baselines", {}).get(w, {}).items() if str(n_) in v]
                L.append(f"\n{w} latent after {n_:,} rollouts")
                L += ResultsTable.markdown(rows, COLUMNS)
    write_json(RESD / "results.json", rep); ReportWriter("LOES without the anchor term", __doc__).write(RESD / "README.md", L)
    print("\n".join(L))


COLUMNS = (("Fall (s)", "fall_s", "{:.3f}"), ("Success", "success", "{:.1f}%", 100), ("MPJPE", "mpjpe", "{:.2f}"), ("MMPJPE", "mmpjpe", "{:.2f}"),
           ("MPJAE", "mpjae", "{:.2f}"), ("drift (m)", "drift", "{:.2f}"))


def main(argv):
    global BODIES
    BODIES = parse_flag(argv, "--bodies", ",".join(CONF)).split(",")
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[[x for x in argv[1:] if x in ("commands", "prepare", "evalcmds", "analyse")][0]]()


if __name__ == "__main__":
    main(sys.argv)
