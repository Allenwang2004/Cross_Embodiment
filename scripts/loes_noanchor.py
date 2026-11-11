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
from crossenbodiment.bodies import BODIES
from crossenbodiment.paths import CHILD_XML_REL
from crossenbodiment.metrics import MetricSuite, metric_body, world_positions
from crossenbodiment.paths import CHILD_52_TASK, read_clips   # noqa: E402  configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
CLIPS = read_clips(CHILD_52_TASK)
ORDER = sorted(CLIPS, key=lambda c: -len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]))


def child_cmd(c, s):
    return (f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {CHILD_XML_REL} --objective bfm "
            f"--evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy "
            f"--anchor-weight 0.3 --eval-every 0 --init-vel-scale 1 --seed {s}")


def noanchor(cmd):
    assert cmd.count("--anchor-weight 0.3") == 1, cmd
    return cmd.replace("--anchor-weight 0.3", "--anchor-weight 0")


def commands():
    sys.argv = [sys.argv[0], "--set", "52"]                             # g1_small_test reads its set from argv at import
    import body_52_test as B
    import g1_small_test as G1
    st = lambda c: c.split("/")[1]
    for s in range(5):
        for c in ORDER:
            d = f"{OUT}/child/{st(c)}_es_s{s}"; print(f"{d}|{noanchor(child_cmd(c, s))} --out {d}")
    for c in ORDER:
        for b in ("payload", "limit_leg"):
            d = f"{OUT}/{b}/{st(c)}_es_s0"; print(f"{d}|{noanchor(B.zcmd(b, c, 'es'))} --seed 0 --out {d}")
        d = f"{OUT}/g1/{st(c)}_es_s0"; print(f"{d}|{noanchor(G1.zcmd(c, 'es'))} --seed 0 --out {d}")


REP = Path(__file__).resolve().parent.parent
CONF = dict(child=dict(seeds=range(5), G=(4, 8, 16, 32, 64), s=BODIES.scale("child")), payload=dict(seeds=(0,), G=(4, 8, 16, 32, 64, 128), s=BODIES.scale("payload")),
            limit_leg=dict(seeds=(0,), G=(4, 8, 16, 32, 64, 128), s=BODIES.scale("limit_leg")), g1=dict(seeds=(0,), G=(4, 8, 16, 32, 64, 128), s=BODIES.scale("g1")))
KEYS = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "drift")
BODIES = sys.argv[sys.argv.index("--bodies") + 1].split(",") if "--bodies" in sys.argv else list(CONF)


def st_(c):
    return c.split("/")[1]


def done(b, c):
    return all((REP / OUT / b / f"{st_(c)}_es_s{s}" / "summary.json").exists() for s in CONF[b]["seeds"])


def prepare():
    import json
    n = 0
    for b in BODIES:
        for c in CLIPS:
            if not done(b, c):
                continue
            d = REP / OUT / b / "rollouts" / st_(c); d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
            for s in CONF[b]["seeds"]:
                r = f"{st_(c)}_es_s{s}"; t = np.load(REP / OUT / b / r / "z_trace.npz"); zb, zm = t["z_best"], t["z_mean"]; G = CONF[b]["G"]; assert len(zb) == G[-1] + 1, (b, r)
                if b == "child":                                       # every generation, as child_52_rerun.py's prepare
                    best = []
                    for g in range(1, G[-1] + 1):
                        if not best or np.abs(zb[g] - zb[g - 1]).max() > 0:
                            Z.append(zb[g]); best.append(len(Z) - 1)
                        else:
                            best.append(best[-1])
                    idx[r] = dict(best=best)
                else:
                    idx[r] = dict(best=list(range(len(Z), len(Z) + len(G)))); Z += [zb[g] for g in G]
                idx[r]["centre"] = list(range(len(Z), len(Z) + len(G))); Z += [zm[g] for g in G]
            np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(body=b, clip=c, map=idx))); n += 1
    print("prepared", n)


def evalcmds():
    sys.argv = [sys.argv[0], "--set", "52"]
    import body_52_test as B
    import g1_small_test as G1
    for b in BODIES:
        for c in CLIPS:
            d = f"{OUT}/{b}/rollouts/{st_(c)}"
            if not (REP / d / "z.npy").exists():
                continue
            base = {"child": lambda: child_cmd(c, 0), "g1": lambda: G1.zcmd(c, "es")}.get(b, lambda: B.zcmd(b, c, "es"))()
            print(f"{d}|{noanchor(base)}" + ("" if b == "child" else " --seed 0") + f" --eval-z {d}/z.npy --eval-save-qpos --out {d}")


_B = {}


def body_of(b):
    import eval_metrics as E
    if b not in _B:
        sys.argv = [sys.argv[0], "--set", "52"]
        import body_52_test as Bt
        _B[b] = (metric_body(E.XML["child"], True) if b == "child" else metric_body(E.XML["g1"], False) if b == "g1" else metric_body(str(REP / Bt.BODIES[b]["xml"]), True))
    return _B[b]


def ref_of(b, c):
    sys.argv = [sys.argv[0], "--set", "52"]
    import body_52_test as Bt
    p = {"child": f"data/child/retargeting_motion/{c}.npz", "g1": f"outputs/g1/roundtrip/{st_(c)}/g1.npz"}.get(b) or Bt.BODIES[b]["ref"].format(c=c)
    return np.load(REP / p)["qpos"].astype(np.float64)[1:]


def stats(body, q, ref, Pw, Dr, s):
    """body_52_test.py / g1_small_test.py's stats (the Task page's metrics), with the success threshold 0.5 m x the root scale s:
    [fall_s, success, mpjpe, mmpjpe, mpjae, drift (the body's metres)] (crossenbodiment.metrics.MetricSuite.task_page)"""
    return MetricSuite.task_page(body, s).reference(ref, Pw, Dr).values(q)


def clip_job(job):
    import csv, json
    b, c = job; body = body_of(b); ref = ref_of(b, c); s_ = CONF[b]["s"]; st = st_(c); G = CONF[b]["G"]
    Pw = world_positions(body.m, body.d, ref)
    _, Dr = body.part_floor(ref); f = lambda q: stats(body, q, ref, Pw, Dr, s_)
    old_dir = {"child": REP / "outputs/child_52_rerun", "g1": REP / "outputs/g1_52"}.get(b, REP / "outputs/body_52" / b)
    d = REP / OUT / b / "rollouts" / st; ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"]
    cost = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]); chk = dict(reroll_cost=0.0, z0_vs_anchored=0.0, bad=[])
    R = dict(best=np.zeros((len(CONF[b]["seeds"]), len(G), len(KEYS))), centre=np.zeros((len(CONF[b]["seeds"]), len(G), len(KEYS))))
    for i, s in enumerate(CONF[b]["seeds"]):
        r = f"{st}_es_s{s}"; new = json.load(open(REP / OUT / b / r / "summary.json")); old = json.load(open(old_dir / r / "summary.json"))
        skip = {"minutes", "best", "improvement_vs_origin_z", "cos_best_z0", "path_len_deg", "net_deg", "origin_z", "anchor_weight", "xml", "out", "summary", "elapsed_min", "best_from"}
        chk["bad"] += [(r, k, old.get(k), new.get(k)) for k in set(old) | set(new) if k not in skip and old.get(k) != new.get(k)]
        if new.get("anchor_weight") not in (0, 0.0):
            chk["bad"].append((r, "anchor_weight", new.get("anchor_weight")))
        bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(REP / OUT / b / r / "curve.csv"))])
        bi = [ix[r]["best"][g - 1] for g in G] if b == "child" else ix[r]["best"]              # the best latent at each budget of G
        gens = list(range(1, G[-1] + 1)) if b == "child" else list(G)
        chk["reroll_cost"] = max(chk["reroll_cost"], float(np.abs(cost[ix[r]["best"]] - bs[[g - 1 for g in gens]]).max()))
        zn = np.load(REP / OUT / b / r / "origin_z.npz")["qpos"]; zo = np.load(old_dir / r / "origin_z.npz")["qpos"]
        chk["z0_vs_anchored"] = max(chk["z0_vs_anchored"], float(np.abs(zn - zo).max()))
        for j in range(len(G)):
            R["best"][i, j] = f(Q[bi[j]]); R["centre"][i, j] = f(Q[ix[r]["centre"][j]])
    R["frozen"] = np.array(f(np.load(REP / OUT / b / f"{st}_es_s0/origin_z.npz")["qpos"]))
    if b == "child":                                                   # the anchored LOES, the same code (must give the Task page's numbers)
        oix = json.load(open(old_dir / "rollouts" / st / "index.json"))["map"]; OQ = np.load(old_dir / "rollouts" / st / "eval_qpos.npz")["qpos"]
        R["anchored_best"] = np.array([[f(OQ[oix[f"{st}_es_s{s}"]["best"][g - 1]]) for g in G] for s in CONF[b]["seeds"]])
        R["anchored_centre"] = np.array([[f(OQ[oix[f"{st}_es_s{s}"]["centre"][j]]) for j in range(len(G))] for s in CONF[b]["seeds"]])
    return b, st, R, chk


def analyse():
    import json
    from multiprocessing import Pool
    jobs = [(b, c) for b in BODIES for c in CLIPS if (REP / OUT / b / "rollouts" / st_(c) / "eval_qpos.npz").exists()]
    with Pool(16) as p:
        res = p.map(clip_job, jobs, chunksize=1)
    rep, L = {}, []
    for b in BODIES:
        o = [x for x in res if x[0] == b]
        if not o:
            continue
        chk = dict(reroll_cost=max(x[3]["reroll_cost"] for x in o), z0_vs_anchored=max(x[3]["z0_vs_anchored"] for x in o), bad=[y for x in o for y in x[3]["bad"]][:10], motions=len(o))
        G = CONF[b]["G"]; mean = lambda key: {str(16 * g): dict(zip(KEYS, np.mean([x[2][key][:, j].mean(0) for x in o], 0).tolist())) for j, g in enumerate(G)}
        rep[b] = dict(checks=chk, noanchor=dict(best=mean("best"), centre=mean("centre")), frozen=dict(zip(KEYS, np.mean([x[2]["frozen"] for x in o], 0).tolist())))
        if b == "child":
            rep[b]["anchored"] = dict(best=mean("anchored_best"), centre=mean("anchored_centre"))
        else:
            old = json.load(open(REP / ("outputs/g1_52" if b == "g1" else f"outputs/body_52/{b}") / "results.json"))["mean"]
            rep[b]["anchored"] = {w: old[w]["es"] for w in ("best", "centre")}; rep[b]["baselines"] = {w: {k: v for k, v in old[w].items() if k != "es"} for w in ("best", "centre")}
        L.append(f"\n## {b} ({chk['motions']} motions)\nchecks: settings differing beyond anchor_weight: {len(chk['bad'])} {chk['bad'][:3]} | re-roll vs best cost so far: {chk['reroll_cost']:.2g} | z0 rollout vs the anchored runs': {chk['z0_vs_anchored']}")
        for n_ in sorted({1024, 16 * G[-1]}):
            for w in ("best", "centre"):
                L.append(f"\n{w} latent after {n_:,} rollouts\n| method | Fall (s) | Success | MPJPE | MMPJPE | MPJAE | drift (m) |\n|---|---|---|---|---|---|---|")
                rows = [("LOES, anchor 0.3 (Task page)", rep[b]["anchored"][w][str(n_)]), ("LOES, no anchor", rep[b]["noanchor"][w][str(n_)])]
                rows += [(k, v[str(n_)]) for k, v in rep[b].get("baselines", {}).get(w, {}).items() if str(n_) in v]
                for name, v in rows:
                    L.append(f"| {name} | {v['fall_s']:.3f} | {100 * v['success']:.1f}% | {v['mpjpe']:.2f} | {v['mmpjpe']:.2f} | {v['mpjae']:.2f} | {v['drift']:.2f} |")
    (REP / OUT / "results.json").write_text(json.dumps(rep, indent=1)); (REP / OUT / "README.md").write_text("# LOES without the anchor term\n\n" + __doc__ + "\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[[x for x in sys.argv[1:] if x in ("commands", "prepare", "evalcmds", "analyse")][0]]()
