#!/usr/bin/env python3
"""loes_weight_test.py -- would smaller heading / position weights also lower LOES's MPJPE, as x 0.05 did for LSOR
(outputs/lsor_mpjae_test)? Small sample: the 6 child task motions where LOES's MPJPE trails LSO (BFMTrack) most at 1,024 rollouts
(outputs/loes_weight_test/clips.txt), seeds 0-2, LOES as in outputs/child_52_rerun with only the two weights scaled; anchor 0.3 kept.
  w0     heading 0, position 0
  w0.05  heading 0.05, position 0.005
  w1     heading 1.0, position 0.1: the existing runs (outputs/child_52_rerun, best latent after 64 generations)
  commands  "<out dir>|<command>" per search, into outputs/loes_weight_test/<stem>_<cond>_s<seed>
  prepare   outputs/loes_weight_test/rollouts/<stem>/{z.npy, index.json}: each new run's best latent after 64 generations
  evalcmds  one --eval-z command per motion
  analyse   MPJPE, MMPJPE, MPJAE (eval_metrics.tracking_metrics), PHC success (world-frame mean body error never above 0.31 m on the
            child), root drift (mean root xy distance, body metres); checks that the rollouts reproduce each search's best cost.
            Writes outputs/loes_weight_test/results.json.
usage: uv run python scripts/loes_weight_test.py commands | prepare | evalcmds | analyse"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment.bodies import BODIES
from crossembodiment.paths import CHILD_XML_REL
from crossembodiment.metrics import MetricSuite, metric_body, world_positions
from crossembodiment import paths
from crossembodiment.experiment import ClipSet, Experiment, workers
from crossembodiment.reroll import Reroller
from crossembodiment.results import write_json
# refactor step 8: runs (and clips.txt) are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under
# paths.results_root() ($CE_RESULTS_ROOT); both REPO/outputs by default. prepare writes the schema-2 index (crossembodiment/reroll.py)
OUT = paths.output("loes_weight_test"); ROLL = OUT / "rollouts"; RERUN = paths.output("child_52_rerun"); S = BODIES.scale("child")
RES = paths.result("loes_weight_test")
XML = CHILD_XML_REL; SEEDS = range(3)
NEW = {"w0": (0.0, 0.0), "w0.05": (0.05, 0.005)}; COND = ("w0", "w0.05", "w1")
REROLL = Reroller(64, best="last", centre=False)                       # each new run's best latent after 64 generations
body = None


def clips():
    return list(ClipSet.from_file(OUT / "clips.txt"))


def experiment():
    return Experiment("outputs/loes_weight_test", clips=clips(), methods=tuple(NEW), seeds=SEEDS, order=("seed", "method", "clip"))


def base_cmd(c, h, p):
    return (f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {XML} --objective bfm --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 "
            f"--obs-scale exact --heading-weight {h} --pos-weight {p} --anchor data/origin_z/{c}.npy --anchor-weight 0.3 --eval-every 0 --init-vel-scale 1")


def commands():
    experiment().print_commands(lambda r: f"{base_cmd(r.clip, *NEW[r.method])} --seed {r.seed} --out {r.dir}")


def prepare():
    for c in clips():
        stem = c.split("/")[1]
        REROLL.prepare(RES / "rollouts" / stem, [(f"{stem}_{k}_s{s}", OUT / f"{stem}_{k}_s{s}") for k in NEW for s in SEEDS], clip=c)
    print("prepared")


def evalcmds():
    E = experiment()
    for c in clips():
        d = E.rollouts_dir(c.split('/')[1]); print(E.eval_line(d, base_cmd(c, 0.0, 0.0)))


def world(Q):
    """world positions of every body of the module's `body` (crossembodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def stats(q, ref, Pw):
    """{mpjpe, mmpjpe, mpjae, success (PHC, 0.5 m x S), drift (m)} of the module's `body` (crossembodiment.metrics.MetricSuite)"""
    return MetricSuite(body, ("mpjpe", "mmpjpe", "mpjae", "success", "drift"), S).reference(ref, Pw)(q)


def clip(c):
    global body
    import eval_metrics as E
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]; Pw = world(ref)
    d = ROLL / stem; ix = Reroller.read(d, (64,), 64); Q = np.load(d / "eval_qpos.npz")["qpos"]; res = {}; worst = 0.0
    for k in NEW:
        res[k] = []
        for s in SEEDS:
            r = f"{stem}_{k}_s{s}"; i = ix.best_last(r); res[k].append(stats(Q[i], ref, Pw))
            b = np.load(OUT / r / "best.npz")                      # the search's own rollout of its best latent
            if "qpos" in b.files:
                worst = max(worst, float(np.abs(b["qpos"][: len(Q[i])] - Q[i][: len(b["qpos"])]).max()))
    ox = Reroller.read(RERUN / f"rollouts/{stem}"); OQ = np.load(RERUN / f"rollouts/{stem}/eval_qpos.npz")["qpos"]
    res["w1"] = [stats(OQ[ox.best_last(f"{stem}_es_s{s}")], ref, Pw) for s in SEEDS]
    return stem, res, worst


def analyse():
    cl = clips()
    with Pool(workers(len(cl))) as p:
        out = p.map(clip, cl)
    res = {o[0]: o[1] for o in out}; keys = ("mpjpe", "mmpjpe", "mpjae", "success", "drift")
    print("best rollout re-rolled vs the search's saved best rollout, largest qpos difference:", max(o[2] for o in out))
    mean = {k: {m: float(np.mean([r[m] for st in res for r in res[st][k]])) for m in keys} for k in COND}
    print(f"{len(cl)} motions x {len(SEEDS)} seeds, LOES best latent after 1,024 rollouts")
    print(f"{'weights':8s}" + "".join(f"{m:>9s}" for m in keys))
    for k in COND:
        print(f"{k:8s}" + "".join(f"{mean[k][m] * (100 if m == 'success' else 1):9.2f}" for m in keys))
    for st in res:
        print(f"  {st:30s}" + " | ".join(f"{k}: MPJPE {np.mean([r['mpjpe'] for r in res[st][k]]):5.2f} drift {np.mean([r['drift'] for r in res[st][k]]):4.2f} "
                                          f"succ {sum(r['success'] for r in res[st][k]):.0f}/3" for k in COND))
    write_json(RES / "results.json", dict(per_motion=res, mean=mean))


if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[sys.argv[1]]()
