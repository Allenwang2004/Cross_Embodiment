#!/usr/bin/env python3
"""child_52_rerun.py -- Comparison A searched again with the corrected alignment (scripts/single_z_search.py --frame-offset 1): LOES, random search, CEM and CMA-ES, 5 seeds, the 52 motions of outputs/child_52/clips.txt, everything else as in outputs/child_52 and outputs/child_52_cmaes. Frozen z0 is not searched: its rollout is the one every search starts from (origin_z.npz, the same z0 re-projected onto the sphere that the searches and the anchor use). outputs/child_52/<stem>_z0 skipped that projection (z differs by 2.4e-7), and chaos split 11 of the 52 motions.
  (run)    outputs/train_logs/queue/run_child_52_rerun.sh: searches in outputs/child_52_rerun/<stem>_<algo>_s<seed>
  prepare  writes outputs/child_52_rerun/rollouts/<stem>/{z.npy, index.json}: each run's distinct best z so far at every generation and its centre after 4, 8, 16, 32, 64 generations
  (run)    scripts/single_z_search.py --eval-z z.npy --eval-save-qpos on each motion
  analyse  metrics of each rollout with q[t] against ref[t + 1]; writes per_run.npz (best_<algo>: motions x seeds x 64 generations x metrics, mean_<algo>: motions x seeds x 5 x metrics, z0) and means.json. Checks: rollouts reproduce each search's best_so_far; settings match the old runs apart from the alignment; headstand_4 LOES matches outputs/loes_aligned.
usage: uv run python scripts/child_52_rerun.py prepare | analyse"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment import paths   # noqa: E402
from crossembodiment.experiment import workers   # noqa: E402
from crossembodiment.methods import experiment_keys   # noqa: E402
from crossembodiment.metrics import metric_body   # noqa: E402  one Body per (xml, smpl) and process (refactor step 3)
from crossembodiment.reroll import EvalRollouts, Reroller, read_best_so_far   # noqa: E402
from crossembodiment.results import ResultsTable, save_npz, write_json   # noqa: E402
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under paths.results_root()
# ($CE_RESULTS_ROOT); both REPO/outputs by default. prepare writes the schema-2 index (crossembodiment/reroll.py), analyse reads any.
# the run-directory keys here (crossembodiment/methods.py RUNS; also the per_run.npz / means.json keys): LOES with the anchor "es", CEM
# with OUR cost "cem" (cem_ours: Comparison A's, not BFM-Zero's), random search, CMA-ES
K = experiment_keys("outputs/child_52_rerun"); LOES, CEM, RANDOM, CMAES = (K[m] for m in ("loes_anchor", "cem_ours", "random", "cmaes"))
SRC = paths.output("child_52"); OLD_CMA = paths.output("child_52_cmaes"); OUT = paths.output("child_52_rerun"); ROLL = OUT / "rollouts"; ALGOS = (LOES, CEM, RANDOM, CMAES); SEEDS = range(5); G = (4, 8, 16, 32, 64)
RES = paths.result("child_52_rerun"); REROLL = Reroller(64, G, best="every")   # each run's distinct best per generation, centres at G
SAME = ("evals", "generations", "pairs", "sigma", "lr", "steps", "fitness", "sampling", "objective", "init_vel_scale", "heading_weight", "pos_weight", "pos_unit", "pos_div", "anchor_weight", "obs_scale", "init", "bfm_zero_vel", "fall_weight", "seed", "algo")
body = None


def clips():
    from crossembodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    return read_clips(CHILD_52_TASK)


def prepare():
    for c in clips():
        stem = c.split("/")[1]
        REROLL.prepare(RES / "rollouts" / stem, [(f"{stem}_{a}_s{s}", OUT / f"{stem}_{a}_s{s}") for a in ALGOS for s in SEEDS], clip=c)
    print("prepared")


def clip(c):
    global body
    import eval_metrics as E
    from crossembodiment.metrics import LEGACY_KEYS as KEYS, legacy_metrics as full   # child_52_means' (refactor step 4)
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]; d = ROLL / stem
    ev = EvalRollouts(d, G, 64); ix = ev.index; Q = ev.qpos.astype(np.float64); cost = ev.cost; base = ev.base; V = {}
    def m(i):
        if i not in V:
            v = full(body, Q[i], ref, cost[i] / base); V[i] = [v[k] for k in KEYS]
        return V[i]
    B = {a: np.zeros((5, 64, len(KEYS))) for a in ALGOS}; C = {a: np.zeros((5, len(G), len(KEYS))) for a in ALGOS}; worst = 0.0; bad = []
    for a in ALGOS:
        for s in SEEDS:
            r = f"{stem}_{a}_s{s}"; old = json.load(open((OLD_CMA if a == CMAES else SRC) / r / "summary.json")); new = json.load(open(OUT / r / "summary.json")); bad += [(r, k, old.get(k), new.get(k)) for k in SAME if old.get(k) != new.get(k)]
            if new.get("frame_offset") != 1:
                bad.append((r, "frame_offset", new.get("frame_offset")))
            for j, i in enumerate(ix.best_generations(r)):
                B[a][s, j] = m(i)
            for j, i in enumerate(ix.centre(r)):
                C[a][s, j] = m(i)
            worst = max(worst, ix.check_vs_curve(r, cost, read_best_so_far(OUT / r)))
    v = full(body, np.load(OUT / f"{stem}_{LOES}_s0/origin_z.npz")["qpos"].astype(np.float64), ref, 1.0)
    return stem, [v[k] for k in KEYS], B, C, worst, bad


def analyse():
    from crossembodiment.metrics import LEGACY_KEYS as KEYS   # child_52_means.KEYS (refactor step 4)
    cl = clips()
    with Pool(workers(24)) as p:
        res = p.map(clip, cl)
    T = ResultsTable(KEYS, [r[0] for r in res], G, SEEDS); z0 = T.add("z0", np.array([r[1] for r in res]))
    B = {a: T.add(f"best_{a}", np.stack([r[2][a] for r in res])) for a in ALGOS}; C = {a: T.add(f"mean_{a}", np.stack([r[3][a] for r in res])) for a in ALGOS}; bad = [x for r in res for x in r[5]]
    same = max(float(np.abs(np.load(OUT / f"headstand_4_{LOES}_s{s}/best_z.npy") - np.load(paths.output(f"loes_aligned/headstand_4_s{s}/best_z.npy"))).max()) for s in SEEDS)
    print("largest rollout vs search best_so_far:", max(r[4] for r in res), "| settings differing from the old runs:", len(bad), bad[:6], "| headstand_4 LOES against outputs/loes_aligned, largest z difference:", same, flush=True)
    save_npz(RES / "per_run.npz", clips=np.array(cl), keys=np.array(KEYS), gens_mean_z=np.array(G), z0=z0, **{f"best_{a}": B[a] for a in ALGOS}, **{f"mean_{a}": C[a] for a in ALGOS})
    m = lambda A: ResultsTable.pooled(A, KEYS)                          # every motion x seed at once
    test = lambda X, Y: ResultsTable.paired(X, Y, KEYS)                 # per-motion means over seeds, paired Wilcoxon
    rep = dict(z0=m(z0), best={a: {str(g * 16): m(B[a][:, :, g - 1]) for g in G} for a in ALGOS}, centre={a: {str(g * 16): m(C[a][:, :, j]) for j, g in enumerate(G)} for a in ALGOS},
               loes_vs=dict(best={a: test(B[LOES][:, :, -1], B[a][:, :, -1]) for a in ALGOS[1:]}, centre={a: test(C[LOES][:, :, -1], C[a][:, :, -1]) for a in ALGOS[1:]}))
    write_json(RES / "means.json", rep); print("written")


if __name__ == "__main__":
    {"prepare": prepare, "analyse": analyse}[sys.argv[1]]()
