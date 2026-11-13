#!/usr/bin/env python3
"""child_52_lso.py -- the latent-sequence methods on the 52 child task motions (outputs/child_52/clips.txt), to compare
with LOES and the single-latent baselines (outputs/child_52_rerun, outputs/child_52_cem_phc):
  native tracking   Motivo's ER L = 8 sequence, the zero-shot rollout every scripts/lso_search.py run makes first (q_ER8)
  lso               LSO (BFMTrack): BFMTrack's values and its own cost (the BFM term alone)
  lsor              LSO with centred ranks per step (--rank), BFMTrack's cost
  lsor_ours         LSOR with our cost: BFM term + 1.0 x heading + 0.1 x position (adult-equivalent metres), per step; no anchor
  lsor_w005         LSOR with our cost x 0.05: 0.05 x heading + 0.005 x position (the user's choice of 11/08, outputs/lsor_mpjae_test)
  (lso_ours, our cost without ranks, was stopped by the user at 24 of 260 runs on 11/08; not analysed)
                    (the anchor pulls a single latent towards z0; a latent sequence starts from B(g_t+1), not z0)
  ta                Trajectory Adaptation (BFM-Zero): --method ta --objective phc (PHC's tracking reward)
5 seeds, 64 iterations x 16 rollouts (+ one logged rollout of the mean, never the result), the reference velocity at the
start, the state after step t scored against reference frame t + 1.
  commands  "<out dir> <command>" per search, into outputs/child_52_lso/<stem>_<method>_s<seed>
  analyse   metrics (scripts/eval_metrics.py via crossembodiment.metrics.legacy_metrics, child_52_means.full until step 4) of the saved rollouts: the best sequence so far
            (q_best_it<k>) and the mean sequence (q_mean_it<k>) after 4, 8, 16, 32, 64 iterations, and native tracking.
            Writes per_run.npz and means.json (by budget; LOES, frozen z0 and native tracking against each method).
            Checks: settings; 1,024 sample rollouts; the result is never the logged mean; the re-rolled best matches the
            saved best rollout; native tracking is identical across every run of a motion.
usage: uv run python scripts/child_52_lso.py commands | analyse"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment import paths   # noqa: E402
from crossembodiment.experiment import ClipSet, Experiment, workers   # noqa: E402
from crossembodiment.methods import experiment_keys, key_in   # noqa: E402
from crossembodiment.metrics import metric_body   # noqa: E402  one Body per (xml, smpl) and process (refactor step 3)
from crossembodiment.paths import CHILD_52_TASK   # noqa: E402
from crossembodiment.results import Checks, ResultsTable, save_npz, write_json   # noqa: E402
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under paths.results_root()
# ($CE_RESULTS_ROOT); both REPO/outputs by default
SRC = paths.output("child_52"); OUT = paths.output("child_52_lso"); RERUN = paths.output("child_52_rerun"); RES = paths.result("child_52_lso")
# the run-directory keys here (crossembodiment/methods.py RUNS; also the per_run.npz / means.json keys): "lsor" is LSO + ranks with
# BFMTrack's own cost (lsor_bfm) in this experiment, the 0.05 / 0.005 variant is "lsor_w005"; RERUN_LOES: child_52_rerun's LOES ("es")
K = experiment_keys("outputs/child_52_lso"); RERUN_LOES = key_in("outputs/child_52_rerun", "loes_anchor")
LSO, LSOR_BFM, TA, LSOR_OURS, LSOR_W005 = (K[m] for m in ("lso", "lsor_bfm", "ta", "lsor_ours", "lsor_w005"))
METHODS = {LSO: "", LSOR_BFM: "--rank", TA: "--method ta --objective phc", LSOR_OURS: "--rank --heading-weight 1.0 --pos-weight 0.1", LSOR_W005: "--rank --heading-weight 0.05 --pos-weight 0.005"}; SEEDS = range(5); G = (4, 8, 16, 32, 64)
OWN = dict(heading_weight=0.0, pos_weight=0.0); OURS = dict(heading_weight=1.0, pos_weight=0.1, pos_unit="adult"); OURS005 = dict(heading_weight=0.05, pos_weight=0.005, pos_unit="adult")     # BFMTrack's own cost; ours without the anchor
WANT = {LSO: dict(method="lso", objective="bfm", rank=False, **OWN), LSOR_BFM: dict(method="lso", objective="bfm", rank=True, **OWN),
        TA: dict(method="ta", objective="phc", rank=False, **OWN), LSOR_OURS: dict(method="lso", objective="bfm", rank=True, **OURS),
        LSOR_W005: dict(method="lso", objective="bfm", rank=True, **OURS005)}
COMMON = dict(n=16, iters=64, init_vel_scale=1.0, sample_rollouts=1024, antithetic=False, sigma=0.0125)
EXP = Experiment("outputs/child_52_lso", clips=ClipSet.named(CHILD_52_TASK), methods=METHODS, seeds=SEEDS, budgets=G,
                 order=("clip", "method", "seed"), sep=" ")
body = None


def clips():
    from crossembodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    return read_clips(CHILD_52_TASK)


def commands():
    EXP.print_commands(lambda r: f"uv run python scripts/lso_search.py --ref data/child/retargeting_motion/{r.clip}.npz --body child --n 16 --iters 64 "
                                 f"--init-vel-scale 1 --trace budgets --seed {r.seed} {METHODS[r.method]} --out {r.dir}".replace("  ", " "))


def clip(c):
    global body
    import eval_metrics as E
    from crossembodiment.metrics import LEGACY_KEYS as KEYS, legacy_metrics as full   # child_52_means' (refactor step 4)
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]
    ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    met = lambda q, ratio: [full(body, q.astype(np.float64), ref, ratio)[k] for k in KEYS]
    nat = np.load(OUT / f"{stem}_{LSO}_s0/rollouts.npz")["q_ER8"]; native = met(nat, 1.0); bad, nat_diff, reroll = [], 0.0, 0.0
    B = {k: np.zeros((5, len(G), len(KEYS))) for k in METHODS}; C = {k: np.zeros((5, len(G), len(KEYS))) for k in METHODS}
    for k in METHODS:
        for s in SEEDS:
            d = OUT / f"{stem}_{k}_s{s}"; S = json.load(open(d / "summary.json")); R = np.load(d / "rollouts.npz")
            bad += [(d.name, x, S.get(x)) for x in Checks.settings({**COMMON, **WANT[k]}, S)]
            if S.get("best_src") == "mean" or list(S.get("budgets", [])) != list(G):
                bad.append((d.name, "best_src / budgets", S.get("best_src"), S.get("budgets")))
            nat_diff = max(nat_diff, float(np.abs(R["q_ER8"] - nat).max())); reroll = max(reroll, float(np.abs(R["q_LSO_best"] - R["q_best_it64"]).max()))
            start = S["results"]["B(g_t) (LSO start)" if k != TA else "ER L=8 (Motivo zero-shot)"]["cost"]
            for j, g in enumerate(G):
                B[k][s, j] = met(R[f"q_best_it{g}"], S["best_cost_at"][str(g)] / start); C[k][s, j] = met(R[f"q_mean_it{g}"], np.nan)
    return stem, native, B, C, bad, nat_diff, reroll


def analyse():
    from crossembodiment.metrics import LEGACY_KEYS as KEYS   # child_52_means.KEYS (refactor step 4)
    cl = clips()
    with Pool(workers(24)) as p:
        res = p.map(clip, cl)
    T = ResultsTable(KEYS, [r[0] for r in res], G, SEEDS); nat = T.add("native", np.array([r[1] for r in res]))
    B = {k: T.add(f"best_{k}", np.stack([r[2][k] for r in res])) for k in METHODS}; C = {k: T.add(f"mean_{k}", np.stack([r[3][k] for r in res])) for k in METHODS}
    bad = [x for r in res for x in r[4]]
    print("runs:", len(cl) * len(METHODS) * len(SEEDS), "| settings not as intended:", len(bad), bad[:6], "| native tracking across runs, largest difference:",
          max(r[5] for r in res), "| re-rolled best vs saved best rollout, largest difference:", max(r[6] for r in res), flush=True)
    old = np.load(RERUN / "per_run.npz"); assert list(old["clips"]) == cl and list(old["keys"]) == list(KEYS)
    save_npz(RES / "per_run.npz", clips=np.array(cl), keys=np.array(KEYS), gens=np.array(G), native=nat,
             **{f"best_{k}": B[k] for k in METHODS}, **{f"mean_{k}": C[k] for k in METHODS})
    m = lambda A: ResultsTable.pooled(A, KEYS, nan=True)                # every motion x seed at once (nanmean: ratio of centres)
    def test(X, Y):                       # X, Y: motions x seeds x metrics (or motions x metrics) -> per-motion means, paired
        return ResultsTable.paired(X, Y, KEYS, per_metric=False)
    rep = dict(native=m(nat), z0=m(old["z0"]), best={k: {str(g * 16): m(B[k][:, :, j]) for j, g in enumerate(G)} for k in METHODS},
               centre={k: {str(g * 16): m(C[k][:, :, j]) for j, g in enumerate(G)} for k in METHODS},
               loes_best_vs={k: test(old[f"best_{RERUN_LOES}"][:, :, -1], B[k][:, :, -1]) for k in METHODS},
               loes_centre_vs={k: test(old[f"mean_{RERUN_LOES}"][:, :, -1], C[k][:, :, -1]) for k in METHODS},
               best_vs_native={k: test(B[k][:, :, -1], nat) for k in METHODS})
    write_json(RES / "means.json", rep); print("written")


if __name__ == "__main__":
    {"commands": commands, "analyse": analyse}[sys.argv[1]]()
