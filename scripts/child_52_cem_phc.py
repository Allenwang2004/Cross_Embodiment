#!/usr/bin/env python3
"""child_52_cem_phc.py -- Comparison A's CEM (BFM-Zero) searched again with BFM-Zero's tracking reward
(scripts/single_z_search.py --objective phc: 1 - mean_t PHC tracking reward, model/phc_reward.py) instead of our cost:
no heading, position or anchor term; every other setting as outputs/child_52_rerun's CEM (52 motions of
outputs/child_52/clips.txt, 5 seeds, 64 x 16 rollouts, sigma 0.05, 4 elites, std floor 0.01, the reference velocity at the
start, q[t] scored against ref[t + 1]). The old CEM runs stay in outputs/child_52_rerun.
  commands       "<out dir> <command>" per search, into outputs/child_52_cem_phc/<stem>_cem_s<seed>
  prepare        outputs/child_52_cem_phc/rollouts/<stem>/{z.npy, index.json}: each run's distinct best z so far at every
                 generation and its centre after 4, 8, 16, 32, 64 generations (as child_52_rerun.py)
  eval_commands  "<out dir> <command>" per motion: those latents rolled out with --eval-z --eval-save-qpos
  analyse        metrics (q[t] against ref[t + 1]) -> per_run.npz (best_cem, mean_cem: motions x seeds x gens x metrics, z0)
                 and means.json (best / centre by budget; LOES and the old CEM of child_52_rerun against this one).
                 Checks: settings match the old CEM runs apart from the cost; rollouts reproduce each search's best_so_far;
                 the z0 rollout equals child_52_rerun's.
usage: uv run python scripts/child_52_cem_phc.py commands | prepare | eval_commands | analyse"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment import paths   # noqa: E402
from crossenbodiment.experiment import ClipSet, Experiment, workers   # noqa: E402
from crossenbodiment.methods import key_in   # noqa: E402
from crossenbodiment.metrics import metric_body   # noqa: E402  one Body per (xml, smpl) and process (refactor step 3)
from crossenbodiment.paths import CHILD_52_TASK   # noqa: E402
from crossenbodiment.reroll import EvalRollouts, Reroller, read_best_so_far   # noqa: E402
from crossenbodiment.results import Checks, ResultsTable, save_npz, write_json   # noqa: E402
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under paths.results_root()
# ($CE_RESULTS_ROOT); both REPO/outputs by default. prepare writes the schema-2 index (crossenbodiment/reroll.py), analyse reads any.
SRC = paths.output("child_52"); OLD = paths.output("child_52_rerun"); OUT = paths.output("child_52_cem_phc"); ROLL = OUT / "rollouts"
# run-directory keys (crossenbodiment/methods.py RUNS): BFM-Zero's CEM here is "cem"; in OLD (child_52_rerun), "cem" is CEM with our cost
# (its settings are compared, SAME) and "es" LOES with the anchor (whose z0 rollout is compared)
CEM = key_in("outputs/child_52_cem_phc", "cem_phc")
OLD_CEM, OLD_LOES = key_in("outputs/child_52_rerun", "cem_ours"), key_in("outputs/child_52_rerun", "loes_anchor")
RES = paths.result("child_52_cem_phc")
SEEDS = range(5); G = (4, 8, 16, 32, 64); XML = "assets/robots_torque/child/robot_torque_full.xml"
SAME = ("evals", "generations", "pairs", "sigma", "lr", "steps", "fitness", "sampling", "init_vel_scale", "pos_unit", "pos_div",
        "obs_scale", "init", "bfm_zero_vel", "fall_weight", "seed", "algo", "frame_offset")
WANT = dict(objective="phc", heading_weight=0.0, pos_weight=0.0, anchor_weight=0.0)
body = None


def clips():
    from crossenbodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    return read_clips(CHILD_52_TASK)


EXP = Experiment("outputs/child_52_cem_phc", clips=ClipSet.named(CHILD_52_TASK), methods=(CEM,), seeds=SEEDS, budgets=G,
                 order=("clip", "seed", "method"), sep=" ")
REROLL = Reroller(64, G, best="every")


def base_cmd(c):
    return (f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {XML} --objective phc --algo cem --evals 1024 "
            f"--pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --eval-every 0 --init-vel-scale 1")


def commands():
    EXP.print_commands(lambda r: f"{base_cmd(r.clip)} --seed {r.seed} --out {r.dir}")


def prepare():
    for c in clips():
        stem = c.split("/")[1]
        REROLL.prepare(RES / "rollouts" / stem, [(f"{stem}_{CEM}_s{s}", OUT / f"{stem}_{CEM}_s{s}") for s in SEEDS], clip=c)
    print("prepared")


def eval_commands():
    for c in clips():
        d = EXP.rollouts_dir(c.split('/')[1]); print(EXP.eval_line(d, base_cmd(c)))


def clip(c):
    global body
    import eval_metrics as E
    from crossenbodiment.metrics import LEGACY_KEYS as KEYS, legacy_metrics as full   # child_52_means' (refactor step 4)
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]; d = ROLL / stem
    ev = EvalRollouts(d, G, 64); ix = ev.index; Q = ev.qpos.astype(np.float64); cost = ev.cost
    base = ev.base; V = {}
    def m(i):
        if i not in V:
            v = full(body, Q[i], ref, cost[i] / base); V[i] = [v[k] for k in KEYS]
        return V[i]
    B = np.zeros((5, 64, len(KEYS))); C = np.zeros((5, len(G), len(KEYS))); worst = 0.0; bad = []
    for s in SEEDS:
        r = f"{stem}_{CEM}_s{s}"; old = json.load(open(OLD / f"{stem}_{OLD_CEM}_s{s}" / "summary.json")); new = json.load(open(OUT / r / "summary.json"))
        bad += [(r, k, old.get(k), new.get(k)) for k in SAME if old.get(k) != new.get(k)] + [(r, k, new.get(k)) for k in Checks.settings(WANT, new)]
        for j, i in enumerate(ix.best_generations(r)):
            B[s, j] = m(i)
        for j, i in enumerate(ix.centre(r)):
            C[s, j] = m(i)
        worst = max(worst, ix.check_vs_curve(r, cost, read_best_so_far(OUT / r)))
    z0q = np.load(OUT / f"{stem}_{CEM}_s0/origin_z.npz")["qpos"]; same_z0 = float(np.abs(z0q - np.load(OLD / f"{stem}_{OLD_LOES}_s0/origin_z.npz")["qpos"]).max())
    return stem, B, C, worst, bad, same_z0


def analyse():
    from crossenbodiment.metrics import LEGACY_KEYS as KEYS   # child_52_means.KEYS (refactor step 4)
    cl = clips()
    with Pool(workers(24)) as p:
        res = p.map(clip, cl)
    T = ResultsTable(KEYS, [r[0] for r in res], G, SEEDS)
    B = T.add("best_cem", np.stack([r[1] for r in res])); C = T.add("mean_cem", np.stack([r[2] for r in res])); bad = [x for r in res for x in r[4]]
    print("largest rollout vs search best_so_far:", max(r[3] for r in res), "| settings not as intended:", len(bad), bad[:6],
          "| z0 rollout against child_52_rerun's, largest difference:", max(r[5] for r in res), flush=True)
    old = np.load(OLD / "per_run.npz"); assert list(old["clips"]) == cl and list(old["keys"]) == list(KEYS)
    save_npz(RES / "per_run.npz", clips=np.array(cl), keys=np.array(KEYS), gens_mean_z=np.array(G), z0=old["z0"], best_cem=B, mean_cem=C)
    m = lambda A: ResultsTable.pooled(A, KEYS)                          # every motion x seed at once
    test = lambda X, Y: ResultsTable.paired(X, Y, KEYS)                 # per-motion means over seeds, paired Wilcoxon
    rep = dict(z0=m(old["z0"]), best={str(g * 16): m(B[:, :, g - 1]) for g in G}, centre={str(g * 16): m(C[:, :, j]) for j, g in enumerate(G)},
               loes_vs_cem_phc=dict(best=test(old[f"best_{OLD_LOES}"][:, :, -1], B[:, :, -1]), centre=test(old[f"mean_{OLD_LOES}"][:, :, -1], C[:, :, -1])),
               cem_phc_vs_cem_ours=dict(best=test(B[:, :, -1], old[f"best_{OLD_CEM}"][:, :, -1]), centre=test(C[:, :, -1], old[f"mean_{OLD_CEM}"][:, :, -1])))
    write_json(RES / "means.json", rep); print("written")


if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "eval_commands": eval_commands, "analyse": analyse}[sys.argv[1]]()
