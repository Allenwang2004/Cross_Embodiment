#!/usr/bin/env python3
"""objective_optimizer_test.py -- separate the optimizer from the objective for the single-latent search (user, 11/13, advisor's
checklist: "separate the contributions of the optimization algorithm and the objective function"). Child body, 10 of the 52 task
motions (every 5th line of outputs/child_52/clips.txt: lines 1, 6, ..., 46), seeds 0-2, 64 x 16 = 1,024 rollouts each.
New searches (every other setting as the paper's runs: reference velocity at the start, --obs-scale exact, --pairs 8):
  es_phc        our ES (sigma 0.05, lr 0.03) on BFM-Zero's objective, 1 - mean PHC tracking reward (--objective phc)
  es_ours_s0.5  our ES on our objective with a larger sampling step, sigma 0.5 (lr 0.03 unchanged; user 11/14: "if ES uses 0.5 it
                may also converge faster")
  cem_ours      BFM-Zero's cross-entropy search (sigma 0.05, 4 elites, floor sigma/5) on our objective (B term + 1.0 heading +
                0.1 position, no anchor)
  cem_phc_s0.1  BFM-Zero's cross-entropy search on its own objective with a larger starting step, sigma 0.1 (floor 0.02)
  cem_phc_s0.5  the same with sigma 0.5 (floor 0.1)
Reused, same motions and seeds (not re-run):
  es_ours       outputs/noanchor/child/<stem>_es_s<seed>      (the paper's method, no anchor)
  cem_phc       outputs/child_52_cem_phc/<stem>_cem_s<seed>  (BFM-Zero as in the paper, sigma 0.05)
  frozen        outputs/child_52_rerun/<stem>_es_s0/origin_z.npz
Evaluation: each run's best latent so far after 4, 8, 16, 32 and 64 generations (64 ... 1,024 rollouts; z_trace.npz z_best[g])
rolled out again with its own search command (--eval-z --eval-save-qpos, into rollouts_g/); every re-roll must reproduce the
search's own best cost so far. Metrics on q[t] against ref[t + 1]: PHC success (world-frame mean body error never above 0.5 m
adult-equivalent, scripts/success_thresholds.py score), PHC reward (scripts/phc_reward_tables.py reward_fn), MPJPE, MMPJPE, MPJAE
(scripts/child_52_means.py full). The reused runs must reproduce outputs/success_definitions/task_motions_noanchor.json ("global_")
run by run at 1,024 rollouts.
  commands | prepare | evalcmds | analyse        -> outputs/objective_optimizer_test/{results.json, README.md}
usage: uv run python scripts/objective_optimizer_test.py commands"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.results import write_json                            # noqa: E402
# OOP refactor (ported from the main tree, 11/13): the runs (new and reused) and the re-rolls are read under paths.outputs_root()
# ($CE_OUTPUTS_ROOT, default outputs/); prepare's z.npy and analyse's results.json / README.md are written under $CE_RESULTS_ROOT
# (paths.write_path, default outputs/); the command lines keep their repository-relative outputs/... folders. The clip list is
# configs/clips' copy; child_52_means.full is crossenbodiment.metrics.legacy_metrics (refactor step 4). Docstring unchanged (README.md).
OUT_REL = "outputs/objective_optimizer_test"; ROLL_REL = f"{OUT_REL}/rollouts_g"; ROLL = paths.read_path(ROLL_REL); SEEDS = (0, 1, 2); S_CHILD = 0.611036; G = (4, 8, 16, 32, 64)
XML = "assets/robots_torque/child/robot_torque_full.xml"; REF = "data/child/retargeting_motion/{c}.npz"
COMMON = f"--body child --xml {XML} --evals 1024 --pairs 8 --lr 0.03 --obs-scale exact --eval-every 0 --init-vel-scale 1"
PHC = "--objective phc"
OURS = "--objective bfm --heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0"
NEW = {"es_phc": f"--algo es --sigma 0.05 {PHC}", "es_ours_s0.5": f"--algo es --sigma 0.5 {OURS}", "cem_ours": f"--algo cem --sigma 0.05 {OURS}",
       "cem_phc_s0.1": f"--algo cem --sigma 0.1 {PHC}", "cem_phc_s0.5": f"--algo cem --sigma 0.5 {PHC}"}
OLD = {"es_ours": ("outputs/noanchor/child/{st}_es_s{s}", f"--algo es --sigma 0.05 {OURS} --anchor data/origin_z/{{c}}.npy"),
       "cem_phc": ("outputs/child_52_cem_phc/{st}_cem_s{s}", f"--algo cem --sigma 0.05 {PHC}")}
ORDER = ("frozen", "es_ours", "es_ours_s0.5", "es_phc", "cem_ours", "cem_phc", "cem_phc_s0.1", "cem_phc_s0.5")
SEARCH = ("es_ours", "cem_phc") + tuple(NEW)
KEYS = ("success", "reward", "mpjpe", "mmpjpe", "mpjae")


def clips():
    # configs/clips/child_52_task.txt = outputs/child_52/clips.txt: no blank lines, so the line numbers are the list's indices
    return [l for i, l in enumerate(paths.read_clips(paths.CHILD_52_TASK)) if i % 5 == 0][:10]


def run_rel(k, c, s):
    """the run folder, repository-relative (the command's --out)"""
    st = c.split("/")[1]
    return OLD[k][0].format(st=st, s=s) if k in OLD else f"{OUT_REL}/{st}_{k}_s{s}"


def run_dir(k, c, s):
    """the run folder to read (under the read root)"""
    return paths.read_path(run_rel(k, c, s))


def cmd(k, c):
    extra = OLD[k][1].format(c=c) if k in OLD else NEW[k]
    return f"uv run python scripts/single_z_search.py --clip {c} {COMMON} {extra}"


def commands():
    for c in clips():
        for k in NEW:
            for s in SEEDS:
                print(run_rel(k, c, s), f"{cmd(k, c)} --seed {s} --out {run_rel(k, c, s)}")


def prepare():
    for c in clips():
        st = c.split("/")[1]
        for k in SEARCH:
            d = paths.write_path(f"{ROLL_REL}/{st}/{k}"); d.mkdir(parents=True, exist_ok=True)
            Z = [np.load(run_dir(k, c, s) / "z_trace.npz")["z_best"] for s in SEEDS]
            assert all(len(z) == 65 for z in Z), (st, k)
            np.save(d / "z.npy", np.stack([z[g] for z in Z for g in G]))           # row = seed index * len(G) + budget index
    print("prepared")


def evalcmds():
    for c in clips():
        st = c.split("/")[1]
        for k in SEARCH:
            d = f"{ROLL_REL}/{st}/{k}"
            print(d, f"{cmd(k, c)} --seed 0 --eval-z {d}/z.npy --eval-save-qpos --out {d}")


_B = {}


def clip_job(c):
    from crossenbodiment import eval_metrics as E             # scripts/eval_metrics.py re-exports this module (refactor step 8)
    from crossenbodiment.metrics import legacy_metrics as full   # child_52_means.full (refactor step 4)
    from success_thresholds import world, score
    from phc_reward_tables import reward_fn
    if "b" not in _B:
        _B["b"] = E.Body(E.XML["child"], True)
    body = _B["b"]; st = c.split("/")[1]
    ref_all = np.load(REPO / REF.format(c=c))["qpos"].astype(np.float64); ref = ref_all[1:]; Pw = world(body, ref)
    rew = reward_fn(XML, ref_all, S_CHILD)

    def metrics(q):
        sc = score(body, q, Pw, S_CHILD, len(ref)); v = full(body, np.asarray(q, dtype=np.float64), ref, 1.0)
        return dict(success=sc["success"]["0.5"], reward=rew(q), **{k: float(v[k]) for k in ("mpjpe", "mmpjpe", "mpjae")})

    fz = metrics(np.load(paths.read_path(f"outputs/child_52_rerun/{st}_es_s0/origin_z.npz"))["qpos"])
    R = {"frozen": [[fz] * len(G)]}; worst = {}
    for k in SEARCH:
        d = ROLL / st / k; Q = np.load(d / "eval_qpos.npz")["qpos"]; cost = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))])
        R[k] = [[metrics(Q[si * len(G) + gi]) for gi in range(len(G))] for si in range(len(SEEDS))]
        best = []
        for s in SEEDS:
            curve = [float(r["best_so_far"]) for r in csv.DictReader(open(run_dir(k, c, s) / "curve.csv"))]; assert len(curve) == 64, (st, k, s)
            best += [curve[g - 1] for g in G]
        worst[k] = float(np.abs(cost[:len(best)] - np.array(best)).max())
    return st, R, worst


def analyse():
    cl = clips()
    with Pool(10) as p:
        res = p.map(clip_job, cl)
    ref = json.load(open(paths.read_path("outputs/success_definitions/task_motions_noanchor.json")))["per_clip"]
    mism = [(st, k, s) for st, R, _ in res for k, rk in (("es_ours", "es_best"), ("cem_phc", "cem_phc_best"))
            for s in SEEDS if R[k][s][-1]["success"] != ref[st][rk][s]["global_"]]
    mism += [(st, "frozen") for st, R, _ in res if R["frozen"][0][0]["success"] != ref[st]["frozen"]["global_"]]
    worst = max(v for _, _, w in res for v in w.values())
    # means over seeds, then motions, at every budget
    mean = {k: {str(16 * g): {m: float(np.mean([np.mean([x[gi][m] for x in R[k]]) for _, R, _ in res])) for m in KEYS}
                for gi, g in enumerate(G)} for k in ORDER}
    per = {st: {k: {m: float(np.mean([x[-1][m] for x in R[k]])) for m in KEYS} for k in ORDER} for st, R, _ in res}
    rep = dict(clips=cl, seeds=list(SEEDS), budgets=[16 * g for g in G], mean=mean, per_motion_1024=per,
               checks=dict(reused_runs_vs_success_definitions=mism, eval_vs_search_best_cost=worst))
    W = paths.write_path(OUT_REL); write_json(W / "results.json", rep)
    names = {"frozen": "Frozen (z0)", "es_ours": "ES + our objective, sigma 0.05 (paper)", "es_ours_s0.5": "ES + our objective, sigma 0.5",
             "es_phc": "ES + PHC reward", "cem_ours": "CEM + our objective", "cem_phc": "CEM + PHC reward, sigma 0.05 (BFM-Zero, paper)",
             "cem_phc_s0.1": "CEM + PHC reward, sigma 0.1", "cem_phc_s0.5": "CEM + PHC reward, sigma 0.5"}
    B = [str(16 * g) for g in G]; f = mean
    lines = ["# Optimizer vs objective, child, 10 motions x 3 seeds", "", "```", __doc__.strip(), "```", "",
             f"checks: reused runs vs success_definitions mismatches {len(mism)} {mism[:4]} | re-roll vs search best cost so far, largest "
             f"difference {worst:.2e}", "", "best latent after 1,024 rollouts; means over seeds, then motions (frozen: one rollout)", "",
             "| method | PHC success | PHC reward | MPJPE (cm) | MMPJPE (cm) | MPJAE (m/s^2) |", "|---|---|---|---|---|---|"]
    lines += [f"| {names[k]} | {100 * f[k]['1024']['success']:.1f}% | {f[k]['1024']['reward']:.3f} | {f[k]['1024']['mpjpe']:.2f} | "
              f"{f[k]['1024']['mmpjpe']:.2f} | {f[k]['1024']['mpjae']:.2f} |" for k in ORDER]
    for m, lab, fmt in (("success", "PHC success", lambda v: f"{100 * v:.1f}%"), ("reward", "PHC reward", lambda v: f"{v:.3f}"),
                        ("mpjpe", "MPJPE (cm)", lambda v: f"{v:.2f}")):
        lines += ["", f"{lab} of the best latent so far, by rollouts", "", "| method | " + " | ".join(B) + " |", "|---|" + "---|" * len(B)]
        lines += [f"| {names[k]} | " + " | ".join(fmt(f[k][b][m]) for b in B) + " |" for k in ORDER]
    lines += ["", "Motions: " + ", ".join(c.split("/")[1] for c in cl)]
    (W / "README.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines[8:]))


if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[sys.argv[1]]()
