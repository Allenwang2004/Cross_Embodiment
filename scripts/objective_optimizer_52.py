#!/usr/bin/env python3
"""objective_optimizer_52.py -- the optimizer-vs-objective test of scripts/objective_optimizer_test.py on all 52 task motions, seeds 0-4,
child and tall body (user, 11/14: confirm the 10-motion result before changing the paper's claims). New searches, 64 x 16 = 1,024
rollouts each, every other setting as the paper's runs on that body:
  es_phc        our ES (sigma 0.05, lr 0.03) on BFM-Zero's objective, 1 - mean PHC tracking reward (--objective phc)
  cem_phc_s0.5  BFM-Zero's cross-entropy search on its own objective, starting step sigma 0.5 (floor 0.1)
Reused (not re-run): es_ours, the paper's method (child outputs/noanchor/child, tall outputs/tall_kp06_52 *_es_*); cem_phc, the paper's
BFM-Zero, sigma 0.05 (child outputs/child_52_cem_phc, tall outputs/tall_kp06_52 *_cem_*); frozen z0 (the es runs' origin_z.npz).
The child's 10 x 3 runs of outputs/objective_optimizer_test are reused for their motions and seeds (same command).
Evaluation: each run's best latent after 1,024 rollouts (z_trace.npz z_best[64]) re-rolled with its own search command
(--eval-z --eval-save-qpos; must reproduce the search's best cost). Metrics, q[t] against ref[t + 1]: PHC success (world-frame mean body
error never above 0.5 m x the root scale s), PHC reward, MPJPE, MMPJPE, MPJAE in the body's cm and m/s^2 and adult-equivalent (/ s).
Reused runs must reproduce the stored success (child outputs/success_definitions/task_motions_noanchor.json "global_", tall
outputs/tall_kp06_52/results.json) run by run.
  commands | prepare | evalcmds | analyse      -> outputs/objective_optimizer_52/{results.json, README.md}
usage: uv run python scripts/objective_optimizer_52.py commands"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment import paths                                         # noqa: E402
from crossembodiment.results import write_json                            # noqa: E402
# OOP refactor (ported from the main tree, 11/13): the runs (new and reused, outputs/objective_optimizer_test's too) and the re-rolls
# are read under paths.outputs_root() ($CE_OUTPUTS_ROOT, default outputs/); prepare's z.npy and analyse's results.json / README.md
# are written under $CE_RESULTS_ROOT (paths.write_path, default outputs/); the command lines keep their repository-relative
# outputs/... folders. The clip list is configs/clips' copy; child_52_means.full is crossembodiment.metrics.legacy_metrics (refactor
# step 4). Docstring unchanged (README.md copies it).
OUT_REL = "outputs/objective_optimizer_52"; SEEDS = range(5); TEST_REL = "outputs/objective_optimizer_test"
BODY = dict(
    child=dict(flag="child", xml="assets/robots_torque/child/robot_torque_full.xml", ref="data/child/retargeting_motion/{c}.npz", s=0.611036,
               es_ours="outputs/noanchor/child/{st}_es_s{s}", cem_phc="outputs/child_52_cem_phc/{st}_cem_s{s}",
               frozen="outputs/child_52_rerun/{st}_es_s0/origin_z.npz"),
    tall=dict(flag="len1.4_wid0.8", xml="assets/robots_torque/len1.4_wid0.8_kp0.6_kd1.0/robot_torque_full.xml",
              ref="data/len1.4_wid0.8/retargeting_motion/{c}.npz", s=1.364408905613045,
              es_ours="outputs/tall_kp06_52/{st}_es_s{s}", cem_phc="outputs/tall_kp06_52/{st}_cem_s{s}",
              frozen="outputs/tall_kp06_52/{st}_es_s0/origin_z.npz"))
PHC = "--objective phc"; OURS = "--objective bfm --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy --anchor-weight 0"
ARGS = {"es_ours": f"--algo es --sigma 0.05 {OURS}", "cem_phc": f"--algo cem --sigma 0.05 {PHC}",
        "es_phc": f"--algo es --sigma 0.05 {PHC}", "cem_phc_s0.5": f"--algo cem --sigma 0.5 {PHC}"}
NEW = ("es_phc", "cem_phc_s0.5"); ALL = ("es_ours", "es_phc", "cem_phc", "cem_phc_s0.5"); ORDER = ("frozen",) + ALL
KEYS = ("success", "reward", "mpjpe", "mmpjpe", "mpjae")


def clips():
    return paths.read_clips(paths.CHILD_52_TASK)          # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)


def run_rel(b, k, c, s):
    """the run folder, repository-relative (the command's --out): the reused runs, the child's 10 x 3 of objective_optimizer_test
    where that run has finished (under the read root), else this test's own"""
    st = c.split("/")[1]
    if k not in NEW:
        return BODY[b][k].format(st=st, s=s)
    t = f"{TEST_REL}/{st}_{k}_s{s}"
    if b == "child" and (paths.read_path(t) / "summary.json").exists():
        return t
    return f"{OUT_REL}/{b}/{st}_{k}_s{s}"


def run_dir(b, k, c, s):
    """the run folder to read (under the read root)"""
    return paths.read_path(run_rel(b, k, c, s))


def cmd(b, k, c):
    B = BODY[b]
    return (f"uv run python scripts/single_z_search.py --clip {c} --body {B['flag']} --xml {B['xml']} --evals 1024 --pairs 8 --lr 0.03 "
            f"--obs-scale exact --eval-every 0 --init-vel-scale 1 {ARGS[k].format(c=c)}")


def commands():
    for s in SEEDS:                       # seed-major: every motion and body gets seed 0 first
        for c in clips():
            for b in BODY:
                for k in NEW:
                    d = run_rel(b, k, c, s)
                    if not d.startswith(TEST_REL):
                        print(d, f"{cmd(b, k, c)} --seed {s} --out {d}")


def prepare():
    for b in BODY:
        for c in clips():
            st = c.split("/")[1]
            for k in ALL:
                d = paths.write_path(f"{OUT_REL}/{b}/rollouts/{st}/{k}"); d.mkdir(parents=True, exist_ok=True)
                Z = [np.load(run_dir(b, k, c, s) / "z_trace.npz")["z_best"] for s in SEEDS]
                assert all(len(z) == 65 for z in Z), (b, st, k)
                np.save(d / "z.npy", np.stack([z[64] for z in Z]))
    print("prepared")


def evalcmds():
    for b in BODY:
        for c in clips():
            for k in ALL:
                d = f"{OUT_REL}/{b}/rollouts/{c.split('/')[1]}/{k}"
                print(d, f"{cmd(b, k, c)} --seed 0 --eval-z {d}/z.npy --eval-save-qpos --out {d}")


_B = {}


def job(arg):
    b, c = arg
    from crossembodiment import eval_metrics as E             # scripts/eval_metrics.py re-exports this module (refactor step 8)
    from crossembodiment.metrics import legacy_metrics as full   # child_52_means.full (refactor step 4)
    from success_thresholds import world, score
    from phc_reward_tables import reward_fn
    B = BODY[b]
    if b not in _B:
        _B[b] = E.Body(B["xml"], True)
    body = _B[b]; st = c.split("/")[1]; S = B["s"]
    ref_all = np.load(REPO / B["ref"].format(c=c))["qpos"].astype(np.float64); ref = ref_all[1:]; Pw = world(body, ref); rew = reward_fn(B["xml"], ref_all, S)

    def metrics(q):
        q = np.asarray(q, dtype=np.float64); sc = score(body, q, Pw, S, len(ref)); v = full(body, q, ref, 1.0)
        return dict(success=sc["success"]["0.5"], reward=rew(q), mpjpe=float(v["mpjpe"]), mmpjpe=float(v["mmpjpe"]), mpjae=float(v["mpjae"]))

    R = {"frozen": [metrics(np.load(paths.read_path(B["frozen"].format(st=st)))["qpos"])]}; worst = 0.0
    for k in ALL:
        d = paths.read_path(f"{OUT_REL}/{b}/rollouts/{st}/{k}"); Q = np.load(d / "eval_qpos.npz")["qpos"]; cost = [float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]
        R[k] = [metrics(Q[i]) for i in range(len(SEEDS))]
        best = [float(list(csv.DictReader(open(run_dir(b, k, c, s) / "curve.csv")))[-1]["best_so_far"]) for s in SEEDS]
        worst = max(worst, float(np.abs(np.array(cost[:len(SEEDS)]) - np.array(best)).max()))
    return b, st, R, worst


def analyse():
    from scipy.stats import wilcoxon
    cl = clips(); st_all = [c.split("/")[1] for c in cl]
    with Pool(24) as p:
        res = p.map(job, [(b, c) for b in BODY for c in cl])
    R = {b: {} for b in BODY}; worst = 0.0
    for b, st, r, w in res:
        R[b][st] = r; worst = max(worst, w)
    # reused runs must reproduce the stored success
    sd = json.load(open(paths.read_path("outputs/success_definitions/task_motions_noanchor.json")))["per_clip"]
    tl = json.load(open(paths.read_path("outputs/tall_kp06_52/results.json")))["per_motion"]
    mism = [("child", st, k, s) for st in st_all for k, rk in (("es_ours", "es_best"), ("cem_phc", "cem_phc_best")) for s in SEEDS
            if R["child"][st][k][s]["success"] != sd[st][rk][s]["global_"]]
    mism += [("tall", st, k, s) for st in st_all for k, tk in (("es_ours", "es"), ("cem_phc", "cem")) for s in SEEDS
             if R["tall"][st][k][s]["success"] != tl[st][tk]["best"][s][4][1]]
    # per body: means over seeds, then motions; distance metrics also adult-equivalent (/ s)
    seedmean = lambda b, st, k, m: float(np.mean([x[m] for x in R[b][st][k]]))
    per = {b: {k: {m: float(np.mean([seedmean(b, st, k, m) for st in st_all])) for m in KEYS} for k in ORDER} for b in BODY}
    for b in BODY:
        for k in ORDER:
            for m in ("mpjpe", "mmpjpe", "mpjae"):
                per[b][k][m + "_ae"] = per[b][k][m] / BODY[b]["s"]
    avg = {k: {m: float(np.mean([per[b][k][m] for b in BODY])) for m in ("success", "reward", "mpjpe_ae", "mmpjpe_ae", "mpjae_ae")} for k in ORDER}
    # success pooled from counts (both bodies have 52 x 5 runs per method, 52 x 1 for frozen, so the plain mean equals the pooled rate)

    def test(b, a, c, m):
        x = np.array([seedmean(b, st, a, m) for st in st_all]); y = np.array([seedmean(b, st, c, m) for st in st_all]); d = x - y
        good = d > 0 if m in ("success", "reward") else d < 0; bad = d < 0 if m in ("success", "reward") else d > 0
        try:
            p = float(wilcoxon(x, y).pvalue)
        except ValueError:
            p = float("nan")
        return dict(diff=float(d.mean()), better=int(good.sum()), worse=int(bad.sum()), p=p)
    pairs = (("es_ours", "es_phc"), ("es_ours", "cem_phc_s0.5"), ("es_ours", "cem_phc"), ("es_phc", "cem_phc"), ("cem_phc_s0.5", "cem_phc"))
    tests = {b: {f"{a} vs {c}": {m: test(b, a, c, m) for m in ("success", "reward", "mpjpe")} for a, c in pairs} for b in BODY}
    rep = dict(clips=cl, seeds=list(SEEDS), per_body=per, average=avg, tests=tests,
               checks=dict(reused_vs_stored_success_mismatches=mism, eval_vs_search_best_cost=worst))
    W = paths.write_path(OUT_REL); write_json(W / "results.json", rep)
    names = {"frozen": "Frozen (z0)", "es_ours": "ES + our objective (paper)", "es_phc": "ES + PHC reward",
             "cem_phc": "CEM + PHC reward, sigma 0.05 (BFM-Zero, paper)", "cem_phc_s0.5": "CEM + PHC reward, sigma 0.5"}
    L = ["# Optimizer vs objective on the 52 task motions, child and tall, 5 seeds", "", "```", __doc__.strip(), "```", "",
         f"checks: reused runs vs stored success, mismatches {len(mism)} {mism[:4]} | re-roll vs search best cost, largest difference {worst:.2e}", ""]
    for b in BODY:
        L += [f"## {b} (s = {BODY[b]['s']:.4f}); best latent after 1,024 rollouts, means over seeds then motions; distances in the body's own units", "",
              "| method | PHC success | PHC reward | MPJPE (cm) | MMPJPE (cm) | MPJAE (m/s^2) |", "|---|---|---|---|---|---|"]
        L += [f"| {names[k]} | {100 * per[b][k]['success']:.1f}% | {per[b][k]['reward']:.3f} | {per[b][k]['mpjpe']:.2f} | {per[b][k]['mmpjpe']:.2f} | "
              f"{per[b][k]['mpjae']:.2f} |" for k in ORDER]
        L += ["", "paired over the 52 motions (seed means; Wilcoxon signed-rank): diff = first - second; better/worse = motions where the first wins/loses", "",
              "| pair | success | PHC reward | MPJPE |", "|---|---|---|---|"]
        L += [f"| {pr} | " + " | ".join(f"{t[m]['diff']:+.3f} ({t[m]['better']}/{t[m]['worse']}, p={t[m]['p']:.2g})" for m in ("success", "reward", "mpjpe")) + " |"
              for pr, t in tests[b].items()]
        L += [""]
    L += ["## Child and tall averaged (distances adult-equivalent: / s)", "",
          "| method | PHC success | PHC reward | MPJPE (cm) | MMPJPE (cm) | MPJAE (m/s^2) |", "|---|---|---|---|---|---|"]
    L += [f"| {names[k]} | {100 * avg[k]['success']:.1f}% | {avg[k]['reward']:.3f} | {avg[k]['mpjpe_ae']:.2f} | {avg[k]['mmpjpe_ae']:.2f} | "
          f"{avg[k]['mpjae_ae']:.2f} |" for k in ORDER]
    (W / "README.md").write_text("\n".join(L) + "\n"); print("\n".join(L[8:]))


if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[sys.argv[1]]()
