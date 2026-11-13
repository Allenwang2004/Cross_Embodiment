#!/usr/bin/env python3
"""amass_baselines.py -- the latent-sequence methods on AMASS motions (user, 11/11: the "AMASS motions" rows of the baseline comparison;
AMASS clips have no task latent z0, so only the latent-sequence methods run). Clips: the 68 ACCAD clips with locomotion of at least
5 s (outputs/amass_render/locomotion_clips_5s, scripts/accad_locomotion.py; 5.1-53 s, 615 s in all). Bodies: the child and the tall,
thin body of the new teaser (len1.4_wid0.8). Seed 0. Methods, as on the task motions (scripts/child_52_lso.py, scripts/body_52_test.py):
  native   Motivo's own tracking (ER, L = 8), the zero-shot rollout every lso_search.py run makes first (q_ER8)
  lso      LSO (BFMTrack): its own cost (the BFM term alone), no ranks
  lsor     LSOR: LSO with centred ranks per step and 0.05 x heading + 0.005 x position (lsor_w005 of the task motions)
  ta       Trajectory Adaptation (BFM-Zero): --method ta --objective phc
  16 rollouts per iteration, the reference's velocity at the start; 64 iterations (1,024 rollouts) on both bodies (user, 11/11: the
  tall body 1,024 too, as its task motions in outputs/tall_body_52); the best sequence so far and the mean after each budget kept.
Root scale s (ExactObs's observation mapping, the adult-equivalent metres of the position term and of the metrics): the child keeps
ExactObs's measured 0.611036; the tall body passes --scale explicitly (lso_search.py's default is the child's value for ANY body),
read off the clip as scripts/single_z_search.py does for non-child bodies: the median ratio of the reference's root x, y to the adult
clip's (constant per body to 1e-16; about 1.364 here).
References: child amass_data/amass_retargeting_motion/ACCAD/<clip>.npz; tall body data/len1.4_wid0.8/amass_retargeting_motion/ACCAD
(scripts/qpos_retarget.py from amass_data/amass_origin_motion with the body's skeleton; the same call reproduces the child's files exactly).
Metrics per rollout, q[t] against ref[t + 1] (as the Task page): MPJPE, MMPJPE (cm), MPJAE (m/s^2, acceleration error), Fall Time (s, non-foot contact while
the reference's is clear), PHC success (world-frame mean body error never above 0.5 m adult-equivalent), drift (m, adult-equivalent).
  commands | analyse      -> outputs/amass_baselines/<body>/<clip>_<method>_s0, outputs/amass_baselines/{results.json, README.md}"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.bodies import BODIES
from crossenbodiment.paths import AMASS_LOCOMOTION_5S, CHILD_XML_REL, read_clips
from crossenbodiment.metrics import MetricSuite, metric_body, world_positions
from crossenbodiment import paths
from crossenbodiment.experiment import ClipSet, Experiment, workers
from crossenbodiment.methods import experiment_keys
from crossenbodiment.results import ReportWriter, ResultsTable, write_json
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under paths.results_root()
# ($CE_RESULTS_ROOT); both REPO/outputs by default
OUT = paths.output("amass_baselines"); RES = paths.result("amass_baselines"); CLIPS = read_clips(AMASS_LOCOMOTION_5S)   # = outputs/amass_baselines/clips.txt (refactor step 4)
BODY = dict(child=dict(ref="amass_data/amass_retargeting_motion/ACCAD/{c}.npz", iters=64, G=(4, 8, 16, 32, 64), xml=CHILD_XML_REL),
            tall=dict(label="len1.4_wid0.8", ref="data/len1.4_wid0.8/amass_retargeting_motion/ACCAD/{c}.npz", iters=64, G=(4, 8, 16, 32, 64),
                      xml="assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml"))
# the run-directory keys (crossenbodiment/methods.py RUNS "outputs/amass_baselines/<body>"; also results.json's keys): LSOR here is the
# 0.05 heading / 0.005 position variant (lsor_w005) under the key "lsor"
K = experiment_keys("outputs/amass_baselines/<body>"); LSO, LSOR, TA = K["lso"], K["lsor_w005"], K["ta"]
METHODS = {LSO: "", LSOR: "--rank --heading-weight 0.05 --pos-weight 0.005", TA: "--method ta --objective phc"}
KEYS = ("mpjpe", "mmpjpe", "mpjae", "fall_s", "success", "drift")
EXP = Experiment("outputs/amass_baselines", clips=ClipSet(CLIPS), methods=tuple(METHODS), bodies=tuple(BODY), name="{body}/{stem}_{method}_s{seed}",
                 order=("body", "clip", "method"))

BFM_DOC = """--set bfm (user, 11/12-13): the same 68 clips, bodies, methods, references, root scales and seed with BFMTrack's own LSO settings
(paper Sec. 5.1: N = 128 parallel rollouts per iteration x 24 iterations = 3,072 rollouts; Adam lr 0.00625, betas (0.8, 0.99), sigma
0.0125, gamma 0.97, pink noise) for LSO, LSOR and TA alike, the best sequence so far and the mean kept after 1/2/4/8/16/24 iterations
(128 ... 3,072 rollouts; 1,024 = 8 iterations, the earlier run's budget). Runs: outputs/amass_baselines_bfm/<body>/<clip>_<method>_s0
(commands.txt, run_queue.sh). Per rollout, besides the metrics above:
  phc      PHC reward: scripts/phc_reward_tables.reward_fn = 1 - model/phc_reward.PHCCost(model, ref[0], ref[1:], scale=s)(q[None], ref[0]),
           the mean over steps of PHC's per-step reward (0-1, higher is better), s = 0.611036 (child) / scale() (tall body)
  within   the share of frames within the success threshold: world-frame mean body error <= 0.5 m adult-equivalent (0.5 x s)
Checks: the settings in every summary.json (n = 128, iters = 24, budgets, the method's flags, 3,072 sample rollouts; pos_div = s); native
q_ER8 identical in a clip's three runs; the TA runs (PHC objective): PHC reward of q_best_it<k> = 1 - best_cost_at[k]; for reference,
q_ER8 against the earlier run's (outputs/amass_baselines).
  analyse --set bfm [--bodies child] [--suffix _child]  -> outputs/amass_baselines_bfm/{results.json, README.md}"""
# run sets other than the default (outputs/amass_baselines, 16 x 64); use() switches OUT (RES, EXP), N and every body's iters and G.
# The bfm runs use the same run-directory keys as outputs/amass_baselines/<body> (K above); they are not in crossenbodiment/methods.py
# RUNS, whose sequence settings are the 16-rollout ones (COMMON n = 16)
SETS = dict(bfm=dict(out="outputs/amass_baselines_bfm", n=128, iters=24, G=(1, 2, 4, 8, 16, 24)))
SET, N = None, 16
KEYS_BFM = KEYS + ("phc", "within")


def use(name):
    global OUT, RES, EXP, SET, N
    if name is None:
        return
    S = SETS[name]; rel = S["out"][len("outputs/"):]; OUT, RES, SET, N = paths.output(rel), paths.result(rel), name, S["n"]
    EXP = Experiment(S["out"], clips=EXP.clips, methods=EXP.methods, bodies=EXP.bodies, name=EXP.name, order=EXP.order)
    for B in BODY.values():
        B.update(iters=S["iters"], G=S["G"])


def scale(b, c):
    """the body's root scale s for clip c: ExactObs's child default, or single_z_search.py's rule (median ref / adult root x, y)
    against amass_data/amass_origin_motion/ACCAD/<c>.npz -- crossenbodiment/bodies.py BodyRegistry.scale (the same expression)"""
    assert BODIES.reference(BODY[b].get("label", b), c, amass=True) == BODY[b]["ref"].format(c=c)
    return BODIES.scale(BODY[b].get("label", b), c, amass=True)


def commands():
    order = EXP.clips.longest_first(lambda c: BODY["child"]["ref"].format(c=c))       # by the child's clip length

    def cmd(r):
        b, c, k, d = r.body, r.clip, r.method, r.dir; B = BODY[b]
        sc = "" if b == "child" else f"--scale {scale(b, c)!r} "
        return " ".join((f"uv run python scripts/lso_search.py --ref {B['ref'].format(c=c)} --body {B.get('label', b)} --n {N} --iters {B['iters']} "
                         f"--init-vel-scale 1 --trace budgets --budgets {' '.join(map(str, B['G']))} --seed 0 {METHODS[k]} {sc}--out {d}").split())
    EXP.print_commands(cmd, clip=order)


_ctx = {}


def one(job):
    from model.exact_obs import ExactObs, ADULT_XML
    b, c = job; B = BODY[b]
    if _ctx.get("b") != b:
        _ctx.update(b=b, body=metric_body(B["xml"], True))
    body, s = _ctx["body"], scale(b, c); m, d = body.m, body.d
    ref = np.load(REPO / B["ref"].format(c=c))["qpos"].astype(np.float64)[1:]
    Pr = world_positions(m, d, ref); _, Dr = body.part_floor(ref)
    # [mpjpe, mmpjpe, mpjae, fall_s, success (PHC, 0.5 m x s), drift (adult-equivalent m)] per rollout (crossenbodiment.metrics)
    suite = MetricSuite(body, ("mpjpe", "mmpjpe", "mpjae", "fall_s", "success", "drift"), s, drift_unit="adult").reference(ref, Pr, Dr)
    if SET is not None:                                      # PHC reward exactly as phc_reward_tables.py: its reward_fn on the full reference
        from phc_reward_tables import reward_fn
        rew = reward_fn(B["xml"], np.load(REPO / B["ref"].format(c=c))["qpos"].astype(np.float64), s)

    def stats(q):
        x = suite.rollout(q); v = [f(x) for _, f in suite.metrics]
        # a run set: + PHC reward (of the rollout as stored) and the share of frames within the success threshold (the same world error)
        return v if SET is None else v + [rew(q), float((x.world_error <= 0.5 * s).mean())]
    R, chk = {}, dict(native_diff=0.0, bad=[])
    if SET is not None:
        chk.update(ta_budget=0.0, pos_div=0.0, native_vs_16x64=None); iphc = KEYS_BFM.index("phc")
    nat = None
    for k in METHODS:
        run = OUT / b / f"{c}_{k}_s0"; S = json.load(open(run / "summary.json")); z = np.load(run / "rollouts.npz")
        want = dict(n=N, iters=B["iters"], seed=0, init_vel_scale=1.0, antithetic=False, sigma=0.0125, beta=1.0,
                    **{LSO: dict(method="lso", objective="bfm", rank=False, heading_weight=0.0, pos_weight=0.0),
                       LSOR: dict(method="lso", objective="bfm", rank=True, heading_weight=0.05, pos_weight=0.005),
                       TA: dict(method="ta", objective="phc", rank=False, heading_weight=0.0, pos_weight=0.0)}[k])
        if SET is not None:
            want.update(budgets=list(B["G"]), sample_rollouts=N * B["iters"])
            chk["pos_div"] = max(chk["pos_div"], abs(S["pos_div"] - s))
        chk["bad"] += [(b, c, k, x, S.get(x), v) for x, v in want.items() if S.get(x) != v]
        nat = z["q_ER8"] if nat is None else nat; chk["native_diff"] = max(chk["native_diff"], float(np.abs(z["q_ER8"] - nat).max()))
        R[k] = dict(best=np.array([stats(z[f"q_best_it{g}"]) for g in B["G"]]), centre=np.array([stats(z[f"q_mean_it{g}"]) for g in B["G"]]))
        if SET is not None and k == TA:                      # TA optimised the PHC cost: its own best_cost_at must come back
            chk["ta_budget"] = max(abs(R[k]["best"][j, iphc] - (1 - S["best_cost_at"][str(g)])) for j, g in enumerate(B["G"]))
    R["native"] = np.array(stats(nat))
    if SET is not None:                                      # for reference: the earlier run's zero-shot rollout (16 x 64)
        old = paths.output("amass_baselines") / b / f"{c}_{LSO}_s0/rollouts.npz"
        if old.exists():
            q = np.load(old)["q_ER8"]; chk["native_vs_16x64"] = float(np.abs(q - nat).max()) if q.shape == nat.shape else float("inf")
    return b, c, R, chk


def analyse(bodies=None, n_clips=None, dest=None, suffix=""):
    """bodies (default all), the first n_clips clips (default all), dest (default RES), results<suffix>.json, README<suffix>.md"""
    bodies = bodies or list(BODY); dest = paths.writable(Path(dest) if dest else RES)
    jobs = [(b, c) for b in BODY if b in bodies for c in CLIPS[:n_clips] if all((OUT / b / f"{c}_{k}_s0/rollouts.npz").exists() for k in METHODS)]
    with Pool(workers(16)) as p:
        res = p.map(one, jobs, chunksize=1)
    if SET is not None:
        return analyse_set(res, bodies, dest, suffix)
    rep, L = {}, []
    names = {"native": "Native tracking", LSO: "LSO (BFMTrack)", LSOR: "LSOR (ours)", TA: "Trajectory Adaptation (BFM-Zero)"}
    for b, B in BODY.items():
        o = [x for x in res if x[0] == b]
        if not o:
            continue
        chk = dict(motions=len(o), native_diff=max(x[3]["native_diff"] for x in o), bad=[y for x in o for y in x[3]["bad"]][:10])
        T = ResultsTable(KEYS, [x[1] for x in o], B["G"], (0,)); T.add("native", np.stack([x[2]["native"] for x in o]))
        for k in METHODS:
            for w in ("best", "centre"):
                T.add(f"{k}_{w}", np.stack([x[2][k][w][None] for x in o]))      # one seed: (motions, 1, budgets, metrics)
        mean = {"native": T.mean("native", 0)}
        for k in METHODS:
            for w in ("best", "centre"):
                mean[f"{k}_{w}"] = T.by_budget(f"{k}_{w}")
        rep[b] = dict(checks=chk, mean=mean, per_motion={x[1]: {k: (v.tolist() if isinstance(v, np.ndarray) else {w: a.tolist() for w, a in v.items()}) for k, v in x[2].items()} for x in o})
        n = str(16 * B["G"][-1]); L.append(f"\n## {b} ({len(o)} AMASS motions, seed 0), the best sequence after {int(n):,} rollouts\nchecks: {json.dumps(chk)}")
        L += ResultsTable.markdown([(names[k], mean["native"] if k == "native" else mean[f"{k}_best"][n]) for k in ("native", LSO, LSOR, TA)], COLUMNS)
    write_json(dest / f"results{suffix}.json", rep); ReportWriter("Latent-sequence methods on AMASS motions", __doc__).write(dest / f"README{suffix}.md", L)
    print("\n".join(L))


NAMES = {"native": "Native (BFM tracking)", LSO: "LSO (BFMTrack)", TA: "Trajectory Adaptation (BFM-Zero)", LSOR: "LSOR (our method)"}   # the run set's README
ORDER = ("native", LSO, TA, LSOR)


def analyse_set(res, bodies, dest, suffix):
    """a run set's means, checks, results<suffix>.json and README<suffix>.md (PHC reward and frames within the threshold added)"""
    rep, L = {}, []
    old_m = json.load(open(paths.read_path("outputs/amass_baselines/results.json")))
    old_p = json.load(open(paths.read_path("outputs/phc_reward_tables/results.json")))
    for b, B in BODY.items():
        o = [x for x in res if x[0] == b]
        if b not in bodies or not o:
            continue
        bud = [str(N * g) for g in B["G"]]; X = [x[3] for x in o]; bad = [y for x in X for y in x["bad"]]
        nv = [x["native_vs_16x64"] for x in X if x["native_vs_16x64"] is not None]
        chk = dict(motions=len(o), scale=sorted({round(scale(b, x[1]), 9) for x in o}), settings_bad=len(bad), settings_bad_first=bad[:10],
                   native_diff_across_runs=max(x["native_diff"] for x in X), ta_runs=len(X),
                   ta_phc_reward_vs_1_minus_best_cost_at=max(x["ta_budget"] for x in X), pos_div_vs_s=max(x["pos_div"] for x in X),
                   native_vs_16x64_run=dict(clips=len(nv), max_abs_diff=max(nv, default=None), identical=sum(v == 0 for v in nv)))
        mean = {"native": dict(zip(KEYS_BFM, np.mean([x[2]["native"] for x in o], 0).tolist()))}
        for k in METHODS:
            for w in ("best", "centre"):
                mean[f"{k}_{w}"] = {n: dict(zip(KEYS_BFM, np.mean([x[2][k][w][j] for x in o], 0).tolist())) for j, n in enumerate(bud)}
        rep[b] = dict(keys=list(KEYS_BFM), budgets=bud, checks=chk, mean=mean,
                      per_motion={x[1]: {k: (v.tolist() if isinstance(v, np.ndarray) else {w: a.tolist() for w, a in v.items()}) for k, v in x[2].items()} for x in o})
        v = lambda k, w, n: mean["native"] if k == "native" else mean[f"{k}_{w}"][n]
        hd = "| method | PHC reward | Success | MPJPE (cm) | MMPJPE (cm) | MPJAE (m/s^2) | Frames within threshold | Drift (m) |\n|---|---|---|---|---|---|---|---|"
        row = lambda k, w, n: (f"| {NAMES[k]} | {v(k, w, n)['phc']:.3f} | {100 * v(k, w, n)['success']:.1f}% | {v(k, w, n)['mpjpe']:.2f} | {v(k, w, n)['mmpjpe']:.2f} | "
                               f"{v(k, w, n)['mpjae']:.2f} | {100 * v(k, w, n)['within']:.1f}% | {v(k, w, n)['drift']:.2f} |")
        L += [f"\n## {b}{'' if b == 'child' else ' (' + B['label'] + ')'}: {len(o)} AMASS motions, seed 0, mean over motions", "",
              f"### Best sequence after {int(bud[-1]):,} rollouts ({N} x {B['G'][-1]})", "", hd] + [row(k, "best", bud[-1]) for k in ORDER]
        n1 = str(1024)
        if n1 in bud:
            L += ["", f"### Best sequence after 1,024 rollouts ({N} x {1024 // N})", "", hd] + [row(k, "best", n1) for k in ORDER]
            om, op = old_m.get(b, {}).get("mean"), old_p.get(f"amass_{b}", {}).get("best", {}).get("1024")
            if om and op:
                L += ["", "For comparison, the earlier run at 1,024 rollouts (16 x 64; outputs/amass_baselines, PHC reward from outputs/phc_reward_tables):", "",
                      "| method | PHC reward | Success | MPJPE (cm) |", "|---|---|---|---|"]
                for k in ORDER:
                    u = om["native"] if k == "native" else om[f"{k}_best"]["1024"]
                    L.append(f"| {NAMES[k]} | {op[k]:.3f} | {100 * u['success']:.1f}% | {u['mpjpe']:.2f} |")
        L += ["", f"### By budget, best sequence so far (rollouts; native is one rollout, the same in every column)"]
        for key, lab, f in (("phc", "PHC reward", lambda x: f"{x:.3f}"), ("success", "Success", lambda x: f"{100 * x:.1f}%"), ("mpjpe", "MPJPE (cm)", lambda x: f"{x:.2f}"),
                            ("mmpjpe", "MMPJPE (cm)", lambda x: f"{x:.2f}"), ("mpjae", "MPJAE (m/s^2)", lambda x: f"{x:.2f}")):
            L += ["", f"{lab}:", "", "| method | " + " | ".join(f"{int(n):,}" for n in bud) + " |", "|---" * (len(bud) + 1) + "|"]
            L += [f"| {NAMES[k]} | " + " | ".join(f(v(k, "best", n)[key]) for n in bud) + " |" for k in ORDER]
        L += ["", f"### Mean sequence after {int(bud[-1]):,} rollouts (native: its one rollout, for reference)", "", hd] + [row(k, "centre", bud[-1]) for k in ORDER]
        L += ["", "### Checks", "", f"- motions: {chk['motions']}; root scale s: {', '.join(f'{x:g}' for x in chk['scale'])}",
              f"- settings in summary.json (n = {N}, iters = {B['iters']}, budgets, method flags, {N * B['iters']:,} sample rollouts): {chk['settings_bad']} mismatches"
              + (f" (first: {chk['settings_bad_first']})" if bad else ""),
              f"- pos_div in summary.json vs s: largest difference {chk['pos_div_vs_s']:.1e}",
              f"- native q_ER8 across a clip's three runs: largest difference {chk['native_diff_across_runs']:.1e}",
              f"- TA (PHC objective, {chk['ta_runs']} runs): PHC reward of q_best_it<k> vs 1 - best_cost_at[k], largest difference "
              f"{chk['ta_phc_reward_vs_1_minus_best_cost_at']:.1e} (rollouts stored as float32)",
              f"- for reference, native q_ER8 vs the earlier 16 x 64 run's: {chk['native_vs_16x64_run']['identical']} of {chk['native_vs_16x64_run']['clips']} clips identical, "
              f"largest difference {chk['native_vs_16x64_run']['max_abs_diff']}"]
    write_json(dest / f"results{suffix}.json", rep)
    head = ["# Latent-sequence methods on AMASS motions, BFMTrack's own settings (128 rollouts x 24 iterations = 3,072)", "",
            "Columns: PHC reward (0-1, higher is better); Success (PHC's test: the world-frame mean body error never above 0.5 m adult-equivalent, "
            "0.5 x s); MPJPE, MMPJPE (cm); MPJAE (m/s^2, acceleration error); Frames within threshold (share of frames whose world-frame mean body "
            "error is within that 0.5 m adult-equivalent); Drift (mean root xy distance, m adult-equivalent). Each rollout's state after step t "
            "against reference frame t + 1. Method details at the bottom."]
    r = paths.writable(dest / f"README{suffix}.md"); r.parent.mkdir(parents=True, exist_ok=True)
    r.write_text("\n".join(head + L + ["", "## Method details", "", "```", __doc__.strip(), "", BFM_DOC, "```"]) + "\n")
    print("\n".join(L))


COLUMNS = (("MPJPE (cm)", "mpjpe", "{:.2f}"), ("MMPJPE", "mmpjpe", "{:.2f}"), ("MPJAE (m/s^2)", "mpjae", "{:.2f}"), ("Fall (s)", "fall_s", "{:.2f}"),
           ("Success", "success", "{:.1f}%", 100), ("drift (m, adult-eq)", "drift", "{:.2f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("commands", "analyse")); ap.add_argument("--set", choices=tuple(SETS))
    ap.add_argument("--bodies", nargs="+", choices=tuple(BODY)); ap.add_argument("--clips", type=int, help="only the first CLIPS clips (tests)")
    ap.add_argument("--dest", help="write results / README here instead of OUT"); ap.add_argument("--suffix", default="")
    a = ap.parse_args(); use(a.set)
    commands() if a.cmd == "commands" else analyse(a.bodies, a.clips, a.dest, a.suffix)
