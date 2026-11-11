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
import json, sys
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
        return " ".join((f"uv run python scripts/lso_search.py --ref {B['ref'].format(c=c)} --body {B.get('label', b)} --n 16 --iters {B['iters']} "
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
    stats = MetricSuite(body, ("mpjpe", "mmpjpe", "mpjae", "fall_s", "success", "drift"), s, drift_unit="adult").reference(ref, Pr, Dr).values
    R, chk = {}, dict(native_diff=0.0, bad=[])
    nat = None
    for k in METHODS:
        run = OUT / b / f"{c}_{k}_s0"; S = json.load(open(run / "summary.json")); z = np.load(run / "rollouts.npz")
        want = dict(n=16, iters=B["iters"], seed=0, init_vel_scale=1.0, antithetic=False, sigma=0.0125, beta=1.0,
                    **{LSO: dict(method="lso", objective="bfm", rank=False, heading_weight=0.0, pos_weight=0.0),
                       LSOR: dict(method="lso", objective="bfm", rank=True, heading_weight=0.05, pos_weight=0.005),
                       TA: dict(method="ta", objective="phc", rank=False, heading_weight=0.0, pos_weight=0.0)}[k])
        chk["bad"] += [(b, c, k, x, S.get(x), v) for x, v in want.items() if S.get(x) != v]
        nat = z["q_ER8"] if nat is None else nat; chk["native_diff"] = max(chk["native_diff"], float(np.abs(z["q_ER8"] - nat).max()))
        R[k] = dict(best=np.array([stats(z[f"q_best_it{g}"]) for g in B["G"]]), centre=np.array([stats(z[f"q_mean_it{g}"]) for g in B["G"]]))
    R["native"] = np.array(stats(nat))
    return b, c, R, chk


def analyse():
    jobs = [(b, c) for b in BODY for c in CLIPS if all((OUT / b / f"{c}_{k}_s0/rollouts.npz").exists() for k in METHODS)]
    with Pool(workers(16)) as p:
        res = p.map(one, jobs, chunksize=1)
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
    write_json(RES / "results.json", rep); ReportWriter("Latent-sequence methods on AMASS motions", __doc__).write(RES / "README.md", L)
    print("\n".join(L))


COLUMNS = (("MPJPE (cm)", "mpjpe", "{:.2f}"), ("MMPJPE", "mmpjpe", "{:.2f}"), ("MPJAE (m/s^2)", "mpjae", "{:.2f}"), ("Fall (s)", "fall_s", "{:.2f}"),
           ("Success", "success", "{:.1f}%", 100), ("drift (m, adult-eq)", "drift", "{:.2f}"))


if __name__ == "__main__":
    {"commands": commands, "analyse": analyse}[sys.argv[1]]()
