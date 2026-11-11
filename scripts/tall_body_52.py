#!/usr/bin/env python3
"""tall_body_52.py -- the tall, thin body of the new teaser (len1.4_wid0.8: length x1.4, width x0.8, isometric, 2.47 m, 109 kg) on the
52 child task motions (user, 11/11): LOES (ours) and CEM (BFM-Zero), seed 0 each, 104 runs, 1,024 rollouts (user, 11/11; first planned at 2,048); not for Notion. Then the motions where
z0 does badly, CEM is not much better, and ours improves clearly: the top 5 candidates for the user to choose from.
  LOES  scripts/single_z_search.py, the child's settings (1,024 rollouts, pairs 8, sigma 0.05, lr 0.03, ExactObs), our cost WITHOUT
        the anchor term (user, 11/11: the method dropped it; --anchor stays given only so anchor_dist is logged):
        BFM term + 1.0 heading + 0.1 position
  CEM   the same search settings with --algo cem --objective phc: BFM-Zero's PHC tracking reward alone (as outputs/child_52_cem_phc,
        scripts/body_52_test.py ccmd); lengths / the body's root scale (pos_div 1.364, set by the script)
  The reference: data/len1.4_wid0.8/retargeting_motion/<clip>.npz (scripts/make_scaled_bodies.py).
Per run: its best rollout (best.npz, the best latent after 1,024 rollouts by the run's own objective) and z0's (origin_z.npz); q[t]
against ref[t + 1] (as the Task page):
  MPJPE, MMPJPE, MPJAE   eval_metrics.tracking_metrics (cm, cm, m/s^2: MPJAE is the per-joint ACCELERATION error)
  fall_s    seconds with a non-foot body within 1 cm of the floor while the reference's is more than 5 cm above it (Task page)
  success   PHC's test: the world-frame mean body error never above 0.5 m adult-equivalent (0.5 x 1.364 m here)
  phc       PHC's tracking reward (1 - model/phc_reward.py's cost; CEM's objective)
  drift     mean pelvis x, y distance to the reference (m, adult-equivalent)
Candidates (rule fixed before looking at the results):
  "z0 bad"       z0 fails PHC's success test or falls for more than 0.5 s
  "CEM not good" CEM fails the success test, or falls for more than 0.5 s, or keeps at least half of z0's MPJPE excess over ours
                 (MPJPE_cem - MPJPE_ours >= 0.5 x (MPJPE_z0 - MPJPE_ours))
  "ours clear"   ours succeeds and falls no longer than z0 and CEM
  ranked by ours' MPJPE gain over the better of z0 and CEM (min(MPJPE_z0, MPJPE_cem) - MPJPE_ours, cm); the top 5 that pass all three,
  then (if fewer than 5 pass) the next best by the same gain, marked.
  commands | analyse      -> outputs/tall_body_52/<stem>_{es,cem}_s0, outputs/tall_body_52/{results.json, README.md}"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.metrics import MetricSuite, metric_body, world_positions   # noqa: E402
from crossenbodiment import paths   # noqa: E402
from crossenbodiment.experiment import ClipSet, Experiment, workers   # noqa: E402
from crossenbodiment.methods import experiment_keys   # noqa: E402
from crossenbodiment.results import ReportWriter, ResultsTable, write_json   # noqa: E402
# refactor step 8: runs are read under paths.outputs_root() ($CE_OUTPUTS_ROOT; OUT, read by one() when called), results written
# under paths.results_root() ($CE_RESULTS_ROOT; RES); both REPO/outputs by default
B = "len1.4_wid0.8"; XML = f"assets/robots_torque/{B}/robot_torque_full.xml"; OUT = paths.output("tall_body_52"); RES = paths.result("tall_body_52")
from crossenbodiment.paths import CHILD_52_TASK, read_clips   # noqa: E402  configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
CLIPS = read_clips(CHILD_52_TASK)
# the run-directory keys of the two methods (crossenbodiment/methods.py RUNS): LOES without the anchor "es", BFM-Zero's CEM "cem"
K = experiment_keys("outputs/tall_body_52"); LOES, CEM = K["loes"], K["cem_phc"]
EXP = Experiment("outputs/tall_body_52", clips=ClipSet(CLIPS), methods=(LOES, CEM), order=("clip", "method"))
KEYS = ("mpjpe", "mmpjpe", "mpjae", "fall_s", "success", "phc", "drift")


def st_(c):
    return c.split("/")[1]


def commands():
    base = lambda c: (f"uv run python scripts/single_z_search.py --clip {c} --body {B} --xml {XML} --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 "
                      f"--obs-scale exact --eval-every 0 --init-vel-scale 1 --seed 0")
    own = {LOES: lambda c: f"--objective bfm --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy --anchor-weight 0",
           CEM: lambda c: "--objective phc --algo cem"}
    EXP.print_commands(lambda r: f"{base(r.clip)} {own[r.method](r.clip)} --out {r.dir}", clip=EXP.clips.longest_first())


_ctx = {}


def one(c):
    from model.phc_reward import PHCCost
    if not _ctx:
        _ctx["body"] = metric_body(XML, True)
    body = _ctx["body"]; m, d = body.m, body.d; st = st_(c)
    ref_all = np.load(REPO / f"data/{B}/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); ref = ref_all[1:]
    S = {a: json.load(open(OUT / f"{st}_{a}_s0/summary.json")) for a in (LOES, CEM)}; s = S[LOES]["pos_div"]
    chk = dict(bad=[k for k, v in {LOES: dict(algo="es", objective="bfm", heading_weight=1.0, pos_weight=0.1, anchor_weight=0.0),
                                   CEM: dict(algo="cem", objective="phc", heading_weight=0.0, pos_weight=0.0, anchor_weight=0.0)}.items()
                    for x, w in v.items() if S[k].get(x) != w]
               + [k for k in S if (S[k]["evals"], S[k]["pairs"], S[k]["sigma"], S[k]["lr"], S[k]["obs_scale"], S[k]["body"], S[k]["seed"]) != (1024, 8, 0.05, 0.03, "exact", B, 0)],
               pos_div=s, z0_same=float(np.abs(np.load(OUT / f"{st}_{LOES}_s0/origin_z.npz")["qpos"] - np.load(OUT / f"{st}_{CEM}_s0/origin_z.npz")["qpos"]).max()))
    phc = PHCCost(m, ref_all[0], ref, scale=s)

    Pr = world_positions(m, d, ref); _, Dr = body.part_floor(ref)
    # {mpjpe, mmpjpe, mpjae, fall_s, success (PHC, 0.5 m x s), phc (PHC's tracking reward), drift (adult-equivalent m)} per rollout
    stats = MetricSuite(body, ("mpjpe", "mmpjpe", "mpjae", "fall_s", "success", ("phc", lambda x: 1.0 - float(phc(x.q[None], ref_all[0])[0])), "drift"),
                        s, drift_unit="adult").reference(ref, Pr, Dr)
    # results.json keys (unchanged): z0 (frozen), cem (cem_phc), ours (loes)
    res = dict(z0=stats(np.load(OUT / f"{st}_{LOES}_s0/origin_z.npz")["qpos"]), cem=stats(np.load(OUT / f"{st}_{CEM}_s0/best.npz")["qpos"]),
               ours=stats(np.load(OUT / f"{st}_{LOES}_s0/best.npz")["qpos"]))
    return st, res, chk


def analyse():
    with Pool(workers(16)) as p:
        out = p.map(one, CLIPS, chunksize=1)
    per = {st: r for st, r, _ in out}
    chk = dict(settings_bad=[(st, c["bad"]) for st, _, c in out if c["bad"]], z0_es_vs_cem=max(c["z0_same"] for _, _, c in out),
               pos_div=sorted({round(c["pos_div"], 6) for _, _, c in out}))
    mean = {a: {k: float(np.mean([per[st][a][k] for st in per])) for k in KEYS} for a in ("z0", "cem", "ours")}
    rows = []
    for st, r in per.items():
        z, c, o = r["z0"], r["cem"], r["ours"]
        z0_bad = z["success"] < 1 or z["fall_s"] > 0.5
        cem_not_good = c["success"] < 1 or c["fall_s"] > 0.5 or (c["mpjpe"] - o["mpjpe"]) >= 0.5 * (z["mpjpe"] - o["mpjpe"])
        ours_clear = o["success"] == 1 and o["fall_s"] <= min(z["fall_s"], c["fall_s"])
        rows.append(dict(motion=st, gain=min(z["mpjpe"], c["mpjpe"]) - o["mpjpe"], z0_bad=z0_bad, cem_not_good=cem_not_good, ours_clear=ours_clear,
                         passes=bool(z0_bad and cem_not_good and ours_clear)))
    rows.sort(key=lambda x: (-x["passes"], -x["gain"]))
    rep = dict(checks=chk, mean=mean, per_motion=per, ranking=rows); write_json(RES / "results.json", rep)
    L = [f"checks: {json.dumps(chk)}", "", "mean over the 52 motions (best after 1,024 rollouts; z0 one rollout)"]
    L += ResultsTable.markdown([(n, mean[a]) for a, n in (("z0", "z0"), ("cem", "CEM (BFM-Zero)"), ("ours", "Ours (LOES)"))], COLUMNS)
    L += ["", f"candidates: {sum(r['passes'] for r in rows)} of 52 pass all three tests; top 10 by ours' MPJPE gain over the better of z0 and CEM",
          "| # | motion | passes | gain (cm) | MPJPE z0 / CEM / ours | Success z0 / CEM / ours | Fall (s) z0 / CEM / ours | PHC reward z0 / CEM / ours |",
          "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows[:10]):
        p = per[r["motion"]]; f = lambda k, fm: " / ".join(fm.format(p[a][k]) for a in ("z0", "cem", "ours"))
        L.append(f"| {i + 1} | {r['motion']} | {'yes' if r['passes'] else 'no (' + ', '.join(n for n, k in (('z0 ok', 'z0_bad'), ('CEM good', 'cem_not_good'), ('ours not clear', 'ours_clear')) if not r[k]) + ')'} "
                 f"| {r['gain']:.2f} | {f('mpjpe', '{:.1f}')} | {f('success', '{:.0f}')} | {f('fall_s', '{:.1f}')} | {f('phc', '{:.3f}')} |")
    ReportWriter("Tall, thin body (len1.4_wid0.8): LOES vs CEM on the 52 task motions", __doc__).write(RES / "README.md", L, gap="\n\n")
    print("\n".join(L))


COLUMNS = (("MPJPE (cm)", "mpjpe", "{:.2f}"), ("MMPJPE", "mmpjpe", "{:.2f}"), ("MPJAE (m/s^2)", "mpjae", "{:.2f}"), ("Fall (s)", "fall_s", "{:.2f}"),
           ("Success", "success", "{:.1f}%", 100), ("PHC reward", "phc", "{:.3f}"), ("drift (m, adult-eq)", "drift", "{:.2f}"))


if __name__ == "__main__":
    {"commands": commands, "analyse": analyse}[sys.argv[1]]()
