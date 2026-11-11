#!/usr/bin/env python3
"""cost_correlation.py -- does our tracking cost track what we care about? (user, 11/11, for the paper's "Analysis of the
Representation-Based Objective"): within each child task motion, the correlation between our cost and the task's HumEnv reward,
MPJPE, MMPJPE and MPJAE over many latents.

Latents: every latent the child 52-motion comparison already re-rolled (outputs/child_52_rerun/rollouts/<stem>/z.npy: the best latent
so far at every generation and the centre latents of LOES, random search, CEM and CMA-ES, 5 seeds; 201-938 per motion) plus z0.
They are rolled out again with the LOES runs' exact settings (scripts/single_z_search.py --eval-z, 16 envs) saving qpos, qvel and
actions (--eval-save-qvel, --eval-save-act); the qpos must equal the earlier re-rolls bit for bit.
Per latent:
  cost      our search cost as the searches computed it (BFM term + 1.0 heading + 0.1 position + 0.3 anchor) and its tracking
            part without the anchor regulariser (BFM term + 1.0 heading + 0.1 position)
  reward    the motion's own HumEnv task reward (humenv.rewards, the clip's task name, e.g. move-ego-0-2), per step on the
            adult-equivalent state (model.exact_obs.ExactObs.adult_state, as scripts/body_reward_eval.py), the actor's action as ctrl,
            averaged over the 300 steps (0-1)
  phc       PHC's tracking reward (model/phc_reward.py: 1 - PHCCost, world-frame body positions, rotations, velocities, lengths / 0.611;
            the reward of the convergence plots and BFM-Zero's objective; user, 11/11: the "task reward" of the paper text is this one)
  MPJPE, MMPJPE, MPJAE  eval_metrics.tracking_metrics, q[t] against ref[t + 1] (as the Task page)
Correlation: Spearman (and Pearson) across a motion's latents; reported as the median over the 52 motions and how many motions have
the expected sign with p < 0.05 (cost vs reward negative; cost vs the errors positive).
  prepare | commands | analyse     -> outputs/cost_corr/<stem>/..., outputs/cost_corr/{results.json, README.md}"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.metrics import metric_body   # noqa: E402  one Body per (xml, smpl) and process (refactor step 3)
from crossenbodiment.geometry import sphere16   # noqa: E402  16 z / |z| (refactor step 3)
from crossenbodiment.bodies import BODIES
from crossenbodiment.paths import CHILD_XML_REL
from crossenbodiment import paths
from crossenbodiment.experiment import Experiment, workers
from crossenbodiment.results import ReportWriter, save_npz, write_json
# refactor step 8: runs and re-rolls are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results (and prepare's z.npy) written
# under paths.results_root() ($CE_RESULTS_ROOT); both REPO/outputs by default
RR = paths.output("child_52_rerun"); OUT = paths.output("cost_corr"); RES = paths.result("cost_corr"); XML = CHILD_XML_REL
EXP = Experiment("outputs/cost_corr")
W_HEAD, W_POS, W_ANCHOR = 1.0, 0.1, 0.3


def clips():
    from crossenbodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    return read_clips(CHILD_52_TASK)


def prepare():
    for c in clips():
        st = c.split("/")[1]; d = paths.writable(RES / st); d.mkdir(parents=True, exist_ok=True)
        z0 = np.load(REPO / f"data/origin_z/{c}.npy").reshape(-1).astype(np.float64); z0 = sphere16(z0)
        np.save(d / "z.npy", np.concatenate([np.load(RR / "rollouts" / st / "z.npy").astype(np.float64), z0[None]]))
    print("prepared", len(clips()))


def commands():
    for c in clips():
        d = f"{EXP.out}/{c.split('/')[1]}"
        print(EXP.eval_line(d, f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {XML} --objective bfm --evals 1024 --pairs 8 --sigma 0.05 "
                               f"--lr 0.03 --obs-scale exact --heading-weight {W_HEAD} --pos-weight {W_POS} --anchor data/origin_z/{c}.npy --anchor-weight {W_ANCHOR} "
                               f"--eval-every 0 --init-vel-scale 1", seed=0, save=("qpos", "act", "qvel")))


_ctx = {}


def one(c):
    import mujoco
    import eval_metrics as E
    from child_reward_eval import reward_of
    from model.exact_obs import ExactObs, ADULT_XML
    from model.phc_reward import PHCCost
    if not _ctx:
        _ctx.update(body=metric_body(E.XML["child"], True), ex=ExactObs(REPO / XML, REPO / ADULT_XML), ma=mujoco.MjModel.from_xml_path(str(REPO / ADULT_XML)),
                    mc=mujoco.MjModel.from_xml_path(str(REPO / XML)))
        _ctx["da"] = mujoco.MjData(_ctx["ma"])
    body, ex, ma, da = _ctx["body"], _ctx["ex"], _ctx["ma"], _ctx["da"]
    task, st = c.split("/"); d = OUT / st; rew = reward_of(task)
    ref_all = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); ref = ref_all[1:]
    phc = PHCCost(_ctx["mc"], ref_all[0], ref_all[1:], scale=BODIES.scale("child"))
    Q = np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64); V = np.load(d / "eval_qvel.npz")["qvel"].astype(np.float64); A = np.load(d / "eval_act.npz")["act"].astype(np.float64)
    rows = list(csv.DictReader(open(d / "eval.csv"))); n = len(rows)
    old = np.load(RR / "rollouts" / st / "eval_qpos.npz")["qpos"]; oldc = [float(r["cost"]) for r in csv.DictReader(open(RR / "rollouts" / st / "eval.csv"))]
    chk = dict(qpos_vs_earlier=float(np.abs(Q[:len(old)].astype(np.float32) - old).max()), cost_vs_earlier=float(np.max(np.abs(np.array([float(r["cost"]) for r in rows[:len(oldc)]]) - oldc))),
               z0_vs_origin=float(np.abs(Q[-1].astype(np.float32) - np.load(RR / f"{st}_es_s0/origin_z.npz")["qpos"]).max()))
    out = dict(cost=[], track=[], reward=[], phc=[], mpjpe=[], mmpjpe=[], mpjae=[])
    for i in range(n):
        r = rows[i]; out["cost"].append(float(r["cost"])); out["track"].append(float(r["bfm"]) + W_HEAD * float(r["head"]) + W_POS * float(r["pos"]))
        tot = 0.0
        for q, v, a in zip(Q[i], V[i], A[i]):
            qa, va = ex.adult_state(q, v); da.qpos[:] = qa; da.qvel[:] = va; da.ctrl[:] = a; mujoco.mj_forward(ma, da); tot += float(rew.compute(ma, da))
        out["reward"].append(tot / len(Q[i])); out["phc"].append(1.0 - float(phc(Q[i][None], ref_all[0])[0]))
        tm = E.tracking_metrics(body, Q[i], ref); out["mpjpe"].append(tm["mpjpe"]); out["mmpjpe"].append(tm["mmpjpe"]); out["mpjae"].append(tm["mpjae"])
    # the reward computed here equals humenv's own call (fresh MjData) on a sample
    i, t = n // 2, 150; qa, va = ex.adult_state(Q[i][t], V[i][t]); fresh = float(rew(ma, qa, va, A[i][t]))
    da.qpos[:] = qa; da.qvel[:] = va; da.ctrl[:] = A[i][t]; mujoco.mj_forward(ma, da); chk["reward_vs_humenv_call"] = abs(fresh - float(rew.compute(ma, da)))
    return st, {k: np.array(v) for k, v in out.items()}, chk


def analyse():
    from scipy.stats import spearmanr, pearsonr
    with Pool(workers(16)) as p:
        res = p.map(one, clips(), chunksize=1)
    chk = {k: max(r[2][k] for r in res) for k in res[0][2]}; print("checks (largest):", chk, flush=True)
    MET = (("phc", -1), ("mpjpe", 1), ("mmpjpe", 1), ("mpjae", 1), ("reward", -1)); rep = dict(checks=chk, per_motion={}, summary={})
    for cname in ("track", "cost"):
        rep["summary"][cname] = {}
        for m, sign in MET:
            S, P, ok_s, ok_p = [], [], 0, 0
            for st, D, _ in res:
                x, y = D[cname], D[m]
                if np.ptp(y) == 0:
                    continue
                s = spearmanr(x, y); p = pearsonr(x, y); S.append(s.statistic); P.append(p.statistic)
                ok_s += int(np.sign(s.statistic) == sign and s.pvalue < 0.05); ok_p += int(np.sign(p.statistic) == sign and p.pvalue < 0.05)
                rep["per_motion"].setdefault(st, {})[f"{cname}_{m}"] = dict(spearman=float(s.statistic), spearman_p=float(s.pvalue), pearson=float(p.statistic), n=len(x))
            rep["summary"][cname][m] = dict(spearman_median=float(np.median(S)), spearman_iqr=[float(np.percentile(S, 25)), float(np.percentile(S, 75))],
                                            pearson_median=float(np.median(P)), expected_sign_p05_spearman=ok_s, expected_sign_p05_pearson=ok_p, motions=len(S))
    rep["n_latents"] = {st: len(D["cost"]) for st, D, _ in res}
    write_json(RES / "results.json", rep); save_npz(RES / "per_latent.npz", **{f"{st}__{k}": v for st, D, _ in res for k, v in D.items()})
    L = [f"latents per motion: median {int(np.median(list(rep['n_latents'].values())))} (total {sum(rep['n_latents'].values())})"]
    for cname, lab in (("track", "tracking cost (BFM + 1.0 heading + 0.1 position)"), ("cost", "full search cost (+ 0.3 anchor)")):
        L.append(f"\n{lab}: correlation across a motion's latents, 52 motions\n| vs | Spearman median [IQR] | Pearson median | motions with the expected sign, p<0.05 (Spearman / Pearson) |\n|---|---|---|---|")
        for m, _ in MET:
            v = rep["summary"][cname][m]
            L.append(f"| {m} | {v['spearman_median']:+.2f} [{v['spearman_iqr'][0]:+.2f}, {v['spearman_iqr'][1]:+.2f}] | {v['pearson_median']:+.2f} | {v['expected_sign_p05_spearman']} / {v['expected_sign_p05_pearson']} of {v['motions']} |")
    ReportWriter("Our cost vs task reward and tracking errors (child, 52 task motions)", __doc__).write(RES / "README.md", L, gap="\n\nchecks: " + json.dumps(chk) + "\n\n")
    print("\n".join(L))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
