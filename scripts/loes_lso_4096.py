#!/usr/bin/env python3
"""loes_lso_4096.py -- does LOES catch up with LSO (BFMTrack) on MPJPE, MMPJPE and MPJAE if both get 4,096 rollouts instead of 1,024?
Child body, 3 task motions where LSO beat LOES on all three metrics at 1,024 (outputs/child_52_rerun, outputs/child_52_lso), 5 seeds.
Same settings as those runs, only the budget is 256 x 16: LOES = scripts/single_z_search.py --evals 4096, LSO = scripts/lso_search.py
--iters 256 (BFMTrack's values, pink noise beta 1). Searches: outputs/loes_lso_4096/<stem>_{es,lso}_s<seed>
(outputs/train_logs/queue/run_loes_lso_4096.sh).
  prepare  writes outputs/loes_lso_4096/rollouts/<stem>/{z.npy, index.json}: each LOES run's best latent so far and its centre after
           64, 128, 192 and 256 generations (1,024 to 4,096 rollouts); checks the first 64 generations against outputs/child_52_rerun
  evalcmds one --eval-z command per motion (rolls out z.npy, saves qpos)
  analyse  MPJPE, MMPJPE, MPJAE (scripts/eval_metrics.tracking_metrics, q[t] against ref[t + 1]) for LOES and LSO, best and centre, at
           each budget; checks LSO's first 64 iterations against outputs/child_52_lso and that the LOES rollouts reproduce the search's
           best cost so far. Writes outputs/loes_lso_4096/means.json.
usage: uv run python scripts/loes_lso_4096.py prepare | evalcmds | analyse"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "outputs/loes_lso_4096"; ROLL = OUT / "rollouts"; RERUN = REPO / "outputs/child_52_rerun"; LSO = REPO / "outputs/child_52_lso"
CLIPS = ("move-ego-0-4/move-ego-0-4_9", "move-ego-90-4/move-ego-90-4_5", "move-ego-0-2-raisearms-h-m/move-ego-0-2-raisearms-h-m_1")
SEEDS = range(5); G = (64, 128, 192, 256); KEYS = ("mpjpe", "mmpjpe", "mpjae")
XML = "assets/robots_torque/child/robot_torque_full.xml"


def base_cmd(c):
    return (f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {XML} --objective bfm --evals 4096 --pairs 8 --sigma 0.05 "
            f"--lr 0.03 --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy --anchor-weight 0.3 --eval-every 0 --init-vel-scale 1")


def prepare():
    worst = 0.0
    for c in CLIPS:
        stem = c.split("/")[1]; d = ROLL / stem; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
        for s in SEEDS:
            r = f"{stem}_es_s{s}"; t = np.load(OUT / r / "z_trace.npz"); b, m = t["z_best"], t["z_mean"]; assert len(b) == 257 and len(m) == 257, r
            o = np.load(RERUN / r / "z_trace.npz"); worst = max(worst, float(np.abs(b[:65] - o["z_best"]).max()), float(np.abs(m[:65] - o["z_mean"]).max()))
            idx[r] = dict(best=list(range(len(Z), len(Z) + len(G)))); Z += [b[g] for g in G]
            idx[r]["centre"] = list(range(len(Z), len(Z) + len(G))); Z += [m[g] for g in G]
        np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, map=idx)))
    print("prepared | LOES first 64 generations against outputs/child_52_rerun, largest z difference:", worst)


def evalcmds():
    for c in CLIPS:
        d = f"outputs/loes_lso_4096/rollouts/{c.split('/')[1]}"
        print(f"{d}|{base_cmd(c)} --seed 0 --eval-z {d}/z.npy --eval-save-qpos --out {d}")


def clip(c):
    import eval_metrics as E
    body = E.Body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    met = lambda q: [E.tracking_metrics(body, q.astype(np.float64), ref)[k] for k in KEYS]
    d = ROLL / stem; ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"]; cost = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))])
    R = {f"{a}_{w}": np.zeros((5, len(G), len(KEYS))) for a in ("loes", "lso") for w in ("best", "centre")}; reroll = lso_same = 0.0
    for s in SEEDS:
        r = f"{stem}_es_s{s}"; bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(OUT / r / "curve.csv"))])
        reroll = max(reroll, float(np.abs(cost[ix[r]["best"]] - bs[[g - 1 for g in G]]).max()))
        for j in range(len(G)):
            R["loes_best"][s, j] = met(Q[ix[r]["best"][j]]); R["loes_centre"][s, j] = met(Q[ix[r]["centre"][j]])
        X = np.load(OUT / f"{stem}_lso_s{s}/rollouts.npz"); old = np.load(LSO / f"{stem}_lso_s{s}/rollouts.npz")
        lso_same = max(lso_same, float(np.abs(X["q_best_it64"] - old["q_best_it64"]).max()), float(np.abs(X["q_mean_it64"] - old["q_mean_it64"]).max()))
        for j, g in enumerate(G):
            R["lso_best"][s, j] = met(X[f"q_best_it{g}"]); R["lso_centre"][s, j] = met(X[f"q_mean_it{g}"])
    return stem, R, reroll, lso_same


def analyse():
    with Pool(3) as p:
        res = p.map(clip, CLIPS)
    print("LOES rollouts vs search best cost so far, largest difference:", max(r[2] for r in res),
          "| LSO first 64 iterations against outputs/child_52_lso, largest qpos difference:", max(r[3] for r in res))
    rep = {}
    for stem, R, _, _ in res:
        rep[stem] = {k: {str(g * 16): {m: float(R[k][:, j, i].mean()) for i, m in enumerate(KEYS)} for j, g in enumerate(G)} for k in R}
        rep[stem]["seeds"] = {k: R[k].tolist() for k in R}
    (OUT / "means.json").write_text(json.dumps(rep, indent=1))
    for stem in rep:
        print(f"== {stem} (mean of 5 seeds)")
        for k in ("loes_best", "lso_best", "loes_centre", "lso_centre"):
            print(f"  {k:12s}" + "  ".join(f"{b:>5s}: " + " ".join(f"{rep[stem][k][b][m]:6.2f}" for m in KEYS) for b in rep[stem][k]))


if __name__ == "__main__":
    {"prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[sys.argv[1]]()
