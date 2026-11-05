#!/usr/bin/env python3
"""child_52_cmaes.py -- CMA-ES added to Comparison A: outputs/child_52_cmaes/<stem>_cmaes_s<seed> (searches, same settings and motions as outputs/child_52), scored like the other methods.
  prepare  writes outputs/child_52_cmaes/rollouts/<stem>/{z.npy, index.json}: each run's distinct best z so far at every generation and its mean z after 4, 8, 16, 32, 64 generations
  (run)    outputs/train_logs/queue: scripts/single_z_search.py --eval-z z.npy --eval-save-qpos on each motion
  analyse  metrics of each rollout, best-so-far costs checked against curve.csv; writes per_run.npz (best_cmaes, mean_cmaes: motions x seeds x generations x metrics, keys as in outputs/child_52_curves/per_run.npz) and report.json (means over 260 runs; Wilcoxon on the 52 motion means against the other methods).
usage: uv run python scripts/child_52_cmaes.py prepare | analyse"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
SRC = REPO / "outputs/child_52"; OUT = REPO / "outputs/child_52_cmaes"; ROLL = OUT / "rollouts"; G = (4, 8, 16, 32, 64); BUD = [g * 16 for g in G]


def clips():
    return [l.strip() for l in open(SRC / "clips.txt")]


def prepare():
    for c in clips():
        stem = c.split("/")[1]; d = ROLL / stem; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
        for s in range(5):
            r = f"{stem}_cmaes_s{s}"; t = np.load(OUT / r / "z_trace.npz"); b, m = t["z_best"], t["z_mean"]; assert len(b) == 65 and len(m) == 65, r; best = []
            for g in range(1, 65):
                if not best or np.abs(b[g] - b[g - 1]).max() > 0:
                    Z.append(b[g]); best.append(len(Z) - 1)
                else:
                    best.append(best[-1])
            centre = list(range(len(Z), len(Z) + len(G))); Z += [m[g] for g in G]; idx[r] = dict(best=best, centre=centre)
        np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, map=idx)))
    print("prepared")


def analyse():
    import eval_metrics as E
    from child_52_means import full, KEYS
    body = E.Body(E.XML["child"], True); cl = clips(); B = np.zeros((len(cl), 5, 64, len(KEYS))); C = np.zeros((len(cl), 5, len(G), len(KEYS))); worst = []
    for ci, c in enumerate(cl):
        stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); z0 = float(next(csv.DictReader(open(SRC / f"{stem}_z0/eval.csv")))["cost"]); d = ROLL / stem
        ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64); cost = [float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]; V = {}
        def m(i):
            if i not in V:
                v = full(body, Q[i], ref, cost[i] / z0); V[i] = [v[k] for k in KEYS]
            return V[i]
        for s in range(5):
            r = f"{stem}_cmaes_s{s}"
            for j, i in enumerate(ix[r]["best"]):
                B[ci, s, j] = m(i)
            for j, i in enumerate(ix[r]["centre"]):
                C[ci, s, j] = m(i)
            bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(OUT / r / "curve.csv"))]); worst.append(float(np.abs(np.array(cost)[ix[r]["best"]] - bs).max()))
        print(stem, flush=True)
    np.savez_compressed(OUT / "per_run.npz", clips=np.array(cl), keys=np.array(KEYS), gens_mean_z=np.array(G), best_cmaes=B, mean_cmaes=C)
    O = np.load(REPO / "outputs/child_52_curves/per_run.npz"); mean = lambda A: {k: float(A[:, :, i].mean()) for i, k in enumerate(KEYS)}
    rep = dict(max_rollout_vs_search_cost=max(worst), best={str(b): mean(B[:, :, g - 1]) for b, g in zip(BUD, G)}, centre={str(b): mean(C[:, :, j]) for j, b in enumerate(BUD)}, pairs=dict(best={}, centre={}))
    for o in ("es", "random", "cem"):
        for t in ("best", "centre"):
            rep["pairs"][t][f"{o}_vs_cmaes"] = {k: {str(b): dict(lower=int((X.mean(1) < Y.mean(1)).sum()), p=float(wilcoxon(X.mean(1), Y.mean(1)).pvalue)) for b, X, Y in ((b, (O[f"best_{o}"][:, :, g - 1, i] if t == "best" else O[f"mean_{o}"][:, :, j, i]), (B[:, :, g - 1, i] if t == "best" else C[:, :, j, i])) for j, (b, g) in enumerate(zip(BUD, G)))} for i, k in enumerate(KEYS) if k in ("ratio", "mpjpe", "mmpjpe", "mpjae")}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1)); print("JSON" + json.dumps(rep))


if __name__ == "__main__":
    {"prepare": prepare, "analyse": analyse}[sys.argv[1]]()
