#!/usr/bin/env python3
"""child_52_aligned.py -- Comparison A's tables with each rollout state compared to the frame it should be at: q[t] is the state after step t, which goes with ref[t + 1]. scripts/child_52_means.py and scripts/child_52_cmaes.py compared q[t] with ref[t].
The saved rollouts are only re-scored, nothing is simulated (the searches themselves stay as they were). Check: the same code with offset 0 must reproduce the old per_run.npz files, on 4 motions.
Writes outputs/child_52_aligned/{per_run.npz, means.json}: best_<algo> (motions, seeds, 64 generations, metrics), mean_<algo> (motions, seeds, generations 4 8 16 32 64, metrics), z0. usage: uv run python scripts/child_52_aligned.py"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import eval_metrics as E                                                  # noqa: E402
from child_52_means import full, KEYS, CUR, MZ, SRC                       # noqa: E402
CM = REPO / "outputs/child_52_cmaes"; OUT = REPO / "outputs/child_52_aligned"; ALGOS = ("es", "cem", "random", "cmaes"); G = (4, 8, 16, 32, 64)
body = None


def rollouts(d):
    return json.load(open(d / "index.json"))["map"], np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64), [float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]


def clip(job):
    global body
    c, offset = job; body = body or E.Body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[offset:]; z0c = float(next(csv.DictReader(open(SRC / f"{stem}_z0/eval.csv")))["cost"])
    vec = lambda q, cost: (lambda m: [m[k] for k in KEYS])(full(body, q, ref, cost / z0c))
    B = {a: np.zeros((5, 64, len(KEYS))) for a in ALGOS}; C = {a: np.zeros((5, len(G), len(KEYS))) for a in ALGOS}; z0 = vec(np.load(SRC / f"{stem}_z0/eval_qpos.npz")["qpos"][0].astype(np.float64), z0c)
    for d, arr in ((CUR / stem, B), (MZ / stem, C)):
        ix, Q, cost = rollouts(d); V = {}
        for r, idx in ix.items():
            a, s = r.rsplit("_", 2)[1:]; s = int(s[1:])
            for j, i in enumerate(idx):
                V[i] = V[i] if i in V else vec(Q[i], cost[i]); arr[a][s, j] = V[i]
    ix, Q, cost = rollouts(CM / "rollouts" / stem); V = {}
    for r, parts in ix.items():
        s = int(r.rsplit("_", 1)[1][1:])
        for key, arr in (("best", B), ("centre", C)):
            for j, i in enumerate(parts[key]):
                V[i] = V[i] if i in V else vec(Q[i], cost[i]); arr["cmaes"][s, j] = V[i]
    return stem, z0, B, C


def main():
    OUT.mkdir(exist_ok=True); clips = [l.strip() for l in open(SRC / "clips.txt")]; old = np.load(CUR / "per_run.npz"); oldc = np.load(CM / "per_run.npz")
    with Pool(24) as p:
        chk = p.map(clip, [(c, 0) for c in clips[:4]]); worst = 0.0
        for ci, (stem, z0, B, C) in enumerate(chk):
            worst = max(worst, float(np.abs(np.array(z0) - old["z0"][ci]).max()))
            for a in ALGOS:
                ob, oc = (oldc["best_cmaes"], oldc["mean_cmaes"]) if a == "cmaes" else (old[f"best_{a}"], old[f"mean_{a}"])
                worst = max(worst, float(np.abs(B[a] - ob[ci]).max()), float(np.abs(C[a] - oc[ci]).max()))
        print("offset 0 against the old per_run files, largest difference:", worst, flush=True); assert worst < 1e-9, worst
        res = p.map(clip, [(c, 1) for c in clips])
    z0 = np.array([r[1] for r in res]); B = {a: np.stack([r[2][a] for r in res]) for a in ALGOS}; C = {a: np.stack([r[3][a] for r in res]) for a in ALGOS}
    np.savez_compressed(OUT / "per_run.npz", clips=np.array(clips), keys=np.array(KEYS), gens_mean_z=np.array(G), z0=z0, **{f"best_{a}": B[a] for a in ALGOS}, **{f"mean_{a}": C[a] for a in ALGOS})
    m = lambda A: {k: float(A[..., i].mean()) for i, k in enumerate(KEYS)}
    rep = dict(z0=m(z0), best={a: {str(g * 16): m(B[a][:, :, g - 1]) for g in G} for a in ALGOS}, centre={a: {str(g * 16): m(C[a][:, :, j]) for j, g in enumerate(G)} for a in ALGOS})
    (OUT / "means.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    main()
