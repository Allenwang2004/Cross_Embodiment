#!/usr/bin/env python3
"""child_52_rerun.py -- Comparison A searched again with the corrected alignment (scripts/single_z_search.py --frame-offset 1): LOES, random search, CEM and CMA-ES, 5 seeds, the 52 motions of outputs/child_52/clips.txt, everything else as in outputs/child_52 and outputs/child_52_cmaes. Frozen z0 is not searched: its rollout is the one every search starts from (origin_z.npz, the same z0 re-projected onto the sphere that the searches and the anchor use). outputs/child_52/<stem>_z0 skipped that projection (z differs by 2.4e-7), and chaos split 11 of the 52 motions.
  (run)    outputs/train_logs/queue/run_child_52_rerun.sh: searches in outputs/child_52_rerun/<stem>_<algo>_s<seed>
  prepare  writes outputs/child_52_rerun/rollouts/<stem>/{z.npy, index.json}: each run's distinct best z so far at every generation and its centre after 4, 8, 16, 32, 64 generations
  (run)    scripts/single_z_search.py --eval-z z.npy --eval-save-qpos on each motion
  analyse  metrics of each rollout with q[t] against ref[t + 1]; writes per_run.npz (best_<algo>: motions x seeds x 64 generations x metrics, mean_<algo>: motions x seeds x 5 x metrics, z0) and means.json. Checks: rollouts reproduce each search's best_so_far; settings match the old runs apart from the alignment; headstand_4 LOES matches outputs/loes_aligned.
usage: uv run python scripts/child_52_rerun.py prepare | analyse"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.metrics import metric_body   # noqa: E402  one Body per (xml, smpl) and process (refactor step 3)
SRC = REPO / "outputs/child_52"; OLD_CMA = REPO / "outputs/child_52_cmaes"; OUT = REPO / "outputs/child_52_rerun"; ROLL = OUT / "rollouts"; ALGOS = ("es", "cem", "random", "cmaes"); SEEDS = range(5); G = (4, 8, 16, 32, 64)
SAME = ("evals", "generations", "pairs", "sigma", "lr", "steps", "fitness", "sampling", "objective", "init_vel_scale", "heading_weight", "pos_weight", "pos_unit", "pos_div", "anchor_weight", "obs_scale", "init", "bfm_zero_vel", "fall_weight", "seed", "algo")
body = None


def clips():
    return [l.strip() for l in open(SRC / "clips.txt")]


def prepare():
    for c in clips():
        stem = c.split("/")[1]; d = ROLL / stem; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
        for a in ALGOS:
            for s in SEEDS:
                r = f"{stem}_{a}_s{s}"; t = np.load(OUT / r / "z_trace.npz"); b, m = t["z_best"], t["z_mean"]; assert len(b) == 65 and len(m) == 65, r; best = []
                for g in range(1, 65):
                    if not best or np.abs(b[g] - b[g - 1]).max() > 0:
                        Z.append(b[g]); best.append(len(Z) - 1)
                    else:
                        best.append(best[-1])
                centre = list(range(len(Z), len(Z) + len(G))); Z += [m[g] for g in G]; idx[r] = dict(best=best, centre=centre)
        np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, map=idx)))
    print("prepared")


def clip(c):
    global body
    import eval_metrics as E
    from child_52_means import full, KEYS
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]; d = ROLL / stem
    ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64); cost = [float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]; base = json.load(open(d / "eval_base.json"))["cost"]; V = {}
    def m(i):
        if i not in V:
            v = full(body, Q[i], ref, cost[i] / base); V[i] = [v[k] for k in KEYS]
        return V[i]
    B = {a: np.zeros((5, 64, len(KEYS))) for a in ALGOS}; C = {a: np.zeros((5, len(G), len(KEYS))) for a in ALGOS}; worst = 0.0; bad = []
    for a in ALGOS:
        for s in SEEDS:
            r = f"{stem}_{a}_s{s}"; old = json.load(open((OLD_CMA if a == "cmaes" else SRC) / r / "summary.json")); new = json.load(open(OUT / r / "summary.json")); bad += [(r, k, old.get(k), new.get(k)) for k in SAME if old.get(k) != new.get(k)]
            if new.get("frame_offset") != 1:
                bad.append((r, "frame_offset", new.get("frame_offset")))
            for j, i in enumerate(ix[r]["best"]):
                B[a][s, j] = m(i)
            for j, i in enumerate(ix[r]["centre"]):
                C[a][s, j] = m(i)
            bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(OUT / r / "curve.csv"))]); worst = max(worst, float(np.abs(np.array(cost)[ix[r]["best"]] - bs).max()))
    v = full(body, np.load(OUT / f"{stem}_es_s0/origin_z.npz")["qpos"].astype(np.float64), ref, 1.0)
    return stem, [v[k] for k in KEYS], B, C, worst, bad


def analyse():
    from child_52_means import KEYS
    cl = clips()
    with Pool(24) as p:
        res = p.map(clip, cl)
    z0 = np.array([r[1] for r in res]); B = {a: np.stack([r[2][a] for r in res]) for a in ALGOS}; C = {a: np.stack([r[3][a] for r in res]) for a in ALGOS}; bad = [x for r in res for x in r[5]]
    same = max(float(np.abs(np.load(OUT / f"headstand_4_es_s{s}/best_z.npy") - np.load(REPO / f"outputs/loes_aligned/headstand_4_s{s}/best_z.npy")).max()) for s in SEEDS)
    print("largest rollout vs search best_so_far:", max(r[4] for r in res), "| settings differing from the old runs:", len(bad), bad[:6], "| headstand_4 LOES against outputs/loes_aligned, largest z difference:", same, flush=True)
    np.savez_compressed(OUT / "per_run.npz", clips=np.array(cl), keys=np.array(KEYS), gens_mean_z=np.array(G), z0=z0, **{f"best_{a}": B[a] for a in ALGOS}, **{f"mean_{a}": C[a] for a in ALGOS})
    m = lambda A: {k: float(A[..., i].mean()) for i, k in enumerate(KEYS)}
    test = lambda X, Y: {k: dict(lower=int((X[..., i].mean(1) < Y[..., i].mean(1)).sum()), p=float(wilcoxon(X[..., i].mean(1), Y[..., i].mean(1)).pvalue)) for i, k in enumerate(KEYS) if k != "ratio"}
    rep = dict(z0=m(z0), best={a: {str(g * 16): m(B[a][:, :, g - 1]) for g in G} for a in ALGOS}, centre={a: {str(g * 16): m(C[a][:, :, j]) for j, g in enumerate(G)} for a in ALGOS},
               loes_vs=dict(best={a: test(B["es"][:, :, -1], B[a][:, :, -1]) for a in ALGOS[1:]}, centre={a: test(C["es"][:, :, -1], C[a][:, :, -1]) for a in ALGOS[1:]}))
    (OUT / "means.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    {"prepare": prepare, "analyse": analyse}[sys.argv[1]]()
