#!/usr/bin/env python3
"""child_52_mean_z.py -- Comparison A (outputs/child_52) if a search handed over its current point instead of the best z so far: rollouts of each run's mean z (z_trace.npz z_mean: ES's and CEM's centre; random search's is its best so far)
after 4, 8, 16, 32 and 64 generations, against the best z so far at the same step (outputs/child_52_curves). Also each generation's best candidate's cost (curve.csv gen_best; its z is not saved).
  prepare  writes outputs/child_52_mean_z/<stem>/{z.npy, index.json}; then scripts/single_z_search.py --eval-z on each motion (queue script)
  analyse  writes outputs/child_52_mean_z/report.json. usage: uv run python scripts/child_52_mean_z.py prepare | analyse"""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import eval_metrics as E                                                  # noqa: E402
SRC = REPO / "outputs/child_52"; CUR = REPO / "outputs/child_52_curves"; OUT = REPO / "outputs/child_52_mean_z"; G = (4, 8, 16, 32, 64); A = ("es", "cem", "random"); K = ("mpjpe", "mmpjpe", "mpjae")


def prepare():
    for c in (l.strip() for l in open(SRC / "clips.txt")):
        stem = c.split("/")[1]; d = OUT / stem; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
        for r in sorted(glob.glob(str(SRC / f"{stem}_*_s*"))):
            m = np.load(f"{r}/z_trace.npz")["z_mean"]; assert len(m) == 65, r; idx[Path(r).name] = list(range(len(Z), len(Z) + len(G))); Z += [m[g] for g in G]
        np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, map=idx)))


def success(q, qr):
    T = min(len(q), len(qr)); q, qr = q[:T], qr[:T]; pos = np.linalg.norm(q[:, :2] - qr[:, :2], axis=1); rot = np.degrees(2 * np.arccos(np.abs((q[:, 3:7] * qr[:, 3:7]).sum(1)).clip(0, 1))); return not ((pos > 1.5) | (rot > 45)).any()


def analyse():
    body = E.Body(E.XML["child"], True); clips = [l.strip() for l in open(SRC / "clips.txt")]; X = {a: {t: {g: {k: [] for k in K + ("ratio", "success")} for g in G} for t in ("mean", "best")} for a in A}; GB = {a: {g: [] for g in G} for a in A}
    for c in clips:
        task, stem = c.split("/"); ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); z0 = float(next(csv.DictReader(open(SRC / f"{stem}_z0/eval.csv")))["cost"])
        d = OUT / stem; ix = json.load(open(d / "index.json"))["map"]; Qm = np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64); Cm = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]); Vm = [dict(E.tracking_metrics(body, q, ref), success=success(q, ref), ratio=Cm[i] / z0) for i, q in enumerate(Qm)]
        e = CUR / stem; bx = json.load(open(e / "index.json"))["map"]; Qb = np.load(e / "eval_qpos.npz", mmap_mode=None)["qpos"]; Cb = np.array([float(r["cost"]) for r in csv.DictReader(open(e / "eval.csv"))]); cache = {}
        def best(i):
            if i not in cache:
                q = Qb[i].astype(np.float64); cache[i] = dict(E.tracking_metrics(body, q, ref), success=success(q, ref), ratio=Cb[i] / z0)
            return cache[i]
        for a in A:
            runs = sorted(r for r in ix if r.rsplit("_", 2)[1] == a); gb = {g: [] for g in G}
            for j, g in enumerate(G):
                vm = [Vm[ix[r][j]] for r in runs]; vb = [best(bx[r][g - 1]) for r in runs]
                for t, v in (("mean", vm), ("best", vb)):
                    for k in K + ("ratio",):
                        X[a][t][g][k].append(float(np.median([x[k] for x in v])))
                    X[a][t][g]["success"].append(sum(x["success"] for x in v))
            for r in runs:
                cur = list(csv.DictReader(open(SRC / r / "curve.csv")))
                for g in G:
                    gb[g].append(float(cur[g - 1]["gen_best"]) / z0)
            for g in G:
                GB[a][g].append(float(np.median(gb[g])))
        print(stem, flush=True)
    rep = {}
    for a in A:
        rep[a] = {}
        for g in G:
            m, b = X[a]["mean"][g], X[a]["best"][g]; rep[a][str(g * 16)] = dict(mean={k: float(np.median(m[k])) for k in K + ("ratio",)}, best={k: float(np.median(b[k])) for k in K + ("ratio",)}, success_mean=int(sum(m["success"])), success_best=int(sum(b["success"])), gen_best_ratio=float(np.median(GB[a][g])), mean_lower_than_best={k: int((np.array(m[k]) < np.array(b[k])).sum()) for k in K + ("ratio",)})
    for g in G:
        rep.setdefault("es_mean_vs_random_best", {})[str(g * 16)] = {k: dict(lower=int((np.array(X["es"]["mean"][g][k]) < np.array(X["random"]["best"][g][k])).sum()), p=float(wilcoxon(X["es"]["mean"][g][k], X["random"]["best"][g][k]).pvalue)) for k in K + ("ratio",)}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))


if __name__ == "__main__":
    {"prepare": prepare, "analyse": analyse}[sys.argv[1]]()
