#!/usr/bin/env python3
"""lsor_mpjae_test.py -- why does LSOR with our cost have a higher MPJAE (16.12) than LOES (12.23) and than LSOR with BFMTrack's cost (12.73)?
Hypothesis: the heading and position terms make the per-step latents correct the body back towards the reference's path and facing,
step by step, and those corrections add acceleration. Small sample: the 6 child task motions whose MPJAE rose most from lsor to
lsor_ours (outputs/lsor_mpjae_test/clips.txt), seeds 0-2, LSO with per-step ranks, 64 x 16, everything else as outputs/child_52_lso.
  existing  lsor (weights x 0) and lsor_ours (heading 1.0, position 0.1) from outputs/child_52_lso
  w0.5, w2  both weights x 0.5 / x 2                       -> prediction 1: MPJAE rises with the weights
  w0.01 ... w0.3  both weights x 0.01, 0.02, 0.05, 0.1, 0.2, 0.3      -> the user's follow-up: a weight that corrects the path without moving MPJPE
  head, pos heading 1.0 only / position 0.1 only           -> prediction 2: each term adds MPJAE
  (red-noise variants, beta 2, were dropped by the user: BFMTrack uses beta 1; the 20 that ran stay on disk, unused)
Per run, the best sequence after 64 iterations: MPJPE, MMPJPE, MPJAE (eval_metrics.tracking_metrics), PHC success (world-frame mean body
error never above 0.5 m adult-equivalent = 0.31 m on the child), root drift (mean root xy distance to the reference, body metres), mean
body acceleration (root-relative, as in MPJAE), latent change between steps |16 (z_t+1 - z_t)|. Also, on all 52 motions (lsor against
lsor_ours, 5 seeds), the rank correlation between the drop in root drift and the rise in MPJAE.
Writes outputs/lsor_mpjae_test/results.json. usage: uv run python scripts/lsor_mpjae_test.py"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.bodies import BODIES
from crossenbodiment.metrics import MetricSuite, drift, metric_body, world_positions
OUT = REPO / "outputs/lsor_mpjae_test"; OLD = REPO / "outputs/child_52_lso"; S = BODIES.scale("child")
COND = ("lsor", "w0.01", "w0.02", "w0.05", "w0.1", "w0.2", "w0.3", "w0.5", "lsor_ours", "w2", "head", "pos"); SEEDS = range(3)
WEIGHTS = {"lsor": "x 0", "w0.01": "x 0.01", "w0.02": "x 0.02", "w0.05": "x 0.05", "w0.1": "x 0.1", "w0.2": "x 0.2", "w0.3": "x 0.3", "w0.5": "x 0.5", "lsor_ours": "x 1", "w2": "x 2", "head": "heading only", "pos": "position only"}
body = None


def world(Q):
    """world positions of every body of the module's `body` (crossenbodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def body_acc(x):
    """mean |body acceleration| (m/s^2, root-relative positions in the path frame, as in MPJAE) of a metrics.Rollout"""
    p, _ = x.body.local(x.q)
    return float(np.linalg.norm(np.diff(p, n=2, axis=0) * 900, axis=-1).mean())


def stats(q, ref, Pw):
    """{mpjpe, mmpjpe, mpjae, success (PHC, 0.5 m x S), drift (m), acc} of the module's `body` (crossenbodiment.metrics.MetricSuite)"""
    return MetricSuite(body, ("mpjpe", "mmpjpe", "mpjae", "success", "drift", ("acc", body_acc)), S).reference(ref, Pw)(q)


def run_dir(stem, k, s):
    return (OLD if k in ("lsor", "lsor_ours") else OUT) / f"{stem}_{k}_s{s}"


def clip(c):
    global body
    import eval_metrics as E
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    Pw = world(ref); res = {}
    for k in COND:
        rows = []
        for s in SEEDS:
            d = run_dir(stem, k, s); v = stats(np.load(d / "rollouts.npz")["q_best_it64"], ref, Pw); z = np.load(d / "mu_trace.npz")["best"]
            v["dz"] = float(np.linalg.norm(16 * np.diff(z, axis=0), axis=1).mean()); rows.append(v)
        res[k] = rows
    return stem, res


def drift_vs_mpjae(c):
    global body
    import eval_metrics as E
    body = body or metric_body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    out = {}
    for k in ("lsor", "lsor_ours"):
        D, A = [], []
        for s in range(5):
            q = np.load(OLD / f"{stem}_{k}_s{s}/rollouts.npz")["q_best_it64"].astype(np.float64); T = min(len(q), len(ref))
            D.append(drift(q[:T], ref[:T])); A.append(E.tracking_metrics(body, q, ref)["mpjae"])
        out[k] = (float(np.mean(D)), float(np.mean(A)))
    return out


def main():
    cl = [l.strip() for l in open(OUT / "clips.txt") if l.strip()]
    with Pool(len(cl)) as p:
        res = dict(p.map(clip, cl))
    keys = ("mpjpe", "mmpjpe", "mpjae", "success", "drift", "acc", "dz")
    mean = {k: {m: float(np.mean([r[m] for st in res for r in res[st][k]])) for m in keys} for k in COND}
    motion = lambda k, m: np.array([np.mean([r[m] for r in res[st][k]]) for st in res])
    print(f"{len(cl)} motions x {len(SEEDS)} seeds, best sequence after 1,024 rollouts (LSO with ranks)")
    print(f"{'condition':12s} {'weights, noise':20s}" + "".join(f"{m:>9s}" for m in keys))
    for k in COND:
        print(f"{k:12s} {WEIGHTS[k]:20s}" + "".join(f"{mean[k][m] * (100 if m == 'success' else 1):9.2f}" for m in keys))
    tests = {}
    for a, b in (("lsor", "w0.01"), ("w0.01", "w0.02"), ("w0.02", "w0.05"), ("w0.05", "w0.1"), ("w0.1", "w0.2"), ("w0.2", "w0.3"), ("w0.3", "w0.5"), ("w0.5", "lsor_ours"), ("lsor_ours", "w2"), ("lsor", "head"), ("lsor", "pos")):
        x, y = motion(a, "mpjae"), motion(b, "mpjae"); tests[f"{a} -> {b}"] = dict(higher=int((y > x).sum()), of=len(x), diff=float((y - x).mean()))
    print("MPJAE, per motion (mean of seeds):", json.dumps(tests))
    from crossenbodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    all52 = read_clips(CHILD_52_TASK)
    with Pool(24) as p:
        dv = p.map(drift_vs_mpjae, all52)
    dd = np.array([x["lsor"][0] - x["lsor_ours"][0] for x in dv]); da = np.array([x["lsor_ours"][1] - x["lsor"][1] for x in dv]); rho = spearmanr(dd, da)
    print(f"52 motions, lsor -> lsor_ours: drift drop {dd.mean():.3f} m, MPJAE rise {da.mean():.2f}; Spearman rho(drift drop, MPJAE rise) = {rho.statistic:.2f}, p = {rho.pvalue:.2g}")
    (OUT / "results.json").write_text(json.dumps(dict(per_motion=res, mean=mean, mpjae_tests=tests, all52=dict(drift_drop=dd.tolist(), mpjae_rise=da.tolist(), rho=float(rho.statistic), p=float(rho.pvalue))), indent=1))


if __name__ == "__main__":
    main()
