#!/usr/bin/env python3
"""axis_subspace_test.py -- small test: search only the 8 ORIGINAL latent coordinates with the largest correction
energy, instead of 8 principal directions (each a mix of ~80-100 coordinates).

  axis8       the 8 coordinates with the largest sum of squared corrections over the 24 training bodies x 16 clips
              x 2 seeds of multibody_corr (the same data as the multi_same PCA basis)
  axis8_rand  8 random coordinates (seed 0), control
Cells: 4 of the analysis-E test bodies x the 8 continuation clips (one per motion kind), seed 0, the full-space
recipe with --subspace; compared with the existing full / multi_same / rand8 runs of the same cells.
  bases | search | report
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from multibody_corr import (BODIES, E_TEST, GENS, OUT, PAIRS, SEEDS, _corr, cell_dir, clips, ref_path,   # noqa: E402
                            run_jobs)

AX = OUT / "axis_test"
BODIES_T = ["child", "gl0500a0550", "gl1500a1400", "short_stocky"]
ARMS = ["axis8", "axis8_rand"]


def clips_t():
    cont = {l.strip() for l in open(REPO / "outputs/continuation_clips.txt") if l.strip()}
    return [c for c in clips() if c in cont]


def bases(a):
    AX.mkdir(parents=True, exist_ok=True)
    train = [b for b in BODIES if b not in E_TEST]
    D = np.array([d for c in clips() for b in train for s in SEEDS for d in [_corr(b, c, s)] if np.linalg.norm(d) > 1e-9])
    e = np.sum(D ** 2, 0)
    top = np.argsort(-e)[:8]
    rnd = np.random.default_rng(0).choice(256, 8, replace=False)
    I = np.eye(256)
    np.save(AX / "axis8.npy", I[top]); np.save(AX / "axis8_rand.npy", I[rnd])
    print(f"axis8 coordinates {top.tolist()} ({e[top].sum() / e.sum():.1%} of training energy); axis8_rand {rnd.tolist()}")


def search(a):
    jobs = []
    for arm in ARMS:
        for b in BODIES_T:
            for c in clips_t():
                n = len(np.load(ref_path(b, c))["qpos"])
                out = AX / arm / b / f"{c.split('/')[1]}_s0"
                cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", c, "--body", b,
                       "--init", "reference", "--steps", str(n), "--pairs", str(PAIRS), "--sigma", "0.05", "--lr", "0.03",
                       "--seed", "0", "--obs-scale", "exact", "--objective", "bfm", "--heading-weight", "1.0",
                       "--pos-weight", "0.1", "--evals", str(GENS * 2 * PAIRS), "--anchor", f"data/origin_z/{c}.npy",
                       "--anchor-weight", "0.3", "--subspace", str(AX / f"{arm}.npy"), "--subspace-dim", "8", "--out", str(out)]
                jobs.append((n, out, cmd))
    jobs.sort(key=lambda j: -j[0])
    print(f"axis test: {len(jobs)} searches")
    run_jobs([(o, c) for _, o, c in jobs], a, "summary.json")


def report(a, at=(2, 4, 8, 16, 32, 64)):
    def curve(d):
        b = {int(r["gen"]): float(r["best_so_far"]) for r in csv.DictReader(open(d / "curve.csv"))}
        return np.minimum.accumulate([b[i] for i in range(max(b) + 1)])
    P = {}
    for c in clips_t():
        st = c.split("/")[1]
        for b in BODIES_T:
            fd = cell_dir(b, c, 0); c0 = json.load(open(fd / "summary.json"))["origin_z"]["cost"]
            full = np.minimum(curve(fd), c0); gain = c0 - full[-1]
            if gain < 0.02 * c0:
                continue
            dirs = {"full": fd, "pca8 (multi_same)": OUT / "esearch/multi_same" / b / f"{st}_s0",
                    "rand8 (random rotation)": OUT / "esearch/rand8" / b / f"{st}_s0",
                    "axis8 (top coordinates)": AX / "axis8" / b / f"{st}_s0", "axis8_rand": AX / "axis8_rand" / b / f"{st}_s0"}
            for k, d in dirs.items():
                if (d / "summary.json").exists():
                    pct = 100 * (c0 - np.minimum(curve(d), c0)) / gain
                    P.setdefault(k, []).append([pct[g - 1] for g in at])
    print(f"% of the full search's 64-gen gain at generations {list(at)} (median over cells)")
    for k, v in P.items():
        print(f"  {k:24s} n {len(v):2d}  " + " / ".join(f"{x:5.1f}" for x in np.median(v, 0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bases", "search", "report"])
    ap.add_argument("--gpus", default="0"); ap.add_argument("--max-par", type=int, default=8)
    ap.add_argument("--min-free-mib", type=int, default=2500); ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--launch-pause", type=float, default=30)
    a = ap.parse_args()
    {"bases": bases, "search": search, "report": report}[a.cmd](a)
