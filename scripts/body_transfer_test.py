#!/usr/bin/env python3
"""body_transfer_test.py -- does the child's 8-dim correction basis also speed up the search on OTHER bodies?

Same search as fast_search_test.py (from the true z0, L_align, exact observations with each body's own root
scale, 8 pairs = 16 rollouts per generation, 10 generations, no anchor), on several bodies, in three spaces:
  child_k8  the child's 8-dim basis, the c60 leave-out one (outputs/fast_search_c60/basis_k8_leaveout.npy,
            built WITHOUT the clips tested here, so no clip's own child solution is in it)
  rand_k8   a random orthonormal 8-dim basis (control: is it the child's directions, or just being 8-dim?)
  full      all 256 dims
Clips: --per-cat per category from the c60 held-out list, among those every body has (0-9 of each motion). No body other than the child has a long reference
search, so gain is measured against the best L_align any arm reached on that body and clip in 10 generations:
  fraction = (L_align(z0) - best so far) / (L_align(z0) - best of the three arms).
  run      launch all searches (round-robin over --gpus) and wait
  report   print the table and write <out>/curve.png
Writes outputs/body_transfer/{clips.txt, basis_rand8.npy, <body>/<arm>/<clip>}.
"""
import argparse, csv, json, os, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/body_transfer"
LOG = REPO / "outputs/train_logs/body_transfer"
CHILD_BASIS = REPO / "outputs/fast_search_c60/basis_k8_leaveout.npy"
ARMS = ("child_k8", "rand_k8", "full")
GENS, PAIRS = 10, 8


def pick(per_cat, bodies):
    rows = [l.split() for l in open(REPO / "outputs/fast_search_c60/clips.txt") if l.strip()]
    # the other bodies only have clips 0-9 of each motion
    rows = [r for r in rows if all((REPO / f"data/{b}/retargeting_motion/{r[0]}/{r[0]}_{r[1]}.npz").exists() for b in bodies)]
    by = defaultdict(list)
    for r in rows:
        by[r[2]].append(r)
    return [r for c in sorted(by) for r in random.Random(0).sample(sorted(by[c]), per_cat)]


def run(args):
    OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    clips = pick(args.per_cat, args.bodies)
    (OUT / "clips.txt").write_text("".join(" ".join(r) + "\n" for r in clips))
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((256, 8)))
    np.save(OUT / "basis_rand8.npy", q.T)
    basis = {"child_k8": CHILD_BASIS, "rand_k8": OUT / "basis_rand8.npy"}
    jobs = [(b, arm, r) for b in args.bodies for r in clips for arm in ARMS]
    gpus = args.gpus.split(",")
    procs, t0 = [], time.time()
    for i, (body, arm, (t, k, cat, n)) in enumerate(jobs):
        stem = f"{t}_{k}"
        out = OUT / body / arm / stem
        if (out / "summary.json").exists():
            continue
        cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{stem}", "--body", body,
               "--init", "reference", "--steps", n, "--pairs", str(PAIRS), "--sigma", "0.05", "--lr", "0.03",
               "--seed", "0", "--best-from-start", "--obs-scale", "exact", "--objective", "align",
               "--evals", str(GENS * 2 * PAIRS), "--z-start", f"data/origin_z/{t}/{stem}.npy", "--out", str(out)]
        if arm in basis:
            cmd += ["--subspace", str(basis[arm]), "--subspace-dim", "8"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]}
        while sum(p.poll() is None for p in procs) >= args.max_par:
            time.sleep(5)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(LOG / f"{body}_{arm}_{stem}.log", "w"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"=== body_transfer done ({(time.time() - t0) / 60:.1f} min, {len(procs)} runs, "
          f"{sum(p.returncode != 0 for p in procs)} failed) ===", flush=True)


def curve(d):
    s = json.load(open(d / "summary.json")); z0 = s["origin_z"]["align"]
    best = np.minimum.accumulate([min(float(r["best_so_far"]), z0) for r in csv.DictReader(open(d / "curve.csv"))])
    return z0, np.concatenate([[z0], best[:GENS]])


def report(args):
    clips = [l.split() for l in open(OUT / "clips.txt") if l.strip()]
    at = [1, 2, 5, 10]
    F = {}
    print(f"fraction of the best gain (best of the 3 arms after {GENS} generations) reached after "
          f"{' / '.join(map(str, at))} generations, median over clips; L_align z0 and best / z0 also medians")
    for body in args.bodies:
        L0, R, frac = [], defaultdict(list), defaultdict(list)
        for t, k, cat, n in clips:
            stem = f"{t}_{k}"
            cs = {a: curve(OUT / body / a / stem) for a in ARMS if (OUT / body / a / stem / "summary.json").exists()}
            if len(cs) < len(ARMS):
                continue
            z0 = cs["full"][0]; ref = min(c[1][-1] for c in cs.values()); L0.append(z0)
            for a, (_, c) in cs.items():
                R[a].append(c[-1] / z0)
                frac[a].append((z0 - c) / (z0 - ref) if z0 - ref > 1e-9 else np.ones_like(c))
        if not L0:
            print(f"{body:13s} not finished"); continue
        F[body] = {a: np.array(v) for a, v in frac.items()}
        print(f"{body:13s} {len(L0)} clips, L_align z0 {np.median(L0):.3f}")
        for a in ARMS:
            m = np.median(F[body][a], 0)
            print(f"   {a:9s} " + " / ".join(f"{m[g]:4.0%}" for g in at) + f"   best/z0 {np.median(R[a]):.2f}")
    if F:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        col = {"child_k8": "#2a7de1", "rand_k8": "#9aa0aa", "full": "#e8590c"}
        fig, axs = plt.subplots(1, len(F), figsize=(3.2 * len(F), 3.2), sharey=True, squeeze=False)
        for ax, (body, fr) in zip(axs[0], F.items()):
            for a in ARMS:
                g = np.arange(GENS + 1)
                ax.plot(g, 100 * np.median(fr[a], 0), color=col[a], lw=2, label=a)
                ax.fill_between(g, 100 * np.percentile(fr[a], 25, 0), 100 * np.percentile(fr[a], 75, 0),
                                color=col[a], alpha=.12, lw=0)
            ax.set_title(f"{body} ({len(fr['full'])} clips)"); ax.set_xlabel("generation (16 rollouts)")
            ax.grid(alpha=.3)
        axs[0][0].set_ylabel("% of best gain"); axs[0][0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(OUT / "curve.png", dpi=130)
        print(f"wrote {OUT / 'curve.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--bodies", nargs="+", default=["child", "teen", "short_limbed", "elderly", "giant"])
    ap.add_argument("--per-cat", type=int, default=2, help="clips per category")
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--max-par", type=int, default=24)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
