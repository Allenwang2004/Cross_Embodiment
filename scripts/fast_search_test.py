#!/usr/bin/env python3
"""fast_search_test.py -- how much of the gain does an L_align search from the true z0 get in its first generations?

exact_train clips, --per-cat per category (default 2 = 12 clips; --tag writes outputs/fast_search_<tag>). Each searched straight from z0 (no bfm stage) on L_align, exact observations,
8 pairs (16 rollouts) per generation, 10 generations, no anchor, in two spaces:
  k8    the 8-dim exact-observation correction basis, recomputed WITHOUT these 12 clips (no leakage of their own
        solutions into the basis)
  full  all 256 dims
Gain is measured against the exact_train search of the same clip (bfm 1024 -> L_align 1024, anchor 1):
  fraction = (L_align(z0) - best L_align so far) / (L_align(z0) - exact_train's best L_align).
  run      launch the 24 searches (12 per GPU, --gpus) and wait
  report   print the table
Writes outputs/fast_search/{basis_k8_leaveout.npy, clips.txt, k8/<clip>, full/<clip>}.
"""
import argparse, csv, json, os, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/fast_search"
LOG = REPO / "outputs/train_logs/fast_search"
PER_CAT = 2
GENS, PAIRS = 10, 8


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def pick():
    rows = [l.split() for l in open(REPO / "outputs/exact_train/clips.txt") if l.strip()]
    rows = [r for r in rows if (REPO / f"outputs/exact_train/{r[0]}_{r[1]}/align/summary.json").exists()]
    by = defaultdict(list)
    for r in rows:
        by[r[2]].append(r)
    return [r for c in sorted(by) for r in random.Random(0).sample(sorted(by[c]), PER_CAT)]


def run(args):
    OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    clips = pick()
    (OUT / "clips.txt").write_text("".join(" ".join(r) + "\n" for r in clips))
    held = {f"{r[0]}_{r[1]}" for r in clips}
    idx = [json.loads(l) for l in open(REPO / "outputs/b500_targets/sup_dataset/index.jsonl")]
    D = [ld(REPO / f"outputs/exact_train/{r['clip']}/align/best_z.npy") - ld(REPO / r["origin_z"]) for r in idx
         if r["split"] == "train" and r["clip"] not in held and (REPO / f"outputs/exact_train/{r['clip']}/align/best_z.npy").exists()]
    _, _, Vt = np.linalg.svd(np.stack(D), full_matrices=False)
    np.save(OUT / "basis_k8_leaveout.npy", Vt[:8])
    full_basis = np.load(REPO / "outputs/lowdim_search/basis_corrPCA_train_exact.npy")[:8]
    print(f"basis from {len(D)} clips ({len(held)} held out); principal cosines vs the all-clip basis: "
          f"{np.linalg.svd(Vt[:8] @ full_basis.T, compute_uv=False).round(2)}", flush=True)
    gpus = args.gpus.split(",")
    procs = []
    for i, (t, k, cat, n) in enumerate(clips):
        stem = f"{t}_{k}"
        for arm in ("k8", "full"):
            cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{stem}", "--body", "child",
                   "--init", "reference", "--steps", n, "--pairs", str(PAIRS), "--sigma", "0.05", "--lr", "0.03",
                   "--seed", "0", "--best-from-start", "--obs-scale", "exact", "--objective", "align",
                   "--evals", str(GENS * 2 * PAIRS), "--z-start", f"data/origin_z/{t}/{stem}.npy",
                   "--out", str(OUT / arm / stem)]
            if arm == "k8":
                cmd += ["--subspace", str(OUT / "basis_k8_leaveout.npy"), "--subspace-dim", "8"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]}
            while sum(p.poll() is None for p in procs) >= args.max_par:
                time.sleep(5)
            procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(LOG / f"{arm}_{stem}.log", "w"),
                                          stderr=subprocess.STDOUT))
    t0 = time.time()
    for p in procs:
        p.wait()
    print(f"=== fast_search done ({(time.time() - t0) / 60:.1f} min, {sum(p.returncode != 0 for p in procs)} failed) ===", flush=True)


def report(args):
    clips = [l.split() for l in open(OUT / "clips.txt") if l.strip()]
    at = [1, 2, 5, 10]
    print(f"{'clip':34s} {'cat':9s} | L_align z0 -> exact_train best | fraction of that gain after 1 / 2 / 5 / 10 generations"
          f" (16 rollouts each), k8  ||  full 256")
    F = defaultdict(list)
    for t, k, cat, n in clips:
        stem = f"{t}_{k}"
        ref = json.load(open(REPO / f"outputs/exact_train/{stem}/align/summary.json"))
        row = f"{stem:34s} {cat:9s} |"
        z0 = None
        for arm in ("k8", "full"):
            d = OUT / arm / stem
            if not (d / "summary.json").exists():
                row += "  not finished  ||"; continue
            s = json.load(open(d / "summary.json")); z0 = s["origin_z"]["align"]
            best = np.minimum.accumulate([min(float(r["best_so_far"]), z0) for r in csv.DictReader(open(d / "curve.csv"))])
            gain = z0 - ref["best"]["align"]
            fr = [(z0 - best[g - 1]) / gain for g in at]
            F[arm].append(fr)
            row += ("" if arm == "k8" else "  ||") + "  " + " / ".join(f"{x:4.0%}" for x in fr)
        print(f"{row[:46]} {z0 if z0 is not None else float('nan'):.3f} -> {ref['best']['align']:.3f}       {row[46:]}")
    for arm in ("k8", "full"):
        a = np.array(F[arm])
        if len(a):
            print(f"{arm:5s} median " + " / ".join(f"{np.median(a[:, j]):.0%}" for j in range(len(at)))
                  + f"   | clips >= 50% after 2 generations: {(a[:, 1] >= 0.5).sum()}/{len(a)}, after 5: {(a[:, 2] >= 0.5).sum()}/{len(a)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--per-cat", type=int, default=2, help="clips per category")
    ap.add_argument("--tag", default="", help="outputs/fast_search<_tag>")
    ap.add_argument("--max-par", type=int, default=24)
    a = ap.parse_args()
    PER_CAT = a.per_cat
    if a.tag:
        OUT, LOG = Path(f"{OUT}_{a.tag}"), Path(f"{LOG}_{a.tag}")
    run(a) if a.cmd == "run" else report(a)
