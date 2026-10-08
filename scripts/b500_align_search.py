#!/usr/bin/env python3
"""b500_align_search.py -- one search per clip, straight from z0 (no bfm stage), anchored to z0.

--clips b500: the b500 clips (outputs/b500_targets/sup_dataset); c540: the 540 of crossenbodiment-child-torque.
--objective align: L_align (losses.functional_equivalence); mse: losses.tracking_mse (pose + ee + root + heading);
            bfm_global: 1 - cos(B(s), B(g)) + 1.0 * (1 - cos dheading) / 2 + 0.1 * |root xy - ref| (m), the old
            two-stage "global" cost in one stage -- cosine alone cannot see heading or root x, y (the proprio is in the
            root's heading frame and has no x, y), so the bfm-only searches ended ~40 deg / ~1 m off.
cost = objective + w * (|z - z0| / 16)^2 (w = --anchor-weight, default 0.3), exact observations, rollouts from
the reference's first frame, 8 pairs = 16 rollouts per generation, --gens generations (default 64 = 1,024
rollouts, the budget of exact_train's L_align stage). Same ES settings as exact_train (sigma 0.05, lr 0.03).
  run      launch the searches (round-robin over --gpus, --max-par at a time) and wait;
           --per-cat N runs only N train clips per category (a pilot), finished clips are skipped
  report   convergence: fraction of the last generation's gain already reached at earlier generations
Writes <out>/<clip>/ (single_z_search.py output) and <out>/clips.txt.
"""
import argparse, csv, json, os, random, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "outputs/b500_targets/sup_dataset/index.jsonl"
C540 = REPO / "datasets/crossenbodiment-child-torque/manifest.jsonl"
PAIRS = 8


def group(task):
    for g in ("crawl", "headstand", "jump", "rotate", "move", "raisearms"):
        if task.startswith(g):
            return g
    return "ground"          # lieonground, sitonground, split, crouch


def clips(per_cat, source="b500"):
    if source == "c540":
        rows = [dict(task=m["reward_name"], trial=m["trial"], clip=f"{m['reward_name']}_{m['trial']}",
                     category=group(m["reward_name"]), split=m["task_split"]) for m in map(json.loads, open(C540))]
    else:
        rows = [json.loads(l) for l in open(INDEX)]
    if per_cat:
        by = defaultdict(list)
        for r in rows:
            if r["split"] == "train":
                by[r["category"]].append(r)
        rows = [r for c in sorted(by) for r in random.Random(0).sample(sorted(by[c], key=lambda r: r["clip"]), per_cat)]
    return rows


def run(args):
    out = REPO / args.out; log = REPO / "outputs/train_logs" / Path(args.out).name
    out.mkdir(parents=True, exist_ok=True); log.mkdir(parents=True, exist_ok=True)
    rows = clips(args.per_cat, args.clips)
    (out / "clips.txt").write_text("".join(f"{r['task']} {r['trial']} {r['category']} {r['split']}\n" for r in rows))
    gpus = args.gpus.split(",")
    procs, t0 = [], time.time()
    for i, r in enumerate(rows):
        t, stem = r["task"], r["clip"]
        if (out / stem / "summary.json").exists():
            continue
        n = len(np.load(REPO / f"data/child/retargeting_motion/{t}/{stem}.npz")["qpos"])
        z0 = f"data/origin_z/{t}/{stem}.npy"
        obj = ["--objective", "bfm", "--heading-weight", "1.0", "--pos-weight", "0.1"] if args.objective == "bfm_global" \
            else ["--objective", args.objective]
        cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{stem}", "--body", "child",
               "--init", "reference", "--steps", str(n), "--pairs", str(PAIRS), "--sigma", "0.05", "--lr", "0.03",
               "--seed", "0", "--obs-scale", "exact", *obj, "--evals", str(args.gens * 2 * PAIRS),
               "--anchor", z0, "--anchor-weight", str(args.anchor_weight), "--out", str(out / stem)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]}
        while sum(p.poll() is None for p in procs) >= args.max_par:
            time.sleep(5)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(log / f"{stem}.log", "w"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"=== {args.out} done ({(time.time() - t0) / 60:.1f} min, {len(procs)} runs, "
          f"{sum(p.returncode != 0 for p in procs)} failed) ===", flush=True)


def report(args):
    out = REPO / args.out
    at = [g for g in (8, 16, 32, 48, 64, 96, 128) if g <= args.gens]
    F, L, A = defaultdict(list), [], []
    for l in open(out / "clips.txt"):
        t, k, cat, split = l.split()
        d = out / f"{t}_{k}"
        if not (d / "summary.json").exists():
            continue
        s = json.load(open(d / "summary.json")); c0 = s["origin_z"]["cost"]
        best = np.minimum.accumulate([min(float(r["best_so_far"]), c0) for r in csv.DictReader(open(d / "curve.csv"))])
        if len(best) < args.gens:
            continue
        gain = c0 - best[args.gens - 1]
        fr = [(c0 - best[g - 1]) / gain if gain > 1e-9 else 1.0 for g in at]
        F[cat].append(fr); F["all"].append(fr)
        L.append((s["origin_z"]["align"], s["best"]["align"], s["origin_z"].get("mse", np.nan), s["best"].get("mse", np.nan),
                  s["origin_z"]["head"], s["best"]["head"], s["origin_z"]["pos"], s["best"]["pos"]))
        A.append(s["best"].get("gen", -1))
    print(f"{out.name}: {len(F['all'])} clips; fraction of the gen-{args.gens} gain (anchored cost) reached by gen "
          + " / ".join(map(str, at)))
    for cat in sorted(F, key=lambda c: (c == "all", c)):
        a = np.array(F[cat])
        print(f"  {cat:9s} n={len(a):3d} median " + " / ".join(f"{np.median(a[:, j]):4.0%}" for j in range(len(at)))
              + "   worst " + " / ".join(f"{a[:, j].min():4.0%}" for j in range(len(at))))
    L = np.array(L)
    print(f"  L_align z0 -> best: median {np.median(L[:, 0]):.3f} -> {np.median(L[:, 1]):.3f} "
          f"(best/z0 {np.median(L[:, 1] / L[:, 0]):.2f}); best found at gen median {np.median(A):.0f}")
    if not np.isnan(L[:, 2]).all():
        print(f"  mse     z0 -> best: median {np.nanmedian(L[:, 2]):.3f} -> {np.nanmedian(L[:, 3]):.3f} "
              f"(best/z0 {np.nanmedian(L[:, 3] / L[:, 2]):.2f})")
    print(f"  heading (1 - cos) / 2 z0 -> best: median {np.median(L[:, 4]):.3f} -> {np.median(L[:, 5]):.3f}; "
          f"root-xy distance {np.median(L[:, 6]):.2f} -> {np.median(L[:, 7]):.2f} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", default="outputs/b500_align_a03")
    ap.add_argument("--clips", default="b500", choices=["b500", "c540"])
    ap.add_argument("--objective", default="align", choices=["align", "mse", "bfm_global"])
    ap.add_argument("--anchor-weight", type=float, default=0.3)
    ap.add_argument("--gens", type=int, default=64)
    ap.add_argument("--per-cat", type=int, default=0, help="pilot: only N train clips per category")
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--max-par", type=int, default=24)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
