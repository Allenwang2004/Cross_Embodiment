#!/usr/bin/env python3
"""pca_subspace_test.py -- does ES in the principal subspace of past corrections need fewer generations?

Basis: uncentred SVD of the corrections 16 z_best/|z_best| - 16 z0/|z0| of a c540 search (--src, 64 generations;
default the MSE one, outputs/c540_mse_a03), train-split clips only and WITHOUT any clip evaluated here.
Evaluated on two held-out sets:
  unseen_motion  --per-test clips of each test-split motion (motions no basis clip comes from)
  unseen_clip    --per-cat train-split clips per category (their motion is in the basis, the clip is not)
Each clip is searched from z0 exactly like the --src run (its --objective: mse, or bfm_global = cosine +
1.0 heading + 0.1 root-xy; + 0.3 (|z - z0| / 16)^2, exact observations,
8 pairs = 16 rollouts per generation, sigma 0.05, lr 0.03, seed 0) for --gens generations, in:
  pca<k>   the first k principal directions, k in --ks
  rand<k>  a random orthonormal k-dim basis (control: is it the principal directions, or just fewer dims?)
  full     all 256 dims
Subspace steps are scaled by sqrt(256 / k) (single_z_search.py), so every arm moves the same Euclidean
distance per step; any speed-up comes from the directions, not a larger step.
Gain is measured against the clip's own 64-generation full-space --src result (same cost):
  fraction = (cost(z0) - best so far) / (cost(z0) - c540 best)     (can exceed 100%)
  run      build the bases, launch the searches (round-robin over --gpus) and wait
  report   table per set and arm, and <out>/curve.png
Writes <out>/{clips.txt, basis_pca.npy, basis_rand<k>.npy, <arm>/<clip>/}.
"""
import argparse, csv, json, os, random, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs/c540_mse_a03"
PAIRS = 8
OBJ = {"mse": ["--objective", "mse"],
       "bfm_global": ["--objective", "bfm", "--heading-weight", "1.0", "--pos-weight", "0.1"]}


def unit16(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return 16 * v / np.linalg.norm(v)


def src_clips():
    return [l.split() for l in open(SRC / "clips.txt") if l.strip()]       # task trial category split


def pick(per_test, per_cat):
    rows = [r for r in src_clips() if (SRC / f"{r[0]}_{r[1]}" / "summary.json").exists()]
    by_task, by_cat = defaultdict(list), defaultdict(list)
    for r in rows:
        (by_task[r[0]] if r[3] == "test" else by_cat[r[2]]).append(r)
    rnd = random.Random(0)
    out = [(*r, "unseen_motion") for t in sorted(by_task) for r in rnd.sample(sorted(by_task[t]), min(per_test, len(by_task[t])))]
    out += [(*r, "unseen_clip") for c in sorted(by_cat) for r in rnd.sample(sorted(by_cat[c]), min(per_cat, len(by_cat[c])))]
    return out


def arms(ks, rand_ks):
    return [f"pca{k}" for k in ks] + [f"rand{k}" for k in rand_ks] + ["full"]


def run(args):
    out = REPO / args.out; log = REPO / "outputs/train_logs" / Path(args.out).name
    out.mkdir(parents=True, exist_ok=True); log.mkdir(parents=True, exist_ok=True)
    ev = pick(args.per_test, args.per_cat)
    (out / "clips.txt").write_text("".join(" ".join(r) + "\n" for r in ev))
    held = {f"{r[0]}_{r[1]}" for r in ev}
    D = [unit16(np.load(SRC / f"{t}_{k}" / "best_z.npy")) - unit16(np.load(REPO / f"data/origin_z/{t}/{t}_{k}.npy"))
         for t, k, c, sp in src_clips()
         if sp == "train" and f"{t}_{k}" not in held and (SRC / f"{t}_{k}" / "summary.json").exists()]
    _, S, Vt = np.linalg.svd(np.stack(D), full_matrices=False)
    np.save(out / "basis_pca.npy", Vt)
    e = np.cumsum(S ** 2) / np.sum(S ** 2)
    print(f"basis from {len(D)} train clips ({len(held)} evaluated clips left out); energy in the top "
          + ", ".join(f"{k}: {e[k - 1]:.0%}" for k in args.ks), flush=True)
    basis = {f"pca{k}": (out / "basis_pca.npy", k) for k in args.ks}
    for k in args.rand_ks:
        q, _ = np.linalg.qr(np.random.default_rng(k).standard_normal((256, k)))
        np.save(out / f"basis_rand{k}.npy", q.T); basis[f"rand{k}"] = (out / f"basis_rand{k}.npy", k)
    gpus, procs, t0 = args.gpus.split(","), [], time.time()
    jobs = [(r, a) for r in ev for a in arms(args.ks, args.rand_ks)]
    for i, ((t, k, cat, sp, s), arm) in enumerate(jobs):
        stem = f"{t}_{k}"; d = out / arm / stem
        if (d / "summary.json").exists():
            continue
        n = len(np.load(REPO / f"data/child/retargeting_motion/{t}/{stem}.npz")["qpos"])
        z0 = f"data/origin_z/{t}/{stem}.npy"
        cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{stem}", "--body", "child",
               "--init", "reference", "--steps", str(n), "--pairs", str(PAIRS), "--sigma", "0.05", "--lr", "0.03",
               "--seed", "0", "--obs-scale", "exact", *OBJ[args.objective], "--evals", str(args.gens * 2 * PAIRS),
               "--anchor", z0, "--anchor-weight", "0.3", "--out", str(d)]
        if arm in basis:
            cmd += ["--subspace", str(basis[arm][0]), "--subspace-dim", str(basis[arm][1])]
        while sum(p.poll() is None for p in procs) >= args.max_par:
            time.sleep(5)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]},
                                      stdout=open(log / f"{arm}_{stem}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"=== {args.out} done ({(time.time() - t0) / 60:.1f} min, {len(procs)} runs, "
          f"{sum(p.returncode != 0 for p in procs)} failed) ===", flush=True)


def report(args):
    out = REPO / args.out
    ev = [l.split() for l in open(out / "clips.txt") if l.strip()]
    A = arms(args.ks, args.rand_ks)
    at = [g for g in (1, 2, 4, 8, 16, 32) if g <= args.gens]
    F = defaultdict(list)                                 # (set, arm) -> (n_clips, gens + 1) fraction curves
    for t, k, cat, sp, s in ev:
        stem = f"{t}_{k}"
        ref = json.load(open(SRC / stem / "summary.json"))
        c0, cref = ref["origin_z"]["cost"], ref["best"]["cost"]
        if not all((out / a / stem / "summary.json").exists() for a in A) or c0 - cref < 1e-9:
            continue
        for a in A:
            b = np.minimum.accumulate([min(float(r["best_so_far"]), c0) for r in csv.DictReader(open(out / a / stem / "curve.csv"))])
            f = np.concatenate([[0.0], (c0 - b[: args.gens]) / (c0 - cref)])
            F[(s, a)].append(f); F[("all", a)].append(f)
    for s in ("unseen_motion", "unseen_clip", "all"):
        if not F.get((s, "full")):
            continue
        n = len(F[(s, "full")])
        print(f"\n{s} ({n} clips): median % of the 64-gen full-space gain after {' / '.join(map(str, at))} generations"
              f"   | generations to 50% / 90% (median; never = {args.gens + 1})")
        for a in A:
            c = np.array(F[(s, a)])
            reach = [np.where((c >= x).any(1), (c >= x).argmax(1), args.gens + 1) for x in (.5, .9)]
            print(f"  {a:7s} " + " / ".join(f"{np.median(c[:, g]):4.0%}" for g in at)
                  + f"   | {np.median(reach[0]):4.0f} / {np.median(reach[1]):4.0f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sets = [s for s in ("unseen_motion", "unseen_clip") if F.get((s, "full"))]
    fig, axs = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 4), sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, s in zip(axs[0], sets):
        g = np.arange(args.gens + 1)
        for i, a in enumerate(A):
            c = 100 * np.median(np.array(F[(s, a)]), 0)
            style = dict(color="#e8590c", lw=2.5) if a == "full" else \
                dict(color="#9aa0aa", lw=1.5, ls="--") if a.startswith("rand") else dict(color=cmap(i / max(len(args.ks), 1)), lw=2)
            ax.plot(g, c, label=a, **style)
        ax.set_title(f"{s} ({len(F[(s, 'full')])} clips)"); ax.set_xlabel("generation (16 rollouts)"); ax.grid(alpha=.3)
    axs[0][0].set_ylabel("% of the 64-gen full-space gain (median)"); axs[0][0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "curve.png", dpi=130)
    print(f"\nwrote {out / 'curve.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", default="outputs/pca_subspace")
    ap.add_argument("--src", default="outputs/c540_mse_a03", help="the 64-gen c540 search: basis and reference")
    ap.add_argument("--objective", default="mse", choices=sorted(OBJ), help="must match --src")
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--rand-ks", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--gens", type=int, default=16)
    ap.add_argument("--per-test", type=int, default=5, help="clips per test-split motion")
    ap.add_argument("--per-cat", type=int, default=5, help="train-split clips per category")
    ap.add_argument("--gpus", default="1,3")
    ap.add_argument("--max-par", type=int, default=24)
    a = ap.parse_args()
    SRC = REPO / a.src
    run(a) if a.cmd == "run" else report(a)
