#!/usr/bin/env python3
"""body_pca_transfer.py -- does the CHILD's 8-dim principal subspace also cut generations on other bodies?

Clips: --n of the 60 held-out clips of outputs/pca_subspace_bfmg (stratified by category, seed 0), so the
child basis (outputs/pca_subspace_bfmg/basis_pca.npy, 455 child clips) contains none of their child solutions.
Every body has clips 0-9 of every motion, so all of them exist everywhere.
Per (body, clip), from z0, cosine + 1.0 heading + 0.1 root-xy + 0.3 (|z - z0| / 16)^2, exact observations with
the body's own root scale, 8 pairs = 16 rollouts per generation, sigma 0.05, lr 0.03, seed 0, --eval-every 0
(no extra mean-z rollouts, so every arm spends exactly 16 rollouts per generation):
  pca8   the child's first 8 principal directions, 16 generations
  rand8  the same random 8-dim basis as the child experiment, 16 generations
  full   all 256 dims, 64 generations -- its first 16 are the full arm, its 64-gen best is the 100% reference
  fraction = (cost(z0) - best so far) / (cost(z0) - full's 64-gen best)
The child row of the report is the child experiment itself (outputs/pca_subspace_bfmg vs outputs/c540_bfmglobal_a03).
  run      launch (round-robin over --gpus) and wait;  report  table + <out>/curve.png
"""
import argparse, csv, json, os, random, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CHILD = REPO / "outputs/pca_subspace_bfmg"
CHILD_REF = REPO / "outputs/c540_bfmglobal_a03"
GENS = {"pca8": 16, "rand8": 16, "full": 64}
BASIS = {"pca8": CHILD / "basis_pca.npy", "rand8": CHILD / "basis_rand8.npy"}


def pick(n):
    rows = [l.split() for l in open(CHILD / "clips.txt") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r[2]].append(r)
    rnd, out, i = random.Random(0), [], 0
    pools = {c: rnd.sample(sorted(v), len(v)) for c, v in sorted(by.items())}
    while len(out) < n and any(pools.values()):                     # round-robin over categories
        c = sorted(pools)[i % len(pools)]; i += 1
        if pools[c]:
            out.append(pools[c].pop())
    return out


def run(args):
    out = REPO / args.out; log = REPO / "outputs/train_logs" / Path(args.out).name
    out.mkdir(parents=True, exist_ok=True); log.mkdir(parents=True, exist_ok=True)
    clips = pick(args.n)
    (out / "clips.txt").write_text("".join(" ".join(r) + "\n" for r in clips))
    jobs = [(b, a, r) for a in ("full", "pca8", "rand8") for b in args.bodies for r in clips]   # long runs first
    gpus, procs, t0 = args.gpus.split(","), [], time.time()
    for i, (body, arm, (t, k, cat, sp, s)) in enumerate(jobs):
        stem = f"{t}_{k}"; d = out / body / arm / stem
        if (d / "summary.json").exists():
            continue
        n = len(np.load(REPO / f"data/{body}/retargeting_motion/{t}/{stem}.npz")["qpos"])
        z0 = f"data/origin_z/{t}/{stem}.npy"
        cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{stem}", "--body", body,
               "--init", "reference", "--steps", str(n), "--pairs", "8", "--sigma", "0.05", "--lr", "0.03",
               "--seed", "0", "--obs-scale", "exact", "--objective", "bfm", "--heading-weight", "1.0",
               "--pos-weight", "0.1", "--evals", str(GENS[arm] * 16), "--eval-every", "0",
               "--anchor", z0, "--anchor-weight", "0.3", "--out", str(d)]
        if arm in BASIS:
            cmd += ["--subspace", str(BASIS[arm]), "--subspace-dim", "8"]
        while sum(p.poll() is None for p in procs) >= args.max_par:
            time.sleep(5)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]},
                                      stdout=open(log / f"{body}_{arm}_{stem}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"=== {args.out} done ({(time.time() - t0) / 60:.1f} min, {len(procs)} runs, "
          f"{sum(p.returncode != 0 for p in procs)} failed) ===", flush=True)


def curve(d, c0, cref, G=16):
    b = np.minimum.accumulate([min(float(r["best_so_far"]), c0) for r in csv.DictReader(open(d / "curve.csv"))])
    return 100 * np.concatenate([[0.0], (c0 - b[:G]) / (c0 - cref)])


def report(args):
    from scipy.stats import wilcoxon
    out = REPO / args.out
    clips = [l.split() for l in open(out / "clips.txt") if l.strip()]
    F = {}
    # child: the child experiment on the same clips
    F["child"] = defaultdict(list)
    for t, k, cat, sp, s in clips:
        st = f"{t}_{k}"; ref = json.load(open(CHILD_REF / st / "summary.json"))
        c0, cr = ref["origin_z"]["cost"], ref["best"]["cost"]
        if c0 - cr > 1e-9:
            for a in ("pca8", "rand8", "full"):
                F["child"][a].append(curve(CHILD / a / st, c0, cr))
    for body in args.bodies:
        F[body] = defaultdict(list)
        for t, k, cat, sp, s in clips:
            st = f"{t}_{k}"
            if not all((out / body / a / st / "summary.json").exists() for a in GENS):
                continue
            ref = json.load(open(out / body / "full" / st / "summary.json"))
            c0, cr = ref["origin_z"]["cost"], ref["best"]["cost"]
            if c0 - cr > 1e-9:
                for a in GENS:
                    F[body][a].append(curve(out / body / a / st, c0, cr))
    reach = lambda c, x: np.where((c >= x).any(1), (c >= x).argmax(1), 17)
    print("median % of the 64-gen full-space gain after 1 / 2 / 4 / 8 / 16 gens | mean gens to 80% (not reached = 17) | "
          "pca8 vs full and vs rand8 at gen 8: clips ahead, p")
    for body, Fb in F.items():
        if not Fb.get("full"):
            continue
        A = {a: np.array(v) for a, v in Fb.items()}; n = len(A["full"])
        print(f"{body} ({n} clips)")
        for a in ("pca8", "full", "rand8"):
            line = f"  {a:6s} " + " / ".join(f"{np.median(A[a][:, g]):3.0f}" for g in (1, 2, 4, 8, 16)) + f" | {reach(A[a], 80).mean():4.1f}"
            if a != "pca8":
                d = A["pca8"][:, 8] - A[a][:, 8]
                line += f" | pca8 ahead {np.sum(d > 0)}/{n}, median {np.median(d):+.0f} pts, p={wilcoxon(d).pvalue:.0e}"
            print(line)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bodies = [b for b in F if F[b].get("full")]
    fig, axs = plt.subplots(1, len(bodies), figsize=(3.3 * len(bodies), 3.6), sharey=True, squeeze=False)
    col = {"pca8": "#2a78d6", "full": "#eb6834", "rand8": "#8d8c88"}
    g = np.arange(17)
    for ax, body in zip(axs[0], bodies):
        for a in ("pca8", "full", "rand8"):
            c = np.array(F[body][a])
            ax.plot(g, np.median(c, 0), color=col[a], lw=2, ls=(0, (4, 3)) if a == "rand8" else "-",
                    label={"pca8": "child PCA 8-d", "full": "full 256-d", "rand8": "random 8-d"}[a])
            if a != "rand8":
                ax.fill_between(g, np.percentile(c, 25, 0), np.percentile(c, 75, 0), color=col[a], alpha=.10, lw=0)
        ax.axhline(80, color="#52514e", lw=1, alpha=.35)
        ax.set_title(f"{body} ({len(F[body]['full'])} clips)", fontsize=10); ax.set_xlabel("generation"); ax.grid(color="#e6e5e1", lw=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0][0].set_ylabel("% of 64-gen full-space gain"); axs[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "curve.png", dpi=140)
    print(f"wrote {out / 'curve.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", default="outputs/body_pca_transfer")
    ap.add_argument("--bodies", nargs="+", default=["teen", "short_limbed", "elderly", "giant"])
    ap.add_argument("--n", type=int, default=24, help="clips (of the child experiment's 60 held-out ones)")
    ap.add_argument("--gpus", default="1,3")
    ap.add_argument("--max-par", type=int, default=24)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
