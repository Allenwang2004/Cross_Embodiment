#!/usr/bin/env python3
"""multibody_pca_test.py -- can principal directions shared ACROSS bodies make the search on an unseen body short?

Leave-one-body-out over child, teen, short_limbed, elderly, giant. Every search uses the c540 cosine + heading
recipe: from z0, bfm + --heading-weight 1.0 --pos-weight 0.1,
+ 0.3 (|z - z0| / 16)^2, exact observations with the body's own root scale, 8 pairs, sigma 0.05, lr 0.03, seed 0).
  search   stage 1: 64-generation full-space searches on --n-basis train clips per non-child body (the same clips
           on every body; none of the 60 held-out clips of outputs/pca_subspace_bfmg). The child's come from
           outputs/c540_bfmglobal_a03 (same recipe), so it needs no new runs.
  bases    stage 2: per body, its corrections 16 z*/|z*| - 16 z0/|z0| (Euclidean); for each held-out body
           pooled = uncentred SVD of the OTHER bodies' corrections (equal clips per body), own = its own; prints how
           much of each body's corrections the others' basis captures and the principal cosines between bodies.
  eval     stage 3: on each held-out body, the 24 clips of outputs/body_pca_transfer, 64 generations in
           pooled k = 8, 16, 32 and own k = 8 (--eval-every 0: exactly 16 rollouts per generation).
  report   % of that body's 64-generation full-space gain (outputs/body_pca_transfer/<body>/full, or
           outputs/c540_bfmglobal_a03 for the child) reached at generations 4 / 8 / 16 / 32 / 64, per arm.
Writes outputs/multibody_pca/{basis_clips.txt, search/<body>/<clip>, bases/<held>_{pooled,own}.npy, eval/<held>/<arm>/<clip>}.
"""
import argparse, csv, json, os, random, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/multibody_pca"
LOG = REPO / "outputs/train_logs/multibody_pca"
BODIES = ["child", "teen", "short_limbed", "elderly", "giant"]
CHILD_SRC = REPO / "outputs/c540_bfmglobal_a03"
HELD60 = REPO / "outputs/pca_subspace_bfmg/clips.txt"
EVAL24 = REPO / "outputs/body_pca_transfer/clips.txt"
ARMS = {"pooled8": ("pooled", 8), "pooled16": ("pooled", 16), "pooled32": ("pooled", 32), "own8": ("own", 8)}
RECIPE = ["--init", "reference", "--pairs", "8", "--sigma", "0.05", "--lr", "0.03", "--seed", "0", "--obs-scale", "exact",
          "--objective", "bfm", "--heading-weight", "1.0", "--pos-weight", "0.1", "--anchor-weight", "0.3"]


def unit16(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return 16 * v / np.linalg.norm(v)


def basis_clips(n):
    held = {f"{l.split()[0]}_{l.split()[1]}" for l in open(HELD60) if l.strip()}
    rows = [l.split() for l in open(CHILD_SRC / "clips.txt") if l.strip()]          # task trial category split
    rows = [r for r in rows if r[3] == "train" and f"{r[0]}_{r[1]}" not in held]
    rare = [r for r in rows if r[2] in ("headstand", "jump")]          # 10 clips each in the 540: take every one left
    rest = [r for r in rows if r[2] not in ("headstand", "jump")]
    return sorted(rare + random.Random(0).sample(rest, n - len(rare)))


def launch(jobs, args, tag):
    LOG.mkdir(parents=True, exist_ok=True)
    gpus, procs, t0 = args.gpus.split(","), [], time.time()
    for i, (out, cmd) in enumerate(jobs):
        if (out / "summary.json").exists():
            continue
        while sum(p.poll() is None for p in procs) >= args.max_par:
            time.sleep(5)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]},
                                      stdout=open(LOG / f"{tag}_{out.parent.name}_{out.name}.log", "w"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"=== {tag}: {len(procs)} runs in {(time.time() - t0) / 60:.1f} min, "
          f"{sum(p.returncode != 0 for p in procs)} failed ===", flush=True)


def ref_len(body, t, k):
    return str(len(np.load(REPO / f"data/{body}/retargeting_motion/{t}/{t}_{k}.npz")["qpos"]))


def search(args):
    OUT.mkdir(parents=True, exist_ok=True)
    clips = basis_clips(args.n_basis)
    (OUT / "basis_clips.txt").write_text("".join(" ".join(r) + "\n" for r in clips))
    jobs = []
    for body in BODIES[1:]:
        for t, k, cat, sp in clips:
            z0 = f"data/origin_z/{t}/{t}_{k}.npy"
            out = OUT / "search" / body / f"{t}_{k}"
            jobs.append((out, ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{t}_{k}", "--body", body,
                               "--steps", ref_len(body, t, k), *RECIPE, "--anchor", z0, "--evals", str(64 * 16), "--out", str(out)]))
    launch(jobs, args, "search")


def corrections(body):
    D = []
    for l in open(OUT / "basis_clips.txt"):
        t, k, cat, sp = l.split()
        d = (CHILD_SRC if body == "child" else OUT / "search" / body) / f"{t}_{k}"
        if (d / "summary.json").exists():
            D.append(unit16(np.load(d / "best_z.npy")) - unit16(np.load(REPO / f"data/origin_z/{t}/{t}_{k}.npy")))
    D = np.stack(D)
    return D[np.linalg.norm(D, axis=1) > 1e-9]


def bases(args):
    (OUT / "bases").mkdir(parents=True, exist_ok=True)
    C = {b: corrections(b) for b in BODIES}
    n = min(len(v) for v in C.values())
    print(f"corrections per body: {', '.join(f'{b} {len(v)}' for b, v in C.items())}; pooled bases use {n} per body")
    V = {}
    for b in BODIES:
        _, _, Vt = np.linalg.svd(C[b], full_matrices=False); V[b] = Vt
        np.save(OUT / "bases" / f"{b}_own.npy", Vt)
        P = np.concatenate([C[o][:n] for o in BODIES if o != b])
        _, _, Pt = np.linalg.svd(P, full_matrices=False)
        np.save(OUT / "bases" / f"{b}_pooled.npy", Pt)
        cap = lambda B, k: np.sum((C[b] @ B[:k].T) ** 2) / np.sum(C[b] ** 2)
        print(f"{b:13s} share of its own corrections inside: own k=8 {cap(Vt, 8):.0%} | pooled-others k=8 {cap(Pt, 8):.0%}, "
              f"16 {cap(Pt, 16):.0%}, 32 {cap(Pt, 32):.0%} | child-only k=8 {cap(V.get('child', Vt), 8):.0%}")
    print("principal cosines of the 8-dim own bases (mean of the 8):")
    for i, a in enumerate(BODIES):
        print("  " + a.ljust(13) + " ".join(f"{np.linalg.svd(V[a][:8] @ V[b][:8].T, compute_uv=False).mean():.2f}" for b in BODIES))


def evaluate(args):
    clips = [l.split() for l in open(EVAL24) if l.strip()]
    jobs = []
    for body in BODIES:
        for arm, (kind, k) in ARMS.items():
            for t, kk, *_ in clips:
                z0 = f"data/origin_z/{t}/{t}_{kk}.npy"
                out = OUT / "eval" / body / arm / f"{t}_{kk}"
                jobs.append((out, ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{t}_{kk}", "--body", body,
                                   "--steps", ref_len(body, t, kk), *RECIPE, "--anchor", z0, "--evals", str(64 * 16),
                                   "--eval-every", "0", "--subspace", str(OUT / "bases" / f"{body}_{kind}.npy"),
                                   "--subspace-dim", str(k), "--out", str(out)]))
    jobs.sort(key=lambda j: int(j[1][j[1].index("--steps") + 1]), reverse=True)      # long clips first
    launch(jobs, args, "eval")


def curve(d, c0, cref, G=64):
    b = np.minimum.accumulate([min(float(r["best_so_far"]), c0) for r in csv.DictReader(open(d / "curve.csv"))])
    b = np.concatenate([b, np.full(max(0, G - len(b)), b[-1])])
    return 100 * np.concatenate([[0.0], (c0 - b[:G]) / (c0 - cref)])


def report(args):
    from scipy.stats import wilcoxon
    clips = [l.split() for l in open(EVAL24) if l.strip()]
    at = [4, 8, 16, 32, 64]
    print(f"% of the body's own 64-generation full-space gain at generations {' / '.join(map(str, at))} (median over clips)")
    for body in BODIES:
        F = defaultdict(list)
        for t, k, *_ in clips:
            st = f"{t}_{k}"
            refd = (CHILD_SRC if body == "child" else REPO / "outputs/body_pca_transfer" / body / "full") / st
            if not (refd / "summary.json").exists():
                continue
            ref = json.load(open(refd / "summary.json")); c0, cr = ref["origin_z"]["cost"], ref["best"]["cost"]
            if c0 - cr < 1e-9 or not all((OUT / "eval" / body / a / st / "summary.json").exists() for a in ARMS):
                continue
            F["full"].append(curve(refd, c0, cr))
            for a in ARMS:
                F[a].append(curve(OUT / "eval" / body / a / st, c0, cr))
        if not F:
            print(f"{body}: not finished"); continue
        n = len(F["full"]); print(f"{body} ({n} clips)")
        for a in ["full", *ARMS]:
            c = np.array(F[a]); line = f"  {a:9s} " + " / ".join(f"{np.median(c[:, g]):4.0f}" for g in at)
            if a != "full":
                d = c[:, 16] - np.array(F["full"])[:, 16]
                line += f"   vs full at gen 16: ahead on {np.sum(d > 0)}/{n}, p={wilcoxon(d).pvalue:.0e}"
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "bases", "eval", "report"])
    ap.add_argument("--n-basis", type=int, default=100, help="train clips per body for the bases")
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--max-par", type=int, default=16)
    a = ap.parse_args()
    {"search": search, "bases": bases, "eval": evaluate, "report": report}[a.cmd](a)
