#!/usr/bin/env python3
"""multibody_corr.py -- are the latent corrections of different bodies really related, and low-dimensional?

Design (fixed before any result):
  bodies  32 existing bodies, all built by the same pipeline (assets/robots_torque/<body>, data/<body>):
            grid   15  leg x arm lattice, legs 0.5 / 0.75 / 1 / 1.25 / 1.5 x arms 0.55 / 1 / 1.4; its centre is the
                       adult rebuilt through the pipeline (m2c_t000), the reference body
            axis    6  the shrinking end of each other beta dim (torso, head, leg / arm / torso / head girth)
            named  11  child, teen, petite, tall_slim, long_limbed, short_limbed, athletic, elderly, pear_shaped,
                       giant, short_stocky
  clips   16  the 8 motion kinds present on every body x 2 trials: the morphology study's continuation trial
              (outputs/continuation_clips.txt) + one more trial per kind drawn with Random(0)
  seeds   0, 1 per (body, clip): two independent searches measure the search-noise floor
  search  the c540 cosine + heading recipe (as outputs/c540_bfmglobal_a03): from z0, exact observations, 8 pairs,
          sigma 0.05, lr 0.03, bfm + 1.0 heading + 0.1 xy + 0.3 (|z - z0| / 16)^2, 64 generations
Corrections are Euclidean, |z| = 16: D(b, m, s) = 16 z*/|z*| - 16 z0/|z0|; the body-induced part is
d(b, m, s) = D(b, m, s) - mean_s D(adult, m, s).
  plan     write outputs/multibody_corr/{bodies.tsv, clips.txt} and check every input file exists
  search   run the searches (skips finished cells; failed cells are retried up to --retries passes)
  status   finished / missing cells per body
  transfer functional test: every body's z* rolled out on every body (see transfer())
  ebases / esearch   analysis E: 8-dim subspace search on 8 test bodies with 4 bases (see ebases())
Writes outputs/multibody_corr/search/<body>/<clip>_s<seed>/ (single_z_search.py output) and transfer/.
"""
import argparse, json, os, random, subprocess, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/multibody_corr"
LOG = REPO / "outputs/train_logs/multibody_corr"
ADULT = "m2c_t000"
GRID = [["gl0500a0550", "ax0m_t1000", "gl0500a1400"],
        ["gl0750a0550", "ax0m_t500", "gl0750a1400"],
        ["ax1m_t1000", ADULT, "ax1p_t1000"],
        ["gl1250a0550", "ax0p_t500", "gl1250a1400"],
        ["gl1500a0550", "ax0p_t1000", "gl1500a1400"]]
AXIS = ["ax2m_t1000", "ax3m_t1000", "ax4m_t1000", "ax5m_t1000", "ax6m_t1000", "ax7m_t1000"]
NAMED = ["child", "teen", "petite", "tall_slim", "long_limbed", "short_limbed", "athletic", "elderly",
         "pear_shaped", "giant", "short_stocky"]
GROUPS = {**{b: "grid" for row in GRID for b in row}, **{b: "axis" for b in AXIS}, **{b: "named" for b in NAMED}}
BODIES = list(GROUPS)
BETA_KEYS = ["leg_scale", "arm_scale", "torso_scale", "head_scale", "leg_girth", "arm_girth", "torso_girth", "head_girth"]
SEEDS = [0, 1]
GENS, PAIRS = 64, 8
N_RANDOM = 16


def beta(body):
    p = json.load(open(REPO / f"assets/robots/{body}/parameter.json"))
    return [float(p[k]) for k in BETA_KEYS]


def choose_clips():
    cont = [l.strip() for l in open(REPO / "outputs/continuation_clips.txt") if l.strip()]
    rng, clips = random.Random(0), []
    for c in sorted(cont):
        kind, stem = c.split("/")
        k0 = int(stem.rsplit("_", 1)[1])
        k1 = rng.choice([k for k in range(10) if k != k0])
        clips += [c, f"{kind}/{kind}_{k1}"]
    return clips


def clips():
    return [l.strip() for l in open(OUT / "clips.txt") if l.strip()]


def ref_path(body, clip):
    return REPO / f"data/{body}/retargeting_motion/{clip}.npz"


def cell_dir(body, clip, seed):
    return OUT / "search" / body / f"{clip.split('/')[1]}_s{seed}"


def plan(args):
    OUT.mkdir(parents=True, exist_ok=True)
    cs = choose_clips()
    (OUT / "clips.txt").write_text("".join(c + "\n" for c in cs))
    rows = ["body\tgroup\t" + "\t".join(BETA_KEYS)]
    missing = []
    for b in BODIES:
        rows.append(f"{b}\t{GROUPS[b]}\t" + "\t".join(f"{v:g}" for v in beta(b)))
        if not (REPO / f"assets/robots_torque/{b}/robot_torque_full.xml").exists():
            missing.append(f"{b}: torque xml")
        missing += [f"{b}: {c}" for c in cs if not ref_path(b, c).exists()]
    missing += [c for c in cs if not (REPO / f"data/origin_z/{c}.npy").exists()]
    (OUT / "bodies.tsv").write_text("\n".join(rows) + "\n")
    B = np.array([beta(b) for b in BODIES])
    dup = [(BODIES[i], BODIES[j]) for i in range(len(B)) for j in range(i) if np.allclose(B[i], B[j])]
    print(f"{len(BODIES)} bodies ({', '.join(f'{g} {sum(v == g for v in GROUPS.values())}' for g in ('grid', 'axis', 'named'))}), "
          f"{len(cs)} clips, seeds {SEEDS} -> {len(BODIES) * len(cs) * len(SEEDS)} searches")
    print("clips:", " ".join(c.split("/")[1] for c in cs))
    print("duplicate betas:", dup or "none", "| missing inputs:", missing or "none")


def jobs_for(bodies, cs, seeds):
    jobs = []
    for b in bodies:
        for c in cs:
            n = len(np.load(ref_path(b, c))["qpos"])
            for s in seeds:
                out = cell_dir(b, c, s)
                cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", c, "--body", b,
                       "--init", "reference", "--steps", str(n), "--pairs", str(PAIRS), "--sigma", "0.05", "--lr", "0.03",
                       "--seed", str(s), "--obs-scale", "exact", "--objective", "bfm", "--heading-weight", "1.0",
                       "--pos-weight", "0.1", "--evals", str(GENS * 2 * PAIRS), "--anchor", f"data/origin_z/{c}.npy",
                       "--anchor-weight", "0.3", "--out", str(out)]
                jobs.append((n, out, cmd))
    jobs.sort(key=lambda j: -j[0])                      # long clips first
    return [(o, c) for _, o, c in jobs]


def free_mib(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def search(args):
    bodies = args.bodies.split(",") if args.bodies else BODIES
    cs = [c for c in clips() if not args.clips or c.split("/")[1] in args.clips.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    run_jobs(jobs_for(bodies, cs, seeds), args, "summary.json")


def run_jobs(jobs, args, done_file):
    """Launch (out_dir, cmd) jobs on args.gpus, at most args.max_par at a time and only while the GPU has
    args.min_free_mib free; a job is done when out_dir/done_file exists; failed jobs get args.retries more passes."""
    LOG.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    for p_i in range(1 + args.retries):
        todo = [(o, c) for o, c in jobs if not (o / done_file).exists()]
        if not todo:
            break
        print(f"== pass {p_i + 1}: {len(todo)} of {len(jobs)} runs, max-par {args.max_par} ({time.ctime()})", flush=True)
        procs, t0 = [], time.time()
        for i, (out, cmd) in enumerate(todo):
            g = gpus[i % len(gpus)]
            # GPU 0 is shared: wait for a free slot AND enough free memory (a search needs ~1.8 GB); the pause
            # after each launch (--launch-pause) lets the new process allocate before the next memory reading
            while sum(p.poll() is None for p in procs) >= args.max_par or free_mib(g) < args.min_free_mib:
                time.sleep(10)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": g}
            procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(LOG / f"{out.parent.name}_{out.name}.log", "w"),
                                          stderr=subprocess.STDOUT))
            time.sleep(args.launch_pause)
        for p in procs:
            p.wait()
        print(f"   pass {p_i + 1}: {len(procs)} runs in {(time.time() - t0) / 60:.1f} min, "
              f"{sum(p.returncode != 0 for p in procs)} failed", flush=True)
    done = sum((o / done_file).exists() for o, _ in jobs)
    print(f"=== done {done}/{len(jobs)} ({time.ctime()})", flush=True)


def transfer(args):
    """Functional test: every body's searched latents rolled out on every other body, same clip, same cost.
    Per (target body, clip): the 32 x 2 searched z* of that clip (own two seeds included: they must reproduce the
    search's best cost) + N_RANDOM controls = z0 moved by a random direction with the norm of a random source
    correction. Writes transfer/<target>/<clip>/{z.npy, labels.json, eval.csv, eval_base.json}."""
    rng = np.random.default_rng(0)
    cs, jobs = clips(), []
    for c in cs:
        z0 = np.load(REPO / f"data/origin_z/{c}.npy").reshape(-1); z0 = 16 * z0 / np.linalg.norm(z0)
        Zs, labels = [], []
        for b in BODIES:
            for s in SEEDS:
                z = np.load(cell_dir(b, c, s) / "best_z.npy").reshape(-1)
                Zs.append(16 * z / np.linalg.norm(z)); labels.append(dict(kind="body", source=b, seed=s))
        norms = [n for n in (np.linalg.norm(z - z0) for z in Zs) if n > 1e-9]
        for _ in range(N_RANDOM):
            r = rng.standard_normal(256); r *= rng.choice(norms) / np.linalg.norm(r)
            z = z0 + r; Zs.append(16 * z / np.linalg.norm(z)); labels.append(dict(kind="random", norm=float(np.linalg.norm(r))))
        for t in BODIES:
            out = OUT / "transfer" / t / c.split("/")[1]
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "z.npy", np.array(Zs)); (out / "labels.json").write_text(json.dumps(labels))
            n = len(np.load(ref_path(t, c))["qpos"])
            cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", c, "--body", t,
                   "--init", "reference", "--steps", str(n), "--pairs", str(PAIRS), "--obs-scale", "exact",
                   "--objective", "bfm", "--heading-weight", "1.0", "--pos-weight", "0.1",
                   "--anchor", f"data/origin_z/{c}.npy", "--anchor-weight", "0.3",
                   "--eval-z", str(out / "z.npy"), "--out", str(out)]
            jobs.append((n, out, cmd))
    jobs.sort(key=lambda j: -j[0])
    print(f"transfer: {len(jobs)} (target, clip) jobs x {len(Zs)} latents")
    run_jobs([(o, c) for _, o, c in jobs], args, "eval.csv")


E_TEST = ["child", "giant", "short_stocky", "gl0500a0550", "gl1500a1400", "gl0500a1400", "ax2m_t1000", "ax4m_t1000"]
E_ARMS = ["multi_same", "multi_loco", "child524", "rand8"]
E_K = 8


def _corr(b, c, s):
    z0 = np.load(REPO / f"data/origin_z/{c}.npy").reshape(-1)
    z = np.load(cell_dir(b, c, s) / "best_z.npy").reshape(-1)
    return 16 * z / np.linalg.norm(z) - 16 * z0 / np.linalg.norm(z0)


def _top(D, k=E_K):
    D = np.asarray([d for d in D if np.linalg.norm(d) > 1e-9])
    return np.linalg.svd(D, full_matrices=False)[2][:k]


def ebases(args):
    """Analysis E bases (8 orthonormal rows each), built only from bodies OUTSIDE the 8 test bodies:
      multi_same   uncentred PCA of the 24 training bodies' corrections on all 16 clips (the clip being searched
                   included: "a motion already solved on other bodies, now a new body")
      multi_loco   per clip: the same without that clip ("new body AND new motion")
      child524     the child's c540 corrections (outputs/c540_bfmglobal_a03, same recipe) on the 524 clips that are
                   NOT among the 16 (one body, many motions; for the test body child itself: new motions only)
      rand8        outputs/pca_subspace_bfmg/basis_rand8.npy (fixed random orthonormal 8)"""
    out = OUT / "ebases"; out.mkdir(parents=True, exist_ok=True)
    cs = clips(); train = [b for b in BODIES if b not in E_TEST]
    allD = {c: [_corr(b, c, s) for b in train for s in SEEDS] for c in cs}
    np.save(out / "multi_same.npy", _top([d for c in cs for d in allD[c]]))
    for c in cs:
        np.save(out / f"multi_loco_{c.split('/')[1]}.npy", _top([d for o in cs if o != c for d in allD[o]]))
    src = REPO / "outputs/c540_bfmglobal_a03"; ours = {c.split("/")[1] for c in cs}; D = []
    for l in open(src / "clips.txt"):
        t, k = l.split()[:2]
        if f"{t}_{k}" in ours or not (src / f"{t}_{k}" / "best_z.npy").exists():
            continue
        z0 = np.load(REPO / f"data/origin_z/{t}/{t}_{k}.npy").reshape(-1); z = np.load(src / f"{t}_{k}" / "best_z.npy").reshape(-1)
        D.append(16 * z / np.linalg.norm(z) - 16 * z0 / np.linalg.norm(z0))
    np.save(out / "child524.npy", _top(D))
    np.save(out / "rand8.npy", np.load(REPO / "outputs/pca_subspace_bfmg/basis_rand8.npy")[:E_K])
    print(f"bases from {len(train)} training bodies; child524 from {len(D)} child corrections -> {out}")
    for a in ("multi_same", "child524", "rand8"):
        U = np.load(out / f"{a}.npy"); print(f"  {a}: {U.shape}, orthonormal err {np.abs(U @ U.T - np.eye(len(U))).max():.1e}")


def esearch(args):
    """Analysis E searches: the 8 test bodies x 16 clips x 4 bases, 8-dim subspace search, otherwise exactly the
    full-space recipe (seed 0); the full-space arm is the existing search/<body>/<clip>_s0."""
    cs, jobs = clips(), []
    for arm in E_ARMS:
        for b in E_TEST:
            for c in cs:
                st = c.split("/")[1]
                basis = OUT / "ebases" / (f"multi_loco_{st}.npy" if arm == "multi_loco" else f"{arm}.npy")
                n = len(np.load(ref_path(b, c))["qpos"])
                out = OUT / "esearch" / arm / b / f"{st}_s0"
                cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", c, "--body", b,
                       "--init", "reference", "--steps", str(n), "--pairs", str(PAIRS), "--sigma", "0.05", "--lr", "0.03",
                       "--seed", "0", "--obs-scale", "exact", "--objective", "bfm", "--heading-weight", "1.0",
                       "--pos-weight", "0.1", "--evals", str(GENS * 2 * PAIRS), "--anchor", f"data/origin_z/{c}.npy",
                       "--anchor-weight", "0.3", "--subspace", str(basis), "--subspace-dim", str(E_K), "--out", str(out)]
                jobs.append((n, out, cmd))
    jobs.sort(key=lambda j: -j[0])
    print(f"esearch: {len(jobs)} subspace searches ({len(E_ARMS)} bases x {len(E_TEST)} bodies x {len(cs)} clips)")
    run_jobs([(o, c) for _, o, c in jobs], args, "summary.json")


def status(args):
    cs = clips()
    tot = 0
    for b in BODIES:
        n = sum(cell_dir(b, c, s).joinpath("summary.json").exists() for c in cs for s in SEEDS)
        tot += n
        print(f"{b:13s} {GROUPS[b]:5s} {n:3d}/{len(cs) * len(SEEDS)}")
    print(f"total {tot}/{len(BODIES) * len(cs) * len(SEEDS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("search")
    p.add_argument("--bodies", default="")
    p.add_argument("--clips", default="", help="comma-separated clip stems (default all)")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--gpus", default="0")
    p.add_argument("--max-par", type=int, default=3)
    p.add_argument("--min-free-mib", type=int, default=2500)
    p.add_argument("--retries", type=int, default=10)
    p.add_argument("--launch-pause", type=float, default=60)
    sub.add_parser("status")
    p = sub.add_parser("transfer")
    p.add_argument("--gpus", default="0")
    p.add_argument("--max-par", type=int, default=3)
    p.add_argument("--min-free-mib", type=int, default=2500)
    p.add_argument("--retries", type=int, default=10)
    p.add_argument("--launch-pause", type=float, default=15, help="eval jobs are short (~1 min): pause less")
    sub.add_parser("ebases")
    p = sub.add_parser("esearch")
    p.add_argument("--gpus", default="0")
    p.add_argument("--max-par", type=int, default=8)
    p.add_argument("--min-free-mib", type=int, default=2500)
    p.add_argument("--retries", type=int, default=10)
    p.add_argument("--launch-pause", type=float, default=30)
    a = ap.parse_args()
    {"plan": plan, "search": search, "status": status, "transfer": transfer, "ebases": ebases, "esearch": esearch}[a.cmd](a)
