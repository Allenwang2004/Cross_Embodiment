"""Tests of OOP refactor step 6 (docs/refactor/PLAN.md): the optimisers and loops against the inline code they replaced.

The search loops of the four scripts at BASE (fa6c5f9, the commit before step 6) are executed from `git show` -- their
verbatim lines, from the RNG's creation to the end of the loop body (as tests/test_rollout.py runs old loops) -- on a toy
cost instead of rollouts, and the new code (crossembodiment/optim.py ES / CEM / CMAES / RandomSearch / LSO /
TrajectoryAdaptation driven by crossembodiment/search.py SearchLoop / SequenceSearchLoop) runs on the same toy cost. Every
batch either one scores (the candidates, the mean-z copies) is recorded and must be np.array_equal (dtype and shape included),
and so must the iterates (z_trace's z_mean / z_best, mu_trace), the best record, the sigma / path-length columns and, for the
sequences, the kept budget rollouts and best_cost_at.
  single_z_search  es independent / antithetic x rank / raw, es in a --subspace, cem, cmaes (pycma, seed + 1), random;
                   mean-z rollouts every 2 steps
  robot_z_search   es independent / antithetic, cem, cmaes, random; and --resume: 3 steps, state through
                   RunRecorder.state / read_state into a fresh ES (load_state_dict), 2 more steps = the old code's 5 straight
  lso_search       lso (pink noise), lso --rank, lso --antithetic, ta; --budgets 2 4, --keep-iters 1
  robot_lso_search lso --rank --best-from all, ta
plus Adam against the scripts' expressions and return_to_go against the inline convolution.

No GPU, no simulation: ~10 s.  Run standalone:  uv run python tests/test_optim.py
"""
import ast
import csv
import json
import subprocess
import sys
import textwrap
import time
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

BASE = "fa6c5f9"                                              # the commit before step 6
WORK = REPO / "outputs/refactor_tests/test_optim"
_SRC = {}


def old_source(rel):
    if rel not in _SRC:
        _SRC[rel] = subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return _SRC[rel]


def block(rel, start, end):
    """the lines of `rel` at BASE from the first one containing `start` to the next one containing `end`, dedented"""
    lines = old_source(rel).split("\n")
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next(k for k in range(i, len(lines)) if end in lines[k])
    return compile(textwrap.dedent("\n".join(lines[i:j + 1])) + "\n", f"{BASE}:{rel}", "exec")


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def check_list(name, A, B):
    assert len(A) == len(B), f"{name}: {len(A)} vs {len(B)}"
    for k, (a, b) in enumerate(zip(A, B)):
        assert same(a, b), f"{name}[{k}] differs (max |d| {np.abs(np.asarray(a, float) - np.asarray(b, float)).max()})"


# ------------------------------------------------------------------ the toy problems

D = 256
TARGET = np.random.default_rng(7).standard_normal(D)
TARGET = 16 * TARGET / np.linalg.norm(TARGET)
Z0 = np.random.default_rng(8).standard_normal(D)
Z0 = 16 * Z0 / np.linalg.norm(Z0)


def toy_cost(Z):
    """(n, 256) -> (n,) float64"""
    return ((np.asarray(Z) - TARGET) ** 2).mean(1) + 0.01 * np.sin(np.asarray(Z)[:, 0])


def toy_reward(Z, tgt):
    """(n, S, 256) unit sequences -> per-step reward (n, S)"""
    return -((Z - tgt[None]) ** 2).sum(-1) + 0.1 * np.cos(Z[..., 1])


def ns_common():
    from crossembodiment.geometry import ang_pair, project_z, rank_normalize, rank_normalize_steps, unit_sphere
    from crossembodiment.optim import colored_noise
    from crossembodiment.rollout import BatchRollout
    return dict(np=np, json=json, csv=csv, time=time, Path=Path, REPO=REPO, print=lambda *a, **k: None, project_z=project_z,
                rank_normalize=rank_normalize, ang_pair=ang_pair, rank_normalize_steps=rank_normalize_steps, unit=unit_sphere,
                colored_noise=colored_noise, batch=types.SimpleNamespace(copies=BatchRollout.copies, with_mean=BatchRollout.with_mean,
                                                                       padded=BatchRollout.padded))


SZ_DEFAULTS = dict(algo="es", sigma=0.05, lr=0.03, pairs=8, sampling="independent", fitness="rank", subspace=None,
                   subspace_dim=None, cem_elites=4, cem_sigma_min=None, z_start=None, best_from_start=False, seed=0,
                   eval_every=2)


# ------------------------------------------------------------------ single latent: single_z_search.py

def run_old_single(**kw):
    args = types.SimpleNamespace(**{**SZ_DEFAULTS, **kw})
    batches = []

    def score(zs):
        batches.append(np.array(zs, copy=True))
        n = len(zs)
        extra["head"], extra["pos"] = np.zeros(n), np.zeros(n)
        return toy_cost(zs), np.zeros(n), np.zeros(n), np.zeros(n, np.float32), np.asarray(zs)
    extra = {}
    n_envs = 2 * args.pairs
    c0 = toy_cost(Z0[None])
    ns = dict(ns_common(), args=args, z0=Z0.copy(), score=score, extra=extra, n_gens=5, n_envs=n_envs,
              logged=lambda i, q, a_, p_: (0.0, 0.0, 0.0), dext=lambda i, with_path=True: {},
              base=dict(cost=float(c0[0]), align=0.0, phys=0.0, bfm=0.0, head=0.0, pos=0.0, mse=0.0), q0=Z0[None].copy())
    exec(block("scripts/single_z_search.py", "rng = np.random.default_rng(args.seed)", 'trace_best.append(best["z"].copy())'), ns)
    return dict(batches=batches, mean=ns["trace_mean"], best_trace=ns["trace_best"], best=ns["best"],
                sigma=[r["sigma"] for r in ns["curve"]], path=[r["path_len_deg"] for r in ns["curve"]],
                mean_z=[r["mean_z_cost"] for r in ns["curve"]])


def run_new_single(robot=False, n_gens=5, opt=None, best=None, **kw):
    from crossembodiment.optim import single_latent_optimizer
    from crossembodiment.search import SearchLoop
    a = types.SimpleNamespace(**{**SZ_DEFAULTS, **kw})
    U = None
    if a.subspace:
        U = np.load(a.subspace).astype(np.float64)[: (a.subspace_dim or None)]
    batches = []

    def score(zs):
        batches.append(np.array(zs, copy=True))
        return types.SimpleNamespace(cost=toy_cost(zs))

    def record(sc, i, z, gen, source):
        return dict(cost=float(sc.cost[i]), z=z.copy(), gen=gen, source=source)
    if opt is None:
        opt = single_latent_optimizer(a.algo, Z0.copy(), a.sigma, a.lr, a.pairs, a.seed, sampling=a.sampling,
                                      fitness=a.fitness, subspace=U, cem_elites=a.cem_elites, cem_sigma_min=a.cem_sigma_min)
    c0 = toy_cost(Z0[None])
    best = best or dict(cost=float(c0[0]), z=Z0.copy(), gen=-1, source="origin_z")
    rows = lambda lp, gen, sc: dict(sigma=lp.opt.step_sigma, path_len_deg=round(lp.path_len, 2), mean_z_cost="")   # noqa: E731
    loop = SearchLoop(opt, score, record, 2 * a.pairs, n_gens, best, eval_every=a.eval_every, row=rows)
    return loop, batches


def compare_single(old, loop, batches, keys=("cost", "gen", "source")):
    check_list("scored batches", old["batches"], batches)
    check_list("z_mean", old["mean"], loop.trace_mean)
    check_list("z_best", old["best_trace"], loop.trace_best)
    for k in keys:
        assert old["best"][k] == loop.best[k], f"best {k}: {old['best'][k]} vs {loop.best[k]}"
    assert same(old["best"]["z"], loop.best["z"])
    assert old["sigma"] == [r["sigma"] for r in loop.curve] and old["path"] == [r["path_len_deg"] for r in loop.curve]
    assert old["mean_z"] == [r["mean_z_cost"] for r in loop.curve]


def _single(**kw):
    old = run_old_single(**kw)
    loop, batches = run_new_single(**kw)
    loop.run()
    compare_single(old, loop, batches)
    assert len(batches) == 5 + 3                                       # 5 steps + mean z at steps 0, 2, 4


def test_single_es_independent_rank():
    _single()


def test_single_es_antithetic_rank():
    _single(sampling="antithetic")


def test_single_es_independent_raw():
    _single(fitness="raw")


def test_single_es_antithetic_raw():
    _single(sampling="antithetic", fitness="raw", seed=3)


def test_single_es_subspace():
    WORK.mkdir(parents=True, exist_ok=True)
    U = np.linalg.qr(np.random.default_rng(11).standard_normal((D, 12)))[0].T          # (12, 256) orthonormal rows
    np.save(WORK / "U.npy", U)
    _single(subspace=str(WORK / "U.npy"), subspace_dim=8)


def test_single_cem():
    _single(algo="cem", sigma=0.3, seed=2)


def test_single_cmaes():
    _single(algo="cmaes", sigma=0.3, seed=1)


def test_single_random():
    _single(algo="random", sigma=0.4, seed=5)


# ------------------------------------------------------------------ single latent: robot_z_search.py

def run_old_robot(n_gens=5, **kw):
    a = types.SimpleNamespace(**{**SZ_DEFAULTS, "resume": None, **kw})
    batches = []

    def score(zs):
        batches.append(np.array(zs, copy=True))
        return dict(cost=toy_cost(zs))

    def pick(r, i, z, gen, source):
        return dict(cost=float(r["cost"][i]), bfm=0.0, bfm_plain=0.0, mse=0.0, head=0.0, pos=0.0, ik=0.0, z=z.copy(), gen=gen,
                    source=source)
    n_envs = 2 * a.pairs
    c0 = toy_cost(Z0[None])
    ns = dict(ns_common(), a=a, z0=Z0.copy(), score=score, pick=pick, n_gens=n_gens, n_envs=n_envs,
              n_eps=a.pairs if a.sampling == "antithetic" else n_envs, al=lambda b: "", t0=time.time(),
              base=dict(cost=float(c0[0]), bfm=0.0, bfm_plain=0.0, mse=0.0, head=0.0, pos=0.0, ik=0.0, z=Z0.copy(), gen=-1,
                        source="origin_z"))
    exec(block("scripts/robot_z_search.py", "rng = np.random.default_rng(a.seed)", "curve.append(row); trace_mean.append"), ns)
    return dict(batches=batches, mean=ns["trace_mean"], best_trace=ns["trace_best"], best=ns["best"],
                sigma=[r["sigma"] for r in ns["curve"]], path=[r["path_len_deg"] for r in ns["curve"]],
                mean_z=[r["mean_z_cost"] for r in ns["curve"]], ns=ns)


def _robot(**kw):
    old = run_old_robot(**kw)
    loop, batches = run_new_single(**kw)
    loop.run()
    compare_single(old, loop, batches)


def test_robot_es_independent():
    _robot()


def test_robot_es_antithetic():
    _robot(sampling="antithetic", seed=4)


def test_robot_cem():
    _robot(algo="cem", sigma=0.3, cem_sigma_min=0.02)


def test_robot_cmaes():
    _robot(algo="cmaes", sigma=0.3, seed=6)


def test_robot_random():
    _robot(algo="random", sigma=0.4)


def test_robot_resume_exact():
    """3 steps, state.npz through RunRecorder, a fresh ES restored from it, 2 more steps == the old code's 5 straight"""
    from crossembodiment.optim import ES
    from crossembodiment.records import RunRecorder
    from crossembodiment.search import SearchLoop
    old = run_old_robot(n_gens=5, eval_every=0)
    # the old code's own state after 5 steps, as robot_z_search.py saved it
    ons = old["ns"]
    loop, b1 = run_new_single(n_gens=3, eval_every=0)
    loop.run()
    rec = RunRecorder(WORK / "resume")
    sd = loop.opt.state_dict()
    rec.state(z=sd["z"], m=sd["m"], v=sd["v"], t_adam=sd["t"], path_len=loop.path_len, best_z=loop.best["z"], rng=sd["rng"])
    S = rec.read_state()
    opt = ES(Z0.copy(), 0.05, 0.03, 8, seed=0)
    opt.load_state_dict(dict(z=S["z"], m=S["m"], v=S["v"], t=int(S["t_adam"]), rng=S["rng"]))
    best = dict(loop.best, z=S["best_z"])
    loop2, b2 = run_new_single(n_gens=5, opt=opt, best=best, eval_every=0)
    loop2.gen0, loop2.path_len = 3, float(S["path_len"])
    loop2.trace_mean, loop2.trace_best, loop2.curve = list(loop.trace_mean), list(loop.trace_best), list(loop.curve)
    loop2.run()
    check_list("scored batches (3 + 2)", old["batches"], b1 + b2)
    check_list("z_mean", old["mean"], loop2.trace_mean)
    assert same(ons["z"], loop2.z) and same(ons["m"], opt.adam.m) and same(ons["v"], opt.adam.v)
    assert json.dumps(ons["rng"].bit_generator.state) == json.dumps(opt.rng.bit_generator.state)
    assert ons["path_len"] == loop2.path_len and opt.adam.t == 5
    assert isinstance(SearchLoop, type)


# ------------------------------------------------------------------ latent sequences: lso_search.py / robot_lso_search.py

S_, N_ = 12, 8
LSO_DEFAULTS = dict(method="lso", n=N_, iters=5, sigma=0.0125, lr=0.00625, betas=(0.8, 0.99), gamma=0.97, beta=1.0,
                    antithetic=False, rank=False, budgets=[2, 4], keep_iters=[1], ta_beta1=0.85, ta_beta2=0.9, ta_temp=0.05,
                    seed=0, best_from="samples")


def seq_fixture():
    from crossembodiment.geometry import unit_sphere
    rng = np.random.default_rng(9)
    tgt = unit_sphere(rng.standard_normal((S_, D)))
    mu0 = unit_sphere(tgt + 0.3 * rng.standard_normal((S_, D)))
    er8 = unit_sphere(tgt + 0.2 * rng.standard_normal((S_, D)))
    return tgt, mu0, er8


def run_old_lso(**kw):
    a = types.SimpleNamespace(**{**LSO_DEFAULTS, **kw})
    tgt, mu0, er8 = seq_fixture()
    batches = []

    def rollout(Z):
        batches.append(np.array(Z, copy=True))
        rollout.parts = dict(bfm=np.zeros(len(Z)))
        return toy_reward(Z, tgt), np.asarray(Z)[..., :4].copy()
    base = {"B(g_t) (LSO start)": mu0, "ER L=8 (Motivo zero-shot)": er8}
    Zb = np.stack([mu0, er8])
    r0, q0 = rollout(Zb); batches.clear()
    res = {k: dict(cost=float(1 - r0[i].sum() / S_)) for i, k in enumerate(base)}
    ns = dict(ns_common(), a=a, base=base, res=res, q=q0, SR=S_, S=S_, rollout=rollout, keep={}, t0=time.time(), mu=mu0)
    exec(block("scripts/lso_search.py", "rng = np.random.default_rng(a.seed)", "|d mu_t| {row['mu_move']:.3f}{dl}"), ns)
    return dict(batches=batches, trace=ns["trace"], best=ns["best"], keep=ns["keep"], budget=ns["budget_cost"],
                mu=ns["mu"], move=[r["mu_move"] for r in ns["curve"]])


def run_old_robot_lso(**kw):
    a = types.SimpleNamespace(**{**LSO_DEFAULTS, **kw})
    tgt, mu0, er8 = seq_fixture()
    batches = []

    def rollout(Z):
        batches.append(np.array(Z, copy=True))
        rollout.parts = dict(bfm=np.zeros(len(Z)), ik_deg=np.zeros(len(Z)))
        rollout.keep = dict(qg=np.asarray(Z)[..., :4].copy())
        return toy_reward(Z, tgt)
    Zb = np.stack([mu0, er8])
    r0 = rollout(Zb); q0 = rollout.keep["qg"]; batches.clear()
    res = {"B(g_t) (LSO start)": dict(cost=float(1 - r0[0].sum() / S_)), "ER L=8 (Motivo zero-shot)": dict(cost=float(1 - r0[1].sum() / S_))}
    q_start = {"B(g_t) (LSO start)": q0[0].copy(), "ER L=8 (Motivo zero-shot)": q0[1].copy()}
    WORK.mkdir(parents=True, exist_ok=True)
    ns = dict(ns_common(), a=a, res=res, q_start=q_start, S=S_, rollout=rollout, keep={}, t0=time.time(), mu=mu0, er8=er8,
              out=WORK)
    exec(block("scripts/robot_lso_search.py", "rng = np.random.default_rng(a.seed)", "w_.writerows(curve)"), ns)
    return dict(batches=batches, trace=ns["trace"], best=ns["best"], keep=ns["keep"], budget=ns["budget_cost"],
                mu=ns["mu"], move=[r["mu_move"] for r in ns["curve"]])


def run_new_lso(robot=False, **kw):
    from crossembodiment.optim import LSO, TrajectoryAdaptation
    from crossembodiment.search import SequenceSearchLoop
    a = types.SimpleNamespace(**{**LSO_DEFAULTS, **kw})
    tgt, mu0, er8 = seq_fixture()
    batches = []

    def score(Z):
        batches.append(np.array(Z, copy=True))
        return types.SimpleNamespace(r=toy_reward(Z, tgt), q=np.asarray(Z)[..., :4].copy())
    r0 = toy_reward(np.stack([mu0, er8]), tgt); q0 = np.stack([mu0, er8])[..., :4].copy()
    if a.method == "ta":
        opt = TrajectoryAdaptation(er8.copy(), a.sigma, a.n, a.iters, seed=a.seed, beta1=a.ta_beta1, beta2=a.ta_beta2, temp=a.ta_temp)
        k = 1
    else:
        opt = LSO(mu0, a.sigma, a.lr, a.n, seed=a.seed, beta=a.beta, gamma=a.gamma, betas=a.betas, rank=a.rank,
                  antithetic=a.antithetic)
        k = 0
    best = dict(cost=float(1 - r0[k].sum() / S_), z=opt.mu.copy(), it=-1, src="start", q=q0[k].copy())
    row = lambda lp, it, sc, costs: dict(mu_move=float(np.linalg.norm(16 * (lp.opt.mu - lp.trace[-1]), axis=1).mean()))  # noqa: E731
    loop = SequenceSearchLoop(opt, score, a.iters, S_, best, budgets=a.budgets, keep={}, keep_iters=[] if robot else a.keep_iters,
                              best_from=a.best_from, row=row)
    loop.run()
    return loop, batches


def compare_seq(old, loop, batches):
    check_list("scored batches", old["batches"], batches)
    check_list("mu trace", old["trace"], loop.trace)
    assert same(old["mu"], loop.opt.mu)
    for k in ("cost", "it", "src"):
        assert old["best"][k] == loop.best[k], f"best {k}"
    assert same(old["best"]["z"], loop.best["z"]) and same(old["best"]["q"], loop.best["q"])
    assert list(old["keep"]) == list(loop.keep), f"kept rollouts {list(old['keep'])} vs {list(loop.keep)}"
    for k in old["keep"]:
        assert same(old["keep"][k], loop.keep[k]), k
    assert old["budget"] == loop.budget_cost and old["move"] == [r["mu_move"] for r in loop.curve]


def test_lso_pink():
    compare_seq(run_old_lso(), *run_new_lso())


def test_lso_rank():
    compare_seq(run_old_lso(rank=True, seed=2), *run_new_lso(rank=True, seed=2))


def test_lso_antithetic():
    compare_seq(run_old_lso(antithetic=True, seed=3), *run_new_lso(antithetic=True, seed=3))


def test_lso_white_noise():
    compare_seq(run_old_lso(beta=0.0, gamma=0.9), *run_new_lso(beta=0.0, gamma=0.9))


def test_ta():
    compare_seq(run_old_lso(method="ta", sigma=0.05), *run_new_lso(method="ta", sigma=0.05))


def test_robot_lso_rank_best_from_all():
    compare_seq(run_old_robot_lso(rank=True, best_from="all"), *run_new_lso(robot=True, rank=True, best_from="all"))


def test_robot_ta():
    compare_seq(run_old_robot_lso(method="ta", sigma=0.05, seed=1), *run_new_lso(robot=True, method="ta", sigma=0.05, seed=1))


# ------------------------------------------------------------------ the pure helpers

def test_adam_and_return_to_go():
    from crossembodiment.optim import Adam, return_to_go
    rng = np.random.default_rng(13)
    G = rng.standard_normal((6, 256))
    m = np.zeros(256); v = np.zeros_like(m); b1, b2, eps_adam = 0.9, 0.999, 1e-8
    ad = Adam(0.03, 256)
    for gen, g in enumerate(G):                                  # single_z_search's expressions
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (gen + 1))
        vh = v / (1 - b2 ** (gen + 1))
        step = 0.03 * mh / (np.sqrt(vh) + eps_adam)
        assert same(step, ad.step(g))
    m = np.zeros((5, 256)); v = np.zeros_like(m); b1, b2 = (0.8, 0.99)
    ad = Adam(0.00625, (5, 256), betas=(0.8, 0.99))
    for it in range(4):                                          # lso_search's
        g = rng.standard_normal((5, 256))
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        assert same(0.00625 * (m / (1 - b1 ** (it + 1))) / (np.sqrt(v / (1 - b2 ** (it + 1))) + 1e-8), ad.step(g))
    r = rng.standard_normal((7, 30)); disc = 0.97 ** np.arange(30)
    assert same(np.stack([np.convolve(x[::-1], disc)[:30][::-1] for x in r]), return_to_go(r, disc))


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except unittest.SkipTest as e:
            print(f"  SKIP  {name}: {e}", flush=True)
        except (AssertionError, KeyError, ValueError, OSError, ImportError, TypeError, StopIteration,
                subprocess.CalledProcessError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
