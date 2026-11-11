"""Tests of OOP refactor step 6 (docs/refactor/PLAN.md): RunRecorder (crossenbodiment/records.py) writes the search files as
the scripts did.

  round trips   every writer, then its reader (or np.load / json / csv): the same arrays (dtype, shape, bits), the same rows,
                the same summary
  golden files  files of the golden baseline (outputs/refactor_golden/baseline, made by the snapshot code) read and written
                again through RunRecorder: summary.json (indent 2: single_z_search / robot_z_search; indent 1: lso_search /
                robot_lso_search), curve.csv (first-row columns; robot_z_search's last-row columns), eval_base.json and
                best_z.npy must be byte-identical; z_trace.npz, state.npz, mu_trace.npz, rollouts.npz, best.npz must hold the
                same arrays under the same keys in the same member order
  old writers   the eval.csv writers of single_z_search ("%.6g") and robot_z_search ("%.9g") and the state.npz / z_trace.npz
                lines, executed from their source at BASE (fa6c5f9) on the same rows: byte-identical eval.csv, equal arrays
No GPU: ~2 s.  Run standalone:  uv run python tests/test_records.py
"""
import csv
import json
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

BASE = "fa6c5f9"
GOLD = REPO / "outputs/refactor_golden/baseline"
WORK = REPO / "outputs/refactor_tests/test_records"


def old_source(rel):
    return subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def rec(name):
    from crossenbodiment.records import RunRecorder
    return RunRecorder(WORK / name)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def npz_equal(fa, fb):
    """same members in the same order, every array the same dtype / shape / bytes"""
    assert zipfile.ZipFile(fa).namelist() == zipfile.ZipFile(fb).namelist(), f"{fa.name}: member order"
    A, B = np.load(fa), np.load(fb)
    for k in A.files:
        assert same(A[k], B[k]), f"{fa.name}: {k}"


def need_gold(*cases):
    for c in cases:
        if not (GOLD / c).is_dir():
            raise unittest.SkipTest(f"no golden baseline {GOLD / c}")


# ------------------------------------------------------------------ round trips

def test_round_trip():
    r = rng = np.random.default_rng(1)
    R = rec("round_trip")
    z = rng.standard_normal(256)
    R.best_z(z)
    assert same(R.read_best_z(), z.astype(np.float32).reshape(1, -1))
    tm, tb = [rng.standard_normal(256) for _ in range(5)], [rng.standard_normal(256) for _ in range(5)]
    R.z_trace(tm, tb)
    T = R.read_z_trace()
    assert list(T) == ["gen", "z_mean", "z_best"] and same(T["gen"], np.arange(-1, 4, dtype=np.int32))
    assert same(T["z_mean"], np.stack(tm).astype(np.float32)) and same(T["z_best"], np.stack(tb).astype(np.float32))
    rows = [dict(gen=g, cost=float(r.standard_normal()), mean_z_cost="" if g % 2 else float(r.random()), sigma=0.05)
            for g in range(4)]
    R.curve(rows)
    back = R.read_curve()
    assert back == [{k: str(v) for k, v in row.items()} for row in rows]
    summ = dict(clip="a/b", best=dict(cost=0.25, gen=-1, source="origin_z"), nan=float("nan"), none=None, flag=True,
                lst=[1, 2.5], best_cost_at={2: 0.5})
    R.summary(summ, indent=2)
    s2 = R.read_summary()
    assert list(s2) == list(summ) and s2["best"] == summ["best"] and np.isnan(s2["nan"]) and s2["best_cost_at"] == {"2": 0.5}
    m, v = rng.standard_normal(256), rng.random(256)
    R.state(z=z, m=m, v=v, t_adam=7, path_len=12.5, best_z=z[::-1].copy(), rng=json.dumps(rng.bit_generator.state))
    S = R.read_state()
    assert list(S) == ["z", "m", "v", "t_adam", "path_len", "best_z", "rng"]
    assert same(S["z"], z) and same(S["m"], m) and int(S["t_adam"]) == 7 and float(S["path_len"]) == 12.5
    g = np.random.default_rng(0); g.bit_generator.state = json.loads(str(S["rng"]))
    assert g.bit_generator.state == rng.bit_generator.state
    trace = [rng.standard_normal((6, 256)).astype(np.float32) for _ in range(5)]
    R.mu_trace(trace, [0, 2, 4], trace[3].astype(np.float64))
    M = R.read_npz("mu_trace.npz")
    assert list(M) == ["mu", "mu_iters", "best"] and same(M["mu"], np.stack([trace[0], trace[2], trace[4]]))
    keep = {"q_Bg": rng.standard_normal((6, 4)).astype(np.float32), "q_best_it2": rng.standard_normal((6, 4)).astype(np.float32)}
    ref = rng.standard_normal((7, 4))
    R.rollouts(ref, keep)
    K = R.read_npz("rollouts.npz")
    assert list(K) == ["ref", "q_Bg", "q_best_it2"] and same(K["ref"], ref.astype(np.float32))
    R.eval_csv(["index", "cost"], [(0, 0.123456789), (1, 2.0)], style="single")
    assert R.path("eval.csv").read_text() == "index,cost\n0,0.123457\n1,2\n"
    R.eval_csv(["index", "cost"], [(0, 0.123456789123), (1, 2.0)], style="robot")
    assert R.path("eval.csv").read_text() == "index,cost\n0,0.123456789\n1,2\n"


# ------------------------------------------------------------------ the golden baseline's files, written again

def _rewrite_json(case, name, indent):
    f = GOLD / case / name
    R = rec(f"gold_{case.replace('/', '_')}")
    out = R.json(name, json.loads(f.read_text()), indent=indent)
    assert out.read_bytes() == f.read_bytes(), f"{case}/{name}: bytes differ"


def test_golden_summary_json():
    need_gold("child_es_anchor", "child_dtw", "g1_es", "g1_es_resume/full", "child_lso", "g1_lsor", "g1_ta", "tall_amass_lso")
    for case in ("child_es_anchor", "child_dtw", "g1_es", "g1_es_resume/full", "g1_cem_phc"):
        _rewrite_json(case, "summary.json", 2)
    for case in ("child_lso", "child_ta", "payload_lso", "tall_amass_lso", "g1_lsor", "g1_ta"):
        _rewrite_json(case, "summary.json", 1)
    for case in ("child_evalz", "g1_evalz"):
        _rewrite_json(case, "eval_base.json", None)


def test_golden_curve_csv():
    need_gold("child_es_anchor", "g1_es", "child_lso")
    for case, cols in (("child_es_anchor", 0), ("child_cmaes", 0), ("child_lso", 0), ("g1_lsor", 0), ("g1_es", -1),
                       ("g1_es_resume/full", -1)):
        f = GOLD / case / "curve.csv"
        rows = list(csv.DictReader(open(f)))
        out = rec(f"gold_{case.replace('/', '_')}").curve(rows, columns_from=cols)
        assert out.read_bytes() == f.read_bytes(), f"{case}/curve.csv: bytes differ"


def test_golden_npy_npz():
    need_gold("child_es_anchor", "g1_es", "child_lso", "g1_lsor")
    for case in ("child_es_anchor", "g1_es"):
        R = rec(f"gold_{case}")
        z = np.load(GOLD / case / "best_z.npy")
        assert R.best_z(z.reshape(-1).astype(np.float64)).read_bytes() == (GOLD / case / "best_z.npy").read_bytes()
        T = np.load(GOLD / case / "z_trace.npz")
        npz_equal(GOLD / case / "z_trace.npz", R.z_trace(list(T["z_mean"]), list(T["z_best"])))
        B = np.load(GOLD / case / "best.npz")
        npz_equal(GOLD / case / "best.npz", R.npz("best.npz", **{k: B[k] for k in B.files}))
    S = np.load(GOLD / "g1_es/state.npz")
    npz_equal(GOLD / "g1_es/state.npz", rec("gold_g1_es").state(**{k: S[k] for k in S.files}))
    for case in ("child_lso", "g1_lsor"):
        R = rec(f"gold_{case}")
        K = np.load(GOLD / case / "rollouts.npz")
        npz_equal(GOLD / case / "rollouts.npz", R.rollouts(K["ref"], {k: K[k] for k in K.files if k != "ref"}))
        M = np.load(GOLD / case / "mu_trace.npz")
        its = list(M["mu_iters"])
        trace = {int(k): M["mu"][i] for i, k in enumerate(its)}
        npz_equal(GOLD / case / "mu_trace.npz", R.mu_trace(trace, [int(k) for k in its], M["best"]))


# ------------------------------------------------------------------ the old inline writers on the same rows

def _old_lines(rel, start, end):
    lines = old_source(rel).split("\n")
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next(k for k in range(i, len(lines)) if end in lines[k])
    import textwrap
    return textwrap.dedent("\n".join(lines[i:j + 1])) + "\n"


def test_old_eval_csv_writers():
    rng = np.random.default_rng(4)
    rows = [(k, *map(float, rng.standard_normal(5) * 10.0 ** rng.integers(-6, 6, 5)), float("nan")) for k in range(7)]
    rows.append((7, 0.0, -0.0, 1e-300, 12345678.9, float("inf"), 3.0))
    # single_z_search: the header from DTW keys, rows "%.6g" for floats, str() otherwise
    out = WORK / "old_single"; out.mkdir(parents=True, exist_ok=True)
    extra = {"dtw": None}
    src = _old_lines("scripts/single_z_search.py", 'with open(out_dir / "eval.csv", "w") as fh:', "fh.writelines(")
    exec(src, dict(out_dir=out, rows=rows, extra=extra, DTW_KEYS=("dtw", "done", "pace", "head_al", "fallen")))
    R = rec("new_single")
    R.eval_csv(["index", "cost", "bfm", "head", "pos", "anchor_dist", "dtw"], rows, style="single")
    assert R.path("eval.csv").read_bytes() == (out / "eval.csv").read_bytes()
    # robot_z_search: "%.9g" for everything but ints
    rows8 = [r[:8] for r in rows]
    out = WORK / "old_robot"; out.mkdir(parents=True, exist_ok=True)
    src = _old_lines("scripts/robot_z_search.py", 'with open(out / "eval.csv", "w") as fh:', "fh.writelines(")
    exec(src, dict(out=out, rows=rows8))
    R = rec("new_robot")
    R.eval_csv(["index", "cost", "bfm", "head", "pos", "anchor_dist", "ik", "mse"], rows8, style="robot")
    assert R.path("eval.csv").read_bytes() == (out / "eval.csv").read_bytes()


def test_old_state_and_trace_writers():
    rng = np.random.default_rng(6)
    z, m, v, bz = rng.standard_normal(256), rng.standard_normal(256), rng.random(256), rng.standard_normal(256)
    g = np.random.default_rng(3); g.standard_normal(10)
    out = WORK / "old_state"; out.mkdir(parents=True, exist_ok=True)
    src = _old_lines("scripts/robot_z_search.py", 'np.savez(out / "state.npz"', "rng=json.dumps(rng.bit_generator.state))")
    exec(src, dict(np=np, json=json, out=out, z=z, m=m, v=v, n_gens=9, gen0=4, t_adam0=4, path_len=3.25, best=dict(z=bz), rng=g))
    R = rec("new_state")
    R.state(z=z, m=m, v=v, t_adam=9 - 4 + 4, path_len=3.25, best_z=bz, rng=json.dumps(g.bit_generator.state))
    npz_equal(out / "state.npz", R.path("state.npz"))
    tm, tb = [rng.standard_normal(256) for _ in range(4)], [rng.standard_normal(256) for _ in range(4)]
    src = _old_lines("scripts/robot_z_search.py", 'np.savez_compressed(out / "z_trace.npz"', "z_mean=np.stack(trace_mean)")
    exec(src, dict(np=np, out=out, trace_mean=tm, trace_best=tb))
    npz_equal(out / "z_trace.npz", R.z_trace(tm, tb))


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except unittest.SkipTest as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}", flush=True)
        except (AssertionError, KeyError, ValueError, OSError, StopIteration, subprocess.CalledProcessError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
