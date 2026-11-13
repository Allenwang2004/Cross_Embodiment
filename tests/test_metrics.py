"""Tests of crossembodiment/metrics.py (OOP refactor step 3, docs/refactor/PLAN.md), exact comparisons (==, np.array_equal):

  * each named metric function against the VERBATIM expression it replaced (frozen below, from commit a997a23), on fixed random
    arrays (values on and next to the thresholds included)
  * the Task-page drivers that tests/golden/metrics_golden.py does not run (loes_weight_test.stats, lsor_mpjae_test.stats,
    amass_baselines.one, kpkd_test.phc_one / contact_one) and the three it does (loes_noanchor / body_52_test / g1_small_test
    stats): the module as it is now against the same module at a997a23 (git show, executed as a module of its own), on the golden
    inputs (tests/golden/inputs/metrics_rollouts.npz, the frozen saved rollouts) and, for amass_baselines, the golden baseline's
    AMASS LSO rollouts. Driver folders they read are rebuilt under outputs/refactor_tests/ (removed afterwards).

Run standalone (no pytest in the environment), like tests/test_losses.py:
    uv run python tests/test_metrics.py
"""
import builtins
import io
import json
import os
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from crossembodiment import metrics as M  # noqa: E402

BASE = "a997a23"                                              # the commit before step 3
INP = REPO / "tests/golden/inputs"
WORK = REPO / "outputs/refactor_tests/test_metrics"
GOLDEN = REPO / "outputs/refactor_golden/baseline"
CHILD = ("headstand/headstand_4", "rotate-x--5-0.8/rotate-x--5-0.8_4", "move-ego-0-2/move-ego-0-2_1")
AMASS = "Male2Walking__B9_-__Walk_turn_left_90"
RNG = lambda k=0: np.random.default_rng(20261112 + k)        # noqa: E731
_R = {}


def rollouts():
    if not _R:
        _R.update(dict(np.load(INP / "metrics_rollouts.npz")))
    return _R


def same(a, b):
    """exactly equal, recursively (floats by ==, NaN = NaN, arrays by dtype, shape and value)"""
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return type(a) is type(b) and a == b


def load_source(name, src, path, files=None):
    """execute `src` as a fresh module `name` with __file__ = path (REPO from it); files: {path suffix: text} served by the
    module's own open() while it is imported (module-level reads of files this worktree does not have)"""
    mod = types.ModuleType(name); mod.__file__ = str(path)
    if files:
        def fake_open(f, *a, **k):
            for suf, text in files.items():
                if str(f).endswith(suf):
                    return io.StringIO(text)
            return builtins.open(f, *a, **k)
        mod.__dict__["open"] = fake_open
    exec(compile(src, str(path), "exec"), mod.__dict__)
    mod.__dict__.pop("open", None)
    return mod


def old_and_new(script, files=None):
    """(the module at BASE, the module now), both executed fresh from source"""
    path = REPO / "scripts" / f"{script}.py"
    old = subprocess.run(["git", "show", f"{BASE}:scripts/{script}.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return load_source(f"old_{script}", old, path, files), load_source(f"new_{script}", path.read_text(), path, files)


# ------------------------------------------------------------------ the old expressions, verbatim (a997a23)

def old_contact(D, Dr, T):                                    # body_52_test / g1_small_test / loes_noanchor / tall / amass / kpkd
    return float(((D <= 0.01) & (Dr[:T] > 0.05)).any(1).sum() / 30)


def old_phc(gw, s):
    return float(not (gw > 0.5 * s).any())


def old_phc_max(body_err, S):                                 # drift_success_child.py
    return float(body_err.max() <= 0.5 * S)


def old_drift_body(q, r):
    return float(np.linalg.norm(q[:, :2] - r[:, :2], axis=1).mean())


def old_drift_adult(q, r, s):
    return float(np.linalg.norm(q[:, :2] - r[:, :2], axis=1).mean() / s)


def old_facing(q):
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)) - np.pi / 2


def old_heading(q, r):                                        # drift_success_child.py
    return float(np.degrees(np.abs(np.angle(np.exp(1j * (old_facing(q) - old_facing(r)))))).mean())


def old_root(q, qr):                                          # eval_metrics.metrics (bad, pos), child_52_means.full, success_definitions
    pos = np.linalg.norm(q[:, :2] - qr[:, :2], axis=1); rot = np.degrees(2 * np.arccos(np.abs((q[:, 3:7] * qr[:, 3:7]).sum(1)).clip(0, 1)))
    bad = (pos > 1.5) | (rot > 45)
    return bad, pos, float(not ((pos > 1.5) | (rot > 45)).any()), not bad.any()


def old_height(q, qr):
    return float((q[:, 2] < 0.5 * qr[:, 2]).sum() / 30)


def old_hand_fall(h, hr, TOUCH=0.01, REF_NEAR=0.05, HAND_MOTION_S=0.5):    # child_52_means.hand_fall after hand_floor
    touch = h <= TOUCH
    hand_motion = (hr <= REF_NEAR).sum() / 30 >= HAND_MOTION_S
    return (float("nan") if hand_motion else float(touch.sum() / 30)), float((touch & (hr > REF_NEAR)).sum() / 30)


def old_world(m, d, Q):                                       # the drivers' world() / xpos()
    P = np.empty((len(Q), m.nbody - 1, 3))
    for t, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); P[t] = d.xpos[1:]
    return P


# ------------------------------------------------------------------ inputs

def floor_like(k, T=300, P=18):
    """floor distances with many values exactly on and next to 0.01 / 0.05"""
    r = RNG(k); D = r.choice([0.0, 0.005, 0.01, np.nextafter(0.01, 1), 0.03, 0.05, np.nextafter(0.05, 1), 0.2, 0.5], (T, P))
    return D + r.choice([0.0, 0.0, 0.0, 1e-3], (T, P))


def qpos_like(k, T=300, nq=76, drift=1.0):
    r = RNG(k); q = np.zeros((T, nq)); q[:, :2] = np.cumsum(r.normal(0, 0.02 * drift, (T, 2)), 0); q[:, 2] = r.uniform(0.2, 1.0, T)
    qt = r.standard_normal((T, 4)); q[:, 3:7] = qt / np.linalg.norm(qt, axis=1, keepdims=True); q[:, 7:] = r.uniform(-1, 1, (T, nq - 7))
    return q


# ------------------------------------------------------------------ the named functions

def test_contact_fall_time():
    for k in range(20):
        D, Dr = floor_like(k), floor_like(100 + k, T=320)
        T = len(D)
        assert same(M.contact_fall_time(D, Dr[:T]), old_contact(D, Dr, T))


def test_phc_success():
    for k in range(40):
        r = RNG(200 + k); s = [1.0, 0.611036, 0.8212611427612194, 1.364408905613045][k % 4]
        gw = r.uniform(0, 0.6, 300) * s
        if k % 3 == 0:
            gw[r.integers(0, 300)] = 0.5 * s                       # exactly on the threshold
        if k % 3 == 1:
            gw[r.integers(0, 300)] = np.nextafter(0.5 * s, 1)      # just above it
        assert same(M.phc_success(gw, s), old_phc(gw, s))
        assert same(M.phc_success(gw, s), old_phc_max(gw, s))     # drift_success_child's form: equal on finite, non-empty input
    assert same(M.phc_success(np.array([0.5, 0.2]), 1.0, 0.5), old_phc(np.array([0.5, 0.2]), 1.0))


def test_drift_and_heading():
    for k in range(10):
        q, r = qpos_like(300 + k), qpos_like(400 + k)
        assert same(M.drift(q, r), old_drift_body(q, r)) and same(M.drift(q, r, "body"), old_drift_body(q, r))
        for s in (0.611036, 1.364408905613045, 1.0):
            assert same(M.drift(q, r, "adult", s), old_drift_adult(q, r, s))
        assert same(M.heading_error(q, r), old_heading(q, r))
        q32 = q.astype(np.float32).astype(np.float64)
        assert same(M.heading_error(q32, r), old_heading(q32, r))
    try:
        M.drift(q, r, "metres")
        raise AssertionError("drift accepted an unknown unit")
    except ValueError:
        pass


def test_legacy_root_and_height():
    for k in range(10):
        q, r = qpos_like(500 + k, drift=8.0), qpos_like(600 + k, drift=8.0)
        if k % 2:
            q[:, 3:7] = r[:, 3:7]                                     # rotation passes: position decides
        bad, pos, s_float, s_bool = old_root(q, r)
        nb, npos = M.root_fail(q, r)
        assert same(nb, bad) and same(npos, pos)
        assert same(M.root_success(q, r), s_bool) and same(float(M.root_success(q, r)), s_float)
        assert same(M.height_fall(q, r), old_height(q, r))


def test_hand_fall():
    for k in range(30):
        h = floor_like(700 + k, P=1)[:, 0]; hr = floor_like(800 + k, P=1)[:, 0] * (3 if k % 2 else 1)
        assert same(M.hand_fall(h, hr), old_hand_fall(h, hr))
        assert same(M.hand_fall(h, hr, touch=0.01, near=0.05, motion_s=0.5), old_hand_fall(h, hr))


def test_world_positions_and_body_error():
    R = rollouts()
    import eval_metrics as E
    for name, xml, smpl in (("child", E.XML["child"], True), ("g1", E.XML["g1"], False)):
        body = M.metric_body(xml, smpl); m = body.m; d2 = mujoco.MjData(m)
        assert M.metric_body(xml, smpl) is body                       # one per (xml, smpl) and process
        st = "move-ego-0-2_1"; ref = R[f"ref__{name}__{st}"].astype(np.float64)[1:]
        q = R[f"{name}__{st}__es_best"]
        Pw = M.world_positions(m, body.d, ref); assert same(Pw, old_world(m, d2, ref))
        P = M.world_positions(m, body.d, q.astype(np.float64)); assert same(P, old_world(m, d2, q.astype(np.float64)))
        assert same(M.world_positions(m, body.d, q), old_world(m, d2, q))                           # float32 rows
        T = min(len(q), len(ref))
        assert same(M.mean_body_error(P[:T], Pw[:T]), np.linalg.norm(P[:T] - Pw[:T], axis=-1).mean(1))


# ------------------------------------------------------------------ the drivers: now vs a997a23

def test_task_page_stats():
    """loes_noanchor / body_52_test / g1_small_test stats(): MetricSuite.task_page now vs the old copies"""
    R = rollouts()
    sys.argv = sys.argv[:1]
    on, nn = old_and_new("loes_noanchor")
    ob, nb = old_and_new("body_52_test")
    og, ng = old_and_new("g1_small_test")
    import eval_metrics as E
    st = "move-ego-0-2_1"
    for name, xml, smpl, s in (("child", E.XML["child"], True, nn.CONF["child"]["s"]), ("payload", nb.BODIES["payload"]["xml"], True, 1.0),
                               ("g1", E.XML["g1"], False, ng.S)):
        body = E.Body(xml, smpl); ref = R[f"ref__{name}__{st}"].astype(np.float64)[1:]
        Pw = old_world(body.m, mujoco.MjData(body.m), ref); _, Dr = body.part_floor(ref)
        for k in sorted(k for k in R if k.startswith(f"{name}__{st}__")):
            q = R[k]; want = on.stats(body, q, ref, Pw, Dr, s)
            assert same(nn.stats(body, q, ref, Pw, Dr, s), want), k
            assert same(M.MetricSuite.task_page(body, s).reference(ref).values(q), want), k          # Pw, Dr computed by the suite
            assert list(M.MetricSuite.task_page(body, s).reference(ref, Pw, Dr)(q)) == list(nn.KEYS)
            if name == "payload":
                ob.body = nb.body = body; assert same(nb.stats(q, ref, Pw, Dr), ob.stats(q, ref, Pw, Dr)), k
            if name == "g1":
                og.body = ng.body = body; assert same(ng.stats(q, ref, Pw, Dr), og.stats(q, ref, Pw, Dr)), k


def test_loes_weight_and_lsor_mpjae_stats():
    R = rollouts()
    ow, nw = old_and_new("loes_weight_test")
    ol, nl = old_and_new("lsor_mpjae_test")
    import eval_metrics as E
    body = E.Body(E.XML["child"], True)
    for mod in (ow, nw, ol, nl):
        mod.body = body
    for c in CHILD:
        st = c.split("/")[1]; ref = R[f"ref__child__{st}"].astype(np.float64)[1:]; Pw = ow.world(ref)
        assert same(nw.world(ref), Pw)
        for k in sorted(k for k in R if k.startswith(f"child__{st}__")):
            q = R[k]
            assert same(nw.stats(q, ref, Pw), ow.stats(q, ref, Pw)), k
            assert same(nl.stats(q, ref, Pw), ol.stats(q, ref, Pw)), k


def _fresh(d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def test_kpkd_phc_and_contact():
    R = rollouts()
    ok, nk = old_and_new("kpkd_test")
    out = _fresh(WORK / "kpkd")
    for mod in (ok, nk):
        mod.OUT = out
    jobs = []
    for c in CHILD:
        st = c.split("/")[1]; d = out / f"child_kp1_kd1/{st}_s0"; d.mkdir(parents=True)
        np.savez_compressed(d / "origin_z.npz", qpos=R[f"child__{st}__z0"]); np.savez_compressed(d / "best.npz", qpos=R[f"child__{st}__es_best"])
        jobs.append(("child", "kp1_kd1", c, 0))
    for j in jobs:
        assert same(nk.phc_one(j), ok.phc_one(j)), j
        assert same(nk.contact_one(j), ok.contact_one(j)), j
    shutil.rmtree(out)


def test_amass_baselines_one():
    files = {"outputs/amass_baselines/clips.txt": AMASS + "\n"}
    oa, na = old_and_new("amass_baselines", files)
    out = _fresh(WORK / "amass")
    for mod in (oa, na):
        mod.OUT = out
    src = np.load(GOLDEN / "tall_amass_lso/rollouts.npz"); summ = json.loads((GOLDEN / "tall_amass_lso/summary.json").read_text())
    pool = [src[k] for k in ("q_best_it2", "q_mean_it2", "q_best_it4", "q_mean_it4", "q_LSO", "q_LSO_best", "q_ER5", "q_Bg")]
    for i, k in enumerate(na.METHODS):
        d = out / "tall" / f"{AMASS}_{k}_s0"; d.mkdir(parents=True); (d / "summary.json").write_text(json.dumps(summ))
        arr = {"q_ER8": src["q_ER8"]}
        for j, g in enumerate(na.BODY["tall"]["G"]):
            arr[f"q_best_it{g}"] = pool[(i + 2 * j) % len(pool)]; arr[f"q_mean_it{g}"] = pool[(i + 2 * j + 1) % len(pool)]
        np.savez_compressed(d / "rollouts.npz", **arr)
    o, n = oa.one(("tall", AMASS)), na.one(("tall", AMASS))
    assert same(n[:2], o[:2]) and same(n[3], o[3])
    assert list(n[2]) == list(o[2])
    for k in o[2]:
        assert same(n[2][k], o[2][k]), k
    shutil.rmtree(out)


# --------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}")
        except (AssertionError, KeyError, ValueError, OSError, ImportError, subprocess.CalledProcessError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    if WORK.exists() and not any(WORK.iterdir()):
        WORK.rmdir()
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
