"""Tests of OOP refactor step 7 (docs/refactor/PLAN.md): the G1-only code path (model/g1_map.py G1Map, model/g1_env.py G1Env,
archived in step 7 to archive/model/) against the generic robot code the G1 goes through now (model/robot_map.py
RobotMap("g1"), model/robot_env.py RobotEnv("g1")), exact comparisons (dtype, shape, every bit: np.array_equal).

The old modules are executed from `git show BASE:model/g1_map.py` / `BASE:model/g1_env.py` (BASE = 543a303, the commit before
step 7) as module objects of their own, with __file__ at their old place (g1_map's FILL_JSON resolves as it did) and the
working directory at the repository (their relative XML paths); g1_env's `from model.g1_map import ...` gets the old g1_map.
  map       calibration (q_nom, calibration residuals, C of every pair, s); SMPL -> G1 (smpl_to_g1_qpos vs
            smpl_to_robot_qpos: G1 qpos and per-link IK residuals) over every frame of the golden clip warm-started as the
            roundtrip and the env do, cold starts, and random SMPL poses that push G1 joints into their limits (the
            per-chain re-solve from the nominal pose); G1 -> SMPL (g1_to_smpl_qpos vs robot_to_smpl_qpos, fill "fixed": the
            only fill G1Map has) on the mapped clip and on random G1 poses; the lowest point (_G1Lowest vs robot_map.Lowest)
            on the same poses. g1_map's _quat / _rotvec / as_euler call scipy without assume_valid, robot_map's with it.
  env       G1Env(gains="perjoint") vs RobotEnv("g1") (its default fill "fixed", gains ROBOTS["g1"]["gains_json"] =
            assets/robots/g1/gains_perjoint.json): every model parameter the physics reads (option, actuators, dofs), the
            reset to the clip's frame 0, then STEPS fixed random actions (some beyond [-1, 1], so the clip is exercised;
            the G1 falls, so the floor contacts are too): per step obs, SMPL-equivalent qpos, G1 qpos / qvel, ctrl,
            last_resid, last_atlim.
  targets   the stored G1 targets of the main tree (outputs/g1/roundtrip/<stem>/g1.npz and smpl_rt.npz, read only):
            move-ego-0-2_4 written by scripts/g1_roundtrip.py (G1Map, 10/20) re-made with RobotMap("g1"), and
            move-ego-0-2_1 written by scripts/robot_roundtrip.py (RobotMap, 11/09) re-made with G1Map; the roundtrip loops
            are those of the two scripts (smpl -> robot warm-started from the previous frame's joints, then back).
  gains     scripts/robot_gain_scaling.py --robot g1 re-makes assets/robots/g1/gains_perjoint.json (scripts/g1_gain_scaling.py's)
            bit for bit with the json's 18 clips in its order (sorted by stem), written to outputs/refactor_tests/test_g1_generic;
            g1_env.GAIN_SOURCE = ROBOTS["g1"]["gain_source"] actuator by actuator (in the env test).
Not covered: G1Env(gains="mass"), the pre-10/20 gains, which have no generic counterpart (dropped, user 11/12);
G1VecEnv / RobotVecEnv, whose workers make these same envs and pass reset / step through unchanged (RobotVecEnv's worker
also accepts a (frame 0, frame 1) tuple for the reference-velocity start, which G1Env does not have).

Run standalone (no pytest in the environment), like tests/test_moves.py (CPU only, ~1-2 min, < 2 GB):
    MUJOCO_GL=egl OMP_NUM_THREADS=1 uv run python tests/test_g1_generic.py
"""
import os
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
os.chdir(REPO)                                               # g1_map / g1_env open their XMLs by repository-relative paths

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

BASE = "543a303"                                             # the commit before step 7
CLIP = "move-ego-0-2/move-ego-0-2_1"                         # the golden clip (300 frames)
STEPS = 60
WORK = REPO / "outputs/refactor_tests/test_g1_generic"
RNG = lambda k=0: np.random.default_rng(20261112 + k)        # noqa: E731
F = {}                                                        # fixtures


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def check(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if same(a, b):
        return
    num = a.shape == b.shape and a.size and a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max() if num else "n/a"
    raise AssertionError(f"{name}: differs (dtype {a.dtype} / {b.dtype}, shape {a.shape} / {b.shape}, max |d| {d})")


def old_module(rel, name, inject=None):
    """the whole file `rel` at BASE executed as a fresh module; inject: {module name: module} visible to its imports"""
    src = subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType(name); mod.__file__ = str(REPO / rel)
    saved = {k: sys.modules.get(k) for k in inject or {}}
    sys.modules.update(inject or {})
    try:
        exec(compile(src, f"{BASE}:{rel}", "exec"), mod.__dict__)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def old_g1_map():
    if "g1_map" not in F:
        F["g1_map"] = old_module("model/g1_map.py", "old_g1_map")
    return F["g1_map"]


def maps():
    """(old G1Map, new RobotMap("g1")), made once"""
    if "maps" not in F:
        from model.robot_map import RobotMap
        F["maps"] = (old_g1_map().G1Map(), RobotMap("g1"))
    return F["maps"]


def clip(stem_path=CLIP):
    return np.load(REPO / f"data/origin_motion/{stem_path}.npz")["qpos"]


def roundtrip(fwd, back, ref, g_qadr):
    """the roundtrip loop of g1_roundtrip.py / robot_roundtrip.py: SMPL -> robot warm-started, then robot -> SMPL"""
    qg, resid, q = [], [], None
    for x in ref:
        g, r = fwd(x, q); qg.append(g); resid.append(r); q = g[g_qadr]
    qg = np.array(qg)
    return qg, np.array(resid), np.array([back(g) for g in qg])


# ------------------------------------------------------------------ 1. the map

def test_map_calibration():
    old, new = maps()
    check("q_nom", old.q_nom, new.q_nom)
    assert list(old.calib) == list(new.calib) and all(old.calib[k] == new.calib[k] for k in old.calib), "calib"
    assert [b for b, _ in old_g1_map().PAIRS] == list(old.C) == list(new.C) and old_g1_map().PAIRS == new.pairs, "pairs"
    for b in old.C:
        check(f"C[{b}]", old.C[b], new.C[b])
    assert type(old.s) is type(new.scale) and old.s == new.scale, (old.s, new.scale)
    for k, k_new in (("g_qadr", "robot_qadr"), ("g_dadr", "robot_dadr"), ("g_range", "robot_range"), ("s_qadr", "smpl_qadr"),
                     ("anc", "anc"), ("w", "w")):
        check(k, getattr(old, k), getattr(new, k_new))
    # step 11: the G1Map-era names stay readable on RobotMap, read-only, as the same objects
    from model.robot_map import OLD_NAMES
    for k, k_new in OLD_NAMES.items():
        x, y = getattr(new, k), getattr(new, k_new)
        assert x is y or (callable(x) and x.__func__ is y.__func__), k
    try:
        new.g_jnt = []
        raise AssertionError("RobotMap.g_jnt is writable")
    except AttributeError:
        pass
    check("link_ids", np.array(old.link_ids), new.link_ids)
    # chain labels differ (g1_map "left_leg", robot_map "Lleg"); the re-solve only uses which joints / links share a chain
    # (chains are disjoint, so the order np.unique visits them in does not matter): the same partition, one label each
    pair = set(zip(old.j_chain, new.j_chain)) | set(zip(old.l_chain, new.l_chain))
    assert len(pair) == len({o for o, _ in pair}) == len({n for _, n in pair}) == 5, sorted(pair)
    assert old.g_jnt == new.robot_jnt and old.s_names == new.smpl_names
    assert list(old.fill) == list(new.fill) and all(same(old.fill[b], new.fill[b]) for b in old.fill), "fill"


def test_map_smpl_to_robot_clip():
    old, new = maps(); ref = clip()
    a = roundtrip(old.smpl_to_g1_qpos, old.g1_to_smpl_qpos, ref, old.g_qadr)
    b = roundtrip(new.smpl_to_robot_qpos, new.robot_to_smpl_qpos, ref, new.robot_qadr)
    for k, x, y in zip(("G1 qpos", "IK residual", "SMPL round trip"), a, b):
        check(k, x, y)
    F["clip_qg"] = a[0]
    for t in range(0, len(ref), 23):                                           # cold starts
        x, y = old.smpl_to_g1_qpos(ref[t]), new.smpl_to_robot_qpos(ref[t])
        check(f"cold qpos t={t}", x[0], y[0]); check(f"cold resid t={t}", x[1], y[1])


def random_smpl(n, k):
    """SMPL poses far from the clip: root orientation random, hinges uniform over 1.5 x their ranges (+-pi if unlimited),
    so the G1 targets run into joint limits and the per-chain re-solve runs"""
    _, new = maps(); ms = new.m_smpl; rng = RNG(k)
    out = np.repeat(ms.qpos0[None], n, 0)
    q = rng.standard_normal((n, 4)); out[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    out[:, :3] = rng.uniform([-2, -2, 0.3], [2, 2, 1.2], (n, 3))
    r = np.where(ms.jnt_limited[new.smpl_jnt][:, None].astype(bool), 1.5 * ms.jnt_range[new.smpl_jnt], [[-np.pi, np.pi]])
    out[:, new.smpl_qadr] = rng.uniform(r[:, 0], r[:, 1], (n, len(new.smpl_jnt)))
    return out


def test_map_smpl_to_robot_limits():
    old, new = maps(); X = random_smpl(40, 1)
    lo, hi = new.robot_range[:, 0], new.robot_range[:, 1]
    q_old = q_new = None; hit = 0
    for i, x in enumerate(X):                                                   # warm-started through unrelated poses
        a, b = old.smpl_to_g1_qpos(x, q_old), new.smpl_to_robot_qpos(x, q_new)
        check(f"random qpos {i}", a[0], b[0]); check(f"random resid {i}", a[1], b[1])
        q_old, q_new = a[0][old.g_qadr], b[0][new.robot_qadr]
        hit += bool(np.any((q_new <= lo + 1e-6) | (q_new >= hi - 1e-6)))
    assert hit >= 10, f"only {hit} of 40 random poses reached a joint limit: the re-solve branch is barely tested"


def random_g1(n, k):
    _, new = maps(); rng = RNG(k); out = np.zeros((n, new.m_robot.nq))
    q = rng.standard_normal((n, 4)); out[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    out[:, :3] = rng.uniform([-2, -2, 0.2], [2, 2, 1.0], (n, 3))
    out[:, new.robot_qadr] = rng.uniform(new.robot_range[:, 0], new.robot_range[:, 1], (n, len(new.robot_jnt)))
    return out


def test_map_robot_to_smpl():
    old, new = maps()
    if "clip_qg" not in F:
        test_map_smpl_to_robot_clip()
    for name, Q in (("clip", F["clip_qg"]), ("random", random_g1(60, 2))):
        for i, g in enumerate(Q):
            check(f"{name} {i}", old.g1_to_smpl_qpos(g), new.robot_to_smpl_qpos(g))
            check(f"{name} {i} rotations", np.stack(list(old.smpl_rots_from_g1(g).values())),
                  np.stack(list(new.smpl_rots_from_robot(g).values())))


def test_lowest_point():
    from model.robot_map import Lowest
    old_map = old_g1_map()
    m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/g1/g1.xml")); d = mujoco.MjData(m)
    lo_old, lo_new = old_map._G1Lowest(m), Lowest(m)
    Q = np.concatenate([random_g1(60, 3), F.get("clip_qg", random_g1(10, 4))])
    for i, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        a, b = lo_old(d), lo_new(d)
        assert type(a) is type(b) and a == b, (i, a, b)


# ------------------------------------------------------------------ 2. the env

def test_env_perjoint():
    old_env = old_module("model/g1_env.py", "old_g1_env", {"model.g1_map": old_g1_map()})
    from model.robot_env import RobotEnv
    from model.robots import ROBOTS
    assert ROBOTS["g1"]["gains_json"] == old_env.GAINS_JSON
    A, B = old_env.G1Env(gains="perjoint"), RobotEnv("g1")
    for k in ("timestep", "integrator", "cone", "jacobian", "solver", "iterations", "tolerance", "noslip_iterations",
              "impratio", "gravity", "disableflags", "enableflags"):
        check(f"opt.{k}", getattr(A.m.opt, k), getattr(B.m.opt, k))
    for k in ("actuator_gaintype", "actuator_biastype", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
              "actuator_ctrllimited", "actuator_forcerange", "actuator_forcelimited", "actuator_gear", "actuator_trnid",
              "actuator_dyntype", "actuator_dynprm", "jnt_actfrcrange", "jnt_actfrclimited", "dof_damping", "dof_armature",
              "dof_frictionloss", "jnt_stiffness", "body_mass", "geom_friction"):
        check(k, getattr(A.m, k), getattr(B.m, k))
    check("kp", A.kp, B.kp); check("kd", A.kd, B.kd)
    check("actuator -> map joint", np.arange(A.m.nu), B.act_j)
    for i in range(A.m.nu):                                  # g1_env.GAIN_SOURCE = ROBOTS["g1"]["gain_source"] (gain scaling)
        j = A.m.actuator(i).name.replace("_joint", "")
        side = "L_" if j.startswith("left") else "R_" if j.startswith("right") else ""
        src = [side + h for h in old_env.GAIN_SOURCE[j.replace("left_", "").replace("right_", "")]]
        assert src == ROBOTS["g1"]["gain_source"][A.m.joint(A.m.actuator_trnid[i, 0]).name], (i, src)
    ref = clip()
    a, b = A.reset(ref[0]), B.reset(ref[0])
    check("reset obs", a[0], b[0]); check("reset qs", a[1], b[1])
    check("reset qpos", A.d.qpos, B.d.qpos); check("reset ctrl", A.d.ctrl, B.d.ctrl)
    U = RNG(5).uniform(-1.2, 1.2, (STEPS, A.ms.nu))
    fell = False
    for t, u in enumerate(U):
        a, b = A.step(u), B.step(u)
        check(f"t={t} obs", a[0], b[0]); check(f"t={t} qs", a[1], b[1])
        for k in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
            check(f"t={t} {k}", getattr(A.d, k), getattr(B.d, k))
        check(f"t={t} last_resid", A.last_resid, B.last_resid)
        assert type(A.last_atlim) is type(B.last_atlim) and A.last_atlim == B.last_atlim, t
        fell |= A.d.qpos[2] < 0.4
    assert fell, "the G1 did not fall: the floor contacts are barely tested"


# ------------------------------------------------------------------ 3. the stored targets

def stored(stem):
    d = REPO / "outputs/g1/roundtrip" / stem
    if not (d / "g1.npz").exists():
        raise unittest.SkipTest(f"{d} missing")
    return np.load(d / "g1.npz")["qpos"], np.load(d / "smpl_rt.npz")["qpos"]


def test_stored_targets():
    old, new = maps()
    for stem_path, fwd, back, qadr in (("move-ego-0-2/move-ego-0-2_4", new.smpl_to_robot_qpos, new.robot_to_smpl_qpos, new.robot_qadr),
                                        ("move-ego-0-2/move-ego-0-2_1", old.smpl_to_g1_qpos, old.g1_to_smpl_qpos, old.g_qadr)):
        qg, rt = stored(stem_path.split("/")[1])
        a = roundtrip(fwd, back, clip(stem_path), qadr)
        check(f"{stem_path} g1.npz", qg, a[0]); check(f"{stem_path} smpl_rt.npz", rt, a[2])


def test_gains_perjoint():
    """scripts/robot_gain_scaling.py --robot g1 re-makes assets/robots/g1/gains_perjoint.json (written by g1_gain_scaling.py)
    bit for bit, given its 18 clips in g1_gain_scaling's order (sorted by stem; the json's "clips"). In
    outputs/g1/search_clips.txt's own order the fit sums in another order: kp / kd then differ in the last bit or two."""
    import json
    want = json.loads((REPO / "assets/robots/g1/gains_perjoint.json").read_text())
    task = {c.split("/")[1]: c for c in (REPO / "outputs/g1/search_clips.txt").read_text().split()} \
        if (REPO / "outputs/g1/search_clips.txt").exists() else {}
    if not all((REPO / f"outputs/g1/roundtrip/{s}/g1.npz").exists() and s in task for s in want["clips"]):
        raise unittest.SkipTest("the G1 targets of the json's clips (outputs/g1, main tree) are missing")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "clips_sorted.txt").write_text("".join(f"{task[s]}\n" for s in want["clips"]))
    out = WORK / "gains_perjoint.json"
    subprocess.run([sys.executable, "scripts/robot_gain_scaling.py", "--robot", "g1", "--stride", str(want["stride"]),
                    "--clips-file", str((WORK / "clips_sorted.txt").relative_to(REPO)), "--out", str(out.relative_to(REPO))],
                   cwd=REPO, check=True, capture_output=True)
    got = json.loads(out.read_text())
    for k in ("kp", "kd", "k_group", "r2_group", "per_clip_k"):
        assert got[k] == want[k] and list(got[k]) == list(want[k]), k


# --------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]   # file order: the clip map first
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except unittest.SkipTest as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}", flush=True)
        except (AssertionError, KeyError, ValueError, OSError, ImportError, subprocess.CalledProcessError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
