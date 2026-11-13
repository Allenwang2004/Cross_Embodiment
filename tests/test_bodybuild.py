"""Tests of OOP refactor step 9 (docs/refactor/PLAN.md): the body-building pipeline (crossembodiment/bodybuild) against the scripts it
replaced, exact comparisons (file bytes, np.array_equal).

The old scripts are BASE's (2bdb2fe, the commit before step 9), from `git show`, run as subprocesses in a scratch repository root
(WORK/oldrepo: their scripts/, assets/robots/adult and data/origin_motion as read-only symlinks to the repository's, every other
folder real), since they find their repository from their own path. The new builders write under WORK/new (robots_dir, data_dir).
  beta        BETA_AXES is one list: model/dataset.py's and scripts/scale_robot.py's AXES are the package's object, equal to BASE's two
  parsers     every wrapper script keeps its CLI: the parser main() builds (parse_args intercepted) == BASE's, action by action
  variants    heavy_arm_body.py (x5 whole arm; x80 Wrist + Hand), lock_joint_body.py (--joint R_Knee_x --deg 10; --user-lock --solref
              0.005; --no-lift), limit_joint_body.py (--lock-body adult_lockRleg_bent), elderly_body.py (default; --restrict both
              --limit-hip-extension), strength_rollout.py build (gear, cap) vs HeavyLimb / LockJoints / LimitJoints / Elderly /
              ActuatorStrength: robot.xml, the references, elderly.json, the slowed walk; parameter.json keeps BASE's keys
  stored      the limited leg of the Task page (assets/robots/adult_lockRleg_bent_lim, its references in data/, read only) and
              adult_s08_gear / adult_s08_cap rebuilt byte for byte
  retarget    qpos_retarget.py's functions (BASE) vs bodybuild.retarget on a clip of every kind of geom contact
  pipeline    BodyPipeline on 2 motion folders: every step, then a second run skips every step; keep_tasks prune and the retarget
              step's done check after it; the registry entry of a new body loads (register() into a copy of configs/bodies.yaml)
The full rebuilds (len0.8_wid1.2, the child, AMASS, the gain variants) are tests/golden/verify_bodies.py.

Run standalone (no pytest in the environment), CPU only, ~2 min:
    MUJOCO_GL=egl OMP_NUM_THREADS=1 uv run python tests/test_bodybuild.py
"""
import argparse
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

import numpy as np  # noqa: E402

BASE = "2bdb2fe"                                              # the commit before step 9
WORK = REPO / "outputs/refactor_tests/test_bodybuild"
CLIP = "move-ego-0-2/move-ego-0-2_4"                         # the clip the variant scripts default to / were built on
OLD = ("heavy_arm_body", "lock_joint_body", "limit_joint_body", "elderly_body", "qpos_retarget", "scale_robot",
       "export_skeleton_json", "torque_ratio_across_motions", "torque_aggregate_motion_k", "torque_scale_actuators",
       "torque_ratio_per_joint", "scale_body_gains", "strength_rollout")


def old_source(rel):
    return subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def same_bytes(a, b):
    a, b = Path(a), Path(b)
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


_ROOT = {}


def oldrepo():
    if "old" not in _ROOT:
        root = WORK / "oldrepo"
        if root.exists():
            shutil.rmtree(root)
        (root / "scripts").mkdir(parents=True)
        for s in OLD:
            (root / "scripts" / f"{s}.py").write_text(old_source(f"scripts/{s}.py"))
        for rel in ("assets/robots/adult", "data/origin_motion"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).symlink_to(REPO / rel)
        (root / "outputs").mkdir()
        _ROOT["old"] = root
    return _ROOT["old"]


def run_old(script, *args):
    root = oldrepo()
    r = subprocess.run([sys.executable, f"scripts/{script}.py", *map(str, args)], cwd=root, capture_output=True, text=True,
                       env=dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="1"))
    assert r.returncode == 0, f"old {script} {args}: {r.stderr[-2000:]}"
    return root


def new_dirs():
    d = WORK / "new"
    return d / "assets/robots", d / "data", d / "outputs"


# ------------------------------------------------------------------ 1. one BETA_AXES

def test_beta_axes_one_source():
    from crossembodiment.bodybuild.beta import BETA_AXES, BODY_GROUPS, PRESETS
    import model.dataset as D
    import scale_robot as S
    assert D.BETA_AXES is BETA_AXES and S.AXES is BETA_AXES
    import ast
    ns = {}
    tree = ast.parse(old_source("model/dataset.py"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "BETA_AXES" for t in n.targets))
    exec(compile(ast.Module([node], []), "old_dataset", "exec"), ns)
    old_sr = types.ModuleType("old_sr"); old_sr.__file__ = str(REPO / "scripts/scale_robot.py")
    exec(compile(old_source("scripts/scale_robot.py"), "old_scale_robot", "exec"), old_sr.__dict__)
    assert BETA_AXES == ns["BETA_AXES"] == old_sr.AXES
    assert BODY_GROUPS == old_sr.BODY_GROUPS and PRESETS == old_sr.PRESETS


# ------------------------------------------------------------------ 2. the wrappers' CLIs

class _Parser(Exception):
    def __init__(self, parser):
        self.parser = parser


def grab_parser(mod_dict, name):
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        raise _Parser(self)
    argparse.ArgumentParser.parse_args = grab
    try:
        mod_dict["main"]()
    except _Parser as e:
        return e.parser
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise AssertionError(f"{name}: main() did not parse")


def module_from(src, name, path):
    mod = types.ModuleType(name); mod.__file__ = str(path)
    exec(compile(src, f"{name}", "exec"), mod.__dict__)
    return mod


FIELDS = ("option_strings", "dest", "default", "choices", "nargs", "required", "help", "const", "metavar", "type")
# deliberate help-text changes (the old text described an older law)
HELP_CHANGED = {("torque_aggregate_motion_k", "joint_dynamics")}


def test_wrapper_parsers():
    old_tsa = module_from(old_source("scripts/torque_scale_actuators.py"), "scripts.torque_scale_actuators",
                          REPO / "scripts/torque_scale_actuators.py")
    sys.modules.setdefault("scripts.torque_scale_actuators", old_tsa)   # the old aggregate script imports it (archived now)
    try:
        for name in ("scale_robot", "export_skeleton_json", "qpos_retarget", "torque_ratio_across_motions",
                     "torque_aggregate_motion_k", "scale_body_gains", "heavy_arm_body", "lock_joint_body", "limit_joint_body",
                     "elderly_body", "strength_rollout"):
            path = REPO / "scripts" / f"{name}.py"
            old = grab_parser(module_from(old_source(f"scripts/{name}.py"), f"old_{name}", path).__dict__, name)
            new = grab_parser(module_from(path.read_text(), f"new_{name}", path).__dict__, name)
            oa = [a for a in old._actions]; na = [a for a in new._actions]
            assert [a.dest for a in oa] == [a.dest for a in na], (name, [a.dest for a in oa], [a.dest for a in na])
            for x, y in zip(oa, na):
                for f in FIELDS:
                    if f == "help" and (name, x.dest) in HELP_CHANGED:
                        continue
                    assert getattr(x, f) == getattr(y, f), (name, x.dest, f, getattr(x, f), getattr(y, f))
                assert type(x) is type(y), (name, x.dest)
    finally:
        if sys.modules.get("scripts.torque_scale_actuators") is old_tsa:
            del sys.modules["scripts.torque_scale_actuators"]


# ------------------------------------------------------------------ 3. the adult-size variants, old script vs builder

def _params_keep_old(new, old):
    n, o = json.loads(Path(new).read_text()), json.loads(Path(old).read_text())
    assert all(k in n and n[k] == o[k] for k in o), (new, {k: (o[k], n.get(k)) for k in o if n.get(k) != o[k]})
    assert {"name", "kind", "parent", "builder", "edit"} <= set(n), n.keys()


def test_heavy_limb():
    from crossembodiment.bodybuild import CopyReference, HeavyLimb
    rob, data, _ = new_dirs()
    for flags, kw in (((), dict(scale=5)), (("--parts", "Wrist,Hand"), dict(scale=80, parts=["Wrist", "Hand"]))):
        root = run_old("heavy_arm_body", "--scale", kw["scale"], *flags, "--clip", CLIP)
        b = HeavyLimb(kw["scale"], "R", kw.get("parts", HeavyLimb.PARTS), robots_dir=rob)
        b.build(log=lambda *_: None)
        old = root / "assets/robots" / b.name
        assert same_bytes(b.xml, old / "robot.xml"), b.name
        _params_keep_old(b.parameter_json, old / "parameter.json")
        ref = CopyReference().write_clip(REPO / f"data/origin_motion/{CLIP}.npz", data / b.name / "retargeting_motion" / f"{CLIP}.npz")
        assert same_bytes(ref, root / f"data/{b.name}/retargeting_motion/{CLIP}.npz")
        assert b.arm_mass()[1] > b.arm_mass()[0]


def _lock(name, locks, solref, lift=True):
    from crossembodiment.bodybuild import LockJoints
    rob, data, _ = new_dirs()
    b = LockJoints(name, locks, solref=solref, robots_dir=rob)
    b.build(log=lambda *_: None)
    p = b.reference(lift=lift).write_clip(REPO / f"data/origin_motion/{CLIP}.npz", data / name / "retargeting_motion" / f"{CLIP}.npz")
    return b, p


def test_lock_joints():
    import lock_joint_body as L
    cases = ((("--joint", "R_Knee_x", "--deg", 10), "adult_lockRKneex10", [("R_Knee_x", 10.0)], None, True),
             (("--user-lock", "--solref", 0.005), "adult_lockRleg_bent", [(k, float(v)) for k, v in L.LOCK_DEG.items()], 0.005, True),
             (("--lock", "R_Knee_x=20,R_Ankle_x=-5", "--label", "adult_lock_nolift", "--no-lift"), "adult_lock_nolift",
              [("R_Knee_x", 20.0), ("R_Ankle_x", -5.0)], None, False))
    for flags, name, locks, solref, lift in cases:
        root = run_old("lock_joint_body", "--clip", CLIP, *flags)
        b, ref = _lock(name, locks, solref, lift)
        old = root / "assets/robots" / name
        assert same_bytes(b.xml, old / "robot.xml"), name
        _params_keep_old(b.parameter_json, old / "parameter.json")
        assert same_bytes(ref, root / f"data/{name}/retargeting_motion/{CLIP}.npz"), name
        assert b.lock_error(np.load(ref)["qpos"][0]) < 5.0


def test_limit_joints():
    from crossembodiment.bodybuild import CopyReference, LimitJoints
    import lock_joint_body as L
    rob, data, _ = new_dirs()
    root = run_old("lock_joint_body", "--clip", CLIP, "--user-lock", "--solref", 0.005)
    root = run_old("limit_joint_body", "--lock-body", "adult_lockRleg_bent", "--clip", CLIP)
    _lock("adult_lockRleg_bent", [(k, float(v)) for k, v in L.LOCK_DEG.items()], 0.005)
    b = LimitJoints("adult_lockRleg_bent", robots_dir=rob)
    b.build(log=lambda *_: None)
    old = root / "assets/robots/adult_lockRleg_bent_lim"
    assert same_bytes(b.xml, old / "robot.xml")
    _params_keep_old(b.parameter_json, old / "parameter.json")
    ref = CopyReference().write_clip(b.reference_source(CLIP, data), data / b.name / "retargeting_motion" / f"{CLIP}.npz")
    assert same_bytes(ref, root / f"data/adult_lockRleg_bent_lim/retargeting_motion/{CLIP}.npz")
    assert b.limit_error(np.load(ref)["qpos"][0]) < 5.0


def test_elderly():
    from crossembodiment.bodybuild import Elderly
    rob, data, outs = new_dirs()
    for flags, kw in (((), {}), (("--restrict", "both", "--limit-hip-extension", "--label", "adult_elderly_both"),
                                 dict(restrict="both", limit_hip_extension=True, name="adult_elderly_both"))):
        root = run_old("elderly_body", "--clip", CLIP, *flags)
        b = Elderly(CLIP, robots_dir=rob, **kw)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            b.plan(); b.build(); ref = b.reference(data, outs)
        old = root / "assets/robots" / b.name
        assert same_bytes(b.xml, old / "robot.xml"), b.name
        assert same_bytes(b.dir / "elderly.json", old / "elderly.json"), b.name
        _params_keep_old(b.parameter_json, old / "parameter.json")
        assert same_bytes(ref, root / f"data/{b.name}/retargeting_motion/{CLIP}.npz"), b.name
        assert same_bytes(outs / "elderly" / f"{b.name}_young_slow.npz", root / f"outputs/elderly/{b.name}_young_slow.npz")


def test_actuator_strength():
    import ast
    from crossembodiment.bodybuild import ActuatorStrength
    import mujoco
    rob, _, _ = new_dirs()
    tree = ast.parse(old_source("scripts/strength_rollout.py"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    old_rob = WORK / "old_strength"
    ns = dict(REPO=old_rob, ADULT=REPO / "assets/robots/adult", mujoco=mujoco)
    exec(compile(ast.Module([fn], []), "old_build", "exec"), ns)
    for what in ("gear", "cap"):
        name = f"adult_s08_{what}"
        old = ns["build"](name, 0.8, what)
        b = ActuatorStrength(name, 0.8, what, robots_dir=rob)
        b.build(log=lambda *_: None)
        assert same_bytes(b.xml, old), name
        assert same_bytes(b.xml, REPO / "assets/robots" / name / "robot.xml"), f"{name}: stored"
        _params_keep_old(b.parameter_json, old.parent / "parameter.json")


def test_stored_limited_leg():
    """the Task page's limited leg (assets/robots/adult_lockRleg_bent_lim) and its references (data/, read only) rebuilt exactly"""
    rob, data, _ = new_dirs()
    from crossembodiment.bodybuild import LimitJoints, LockJoints
    import lock_joint_body as L
    lock = LockJoints("adult_lockRleg_bent", [(k, float(v)) for k, v in L.LOCK_DEG.items()], solref=0.005, robots_dir=rob)
    lock.build(log=lambda *_: None)
    assert same_bytes(lock.xml, REPO / "assets/robots/adult_lockRleg_bent/robot.xml")
    lim = LimitJoints("adult_lockRleg_bent", robots_dir=rob)
    lim.build(log=lambda *_: None)
    assert same_bytes(lim.xml, REPO / "assets/robots/adult_lockRleg_bent_lim/robot.xml")
    stored = REPO / "data/adult_lockRleg_bent_lim/retargeting_motion"
    r = lock.reference(lift=True)
    clips = sorted(stored.rglob("*.npz"))
    for c in clips[::max(1, len(clips) // 6)]:
        clip = str(c.relative_to(stored).with_suffix(""))
        p = r.write_clip(REPO / f"data/origin_motion/{clip}.npz", data / "check_lim" / f"{clip}.npz")
        assert same_bytes(p, c), clip


# ------------------------------------------------------------------ 4. the retarget functions

def test_retarget_functions():
    import mujoco
    from crossembodiment.bodybuild import retarget as R
    old = module_from(old_source("scripts/qpos_retarget.py"), "old_qpos_retarget", REPO / "scripts/qpos_retarget.py")
    m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/child/robot.xml"))
    s = old.load_root_rest_height(REPO / "assets/robots/child/skeleton.json") / old.load_root_rest_height(REPO / "assets/robots/adult/skeleton.json")
    assert s == R.SameTopologyRetargeter(REPO / "assets/robots/adult/skeleton.json", REPO / "assets/robots/child/skeleton.json",
                                         REPO / "assets/robots/child/robot.xml").scale
    for clip in ("headstand/headstand_0", "crawl-0.4-0-d/crawl-0.4-0-d_0", CLIP):
        q = np.load(REPO / f"data/origin_motion/{clip}.npz")["qpos"]
        a, b = old.retarget_qpos(q, s), R.retarget_qpos(q, s)
        assert np.array_equal(a, b)
        oa, ob = old.ground_correct_qpos(m, a), R.ground_correct_qpos(m, b)
        assert np.array_equal(oa[0], ob[0]) and oa[1:] == ob[1:], clip


# ------------------------------------------------------------------ 5. the pipeline: steps, skips, prune, registry

def test_pipeline_steps_and_skips():
    from crossembodiment.bodybuild import BodyPipeline, BuildRoots, ScaledBody, registry_entry, register
    from crossembodiment.bodies import BodyRegistry, CONFIG
    roots = BuildRoots(WORK / "pipe")
    if roots.root.exists():
        shutil.rmtree(roots.root)
    motions = ["jump-2", "move-ego-0-2", "headstand"]
    keep = WORK / "keep_tasks.txt"; keep.parent.mkdir(parents=True, exist_ok=True); keep.write_text("move-ego-0-2\n")

    def pipe():
        b = ScaledBody.from_len_wid(0.9, 1.1, name="test_len0.9_wid1.1", robots_dir=roots.robots)
        return BodyPipeline(b, roots, motions=motions, workers=2, keep_tasks=str(keep), quiet=True)
    res = {r.step: r.status for r in pipe().run()}
    assert res == dict(shape="done", skeleton="done", retarget="done", amass="n/a", torque_ratio="done", aggregate="done",
                       physics="done", prune="done", registry="done"), res
    p = pipe()
    assert sorted(d.name for d in p.retarget_dir.iterdir()) == ["move-ego-0-2"]
    res2 = {r.step: r.status for r in p.run()}
    assert all(res2[s] == "skipped" for s in ("shape", "skeleton", "retarget", "torque_ratio", "aggregate", "physics", "prune")), res2
    k = json.loads((roots.robots / "test_len0.9_wid1.1/parameter.json").read_text())
    assert k["kind"] == "smpl_scaled" and k["edit"]["len"] == 0.9 and k["leg_girth"] == round(0.9 * 1.1, 6) and "split" not in k
    cfg = WORK / "bodies.yaml"; shutil.copy(CONFIG, cfg)
    register(registry_entry(p.b), cfg)
    reg = BodyRegistry.load(cfg)
    spec = reg["test_len0.9_wid1.1"]
    assert spec.kind == "smpl_scaled" and spec.physics_xml == "assets/robots_torque/test_len0.9_wid1.1/robot_torque_full.xml"
    assert set(BodyRegistry.load(CONFIG).names()) | {"test_len0.9_wid1.1"} == set(reg.names())
    assert (roots.logs / "test_len0.9_wid1.1.bodies.yaml").exists() and (roots.logs / "test_len0.9_wid1.1.log").exists()


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except unittest.SkipTest as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}", flush=True)
        except (AssertionError, KeyError, ValueError, OSError, ImportError, RuntimeError, StopIteration,
                subprocess.CalledProcessError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
