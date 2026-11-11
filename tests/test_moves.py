"""Tests of OOP refactor step 4 (docs/refactor/PLAN.md): library code moved out of scripts into the package, exact comparisons.

Every moved helper is run against its own source at BASE (68c95f4, the commit before step 4: `git show`, the function's
definition extracted with ast and executed in a namespace of its own) on fixed inputs, and must give the same result (==,
np.array_equal, dtypes included):
  compute_batch_cost / _phys_weight_table / ESConfig  -> crossenbodiment.align_cost (AlignCostConfig); model/simple re-exports
  child_52_means.full / hand_fall / KEYS              -> crossenbodiment.metrics legacy_metrics / body_hand_fall / LEGACY_KEYS
  mse_vs_align_test.metrics                           -> crossenbodiment.metrics.pose_similarity
  bfm_velocity_check.resample                         -> crossenbodiment.retiming.resample
  g1_z0_rollout.act / motion / g1_up_z                -> crossenbodiment.robot_rollout
  rollout_z_trace.label, g1_axes_demo px / font / cam / project / label / hcat -> crossenbodiment.draw
  survey_body_videos CLIPS / BODY / ADULT(_H) / SUFFIXES / run_of / colours   -> crossenbodiment.survey, palette, configs
  multibody_corr's layout                             -> crossenbodiment.multibody_study
  the clip lists                                      -> configs/clips/*.txt, byte-identical to the main tree's files
Inputs: tests/golden/inputs/metrics_rollouts.npz (frozen saved rollouts), fixed random arrays. Existing runs and the main tree are
READ ONLY (MAIN, as in tests/test_registry.py).

Run standalone (no pytest in the environment), like tests/test_metrics.py:
    uv run python tests/test_moves.py
"""
import ast
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

BASE = "68c95f4"                                              # the commit before step 4
MAIN = Path(os.environ.get("CROSSENBODIMENT_MAIN", "/home/allen19/crossenbodiment"))   # the main tree (read only)
INP = REPO / "tests/golden/inputs"
WORK = REPO / "outputs/refactor_tests/test_moves"
RNG = lambda k=0: np.random.default_rng(20261113 + k)        # noqa: E731
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


def old_source(rel):
    return subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def old_defs(rel, names, ns=None):
    """the top-level definitions (functions, classes, assignments) `names` of file `rel` at BASE, executed in order into a
    namespace (ns: the globals they need); returns the namespace"""
    ns = dict(ns or {}); ns.setdefault("__name__", f"old_{Path(rel).stem}")
    tree = ast.parse(old_source(rel)); want = set(names); found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
            hit = {node.name}
        elif isinstance(node, ast.Assign):
            hit = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)} & want
        else:
            continue
        if hit:
            exec(compile(ast.Module([node], []), f"{BASE}:{rel}", "exec"), ns); found |= hit
    assert found == want, f"{rel}: not found at {BASE}: {sorted(want - found)}"
    return ns


def old_module(rel, name):
    """the whole file `rel` at BASE executed as a fresh module (its __main__ block not run)"""
    mod = types.ModuleType(name); mod.__file__ = str(REPO / rel)
    exec(compile(old_source(rel), f"{BASE}:{rel}", "exec"), mod.__dict__)
    return mod


def child_cases():
    R = rollouts()
    for k in sorted(R):
        if k.startswith("child__"):
            stem = k.split("__")[1]
            yield k, R[k].astype(np.float64), R[f"ref__child__{stem}"].astype(np.float64)


# ------------------------------------------------------------------ 1. compute_batch_cost, AlignCostConfig

def test_align_cost_config_defaults():
    from crossenbodiment.align_cost import AlignCostConfig
    import dataclasses
    es = old_module("model/simple/config.py", "old_config").ESConfig(device="cpu"); new = AlignCostConfig()
    for f in dataclasses.fields(new):
        if f.name in ("phys_weights", "phys_fall_ref"):         # not ESConfig fields: compute_batch_cost's getattr defaults
            continue
        assert same(getattr(es, f.name), getattr(new, f.name)), (f.name, getattr(es, f.name), getattr(new, f.name))
    assert not hasattr(es, "phys_weights") and not hasattr(es, "phys_fall_ref")
    assert (new.phys_weights, new.phys_fall_ref) == ("default", False)
    src = old_source("model/simple/train.py")
    assert 'getattr(cfg, "phys_weights", "default")' in src and 'getattr(cfg, "phys_fall_ref", False)' in src
    assert 'getattr(cfg, "align_discount", 1.0)' in src


def test_compute_batch_cost():
    from crossenbodiment import align_cost as A
    from model import losses
    old = old_defs("model/simple/train.py", ["_phys_weight_table", "compute_batch_cost"], {"np": np, "losses": losses})
    fk = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots_torque/child/robot_torque_full.xml"))
    by_stem = {}
    for k, q, ref in child_cases():
        by_stem.setdefault(k.split("__")[1], (ref, []))[1].append(q)
    for (ref, qs), shift in zip(by_stem.values(), (0, 1, 2)):     # frame offsets 0 and 1 (single_z_search --frame-offset)
        goal = ref[shift:]; T = min(min(len(q) for q in qs), len(goal))
        Q = np.stack([q[:T] for q in qs[:4]])
        for pw in losses.PHYS_WEIGHT_TABLES:
            for fall_ref in (False, True):
                for disc, la, lp in ((1.0, 1.0, 0.0), (0.97, 1.0, 1.0), (1.0, 0.0, 1.0)):
                    cfg = A.AlignCostConfig(lambda_align=la, lambda_phys=lp, phys_weights=pw, phys_fall_ref=fall_ref, align_discount=disc)
                    for rt in (False, True):
                        n = A.compute_batch_cost(fk, cfg, Q, [goal] * len(Q), return_terms=rt)
                        o = old["compute_batch_cost"](fk, cfg, Q, [goal] * len(Q), return_terms=rt)
                        assert same(list(n), list(o)), (pw, fall_ref, disc, la, lp, rt)
        # single_z_search's reference line: the goal's first frames scored against the goal itself
        cfg = A.AlignCostConfig(lambda_align=1.0, lambda_phys=0.0)
        g = goal[None, :120]
        assert same(list(A.compute_batch_cost(fk, cfg, g, [goal])), list(old["compute_batch_cost"](fk, cfg, g, [goal])))
    for pw in losses.PHYS_WEIGHT_TABLES:
        cfg = A.AlignCostConfig(phys_weights=pw)
        assert A._phys_weight_table(cfg) is old["_phys_weight_table"](cfg)


def test_model_simple_reexports():
    from crossenbodiment import align_cost as A
    import model.simple.train as T
    assert T.compute_batch_cost is A.compute_batch_cost and T._phys_weight_table is A._phys_weight_table


# ------------------------------------------------------------------ 2. child_52_means -> metrics

def test_legacy_metrics():
    import eval_metrics as E
    from crossenbodiment import metrics as M
    old = old_module("scripts/child_52_means.py", "old_child_52_means")
    assert M.LEGACY_KEYS == old.KEYS
    body = E.Body(E.XML["child"], True)
    for k, q, ref in child_cases():
        for r in (ref[1:], ref):
            for ratio in (1.0, 0.37):
                assert same(M.legacy_metrics(body, q, r, ratio), old.full(body, q, r, ratio)), k
            assert same(M.body_hand_fall(body, q, r), old.hand_fall(body, q, r)), k


# ------------------------------------------------------------------ 5. pose_similarity, resample, robot_rollout, draw

def test_pose_similarity():
    from crossenbodiment.metrics import pose_similarity
    from model import losses, kinematics as kin
    old = old_defs("scripts/mse_vs_align_test.py", ["metrics"], {"np": np})
    fk = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots_torque/child/robot_torque_full.xml"))
    for k, q, ref in child_cases():
        assert same(pose_similarity(fk, q, ref[1:], losses, kin), old["metrics"](fk, q, ref[1:], losses, kin)), k


def test_resample():
    from crossenbodiment.retiming import resample
    from scipy.spatial.transform import Rotation, Slerp
    old = old_defs("scripts/bfm_velocity_check.py", ["resample"], {"np": np, "Rotation": Rotation, "Slerp": Slerp})
    for k, q, ref in child_cases():
        for r in (0.5, 0.75, 1.0, 1.3, 2.0):
            assert same(resample(ref, r), old["resample"](ref, r)), (k, r)


def test_robot_rollout():
    import torch
    from crossenbodiment import robot_rollout as RR
    old = old_defs("scripts/g1_z0_rollout.py", ["act", "g1_up_z", "motion"], {"np": np, "torch": torch})
    R = rollouts()
    ref = R["ref__g1__move-ego-0-2_1"].astype(np.float64)
    for k in sorted(R):
        if k.startswith("g1__"):
            q = R[k].astype(np.float64)
            for r in (ref, ref[1:], ref[:100]):
                assert same(RR.motion(q, r), old["motion"](q, r)), k
            for x in q[::37, 3:7]:
                assert same(RR.g1_up_z(x), old["g1_up_z"](x))
    W = torch.as_tensor(RNG(1).standard_normal((40, 6)), dtype=torch.float32)
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(actor_std=0.2), _normalize=lambda o: 0.5 * o,
                                  _actor=lambda o, z, std: types.SimpleNamespace(mean=torch.tanh(torch.cat([o, z], 1) @ W + std)))
    z = torch.as_tensor(RNG(2).standard_normal((1, 8)), dtype=torch.float32)
    for obs in RNG(3).standard_normal((5, 32)):
        for batch in (16, 1, 3):
            assert same(RR.act(model, obs, z, "cpu", batch), old["act"](model, obs, z, "cpu", batch))


def test_draw_frame_label():
    from PIL import Image, ImageDraw
    from crossenbodiment.draw import frame_label
    old = old_defs("scripts/rollout_z_trace.py", ["label"], {"np": np, "Image": Image, "ImageDraw": ImageDraw})
    frame = (RNG(4).random((120, 160, 3)) * 255).astype(np.uint8)
    for text, sub in (("gen 4", None), ("gen -1 (z0)", "cost 0.31"), ("", "")):
        assert same(frame_label(frame, text, sub), old["label"](frame, text, sub))


def test_draw_panels():
    from PIL import Image, ImageDraw, ImageFont
    from crossenbodiment import draw as D
    old = old_defs("scripts/g1_axes_demo.py", ["FONT", "S", "K", "px", "cam", "font", "project", "label", "hcat"],
                   {"np": np, "mujoco": mujoco, "Path": Path, "Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont})
    assert (D.FONT, D.S, D.K) == (old["FONT"], old["S"], old["K"])
    if not Path(D.FONT).exists():
        raise unittest.SkipTest(f"{D.FONT} missing")
    K0 = D.K
    try:
        for K in (old["K"], 1080 / 1.25 / 420):                   # g1_axes_demo's own, robot_pairs.py's
            D.K = old["K"] = K
            assert [D.px(v) for v in (0, 1, 4.5, 22, 37.25)] == [old["px"](v) for v in (0, 1, 4.5, 22, 37.25)]
            assert D.font(22).size == old["font"](22).size
            img = (RNG(5).random((200, 300, 3)) * 255).astype(np.uint8)
            for lines in (["title"], ["a long title " * 6, "the subtitle, which wraps as well " * 3]):
                assert same(D.label(img, lines), old["label"](img, lines))
            P = [(img[:150], ), (img, )]
            assert same(D.hcat([p[0] for p in P]), old["hcat"]([p[0] for p in P]))
    finally:
        D.K = K0
    c, co = D.cam(140, (0, 0, 0.6), dist=1.9, el=-5), old["cam"](140, (0, 0, 0.6), dist=1.9, el=-5)
    assert (c.type, c.azimuth, c.elevation, c.distance, list(c.lookat)) == (co.type, co.azimuth, co.elevation, co.distance, list(co.lookat))
    cam = types.SimpleNamespace(forward=[0.1, 0.9, -0.2], up=[0.0, 0.2, 0.97], pos=[0.3, -2.5, 1.1], frustum_near=0.01, frustum_top=0.004)
    r = types.SimpleNamespace(scene=types.SimpleNamespace(camera=[cam]))
    pts = RNG(6).standard_normal((5, 3))
    assert same(D.project(r, pts, 1080, 1080), old["project"](r, pts, 1080, 1080))


# ------------------------------------------------------------------ 3. survey data, palette

def test_survey_data():
    from crossenbodiment import palette, survey
    old = old_module("scripts/survey_body_videos.py", "old_survey_body_videos")
    assert same(survey.CLIPS, old.CLIPS) and same(survey.ADULT, old.ADULT) and same(survey.ADULT_H, old.ADULT_H)
    assert same(survey.SUFFIXES, old.SUFFIXES)
    assert same(survey.BODY, old.BODY) and all(list(survey.BODY[b]) == list(old.BODY[b]) for b in old.BODY)
    assert json.dumps(survey.BODY) == json.dumps(old.BODY)                 # key order and float spelling (specs are json)
    assert same([palette.GREY, palette.RED, palette.BLUE], [old.GREY, old.RED, old.BLUE])
    import survey_body_videos as S                                         # the script imports them back
    assert S.BODY is survey.BODY and S.CLIPS is survey.CLIPS and (S.GREY, S.RED, S.BLUE) == (palette.GREY, palette.RED, palette.BLUE)
    assert S.REPO == old.REPO and S.PREV == old.PREV and S.ENV == old.ENV


def test_survey_run_of():
    from crossenbodiment import survey
    old = old_module("scripts/survey_body_videos.py", "old_survey_body_videos")
    root = WORK / "survey_root"
    if root.exists():
        shutil.rmtree(root)
    costs = {("fat", "jump-2_6"): {"_s0": 0.4, "_sig015_s1": 0.2, "_g512_s0": 0.3}, ("limit_leg", "headstand_4"): {"_s0": 0.5},
             ("heavy_arm", "move-ego-0-2_4"): {"_sig015_cont2_s2": 0.1, "_sig015_s3": 0.1}}        # a tie: the first in SUFFIXES
    for (name, stem), d in costs.items():
        for suf, c in d.items():
            f = root / f"outputs/user_survey_search/{name}/{stem}{suf}/summary.json"; f.parent.mkdir(parents=True)
            f.write_text(json.dumps({"best": {"cost": c}}))
    old.REPO = root
    import survey_body_videos as S
    S0 = S.REPO
    try:
        S.REPO = root
        for name, stem in list(costs) + [("child", "rotate-x--5-0.8_4"), ("child", "jump-2_7")]:
            o = old.run_of(name, stem)
            assert survey.run_of(name, stem, root=root) == o == S.run_of(name, stem), (name, stem)
    finally:
        S.REPO = S0
    shutil.rmtree(root)


# ------------------------------------------------------------------ 5. multibody_corr's layout

def test_multibody_study():
    from crossenbodiment import multibody_study as MB
    old = old_module("scripts/multibody_corr.py", "old_multibody_corr")
    for k in ("OUT", "LOG", "ADULT", "GRID", "AXIS", "NAMED", "GROUPS", "BODIES", "BETA_KEYS", "SEEDS", "GENS", "PAIRS", "N_RANDOM",
              "E_TEST", "E_ARMS", "E_K"):
        assert same(getattr(MB, k), getattr(old, k)), k
    for b, c, s in (("child", "jump-2/jump-2_6", 0), ("gl0500a0550", "crawl-0.4-2-d/crawl-0.4-2-d_8", 1)):
        assert MB.cell_dir(b, c, s) == old.cell_dir(b, c, s) and MB.ref_path(b, c) == old.ref_path(b, c)
    assert MB.beta("child") == old.beta("child")
    if (MB.OUT / "clips.txt").exists():
        assert MB.clips() == old.clips()
    import multibody_corr as P                                             # the driver imports them back
    assert P.OUT is MB.OUT and P.E_TEST is MB.E_TEST and P.cell_dir is MB.cell_dir


# ------------------------------------------------------------------ 6. clip lists

def test_clip_lists():
    from crossenbodiment import paths
    for name, rel in ((paths.CHILD_52_TASK, "outputs/child_52/clips.txt"), (paths.AMASS_LOCOMOTION_5S, "outputs/amass_baselines/clips.txt")):
        mine, main = REPO / paths.clips_file(name), MAIN / rel
        if not main.exists():
            raise unittest.SkipTest(f"{main} missing")
        assert mine.read_bytes() == main.read_bytes(), f"{mine} differs from {main}"
        c = paths.read_clips(name)
        assert c == [l.strip() for l in open(main)] == main.read_text().split() == [l.strip() for l in open(main) if l.strip()], name
        assert len(c) == {paths.CHILD_52_TASK: 52, paths.AMASS_LOCOMOTION_5S: 68}[name]
    old = old_module("scripts/survey_body_videos.py", "old_survey_body_videos")
    assert paths.read_clips(paths.USER_SURVEY) == old.CLIPS


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
