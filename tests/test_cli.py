"""Tests of OOP refactor step 6 (docs/refactor/PLAN.md): the four search scripts keep their CLI.

The old parser of each script is taken from its source at BASE (fa6c5f9, the commit before step 6): the module is executed
from `git show` and its main() run with ArgumentParser.parse_args patched to hand back the parser instead of parsing. The new
parser is the script's build_parser(). Every action must agree: option strings, dest, default, type (the same function, or
for a lambda the same code), choices, nargs, required, help, const, the action's class; and so must the parser's
description / prog settings that argparse prints. Also checked: the names other scripts import from these modules
(single_z_search.rollout / device_arg / project_z / rank_normalize / ang_pair / OBJECTIVES, lso_search.colored_noise,
robot_z_search.dtw_align) still resolve, and colored_noise / dtw_align are the package's (crossembodiment.optim /
crossembodiment.costs) with the old source's results on fixed inputs.

Flags added in the main tree after the refactor and ported (ADDED): the expected parser is BASE's source with the main tree's
add_argument lines inserted verbatim where the main tree has them. So far one: single_z_search.py --exact-clearance-vel (main tree
11/12, after --obs-scale-ref; model/exact_obs.py clearance_vel). test_single_z_search_flags_main_tree also compares the new parser
with the main tree's own scripts/single_z_search.py (read only; skipped when that file is not there).

No GPU, no simulation: ~10 s.  Run standalone:  uv run python tests/test_cli.py
"""
import argparse
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

import numpy as np  # noqa: E402

BASE = "fa6c5f9"                                              # the commit before step 6
SCRIPTS = ("single_z_search", "robot_z_search", "lso_search", "robot_lso_search")
MAIN_TREE = Path("/home/allen19/crossembodiment")             # the main tree (feat/multibody): read only
# script -> [(the BASE line after which the main tree added flags, the added lines verbatim)]
ADDED = {"single_z_search": [(
    '    p.add_argument("--obs-scale-ref", default="assets/robots/adult/robot.xml")\n',
    '    p.add_argument("--exact-clearance-vel", action="store_true",\n'
    '                   help="with --obs-scale exact: also add d/dt of ExactObs\'s ground-clearance height correction to the root\'s "\n'
    '                        "vertical velocity (backward difference over the 1/30 s control step, 0 at the start state; "\n'
    '                        "model/exact_obs.py clearance_vel). Off by default (the derivative-free interface)")\n')]}


def old_source(rel):
    return subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def expected_source(name):
    """BASE's scripts/<name>.py with the flags the main tree added since (ADDED) inserted"""
    src = old_source(f"scripts/{name}.py")
    for after, lines in ADDED.get(name, ()):
        assert src.count(after) == 1, f"{name}: the anchor line of an added flag is not in {BASE}'s source once"
        src = src.replace(after, after + lines)
    return src


class _Parser(Exception):
    def __init__(self, parser):
        self.parser = parser


def old_parser(name, src=None, label=None):
    """the parser main() of scripts/<name>.py at BASE builds (+ the flags added since, ADDED); src / label: another old source"""
    mod = types.ModuleType(f"old_{name}")
    mod.__file__ = str(REPO / "scripts" / f"{name}.py")
    src = expected_source(name) if src is None else src
    exec(compile(src, label or f"{BASE}:scripts/{name}.py", "exec"), mod.__dict__)
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        raise _Parser(self)
    argparse.ArgumentParser.parse_args = grab
    try:
        mod.main()
    except _Parser as e:
        return e.parser
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise AssertionError(f"{name}: main() did not parse")


def same_type(a, b):
    """the same type callable: the same object (int, float, ...), or Python functions with the same code (a lambda, or a
    helper like single_z_search.device_arg that the old module defined itself)"""
    if a is b:
        return True
    ca, cb = getattr(a, "__code__", None), getattr(b, "__code__", None)
    return ca is not None and cb is not None and a.__name__ == b.__name__ and ca.co_code == cb.co_code \
        and ca.co_consts == cb.co_consts and ca.co_names == cb.co_names


FIELDS = ("option_strings", "dest", "default", "choices", "nargs", "required", "help", "const", "metavar")


def compare(name, src=None, label=None):
    import importlib
    new = importlib.import_module(name).build_parser()
    old = old_parser(name, src, label)
    for k in ("description", "epilog", "prefix_chars", "add_help", "allow_abbrev"):
        assert getattr(old, k) == getattr(new, k), f"{name}: parser.{k} differs"
    A, B = old._actions, new._actions
    assert [x.option_strings for x in A] == [x.option_strings for x in B], f"{name}: the flags differ (or their order)"
    for x, y in zip(A, B):
        for f in FIELDS:
            assert getattr(x, f) == getattr(y, f), f"{name} {x.option_strings}: {f} {getattr(x, f)!r} != {getattr(y, f)!r}"
        assert type(x) is type(y), f"{name} {x.option_strings}: action {type(x).__name__} != {type(y).__name__}"
        assert same_type(x.type, y.type), f"{name} {x.option_strings}: type {x.type} != {y.type}"
    assert old.format_help() == new.format_help(), f"{name}: --help text differs"
    return len(B)


def test_single_z_search_flags():
    assert compare("single_z_search") > 40


def test_single_z_search_flags_main_tree():
    """the new parser against the main tree's current scripts/single_z_search.py (its main() builds the parser inline)"""
    f = MAIN_TREE / "scripts/single_z_search.py"
    if not f.exists():
        raise unittest.SkipTest(f"{f} not found")
    assert compare("single_z_search", f.read_text(), str(f)) > 40


def test_robot_z_search_flags():
    assert compare("robot_z_search") > 30


def test_lso_search_flags():
    assert compare("lso_search") > 30


def test_robot_lso_search_flags():
    assert compare("robot_lso_search") > 20


def test_reexports():
    import single_z_search as sz
    import lso_search as ls
    import robot_z_search as rz
    import robot_lso_search as rl  # noqa: F401
    from crossembodiment import costs, geometry, optim, problems
    for n in ("rollout", "device_arg", "project_z", "rank_normalize", "ang_pair", "smpl_heading", "OBJECTIVES"):
        assert hasattr(sz, n), f"single_z_search.{n} missing"
    assert sz.OBJECTIVES is problems.OBJECTIVES and sz.project_z is geometry.project_z
    assert sz.device_arg("1") == "cuda:1" and sz.device_arg("cpu") == "cpu"
    assert ls.colored_noise is optim.colored_noise and rz.dtw_align is costs.dtw_align


def test_colored_noise_and_dtw_align_old_source():
    """the moved pure functions against their BASE source (lso_search.colored_noise, robot_z_search.dtw_align)"""
    import ast
    from crossembodiment import costs, optim
    for rel, fn, new in (("scripts/lso_search.py", "colored_noise", optim.colored_noise),
                         ("scripts/robot_z_search.py", "dtw_align", costs.dtw_align)):
        tree = ast.parse(old_source(rel))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        ns = {"np": np}
        exec(compile(ast.Module([node], []), f"{BASE}:{rel}", "exec"), ns)
        if fn == "colored_noise":
            for beta, shape in ((1.0, (5, 7, 30)), (0.0, (3, 4, 31)), (2.0, (2, 256, 299))):
                a, b = ns[fn](beta, shape, np.random.default_rng(3)), new(beta, shape, np.random.default_rng(3))
                assert a.dtype == b.dtype and np.array_equal(a, b), f"colored_noise beta {beta} {shape}"
        else:
            rng = np.random.default_rng(5)
            Bs, Bg = rng.standard_normal((4, 40, 16)), rng.standard_normal((37, 16))
            for lam in (0.0, 1.0, 3.0):
                for x, y in zip(ns[fn](Bs, Bg, lam), new(Bs, Bg, lam)):
                    assert x.dtype == y.dtype and np.array_equal(x, y), f"dtw_align lam {lam}"


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except unittest.SkipTest as e:
            print(f"  SKIP  {name}: {e}", flush=True)
        except (AssertionError, KeyError, ValueError, OSError, ImportError, AttributeError, subprocess.CalledProcessError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
