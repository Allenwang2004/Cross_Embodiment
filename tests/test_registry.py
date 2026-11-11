"""Tests of the body and method registries (crossenbodiment/, OOP refactor step 2, docs/refactor/PLAN.md).

The registries are data meant to replace literals without changing a bit, so every value is checked against what the
current code uses and what the existing runs recorded:
  * every body's XMLs, gains and references exist; robots agree with model/robots.py ROBOTS and target_path
  * the root scale s: the child's literal = ExactObs's default = every child run's pos_div; each len*_wid* body, the payload,
    the limited leg and the gain variants: scale(body, clip) == the pos_div single_z_search.py / lso_search.py wrote into the
    runs' summary.json (==, to the bit); the robots: == RobotMap(name).s
  * paths.py's spellings = the literals of the code they replace
  * every (experiment, run-directory key) of crossenbodiment/methods.py RUNS: one existing run's summary.json has the
    method's settings
Existing runs are READ ONLY: outputs/ folders that are not linked into this worktree are read from the main tree (MAIN).

Run standalone (no pytest in the environment), like tests/test_losses.py:
    uv run python tests/test_registry.py
"""
import inspect
import json
import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
MAIN = Path(os.environ.get("CROSSENBODIMENT_MAIN", "/home/allen19/crossenbodiment"))   # where the runs live (read only)
GOLDEN = REPO / "outputs/refactor_golden/baseline"                                       # tests/golden's baseline runs

from crossenbodiment import paths  # noqa: E402
from crossenbodiment.bodies import BODIES, KINDS, robot_specs  # noqa: E402
from crossenbodiment.methods import COMMON, METHODS, RUNS  # noqa: E402

WALK = "move-ego-0-2/move-ego-0-2_1"                     # every body has its reference (outputs/walk_bodies)
AMASS = "Male2Walking__B9_-__Walk_turn_left_90"          # the golden AMASS case's clip
CLIPS = paths.read_clips(paths.CHILD_52_TASK)              # = outputs/child_52/clips.txt (tests/test_moves.py)


def out(rel):
    """an outputs/ path: in this tree if it is there (symlinks into the main tree included), else in the main tree"""
    p = REPO / rel
    return p if p.exists() else MAIN / rel


def summary(d):
    return json.loads((Path(d) / "summary.json").read_text())


def stem(c):
    return c.split("/")[1]


# ------------------------------------------------------------------ paths

def test_paths_match_the_literals():
    from model import exact_obs, robots
    assert paths.REPO == REPO
    assert paths.ADULT_XML_REL == exact_obs.ADULT_XML == "assets/robots/adult/robot.xml"
    assert paths.ADULT_XML == robots.ADULT_XML
    assert paths.CHILD_XML_REL == "assets/robots_torque/child/robot_torque_full.xml"
    src = (REPO / "crossenbodiment/eval_metrics.py").read_text()          # scripts/eval_metrics.py until refactor step 8 (it re-exports it)
    assert f'"child": "{paths.CHILD_XML_REL}"' in src
    m = re.search(r'add_argument\("--metamotivo", default="([^"]+)"\)', (REPO / "scripts/single_z_search.py").read_text())
    assert m and m.group(1) == paths.MOTIVO_CKPT == "facebook/metamotivo-M-1"
    for p in (paths.ADULT_XML, paths.CHILD_XML, paths.ORIGIN_MOTION, paths.ORIGIN_Z, paths.AMASS_ORIGIN_MOTION,
              paths.AMASS_RETARGETING_MOTION):
        assert p.exists(), p
    assert paths.origin_motion(WALK) == f"data/origin_motion/{WALK}.npz" and (REPO / paths.origin_motion(WALK)).exists()
    assert paths.origin_z(WALK) == f"data/origin_z/{WALK}.npy" and (REPO / paths.origin_z(WALK)).exists()
    assert paths.retargeting_motion("child", WALK) == f"data/child/retargeting_motion/{WALK}.npz"
    assert paths.torque_xml("child") == paths.CHILD_XML_REL and paths.shape_xml("adult") == paths.ADULT_XML_REL
    assert (REPO / paths.amass_origin_motion(AMASS)).exists() and (REPO / paths.amass_retargeting_motion(AMASS)).exists()
    for r in ("g1", "h1", "apollo"):
        assert REPO / paths.roundtrip(r, "x") == robots.target_path(r, "x"), r


# ------------------------------------------------------------------ bodies: data and paths

def test_roster():
    names = set(BODIES.names())
    want = {"adult", "child", "payload", "limit_leg", "g1", "h1", "apollo"} | {f"len{L}_wid{W}" for L in (0.6, 0.8, 1.2, 1.4) for W in (0.8, 1.2)}
    want |= {f"{p}_{c}" for p in ("child", "limit_leg", "payload", "g1") for c in ("kp0.8_kd1.2", "kp1.2_kd0.8", "kp0.8_kd0.8", "kp1.2_kd1.2")}
    assert want <= names, sorted(want - names)
    for b in BODIES:
        assert b.kind in KINDS, b.name
        if b.kind == "gain_variant":
            p = BODIES[b.parent]
            assert (b.label, b.ref_task, b.ref_amass, b.s, b.s_default, b.shape_xml) == (p.label, p.ref_task, p.ref_amass, p.s, p.s_default, p.shape_xml)
    for alias, name in (("tall", "len1.4_wid0.8"), ("heavy_arm", "payload"), ("adult_heavyRWristHand_x80", "payload"),
                        ("adult_lockRleg_bent_lim", "limit_leg"), ("heavy_arm_kp0.8_kd1.2", "payload_kp0.8_kd1.2"),
                        ("adult_lockRleg_bent_lim_kp1.2_kd0.8", "limit_leg_kp1.2_kd0.8"), ("child_kp1_kd1", "child")):
        assert BODIES[alias].name == name, alias
    assert BODIES["payload"].physics_xml == BODIES["heavy_arm"].physics_xml == "assets/robots/adult_heavyRWristHand_x80/robot.xml"
    assert BODIES["payload"].reference(WALK) == f"data/origin_motion/{WALK}.npz"
    assert BODIES["limit_leg"].reference(WALK) == f"data/adult_lockRleg_bent_lim/retargeting_motion/{WALK}.npz"
    assert BODIES["tall"].reference(AMASS) == f"data/len1.4_wid0.8/amass_retargeting_motion/ACCAD/{AMASS}.npz"
    try:
        BODIES["no_such_body"]
        raise AssertionError("unknown body did not raise")
    except KeyError:
        pass


def test_every_body_path_exists():
    for b in BODIES:
        for f in (b.shape_xml, b.physics_xml, b.gains):
            assert f is None or (REPO / f).exists(), (b.name, f)
        if b.smpl:
            assert (REPO / b.reference(WALK)).exists(), (b.name, b.reference(WALK))
        else:                                                           # robots: the roundtrip targets (outputs/<robot>)
            stems = [stem(c) for c in CLIPS if out(b.reference(c)).exists()]
            assert stems, (b.name, b.ref_task)
            if b.label == "g1":
                assert len(stems) == len(CLIPS), "the G1 has all 52 targets (11/09)"
        if b.ref_amass is not None:
            assert (REPO / b.reference(AMASS)).exists(), (b.name, b.reference(AMASS))


def test_robots_agree_with_model_robots():
    from model.robots import ROBOTS, target_path
    specs = robot_specs()
    assert set(specs) == set(ROBOTS)
    for r, cfg in ROBOTS.items():
        s, b = specs[r], BODIES[r]
        assert (s.xml, s.scene, s.gains_json, s.root, s.display) == (cfg["xml"], cfg["scene"], cfg["gains_json"], cfg["root"], cfg["name"])
        assert s.damping == cfg.get("damping", "robot") and list(s.pairs) == [tuple(p) for p in cfg["pairs"]]
        assert (b.kind, b.smpl, b.shape_xml, b.physics_xml, b.gains) == ("robot", False, cfg["xml"], cfg["scene"], cfg["gains_json"])
        assert REPO / b.reference(WALK) == target_path(r, stem(WALK))
        assert BODIES.robot(r) is s
    assert BODIES.robot("g1_kp0.8_kd1.2") is specs["g1"]


# ------------------------------------------------------------------ the root scale s

def test_child_scale_is_the_literal():
    from model.exact_obs import ExactObs, ExactObsWrapper
    s = BODIES.scale("child")
    assert s == 0.611036 and type(s) is float
    assert s == inspect.signature(ExactObs.__init__).parameters["scale"].default
    assert s == inspect.signature(ExactObsWrapper.__init__).parameters["scale"].default
    assert BODIES.scale("child", WALK) == s                            # NOT recomputed from the clip
    n = 0
    for d in (out("outputs/walk_bodies/child_es_s0"), *[out(f"outputs/child_52_rerun/{stem(c)}_es_s0") for c in CLIPS],
              *[out(f"outputs/child_52_lso/{stem(c)}_lso_s0") for c in CLIPS[:5]], GOLDEN / "child_es_anchor", GOLDEN / "child_lso"):
        assert summary(d)["pos_div"] == s, d; n += 1
    for v in BODIES.names("gain_variant"):
        if BODIES[v].parent == "child":
            assert BODIES.scale(v) == s
    print(f"    child s {s!r} = {n} runs' pos_div")


def test_scaled_bodies_scale_matches_runs():
    for b in BODIES.names("smpl_scaled"):
        if BODIES[b].s_rule != "clip_ratio":
            continue
        S = summary(out(f"outputs/walk_bodies/{b}_es_s0"))
        assert S["body"] == b and S["clip"] == WALK and S["xml"] == BODIES[b].physics_xml
        assert BODIES.scale(b, WALK) == S["pos_div"], (b, BODIES.scale(b, WALK), S["pos_div"])
        try:
            BODIES.scale(b)
            raise AssertionError(f"{b}: scale without a clip should raise")
        except ValueError:
            pass
    tall = BODIES.scale("tall", WALK)
    assert tall == summary(GOLDEN / "tall_es")["pos_div"] == summary(GOLDEN / "tall_cem_phc")["pos_div"] == 1.364408905613045
    S = summary(GOLDEN / "tall_amass_lso")
    assert S["ref"] == BODIES["tall"].reference(AMASS)
    assert BODIES.scale("tall", AMASS) == S["pos_div"] == 1.364408905613045      # amass_baselines.py --scale, lso_search pos_div


def test_adult_size_bodies_scale_matches_runs():
    n = 0
    for b, folder in (("payload", "outputs/body_52/payload"), ("limit_leg", "outputs/body_52/limit_leg")):
        assert BODIES.scale(b) == 1.0                                   # lso_search.py --scale 1 (body_52_test.py lcmd)
        for c in CLIPS:
            S = summary(out(f"{folder}/{stem(c)}_es_s0"))
            assert S["ref"] == BODIES[b].reference(c) and S["body"] == BODIES[b].label and S["xml"] == BODIES[b].physics_xml, (b, c)
            assert BODIES.scale(b, c) == S["pos_div"] == 1.0, (b, c, BODIES.scale(b, c), S["pos_div"])
            assert summary(out(f"{folder}/{stem(c)}_lso_s0"))["pos_div"] == BODIES.scale(b), (b, c); n += 2
    assert BODIES.scale("payload", WALK) == summary(GOLDEN / "payload_es")["pos_div"]
    assert BODIES.scale("payload") == summary(GOLDEN / "payload_lso")["pos_div"]
    print(f"    payload / limited leg: {n} runs' pos_div")


def test_gain_variants_match_kpkd_runs():
    """outputs/kpkd/<folder>: the folder name resolves to the body; its runs' --body, --xml / --gains and pos_div agree."""
    n = 0
    for d in sorted(p for p in out("outputs/kpkd").iterdir() if re.fullmatch(r".+_kp[0-9.]+_kd[0-9.]+", p.name)):   # not search/
        assert d.name in BODIES, d.name
        b = BODIES[d.name]
        runs = sorted(d.glob("*_s0"))
        assert runs, d
        for r in runs:
            S = summary(r); c = S["clip"]
            if b.smpl:
                assert (S["body"], S["xml"]) == (b.label, b.physics_xml), (d.name, S["body"], S["xml"])
                assert BODIES.scale(b, c) == S["pos_div"], (d.name, c)
            else:
                assert (S["body"], S["gains"]) == (b.label, b.gains), (d.name, S["gains"])
            n += 1
    print(f"    {n} kpkd runs")


def test_robot_scale_is_robotmap():
    from model.robot_map import RobotMap
    for r in ("g1", "h1", "apollo"):
        s = RobotMap(r).s
        assert BODIES.scale(r) == s and float(s).hex() == BODIES.scale(r).hex(), (r, BODIES.scale(r), s)
    assert BODIES.scale("g1") == 0.8212611427612194                      # the literal of kpkd_test / g1_small_test / loes_noanchor
    for v in BODIES.names("gain_variant"):
        if BODIES[v].parent == "g1":
            assert BODIES.scale(v) == BODIES.scale("g1")


# ------------------------------------------------------------------ methods

def _expand(experiment):
    """RUNS keys with <body> / <cond> -> existing folders (this tree's golden baseline stands in for runs not made yet)"""
    if "<body>" in experiment and not experiment.startswith("outputs/kpkd"):
        base = out(experiment.split("/<body>")[0])
        return [p for p in sorted(base.iterdir()) if p.is_dir() and p.name != "rollouts"] if base.exists() else []
    if experiment.startswith("outputs/kpkd"):
        return [p for p in sorted(out("outputs/kpkd").iterdir()) if re.fullmatch(r".+_kp[0-9.]+_kd[0-9.]+", p.name)]
    return [out(experiment)] if out(experiment).exists() else []


def _runs(folder, key, pattern):
    """the run folders of `key` in `folder`, by exact name (task stems, AMASS clips or body names, seed 0)"""
    if pattern.startswith("<body>"):                                  # outputs/walk_bodies/<body>_<key>_s0
        names = [f"{b}_{key}_s0" for b in BODIES.names()]
    elif pattern.startswith("<clip>"):                                # outputs/amass_baselines/<body>/<clip>_<key>_s0
        names = [f"{c}_{key}_s0" for c in paths.read_clips(paths.AMASS_LOCOMOTION_5S)]   # = outputs/amass_baselines/clips.txt
    elif not key:                                                     # outputs/kpkd/<body>_<cond>/<stem>_s<seed>
        names = [f"{stem(c)}_s0" for c in CLIPS]
    else:
        names = [f"{stem(c)}_{key}_s0" for c in CLIPS]
    return [folder / n for n in names if (folder / n / "summary.json").exists()]


GOLDEN_RUNS = {("outputs/tall_body_52", "es"): "tall_es", ("outputs/tall_body_52", "cem"): "tall_cem_phc",
               ("outputs/amass_baselines/<body>", "lso"): "tall_amass_lso"}


def test_methods_match_runs():
    """every (experiment, key) of RUNS: up to 3 existing runs carry the method's settings (and the kind's shared ones)"""
    checked, golden = 0, []
    for experiment, (pattern, keys) in RUNS.items():
        for key, mkey in keys.items():
            m = METHODS[mkey]
            runs = [r for f in _expand(experiment) for r in _runs(f, key, pattern)[:3]]
            if not runs and (experiment, key) in GOLDEN_RUNS:
                runs = [GOLDEN / GOLDEN_RUNS[(experiment, key)]]; golden.append(f"{experiment} {key}")
            if not runs:
                golden.append(f"{experiment} {key} (no run yet)"); continue
            for r in runs:
                S = summary(r)
                robot = "robot" in S or S.get("body") == "g1"
                want = dict(m.robot_settings if robot else m.settings)
                kind = ("robot_" if robot else "") + m.kind
                want.update(COMMON[kind]["settings"])
                # the G1 runs of outputs/kpkd were made before robot_z_search.py recorded `algo`; kpkd_test.py passes no --algo, so they ran the default, es
                bad = {k: (v, S.get(k, "es" if k == "algo" else None)) for k, v in want.items() if S.get(k, "es" if k == "algo" else None) != v}
                assert not bad, (experiment, key, r.name, bad)
                checked += 1
    print(f"    {checked} runs checked; from the golden baseline / not run yet: {golden}")


def test_method_keys():
    from crossenbodiment.methods import method_of, run_keys
    assert method_of("outputs/child_52_rerun", "cem").key == "cem_ours"
    assert method_of("outputs/body_52/<body>", "cem").key == method_of("outputs/child_52_cem_phc", "cem").key == "cem_phc"
    assert method_of("outputs/child_52_lso", "lsor").key == "lsor_bfm"
    assert method_of("outputs/g1_52", "lsor").key == method_of("outputs/child_52_lso", "lsor_w005").key == "lsor_w005"
    assert method_of("outputs/noanchor/<body>", "es").key == "loes" and method_of("outputs/child_52_rerun", "es").key == "loes_anchor"
    assert ("outputs/tall_body_52", "cem") in run_keys("cem_phc")
    src = (REPO / "scripts/body_52_test.py").read_text()               # the flags are the drivers' own
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import body_52_test as B
    assert list(B.SEQ) == ["lso", "lsor", "ta"], list(B.SEQ)
    for k, flags in B.SEQ.items():
        assert flags == METHODS[method_of("outputs/body_52/<body>", k).key].flags, (k, flags)
    assert "--objective phc --algo cem" in src and METHODS["cem_phc"].flags == "--objective phc --algo cem"


def test_step11_names():
    """step 11: one name per method / root scale in the code, the old names still read (run keys, yaml keys, palette keys)"""
    from crossenbodiment.methods import AMBIGUOUS, RUNS, canonical, experiment_keys, key_in, method_of
    for experiment, (_, keys) in RUNS.items():                          # key_in inverts method_of; experiment_keys lists them
        for k, m in keys.items():
            assert method_of(experiment, k).key == m and key_in(experiment, m) == k, (experiment, k, m)
        assert dict(experiment_keys(experiment)) == {m: k for k, m in keys.items()}, experiment
    for m in METHODS.values():                                          # run_key: the key most experiments give the method
        ks = {k for _, (_, keys) in RUNS.items() for k, mm in keys.items() if mm == m.key and k}
        assert (m.run_key is None) == (m.key in ("native", "frozen")) and (not ks or m.run_key in ks), (m.key, m.run_key, ks)
    assert canonical("z0") == "frozen" and canonical("cem_phc") == "cem_phc"
    for k in AMBIGUOUS:
        try:
            canonical(k)
            raise AssertionError(f"canonical({k!r}) did not raise")
        except KeyError:
            pass
    assert experiment_keys("outputs/tall_body_52")["cem_phc"] == "cem" and key_in("outputs/child_52_lso", "lsor_w005") == "lsor_w005"
    from crossenbodiment.experiment import Experiment
    assert Experiment("outputs/body_52").key("lsor_w005") == "lsor" and Experiment("outputs/child_52_rerun").key("cem_ours") == "cem"
    for b in BODIES:                                                    # BodySpec's root-scale fields and their yaml-key aliases
        assert (b.s, b.s_rule, b.s_default) == (b.scale, b.scale_rule, b.scale_default), b.name
    from crossenbodiment.palette import OLD_KEYS, OLD_VARIANT_KEYS, Palette
    for old, new in OLD_KEYS.items():
        assert Palette.method(old) is Palette.method(new) and Palette.method(new).key == new
    for (old, v), new in OLD_VARIANT_KEYS.items():
        assert Palette.method(old, v) is Palette.method(*new)
    assert Palette.hex("z0", "k_scale") == Palette.hex("frozen", "k_scale") == "#eb6834"
    from crossenbodiment import costs, obs, paths as P
    assert costs.Position(None, 2.5).pos_div == 2.5 and obs.CHILD_EXACT_S == obs.CHILD_SCALE == BODIES.scale("child")
    assert P.origin_z is P.task_z0 and P.z0_rollout("x") == "x/origin_z.npz" and P.Z0_KEY == "origin_z"


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
        except (AssertionError, KeyError, ValueError, OSError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
