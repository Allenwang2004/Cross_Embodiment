"""Tests of OOP refactor step 8 (docs/refactor/PLAN.md): the experiment layer's command builders, re-roll index and results helpers.

  builders   every command list the ported drivers print (commands, cemcommands, algocommands, evalcmds, eval_commands, with the
             --set / --methods / --algos / --bodies variants the queues used) is byte-identical to the driver's own output at BASE
             (d299653, the commit before step 8). Each side runs in a fresh process, as the queues ran them: the new driver as
             `python scripts/<name>.py ...`; the old one from `git show BASE:scripts/<name>.py`, executed as __main__ with an import
             hook that serves the BASE source of every driver it imports (loes_noanchor imports body_52_test / g1_small_test, whose
             import-time argv reads are part of what is compared). The two drivers whose builders read experiment folders
             (loes_noanchor evalcmds: which z.npy exist; loes_weight_test: clips.txt) see a small fake repository under
             outputs/refactor_tests/test_experiments (the new code through CE_OUTPUTS_ROOT, the old one through its REPO).
  reroll     Reroller.prepare on synthetic z_traces against the three old prepare loops (BASE source): z.npy byte-identical, the
             schema-2 index read back gives the old lists; RerollIndex reads every old layout; check_vs_curve
  results    ResultsTable's reductions against the drivers' expressions (bitwise), markdown rows, Checks, ReportWriter
No GPU. Run standalone:  uv run python tests/test_experiments.py       (~2 min: about 60 driver processes)
"""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

BASE = "d299653"
MAIN = Path(os.environ.get("CROSSENBODIMENT_MAIN", "/home/allen19/crossenbodiment"))   # read only
WORK = REPO / "outputs/refactor_tests/test_experiments"
PY = sys.executable
DRIVERS = ("child_52_rerun", "child_52_cem_phc", "child_52_lso", "body_52_test", "g1_small_test", "kpkd_test", "loes_noanchor",
           "loes_weight_test", "lsor_mpjae_test", "tall_body_52", "amass_baselines", "cost_correlation", "drift_success_child",
           "success_definitions", "fall_definitions")

# run one BASE driver as __main__ in this process: argv[1] the fake repository root its __file__ is placed in, argv[2] the driver,
# the rest its CLI arguments
OLD_RUNNER = r'''
import importlib.abc, importlib.util, subprocess, sys
from pathlib import Path
REPO, ROOT, BASE, NAME = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
OLD = set(sys.argv[5].split(","))
def src(n):
    return subprocess.run(["git", "show", f"{BASE}:scripts/{n}.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
class Old(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        return importlib.util.spec_from_loader(fullname, self) if fullname in OLD else None
    def create_module(self, spec):
        return None
    def exec_module(self, m):
        m.__file__ = str(ROOT / "scripts" / f"{m.__name__}.py")
        exec(compile(src(m.__name__), m.__file__, "exec"), m.__dict__)
sys.meta_path.insert(0, Old())
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
f = str(ROOT / "scripts" / f"{NAME}.py"); sys.argv = [f] + sys.argv[6:]
exec(compile(src(NAME), f, "exec"), {"__name__": "__main__", "__file__": f})
'''
ENV = dict(os.environ, MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="2", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
for k in ("CE_OUTPUTS_ROOT", "CE_RESULTS_ROOT"):
    ENV.pop(k, None)


def fake_repo():
    """outputs/refactor_tests/test_experiments/repo: data -> the worktree's data (read only), outputs/loes_weight_test/clips.txt
    (a copy of the main tree's), outputs/noanchor/<body>/rollouts/<stem>/z.npy for some clips (empty: only existence is read)"""
    root = WORK / "repo"
    if root.exists():
        shutil.rmtree(root)
    (root / "scripts").mkdir(parents=True)
    (root / "data").symlink_to(REPO / "data")
    lw = root / "outputs/loes_weight_test"; lw.mkdir(parents=True)
    shutil.copy(MAIN / "outputs/loes_weight_test/clips.txt", lw / "clips.txt")
    from crossenbodiment.paths import CHILD_52_TASK, read_clips
    clips = read_clips(CHILD_52_TASK)
    for b, keep in (("child", clips[::2]), ("payload", clips[:10]), ("limit_leg", clips[5:9]), ("g1", clips[-3:])):
        for c in keep:
            d = root / "outputs/noanchor" / b / "rollouts" / c.split("/")[1]; d.mkdir(parents=True); (d / "z.npy").write_bytes(b"")
    return root


class TestBuilders(unittest.TestCase):
    CASES = [
        ("child_52_cem_phc", ["commands"]), ("child_52_cem_phc", ["eval_commands"]),
        ("child_52_lso", ["commands"]),
        ("body_52_test", ["commands"]), ("body_52_test", ["cemcommands"]), ("body_52_test", ["algocommands"]),
        ("body_52_test", ["algocommands", "--algos", "random,cmaes"]), ("body_52_test", ["algocommands", "--algos", "cem"]),
        ("body_52_test", ["evalcmds"]), ("body_52_test", ["evalcmds", "--algos", "random,cmaes"]), ("body_52_test", ["--algos", "cem", "evalcmds"]),
        ("g1_small_test", ["commands"]), ("g1_small_test", ["evalcmds"]), ("g1_small_test", ["--set", "52", "commands"]),
        ("g1_small_test", ["--set", "52", "evalcmds"]), ("g1_small_test", ["--set", "52", "--methods", "es,lso,lsor,ta", "commands"]),
        ("g1_small_test", ["--set", "52", "--methods", "es,lso,lsor,ta", "evalcmds"]), ("g1_small_test", ["--methods", "es,lso", "commands"]),
        ("kpkd_test", ["commands"]),
        ("loes_noanchor", ["commands"]), ("loes_noanchor", ["evalcmds"]), ("loes_noanchor", ["evalcmds", "--bodies", "child"]),
        ("loes_noanchor", ["evalcmds", "--bodies", "payload,limit_leg,g1"]),
        ("loes_weight_test", ["commands"]), ("loes_weight_test", ["evalcmds"]),
        ("tall_body_52", ["commands"]), ("amass_baselines", ["commands"]), ("cost_correlation", ["commands"]),
    ]
    FAKE = {"loes_noanchor", "loes_weight_test"}               # their builders read experiment folders

    @classmethod
    def setUpClass(cls):
        cls.root = fake_repo()

    def run_new(self, name, args):
        env = dict(ENV, CE_OUTPUTS_ROOT=str(self.root / "outputs")) if name in self.FAKE else ENV
        r = subprocess.run([PY, str(REPO / "scripts" / f"{name}.py"), *args], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return r.stdout

    def run_old(self, name, args):
        root = self.root if name in self.FAKE else REPO
        r = subprocess.run([PY, "-c", OLD_RUNNER, str(REPO), str(root), BASE, name, ",".join(DRIVERS), *args], cwd=REPO, env=ENV,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return r.stdout

    def test_builders(self):
        for name, args in self.CASES:
            with self.subTest(name=name, args=args):
                old, new = self.run_old(name, args), self.run_new(name, args)
                self.assertTrue(old.strip(), f"{name} {args}: the old builder printed nothing")
                self.assertEqual(new, old, f"{name} {args}")
                print(f"  identical: {name} {' '.join(args)} ({len(old.splitlines())} lines)", flush=True)


def old_source(rel):
    return subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def old_function(rel, name, env):
    """the BASE source of one top-level function of a script, executed in `env` (its globals)"""
    import ast
    tree = ast.parse(old_source(rel))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    mod = ast.Module(body=[node], type_ignores=[])
    exec(compile(mod, rel, "exec"), env)
    return env[name]


def traces(d, runs, gens, seed=0):
    """synthetic z_trace.npz files: z_best constant over random stretches (repeats share an index), z_mean random"""
    rng = np.random.default_rng(20261111 + seed)
    for r in runs:
        b = rng.standard_normal((gens + 1, 256)).astype(np.float32)
        for g in range(1, gens + 1):
            if rng.random() < 0.6:
                b[g] = b[g - 1]
        m = rng.standard_normal((gens + 1, 256)).astype(np.float32)
        (d / r).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / r / "z_trace.npz", gen=np.arange(-1, gens, dtype=np.int32), z_mean=m, z_best=b)


class TestReroll(unittest.TestCase):
    def setUp(self):
        self.d = WORK / "reroll"
        if self.d.exists():
            shutil.rmtree(self.d)
        self.d.mkdir(parents=True)

    def compare(self, rel, fn_name, env, run_old, prepare_new, budgets, gens, lists):
        """run the BASE prepare loop (env: its module globals) and the new one; z.npy byte-identical, index lists equal"""
        fn = old_function(rel, fn_name, env)
        run_old(fn)
        I = prepare_new()
        old_dir, new_dir = env["_old_dir"], env["_new_dir"]
        self.assertEqual((new_dir / "z.npy").read_bytes(), (old_dir / "z.npy").read_bytes())
        from crossenbodiment.reroll import Reroller
        O = Reroller.read(old_dir, budgets, gens); N = Reroller.read(new_dir)
        self.assertEqual(N.schema, 2); self.assertEqual(O.schema, 1)
        self.assertEqual(O.runs, N.runs)
        for r in O.runs:
            lists(self, O, N, r)
        return O, N

    def test_child_style(self):
        """child_52_rerun.prepare: every generation's distinct best + centres (schema 'generation')"""
        from crossenbodiment.reroll import Reroller
        runs = [f"x_{a}_s{s}" for a in ("es", "cem") for s in range(2)]; traces(self.d / "runs", runs, 64)
        old_dir, new_dir = self.d / "old/rollouts/x", self.d / "new/rollouts/x"
        G = (4, 8, 16, 32, 64)
        env = dict(np=np, json=json, clips=lambda: ["t/x"], ROLL=self.d / "old/rollouts", ALGOS=("es", "cem"), SEEDS=range(2), G=G,
                   OUT=self.d / "runs", print=lambda *a: None, _old_dir=old_dir, _new_dir=new_dir)

        def lists(t, O, N, r):
            t.assertEqual(O.map[r]["best"], N.best_generations(r)); t.assertEqual(O.map[r]["centre"], N.centre(r))
            t.assertEqual(O.best_budgets(r), N.best_budgets(r)); t.assertEqual(O.best_last(r), N.best_last(r))
            t.assertEqual(O.rerolled(r), N.rerolled(r))
        O, N = self.compare("scripts/child_52_rerun.py", "prepare", env, lambda f: f(),
                            lambda: Reroller(64, G).prepare(new_dir, [(r, self.d / "runs" / r) for r in runs], clip="t/x"), G, 64, lists)
        self.assertEqual(N.data["clip"], "t/x"); self.assertEqual(N.budgets, list(G)); self.assertEqual(O.layout(runs[0]), "generation")
        self.assertLess(len(np.load(new_dir / "z.npy")), 4 * (64 + 5))   # repeats were shared

    def test_budget_style(self):
        """body_52_test.prepare / g1_small_test.prepare: the best and the centre at each budget (schema 'budget')"""
        from crossenbodiment.reroll import Reroller
        G = (4, 8, 16, 32, 64, 128); runs = ["x_es_s0"]; traces(self.d / "runs/payload", runs, 128, 1)
        old_dir, new_dir = self.d / "old/payload/rollouts/x", self.d / "new/payload/rollouts/x"
        env = dict(np=np, json=json, BODIES={"payload": None}, CLIPS=("t/x",), ALGOS=["es"], SEEDS=(0,), G=G, GENS=128, OUT=self.d / "runs",
                   rdir=lambda b, c, k: old_dir, stem=lambda c: c.split("/")[1], print=lambda *a: None, _old_dir=old_dir, _new_dir=new_dir)

        def lists(t, O, N, r):
            t.assertEqual(O.map[r]["best"], N.best_budgets(r)); t.assertEqual(O.map[r]["centre"], N.centre(r))
            t.assertEqual(O.best_generations(r), N.best_generations(r)); t.assertEqual(O.best_last(r), N.best_last(r))
            t.assertEqual(O.rerolled(r), N.rerolled(r))
        O, N = self.compare("scripts/body_52_test.py", "prepare", env, lambda f: f(),
                            lambda: Reroller(128, G, best="budgets").prepare(new_dir, [("x_es_s0", self.d / "runs/payload/x_es_s0")],
                                                                              body="payload", clip="t/x"), G, 128, lists)
        self.assertEqual(O.layout("x_es_s0"), "budget"); self.assertEqual(N.data["body"], "payload")
        self.assertEqual(sum(i is not None for i in N.best_generations("x_es_s0")), len(G))

    def test_last_style(self):
        """loes_weight_test.prepare: map[run] = the best latent after 64 generations (schema 'last')"""
        from crossenbodiment.reroll import Reroller
        runs = [f"x_{k}_s{s}" for k in ("w0", "w0.05") for s in range(3)]; traces(self.d / "runs", runs, 64, 2)
        old_dir, new_dir = self.d / "old/rollouts/x", self.d / "new/rollouts/x"
        env = dict(np=np, json=json, clips=lambda: ["t/x"], ROLL=self.d / "old/rollouts", NEW={"w0": 0, "w0.05": 0}, SEEDS=range(3),
                   OUT=self.d / "runs", print=lambda *a: None, _old_dir=old_dir, _new_dir=new_dir)

        def lists(t, O, N, r):
            t.assertEqual(O.map[r], N.best_last(r)); t.assertEqual(O.best_generations(r), N.best_generations(r))
        self.compare("scripts/loes_weight_test.py", "prepare", env, lambda f: f(),
                     lambda: Reroller(64, best="last", centre=False).prepare(new_dir, [(r, self.d / "runs" / r) for r in runs], clip="t/x"),
                     (64,), 64, lists)

    def test_readers_and_curve_check(self):
        from crossenbodiment.reroll import RerollIndex, check_vs_curve
        tiny = RerollIndex({"map": {"a_es_s0": {"best": [0]}}})          # tests/golden/metrics_golden.py's minimal index
        self.assertEqual(tiny.best_last("a_es_s0"), 0); self.assertEqual(tiny.layout("a_es_s0"), "unknown")
        with self.assertRaises(ValueError):
            tiny.best_generations("a_es_s0")
        cost = [0.5, 0.4, 0.3, 0.2]; bs = np.array([0.5, 0.45, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2])
        I = RerollIndex({"map": {"r": {"best": [0, 2], "centre": [1, 3]}}}, budgets=(5, 8))
        self.assertEqual(I.rerolled("r"), [(5, 0), (8, 2)])
        self.assertEqual(I.check_vs_curve("r", cost, bs), float(np.abs(np.array(cost)[[0, 2]] - bs[[4, 7]]).max()))
        self.assertEqual(check_vs_curve(cost, bs, [0, 1], [1, 2]), float(np.abs(np.array([0.5, 0.4]) - np.array([0.5, 0.45])).max()))


class TestExperiment(unittest.TestCase):
    def test_grid_names_lines(self):
        from crossenbodiment.experiment import ClipSet, Experiment
        E = Experiment("outputs/body_52", clips=ClipSet(["a/a_1", "b/b_2"]), methods=("es", "lso"), seeds=(0, 1), bodies=("payload",),
                       name="{body}/{stem}_{method}_s{seed}", order=("seed", "clip", "method"))
        self.assertEqual([r.dir for r in E.runs()][:3], ["outputs/body_52/payload/a_1_es_s0", "outputs/body_52/payload/a_1_lso_s0",
                                                         "outputs/body_52/payload/b_2_es_s0"])
        self.assertEqual(E.commands(lambda r: f"run {r.seed}", order=("clip",), clip=("b/b_2",)), ["outputs/body_52/payload/b_2_es_s0|run 0"])
        self.assertEqual(E.eval_line("outputs/x/rollouts/a", "base"), "outputs/x/rollouts/a|base --seed 0 --eval-z outputs/x/rollouts/a/z.npy "
                                                                          "--eval-save-qpos --out outputs/x/rollouts/a")
        self.assertEqual(E.rollouts_dir("a_1_cem", body="payload"), "outputs/body_52/payload/rollouts/a_1_cem")
        self.assertEqual(E.method_spec("cem").key, "cem_phc"); self.assertEqual(E.method_spec("lsor").key, "lsor_w005")
        self.assertEqual(Experiment("outputs/child_52_rerun").method_spec("cem").key, "cem_ours")
        self.assertEqual(Experiment("outputs/child_52_lso").method_spec("lsor").key, "lsor_bfm")
        self.assertEqual(Experiment("outputs/kpkd").method_spec("").key, "loes_anchor")
        c = ClipSet.named("child_52_task")
        self.assertEqual(list(c.longest_first()), sorted(c, key=lambda x: -len(np.load(REPO / f"data/origin_motion/{x}.npz")["qpos"])))

    def test_roots(self):
        """CE_OUTPUTS_ROOT / CE_RESULTS_ROOT; a write outside the repository is refused"""
        from crossenbodiment import paths
        from crossenbodiment.results import write_json
        old = {k: os.environ.pop(k, None) for k in (paths.OUTPUTS_ROOT_ENV, paths.RESULTS_ROOT_ENV)}
        try:
            self.assertEqual(paths.output("x"), REPO / "outputs/x"); self.assertEqual(paths.result("x"), REPO / "outputs/x")
            os.environ[paths.OUTPUTS_ROOT_ENV] = str(MAIN / "outputs")
            self.assertEqual(paths.output("x"), MAIN / "outputs/x"); self.assertEqual(paths.result("x"), MAIN / "outputs/x")
            with self.assertRaises(RuntimeError):
                write_json(paths.result("never_written.json"), {})
            self.assertFalse((MAIN / "outputs/never_written.json").exists())
            os.environ[paths.RESULTS_ROOT_ENV] = str(WORK / "roots")
            write_json(paths.result("a/b.json"), {"x": 1}); self.assertEqual((WORK / "roots/a/b.json").read_text(), '{\n "x": 1\n}')
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


def old_module(name, alias):
    """the BASE source of scripts/<name>.py as a module registered as `alias` (so Pool workers, forked, find its functions)"""
    import types
    m = types.ModuleType(alias); m.__file__ = str(REPO / "scripts" / f"{name}.py"); sys.modules[alias] = m
    exec(compile(old_source(f"scripts/{name}.py"), m.__file__, "exec"), m.__dict__)
    return m


class TestAmassAnalyse(unittest.TestCase):
    """amass_baselines.analyse (its queue is still running, so its real folders are not read): old and new on a synthetic experiment
    built from the golden baseline's tall_amass_lso run (copied as lso, lsor and ta; budgets 2, 4): results.json, README.md identical"""

    def test_old_and_new_identical(self):
        gold = REPO / "outputs/refactor_golden/baseline/tall_amass_lso"
        if not gold.exists():
            self.skipTest("no golden baseline")
        clip = "Male2Walking__B9_-__Walk_turn_left_90"; d = WORK / "amass"
        if d.exists():
            shutil.rmtree(d)
        for k in ("lso", "lsor", "ta"):
            (d / "runs/tall" / f"{clip}_{k}_s0").mkdir(parents=True)
            for f in ("summary.json", "rollouts.npz"):
                shutil.copy(gold / f, d / "runs/tall" / f"{clip}_{k}_s0" / f)
        import amass_baselines as new
        old = old_module("amass_baselines", "_old_amass_baselines")
        outs = {}
        for tag, m in (("old", old), ("new", new)):
            m.BODY["tall"].update(iters=4, G=(2, 4)); m.OUT = d / "runs"
            if tag == "new":
                m.RES = d / "new"
            else:
                shutil.copytree(d / "runs/tall", d / "old/tall"); m.OUT = d / "old"
            m.analyse()
            outs[tag] = m.RES if tag == "new" else m.OUT
        for f in ("results.json", "README.md"):
            self.assertEqual((outs["new"] / f).read_bytes(), (outs["old"] / f).read_bytes(), f)
        self.assertEqual(json.loads((outs["new"] / "results.json").read_text())["tall"]["checks"]["motions"], 1)


class TestResults(unittest.TestCase):
    def test_reductions_match_the_drivers(self):
        from crossenbodiment.results import ResultsTable
        rng = np.random.default_rng(3); K = ("ratio", "fall_s", "success", "mpjpe")
        A = rng.standard_normal((52, 5, 64, 4)); C = rng.standard_normal((52, 5, 5, 4))
        m = lambda A: {k: float(A[..., i].mean()) for i, k in enumerate(K)}                       # child_52_rerun's m
        self.assertEqual(ResultsTable.pooled(A[:, :, 7], K), m(A[:, :, 7]))
        from scipy.stats import wilcoxon
        test = lambda X, Y: {k: dict(lower=int((X[..., i].mean(1) < Y[..., i].mean(1)).sum()), p=float(wilcoxon(X[..., i].mean(1), Y[..., i].mean(1)).pvalue)) for i, k in enumerate(K) if k != "ratio"}
        self.assertEqual(ResultsTable.paired(A[:, :, -1], C[:, :, -1], K), test(A[:, :, -1], C[:, :, -1]))
        res = {f"m{i}": dict(es_best=rng.standard_normal((1, 6, 4)), frozen=rng.standard_normal((1, 4))) for i in range(52)}
        G = (4, 8, 16, 32, 64, 128)
        get = lambda st, k: np.repeat(res[st][k][:, None], len(G), axis=1) if k == "frozen" else res[st][k]
        T = ResultsTable(K, list(res), G)
        T.add("es_best", np.stack([res[st]["es_best"] for st in res])); T.add("frozen", np.stack([res[st]["frozen"][0] for st in res]))
        for k in ("es_best", "frozen"):                                                              # body_52_test's mean / per_motion
            want = {str(g * 16): dict(zip(K, np.mean([get(st, k)[:, j].mean(0) for st in res], 0).tolist())) for j, g in enumerate(G)}
            self.assertEqual(T.by_budget(k), want)
            self.assertEqual(T.per_motion(k, -1), {st: dict(zip(K, get(st, k)[:, -1].mean(0).tolist())) for st in res})
        Y = np.array([get(st, "es_best")[:, 2].mean(0) for st in res])                               # plot_child_convergence's band
        self.assertEqual(T.se("es_best", 2), dict(zip(K, (Y.std(0, ddof=1) / np.sqrt(len(Y))).tolist())))

    def test_markdown_checks_report(self):
        from crossenbodiment.results import Checks, ReportWriter, ResultsTable
        v = dict(fall_s=0.12345, success=0.5, mpjpe=3.14159)
        cols = (("Fall (s)", "fall_s", "{:.3f}"), ("Success", "success", "{:.1f}%", 100), ("MPJPE", "mpjpe", "{:.2f}"))
        L = ResultsTable.markdown([("LOES", v)], cols)
        self.assertEqual(L, ["| method | Fall (s) | Success | MPJPE |", "|---|---|---|---|",
                             f"| LOES | {v['fall_s']:.3f} | {100 * v['success']:.1f}% | {v['mpjpe']:.2f} |"])
        c = Checks(("a",), ("bad",)); c.diff("a", np.array([1.0, 2.0]), np.array([1.0, 2.5])); c.diff("a", np.zeros(2), np.zeros(2))
        c.add("bad", [("r", "x")]); self.assertEqual(json.dumps(c), json.dumps(dict(a=0.5, bad=[["r", "x"]])))
        self.assertEqual(Checks.settings(dict(x=1, y=2), dict(x=1, y=3)), ["y"])
        self.assertEqual(ReportWriter("T", "doc").render(["a", "b"]), "# T\n\ndoc\na\nb\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
