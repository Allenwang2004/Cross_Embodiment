"""experiment.py -- what an experiment runs: ClipSet, RunSpec, Experiment (OOP refactor step 8, docs/refactor/PLAN.md).

The drivers (scripts/child_52_*.py, body_52_test.py, g1_small_test.py, kpkd_test.py, loes_*.py, tall_body_52.py, amass_baselines.py,
cost_correlation.py, ...) each built their command lists with their own nested loops, folder names and separators. An Experiment holds
those facts once:

  out        the experiment folder as the command lines spell it ("outputs/child_52_cem_phc"); read under paths.outputs_root(),
             results written under paths.results_root() (crossenbodiment/paths.py)
  clips      a ClipSet (file order, or longest_first() as the drivers sorted them)
  methods    run-directory keys (crossenbodiment/methods.py RUNS says which MethodSpec each one is in which experiment; the drivers
             take them from methods.experiment_keys(experiment)[canonical method]; method_spec(key) / key(method) convert)
  seeds, bodies, budgets (generations), gens, evals, frame_offset: the experiment's grid and budget
  name       the run folder's name: a pattern ("{stem}_{method}_s{seed}", "{body}/{stem}_{method}_s{seed}") or a function of the RunSpec
  order      the loop order of commands() ("seed", "clip", "body", "method"; the first is the outermost)
  sep        between the folder and the command: "|" (the queues' "dir|cmd" lines) or " " (the xargs-era drivers' "dir cmd")

Experiment.runs() -> RunSpec per grid point in `order`; commands(command) -> the "dir|cmd" lines, `command(run)` giving each command
line exactly as the driver's builder wrote it (nothing is added to it); eval_line() -> one --eval-z re-roll line; done(run) -> its
summary.json (or another marker) exists. Nothing here reads sys.argv: the drivers parse their own flags in main.

    EXP = Experiment("outputs/child_52_cem_phc", clips=ClipSet.named(CHILD_52_TASK), methods=("cem",), seeds=range(5),
                     order=("clip", "seed", "method"), sep=" ")
    EXP.print_commands(lambda r: f"{base_cmd(r.clip)} --seed {r.seed} --out {r.dir}")
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import paths

WORKERS_ENV = "CE_WORKERS"


def workers(default):
    """the size of an analysis Pool: $CE_WORKERS, else the driver's own default (the results do not depend on it: Pool.map keeps
    the order and every worker computes the same numbers)"""
    v = os.environ.get(WORKERS_ENV)
    return int(v) if v else default


class ClipSet(tuple):
    """clip names in order ('<task>/<stem>' task clips or bare AMASS names); a tuple, so it compares and iterates like the
    drivers' lists"""

    def __new__(cls, clips=(), name=None):
        self = super().__new__(cls, tuple(clips))
        self.name = name
        return self

    @classmethod
    def named(cls, name):
        """configs/clips/<name>.txt (paths.read_clips)"""
        return cls(paths.read_clips(name), name)

    @classmethod
    def from_file(cls, path):
        """one clip per line, stripped, blank lines skipped (outputs/loes_weight_test/clips.txt and the like)"""
        with open(path) as f:
            return cls([l.strip() for l in f if l.strip()], str(path))

    @property
    def stems(self):
        return [paths.stem(c) for c in self]

    def longest_first(self, motion=None):
        """sorted by the clip's length, longest first, ties in file order (sorted is stable): the drivers'
        `sorted(CLIPS, key=lambda c: -len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]))`. motion(c) -> the
        repository-relative clip file whose length counts (default: the adult's task clip)"""
        motion = motion or paths.origin_motion
        return ClipSet(sorted(self, key=lambda c: -len(np.load(paths.REPO / motion(c))["qpos"])), self.name)


@dataclass(frozen=True)
class RunSpec:
    """one search of an experiment: its grid point and, through the experiment, its folder"""
    exp: "Experiment" = field(repr=False, compare=False)
    clip: Any = None
    method: Any = None
    seed: Any = None
    body: Any = None

    @property
    def stem(self):
        return paths.stem(self.clip) if isinstance(self.clip, str) else None

    @property
    def name(self):
        return self.exp.run_name(self)

    @property
    def dir(self):
        """the run folder as the command line spells it (relative: outputs/<experiment>/<name>)"""
        return f"{self.exp.out}/{self.name}"

    @property
    def path(self):
        """the run folder to read (under paths.outputs_root())"""
        return self.exp.read_dir / self.name

    def line(self, cmd, sep=None):
        return f"{self.dir}{self.exp.sep if sep is None else sep}{cmd}"


class Experiment:
    """an experiment's grid, folder names and command lines (module docstring)"""

    AXES = ("seed", "clip", "body", "method")

    def __init__(self, out, clips=(), methods=(None,), seeds=(0,), bodies=(None,), budgets=(), gens=None, evals=None,
                 frame_offset=1, name="{stem}_{method}_s{seed}", order=AXES, sep="|"):
        assert out.startswith(paths.OUTPUTS_REL + "/"), out
        self.out = out
        self.clips = clips if isinstance(clips, ClipSet) else ClipSet(clips)
        self.methods, self.seeds, self.bodies = tuple(methods), tuple(seeds), tuple(bodies)
        self.budgets = tuple(budgets)
        self.gens = gens if gens is not None else (self.budgets[-1] if self.budgets else None)
        self.evals, self.frame_offset = evals, frame_offset
        self.name, self.order, self.sep = name, tuple(order), sep
        assert set(self.order) <= set(self.AXES), self.order

    # ------------------------------------------------------------------ folders
    @property
    def rel(self):
        """the folder below outputs/ ("child_52_cem_phc")"""
        return self.out[len(paths.OUTPUTS_REL) + 1:]

    @property
    def read_dir(self):
        return paths.output(self.rel)

    @property
    def result_dir(self):
        return paths.result(self.rel)

    def run_name(self, run):
        if callable(self.name):
            return self.name(run)
        return self.name.format(stem=run.stem, clip=run.clip, method=run.method, seed=run.seed, body=run.body)

    def run(self, clip=None, method=None, seed=None, body=None):
        return RunSpec(self, clip, method, seed, body)

    def run_dir(self, clip=None, method=None, seed=None, body=None):
        """the relative run folder (the command's --out)"""
        return self.run(clip, method, seed, body).dir

    def path(self, clip=None, method=None, seed=None, body=None):
        """the run folder to read"""
        return self.run(clip, method, seed, body).path

    def rollouts_dir(self, name, body=None):
        """the re-roll folder of one clip (and method), relative: <out>[/<body>]/rollouts/<name>"""
        return f"{self.out}/{body + '/' if body else ''}rollouts/{name}"

    def method_spec(self, key):
        """the MethodSpec behind run-directory key `key` in this experiment (crossenbodiment/methods.py RUNS: "cem" is CEM with
        our cost in outputs/child_52_rerun, BFM-Zero's in outputs/body_52/<body>, ...)"""
        from .methods import RUNS, method_of
        for e in (self.out, f"{self.out}/<body>", f"{self.out}/<body>_<cond>"):
            if e in RUNS:
                return method_of(e, key)
        raise KeyError(f"{self.out} is not in crossenbodiment/methods.py RUNS")

    def key(self, method):
        """the run-directory key of canonical `method` in this experiment (crossenbodiment/methods.py key_in, the inverse of
        method_spec): Experiment("outputs/tall_body_52", ...).key("cem_phc") -> "cem" """
        from .methods import RUNS, key_in
        for e in (self.out, f"{self.out}/<body>", f"{self.out}/<body>_<cond>"):
            if e in RUNS:
                return key_in(e, method)
        raise KeyError(f"{self.out} is not in crossenbodiment/methods.py RUNS")

    # ------------------------------------------------------------------ the grid
    def runs(self, order=None, **axes):
        """RunSpec per grid point, `order` (default self.order) outermost first; axes given here replace the experiment's
        (clip=..., method=..., seed=..., body=...); an axis not in `order` takes its single value (first of the axis)"""
        order = tuple(order or self.order)
        ax = dict(seed=self.seeds, clip=self.clips, body=self.bodies, method=self.methods)
        ax.update({k: tuple(v) for k, v in axes.items()})
        fixed = {a: ax[a][0] for a in self.AXES if a not in order}
        for combo in itertools.product(*(ax[a] for a in order)):
            yield RunSpec(self, **fixed, **dict(zip(order, combo)))

    def commands(self, command: Callable[[RunSpec], str], order=None, **axes):
        """["<run dir><sep><command(run)>"] over runs(order, **axes)"""
        return [r.line(command(r)) for r in self.runs(order, **axes)]

    def print_commands(self, command, order=None, **axes):
        for line in self.commands(command, order, **axes):
            print(line)

    @staticmethod
    def eval_suffix(d, seed=0, save=("qpos",)):
        """the drivers' --eval-z tail: " --seed 0 --eval-z <d>/z.npy --eval-save-qpos [...] --out <d>" (seed None: no --seed)"""
        s = "" if seed is None else f" --seed {seed}"
        return s + f" --eval-z {d}/z.npy" + "".join(f" --eval-save-{k}" for k in save) + f" --out {d}"

    def eval_line(self, d, base, seed=0, save=("qpos",), sep=None):
        """one re-roll line: "<d><sep><base> --seed 0 --eval-z <d>/z.npy --eval-save-qpos --out <d>" """
        return f"{d}{self.sep if sep is None else sep}{base}{self.eval_suffix(d, seed, save)}"

    # ------------------------------------------------------------------ state
    def done(self, run, marker="summary.json"):
        """the run has written its marker (summary.json: every search writes it last)"""
        return (run.path / marker).exists()

    def all_done(self, runs, marker="summary.json"):
        return all(self.done(r, marker) for r in runs)


def parse_flag(argv, flag, default=None, cast=str):
    """the value after `flag` in argv (the drivers' `sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default`)"""
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def action(argv, actions):
    """the first argv word that names an action (the drivers' `[x for x in sys.argv[1:] if x in A][0]`)"""
    return [x for x in argv[1:] if x in actions][0]


def output_arg(rel):
    """a repository-relative folder given on a driver's command line (e.g. --loes-dir outputs/noanchor/child) as the path to READ:
    under paths.outputs_root() when it is an "outputs/..." path, else REPO / rel"""
    pre = paths.OUTPUTS_REL + "/"
    return paths.output(rel[len(pre):]) if rel.startswith(pre) else paths.REPO / rel
