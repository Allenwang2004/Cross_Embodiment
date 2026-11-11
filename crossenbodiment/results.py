"""results.py -- result tables of an experiment: ResultsTable, Checks, ReportWriter and the file writers (OOP refactor step 8).

ResultsTable  per method an array motions x seeds x budgets x metrics (as the drivers stacked it: np.stack of the per-motion
              arrays) or motions x metrics (one rollout per motion: frozen z0, native tracking), with labels (motions, metrics,
              budgets). Its reductions are the drivers' own numpy expressions, one method each, because the order of a float sum
              decides the last bit and the old result files must come out identical:
                mean(m, j)      seeds first, then motions: np.mean([A[motion][:, j].mean(0) for motion], 0)
                                (body_52_test, g1_small_test, loes_noanchor results.json "mean")
                per_motion(m, j) {motion: A[motion][:, j].mean(0)}  (their "per_motion")
                pooled(A)       every motion x seed at once: float(A[..., i].mean()) per metric (child_52_rerun / cem_phc / lso)
                se(m, j)        the standard error over motions of the seed means (ddof 1; plot_child_convergence.py's band)
                paired(X, Y)    per-motion means over seeds, then the paired Wilcoxon test against a reference method and how many
                                motions the reference is lower on (child_52_* means.json)
              markdown() renders rows of (name, {metric: value}) with per-column formats (the drivers' README / analyse tables).
Checks        an analysis's consistency checks: largest differences (diff / high) and mismatch records (settings that are not as
              intended); a dict, so json.dumps writes what the drivers wrote.
ReportWriter  README.md: "# <title>\n\n" + the driver's docstring + a gap + the table lines.
write_json, save_npz   results.json / means.json (json.dumps, indent 1 as the drivers) and per_run.npz (np.savez_compressed, keys in
              the order given), parents created; paired_wilcoxon is the plain Wilcoxon wrapper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths


def write_json(path, obj, indent=1):
    path = paths.writable(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent))
    return path


def save_npz(path, **arrays):
    """np.savez_compressed with the keys in the order given (per_run.npz)"""
    path = paths.writable(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def paired_wilcoxon(x, y):
    """dict(lower = how many pairs have x < y, p = the two-sided Wilcoxon signed-rank p of x against y)"""
    from scipy.stats import wilcoxon
    return dict(lower=int((x < y).sum()), p=float(wilcoxon(x, y).pvalue))


class ResultsTable:
    """an experiment's per-run metrics by method (module docstring)"""

    def __init__(self, metrics, motions=(), budgets=(), seeds=()):
        self.metrics, self.motions, self.budgets, self.seeds = tuple(metrics), list(motions), list(budgets), list(seeds)
        self.data = {}

    # ------------------------------------------------------------------ filling
    def add(self, method, values):
        """values: (motions, seeds, budgets, metrics) or, one rollout per motion, (motions, metrics)"""
        self.data[method] = values
        return values

    @classmethod
    def from_motions(cls, per_motion, metrics, budgets, keys, seeds=()):
        """{motion: {key: array (seeds, budgets, metrics) or (1, metrics)}} -> a table of the keys, np.stack over motions in
        the dict's order (the drivers' res[st][f"{k}_{w}"])"""
        t = cls(metrics, list(per_motion), budgets, seeds)
        for k in keys:
            t.add(k, np.stack([per_motion[m][k] for m in per_motion]))
        return t

    def __getitem__(self, method):
        return self.data[method]

    def __contains__(self, method):
        return method in self.data

    def single(self, method):
        return self.data[method].ndim == 2

    def motion(self, method, i):
        """(seeds, budgets, metrics) of motion i; one-rollout methods repeated over the budgets as (1, budgets, metrics)
        (the drivers' np.repeat(res[st][k][:, None], len(G), axis=1))"""
        A = self.data[method]
        if A.ndim == 2:
            return np.repeat(A[i][None][:, None], len(self.budgets), axis=1)
        return A[i]

    # ------------------------------------------------------------------ reductions (the drivers' expressions)
    def seed_means(self, method, j):
        """[(metrics,)] per motion: the mean over seeds at budget index j"""
        return [self.motion(method, i)[:, j].mean(0) for i in range(len(self.motions))]

    def mean(self, method, j):
        """{metric: mean over motions of the seed means} at budget index j"""
        return dict(zip(self.metrics, np.mean(self.seed_means(method, j), 0).tolist()))

    def se(self, method, j):
        """{metric: standard error over motions of the seed means} (std ddof 1 / sqrt(motions))"""
        Y = np.array(self.seed_means(method, j))
        return dict(zip(self.metrics, (Y.std(0, ddof=1) / np.sqrt(len(Y))).tolist()))

    def mean_se(self, method, j):
        return self.mean(method, j), self.se(method, j)

    def by_budget(self, method, label=lambda g: str(16 * g)):
        """{label(budget): mean(method, j)} over the budgets (rollouts = 16 x generations by default)"""
        return {label(g): self.mean(method, j) for j, g in enumerate(self.budgets)}

    def per_motion(self, method, j=-1):
        """{motion: {metric: mean over seeds}} at budget index j"""
        return {m: dict(zip(self.metrics, self.motion(method, i)[:, j].mean(0).tolist())) for i, m in enumerate(self.motions)}

    @staticmethod
    def pooled(A, metrics, nan=False):
        """{metric: float(A[..., i].mean())} over everything A holds for that metric (np.nanmean with nan=True)"""
        f = np.nanmean if nan else np.mean
        return {k: float(f(A[..., i])) for i, k in enumerate(metrics)}

    @staticmethod
    def paired(X, Y, metrics, skip=("ratio",), per_metric=True):
        """{metric: paired_wilcoxon(x, y)} with x, y the per-motion means over seeds of X, Y (motions x seeds x metrics, or
        motions x metrics for a one-rollout method). per_metric: x = X[..., i].mean(1) (child_52_rerun, child_52_cem_phc);
        otherwise x = X.mean(1)[:, i] (child_52_lso) -- the same numbers up to the last bit, kept as each driver had them"""
        out = {}
        if per_metric:
            for i, k in enumerate(metrics):
                if k not in skip:
                    out[k] = paired_wilcoxon(X[..., i].mean(1), Y[..., i].mean(1))
            return out
        x = X.mean(1) if X.ndim == 3 else X; y = Y.mean(1) if Y.ndim == 3 else Y
        for i, k in enumerate(metrics):
            if k not in skip:
                out[k] = paired_wilcoxon(x[:, i], y[:, i])
        return out

    # ------------------------------------------------------------------ rendering
    @staticmethod
    def markdown(rows, columns, header="method"):
        """the table lines: '| header | col ... |', '|---|...|', then '| name | value ... |' per (name, values) row; columns:
        [(title, key, fmt[, scale])] with fmt a format spec ("{:.2f}", "{:.1f}%") and the value multiplied by scale first"""
        L = ["| " + " | ".join([header] + [c[0] for c in columns]) + " |", "|" + "---|" * (len(columns) + 1)]
        for name, v in rows:
            L.append("| " + " | ".join([str(name)] + [_cell(v, c) for c in columns]) + " |")
        return L


def _cell(v, c):
    x = v[c[1]]
    if len(c) > 3 and c[3] is not None:
        x = c[3] * x
    return c[2].format(x)


class Checks(dict):
    """consistency checks of an analysis: largest differences and mismatch records (module docstring)"""

    def __init__(self, maxes=(), lists=(), **values):
        super().__init__()
        for k in maxes:
            self[k] = 0.0
        for k in lists:
            self[k] = []
        self.update(values)

    def high(self, key, v):
        """self[key] = max(self[key], v) (Python's max, in that order, as the drivers wrote it)"""
        self[key] = max(self[key], v)
        return self[key]

    def diff(self, key, a, b):
        """record float(np.abs(a - b).max()) under the largest-so-far rule; returns this difference"""
        v = float(np.abs(a - b).max())
        self.high(key, v)
        return v

    def add(self, key, records):
        self[key] += list(records)
        return self[key]

    @staticmethod
    def settings(want, got):
        """the keys of `want` whose value `got` (a summary.json) does not have"""
        return [x for x, v in want.items() if got.get(x) != v]

    @staticmethod
    def worst(dicts, key):
        """max over the per-clip check dicts of one key"""
        return max(d[key] for d in dicts)


class ReportWriter:
    """README.md of an analysis: '# title\\n\\n' + doc + gap + '\\n'.join(lines) + '\\n'"""

    def __init__(self, title, doc):
        self.title, self.doc = title, doc

    def render(self, lines, gap="\n"):
        return f"# {self.title}\n\n" + self.doc + gap + "\n".join(lines) + "\n"

    def write(self, path, lines, gap="\n"):
        path = paths.writable(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(lines, gap))
        return path
