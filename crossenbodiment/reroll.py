"""reroll.py -- the latents a finished search is re-rolled with, and the index that finds them again (OOP refactor step 8).

Every Task-page driver had a `prepare` that walked each run's z_trace.npz and wrote rollouts/<name>/{z.npy, index.json}, then
`single_z_search.py --eval-z z.npy --eval-save-qpos` rolled the latents out, and `analyse` looked them up in index.json. Three
index layouts grew (all {"map": {run: ...}} plus a few meta keys such as clip / body / algo):
  last         map[run] = i                                      the best latent after the last generation (loes_weight_test)
  generation   map[run] = {best: [i per generation 1..G], centre: [i per budget]}
                                                                 child_52_rerun, child_52_cem_phc, loes_noanchor's child
  budget       map[run] = {best: [i per budget], centre: [i per budget]}
                                                                 body_52_test, g1_small_test, loes_noanchor's payload / limit_leg / g1
Reroller.prepare() writes one versioned layout (schema 2) for all of them:
  {"schema": 2, <meta>, "generations": G, "budgets": [g, ...], "map": {run: {"best": [i or null per generation 1..G],
                                                                             "centre": [i or null per budget]}}}
with null where that generation's best (or that budget's centre) was not re-rolled, so z.npy -- the latents, in the order the
drivers wrote them -- stays byte-identical to theirs (mode "every": each run's distinct best latents, a latent repeated from the
previous generation shares its index, then its centres at the budgets; "budgets": the best and the centre latent at each budget;
"last": the best latent after G generations only). best[-1] and centre[-1] are the last generation's / budget's in every layout,
which is all that other readers (fall_definitions, success_definitions, drift_success_child, plot_child_convergence) use.
RerollIndex (Reroller.read) reads any of the four and answers best_generations / best_budgets / best_last / centre / rerolled;
check_vs_curve compares the re-rolled costs with the search's own best_so_far (curve.csv) at those generations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import paths

SCHEMA = 2
MODES = ("every", "budgets", "last")


class Reroller:
    """the latents of finished single-latent searches to re-roll (module docstring)

    gens     the searches' generations (z_trace.npz has gens + 1 rows: gen -1 is the start)
    budgets  the generations whose centre (and, in mode "budgets", best) latent is re-rolled
    best     "every" | "budgets" | "last"; centre: re-roll the centre (z_mean) latent at each budget
    """

    def __init__(self, gens, budgets=(), best="every", centre=True):
        assert best in MODES, best
        self.gens, self.budgets, self.best, self.centre = int(gens), [int(g) for g in budgets], best, centre
        if best == "last":
            self.budgets = self.budgets or [self.gens]

    def collect(self, traces):
        """traces: [(run name, z_trace.npz path)] -> (Z list of latents, index map) in the drivers' order"""
        Z, idx = [], {}
        for r, f in traces:
            t = np.load(f); b = t["z_best"]
            assert len(b) == self.gens + 1, (r, len(b), self.gens + 1)
            if self.best == "every":
                best = []
                for g in range(1, self.gens + 1):
                    if not best or np.abs(b[g] - b[g - 1]).max() > 0:
                        Z.append(b[g]); best.append(len(Z) - 1)
                    else:
                        best.append(best[-1])
            elif self.best == "budgets":
                best = [None] * self.gens
                for g in self.budgets:
                    best[g - 1] = len(Z); Z.append(b[g])
            else:
                best = [None] * self.gens; best[-1] = len(Z); Z.append(b[self.gens])
            centre = [None] * len(self.budgets)
            if self.centre:
                m = t["z_mean"]; assert len(m) == self.gens + 1, (r, len(m), self.gens + 1)
                for j, g in enumerate(self.budgets):
                    centre[j] = len(Z); Z.append(m[g])
            idx[r] = dict(best=best, centre=centre)
        return Z, idx

    def index(self, idx, **meta):
        return dict(schema=SCHEMA, **meta, generations=self.gens, budgets=self.budgets, best_mode=self.best, map=idx)

    def prepare(self, out_dir, traces, **meta):
        """write <out_dir>/z.npy (np.save of the stacked latents) and <out_dir>/index.json (schema 2, `meta` first);
        traces: [(run name, run folder or its z_trace.npz)]. Returns the RerollIndex."""
        out_dir = paths.writable(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        traces = [(r, Path(p) / "z_trace.npz" if Path(p).suffix != ".npz" else Path(p)) for r, p in traces]
        Z, idx = self.collect(traces)
        np.save(out_dir / "z.npy", np.stack(Z))
        data = self.index(idx, **meta)
        (out_dir / "index.json").write_text(json.dumps(data))
        return RerollIndex(data)

    @staticmethod
    def read(path, budgets=None, gens=None):
        """RerollIndex of an index.json (or its folder) of any layout; old layouts need the experiment's budgets (and gens,
        default the last budget) to tell per-generation from per-budget lists"""
        path = Path(path)
        if path.is_dir():
            path = path / "index.json"
        with open(path) as f:
            return RerollIndex(json.load(f), budgets, gens)


class RerollIndex:
    """index.json, any layout (module docstring)"""

    def __init__(self, data, budgets=None, gens=None):
        self.data, self.map = data, data["map"]
        self.schema = data.get("schema", 1)
        if self.schema == SCHEMA:
            self.budgets, self.gens = list(data["budgets"]), int(data["generations"])
        else:
            assert self.schema == 1, f"unknown index schema {self.schema}"
            self.budgets = [int(g) for g in budgets] if budgets is not None else None
            self.gens = int(gens) if gens is not None else (self.budgets[-1] if self.budgets else None)

    def __contains__(self, run):
        return run in self.map

    @property
    def runs(self):
        return list(self.map)

    def layout(self, run):
        """'v2' | 'last' | 'generation' | 'budget' (old layouts, told apart by the list lengths) | 'unknown' (no budgets given)"""
        e = self.map[run]
        if self.schema == SCHEMA:
            return "v2"
        if not isinstance(e, dict):
            return "last"
        n = len(e["best"])
        if self.budgets is None:
            return "unknown"
        if n == len(self.budgets) and n != self.gens:
            return "budget"
        if n == self.gens:
            return "generation"
        raise ValueError(f"{run}: {n} best latents fit neither {len(self.budgets)} budgets nor {self.gens} generations")

    def best_generations(self, run):
        """[index or None] per generation 1..gens (None: not re-rolled)"""
        e, lay = self.map[run], self.layout(run)
        if lay in ("v2", "generation"):
            return list(e["best"])
        if lay == "last":
            return [None] * (self.gens - 1) + [e]
        if lay == "budget":
            out = [None] * self.gens
            for g, i in zip(self.budgets, e["best"]):
                out[g - 1] = i
            return out
        raise ValueError(f"{run}: pass the experiment's budgets to read this index")

    def best_at(self, run, gen):
        i = self.best_generations(run)[gen - 1]
        if i is None:
            raise KeyError(f"{run}: the best latent after {gen} generations was not re-rolled")
        return i

    def best_budgets(self, run):
        """[index] of the best latent at each budget"""
        if self.layout(run) == "budget":
            return list(self.map[run]["best"])
        return [self.best_at(run, g) for g in self.budgets]

    def best_last(self, run):
        e = self.map[run]
        return e if not isinstance(e, dict) else e["best"][-1]

    def centre(self, run):
        """[index or None] of the centre latent at each budget"""
        e = self.map[run]
        return [None] * len(self.budgets or [None]) if not isinstance(e, dict) else list(e.get("centre", []))

    def centre_last(self, run):
        return self.map[run]["centre"][-1]

    def rerolled(self, run):
        """[(generation, index)] of every re-rolled best latent, in generation order"""
        return [(g, i) for g, i in enumerate(self.best_generations(run), 1) if i is not None]

    def check_vs_curve(self, run, cost, best_so_far):
        """the largest |re-rolled cost - the search's best_so_far| over the re-rolled best latents of `run`"""
        gi = self.rerolled(run)
        return check_vs_curve(cost, best_so_far, [i for _, i in gi], [g for g, _ in gi])


def check_vs_curve(cost, best_so_far, idx, gens):
    """float(max |cost[idx] - best_so_far[gens - 1]|): the drivers' `np.abs(np.array(cost)[ix[r]["best"]] - bs[...]).max()`"""
    return float(np.abs(np.asarray(cost)[idx] - np.asarray(best_so_far)[[g - 1 for g in gens]]).max())


def read_best_so_far(run_dir):
    """curve.csv's best_so_far column as float64 (one row per generation)"""
    with open(Path(run_dir) / "curve.csv") as f:
        return np.array([float(x["best_so_far"]) for x in csv.DictReader(f)])


class EvalRollouts:
    """a re-roll folder's outputs (single_z_search / robot_z_search --eval-z): qpos (as saved, float32), cost (list of floats, one
    per latent, from eval.csv), base (eval_base.json's cost: z0's), index (RerollIndex)"""

    def __init__(self, d, budgets=None, gens=None):
        self.dir = Path(d)
        self.index = Reroller.read(self.dir / "index.json", budgets, gens)

    @property
    def qpos(self):
        if not hasattr(self, "_q"):
            self._q = np.load(self.dir / "eval_qpos.npz")["qpos"]
        return self._q

    @property
    def cost(self):
        with open(self.dir / "eval.csv") as f:
            return [float(r["cost"]) for r in csv.DictReader(f)]

    @property
    def base(self):
        with open(self.dir / "eval_base.json") as f:
            return json.load(f)["cost"]
