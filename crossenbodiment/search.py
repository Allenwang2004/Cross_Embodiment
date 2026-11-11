"""search.py -- the search loops: SearchLoop (one latent) and SequenceSearchLoop (a latent per step) (OOP refactor step 6,
docs/refactor/PLAN.md).

A loop owns what every search did around its optimiser (crossenbodiment/optim.py) and its scorer (crossenbodiment/problems.py):

SearchLoop (scripts/single_z_search.py, robot_z_search.py), per generation gen = gen0 .. n_gens - 1:
    cand = opt.ask(best z)                       2 * pairs candidates
    sc = score(cand)                             sc.cost: (2 * pairs,) costs
    i = first argmin; if cost[i] < best cost (strict): best = record(sc, i, cand[i], gen, "sample")
    opt.tell(cand, cost, best z); z = opt.z; path_len += angle(z_prev, z)
    row = row(loop, gen, sc)                     the curve row (built before the mean-z rollout, mean_z_cost "" unless rolled out)
    every eval_every steps and at the last: sc_mean = score(copies of z); row mean_z_cost; best may become it ("mean_z")
    curve += row; z trace (z_mean, z_best) += z, best z      (the traces start with the start latent: gen -1)
  Resume (robot_z_search --resume): construct the loop with gen0, the earlier curve rows, traces, best and path_len.

SequenceSearchLoop (scripts/lso_search.py, robot_lso_search.py), per iteration it = 0 .. iters - 1:
    Z = opt.ask()                                n + 1 sequences, slot 0 = the mean
    sc = score(Z)                                sc.r: per-step rewards (n + 1, S); sc.q: the rollouts kept for --budgets
    keep q_it<it> (--keep-iters) and q_mean_it<it> (it in --budgets) of slot 0
    costs = 1 - sum_t r / S; i = first argmin over the samples (best_from "samples") or every slot ("all"); strict <
    it + 1 in --budgets: keep q_best_it<it + 1> = the best so far, best_cost_at[it + 1]
    opt.tell(Z, r, costs); row = row(loop, it, sc, costs); curve += row; trace += mu (float32); report; on_iter
Both loops only orchestrate: every number is computed by the scorer / optimiser exactly as the scripts did.
"""
from __future__ import annotations

import numpy as np

from .geometry import ang_pair


def copies(z, n):
    """every slot one latent: np.repeat(z[None], n, axis=0) (BatchRollout.copies)"""
    return np.repeat(z[None], n, axis=0)


class SearchLoop:
    """the generation loop of a single-latent search (module docstring).

    opt         an optimiser of crossenbodiment/optim.py (ask(best_z), tell(cand, cost, best_z), .z)
    score       cand (n, 256) -> an object with .cost (n,)
    record      (sc, i, z, gen, source) -> the best-so-far record (a dict with at least cost and z; it copies z)
    best        the starting best record (z0's rollout, gen -1)
    row         (loop, gen, sc) -> the curve row (dict); report (loop, gen, sc, sc_mean, row): after a mean-z rollout
    """

    def __init__(self, opt, score, record, n_slots, n_gens, best, eval_every=0, row=None, report=None, gen0=0, path_len=0.0,
                 curve=None, trace_mean=None, trace_best=None):
        self.opt, self.score, self.record = opt, score, record
        self.n, self.n_gens, self.best = n_slots, n_gens, best
        self.eval_every, self.row, self.report = eval_every, row, report
        self.gen0, self.path_len = gen0, path_len
        self.z = opt.z
        self.curve = [] if curve is None else curve
        self.trace_mean = [self.z.copy()] if trace_mean is None else trace_mean
        self.trace_best = [self.z.copy()] if trace_best is None else trace_best

    def mean_due(self, gen):
        return bool(self.eval_every) and (gen % self.eval_every == 0 or gen == self.n_gens - 1)

    def step(self, gen):
        cand = self.opt.ask(self.best["z"])
        sc = self.score(cand)
        cost = sc.cost
        i = int(np.argmin(cost))
        if cost[i] < self.best["cost"]:
            self.best = self.record(sc, i, cand[i], gen, "sample")
        z_prev = self.z
        self.opt.tell(cand, cost, self.best["z"])
        self.z = self.opt.z
        self.path_len += ang_pair(z_prev, self.z)
        row = self.row(self, gen, sc)
        if self.mean_due(gen):
            # the iterate itself, not just its samples -- one extra rollout
            sm = self.score(copies(self.z, self.n))
            row["mean_z_cost"] = float(sm.cost[0])
            if sm.cost[0] < self.best["cost"]:
                self.best = self.record(sm, 0, self.z, gen, "mean_z")
            if self.report is not None:
                self.report(self, gen, sc, sm, row)
        self.curve.append(row)
        self.trace_mean.append(self.z.copy())
        self.trace_best.append(self.best["z"].copy())
        return sc

    def run(self):
        for gen in range(self.gen0, self.n_gens):
            self.step(gen)
        return self.best


class SequenceSearchLoop:
    """the iteration loop of a latent-sequence search (module docstring).

    opt         LSO / TrajectoryAdaptation (ask(), tell(Z, r, costs), .mu)
    score       Z (n + 1, S, 256) -> an object with .r (n + 1, S) per-step rewards and .q (n + 1, ...) the rollouts to keep
    S           steps per sequence (the costs are 1 - sum_t r / S)
    best        the starting record: dict(cost, z, it=-1, src="start", q)
    budgets     iteration counts after which the best and the mean rollouts are kept (q_best_it<k>, q_mean_it<k>)
    keep        the rollouts.npz dict the loop adds to (insertion order = the file's key order)
    """

    def __init__(self, opt, score, iters, S, best, budgets=(), keep=None, keep_iters=(), best_from="samples", row=None,
                 report=None, on_iter=None):
        self.opt, self.score, self.iters, self.S, self.best = opt, score, iters, S, best
        self.budgets, self.keep_iters, self.best_from = budgets, keep_iters, best_from
        self.keep = {} if keep is None else keep
        self.row, self.report, self.on_iter = row, report, on_iter
        self.curve, self.trace, self.budget_cost = [], [opt.mu.astype(np.float32)], {}

    def step(self, it):
        Z = self.opt.ask()
        sc = self.score(Z)
        if it in self.keep_iters:                                    # slot 0: the mean after `it` updates
            self.keep[f"q_it{it}"] = sc.q[0].astype(np.float32)
        if it in self.budgets:
            self.keep[f"q_mean_it{it}"] = sc.q[0].astype(np.float32)
        costs = 1 - sc.r.sum(1) / self.S
        # samples only (slot 0, the mean, is logged, never the result) unless best_from "all"
        i = int(np.argmin(costs)) if self.best_from == "all" else 1 + int(np.argmin(costs[1:]))
        if costs[i] < self.best["cost"]:
            self.best = dict(cost=float(costs[i]), z=Z[i].copy(), it=it, src="mean" if i == 0 else "sample", q=sc.q[i].copy())
        if it + 1 in self.budgets:
            self.keep[f"q_best_it{it + 1}"] = self.best["q"].astype(np.float32); self.budget_cost[it + 1] = self.best["cost"]
        self.opt.tell(Z, sc.r, costs)
        row = self.row(self, it, sc, costs)
        self.curve.append(row); self.trace.append(self.opt.mu.astype(np.float32))
        if self.report is not None:
            self.report(self, it, sc, costs, row)
        if self.on_iter is not None:
            self.on_iter(self, it)
        return sc

    def run(self):
        for it in range(self.iters):
            self.step(it)
        return self.best

    def traced_iters(self, trace="all"):
        """the iterations mu_trace.npz holds: every one, or 0 and the --budgets reached"""
        return list(range(len(self.trace))) if trace == "all" else [0] + [k for k in self.budgets if k < len(self.trace)]
