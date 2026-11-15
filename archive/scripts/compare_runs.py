#!/usr/bin/env python3
"""compare_runs.py -- per-category RELATIVE cost curves for several training runs.

Why relative. The balanced set's z0 cost is 0.674 on move and 0.305 on
raisearms, so a run's mean cost is mostly a statement about how much move it
contains: a mean that falls from 0.52 to 0.35 can be move improving by a third
while every other family sits exactly where it started. Dividing each clip by
ITS OWN z0 cost puts every family on the same axis -- 1.0 is "the adapter did
nothing", below 1.0 is real improvement -- which is the only way to see whether
a fix for "move dominates the batch" actually moved the other families.

Reads each run's eval_history.json (written by model/simple/train_es.py, which
records every clip's cost at every eval) and the z0 cost CSV from
scripts/rank_initial_cost.py. Nothing has to be re-rolled out.

Usage:
    uv run scripts/compare_runs.py \
        --runs baseline=outputs/simple_es/child_balanced/lr3e-4_pairs16 \
               conf=outputs/simple_es/child_balanced/exp1_conf \
        --z0-csv outputs/initial_cost/child_balanced500/z0_cost.csv \
        --out outputs/simple_es/child_balanced/compare.png
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same prefix table as scripts/write_balanced_clips.py, and for the same
# reason: move-ego must be tested before the motion names its own name
# contains (move-ego-0-2-raisearms-h-l is a walking clip).
PREFIXES = [("move-ego", "move"), ("rotate", "rotate"), ("crawl", "crawl"),
            ("raisearms", "raisearms"), ("headstand", "headstand"), ("jump", "jump")]


def category(task: str) -> str:
    for pre, cat in PREFIXES:
        if task.startswith(pre):
            return cat
    return "other"


def load_z0(path):
    rows = list(csv.DictReader(open(REPO_ROOT / path if not Path(path).is_absolute() else path)))
    col = [c for c in rows[0] if c not in ("rank", "task", "clip", "mean")]
    return {r["clip"]: float(r[col[0]]) for r in rows}


def task_of(stem, z0):
    """clip stem -> task. rsplit cannot do it for a task whose name has an
    underscore, so the stem is matched against the tasks we actually know."""
    return stem.rsplit("_", 1)[0]


def series(hist, row, z0):
    """(updates, {category: [mean ratio]}, [share of clips beating z0])."""
    ups, per_cat, share = [], collections.defaultdict(list), []
    cats = sorted({category(task_of(c["clip"], z0)) for e in hist for c in e[row]["per_clip"]})
    for e in hist:
        g = collections.defaultdict(list)
        beat = 0
        n = 0
        for c in e[row]["per_clip"]:
            base = z0.get(c["clip"])
            if base is None:
                continue
            r = c["cost"] / base
            g[category(task_of(c["clip"], z0))].append(r)
            beat += r < 1.0
            n += 1
        ups.append(e["update"])
        for k in cats:
            per_cat[k].append(st.mean(g[k]) if g[k] else float("nan"))
        share.append(beat / max(n, 1))
    return ups, per_cat, share


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="label=path/to/ckpt_dir")
    p.add_argument("--z0-csv", default="outputs/initial_cost/child_balanced500/z0_cost.csv")
    p.add_argument("--row", default="child:unseen", help="'child:unseen' = held out, 'child' = seen")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z0 = load_z0(args.z0_csv)
    runs = {}
    for spec in args.runs:
        lab, _, path = spec.partition("=")
        f = Path(path if Path(path).is_absolute() else REPO_ROOT / path) / "eval_history.json"
        if not f.exists():
            print(f"skip {lab}: no {f}")
            continue
        runs[lab] = series(json.load(open(f)), args.row, z0)

    cats = sorted({c for _, pc, _ in runs.values() for c in pc})
    print(f"row '{args.row}' -- mean cost / cost_z0 (1.000 = adapter did nothing)")
    hdr = f"{'run':14s} {'upd':>5s} " + " ".join(f"{c:>10s}" for c in cats) + f" {'ALL':>8s} {'beat z0':>8s}"
    print(hdr)
    for lab, (ups, pc, share) in runs.items():
        for i in (0, len(ups) - 1):
            row = " ".join(f"{pc[c][i]:10.3f}" for c in cats)
            allm = st.mean([pc[c][i] for c in cats if pc[c][i] == pc[c][i]])
            print(f"{lab:14s} {ups[i]:5d} {row} {allm:8.3f} {share[i]:8.1%}")
        print()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cats) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.3 * n, 4.0), sharey=True)
    for j, c in enumerate(cats):
        for lab, (ups, pc, _) in runs.items():
            ax[j].plot(ups, pc[c], lw=1.4, label=lab)
        ax[j].axhline(1.0, c="k", ls=":", lw=.9)
        ax[j].set_title(c); ax[j].set_xlabel("update"); ax[j].grid(alpha=.3)
    for lab, (ups, _, share) in runs.items():
        ax[-1].plot(ups, share, lw=1.4, label=lab)
    ax[-1].set_title("share of clips beating z0"); ax[-1].set_xlabel("update"); ax[-1].grid(alpha=.3)
    ax[0].set_ylabel(f"cost / cost_z0   ({args.row})")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"per-category relative cost, row '{args.row}' -- 1.0 = no better than z0")
    fig.tight_layout()
    out = Path(args.out if Path(args.out).is_absolute() else REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
