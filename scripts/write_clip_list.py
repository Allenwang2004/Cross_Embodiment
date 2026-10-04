#!/usr/bin/env python3
"""write_clip_list.py -- the clips whose z0 cost is above a threshold, as a
clip list for train_es.py --clip-list.

scripts/rank_initial_cost.py scores z0 on every clip. Most of them are already
cheap -- on child 271 of 540 are under 0.3 -- and a cell with nothing to gain
contributes no gradient while still taking its share of the rollout budget.
This cuts them.

Usage:
    uv run scripts/write_clip_list.py --csv outputs/initial_cost/child/z0_cost.csv \
        --body child --min-cost 0.3 --out datasets/crossenbodiment-10bodies/splits/child_z0gt03_clips.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", required=True, help="z0_cost.csv from rank_initial_cost.py")
    p.add_argument("--body", required=True, help="which column to threshold ('mean' works too)")
    p.add_argument("--min-cost", type=float, default=0.3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = list(csv.DictReader(open(REPO_ROOT / args.csv if not Path(args.csv).is_absolute() else args.csv)))
    if args.body not in rows[0]:
        raise SystemExit(f"column '{args.body}' not in {args.csv}: {list(rows[0])}")
    keep = [r for r in rows if float(r[args.body]) > args.min_cost]
    out = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # "<task> <trial>": the manifest's key is (reward_name, trial), and the
    # stem is <task>_<trial>, which rsplit cannot undo for tasks whose own name
    # contains an underscore -- so write the two fields apart.
    lines = []
    for r in keep:
        task, stem = r["task"], r["clip"]
        trial = stem[len(task) + 1:] if stem.startswith(task + "_") else stem.rsplit("_", 1)[1]
        lines.append(f"{task} {trial}")
    out.write_text("\n".join(lines) + "\n")
    costs = [float(r[args.body]) for r in keep]
    print(f"{len(keep)} of {len(rows)} clips with {args.body} z0 > {args.min_cost} "
          f"(mean {sum(costs) / len(costs):.3f}, min {min(costs):.3f}, max {max(costs):.3f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
