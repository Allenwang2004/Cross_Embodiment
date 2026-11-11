#!/usr/bin/env python3
"""analyze_search_convergence.py -- after how many evals does each b500_targets search stop paying?

For every finished search (outputs/b500_targets/<clip>/{bfm,global,align}): start cost c_s (the
"warm start" line of its log: z0 for stage 1, the stage-1 best for stage 2), the best-so-far curve
(curve.csv, 16 evals per generation) and the final best c_f. Reported per stage and family:
  - evals until the best-so-far has made 90 / 95 / 99 / 100 % of its total gain c_s - c_f
  - what stopping at a smaller budget costs: best-so-far at that budget minus c_f, in units of
    the clip's z0 cost (so 0.01 = one percentage point of the "cost / z0" ratio)
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT, LOG = REPO / "outputs/b500_targets", REPO / "outputs/train_logs/b500_targets"
FRACS = (0.9, 0.95, 0.99, 1.0)
BUDGETS = {"bfm": (256, 512, 1024, 2048, 3072, 4096), "global": (256, 512, 1024, 1536, 2048),
           "align": (256, 512, 1024, 1536, 2048)}


def family(stem):
    t = stem.rsplit("_", 1)[0]
    if t.startswith("move"):
        return "move"
    return t.split("-")[0]


R = defaultdict(list)       # (stage, family) -> rows
for d in sorted(ROOT.iterdir()):
    for st in ("bfm", "global", "align"):
        S = d / st / "summary.json"
        if not S.exists():
            continue
        S = json.loads(S.read_text())
        m = re.search(r"warm start: cost\s+([0-9.]+)", (LOG / f"{d.name}_{st}.log").read_text())
        c_s = float(m.group(1)) if m else S["origin_z"]["cost"]
        rows = list(csv.DictReader(open(d / st / "curve.csv")))
        ev = np.array([int(r["evals"]) for r in rows]); bsf = np.array([float(r["best_so_far"]) for r in rows])
        c_f, z0c = S["best"]["cost"], S["origin_z"]["cost"]
        gain = c_s - c_f
        hit = {}
        for f in FRACS:
            ok = np.nonzero(c_s - bsf >= f * gain - 1e-12)[0]
            hit[f] = int(ev[ok[0]]) if len(ok) else int(ev[-1])
        if gain <= 0:                          # never beat its start
            hit = {f: 0 for f in FRACS}
        loss = {b: float((bsf[min(np.searchsorted(ev, b), len(ev) - 1)] - c_f) / z0c) for b in BUDGETS[st]}
        R[(st, family(d.name))].append(dict(hit=hit, loss=loss, gain=gain / z0c, none=gain <= 0))

fams = sorted({f for _, f in R})
for st, lab in (("bfm", "stage 1 (bfm + anchor, 4096 evals)"), ("global", "stage 2 global (2048)"),
                ("align", "stage 2 L_align (2048)")):
    print(f"\n=== {lab} ===")
    print(f"{'family':10s} {'n':>4s} {'no gain':>7s} | evals to reach x% of the final gain: median [p75, p90]"
          f"{'':4s}| gap left at budget (cost / z0 units, median / p90)")
    print(f"{'':24s}| " + "   ".join(f"{int(f * 100):>3d}%{'':10s}" for f in FRACS) + " | "
          + "  ".join(f"{b:>11d}" for b in BUDGETS[st]))
    for fam in fams + ["ALL"]:
        rows = [r for (s, f), v in R.items() if s == st and (fam == "ALL" or f == fam) for r in v]
        if not rows:
            continue
        h = " ".join(f"{np.median([r['hit'][f] for r in rows]):5.0f} [{np.percentile([r['hit'][f] for r in rows], 75):4.0f},"
                     f"{np.percentile([r['hit'][f] for r in rows], 90):4.0f}]" for f in FRACS)
        g = "  ".join(f"{np.median([r['loss'][b] for r in rows]):.3f}/{np.percentile([r['loss'][b] for r in rows], 90):.3f}"
                      for b in BUDGETS[st])
        print(f"{fam:10s} {len(rows):4d} {sum(r['none'] for r in rows):7d} | {h} | {g}")
