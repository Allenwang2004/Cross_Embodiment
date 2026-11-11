#!/usr/bin/env python3
"""k_scale_compare.py -- does the search still help when the body's gains change? Per clip: z0's cost and the
searched z's cost on the body as built (k x 1, outputs/c540_bfmglobal_g128: same objective, budget and seed) and on
the body with every joint's k multiplied (scripts/scale_body_gains.py; outputs/k_scale/<body>_k<m>/), with the
bfm / heading / pos parts and the reduction (1 - best / z0). Writes outputs/k_scale/<body>_k<m>/compare.json.
usage: uv run python scripts/k_scale_compare.py outputs/k_scale/child_k1.2 [--base outputs/c540_bfmglobal_g128]"""
import argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CLIPS = ["crawl-0.4-2-d_8", "crouch-0_8", "headstand_4", "jump-2_6", "jump-2_7", "move-ego-0-2_2", "move-ego-0-4_8",
         "rotate-x--5-0.8_4", "rotate-x-5-0.8_9", "rotate-y-5-0.8_7", "rotate-z--5-0.8_6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--base", default="outputs/c540_bfmglobal_g128")
    a = ap.parse_args()
    run, base = REPO / a.run, REPO / a.base
    rows, keys = {}, ("cost", "bfm", "head", "pos")
    print(f"{'clip':20s} | {'k x 1: z0 -> best':>22s} {'cut':>5s} | {'scaled: z0 -> best':>22s} {'cut':>5s} | bfm best (x1 / scaled)")
    for c in CLIPS:
        r = {}
        for tag, d in (("k1", base / c), ("scaled", run / c)):
            f = d / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            r[tag] = dict(z0={k: s["origin_z"][k] for k in keys}, best={k: s["best"][k] for k in keys},
                          cut=1 - s["best"]["cost"] / s["origin_z"]["cost"], gens=s["generations"], evals=s["evals"])
        if len(r) < 2:
            print(f"{c:20s} | (not finished)"); continue
        rows[c] = r
        k1, sc = r["k1"], r["scaled"]
        print(f"{c:20s} | {k1['z0']['cost']:9.3f} -> {k1['best']['cost']:9.3f} {100 * k1['cut']:4.0f}% | "
              f"{sc['z0']['cost']:9.3f} -> {sc['best']['cost']:9.3f} {100 * sc['cut']:4.0f}% | {k1['best']['bfm']:.3f} / {sc['best']['bfm']:.3f}")
    if rows:
        med = lambda t, k: float(np.median([rows[c][t][k] if k == "cut" else rows[c][t]["best"][k] for c in rows]))
        z0m = lambda t: float(np.median([rows[c][t]["z0"]["cost"] for c in rows]))
        summ = {t: dict(z0_cost_median=z0m(t), best_cost_median=med(t, "cost"), best_bfm_median=med(t, "bfm"), cut_median=med(t, "cut"))
                for t in ("k1", "scaled")}
        print(f"{'median':20s} | {summ['k1']['z0_cost_median']:9.3f} -> {summ['k1']['best_cost_median']:9.3f} {100 * summ['k1']['cut_median']:4.0f}% | "
              f"{summ['scaled']['z0_cost_median']:9.3f} -> {summ['scaled']['best_cost_median']:9.3f} {100 * summ['scaled']['cut_median']:4.0f}% | "
              f"{summ['k1']['best_bfm_median']:.3f} / {summ['scaled']['best_bfm_median']:.3f}")
        (run / "compare.json").write_text(json.dumps(dict(base=a.base, run=a.run, clips=rows, median=summ), indent=1))


if __name__ == "__main__":
    main()
