#!/usr/bin/env python3
"""compare_exact_search.py -- the b500_targets search with the fixed obs multiplier vs with exact observations.

  multiplier  outputs/b500_targets/<clip>/{bfm,global,align}   (--obs-scale auto)
  exact       outputs/exact_search/<clip>/{bfm,global,align}   (--obs-scale exact)
Same search otherwise. Per clip: L_align of z0 and of the searched z (joint space, comparable across the two),
the global stage's heading / root-xy terms, and how far the search moved from z0. Across the clips: how alike
the corrections z_align - z0 of different clips are (pairwise cos), under each setting.
"""
import itertools, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CLIPS = [("move-ego-0-2", "move-ego-0-2_4"), ("move-ego-0-2", "move-ego-0-2_0"), ("headstand", "headstand_3"),
         ("headstand", "headstand_0"), ("jump-2", "jump-2_0"), ("jump-2", "jump-2_1"),
         ("crawl-0.4-0-d", "crawl-0.4-0-d_0"), ("rotate-z-5-0.8", "rotate-z-5-0.8_0")]
SETS = (("multiplier", "outputs/b500_targets"), ("exact", "outputs/exact_search"))


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def main():
    D = {name: [] for name, _ in SETS}
    print(f"{'clip':18s} {'obs':10s} | z0 L_align | L_align stage (x z0) | global stage L_align | heading / pos (m) | |z_align - z0|")
    for t, stem in CLIPS:
        z0 = ld(REPO / f"data/origin_z/{t}/{stem}.npy")
        for name, root in SETS:
            R = REPO / root / stem
            if not (R / "align/summary.json").exists():
                print(f"{stem:18s} {name:10s} | not finished"); continue
            S = {v: json.loads((R / v / "summary.json").read_text()) for v in ("global", "align")}
            la0, la, lg = S["align"]["origin_z"]["align"], S["align"]["best"]["align"], S["global"]["best"]["align"]
            za = ld(R / "align/best_z.npy"); D[name].append(za - z0)
            print(f"{stem:18s} {name:10s} | {la0:10.3f} | {la:8.3f} ({la / la0:.2f})     | {lg:20.3f} "
                  f"| {S['global']['best']['head']:.3f} / {S['global']['best']['pos']:.2f}    | {np.linalg.norm(za - z0):.2f}")
    for name, d in D.items():
        if len(d) < 2:
            continue
        u = np.stack(d); u = u / np.linalg.norm(u, axis=1, keepdims=True)
        c = [u[i] @ u[j] for i, j in itertools.combinations(range(len(u)), 2)]
        print(f"{name:10s}: correction size {np.mean([np.linalg.norm(x) for x in d]):.2f}, "
              f"pairwise cos between clips' corrections {np.mean(c):+.3f} [{min(c):+.2f}..{max(c):+.2f}]")


if __name__ == "__main__":
    main()
