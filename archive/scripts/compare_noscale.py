#!/usr/bin/env python3
"""compare_noscale.py -- search with vs without obs scaling, same settings otherwise.

  scaled     outputs/b500_targets/<clip>/{bfm,global,align}   (--obs-scale auto)
  no scale   outputs/noscale_test/<clip>/{bfm,global,align}   (--obs-scale none)
L_align is joint-space, so it compares across the two; bfm does not (B sees scaled vs raw obs), so it is
only reported within a setting. Heading / root-xy terms of the global stage are scale-independent too.
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CLIPS = [("move-ego-0-2", "move-ego-0-2_4"), ("move-ego-0-2", "move-ego-0-2_0"), ("headstand", "headstand_3"),
         ("headstand", "headstand_0")]


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def main():
    print(f"{'clip':16s} {'setting':9s} | {'z0 L_align':>10s} | {'L_align stage: L_align':>22s} (x z0) | {'global stage: L_align':>21s} "
          f"| heading / pos | bfm stage: bfm (x z0, within setting) | |z_align - z0|")
    for t, stem in CLIPS:
        z0 = ld(REPO / f"data/origin_z/{t}/{stem}.npy")
        for name, root in (("scaled", "outputs/b500_targets"), ("no scale", "outputs/noscale_test")):
            R = REPO / root / stem
            if not (R / "align/summary.json").exists():
                print(f"{stem:16s} {name:9s} | not finished"); continue
            S = {v: json.loads((R / v / "summary.json").read_text()) for v in ("bfm", "global", "align")}
            la0 = S["align"]["origin_z"]["align"]; la = S["align"]["best"]["align"]; lg = S["global"]["best"]["align"]
            print(f"{stem:16s} {name:9s} | {la0:10.3f} | {la:15.3f} ({la / la0:.2f}) | {lg:14.3f} ({lg / la0:.2f}) "
                  f"| {S['global']['best']['head']:.3f} / {S['global']['best']['pos']:.3f} "
                  f"| {S['bfm']['best']['bfm']:.3f} ({S['bfm']['best']['bfm'] / S['bfm']['origin_z']['bfm']:.2f}) "
                  f"| {np.linalg.norm(ld(R / 'align/best_z.npy') - z0):.2f}   [obs_scale={S['align']['obs_scale']}]")


if __name__ == "__main__":
    main()
