#!/usr/bin/env python3
"""compare_lowdim_exact.py -- the 8-start walking test in k = 8, fixed multiplier vs exact observations.

  unconstrained, multiplier   outputs/latent_transfer_own/move-ego-0-2_4/align
  k = 8, multiplier           outputs/lowdim_search/move-ego-0-2_4/k8/align       (multiplier basis)
  k = 8, exact                outputs/lowdim_exact/move-ego-0-2_4/align          (exact basis)
Per setting: L_align of each result against its own target (joint space, comparable across settings), the
correction d = z* - start (size, difference from the z0 start's, difference / size, cos), and the start-distance
-> correction-difference relation over all 28 pairs.
"""
import itertools, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
STEM = "move-ego-0-2_4"
RUNS = {"unconstrained, multiplier": f"outputs/latent_transfer_own/{STEM}/align",
        "k = 8, multiplier": f"outputs/lowdim_search/{STEM}/k8/align",
        "k = 8, exact": f"outputs/lowdim_exact/{STEM}/align"}


def ld(p):
    v = np.load(REPO / p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def main():
    S = np.stack([ld(f"outputs/latent_transfer/{STEM}/starts/{n}.npy") for n in N])
    P = list(itertools.combinations(range(8), 2))
    xs = np.array([np.linalg.norm(S[i] - S[j]) for i, j in P])
    print(f"{'setting':26s} | L_align own (mean, z0 start's z0) | |d|   | diff to z0 start's d, starts 0.14 / 1.40 / 8.28 apart | "
          f"28 pairs: diff/size, cos, corr(start dist, diff)")
    for name, root in RUNS.items():
        if not all((REPO / root / n / "summary.json").exists() for n in N):
            print(f"{name:26s} | not finished"); continue
        Sm = [json.loads((REPO / root / n / "summary.json").read_text()) for n in N]
        d = np.stack([ld(f"{root}/{n}/best_z.npy") for n in N]) - S
        la = np.mean([s["best"]["align"] for s in Sm]); la0 = Sm[0]["origin_z"]["align"]
        size = np.linalg.norm(d, axis=1)
        diff = np.array([np.linalg.norm(d[i] - d[j]) for i, j in P])
        rms = np.array([np.sqrt((size[i] ** 2 + size[j] ** 2) / 2) for i, j in P])
        cos = np.array([d[i] @ d[j] / (size[i] * size[j]) for i, j in P])
        z0d = [np.linalg.norm(d[k] - d[0]) for k in (1, 4, 7)]
        print(f"{name:26s} | {la:.3f} ({la0:.3f})                    | {size.mean():5.2f} | {z0d[0]:5.2f} / {z0d[1]:5.2f} / {z0d[2]:5.2f}"
              f"                              | {np.mean(diff / rms):.2f}, {cos.mean():+.2f}, {np.corrcoef(xs, diff)[0, 1]:+.2f}")
    print("(L_align own: each result against its own target; in brackets z0 itself against the z0 start's target, under that setting's observation)")


if __name__ == "__main__":
    main()
