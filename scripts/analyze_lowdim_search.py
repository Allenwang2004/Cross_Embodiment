#!/usr/bin/env python3
"""analyze_lowdim_search.py -- does restricting the search to a k-dim subspace make the correction a
function of the start, and does the loss still come down?

Walking (move-ego-0-2_4), 8 starts each tracking its own adult rollout, two-stage L_align, no penalty.
  k = 256  unconstrained          outputs/latent_transfer_own/move-ego-0-2_4/align
  k = 8 / 16 / 32 / 64 / 128      outputs/lowdim_search/move-ego-0-2_4/k<k>/align
(subspace = top-k of the uncentered PCA of the b500 corrections, this clip left out).
Per k: L_align of each result against its own target (the search's own score), the correction
d = z* - start (size, pairwise difference, difference / size, cos), and how the result-to-result
distance follows the start-to-start distance. Writes outputs/lowdim_search/lowdim_summary.csv and
outputs/lowdim_search/lowdim_search.png.
"""
import csv, itertools, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
STEM = "move-ego-0-2_4"
RUNS = {8: REPO / f"outputs/lowdim_search/{STEM}/k8/align", 16: REPO / f"outputs/lowdim_search/{STEM}/k16/align",
        32: REPO / f"outputs/lowdim_search/{STEM}/k32/align", 64: REPO / f"outputs/lowdim_search/{STEM}/k64/align",
        128: REPO / f"outputs/lowdim_search/{STEM}/k128/align", 256: REPO / f"outputs/latent_transfer_own/{STEM}/align"}
COL = {8: "#e87ba4", 16: "#008300", 32: "#2a78d6", 64: "#1baf7a", 128: "#eda100", 256: "#eb6834"}


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def main():
    S = np.stack([ld(REPO / f"outputs/latent_transfer/{STEM}/starts/{n}.npy") for n in N])
    P = list(itertools.combinations(range(8), 2))
    xs = np.array([np.linalg.norm(S[i] - S[j]) for i, j in P])
    rows, res = [], {}
    for k, D in RUNS.items():
        if not all((D / n / "summary.json").exists() for n in N):
            print(f"k={k}: not complete"); continue
        Sm = [json.loads((D / n / "summary.json").read_text()) for n in N]
        Z = np.stack([ld(D / n / "best_z.npy") for n in N]); d = Z - S
        la = np.array([s["best"]["align"] for s in Sm])
        la0 = Sm[0]["origin_z"]["align"]                       # z0 itself against the z0 start's target
        size = np.linalg.norm(d, axis=1)
        diff = np.array([np.linalg.norm(d[i] - d[j]) for i, j in P])
        rms = np.array([np.sqrt((size[i] ** 2 + size[j] ** 2) / 2) for i, j in P])
        cos = np.array([d[i] @ d[j] / (size[i] * size[j]) for i, j in P])
        ys = np.array([np.linalg.norm(Z[i] - Z[j]) for i, j in P])
        res[k] = dict(la=la, ratio=diff / rms, diff=diff)
        rows.append(dict(k=k, l_align_mean=la.mean(), l_align_max=la.max(), l_align_z0=la0,
                         corr_size=size.mean(), corr_diff=diff.mean(), diff_over_size=(diff / rms).mean(),
                         cos_mean=cos.mean(), result_pair=ys.mean(), corr_start_result=float(np.corrcoef(xs, ys)[0, 1]),
                         diff_closest_pair=diff[0], start_closest_pair=xs[0]))
    out = REPO / "outputs/lowdim_search"
    with open(out / "lowdim_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"starts: pairwise {xs.mean():.2f} [{xs.min():.2f}-{xs.max():.2f}]")
    print(f"{'k':>4s} | {'L_align own':>18s} | {'|d|':>5s} | {'|d_i-d_j|':>9s} | {'diff/size':>9s} | {'cos(d_i,d_j)':>12s} | "
          f"{'z0 vs 0.5deg: start -> diff':>28s} | corr(start dist, result dist)")
    for r in sorted(rows, key=lambda r: r["k"]):
        print(f"{r['k']:4d} | {r['l_align_mean']:.3f} (max {r['l_align_max']:.3f}) | {r['corr_size']:5.2f} | {r['corr_diff']:9.2f} | "
              f"{r['diff_over_size']:9.2f} | {r['cos_mean']:+12.2f} | {r['start_closest_pair']:12.2f} -> {r['diff_closest_pair']:5.2f}"
              f"      | {r['corr_start_result']:+.2f}")
    print(f"(z0's own L_align against the z0 start's target: {rows[0]['l_align_z0']:.3f})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor(SURF)
    for k in sorted(res):
        r = next(x for x in rows if x["k"] == k)
        a1.scatter(xs, res[k]["ratio"], s=34, color=COL[k], edgecolor=SURF, lw=.8, zorder=4,
                   label=(f"k = {k}" if k < 256 else "k = 256 (unconstrained)")
                         + f":  |d| {r['corr_size']:.1f}, difference {r['corr_diff']:.1f}, cos {r['cos_mean']:+.2f}")
    a1.axhline(np.sqrt(2), color=INK3, lw=1, ls=":"); a1.text(9.5, np.sqrt(2) + .03, "perpendicular (1.41)", fontsize=8.5, color=INK2, ha="right")
    a1.axhline(0, color=INK3, lw=1, ls=":"); a1.text(9.5, .03, "same direction (0): what learning needs", fontsize=8.5, color=INK2, ha="right")
    a1.set_xscale("log"); a1.set_xlim(0.1, 10); a1.set_ylim(-0.05, 1.6)
    a1.set_xlabel("start distance |s_i − s_j|  (log scale)", fontsize=9.5, color=INK2)
    a1.set_ylabel("correction difference ÷ correction size", fontsize=9.5, color=INK2)
    a1.set_title("does a smaller search space make nearby starts get the same correction?", fontsize=10, color=INK)
    a1.legend(frameon=False, fontsize=8.3, loc="lower left", bbox_to_anchor=(0, .08), labelcolor=INK)
    ks = sorted(res)
    for x, k in enumerate(ks):
        a2.scatter(np.full(8, x) + np.linspace(-.18, .18, 8), res[k]["la"], s=30, color=COL[k], edgecolor=SURF, lw=.8, zorder=4)
        a2.plot([x - .28, x + .28], [res[k]["la"].mean()] * 2, color=INK, lw=1.6)
    a2.axhline(rows[0]["l_align_z0"], color=INK3, lw=1, ls=":")
    a2.text(len(ks) - .5, rows[0]["l_align_z0"], f"z0 without search ({rows[0]['l_align_z0']:.2f})", fontsize=8.5, color=INK2,
            ha="right", va="bottom")
    a2.set_xticks(range(len(ks))); a2.set_xticklabels([f"k = {k}" if k < 256 else "256\n(unconstrained)" for k in ks])
    a2.set_ylabel("L_align of the result against its own target", fontsize=9.5, color=INK2)
    a2.set_ylim(0, None)
    a2.set_title("does the loss still come down? (8 starts each, bar = mean)", fontsize=10, color=INK)
    for ax in (a1, a2):
        ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)
    fig.tight_layout(); fig.savefig(out / "lowdim_search.png", dpi=150, facecolor=SURF)
    print(f"-> {out / 'lowdim_summary.csv'}\n-> {out / 'lowdim_search.png'}")


if __name__ == "__main__":
    main()
