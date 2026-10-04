#!/usr/bin/env python3
"""plot_b500_pca.py -- does the stage-2 L_align search (outputs/b500_targets/<clip>/align) keep
the motion-category structure of z0?

One PCA fit on z0 and z_align together (all latents at radius 16), two panels on the same
axes: z0 left, z_align right, colour + marker = motion category
(datasets/crossenbodiment-child-balanced/splits/balanced500_categories.txt).
Numbers in the full 256-d space, z0 vs z_align:
  - 5-NN category agreement: share of each latent's 5 nearest neighbours in its own category
  - silhouette by category (Euclidean)
  - correlation between the two sets' pairwise distances (same clip pairs)
  - share of z_align whose nearest z0 is its own z0
Writes outputs/b500_targets/pca_align.png.
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "outputs/b500_targets"
SPL = REPO / "datasets/crossenbodiment-child-balanced/splits"
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
CATS = ["crawl", "headstand", "jump", "move", "raisearms", "rotate"]
COL = dict(zip(CATS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MRK = dict(zip(CATS, ["o", "s", "^", "D", "v", "P"]))


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def knn_agree(X, lab, k=5):
    D = np.linalg.norm(X[:, None] - X[None], axis=-1); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    return float((lab[nn] == lab[:, None]).mean())


def silhouette(X, lab):
    D = np.linalg.norm(X[:, None] - X[None], axis=-1); s = []
    for i in range(len(X)):
        own = (lab == lab[i]); own[i] = False
        a = D[i, own].mean()
        b = min(D[i, lab == c].mean() for c in set(lab) if c != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def main():
    cat_of = dict(l.split() for l in open(SPL / "balanced500_categories.txt") if l.strip())
    names, Z0, ZA, lab = [], [], [], []
    for t, k in (l.split() for l in open(SPL / "balanced500_clips.txt") if l.strip()):
        d = ROOT / f"{t}_{k}" / "align"
        if not (d / "summary.json").exists():
            continue
        names.append(f"{t}_{k}"); lab.append(cat_of[t])
        Z0.append(ld(REPO / f"data/origin_z/{t}/{t}_{k}.npy")); ZA.append(ld(d / "best_z.npy"))
    Z0, ZA, lab = np.stack(Z0), np.stack(ZA), np.array(lab)
    n = len(names)
    P = np.concatenate([Z0, ZA]); mu = P.mean(0)
    _, sv, Vt = np.linalg.svd(P - mu, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum(); Y0, YA = (Z0 - mu) @ Vt[:2].T, (ZA - mu) @ Vt[:2].T

    iu = np.triu_indices(n, 1)
    D0 = np.linalg.norm(Z0[:, None] - Z0[None], axis=-1)[iu]
    DA = np.linalg.norm(ZA[:, None] - ZA[None], axis=-1)[iu]
    own = np.argmin(np.linalg.norm(ZA[:, None] - Z0[None], axis=-1), axis=1) == np.arange(n)
    moved = np.linalg.norm(ZA - Z0, axis=1)
    m = dict(knn0=knn_agree(Z0, lab), knnA=knn_agree(ZA, lab), sil0=silhouette(Z0, lab), silA=silhouette(ZA, lab),
             corr=float(np.corrcoef(D0, DA)[0, 1]), own=float(own.mean()), moved=float(np.median(moved)),
             nn0=float(np.median(np.sort(np.linalg.norm(Z0[:, None] - Z0[None], axis=-1), axis=1)[:, 1])))
    print(f"{n} clips | PCA: PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%}")
    print(f"5-NN same-category share: z0 {m['knn0']:.2f} -> z_align {m['knnA']:.2f}")
    print(f"silhouette by category:  z0 {m['sil0']:+.2f} -> z_align {m['silA']:+.2f}")
    print(f"pairwise-distance correlation z0 vs z_align: {m['corr']:+.2f}")
    print(f"z_align whose nearest z0 is its own: {m['own']:.0%} | moved from z0 median {m['moved']:.2f} "
          f"(z0's nearest other z0, median {m['nn0']:.2f})")
    for c in CATS:
        s = lab == c
        w0 = np.linalg.norm(Z0[s][:, None] - Z0[s][None], axis=-1)[np.triu_indices(s.sum(), 1)].mean()
        wA = np.linalg.norm(ZA[s][:, None] - ZA[s][None], axis=-1)[np.triu_indices(s.sum(), 1)].mean()
        print(f"  {c:10s} n={s.sum():3d}  within-category distance {w0:5.2f} -> {wA:5.2f}   moved {np.median(moved[s]):.2f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True, sharey=True); fig.patch.set_facecolor(SURF)
    for ax, Y, ttl, k, s in ((axs[0], Y0, "z0 (adult latents)", m["knn0"], m["sil0"]),
                             (axs[1], YA, "after stage-2 L_align search on the child", m["knnA"], m["silA"])):
        for c in CATS:
            q = lab == c
            ax.scatter(Y[q, 0], Y[q, 1], s=26, marker=MRK[c], color=COL[c], edgecolor=SURF, lw=.6,
                       alpha=.9, label=f"{c} ({q.sum()})", zorder=3)
        ax.set_title(f"{ttl}\n5-NN same category {k:.2f}, silhouette {s:+.2f} (256-d)", fontsize=10.5, color=INK)
        ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)", color=INK2, fontsize=9.5)
    axs[0].set_ylabel(f"PC2 ({ev[1]:.0%} of variance)", color=INK2, fontsize=9.5)
    axs[1].legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5), labelcolor=INK)
    fig.suptitle(f"balanced500 on the child body, {n} clips: one PCA fit on z0 and z_align together | "
                 f"pairwise-distance correlation {m['corr']:+.2f}, moved {m['moved']:.2f} from z0 (median), "
                 f"nearest z0 is its own for {m['own']:.0%}", fontsize=10, color=INK)
    fig.tight_layout(); out = ROOT / "pca_align.png"
    fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight")
    json.dump(m, open(ROOT / "pca_align_metrics.json", "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
