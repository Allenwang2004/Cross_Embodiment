#!/usr/bin/env python3
"""multibody_corr_analyze.py -- analyses A-D of the multi-body correction experiment (scripts/multibody_corr.py).

All on the searched latents, no new rollouts. D(b, m, s) = 16 z*/|z*| - 16 z0/|z0| (Euclidean, |z| = 16) for body b,
clip m, seed s; Dbar(b, m) = seed mean; beta = the body's 8 parameters, ||dbeta|| Euclidean.
  A  noise floor vs body signal: cosine between corrections of the SAME clip
       seed pair  (same body, seeds 0 vs 1)          -> ceiling: how alike two searches of one cell are
       body pairs (different bodies, cross seeds)    -> as a function of ||dbeta||
       clip pairs (same body, different clips)       -> floor
  B  variance decomposition per clip: one-way ANOVA of D over bodies with the 2 seeds as replicates. ICC =
     share of correction variance that belongs to the body (rest = search noise); permutation p (shuffle the 64
     corrections across body labels). Also the share of |D|^2 carried by the clip's mean over bodies (a correction
     every body makes for that clip, adult included).
  C  dimensionality: body effects E_m = Dbar(b, m) - mean_b Dbar(b, m) (32 x 256 per clip) vs the matching seed-noise
     matrix N_m = (D(b, m, 0) - D(b, m, 1)) / 2 (same expected energy if there were no body signal). PCA spectra;
     body-signal energy; overlap of the top-k body-effect subspaces between clips (same kind / other kinds vs
     random k / 256); leave-one-clip-out capture of E_m by the other clips' pooled top-k.
  D  prediction from beta, leave-one-body-out per clip: predict D(b, m, 0) from the other 31 bodies by
     clip mean / nearest beta / linear in beta / RBF kernel in beta; reference: the cell's other seed D(b, m, 1).
     Score: |pred - D(b, m, 0)| / |D(b, m, 0)| (1 = no better than z0) and cosine.
Writes outputs/multibody_corr/analysis/{summary.json, figure_A.png}.
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from multibody_corr import ADULT, BODIES, GROUPS, OUT, SEEDS, beta, cell_dir, clips   # noqa: E402

AN = OUT / "analysis"
rng = np.random.default_rng(0)


def u16(v):
    v = np.asarray(v, float).reshape(-1)
    return 16 * v / np.linalg.norm(v)


def cos(a, b):
    """Cosine; NaN when either correction is zero (the search never beat z0: 21 of 1024 cells), so nanmean skips it."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def load():
    cs = clips()
    D = np.zeros((len(BODIES), len(cs), len(SEEDS), 256))
    c0 = np.zeros((len(BODIES), len(cs)))
    cb = np.zeros((len(BODIES), len(cs), len(SEEDS)))
    for j, c in enumerate(cs):
        z0 = u16(np.load(REPO / f"data/origin_z/{c}.npy"))
        for i, b in enumerate(BODIES):
            for k, s in enumerate(SEEDS):
                d = cell_dir(b, c, s)
                D[i, j, k] = u16(np.load(d / "best_z.npy")) - z0
                sm = json.load(open(d / "summary.json"))
                c0[i, j], cb[i, j, k] = sm["origin_z"]["cost"], sm["best"]["cost"]
    return cs, D, c0, cb


def analysis_A(cs, D, Bt):
    nb, nc = D.shape[:2]
    seed = np.array([[cos(D[i, j, 0], D[i, j, 1]) for j in range(nc)] for i in range(nb)])
    pairs = []                       # (clip, i, k, ||dbeta||, cos averaged over the two cross-seed pairings)
    for j in range(nc):
        for i in range(nb):
            for k in range(i):
                cc = np.nanmean([cos(D[i, j, 0], D[k, j, 1]), cos(D[i, j, 1], D[k, j, 0])])
                if np.isfinite(cc):
                    pairs.append((j, i, k, float(np.linalg.norm(Bt[i] - Bt[k])), cc))
    clip_pairs = [x for x in (cos(D[i, j, 0], D[i, l, 0]) for i in range(nb) for j in range(nc) for l in range(j)) if np.isfinite(x)]
    P = np.array([(p[3], p[4]) for p in pairs])
    bins = [0, 0.15, 0.3, 0.45, 0.6, 0.8, 2.0]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (P[:, 0] >= lo) & (P[:, 0] < hi)
        if sel.sum():
            per_clip = [np.mean([p[4] for p in pairs if p[0] == j and lo <= p[3] < hi]) for j in range(nc)
                        if any(p[0] == j and lo <= p[3] < hi for p in pairs)]
            bs = [np.mean(rng.choice(per_clip, len(per_clip))) for _ in range(2000)]
            rows.append(dict(lo=lo, hi=hi, n=int(sel.sum()), mean_cos=float(P[sel, 1].mean()),
                             ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
    # with the adult specifically
    a = BODIES.index(ADULT)
    adult = [(float(np.linalg.norm(Bt[i] - Bt[a])), float(np.nanmean([[cos(D[i, j, 0], D[a, j, 1]), cos(D[i, j, 1], D[a, j, 0])]
                                                                        for j in range(nc)]))) for i in range(nb) if i != a]
    rho = float(np.corrcoef(P[:, 0], P[:, 1])[0, 1])
    return dict(seed_pair_cos=float(np.nanmean(seed)), seed_pair_cos_per_clip=np.nanmean(seed, 0).tolist(),
                body_pair_bins=rows, body_pair_corr_with_dbeta=rho,
                clip_pair_cos=float(np.mean(clip_pairs)), adult_vs_body=adult), seed, pairs


def analysis_B(cs, D):
    nb, nc, ns, _ = D.shape
    out = []
    for j in range(nc):
        X = D[:, j]                                   # nb x ns x 256
        icc, F = _icc(X)
        null = []
        flat = X.reshape(nb * ns, -1)
        for _ in range(500):
            null.append(_icc(flat[rng.permutation(nb * ns)].reshape(nb, ns, -1))[0])
        mean_share = float(np.sum(X.mean((0, 1)) ** 2) / np.mean(np.sum(X ** 2, -1)))
        out.append(dict(clip=cs[j].split("/")[1], icc=icc, F=F, p=float((np.sum(np.array(null) >= icc) + 1) / (len(null) + 1)),
                        clip_mean_share=mean_share, mean_norm=float(np.linalg.norm(X, axis=-1).mean())))
    return out


def _icc(X):
    nb, ns = X.shape[:2]
    gm = X.mean((0, 1))
    bm = X.mean(1)
    ssb = ns * np.sum((bm - gm) ** 2)
    ssw = np.sum((X - bm[:, None]) ** 2)
    msb, msw = ssb / (nb - 1), ssw / (nb * (ns - 1))
    return float((msb - msw) / (msb + (ns - 1) * msw)), float(msb / msw)


def analysis_C(cs, D):
    nb, nc = D.shape[:2]
    E = D.mean(2) - D.mean(2).mean(0, keepdims=True)          # nb x nc x 256
    N = (D[:, :, 0] - D[:, :, 1]) / 2
    spec = []
    for j in range(nc):
        se = np.linalg.svd(E[:, j], compute_uv=False) ** 2
        sn = np.linalg.svd(N[:, j], compute_uv=False) ** 2
        spec.append(dict(clip=cs[j].split("/")[1], energy_E=float(se.sum()), energy_N=float(sn.sum()),
                         signal_share=float(max(0.0, 1 - sn.sum() * (nb - 1) / nb / se.sum())),
                         cum_E=(np.cumsum(se) / se.sum())[:10].tolist(), cum_N=(np.cumsum(sn) / sn.sum())[:10].tolist()))
    k = 4
    U = [np.linalg.svd(E[:, j], full_matrices=False)[2][:k] for j in range(nc)]
    kinds = [c.split("/")[0] for c in cs]
    same, other = [], []
    for j in range(nc):
        for l in range(j):
            ov = float(np.sum((U[j] @ U[l].T) ** 2) / k)
            (same if kinds[j] == kinds[l] else other).append(ov)
    cap = {}
    for kk in (4, 8, 16, 32):
        ce, cn = [], []
        for j in range(nc):
            P = np.concatenate([E[:, l] for l in range(nc) if l != j])
            V = np.linalg.svd(P, full_matrices=False)[2][:kk]
            ce.append(np.sum((E[:, j] @ V.T) ** 2) / np.sum(E[:, j] ** 2))
            cn.append(np.sum((N[:, j] @ V.T) ** 2) / np.sum(N[:, j] ** 2))
        cap[kk] = dict(body_effect=float(np.mean(ce)), seed_noise=float(np.mean(cn)), random=kk / 256)
    return dict(per_clip=spec, subspace_overlap_k4=dict(same_kind=float(np.mean(same)), other_kind=float(np.mean(other)),
                                                         random=k / 256), loco_capture=cap)


def analysis_D(cs, D, Bt):
    nb, nc = D.shape[:2]
    Db = D.mean(2)
    res = {m: [] for m in ("z0", "clip_mean", "nearest", "linear", "kernel", "other_seed")}
    cosr = {m: [] for m in res}
    X = Bt - 1.0
    for j in range(nc):
        for i in range(nb):
            tgt = D[i, j, 0]
            if np.linalg.norm(tgt) < 1e-9:
                continue
            o = [k for k in range(nb) if k != i]
            Y = Db[o, j]
            preds = {"z0": np.zeros(256), "clip_mean": Y.mean(0), "other_seed": D[i, j, 1],
                     "nearest": Y[int(np.argmin(np.linalg.norm(X[o] - X[i], axis=1)))]}
            A = np.c_[np.ones(len(o)), X[o]]
            W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(9), A.T @ Y)
            preds["linear"] = np.r_[1.0, X[i]] @ W
            d2 = np.sum((X[o] - X[i]) ** 2, 1)
            w = np.exp(-d2 / (2 * 0.15 ** 2)); w /= w.sum()
            preds["kernel"] = w @ Y
            for m, p in preds.items():
                res[m].append(np.linalg.norm(p - tgt) / np.linalg.norm(tgt))
                cosr[m].append(cos(p, tgt) if np.linalg.norm(p) > 1e-9 else 0.0)
    return {m: dict(rel_err_median=float(np.median(v)), rel_err_mean=float(np.mean(v)), cos_mean=float(np.nanmean(cosr[m])))
            for m, v in res.items()}


def figure(A, seed, pairs, B, C, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    P = np.array([(p[3], p[4]) for p in pairs])
    ax[0].scatter(P[:, 0], P[:, 1], s=3, alpha=0.15, color="#4a6fa5", label="body pairs (same clip)")
    for r in A["body_pair_bins"]:
        x = (r["lo"] + min(r["hi"], 1.2)) / 2
        ax[0].errorbar(x, r["mean_cos"], yerr=[[r["mean_cos"] - r["ci"][0]], [r["ci"][1] - r["mean_cos"]]], fmt="o", color="#1b2a41")
    ax[0].axhline(A["seed_pair_cos"], color="#c0504d", ls="--", label=f"same body, other seed ({A['seed_pair_cos']:.2f})")
    ax[0].axhline(A["clip_pair_cos"], color="#888", ls=":", label=f"same body, other clip ({A['clip_pair_cos']:.2f})")
    ax[0].set_xlabel("||Δβ|| between the two bodies"); ax[0].set_ylabel("cosine of corrections")
    ax[0].set_title("A. correction similarity vs body distance"); ax[0].legend(fontsize=8, loc="upper right")
    icc = [b["icc"] for b in B]
    ax[1].bar(range(len(B)), icc, color=["#4a6fa5" if b["p"] < 0.01 else "#bbb" for b in B])
    ax[1].set_xticks(range(len(B))); ax[1].set_xticklabels([b["clip"][:14] for b in B], rotation=70, fontsize=7)
    ax[1].set_ylabel("ICC (share of variance from the body)"); ax[1].set_title("B. body vs search noise, per clip")
    ax[1].axhline(0, color="k", lw=0.5)
    ks = np.arange(1, 11)
    ce = np.mean([c["cum_E"] for c in C["per_clip"]], 0); cn = np.mean([c["cum_N"] for c in C["per_clip"]], 0)
    ax[2].plot(ks, ce, "o-", color="#4a6fa5", label="body effects")
    ax[2].plot(ks, cn, "s--", color="#c0504d", label="seed noise")
    ax[2].set_xlabel("principal components (per clip, 32 bodies)"); ax[2].set_ylabel("cumulative variance")
    ax[2].set_title("C. how low-dimensional are body effects"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main():
    AN.mkdir(parents=True, exist_ok=True)
    cs, D, c0, cb = load()
    Bt = np.array([beta(b) for b in BODIES])
    A, seed, pairs = analysis_A(cs, D, Bt)
    B = analysis_B(cs, D)
    C = analysis_C(cs, D)
    Dres = analysis_D(cs, D, Bt)
    gain = 1 - cb.min(2) / c0
    out = dict(A=A, B=B, C=C, D=Dres,
               search=dict(best_over_z0_median=float(np.median(cb / c0[..., None])),
                           seed_gap_median=float(np.median(np.abs(cb[..., 0] - cb[..., 1]) / c0)),
                           gain_by_group={g: float(np.median(gain[[i for i, b in enumerate(BODIES) if GROUPS[b] == g]])) for g in ("grid", "axis", "named")},
                           norm_D_median=float(np.median(np.linalg.norm(D, axis=-1)))))
    (AN / "summary.json").write_text(json.dumps(out, indent=1))
    figure(A, seed, pairs, B, C, AN / "figure_A.png")
    print(json.dumps({k: v for k, v in out.items() if k != "B"}, indent=1)[:6000])
    print("B (per clip): " + " | ".join(f"{b['clip'][:16]} icc {b['icc']:.2f} p {b['p']:.3f} mean-share {b['clip_mean_share']:.2f}" for b in B))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def analysis_transfer():
    """Functional transfer (multibody_corr.py transfer): cost of every body's z* (both seeds) and of 16 random
    same-size directions, rolled out on every body, same clip, as a ratio to that body's z0 cost.
    Also checks that the own-z* rows reproduce the search's best cost."""
    from scipy.stats import spearmanr
    import csv as _csv
    Bt = {b: np.array(beta(b)) for b in BODIES}
    rows, repro = [], []
    for t in BODIES:
        for c in clips():
            d = OUT / "transfer" / t / c.split("/")[1]
            lab = json.load(open(d / "labels.json")); base = json.load(open(d / "eval_base.json"))["cost"]
            cost = [float(r["cost"]) for r in _csv.DictReader(open(d / "eval.csv"))]
            for l, x in zip(lab, cost):
                if l["kind"] == "random":
                    rows.append(dict(t=t, c=c, kind="random", r=x / base))
                else:
                    rows.append(dict(t=t, c=c, kind="own" if l["source"] == t else "other", s=l["source"], seed=l["seed"],
                                     d=float(np.linalg.norm(Bt[l["source"]] - Bt[t])), r=x / base))
                    if l["source"] == t:
                        sb = json.load(open(cell_dir(t, c, l["seed"]) / "summary.json"))["best"]["cost"]
                        repro.append(abs(x - sb))
    R = lambda sel: np.array([r["r"] for r in rows if sel(r)])
    own, rnd = R(lambda r: r["kind"] == "own"), R(lambda r: r["kind"] == "random")
    out = dict(reproduce_max_abs=float(max(repro)), own=dict(median=float(np.median(own)), better=float(np.mean(own < 1))),
               random=dict(median=float(np.median(rnd)), better=float(np.mean(rnd < 1))), bins=[])
    for lo, hi in [(0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 2.0)]:
        x = R(lambda r: r["kind"] == "other" and lo <= r["d"] < hi)
        if len(x):
            out["bins"].append(dict(lo=lo, hi=hi, n=int(len(x)), median=float(np.median(x)), better=float(np.mean(x < 1))))
    oth = [r for r in rows if r["kind"] == "other"]
    out["spearman_ratio_vs_dbeta"] = float(spearmanr([r["d"] for r in oth], [r["r"] for r in oth])[0])
    # per (target, clip): own best, best / nearest / adult source, random median
    cells = {}
    for r in rows:
        cells.setdefault((r["t"], r["c"]), []).append(r)
    best_src, near_src, adult_src, own_best, rnd_med = [], [], [], [], []
    for (t, c), rr in cells.items():
        own_best.append(min(r["r"] for r in rr if r["kind"] == "own"))
        o = [r for r in rr if r["kind"] == "other"]
        best_src.append(min(r["r"] for r in o))
        dmin = min(r["d"] for r in o if r["s"] not in ("ax3m_t1000", "ax7m_t1000", ADULT) or t in ("ax3m_t1000", "ax7m_t1000", ADULT))
        near_src.append(np.mean([r["r"] for r in o if r["d"] == dmin]))
        if t != ADULT:
            adult_src.append(np.mean([r["r"] for r in o if r["s"] == ADULT]))
        rnd_med.append(np.median([r["r"] for r in rr if r["kind"] == "random"]))
    out["per_cell"] = dict(own_best=float(np.median(own_best)), best_of_31_sources=float(np.median(best_src)),
                           nearest_source=float(np.median(near_src)), adult_source=float(np.median(adult_src)),
                           random=float(np.median(rnd_med)))
    kinds = sorted({c.split("/")[0] for c in clips()})
    out["by_kind"] = {k: dict(other=float(np.median(R(lambda r, k=k: r["kind"] == "other" and r["c"].startswith(k + "/")))),
                             random=float(np.median(R(lambda r, k=k: r["kind"] == "random" and r["c"].startswith(k + "/"))))) for k in kinds}
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "transfer":
    T = analysis_transfer()
    (AN / "transfer.json").write_text(json.dumps(T, indent=1))
    print(json.dumps(T, indent=1))


def analysis_E(at=(2, 4, 8, 16, 32, 64)):
    """Analysis E: 8-dim subspace searches (multibody_corr.py esearch) vs the full-space search of the same cell
    (search/<body>/<clip>_s0, same seed and recipe). Per cell, % of the FULL search's 64-generation gain
    (c0 - best_full) reached by generation g: 100 (c0 - best_arm(g)) / (c0 - best_full(64)). Cells where the full
    search gained < 2% of c0 are skipped (no gain to measure). Medians over cells; paired Wilcoxon vs rand8 / full."""
    from scipy.stats import wilcoxon
    from multibody_corr import E_ARMS, E_TEST

    def curve(d):
        b = {}
        for r in csv.DictReader(open(d / "curve.csv")):
            b[int(r["gen"])] = float(r["best_so_far"])
        g = max(b)
        return np.minimum.accumulate([b.get(i, b[min(b, key=lambda k: abs(k - i))]) for i in range(g + 1)])

    res, P = {}, {}
    for c in clips():
        st = c.split("/")[1]
        for b in E_TEST:
            fd = cell_dir(b, c, 0)
            sm = json.load(open(fd / "summary.json")); c0 = sm["origin_z"]["cost"]
            full = np.minimum(curve(fd), c0); gain = c0 - full[-1]
            if gain < 0.02 * c0:
                continue
            arms = {"full": full}
            for a in E_ARMS:
                d = OUT / "esearch" / a / b / f"{st}_s0"
                if (d / "summary.json").exists():
                    arms[a] = np.minimum(curve(d), c0)
            for a, cu in arms.items():
                P.setdefault(a, {})[(b, st)] = [100 * (c0 - cu[min(g - 1, len(cu) - 1)]) / gain for g in at]
    for a, cells in P.items():
        M = np.array(list(cells.values()))
        res[a] = dict(n=len(cells), median=[float(x) for x in np.median(M, 0)])
        for ref in ("rand8", "full"):
            if ref != a and ref in P:
                common = [k for k in cells if k in P[ref]]
                if len(common) >= 8:
                    X = np.array([cells[k] for k in common]); Y = np.array([P[ref][k] for k in common])
                    res[a][f"vs_{ref}"] = dict(n=len(common), win=[float(np.mean(X[:, i] > Y[:, i])) for i in range(len(at))],
                                               p=[float(wilcoxon(X[:, i], Y[:, i]).pvalue) if np.any(X[:, i] != Y[:, i]) else 1.0 for i in range(len(at))])
    return dict(generations=list(at), arms=res)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "E":
    import csv
    E = analysis_E()
    (AN / "E.json").write_text(json.dumps(E, indent=1))
    print("% of the full-space 64-generation gain reached at generations", E["generations"])
    for a, r in E["arms"].items():
        line = f"  {a:11s} n {r['n']:3d}  " + " / ".join(f"{x:5.1f}" for x in r["median"])
        for ref in ("rand8", "full"):
            if f"vs_{ref}" in r:
                v = r[f"vs_{ref}"]; line += f" | vs {ref} (n {v['n']}) win " + "/".join(f"{w:.0%}" for w in v["win"])
        print(line)
