"""Learn a map between the tracking embeddings of two DIFFERENT bodies.

The same AMASS clip is retargeted onto two skeletons, and `batch_infer_z.py`
runs `tracking_inference` separately on each. Because the env is built with
`--xml`, the proprio features -- and therefore z -- come off the body that
actually moved. This script asks whether the two bodies' z are related by a
fixed map:

    z_dst_t  ~=  project_z( unit(z_src_t) @ W )

Unlike `fit_z_map.py` (tracking z -> reward z0, SAME body, one z0 per clip),
both sides here are full (T, 256) trajectories of the SAME clip, so the pairing
is per FRAME and the sample count is ~131k rather than 540.

Four things are load-bearing and easy to get wrong:

  * Fit PER FRAME, not on the clip mean. Both trees are trajectories; collapsing
    each to its mean before fitting throws the signal away (held-out cosine
    0.89 -> 0.75). `fit_z_map.py` averages because ITS target is a single z0;
    that reasoning does not carry over.
  * Split over TASKS, not trials or frames. The 10 trials of a task are variants
    of one motion and consecutive frames are near-duplicates, so both a
    trial-level and a frame-level split leak. The effective sample size is 54
    tasks, not 131,700 frames.
  * Both trees must agree on T per clip. They do (the retarget preserves frame
    count), and the loader asserts it -- a silent off-by-one would pair frame t
    with frame t+k and quietly destroy the fit.
  * Metamotivo keeps latents on the radius-sqrt(d) sphere (fb/model.py:126), so
    cosine is the metric and predictions must be projected back onto it.

The map is NOT low-rank (rank 128 of 256 recovers only ~0.83 of the full-rank
0.89), and the two directions are NOT symmetric: the child latents occupy a
much lower-dimensional manifold, so src->dst -- mapping INTO the child -- is
markedly easier than dst->src. The default is the easy direction on purpose:
src=infer_origin_z, dst=child/infer_retargeting_z, because origin->child is what
deployment actually needs ("convert an adult embedding so the child body can
execute it"). So W_full is the map to deploy and W_full_reverse is the
child->origin direction, kept for analysis. Task-level held-out ridge cosine is
0.961 for origin->child against 0.894 for child->origin -- the asymmetry is the
point, and it is why the deployable direction is also the accurate one.
The rank sweep and the PCA spectrum in the outputs are what those claims rest on.

Usage (from project root):
    uv run scripts/fit_cross_body_z_map.py
    uv run scripts/fit_cross_body_z_map.py --src child/infer_retargeting_z \
        --dst infer_origin_z --n-folds 6      # the child->origin direction

Writes to outputs/fit_cross_body_z_map/ (override with --out-dir):
    W.npz          maps + the held-out task list, for a rollout eval
    results.json   every number behind the plots
    *.png          model ladder, per-task, rank sweep, PCA spectrum/scatter,
                   cosine histogram
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath

# Must precede the numpy import. The solves here are small (256x256); letting
# OpenBLAS fan out to its default thread count makes them dramatically SLOWER
# through thread thrashing. Same guard, same reason, as fit_z_map.py.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def project_z(v):
    """Back onto the radius-sqrt(d) sphere Metamotivo's actor expects."""
    v = np.asarray(v, dtype=np.float64)
    return unit(v) * np.sqrt(v.shape[-1])


def row_cosine(A, B):
    return (unit(A) * unit(B)).sum(axis=-1)


def apply_map(z, W):
    """Map a (…, d) latent through W and return it on the sqrt(d) sphere.
    W is fitted on unit vectors, so normalise first -- keep this in sync with
    whatever consumes W.npz."""
    return project_z(unit(np.asarray(z, dtype=np.float64)) @ W)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def tree_label(tree):
    """Short body label for a z-tree path, used to NAME the maps in W.npz.

        'child/infer_retargeting_z' -> 'child'    a body subdirectory names the body
        'infer_origin_z'            -> 'origin'   else strip the inference affixes

    The names exist so a consumer states a direction instead of a position:
    `W_origin_to_child` cannot be applied backwards by accident, `W_full` can.
    key_forward/key_reverse are written alongside, so nothing downstream has to
    reproduce this rule -- it only has to be stable and distinct.
    """
    parts = PurePosixPath(tree).parts
    if len(parts) > 1:
        return parts[0]
    s = parts[0]
    if s.startswith("infer_"):
        s = s[len("infer_"):]
    if s.endswith("_z"):
        s = s[:-len("_z")]
    return s or parts[0]


def family_of(task):
    """Coarse motion family, from this tree's naming convention. Clip length is
    a task property here and tracks the family exactly (locomotion 300 frames,
    static 120, jump 150), which is why the two are reported together."""
    if task.startswith(("crawl", "move-ego", "rotate")):
        return "locomotion"
    if task.startswith("jump"):
        return "jump"
    if task.startswith(("raisearms", "crouch", "headstand", "sitonground",
                        "split", "lieonground")):
        return "static"
    return "other"


def load_pairs(src_dir, dst_dir):
    """Frame-aligned (X, Y) over every clip present in BOTH trees.

    Returns unit vectors: the objective is cosine (scale invariant) but the
    ridge penalty is not -- keeping the sqrt(d)=16 radius would scale X.T@X by
    d and silently shift the meaningful lambda range by the same factor.
    """
    tasks = sorted(p.name for p in src_dir.iterdir() if p.is_dir())
    xs, ys, ftid, fcid, clip_len, clip_tid = [], [], [], [], [], []
    for ti, task in enumerate(tasks):
        for npy in sorted((src_dir / task).glob("*.npy")):
            dst_path = dst_dir / task / npy.name
            if not dst_path.exists():
                print(f"  warn: no dst for {task}/{npy.name}, skipping")
                continue
            a = np.load(npy).astype(np.float64)
            b = np.load(dst_path).astype(np.float64)
            if a.ndim == 1:
                a, b = a[None], b[None]
            if a.shape != b.shape:
                raise SystemExit(
                    f"{task}/{npy.name}: src {a.shape} != dst {b.shape}. The "
                    "per-frame pairing this script rests on is not valid.")
            ci = len(clip_len)
            xs.append(a); ys.append(b)
            ftid.append(np.full(len(a), ti)); fcid.append(np.full(len(a), ci))
            clip_len.append(len(a)); clip_tid.append(ti)
    if not xs:
        raise SystemExit(f"no overlapping clips between {src_dir} and {dst_dir}")
    return (unit(np.concatenate(xs)), unit(np.concatenate(ys)),
            np.concatenate(ftid), np.concatenate(fcid),
            np.array(clip_len), np.array(clip_tid), tasks)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def fit_ridge(X, Y, lam, w=None):
    d = X.shape[1]
    if w is None:
        A, B = X.T @ X, X.T @ Y
    else:
        Xw = X * w[:, None]
        A, B = Xw.T @ X, Xw.T @ Y
    return np.linalg.solve(A + lam * np.eye(d), B)


def fit_procrustes(X, Y):
    """Best ORTHOGONAL map -- rotation only, lengths preserved. The gap between
    this and ridge is the part of the body swap that is not a rigid rotation."""
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return U @ Vt


def fit_reduced_rank(X, Y, lam, r):
    """Reduced-rank ridge: ridge, then keep only the top-r directions of the
    predicted response. Sweeping r measures how many latent directions the body
    swap actually touches."""
    W = fit_ridge(X, Y, lam)
    P = X @ W
    _, _, Vt = np.linalg.svd(P.T @ P)
    V = Vt[:r].T
    return W @ V @ V.T


def fit_affine(X, Y, lam):
    """Ridge with a bias column. Returns (W, b)."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    Wa = np.linalg.solve(Xa.T @ Xa + lam * np.eye(X.shape[1] + 1), Xa.T @ Y)
    return Wa[:-1], Wa[-1]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def task_folds(n_tasks, k, seed):
    """Group k-fold over tasks: every task is held out exactly once, which is
    what makes the per-task numbers honest."""
    return np.array_split(np.random.default_rng(seed).permutation(n_tasks), k)


def run_cv(X, Y, ftid, wclip, tasks, folds, lam, lams, ranks, per_task=True):
    n = len(tasks)
    ladder = {m: [] for m in ["identity", "mean-shift", "procrustes", "ridge",
                              "affine", "ridge (clip-weighted)"]}
    rank_curve = {r: [] for r in ranks}
    lam_curve = {l: [] for l in lams}
    train_cos = []
    pt_id = np.zeros(n); pt_rg = np.zeros(n)
    hist_id, hist_rg = [], []

    for fold in folds:
        te_t = set(fold.tolist())
        te = np.array([t in te_t for t in ftid]); tr = ~te
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

        ladder["identity"].append(row_cosine(Xte, Yte).mean())
        # translate the src cloud onto the dst cloud; cosine renormalises, so
        # this isolates how much of the gap is a constant offset
        shift = Ytr.mean(0) - Xtr.mean(0)
        ladder["mean-shift"].append(row_cosine(Xte + shift, Yte).mean())
        ladder["procrustes"].append(
            row_cosine(Xte @ fit_procrustes(Xtr, Ytr), Yte).mean())

        for l in lams:
            lam_curve[l].append(row_cosine(Xte @ fit_ridge(Xtr, Ytr, l), Yte).mean())

        W = fit_ridge(Xtr, Ytr, lam)
        ladder["ridge"].append(row_cosine(Xte @ W, Yte).mean())
        train_cos.append(row_cosine(Xtr @ W, Ytr).mean())

        # 1/T weighting makes the 540 clips equal. It scores WORSE, partly
        # because the metric below is itself frame-averaged -- to use this
        # variant, weight the metric to match.
        Wc = fit_ridge(Xtr, Ytr, lam, w=wclip[tr])
        ladder["ridge (clip-weighted)"].append(row_cosine(Xte @ Wc, Yte).mean())

        Wa, ba = fit_affine(Xtr, Ytr, lam)
        ladder["affine"].append(row_cosine(Xte @ Wa + ba, Yte).mean())

        for r in ranks:
            rank_curve[r].append(
                row_cosine(Xte @ fit_reduced_rank(Xtr, Ytr, lam, r), Yte).mean())

        if per_task:
            ci = row_cosine(Xte, Yte); cr = row_cosine(Xte @ W, Yte)
            hist_id.append(ci); hist_rg.append(cr)
            sub = ftid[te]
            for ti in fold:
                m = sub == ti
                pt_id[ti] = ci[m].mean(); pt_rg[ti] = cr[m].mean()

    out = {
        "ladder": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                       "folds": [float(x) for x in v]} for k, v in ladder.items()},
        "ridge_train": float(np.mean(train_cos)),
        "lam_curve": {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                      for k, v in lam_curve.items()},
        "rank_curve": {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                       for k, v in rank_curve.items()},
    }
    if per_task:
        out["per_task"] = {"task": tasks,
                           "family": [family_of(t) for t in tasks],
                           "identity": pt_id.tolist(), "ridge": pt_rg.tolist()}
        out["_hist"] = (np.concatenate(hist_id), np.concatenate(hist_rg))
    return out


def pca_summary(A, keep=256):
    s = np.linalg.svd(A - A.mean(0), compute_uv=False)
    ev = s ** 2; ev /= ev.sum()
    cum = np.cumsum(ev)
    return {"cum": cum[:keep].tolist(),
            "participation_ratio": float(1.0 / np.sum(ev ** 2)),
            "dims_90pct": int(np.searchsorted(cum, 0.90)) + 1}


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
BASE, FIT, SRC_C, DST_C, MAP_C = "#9aa1b2", "#2a78d6", "#eb6834", "#2a78d6", "#1baf7a"


def plot_ladder(res, rev, floor, out):
    names = sorted(res["ladder"], key=lambda k: res["ladder"][k]["mean"])
    means = [res["ladder"][n]["mean"] for n in names]
    errs = [res["ladder"][n]["std"] for n in names]
    plt.figure(figsize=(9, 5))
    bars = plt.barh(names, means, xerr=errs, capsize=4,
                    color=[FIT if n == "ridge" else BASE for n in names], height=.62)
    for b, m, e in zip(bars, means, errs):
        plt.text(m + e + .012, b.get_y() + b.get_height() / 2, f"{m:.3f}",
                 va="center", fontsize=9)
    plt.axvline(res["ridge_train"], color="#555b6b", lw=1)
    plt.text(res["ridge_train"] - .01, -.45, f"ridge train {res['ridge_train']:.3f}",
             fontsize=8, color="#555b6b", ha="right")
    plt.axvline(floor, color="#c0392b", lw=1)
    plt.text(floor + .008, -.45, f"shuffled floor {floor:.3f}", fontsize=8, color="#c0392b")
    plt.xlim(0, 1.06)
    plt.xlabel("frame cosine on held-out tasks")
    title = f"src -> dst  ({res['ladder']['ridge']['mean']:.3f})"
    if rev is not None:
        title += f"   |   reverse dst -> src ridge = {rev['ladder']['ridge']['mean']:.3f}"
    plt.title(title)
    plt.grid(axis="x", ls="--", alpha=.35); plt.tight_layout()
    plt.savefig(out, dpi=200); plt.close()


def plot_per_task(res, out):
    pt = res["per_task"]
    order = np.argsort(pt["ridge"])
    t = [pt["task"][i] for i in order]
    a = np.array(pt["identity"])[order]; b = np.array(pt["ridge"])[order]
    y = np.arange(len(t))
    plt.figure(figsize=(9, max(6, .22 * len(t))))
    plt.hlines(y, np.minimum(a, b), np.maximum(a, b), color="#d8dce4", lw=2, zorder=1)
    plt.scatter(a, y, s=26, color=BASE, label="identity", zorder=2)
    plt.scatter(b, y, s=26, color=FIT, label="ridge", zorder=3)
    plt.yticks(y, t, fontsize=7)
    plt.xlabel("frame cosine (task held out)"); plt.xlim(0, 1.02)
    plt.title("per-task, identity vs fitted linear map")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(axis="x", ls="--", alpha=.35); plt.tight_layout()
    plt.savefig(out, dpi=200); plt.close()


def plot_rank(res, out):
    ranks = sorted(int(k) for k in res["rank_curve"])
    m = [res["rank_curve"][str(r)]["mean"] for r in ranks]
    e = [res["rank_curve"][str(r)]["std"] for r in ranks]
    full = res["ladder"]["ridge"]["mean"]
    plt.figure(figsize=(8, 4.6))
    plt.errorbar(ranks, m, yerr=e, marker="o", ms=5, color=FIT, capsize=3, lw=2)
    plt.axhline(full, color="#555b6b", lw=1)
    plt.text(ranks[0], full + .012, f"full rank {full:.3f}", fontsize=8, color="#555b6b")
    plt.xscale("log", base=2); plt.xticks(ranks, [str(r) for r in ranks], fontsize=7)
    plt.xlabel("rank r of W"); plt.ylabel("frame cosine on held-out tasks")
    plt.title("reduced-rank ridge: how many latent directions the body swap touches")
    plt.grid(ls="--", alpha=.35); plt.tight_layout()
    plt.savefig(out, dpi=200); plt.close()


def plot_spectrum(spec, src_name, dst_name, out):
    plt.figure(figsize=(8, 4.6))
    for key, c, nm in ((src_name, SRC_C, "src"), (dst_name, DST_C, "dst")):
        s = spec[key]
        plt.plot(np.arange(1, len(s["cum"]) + 1), s["cum"], color=c, lw=2,
                 label=f"{nm}: {key}  (PR {s['participation_ratio']:.1f})")
        plt.scatter([s["dims_90pct"]], [.9], color=c, s=44, zorder=3)
        plt.annotate(f"{s['dims_90pct']} dims", (s["dims_90pct"], .9), color=c,
                     textcoords="offset points", xytext=(6, -13), fontsize=9)
    plt.axhline(.9, color="#888", lw=1)
    plt.xlabel("number of principal components"); plt.ylabel("cumulative explained variance")
    plt.title("latent dimensionality of the two bodies")
    plt.legend(fontsize=8, loc="lower right"); plt.grid(ls="--", alpha=.35)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def plot_scatter(cm_src, cm_dst, cm_map, out):
    mu = cm_dst.mean(0)
    _, _, Vt = np.linalg.svd(cm_dst - mu)
    B = Vt[:2].T
    p = lambda A: (A - mu) @ B
    plt.figure(figsize=(7.6, 6.4))
    for A, c, nm in ((cm_src, SRC_C, "z src"), (cm_map, MAP_C, "z src @ W"),
                     (cm_dst, DST_C, "z dst")):
        q = p(A)
        plt.scatter(q[:, 0], q[:, 1], s=17, color=c, alpha=.6, edgecolors="none", label=nm)
    plt.xlabel("PC1 (dst latent basis)"); plt.ylabel("PC2")
    plt.title("clip-mean latents, projected onto the dst principal axes")
    plt.legend(fontsize=9); plt.grid(ls="--", alpha=.3)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def plot_hist(ci, cr, out):
    plt.figure(figsize=(8, 4.4))
    bins = np.linspace(-.2, 1., 61)
    plt.hist(ci, bins=bins, histtype="step", lw=2, color=BASE, label="identity")
    plt.hist(cr, bins=bins, histtype="step", lw=2, color=FIT, label="ridge")
    plt.xlabel("frame cosine"); plt.ylabel("frames")
    plt.title("distribution over every held-out frame")
    plt.legend(fontsize=9); plt.grid(ls="--", alpha=.35)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


# --------------------------------------------------------------------------
# refinement 1: temporal window
#
# The per-frame fit throws the time axis away, but tracking_inference builds
# each z by averaging B(s) over the next cfg.seq_length (8) frames, so
# neighbouring frames carry real information about what z_t should be. Stacking
# a +/-k window keeps the estimator LINEAR -- it just widens the input from
# d to d*(2k+1) -- so this is still one least-squares solve, not a new model
# class. It also softens the clip-boundary effect, where that averaging window
# has collapsed to a single frame.
# --------------------------------------------------------------------------
def window_cols(fcid, n_clips, k):
    """Index arrays for offsets -k..+k, clamped to each clip's own boundaries so
    a window never reads across into the neighbouring clip."""
    n = len(fcid)
    idx = np.arange(n)
    starts = np.searchsorted(fcid, np.arange(n_clips))
    start_of = starts[fcid]
    end_of = (np.append(starts[1:], n) - 1)[fcid]
    return [np.clip(idx + o, start_of, end_of) for o in range(-k, k + 1)]


def _stack(X, cols, rows):
    return np.concatenate([X[c[rows]] for c in cols], axis=1)


def gram_windowed(X, Y, cols, rows, chunk=20000):
    """Accumulate (X'X, X'Y) for the stacked window, in blocks.

    The stacked matrix is never materialised: at k=4 it would be
    131700 x 2304 float64 = 2.4 GB. Blocking keeps peak memory at
    chunk x d(2k+1). Returning the Gram rather than a fitted W makes the lambda
    sweep almost free -- only the solve is repeated, not the accumulation."""
    D = X.shape[1] * len(cols)
    A = np.zeros((D, D)); B = np.zeros((D, Y.shape[1]))
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        Xw = _stack(X, cols, r)
        A += Xw.T @ Xw
        B += Xw.T @ Y[r]
    return A, B


def fit_ridge_windowed(X, Y, cols, rows, lam, chunk=20000):
    A, B = gram_windowed(X, Y, cols, rows, chunk)
    return np.linalg.solve(A + lam * np.eye(len(A)), B)


def cosine_windowed(X, Y, cols, rows, W, chunk=20000):
    out = np.empty(len(rows))
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        out[i:i + len(r)] = row_cosine(_stack(X, cols, r) @ W, Y[r])
    return out


def run_window_sweep(X, Y, ftid, fcid, n_clips, folds, ks, lams):
    """Sweep the window half-width. Each k is scored at its own best lambda,
    chosen on the fold means -- a fixed lambda would be an unfair comparison,
    since the same penalty is relatively weaker as the input widens from d to
    d(2k+1). k=0 gets the same advantage, so the comparison across k is fair
    (and each number is a slight over-estimate in the same direction)."""
    res = {}
    for k in ks:
        cols = window_cols(fcid, n_clips, k)
        per_lam = {l: [] for l in lams}
        for fold in folds:
            te = np.isin(ftid, fold)
            tr_rows = np.flatnonzero(~te); te_rows = np.flatnonzero(te)
            A, B = gram_windowed(X, Y, cols, tr_rows)
            I = np.eye(len(A))
            for l in lams:
                W = np.linalg.solve(A + l * I, B)
                per_lam[l].append(cosine_windowed(X, Y, cols, te_rows, W).mean())
        best = max(lams, key=lambda l: np.mean(per_lam[l]))
        v = np.array(per_lam[best])
        res[k] = {"mean": float(v.mean()), "std": float(v.std()),
                  "lam": float(best), "n_features": int(X.shape[1] * len(cols)),
                  "by_lam": {str(l): float(np.mean(per_lam[l])) for l in lams}}
        print(f"    k={k:<2d} ({res[k]['n_features']:5d} features)  "
              f"{res[k]['mean']:.4f} +/- {res[k]['std']:.4f}   at lam={best:g}   "
              f"[{'  '.join(f'{l:g}:{np.mean(per_lam[l]):.4f}' for l in lams)}]")
    return res


# --------------------------------------------------------------------------
# refinement 2: per-task oracle
#
# Upper bound on what per-task CONDITIONING can buy over one shared map.
#
# Fitting an independent W per task does NOT answer this: 8 trials is 960-2400
# highly correlated frames for a 65k-parameter map, so it is data-starved and
# scores BELOW the global map (measured: 0.822 vs 0.895) -- which says nothing
# about whether per-task structure exists. Instead each task gets a shrunk
# CORRECTION on top of the global map,
#
#     W_task = W_global + Delta,   Delta = argmin ||X Delta - (Y - X W_global)||^2
#                                          + lam ||Delta||^2
#
# with lam swept up to where Delta vanishes. That makes it a true bound: at
# large lam it recovers the global map exactly, so the oracle can never lose.
# Deliberately optimistic -- trials of one task are near-duplicates and lam is
# picked on the held-out trials. If even this barely clears the global map, the
# residual is information the source latents do not carry, and no amount of
# conditioning or model capacity recovers it.
# --------------------------------------------------------------------------
def per_task_oracle(X, Y, ftid, fcid, clip_tid, tasks, folds, lam,
                    lams=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6), n_trial_folds=5):
    fold_of = {}
    fold_W = []
    for fi, fold in enumerate(folds):
        for ti in fold:
            fold_of[int(ti)] = fi
        tr = np.flatnonzero(~np.isin(ftid, fold))
        fold_W.append(fit_ridge(X[tr], Y[tr], lam))

    oracle, glob = np.zeros(len(tasks)), np.zeros(len(tasks))
    for ti in range(len(tasks)):
        clips = np.flatnonzero(clip_tid == ti)
        groups = np.array_split(clips, min(n_trial_folds, len(clips)))
        o_acc, g_acc = [], []
        for g in groups:
            te = np.flatnonzero(np.isin(fcid, g))
            tr = np.flatnonzero(np.isin(fcid, np.setdiff1d(clips, g)))
            if len(tr) == 0 or len(te) == 0:
                continue
            Wg = fold_W[fold_of[ti]]
            g_acc.append(row_cosine(X[te] @ Wg, Y[te]).mean())
            # correction on top of the global map, shrunk toward no-correction
            R = Y[tr] - X[tr] @ Wg
            A, Bm = X[tr].T @ X[tr], X[tr].T @ R
            I = np.eye(X.shape[1])
            o_acc.append(max(
                row_cosine(X[te] @ (Wg + np.linalg.solve(A + l * I, Bm)), Y[te]).mean()
                for l in lams))
        oracle[ti] = np.mean(o_acc); glob[ti] = np.mean(g_acc)
    return {"task": tasks, "family": [family_of(t) for t in tasks],
            "oracle": oracle.tolist(), "global": glob.tolist(),
            "oracle_mean": float(oracle.mean()), "global_mean": float(glob.mean())}


# --------------------------------------------------------------------------
# refinement 3: per-task correction bank
#
# Turns the oracle of refinement 2 into something usable. Each task gets a
# correction on top of the global map,
#
#     W_task = W_global + Delta_task
#
# fitted on that task's own trials with a single shared lambda. Unlike the
# oracle this is deployable, but only where the task is KNOWN -- which it is for
# every clip in these trees (it is the directory name) and is not for a new
# motion. On an unseen task there is no Delta and you get W_global back, so the
# bank buys accuracy on the 54 known motions, NOT better generalisation.
#
# One shared lambda rather than one per task: it is a single hyperparameter over
# 270 held-out evaluations, so picking it on those evaluations is only slightly
# optimistic, whereas 54 separately-tuned lambdas would not be.
# --------------------------------------------------------------------------
def fit_task_maps(X, Y, fcid, clip_tid, tasks, W_deploy, lam_global, lams,
                  n_trial_folds=5):
    """Per-task corrections, and an honest test of whether they are worth having.

    The comparison must hold the TRIALS out from both models. Scoring against a
    global map that was fitted on all frames (W_full) is meaningless: it trained
    on the very frames being scored, so it reads ~0.96 in-sample and no
    correction can appear to help. Instead, for each trial-fold f:

        W_shared^f  = ridge on every task's OTHER trials      (saw this task)
        W_task^f    = W_shared^f + Delta_task fitted on this task's other trials

    both scored on fold f. That isolates the one question a bank answers -- given
    8 trials of every task, does splitting the map per task beat sharing one?
    """
    d = X.shape[1]
    I = np.eye(d)
    n_clips = len(np.unique(fcid))

    # trial-level folds, applied inside every task
    fold_of_clip = np.zeros(n_clips, dtype=int)
    for ti in range(len(tasks)):
        for j, c in enumerate(np.flatnonzero(clip_tid == ti)):
            fold_of_clip[c] = j % n_trial_folds
    frame_fold = fold_of_clip[fcid]
    task_of_frame = clip_tid[fcid]

    shared, per_lam = [], {l: [] for l in lams}
    shared_t = np.zeros((n_trial_folds, len(tasks)))
    bank_t = {l: np.zeros((n_trial_folds, len(tasks))) for l in lams}
    for f in range(n_trial_folds):
        te_all = frame_fold == f
        Wg = fit_ridge(X[~te_all], Y[~te_all], lam_global)
        for ti in range(len(tasks)):
            te = np.flatnonzero(te_all & (task_of_frame == ti))
            tr = np.flatnonzero((~te_all) & (task_of_frame == ti))
            if not len(te) or not len(tr):
                continue
            c0 = row_cosine(X[te] @ Wg, Y[te]).mean()
            shared.append(c0); shared_t[f, ti] = c0
            R = Y[tr] - X[tr] @ Wg
            A, B = X[tr].T @ X[tr], X[tr].T @ R
            for l in lams:
                c = row_cosine(X[te] @ (Wg + np.linalg.solve(A + l * I, B)), Y[te]).mean()
                per_lam[l].append(c); bank_t[l][f, ti] = c
    best = max(lams, key=lambda l: np.mean(per_lam[l]))

    # deploy on top of the map that saw everything, using all 10 trials
    delta = np.zeros((len(tasks), d, d), dtype=np.float32)
    for ti in range(len(tasks)):
        allf = np.flatnonzero(task_of_frame == ti)
        R = Y[allf] - X[allf] @ W_deploy
        A, B = X[allf].T @ X[allf], X[allf].T @ R
        delta[ti] = np.linalg.solve(A + best * I, B).astype(np.float32)

    return {"delta": delta, "lam": float(best),
            "mean": float(np.mean(per_lam[best])),
            "shared_mean": float(np.mean(shared)),
            "per_task_shared": shared_t.mean(0).tolist(),
            "per_task_bank": bank_t[best].mean(0).tolist(),
            "by_lam": {str(l): float(np.mean(per_lam[l])) for l in lams}}


def plot_window(win, base, out):
    ks = sorted(win)
    m = [win[k]["mean"] for k in ks]; e = [win[k]["std"] for k in ks]
    plt.figure(figsize=(8, 4.6))
    plt.errorbar(ks, m, yerr=e, marker="o", ms=6, color=FIT, capsize=3, lw=2)
    plt.axhline(base, color="#555b6b", lw=1)
    plt.text(ks[0], base - .006, f"single frame {base:.3f}", fontsize=8,
             color="#555b6b", va="top")
    for k, v in zip(ks, m):
        plt.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8)
    plt.xticks(ks, [f"k={k}\n{win[k]['n_features']}d" for k in ks], fontsize=8)
    plt.xlabel("temporal window half-width (input is z[t-k .. t+k])")
    plt.ylabel("frame cosine on held-out tasks")
    plt.title("does the map improve if it can see neighbouring frames?")
    plt.grid(ls="--", alpha=.35); plt.tight_layout()
    plt.savefig(out, dpi=200); plt.close()


def plot_oracle(orc, out):
    o = np.array(orc["oracle"]); g = np.array(orc["global"])
    order = np.argsort(g)
    t = [orc["task"][i] for i in order]
    y = np.arange(len(t))
    plt.figure(figsize=(9, max(6, .22 * len(t))))
    plt.hlines(y, np.minimum(g, o)[order], np.maximum(g, o)[order],
               color="#d8dce4", lw=2, zorder=1)
    plt.scatter(g[order], y, s=26, color=FIT, label="global W (task held out)", zorder=2)
    plt.scatter(o[order], y, s=26, color=MAP_C, label="+ per-task correction", zorder=3)
    plt.yticks(y, t, fontsize=7)
    plt.xlabel("frame cosine on the same held-out trials"); plt.xlim(0, 1.02)
    plt.title(f"headroom above one shared map: "
              f"{orc['global_mean']:.3f} -> {orc['oracle_mean']:.3f} "
              f"({orc['oracle_mean'] - orc['global_mean']:+.3f})\n"
              f"per-task = global map + shrunk per-task correction")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(axis="x", ls="--", alpha=.35); plt.tight_layout()
    plt.savefig(out, dpi=200); plt.close()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--src", default="infer_origin_z",
                    help="tree of (T,256) z to map FROM, relative to --data-root")
    ap.add_argument("--dst", default="child/infer_retargeting_z",
                    help="tree of (T,256) z to map TO, relative to --data-root")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/<script name>")
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0, help="ridge lambda for the reported map")
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0, 10.0, 100.0], help="lambda sweep")
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    ap.add_argument("--no-reverse", action="store_true",
                    help="skip the dst -> src direction")
    ap.add_argument("--window-ks", type=int, nargs="*", default=[0, 1, 2, 4],
                    help="temporal window half-widths to sweep; k=0 is the "
                         "single-frame fit, kept so every k is compared at its "
                         "own best lambda. Empty to skip")
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the per-task oracle upper bound")
    ap.add_argument("--no-task-maps", action="store_true",
                    help="skip the deployable per-task correction bank")
    ap.add_argument("--task-map-lams", type=float, nargs="+",
                    default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
                    help="shrinkage sweep for the per-task corrections")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.is_absolute():
        root = ROOT / root
    src_dir, dst_dir = root / args.src, root / args.dst
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / Path(__file__).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y, ftid, fcid, clip_len, clip_tid, tasks = load_pairs(src_dir, dst_dir)
    n_tasks, d = len(tasks), X.shape[1]
    fams = [family_of(t) for t in tasks]
    print(f"{len(X)} frame pairs over {len(clip_len)} clips / {n_tasks} tasks, dim {d}")
    print(f"  src {src_dir}\n  dst {dst_dir}")
    lens = {int(L): int((clip_len == L).sum()) for L in sorted(set(clip_len.tolist()))}
    print(f"  clip lengths (T: n_clips): {lens}")

    wclip = (1.0 / clip_len)[fcid]
    folds = task_folds(n_tasks, args.n_folds, args.seed)
    ranks = [r for r in args.ranks if r <= d]

    print(f"\n{args.n_folds}-fold task-level CV (every task held out exactly once)")
    res = run_cv(X, Y, ftid, wclip, tasks, folds, args.lam, args.lams, ranks)
    hist_id, hist_rg = res.pop("_hist")

    rev = None
    if not args.no_reverse:
        rev = run_cv(Y, X, ftid, wclip, tasks, folds, args.lam, args.lams,
                     ranks, per_task=False)

    # chance level: how close two unrelated latents are
    rp = np.random.default_rng(args.seed + 1).permutation(len(X))
    floor = float(row_cosine(X[rp], Y).mean())

    for name, v in sorted(res["ladder"].items(), key=lambda kv: kv[1]["mean"]):
        ang = np.degrees(np.arccos(np.clip(v["mean"], -1, 1)))
        print(f"  {name:24s} {v['mean']:.4f} +/- {v['std']:.4f}   ({ang:4.1f} deg)")
    print(f"  {'ridge TRAIN':24s} {res['ridge_train']:.4f}")
    print(f"  {'shuffled floor':24s} {floor:.4f}")
    if rev:
        print(f"\n  reverse dst -> src ridge  {rev['ladder']['ridge']['mean']:.4f} "
              f"+/- {rev['ladder']['ridge']['std']:.4f}")

    # per-family, frame-weighted (clip length is constant within a task)
    nfr = np.array([clip_len[clip_tid == ti].sum() for ti in range(n_tasks)])
    pid = np.array(res["per_task"]["identity"]); prg = np.array(res["per_task"]["ridge"])
    fam_break = {}
    for f in sorted(set(fams)):
        m = np.array([x == f for x in fams]); w = nfr[m]
        fam_break[f] = {"identity": float((pid[m] * w).sum() / w.sum()),
                        "ridge": float((prg[m] * w).sum() / w.sum()),
                        "n_tasks": int(m.sum()), "n_frames": int(w.sum())}
    print("\n  by motion family (frame-weighted):")
    for f, v in fam_break.items():
        print(f"    {f:11s} {v['n_tasks']:2d} tasks  {v['n_frames']:6d} frames  "
              f"identity {v['identity']:.4f} -> ridge {v['ridge']:.4f}")
    n_improved = int((prg > pid).sum())
    print(f"  {n_improved}/{n_tasks} tasks improved; worst "
          f"{res['per_task']['task'][int(prg.argmin())]} {prg.min():.4f}")

    win = None
    if args.window_ks:
        print("\n  temporal window (still a single linear solve, wider input):")
        win = run_window_sweep(X, Y, ftid, fcid, len(clip_len), folds,
                               sorted(args.window_ks), args.lams)
        best_k = max(win, key=lambda k: win[k]["mean"])
        base = win[0]["mean"] if 0 in win else res["ladder"]["ridge"]["mean"]
        print(f"    best k={best_k}: {win[best_k]['mean']:.4f} "
              f"({win[best_k]['mean'] - base:+.4f} vs single frame {base:.4f})")

    orc = None
    if not args.no_oracle:
        print("\n  per-task oracle (upper bound on any shared map):")
        orc = per_task_oracle(X, Y, ftid, fcid, clip_tid, tasks, folds, args.lam)
        gap = orc["oracle_mean"] - orc["global_mean"]
        print(f"    global W {orc['global_mean']:.4f} -> per-task oracle "
              f"{orc['oracle_mean']:.4f}  (headroom {gap:+.4f})")
        print("    " + ("headroom is small: one shared map is already close to what "
                        "per-task\n    conditioning could achieve, so the residual is "
                        "mostly information the\n    source latents do not carry."
                        if gap < 0.03 else
                        "headroom is real: a shared map is leaving accuracy on the "
                        "table,\n    so conditioning or a richer model class is worth "
                        "trying."))

    spec = {args.src: pca_summary(X), args.dst: pca_summary(Y)}

    # Two maps, mirroring fit_z_map.py: one fitted on everything (for downstream
    # use on new motions) and one that never saw fold 0's tasks (so a rollout on
    # those tasks is a genuine held-out test).
    W_full = fit_ridge(X, Y, args.lam)
    held_out = sorted(folds[0].tolist())
    tr0 = np.array([t not in set(held_out) for t in ftid])
    W_heldout = fit_ridge(X[tr0], Y[tr0], args.lam)
    save = dict(W_full=W_full, W_heldout=W_heldout, lam=args.lam,
                src=args.src, dst=args.dst, split_seed=args.seed,
                tasks=np.array(tasks),
                held_out_tasks=np.array([tasks[i] for i in held_out]),
                train_tasks=np.array([tasks[i] for i in range(n_tasks)
                                      if i not in set(held_out)]),
                floor=floor)
    W_full_rev = fit_ridge(Y, X, args.lam) if rev else None
    if rev:
        save["W_full_reverse"] = W_full_rev

    # Directional aliases. W_full/W_full_reverse are POSITIONAL -- they only mean
    # something if you already know which tree was src, which is exactly the fact
    # that goes missing. These name the direction instead, so apply_z_map --key
    # states intent that cannot be silently inverted.
    lab_s, lab_d = tree_label(args.src), tree_label(args.dst)
    if lab_s == lab_d:
        raise SystemExit(f"--src {args.src} and --dst {args.dst} both label as "
                         f"{lab_s!r}; the maps would collide in W.npz")
    save["key_forward"] = f"W_{lab_s}_to_{lab_d}"
    save["key_reverse"] = f"W_{lab_d}_to_{lab_s}"
    save[save["key_forward"]] = W_full
    if rev:
        save[save["key_reverse"]] = W_full_rev
    np.savez(out_dir / "W.npz", **save)
    print(f"\n  W.npz maps: {save['key_forward']}"
          + (f", {save['key_reverse']}" if rev else ""))

    bank = None
    if not args.no_task_maps:
        print("\n  per-task correction bank (deployable; known tasks only):")
        bank = {"forward": fit_task_maps(X, Y, fcid, clip_tid, tasks, W_full,
                                         args.lam, args.task_map_lams)}
        if W_full_rev is not None:
            bank["reverse"] = fit_task_maps(Y, X, fcid, clip_tid, tasks, W_full_rev,
                                            args.lam, args.task_map_lams)
        store = {"tasks": np.array(tasks), "W_global": W_full}
        for name, b in bank.items():
            sfx = "" if name == "forward" else "_reverse"
            store[f"delta{sfx}"] = b["delta"]
            store[f"lam{sfx}"] = b["lam"]
            store[f"per_task_bank{sfx}"] = np.array(b["per_task_bank"])
            store[f"per_task_shared{sfx}"] = np.array(b["per_task_shared"])
            print(f"    {name:8s} one shared map {b['shared_mean']:.4f} -> per-task "
                  f"bank {b['mean']:.4f}  ({b['mean'] - b['shared_mean']:+.4f})  "
                  f"at shrinkage lam={b['lam']:g}")
        if W_full_rev is not None:
            store["W_global_reverse"] = W_full_rev
        store["key_forward"] = save["key_forward"]
        store["key_reverse"] = save["key_reverse"]
        np.savez(out_dir / "W_by_task.npz", **store)
        print(f"    -> {out_dir / 'W_by_task.npz'}  "
              f"({len(tasks)} corrections; unknown tasks fall back to W_global)")

    # All three clouds must sit on the SAME sphere or the scatter is unreadable:
    # X and Y are unit vectors, so their per-clip means have norm <= 1, while
    # apply_map returns radius sqrt(d)=16. project_z puts all three on radius 16.
    cm_src = project_z(np.stack([X[fcid == i].mean(0) for i in range(len(clip_len))]))
    cm_dst = project_z(np.stack([Y[fcid == i].mean(0) for i in range(len(clip_len))]))
    cm_map = apply_map(cm_src, W_full)

    plot_ladder(res, rev, floor, out_dir / "cosine_by_method.png")
    plot_per_task(res, out_dir / "per_task.png")
    plot_rank(res, out_dir / "rank_sweep.png")
    plot_spectrum(spec, args.src, args.dst, out_dir / "pca_spectrum.png")
    plot_scatter(cm_src, cm_dst, cm_map, out_dir / "pca_scatter.png")
    plot_hist(hist_id, hist_rg, out_dir / "cosine_hist.png")
    if win:
        plot_window(win, win[0]["mean"] if 0 in win else res["ladder"]["ridge"]["mean"],
                    out_dir / "window_sweep.png")
    if orc:
        plot_oracle(orc, out_dir / "per_task_oracle.png")

    payload = {
        "src": str(src_dir), "dst": str(dst_dir),
        "n_frames": int(len(X)), "n_clips": int(len(clip_len)), "n_tasks": n_tasks,
        "dim": int(d), "lam": args.lam, "n_folds": args.n_folds, "seed": args.seed,
        "clip_lengths": lens, "floor": floor,
        "forward": res, "reverse": rev,
        "family_breakdown": fam_break, "n_tasks_improved": n_improved,
        "window_sweep": {str(k): v for k, v in win.items()} if win else None,
        "per_task_oracle": orc,
        "task_map_bank": ({k: {kk: vv for kk, vv in v.items() if kk != "delta"}
                           for k, v in bank.items()} if bank else None),
        "spectra": spec,
        "held_out_tasks": [tasks[i] for i in held_out],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {out_dir}/")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
