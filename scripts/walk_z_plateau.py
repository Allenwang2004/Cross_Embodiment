#!/usr/bin/env python3
"""walk_z_plateau.py -- sample the low-cost plateau of one (body, clip) and
find its dimension and a basis.

scripts/probe_z_flat.py showed that the 20 seeds' best z's for one clip sit on
ONE connected plateau (cost stays at the floor along every great circle between
them) that is NOT isotropic (a random direction climbs). 20 points cannot tell
how many directions the plateau has, so this script collects hundreds more:

  1. SAMPLE  (--sampler trace, default) harvest the seeds' own ES traces:
     once the ES mean reaches the plateau it keeps drifting 50-60 deg along it
     at the floor (the gradient term holds it inside, the noise term explores
     the flat directions), so z_mean[g0:] of every seed, every --every gens, is
     a free plateau sample. Each is re-rolled out and kept if cost <= --thr.
     (--sampler walk) a run-and-tumble random walk with hard acceptance at
     --thr. Kept for reference; in 255 dims it drifts to the sublevel set's
     boundary within ~50 steps and then rejects ~90% of proposals, so it is a
     poor sampler -- an ES mean is the same walk with a restoring force.

  2. PCA  log-map every accepted point (and the seed bests) to the tangent
     space at their spherical mean, SVD, scree + participation ratio against
     a random-sphere baseline of the same spread.

  3. VALIDATE  what a scree cannot prove: for k in --ks, walk from the mean
     along random directions INSIDE span(U_k) and along random directions of
     the full tangent space, at increasing angles, and roll them out. The
     smallest k whose in-span curve stays at the floor while the random curve
     climbs is the plateau's dimension; U_k is its basis.

Usage:
    uv run scripts/walk_z_plateau.py --root outputs/single_z_seeds_s005_5k --clip move-ego-0-2_4 \
        --steps 600 --walkers 16 --thr 0.26
    uv run scripts/walk_z_plateau.py --root ... --clip ... --analyze-only     # redo 2+3 from saved samples

Writes under <root>/plateau_<clip>/:
    samples.npz   z (N,256), cost (N,), accepted (N,), walker (N,), step (N,)
    basis.npz     mean (256,), U (256,K), sv (K,), thr
    scree.png, validate.png, walk.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from single_z_search import device_arg, project_z, rollout

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- sphere helpers ---------------------------------------------------------

def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def tangent(v, u):
    """component of v orthogonal to unit vector u."""
    return v - (v @ u) * u


def step_along(z, d, angle):
    """move `angle` radians from z along unit tangent d, staying on z's sphere."""
    r = np.linalg.norm(z)
    return r * (np.cos(angle) * z / r + np.sin(angle) * d)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1)))


def log_map(Z, m):
    """tangent vectors at unit m whose exp-map gives each row of unit Z;
    |v| = angle in radians."""
    Zu = unit(Z)
    c = np.clip(Zu @ m, -1, 1)
    th = np.arccos(c)
    t = Zu - c[:, None] * m
    return t * (th / np.maximum(np.linalg.norm(t, axis=1), 1e-12))[:, None]


def spherical_mean(Z, iters=20):
    m = unit(unit(Z).mean(0))
    for _ in range(iters):
        v = log_map(Z, m).mean(0)
        if np.linalg.norm(v) < 1e-9:
            break
        m = unit(np.cos(np.linalg.norm(v)) * m + np.sin(np.linalg.norm(v)) * unit(v))
    return m


# --- scorer ---------------------------------------------------------------------

class Scorer:
    """rolls a batch of z's out on the body and scores them with the bfm loss
    exactly as single_z_search / probe_z_flat do."""

    def __init__(self, s0, device, metamotivo, envs):
        from humenv import make_humenv
        from metamotivo.fb_cpr.huggingface import FBcprModel
        from model import bfm_align
        from model.obs_scale import build_obs_multiplier
        from model.simple.config import ESConfig
        import mujoco

        self.bfm_align = bfm_align
        self.device, self.envs, self.steps = device, envs, s0["steps"]
        task, stem = s0["clip"].split("/")
        xml = Path(s0["xml"])
        ref = np.load(REPO_ROOT / "data" / s0["body"] / "retargeting_motion" / task / f"{stem}.npz")["qpos"]
        cfg = ESConfig(device=device)
        self.model = FBcprModel.from_pretrained(metamotivo).to(device); self.model.eval()
        self.obs_mul = build_obs_multiplier(xml, REPO_ROOT / "assets/robots/adult/robot.xml",
                                            mode=s0.get("obs_scale", "auto"), parts=cfg.obs_scale_parts, verbose=False)
        env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
        self.Bg = bfm_align.reference_embeddings(self.model, env1, ref, device, self.obs_mul)
        env1.close()
        self.env, _ = make_humenv(num_envs=envs, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")
        self.nv = mujoco.MjModel.from_xml_path(str(xml)).nv
        self.init_qpos = ref[0] if s0.get("init", "reference") == "reference" else None
        self.n_rollouts = 0

    def __call__(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        out = np.empty(len(Z))
        for b0 in range(0, len(Z), self.envs):
            zb = Z[b0:b0 + self.envs]
            pad = np.concatenate([zb, np.repeat(zb[-1:], self.envs - len(zb), axis=0)]) if len(zb) < self.envs else zb
            zt = torch.as_tensor(pad, dtype=torch.float32, device=self.device)
            _, o = rollout(self.model, self.env, zt, self.steps, self.device, self.obs_mul,
                           init_qpos=self.init_qpos, nv=self.nv, return_obs=True)
            out[b0:b0 + len(zb)] = self.bfm_align.batch_bfm_align(self.model, o[:len(zb)], self.Bg, self.device)
            self.n_rollouts += self.envs
        return out

    def close(self):
        self.env.close()


# --- loading -----------------------------------------------------------------------

def load_seed_bests(root, clip):
    runs = []
    for d in sorted(root.iterdir()):
        m = re.match(rf"^{re.escape(clip)}_(\w+?)_(s\d+)$", d.name)
        if m and (d / "best_z.npy").exists():
            runs.append((m.group(2), d))
    if not runs:
        raise SystemExit(f"no seed runs for {clip} under {root}")
    Z = np.stack([np.load(d / "best_z.npy").reshape(-1).astype(np.float64) for _, d in runs])
    s0 = json.loads((runs[0][1] / "summary.json").read_text())
    assert s0["objective"] == "bfm", "this probe scores with the bfm loss; the runs must be bfm runs"
    return [t for t, _ in runs], Z, s0


# --- 1. walk ---------------------------------------------------------------------------

def walk(score, Z0, args, out, rng):
    n = args.walkers
    r = np.linalg.norm(Z0[0])
    starts = np.concatenate([Z0] * (n // len(Z0) + 1))[:n]
    z = starts.copy()
    d = np.stack([unit(tangent(rng.standard_normal(256), unit(zi))) for zi in z])
    step = np.radians(args.step)
    jitter = np.radians(args.jitter)
    rec = dict(z=[], cost=[], accepted=[], walker=[], step=[])
    n_acc = 0
    t0 = time.time()
    for s in range(args.steps):
        prop = np.stack([step_along(z[i], d[i], step) for i in range(n)])
        c = score(prop)
        acc = c <= args.thr
        for i in range(n):
            rec["z"].append(prop[i]); rec["cost"].append(c[i]); rec["accepted"].append(acc[i])
            rec["walker"].append(i); rec["step"].append(s)
            zi_u = unit(prop[i]) if acc[i] else unit(z[i])
            if acc[i]:
                z[i] = prop[i]
                # keep going: transport d to the new point, mix in a little noise
                dn = unit(tangent(d[i], zi_u))
                d[i] = unit(tangent(np.cos(jitter) * dn + np.sin(jitter) * unit(tangent(rng.standard_normal(256), zi_u)), zi_u))
            else:
                d[i] = unit(tangent(rng.standard_normal(256), zi_u))
        n_acc += int(acc.sum())
        if (s + 1) % args.log_every == 0 or s == args.steps - 1:
            dist = np.array([angle_between(z[i], starts[i]) for i in range(n)])
            pair = np.array([angle_between(z[i], z[j]) for i in range(n) for j in range(i + 1, n)])
            el = time.time() - t0
            print(f"[{s + 1:4d}/{args.steps}] acc {n_acc / ((s + 1) * n):.2f}  cost now {c.mean():.3f} "
                  f"(acc'd {c[acc].mean() if acc.any() else float('nan'):.3f})  "
                  f"dist from start {dist.mean():.0f} [{dist.min():.0f}-{dist.max():.0f}] deg  "
                  f"pairwise {pair.mean():.0f} deg  {el / (s + 1):.1f}s/step", flush=True)
            np.savez(out / "samples.npz", **{k: np.asarray(v) for k, v in rec.items()}, thr=args.thr, starts=starts)
    return {k: np.asarray(v) for k, v in rec.items()}


def sample_traces(score, root, clip, tags, args):
    """z_mean tail of every seed's ES run, re-scored. Returns rec like walk()."""
    import csv
    Z, walker, step = [], [], []
    for i, t in enumerate(tags):
        d = root / f"{clip}_bfm_{t}"
        f = np.load(d / "z_trace.npz")
        zm = f["z_mean"].astype(np.float64)
        # first logged generation whose mean cost is on the plateau
        rows = [(int(r["gen"]), float(r["mean_z_cost"])) for r in csv.DictReader(open(d / "curve.csv")) if r["mean_z_cost"]]
        g0 = next((g for g, c in rows if c <= args.thr), None)
        if g0 is None:
            print(f"  {t}: mean never reached thr, skipped"); continue
        gens = np.arange(g0, len(zm), args.every)
        Z.append(zm[gens]); walker += [i] * len(gens); step += list(gens)
    Z = project_z(np.concatenate(Z))
    print(f"trace sampler: {len(Z)} z_mean points from {len(tags)} seeds (every {args.every} gens after reaching {args.thr}), re-scoring...")
    c = score(Z)
    acc = c <= args.thr
    print(f"  cost {c.mean():.3f} [{c.min():.3f}-{c.max():.3f}], {acc.sum()}/{len(c)} <= thr")
    return dict(z=Z, cost=c, accepted=acc, walker=np.array(walker), step=np.array(step))


# --- 2. pca ------------------------------------------------------------------------------

def pca(P, Zbest, rng, out, clip, thr):
    """tangent-space PCA of the plateau points at their spherical mean."""
    m = spherical_mean(P)
    V = log_map(P, m)                       # (N, 256) tangent vectors, |v| = angle
    ang = np.degrees(np.linalg.norm(V, axis=1))
    Vc = V - V.mean(0)
    _, sv, Vt = np.linalg.svd(Vc, full_matrices=False)
    U = Vt.T                                 # (256, K) principal directions in the tangent space
    pr = sv.sum() ** 2 / (sv ** 2).sum()
    # baseline: same N random points on the sphere with the same angles from m
    R = np.stack([step_along(m, unit(tangent(rng.standard_normal(256), m)), np.radians(a)) for a in ang])
    Vr = log_map(R, m); Vr -= Vr.mean(0)
    svr = np.linalg.svd(Vr, compute_uv=False)
    prr = svr.sum() ** 2 / (svr ** 2).sum()
    var = sv ** 2 / (sv ** 2).sum()
    cum = np.cumsum(var)
    kk = {q: int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.8, 0.9, 0.95)}
    best_ang = np.degrees(np.linalg.norm(log_map(Zbest, m), axis=1))
    print(f"\nPCA at the spherical mean of {len(P)} plateau points: angle from mean {ang.mean():.1f} "
          f"[{ang.min():.1f}-{ang.max():.1f}] deg (seed bests {best_ang.mean():.1f} deg)")
    print(f"  participation ratio {pr:.1f}  (random sphere, same spread: {prr:.1f})")
    print(f"  components for 50/80/90/95% variance: {kk[0.5]}/{kk[0.8]}/{kk[0.9]}/{kk[0.95]}")
    print(f"  top singular values: {np.round(sv[:12], 2)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    n_show = min(60, len(sv))
    ax[0].plot(np.arange(1, n_show + 1), sv[:n_show], "o-", ms=3, label="plateau samples")
    ax[0].plot(np.arange(1, n_show + 1), svr[:n_show], "s--", ms=3, c="gray", label="random sphere, same spread")
    ax[0].set_title("singular values (tangent space at the mean)"); ax[0].set_xlabel("index"); ax[0].legend()
    ax[1].plot(np.arange(1, n_show + 1), cum[:n_show], "o-", ms=3)
    ax[1].plot(np.arange(1, n_show + 1), np.cumsum(svr ** 2 / (svr ** 2).sum())[:n_show], "s--", ms=3, c="gray")
    for q in (0.8, 0.9, 0.95):
        ax[1].axhline(q, c="k", lw=.5, ls=":")
    ax[1].set_title("cumulative variance"); ax[1].set_xlabel("k"); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=.3)
    ax[2].hist(ang, bins=30, alpha=.7, label="walk samples")
    ax[2].hist(best_ang, bins=10, alpha=.7, label="seed bests")
    ax[2].set_title("angle from the plateau's spherical mean (deg)"); ax[2].legend()
    fig.suptitle(f"{clip}: {len(P)} plateau points (cost <= {thr})  --  participation ratio {pr:.1f} vs random {prr:.1f}  "
                 f"--  k for 80/90/95%: {kk[0.8]}/{kk[0.9]}/{kk[0.95]}")
    fig.tight_layout(); fig.savefig(out / "scree.png", dpi=130)
    return m, U, sv, ang


# --- 3. validate -----------------------------------------------------------------------

def validate(score, m, U, sv, r, args, rng, out, clip, floor):
    """cost vs angle from the mean, walking inside span(U_k) vs random."""
    angles = np.array(args.angles, dtype=float)
    fams = [f"k={k}" for k in args.ks] + ["random"]
    dirs = {}
    for k in args.ks:
        Uk = U[:, :k]
        dirs[f"k={k}"] = np.stack([unit(tangent(Uk @ rng.standard_normal(k), m)) for _ in range(args.ndir)])
    # bands: directions drawn only from components a..b -- flatness as a function of component index
    rank = int(np.sum(sv > sv[0] * 1e-6))
    for a, b in args.bands:
        if a > rank:
            print(f"  band c{a}-{b} skipped: beyond the PCA rank ({rank} = #points - 1)")
            continue
        Ub = U[:, a - 1:b]
        dirs[f"c{a}-{b}"] = np.stack([unit(tangent(Ub @ rng.standard_normal(Ub.shape[1]), m)) for _ in range(args.ndir)])
        fams.append(f"c{a}-{b}")
    dirs["random"] = np.stack([unit(tangent(rng.standard_normal(256), m)) for _ in range(args.ndir)])
    # the complement control: a random direction orthogonal to span(U_kmax)
    Ukm = U[:, :max(args.ks)]
    comp = []
    for _ in range(args.ndir):
        v = rng.standard_normal(256); v -= Ukm @ (Ukm.T @ v); comp.append(unit(tangent(v, m)))
    dirs[f"orth(k={max(args.ks)})"] = np.stack(comp); fams.append(f"orth(k={max(args.ks)})")
    allz, idx = [], []
    for f in fams:
        for j in range(args.ndir):
            for a in angles:
                allz.append(step_along(r * m, dirs[f][j], np.radians(a))); idx.append((f, j, a))
    allz.append(r * m); idx.append(("mean", 0, 0.0))
    print(f"\nvalidation: {len(fams)} families x {args.ndir} dirs x {len(angles)} angles = {len(allz)} rollouts")
    c = score(np.stack(allz))
    cost = {f: np.full((args.ndir, len(angles)), np.nan) for f in fams}
    for (f, j, a), v in zip(idx, c):
        if f == "mean":
            c_mean = v
        else:
            cost[f][j, list(angles).index(a)] = v
    print(f"  mean point cost {c_mean:.3f}   (floor {floor:.3f}, thr {args.thr})")
    print("  family        " + "".join(f"{a:>7.0f}d" for a in angles))
    for f in fams:
        print(f"  {f:13s} " + "".join(f"{v:8.3f}" for v in cost[f].mean(0)) + f"   frac<=thr {np.mean(cost[f] <= args.thr):.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for f in fams:
        mu, sd = cost[f].mean(0), cost[f].std(0)
        ls = "--" if f.startswith(("random", "orth")) else (":" if f.startswith("c") else "-")
        ax.errorbar(angles, mu, yerr=sd, marker="o", ms=4, lw=1.5, ls=ls, capsize=3, label=f)
    ax.axhline(args.thr, c="k", lw=.7, ls=":", label=f"thr {args.thr}")
    ax.axhline(floor, c="gray", lw=.7, ls=":", label=f"seed floor {floor:.3f}")
    ax.set_xlabel("angle from the plateau mean (deg)"); ax.set_ylabel("bfm cost  1 - mean cos(B(s),B(g))")
    ax.set_title(f"{clip}: walking from the mean inside span(U_k) vs random directions")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "validate.png", dpi=130)
    np.savez(out / "validate.npz", angles=angles, **{f: cost[f] for f in fams}, mean_cost=c_mean)
    band_figure(cost, angles, args, out, clip, floor)
    return cost


def crossing_angle(angles, c, thr):
    """first angle at which a cost curve exceeds thr (linear interpolation);
    the last angle if it never does."""
    over = np.where(c > thr)[0]
    if len(over) == 0:
        return angles[-1]
    i = over[0]
    if i == 0:
        return angles[0]
    a0, a1, c0, c1 = angles[i - 1], angles[i], c[i - 1], c[i]
    return a0 + (a1 - a0) * (thr - c0) / max(c1 - c0, 1e-9)


def band_figure(cost, angles, args, out, clip, floor):
    """the plateau's extent as a function of component index: for every
    --bands group, how far one can walk inside it before leaving the plateau,
    and the cost at a few fixed angles. A step in either curve is the edge of
    the low-dimensional subspace."""
    bands = [(a, b) for a, b in args.bands if f"c{a}-{b}" in cost]
    if not bands:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mid = np.array([(a + b) / 2 for a, b in bands])
    width = np.array([b - a + 1 for a, b in bands])
    rad = np.array([[crossing_angle(angles, row, args.thr) for row in cost[f"c{a}-{b}"]] for a, b in bands])
    rad_rand = np.array([crossing_angle(angles, row, args.thr) for row in cost["random"]])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].errorbar(mid, rad.mean(1), yerr=rad.std(1), xerr=width / 2, marker="o", ms=4, capsize=3, lw=1.5, label="within the band")
    ax[0].axhspan(rad_rand.mean() - rad_rand.std(), rad_rand.mean() + rad_rand.std(), color="gray", alpha=.2)
    ax[0].axhline(rad_rand.mean(), c="gray", ls="--", lw=1, label=f"random direction ({rad_rand.mean():.0f} deg)")
    ax[0].set_xlabel("PCA component index (band)"); ax[0].set_ylabel(f"angle at which cost exceeds {args.thr} (deg)")
    ax[0].set_title("plateau radius per direction group"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_xscale("log"); ax[0].set_ylim(0, angles[-1] * 1.05)
    show = [a for a in (30, 40, 50) if a in list(angles)] or list(angles[len(angles) // 3: 2 * len(angles) // 3 + 1])
    for a in show:
        j = list(angles).index(a)
        mu = np.array([cost[f"c{x}-{y}"][:, j].mean() for x, y in bands])
        sd = np.array([cost[f"c{x}-{y}"][:, j].std() for x, y in bands])
        ax[1].errorbar(mid, mu, yerr=sd, marker="o", ms=4, capsize=3, lw=1.5, label=f"walk {a:.0f} deg")
        ax[1].axhline(cost["random"][:, j].mean(), ls="--", lw=.8, c=ax[1].lines[-1].get_color())
    ax[1].axhline(args.thr, c="k", ls=":", lw=.8, label=f"thr {args.thr}")
    ax[1].axhline(floor, c="gray", ls=":", lw=.8, label=f"floor {floor:.3f}")
    ax[1].set_xlabel("PCA component index (band)"); ax[1].set_ylabel("bfm cost"); ax[1].set_xscale("log")
    ax[1].set_title("cost after walking a fixed angle inside the band (dashed: random direction)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle(f"{clip}: flatness as a function of PCA component index  --  {args.ndir} directions per band")
    fig.tight_layout(); fig.savefig(out / "bands.png", dpi=130)
    print("\n  band        radius(deg)   " + "".join(f"c@{a:.0f}d " for a in show))
    for (a, b), r in zip(bands, rad):
        print(f"  c{a:3d}-{b:3d}    {r.mean():5.1f} +- {r.std():4.1f}   " + "".join(f"{cost[f'c{a}-{b}'][:, list(angles).index(x)].mean():6.3f} " for x in show))
    print(f"  random       {rad_rand.mean():5.1f} +- {rad_rand.std():4.1f}   " + "".join(f"{cost['random'][:, list(angles).index(x)].mean():6.3f} " for x in show))


def walk_figure(rec, starts, out, clip, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = int(rec["walker"].max()) + 1
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for i in range(n):
        sel = rec["walker"] == i
        acc = sel & rec["accepted"]
        Zi = rec["z"][acc]
        if len(Zi) == 0:
            continue
        ax[0].plot(rec["step"][acc], [angle_between(z, Zi[0]) for z in Zi], lw=.8)
        ax[1].plot(rec["step"][acc], rec["cost"][acc], lw=.6, alpha=.7)
    ax[0].set_title("angle from the first plateau point of each seed/walker"); ax[0].set_xlabel("step"); ax[0].set_ylabel("deg"); ax[0].grid(alpha=.3)
    ax[1].axhline(thr, c="k", ls=":", lw=.7); ax[1].set_title("cost of accepted points"); ax[1].set_xlabel("step"); ax[1].grid(alpha=.3)
    ax[2].hist(rec["cost"][rec["accepted"]], bins=40, alpha=.7, label="accepted")
    ax[2].hist(np.clip(rec["cost"][~rec["accepted"]], 0, 1.2), bins=40, alpha=.7, label="rejected")
    ax[2].axvline(thr, c="k", ls=":", lw=.7); ax[2].set_title("proposal costs"); ax[2].legend()
    fig.suptitle(f"{clip}: {n} seeds/walkers, {rec['accepted'].sum()} of {len(rec['z'])} samples <= thr")
    fig.tight_layout(); fig.savefig(out / "walk.png", dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="outputs/single_z_seeds_s005_5k")
    p.add_argument("--clip", required=True, help="stem, e.g. move-ego-0-2_4")
    p.add_argument("--thr", type=float, default=None, help="cost <= thr counts as on the plateau (default: floor * --thr-rel)")
    p.add_argument("--thr-rel", type=float, default=1.1, help="thr as a multiple of the plateau level, used when --thr is not given")
    p.add_argument("--thr-base", choices=["mean", "best"], default="mean",
                   help="plateau level = 'mean': median cost of the ES mean z over the last quarter of every seed run "
                        "(where ES settles; the best sample sits 5-15%% below it); 'best': mean of the seeds' best costs")
    p.add_argument("--sampler", choices=["trace", "walk"], default="trace")
    p.add_argument("--every", type=int, default=4, help="trace sampler: take z_mean every N gens after it first reaches thr")
    p.add_argument("--walkers", type=int, default=16)
    p.add_argument("--steps", type=int, default=600, help="proposals per walker")
    p.add_argument("--step", type=float, default=8.0, help="step angle, deg")
    p.add_argument("--jitter", type=float, default=15.0, help="deg of isotropic noise mixed into the kept direction after an accept")
    p.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--angles", type=float, nargs="+", default=[10, 20, 30, 45, 60, 75, 90])
    p.add_argument("--ndir", type=int, default=8, help="random directions per family in validation")
    p.add_argument("--bands", type=lambda t: tuple(int(x) for x in t.split("-")), nargs="*",
                   default=[(1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 255)],
                   help="component ranges a-b (1-based, inclusive) to test as their own families")
    p.add_argument("--analyze-only", action="store_true", help="skip the walk; reuse samples.npz")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=device_arg, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metamotivo", default="facebook/metamotivo-M-1")
    p.add_argument("--envs", type=int, default=16)
    args = p.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    out = root / f"plateau_{args.clip}"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tags, Zbest, s0 = load_seed_bests(root, args.clip)
    r = np.linalg.norm(Zbest[0])
    import csv as _csv
    floor = float(np.mean([json.loads((root / f"{args.clip}_bfm_{t}" / "summary.json").read_text())["best"]["cost"] for t in tags]))
    settle = []
    for t in tags:
        rows = [float(r["mean_z_cost"]) for r in _csv.DictReader(open(root / f"{args.clip}_bfm_{t}" / "curve.csv")) if r["mean_z_cost"]]
        settle += rows[-max(1, len(rows) // 4):]
    settle = float(np.median(settle))
    if args.thr is None:
        args.thr = round((settle if args.thr_base == "mean" else floor) * args.thr_rel, 4)
    print(f"{args.clip} on {s0['body']}: {len(Zbest)} seed bests, best floor {floor:.4f}, ES-mean settle level {settle:.4f}, thr {args.thr}")

    score = Scorer(s0, args.device, args.metamotivo, args.envs)
    t0 = time.time()
    if args.analyze_only:
        f = np.load(out / "samples.npz")
        rec = {k: f[k] for k in ("z", "cost", "accepted", "walker", "step")}
        starts = f["starts"]
        print(f"loaded {len(rec['z'])} samples ({rec['accepted'].sum()} accepted)")
    elif args.sampler == "trace":
        rec = sample_traces(score, root, args.clip, tags, args)
        starts = Zbest
        np.savez(out / "samples.npz", **rec, thr=args.thr, starts=starts)
    else:
        rec = walk(score, Zbest, args, out, rng)
        starts = np.concatenate([Zbest] * (args.walkers // len(Zbest) + 1))[:args.walkers]
        print(f"walk done: {rec['accepted'].sum()}/{len(rec['z'])} accepted in {(time.time() - t0) / 60:.1f} min")
    walk_figure(rec, starts, out, args.clip, args.thr)

    P = np.concatenate([Zbest, rec["z"][rec["accepted"]]])
    m, U, sv, ang = pca(P, Zbest, rng, out, args.clip, args.thr)
    np.savez(out / "basis.npz", mean=m, U=U, sv=sv, thr=args.thr, radius=r, n_points=len(P))
    validate(score, m, U, sv, r, args, rng, out, args.clip, floor)
    score.close()
    print(f"\n{score.n_rollouts} rollouts, {(time.time() - t0) / 60:.1f} min\n-> {out}/")


if __name__ == "__main__":
    main()
