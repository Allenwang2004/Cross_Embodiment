#!/usr/bin/env python3
"""build_correction_transfer_jobs.py -- move a latent CORRECTION, not a latent.

The old cross-motion test (cross_clip_z_transfer.py) rolled clip A's whole
searched latent z*_A on clip B. That throws away B's own z0, i.e. it asks B to
perform A's motion, so its failure says nothing. Along the body axis the same
operation was fair only because z0 is shared by every body for a given motion.

Here the thing transferred is the correction: the step from z0_A to z*_A. On the
sphere that step is an angle theta_A along a unit tangent direction u_A at z0_A.
It is parallel-transported along the great circle from z0_A to z0_B and taken
from z0_B, scaled by alpha:

    z = cos(alpha * theta_A) * z0_B + sin(alpha * theta_A) * transport(u_A)

For the body axis z0_A == z0_B, so transport is the identity and alpha = 1
reproduces the neighbour's own z* exactly.

Sets (--set):
  motion    8 motions on the child end of the m2c path (m2c_t1000), every
            ordered pair A != B, corrections from the cont and indep searches
  headstand 10 headstand trials on child (outputs/single_z_floor)
  walk      10 move-ego-0-2 trials on child (outputs/single_z_floor)
  body      every adjacent pair of bodies on the four morph paths, both
            directions, cont and indep corrections

--random: instead of the source's direction, a random tangent direction at the
target's z0, taken at the source's full angle (one draw per transfer). The control
for "a correction carries information": if a random step of the same length does
as well, what transferred was only the step length.

Labels:  <set>|<method>|<source>|a<alpha>   (|rand with --random)   plus, for the motion sets,
         <set>|<method>|<source>|abs  (the old test: z*_A as is) and
         own|<method>  (B's own z*, the ceiling).
"""
import argparse
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
ALPHAS = (0.25, 0.5, 1.0)
R = 16.0
HEADSTAND = [0, 2, 3, 4, 9, 15, 20, 23, 30, 34]      # the clips of the old 10x10 matrix


def unit(v):
    return v / np.linalg.norm(v)


def transport(z0a, zsa, z0b, alpha):
    p, s, q = unit(z0a), unit(zsa), unit(z0b)
    th = np.arccos(np.clip(p @ s, -1, 1))
    v = s - (p @ s) * p
    if np.linalg.norm(v) < 1e-9:
        return R * q
    u = v / np.linalg.norm(v)
    # parallel transport of the tangent vector u from p to q along their great circle
    u2 = u - (u @ q) / (1.0 + p @ q) * (p + q)
    return R * (np.cos(alpha * th) * q + np.sin(alpha * th) * unit(u2))


RNG = np.random.default_rng(0)


def random_step(z0a, zsa, z0b):
    p, s, q = unit(z0a), unit(zsa), unit(z0b)
    th = np.arccos(np.clip(p @ s, -1, 1))
    r = RNG.normal(size=q.shape); r = unit(r - (r @ q) * q)
    return R * (np.cos(th) * q + np.sin(th) * r)


def z0_of(stem):
    task = stem.rsplit("_", 1)[0]
    return np.load(REPO / "data/origin_z" / task / f"{stem}.npy").reshape(-1).astype(np.float64)


def zs_of(d):
    return np.load(d / "best_z.npy").reshape(-1).astype(np.float64)


ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["motion", "headstand", "walk", "body"])
ap.add_argument("--out", required=True)
ap.add_argument("--random", action="store_true")
a = ap.parse_args()
C, B, L, Z = [], [], [], []


def add(stem, body, label, z):
    C.append(f"{stem.rsplit('_', 1)[0]}/{stem}"); B.append(body); L.append(label); Z.append(z)


def motion_pairs(tag, src, body):
    """src: {method: {stem: dir with best_z.npy}}"""
    for m, dirs in src.items():
        for sb, db in dirs.items():
            if not a.random:
                add(sb, body, f"own|{m}", zs_of(db))
            for sa, da in dirs.items():
                if sa == sb:
                    continue
                za, zsa, zb = z0_of(sa), zs_of(da), z0_of(sb)
                if a.random:
                    add(sb, body, f"{tag}|{m}|{sa}|rand", random_step(za, zsa, zb))
                    continue
                add(sb, body, f"{tag}|{m}|{sa}|abs", zsa)
                for al in ALPHAS:
                    add(sb, body, f"{tag}|{m}|{sa}|a{al}", transport(za, zsa, zb, al))


if a.set == "motion":
    root = REPO / "outputs/continuation/m2c"
    stems = sorted(p.name.split("__")[0] for p in (root / "cont").glob("*__m2c_t1000"))
    motion_pairs("motion", {m: {s: root / m / f"{s}__m2c_t1000" for s in stems} for m in ("cont", "indep")},
                 "m2c_t1000")
elif a.set in ("headstand", "walk"):
    ks = HEADSTAND if a.set == "headstand" else range(10)
    task = "headstand" if a.set == "headstand" else "move-ego-0-2"
    dirs = {f"{task}_{k}": REPO / f"outputs/single_z_floor/{task}_{k}_bfm_s0" for k in ks}
    dirs = {s: d for s, d in dirs.items() if (d / "best_z.npy").exists()}
    motion_pairs(a.set, {"floor": dirs}, "child")
else:
    for prefix in ("m2c", "m2s", "m2g", "m2k"):
        root = REPO / "outputs/continuation" / prefix
        bodies = sorted({p.name.split("__")[1] for p in (root / "cont").iterdir()},
                        key=lambda b: int(b.rsplit("_t", 1)[1]))
        stems = sorted({p.name.split("__")[0] for p in (root / "cont").iterdir()})
        for m in ("cont", "indep"):
            for s in stems:
                z0 = z0_of(s)
                for i, bi in enumerate(bodies):
                    d = root / m / f"{s}__{bi}"
                    if not (d / "best_z.npy").exists():
                        continue
                    for j in (i - 1, i + 1):
                        if 0 <= j < len(bodies):
                            if a.random:
                                add(s, bodies[j], f"body|{m}|{bi}|rand", random_step(z0, zs_of(d), z0))
                                continue
                            for al in ALPHAS:
                                add(s, bodies[j], f"body|{m}|{bi}|a{al}", transport(z0, zs_of(d), z0, al))
np.savez(REPO / a.out, clip=np.array(C), body=np.array(B), label=np.array(L), z=np.stack(Z))
print(f"{a.set}: {len(C)} jobs")
