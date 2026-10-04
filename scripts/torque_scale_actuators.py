#!/usr/bin/env python3
"""torque_scale_actuators.py — write a child MJCF whose actuators are scaled by the
per-joint torque ratio k measured against the adult.

The child's actuators are byte-identical to the adult's (every gainprm /
biasprm / forcerange ratio is exactly 1.0000), so the child carries adult-sized
motors on a 38 kg body. This applies k from
outputs/torque_ratio_per_joint/<clip>/_ratios.csv, so each actuator's torque authority
matches that joint's measured demand ratio.

All four affine terms scale, biasprm[0] included
------------------------------------------------
The actuator is an affine position servo:

    force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*length + biasprm[2]*velocity

For the scaled actuator to deliver exactly k times the force in every state,
ALL FOUR coefficients must be multiplied by k. Then the equilibrium angle
    q* = -(gainprm[0]*ctrl + biasprm[0]) / biasprm[1]
is unchanged, so ctrl keeps its meaning and only the torque authority moves.
Leaving biasprm[0] alone (as utils/actuator.py:apply_actuator_scale does)
instead shifts the neutral angle by biasprm[0]*(1/k - 1) -- at k=0.21 that is a
~4.7x amplification of an offset that reaches 262 N*m on this model, so the body
would sag into a different rest pose. forcerange scales too: it is the torque
ceiling, and the point of the exercise is to move it.

Which k, and the two joints that cannot use the fitted one
----------------------------------------------------------
k_fitted is trustworthy only where the fit had signal. Two joints have almost no
gravity torque all clip, so their least-squares k is noise: Torso_y (R^2 0.878)
and Chest_y (R^2 0.003, k_fitted = -0.057). A NEGATIVE gain would inverse the
position servo into positive feedback and the model would diverge on contact, so
those fall back to k_predicted_subtree -- the physical prediction (downstream
subtree mass x lever arm) read from the two MJCFs, which is positive for every
joint by construction. --r2-min controls the threshold.

Left and right get the SAME k
-----------------------------
The fit is per actuator, so L_Shoulder_y and R_Shoulder_y come out at 0.2161 and
0.2122 on move-ego-90 -- a 1.8% difference that is entirely an artefact of the
clip turning one way. The body itself is symmetric: k_predicted_subtree, which
is read from the two MJCFs and knows nothing about any motion, agrees left/right
to 0.026% on every pair. Letting the clip's bias through would size the two legs'
motors differently for no physical reason, so --symmetry pair (the default) gives
each of the 27 mirrored pairs one shared k; the 15 midline actuators
(Torso/Spine/Chest/Neck/Head) have no partner and keep their own.

  both sides cleared --r2-min   ->  mean of the two fitted k
  one side cleared it           ->  that side's k (the other is noise)
  neither                       ->  mean of the two fallbacks

Scaling all four affine terms preserves the mirror exactly. The two MJCFs are
mirror images rather than copies -- gainprm, biasprm[1..2] and forcerange are
identical L/R while biasprm[0] flips sign on the y and z axes -- and k multiplies
+b and -b alike, so an equal k leaves the pair an exact mirror. check_mirror()
asserts that on the written file rather than trusting it.

Joint dynamics: --joint-dynamics applies the isometric scaling law, measured
-----------------------------------------------------------------------------
Scaling only the actuator leaves the joint's own inertia, damping and passive
spring sized for the motor the body no longer has. The classical law for a
geometrically similar body of length scale s is

    inertia   ~ s^5      stiffness ~ s^4      damping ~ sqrt(stiffness*inertia)

with s^4 (not s^3) for stiffness because a rotational stiffness is a torque:
N*m/rad, and gravity torque is m*g*d ~ s^3 * s = s^4.

These bodies are NOT geometrically similar, so no single s exists: scale_robot.py
scales length and girth independently per body group, and the implied s per joint
spans 0.240-1.250 on the child alone (leg 0.692, arm 0.668, torso 0.866, head
1.144 -- matching neither the nominal length nor the girth factor). Driving the
exponents off a nominal s therefore misfires badly: measured zeta lands in
0.32-1.24, i.e. some joints turn sharply underdamped and will ring.

So the law is applied through what the exponents MEAN, read off the two compiled
models, instead of through s:

    s^5  ->  Ir, the measured subtree-inertia ratio (mass-matrix diagonal minus
             armature, in the default pose)
    s^4  ->  k, the measured gravity-load ratio -- already the actuator's k
    sqrt ->  sqrt(k*Ir)

    armature  = adult_armature  * Ir
    damping   = adult_damping   * sqrt(k*Ir)
    stiffness = adult_stiffness * k

Scaling armature by Ir makes the TOTAL inertia scale by exactly Ir, because the
subtree term already does: I = Isub_a*Ir + a_a*Ir = Ir*I_a. The damping ratio
zeta = C/(2*sqrt(K*I)) then reduces to sqrt(k*Ir)/sqrt(k*Ir) = 1 identically, so
zeta is preserved exactly on every joint rather than approximately -- zeta is the
quantity that decides whether a joint overshoots or rings, which is what breaks a
frozen policy. tau = C/K = sqrt(Ir/k) = sqrt(s) and omega_n = 1/sqrt(s), i.e. the
body keeps its own Froude clock: the child runs 1.21x faster than the adult,
which is what a smaller body physically does.

Base values come from the REFERENCE body (adult), not from src. src's own joint
values were scaled by scale_robot.py's length^3*girth^2 heuristic, and applying
anything on top of that compounds two unrelated factors -- the earlier "multiply
src by k" version did exactly that and drove tau to 0.46 and zeta to 0.56. The
reference is also where the base is resolved through <default> and class=
inheritance: 18 of the 69 joints in child/robot.xml (toes, wrists, hands) carry
no damping attribute at all, so a text-level edit would have skipped them.

Because damping and biasprm[2] must share a factor, biasprm[2] scales by
sqrt(k*Ir) while the other three affine terms keep k. That does NOT disturb the
equilibrium identity above: q* = -(gainprm[0]*ctrl + biasprm[0])/biasprm[1] has
no biasprm[2] in it, so ctrl keeps its meaning. verify() splits the check into a
static part and a velocity part accordingly.

Usage:
  uv run scripts/torque_scale_actuators.py
  uv run scripts/torque_scale_actuators.py --out assets/robots/child/robots_torque.xml
  uv run scripts/torque_scale_actuators.py --symmetry none   # per-actuator k
  uv run scripts/torque_scale_actuators.py --joint-dynamics  # + joint dynamics
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "outputs/torque_ratio_per_joint/move-ego-90-2_0_gravity/_ratios.csv"
DEFAULT_SRC = ROOT / "assets/robots/child/robot.xml"
DEFAULT_OUT = ROOT / "assets/robots/child/robots_torque.xml"


def fmt(x: float) -> str:
    """Round-trippable float text, without numpy's array repr."""
    return repr(float(x))


def choose_k(rows, r2_min):
    """Per-actuator scale factor, with the fallback rule applied and reported.

    Returns (kmap, notes, trusted); trusted holds the actuators whose own fitted
    k survived, which is what symmetrize_k needs to know when the two sides of a
    pair disagree about whether their fit meant anything.
    """
    out, notes, trusted = {}, [], set()
    for r in rows:
        name = r["actuator"]
        kf, r2, kp = (float(r["k_fitted"]), float(r["r2"]),
                      float(r["k_predicted_subtree"]))
        if r2 >= r2_min and kf > 0:
            out[name] = kf
            trusted.add(name)
            continue
        if not (kp > 0) or not np.isfinite(kp):
            out[name] = 1.0
            notes.append(f"{name}: R²={r2:.3f}, k_fitted={kf:+.4f}, predictor "
                         f"unusable too -> left unscaled (k=1)")
            continue
        out[name] = kp
        why = "k_fitted<=0" if kf <= 0 else f"R²={r2:.3f}<{r2_min}"
        notes.append(f"{name}: {why}, k_fitted={kf:+.4f} -> using predictor {kp:.4f}")
    return out, notes, trusted


def mirror_pairs(names):
    """Split actuator names into (left, right) pairs and midline singles.

    Pairing is by the L_/R_ prefix the skeleton already uses. A half-pair means
    the naming convention broke, which would silently leave that joint
    asymmetric, so it aborts instead of skipping.
    """
    have = set(names)
    pairs, midline, orphans = [], [], []
    for n in names:
        if n.startswith("L_"):
            partner = "R_" + n[2:]
            (pairs.append((n, partner)) if partner in have else orphans.append(n))
        elif n.startswith("R_"):
            if "L_" + n[2:] not in have:
                orphans.append(n)
        else:
            midline.append(n)
    if orphans:
        raise SystemExit(f"{len(orphans)} actuator(s) have no mirror partner: "
                         f"{orphans[:5]}")
    return pairs, midline


def symmetrize_k(kmap, trusted):
    """Give both actuators of every mirrored pair one shared k, in place.

    The rule is in the module docstring. The one-sided case matters more than it
    looks: a joint that carries almost no gravity torque in a clip fits an
    arbitrary k, and averaging that into the good side would corrupt both
    actuators instead of neither.
    """
    pairs, midline = mirror_pairs(list(kmap))
    report = []
    for lname, rname in pairs:
        kl, kr = kmap[lname], kmap[rname]
        tl, tr = lname in trusted, rname in trusted
        if tl == tr:                      # both fitted, or both fell back
            k, why = 0.5 * (kl + kr), "mean" if tl else "mean of fallbacks"
        elif tl:
            k, why = kl, "L only (R untrusted)"
        else:
            k, why = kr, "R only (L untrusted)"
        kmap[lname] = kmap[rname] = k
        moved = max(abs(kl - k), abs(kr - k)) / max(abs(k), 1e-12)
        report.append((lname[2:], kl, kr, k, moved, why))
    return report, midline


def print_symmetry_report(report, midline, top=6):
    moved_any = [r for r in report if r[4] > 1e-12]
    print(f"symmetry: {len(report)} mirrored pairs share one k, "
          f"{len(midline)} midline actuators keep their own")
    if not moved_any:
        print("  every pair already agreed; no k changed")
        return
    worst = sorted(moved_any, key=lambda r: -r[4])[:top]
    print(f"  {len(moved_any)} pair(s) moved; largest {len(worst)}:")
    print(f"    {'joint':14s} {'k_L':>8s} {'k_R':>8s} {'k_pair':>8s} {'moved':>7s}  rule")
    for j, kl, kr, k, moved, why in worst:
        print(f"    {j:14s} {kl:8.4f} {kr:8.4f} {k:8.4f} {moved:6.2%}  {why}")
    special = [r for r in report if not r[5].startswith("mean")]
    if special:
        print(f"  {len(special)} pair(s) used one side only:")
        for j, kl, kr, k, _, why in special:
            print(f"    {j:14s} k_L {kl:.4f}  k_R {kr:.4f} -> {k:.4f}  ({why})")


def check_mirror(model, tol=1e-9):
    """Assert the written actuators are still exact mirror images.

    gainprm, biasprm[1..2] and forcerange are equal L/R in the source MJCF while
    biasprm[0] flips sign on the y and z axes, so the invariant is "equal, except
    biasprm[0] which is equal in magnitude". Scaling by an equal k preserves all
    four; an unequal k breaks every one of them, which is what this catches.
    """
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    idx = {n: i for i, n in enumerate(names)}
    pairs, _ = mirror_pairs(names)
    worst = 0.0
    for lname, rname in pairs:
        a, b = idx[lname], idx[rname]
        d = max(
            np.abs(model.actuator_gainprm[a] - model.actuator_gainprm[b]).max(),
            np.abs(model.actuator_biasprm[a, 1:3]
                   - model.actuator_biasprm[b, 1:3]).max(),
            np.abs(np.abs(model.actuator_forcerange[a])
                   - np.abs(model.actuator_forcerange[b])).max(),
            abs(abs(model.actuator_biasprm[a, 0])
                - abs(model.actuator_biasprm[b, 0])),
        )
        if d > tol:
            raise SystemExit(f"  !! {lname}/{rname} are not mirror images "
                             f"(max term difference {d:.3e})")
        worst = max(worst, float(d))
    return len(pairs), worst


def scale_line(line: str, k: float, kv: float | None = None) -> str:
    """Scale one <general> line: gainprm, biasprm and forcerange.

    kv is the factor for biasprm[2], the velocity term, which --joint-dynamics
    needs to move with the joint's damping rather than with k. It defaults to k,
    which is the plain "everything delivers k times the force" case.
    """
    kv = k if kv is None else kv

    def one(m):
        return f'{m.group(1)}="{fmt(float(m.group(2)) * k)}"'

    def bias(m):
        vals = m.group(2).split()
        f = [k] * len(vals)
        if len(vals) > 2:
            f[2] = kv
        return f'{m.group(1)}="{" ".join(fmt(float(v) * s) for v, s in zip(vals, f))}"'

    def many(m):
        vals = [fmt(float(v) * k) for v in m.group(2).split()]
        return f'{m.group(1)}="{" ".join(vals)}"'

    line = re.sub(r'(gainprm)="([^"]+)"', one, line)
    line = re.sub(r'(biasprm)="([^"]+)"', bias, line)
    line = re.sub(r'(forcerange)="([^"]+)"', many, line)
    return line


def verify(src_xml, out_xml, kmap, names, n_states=40, seed=0, kvmap=None):
    """Confirm the scaled actuator reproduces the intended affine map exactly.

    Random ctrl at random poses with random velocities, because that is the only
    way to catch a term that was left unscaled -- a pure-pose check would miss
    biasprm[2]. The expectation is built from the SOURCE model's own
    coefficients rather than from k*force, so it stays exact when
    --joint-dynamics gives the velocity term its own factor:

        expected = k * (gainprm[0]*ctrl + biasprm[0] + biasprm[1]*length)
                 + kv * (biasprm[2]*velocity)
    """
    ma = mujoco.MjModel.from_xml_path(str(src_xml))
    mb = mujoco.MjModel.from_xml_path(str(out_xml))
    da, db = mujoco.MjData(ma), mujoco.MjData(mb)
    rng = np.random.default_rng(seed)
    k = np.array([kmap[n] for n in names])
    kv = k if kvmap is None else np.array([kvmap[n] for n in names])

    worst = 0.0
    for _ in range(n_states):
        q = np.zeros(ma.nq); q[3] = 1.0
        q[7:] = rng.uniform(ma.jnt_range[1:, 0], ma.jnt_range[1:, 1])
        v = rng.normal(0, 1.5, ma.nv)
        c = rng.uniform(-1, 1, ma.nu)
        for m, d in ((ma, da), (mb, db)):
            d.qpos[:] = q; d.qvel[:] = v; d.ctrl[:] = c
            mujoco.mj_forward(m, d)
        static = (ma.actuator_gainprm[:, 0] * c + ma.actuator_biasprm[:, 0]
                  + ma.actuator_biasprm[:, 1] * da.actuator_length)
        velocity = ma.actuator_biasprm[:, 2] * da.actuator_velocity
        expected = k * static + kv * velocity
        # forcerange clamps, so compare only where neither model is saturated.
        fa, fb = da.actuator_force.copy(), db.actuator_force.copy()
        lim_a = np.abs(ma.actuator_forcerange).min(axis=1)
        lim_b = np.abs(mb.actuator_forcerange).min(axis=1)
        free = (np.abs(fa) < 0.999 * lim_a) & (np.abs(fb) < 0.999 * lim_b)
        if free.any():
            rel = np.abs(fb[free] - expected[free]) / np.maximum(
                np.abs(expected[free]), 1e-9)
            worst = max(worst, float(rel.max()))
    return worst, mb


JOINT_LINE_RE = re.compile(r'<joint\s+name="([^"]+)"')
REFERENCE_XML = ROOT / "assets/robots/adult/robot.xml"


def set_attr(line: str, attr: str, value: float) -> str:
    """Set attr="value" on one XML element line, inserting it if absent.

    Insertion matters more than replacement: the joints that most need damping
    scaled are exactly the ones that never had a damping attribute to rewrite.
    """
    pat = rf'(?<![-\w]){attr}="[^"]*"'
    if re.search(pat, line):
        return re.sub(pat, f'{attr}="{fmt(value)}"', line)
    return re.sub(r'\s*/>', f' {attr}="{fmt(value)}"/>', line, count=1)


def joint_table(model):
    """Per actuated joint: total diagonal inertia, armature, damping, stiffness.

    Read from the compiled model rather than the XML text so a joint inheriting
    its value from <default> or from a class= is resolved to what MuJoCo
    actually uses; see the module docstring for the 18 joints this catches.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    out = {}
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        dof = model.jnt_dofadr[jid]
        out[name] = dict(I=float(M[dof, dof]),
                         armature=float(model.dof_armature[dof]),
                         damping=float(model.dof_damping[dof]),
                         stiffness=float(model.jnt_stiffness[jid]))
    return out


def joint_targets(src_model, ref_model, kmap, symmetrize=True):
    """Absolute armature/damping/stiffness per joint, plus biasprm[2]'s factor.

    Implements the measured isometric law from the module docstring. Ir is read
    off the two compiled models as the subtree-inertia ratio -- the mass-matrix
    diagonal minus armature, so it is pure geometry and carries none of the
    armature that is about to be overwritten.

    symmetrize gives each mirrored pair one shared Ir, for the same reason k is
    symmetrised: the body is mirror-symmetric by construction, so an L/R spread
    in a per-joint measurement is an artefact, and here it is a large enough one
    to matter -- up to 2.2% on the child (L_Ankle_z). Left unsymmetrised it also
    reaches biasprm[2] through kv and makes check_mirror fail outright.
    """
    src, ref = joint_table(src_model), joint_table(ref_model)
    if set(src) != set(ref):
        raise SystemExit("src and reference do not share the same actuated "
                         "joints; the scaling law has nothing to measure against")
    gain_dev = np.abs(src_model.actuator_gainprm[:, 0]
                      / ref_model.actuator_gainprm[:, 0] - 1).max()
    if gain_dev > 1e-9:
        raise SystemExit(
            f"src actuators already differ from the reference by up to "
            f"{gain_dev:.3e}; k would be applied on top of an earlier scaling "
            f"and would no longer mean 'ratio to the reference body'")

    Ir = {name: ((src[name]["I"] - src[name]["armature"])
                 / (ref[name]["I"] - ref[name]["armature"])) for name in kmap}
    if symmetrize:
        moved = 0.0
        for lname, rname in mirror_pairs(list(Ir))[0]:
            shared = 0.5 * (Ir[lname] + Ir[rname])
            moved = max(moved, abs(Ir[lname] - shared) / shared)
            Ir[lname] = Ir[rname] = shared
        print(f"  Ir symmetrised over mirrored pairs (largest move {moved:.2%})")

    targets = {}
    for name, k in kmap.items():
        kv = float(np.sqrt(k * Ir[name]))
        targets[name] = dict(
            armature=ref[name]["armature"] * Ir[name],
            damping=ref[name]["damping"] * kv,
            stiffness=ref[name]["stiffness"] * k,
            kv=kv, Ir=Ir[name])
    return targets


def closed_loop(model):
    """Per joint: tau = C/K, zeta, omega_n and the integrator margin.

    C sums the joint's own damping and the servo's velocity gain (the joint term
    is the larger of the two, a median 78.9% on the adult); K sums the servo's
    position gain and the passive spring (here the servo dominates, 90%). Both
    have to be counted or the numbers describe neither system.
    """
    t = joint_table(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        Kp = -float(model.actuator_biasprm[a, 1])
        Kd = -float(model.actuator_biasprm[a, 2])
        I = max(t[name]["I"], 1e-12)
        K = max(Kp + t[name]["stiffness"], 1e-12)
        C = t[name]["damping"] + Kd
        out[name] = dict(tau=C / K, zeta=C / (2 * np.sqrt(K * I)),
                         wn=np.sqrt(K / I),
                         margin=model.opt.timestep * np.sqrt(Kp / I))
    return out


def report_joint_dynamics(out_model, ref_model, targets, symmetrized=True):
    """Check the written joints against the law, and against the reference.

    zeta is preserved identically only when Ir is each joint's own measurement:
    armature*Ir makes the total inertia scale by exactly Ir, and zeta reduces to
    sqrt(k*Ir)/sqrt(k*Ir). Sharing one Ir across a mirrored pair breaks that
    cancellation by however much the two sides disagreed, which is ~0.2% here --
    a trade worth making, since an exact left/right mirror matters more than the
    third decimal of zeta. The tolerance below follows which case applies.
    """
    got = joint_table(out_model)
    worst = {a: 0.0 for a in ("armature", "damping", "stiffness")}
    for name, want in targets.items():
        for attr in worst:
            ref = max(abs(want[attr]), 1e-12)
            worst[attr] = max(worst[attr], abs(got[name][attr] - want[attr]) / ref)
    print("\njoint dynamics (measured isometric law: armature x Ir, "
          "damping x sqrt(k*Ir), stiffness x k):")
    for attr, w in worst.items():
        print(f"  {attr:9s} matches the law: {w:.2e}")
    Ir = np.array([t["Ir"] for t in targets.values()])
    print(f"  Ir (subtree inertia ratio): median {np.median(Ir):.4f}  "
          f"range {Ir.min():.4f}-{Ir.max():.4f}")

    a, b = closed_loop(ref_model), closed_loop(out_model)
    shared = [n for n in b if n in a]
    for key, label, want in (("zeta", "zeta  (damping ratio)", "1.00 exactly"),
                             ("tau", "tau   = C/K", "sqrt(s), the Froude clock"),
                             ("wn", "omega_n", "1/sqrt(s)")):
        r = np.array([b[n][key] / a[n][key] for n in shared])
        print(f"  {label:22s} vs reference: median {np.median(r):.3f}  "
              f"range {r.min():.3f}-{r.max():.3f}   (expect {want})")
    zr = np.array([b[n]["zeta"] / a[n]["zeta"] for n in shared])
    tol = 1e-2 if symmetrized else 1e-9
    if np.abs(zr - 1.0).max() > tol:
        raise SystemExit(f"  !! zeta off by {np.abs(zr - 1).max():.3e}, above the "
                         f"{tol:.0e} allowed with symmetrize="
                         f"{symmetrized}; the law was not applied consistently")
    if symmetrized:
        print(f"  (zeta exact to {np.abs(zr - 1).max():.1e}, not to machine "
              f"precision, because mirrored pairs share one Ir)")
    m = np.array([b[n]["margin"] for n in shared])
    print(f"  dt*sqrt(Kp/I) max {m.max():.3f} (explicit integrator needs < 2)")


def write_scaled_xml(src, out, kmap, check_symmetry=True, joint_dynamics=False,
                     reference=REFERENCE_XML):
    """Apply kmap to every <general> line of src, write out, and verify exactly.

    Shared with scripts/aggregate_motion_k.py, which supplies a k averaged over
    many motions instead of a single clip's fit; everything downstream of "here
    is one k per actuator" is identical, and the verification below is the part
    that must not be duplicated and drift.

    check_symmetry asserts the mirrored pairs came out identical, so it must be
    off when the caller deliberately used a per-actuator k (--symmetry none).

    joint_dynamics additionally rewrites each joint's armature, damping and
    stiffness from the reference body under the measured isometric law -- see
    the module docstring for the three exponents and why the base is the
    reference rather than src.
    """
    src, out = Path(src), Path(out)
    model = mujoco.MjModel.from_xml_path(str(src))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    missing = [n for n in names if n not in kmap]
    if missing:
        raise SystemExit(f"no k for {len(missing)} actuator(s): {missing[:5]}")

    ref_model = targets = None
    if joint_dynamics:
        ref_model = mujoco.MjModel.from_xml_path(str(reference))
        targets = joint_targets(model, ref_model, kmap, symmetrize=check_symmetry)

    text = src.read_text().splitlines(keepends=True)
    edited = edited_joints = 0
    for i, line in enumerate(text):
        m = re.search(r'<general\s+name="([^"]+)"', line)
        if m and m.group(1) in kmap:
            name = m.group(1)
            kv = targets[name]["kv"] if joint_dynamics else None
            text[i] = scale_line(line, kmap[name], kv)
            edited += 1
            continue
        j = JOINT_LINE_RE.search(line)
        if joint_dynamics and j and j.group(1) in kmap:
            want = targets[j.group(1)]
            for attr in ("armature", "damping", "stiffness"):
                line = set_attr(line, attr, want[attr])
            text[i] = line
            edited_joints += 1
    if edited != model.nu:
        raise SystemExit(f"edited {edited} actuator lines but the model has "
                         f"{model.nu}; aborting rather than writing a partial file")
    if joint_dynamics and edited_joints != model.nu:
        raise SystemExit(f"edited {edited_joints} joint lines but {model.nu} "
                         f"actuated joints need one; aborting rather than "
                         f"writing a body whose joints disagree with its motors")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        backup = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, backup)
        print(f"existing {out.name} backed up to {backup.name}")
    out.write_text("".join(text))

    kv = np.array([kmap[n] for n in names])
    print(f"wrote {out}  ({edited} actuators scaled)")
    print(f"  k: median {np.median(kv):.4f}  range {kv.min():.4f}–{kv.max():.4f}")

    kvmap = {n: t["kv"] for n, t in targets.items()} if joint_dynamics else None
    worst, mb = verify(src, out, kmap, names, kvmap=kvmap)
    print(f"\nverification (random poses/velocities/ctrl, unsaturated actuators):")
    print(f"  max relative error of force_scaled / (k * force_original): {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("  !! scaling is NOT exact -- an affine term was missed")
    print("  exact: gainprm, biasprm[0..2] and forcerange all scaled consistently"
          + (" (velocity term on its own factor)" if joint_dynamics else ""))

    gain_r = mb.actuator_gainprm[:, 0] / model.actuator_gainprm[:, 0]
    print(f"  neutral-angle shift vs original: "
          f"{np.abs(mb.actuator_biasprm[:, 0] / mb.actuator_biasprm[:, 1] - model.actuator_biasprm[:, 0] / model.actuator_biasprm[:, 1]).max():.2e} rad "
          f"(0 = ctrl keeps its meaning)")
    print(f"  gainprm ratio matches k: {np.abs(gain_r - kv).max():.2e}")
    print(f"  new forcerange: mean ±{np.abs(mb.actuator_forcerange[:, 1]).mean():.1f} N·m "
          f"(was ±{np.abs(model.actuator_forcerange[:, 1]).mean():.1f})")
    if check_symmetry:
        n_pairs, worst_mirror = check_mirror(mb)
        print(f"  left/right mirror holds for all {n_pairs} pairs "
              f"(max term difference {worst_mirror:.2e})")
    if joint_dynamics:
        report_joint_dynamics(mb, ref_model, targets, symmetrized=check_symmetry)
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--r2-min", type=float, default=0.9,
                   help="below this the fitted k is replaced by the predictor")
    p.add_argument("--symmetry", choices=["pair", "none"], default="pair",
                   help="'pair' gives each mirrored L/R actuator pair one shared "
                        "k; 'none' keeps the raw per-actuator fit")
    p.add_argument("--joint-dynamics", action="store_true",
                   help="also scale each joint's armature and damping by its k "
                        "(passive stiffness is never scaled)")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    src, out = Path(args.src), Path(args.out)
    if not rows:
        raise SystemExit(f"{args.csv} is empty")

    model = mujoco.MjModel.from_xml_path(str(src))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    csv_names = [r["actuator"] for r in rows]
    if set(csv_names) != set(names):
        raise SystemExit(f"csv covers {len(csv_names)} actuators, model has "
                         f"{len(names)}; names do not match")

    kmap, notes, trusted = choose_k(rows, args.r2_min)
    if notes:
        print(f"{len(notes)} actuator(s) did not use the fitted k:")
        for n in notes:
            print(f"    {n}")

    if args.symmetry == "pair":
        report, midline = symmetrize_k(kmap, trusted)
        print()
        print_symmetry_report(report, midline)
    print()
    write_scaled_xml(src, out, kmap, check_symmetry=(args.symmetry == "pair"),
                     joint_dynamics=args.joint_dynamics)


if __name__ == "__main__":
    main()
