"""torque.py -- a scaled body's physics: TorqueCalibrator (per-joint torque ratio k -> robot_torque_full.xml), OOP refactor step 9.

The body from bodybuild/builders.py ScaledBody has the adult's actuators and joint coefficients. TorqueCalibrator sizes them for the
body's own weight, in three steps that were three scripts:
  ratio_matrix()  scripts/torque_ratio_across_motions.py: on the first clip of every motion folder, the gravity torque of every
                  actuated joint (qfrc_bias at qvel = 0) on the adult and on the body (its retargeted clip, same frames), fitted
                  as body = k * adult through the origin -> <matrix>/_k_matrix.csv, _r2_matrix.csv, _corr.csv (motion x motion
                  correlation of the k vectors), _corr.png
  aggregate()     scripts/torque_aggregate_motion_k.py: drop the motions whose mean correlation is a low outlier (median - z x
                  scaled MAD), average each joint's k over the kept motions where its fit reached R^2 >= r2_min (else the
                  geometric prediction k_predicted_subtree), give each mirrored L/R pair one k -> <matrix>/_ratios_aggregate.csv
  write_xml()     write_scaled_xml (scripts/torque_scale_actuators.py's, archived) with joint dynamics: actuator gainprm / biasprm[0,1] /
                  forcerange x k, biasprm[2] x sqrt(k Ir), joint armature = adult x Ir, damping = adult x sqrt(k Ir),
                  stiffness = adult x k (Ir: the measured subtree-inertia ratio) -> assets/robots_torque/<body>/robot_torque_full.xml,
                  then its checks: forces exact on random states, mirror pairs, the law, zeta / tau / omega_n vs the adult,
                  the integrator margin
  check()         the runbook's step 5 (docs/new_body.md): measured k vs the geometric prediction (make_scaled_bodies.py's line)

The fitting math stays plain functions (joint_torques, fit_ratio, predicted_ratios: scripts/torque_ratio_per_joint.py's), as do
the matrix / aggregation / XML helpers; every function is the old code moved verbatim (the outputs are byte-identical:
tests/golden/verify_bodies.py). Failures raise CalibrationError, a SystemExit as before (the scripts exit with the message); the
"zeta off by" check fails after the XML is written, as before.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import paths  # noqa: E402

REFERENCE_XML = paths.ADULT_XML
MATRIX_ROOT_REL = "outputs/torque_ratio_across_motions"       # <mode>/<body>/ (docs/new_body.md step 4)


class CalibrationError(SystemExit):
    """a calibration check failed (a SystemExit, as the scripts raised it: the CLI exits with the message)"""


# ======================================================================== fitting math (scripts/torque_ratio_per_joint.py)

def joint_torques(xml, qpos, dt, *, mode="gravity"):
    """Per-joint torque demand for the 69 actuated hinges. Returns (T,69), ncon, names, model.

    mode "gravity" -- qfrc_bias at qvel=0: the PURE gravity torque of the pose (no constraint force: the retargeted feet are in
    contact on every frame, so anything from mj_inverse is contaminated, and differently on the two bodies; torque_ratio_per_joint.py
    has the full discussion). "quasi" / "full" (mj_inverse) are not implemented (they were commented out there too).
    """
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             or f"act{i}" for i in range(model.nu)]
    dofs = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    # forcerange is in pre-gear actuator units; gear is 1 here, but divide anyway
    # so a future non-unit gear does not silently corrupt the comparison.
    gear = model.actuator_gear[:, 0]

    if mode == "gravity":
        T = len(qpos)
        tau = np.zeros((T, model.nv))
        ncon = np.zeros(T, dtype=int)
        for t in range(T):
            data.qpos[:] = qpos[t]
            data.qvel[:] = 0.0          # qvel=0 -> qfrc_bias is gravity + springs
            mujoco.mj_forward(model, data)
            tau[t] = data.qfrc_bias
            ncon[t] = data.ncon         # recorded for the plot, not used in the fit

    return tau[:, dofs] / gear, ncon, names, model


def fit_ratio(x, y):
    """Least-squares k for y ~ k*x through the origin, plus R^2 about that fit.

    Through the origin, not a free intercept: the claim being tested is
    "child torque is a fixed multiple of adult torque", and a fitted offset
    would let a joint pass with a constant bias it has no physical reason to
    carry. R^2 is 1 - SSE/SS_total-about-zero, so it is comparable to that claim.
    """
    denom = float(np.dot(x, x))
    if denom < 1e-18:
        return np.nan, np.nan
    k = float(np.dot(x, y) / denom)
    sse = float(np.sum((y - k * x) ** 2))
    sst = float(np.sum(y ** 2))
    return k, (1.0 - sse / sst if sst > 1e-18 else np.nan)


def predicted_ratios(m_adult, m_child):
    """Per-joint child/adult gravity-torque ratio predicted from the two models.

    Gravity torque about a hinge is (downstream subtree mass) x g x (lever arm
    from the joint axis to that subtree's centre of mass), so the ratio needs no
    scaling exponent guessed a priori -- both factors are read straight out of a
    forward pass on each model.
    """
    da, dc = mujoco.MjData(m_adult), mujoco.MjData(m_child)
    mujoco.mj_forward(m_adult, da)
    mujoco.mj_forward(m_child, dc)
    out = np.full(m_adult.nu, np.nan)
    for i in range(m_adult.nu):
        ja, jc = m_adult.actuator_trnid[i, 0], m_child.actuator_trnid[i, 0]
        ba, bc = m_adult.jnt_bodyid[ja], m_child.jnt_bodyid[jc]
        arm_a = np.linalg.norm(da.subtree_com[ba] - da.xanchor[ja])
        arm_c = np.linalg.norm(dc.subtree_com[bc] - dc.xanchor[jc])
        num = m_child.body_subtreemass[bc] * arm_c
        den = m_adult.body_subtreemass[ba] * arm_a
        out[i] = num / den if den > 1e-12 else np.nan
    return out


# ======================================================================== k per motion (scripts/torque_ratio_across_motions.py)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
# diverging blue <-> red pinned to [-1, 1] when any correlation is negative; a sequential (YlGnBu, dark -> light) ramp over the
# observed range when they are all positive (torque_ratio_across_motions.py's reasons)
DIVERGING_COLORS = ["#7a1513", "#c0332f", "#e34948", "#ef8a84", "#f7c4bf", "#f0efec",
                    "#b7d3f6", "#86b6ef", "#3987e5", "#256abf", "#104281"]
SEQUENTIAL_COLORS = ["#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#122160"]


def first_clip(motion_dir):
    """The first .npz of a motion, i.e. the *_0 take when the names are padded."""
    clips = sorted(motion_dir.glob("*.npz"))
    return clips[0] if clips else None


def k_vector(a_path, c_path, adult_xml, child_xml, dt, mode):
    """One motion -> (k[69], r2[69], names, frames). None if the pair is unusable."""
    qa = np.load(a_path)["qpos"]
    qc = np.load(c_path)["qpos"]
    if len(qa) != len(qc):
        return None
    ta, _, names, _ = joint_torques(adult_xml, qa, dt, mode=mode)
    tc, _, names_c, _ = joint_torques(child_xml, qc, dt, mode=mode)
    if names != names_c:
        return None
    k = np.full(len(names), np.nan)
    r2 = np.full(len(names), np.nan)
    for i in range(len(names)):
        k[i], r2[i] = fit_ratio(ta[:, i], tc[:, i])
    return k, r2, names, len(qa)


def rankdata(x):
    """Average-tie ranks, so --metric spearman needs no scipy."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average the ranks inside each run of equal values
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def correlate(K, metric):
    """Motion x motion correlation of the k vectors (pearson on the values, spearman on the per-joint ranks)."""
    X = np.array([rankdata(row) for row in K]) if metric == "spearman" else K
    return np.corrcoef(X)


def draw_corr(labels, C, out_path, subtitle):
    """the motion x motion correlation heatmap (_corr.png), torque_ratio_across_motions.py's figure"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    DIVERGING = LinearSegmentedColormap.from_list("corr_div", DIVERGING_COLORS)
    SEQUENTIAL = LinearSegmentedColormap.from_list("corr_seq", SEQUENTIAL_COLORS)
    n = len(labels)
    size = max(9.0, 0.20 * n + 3.2)
    fig, ax = plt.subplots(figsize=(size + 1.4, size), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    off = C[~np.eye(n, dtype=bool)]
    if off.min() < 0:
        cmap, vmin, vmax = DIVERGING, -1.0, 1.0
        ticks = [-1, -0.5, 0, 0.5, 1]
    else:
        # floor to the next 0.05 below the weakest pair, so the colorbar states
        # the range being spent and nobody reads dark blue as "correlation 1".
        vmin = float(np.floor(off.min() * 20) / 20)
        cmap, vmax = SEQUENTIAL, 1.0
        ticks = list(np.round(np.linspace(vmin, 1.0, 5), 3))
    im = ax.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5, color=INK_2)
    ax.set_yticklabels(labels, fontsize=6.5, color=INK_2)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.0)  # 1px surface gap between cells
    ax.tick_params(which="both", colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_color(AXIS)

    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015, ticks=ticks)
    cb.outline.set_color(AXIS)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label("correlation of the two motions' k vectors", color=INK_2, fontsize=9)

    ax.set_title("Per-joint torque-ratio vector, motion vs motion",
                 loc="left", fontsize=13, color=INK, pad=30)
    ax.text(0, 1.006, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=INK_2)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_matrix(path, header, rows, labels):
    """one of the motion x joint / motion x motion csv files ("%.6f")"""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion"] + header)
        for m, row in zip(labels, rows):
            w.writerow([m] + [f"{v:.6f}" for v in row])


def read_matrix(path):
    rows = list(csv.reader(open(path)))
    return rows[0][1:], [r[0] for r in rows[1:]], np.array(
        [[float(v) for v in r[1:]] for r in rows[1:]])


# ======================================================================== aggregation (scripts/torque_aggregate_motion_k.py)

def keep_mask(C, z):
    """Motions whose mean correlation to the others is not a low outlier.

    Robust (median / scaled MAD) rather than mean / std: the tail being cut is
    exactly what would inflate a standard deviation and hide itself.
    """
    mean_c = (C.sum(axis=1) - np.diag(C)) / (len(C) - 1)
    med = np.median(mean_c)
    mad = np.median(np.abs(mean_c - med)) * 1.4826
    if mad < 1e-12:
        return np.ones(len(C), dtype=bool), mean_c, med, mad
    return mean_c >= med - z * mad, mean_c, med, mad


# ======================================================================== the XML writer (archive/scripts/torque_scale_actuators.py's)

def fmt(x: float) -> str:
    """Round-trippable float text, without numpy's array repr."""
    return repr(float(x))


def choose_k(rows, r2_min):
    """Per-actuator scale factor from one clip's _ratios.csv rows, with the fallback rule applied and reported (the one-clip route of
    torque_scale_actuators.py, archived in step 9). Returns (kmap, notes, trusted)."""
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
    """Split actuator names into (left, right) pairs and midline singles (by the L_ / R_ prefix; a half pair aborts)."""
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
        raise CalibrationError(f"{len(orphans)} actuator(s) have no mirror partner: "
                               f"{orphans[:5]}")
    return pairs, midline


def symmetrize_k(kmap, trusted):
    """Give both actuators of every mirrored pair one shared k, in place.

      both sides trusted (or neither) -> the mean of the two;  one side trusted -> that side's k
    (a joint carrying almost no gravity torque in a clip fits an arbitrary k; averaging it into the good side would corrupt both).
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
    """Assert the written actuators are still exact mirror images (equal, except biasprm[0], equal in magnitude)."""
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
            raise CalibrationError(f"  !! {lname}/{rname} are not mirror images "
                                   f"(max term difference {d:.3e})")
        worst = max(worst, float(d))
    return len(pairs), worst


def scale_line(line: str, k: float, kv: float | None = None) -> str:
    """Scale one <general> line: gainprm, biasprm and forcerange (biasprm[2], the velocity term, by kv; default k)."""
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
    """Confirm the scaled actuator reproduces the intended affine map exactly, on random poses / velocities / ctrl:
        expected = k * (gainprm[0]*ctrl + biasprm[0] + biasprm[1]*length) + kv * (biasprm[2]*velocity)
    (unsaturated actuators only). Returns (worst relative error, the scaled model)."""
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


def set_attr(line: str, attr: str, value: float) -> str:
    """Set attr="value" on one XML element line, inserting it if absent (the joints that most need damping scaled are exactly
    the ones that never had a damping attribute)."""
    pat = rf'(?<![-\w]){attr}="[^"]*"'
    if re.search(pat, line):
        return re.sub(pat, f'{attr}="{fmt(value)}"', line)
    return re.sub(r'\s*/>', f' {attr}="{fmt(value)}"/>', line, count=1)


def joint_table(model):
    """Per actuated joint: total diagonal inertia, armature, damping, stiffness (from the compiled model, <default> resolved)."""
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
    """Absolute armature / damping / stiffness per joint, plus biasprm[2]'s factor: the measured isometric law
        armature = ref armature x Ir,  damping = ref damping x sqrt(k Ir),  stiffness = ref stiffness x k
    Ir = (M_ii - armature) of src / of the reference (subtree inertia, pure geometry), one shared Ir per mirrored pair."""
    src, ref = joint_table(src_model), joint_table(ref_model)
    if set(src) != set(ref):
        raise CalibrationError("src and reference do not share the same actuated "
                               "joints; the scaling law has nothing to measure against")
    gain_dev = np.abs(src_model.actuator_gainprm[:, 0]
                      / ref_model.actuator_gainprm[:, 0] - 1).max()
    if gain_dev > 1e-9:
        raise CalibrationError(
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
    """Per joint: tau = C/K, zeta, omega_n and the integrator margin (C = joint damping + servo kd, K = servo kp + spring)."""
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
    """Check the written joints against the law, and against the reference (zeta within 1e-2 with shared pair Ir)."""
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
        raise CalibrationError(f"  !! zeta off by {np.abs(zr - 1).max():.3e}, above the "
                               f"{tol:.0e} allowed with symmetrize="
                               f"{symmetrized}; the law was not applied consistently")
    if symmetrized:
        print(f"  (zeta exact to {np.abs(zr - 1).max():.1e}, not to machine "
              f"precision, because mirrored pairs share one Ir)")
    m = np.array([b[n]["margin"] for n in shared])
    print(f"  dt*sqrt(Kp/I) max {m.max():.3f} (explicit integrator needs < 2)")


def write_scaled_xml(src, out, kmap, check_symmetry=True, joint_dynamics=False,
                     reference=REFERENCE_XML):
    """Apply kmap to every <general> line of src (and, with joint_dynamics, rewrite each actuated joint's armature / damping /
    stiffness from the reference body under the measured law), write out (an existing file is backed up to <out>.bak first), and
    verify exactly (forces, gains, neutral angles, mirror pairs, the law). Returns the actuator names."""
    src, out = Path(src), Path(out)
    model = mujoco.MjModel.from_xml_path(str(src))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    missing = [n for n in names if n not in kmap]
    if missing:
        raise CalibrationError(f"no k for {len(missing)} actuator(s): {missing[:5]}")

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
        raise CalibrationError(f"edited {edited} actuator lines but the model has "
                               f"{model.nu}; aborting rather than writing a partial file")
    if joint_dynamics and edited_joints != model.nu:
        raise CalibrationError(f"edited {edited_joints} joint lines but {model.nu} "
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
        raise CalibrationError("  !! scaling is NOT exact -- an affine term was missed")
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


# ======================================================================== the calibrator

class TorqueCalibrator:
    """One body's torque calibration against the adult (module docstring).

        cal = TorqueCalibrator(shape_xml, retarget_dir, matrix_dir)
        cal.ratio_matrix()                     # _k_matrix.csv, _r2_matrix.csv, _corr.csv, _corr.png
        kmap = cal.aggregate()                 # _ratios_aggregate.csv; the k per actuator
        cal.write_xml(out, kmap)               # robot_torque_full.xml (joint dynamics on), checks printed
        cal.motion_choice(kmap)                # how much dropping motions moved k (printed)
        cal.check()                            # the measured / predicted line (docs/new_body.md step 5)

    Defaults are the pipeline's (docs/new_body.md): gravity torque, pearson, every motion folder's first clip, z = 3,
    R^2 >= 0.9, mean, mirrored pairs, joint dynamics. The adult is paths.ADULT_XML and its clips paths.ORIGIN_MOTION.
    """

    def __init__(self, body_xml, retarget_dir, matrix_dir, *, adult_xml=paths.ADULT_XML, origin=paths.ORIGIN_MOTION,
                 mode="gravity", fps=30.0, reference=REFERENCE_XML):
        self.body_xml, self.adult_xml = body_xml, adult_xml
        self.retarget_dir = Path(retarget_dir) if retarget_dir is not None else None     # ratio_matrix only
        self.origin, self.matrix_dir = Path(origin), Path(matrix_dir)
        self.mode, self.fps, self.reference = mode, fps, reference
        self._agg = None

    # ------------------------------------------------------------ step 4: k per motion and joint
    MATRIX_FILES = ("_k_matrix.csv", "_r2_matrix.csv", "_corr.csv")

    def ratio_done(self):
        return all((self.matrix_dir / f).exists() for f in self.MATRIX_FILES)

    def ratio_matrix(self, metric="pearson", r2_min=0.0, figure=True, log=print):
        """torque_ratio_across_motions.py main: the k / R^2 matrices, the motion correlation, its figure, the summary lines"""
        origin, retarget = self.origin, self.retarget_dir
        motions = sorted(d.name for d in origin.iterdir() if d.is_dir())
        outdir = self.matrix_dir
        outdir.mkdir(parents=True, exist_ok=True)

        dt = 1.0 / self.fps
        used, K, R2, names = [], [], [], None
        for m in motions:
            a = first_clip(origin / m)
            c_dir = retarget / m
            c = first_clip(c_dir) if c_dir.is_dir() else None
            if a is None or c is None or a.name != c.name:
                log(f"  skip {m}: no matching first clip")
                continue
            got = k_vector(a, c, str(self.adult_xml), str(self.body_xml), dt, self.mode)
            if got is None:
                log(f"  skip {m}: frame count or actuator names differ")
                continue
            k, r2, names, T = got
            used.append(m)
            K.append(k)
            R2.append(r2)
            log(f"  {m:34s} {a.name:34s} {T:4d} frames   "
                f"k median {np.nanmedian(k):.3f}  R² median {np.nanmedian(r2):.4f}")

        if len(used) < 2:
            raise CalibrationError("need at least two usable motions")
        K = np.array(K)
        R2 = np.array(R2)

        # A joint enters the correlation only if it is finite in EVERY motion:
        # a per-motion mask would make each pair of vectors a different quantity.
        keep = np.isfinite(K).all(axis=0) & np.isfinite(R2).all(axis=0)
        if r2_min > 0:
            keep &= (R2 >= r2_min).all(axis=0)
        dropped = [names[i] for i in range(len(names)) if not keep[i]]
        log(f"\n{len(used)} motions x {int(keep.sum())}/{len(names)} joints"
            + (f"   dropped: {', '.join(dropped)}" if dropped else ""))

        C = correlate(K[:, keep], metric)
        off = C[~np.eye(len(C), dtype=bool)]
        subtitle = (f"{len(used)} motions (first clip of each) x {int(keep.sum())} joints "
                    f"— {metric}, {self.mode} torque"
                    + (f", R² ≥ {r2_min:g} in every motion" if r2_min > 0 else ""))
        if figure:
            draw_corr(used, C, outdir / "_corr.png", subtitle)

        write_matrix(outdir / "_k_matrix.csv", names, K, used)
        write_matrix(outdir / "_r2_matrix.csv", names, R2, used)
        write_matrix(outdir / "_corr.csv", used, C, used)

        log("\n=== is k a body property or a motion property? ===")
        log(f"off-diagonal correlation : median {np.median(off):.4f}   "
            f"min {off.min():.4f}   p05 {np.percentile(off, 5):.4f}")
        worst_pair = np.unravel_index(np.argmin(np.where(np.eye(len(C), dtype=bool),
                                                         np.inf, C)), C.shape)
        log(f"least agreeing pair      : {used[worst_pair[0]]} vs "
            f"{used[worst_pair[1]]}  ({C[worst_pair]:.4f})")
        mean_c = (C.sum(axis=1) - 1) / (len(C) - 1)
        odd = np.argsort(mean_c)[:5]
        log("most atypical motions    : "
            + ", ".join(f"{used[i]} ({mean_c[i]:.3f})" for i in odd))
        spread = K[:, keep].std(axis=0) / np.abs(K[:, keep].mean(axis=0))
        ji = np.array([i for i in range(len(names)) if keep[i]])
        wj = np.argsort(spread)[-5:][::-1]
        log("joints whose k moves most across motions (CV): "
            + ", ".join(f"{names[ji[i]]} {spread[i]:.1%}" for i in wj))
        log(f"\n-> {outdir}/_corr.png, _corr.csv, _k_matrix.csv, _r2_matrix.csv")
        return dict(motions=used, K=K, R2=R2, names=names, C=C)

    # ------------------------------------------------------------ step 6a: one k per actuator
    def aggregate_csv(self, motions=None):
        suffix = ("_" + re.sub(r"[^0-9A-Za-z]+", "", motions)) if motions else ""
        return self.matrix_dir / f"_ratios_aggregate{suffix}.csv"

    def aggregate(self, z=3.0, r2_min=0.9, agg="mean", motions=None, symmetry="pair", src=None, write_csv=True, log=print):
        """torque_aggregate_motion_k.py up to its per-actuator table: returns kmap {actuator: k} (also kept for motion_choice)"""
        mdir = self.matrix_dir
        joints, motions_, K = read_matrix(mdir / "_k_matrix.csv")
        joints_r, motions_r, R2 = read_matrix(mdir / "_r2_matrix.csv")
        corr_motions, _, C = read_matrix(mdir / "_corr.csv")
        if joints != joints_r or motions_ != motions_r or corr_motions != motions_:
            raise CalibrationError(f"{mdir} matrices disagree on their rows/columns")
        motions_all = motions_

        if motions:
            sel = [i for i, m in enumerate(motions_) if re.search(motions, m)]
            if not sel:
                raise CalibrationError(f"--motions {motions!r} matched none of the "
                                       f"{len(motions_)} motions")
            # C is motion x motion, so it has to be subset on BOTH axes or the
            # agreement cut would score each kept motion against the excluded ones.
            motions_ = [motions_[i] for i in sel]
            K, R2, C = K[sel], R2[sel], C[np.ix_(sel, sel)]
            log(f"--motions {motions!r}: {len(motions_)} of "
                f"{len(corr_motions)} motions")

        keep, mean_c, med, mad = (keep_mask(C, z) if z > 0
                                  else (np.ones(len(C), bool), None, None, None))
        dropped = [motions_[i] for i in np.where(~keep)[0]]
        log(f"{len(motions_)} motions, {len(joints)} joints")
        if z > 0:
            log(f"agreement cut: mean-corr < {med - z * mad:.4f} "
                f"(median {med:.4f} - {z:g} x scaled MAD {mad:.4f})")
        if dropped:
            log(f"dropped {len(dropped)}:")
            for i in np.argsort(mean_c):
                if not keep[i]:
                    log(f"    {motions_[i]:34s} mean-corr {mean_c[i]:.4f}")
        log(f"kept {int(keep.sum())} motions"
            + (f"  (weakest kept: {motions_[int(np.argmin(np.where(keep, mean_c, np.inf)))]} "
               f"{np.where(keep, mean_c, np.inf).min():.4f})" if z > 0 else ""))

        src = str(self.body_xml if src is None else src)
        m_child = mujoco.MjModel.from_xml_path(src)
        m_adult = mujoco.MjModel.from_xml_path(str(self.reference))
        k_pred = predicted_ratios(m_adult, m_child)
        model_names = [mujoco.mj_id2name(m_child, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                       for i in range(m_child.nu)]
        if model_names != joints:
            raise CalibrationError("the k matrix's joints do not match the model's actuators")

        Kk, R2k = K[keep], R2[keep]
        kmap, spread, notes, trusted = {}, {}, [], set()
        for j, name in enumerate(joints):
            ok = np.isfinite(Kk[:, j]) & (R2k[:, j] >= r2_min) & (Kk[:, j] > 0)
            if ok.sum() == 0:
                kp = k_pred[j]
                kmap[name] = float(kp) if np.isfinite(kp) and kp > 0 else 1.0
                spread[name] = (0, np.nan)
                notes.append(f"{name}: no kept motion reached R² {r2_min:g} "
                             f"(best {np.nanmax(R2k[:, j]):.3f}) -> predictor "
                             f"{kmap[name]:.4f}")
                continue
            vals = Kk[ok, j]
            kmap[name] = float(np.mean(vals) if agg == "mean" else np.median(vals))
            spread[name] = (int(ok.sum()), float(vals.std()))
            trusted.add(name)

        if symmetry == "pair":
            log("")
            report, midline = symmetrize_k(kmap, trusted)
            print_symmetry_report(report, midline)

        kv = np.array([kmap[n] for n in joints])
        cv = np.array([spread[n][1] / abs(kmap[n]) if spread[n][0] else np.nan
                       for n in joints])
        used = np.array([spread[n][0] for n in joints])
        log(f"\nk per actuator: {agg} over the kept motions that cleared "
            f"R² {r2_min:g} (median {int(np.median(used))} motions/joint, "
            f"min {used.min()})")
        log(f"  k median {np.median(kv):.4f}   range {kv.min():.4f}–{kv.max():.4f}")
        worst = np.argsort(np.where(np.isfinite(cv), cv, -1))[-6:][::-1]
        log("  least settled joints (std/mean of k across kept motions):")
        for i in worst:
            log(f"    {joints[i]:14s} k {kv[i]:7.4f}  ±{cv[i]:.1%}  "
                f"from {used[i]} motions")
        if notes:
            log(f"  {len(notes)} actuator(s) fell back to the predictor:")
            for n in notes:
                log(f"    {n}")

        # A filtered run describes a different subset, so it must not overwrite the
        # all-motion table that the unfiltered run writes.
        if write_csv:
            agg_csv = self.aggregate_csv(motions)
            with open(agg_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["actuator", "k_aggregate", "n_motions", "std_across_motions",
                            "k_predicted_subtree"])
                for j, name in enumerate(joints):
                    w.writerow([name, f"{kv[j]:.6f}", used[j],
                                f"{spread[name][1]:.6f}" if used[j] else "",
                                f"{k_pred[j]:.6f}"])
            log(f"  per-actuator table -> {agg_csv}")
        self._agg = dict(kmap=kmap, kv=kv, joints=joints, K=K, R2=R2, motions=motions_, motions_all=motions_all,
                         r2_min=r2_min, symmetry=symmetry)
        return kmap

    # ------------------------------------------------------------ step 6b: the physics XML
    def write_xml(self, out, kmap, symmetry="pair", joint_dynamics=True, src=None):
        """write_scaled_xml(src = the body's shape XML) -> out (robot_torque_full.xml), with its checks"""
        return write_scaled_xml(self.body_xml if src is None else src, out, kmap, check_symmetry=(symmetry == "pair"),
                                joint_dynamics=joint_dynamics, reference=self.reference)

    def motion_choice(self, log=print):
        """torque_aggregate_motion_k.py's last line: the aggregate k vs averaging every motion (both symmetrised the same way)"""
        a = self._agg
        if a is None:
            raise RuntimeError("motion_choice() needs aggregate() first")
        K, R2, joints, kv, r2_min = a["K"], a["R2"], a["joints"], a["kv"], a["r2_min"]
        k_all = {name: float(np.mean(K[np.isfinite(K[:, j]) & (R2[:, j] >= r2_min)
                                       & (K[:, j] > 0), j]))
                 if (R2[:, j] >= r2_min).any() else np.nan
                 for j, name in enumerate(joints)}
        if a["symmetry"] == "pair":
            symmetrize_k(k_all, {n for n, v in k_all.items() if np.isfinite(v)})
        k_all = np.array([k_all[n] for n in joints])
        d = np.abs(kv - k_all) / np.abs(k_all)
        log(f"\nvs averaging all {len(a['motions'])} motions: median |Δk|/k "
            f"{np.nanmedian(d):.2%}, worst {joints[int(np.nanargmax(d))]} "
            f"{np.nanmax(d):.1%}")
        return d

    def calibrate(self, out, z=3.0, r2_min=0.9, agg="mean", motions=None, symmetry="pair", joint_dynamics=True, src=None):
        """torque_aggregate_motion_k.py main: aggregate (csv) -> write_xml -> motion_choice"""
        kmap = self.aggregate(z=z, r2_min=r2_min, agg=agg, motions=motions, symmetry=symmetry, src=src)
        print()
        self.write_xml(out, kmap, symmetry=symmetry, joint_dynamics=joint_dynamics, src=src)
        self.motion_choice()
        return kmap

    # ------------------------------------------------------------ step 5: is k a body property?
    def check(self):
        """the runbook's step-5 numbers from _ratios_aggregate.csv (scripts/make_scaled_bodies.py's line): k median, independent
        fits (joints whose k did not fall back to the prediction), measured / predicted median and range"""
        r = list(csv.DictReader(open(self.matrix_dir / "_ratios_aggregate.csv")))
        km = np.array([float(x["k_aggregate"]) for x in r]); kp = np.array([float(x["k_predicted_subtree"]) for x in r])
        nm = np.array([int(x["n_motions"]) for x in r]); real = (nm > 0) & (~np.isclose(km, kp, rtol=1e-9))
        q = km[real] / kp[real] if real.any() else np.array([1.0])
        return dict(k_median=float(np.median(km)), independent=int(real.sum()), joints=len(r), ratio_median=float(np.median(q)),
                    ratio_min=float(q.min()), ratio_max=float(q.max()),
                    line=(f"k median {np.median(km):.4f} | independent fits {int(real.sum())}/{len(r)}, measured/predicted median "
                          f"{np.median(q):.4f} range {q.min():.3f}-{q.max():.3f}"))
