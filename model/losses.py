"""L_align = d_root + d_ee + d_contact + d_pose + d_velocity (the
functional-equivalence/alignment loss between a generated rollout and a
retargeted reference trajectory -- called D in earlier revisions) plus
L_phys (physical-feasibility/"is this falling over" penalty on the generated
rollout alone -- limit/fall/com_support/foot_slide/penetrate/smooth, see
physics_penalty's docstring for each term).

All computed on qpos arrays (numpy) via forward kinematics -- these are used
as scalar terms inside the REINFORCE return in train.py/train_explore.py,
not backpropagated through directly (see train.py docstring for why: the
MuJoCo rollout that produced qpos_beta isn't autodiff-differentiable). This
also means physics_penalty's terms don't need a differentiable simulator the
way an RL-reward or backprop-through-sim approach would -- they're plain
post-hoc forward-kinematics computations on the recorded qpos, same as L_align.

Both functional_equivalence() and physics_penalty() return (total, terms):
terms is a dict of the unweighted per-component values, kept around for
logging/diagnosis rather than only ever seeing one aggregate number.

Known v1 simplifications (flagged, not silently swept under the rug):
- trajectories of different length are compared by truncating to the
  shorter one, not proper temporal alignment (e.g. DTW).
- d_velocity uses finite-difference qpos deltas as a stand-in for qvel,
  since reference trajectories only store qpos. It also differences the root
  quaternion componentwise, which is not an angular velocity; measured 0
  double-cover sign flips across all 54 reference clips, so it is currently
  harmless, but it is not a quantity with units.
- physics_penalty's com_support term is a static CoM-over-support-base proxy,
  not full ZMP (no CoM acceleration term) -- see its docstring for why.
- com_support and foot_slide gate on FOOT_CONTACT_HEIGHT, a fixed 0.05 m that
  does not scale with the body. Measured rest-pose toe heights span 0.0123
  (tall_slim) to 0.0363 (adult), and on jump-2 the flight phase is detected for
  30% of frames on giant but 0% on child -- i.e. the small bodies get their
  airborne foot penalized for sliding. Fixing this needs a per-body threshold,
  which is a separate change from the three corrected here. foot_slide is
  DISABLED (weight 0) until that gate is fixed -- see PHYS_DISABLED_TERMS.
  com_support gates on the same constant and is still on: it is a position
  comparison, so a mis-gated frame moves it far less than it moves a squared
  velocity.

Three defects were corrected together, all of the same kind -- the loss was not
measuring what it claimed, so any weight tuning or ES search done on top of it
was invalid rather than merely suboptimal:
  1. d_root's heading unwrapped each trajectory separately before subtracting,
     which let the two pick different branches. See d_root.
  2. d_root had no translation term at all, so it was blind to speed -- the one
     parameter that distinguishes the move-ego-* family. See d_root.
  3. dt defaulted to 1.0 instead of 1/30, distorting each term by 30^(-2k) in
     its derivative order k. See DEFAULT_DT and PHYS_DEFAULT_WEIGHTS.
tests/test_losses.py pins all three.
"""

import mujoco
import numpy as np

from . import kinematics as kin


# Control-step duration. The clips in datasets/ all carry fps 30.0 and the env
# runs 15 physics steps of 1/450 s per control step, so this is the real dt for
# everything scored here. It used to default to 1.0, which did NOT rescale the
# cost by a constant: a term built from the k-th time derivative and then
# squared is off by 30^(-2k), so zero-order terms were exact, foot_slide and
# d_velocity were 900x too small, and smooth was 810000x too small. No global
# reweighting can undo that, because the distortion differs per term. It also
# silently bound the loss to 30 fps -- a 60 fps clip would have moved every
# velocity term again with no error raised.
DEFAULT_DT = 1.0 / 30.0

GRAVITY = 9.81

# d_root's fourth sub-term (see its docstring). Set to 0.0 to recover the
# heading/yaw-rate/curvature-only d_root, which is what every number recorded
# before this term existed was measured against.
ROOT_TRAVEL_WEIGHT = 1.0

_LEG_LENGTH_CACHE = {}


def _leg_length(model) -> float:
    """Rest-pose pelvis-to-toe height, the length scale root motion is
    normalized by.

    Both trajectories compared in d_root live on the SAME body, so dividing by
    L does not change which of them is closer -- it makes the term commensurable
    ACROSS bodies, which is the whole point in a pooled cross-embodiment
    objective: a giant's 0.1 m of root error should not automatically outrank a
    child's, they should be compared in body-lengths.

    Cached per MjModel, holding the model so the id() key cannot be recycled.
    """
    key = id(model)
    hit = _LEG_LENGTH_CACHE.get(key)
    if hit is not None:
        return hit[1]
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    pelvis_z = float(data.body(kin.ROOT_BODY).xpos[2])
    toe_z = float(np.mean([data.body(n).xpos[2] for n in kin.FOOT_BODIES]))
    L = max(pelvis_z - toe_z, 1e-6)
    _LEG_LENGTH_CACHE[key] = (model, L)
    return L


def _align_length(a, b):
    T = min(len(a), len(b))
    return a[:T], b[:T]


def _discounted_mean(err, discount: float = 1.0, mask=None) -> float:
    """Mean over frames, weighted by discount ** t.

    Every L_align term is a per-frame error averaged uniformly, and the error a
    tracking rollout makes is CUMULATIVE: the trajectory drifts from the
    reference, so frame 250 is mostly carrying the consequences of frames 0-249
    rather than any decision that could still be made at 250. A uniform mean
    hands most of the objective to that tail. discount < 1 pulls the weight back
    toward the part of the episode a z can still influence -- gamma = 0.99 gives
    frame 300 about 5% of frame 0's weight, gamma = 0.995 about 22%.

    Normalised by the weights' own sum, so the result stays on the same scale as
    the uniform mean instead of shrinking with gamma -- otherwise lambda_align
    would silently need retuning every time gamma moved.

    discount == 1.0 returns np.mean's exact value, not an equivalent computed a
    different way, so nothing recorded before this parameter existed changes.

    err: (T,) or (T, ...) -- extra axes are averaged first, then time is
    weighted. mask: optional per-frame boolean; excluded frames drop out but the
    survivors keep their ORIGINAL t, so a late frame is still discounted as late.
    """
    e = np.asarray(err, dtype=np.float64)
    if e.ndim > 1:
        e = e.reshape(len(e), -1).mean(axis=1)
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(e))
    if mask is not None:
        e = e[mask]
    if len(e) == 0:
        return 0.0
    if discount == 1.0:
        return float(np.mean(e))
    w = np.power(float(discount), idx.astype(np.float64))
    return float(np.dot(w, e) / w.sum())


def d_pose(qpos_a: np.ndarray, qpos_b: np.ndarray, discount: float = 1.0) -> float:
    a, b = _align_length(qpos_a[:, 7:], qpos_b[:, 7:])
    return _discounted_mean((a - b) ** 2, discount)


def d_velocity(qpos_a: np.ndarray, qpos_b: np.ndarray, dt: float = DEFAULT_DT,
               discount: float = 1.0) -> float:
    va = np.diff(qpos_a, axis=0) / dt
    vb = np.diff(qpos_b, axis=0) / dt
    va, vb = _align_length(va, vb)
    return _discounted_mean((va - vb) ** 2, discount)


def d_root(model, qpos_a: np.ndarray, qpos_b: np.ndarray,
           dt: float = DEFAULT_DT, discount: float = 1.0) -> float:
    """heading + yaw_rate + curvature + travel, on the root body.

    heading is 1 - cos(yaw_a - yaw_b), NOT (unwrap(yaw_a) - unwrap(yaw_b))^2.
    np.unwrap is path-dependent: it walks the sequence and adds +-2pi whenever a
    step exceeds pi, so the branch index k is accumulated state. Unwrapping the
    two trajectories SEPARATELY and subtracting afterwards lets each pick its own
    k, and there are two ways that goes wrong:
      * the branch cut lands between them -- both hover near +-pi, noise pushes A
        across on some frame and B not, and every later frame carries a spurious
        2pi. A true error of 0.01 rad reads as 6.29.
      * the true turn counts differ -- rotate-z-5 against rotate-z--5 spin
        opposite ways, so the unwrapped difference diverges LINEARLY in time.
        That is where the measured 578 came from.
    1 - cos is periodic by construction, so the branch is never chosen at all:
    no subtraction-then-wrap, no kink at +-pi (a wrapped d^2 has a derivative
    discontinuity exactly there), and it equals d^2/2 for small d. It does NOT
    carry more gradient at large error -- its slope sin(d) vanishes at d = pi,
    where a wrapped d^2 is steepest. What it buys is that no artifact of the
    representation is being measured.

    How many turns a trajectory made is not lost: yaw_rate is exactly that, in
    bounded per-frame form. Measuring it twice, once unboundedly, was redundant.

    travel is the Froude number, speed / sqrt(g * L) with L the rest-pose
    pelvis-to-toe height, plus root height in units of L. Without it d_root is
    invariant to speed: curvature is a shape descriptor (invariant to
    parameterization and to scale by definition), so "walk 2 m/s straight" and
    "walk 0.5 m/s straight" are the SAME zero-curvature line and score
    identically -- and speed is the only parameter that distinguishes the
    move-ego-* family, more than half the task set. Froude rather than a plain
    speed ratio because dynamic similarity is what makes legged gaits on
    differently-sized bodies actually equivalent (stride frequency and contact
    phase included), which is the claim "functional equivalence" is making.
    """
    pos_a, quat_a = kin.batch_forward_pose(model, qpos_a, [kin.ROOT_BODY])
    pos_b, quat_b = kin.batch_forward_pose(model, qpos_b, [kin.ROOT_BODY])
    root_a, root_b = _align_length(pos_a[kin.ROOT_BODY], pos_b[kin.ROOT_BODY])

    raw_yaw_a = kin.quat_to_yaw(quat_a[kin.ROOT_BODY])
    raw_yaw_b = kin.quat_to_yaw(quat_b[kin.ROOT_BODY])
    ya, yb = _align_length(raw_yaw_a, raw_yaw_b)
    heading_err = _discounted_mean(1.0 - np.cos(ya - yb), discount)

    # unwrap IS correct here: diff of an unwrapped sequence is the per-frame
    # rotation increment, already confined to [-pi, pi]. It is only the
    # *difference of two independently unwrapped sequences* that is unsound.
    yaw_a = np.unwrap(raw_yaw_a)
    yaw_b = np.unwrap(raw_yaw_b)
    yaw_a, yaw_b = _align_length(yaw_a, yaw_b)

    yaw_rate_a, yaw_rate_b = _align_length(np.diff(yaw_a), np.diff(yaw_b))
    yaw_rate_err = _discounted_mean((yaw_rate_a - yaw_rate_b) ** 2, discount)

    def curvature(traj_xy, min_speed=1e-3, clip=50.0):
        # dheading/speed is inherently ill-conditioned as speed -> 0 (e.g.
        # crawling/crouching, near-stationary root): raising min_speed alone
        # just moves the blow-up to "slightly above threshold" samples,
        # since bounded dheading (~pi) over a tiny speed still explodes.
        # Clip to a fixed physically-sane range instead -- +-50 rad/m
        # already covers a sharp human U-turn (~pi over ~0.1 m), so no
        # genuine turning behavior gets clipped, only division-by-~0 noise.
        # Also wrap dheading to [-pi, pi]: arctan2's branch cut otherwise
        # turns a small turn crossing +-pi into a fake ~2*pi jump.
        #
        # Normalized to [-1, 1] by dividing by clip: curvature's natural
        # units (rad/m) aren't comparable to heading_err/yaw_rate_err's
        # (rad^2), so leaving it in raw units would let this one sub-term
        # dominate D_root by orders of magnitude regardless of weighting.
        v = np.diff(traj_xy, axis=0)
        speed = np.linalg.norm(v, axis=-1)
        heading = np.arctan2(v[:, 1], v[:, 0])
        dheading = np.diff(heading)
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
        valid = speed[:-1] > min_speed
        curv = np.zeros_like(dheading)
        curv[valid] = np.clip(dheading[valid] / speed[:-1][valid], -clip, clip) / clip
        return curv, valid

    curv_a, valid_a = curvature(root_a[:, :2])
    curv_b, valid_b = curvature(root_b[:, :2])
    curv_a, curv_b = _align_length(curv_a, curv_b)
    valid_a, valid_b = _align_length(valid_a, valid_b)
    both_valid = valid_a & valid_b
    curv_err = _discounted_mean((curv_a - curv_b) ** 2, discount, mask=both_valid)

    L = _leg_length(model)
    v_ref = np.sqrt(GRAVITY * L)
    froude_a = np.linalg.norm(np.diff(root_a[:, :2], axis=0), axis=-1) / dt / v_ref
    froude_b = np.linalg.norm(np.diff(root_b[:, :2], axis=0), axis=-1) / dt / v_ref
    froude_a, froude_b = _align_length(froude_a, froude_b)
    speed_err = _discounted_mean((froude_a - froude_b) ** 2, discount)
    height_err = _discounted_mean(((root_a[:, 2] - root_b[:, 2]) / L) ** 2, discount)
    travel_err = speed_err + height_err

    return heading_err + yaw_rate_err + curv_err + ROOT_TRAVEL_WEIGHT * travel_err


def d_ee(model, qpos_a: np.ndarray, qpos_b: np.ndarray,
         discount: float = 1.0) -> float:
    bodies = kin.EE_BODIES + [kin.ROOT_BODY]
    pos_a, _ = kin.batch_forward_pose(model, qpos_a, bodies)
    pos_b, _ = kin.batch_forward_pose(model, qpos_b, bodies)
    err = 0.0
    for name in kin.EE_BODIES:
        rel_a = pos_a[name] - pos_a[kin.ROOT_BODY]
        rel_b = pos_b[name] - pos_b[kin.ROOT_BODY]
        rel_a, rel_b = _align_length(rel_a, rel_b)
        err += _discounted_mean((rel_a - rel_b) ** 2, discount)
    return err / len(kin.EE_BODIES)


def d_contact(model, qpos_a: np.ndarray, qpos_b: np.ndarray,
              discount: float = 1.0) -> float:
    pos_a, _ = kin.batch_forward_pose(model, qpos_a, kin.FOOT_BODIES)
    pos_b, _ = kin.batch_forward_pose(model, qpos_b, kin.FOOT_BODIES)
    err = 0.0
    for name in kin.FOOT_BODIES:
        za, zb = _align_length(pos_a[name][:, 2], pos_b[name][:, 2])
        err += _discounted_mean((za - zb) ** 2, discount)
    return err / len(kin.FOOT_BODIES)


def functional_equivalence(model, qpos_beta: np.ndarray, qpos_ref, weights: dict,
                           dt: float = DEFAULT_DT, discount: float = 1.0):
    """weights: dict with keys root/ee/contact/pose/velocity.
    qpos_ref may be None (no retargeted reference attached yet for this
    sample) -> returns (0.0, {}).

    discount: per-frame weight decay, gamma ** t, applied inside every sub-term
    -- see _discounted_mean for why a uniform mean over 300 frames hands the
    objective to accumulated drift. 1.0 (the default) is the plain mean and
    reproduces every number recorded before this existed."""
    if qpos_ref is None:
        return 0.0, {}
    terms = {
        "root": d_root(model, qpos_beta, qpos_ref, dt, discount),
        "ee": d_ee(model, qpos_beta, qpos_ref, discount),
        "contact": d_contact(model, qpos_beta, qpos_ref, discount),
        "pose": d_pose(qpos_beta, qpos_ref, discount),
        "velocity": d_velocity(qpos_beta, qpos_ref, dt, discount),
    }
    total = sum(weights[k] * v for k, v in terms.items())
    return total, terms


# MIGRATION NOTE. These are the weights that reproduce the OLD (dt = 1.0)
# objective exactly, now that dt is real. A term built from the k-th time
# derivative and squared scales by 30^(2k) when dt goes from 1.0 to 1/30, so the
# weights that were tuned against the broken dt have been divided by that same
# factor -- foot_slide (k=1) by 900, smooth (k=2) by 810000. Nothing about the
# cost changes today; what changes is that the number is now correct in SI units
# and no longer silently tied to 30 fps.
#
# This is deliberately NOT the retune. Flipping dt without this would have
# multiplied smooth by 810000 and handed the whole objective to it, mixing "fix
# a bug" with "change what is optimized". Retune from here, one term at a time,
# with the equivalence point as the baseline: the pre-dt-fix values were
# foot_slide 1.0 and smooth 0.01, and going back to those is the first
# experiment worth running.
#
# foot_slide is OFF (weight 0) -- see PHYS_DISABLED_TERMS below.
PHYS_DEFAULT_WEIGHTS = {
    "limit": 1.0,           # joint-range violation                        (k=0)
    "fall": 5.0,            # pelvis height + torso tilt                   (k=0)
    "com_support": 1.0,     # CoM_xy vs. the grounded-foot support base    (k=0)
    "foot_slide": 0.0,      # DISABLED, was 1.0 / 30 ** 2                  (k=1)
    "penetrate": 1.0,       # foot z < 0 -- clipping through the floor     (k=0)
    "smooth": 0.01 / 30 ** 4,         # joint angular-acceleration proxy   (k=2)
}

# Weighted 0, not deleted: the term is still computed and still comes back in
# physics_penalty's terms dict, so terms.csv / audit.csv keep recording it and
# turning it back on is a one-line change rather than a re-implementation. It
# just does not enter the total.
#
# Why foot_slide is off. Its gate is FOOT_CONTACT_HEIGHT, a fixed 0.05 m that
# does not scale with the body (see this module's docstring). The child's
# rest-pose toe sits at 0.014 m and 55% of its reference foot-frames fall under
# 0.05 m, so a large part of what the term calls "a planted foot sliding" is the
# reference's own swing phase passing low over the ground. It shows up in the
# separation: measured on upright clips the retargeted reference is beaten by
# only 1.6x on condition B, against 12-29x for fall and 9-88x for smooth -- the
# term barely distinguishes a rollout from a motion that is correct by
# construction, which is the one thing L_phys has to do. Re-enable it after the
# gate is per-body (rest-pose toe height + a fraction of leg length), not before.
PHYS_DISABLED_TERMS = ("foot_slide",)

# Every ACTIVE term weighted 1.0 -- the "let each term speak at its own
# magnitude" ablation. Read the magnitudes before reading the total: the terms
# are NOT commensurable. Measured over the 540 child clips of
# scripts/loss_test.py, unweighted means on the retargeted reference are limit
# 9e-4, fall 4e-1, com_support 3e-2 and smooth 1.4e3 (1.5e4 on a rollout), so
# with equal weights L_phys is smooth to within 0.1% and the three terms the
# metric exists to measure contribute nothing. Kept as a named constant because
# that collapse is worth being able to reproduce, not because it is a good
# objective.
PHYS_EQUAL_WEIGHTS = {k: (0.0 if k in PHYS_DISABLED_TERMS else 1.0)
                      for k in PHYS_DEFAULT_WEIGHTS}

# Each active term scaled so it contributes ~1.0 at a TYPICAL ROLLOUT, which is
# what "every term gets an equal say" actually requires -- equal weights give it
# to whichever term happens to carry the largest units (smooth, by 10^3).
#
# w = 1 / median(term), medians pooled over conditions A/B/C/D of
# scripts/loss_test.py on the 300 upright child clips
# (outputs/loss_test_upright/terms.csv):
#
#     fall         0.4579     ->    2.18
#     com_support  0.04832    ->   20.7
#     smooth       4.412e+04  ->    2.27e-05
#
# The anchor is the rollouts, not the reference: the reference's own value is
# near zero on every term (that is the point of it), so normalising there would
# divide by noise. Pooling all four conditions keeps the weights from moving
# when one of them improves.
#
# limit and penetrate are NOT median-normalised, because they are not quantities
# with a typical value -- they are VIOLATIONS, zero whenever nothing is wrong.
# Dividing by their median promotes the smallest routine violation to a full
# unit of cost (penetrate's pooled median is 2.8e-07, so 1/median is 3.5e6 and
# one rare frame would swamp the objective; limit's 1/median is 239, which made
# it 45.8% of B's total). They get a physical scale instead -- "this much
# violation, sustained, costs one unit":
#
#     penetrate    0.01 m of foot below the floor   ->  1 / 0.01^2  = 1e4
#     limit        0.1 rad (5.7 deg) past the stop  ->  1 / 0.1^2   = 100
#
# What this changes, measured on the 300 upright clips. Under equal weights
# L_phys was smooth to within 0.1% and A->C read as -63%; almost all of that was
# smooth alone. Under these weights every term is visible, and the ladder does
# NOT hold: B is worse than A on fall (0.900 -> 1.168), com_support (1.108 ->
# 1.245) and limit, better only on smooth. Lowering limit stops it dominating
# but does not restore the ladder, because limit was never the only reason --
# equal weights were hiding three separate regressions behind one improvement.
PHYS_BALANCED_WEIGHTS = {
    "limit": 100.0,         # 1 / (0.1 rad)^2 -- violation scale, not median
    "fall": 2.18,
    "com_support": 20.7,
    "foot_slide": 0.0,      # disabled, see PHYS_DISABLED_TERMS
    "penetrate": 1.0e4,     # 1 / (0.01 m)^2 -- violation scale, not median
    "smooth": 2.27e-5,
}

PHYS_WEIGHT_TABLES = {
    "default": PHYS_DEFAULT_WEIGHTS,
    "equal": PHYS_EQUAL_WEIGHTS,
    "balanced": PHYS_BALANCED_WEIGHTS,
}

# Foot-body world z below this counts as "grounded" for com_support/foot_slide
# gating -- approximate (real ground contact needs mj_forward's contact
# solver, not available from qpos alone), calibrated to L_Toe/R_Toe's own
# geom half-height in robot_child.xml (~0.02-0.03m) with slack for the body
# origin sitting slightly above the geom's bottom face.
FOOT_CONTACT_HEIGHT = 0.05


def _point_segment_dist_xy(p, a, b):
    """p, a, b: (T, 2). Distance from each p[t] to the segment a[t]-b[t]."""
    ab = b - a
    ab_len_sq = np.sum(ab ** 2, axis=-1)
    t = np.zeros(p.shape[0])
    valid = ab_len_sq > 1e-8
    t[valid] = np.clip(np.sum((p[valid] - a[valid]) * ab[valid], axis=-1) / ab_len_sq[valid], 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(p - closest, axis=-1)


def _joint_limit_penalty(model, qpos_seq):
    penalty = 0.0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qadr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        if lo == hi:
            continue
        vals = qpos_seq[:, qadr]
        viol = np.clip(lo - vals, 0, None) + np.clip(vals - hi, 0, None)
        penalty += float(np.mean(viol ** 2))
    return penalty


def _fall_penalty(qpos_seq, qpos_ref=None):
    """Smooth, continuous fall signal (see prior docstring note: a hard
    pelvis-height threshold saturates almost immediately on the child body
    and stops carrying gradient). Two components:
      - height_ratio: clip(pelvis_z / ref_h, 0, 1) -> (1-ratio)^2, where ref_h
        is the height the pelvis is SUPPOSED to be at.

        With qpos_ref (the retargeted reference for this clip) the reference is
        that clip's own PER-FRAME pelvis height, so a motion that is meant to be
        low -- crawl, lieonground, headstand -- is scored against where it should
        be rather than against its own first frame. Measured on the 540 child
        clips, the old self-referenced version gave the retargeted reference
        itself a mean fall of 1.100 on the 240 ground clips (headstand alone
        scored 17.6, WORSE than a failed rollout at 15.0) purely because the
        first frame is standing and everything after it legitimately is not.
        Only being LOWER than the reference is penalized -- the clip at 1.0 is
        the ceiling, so a rollout that stays up while the reference goes down
        scores 0 here, and it is d_root/d_pose's job to notice that.

        Without qpos_ref the fallback is the old qpos_seq[0, 2], which is what
        train_explore.py (no reference by construction) still uses.
      - tilt: torso "up" axis vs. world z. Deliberately NOT referenced -- see
        the note under physics_penalty. NOT the naive
        up_z = 1 - 2*(qx^2 + qy^2) (that assumes local +Z is anatomical
        "up" at identity quaternion) -- this asset's free joint bakes in a
        SMPL-style Y-up rest pose, so the model's own standing/rest qpos has
        quat (0.70710678, 0.70710678, 0, 0), NOT identity. Verified via the
        rotation matrix: that quat maps local Y (not Z) to world +Z, so the
        correct closed-form up_z (z-component of the rotated local-Y axis)
        is up_z = 2*(qy*qz + qw*qx) -> (1 - up_z)^2, 0 when upright, up to 4
        when upside down.
    """
    pelvis_z = qpos_seq[:, 2]
    if qpos_ref is not None:
        pelvis_z, ref_h = _align_length(pelvis_z, qpos_ref[:, 2])
        ref_h = np.maximum(ref_h, 1e-6)
    else:
        ref_h = max(float(qpos_seq[0, 2]), 1e-6)
    height_ratio = np.clip(pelvis_z / ref_h, 0.0, 1.0)
    height_term = float(np.mean((1.0 - height_ratio) ** 2))

    qw, qx, qy, qz = qpos_seq[:, 3], qpos_seq[:, 4], qpos_seq[:, 5], qpos_seq[:, 6]
    up_z = 2.0 * (qy * qz + qw * qx)
    tilt_term = float(np.mean((1.0 - up_z) ** 2))

    return height_term + tilt_term


def _com_support_penalty(com_xy, foot_pos):
    """CoM_xy distance outside the grounded-foot support base -- a static
    (non-ZMP) stability proxy: while at least one foot is near the ground,
    the CoM should stay roughly over it/the segment between both grounded
    feet. Frames where NEITHER foot is grounded (flight/jump) are excluded
    entirely rather than penalized, since there's no legitimate "support
    base" to compare against there."""
    l_xy, r_xy = foot_pos["L_Toe"][:, :2], foot_pos["R_Toe"][:, :2]
    l_grounded = foot_pos["L_Toe"][:, 2] < FOOT_CONTACT_HEIGHT
    r_grounded = foot_pos["R_Toe"][:, 2] < FOOT_CONTACT_HEIGHT

    both = l_grounded & r_grounded
    only_l = l_grounded & ~r_grounded
    only_r = r_grounded & ~l_grounded

    dist = np.zeros(com_xy.shape[0])
    if both.any():
        dist[both] = _point_segment_dist_xy(com_xy[both], l_xy[both], r_xy[both])
    if only_l.any():
        dist[only_l] = np.linalg.norm(com_xy[only_l] - l_xy[only_l], axis=-1)
    if only_r.any():
        dist[only_r] = np.linalg.norm(com_xy[only_r] - r_xy[only_r], axis=-1)

    grounded_any = both | only_l | only_r
    if not grounded_any.any():
        return 0.0
    return float(np.mean(dist[grounded_any] ** 2))


def _foot_slide_penalty(foot_pos, dt):
    """Horizontal speed of a foot while it's grounded -- discourages the
    model from learning to "skate" a planted foot instead of holding it
    still, a physically-implausible way to satisfy other loss terms."""
    total = 0.0
    for name in ("L_Toe", "R_Toe"):
        pos = foot_pos[name]
        grounded = pos[:, 2] < FOOT_CONTACT_HEIGHT
        vel_xy = np.diff(pos[:, :2], axis=0) / dt
        grounded_step = grounded[:-1] & grounded[1:]
        if grounded_step.any():
            total += float(np.mean(np.sum(vel_xy[grounded_step] ** 2, axis=-1)))
    return total / 2.0


def _penetration_penalty(foot_pos):
    total = 0.0
    for name in ("L_Toe", "R_Toe"):
        total += float(np.mean(np.clip(-foot_pos[name][:, 2], 0, None) ** 2))
    return total / 2.0


def _smoothness_penalty(qpos_seq, dt):
    """Mean squared second-difference of the actuated joint angles --
    approximates angular acceleration/jerk. Large values correspond to
    abrupt, unstable-looking motion that's also more likely to destabilize
    balance than a smooth trajectory achieving the same pose sequence."""
    joints = qpos_seq[:, 7:]
    if joints.shape[0] < 3:
        return 0.0
    accel = np.diff(joints, n=2, axis=0) / (dt ** 2)
    return float(np.mean(accel ** 2))


def physics_penalty(model, qpos_seq: np.ndarray, weights: dict = None,
                    dt: float = DEFAULT_DT, qpos_ref: np.ndarray = None):
    """Kinematics-only physical-plausibility cost, computed entirely from a
    single rollout's own qpos (no reference trajectory needed) via forward
    kinematics -- everything here is a proxy for "is this rollout headed
    toward falling over", broken into terms that (unlike a single scalar
    reward) can be individually inspected/reweighted:

      limit        joint-range violation
      fall         pelvis height + torso tilt (graded, not a hard threshold).
                   With qpos_ref, the height half is referenced to the
                   retargeted clip's own per-frame pelvis height instead of
                   the rollout's first frame -- see _fall_penalty.
      com_support  CoM_xy vs. the grounded-foot support base (static
                   stability proxy -- see _com_support_penalty for why this
                   is NOT full ZMP: no CoM acceleration term, just position.
                   ZMP would additionally need CoM ddot, which is a further
                   finite-difference of an already-finite-differenced
                   quantity -- noisy from a single qpos rollout -- and is a
                   natural follow-up if this static proxy proves too weak a
                   signal in practice.)
      foot_slide   grounded-foot horizontal speed (penalizes "skating").
                   DISABLED -- still computed and still returned in terms, but
                   weighted 0 in the total. See PHYS_DISABLED_TERMS for why.
      penetrate    foot clipping below the floor
      smooth       joint angular-acceleration proxy (discourages jerky output)

    Deliberately excludes any FK-vs-reference reconstruction term (that is
    L_align's job) and any closed-loop/RL-style disturbance-recovery signal
    (would need actual perturbations injected during rollout, out of scope for
    this kinematics-only cost).

    qpos_ref, when given, is used by ONE term and for one purpose: to tell the
    fall term what height this clip is supposed to be at. It is not a
    reconstruction target and no other term sees it, so passing it does not
    turn L_phys into a second L_align. Omitting it (train_explore.py, which has
    no reference by construction) keeps the original self-referenced behaviour.
    The tilt half of fall is NOT referenced: on the 240 ground clips it is the
    larger of the two (0.749 vs 0.352 on the reference itself), so referencing
    the height alone does not make ground motion score clean -- that is a
    separate change, not silently folded in here.

    weights: dict with a subset/all of PHYS_DEFAULT_WEIGHTS's keys; missing
    keys fall back to the default. Returns (total, terms) -- terms is a
    dict of the unweighted per-component values, for logging/diagnosis
    (mirrors functional_equivalence's (total, terms) return).
    """
    w = dict(PHYS_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    foot_pos, _ = kin.batch_forward_pose(model, qpos_seq, kin.FOOT_BODIES)
    com = kin.batch_com(model, qpos_seq)

    terms = {
        "limit": _joint_limit_penalty(model, qpos_seq),
        "fall": _fall_penalty(qpos_seq, qpos_ref),
        "com_support": _com_support_penalty(com[:, :2], foot_pos),
        "foot_slide": _foot_slide_penalty(foot_pos, dt),
        "penetrate": _penetration_penalty(foot_pos),
        "smooth": _smoothness_penalty(qpos_seq, dt),
    }
    total = sum(w[k] * v for k, v in terms.items())
    return total, terms
