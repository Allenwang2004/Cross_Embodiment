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
  which is a separate change from the three corrected here.

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


def d_pose(qpos_a: np.ndarray, qpos_b: np.ndarray) -> float:
    a, b = _align_length(qpos_a[:, 7:], qpos_b[:, 7:])
    return float(np.mean((a - b) ** 2))


def d_velocity(qpos_a: np.ndarray, qpos_b: np.ndarray, dt: float = DEFAULT_DT) -> float:
    va = np.diff(qpos_a, axis=0) / dt
    vb = np.diff(qpos_b, axis=0) / dt
    va, vb = _align_length(va, vb)
    return float(np.mean((va - vb) ** 2))


def d_root(model, qpos_a: np.ndarray, qpos_b: np.ndarray,
           dt: float = DEFAULT_DT) -> float:
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
    heading_err = float(np.mean(1.0 - np.cos(ya - yb)))

    # unwrap IS correct here: diff of an unwrapped sequence is the per-frame
    # rotation increment, already confined to [-pi, pi]. It is only the
    # *difference of two independently unwrapped sequences* that is unsound.
    yaw_a = np.unwrap(raw_yaw_a)
    yaw_b = np.unwrap(raw_yaw_b)
    yaw_a, yaw_b = _align_length(yaw_a, yaw_b)

    yaw_rate_a, yaw_rate_b = _align_length(np.diff(yaw_a), np.diff(yaw_b))
    yaw_rate_err = float(np.mean((yaw_rate_a - yaw_rate_b) ** 2)) if len(yaw_rate_a) else 0.0

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
    curv_err = float(np.mean((curv_a[both_valid] - curv_b[both_valid]) ** 2)) if both_valid.any() else 0.0

    L = _leg_length(model)
    v_ref = np.sqrt(GRAVITY * L)
    froude_a = np.linalg.norm(np.diff(root_a[:, :2], axis=0), axis=-1) / dt / v_ref
    froude_b = np.linalg.norm(np.diff(root_b[:, :2], axis=0), axis=-1) / dt / v_ref
    froude_a, froude_b = _align_length(froude_a, froude_b)
    speed_err = float(np.mean((froude_a - froude_b) ** 2)) if len(froude_a) else 0.0
    height_err = float(np.mean(((root_a[:, 2] - root_b[:, 2]) / L) ** 2))
    travel_err = speed_err + height_err

    return heading_err + yaw_rate_err + curv_err + ROOT_TRAVEL_WEIGHT * travel_err


def d_ee(model, qpos_a: np.ndarray, qpos_b: np.ndarray) -> float:
    bodies = kin.EE_BODIES + [kin.ROOT_BODY]
    pos_a, _ = kin.batch_forward_pose(model, qpos_a, bodies)
    pos_b, _ = kin.batch_forward_pose(model, qpos_b, bodies)
    err = 0.0
    for name in kin.EE_BODIES:
        rel_a = pos_a[name] - pos_a[kin.ROOT_BODY]
        rel_b = pos_b[name] - pos_b[kin.ROOT_BODY]
        rel_a, rel_b = _align_length(rel_a, rel_b)
        err += float(np.mean((rel_a - rel_b) ** 2))
    return err / len(kin.EE_BODIES)


def d_contact(model, qpos_a: np.ndarray, qpos_b: np.ndarray) -> float:
    pos_a, _ = kin.batch_forward_pose(model, qpos_a, kin.FOOT_BODIES)
    pos_b, _ = kin.batch_forward_pose(model, qpos_b, kin.FOOT_BODIES)
    err = 0.0
    for name in kin.FOOT_BODIES:
        za, zb = _align_length(pos_a[name][:, 2], pos_b[name][:, 2])
        err += float(np.mean((za - zb) ** 2))
    return err / len(kin.FOOT_BODIES)


def functional_equivalence(model, qpos_beta: np.ndarray, qpos_ref, weights: dict,
                           dt: float = DEFAULT_DT):
    """weights: dict with keys root/ee/contact/pose/velocity.
    qpos_ref may be None (no retargeted reference attached yet for this
    sample) -> returns (0.0, {})."""
    if qpos_ref is None:
        return 0.0, {}
    terms = {
        "root": d_root(model, qpos_beta, qpos_ref, dt),
        "ee": d_ee(model, qpos_beta, qpos_ref),
        "contact": d_contact(model, qpos_beta, qpos_ref),
        "pose": d_pose(qpos_beta, qpos_ref),
        "velocity": d_velocity(qpos_beta, qpos_ref, dt),
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
PHYS_DEFAULT_WEIGHTS = {
    "limit": 1.0,           # joint-range violation                        (k=0)
    "fall": 5.0,            # pelvis height + torso tilt                   (k=0)
    "com_support": 1.0,     # CoM_xy vs. the grounded-foot support base    (k=0)
    "foot_slide": 1.0 / 30 ** 2,      # grounded-foot horizontal speed     (k=1)
    "penetrate": 1.0,       # foot z < 0 -- clipping through the floor     (k=0)
    "smooth": 0.01 / 30 ** 4,         # joint angular-acceleration proxy   (k=2)
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


def _fall_penalty(qpos_seq):
    """Smooth, continuous fall signal (see prior docstring note: a hard
    pelvis-height threshold saturates almost immediately on the child body
    and stops carrying gradient). Two components:
      - height_ratio: clip(pelvis_z / initial_pelvis_z, 0, 1) -> (1-ratio)^2
      - tilt: torso "up" axis vs. world z. NOT the naive
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
    ref_h = qpos_seq[0, 2]
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
                    dt: float = DEFAULT_DT):
    """Kinematics-only physical-plausibility cost, computed entirely from a
    single rollout's own qpos (no reference trajectory needed) via forward
    kinematics -- everything here is a proxy for "is this rollout headed
    toward falling over", broken into terms that (unlike a single scalar
    reward) can be individually inspected/reweighted:

      limit        joint-range violation
      fall         pelvis height + torso tilt (graded, not a hard threshold)
      com_support  CoM_xy vs. the grounded-foot support base (static
                   stability proxy -- see _com_support_penalty for why this
                   is NOT full ZMP: no CoM acceleration term, just position.
                   ZMP would additionally need CoM ddot, which is a further
                   finite-difference of an already-finite-differenced
                   quantity -- noisy from a single qpos rollout -- and is a
                   natural follow-up if this static proxy proves too weak a
                   signal in practice.)
      foot_slide   grounded-foot horizontal speed (penalizes "skating")
      penetrate    foot clipping below the floor
      smooth       joint angular-acceleration proxy (discourages jerky output)

    Deliberately excludes any FK-vs-reference reconstruction term (would
    need a retargeted-motion qpos_ref, which train_explore.py's whole point
    is to NOT depend on) and any closed-loop/RL-style disturbance-recovery
    signal (would need actual perturbations injected during rollout, out of
    scope for this kinematics-only cost).

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
        "fall": _fall_penalty(qpos_seq),
        "com_support": _com_support_penalty(com[:, :2], foot_pos),
        "foot_slide": _foot_slide_penalty(foot_pos, dt),
        "penetrate": _penetration_penalty(foot_pos),
        "smooth": _smoothness_penalty(qpos_seq, dt),
    }
    total = sum(w[k] * v for k, v in terms.items())
    return total, terms
