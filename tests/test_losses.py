"""Regression tests for the three defects fixed in model/losses.py.

All three were the same kind of failure -- the loss did not measure what it
claimed -- so each test pins an INVARIANT the loss must satisfy, not a number it
happened to produce. A number would have to be re-blessed after every retune;
an invariant survives it.

Run standalone (no pytest dependency in pyproject.toml):
    uv run tests/test_losses.py
or under pytest if it is installed:
    uv run pytest tests/test_losses.py -v
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np

from model import losses

XML = str(REPO_ROOT / "assets" / "robots" / "adult" / "robot.xml")
WEIGHTS = {"root": 1.0, "ee": 1.0, "contact": 1.0, "pose": 1.0, "velocity": 1.0}

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_path(XML)
    return _MODEL


def yaw_quat(yaw):
    """(T,) yaw -> (T, 4) wxyz, composed onto this asset's Y-up rest pose.

    The free joint's rest quaternion is (0.70710678, 0.70710678, 0, 0), which
    maps local +Y to world +Z (see _fall_penalty's docstring). A yaw about world
    z is applied on the LEFT of that, so the trajectory stays upright and
    quat_to_yaw reads back the yaw we put in.
    """
    rest = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    out = np.empty((len(yaw), 4))
    for t, a in enumerate(yaw):
        z = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
        mujoco.mju_mulQuat(out[t], z, rest)
    return out


def traj(T, *, yaw=None, xy_speed=0.0, dt=None, jitter=0.0, seed=0):
    """A rigid-root trajectory: straight along +x at xy_speed, heading `yaw`."""
    dt = losses.DEFAULT_DT if dt is None else dt
    m = model()
    q = np.tile(m.qpos0, (T, 1)).astype(np.float64)
    t = np.arange(T) * dt
    q[:, 0] = xy_speed * t
    if yaw is None:
        yaw = np.zeros(T)
    q[:, 3:7] = yaw_quat(yaw)
    if jitter:
        rng = np.random.default_rng(seed)
        q[:, 7:] += jitter * rng.standard_normal(q[:, 7:].shape)
    return q


# --------------------------------------------------------------------------
# defect 1: heading unwrapped each trajectory separately before subtracting
# --------------------------------------------------------------------------

def test_heading_is_bounded_under_opposite_spin():
    """Two trajectories spinning opposite ways must not diverge without bound.

    This is the rotate-z-5 / rotate-z--5 case. Separately unwrapping made the
    heading error grow QUADRATICALLY in the number of turns (measured 578 on
    real clips); a periodic heading metric cannot exceed 2 per frame no matter
    how many turns separate the two.
    """
    T = 300
    # both trajectories are stationary in xy, so curvature and travel are 0 and
    # d_root is heading + yaw_rate. yaw_rate grows only like turns^2 / T^2, which
    # at 8 turns over 300 frames is ~0.11 -- everything above that is heading.
    for turns in (1, 2, 4, 8):
        a = traj(T, yaw=np.linspace(0, turns * 2 * np.pi, T))
        b = traj(T, yaw=np.linspace(0, -turns * 2 * np.pi, T))
        d = losses.d_root(model(), a, b)
        assert d < 2.5, f"{turns} opposite turns -> d_root {d}"


def test_heading_ignores_the_branch_cut():
    """Two trajectories 0.02 rad apart must score ~0.02 rad apart, even when
    they straddle +-pi and cross it on different frames.

    That crossing is what made separate unwrapping put them on different
    branches: a true error of 0.01 rad read as 6.29.
    """
    T = 400
    # quat_to_yaw returns arctan2, i.e. (-pi, pi]. Put a just below the cut and b
    # just above it: 0.02 rad apart physically, but b reads back as ~-pi and a as
    # ~+pi, and unwrap has nothing to correct because each is constant on its own
    # side. Separate unwrapping therefore reports a constant 2pi - 0.02 gap.
    a = traj(T, yaw=np.full(T, np.pi - 0.01))
    b = traj(T, yaw=np.full(T, np.pi + 0.01))

    d = losses.d_root(model(), a, b)
    # 1 - cos(0.02) ~ 2e-4; the separate-unwrap answer was (2pi - 0.02)^2 ~ 39
    assert d < 0.1, f"d_root {d} -- branch artifact leaked in"


def test_heading_is_periodic():
    """Adding whole turns to the relative heading must change nothing.

    A guard, not a detector: 2pi*k lands on the same quaternion, so the old code
    passed this too. It pins the property the fix is supposed to have.
    """
    T = 200
    a = traj(T, yaw=np.full(T, 0.3))
    for k in (0, 1, -1, 3):
        b = traj(T, yaw=np.full(T, 0.3 + 2 * np.pi * k))
        assert losses.d_root(model(), a, b) < 1e-9, f"k={k}"


# --------------------------------------------------------------------------
# defect 2: no translation term, so d_root was blind to speed
# --------------------------------------------------------------------------

def test_same_shape_different_speed_is_not_free():
    """The move-ego-* family varies ONLY in speed. Two straight walks at 1 and
    2 m/s have identical heading (0) and identical curvature (0), so before the
    travel term d_root scored them as identical."""
    T = 300
    slow = traj(T, xy_speed=1.0)
    fast = traj(T, xy_speed=2.0)

    d = losses.d_root(model(), fast, slow)
    assert d > 1e-3, f"d_root {d} -- speed is invisible again"

    # and it must be monotone in the speed gap
    closer = losses.d_root(model(), traj(T, xy_speed=1.2), slow)
    assert closer < d


def test_travel_term_can_be_switched_off():
    """ROOT_TRAVEL_WEIGHT = 0 must reproduce the pre-travel d_root exactly, so
    every number recorded before this term existed stays reproducible. A guard,
    not a detector."""
    T = 300
    a, b = traj(T, xy_speed=2.0), traj(T, xy_speed=1.0)
    saved = losses.ROOT_TRAVEL_WEIGHT
    try:
        losses.ROOT_TRAVEL_WEIGHT = 0.0
        assert losses.d_root(model(), a, b) < 1e-9
    finally:
        losses.ROOT_TRAVEL_WEIGHT = saved


def test_speed_error_is_in_body_lengths():
    """The same absolute speed error must cost the same on a big body and a
    small one -- that is what Froude normalization is for. On raw m/s the giant
    would be scored identically to the child despite covering the gap in far
    fewer body lengths."""
    T = 200
    child = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / "assets" / "robots" / "child" / "robot.xml"))
    giant = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / "assets" / "robots" / "giant" / "robot.xml"))

    def gap(m):
        q = np.tile(m.qpos0, (T, 1)).astype(np.float64)
        t = np.arange(T) * losses.DEFAULT_DT
        a, b = q.copy(), q.copy()
        a[:, 0], b[:, 0] = 2.0 * t, 1.0 * t
        return losses.d_root(m, a, b)

    # same 1 m/s gap costs the SHORT body more, because it is more body lengths
    assert gap(child) > gap(giant), f"child {gap(child)} giant {gap(giant)}"


# --------------------------------------------------------------------------
# defect 3: dt defaulted to 1.0, distorting each term by 30^(-2k)
# --------------------------------------------------------------------------

def test_default_dt_is_the_real_control_step():
    assert abs(losses.DEFAULT_DT - 1.0 / 30.0) < 1e-12


def test_physics_penalty_is_fps_invariant():
    """The same physical motion sampled at 30 and 60 fps must score the same.

    Built from a closed-form q(t) at both rates rather than by resampling, so
    interpolation error is not what is being measured. With a hardwired dt = 1
    the 60 fps version scored foot_slide 4x lower and smooth 16x lower, silently.
    """
    m = model()

    def clip(fps, seconds=4.0):
        T = int(fps * seconds)
        t = np.arange(T) / fps
        q = np.tile(m.qpos0, (T, 1)).astype(np.float64)
        q[:, 0] = 0.4 * np.sin(2 * np.pi * 0.5 * t)          # root sway
        q[:, 7:] += 0.15 * np.sin(2 * np.pi * 0.8 * t)[:, None]
        return q

    # 30 fps goes through the DEFAULT, which is the path every caller in the repo
    # takes -- that default being 1.0 was the whole defect.
    _, t30 = losses.physics_penalty(m, clip(30))
    _, t60 = losses.physics_penalty(m, clip(60), dt=1.0 / 60)
    for k in ("foot_slide", "smooth"):
        lo, hi = t30[k], t60[k]
        if max(lo, hi) < 1e-12:
            continue
        assert abs(lo - hi) / max(lo, hi) < 0.05, f"{k}: 30fps {lo} vs 60fps {hi}"


def test_d_velocity_is_fps_invariant():
    T30, T60 = 120, 240
    m = model()

    def pair(T, fps):
        t = np.arange(T) / fps
        a = np.tile(m.qpos0, (T, 1)).astype(np.float64)
        b = a.copy()
        a[:, 7:] += 0.2 * np.sin(2 * np.pi * 0.7 * t)[:, None]
        b[:, 7:] += 0.2 * np.sin(2 * np.pi * 0.7 * t + 0.3)[:, None]
        return a, b

    v30 = losses.d_velocity(*pair(T30, 30))            # default dt
    v60 = losses.d_velocity(*pair(T60, 60), dt=1.0 / 60)
    assert abs(v30 - v60) / max(v30, v60) < 0.05, f"30fps {v30} vs 60fps {v60}"


def test_weight_migration_preserves_the_old_objective():
    """PHYS_DEFAULT_WEIGHTS was divided by 30^(2k) exactly as dt was fixed, so
    today's cost equals what the dt=1.0 code produced. This pins that the bug
    fix and the retune stayed separate.

    Terms in PHYS_DISABLED_TERMS are excluded from BOTH sides: switching a term
    off is a deliberate change to what is optimized, not a distortion of it, and
    folding it in here would make this test fail for the one reason it is not
    meant to catch."""
    m = model()
    T = 200
    rng = np.random.default_rng(3)
    q = np.tile(m.qpos0, (T, 1)).astype(np.float64)
    q[:, 7:] += 0.1 * rng.standard_normal((T, q.shape[1] - 7))

    _, terms = losses.physics_penalty(m, q)
    new = sum(losses.PHYS_DEFAULT_WEIGHTS[k] * v for k, v in terms.items())

    old_weights = {"limit": 1.0, "fall": 5.0, "com_support": 1.0,
                   "foot_slide": 1.0, "penetrate": 1.0, "smooth": 0.01}
    old_weights = {k: (0.0 if k in losses.PHYS_DISABLED_TERMS else v)
                   for k, v in old_weights.items()}
    _, old_terms = losses.physics_penalty(m, q, weights=old_weights, dt=1.0)
    old = sum(old_weights[k] * v for k, v in old_terms.items())

    assert abs(new - old) <= 1e-6 * max(abs(old), 1.0), f"{new} vs {old}"


def test_disabled_terms_are_still_reported():
    """Weighted 0, not deleted -- the diagnostic has to survive the switch-off,
    otherwise terms.csv silently loses a column and re-enabling the term means
    re-implementing it."""
    m = model()
    T = 60
    rng = np.random.default_rng(5)
    q = np.tile(m.qpos0, (T, 1)).astype(np.float64)
    q[:, 7:] += 0.1 * rng.standard_normal((T, q.shape[1] - 7))

    total, terms = losses.physics_penalty(m, q)
    for k in losses.PHYS_DISABLED_TERMS:
        assert k in terms, f"{k} vanished from the terms dict"
        assert losses.PHYS_DEFAULT_WEIGHTS[k] == 0.0
        assert losses.PHYS_EQUAL_WEIGHTS[k] == 0.0
    # and it really is out of the total
    assert abs(total - sum(losses.PHYS_DEFAULT_WEIGHTS[k] * v
                           for k, v in terms.items())) < 1e-12


# --------------------------------------------------------------------------
# the per-frame discount
# --------------------------------------------------------------------------

def test_discount_1_is_exactly_the_plain_mean():
    """The default must not perturb anything recorded before it existed --
    equal to np.mean bit for bit, not merely close."""
    m = model()
    rng = np.random.default_rng(11)
    T = 200
    a = np.tile(m.qpos0, (T, 1)).astype(np.float64)
    b = a.copy()
    a[:, 7:] += 0.1 * rng.standard_normal((T, a.shape[1] - 7))
    b[:, 7:] += 0.1 * rng.standard_normal((T, b.shape[1] - 7))
    b[:, :3] += 0.05 * rng.standard_normal((T, 3))
    w = {"root": 1.0, "ee": 1.0, "contact": 1.0, "pose": 1.0, "velocity": 1.0}

    plain, pt = losses.functional_equivalence(m, a, b, w)
    same, st = losses.functional_equivalence(m, a, b, w, discount=1.0)
    assert plain == same, f"{plain} vs {same}"
    for k in pt:
        assert pt[k] == st[k], f"{k}: {pt[k]} vs {st[k]}"


def test_discount_moves_weight_off_the_tail():
    """A trajectory that is clean early and bad late must score LOWER under a
    discount, and the mirror image must score higher -- that asymmetry is the
    whole point, so a discount that merely rescaled would pass a same-value
    check while doing nothing."""
    m = model()
    T = 200
    late = np.tile(m.qpos0, (T, 1)).astype(np.float64)
    ref = late.copy()
    late[T // 2:, 7:] += 0.3                      # error only in the tail
    early = np.tile(m.qpos0, (T, 1)).astype(np.float64)
    early[:T // 2, 7:] += 0.3                     # error only at the start

    flat_late = losses.d_pose(late, ref)
    flat_early = losses.d_pose(early, ref)
    assert abs(flat_late - flat_early) < 1e-12, "uniform mean cannot tell them apart"

    g = 0.98
    assert losses.d_pose(late, ref, discount=g) < flat_late
    assert losses.d_pose(early, ref, discount=g) > flat_early
    assert losses.d_pose(late, ref, discount=g) < losses.d_pose(early, ref, discount=g)


def test_discount_keeps_the_scale():
    """Normalised by the weights' sum, so a constant error scores the same
    whatever gamma is -- otherwise lambda_align would need retuning per gamma."""
    m = model()
    T = 150
    ref = np.tile(m.qpos0, (T, 1)).astype(np.float64)
    const = ref.copy()
    const[:, 7:] += 0.2
    base = losses.d_pose(const, ref)
    for g in (0.999, 0.99, 0.95, 0.9):
        assert abs(losses.d_pose(const, ref, discount=g) - base) < 1e-9, g


# --------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
