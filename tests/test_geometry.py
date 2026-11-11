"""Tests of crossenbodiment/geometry.py (OOP refactor step 3, docs/refactor/PLAN.md): every shared function against VERBATIM
copies of the code it replaced (frozen below, from commit a997a23), on fixed random inputs, compared exactly (np.array_equal /
==, so NaN or a last-bit difference fails). Also: the old names still exist and are the shared functions, and the copies kept
apart on purpose really differ (so merging them would have changed bits).

Run standalone (no pytest in the environment), like tests/test_losses.py:
    uv run python tests/test_geometry.py
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from crossenbodiment import geometry as G  # noqa: E402

ADULT_XML = REPO / "assets/robots/adult/robot.xml"
CHILD_XML = REPO / "assets/robots_torque/child/robot_torque_full.xml"
G1_XML = REPO / "assets/robots/g1/scene.xml"
RNG = lambda k=0: np.random.default_rng(20261111 + k)                       # noqa: E731


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


# ------------------------------------------------------------------ the old copies, verbatim (a997a23)

def old_project_z(z: np.ndarray) -> np.ndarray:                      # scripts/single_z_search.py
    a = np.atleast_2d(z)
    r = np.sqrt(a.shape[-1])
    out = a * (r / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12))
    return out.reshape(z.shape)


def old_rank_normalize(f: np.ndarray) -> np.ndarray:                 # scripts/single_z_search.py
    order = np.argsort(np.argsort(f))
    return order / max(len(f) - 1, 1) - 0.5


def old_ang_pair(a, b):                                              # scripts/single_z_search.py
    import numpy as _np
    return float(_np.degrees(_np.arccos(_np.clip(
        a @ b / (_np.linalg.norm(a) * _np.linalg.norm(b)), -1, 1))))


def old_ang(a, b):                                                   # scripts/single_z_search.py main()
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


old_unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731  lso_search.py / robot_lso_search.py


def old_lso_ranks(R, n):                                             # lso_search.py / robot_lso_search.py --rank (a.n = n)
    return np.argsort(np.argsort(R, axis=0), axis=0) / (n - 1) - 0.5


def old_ld(v):                                                       # plot_*.py ld(), multibody_corr.py, cost_correlation.py, ...
    return 16 * v / np.linalg.norm(v)


def old_quat_mul(a, b):                                              # model/losses.py _quat_mul (= model/phc_reward.py qmul)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def old_qmul(a, b):                                                  # model/phc_reward.py qmul
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def old_root_heading_xy(fk, qs):                                     # single_z_search.py (also robot_lso_search / g1_z_search heading_xy)
    import humenv.utils as hu
    flat = qs.reshape(-1, qs.shape[-1]); dk = mujoco.MjData(fk); h = np.empty(len(flat))
    for i, qq in enumerate(flat):
        dk.qpos[:] = qq; mujoco.mj_kinematics(fk, dk)
        h[i] = hu.calc_heading(hu.remove_base_rot(dk.xquat[1][None].copy(), "smpl"))[0]
    return h.reshape(qs.shape[:-1]), qs[..., :2]


def old_lso_heading(fk, dk, q):                                      # lso_search.py heading
    import humenv.utils as hu
    flat = q.reshape(-1, q.shape[-1]); Q = np.empty((len(flat), 4))
    for i, qq in enumerate(flat):
        dk.qpos[:] = qq; mujoco.mj_kinematics(fk, dk); Q[i] = dk.xquat[1]
    base = np.repeat(hu.quat_conjugate(np.array([[0.5, 0.5, 0.5, 0.5]])), len(Q), axis=0)   # hu.remove_base_rot, batched
    return hu.calc_heading(hu.quat_mul(Q, base)).reshape(q.shape[:-1])


def old_robot_z_heading_xy(fk, dk, qs):                              # robot_z_search.py heading_xy
    import humenv.utils as hu
    base_rot = hu.quat_conjugate(np.array([[0.5, 0.5, 0.5, 0.5]]))     # hu.remove_base_rot's, SMPL
    flat = qs.reshape(-1, qs.shape[-1]); r = np.empty((len(flat), 4))
    for i, qq in enumerate(flat):
        dk.qpos[:] = qq; mujoco.mj_kinematics(fk, dk); r[i] = dk.xquat[1]
    return hu.calc_heading(hu.quat_mul(r, np.repeat(base_rot, len(r), axis=0))).reshape(qs.shape[:-1]), qs[..., :2]


def old_eval_heading(d, smpl):                                       # eval_metrics.Body.local, per frame
    import humenv.utils as hu
    if smpl:
        h = hu.calc_heading(hu.remove_base_rot(d.xquat[1][None].copy(), "smpl"))[0]
    else:
        R = d.xmat[1].reshape(3, 3); h = np.arctan2(R[1, 0], R[0, 0])
    return h


def old_robot_yaw(gd, root):                                         # robot_z_search.py / g1_z_search.py g1_points
    R = gd.xmat[root].reshape(3, 3); h = np.arctan2(R[1, 0], R[0, 0])
    return h


def old_facing(q):                                                   # drift_success_child.py, child_topview.py
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)) - np.pi / 2


def old_yaw(q):                                                      # strobe_final_blender.py
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


# ------------------------------------------------------------------ inputs

def latents(k=0, n=64):
    """(n, 256) float64 latents at mixed radii (z0-like, perturbed, tiny), and the same in float32"""
    r = RNG(k); Z = r.standard_normal((n, 256)) * r.uniform(0.01, 30, (n, 1))
    return Z


def qpos_batch(m, k=0, shape=(3, 40)):
    """random poses: root position, a random unit root quaternion, joint angles within +-1.2 rad"""
    r = RNG(k); n = int(np.prod(shape)); Q = np.zeros((n, m.nq))
    Q[:, :3] = r.uniform(-2, 2, (n, 3)); Q[:, 2] = r.uniform(0.2, 1.5, n)
    qt = r.standard_normal((n, 4)); Q[:, 3:7] = qt / np.linalg.norm(qt, axis=1, keepdims=True)
    Q[:, 7:] = r.uniform(-1.2, 1.2, (n, m.nq - 7))
    return Q.reshape(shape + (m.nq,))


def quats(k=0, n=5000):
    r = RNG(k); q = r.standard_normal((n, 4)); return q / np.linalg.norm(q, axis=1, keepdims=True)


# ------------------------------------------------------------------ tests

def test_project_z():
    Z = latents(0)
    assert same(G.project_z(Z), old_project_z(Z))
    for z in Z[:16]:
        assert same(G.project_z(z), old_project_z(z))
    Z32 = Z.astype(np.float32)
    assert same(G.project_z(Z32), old_project_z(Z32))
    assert same(G.project_z(np.zeros(256)), old_project_z(np.zeros(256)))          # the 1e-12 guard
    big = np.concatenate([Z + 0.05 * latents(1), Z - 0.05 * latents(1)])           # the [z + s e, z - s e] batch
    assert same(G.project_z(big), old_project_z(big))


def test_sphere16_and_unit_sphere():
    Z = latents(2)
    for z in Z:
        assert same(G.sphere16(z), old_ld(z)) and same(G.sphere16(z.astype(np.float32)), old_ld(z.astype(np.float32)))
        for x in (z, z.astype(np.float32)):                                        # mse_vs_align_test.py: 16 = 2^4 scales exactly
            assert same(G.sphere16(x), 16 * (lambda v: v / np.linalg.norm(v))(x))
    R = RNG(3).standard_normal((4000, 256))
    assert same(G.sphere16_rows(R), 16 * R / np.linalg.norm(R, axis=1, keepdims=True))      # plot_pair_distances.py
    X = RNG(4).standard_normal((17, 40, 256))
    assert same(G.unit_sphere(X), old_unit(X)) and same(G.unit_sphere(X[0, 0]), old_unit(X[0, 0]))
    # kept apart: project_z (row-norm reduction, v * (16 / |v|)) and sphere16 ((16 v) / dot-norm) differ in the last bit
    differ = sum(not np.array_equal(G.project_z(z), G.sphere16(z)) for z in Z)
    assert differ > 0, "project_z and sphere16 never differ: they could be merged"


def test_ang_pair():
    Z = latents(5)
    for a, b in zip(Z[:-1], Z[1:]):
        v = G.ang_pair(a, b)
        assert v == old_ang_pair(a, b) == old_ang(a, b) and type(v) is float
    assert G.ang_pair(Z[0], Z[0]) == old_ang_pair(Z[0], Z[0])                       # the clip at 1
    assert G.ang_pair(Z[0], -Z[0]) == old_ang_pair(Z[0], -Z[0])


def test_rank_normalize():
    r = RNG(6)
    for f in (r.standard_normal(16), r.standard_normal(32), r.integers(0, 4, 16).astype(float), np.array([1.0]), np.array([3.0, 3.0])):
        assert same(G.rank_normalize(f), old_rank_normalize(f))
    for n, S in ((16, 300), (16, 151), (2, 5), (8, 1)):
        R = r.standard_normal((n, S)); assert same(G.rank_normalize_steps(R), old_lso_ranks(R, n))
        Ri = r.integers(0, 3, (n, S)).astype(float); assert same(G.rank_normalize_steps(Ri), old_lso_ranks(Ri, n))   # ties


def test_quat_mul():
    a, b = quats(7), quats(8)
    assert same(G.quat_mul(a, b), old_quat_mul(a, b)) and same(G.quat_mul(a, b), old_qmul(a, b))
    base = np.array([0.5, -0.5, -0.5, -0.5])                                        # losses.root_heading's broadcast
    assert same(G.quat_mul(a, base), old_quat_mul(a, base))
    A, B = a.reshape(50, 100, 4), b.reshape(50, 100, 4)                              # phc_reward: (T, nb, 4)
    assert same(G.quat_mul(A, B), old_qmul(A, B))
    from model import losses, phc_reward
    assert losses._quat_mul is G.quat_mul and phc_reward.qmul is G.quat_mul


def test_heading_humenv():
    import humenv.utils as hu
    assert same(G.SMPL_BASE_CONJ, hu.quat_conjugate(np.array([[0.5, 0.5, 0.5, 0.5]])))
    Q = quats(9, 20000)
    one = np.array([hu.calc_heading(hu.remove_base_rot(q[None].copy(), "smpl"))[0] for q in Q])
    base = np.repeat(hu.quat_conjugate(np.array([[0.5, 0.5, 0.5, 0.5]])), len(Q), axis=0)
    assert same(G.heading_humenv(Q), hu.calc_heading(hu.quat_mul(Q, base)))
    assert all(G.heading_humenv_one(q) == o for q, o in zip(Q[:2000], one[:2000]))
    assert same(G.heading_humenv(Q), one)            # the two call patterns agree here too (they are still kept as two)


def test_smpl_heading():
    for xml in (ADULT_XML, CHILD_XML):
        m = mujoco.MjModel.from_xml_path(str(xml)); d = mujoco.MjData(m)
        Q = qpos_batch(m, 10, (3, 40)); h_old, xy_old = old_root_heading_xy(m, Q)
        assert same(G.smpl_heading(m, Q), h_old) and same(G.smpl_heading(m, Q, d), h_old)
        assert same(G.smpl_heading(m, Q, d, batched=True), old_lso_heading(m, mujoco.MjData(m), Q))
        assert same(G.smpl_heading(m, Q, d, batched=True), old_robot_z_heading_xy(m, mujoco.MjData(m), Q)[0])
        assert same(G.smpl_heading(m, Q[0]), old_root_heading_xy(m, Q[0])[0])                       # (T, nq)
        Q32 = Q.astype(np.float32)                                                                  # saved rollouts are float32
        assert same(G.smpl_heading(m, Q32), old_root_heading_xy(m, Q32)[0])
        for q in Q.reshape(-1, m.nq)[:50]:                                                          # eval_metrics.Body.local
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            assert G.heading_humenv_one(d.xquat[1]) == old_eval_heading(d, True)


def test_robot_yaw():
    m = mujoco.MjModel.from_xml_path(str(G1_XML)); d = mujoco.MjData(m)
    root = m.body("pelvis").id
    for q in qpos_batch(m, 11, (200,)):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        y = G.robot_yaw(d.xmat[root])
        assert y == old_robot_yaw(d, root) and type(y) is type(old_robot_yaw(d, root))
        assert G.robot_yaw(d.xmat[1]) == old_eval_heading(d, False)


def test_root_yaw_and_facing():
    m = mujoco.MjModel.from_xml_path(str(CHILD_XML))
    Q = qpos_batch(m, 12, (500,)); Q32 = Q.astype(np.float32)
    assert same(G.facing(Q), old_facing(Q)) and same(G.facing(Q32), old_facing(Q32))
    assert same(G.root_yaw(Q), old_yaw(Q))
    import child_topview
    import drift_success_child
    assert child_topview.facing is G.facing and same(drift_success_child.facing(Q), old_facing(Q))


def test_old_names_delegate():
    import single_z_search as S
    assert S.project_z is G.project_z and S.rank_normalize is G.rank_normalize and S.ang_pair is G.ang_pair


# --------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}")
        except (AssertionError, KeyError, ValueError, OSError, ImportError) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
