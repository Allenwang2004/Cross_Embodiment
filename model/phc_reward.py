"""phc_reward.py -- PHC's motion-imitation reward (Luo et al. 2023, Eq. 2) as a cost, the tracking reward BFM-Zero's
trajectory adaptation uses (Li et al. 2026, Sec. 3.3). Follows PHC's public code (ZhengyiLuo/PHC,
phc/env/tasks/humanoid_im.py: compute_imitation_reward, defaults from HumanoidIm.__init__), which squares and averages
the errors where the paper's Eq. 2 writes unsquared norms:
  r_t = 0.5 exp(-100 mean_b,xyz dp^2) + 0.3 exp(-10 mean_b dtheta^2) + 0.1 exp(-0.1 mean_b,xyz dv^2) + 0.1 exp(-0.1 mean_b,xyz dw^2)
over every body of the model (the world excluded): world-frame body positions p, rotations (dtheta = the angle of
q_ref * q_sim^-1, in [0, pi]), linear velocities v and angular velocities w.
Ours, not PHC's: velocities are finite differences at 30 Hz of the body poses, on both sides alike, with each side's
previous pose before its first frame (the rollout's start state; the reference frame before the first goal); lengths
are divided by `scale` (ExactObs's root scale s, adult-equivalent metres as single_z_search.py's position term) because
PHC's constants are set for an adult. Cost = 1 - mean_t r_t: 0 is a perfect match, 1 matches nothing.
BFM-Zero's few-shot objective also subtracts 0.02 x auxiliary penalties designed for its robot; they are left out."""
import mujoco
import numpy as np

K = dict(pos=100.0, rot=10.0, vel=0.1, ang=0.1)
W = dict(pos=0.5, rot=0.3, vel=0.1, ang=0.1)


def qmul(a, b):
    """Hamilton product of (..., 4) wxyz quaternions"""
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(a):
    return a * np.array([1.0, -1.0, -1.0, -1.0])


def rotvec(q):
    """(..., 4) unit wxyz quaternions -> rotation vectors (..., 3), angle in [0, pi]"""
    q = np.where(q[..., :1] < 0, -q, q); s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., :1])
    return np.where(s > 1e-12, q[..., 1:] / np.maximum(s, 1e-12) * ang, 0.0)


def kinematics(m, prev, qs, dt=1.0 / 30, scale=1.0):
    """prev (nq,) the pose before the first frame, qs (T, nq) -> body positions (T, nb, 3), quaternions (T, nb, 4),
    linear (T, nb, 3) and angular (T, nb, 3) velocities, world frame, lengths / scale"""
    d = mujoco.MjData(m); allq = np.concatenate([np.asarray(prev)[None], qs]); nb = m.nbody - 1
    P = np.empty((len(allq), nb, 3)); Q = np.empty((len(allq), nb, 4))
    for i, q in enumerate(allq):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); P[i] = d.xpos[1:]; Q[i] = d.xquat[1:]
    V = np.diff(P, axis=0) / dt; Wv = rotvec(qmul(Q[1:], qconj(Q[:-1]))) / dt
    return P[1:] / scale, Q[1:], V / scale, Wv


class PHCCost:
    """cost(qs, prev) for rollouts against a fixed reference: goal (T, nq) the frames the states after steps 1..T are
    scored against, goal_prev the reference frame before goal[0]"""

    def __init__(self, m, goal_prev, goal, dt=1.0 / 30, scale=1.0):
        self.m, self.dt, self.scale = m, dt, scale
        self.ref = kinematics(m, goal_prev, goal, dt, scale)

    def parts(self, q, prev):
        """one rollout q (T, nq) from the start state prev -> per-step reward r (T,) and the four terms' mean errors"""
        T = min(len(q), len(self.ref[0]))
        P, Q, V, Wv = kinematics(self.m, prev, q[:T], self.dt, self.scale); Pr, Qr, Vr, Wr = (x[:T] for x in self.ref)
        e = dict(pos=((P - Pr) ** 2).mean(-1).mean(-1), rot=(np.linalg.norm(rotvec(qmul(Qr, qconj(Q))), axis=-1) ** 2).mean(-1),
                 vel=((V - Vr) ** 2).mean(-1).mean(-1), ang=((Wv - Wr) ** 2).mean(-1).mean(-1))
        r = sum(W[k] * np.exp(-K[k] * e[k]) for k in K)
        return r, e

    def __call__(self, qs, prev):
        """qs (n, T, nq) rollouts from the start state prev (nq,) -> cost (n,) = 1 - mean_t r_t"""
        return np.array([1.0 - self.parts(q, prev)[0].mean() for q in qs])
