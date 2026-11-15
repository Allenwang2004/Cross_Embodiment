"""dtw.py -- time-aligned comparison of a rollout with its reference that lets the rollout be SLOWER, never faster.

Each rollout frame i is matched to one reference frame j(i): j(0) = 0, j advances by 0 or 1 per rollout frame and
never stays twice in a row, so the rollout runs at 0.5x to 1x the reference's pace. The rollout may stop matching at
any frame (what it does after that is ignored) and need not reach the reference's end. Scored per REFERENCE frame:
sim(j) = mean similarity of the rollout frames matched to j (1 or 2 of them), a reference frame never reached counts
0, score = sum_j sim(j) / R. The path maximises the score (exact dynamic programme).

With a rollout twice the reference's length even the slowest pace (0.5x) can reach the end, so being slower is not
penalised by itself; it cannot catch up by skipping reference frames either (no steps of 2), so "done" and "pace"
report the pace it really kept.
"""
import numpy as np


def align_slow(S):
    """S (n, T, R) similarity of every rollout / reference frame pair -> score (n,), j_end (n,) the last reference
    frame reached, i_end (n,) the rollout frame that reached it, path (n, T) reference frame per rollout frame (-1
    after i_end)."""
    n, T, R = S.shape
    NEG = -1e18
    V = np.full((n, R, 2), NEG); V[:, 0, 0] = S[:, 0, 0]          # [j, s]: s = 1 if frame i is the 2nd on j
    bp = np.zeros((T, n, R), dtype=bool)                          # advanced into j from (j - 1, s = 1)
    best = S[:, 0, 0].astype(np.float64).copy()
    bi, bj, bs = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    for i in range(1, T):
        Vn = np.full((n, R, 2), NEG)
        a0, a1 = V[:, :-1, 0], V[:, :-1, 1]
        take1 = a1 > a0
        Vn[:, 1:, 0] = np.where(take1, a1, a0) + S[:, i, 1:]
        bp[i, :, 1:] = take1
        Vn[:, :, 1] = V[:, :, 0] + (S[:, i, :] - S[:, i - 1, :]) / 2  # j's similarity becomes the mean of 2 frames
        V = Vn
        flat = V.reshape(n, -1); k = flat.argmax(1); v = flat[np.arange(n), k]
        up = v > best
        best = np.where(up, v, best); bi = np.where(up, i, bi); bj = np.where(up, k // 2, bj); bs = np.where(up, k % 2, bs)
    path = np.full((n, T), -1, dtype=np.int64)
    for r in range(n):
        i, j, s = int(bi[r]), int(bj[r]), int(bs[r])
        while True:
            path[r, i] = j
            if i == 0:
                break
            if s == 1:
                s = 0
            else:
                s = int(bp[i, r, j]); j -= 1
            i -= 1
        assert j == 0
    return best / R, bj, bi, path


def cosine_matrix(Bs, Bg):
    """Bs (n, T, d), Bg (R, d) -> (n, T, R) cosine similarities."""
    a = Bs / (np.linalg.norm(Bs, axis=-1, keepdims=True) + 1e-12)
    b = Bg / (np.linalg.norm(Bg, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T
