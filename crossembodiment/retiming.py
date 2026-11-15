"""retiming.py -- a qpos clip played at another speed (scripts/bfm_velocity_check.py resample until OOP refactor step 4, unchanged).

    resample(q, r)   (T, nq) qpos at 30 Hz -> the clip at r x its speed, 30 Hz: frame i is the original's frame i * r; positions
                     and joint angles interpolated linearly, the root quaternion (qpos[3:7], w, x, y, z) by slerp

lso_search.py --init-pace starts from it; bfm_velocity_check.py imports it back.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample(q, r):
    """the clip played at r times its speed, 30 Hz: frame i is the original's frame i * r."""
    f = np.arange(0, len(q) - 1 + 1e-9, r); k = np.minimum(np.floor(f).astype(int), len(q) - 2); w = (f - k)[:, None]
    out = q[k] * (1 - w) + q[k + 1] * w
    rot = Rotation.from_quat(q[:, [4, 5, 6, 3]])                        # qpos quaternion is w, x, y, z
    out[:, 3:7] = Slerp(np.arange(len(q)), rot)(f).as_quat()[:, [3, 0, 1, 2]]
    return out
