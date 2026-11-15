"""robot_rollout.py -- the robot rollouts' actor call and motion summary (scripts/g1_z0_rollout.py act / motion until OOP refactor
step 4, unchanged).

robot_z_search.py, robot_z0.py, robot_feasibility_compare.py (and the G1 line: g1_z_search.py, g1_feasibility_compare.py)
imported these from g1_z0_rollout.py, which pulled model/g1_env.py, model/g1_map.py, model/bfm_align.py and single_z_search.py
in at import; they import them from here. The G1 line (g1_z0_rollout.py included, which imported them back) was archived in
step 7: the G1 goes through robot_z_search.py / robot_z0.py / robot_feasibility_compare.py --robot g1.

  act(model, obs, z, device, batch=16)   the frozen actor's mean action for one observation, computed in a batch of `batch`
                                         identical rows (the GPU's last bits depend on the batch size; the robot rollouts are
                                         chaotic enough for that to show, so keep batch as the caller had it)
  motion(q, ref)                         fallen frames, first fall, travel, end-point distance, pelvis height of a robot rollout
                                         against its target (both (T, nq) qpos with a free root, quaternion w, x, y, z)
  g1_up_z(quat_wxyz)                     world-z of the pelvis's local +Z axis (the robots' up; SMPL assets use local +Y)
"""
import numpy as np
import torch


def act(model, obs, z, device, batch=16):
    """the actor's mean action, computed in a batch of `batch` identical rows as g1_z_search did: the GPU's result
    depends on the batch size in the last bits, and the G1 rollout is chaotic enough for that to show"""
    o = torch.as_tensor(np.repeat(obs[None], batch, 0), dtype=torch.float32, device=device)
    with torch.no_grad():
        return model._actor(model._normalize(o), z.expand(batch, -1), model.cfg.actor_std).mean[0].cpu().numpy()


def g1_up_z(quat_wxyz):
    """World-z component of the G1 pelvis's up axis, its local +Z (model.bodies.up_z is the SMPL asset's, local +Y)."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    return float(1.0 - 2.0 * (x * x + y * y))


def motion(q, ref):
    T = min(len(q), len(ref)); q, ref = q[:T], ref[:T]
    u = np.array([g1_up_z(x) for x in q[:, 3:7]]); ur = np.array([g1_up_z(x) for x in ref[:, 3:7]])
    down = (u < ur - 0.8) | (q[:, 2] < 0.5 * ref[:, 2])
    return dict(fallen=int(down.sum()), first_fall=int(np.argmax(down)) if down.any() else None,
                travel=float(np.linalg.norm(q[-1, :2] - q[0, :2])), ref_travel=float(np.linalg.norm(ref[-1, :2] - ref[0, :2])),
                end_xy=float(np.linalg.norm(q[-1, :2] - ref[-1, :2])),
                pelvis=float(q[:, 2].mean()), ref_pelvis=float(ref[:, 2].mean()))
