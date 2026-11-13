#!/usr/bin/env python3
"""ik_convergence_check.py -- does RobotMap's IK (orientation-only damped least squares) converge, and to the best answer?

On the adult's original clips (the survey's 11), warm-started frame by frame exactly as robot_roundtrip / RobotEnv do:
  stops      every _solve call: iterations used, whether it stopped on its own (|dq|^2 < tol) or ran out (50, or 10 for
             the polish after a limit re-solve), and the last step (deg) when it ran out
  limit      how often a joint ends at a limit, triggering the re-solve from the nominal pose, and how often that wins
  restarts   every 10th frame: 8 extra solves from random poses inside the joint ranges, 200 iterations each; is the best
             of them clearly better (objective = sum of squared link errors, as the IK minimises) than the pipeline's?
Writes outputs/map_error/ik_convergence.json.
usage: uv run python scripts/ik_convergence_check.py
"""
import json, sys, types
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robot_map import RobotMap                    # noqa: E402
from crossembodiment.survey import CLIPS   # noqa: E402  survey_body_videos' (refactor step 4)

LOG = []


def counting_solve(self, Rt, root, q0, iters, damping, tol):
    """RobotMap._solve, line for line, plus a record of how it stopped."""
    q = np.array(q0, dtype=float)
    lo, hi = self.robot_range[:, 0], self.robot_range[:, 1]
    used, stopped, last, moved, obj = iters, False, 0.0, 0.0, []
    for k in range(iters):
        e = self._link_err(Rt, root, q).ravel()
        ax = self.d_robot.xaxis[self.robot_jnt]
        J = (self.anc[:, None, :] * ax.T[None]).reshape(-1, len(self.robot_jnt))
        J, e = J * self.w[:, None], e * self.w
        obj.append(float(e @ e))
        dq = np.linalg.solve(J.T @ J + damping * np.eye(len(q)), J.T @ e)
        q_new = np.clip(q + dq, lo, hi)
        last = float(np.degrees(np.linalg.norm(dq)))                  # the step it asked for
        moved = float(np.degrees(np.linalg.norm(q_new - q)))          # the step it took, after the joint limits
        q = q_new
        if dq @ dq < tol:
            used, stopped = k + 1, True
            break
    e = self._link_err(Rt, root, q).ravel() * self.w; obj.append(float(e @ e))
    # how much the objective still fell over the last 10 iterations, relative (0 = flat)
    tail = (obj[-11] - obj[-1]) / max(obj[-11], 1e-12) if len(obj) > 11 else 0.0
    LOG.append((iters, used, stopped, last, moved, tail))
    return q


def objective(resid_deg):
    return float((np.radians(resid_deg) ** 2).sum())


def main():
    rng = np.random.default_rng(0)
    out = {}
    for key in ("g1", "h1", "apollo"):
        M = RobotMap(key)
        M._solve = types.MethodType(counting_solve, M)
        lo, hi = M.robot_range[:, 0], M.robot_range[:, 1]
        LOG.clear()
        n_frames = n_limit = n_swap = 0
        better, gain_deg, n_rs = 0, [], 0
        for stem in CLIPS:
            ref = np.load(REPO / f"data/origin_motion/{stem.rsplit('_', 1)[0]}/{stem}.npz")["qpos"]
            q = None
            for t in range(len(ref)):
                k0 = len(LOG)
                qg, resid = M.smpl_to_robot_qpos(ref[t], q)
                calls = LOG[k0:]
                n_frames += 1
                if len(calls) > 1:                      # a joint ended at a limit: re-solved from the nominal pose
                    n_limit += 1
                    n_swap += len(calls) == 3           # the polish runs only when the nominal re-solve won somewhere
                q = qg[M.robot_qadr]
                if t % 10 == 0:                         # random restarts: is there a clearly better answer?
                    M.d_smpl.qpos[:] = ref[t]
                    import mujoco
                    mujoco.mj_kinematics(M.m_smpl, M.d_smpl)
                    Rt = [M.d_smpl.xmat[M.smpl_bid[b]].reshape(3, 3) @ M.C[b].T for b, _ in M.pairs]
                    from model.robot_map import _quat
                    root = _quat(Rt[0])
                    n_log = len(LOG)
                    best = min((M._resid(Rt, root, M._solve(Rt, root, rng.uniform(lo, hi), 200, 1e-4, 1e-7)) for _ in range(8)),
                               key=objective)
                    del LOG[n_log:]                     # restarts are not part of the pipeline's statistics
                    n_rs += 1
                    if objective(best) < objective(resid) * 0.9 and resid.mean() - best.mean() > 0.5:
                        better += 1; gain_deg.append(float(resid.mean() - best.mean()))
        main_calls = [c for c in LOG if c[0] == 50]
        out[key] = dict(
            frames=n_frames,
            solves=len(main_calls),
            stopped_on_own=float(np.mean([c[2] for c in main_calls])),
            iters_median=float(np.median([c[1] for c in main_calls])),
            ran_out=int(sum(not c[2] for c in main_calls)),
            ran_out_last_step_deg_median=float(np.median([c[3] for c in main_calls if not c[2]] or [0])),
            ran_out_moved_deg_median=float(np.median([c[4] for c in main_calls if not c[2]] or [0])),
            ran_out_moved_deg_p95=float(np.percentile([c[4] for c in main_calls if not c[2]] or [0], 95)),
            ran_out_last10_fall_median=float(np.median([c[5] for c in main_calls if not c[2]] or [0])),
            ran_out_last10_fall_p95=float(np.percentile([c[5] for c in main_calls if not c[2]] or [0], 95)),
            limit_resolve_frames=n_limit / n_frames, limit_resolve_won=n_swap / n_frames,
            restart_frames=n_rs, restart_clearly_better=better / n_rs,
            restart_gain_deg_median=float(np.median(gain_deg)) if gain_deg else 0.0)
        print(key, json.dumps(out[key]), flush=True)
    (REPO / "outputs/map_error/ik_convergence.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
