#!/usr/bin/env python3
"""robot_torque_limit.py -- how often do the robots' motors hit their torque limits? The gains are rescaled
(scripts/robot_gain_scaling.py) but every robot keeps its own torque limits, so a joint asked for more than its
motor can give is clipped.

For a robot_z_search.py run, z0 and the best z are rolled out again in the exact batch the search scored them in
(z0: 16 copies; the best z: its own step's 16 candidates, rebuilt from the noise stream and the saved search
point), the reroll checked against the saved qpos. Every physics substep (15 per action) records, per motor, the
torque the PD servo asks for, kp (target - q) - kd qdot, against the limit MuJoCo clips it to (actuator force range
and/or joint actuator force range, whichever is tighter). A motor is at its limit when the demand reaches it.
Per run: outputs/torque_limit/<robot>_<stem>.json (per motor: share of substeps at the limit, 99th percentile and
max of |demand| / limit; overall: share of motor-substeps at the limit, share of frames with any motor there).
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/robot_torque_limit.py outputs/user_survey_search/g1_ridge/<run> [...]
"""
import json, sys
from pathlib import Path
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robot_env import N_SUB, RobotEnv          # noqa: E402
from model.robots import ROBOTS                      # noqa: E402
from crossenbodiment.geometry import project_z   # noqa: E402  single_z_search re-exported it (refactor step 4)

OUT = REPO / "outputs/torque_limit"


def limits(m):
    """per actuator: the tightest (lo, hi) torque MuJoCo applies (gear is 1 on every robot here)."""
    lo, hi = np.full(m.nu, -np.inf), np.full(m.nu, np.inf)
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        if m.actuator_forcelimited[i]:
            lo[i], hi[i] = max(lo[i], m.actuator_forcerange[i, 0]), min(hi[i], m.actuator_forcerange[i, 1])
        if m.jnt_actfrclimited[j]:
            lo[i], hi[i] = max(lo[i], m.jnt_actfrcrange[j, 0]), min(hi[i], m.jnt_actfrcrange[j, 1])
    return lo, hi


def _worker(conn, name, fill):
    env = RobotEnv(name, fill=fill); m, d = env.m, env.d
    assert np.all(m.actuator_gear[:, 0] == 1)
    lo, hi = limits(m)

    def demand():                                    # the unclipped servo torque, as MuJoCo computes it
        return m.actuator_gainprm[:, 0] * d.ctrl + m.actuator_biasprm[:, 1] * d.actuator_length \
            + m.actuator_biasprm[:, 2] * d.actuator_velocity
    while True:
        cmd, arg = conn.recv()
        if cmd == "close":
            break
        if cmd == "reset":
            obs, qs = env.reset(arg); tq = np.zeros((N_SUB, m.nu))
        else:                                        # RobotEnv.step, one substep at a time
            d.ctrl[:] = env.action_to_ctrl(arg); tq = np.empty((N_SUB, m.nu))
            for k in range(N_SUB - 1):
                mujoco.mj_step(m, d); tq[k] = demand()
            q_prev = d.qpos.copy()
            mujoco.mj_step(m, d); tq[-1] = demand()
            qs_prev, qs = env.map.robot_to_smpl_qpos(q_prev), env.map.robot_to_smpl_qpos(d.qpos)
            vs = np.zeros(env.ms.nv); mujoco.mj_differentiatePos(env.ms, vs, 1 / 450, qs_prev, qs)
            obs = env.smpl_obs(qs, vs)
        conn.send((obs, d.qpos.copy(), tq, lo, hi))


class Vec:
    def __init__(self, name, n, fill):
        import multiprocessing as mp, os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        ctx = mp.get_context("spawn"); self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe(); p = ctx.Process(target=_worker, args=(b, name, fill), daemon=True); p.start()
            self.conns.append(a); self.procs.append(p)

    def call(self, cmd, args):
        for c, x in zip(self.conns, args):
            c.send((cmd, x))
        return [c.recv() for c in self.conns]

    def close(self):
        for c in self.conns:
            c.send(("close", None))


def best_batch(run, s):
    """the 16 candidates the search scored at the best z's step, and the best's index among them."""
    S, zt = np.load(run / "state.npz"), np.load(run / "z_trace.npz")["z_mean"]
    zb, g, p, sig = S["best_z"].astype(np.float64), s["best"]["gen"], s["pairs"], s["sigma"]
    if s["best"]["source"] == "mean_z":
        return np.repeat(zb[None], 2 * p, 0), 0
    rng = np.random.default_rng(s["seed"])
    anti = s.get("sampling", "antithetic") == "antithetic"     # runs before 11/04 record none: all antithetic
    for _ in range(g + 1):
        eps = rng.standard_normal((p if anti else 2 * p, zb.size))
    E = np.concatenate([eps, -eps]) if anti else eps; r2 = zb.size
    fits = []
    for i, e in enumerate(E):                        # zb = proj(z + sig e), |z| = 16: solve for z, match the trace
        b = sig * zb @ e; c = (b + np.sqrt(b * b - r2 * (sig * sig * e @ e - r2))) / r2
        z = c * zb - sig * e
        fits.append((np.abs(z - zt[g]).max(), i, z))
    err, i, z = min(fits, key=lambda f: f[0])
    assert err < 1e-5, f"{run.name}: no candidate reproduces the search point ({err:.1e})"
    cand = project_z(z + sig * E)
    assert np.abs(cand[i] - zb).max() < 1e-9
    return cand, i


def main():
    from metamotivo.fb_cpr.huggingface import FBcprModel
    runs = [REPO / r for r in sys.argv[1:]]
    by = {}
    for run in runs:
        s = json.loads((run / "summary.json").read_text()); by.setdefault((s["body"], s["fill"]), []).append((run, s))
    OUT.mkdir(parents=True, exist_ok=True)
    model = None
    for (robot, fill), items in by.items():
        vec = Vec(robot, 2 * items[0][1]["pairs"], fill)       # the workers before CUDA
        if model is None:
            model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
        mr = mujoco.MjModel.from_xml_path(str(REPO / ROBOTS[robot]["scene"])); names = [mr.actuator(k).name for k in range(mr.nu)]
        for run, s in items:
            ref = np.load(REPO / f"data/origin_motion/{s['clip']}.npz")["qpos"]; T = len(ref)
            z0 = project_z(np.load(REPO / f"data/origin_z/{s['clip']}.npy").reshape(-1).astype(np.float64))
            res = {}
            for key, (Z, i), saved in (("z0", (np.repeat(z0[None], 2 * s["pairs"], 0), 0), "origin_z.npz"),
                                       ("best", best_batch(run, s), "best.npz")):
                zt = torch.as_tensor(Z, dtype=torch.float32, device="cuda")
                out = vec.call("reset", [ref[0]] * len(Z))
                QG, TQ = [], []
                with torch.no_grad():
                    for _ in range(T):
                        o = torch.as_tensor(np.stack([x[0] for x in out]), dtype=torch.float32, device="cuda")
                        u = model._actor(model._normalize(o), zt, model.cfg.actor_std).mean.cpu().numpy()
                        out = vec.call("step", list(u))
                        QG.append(out[i][1]); TQ.append(out[i][2])
                lo, hi = out[i][3], out[i][4]
                QG, TQ = np.stack(QG), np.stack(TQ)                   # (T, nq), (T, 15, nu)
                dev = float(np.abs(QG - np.load(run / saved)["qpos"]).max())
                lim = np.where(TQ >= 0, hi, -lo)                      # the limit on the side the demand is on
                ratio = np.abs(TQ) / lim
                at = ratio >= 1 - 1e-9
                mot = {names[k]: dict(at_limit=float(at[:, :, k].mean()), p99=float(np.percentile(ratio[:, :, k], 99)),
                                                      max=float(ratio[:, :, k].max()), limit_Nm=float(hi[k]))
                       for k in range(TQ.shape[2])}
                res[key] = dict(reroll_vs_saved_max_dq=dev, at_limit=float(at.mean()), frames_any=float(at.any((1, 2)).mean()),
                                motors_ever=int(at.any((0, 1)).sum()), n_motors=int(TQ.shape[2]), motors=mot)
                np.savez_compressed(OUT / f"{robot}_{run.name.removesuffix('_s0')}_{key}.npz", ratio_frame_max=ratio.max(1).astype(np.float32))
                worst = sorted(mot.items(), key=lambda kv: -kv[1]["at_limit"])[:3]
                print(f"{robot:6s} {run.name:28s} {key:4s} reroll dev {dev:.1e} | at limit {100 * res[key]['at_limit']:.2f}% of motor-substeps, "
                      f"{100 * res[key]['frames_any']:.1f}% of frames, {res[key]['motors_ever']}/{TQ.shape[2]} motors ever | worst "
                      + ", ".join(f"{n} {100 * v['at_limit']:.1f}%" for n, v in worst), flush=True)
            (OUT / f"{robot}_{run.name.removesuffix('_s0')}.json").write_text(json.dumps(dict(run=str(run.relative_to(REPO)), **res), indent=1))
        vec.close()


if __name__ == "__main__":
    main()
