#!/usr/bin/env python3
"""locked_action_check.py -- does a searched z make the policy STOP commanding the locked joints?

A locked joint ignores its action either way (lock_joint_body.py: gear 0; limit_joint_body.py: the servo pushes into
the joint limit). Whether the POLICY has given that joint up is a separate question: its commanded PD target angle
target = (gain * u + bias0) / kp (each actuator maps u in [-1, 1] onto its joint's range) either stays at the lock
angle, or keeps swinging as if the joint still moved.

Per search run (outputs/lock_knee/<run>): z0 and best_z.npy are re-rolled through single_z_search.py --eval-z with the
run's own settings (actions saved; the rollouts are checked against the run's origin_z.npz / best.npz), and for the
locked joints this reports, per z,
  off     mean_t |target - lock angle| (deg), averaged over the locked joints
  swing   std_t of the target (deg), averaged over the locked joints
  within  share of (step, joint) whose target is within --tol deg of the lock angle
next to the same joints' target swing in the adult's own walk (data/origin_action: z0 on the unlocked adult).
usage: uv run python scripts/locked_action_check.py adult_lockRleg_s1_cont adult_lockRleg_lim_s0 ... [--tol 5]
"""
import argparse, json, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "outputs/lock_knee"


def locks_of(body):
    """{joint: lock angle (rad)} from the lock body's equality constraints (a *_lim body: its source lock body)."""
    m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots" / body.removesuffix("_lim") / "robot.xml"))
    return {m.joint(int(m.eq_obj1id[i])).name: float(m.eq_data[i, 0]) for i in range(m.neq)
            if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT}


def targets(m, act, joints):
    """(T, n_act) actions -> (T, len(joints)) commanded target angles (rad)."""
    ids = [m.actuator(j).id for j in joints]
    g, b0, kp = m.actuator_gainprm[ids, 0], m.actuator_biasprm[ids, 0], -m.actuator_biasprm[ids, 1]
    return (g * act[:, ids] + b0) / kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--gpu", default="2")
    a = ap.parse_args()
    m_adult = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml"))
    res = {}
    for run in a.runs:
        d = RUNS / run; s = json.loads((d / "summary.json").read_text())
        out = d / "act_check"; out.mkdir(exist_ok=True)
        z0 = np.load(REPO / f"data/origin_z/{s['clip']}.npy").reshape(-1)
        np.save(out / "z.npy", np.stack([z0, np.load(d / "best_z.npy").reshape(-1)]))
        if not (out / "eval_act.npz").exists():
            cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", s["clip"], "--body", s["body"],
                   "--xml", s["xml"], "--ref", s["ref"], "--init", s["init"], "--obs-scale", s["obs_scale"],
                   "--objective", s["objective"], "--pairs", str(s["pairs"]), "--eval-z", str(out / "z.npy"),
                   "--eval-save-qpos", "--eval-save-act", "--out", str(out)]
            subprocess.run(cmd, cwd=REPO, check=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": a.gpu},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        q = np.load(out / "eval_qpos.npz")["qpos"]; act = np.load(out / "eval_act.npz")["act"]
        for i, f in enumerate(("origin_z.npz", "best.npz")):
            err = np.abs(q[i] - np.load(d / f)["qpos"][:len(q[i])]).max()
            assert err < 1e-3, f"{run}: re-rolled {f} differs from the search's by {err}"
        lk = locks_of(s["body"]); J = list(lk); lock = np.array([lk[j] for j in J])
        mb = mujoco.MjModel.from_xml_path(s["xml"])
        jq = [mb.joint(j).qposadr[0] for j in J]
        ad = np.load(REPO / f"data/origin_action/{s['clip']}.npz")["action"]
        r = {"joints": J, "adult_swing": float(np.degrees(targets(m_adult, ad, J).std(0)).mean())}
        for i, name in enumerate(("z0", "search")):
            tg = targets(mb, act[i], J)
            r[name] = dict(off=float(np.degrees(np.abs(tg - lock)).mean()), swing=float(np.degrees(tg.std(0)).mean()),
                           within=float((np.degrees(np.abs(tg - lock)) < a.tol).mean()),
                           joint_off=float(np.degrees(np.abs(q[i][:, jq] - lock)).mean()),
                           per_joint_off={j: float(v) for j, v in zip(J, np.degrees(np.abs(tg - lock)).mean(0))})
        res[run] = r
        print(f"{run:30s} {len(J):2d} locked joints | adult walk target swing {r['adult_swing']:5.1f} deg")
        for name in ("z0", "search"):
            x = r[name]
            print(f"   {name:6s} target off lock {x['off']:5.1f} deg  swing {x['swing']:5.1f} deg  "
                  f"within {a.tol:g} deg {x['within']:4.0%}   (actual joint off lock {x['joint_off']:.2f} deg)")
    (RUNS / "locked_action_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
