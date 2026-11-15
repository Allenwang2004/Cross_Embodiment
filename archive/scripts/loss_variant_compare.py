#!/usr/bin/env python3
"""loss_variant_compare.py -- the same running clip searched with three costs, measured with one ruler.

Costs (per body: the original search, then two variants without the floor-position term):
  original   bfm + heading + 0.1 pos (+ anchor; robots + ik)
  zerovel    bfm with B's velocity inputs set to 0 on both sides + heading (single_z_search / robot_z_search
             --bfm-zero-vel --pos-weight 0)
  mse        pose MSE (joint angles, hands / feet / head relative to the pelvis, pelvis height, heading) against
             the retargeted target (--objective mse --mse-no-xy)
Measured on z0 and on the best z of each run, from the saved trajectories (velocities by 30 Hz differences on
both sides, so the bfm here differs a little from a search's logged value):
  bfm        1 - mean_t cos(B(state_t), B(reference_t)), the adult-equivalent state (child: ExactObs; robot: its
             SMPL-equivalent qpos) against the reference the search used (child: its retarget through ExactObs;
             robot: the adult's clip)
  bfm0       the same with the velocity inputs set to 0
  pose       mean distance (cm) of the 23 body parts from where the reference has them, relative to the pelvis and
             facing (obs[1:70]): a pose-only measure that none of the costs uses directly
  heading    mean |facing - reference facing| (deg)
  speed      mean horizontal pelvis speed of the body itself (m/s), against its target's
  down       frames with the pelvis below half the target's pelvis height (fallen)
-> outputs/loss_variants/move-ego-0-4_8.json
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/loss_variant_compare.py"""
import json, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model import bfm_align                                             # noqa: E402

CLIP = "move-ego-0-4/move-ego-0-4_8"; STEM = CLIP.split("/")[1]
ADULT = REPO / "assets/robots/adult/robot.xml"
CHILD = REPO / "assets/robots_torque/child/robot_torque_full.xml"
VEL, POS = slice(214, 358), slice(1, 70)
RUNS = {"child": {"original": "outputs/c540_bfmglobal_a03/move-ego-0-4_8",
                  "zerovel": "outputs/child_loss_test/zerovel_nopos/move-ego-0-4_8",
                  "mse": "outputs/child_loss_test/mse_noxy/move-ego-0-4_8"},
        "h1": {"original": "outputs/user_survey_search/h1_ridge/move-ego-0-4_8_s0",
               "zerovel": "outputs/user_survey_search/h1_ridge_zerovel_nopos/move-ego-0-4_8_s0",
               "mse": "outputs/user_survey_search/h1_ridge_mse_noxy/move-ego-0-4_8_s0"}}


def unit(B):
    return B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12)


def fd_vel(m, q):
    v = np.zeros((len(q), m.nv))
    for t in range(1, len(q)):
        mujoco.mj_differentiatePos(m, v[t], bfm_align.DEFAULT_DT, q[t - 1], q[t])
    return v


def heading(m, q):
    import humenv.utils as hu
    d = mujoco.MjData(m); h = np.empty(len(q))
    for i, qq in enumerate(q):
        d.qpos[:] = qq; mujoco.mj_kinematics(m, d)
        h[i] = hu.calc_heading(hu.remove_base_rot(d.xquat[1][None].copy(), "smpl"))[0]
    return h


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.exact_obs import ExactObs
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    env, _ = make_humenv(num_envs=1, task=None, xml=str(ADULT), state_init="Default")
    ma, mc = mujoco.MjModel.from_xml_path(str(ADULT)), mujoco.MjModel.from_xml_path(str(CHILD))
    exact = ExactObs(CHILD, ADULT)
    emb = lambda O: unit(bfm_align.embed(model, O, "cuda"))

    def zero(O):
        O = np.array(O, copy=True); O[:, VEL] = 0; return O

    def child_obs(q):
        return np.stack([exact(qq, vv) for qq, vv in zip(q, fd_vel(mc, q))]).astype(np.float32)
    out = {}
    for body, runs in RUNS.items():
        if body == "child":
            ref = np.load(REPO / f"data/child/retargeting_motion/{CLIP}.npz")["qpos"].astype(np.float64)
            Or, ref_h, tgt_q = child_obs(ref), heading(mc, ref), ref
        else:
            ref = np.load(REPO / f"data/origin_motion/{CLIP}.npz")["qpos"].astype(np.float64)
            Or, ref_h = bfm_align.obs_from_qpos(env, ref), heading(ma, ref)
            tgt_q = np.load(REPO / f"outputs/h1/roundtrip/{STEM}/robot.npz")["qpos"].astype(np.float64)
        Br, Br0 = emb(Or), emb(zero(Or))
        tgt_speed = float(np.linalg.norm(np.diff(tgt_q[:, :2], axis=0), axis=1).mean() * 30)
        out[body] = {"target_speed": tgt_speed}
        for name, run in runs.items():
            s = json.loads((REPO / run / "summary.json").read_text())
            out[body][name] = {"search_cost": {k: s[k]["cost"] for k in ("origin_z", "best")}, "best_gen": s["best"]["gen"]}
            for k, f in (("z0", "origin_z.npz"), ("best", "best.npz")):
                z = np.load(REPO / run / f)
                q = z["qpos"].astype(np.float64)
                if body == "child":
                    O, h = child_obs(q), heading(mc, q)
                else:
                    qs = z["qpos_smpl"].astype(np.float64)
                    O, h = bfm_align.obs_from_qpos(env, qs), heading(ma, qs)
                n = min(len(O), len(Or))
                pose = np.linalg.norm(O[:n, POS].reshape(n, 23, 3) - Or[:n, POS].reshape(n, 23, 3), axis=-1).mean()
                out[body][name][k] = dict(
                    bfm=float(1 - (emb(O)[:n] * Br[:n]).sum(-1).mean()), bfm0=float(1 - (emb(zero(O))[:n] * Br0[:n]).sum(-1).mean()),
                    pose_cm=float(100 * pose), heading_deg=float(np.degrees(np.abs((h[:n] - ref_h[:n] + np.pi) % (2 * np.pi) - np.pi).mean())),
                    speed=float(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1).mean() * 30),
                    down=int((q[:n, 2] < 0.5 * tgt_q[:n, 2]).sum()))
        print(f"== {body} (target speed {tgt_speed:.2f} m/s)")
        for name in runs:
            r = out[body][name]
            for k in ("z0", "best"):
                v = r[k]
                print(f"  {name:8s} {k:4s} bfm {v['bfm']:.3f} | bfm0 {v['bfm0']:.3f} | pose {v['pose_cm']:5.1f} cm | heading {v['heading_deg']:5.1f} deg | "
                      f"speed {v['speed']:.2f} m/s | down {v['down']:3d}/300" + (f" | search cost {r['search_cost']['best']:.3f} (gen {r['best_gen']})" if k == "best" else ""))
    d = REPO / "outputs/loss_variants"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{STEM}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
