#!/usr/bin/env python3
"""robot_feasibility_compare.py -- scripts/g1_feasibility_compare.py (archived in OOP refactor step 7) for any robot in
model/robots.py: everything below reads "G1" for the robot chosen with --robot (model/robot_env.py, model/robot_map.py).

---- g1_feasibility_compare.py: did the IK-feasibility penalty make the searched latent ask the G1 for reachable poses?

For the latents of one clip -- z0, the best of a search with the penalty (--new) and, if given, the best of one
without it (--old) -- this rolls each out
  on the G1   RobotVecEnv with 16 copies of the latent (the batch the search used, so the search's own rollout is
              reproduced; checked against its best.npz / origin_z.npz), recording per step the IK residual of the
              commanded pose (RobotEnv.last_resid) and how many G1 targets sit at a joint limit
  on the adult  humenv, same start (the clip's frame 0 at rest), the actor's mean action; the adult's actual poses are
              then mapped onto the G1 kinematically (RobotMap.smpl_to_robot_joints) to measure how much of what the latent
              makes the ADULT do is something the G1 could not do
and reports bfm, the residuals (all links / arms), the share of targets at a limit and the joints that hit limits most.
Writes <out>/{compare.json, *_adult.npz} and <out>/feasibility.mp4: row 1 the adult (original clip / z0 or old z /
new z), row 2 the G1 (clip mapped kinematically / z0 or old z / new z).
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/robot_feasibility_compare.py --robot g1 \\
         --clip jump-2/jump-2_8 --new outputs/g1/search/jump-2_8_s0 [--old <search without the penalty>]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robot_env import RobotVecEnv        # noqa: E402
from model.robot_map import RobotMap           # noqa: E402
from model.robots import ROBOTS, target_path   # noqa: E402
from model import bfm_align                    # noqa: E402
from crossenbodiment.robot_rollout import motion   # noqa: E402  g1_z0_rollout's (refactor step 4)
from crossenbodiment.geometry import project_z   # noqa: E402  single_z_search re-exported it (refactor step 4)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--clip", default="jump-2/jump-2_8")
    ap.add_argument("--old", default=None, help="a search without the penalty (omitted: the middle column is z0)")
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    stem = a.clip.split("/")[1]
    old, new = (REPO / a.old if a.old else None), REPO / a.new
    out = Path(a.out) if a.out else new / "compare"; out.mkdir(parents=True, exist_ok=True)
    vec = RobotVecEnv(a.robot, 16)
    cfg = ROBOTS[a.robot]; RNAME, SCENE = cfg["name"], cfg["scene"]
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]; T = len(ref)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device)
    adult, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
    Bg = bfm_align.embed(model, bfm_align.obs_from_qpos(adult, ref), a.device)
    gref = np.load(target_path(a.robot, stem))["qpos"]
    gm = RobotMap(a.robot); names = [gm.m_robot.joint(j).name.replace("_joint", "") for j in gm.robot_jnt]
    n_link, n_jnt = len(gm.pairs) - 1, len(gm.robot_jnt)
    ARM = [i for i, (b, _) in enumerate(gm.pairs[1:]) if b.split("_")[-1] in ("Shoulder", "Elbow", "Hand")]
    lo, hi = gm.robot_range[:, 0], gm.robot_range[:, 1]
    # z0 re-projected onto the sphere as g1_z_search does: the stored z0 can differ from it in the 7th digit, and the
    # G1 rollout is chaotic enough that this alone moves the end point by metres
    z0 = project_z(np.load(REPO / f"data/origin_z/{a.clip}.npy").reshape(-1).astype(np.float64))
    zs = {"z0": z0, "new": np.load(new / "best_z.npy").reshape(-1)}
    saved = {"z0": new / "origin_z.npz", "new": new / "best.npz"}
    if old:
        zs["old"], saved["old"] = np.load(old / "best_z.npy").reshape(-1), old / "best.npz"

    @torch.no_grad()
    def act(o, z):
        return model._actor(model._normalize(torch.as_tensor(o, dtype=torch.float32, device=a.device)), z,
                            model.cfg.actor_std).mean.cpu().numpy()

    res = {}
    for k, z in zs.items():
        zt = torch.as_tensor(np.repeat(z[None], 16, 0), dtype=torch.float32, device=a.device)
        obs, _, _, _ = vec.reset(ref[0]); O, QG, FZ = [], [], []
        for _ in range(T):
            obs, _, qg, fz = vec.step(act(obs, zt))
            O.append(obs[0]); QG.append(qg[0]); FZ.append(fz[0])
        O, QG, FZ = np.array(O, np.float32), np.array(QG), np.array(FZ)
        rep = float(np.abs(QG - np.load(saved[k])["qpos"][:T]).max())
        # the adult with the same latent, and how much of its actual pose the G1 cannot make
        ad = adult.unwrapped; ad.set_physics(qpos=ref[0], qvel=np.zeros(ad.model.nv)); o = ad.get_obs()["proprio"]
        zt1 = torch.as_tensor(z[None], dtype=torch.float32, device=a.device); qa = []
        for _ in range(T):
            o = ad.step(act(o[None], zt1)[0])[0]["proprio"]; qa.append(ad.data.qpos.copy())
        qa = np.array(qa); np.savez_compressed(out / f"{k}_adult.npz", qpos=qa, fps=30)
        q = None; ares, hit = [], []
        for x in qa:
            q, r = gm.smpl_to_robot_joints(x, q); ares.append(r)
            hit.append((q <= lo + 1e-6) | (q >= hi - 1e-6))
        ares, hit = np.array(ares), np.array(hit)
        alim = hit.sum(1)
        top = {names[j]: float(hit[:, j].mean()) for j in np.argsort(-hit.mean(0))[:5] if hit[:, j].any()}
        res[k] = dict(
            reproduces_search=rep,
            bfm=bfm_align.bfm_align_loss(bfm_align.embed(model, O, a.device), Bg),
            g1_cmd_ik_deg=float(FZ[:, :n_link].mean()), g1_cmd_ik_arm_deg=float(FZ[:, ARM].mean()),
            g1_cmd_ik_arm_max_deg=float(FZ[:, ARM].max()), g1_cmd_at_limit=float(FZ[:, n_link].mean() / n_jnt),
            adult_pose_ik_deg=float(ares.mean()), adult_pose_ik_arm_deg=float(ares[:, ARM].mean()),
            adult_pose_ik_arm_max_deg=float(ares[:, ARM].max()), adult_pose_at_limit=float(np.mean(alim) / n_jnt),
            adult_pose_joints_at_limit=top,
            motion=motion(QG, gref))
        print(f"{k:4s} (reproduces search to {rep:.1e}) bfm {res[k]['bfm']:.3f} | G1 commands: IK residual {res[k]['g1_cmd_ik_deg']:.1f} deg, "
              f"arms {res[k]['g1_cmd_ik_arm_deg']:.1f} (max {res[k]['g1_cmd_ik_arm_max_deg']:.0f}), at limit {100 * res[k]['g1_cmd_at_limit']:.1f}% | "
              f"adult's own pose on the G1: residual {res[k]['adult_pose_ik_deg']:.1f}, arms {res[k]['adult_pose_ik_arm_deg']:.1f} "
              f"(max {res[k]['adult_pose_ik_arm_max_deg']:.0f}), at limit {100 * res[k]['adult_pose_at_limit']:.1f}% | "
              f"fallen {res[k]['motion']['fallen']}, end-point off {res[k]['motion']['end_xy']:.2f} m", flush=True)
        print("     adult pose: G1 joints most often at a limit:", {j: f"{100 * v:.0f}%" for j, v in top.items()}, flush=True)
    vec.close()
    (out / "compare.json").write_text(json.dumps(res, indent=1))

    rel = lambda p: str(Path(p).resolve().relative_to(REPO))
    f = lambda k, key: res[k][key]
    mid, mid_name = ("old", "the old searched z") if old else ("z0", "z0")
    ns = json.loads((new / "summary.json").read_text()); obj, w = ns.get("objective", "bfm"), ns.get("ik_weight", 0)
    new_name = f"the new z ({'bfm' if obj == 'bfm' else 'target MSE'}, IK penalty {w:g})"
    spec = {"out": rel(out / "feasibility.mp4"), "cols": 3, "size": 1080, "label_px": 36, "quality": 8,
            "track": {"distance": 3.6, "elevation": -12, "azimuth": 140, "lookat_z": 0.7},
            "panels": [
                {"title": "adult: original clip", "sub": a.clip, "xml": "assets/robots/adult/robot.xml",
                 "qpos": f"data/origin_motion/{a.clip}.npz"},
                {"title": f"adult with {mid_name}", "sub": f"arms the {RNAME} can't follow: {f(mid, 'adult_pose_ik_arm_deg'):.0f} deg avg",
                 "xml": "assets/robots/adult/robot.xml", "qpos": rel(out / f"{mid}_adult.npz")},
                {"title": f"adult with {new_name}", "sub": f"arms the {RNAME} can't follow: {f('new', 'adult_pose_ik_arm_deg'):.0f} deg avg",
                 "xml": "assets/robots/adult/robot.xml", "qpos": rel(out / "new_adult.npz")},
                {"title": f"{RNAME}: clip mapped kinematically", "sub": "the target, no physics", "xml": SCENE,
                 "qpos": rel(target_path(a.robot, stem))},
                {"title": f"{RNAME} with {mid_name}", "sub": f"bfm {f(mid, 'bfm'):.3f}, arm cmd residual {f(mid, 'g1_cmd_ik_arm_deg'):.0f} deg",
                 "xml": SCENE, "qpos": rel(saved[mid])},
                {"title": f"{RNAME} with {new_name}", "sub": f"bfm {f('new', 'bfm'):.3f}, arm cmd residual {f('new', 'g1_cmd_ik_arm_deg'):.0f} deg",
                 "xml": SCENE, "qpos": rel(new / "best.npz")}]}
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    subprocess.run(["uv", "run", "python", "scripts/render_panels.py", str(out / "spec.json")], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
