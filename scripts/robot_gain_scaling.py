#!/usr/bin/env python3
"""robot_gain_scaling.py -- scripts/g1_gain_scaling.py for any robot in model/robots.py: per-joint PD gains.

  k     per joint group (ROBOTS[name]["groups"], left = right): the static gravity moment of everything downstream of
        the joint about it, on the robot's kinematic targets (scripts/robot_roundtrip.py) against the adult's original
        clips on the same frames, fitted as robot = k * adult through the origin
  kp    k * the SMPL source hinges' MOTOR kp (ROBOTS[name]["gain_source"]; several in series)
  kd    so that each joint has the SMPL hinge's damping ratio (motor + passive damping, spring in the stiffness):
        zeta_SMPL * 2 sqrt(kp I_robot), minus the robot's own passive damping (Apollo models its transmission's;
        the G1 has none), floored at 0. I = mass-matrix diagonal (armature included), median over the frames.
--damping zero sets the robot's passive damping to 0 (in the json, as below) and motor kd gives all of it.
--damping smpl splits that damping the way the SMPL hinge does, as the child bodies' scaling did (passive and motor
damping both scaled so the ratio stays): passive = SMPL passive share x total, replacing the robot's own passive damping
("passive_damping" in the json, applied by model/robot_env.py), motor kd = the rest.
The robot keeps its armature, friction loss and torque limits (and, by default, its passive damping). Writes ROBOTS[name]["gains_json"]
(or --out). With --robot g1 and the G1's 18 clips this reproduces assets/robots/g1/gains_perjoint.json.
usage: uv run python scripts/robot_gain_scaling.py --robot h1 [--clips-file outputs/g1/search_clips.txt] [--out ...]
"""
import argparse, json, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.robots import ROBOTS, target_path   # noqa: E402

SMPL_XML = REPO / "assets/robots/adult/robot.xml"


def moment(m, d, body, anchor):
    r = d.subtree_com[body] - anchor
    return float(np.linalg.norm(np.cross(r, [0.0, 0.0, -9.81 * m.body_subtreemass[body]])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--clips-file", default="outputs/g1/search_clips.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--damping", default=None, choices=["robot", "smpl", "zero"],
                    help="robot: keep the robot's passive damping, motor kd tops it up; smpl: SMPL's passive / motor split; "
                         "zero: the robot's passive damping set to 0, motor kd gives all of it (as on the G1, which has none)")
    a = ap.parse_args()
    cfg = ROBOTS[a.robot]
    a.damping = a.damping or cfg.get("damping", "robot")       # the robot's default (Apollo: smpl)
    mg = mujoco.MjModel.from_xml_path(str(REPO / cfg["xml"])); dg = mujoco.MjData(mg)
    ms = mujoco.MjModel.from_xml_path(str(SMPL_XML)); ds = mujoco.MjData(ms)
    clips = [c for c in (REPO / a.clips_file).read_text().split() if c]
    origin = {p.stem: p for p in (REPO / "data/origin_motion").glob("*/*.npz")}
    sides = cfg["sides"]
    groups = cfg["groups"]
    s_act = {ms.actuator(i).name: i for i in range(ms.nu)}
    acts = [mg.actuator(i).name for i in range(mg.nu)]
    ajnt = {n: mg.joint(mg.actuator(n).trnid[0]).name for n in acts}

    def group_of(jname):
        return next(g for g, keys in cfg["group_keys"].items() if any(k in jname for k in keys))

    tau = {g: ([], []) for g in groups}; per_clip = {g: {} for g in groups}
    inertia = {n: ([], []) for n in acts}
    Mg, Ms = np.zeros((mg.nv, mg.nv)), np.zeros((ms.nv, ms.nv))
    for clip in clips:
        stem = clip.split("/")[1]
        qg_all = np.load(target_path(a.robot, stem))["qpos"]; qs_all = np.load(origin[stem])["qpos"]
        T = min(len(qg_all), len(qs_all))
        cg, cs = {g: [] for g in groups}, {g: [] for g in groups}
        for t in range(0, T, a.stride):
            dg.qpos[:] = qg_all[t]; mujoco.mj_forward(mg, dg)
            ds.qpos[:] = qs_all[t]; mujoco.mj_forward(ms, ds)
            for g, (gb, gj, sb) in groups.items():
                for S, s in (sides if "{s}" in gb else (("", ""),)):
                    bg, jg, bs = mg.body(gb.format(s=s)).id, mg.joint(gj.format(s=s)).id, ms.body(sb.format(S=S)).id
                    cg[g].append(moment(mg, dg, bg, dg.xanchor[jg])); cs[g].append(moment(ms, ds, bs, ds.xpos[bs]))
            if t % (5 * a.stride) == 0:
                mujoco.mj_fullM(mg, dg, Mg); mujoco.mj_fullM(ms, ds, Ms)
                for n in acts:
                    dofg = mg.jnt_dofadr[mg.actuator(n).trnid[0]]
                    dofs = ms.jnt_dofadr[ms.actuator_trnid[s_act[cfg["gain_source"][ajnt[n]][0]], 0]]
                    inertia[n][0].append(Mg[dofg, dofg]); inertia[n][1].append(Ms[dofs, dofs])
        for g in groups:
            x, y = np.array(cs[g]), np.array(cg[g])
            tau[g][0].extend(y); tau[g][1].extend(x)
            per_clip[g][stem] = float(x @ y / (x @ x)) if x @ x > 1e-9 else float("nan")

    k, r2 = {}, {}
    print(f"{cfg['name']}: gravity-moment ratio robot / adult over {len(clips)} clips (every {a.stride}th frame):")
    for g in groups:
        y, x = np.array(tau[g][0]), np.array(tau[g][1])
        k[g] = float(x @ y / (x @ x)); r2[g] = float(1 - np.sum((y - k[g] * x) ** 2) / np.sum(y ** 2))
        pc = np.array([v for v in per_clip[g].values() if np.isfinite(v)])
        print(f"  {g:9s} k {k[g]:.3f}  R^2 {r2[g]:.3f}  per clip {np.median(pc):.3f} [{pc.min():.3f}, {pc.max():.3f}]")

    series = lambda v: 1.0 / np.sum(1.0 / v)
    kp, kd, rows, passive = {}, {}, {}, {}
    for n in acts:
        jn = ajnt[n]; src = [s_act[h] for h in cfg["gain_source"][jn]]
        s_kp = -ms.actuator_biasprm[src, 1]; s_kd = -ms.actuator_biasprm[src, 2]
        trn = ms.actuator_trnid[src, 0]
        Ks = series(s_kp + ms.jnt_stiffness[trn]); Ds = series(s_kd + ms.dof_damping[ms.jnt_dofadr[trn]]); Kmotor = series(s_kp)
        Ig, Is = float(np.median(inertia[n][0])), float(np.median(inertia[n][1]))
        zeta = Ds / (2 * np.sqrt(Ks * Is))
        d_robot = float(mg.dof_damping[mg.jnt_dofadr[mg.actuator(n).trnid[0]]])
        kp[n] = float(k[group_of(jn)] * Kmotor)
        total = zeta * 2 * np.sqrt(kp[n] * Ig)
        if a.damping == "smpl":
            share = series(ms.dof_damping[ms.jnt_dofadr[trn]]) / Ds          # SMPL's passive share of its damping
            passive[n] = float(share * total); d_robot = passive[n]
            kd[n] = float(total - passive[n])
        elif a.damping == "zero":
            passive[n] = 0.0; d_robot = 0.0
            kd[n] = float(total)
        else:
            kd[n] = float(max(total - d_robot, 0.0))
        rows[n] = dict(group=group_of(jn), k=k[group_of(jn)], smpl_src=cfg["gain_source"][jn], smpl_K=float(Ks),
                       smpl_D=float(Ds), smpl_motor_kp=float(Kmotor), I_robot=Ig, I_smpl=Is, Ir=Ig / Is, zeta=float(zeta),
                       robot_passive_damping=d_robot, zeta_robot=float((kd[n] + d_robot) / (2 * np.sqrt(kp[n] * Ig))),
                       f_smpl=float(np.sqrt(Ks / Is) / (2 * np.pi)), f_robot=float(np.sqrt(kp[n] / Ig) / (2 * np.pi)))
    print("\nper actuator (left side; right = mirror):  kp  kd (+ robot's passive) | zeta target -> got | natural freq SMPL -> robot")
    left = cfg["sides"][0][1] + "_"
    for n in acts:
        if n.startswith(cfg["sides"][1][1] + "_"):
            continue
        r = rows[n]
        print(f"  {n:24s} kp {kp[n]:7.1f}  kd {kd[n]:6.2f} (+{r['robot_passive_damping']:.1f}) | zeta {r['zeta']:.2f} -> "
              f"{r['zeta_robot']:.2f} | f {r['f_smpl']:.1f} -> {r['f_robot']:.1f} Hz")
    out = REPO / (a.out or cfg["gains_json"]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        note=f"per-joint {cfg['name']} PD gains (scripts/robot_gain_scaling.py): kp = k_group * SMPL motor kp; kd gives the "
             "SMPL hinge's damping ratio together with the robot's own passive damping; robot keeps its armature, "
             "friction loss and torque limits", robot=a.robot, clips=clips, stride=a.stride, k_group=k, r2_group=r2,
        per_clip_k=per_clip, kp=kp, kd=kd, damping=a.damping, **({"passive_damping": passive} if passive else {}),
        detail=rows), indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
