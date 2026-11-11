#!/usr/bin/env python3
"""g1_gain_scaling.py -- per-joint PD gains for the G1, the way assets/robots_torque/<body>/ scales the SMPL bodies.

The SMPL-family bodies get their actuators from scripts/torque_aggregate_motion_k.py --joint-dynamics: per joint, the
ratio k of the body's gravity torque to the adult's over the same poses scales kp (and the torque limit), and damping
moves with sqrt(k * Ir) so the damping ratio stays the adult's. Here the same idea for the G1, with three differences:

  k        the G1's joints are not the SMPL's, so k is measured per JOINT GROUP (hip, knee, ankle, waist, shoulder,
           elbow, wrist; left = right): the static gravity moment of everything downstream of the joint about the
           joint, |(subtree COM - joint) x subtree weight|, on the G1's kinematically retargeted clips
           (outputs/g1/roundtrip/<stem>/g1.npz) against the adult's original clips on the same frames, fitted as
           G1 = k * adult through the origin (R^2 about zero, as scripts/torque_ratio_per_joint.py).
  kp       k * the SMPL source hinge's MOTOR kp (GAIN_SOURCE in model/g1_env.py; the waist: Torso, Spine, Chest in
           series; each wrist hinge: the SMPL wrist and hand hinges in series). The SMPL passive spring is left out of the G1 (it stays only in the target angle, see g1_env.py).
  kd       chosen so that each G1 joint keeps the damping ratio the SMPL hinge actually has, passive damping and spring
           included:  zeta_SMPL = (kd + d) / (2 sqrt((kp + k_s) I_SMPL)),  kd_G1 = zeta_SMPL * 2 sqrt(kp_G1 I_G1),
           I = the joint's effective inertia (mass-matrix diagonal, armature included; median over the same frames).
           The G1 keeps its armature (0.01) and its own motor torque limits.
Writes assets/robots/g1/gains_perjoint.json (kp, kd per G1 actuator + the measurements behind them).
usage: uv run python scripts/g1_gain_scaling.py [--stride 2] [--clips-file outputs/g1/search_clips.txt]
"""
import argparse, json, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.g1_env import GAIN_SOURCE, G1_XML   # noqa: E402

SMPL_XML = REPO / "assets/robots/adult/robot.xml"
OUT = REPO / "assets/robots/g1/gains_perjoint.json"
# joint group -> (G1 subtree root body, G1 joint at its root, SMPL body whose subtree it matches); {s} = left / right
GROUPS = {"hip": ("{s}_hip_pitch_link", "{s}_hip_pitch_joint", "{S}_Hip"),
          "knee": ("{s}_knee_link", "{s}_knee_joint", "{S}_Knee"),
          "ankle": ("{s}_ankle_pitch_link", "{s}_ankle_pitch_joint", "{S}_Ankle"),
          "shoulder": ("{s}_shoulder_pitch_link", "{s}_shoulder_pitch_joint", "{S}_Shoulder"),
          "elbow": ("{s}_elbow_link", "{s}_elbow_joint", "{S}_Elbow"),
          "wrist": ("{s}_wrist_pitch_link", "{s}_wrist_pitch_joint", "{S}_Wrist"),
          "waist": ("waist_yaw_link", "waist_yaw_joint", "Torso")}


def group_of(name):
    for g in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist", "waist"):
        if g in name:
            return g


def moment(m, d, body, anchor):
    """|gravity moment of body's subtree about the point anchor| (N m)."""
    r = d.subtree_com[body] - anchor
    w = np.array([0.0, 0.0, -9.81 * m.body_subtreemass[body]])
    return float(np.linalg.norm(np.cross(r, w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--clips-file", default=None, help="task/stem per line (default: every clip with a G1 target)")
    a = ap.parse_args()
    mg = mujoco.MjModel.from_xml_path(str(REPO / G1_XML)); dg = mujoco.MjData(mg)
    ms = mujoco.MjModel.from_xml_path(str(SMPL_XML)); ds = mujoco.MjData(ms)
    clips = sorted(p.parent.name for p in (REPO / "outputs/g1/roundtrip").glob("*/g1.npz"))
    if a.clips_file:
        clips = sorted(Path(c.strip()).name for c in (REPO / a.clips_file).read_text().split() if c.strip())
    origin = {p.stem: p for p in (REPO / "data/origin_motion").glob("*/*.npz")}
    sides = (("left", "L"), ("right", "R"))
    tau = {g: ([], []) for g in GROUPS}            # (G1, SMPL) gravity moments, both sides pooled
    per_clip = {g: {} for g in GROUPS}
    s_act = {ms.actuator(i).name: i for i in range(ms.nu)}
    g_act = [mg.actuator(i).name for i in range(mg.nu)]
    inertia = {n: ([], []) for n in g_act}         # per G1 actuator: (I_G1, I_SMPL source) per frame
    Mg, Ms = np.zeros((mg.nv, mg.nv)), np.zeros((ms.nv, ms.nv))

    def src_hinges(gname):
        key = gname.replace("_joint", "").replace("left_", "").replace("right_", "")
        side = "L_" if gname.startswith("left") else "R_" if gname.startswith("right") else ""
        return [side + h for h in GAIN_SOURCE[key]]

    for stem in clips:
        qg_all = np.load(REPO / f"outputs/g1/roundtrip/{stem}/g1.npz")["qpos"]
        qs_all = np.load(origin[stem])["qpos"]
        T = min(len(qg_all), len(qs_all))
        cg, cs = {g: [] for g in GROUPS}, {g: [] for g in GROUPS}
        for t in range(0, T, a.stride):
            dg.qpos[:] = qg_all[t]; mujoco.mj_forward(mg, dg)
            ds.qpos[:] = qs_all[t]; mujoco.mj_forward(ms, ds)
            for g, (gb, gj, sb) in GROUPS.items():
                for s, S in (sides if "{s}" in gb else (("", ""),)):
                    bg, jg, bs = mg.body(gb.format(s=s)).id, mg.joint(gj.format(s=s)).id, ms.body(sb.format(S=S)).id
                    cg[g].append(moment(mg, dg, bg, dg.xanchor[jg])); cs[g].append(moment(ms, ds, bs, ds.xpos[bs]))
            if t % (5 * a.stride) == 0:
                mujoco.mj_fullM(mg, dg, Mg); mujoco.mj_fullM(ms, ds, Ms)
                for n in g_act:
                    jg = mg.actuator(n).trnid[0]; dofg = mg.jnt_dofadr[jg]
                    src = src_hinges(n)
                    dofs = ms.jnt_dofadr[ms.actuator_trnid[s_act[src[0]], 0]]   # the waist: Torso, the lowest segment
                    inertia[n][0].append(Mg[dofg, dofg]); inertia[n][1].append(Ms[dofs, dofs])
        for g in GROUPS:
            x, y = np.array(cs[g]), np.array(cg[g])
            tau[g][0].extend(y); tau[g][1].extend(x)
            per_clip[g][stem] = float(x @ y / (x @ x)) if x @ x > 1e-9 else float("nan")

    k, r2 = {}, {}
    print(f"gravity-moment ratio G1 / adult over {len(clips)} clips (every {a.stride}th frame), both sides pooled:")
    for g in GROUPS:
        y, x = np.array(tau[g][0]), np.array(tau[g][1])
        k[g] = float(x @ y / (x @ x)); r2[g] = float(1 - np.sum((y - k[g] * x) ** 2) / np.sum(y ** 2))
        pc = np.array([v for v in per_clip[g].values() if np.isfinite(v)])
        print(f"  {g:9s} k {k[g]:.3f}  R^2 {r2[g]:.3f}  per clip {np.median(pc):.3f} [{pc.min():.3f}, {pc.max():.3f}]")

    kp, kd, rows = {}, {}, {}
    for i, n in enumerate(g_act):
        src = [s_act[h] for h in src_hinges(n)]
        s_kp = -ms.actuator_biasprm[src, 1]; s_kd = -ms.actuator_biasprm[src, 2]
        trn = ms.actuator_trnid[src, 0]
        s_ks, s_d = ms.jnt_stiffness[trn], ms.dof_damping[ms.jnt_dofadr[trn]]
        series = lambda v: 1.0 / np.sum(1.0 / v)
        Ks, Ds, Kmotor = series(s_kp + s_ks), series(s_kd + s_d), series(s_kp)
        Ig, Is = float(np.median(inertia[n][0])), float(np.median(inertia[n][1]))
        zeta = Ds / (2 * np.sqrt(Ks * Is))
        kp[n] = float(k[group_of(n)] * Kmotor)
        kd[n] = float(zeta * 2 * np.sqrt(kp[n] * Ig))
        rows[n] = dict(group=group_of(n), k=k[group_of(n)], smpl_src=src_hinges(n), smpl_K=float(Ks), smpl_D=float(Ds),
                       smpl_motor_kp=float(Kmotor), I_g1=Ig, I_smpl=Is, Ir=Ig / Is, zeta=float(zeta),
                       f_smpl=float(np.sqrt(Ks / Is) / (2 * np.pi)), f_g1=float(np.sqrt(kp[n] / Ig) / (2 * np.pi)))
    print("\nper actuator (left side; right = mirror):  kp  kd  | zeta (kept)  | natural freq SMPL -> G1")
    for n in g_act:
        if n.startswith("right"):
            continue
        r = rows[n]
        print(f"  {n.replace('_joint', ''):20s} kp {kp[n]:6.1f}  kd {kd[n]:5.2f} | zeta {r['zeta']:.2f} | "
              f"Ir {r['Ir']:.3f} | f {r['f_smpl']:.1f} -> {r['f_g1']:.1f} Hz")
    OUT.write_text(json.dumps(dict(
        note="per-joint G1 PD gains (scripts/g1_gain_scaling.py): kp = k_group * SMPL motor kp; kd keeps the SMPL "
             "hinge's damping ratio (motor + passive damping, spring included in its stiffness); armature 0.01 kept",
        clips=clips, stride=a.stride, k_group=k, r2_group=r2, per_clip_k=per_clip, kp=kp, kd=kd, detail=rows), indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
