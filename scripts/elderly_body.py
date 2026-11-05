#!/usr/bin/env python3
"""elderly_body.py -- the original adult turned into a stooped elderly walker by RESTRICTING a band per joint (nothing is
locked: every actuator still acts), plus a reference walk for it.

  --restrict joint   the joint's range (MJCF range) is the band; the policy may still command any target angle and the
                     servo then pushes against the limit
  --restrict action  the actuator's ctrlrange is cut so the PD target angle it can command spans exactly the band; the
                     joint range stays the adult's (each actuator maps ctrl in [-1, 1] linearly onto its joint's range:
                     target = (gain * u + bias0) / kp)
  --restrict both    both

  body      assets/robots/<label>/robot.xml: assets/robots/adult/robot.xml with
              spine  Torso_x / Spine_x / Chest_x limited to +-SPINE_BAND deg around a forward bend whose pelvis -> neck
                     lean is --stoop deg (the original walk leans ~3), the added flexion split SPINE_SPLIT (mostly lumbar,
                     so the chest turns little more than the lean and the neck can still bring the face level): it
                     cannot straighten up
              head   Neck_x / Head_x limited to the band the reference uses to keep the face level (looking ahead)
              arms   both Shoulder_x/y/z and Elbow_y limited to half the range the walk uses (plus MARGIN)
              hips   with --limit-hip-extension, both Hip_x capped at the reference's maximum (+MARGIN)
            joint limits made stiff (solreflimit --solref); parameter.json / skeleton.json copied from the adult.
            (Bands are described below as joint ranges; with --restrict action they are target-angle ranges.)
  retarget  data/<label>/retargeting_motion/<clip>.npz: the adult's own walk slowed --slow x (its first 1/slow
            stretched to the full length: root position and joints linear, root rotation slerp), then
              spine  held at the stoop
              head   Neck_x / Head_x pitched back so the head faces the way the original head faced
              arms   centre + 0.5 x (walk angle - its median) on Shoulder_x/y/z + Elbow_y: half the swing; the centre
                     solved by IK so the arms hang from the bent trunk the way they hung from the upright one
            and ground-corrected like the other retargets. Bands come from the steady walk (after the clip's opening
            squat); the opening frames are clipped into them.
  also      outputs/elderly/<label>_young_slow.npz: the slowed walk on the unchanged adult (for side-by-side videos).
usage: uv run python scripts/elderly_body.py --clip move-ego-0-2/move-ego-0-2_4 [--stoop 40 --slow 2]
"""
import argparse, json, shutil, sys
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from qpos_retarget import ground_correct_qpos   # noqa: E402

SPINE = ("Torso_x", "Spine_x", "Chest_x")
SPINE_SPLIT = {"Torso_x": 0.7, "Spine_x": 0.15, "Chest_x": 0.15}   # share of the added flexion
HEAD = ("Neck_x", "Head_x")
ARM = ("Shoulder_x", "Shoulder_y", "Shoulder_z", "Elbow_y")
SPINE_BAND = 5.0   # deg either side of the stoop
MARGIN = 3.0       # deg added to each side of the reference's own min / max for the head and arm bands
OPENING = 30       # original frames of the clip's opening squat: left out of the bands (the reference is clipped into them)
TRUNK = ("Pelvis", "Torso", "Spine", "Chest", "Neck", "Head", "L_Hip", "R_Hip", "L_Knee", "R_Knee")


def slerp(a, b, w):
    if a @ b < 0:
        b = -b
    c = np.clip(a @ b, -1.0, 1.0)
    if c > 0.9995:
        r = a + w * (b - a)
    else:
        th = np.arccos(c)
        r = (np.sin((1 - w) * th) * a + np.sin(w * th) * b) / np.sin(th)
    return r / np.linalg.norm(r)


def slow_down(q, k):
    """the first len(q) / k frames stretched to len(q) frames"""
    T = len(q)
    s = np.arange(T) / k
    i0 = np.floor(s).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = s - i0
    out = (1 - w)[:, None] * q[i0] + w[:, None] * q[i1]
    for t in range(T):
        out[t, 3:7] = slerp(q[i0[t], 3:7], q[i1[t], 3:7], w[t])
    return out


class Kin:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.adr = {m.joint(j).name: int(m.jnt_qposadr[j]) for j in range(m.njnt)}
        self.jid = {m.joint(j).name: j for j in range(m.njnt)}
        self.bid = {m.body(b).name: b for b in range(m.nbody)}

    def set(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)

    def p(self, b):
        return self.d.xpos[self.bid[b]].copy()

    def heading(self):
        """columns: forward, left, up (forward horizontal, perpendicular to the hip line)"""
        left = self.p("L_Hip") - self.p("R_Hip"); left[2] = 0; left /= np.linalg.norm(left)
        up = np.array([0.0, 0.0, 1.0])
        return np.stack([np.cross(left, up), left, up], 1)

    def lean(self):
        """signed forward lean (deg) of the pelvis -> neck line"""
        v = self.heading().T @ (self.p("Neck") - self.p("Pelvis"))
        return float(np.degrees(np.arctan2(v[0], v[2])))

    def head_pitch(self, f_loc):
        return float(np.degrees(np.arcsin(np.clip(self.d.xmat[self.bid["Head"]].reshape(3, 3) @ f_loc, -1, 1)[2])))

    def arm_dirs(self, s):
        H = self.heading()
        u = H.T @ (self.p(f"{s}_Elbow") - self.p(f"{s}_Shoulder"))
        f = H.T @ (self.p(f"{s}_Wrist") - self.p(f"{s}_Elbow"))
        return u / np.linalg.norm(u), f / np.linalg.norm(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--label", default="adult_elderly")
    ap.add_argument("--stoop", type=float, default=40.0, help="median pelvis -> neck forward lean (deg)")
    ap.add_argument("--slow", type=float, default=2.0, help="slow the walk down this many times")
    ap.add_argument("--solref", type=float, default=0.005, help="joint-limit time constant (s) of the restricted joints")
    ap.add_argument("--spine-band", type=float, default=SPINE_BAND, help="deg either side of the stoop the spine may move")
    ap.add_argument("--limit-hip-extension", action="store_true",
                    help="also cap both Hip_x at the reference's own maximum (+MARGIN): the pelvis cannot tilt back "
                         "(extending the hips) to undo the stoop")
    ap.add_argument("--restrict", choices=["joint", "action", "both"], default="joint",
                    help="what carries the band: the joint range, the actuator's ctrlrange (commandable target angles), "
                         "or both")
    a = ap.parse_args()

    src = REPO / "assets/robots/adult"
    m0 = mujoco.MjModel.from_xml_path(str(src / "robot.xml"))
    k = Kin(m0)
    orig = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
    young = slow_down(orig, a.slow)
    T = len(young)
    steady = np.arange(int(OPENING * a.slow), T)
    rng = {j: np.degrees(m0.jnt_range[k.jid[j]]) for j in k.jid if m0.jnt_type[k.jid[j]] == mujoco.mjtJoint.mjJNT_HINGE}
    D = lambda j, t=slice(None): np.degrees(young[t, k.adr[j]])   # noqa: E731

    k.set(young[0]); h0 = k.heading()
    assert (young[-1, :2] - young[0, :2]) @ h0[:2, 0] > 0, "heading frame's forward is not the walking direction"
    speed = np.linalg.norm(young[-1, :2] - young[0, :2]) / (T / 30)

    # 1) spine: one added flexion, split SPINE_SPLIT over the three axes, so the median lean hits --stoop
    med = {j: float(np.median(D(j, steady))) for j in SPINE}
    frames = steady[::5]

    def lean_at(delta):
        L = []
        for t in frames:
            qq = young[t].copy()
            for j in SPINE:
                qq[k.adr[j]] = np.radians(med[j] + SPINE_SPLIT[j] * delta)
            k.set(qq); L.append(k.lean())
        return float(np.median(L)) - a.stoop
    hi = min((rng[j][1] - a.spine_band - med[j]) / SPINE_SPLIT[j] for j in SPINE)
    lo = max((rng[j][0] + a.spine_band - med[j]) / SPINE_SPLIT[j] for j in SPINE)
    delta = brentq(lean_at, 0.0, hi) if lean_at(0.0) < 0 < lean_at(hi) else brentq(lean_at, lo, 0.0)
    stoop = {j: med[j] + SPINE_SPLIT[j] * delta for j in SPINE}
    out = young.copy()
    for j in SPINE:
        out[:, k.adr[j]] = np.radians(stoop[j])
    lean_young = np.median([(k.set(young[t]), k.lean())[1] for t in frames])

    # 2) head: Neck_x and Head_x shifted by one per-frame offset so the head's pitch equals the original's
    k.set(young[T // 2]); f_loc = k.d.xmat[k.bid["Head"]].reshape(3, 3).T @ k.heading()[:, 0]
    for t in range(T):
        k.set(young[t]); target = k.head_pitch(f_loc)
        base = out[t].copy()

        def err(e):
            qq = base.copy()
            for j in HEAD:
                qq[k.adr[j]] = np.radians(np.clip(D(j, t) + e, *rng[j]))
            k.set(qq)
            return (k.head_pitch(f_loc) - target) ** 2
        e = minimize_scalar(err, bounds=(-80, 80), method="bounded", options={"xatol": 0.05}).x
        for j in HEAD:
            out[t, k.adr[j]] = np.radians(np.clip(D(j, t) + e, *rng[j]))

    # 3) arms: half the joint-space swing, about a centre solved so the arms hang the same way from the bent trunk
    for s in "LR":
        names = [f"{s}_{j}" for j in ARM]
        idx = [k.adr[n] for n in names]
        lb = np.radians([rng[n][0] for n in names]); ub = np.radians([rng[n][1] for n in names])
        Y = young[:, idx]; ymed = np.median(Y[steady], 0)
        U, F = [], []
        for t in range(T):
            k.set(young[t]); u, f = k.arm_dirs(s); U.append(u); F.append(f)
        U, F = np.array(U), np.array(F)
        Ut = U[steady].mean(0) + 0.5 * (U - U[steady].mean(0)); Ut /= np.linalg.norm(Ut, axis=1, keepdims=True)
        Ft = F[steady].mean(0) + 0.5 * (F - F[steady].mean(0)); Ft /= np.linalg.norm(Ft, axis=1, keepdims=True)

        def res(c):
            r = []
            for t in frames:
                qq = out[t].copy(); qq[idx] = c + 0.5 * (Y[t] - ymed); k.set(qq); u, f = k.arm_dirs(s)
                r += [u - Ut[t], f - Ft[t]]
            return np.concatenate(r)
        c = least_squares(res, np.clip(ymed, lb + 1e-6, ub - 1e-6), bounds=(lb, ub)).x
        ang = np.degrees(np.arccos(np.clip(1 - 0.5 * (res(c).reshape(-1, 2, 3) ** 2).sum(-1), -1, 1)))
        print(f"  {s} arm: centre moved " + ", ".join(f"{n.split('_', 1)[1]} {np.degrees(c[i] - ymed[i]):+.0f}"
                                                  for i, n in enumerate(names))
              + f" deg; direction error vs half-swing target: upper arm median {np.median(ang[:, 0]):.1f}, "
                f"forearm {np.median(ang[:, 1]):.1f} deg")
        out[:, idx] = c + 0.5 * (Y - ymed)

    # 4) bands (from the steady walk; the whole reference is clipped into them)
    band = {j: (stoop[j] - a.spine_band, stoop[j] + a.spine_band) for j in SPINE}
    for j in list(HEAD) + [f"{s}_{x}" for s in "LR" for x in ARM]:
        v = np.degrees(out[steady, k.adr[j]])
        band[j] = (max(v.min() - MARGIN, rng[j][0]), min(v.max() + MARGIN, rng[j][1]))
        out[:, k.adr[j]] = np.radians(np.clip(np.degrees(out[:, k.adr[j]]), *band[j]))
    if a.limit_hip_extension:   # Hip_x > 0 is extension (the walk's push-off reaches ~ +11)
        for s in "LR":
            j = f"{s}_Hip_x"; v = np.degrees(out[steady, k.adr[j]])
            band[j] = (rng[j][0], min(v.max() + MARGIN, rng[j][1]))
            out[:, k.adr[j]] = np.radians(np.clip(np.degrees(out[:, k.adr[j]]), *band[j]))

    # 5) body
    dst = REPO / "assets/robots" / a.label; dst.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(src / "robot.xml"))
    unit = (lambda deg: deg) if spec.compiler.degree else np.radians
    J = {j.name: j for j in spec.joints}
    A = {x.name: x for x in spec.actuators}
    ctrl = {}
    for j, (l, h) in band.items():
        if a.restrict in ("joint", "both"):
            J[j].range = [unit(l), unit(h)]
            J[j].limited = mujoco.mjtLimited.mjLIMITED_TRUE
            J[j].solref_limit = [a.solref, 1.0]
        if a.restrict in ("action", "both"):
            i = m0.actuator(j).id   # actuators are named after their joints
            assert m0.actuator_trnid[i, 0] == k.jid[j]
            g, b0, kp = m0.actuator_gainprm[i, 0], m0.actuator_biasprm[i, 0], -m0.actuator_biasprm[i, 1]
            u = sorted([(np.radians(l) * kp - b0) / g, (np.radians(h) * kp - b0) / g])
            ctrl[j] = (max(u[0], -1.0), min(u[1], 1.0))
            A[j].ctrlrange = list(ctrl[j])
    m = spec.compile()
    for j, (l, h) in band.items():
        if a.restrict in ("joint", "both"):
            assert np.allclose(np.degrees(m.jnt_range[m.joint(j).id]), [l, h], atol=1e-3), j
        if a.restrict in ("action", "both"):   # the commandable targets are exactly the band
            i = m.actuator(j).id
            tg = [np.degrees((m.actuator_gainprm[i, 0] * u + m.actuator_biasprm[i, 0]) / -m.actuator_biasprm[i, 1])
                  for u in m.actuator_ctrlrange[i]]
            assert np.allclose(sorted(tg), [max(l, np.degrees(m0.jnt_range[k.jid[j]][0])),
                                            min(h, np.degrees(m0.jnt_range[k.jid[j]][1]))], atol=1e-3), (j, tg, l, h)
    (dst / "robot.xml").write_text(spec.to_xml())
    for f in ("parameter.json", "skeleton.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    # 6) reference: ground correction, arm-trunk contacts
    out, n, shift = ground_correct_qpos(m, out)
    d = mujoco.MjData(m)
    arm = {m.body(f"{s}_{b}").id for s in "LR" for b in ("Shoulder", "Elbow", "Wrist", "Hand")}
    trunk = {m.body(b).id for b in TRUNK}; hits = 0
    for t in range(T):
        d.qpos[:] = out[t]; mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            hits += int((b1 in arm and b2 in trunk) or (b2 in arm and b1 in trunk))
    p = REPO / f"data/{a.label}/retargeting_motion/{a.clip}.npz"; p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, qpos=out, fps=30.0)
    yp = REPO / f"outputs/elderly/{a.label}_young_slow.npz"; yp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(yp, qpos=young, fps=30.0)
    kb = Kin(m)
    lean_old = np.median([(kb.set(out[t]), kb.lean())[1] for t in frames])
    dp = (np.array([(kb.set(out[t]), kb.head_pitch(f_loc))[1] for t in steady])
          - np.array([(k.set(young[t]), k.head_pitch(f_loc))[1] for t in steady]))

    print(f"{a.label}: bands carried by {a.restrict} ({'joint range' if a.restrict == 'joint' else 'ctrlrange' if a.restrict == 'action' else 'joint range + ctrlrange'}), nothing locked -> {dst / 'robot.xml'}")
    for j, (lo_u, hi_u) in ctrl.items():
        print(f"  ctrlrange {j:12s} [-1, 1] -> [{lo_u:+.3f}, {hi_u:+.3f}]")
    print(f"  walk slowed {a.slow:g}x: {speed:.2f} m/s; lean pelvis->neck {lean_young:.1f} -> {lean_old:.1f} deg")
    print(f"  head pitch vs the young walk (steady part): median {np.median(dp):+.1f} deg, worst {dp.min():+.1f} (- = looking lower)")
    for j, (l, h) in band.items():
        y = D(j, steady)
        print(f"  {j:12s} {rng[j][0]:6.0f} .. {rng[j][1]:4.0f}  ->  {l:6.1f} .. {h:6.1f} deg   "
              f"(young walk {y.min():6.1f} .. {y.max():6.1f}; width {h - l:4.1f} vs {np.ptp(y):4.1f})")
    print(f"  reference: lifted {n}/{T} frames (max {shift * 100:.1f} cm), arm-trunk contacts {hits} -> {p}")
    print(f"  young slowed walk on the plain adult -> {yp}")
    json.dump(dict(clip=a.clip, stoop=a.stoop, slow=a.slow, speed=speed, bands=band, restrict=a.restrict, ctrlrange=ctrl),
              open(dst / "elderly.json", "w"), indent=1)


if __name__ == "__main__":
    main()
