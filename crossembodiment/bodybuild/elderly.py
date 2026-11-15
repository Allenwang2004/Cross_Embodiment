"""elderly.py -- the Elderly builder: the adult turned into a stooped elderly walker by restricting a band per joint (nothing is
locked: every actuator still acts), and its reference walk; moved from scripts/elderly_body.py (OOP refactor step 9).

The body and its reference are planned together (plan()): the bands come from the modified walk, so the reference is part of the
build here (reference()), not a Retargeter. The edit (restrict):
  "joint"   the joint's range (MJCF range) is the band; the policy may still command any target angle and the servo pushes
            against the limit
  "action"  the actuator's ctrlrange is cut so the PD target angle it can command spans exactly the band (target =
            (gain * u + bias0) / kp); the joint range stays the adult's
  "both"    both
  spine  Torso_x / Spine_x / Chest_x limited to +-spine_band deg around a forward bend whose pelvis -> neck lean is `stoop` deg
         (the added flexion split SPINE_SPLIT, mostly lumbar)
  head   Neck_x / Head_x limited to the band the reference uses to keep the face level
  arms   both Shoulder_x/y/z and Elbow_y limited to half the range the walk uses (plus MARGIN)
  hips   with limit_hip_extension, both Hip_x capped at the reference's maximum (+MARGIN)
The reference: the adult's walk slowed `slow` x (its first 1/slow stretched to the full length), the spine held at the stoop, the head
pitched back to face where it faced, half the arm swing about an IK-solved centre, clipped into the bands, then ground-corrected on
the built body. Also outputs/elderly/<name>_young_slow.npz (the slowed walk on the plain adult) and <robots>/<name>/elderly.json.
Every number is computed as elderly_body.py did (tests/test_bodybuild.py runs both).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .. import paths
from .builders import SpecBody
from .retarget import ground_correct_qpos, save_clip

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


class Elderly(SpecBody):
    """elderly_body.py (module docstring); build() plans the reference first when it was not planned"""

    def __init__(self, clip, name="adult_elderly", stoop=40.0, slow=2.0, solref=0.005, spine_band=SPINE_BAND,
                 limit_hip_extension=False, restrict="joint", **kw):
        assert restrict in ("joint", "action", "both"), restrict
        super().__init__(name, **kw)
        self.clip, self.stoop, self.slow, self.solref, self.spine_band = clip, stoop, slow, solref, spine_band
        self.limit_hip_extension, self.restrict = limit_hip_extension, restrict
        self.planned = False

    # ------------------------------------------------------------ steps 1-4: the reference (before the lift) and the bands
    def plan(self, log=print):
        from scipy.optimize import brentq, least_squares, minimize_scalar
        a = self
        m0 = mujoco.MjModel.from_xml_path(str(self.source_xml))
        k = Kin(m0)
        orig = np.load(paths.REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
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
            log(f"  {s} arm: centre moved " + ", ".join(f"{n.split('_', 1)[1]} {np.degrees(c[i] - ymed[i]):+.0f}"
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
        self.m0, self.k, self.young, self.out, self.band, self.rng = m0, k, young, out, band, rng
        self.steady, self.frames, self.f_loc, self.D = steady, frames, f_loc, D
        self.speed, self.lean_young = speed, lean_young
        self.planned = True
        return out, band

    # ------------------------------------------------------------ step 5: the body
    def edit(self, spec):
        if not self.planned:
            self.plan()
        a, m0, k = self, self.m0, self.k
        unit = (lambda deg: deg) if spec.compiler.degree else np.radians
        J = {j.name: j for j in spec.joints}
        A = {x.name: x for x in spec.actuators}
        ctrl = {}
        for j, (l, h) in self.band.items():
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
        self.ctrl = ctrl

    def check(self, m):
        a, m0, k = self, self.m0, self.k
        for j, (l, h) in self.band.items():
            if a.restrict in ("joint", "both"):
                assert np.allclose(np.degrees(m.jnt_range[m.joint(j).id]), [l, h], atol=1e-3), j
            if a.restrict in ("action", "both"):   # the commandable targets are exactly the band
                i = m.actuator(j).id
                tg = [np.degrees((m.actuator_gainprm[i, 0] * u + m.actuator_biasprm[i, 0]) / -m.actuator_biasprm[i, 1])
                      for u in m.actuator_ctrlrange[i]]
                assert np.allclose(sorted(tg), [max(l, np.degrees(m0.jnt_range[k.jid[j]][0])),
                                                min(h, np.degrees(m0.jnt_range[k.jid[j]][1]))], atol=1e-3), (j, tg, l, h)

    def edit_parameters(self):
        return dict(clip=self.clip, stoop_deg=self.stoop, slow=self.slow, restrict=self.restrict, spine_band_deg=self.spine_band,
                    solref_limit=self.solref, limit_hip_extension=self.limit_hip_extension,
                    bands_deg={j: [float(lo), float(hi)] for j, (lo, hi) in self.band.items()},
                    ctrlrange={j: [float(lo), float(hi)] for j, (lo, hi) in self.ctrl.items()})

    def build(self, log=print):
        xml = super().build(log)
        with open(self.dir / "elderly.json", "w") as f:
            json.dump(dict(clip=self.clip, stoop=self.stoop, slow=self.slow, speed=self.speed, bands=self.band,
                           restrict=self.restrict, ctrlrange=self.ctrl), f, indent=1)
        return xml

    # ------------------------------------------------------------ step 6: the reference
    def reference(self, data_dir=None, outputs_dir=None, log=print):
        """ground-correct the planned walk on the built body, count arm-trunk contacts, write data/<name>/retargeting_motion/<clip>.npz
        and outputs/elderly/<name>_young_slow.npz; print elderly_body.py's report. Returns the reference path."""
        a, m, k, T = self, self.model, self.k, len(self.young)
        data_dir = Path(data_dir) if data_dir is not None else paths.REPO / paths.DATA_REL
        outputs_dir = Path(outputs_dir) if outputs_dir is not None else paths.OUTPUTS
        out, n, shift = ground_correct_qpos(m, self.out)
        d = mujoco.MjData(m)
        arm = {m.body(f"{s}_{b}").id for s in "LR" for b in ("Shoulder", "Elbow", "Wrist", "Hand")}
        trunk = {m.body(b).id for b in TRUNK}; hits = 0
        for t in range(T):
            d.qpos[:] = out[t]; mujoco.mj_forward(m, d)
            for c in d.contact[:d.ncon]:
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                hits += int((b1 in arm and b2 in trunk) or (b2 in arm and b1 in trunk))
        p = save_clip(data_dir / a.name / "retargeting_motion" / f"{a.clip}.npz", out)
        yp = save_clip(outputs_dir / "elderly" / f"{a.name}_young_slow.npz", self.young)
        kb = Kin(m)
        frames, steady, f_loc, D = self.frames, self.steady, self.f_loc, self.D
        lean_old = np.median([(kb.set(out[t]), kb.lean())[1] for t in frames])
        dp = (np.array([(kb.set(out[t]), kb.head_pitch(f_loc))[1] for t in steady])
              - np.array([(k.set(self.young[t]), k.head_pitch(f_loc))[1] for t in steady]))
        what = 'joint range' if a.restrict == 'joint' else 'ctrlrange' if a.restrict == 'action' else 'joint range + ctrlrange'
        log(f"{a.name}: bands carried by {a.restrict} ({what}), nothing locked -> {self.xml}")
        for j, (lo_u, hi_u) in self.ctrl.items():
            log(f"  ctrlrange {j:12s} [-1, 1] -> [{lo_u:+.3f}, {hi_u:+.3f}]")
        log(f"  walk slowed {a.slow:g}x: {self.speed:.2f} m/s; lean pelvis->neck {self.lean_young:.1f} -> {lean_old:.1f} deg")
        log(f"  head pitch vs the young walk (steady part): median {np.median(dp):+.1f} deg, worst {dp.min():+.1f} (- = looking lower)")
        for j, (l, h) in self.band.items():
            y = D(j, steady)
            log(f"  {j:12s} {self.rng[j][0]:6.0f} .. {self.rng[j][1]:4.0f}  ->  {l:6.1f} .. {h:6.1f} deg   "
                f"(young walk {y.min():6.1f} .. {y.max():6.1f}; width {h - l:4.1f} vs {np.ptp(y):4.1f})")
        log(f"  reference: lifted {n}/{T} frames (max {shift * 100:.1f} cm), arm-trunk contacts {hits} -> {p}")
        log(f"  young slowed walk on the plain adult -> {yp}")
        return p
