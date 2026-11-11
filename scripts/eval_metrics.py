#!/usr/bin/env python3
"""eval_metrics.py -- the evaluation metrics of the baseline comparison, on saved trajectories, on the body itself: every rollout is compared, in that body's own
metres and joint angles, with the reference on that body (reference_body: the child's retargeted clip, or the clip mapped onto the robot). Nothing is mapped back to the adult.
Local path frame: origin at the root, x forward, z up (BFMTrack Sec. 6): p = Rz(-heading) (body position - root position), R = Rz(-heading) body rotation.
Pose vector x = (root height, p of every non-root body, first and third columns of R of every body), the layout of humenv's proprio[:214] (reproduces its emd_numpy on the adult).
  fall_s    seconds with the root below half of the reference's: sum_t [z_t < 0.5 z_ref,t] / 30 (reported for one latent per clip only)
  success   the clip never exceeds 1.5 m of root xy distance or 45 deg of root rotation (geodesic) from the reference; over the clips, the share that succeed
  mpjpe     mean over frames and bodies of |p - p_ref|, cm (BFMTrack Sec. 7.1)
  mmpjpe    each body's largest |p - p_ref| in the clip, averaged over the bodies, cm (BFMTrack Sec. 7.1)
  emd       optimal transport cost (ot.emd2, uniform weights) with the L2 distance between pose vectors (Rubner 2000; Motivo's d_smpl, as humenv.bench.utils.metrics.emd_numpy)
  dtw       L1 distance on the body's joint angles (rad, summed over joints) under the optimal time alignment, steps (1,0) (0,1) (1,1), / the warping path's length (BFMTrack Sec. 7.1)
  mpjae     mean per-joint acceleration error: mean over frames and bodies of |a - a_ref|, a = second difference of p x 30^2, m/s^2 (BFMTrack Sec. 7.1, root-relative reading; not "angle" error)
Writes outputs/eval_metrics/<set>.json. usage: uv run python scripts/eval_metrics.py [child g1 child_amass g1_amass]"""
import glob, json, sys
from pathlib import Path
import mujoco
import numpy as np
import ot
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
XML = {"child": "assets/robots_torque/child/robot_torque_full.xml", "g1": "assets/robots/g1/scene.xml"}
SETS = {"child": ("child", "z0", "best"), "g1": ("g1", "z0", "best"), "child_amass": ("child", "zero_shot", "lso_best"), "g1_amass": ("g1", "zero_shot", "lso_best")}
METRICS = ("fall_s", "mpjpe", "mmpjpe", "emd", "dtw", "mpjae")


def dtw(a, b):
    C = cdist(a, b, "cityblock"); n, m = C.shape; D = np.full((n, m), np.inf); L = np.zeros((n, m), dtype=np.int64); D[0, 0] = C[0, 0]; L[0, 0] = 1
    for i in range(n):
        for j in range(m):
            if i or j:
                c = [(D[x, y], L[x, y]) for x, y in ((i - 1, j), (i, j - 1), (i - 1, j - 1)) if x >= 0 and y >= 0]
                d, ln = min(c); D[i, j] = d + C[i, j]; L[i, j] = ln + 1
    return float(D[-1, -1] / L[-1, -1])


class Body:
    def __init__(self, xml, smpl):
        self.m = mujoco.MjModel.from_xml_path(str(REPO / xml)); self.d = mujoco.MjData(self.m); self.smpl = smpl
        assert self.m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and self.m.body_jntadr[1] == 0, xml

    def hand_floor(self, Q):
        """(T,) per frame, the smallest distance (m) between the floor and the hands: the collision geoms of the bodies named
        *wrist* or *hand* (the human's Wrist and Hand, the G1's wrist roll / pitch / yaw links), mj_geomDistance, capped at 0.5 m"""
        m, d = self.m, self.d
        if not hasattr(self, "_hands"):
            fl = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]; assert len(fl) == 1, "one floor plane"
            hb = {b for b in range(m.nbody) if any(k in m.body(b).name.lower() for k in ("wrist", "hand"))}
            self._floor = fl[0]; self._hands = [g for g in range(m.ngeom) if m.geom_bodyid[g] in hb and (m.geom_contype[g] or m.geom_conaffinity[g])]
            assert self._hands, "no hand geoms"
        out = np.empty(len(Q)); ft = np.zeros(6)
        for t, q in enumerate(Q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            out[t] = min(mujoco.mj_geomDistance(m, d, self._floor, g, 0.5, ft) for g in self._hands)
        return out

    def part_floor(self, Q):
        """(names, (T, P)): per frame, each non-foot body's smallest collision-geom distance (m) to the floor, capped at 0.5 m.
        Feet = bodies named *ankle* or *toe*; the world body is skipped"""
        m, d = self.m, self.d
        if not hasattr(self, "_parts"):
            fl = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]; assert len(fl) == 1, "one floor plane"
            P = {}
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g]); n = m.body(b).name
                if b and not any(k in n.lower() for k in ("ankle", "toe")) and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    P.setdefault(n, []).append(g)
            self._pfloor, self._parts = fl[0], P
        names = list(self._parts); out = np.empty((len(Q), len(names))); ft = np.zeros(6)
        for t, q in enumerate(Q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            for j, n in enumerate(names):
                out[t, j] = min(mujoco.mj_geomDistance(m, d, self._pfloor, g, 0.5, ft) for g in self._parts[n])
        return names, out

    def local(self, Q):
        """(T, B, 3) non-root positions, (T, B + 1, 6) rotations, in each frame's path frame"""
        import humenv.utils as hu
        m, d = self.m, self.d; B = m.nbody - 2; P = np.empty((len(Q), B, 3)); O = np.empty((len(Q), B + 1, 6))
        for t, q in enumerate(Q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            if self.smpl:
                h = hu.calc_heading(hu.remove_base_rot(d.xquat[1][None].copy(), "smpl"))[0]
            else:
                R = d.xmat[1].reshape(3, 3); h = np.arctan2(R[1, 0], R[0, 0])
            c, s = np.cos(-h), np.sin(-h); Z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            P[t] = (d.xpos[2:] - d.xpos[1]) @ Z.T
            O[t] = (Z @ d.xmat[1:].reshape(-1, 3, 3))[:, :, [0, 2]].transpose(0, 2, 1).reshape(-1, 6)
        return P, O


def metrics(body, q, qr):
    T = min(len(q), len(qr)); q, qr = q[:T], qr[:T]; (p, o), (pr, or_) = body.local(q), body.local(qr); e = np.linalg.norm(p - pr, axis=-1)
    pos = np.linalg.norm(q[:, :2] - qr[:, :2], axis=1); rot = np.degrees(2 * np.arccos(np.abs((q[:, 3:7] * qr[:, 3:7]).sum(1)).clip(0, 1)))
    bad = (pos > 1.5) | (rot > 45); why = None
    if bad.any():
        t = int(np.argmax(bad)); why = dict(t_s=t / 30, by="position" if pos[t] > 1.5 else "rotation")
    a, ar = np.diff(p, n=2, axis=0) * 900, np.diff(pr, n=2, axis=0) * 900
    x = np.concatenate([q[:, 2:3], p.reshape(T, -1), o.reshape(T, -1)], 1); xr = np.concatenate([qr[:, 2:3], pr.reshape(T, -1), or_.reshape(T, -1)], 1); w = np.ones(T) / T
    return dict(frames=T, fall_s=float((q[:, 2] < 0.5 * qr[:, 2]).sum() / 30), success=not bad.any(), first_fail=why, mpjpe=float(100 * e.mean()), mmpjpe=float(100 * e.max(0).mean()),
                emd=float(ot.emd2(w, w, cdist(x, xr), numItermax=100000)), dtw=dtw(q[:, 7:], qr[:, 7:]), mpjae=float(np.linalg.norm(a - ar, axis=-1).mean()))


def tracking_metrics(body, q, qr):
    """only mpjpe, mmpjpe and mpjae of metrics(), for many rollouts (no optimal transport or time alignment)"""
    T = min(len(q), len(qr)); (p, _), (pr, _) = body.local(q[:T]), body.local(qr[:T]); e = np.linalg.norm(p - pr, axis=-1)
    return dict(mpjpe=float(100 * e.mean()), mmpjpe=float(100 * e.max(0).mean()), mpjae=float(np.linalg.norm(np.diff(p, n=2, axis=0) * 900 - np.diff(pr, n=2, axis=0) * 900, axis=-1).mean()))


def main():
    out = REPO / "outputs/eval_metrics"; out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or list(SETS):
        kind, a_key, b_key = SETS[name]; body = Body(XML[kind], kind == "child"); res = {}
        for f in sorted(glob.glob(str(REPO / f"outputs/user_survey/{name}/trajectories/*/video.json"))):
            D = Path(f).parent; load = lambda k: np.load(D / f"{k}.npz")["qpos"].astype(np.float64)
            qr = load("reference_body"); res[D.name] = {k: metrics(body, load(k), qr) for k in (a_key, b_key)}
        keys = (a_key, b_key)
        summ = {k: dict(success=sum(r[k]["success"] for r in res.values()), clips_fall_ge_0p1s=sum(r[k]["fall_s"] >= 0.1 for r in res.values()), fall_total_s=float(sum(r[k]["fall_s"] for r in res.values())),
                        **{m: float(np.median([r[k][m] for r in res.values()])) for m in METRICS}) for k in keys}
        better = {m: sum(r[b_key][m] < r[a_key][m] for r in res.values()) for m in METRICS[1:]}
        (out / f"{name}.json").write_text(json.dumps(dict(set=name, body=kind, bodies=body.m.nbody - 2, joints=body.m.nq - 7, keys=keys, n=len(res), summary=summ, lower_after=better, clips=res), indent=1))
        print(f"== {name} n={len(res)} bodies {body.m.nbody - 2} joints {body.m.nq - 7} lower after {better}")
        for k in keys:
            s = summ[k]; print(f"  {k:10s} success {s['success']}/{len(res)} fall clips {s['clips_fall_ge_0p1s']} total {s['fall_total_s']:.1f}s | MPJPE {s['mpjpe']:.2f} MMPJPE {s['mmpjpe']:.2f} EMD {s['emd']:.3f} DTW {s['dtw']:.3f} MPJAE {s['mpjae']:.3f}", flush=True)
        print("  fails after search:", [(c[:26], r[b_key]["first_fail"]["by"], round(r[b_key]["first_fail"]["t_s"], 1)) for c, r in res.items() if not r[b_key]["success"]])


if __name__ == "__main__":
    main()
