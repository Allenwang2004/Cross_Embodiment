#!/usr/bin/env python3
"""g1_static_torque.py -- can the G1 hold a clip's support poses at all, given its motor torque limits?

Quasi-static check, frame by frame, of the clip mapped onto the G1 kinematically (outputs/g1/roundtrip/<stem>/g1.npz)
and, for comparison, of the adult's own original clip:
  contacts  the body parts the ADULT has on the floor in that frame (hands, head, feet; geometry within 2 cm), put on
            the same parts of the body checked as a contact PATCH: the part's collision-geometry points (box corners,
            capsule / sphere bottoms, mesh vertices) within 1.5 cm of its lowest point, up to 8 spread over the patch,
            each with its own force -- so a flat hand can shift its centre of pressure instead of the wrist doing all
            the work. A G1 hand hovering a few cm is treated as touching, as a contact-aware target would make it
  forces    contact forces f (friction pyramid, mu = 1, MuJoCo's floor) that balance gravity on the free base,
            qfrc_bias[:6] = J_base^T f at rest, chosen to minimise sum_j (tau_j / tau_max_j)^2 over the motors,
            tau = qfrc_bias[6:] - J_joint^T f  (scipy SLSQP)
  verdict   per frame the joint torques that this needs against the motor limits (G1: actuatorfrcrange; adult: its
            actuator forcerange); a frame whose base no such f balances (residual > 2% of the body weight: the COM is
            outside the support, so the frame is a moment of a dynamic motion) is counted as 'no static balance' and
            left out of the torque statistics
Static gravity only: the moving motion needs at least this much. Prints per joint group the median / 90th percentile
of |tau| / limit and the share of frames above the limit. Writes outputs/g1/static_torque/<stem>.json.
usage: uv run python scripts/g1_static_torque.py --clip headstand/headstand_37 [--clip ...]
"""
import argparse, json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
MU, NEAR, PATCH, NPTS = 1.0, 0.02, 0.015, 8
# contact part -> body-name keys on each body
PARTS = {"adult": {"L_hand": ("L_Wrist", "L_Hand"), "R_hand": ("R_Wrist", "R_Hand"), "head": ("Head",),
                   "L_foot": ("L_Ankle", "L_Toe"), "R_foot": ("R_Ankle", "R_Toe")},
         "g1": {"L_hand": ("left_wrist",), "R_hand": ("right_wrist",), "head": ("torso_link",),
                "L_foot": ("left_ankle_roll",), "R_foot": ("right_ankle_roll",)}}
GROUPS = {"adult": {"shoulder": ("Shoulder",), "elbow": ("Elbow",), "wrist": ("Wrist",), "neck/head": ("Neck", "Head"),
                    "waist": ("Torso", "Spine", "Chest"), "hip": ("Hip",), "knee": ("Knee",), "ankle": ("Ankle", "Toe")},
          "g1": {"shoulder": ("shoulder",), "elbow": ("elbow",), "wrist": ("wrist",), "waist": ("waist",),
                 "hip": ("hip",), "knee": ("knee",), "ankle": ("ankle",)}}


class Body:
    def __init__(self, kind, xml):
        self.kind = kind
        self.m = mujoco.MjModel.from_xml_path(str(REPO / xml)); self.d = mujoco.MjData(self.m)
        m = self.m
        self.floor = m.geom("floor").id
        coll = lambda g: m.geom_contype[g] | m.geom_conaffinity[g]
        self.part_geoms = {p: [g for g in range(m.ngeom) if coll(g) and any(k in m.body(m.geom_bodyid[g]).name for k in keys)]
                           for p, keys in PARTS[kind].items()}
        dof = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.act_dof = dof
        lim = m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1] if kind == "g1" else m.actuator_forcerange[:, 1]
        self.lim = lim
        names = [m.actuator(i).name for i in range(m.nu)]
        self.groups = {g: [i for i, n in enumerate(names) if any(k in n for k in keys)] for g, keys in GROUPS[kind].items()}
        self.ft = np.zeros(6)
        self.weight = 9.81 * m.body_subtreemass[1]

    def points(self, part):
        """the part's contact patch: world points within PATCH of its lowest collision point, up to NPTS spread out."""
        m, d = self.m, self.d
        P, B = [], []
        for g in self.part_geoms[part]:
            c, R, sz, t = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g], m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                loc = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * sz
                pts = c + loc @ R.T
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                pts = np.array([c + R[:, 2] * h * sz[1] for h in (-1, 1)]) - [0, 0, sz[0]]
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                pts = (c - [0, 0, sz[0]])[None]
            elif t == mujoco.mjtGeom.mjGEOM_MESH:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                pts = c + m.mesh_vert[a:a + n].astype(np.float64) @ R.T
            else:
                continue
            P.append(pts); B += [m.geom_bodyid[g]] * len(pts)
        P, B = np.concatenate(P), np.array(B)
        keep = P[:, 2] < P[:, 2].min() + PATCH
        P, B = P[keep], B[keep]
        pick = {int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0])), int(np.argmin(P[:, 1])), int(np.argmax(P[:, 1])),
                int(np.argmin(P[:, 0] + P[:, 1])), int(np.argmax(P[:, 0] + P[:, 1])), int(np.argmin(P[:, 0] - P[:, 1])),
                int(np.argmax(P[:, 0] - P[:, 1]))}
        return [(B[i], P[i]) for i in sorted(pick)][:NPTS]

    def near(self, part):
        """distance of the part to the floor and the part's point nearest it (body id, world point)."""
        best = (np.inf, None, None)
        for g in self.part_geoms[part]:
            dist = mujoco.mj_geomDistance(self.m, self.d, g, self.floor, 1.0, self.ft)
            if dist < best[0]:
                best = (dist, self.m.geom_bodyid[g], self.ft[:3].copy())
        return best

    def set(self, q):
        self.d.qpos[:] = q; self.d.qvel[:] = 0; mujoco.mj_forward(self.m, self.d)

    def solve(self, parts):
        """min-torque static contact forces on the given parts; returns (|tau| / limit per actuator, base residual N)."""
        m, d = self.m, self.d
        J = []
        for p in parts:
            for b, pt in self.points(p):
                jp = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, None, pt, b); J.append(jp)
        bias = d.qfrc_bias.copy()
        if not J:
            tau = bias[self.act_dof]; return np.abs(tau) / self.lim, float(np.linalg.norm(bias[:6]) / self.weight)
        J = np.concatenate(J, 0)                                   # (3k, nv)
        Jb, Ja = J[:, :6].T, J[:, self.act_dof].T                   # base (6, 3k), motors (nu, 3k)
        W = 1.0 / self.lim
        k = len(J) // 3
        A = W[:, None] * Ja; c = W * bias[self.act_dof]
        obj = lambda f: float(np.sum((c - A @ f) ** 2)), lambda f: -2 * A.T @ (c - A @ f)
        cons = [{"type": "eq", "fun": lambda f: Jb @ f - bias[:6], "jac": lambda f: Jb}]
        G = []
        for i in range(k):                                        # friction pyramid: mu fz -/+ fx >= 0, mu fz -/+ fy >= 0
            for ax in (0, 1):
                for sgn in (1, -1):
                    row = np.zeros(3 * k); row[3 * i + 2] = MU; row[3 * i + ax] = -sgn; G.append(row)
            row = np.zeros(3 * k); row[3 * i + 2] = 1; G.append(row)
        G = np.array(G)
        cons.append({"type": "ineq", "fun": lambda f: G @ f, "jac": lambda f: G})
        f0 = np.zeros(3 * k); f0[2::3] = max(bias[2], 1.0) / k
        r = minimize(obj[0], f0, jac=obj[1], constraints=cons, method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
        f = r.x
        resid = float(np.linalg.norm(Jb @ f - bias[:6]) / self.weight)     # in body weights
        tau = bias[self.act_dof] - Ja @ f
        return np.abs(tau) / self.lim, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True)
    a = ap.parse_args()
    out_dir = REPO / "outputs/g1/static_torque"; out_dir.mkdir(parents=True, exist_ok=True)
    adult, g1 = Body("adult", "assets/robots/adult/robot.xml"), Body("g1", "assets/robots/g1/scene.xml")
    for clip in a.clip:
        stem = clip.split("/")[1]
        ref = np.load(REPO / f"data/origin_motion/{clip}.npz")["qpos"]
        gq = np.load(REPO / f"outputs/g1/roundtrip/{stem}/g1.npz")["qpos"]
        T = min(len(ref), len(gq))
        res = {"adult": [], "g1": []}; frames = []
        for t in range(T):
            adult.set(ref[t])
            parts = [p for p in PARTS["adult"] if adult.near(p)[0] < NEAR]
            if not any("hand" in p for p in parts):
                continue                                           # only the frames the adult has a hand down
            frames.append((t, parts))
            for name, b, q in (("adult", adult, ref[t]), ("g1", g1, gq[t])):
                b.set(q); r, resid = b.solve(parts); res[name].append((r, resid))
        summ = {"clip": clip, "frames": len(frames), "contacts": {p: sum(p in ps for _, ps in frames) for p in PARTS["adult"]}}
        print(f"\n{stem}: {len(frames)} frames with an adult hand down; contacts used: "
              + ", ".join(f"{p} {n}" for p, n in summ["contacts"].items()))
        for name, b in (("adult", adult), ("g1", g1)):
            bal = np.array([e < 0.02 for _, e in res[name]])
            R = np.array([r for r, _ in res[name]])[bal]
            rows = {}
            for g, idx in b.groups.items():
                x = R[:, idx].max(1)                               # the group's worst joint per frame
                rows[g] = dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), over=float(np.mean(x > 1)),
                               tau_med_Nm=float(np.median((R[:, idx] * b.lim[idx]).max(1))))
            summ[name] = dict(static_balance=float(bal.mean()), groups=rows,
                              any_over=float(np.mean(R.max(1) > 1)))
            print(f"  {name:5s} static balance possible in {bal.sum()} / {len(bal)} frames; in those, some joint over its "
                  f"limit in {100 * np.mean(R.max(1) > 1):.0f}%")
            for g, x in rows.items():
                print(f"      {g:9s} |tau|/limit median {x['median']:.2f}  p90 {x['p90']:.2f}  over limit {100 * x['over']:3.0f}%"
                      f"   (median {x['tau_med_Nm']:.1f} Nm)")
        (out_dir / f"{stem}.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
