#!/usr/bin/env python3
"""smpl_skin.py -- dress any of our humanoid bodies in an SMPL mesh, for rendering only.

The adult humanoid is SMPL-neutral at betas = 0: its 24 bodies sit on SMPL's 24 rest joints (0.1 mm, after SMPL's
y-up -> z-up rotation, +90 deg about x). So SMPL's own skinning weights bind the mesh to our bodies directly.
A scaled body (child, giant, ...) gets the same mesh deformed part by part: for each body j an affine map
(A_j, t_j) is least-squares fitted from the ADULT's to THIS body's rest-pose landmarks of that body (its joint, its
child joints, its geoms' end caps / radii / box corners), and each vertex moves by the skin-weighted blend
sum_j w_j (A_j v + t_j). Limb length, girth, torso and head size therefore follow the body's own XML.
The mesh is added as a MuJoCo <skin> (visual only: physics, qpos, contacts are untouched) and the body's capsules
can be hidden (moved to geom group 5, not drawn by default).

SMPL model: data/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl (from SMPL_python_v.1.1.0.zip, licence in
data/smpl/LICENSE.txt, not redistributed; data/ is git-ignored); a compact copy data/smpl/smpl_neutral_min.npz is
written on first use.
  python scripts/smpl_skin.py check <body>   # physics unchanged with the skin, landmark fit residuals
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SMPL_PKL = REPO / "data/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
SMPL_NPZ = REPO / "data/smpl/smpl_neutral_min.npz"
ADULT_XML = REPO / "assets/robots_torque/m2c_t000/robot_torque_full.xml"
SMPL_ID = {"Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Torso": 3, "L_Knee": 4, "R_Knee": 5, "Spine": 6, "L_Ankle": 7,
           "R_Ankle": 8, "Chest": 9, "L_Toe": 10, "R_Toe": 11, "Neck": 12, "L_Thorax": 13, "R_Thorax": 14,
           "Head": 15, "L_Shoulder": 16, "R_Shoulder": 17, "L_Elbow": 18, "R_Elbow": 19, "L_Wrist": 20,
           "R_Wrist": 21, "L_Hand": 22, "R_Hand": 23}
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
SKIN_RGBA = (0.70, 0.74, 0.80, 1.0)


def _load_pkl():
    class Ch:                                       # the v1.1.0 pickle stores chumpy arrays; keep their data only
        def __setstate__(self, st):
            self.__dict__.update(st if isinstance(st, dict) else {"x": st})

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            return Ch if mod.startswith("chumpy") else super().find_class(mod, name)

    d = Unpickler(open(SMPL_PKL, "rb"), encoding="latin1").load()
    g = lambda v: np.asarray(v.__dict__["x"] if isinstance(v, Ch) else v)
    return dict(v_template=g(d["v_template"]), f=g(d["f"]).astype(np.int32), weights=g(d["weights"]),
                J_regressor=d["J_regressor"].toarray(), shapedirs=g(d["shapedirs"])[:, :, :10])


def smpl():
    if not SMPL_NPZ.exists():
        np.savez_compressed(SMPL_NPZ, **_load_pkl())
    z = np.load(SMPL_NPZ)
    if "shapedirs" not in z:                        # older cache without the shape basis
        SMPL_NPZ.unlink()
        return smpl()
    return {k: z[k] for k in ("v_template", "f", "weights", "J_regressor", "shapedirs")}


KID_NPY = REPO / "data/smpl/smpl_kid_template.npy"      # AGORA's SMPL-topology kid (SMIL-like) template


def _segments():
    """SMPL joint -> its child joints (kintree)."""
    parent = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13,
              17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
    ch = {j: [c for c, p in parent.items() if p == j] for j in range(24)}
    return parent, ch


def _taubin_torso(v, iters):
    """Taubin (lambda / mu) smoothing away from the head, neck, hands and feet: removes the kid template's stretched
    infant skin folds without shrinking the body or touching the face."""
    import scipy.sparse as sp
    S = smpl()
    F, W = S["f"], S["weights"]
    r = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    c = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(v), len(v)))
    A.data[:] = 1
    L = sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
    keep = W[:, [15, 12, 20, 21, 22, 23, 7, 8, 10, 11]].sum(1)
    a = np.clip(1 - keep / 0.5, 0, 1)[:, None]
    for _ in range(iters):
        v = v + 0.5 * a * (L @ v - v)
        v = v - 0.53 * a * (L @ v - v)
    return v


def kid_template(t, t_head=None):
    """SMPL rest mesh with child features: the adult template blended a fraction t toward AGORA's kid template
    (t = 1: the infant-like template itself), then every part scaled UNIFORMLY about its joint and moved so the
    joints land on the adult's. The skeleton is the adult's again (the per-body maps set the real proportions), while
    the face, head shape and soft-tissue roundness of the kid survive. Part scale = adult / kid length of the part's
    bone (mean over its child bones); leaf parts (hands, toes) take their parent's; the head takes the ratio of head
    heights. t_head (default t) is the blend on the head and neck (by their skinning weight): the face carries most of
    the child look, while the kid's infant torso creases once stretched to a child's height (t >~ 0.7)."""
    S = smpl()
    va, Jr, W = S["v_template"], S["J_regressor"], S["weights"]
    k = np.load(KID_NPY)
    k = k - k.mean(0) + va.mean(0)
    th = t if t_head is None else t_head
    a = np.clip(W[:, 15] + W[:, 12], 0, 1)[:, None]                  # 1 on the head, fading through the neck
    v = va + (t + (th - t) * a) * (k - va)
    Ja, Jk = Jr @ va, Jr @ v
    parent, ch = _segments()
    s = np.ones(24)
    for j in range(24):
        if ch[j] and j != 12:
            s[j] = np.mean([np.linalg.norm(Ja[c] - Ja[j]) / np.linalg.norm(Jk[c] - Jk[j]) for c in ch[j]])
    head = W[:, 15] > 0.5
    s[15] = np.ptp(va[head, 1]) / np.ptp(v[head, 1])
    s[12] = s[15]                                      # neck follows the head (its bone to the head is tiny)
    for j in (10, 11, 22, 23):
        s[j] = s[parent[j]]
    return sum(W[:, j:j + 1] * (Ja[j] + s[j] * (v - Jk[j])) for j in range(24))


def rest_pose(m):
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return d


def landmarks(m, d, body):
    """Rest-pose world points that pin down body `body`'s shape: its joint, its children's joints, and per geom the
    capsule end caps pushed out by the radius along 3 axes / the 8 box corners / the sphere's 6 poles."""
    P = [d.xpos[body]] + [d.xpos[c] for c in range(m.nbody) if m.body_parentid[c] == body and c != body]
    for g in np.where(m.geom_bodyid == body)[0]:
        c, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ax = R[:, 2]
            for e in (-1, 1):
                end = c + e * s[1] * ax
                P += [end + e * s[0] * ax, end + s[0] * R[:, 0], end - s[0] * R[:, 0], end + s[0] * R[:, 1], end - s[0] * R[:, 1]]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            P += [c + R @ (np.array([i, j, k]) * s) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            P += [c + e * s[0] * R[:, a] for a in range(3) for e in (-1, 1)]
    return np.array(P)


def body_maps(m_adult, m):
    """Per body (1..24): affine (A, t) with A p_adult + t ~= p_this over the body's landmarks."""
    da, db = rest_pose(m_adult), rest_pose(m)
    maps, resid = {}, {}
    for b in range(1, m.nbody):
        Pa, Pb = landmarks(m_adult, da, b), landmarks(m, db, b)
        X = np.c_[Pa, np.ones(len(Pa))]
        sol, *_ = np.linalg.lstsq(X, Pb, rcond=None)
        maps[b] = (sol[:3].T, sol[3])
        resid[m.body(b).name] = float(np.abs(X @ sol - Pb).max())
    return maps, resid


def skin_vertices(m, m_adult=None, betas=None, kid=0.0, kid_head=None, smooth=0):
    """Bind-pose vertices (in m's rest pose), faces, (n_vert, 24) weights in m's body order, landmark residuals.
    betas (up to 10 SMPL shape coefficients) restyle the surface before the per-body maps; every part is then shifted
    back so the joints stay on the adult's (betas = 0) joints, so the maps and the body's proportions still apply.
    beta1 > 0 slims the torso (waist / chest depth -25% at +2) without changing height."""
    S = smpl()
    m_adult = m_adult or mujoco.MjModel.from_xml_path(str(ADULT_XML))
    da = rest_pose(m_adult)
    J = S["J_regressor"] @ S["v_template"]
    v = kid_template(kid, kid_head) if kid > 0 or kid_head else S["v_template"]
    if smooth:
        v = _taubin_torso(v, smooth)
    if betas is not None and np.any(betas):
        b = np.zeros(10); b[:len(betas)] = betas
        v = v + S["shapedirs"] @ b
        v = v + S["weights"] @ (J - S["J_regressor"] @ v)
    V = (v - J[0]) @ Y_UP_TO_Z_UP.T + da.xpos[1]                       # SMPL rest mesh on the adult's rest pose
    order = [SMPL_ID[m.body(b).name] for b in range(1, m.nbody)]
    W = S["weights"][:, order]
    maps, resid = body_maps(m_adult, m)
    Vb = np.zeros_like(V)
    for k, b in enumerate(range(1, m.nbody)):
        A, t = maps[b]
        Vb += W[:, k:k + 1] * (V @ A.T + t)
    return Vb, S["f"], W, resid


def style(body):
    """Visual style per body: (kid blend, SMPL betas, torso smoothing passes). Child-like features only where the body
    IS a child (picked by eye on child: kid 0.75 + beta1 2.0 + 30 Taubin passes on the torso; above ~0.7 the infant
    template's folds need the smoothing, and kid 1.0 still creases the shoulders). Small adult bodies (petite,
    leg-0.5 grid ...) keep the adult look; the adult -> child morph path m2c_tXXX ramps in with t."""
    if body == "child":
        return 0.75, [0, 2.0], 30
    if body == "teen":
        return 0.3, [0, 1.0], 0
    if body.startswith("m2c_t"):
        t = int(body.split("_t")[1]) / 1000
        return 0.75 * t, [0, 2.0 * t], int(round(30 * t))
    return 0.0, None, 0


def with_skin(xml, hide_geoms=True, rgba=SKIN_RGBA, min_weight=1e-3, betas=None, kid=None, smooth=None):
    """MjModel of `xml` plus the SMPL skin; physics fields are identical to the plain model. kid in [0, 1] blends in
    the kid template's features (kid_template); kid=None takes style(<body dir name>) for kid, betas and smoothing."""
    if kid is None:
        kid, b, sm = style(Path(xml).parent.name)
        betas = b if betas is None else betas
        smooth = sm if smooth is None else smooth
    spec = mujoco.MjSpec.from_file(str(xml))
    m = spec.compile()
    V, F, W, _ = skin_vertices(m, betas=betas, kid=kid, smooth=smooth or 0)
    d = rest_pose(m)
    sk = spec.add_skin()
    sk.name, sk.rgba, sk.inflate = "smpl", list(rgba), 0.0
    sk.vert = V.reshape(-1).tolist()
    sk.face = F.reshape(-1).tolist()
    W = np.where(W >= min_weight, W, 0.0)
    W /= W.sum(1, keepdims=True)
    names, bpos, bquat, vid, vw = [], [], [], [], []
    for k, b in enumerate(range(1, m.nbody)):
        ids = np.nonzero(W[:, k])[0]
        if len(ids) == 0:
            continue
        names.append(m.body(b).name); bpos += d.xpos[b].tolist(); bquat += d.xquat[b].tolist()
        vid.append(ids.tolist()); vw.append(W[ids, k].tolist())
    sk.bodyname, sk.bindpos, sk.bindquat, sk.vertid, sk.vertweight = names, bpos, bquat, vid, vw
    ms = spec.compile()
    if hide_geoms:
        # move the body's geoms to group 5, which the default render options do not draw (an alpha-0 geom would still
        # write depth and punch holes in the skin); geom_group has no effect on physics
        for g in range(ms.ngeom):
            if ms.geom_bodyid[g] != 0:
                ms.geom_group[g] = 5
    return ms


def check(body):
    xml = REPO / f"assets/robots_torque/{body}/robot_torque_full.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    ms = with_skin(xml)
    _, _, _, resid = skin_vertices(m)
    print(f"{body}: nskin {ms.nskin}, skin verts {ms.nskinvert}; landmark fit residual max "
          f"{max(resid.values()) * 1000:.2f} mm (worst {max(resid, key=resid.get)})")
    d, ds = mujoco.MjData(m), mujoco.MjData(ms)
    rng = np.random.default_rng(0)
    for x in (d, ds):
        x.qpos[:] = m.qpos0
    for _ in range(300):
        u = rng.uniform(-0.3, 0.3, m.nu)
        d.ctrl[:] = u; ds.ctrl[:] = u
        mujoco.mj_step(m, d); mujoco.mj_step(ms, ds)
    print(f"  300 physics steps with random controls: max |qpos diff| {np.abs(d.qpos - ds.qpos).max():.3g}, "
          f"max |qvel diff| {np.abs(d.qvel - ds.qvel).max():.3g}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        for b in sys.argv[2:]:
            check(b)
