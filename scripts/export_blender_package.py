#!/usr/bin/env python3
"""export_blender_package.py -- the paper figures' motion panels as a folder that renders on any machine with Blender, needing nothing from this repo: outputs/blender_package/ and outputs/blender_package.zip.
scene.json holds, per panel and pose, every visible geom's world pose, shape and colour; meshes.npz the robot's meshes; render_blender.py (scripts/render_blender.py) builds and renders them. Rollouts are the same recorded ones as scripts/plot_teaser.py and scripts/plot_method_figure.py use; run this again when they change.
The results go in docs/figures/blender_renders/<panel>.png; those two scripts then draw them instead of the MuJoCo cut-outs.
usage: uv run python scripts/export_blender_package.py"""
import json, shutil, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts")); OUT = REPO / "outputs/blender_package"; S = REPO / "outputs/user_survey"; DPI = 600
GEOM = {mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid", mujoco.mjtGeom.mjGEOM_MESH: "mesh"}
WALK2, WALK4 = "move-ego-0-2_2", "move-ego-0-2_4"; ADULT, CHILD, LIMIT, HEAVY = (S / f"models/{n}.xml" for n in ("adult", "child", "limit_leg", "heavy_arm")); G1 = S / "models/g1/scene.xml"; H1 = REPO / "assets/robots/h1/scene.xml"                # the survey's videos have H1, its folder no rollouts: the search behind them is outputs/user_survey_search/h1_ridge, as g1_ridge is for G1
TEASER_F = (100, 112, 124); TEASER_T = (0.58, 0.3, 0.0); BAND_U = (550, 107, 12, 50.5); BAND_LEFT = 0.12; BAND_DX = 1.65; BAND_POSE_DX = 0.57; BAND_ALPHA = (0.2, 0.42, 0.7, 1.0); BAND_START, BAND_DT = 40, 12; BAND_H1_BACK = 33; RESULT = (60, 102, 144, 186, 228, 270); STAND = 165
NEW_BODY_T = 150                                                          # a still frame of move-ego-0-0_2 (method figure, New body box)
FIRST = (60, 165, 270)                                                    # the first layout of the method figure (scripts/plot_method_figure.py first), on the same run as the newer layout
RUN = "move-ego-0-4_8"; M_ADULT, M_CHILD, M_G1 = S / "child/trajectories" / RUN / "reference_adult.npz", S / "child/trajectories" / RUN / "best.npz", S / "g1/trajectories" / RUN / "best.npz"
meshes = {}


def load(xml, payload=False):
    spec = mujoco.MjSpec.from_file(str(xml))
    if payload:                                 # drawn only, as in plot_teaser.py
        hand = spec.compile(); ids = [i for i in range(hand.ngeom) if hand.body(hand.geom_bodyid[i]).name == "R_Hand"]; ctr = hand.geom_pos[ids].mean(0)
        g = spec.body("R_Hand").add_geom(); g.name = "payload"; g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.pos = ctr.tolist(); g.size = [max(0.14, 1.1 * float(hand.geom_rbound[ids].max())), 0, 0]; g.contype = 0; g.conaffinity = 0; g.mass = 0
    return spec.compile()


def color_of(m, g, red_leg):
    if m.geom(g).name == "payload":
        return "amber"
    if red_leg and m.body(m.geom_bodyid[g]).name in ("R_Hip", "R_Knee", "R_Ankle", "R_Toe"):
        return "red"
    return "body"


def pose(m, q, t, x, tint, heading, red_leg, tag):
    """the visible geoms at frame t, the walk turned onto +x and the root put at (x, 0)."""
    d = mujoco.MjData(m); d.qpos[:] = q[min(t, len(q) - 1)]; mujoco.mj_forward(m, d); c, s = np.cos(-heading), np.sin(-heading); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]); root = d.subtree_com[1][:2].copy(); gs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or m.geom_group[g] >= 3 or m.geom_rgba[g, 3] == 0 or m.geom_type[g] not in GEOM:
            continue
        p = d.geom_xpos[g].copy(); p[:2] -= root; p = Rz @ p; p[0] += x; e = dict(kind=GEOM[m.geom_type[g]], size=[round(float(v), 5) for v in m.geom_size[g]], p=[round(float(v), 5) for v in p], R=[round(float(v), 6) for v in (Rz @ d.geom_xmat[g].reshape(3, 3)).ravel()], color=color_of(m, g, red_leg))
        if e["kind"] == "mesh":
            i = m.geom_dataid[g]; name = f"{tag}_{m.mesh(i).name}"; e["mesh"] = name
            if name + "__v" not in meshes:
                a, n, fa, fn = m.mesh_vertadr[i], m.mesh_vertnum[i], m.mesh_faceadr[i], m.mesh_facenum[i]; meshes[name + "__v"] = m.mesh_vert[a:a + n].astype(np.float32); meshes[name + "__f"] = m.mesh_face[fa:fa + fn].astype(np.int32)
        gs.append(e)
    return dict(x=x, tint=tint, geoms=gs)


def contact_frames(m, q, frames, tol=0.01):
    """for each frame, the nearest one with a foot on the ground: a run spends half its time in the air, and a fixed spacing can land on that every time. Also returns the lowest point per frame."""
    sys.path.insert(0, str(REPO))
    from model.robot_map import Lowest
    low = Lowest(m); d = mujoco.MjData(m); lo = np.empty(len(q))
    for t in range(len(q)):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(m, d); lo[t] = low(d)
    on = np.flatnonzero(lo <= tol); return [int(on[np.abs(on - f).argmin()]) for f in frames], lo


def heading(q, frames):
    a, b = max(frames[0] - 10, 0), min(frames[-1] + 10, len(q) - 1); v = q[b, :2] - q[a, :2]; return float(np.arctan2(v[1], v[0]))


def panel(figure, w_u, h_u, feet_u, per_m, poses_, white=False):
    return dict(figure=figure, width_m=w_u / per_m, height_m=h_u / per_m, feet_m=feet_u / per_m, px=[round(w_u / 100 * DPI), round(h_u / 100 * DPI)], white=white, poses=poses_)


def strip(xml, qpath, frames, tints, dx_m, tag, payload=False, red_leg=False, ground=False):
    """ground: use the nearest frames with a foot down, set exactly on the floor (at most tol away)."""
    m = load(xml, payload); q = np.load(qpath)["qpos"].astype(np.float64); assert q.shape[1] == m.nq, (xml, q.shape, m.nq); h = heading(q, frames); mid = (len(frames) - 1) / 2; lo = None
    if ground:
        frames, lo = contact_frames(m, q, frames); print(" ", tag, "frames", frames, "feet at cm", (lo[frames] * 100).round(1).tolist())
    out = [pose(m, q, t, (i - mid) * dx_m, k, h, red_leg, tag) for i, (t, k) in enumerate(zip(frames, tints))]
    if ground:
        for po, t in zip(out, frames):
            for g in po["geoms"]:
                g["p"][2] = round(g["p"][2] - float(lo[t]), 5)
    return out


def fall_strip(xml, qpath, tag, n=6, span=40, width_m=80 / 42):
    """the fall of a z0 rollout, up to its lowest point: standing pelvis height from the clip's start, the first drop below 60% of it.
    n = 6 poses, as the other method panels (RESULT); width_m: first to last pose, kept at the width of the earlier 5-pose strip so the panel's layout stays."""
    m = load(xml); q = np.load(qpath)["qpos"].astype(np.float64); z = q[:, 2]; stand = np.median(z[10:50]); first = int(np.flatnonzero(z < 0.6 * stand)[0]); low = first + int(np.argmin(z[first:first + span])); frames = [int(f) for f in np.linspace(low - span, low, n).round()]; print(" ", tag, "fall: stands", round(float(stand), 2), "first down", first, "lowest", low, round(float(z[low]), 2), "frames", frames); h = heading(q, frames); mid = (n - 1) / 2; tints = np.linspace(0.6, 0, n)
    return [pose(m, q, t, (i - mid) * width_m / (n - 1), float(k), h, False, tag) for i, (t, k) in enumerate(zip(frames, tints))]


def main():
    OUT.mkdir(parents=True, exist_ok=True); P = {}; U = 55.0; dx = 33 / U; tr = lambda b, c, f: S / b / "trajectories" / c / f
    P["teaser_adult"] = panel("teaser", 142, 130, 10, U, strip(ADULT, tr("child", WALK2, "reference_adult.npz"), TEASER_F, TEASER_T, dx, "adult"))
    for key, body, xml, clip, kw in (("payload", "heavy_arm", HEAVY, WALK4, dict(payload=True)), ("joint", "limit_leg", LIMIT, WALK2, dict(red_leg=True)), ("child", "child", CHILD, WALK2, {}), ("g1", "g1", G1, WALK2, {})):
        P[f"teaser_{key}"] = panel("teaser", 128, 125, 9, U, strip(xml, tr(body, clip, "best.npz"), TEASER_F, TEASER_T, dx, body, **kw))
    # the teaser as one band, as in ReActor's Fig. 1: one walk carrying on across the conditions, each fading in from see-through ghosts that start behind the pose before, on one white floor. shift: metres sideways in the image. The clip is 300 frames: H1 goes back BAND_H1_BACK frames, three of its gait periods (11 frames), to stay in phase
    band = [("Origin", ADULT, tr("child", WALK2, "reference_adult.npz"), "adult", {}), ("Payload", HEAVY, tr("heavy_arm", WALK4, "best.npz"), "heavy_arm", dict(payload=True)), ("Restricted joint", LIMIT, tr("limit_leg", WALK2, "best.npz"), "limit_leg", dict(red_leg=True)), ("Reshaped body", CHILD, tr("child", WALK2, "best.npz"), "child", {}), ("G1", G1, tr("g1", WALK2, "best.npz"), "g1", {}), ("H1", H1, REPO / "outputs/user_survey_search/h1_ridge" / f"{WALK2}_s0/best.npz", "h1", {})]; poses_ = []; groups = []; n = len(BAND_ALPHA)
    for i, (label, xml, qpath, tag, kw) in enumerate(band):
        shift = (i - (len(band) - 1) / 2) * BAND_DX - BAND_LEFT; frames = [BAND_START + (i * n + k) * BAND_DT - (BAND_H1_BACK if label == "H1" else 0) for k in range(n)]; groups.append(dict(label=label, x_m=shift + (n - 1) / 2 * BAND_POSE_DX * np.sin(np.radians(125.0))))        # over the solid pose; 125: AZIMUTH in render_blender.py
        for po, a in zip(strip(xml, qpath, frames, (0.0,) * n, BAND_POSE_DX, tag, ground=True, **kw), BAND_ALPHA):
            po["shift"] = shift; po["alpha"] = a; poses_.append(po)
    P["teaser_band"] = dict(panel("teaser", *BAND_U, poses_, white=True), cam_distance=60.0, groups=groups)
    P["teaser_band_grey"] = dict(P["teaser_band"], white=False)
    U = 42.0; tints = tuple(0.62 * (1 - i / (len(RESULT) - 1)) for i in range(len(RESULT)))
    P["method_fall_child"] = panel("method", 110, 63, 7, U, fall_strip(CHILD, S / "child/trajectories" / RUN / "z0.npz", "child"), white=True)
    P["method_fall_g1"] = panel("method", 110, 63, 7, U, fall_strip(G1, REPO / "outputs/g1/search/move-ego-90-4_9_s0/origin_z.npz", "g1"), white=True)
    P["method_adult"] = panel("method", 130, 100, 8, U, strip(ADULT, M_ADULT, RESULT, tints, 18 / U, "adult", ground=True), white=True)
    P["method_child"] = panel("method", 128, 98, 8, U, strip(CHILD, M_CHILD, RESULT, tints, 18 / U, "child", ground=True), white=True)
    P["method_g1"] = panel("method", 128, 98, 8, U, strip(G1, M_G1, RESULT, tints, 18 / U, "g1", ground=True), white=True)
    for name, xml, qpath, tint in (("adult", ADULT, M_ADULT, 0.5), ("child", CHILD, M_CHILD, 0.0), ("g1", G1, M_G1, 0.0)):
        for t, po in zip(FIRST, strip(xml, qpath, FIRST, (tint,) * len(FIRST), 0.0, name, ground=True)):
            P[f"method_first_{name}_{t}"] = dict(panel("method", 100, 100, 12, U, [po], white=True), cutout=True)
    # the method figure's "New body" box: each body standing still (user, 11/11), frame NEW_BODY_T of the stand-still task motion move-ego-0-0_2
    # (the child's retargeted reference; the G1's target from the SMPL round trip), arms down, as the first layout's other cut-outs
    for name, xml, qpath in (("child", CHILD, REPO / "data/child/retargeting_motion/move-ego-0-0/move-ego-0-0_2.npz"), ("g1", G1, REPO / "outputs/g1/roundtrip/move-ego-0-0_2/g1.npz")):
        po = strip(xml, qpath, (NEW_BODY_T,), (0.0,), 0.0, name, ground=True)[0]
        P[f"method_first_{name}_stand"] = dict(panel("method", 100, 100, 12, U, [po], white=True), cutout=True)
    (OUT / "scene.json").write_text(json.dumps(dict(panels=P))); np.savez_compressed(OUT / "meshes.npz", **meshes); shutil.copy(REPO / "scripts/render_blender.py", OUT / "render_blender.py"); shutil.copy(REPO / "scripts/blender_package_README.md", OUT / "README.md")
    shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)
    for k, v in P.items():
        print(k, "poses", len(v["poses"]), "geoms per pose", len(v["poses"][0]["geoms"]), "px", v["px"], "m", round(v["width_m"], 2), flush=True)
    print("meshes", len(meshes) // 2, "scene MB", round((OUT / "scene.json").stat().st_size / 1e6, 2), "meshes MB", round((OUT / "meshes.npz").stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
