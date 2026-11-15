#!/usr/bin/env python3
"""render_crowd.py -- every body of the multi-body experiment in ONE scene, replaying the same clip side by side.

Nothing is re-simulated: each body replays a rollout its search recorded (search/<body>/<clip>_s<seed>/origin_z.npz
for z0, best.npz for the searched z of the seed with the lower best cost). All 32 humanoids are attached into one
MjModel (prefix b<i>_), contacts off, kinematics only. Each body starts from its own reference start, shifted into its own lane (perpendicular to the
adult reference's overall walking direction, --spacing apart), lanes ordered by standing height. The camera follows
the lane-aligned REFERENCE trajectories (midpoint of the front-most and back-most body, backing off as the group
spreads in depth; smoothed), so the z0 and searched-z videos share one camera path.
Bodies are drawn as their capsules, coloured by group; --mesh dresses them in SMPL skins instead.
Labels above each head: name and that rollout's cost (/ z0 cost for the searched z).
  python scripts/render_crowd.py --clip move-ego-0-2-raisearms-m-h/move-ego-0-2-raisearms-m-h_9 --which best
Writes outputs/multibody_corr/crowd/<clip>_<which>.mp4 (+ .png contact sheet); with --mesh, crowd/mesh/.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO))
import smpl_skin as sk                                                   # noqa: E402
from multibody_corr import ADULT, BODIES, GROUPS, OUT, SEEDS, beta, cell_dir   # noqa: E402

GROUP_RGBA = {"grid": (0.36, 0.56, 0.84, 1), "axis": (0.30, 0.70, 0.55, 1), "named": (0.90, 0.58, 0.40, 1),
              "adult": (0.60, 0.60, 0.62, 1)}
AXIS_NAME = {"ax2m_t1000": "torso .75", "ax3m_t1000": "head .95", "ax4m_t1000": "leg girth .7",
             "ax5m_t1000": "arm girth .7", "ax6m_t1000": "torso girth .82", "ax7m_t1000": "head girth .9"}
BASE_XML = """<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.82 0.9 0.95" rgb2="0.98 0.98 0.98" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.85" rgb2="0.74 0.74 0.73" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 10" dir="0.2 0.3 -1" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
  </worldbody>
</mujoco>"""


def label_of(b):
    if b == ADULT:
        return "adult"
    if b.startswith("gl"):
        return f"L{int(b[2:6]) / 1000:g} A{int(b[7:11]) / 1000:g}"
    if b.startswith("ax0"):
        return f"L{beta(b)[0]:g}"
    if b.startswith("ax1"):
        return f"A{beta(b)[1]:g}"
    return AXIS_NAME.get(b, b.replace("_", " "))


def group_of(b):
    return "adult" if b == ADULT else ("grid" if GROUPS[b] == "grid" else GROUPS[b])


def build(order, W, H, mesh):
    spec = mujoco.MjSpec.from_string(BASE_XML.format(W=W, H=H))
    skins = []
    for i, b in enumerate(order):
        xml = REPO / f"assets/robots_torque/{b}/robot_torque_full.xml"
        cs = mujoco.MjSpec.from_file(str(xml)); m1 = cs.compile()
        spec.worldbody.add_frame().attach_body(cs.body("Pelvis"), f"b{i}_", "")
        if mesh:
            kid, betas, smooth = sk.style(b)
            V, F, Wt, _ = sk.skin_vertices(m1, betas=betas, kid=kid, smooth=smooth)
            skins.append((i, b, m1, sk.rest_pose(m1), V, F, Wt))
    for i, b, m1, d1, V, F, Wt in skins:
        s = spec.add_skin()
        s.name, s.rgba = f"skin{i}", list(GROUP_RGBA[group_of(b)])
        s.vert, s.face = V.reshape(-1).tolist(), F.reshape(-1).tolist()
        Wt = np.where(Wt >= 1e-3, Wt, 0.0); Wt /= Wt.sum(1, keepdims=True)
        names, bp, bq, vid, vw = [], [], [], [], []
        for k, bb in enumerate(range(1, m1.nbody)):
            ids = np.nonzero(Wt[:, k])[0]
            names.append(f"b{i}_{m1.body(bb).name}"); bp += d1.xpos[bb].tolist(); bq += d1.xquat[bb].tolist()
            vid.append(ids.tolist()); vw.append(Wt[ids, k].tolist())
        s.bodyname, s.bindpos, s.bindquat, s.vertid, s.vertweight = names, bp, bq, vid, vw
    m = spec.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        if mesh:
            m.geom_group[g] = 5
        else:                                       # capsules in the body's group colour (material off so rgba shows)
            i = int(m.body(int(m.geom_bodyid[g])).name.split("_")[0][1:])
            m.geom_matid[g] = -1; m.geom_rgba[g] = GROUP_RGBA[group_of(order[i])]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--which", choices=["z0", "best"], required=True)
    ap.add_argument("--spacing", type=float, default=0.9)
    ap.add_argument("--W", type=int, default=2560)
    ap.add_argument("--H", type=int, default=1080)
    ap.add_argument("--elevation", type=float, default=-30)
    ap.add_argument("--yaw", type=float, default=0, help="camera azimuth off the walking direction (deg); 0 = head-on")
    ap.add_argument("--preview", action="store_true", help="render only the 3 contact-sheet frames")
    ap.add_argument("--mesh", action="store_true", help="SMPL skins instead of the capsule bodies")
    ap.add_argument("--mark-falls", action="store_true",
                    help="red 'fallen' label while a body is down: pelvis below half the reference's, or tipped "
                         "(up-axis z more than 0.8 below the reference's) -- relative, so a headstand is not a fall")
    ap.add_argument("--max-limb-ratio", type=float, default=None,
                    help="keep only bodies whose arm / leg length ratio is within [1/r, r] (e.g. 1.15 drops the 11 leg x arm "
                         "grid bodies with mismatched limbs; every other body is within 12%%)")
    a = ap.parse_args()
    st = a.clip.split("/")[1]
    bodies = [b for b in BODIES if a.max_limb_ratio is None or abs(np.log(beta(b)[1] / beta(b)[0])) <= np.log(a.max_limb_ratio)]
    # clips outside the experiment's 16 were searched only on the bodies that have them (and maybe one seed)
    bodies = [b for b in bodies if any((cell_dir(b, a.clip, s) / "summary.json").exists() for s in SEEDS)]
    print(f"{len(bodies)} bodies" + (f" (dropped {', '.join(label_of(b) for b in BODIES if b not in bodies)})" if len(bodies) < len(BODIES) else ""))
    refs = {b: np.load(REPO / f"data/{b}/retargeting_motion/{a.clip}.npz")["qpos"] for b in bodies}
    order = sorted(bodies, key=lambda b: refs[b][0, 2])                  # lanes by pelvis height = stature
    traj, cost = {}, {}
    for b in bodies:
        # every finished search of this cell: the seeds, plus continuations (<cell>_cont, started from a seed's best z)
        dirs = [cell_dir(b, a.clip, s) for s in SEEDS] + [Path(str(cell_dir(b, a.clip, s)) + "_cont") for s in SEEDS]
        dirs = [d for d in dirs if (d / "summary.json").exists()]
        sm = [json.load(open(d / "summary.json")) for d in dirs]
        if a.which == "z0":
            traj[b] = np.load(dirs[0] / "origin_z.npz")["qpos"]; cost[b] = (sm[0]["origin_z"]["cost"], None)
        else:
            k = int(np.argmin([x["best"]["cost"] for x in sm]))
            traj[b] = np.load(dirs[k] / "best.npz")["qpos"]
            cost[b] = (sm[k]["best"]["cost"], sm[k]["best"]["cost"] / sm[k]["origin_z"]["cost"])
    ra = refs[ADULT]
    d = ra[-1, :2] - ra[0, :2]
    if np.linalg.norm(d) < 1.0:
        # an in-place clip (headstand, raise arms, rotate ...): use the adult reference's initial facing instead, so
        # the bodies face the camera. Facing = the pelvis x axis after removing the SMPL base rotation (humenv heading)
        # A lying body (crawl) has that axis pointing at the floor: there, forward = the pelvis -> head direction.
        qrel = np.zeros(4); mujoco.mju_mulQuat(qrel, ra[0, 3:7], np.array([0.5, -0.5, -0.5, -0.5]))
        f = np.zeros(3); mujoco.mju_rotVecQuat(f, np.array([1.0, 0, 0]), qrel); d = f[:2]
        ma = mujoco.MjModel.from_xml_path(str(REPO / f"assets/robots_torque/{ADULT}/robot_torque_full.xml")); da = mujoco.MjData(ma)
        PH = []
        for t in range(0, len(ra), 10):                 # the crawl stands at frame 0, then lies down for the rest
            da.qpos[:] = ra[t]; mujoco.mj_kinematics(ma, da)
            PH.append((da.xpos[ma.body("Head").id] - da.xpos[ma.body("Pelvis").id])[:2])
        PH = np.array(PH)
        if np.median(np.linalg.norm(PH, axis=1)) > 0.3:
            d = PH[np.linalg.norm(PH, axis=1) > 0.3].mean(0)
    d = d / np.linalg.norm(d); p = np.array([-d[1], d[0]])
    n = len(order); lane = {b: (i - (n - 1) / 2) * a.spacing * p for i, b in enumerate(order)}
    T = max(len(traj[b]) for b in bodies)
    # camera path from the lane-aligned references (identical for z0 and best videos), 1 s moving average
    # camera: aim at the middle of the front-most and back-most body along the walk (reference positions), and back off
    # as the group spreads in depth (tall bodies cover ~20 m in 10 s, small ones ~11 m), 1 s moving average
    Pref = np.array([[refs[b][min(t, len(refs[b]) - 1), :2] - refs[b][0, :2] + lane[b] for b in bodies] for t in range(T)])
    along = Pref @ d
    mid = (along.max(1) + along.min(1)) / 2
    C = (Pref.mean(1) - np.outer(Pref.mean(1) @ d, d)) + np.outer(mid, d)
    depth = along.max(1) - along.min(1)
    k = 15
    C = np.array([C[max(0, t - k):t + k + 1].mean(0) for t in range(T)])
    depth = np.array([depth[max(0, t - k):t + k + 1].mean() for t in range(T)])
    m = build(order, a.W, a.H, a.mesh); dd = mujoco.MjData(m)
    r = mujoco.Renderer(m, a.H, a.W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    span = n * a.spacing
    hfov = 2 * np.arctan(np.tan(np.radians(m.vis.global_.fovy) / 2) * a.W / a.H)
    base_dist = (span / 2 + 2.5) / np.tan(hfov / 2) + 2.0; cam.elevation = a.elevation      # whole row in frame
    cam.azimuth = np.degrees(np.arctan2(-d[1], -d[0])) + a.yaw        # look back along the walk: bodies face the camera
    qadr = [int(m.jnt_qposadr[m.body(f"b{i}_Pelvis").jntadr[0]]) for i in range(n)]
    head = [m.body(f"b{i}_Head").id for i in range(n)]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13); font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = font_b = ImageFont.load_default()
    title = f"{st}  ·  " + ("z0 (no search)" if a.which == "z0" else "searched z")
    from model.bodies import up_z

    def fallen(b, t):
        q, r = traj[b][min(t, len(traj[b]) - 1)], refs[b][min(t, len(refs[b]) - 1)]
        return bool(q[2] < 0.5 * r[2] or up_z(q[3:7]) < up_z(r[3:7]) - 0.8)

    frames = []
    ts = [int(T * f) for f in (0.1, 0.5, 0.9)] if a.preview else range(T)
    for t in ts:
        for i, b in enumerate(order):
            q = traj[b][min(t, len(traj[b]) - 1)].copy()
            q[:2] += -refs[b][0, :2] + lane[b]
            dd.qpos[qadr[i]:qadr[i] + 76] = q
        mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
        cam.lookat[:] = [C[t, 0], C[t, 1], 0.9]
        cam.distance = base_dist + 0.45 * depth[t]
        r.update_scene(dd, camera=cam)
        if a.mesh:                                  # skin self-shadow bands; capsules keep their shadows
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        img = Image.fromarray(r.render()); dr = ImageDraw.Draw(img)
        gl = r.scene.camera[0]
        fwd, up = np.array(gl.forward), np.array(gl.up); right = np.cross(fwd, up); pos = np.array(gl.pos)
        hw = (gl.frustum_top - gl.frustum_bottom) / 2 * a.W / a.H        # frustum_width is 0: width follows the aspect
        L, R = gl.frustum_center - hw, gl.frustum_center + hw
        for i, b in enumerate(order):
            v = dd.xpos[head[i]] + np.array([0, 0, 0.30 + 0.42 * (i % 2)]) - pos       # alternate label heights
            z = v @ fwd
            if z <= 0.1:
                continue
            x = (gl.frustum_near * (v @ right) / z - L) / (R - L) * a.W
            y = (gl.frustum_top - gl.frustum_near * (v @ up) / z) / (gl.frustum_top - gl.frustum_bottom) * a.H
            c, rr = cost[b]
            txt = f"{label_of(b)}\n" + (f"{c:.2f}" if rr is None else f"{rr:.2f}x")
            down = a.mark_falls and fallen(b, t)
            if down:
                txt += "\nfallen"
            dr.multiline_text((x, y), txt, fill=(200, 30, 30) if down else (30, 30, 30), font=font, anchor="md",
                              align="center", stroke_width=3, stroke_fill=(250, 250, 250))
        if a.mark_falls:
            nd = sum(fallen(b, t) for b in order)
            dr.text((24, 78), f"fallen now: {nd} / {len(order)}", fill=(200, 30, 30) if nd else (60, 60, 60), font=font_b,
                    stroke_width=3, stroke_fill=(250, 250, 250))
        dr.text((24, 18), title, fill=(20, 20, 20), font=font_b, stroke_width=3, stroke_fill=(250, 250, 250))
        dr.text((24, 52), f"t = {t / 30:4.1f} s   ·   colour: grid (leg x arm) blue, single-dim shrink green, named bodies orange, "
                          f"adult grey   ·   label: " + ("z0 cost" if a.which == "z0" else "best cost / z0 cost"),
                fill=(60, 60, 60), font=font, stroke_width=2, stroke_fill=(250, 250, 250))
        frames.append(np.asarray(img))
    r.close()
    out = OUT / "crowd" / ("mesh" if a.mesh else "") / f"{st}_{a.which}.mp4"; out.parent.mkdir(parents=True, exist_ok=True)
    sheet = frames if a.preview else [frames[int(T * f)] for f in (0.1, 0.5, 0.9)]
    Image.fromarray(np.concatenate(sheet, 0)).save(str(out) + ".png")
    if a.preview:
        print(f"-> {out}.png (preview)"); return
    imageio.mimsave(out, frames, fps=30, quality=8)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
