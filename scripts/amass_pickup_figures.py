#!/usr/bin/env python3
"""amass_pickup_figures.py -- two Blender figures of the AMASS ACCAD clip Male2Walking__B19_-__Walk_to_pickup_box on the child body
(user, 11/12): the person walks to a box and bends down to pick it up.

Bodies (seed 0, 1,024 rollouts, outputs/amass_baselines/child/<clip>_{lso,lsor,ta}_s0/rollouts.npz): the adult reference
(amass_data/amass_origin_motion, adult body), native tracking (q_ER8, the same in all three files), LSO (BFMTrack, q_best_it64 of _lso_),
LSOR (ours, q_best_it64 of _lsor_) and Trajectory Adaptation (BFM-Zero, q_best_it64 of _ta_). Alignment: rollout frame t against reference
frame t + 1 (the reference's first frame dropped), as the strobe figures; the aligned copies are written to <OUT>/trajectories.

Figure 1 (moment): one pose per panel at the pick-up frame (the adult reference's lowest hand, FRAME), all five at that same aligned frame,
one world camera side-on to the adult reference's lean at that frame (its head minus its centre of mass), so the bend is seen in profile
and the walk runs to the right. Each pose is root-centred (centre of mass x, y removed, as every strobe panel) and then centred in its panel
by its own horizontal extent. The adult and the child at their real sizes, one scale for all panels.
Figure 2 (topdown): as scripts/topdown_blender.py --layout row --uniform-color: every panel its own poses at their real places on the floor
(TOP_FRAMES, light = early) and its own pelvis path in the body colour, the adult reference's path as a thin black line in the four method
panels; turned so the reference's start-to-end direction runs left to right; one scale and centre for every panel. Units: adult-equivalent
metres -- the child's scene (positions and bodies) is scaled up uniformly by 1 / 0.611036, the adult reference is drawn as is.
Look and renderer: scripts/render_blender.py (Cycles, 256 samples, white floor, one body colour).
Out: outputs/strobe_task52_blender/amass_pickup_box/{trajectories/, frames.json, scene.json, meshes.npz, render_blender.py, renders/} and
outputs/strobe_task52_blender/combined/amass_child_pickup_box_{moment,topdown}.png (five panels in one row, 24 px apart, no labels:
adult reference | native | LSO | LSOR | TA).
usage: uv run python scripts/amass_pickup_figures.py [export] [render] [combine] [--samples 256] [--threads 16] [--scale 1]
       (threads also from STROBE_THREADS; render one Blender job at a time on the shared machine)

Other clips and bodies (11/12; with no options everything above is unchanged):
  --clip NAME          the ACCAD clip (default the pick-up clip)
  --body child|tall    the rollouts' body: outputs/amass_baselines/<body>; tall = len1.4_wid0.8 (root scale 1.364408905613045), drawn from
                       assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml (no survey copy exists)
  --moment pickup|kick the key moment: pickup = the adult reference's lowest hand, camera side-on to its lean; kick = the adult reference's
                       highest foot (the higher of ankle and toe), camera side-on to its kick (centre of mass to that toe, horizontal)
  --tag NAME           folder outputs/strobe_task52_blender/amass_<tag>/, combined/amass_<body>_<tag>_{moment|<moment>_moment,topdown}.png
                       (the moment figure is named "moment" for pickup, "<moment>_moment" otherwise); panels <moment>_{moment,topdown}_<body>
  --cam-azimuth DEG|fit  the moment figure's camera, world azimuth (default: side-on to the adult reference, as above); fit: one world view
                       that keeps every panel's kick (lean) as close to the image plane as it can (smallest largest angle out of it), the
                       reference's pointing right -- for bodies that face other ways at that frame
  --align A,B,..       moment figure: turn these method panels (e.g. lsor) about their root so the root yaw equals the adult reference's at
                       that frame (as strobe_final_blender.py align=True: heading += yaw(q[t]) - yaw(ref[t])); the others keep their world
                       heading; the camera is the shared one, chosen as without --align
  --ground             moment figure: lower the adult reference by its own lowest point at that frame (model.robot_map.Lowest on the render
                       model outputs/user_survey/models/adult.xml; the AMASS origin motion's floor is off by a few cm); the simulated bodies
                       stand on the physics floor and are not moved
  --top-frames auto|A,B,..  the top-down poses; auto (the default off the pick-up clip): the start, the end and the frames between them
                       whose poses overlap least on the floor, summed over the panels (see auto_frames)
  --top-style strobe|final  the top-down figure: strobe (default) = the poses at the top-down frames; final = only the last frame's pose
                       on the full path, and on the method panels the body's own reference (data/<body>/amass_retargeting_motion, aligned as the
                       rollouts, same adult-equivalent scaling) at that frame as a see-through grey ghost (GHOST), the own path lighter
                       (PATH_TINT), a dot at its start
  --top-lsor NPZ:KEY   the top-down figure's LSOR rollout (its path and pose) from another run, e.g.
                       outputs/amass_baselines/tall_extra/<clip>_lsor_s0_2048/rollouts.npz:q_best_it128; the moment figure keeps the default
  --top-n N            how many poses auto picks (default 3: a form that goes back and forth over ~1.5 m has room for few)
  Off the pick-up clip the moment panel is sized to the poses, the top-down panel to the paths, the drawn poses and their shadows (PAD around), so nothing is cropped.
  e.g. (the tall martial-arts figures, combined/amass_tall_martial_arts_{kick_moment,topdown}.png)
       uv run python scripts/amass_pickup_figures.py --clip Male2MartialArtsExtended__Extended_1 --body tall --moment kick --tag martial_arts --cam-azimuth fit --align lsor --ground
       (combined/amass_tall_martial_arts_kick_moment_unaligned.png: the same without --align lsor --ground)
       --top-style final --top-lsor outputs/amass_baselines/tall_extra/Male2MartialArtsExtended__Extended_1_lsor_s0_2048/rollouts.npz:q_best_it128
       added for combined/amass_tall_martial_arts_topdown.png (..._topdown_strobe.png: without them)

A kick filmstrip (11/12; needs --moment kick; replaces the moment and top-down figures in that run):
  --sequence N         N panels in one row, one body only, our method (SEQ_METHOD = lsor, q_best_it64), at N frames around the kick that
                       --moment kick picks, chosen from the adult reference's kicking-foot height (see seq_frames; N = 5: wind-up, leg
                       rising, kick peak, retraction, foot back down); one world camera for every panel (--cam-azimuth as for the moment
                       figure, fit over the adult reference and SEQ_METHOD at the kick peak), each panel centred on its own bodies, one
                       scale for all (the widest and the tallest panel, PAD around, grown to SEQ_PX = the tall martial-arts moment panels' size)
  --overlay-adult      in every panel also the adult reference (the AMASS origin motion, full adult size) at the same frame, moved
                       horizontally only so its root (pelvis, the free joint's x, y) is on the method's root; no turn, no scaling; --ground
                       lowers it as for the moment figure; drawn light and see-through (ADULT_LOOK: the body colour tinted towards white,
                       alpha) so the solid method body shows inside it
  Out: outputs/strobe_task52_blender/amass_<body>_<tag>_kick_sequence/{trajectories/, frames.json, source.json (frames, run file,
       alignment offsets, heading differences), scene.json, meshes.npz, render_blender.py, renders/}, combined/amass_<body>_<tag>_kick_sequence.png
       (24 px apart, no labels) and every panel as panels/amass_<body>_<tag>_kick_sequence_f<frame>.png
  e.g. uv run python scripts/amass_pickup_figures.py --clip Male2MartialArtsExtended__Extended_1 --body child --moment kick --tag martial_arts --cam-azimuth fit --ground --sequence 5 --overlay-adult
OOP refactor (ported from the main tree, 11/12): the pose export is crossenbodiment/blender.py's (BlenderScene, load_model: the functions
export_blender_package.py had), so scene.json carries the "version" field every refactored Blender export writes (render_blender.py
ignores it); runs and the survey models are read through crossenbodiment.paths.read_path ($CE_OUTPUTS_ROOT), the figure folders are
written under $CE_RESULTS_ROOT (default outputs/)."""
import itertools, json, math, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO))
import mujoco                                                             # noqa: E402
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.blender import BlenderScene, _geom_kinds, load_model  # noqa: E402  (export_blender_package's pose / load / GEOM, step 10)
SCENE = BlenderScene()                                                    # the panels and the meshes their poses use (export_blender_package.meshes)

ARG = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
PICKUP = "Male2Walking__B19_-__Walk_to_pickup_box"
CLIP = ARG("--clip", PICKUP); BODY = ARG("--body", "child"); MOMENT = ARG("--moment", "pickup"); TAG = ARG("--tag", "pickup_box")
LEGACY = (CLIP, BODY) == (PICKUP, "child")                               # the pick-up figures' fixed panel sizes and top-down frames
K = dict(child=0.611036, tall=1.364408905613045)[BODY]                     # body / adult size
ST = paths.write_path("outputs/strobe_task52_blender"); OUT = ST / f"amass_{TAG}"; RUNS_REL = f"outputs/amass_baselines/{BODY}"; RUNS = paths.read_path(RUNS_REL)
PFX = MOMENT; FIG = dict(moment="moment" if MOMENT == "pickup" else f"{MOMENT}_moment", topdown="topdown")   # panel prefix, combined figure names
SAMPLES = ARG("--samples", 256); THREADS = ARG("--threads", int(os.environ.get("STROBE_THREADS", 16))); SCALE = ARG("--scale", 1.0)
MODELS = dict(adult=paths.read_path("outputs/user_survey/models/adult.xml"), child=paths.read_path("outputs/user_survey/models/child.xml"),
              tall=REPO / "assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml")     # no survey copy of the tall body
BODIES = (("adult", "Adult reference"), ("native", "Native tracking"), ("lso", "LSO (BFMTrack)"), ("ta", "Trajectory Adaptation (BFM-Zero)"), ("lsor", "LSOR (ours)"))   # ours last (user, 11/12)
AZ = 125.0                                                                # the renderer's camera azimuth (render_blender.py AZIMUTH)
# figure 1: panel size in metres (one scale for all panels), image width
M_W, M_H, M_FEET, M_PX = 1.2, 1.4, 0.22, 1200
# figure 2: the poses drawn (aligned frames): the start, then about every 0.96 m along the reference's pelvis path up to the pick-up (frame 137, 3.85 m), and the end
TOP_FRAMES = (0, 50, 71, 93, 137, 199); T_MARGIN, T_BELOW, T_PX = 0.75, 1.05, 1200; R_OWN, R_REF = 0.022, 0.013   # margin (m; below the paths: room for the shadows), panel width (px), path line radii (m)
TOPF = ARG("--top-frames", ",".join(map(str, TOP_FRAMES)) if LEGACY else "auto"); TOP_N = ARG("--top-n", 3)
FINAL = ARG("--top-style", "strobe") == "final"; TOP_LSOR = ARG("--top-lsor", "")   # top-down figure: see the docstring
GHOST = ("z0", 0.6); PATH_TINT = 0.45; R_DOT = 0.045; BODY_DATA = dict(child="child", tall="len1.4_wid0.8")   # the ghost's colour (render_blender.py COLORS, mid grey) and alpha (0.3 vanishes on the white floor), the own path's tint (lighter, so the pose stands out), the start dot's radius (m), data/<folder> per body
ALIGN, GROUND = set(ARG("--align", "").split(",")) - {""}, "--ground" in sys.argv             # moment figure: see the docstring
PAD = 0.15                                                                # off the pick-up clip: room around the drawn poses and their shadows (m)
SUN = np.array([2.2, -2.2, 0.8 - 6.5]); SUN = SUN / -SUN[2]               # render_blender.py's key light (KEY_POS towards (x, 0, 0.8)): a point at height z shadows at xy + z * SUN[:2]
SEQ, OVERLAY, SEQ_METHOD = ARG("--sequence", 0), "--overlay-adult" in sys.argv, "lsor"     # the kick filmstrip: see the docstring
BODY_RGB = (0.47, 0.61, 0.70)                                             # render_blender.py COLORS["body"] (sRGB), for the record in source.json
ADULT_LOOK = (0.45, 0.35)                                                 # --overlay-adult: tint towards white (paler, less saturated), alpha
SEQ_PX = (M_PX, 1576)                                                     # --sequence panel size (px): the tall martial-arts moment panels' (combined/amass_tall_martial_arts_kick_moment.png)
NEAR, HALF = 0.05, 0.5                                                    # seq_frames: "on the floor" = within NEAR m of the standing foot height; HALF of the kick's rise
if SEQ:
    assert MOMENT == "kick", "--sequence needs --moment kick"; OUT = ST / f"amass_{BODY}_{TAG}_{MOMENT}_sequence"


def trajectories():
    """aligned copies: reference frame t + 1 against rollout frame t"""
    d = OUT / "trajectories"; d.mkdir(parents=True, exist_ok=True)
    R = {m: np.load(RUNS / f"{CLIP}_{m}_s0/rollouts.npz") for m in ("lso", "lsor", "ta")}
    assert np.array_equal(R["lso"]["q_ER8"], R["lsor"]["q_ER8"]) and np.array_equal(R["lso"]["q_ER8"], R["ta"]["q_ER8"])
    Q = dict(adult=np.load(REPO / f"amass_data/amass_origin_motion/ACCAD/{CLIP}.npz")["qpos"][1:], native=R["lso"]["q_ER8"],
             lso=R["lso"]["q_best_it64"], lsor=R["lsor"]["q_best_it64"], ta=R["ta"]["q_best_it64"])
    Q = {k: np.asarray(v, np.float64) for k, v in Q.items()}; n = {len(v) for v in Q.values()}; assert n == {len(Q["adult"])} and len(R["lso"]["ref"]) == len(Q["adult"]) + 1, n
    for k, v in Q.items():
        np.savez_compressed(d / f"{k}.npz", qpos=v.astype(np.float32), fps=np.float32(30))
    (d / "source.json").write_text(json.dumps(dict(clip=CLIP, **({} if LEGACY else dict(body=BODY, runs=RUNS_REL)), adult=f"amass_data/amass_origin_motion/ACCAD/{CLIP}.npz [1:]", native=f"{CLIP}_lso_s0/rollouts.npz q_ER8",
                                                   lso=f"{CLIP}_lso_s0/rollouts.npz q_best_it64", lsor=f"{CLIP}_lsor_s0/rollouts.npz q_best_it64",
                                                   ta=f"{CLIP}_ta_s0/rollouts.npz q_best_it64"), indent=1))
    return Q


def model(k):
    return load_model(MODELS["adult" if k == "adult" else BODY])


def measures(Q):
    """per body: lower hand height, lean direction (head minus centre of mass, horizontal) and centre of mass, adult-equivalent metres;
    for --moment kick also each foot's height (the higher of ankle and toe; left, right), the direction from the centre of mass to each toe
    and the facing (the hips' left-to-right line turned by 90 deg), degrees"""
    out = {}
    for k, q in Q.items():
        m = model(k); d = mujoco.MjData(m); s = 1.0 if k == "adult" else K; hl, hr, hd = (m.body(b).id for b in ("L_Hand", "R_Hand", "Head")); H, L, C = [], [], []
        ft = [[m.body(f"{x}_{b}").id for b in ("Ankle", "Toe")] for x in "LR"]; pl, pr = m.body("L_Hip").id, m.body("R_Hip").id; Fh, Fd, Fa = [], [], []
        for t in range(len(q)):
            d.qpos[:] = q[t]; mujoco.mj_forward(m, d); H.append(min(d.xpos[hl, 2], d.xpos[hr, 2]) / s); v = d.xpos[hd] - d.subtree_com[1]
            L.append(math.degrees(math.atan2(v[1], v[0]))); C.append(d.subtree_com[1][:2] / s)
            if MOMENT == "kick":
                Fh.append([max(d.xpos[i, 2] for i in f) / s for f in ft]); w = [d.xpos[f[1]] - d.subtree_com[1] for f in ft]; Fd.append([math.degrees(math.atan2(x[1], x[0])) for x in w])
                lr = d.xpos[pl] - d.xpos[pr]; Fa.append(math.degrees(math.atan2(-lr[0], lr[1])))
        out[k] = dict(hand=np.array(H), lean=np.array(L), com=np.array(C))
        if MOMENT == "kick":
            out[k].update(foot=np.array(Fh), kick_dir=np.array(Fd), facing=np.array(Fa))
    return out


def peaks(h, low, w=8):
    """local maxima over +-w frames above low"""
    return [t for t in range(len(h)) if h[t] > low and h[t] >= h[max(t - w, 0):t + w + 1].max()]


def kick_frame(M, win=10):
    """the adult reference's highest foot over the clip; every method's own peak of that foot within +-win frames"""
    a = M["adult"]["foot"]; side = int(np.argmax(a.max(0))); F = int(np.argmax(a[:, side])); leg = ("left", "right")[side]
    info = dict(frame=F, frame_in_reference_file=F + 1, seconds=round(F / 30, 2), kick_leg=leg, reference_foot_peaks_above_20cm={x: [[t, round(float(a[t, i]), 2)] for t in peaks(a[:, i], 0.2)] for i, x in enumerate(("left", "right"))}, methods={})
    for k, _ in BODIES:
        h = M[k]["foot"][:, side]; lo, hi = max(F - win, 0), min(F + win + 1, len(h)); p = lo + int(np.argmax(h[lo:hi])); top = int(np.argmax(h))
        info["methods"][k] = dict(kick_peak_frame=p, kick_peak_offset_frames=p - F, kick_peak_m=round(float(h[p]), 3), foot_at_frame_m=round(float(h[F]), 3), highest_foot_frame=top, highest_foot_m=round(float(h[top]), 3),
                                  kick_dir_deg_at_frame=round(float(M[k]["kick_dir"][F, side]), 1), facing_deg_at_frame=round(float(M[k]["facing"][F]), 1),
                                  com_off_reference_at_frame_m=round(float(np.linalg.norm(M[k]["com"][F] - M["adult"]["com"][F])), 2))
    return info


def pick_frame(Q):
    M = measures(Q)
    if MOMENT == "kick":
        return kick_frame(M), M
    F = int(np.argmin(M["adult"]["hand"])); info = dict(frame=F, frame_in_reference_file=F + 1, seconds=round(F / 30, 2), methods={})
    for k, _ in BODIES:
        h = M[k]["hand"]; low = int(np.argmin(h)); down = np.flatnonzero(h < 0.3)
        info["methods"][k] = dict(lowest_hand_frame=low, lowest_hand_m=round(float(h[low]), 3), hand_at_frame_m=round(float(h[F]), 3),
                                  hand_below_30cm_frames=[int(down.min()), int(down.max())] if len(down) else None, lean_deg_at_frame=round(float(M[k]["lean"][F]), 1),
                                  com_off_reference_at_frame_m=round(float(np.linalg.norm(M[k]["com"][F] - M["adult"]["com"][F])), 2))
    return info, M


def h_offset(po, right):
    """horizontal image extent (along the camera's right) of a pose's geoms, roughly (centres +- bounding radius)"""
    xs = [np.dot(g["p"][:2], right) for g in po["geoms"]]; r = [max(g["size"]) for g in po["geoms"]]
    lo = min(x - a for x, a in zip(xs, r)); hi = max(x + a for x, a in zip(xs, r)); return lo, hi


def yaw(q):
    """root yaw (rad) per frame, from the free joint's quaternion (as strobe_final_blender.py)"""
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def fit_view(info, key, keys=None):
    """--cam-azimuth fit: the world camera azimuth that keeps every panel's lean (kick) direction closest to the image plane (the smallest
    largest angle out of it), turned so that the adult reference's points right; keys: the bodies counted (default all five; must hold adult)"""
    dirs = {k: info["methods"][k][key] for k in (keys or [k for k, _ in BODIES])}; off = lambda d, c: abs((d - c + 90.0) % 180.0 - 90.0)       # angle between direction d and the line c (deg)
    c = min(np.arange(0.0, 180.0, 0.1), key=lambda c: max(off(d, c) for d in dirs.values()))                                 # the image plane's line
    cam = min((c + 90.0, c + 270.0), key=lambda z: abs((dirs["adult"] - (z - 90.0) + 180.0) % 360.0 - 180.0))                  # camera at z: direction z - 90 runs right
    info["moment_camera_fit"] = dict(rule="smallest largest angle between a panel's kick (lean) and the image plane", out_of_image_plane_deg={k: round(off(d, cam - 90.0), 1) for k, d in dirs.items()})
    return float(round(cam % 360.0, 1))


def export_moment(Q, info, P):
    F = info["frame"]; key = "lean_deg_at_frame" if MOMENT == "pickup" else "kick_dir_deg_at_frame"; lean = info["methods"]["adult"][key]; c_arg = ARG("--cam-azimuth", "")
    cam_world = lean + 90.0 if c_arg == "" else fit_view(info, key) if c_arg == "fit" else float(c_arg); h = math.radians(cam_world - AZ)     # camera side-on to the adult's lean (kick), or --cam-azimuth
    right = np.array([math.sin(math.radians(AZ)), -math.cos(math.radians(AZ))]); po_, size = {}, (M_W, M_H, M_FEET)
    for k, _ in BODIES:
        m = model(k); dh = 0.0
        if k in ALIGN and k != "adult":                                   # turned about its root to the reference's root yaw at this frame
            dh = float(yaw(Q[k][F:F + 1])[0] - yaw(Q["adult"][F:F + 1])[0]); info.setdefault("moment_align_yaw_deg", {})[k] = round(math.degrees(math.atan2(math.sin(dh), math.cos(dh))), 1)
        po = SCENE.pose(m, Q[k], F, 0.0, 0.0, h + dh, False, "human")
        if GROUND and k == "adult":                                       # stood on the floor: lowered by its own lowest point at this frame
            from model.robot_map import Lowest
            d = mujoco.MjData(m); d.qpos[:] = Q[k][F]; mujoco.mj_kinematics(m, d); low = float(Lowest(m)(d))
            for g in po["geoms"]:
                g["p"][2] = round(g["p"][2] - low, 5)
            po["note"] = (f"adult reference lowered by {low:.4f} m: its lowest point at this frame (model.robot_map.Lowest on outputs/user_survey/models/adult.xml), "
                          "the AMASS origin motion's floor being off by a few cm; the simulated bodies are not moved"); info["moment_adult_lowered_m"] = round(low, 4)
        lo, hi = h_offset(po, right); top = max(g["p"][2] + max(g["size"]) for g in po["geoms"])
        po["shift"] = -(lo + hi) / 2; print(f"  moment {k}: frame {F}, width {hi - lo:.2f} m, top {top:.2f} m, centred by {po['shift']:+.2f} m", flush=True); po_[k] = (po, hi - lo, top)
    if not LEGACY:                                                        # one scale for all panels: the widest and the tallest pose, PAD around
        size = (max(w for _, w, _ in po_.values()) + 2 * PAD, max(t for _, _, t in po_.values()) + M_FEET + PAD, M_FEET); info["moment_panel_m"] = [round(size[0], 2), round(size[1], 2)]
    for k, _ in BODIES:
        P[f"{PFX}_moment_{k}"] = dict(figure="moment", width_m=size[0], height_m=size[1], feet_m=size[2], px=[M_PX, round(M_PX * size[1] / size[0])], white=True, poses=[po_[k][0]], frames=[F], view="fixed", cam_azimuth_world=cam_world)
    info["moment_camera_world_azimuth_deg"] = round(cam_world, 1)


def seq_frames(h, F):
    """--sequence 5: the kick's frames from the adult reference's kicking-foot height h (the higher of ankle and toe, m) and its peak F.
    Standing height b = the median of h over the clip; wind-up = the last frame before F with the foot within NEAR of b (still on the
    floor); leg rising = the first frame before F at or above b + HALF (h[F] - b); kick peak = F; retraction = the first frame after F at or
    below that; foot back down = the first frame after F within NEAR of b again"""
    assert SEQ == 5, "seq_frames picks 5 frames"
    b = float(np.median(h)); mid = b + HALF * (h[F] - b); pre, post = np.arange(F), np.arange(F + 1, len(h))
    w = int(pre[h[pre] <= b + NEAR].max()); fr = [w, int(pre[(h[pre] >= mid) & (pre > w)].min()), F, int(post[h[post] <= mid].min()), int(post[h[post] <= b + NEAR].min())]
    names = ("wind-up", "leg rising", "kick peak", "retraction", "foot back down"); assert fr == sorted(fr) and len(set(fr)) == 5, fr
    rule = (f"from the adult reference's kicking-foot height h (the higher of ankle and toe): standing height b = median of h over the clip "
            f"({b:.3f} m); wind-up = last frame before the peak with h <= b + {NEAR} m; leg rising = first frame after that with h >= b + {HALF} (h_peak - b) "
            f"({mid:.3f} m); kick peak = the --moment kick frame (the reference's highest foot); retraction = first frame after the peak with h <= {mid:.3f} m; "
            f"foot back down = first frame after the peak with h <= b + {NEAR} m")
    return fr, names, rule, b


def export_sequence(Q, info, M, P):
    """--sequence: SEQ_METHOD's pose at each of the kick's frames, one panel each, with the adult reference overlaid (--overlay-adult)"""
    F = info["frame"]; side = ("left", "right").index(info["kick_leg"]); h_ad = M["adult"]["foot"][:, side]; frames, names, rule, b = seq_frames(h_ad, F)
    key = "kick_dir_deg_at_frame"; c_arg = ARG("--cam-azimuth", ""); keys = ["adult", SEQ_METHOD]
    cam_world = info["methods"]["adult"][key] + 90.0 if c_arg == "" else fit_view(info, key, keys) if c_arg == "fit" else float(c_arg); h = math.radians(cam_world - AZ)
    right = np.array([math.sin(math.radians(AZ)), -math.cos(math.radians(AZ))]); c, s = math.cos(-h), math.sin(-h); Rz = np.array([[c, -s], [s, c]])   # the pose export's turn
    ya, ym = yaw(Q["adult"]), yaw(Q[SEQ_METHOD]); rows, built = [], []
    for i, t in enumerate(frames):
        poses, row = [], dict(panel=f"{PFX}_sequence_f{t:03d}", position=i + 1, moment=names[i], frame=t, frame_in_reference_file=t + 1, seconds=round(t / 30, 2),
                              adult_kick_foot_m=round(float(h_ad[t]), 3), method_kick_foot_m=round(float(M[SEQ_METHOD]["foot"][t, side] * K), 3))
        for k in (["adult"] if OVERLAY else []) + [SEQ_METHOD]:
            m = model(k); d = mujoco.MjData(m); d.qpos[:] = Q[k][t]; mujoco.mj_forward(m, d); com, root = d.subtree_com[1][:2].copy(), Q[k][t, :2].copy()
            po = SCENE.pose(m, Q[k], t, 0.0, 0.0, h, False, "human"); dxy = Rz @ (com - root)         # the pose export centres on the centre of mass: put the root (pelvis) at the origin instead
            for g in po["geoms"]:
                g["p"][0] = round(g["p"][0] + float(dxy[0]), 5); g["p"][1] = round(g["p"][1] + float(dxy[1]), 5)
            row[f"{k}_root_xy_m"] = [round(float(x), 4) for x in root]
            if k == "adult":                                              # moved by (method root - adult root): both roots at the origin; light and see-through
                po.update(tint=ADULT_LOOK[0], alpha=ADULT_LOOK[1])
                if GROUND:                                                # as --ground on the moment figure
                    from model.robot_map import Lowest
                    d2 = mujoco.MjData(m); d2.qpos[:] = Q[k][t]; mujoco.mj_kinematics(m, d2); low = float(Lowest(m)(d2)); row["adult_lowered_m"] = round(low, 4)
                    for g in po["geoms"]:
                        g["p"][2] = round(g["p"][2] - low, 5)
            poses.append(po)
        if OVERLAY:
            off = Q[SEQ_METHOD][t, :2] - Q["adult"][t, :2]; dy = float(ym[t] - ya[t])
            row.update(adult_offset_xy_m=[round(float(x), 4) for x in off], heading_method_minus_adult_deg=round(math.degrees(math.atan2(math.sin(dy), math.cos(dy))), 1))
        gs = [g for po in poses for g in po["geoms"]]; lo, hi = h_offset(dict(geoms=gs), right); top = max(g["p"][2] + max(g["size"]) for g in gs)
        for po in poses:
            po["shift"] = -(lo + hi) / 2
        row.update(width_m=round(hi - lo, 3), top_m=round(top, 3)); rows.append(row); built.append((poses, hi - lo, top))
        print(f"  {row['panel']} ({names[i]}): adult foot {h_ad[t]:.2f} m, {SEQ_METHOD} foot {row['method_kick_foot_m']:.2f} m, width {hi - lo:.2f} m, top {top:.2f} m"
              + (f", adult moved {row['adult_offset_xy_m']}, heading off {row['heading_method_minus_adult_deg']:+.1f} deg" if OVERLAY else ""), flush=True)
    w_, h_ = max(w for _, w, _ in built) + 2 * PAD, max(t for _, _, t in built) + M_FEET + PAD; r_ = SEQ_PX[1] / SEQ_PX[0]
    size = (max(w_, h_ / r_), max(h_, w_ * r_), M_FEET)                  # grown to SEQ_PX's shape (room added at the sides or the top)
    for (poses, _, _), row in zip(built, rows):
        P[row["panel"]] = dict(figure="sequence", width_m=size[0], height_m=size[1], feet_m=size[2], px=list(SEQ_PX), white=True, poses=poses,
                               frames=[row["frame"]], view="fixed", cam_azimuth_world=cam_world)
    info["sequence"] = dict(method=SEQ_METHOD, frames=frames, rule=rule, standing_foot_m=round(b, 3), panel_m=[round(size[0], 2), round(size[1], 2)],
                            camera_world_azimuth_deg=round(cam_world, 1), panels=rows)
    src = dict(clip=CLIP, body=BODY, method=f"{SEQ_METHOD} (ours)", run=f"{RUNS_REL}/{CLIP}_{SEQ_METHOD}_s0/rollouts.npz", key="q_best_it64",
               adult=f"amass_data/amass_origin_motion/ACCAD/{CLIP}.npz qpos[1:] (full adult size)" if OVERLAY else None,
               frame_alignment="simulated q[t] against the adult reference's qpos[1:][t] (its file frame t + 1); seconds = t / 30",
               kick=dict(frame=F, leg=info["kick_leg"], adult_peak_foot_m=round(float(h_ad[F]), 3)), frames=frames, rule=rule,
               alignment=("the adult moved horizontally only, its root (pelvis, free joint x, y) onto the method's root at that frame; no turn, no scaling"
                          + ("; lowered by its own lowest point at that frame (model.robot_map.Lowest on outputs/user_survey/models/adult.xml)" if GROUND else "")) if OVERLAY else None,
               camera=dict(world_azimuth_deg=round(cam_world, 1), rule=("fit over adult and " + SEQ_METHOD + " kick directions at the kick peak") if c_arg == "fit" else c_arg or "side-on to the adult's kick"),
               look=dict(method=dict(colour="body", srgb=list(BODY_RGB), alpha=1.0), adult=dict(colour="body", tint_to_white=ADULT_LOOK[0],
                         srgb=[round((1 - ADULT_LOOK[0]) * x + ADULT_LOOK[0], 3) for x in BODY_RGB], alpha=ADULT_LOOK[1]) if OVERLAY else None),
               panels=rows)
    (OUT / "source.json").write_text(json.dumps(src, indent=1))


def trail(xy, color, r):
    """thin capsules on the floor along a path (as scripts/topdown_blender.py)"""
    gs = []
    for a, b in zip(xy[:-1], xy[1:]):
        d = b - a; L = float(np.linalg.norm(d))
        if L < 1e-4:
            continue
        dx, dy = d / L; mid = (a + b) / 2
        gs.append(dict(kind="capsule", size=[r, L / 2], p=[float(mid[0]), float(mid[1]), r], R=[-dy, 0.0, dx, dx, 0.0, dy, 0.0, 1.0, 0.0], color=color))
    return gs


def auto_frames(Q, S, root, info, step=10, cell=0.02):
    """--top-frames auto: the first and the last frame and TOP_N - 2 frames between them (candidates every step frames), the set whose poses
    overlap least on the floor plane: the sum, over the five panels and every pair of poses, of their shared footprint as a fraction of the
    smaller one (footprints rasterised at cell m, adult-equivalent; a capsule as 5 discs along its axis, a sphere as its disc, a box as its
    bounding sphere), so a punching arm or a kicking leg counts. Searched over every such set."""
    T = len(Q["adult"]); cand = sorted(set(range(0, T, step)) | {T - 1}); u = np.linspace(-1.0, 1.0, 5); D = {}; GEOM = _geom_kinds()
    for k, _ in BODIES:
        m = model(k); d = mujoco.MjData(m); vis = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_group[g] < 3 and m.geom_rgba[g, 3] != 0 and m.geom_type[g] in GEOM]
        cap = [g for g in vis if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]; rest = [g for g in vis if g not in cap]; assert {GEOM[m.geom_type[g]] for g in rest} <= {"sphere", "box"}
        r = np.concatenate([np.repeat(m.geom_size[cap, 0], len(u)), [m.geom_size[g, 0] if GEOM[m.geom_type[g]] == "sphere" else np.linalg.norm(m.geom_size[g]) for g in rest]]) * S[k]; P_ = []
        for t in cand:
            d.qpos[:] = Q[k][t]; mujoco.mj_forward(m, d); ax = d.geom_xmat[cap].reshape(-1, 3, 3)[:, :, 2] * m.geom_size[cap, 1:2]
            P_.append(np.concatenate([(d.geom_xpos[cap][:, None, :2] + u[None, :, None] * ax[:, None, :2]).reshape(-1, 2), d.geom_xpos[rest, :2]]) * S[k])
        D[k] = (np.array(P_), r)
    lo = np.min([p.reshape(-1, 2).min(0) for p, _ in D.values()], 0) - 0.3; hi = np.max([p.reshape(-1, 2).max(0) for p, _ in D.values()], 0) + 0.3
    gx, gy = (lo[i] + cell * np.arange(int((hi[i] - lo[i]) / cell) + 1) for i in range(2)); O = {}
    for k, (P_, r) in D.items():
        A_ = np.zeros((len(cand), len(gx) * len(gy)), np.float32)
        for i in range(len(cand)):
            mk = np.zeros((len(gx), len(gy)), bool)
            for (x, y), rr in zip(P_[i], r):
                a0, a1 = np.searchsorted(gx, [x - rr, x + rr]); b0, b1 = np.searchsorted(gy, [y - rr, y + rr])
                mk[a0:a1, b0:b1] |= (gx[a0:a1, None] - x) ** 2 + (gy[None, b0:b1] - y) ** 2 <= rr * rr
            A_[i] = mk.ravel()
        inter = A_ @ A_.T; area = np.diag(inter); O[k] = inter / np.minimum(area[:, None], area[None, :])
    tot = sum(O.values()); sets = np.array([[0, len(cand) - 1, *c] for c in itertools.combinations(range(1, len(cand) - 1), TOP_N - 2)])
    pairs = list(itertools.combinations(range(TOP_N), 2)); score = sum(tot[sets[:, a], sets[:, b]] for a, b in pairs); best = sorted(sets[int(np.argmin(score))])
    frames = tuple(cand[i] for i in best); worst = {k: round(float(max(o[a, b] for a, b in itertools.combinations(best, 2))), 2) for k, o in O.items()}
    info["top_frames_auto"] = dict(rule=f"start, end and {TOP_N - 2} more (candidates every {step} frames): least summed footprint overlap over the panels", summed_overlap=round(float(score.min()), 2),
                                   largest_pair_overlap_per_panel=worst)
    print("  top-down frames (auto):", frames, "summed overlap", round(float(score.min()), 2), "largest pair overlap per panel", worst, flush=True); return frames


def export_topdown(Q, info, P):
    r_hat = np.array([math.sin(math.radians(AZ)), -math.cos(math.radians(AZ))]); u_hat = np.array([math.cos(math.radians(AZ)), math.sin(math.radians(AZ))])   # image right, up
    S = {k: 1.0 if k == "adult" else 1.0 / K for k, _ in BODIES}                      # to adult-equivalent metres
    if TOP_LSOR:                                                          # --top-lsor: this figure's LSOR from another run
        f, key = TOP_LSOR.rsplit(":", 1); Q = dict(Q, lsor=np.asarray(np.load(paths.read_path(f))[key], np.float64)); assert Q["lsor"].shape == Q["adult"].shape, Q["lsor"].shape
        np.savez_compressed(OUT / "trajectories/lsor_topdown.npz", qpos=Q["lsor"].astype(np.float32), fps=np.float32(30)); info["topdown_lsor"] = f"{f} {key} (trajectories/lsor_topdown.npz)"
    root, pel = {}, {k: Q[k][:, :2] * S[k] for k, _ in BODIES}             # pelvis (free joint) path: the drawn lines, direction and framing (as topdown_blender.py)
    for k, _ in BODIES:                                                   # centre of mass, every frame: where the pose export's root-centred pose goes back to
        m = model(k); d = mujoco.MjData(m); c = []
        for t in range(len(Q[k])):
            d.qpos[:] = Q[k][t]; mujoco.mj_forward(m, d); c.append(d.subtree_com[1][:2] * S[k])
        root[k] = np.array(c)
    ref = pel["adult"]; v = ref[-1] - ref[0]; th = math.atan2(r_hat[1], r_hat[0]) - math.atan2(v[1], v[0])
    Rz = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    TOP = (len(Q["adult"]) - 1,) if FINAL else tuple(int(x) for x in TOPF.split(",")) if TOPF != "auto" else auto_frames(Q, S, root, info)
    if FINAL:                                                             # the reference body's last pose, a ghost on the method panels: [model, qpos, scale, centre of mass at that frame]
        gq = np.asarray(np.load(REPO / f"data/{BODY_DATA[BODY]}/amass_retargeting_motion/ACCAD/{CLIP}.npz")["qpos"][1:], np.float64); assert gq.shape == Q["adult"].shape, gq.shape
        gm = model(BODY); d = mujoco.MjData(gm); d.qpos[:] = gq[TOP[0]]; mujoco.mj_forward(gm, d); ghost = (gm, gq, 1.0 / K, d.subtree_com[1][:2] / K)
        np.savez_compressed(OUT / "trajectories/reference_body.npz", qpos=gq.astype(np.float32), fps=np.float32(30))
    allxy = np.concatenate(list(pel.values())) @ Rz.T; a, b = allxy @ r_hat, allxy @ u_hat
    ctr = (a.min() + a.max()) / 2 * r_hat + (b.min() - T_BELOW + b.max() + T_MARGIN) / 2 * u_hat; place = lambda xy: xy @ Rz.T - ctr
    W = float(np.ptp(a)) + 2 * T_MARGIN; H = float(np.ptp(b)) + T_MARGIN + T_BELOW
    if not LEGACY:                                                        # framed instead to the paths, every drawn geom and its shadow (penumbra: the key light's 14 deg), PAD around
        ga, gb = [a.min() - PAD, a.max() + PAD], [b.min() - PAD, b.max() + PAD]
        items = [(model(k), Q[k], t, S[k], root[k][t]) for k, _ in BODIES for t in TOP] + ([(ghost[0], ghost[1], TOP[0], ghost[2], ghost[3])] if FINAL else [])
        for m, q, t, sk, rt in items:
            p = SCENE.pose(m, q, t, 0.0, 0.0, -th, False, "human"); off = rt @ Rz.T
            for g in p["geoms"]:
                c = np.array([sk * g["p"][0] + off[0], sk * g["p"][1] + off[1], sk * g["p"][2]]); r = sk * (g["size"][0] + g["size"][1] if g["kind"] == "capsule" else float(np.linalg.norm(g["size"])))
                for xy, rr in ((c[:2], r), (c[:2] + c[2] * SUN[:2], r + 0.123 * c[2])):      # the geom and its shadow (the poses are turned already, the light is not)
                    ga += [xy @ r_hat - rr - PAD, xy @ r_hat + rr + PAD]; gb += [xy @ u_hat - rr - PAD, xy @ u_hat + rr + PAD]
        ctr = (min(ga) + max(ga)) / 2 * r_hat + (min(gb) + max(gb)) / 2 * u_hat; W = max(ga) - min(ga); H = max(gb) - min(gb)
    tints = [0.6 * (1 - i / (len(TOP) - 1)) for i in range(len(TOP))] if len(TOP) > 1 else [0.0]
    base = dict(figure="topdown", width_m=W, height_m=H, feet_m=H / 2, px=[T_PX, round(H * T_PX / W)], white=True, elevation=-89.5, cam_distance=80.0)
    for k, _ in BODIES:
        m = model(k); ps = []
        for t, tint in zip(TOP, tints):
            p = SCENE.pose(m, Q[k], t, 0.0, tint, -th, False, "human"); off = place(root[k][t][None])[0]      # root-centred and turned by th
            for g in p["geoms"]:                                          # scaled S about the centre of mass's floor point, then placed
                assert g["kind"] in ("box", "capsule", "sphere"), g["kind"]
                g["p"] = [S[k] * g["p"][0] + float(off[0]), S[k] * g["p"][1] + float(off[1]), S[k] * g["p"][2]]; g["size"] = [S[k] * x for x in g["size"]]; g["color"] = "body"
            p["x"] = 0.0; ps.append(p)
        if FINAL and k != "adult":                                        # where it should be: the reference body at that frame, see-through grey
            p = SCENE.pose(ghost[0], ghost[1], TOP[0], 0.0, 0.0, -th, False, "human"); off = place(ghost[3][None])[0]
            for g in p["geoms"]:
                g["p"] = [ghost[2] * g["p"][0] + float(off[0]), ghost[2] * g["p"][1] + float(off[1]), ghost[2] * g["p"][2]]; g["size"] = [ghost[2] * x for x in g["size"]]; g["color"] = GHOST[0]
            p.update(x=0.0, alpha=GHOST[1]); ps.append(p)
        own = trail(place(pel[k]), "body", R_OWN); extra = [] if k == "adult" else trail(place(ref), "black", R_REF)
        if FINAL:                                                         # the own path lighter, a dot at its start
            a0 = place(pel[k][:1])[0]; extra.append(dict(kind="sphere", size=[R_DOT, 0.0, 0.0], p=[float(a0[0]), float(a0[1]), R_DOT], R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], color="body"))
        lines = [dict(x=0.0, tint=PATH_TINT, geoms=own), dict(x=0.0, tint=0.0, geoms=extra)] if FINAL else [dict(x=0.0, tint=0.0, geoms=own + extra)]
        P[f"{PFX}_topdown_{k}"] = dict(base, poses=ps + lines, frames=list(TOP))
    end = {k: round(float(np.linalg.norm(pel[k][-1] - ref[-1])), 2) for k, _ in BODIES}; mean = {k: round(float(np.linalg.norm(pel[k] - ref, axis=1).mean()), 2) for k, _ in BODIES}
    info["topdown"] = dict(frames=list(TOP), panel_m=[round(W, 2), round(H, 2)], units=f"adult-equivalent metres ({BODY} scene x 1/{K})",
                           pelvis_distance_to_reference_end_m=end, pelvis_distance_to_reference_mean_m=mean)
    print(f"  topdown: {W:.2f} x {H:.2f} m, frames {TOP}; distance to the reference at the end {end}, mean {mean}", flush=True)


def export():
    Q = trajectories(); info, M = pick_frame(Q); P = SCENE.panels
    print(f"{MOMENT} frame", info["frame"], json.dumps(info["methods"], indent=None), flush=True)
    if SEQ:                                                               # --sequence: the kick filmstrip only
        export_sequence(Q, info, M, P)
    else:
        export_moment(Q, info, P); export_topdown(Q, info, P)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "frames.json").write_text(json.dumps(info, indent=1))
    SCENE.write(OUT, placeholder=True)                                    # scene.json (+ "version"), meshes.npz (+ "_"), render_blender.py
    print("exported", len(P), "panels", flush=True)


def render():
    sel = sum([["--panel", n] for n in json.loads((OUT / "scene.json").read_text())["panels"] if n.startswith(tuple(ARG("--only", f"{PFX}_").split(",")))], [])
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--samples", str(SAMPLES),
                    "--scale", str(SCALE), "--device", "CPU", "--threads", str(THREADS), "--out", "renders"] + sel, cwd=OUT, env=dict(os.environ, OMP_NUM_THREADS="1"),
                   check=True, stdout=open(OUT / "render.log", "w"), stderr=subprocess.STDOUT)


def combine(gap=24):
    from PIL import Image
    (ST / "combined").mkdir(exist_ok=True)
    if SEQ:                                                               # --sequence: its panels left to right (the frames' order), each also in panels/
        names = sorted((n for n in json.loads((OUT / "scene.json").read_text())["panels"] if n.startswith(f"{PFX}_sequence_")), key=lambda n: int(n.rsplit("_f", 1)[1])); ims = [Image.open(OUT / "renders" / f"{n}.png").convert("RGB") for n in names]
        out = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), max(im.height for im in ims)), "white"); x = 0
        for n, im in zip(names, ims):
            out.paste(im, (x, 0)); x += im.width + gap; shutil.copy(OUT / "renders" / f"{n}.png", ST / "panels" / f"amass_{BODY}_{TAG}_{n}.png")
        out.save(ST / "combined" / f"amass_{BODY}_{TAG}_{MOMENT}_sequence.png"); print("sequence", out.size, names, flush=True); return
    for fig in [f for f in ("moment", "topdown") if f"{PFX}_{f}_".startswith(tuple(ARG("--only", f"{PFX}_").split(",")))]:
        ims = [Image.open(OUT / "renders" / f"{PFX}_{fig}_{k}.png").convert("RGB") for k, _ in BODIES]
        out = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), max(im.height for im in ims)), "white"); x = 0
        for im in ims:
            out.paste(im, (x, 0)); x += im.width + gap
        out.save(ST / "combined" / f"amass_{BODY}_{TAG}_{FIG[fig]}.png"); print(fig, out.size, flush=True)


if __name__ == "__main__":
    steps = {x for x in sys.argv[1:] if x in ("export", "render", "combine")} or {"export", "render", "combine"}
    for s_ in ("export", "render", "combine"):
        if s_ in steps:
            globals()[s_]()
