#!/usr/bin/env python3
"""retarget_consistency_check.py -- are the retargeted targets' root path and heading consistent with their own
bodies? The root path is the adult's scaled by s (child 0.611 by pelvis height; robots by leg length), the heading
copied; if a body's stride does not scale by the same s, the target's planted feet slide.
Per motion (the adult's original clip, the child's retarget data/child/retargeting_motion, the G1 / H1 kinematic
targets model.robots.target_path: outputs/g1/roundtrip/<stem>/g1.npz, outputs/h1/roundtrip/<stem>/robot.npz):
  speed      mean horizontal pelvis speed, m/s
  slide      mean horizontal speed of an ankle while it is planted (within 3 cm of its lowest height in the clip,
             both frames of the difference), m/s, and as a share of the pelvis speed
  heading    pelvis yaw against the adult's, after removing a constant offset (the bodies' forward axes differ):
             mean |difference| (deg), and the offset
  drift      angle between the direction of travel (pelvis velocity over 5 frames) and the facing, mean |.| (deg),
             frames moving faster than 0.3 m/s; compared with the adult's own
usage: uv run python scripts/retarget_consistency_check.py [task/stem ...]"""
import sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FPS = 30
BODIES = {"adult": ("assets/robots/adult/robot.xml", "data/origin_motion/{clip}.npz", "Pelvis", ("L_Ankle", "R_Ankle"), "x"),
          "child": ("assets/robots/child/robot.xml", "data/child/retargeting_motion/{clip}.npz", "Pelvis", ("L_Ankle", "R_Ankle"), "x"),
          "G1": ("assets/robots/g1/g1.xml", "outputs/g1/roundtrip/{stem}/g1.npz", "pelvis",
                 ("left_ankle_roll_link", "right_ankle_roll_link"), "x"),
          "H1": ("assets/robots/h1/scene.xml", "outputs/h1/roundtrip/{stem}/robot.npz", "pelvis", ("left_ankle_link", "right_ankle_link"), "x")}


def track(xml, q, root, feet):
    m = mujoco.MjModel.from_xml_path(str(REPO / xml)); d = mujoco.MjData(m)
    r, f = m.body(root).id, [m.body(n).id for n in feet]
    P, F, Rm = [], [], []
    for qq in q:
        d.qpos[:] = qq; mujoco.mj_kinematics(m, d)
        P.append(d.xpos[r].copy()); F.append(d.xpos[f].copy()); Rm.append(d.xmat[r].reshape(3, 3).copy())
    return np.array(P), np.array(F), np.array(Rm)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    clips = sys.argv[1:] or ["move-ego-0-4/move-ego-0-4_8", "move-ego-0-2/move-ego-0-2_2"]
    for clip in clips:
        stem = clip.split("/")[1]; print(f"== {clip}")
        yaw_adult = None
        for name, (xml, path, root, feet, _) in BODIES.items():
            q = np.load(REPO / path.format(clip=clip, stem=stem))["qpos"].astype(np.float64)
            P, F, Rm = track(xml, q, root, feet)
            v = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) * FPS
            slides = []
            for k in range(F.shape[1]):
                z = F[:, k, 2]; low = z < z.min() + 0.03
                both = low[1:] & low[:-1]
                slides.append(np.linalg.norm(np.diff(F[:, k, :2], axis=0), axis=1)[both] * FPS)
            slide = np.concatenate(slides)
            # facing: the pelvis axis whose horizontal direction best matches the travel direction over the clip
            vel = (P[5:, :2] - P[:-5, :2]) * FPS / 5; mv = np.linalg.norm(vel, axis=1) > 0.3
            trav = np.arctan2(vel[:, 1], vel[:, 0])
            best = None
            for ax in range(3):
                for sgn in (1, -1):
                    a = sgn * Rm[:, :, ax]; yaw = np.arctan2(a[:, 1], a[:, 0])
                    e = np.abs(wrap(trav - yaw[:-5]))[mv].mean() if mv.any() else np.inf
                    if best is None or e < best[0]:
                        best = (e, yaw)
            drift, yaw = np.degrees(best[0]), best[1]
            if yaw_adult is None:
                yaw_adult = yaw
            dy = wrap(yaw - yaw_adult[:len(yaw)]); off = np.angle(np.exp(1j * dy).mean())
            hd = np.degrees(np.abs(wrap(dy - off)).mean())
            print(f"  {name:6s} speed {v.mean():4.2f} m/s | planted-foot slide {slide.mean():.3f} m/s ({100 * slide.mean() / v.mean():4.1f}% of speed, "
                  f"{len(slide)} foot-frames) | heading vs adult {hd:4.1f} deg (offset {np.degrees(off):+.0f}) | travel vs facing {drift:4.1f} deg")


if __name__ == "__main__":
    main()
