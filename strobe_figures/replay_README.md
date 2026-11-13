# User survey videos: data to re-render them

A frozen motion model (Meta Motivo M-1, trained on an adult human body) controls a body it was never trained on. For
each motion clip it is run twice: with the clip's original latent `z0`, and with a latent found by search on the new
body (`best`). This folder holds the videos and everything needed to re-render them exactly.

## Folders

```
README.md, replay.py          this file; the renderer
models/                       every MuJoCo model the videos use
<body>/side_by_side/<clip>.mp4
<body>/overlay/<clip>.mp4
<body>/trajectories/<clip>/   reference_adult.npz, reference_body.npz, z0.npz, best.npz, video.json
```

Bodies: `child`, `fat`, `tall`, `heavy_arm` (heavy right hand), `limit_leg` (right leg locked, bent) are variants of the
same humanoid; `g1` is the Unitree G1 robot (driven through a human-to-robot mapping). `heavy_arm` walks
`move-ego-0-2_4` instead of `move-ego-0-2_2`.

## What the videos show

| video | layout | colours |
|---|---|---|
| side_by_side | 3 square panels (1080 px), left to right: `reference_adult`, `z0`, `best` | grey, red, blue |
| overlay | one scene (1920 x 1080): `reference_body` as a translucent ghost, `z0`, `best` | grey (translucent), red, blue |

- `reference_adult`: the adult doing the original motion, on `models/adult.xml`.
- `reference_body`: the target on the new body: the motion retargeted to the body (`child`, `fat`, `tall`), the
  adult's motion (`heavy_arm`, `limit_leg`), or the motion mapped onto the robot kinematically (`g1`).
- `z0`, `best`: physics rollouts (MuJoCo) of the model on the new body with the original and the searched latent.

## Trajectory files

Each `.npz` holds `qpos` (frames x nq, float32) and `fps` (30). Clips are 120-300 frames.

- Humanoid bodies (nq = 76): root position x, y, z (m), root orientation as a quaternion (w, x, y, z), then 69 hinge
  angles (rad) in the model's joint order.
- G1 (nq = 36): root position (3), root quaternion (4), 29 joint angles (rad).

The rollouts are stored as joint positions, so replaying them is purely kinematic: set `qpos`, run forward kinematics,
render. `video.json` gives each video's models (paths relative to this folder), colours (RGBA) and camera (free camera
with distance, elevation, azimuth; it follows the body horizontally at a fixed look-at height).

## Re-rendering

Requirements: Python 3, `mujoco` (made with 3.11.0), `numpy`, `imageio`, `imageio-ffmpeg`. Headless machines need
`MUJOCO_GL=egl` (or `osmesa`).

```
cd <this folder>
MUJOCO_GL=egl python replay.py child/trajectories/jump-2_6 --out-dir out
#   -> out/jump-2_6_side_by_side.mp4, out/jump-2_6_overlay.mp4
python replay.py g1/trajectories/headstand_4 --only overlay --out-dir out
```

Checked: re-rendered videos match the ones here to a mean pixel difference below 0.2 (of 255), i.e. video encoding
noise.

The G1 model needs `models/g1/` and `models/g1_menagerie/assets/` together (`g1.xml` loads its meshes from
`../g1_menagerie/assets`).

Loading one trajectory by hand:

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("models/child.xml"); d = mujoco.MjData(m)
q = np.load("child/trajectories/jump-2_6/best.npz")["qpos"]
for t in range(len(q)):
    d.qpos[:] = q[t]; mujoco.mj_forward(m, d)   # body positions in d.xpos, orientations in d.xquat
```
