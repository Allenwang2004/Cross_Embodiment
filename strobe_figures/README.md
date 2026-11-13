# Strobe figures — how to render them, exactly

Self-contained package for the qualitative humanoid figures in the paper: the
multi-pose **strobe PNGs** (ghost poses on a white plate) and the **side-by-side
comparison MP4s** (`adult | z0 | z*`).

Everything needed is here: the two entry points, the shared rendering library,
every frozen figure recipe (`configs/strobe/`), the MuJoCo models those recipes
load, and the recorded joint trajectories the paper figures replay. **No result
images are shipped** — you regenerate them.

Verified on 2026-11-09 with Python 3.11.15 + MuJoCo 3.11.0: all 6 replayable
recipes render in ~15 s each and reproduce the published PNGs **bit-identical or
within 1/255 on at most 96 of 921,600 pixels** (see *Fidelity* below).

---

## 1. The two pipelines

| | `replay_strobe.py` | `render_strobe.py` |
|---|---|---|
| motion comes from | recorded `qpos` dumps in a clip folder (`data.qpos` + `mj_forward`) | a frozen MetaMotivo actor rolled out live under a latent |
| needs | `mujoco numpy imageio imageio-ffmpeg pillow PyYAML gymnasium` | the above **plus** `humenv`, `metamotivo`, `torch`, a GPU-friendly env, the latents and `data/origin_motion` |
| physics | none — purely kinematic replay | MuJoCo rollouts through humenv |
| status in this package | **fully working** | code is here, the model weights / latents / humenv task models are **not** |

The published figures split across both pipelines (§5). For anything replayable
use `replay_strobe.py`: it is deterministic, needs no weights, and its outputs
are byte-comparable to the paper's.

Both scripts share one config schema and one rendering library
(`tools/strobe_wrapper.py`), so **the same knobs render the same figure in
either script** — that is deliberate.

## 2. Layout

```
strobe_figures/
├── README.md                  this file
├── requirements.txt           pinned deps for the replay path
├── run_all_figures.sh         renders every recipe in configs/strobe/
├── replay_strobe.py           ENTRY POINT 1 — figure from recorded trajectories
├── render_strobe.py           ENTRY POINT 2 — figure from a latent rollout (needs full repo)
├── replay.py, replay_README.md   provenance: the original survey videos these
│                                trajectories were cut from (not needed for figures)
├── tools/strobe_wrapper.py    the renderer: colour gradient, camera orbit, compositing
├── model/exact_obs.py         observation helper, used only by render_strobe.py
├── configs/
│   ├── render_strobe.yaml     base config (every knob, documented inline)
│   ├── running_strobe.md      the author's 4-line original workflow
│   └── strobe/                one frozen YAML per published figure  <-- recipes
├── models/                    MuJoCo models the recipes load
│   ├── adult.xml child.xml fat.xml tall.xml heavy_arm.xml limit_leg.xml
│   ├── g1/{g1.xml,scene.xml}
│   └── g1_menagerie/assets/   the G1 STL meshes (g1.xml loads ../g1_menagerie/assets)
└── outputs/<body>/trajectories/<clip>/
    ├── reference_adult.npz    the adult doing the original motion   -> panel "adult"
    ├── reference_body.npz     the motion retargeted onto this body  -> panel "body" (optional)
    ├── z0.npz                 rollout under the clip's own latent    -> panel "z0"
    ├── best.npz               rollout under the searched latent      -> panel "zstar"
    └── video.json             models, colours, camera, fps used for that clip
```

Paths in the configs are relative to the package root, so **keep the tree as
shipped** and run from inside `strobe_figures/`.

## 3. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # mujoco==3.11.0 numpy imageio imageio-ffmpeg pillow PyYAML gymnasium
```

Headless / no X display (servers, CI, containers):

```bash
export MUJOCO_GL=egl        # needs libEGL (nvidia drivers); osmesa is the CPU fallback
```

`replay_strobe.py` already defaults `MUJOCO_GL=egl` if you did not set it.
`MUJOCO_GL=osmesa` works without a GPU (slower, and pixel-level it can differ
slightly from the EGL renders).

## 4. Reproduce a paper figure

```bash
cd strobe_figures
MUJOCO_GL=egl python replay_strobe.py --config configs/strobe/heavy_arm/move-ego-0-2_strobe.yaml
```

That writes into `outputs/figures/strobe/`:

```
heavy_arm_move-ego-0-2_orbit_auto_adult_strobe.png     ghost-pose figure, adult panel
heavy_arm_move-ego-0-2_orbit_auto_z0_strobe.png        ghost-pose figure, z0 panel
heavy_arm_move-ego-0-2_orbit_auto_zstar_strobe.png     ghost-pose figure, z* panel
heavy_arm_move-ego-0-2_orbit_auto_compare_adult_z0_zstar.mp4
```

Output names are always
`<task_name>_<camera.type>_<label>_strobe.png` and
`<task_name>_<camera.type>_compare_<labels>.mp4`, where `task_name` and
`camera.type` come from the YAML.

The whole set:

```bash
bash run_all_figures.sh                  # -> outputs/figures/strobe
OUT_DIR=check bash run_all_figures.sh    # scratch dir, keep outputs/figures clean
```

A single recipe takes ~15 s (3 sources x 300 frames at 1280x720).

## 5. Recipe index (`configs/strobe/`)

Each frozen YAML *is* a figure recipe: base config + overlays + the exact
command line, flattened. Reproducible here:

| recipe | clip replayed | output prefix | ghost steps | camera |
|---|---|---|---|---|
| `strobe/g1/headstand_strobe.yaml` | `outputs/g1/trajectories/headstand_4` | `g1_headstand` | 1,8,…,43 | d 3.5, az +90 |
| `strobe/g1/move-ego-0-4_strobe.yaml` | `outputs/g1/trajectories/move-ego-0-4_8` | `g1_move-ego-0-4` | 15,20,…,55 | d 5, az +90 |
| `strobe/heavy_arm/move-ego-0-2_strobe.yaml` | `outputs/heavy_arm/trajectories/move-ego-0-2_4` | `heavy_arm_move-ego-0-2` | 100,115,…,175 | d 8.5, az +90 |
| `strobe/heavy_arm/rotate-y-5-0.8_strobe.yaml` | `outputs/heavy_arm/trajectories/rotate-y-5-0.8_7` | `heavy_arm_rotate-y-5-0.8` | 150,157,…,192 | d 3.5, az +90 |
| `strobe/limit_leg/move-ego-0-2_strobe.yaml` | `outputs/limit_leg/trajectories/move-ego-0-2_2` | `limit_leg_move-ego-0-2` | 37,44,…,72 | d 5, az +80 |
| `strobe/limit_leg/rotate-y-5-0.8_strobe.yaml` | `outputs/limit_leg/trajectories/rotate-y-5-0.8_7` | `limit_leg_rotate-y-5-0.8` | 1,8,…,43 | d 3.5, az +90 |

`run_all_figures.sh` runs exactly these six.

### Rollout-only recipes (need the full repository)

These published figures came from `render_strobe.py` driving the frozen
MetaMotivo actor on the *child* body; the latents, `data/origin_motion` clips and
humenv task models are not in this package, so the YAMLs are shipped as
documentation only:

- `strobe/jump-2_strobe.yaml` → `jump-2_orbit_auto_{adult,z0,zstar}_strobe.png`
  (z0 = `data/origin_z/jump-2/jump-2_2.npy`, z* = `outputs/my_search/jump-2_2/best_z.npy`)
- `strobe/move-ego-0-4_strobe.yaml` → `move-ego-0-4_orbit_auto_*`
- the child-body figures (`headstand_orbit_auto_*`, `crawl-0.4-2-u_0_s0_orbit_auto_*`)

In the full repo these re-render with e.g.

```bash
MUJOCO_GL=egl python render_strobe.py --config configs/strobe/jump-2_strobe.yaml
```

### Stale recipe — do not replay

`strobe/headstand_strobe.yaml` has `task_name: headstand` but a
`replay.clip` pointing at `heavy_arm/trajectories/move-ego-0-2_4`: it was
re-pointed later for a different figure. Replaying it would write
`headstand_orbit_auto_*_strobe.png` files that are **not** the paper's headstand
figure. `run_all_figures.sh` skips it. Use
`strobe/heavy_arm/move-ego-0-2_strobe.yaml` for that clip.

## 6. Fidelity (measured, 2026-11-09)

All six recipes were re-rendered from inside this package and diffed against the
published PNGs (1280x720x3):

- 18/18 figures match; **4 bit-identical** (max channel delta 0)
- the other 14 differ by **at most 1/255 on ≤96 of 921,600 pixels** (0.01 %)
- re-running the *same* command twice differs by the same ±1 LSB on 2–6 pixels,
  so this is GPU antialiasing nondeterminism under EGL, not config drift

So: same config + same MuJoCo version ⇒ the paper figure. Expect pixel-identical
or one-level-off on antialiased edges; anything larger means a version, a knob,
or a model file drifted. Check it yourself with:

```python
import numpy as np, imageio.v3 as iio
a = iio.imread("outputs/figures/strobe/g1_headstand_orbit_auto_z0_strobe.png").astype(int)
b = iio.imread("published/g1_headstand_orbit_auto_z0_strobe.png").astype(int)
print(np.abs(a - b).max(), (np.abs(a - b) > 0).any(-1).sum())   # -> 0 or 1, few pixels
```

The MP4s are *not* byte-reproducible (video encoding), but the frames are the
same renders and the same 30 fps.

## 7. What the knobs mean

Everything lives in `configs/render_strobe.yaml` (read it — each key is
annotated). The knobs that actually change a figure's look:

```yaml
task_name: heavy_arm_move-ego-0-2      # output filename prefix
camera:
  type: orbit_auto                     # orbit_auto | tracking | fixed
  distance: 8.5                        # metres from the look-at point
  elevation: 0.0                       # deg, 0 = eye level with the root
  azimuth_offset: 90.0                 # 0 behind the path, 90 side-on, 180 facing it
  lookat_height: 0.6                   # look-at raised above the root (m)
  plan_window: strobe                  # "strobe" aims the camera at the captured window
                                       #   only; "full" uses the whole clip (a long clip
                                       #   then drags the poses to the frame edge)
strobe:
  start_step: 100                      # first captured frame (inclusive)
  end_step: 175                        # last captured frame; null = end of each source
  interval: 15                         # steps between ghost poses (the ladder always
                                       #   ends on end_step)
  target_frames: null                  # explicit step list, overrides interval
  shift_px: -60                        # horizontal spacing between ghost poses
  x_offset_px: 0                       # slide the whole trail (0 = trail centred on the
                                       #   camera axis; +right / -left), spacing untouched
  threshold: 5.0                       # foreground mask: summed |frame - background| > this
  start_color: [1.0, 0.88, 0.66, 0.7]  # RGBA at start_step (fades to final_color)
  final_color: [0.627, 0.47, 0.318, 1.0]
  middle_color: null                   # optional third gradient stop
  crop_floor: false                    # whiten the bottom crop_floor_frac of the plate
render:
  width: 1280
  height: 720
  fps: 30
  white_background: false              # unlit white stage (no shadows / sky / floor texture)
  burn_in_frame_index: true            # step badge on VIDEO frames only; green = captured
  out_dir: outputs/figures/strobe
replay:
  clip: outputs/limit_leg/trajectories/move-ego-0-4_8   # used by replay_strobe.py
```

How a ghost figure is built: render the clip frame by frame
(`data.qpos = qpos[t]; mj_forward`), diff each captured frame against a
robot-hidden background to get a mask, shift ghost *k* by `x_offset_px -
((n-1)*shift_px)//2 + k*shift_px` columns, and stamp it on a white plate. The
RGBA of the robot itself fades from `start_color` to `final_color` across the
window, which is why the trail reads as a temporal gradient.

## 8. Tuning a figure you are authoring

1. Render the comparison video with the burn-in on (`render.burn_in_frame_index:
   true`, the default). Each panel is badged with its own step index and the
   badge turns **green on the steps that become ghost poses** — so
   `start_step` / `end_step` / `interval` are readable straight off the footage.
2. Sweep from the CLI (named flags beat the YAML, `--set` beats the flags):

   ```bash
   MUJOCO_GL=egl python replay_strobe.py \
       --config configs/strobe/g1/headstand_strobe.yaml \
       --start-step 12 --end-step 44 --interval 4 \
       --shift-px 90 --x-offset-px 20 --distance 3.0 \
       --out-dir scratch
   ```
   Generic overrides work for anything without a flag:
   `--set camera.azimuth_offset=45 --set strobe.middle_color='[0.5,0.5,0.5,0.6]'`.
3. Freeze the recipe once it looks right, then re-render from that file alone:

   ```bash
   MUJOCO_GL=egl python replay_strobe.py \
       --set replay.clip=outputs/heavy_arm/trajectories/headstand_4 \
       --set task_name=heavy_arm_headstand \
       --set strobe.start_step=20 --set strobe.end_step=60 --set strobe.interval=8 \
       --set strobe.shift_px=90 --distance 3.5 \
       --save-config configs/strobe/heavy_arm/headstand_strobe.yaml
   MUJOCO_GL=egl python replay_strobe.py --config configs/strobe/heavy_arm/headstand_strobe.yaml
   ```

   ⚠️ `--save-config` runs *before* the positional clip argument and `--task-name`
   are folded in, so those two never land in the saved YAML. Pass them as
   `--set replay.clip=…` / `--set task_name=…` (shown above, and what the frozen
   recipes do) or edit the saved file afterwards. Otherwise the frozen recipe
   silently re-renders the base config's clip.
4. `--config A --config B` layers configs (last file wins), which is how the
   per-body overlays are built from the base config.

## 9. Any clip, any panels

Available clips (11 per body, `fps` 30; `nq` 76 for the humanoid variants, 36 for
G1): `crawl-0.4-2-d_8`, `crouch-0_8`, `headstand_4`, `jump-2_6`, `jump-2_7`,
`move-ego-0-2_{2,4}`, `move-ego-0-4_8`, `rotate-x--5-0.8_4`, `rotate-x-5-0.8_9`,
`rotate-y-5-0.8_7`, `rotate-z--5-0.8_6` under `outputs/{g1,heavy_arm,limit_leg}/trajectories/`.

```bash
# panel order and labels are yours; `label=file.npz` overrides the name convention
MUJOCO_GL=egl python replay_strobe.py outputs/heavy_arm/trajectories/jump-2_6 \
    --sources adult,body,z0,zstar --task-name heavy_arm_jump-2 --burn-in

# only the searched latent vs the adult, with an explicit model per source
MUJOCO_GL=egl python replay_strobe.py outputs/limit_leg/trajectories/crouch-0_8 \
    --sources "adult, zstar=best.npz"
```

Per-source model resolution (first hit wins): `replay.sources[].xml` → the
`model` recorded for that trajectory in `video.json` → `models/<adult|body>.xml`
→ the config's `xml_path` / `baseline.xml` (with a warning). `--models-root`
overrides where `models/` is.

`--video-camera` re-aims the camera from the framing recorded in `video.json`
(its absolute azimuth is converted to our heading-relative `azimuth_offset`),
useful when a strobe figure has to match a survey video's viewpoint.

To use a different file layout entirely, list the sources in the YAML:

```yaml
replay:
  clip: outputs/heavy_arm/trajectories/jump-2_6
  sources:
    - { label: "adult", qpos: "reference_adult.npz", xml: "models/adult.xml" }
    - { label: "z0",    qpos: "z0.npz" }
    - { label: "zstar", qpos: "best.npz",  xml: "models/heavy_arm.xml" }
```

## 10. Troubleshooting

- **`GLFW` / `EGL` init failure** → `export MUJOCO_GL=egl` (needs libEGL +
  drivers) or `export MUJOCO_GL=osmesa` (CPU, `libosmesa6`).
- **`trajectory has nq=76 but the model has nq=36`** → wrong model for that
  dump: G1 dumps go with `models/g1/scene.xml`, humanoid dumps with the body XML.
- **`warning: N poses ... wider than the 1280px plate`** → the trail does not
  fit and ghosts are clipped at the edge. The published G1 figures were rendered
  with this warning on, so seeing it is expected; to remove it, lower `shift_px`,
  cut poses, or raise `render.width`.
- **`x_offset_px=0 does not fit ... using x_offset_px=<n>`** → the centred layout
  was clamped to the nearest feasible offset. Same fix.
- **burn-in badge font falls back** → the renderer tries `DejaVuSans-Bold.ttf`
  and falls back to Pillow's built-in font; only the badge changes, the figure
  itself is unaffected (PNGs never carry the badge).
- **`video: panels differ in length ...`** → sources of different lengths are
  truncated to the shortest panel; the burn-in still shows each panel's own step.
- **`no MuJoCo model found for ...`** → the clip's `video.json` points at a model
  that is not under `models/`; pass `--models-root` or set `sources[].xml`.

## 11. Provenance

`replay_README.md` + `replay.py` document where the clip folders come from (the
user-survey video package: same trajectories, side-by-side and overlay videos).
Not needed for the figures, but it explains `video.json` and what
`reference_adult` / `reference_body` / `z0` / `best` are.

Code and configs in this package are byte-identical copies of the working tree
that produced the published figures (nothing was edited for packaging), so a
difference you see is a version or environment difference, not a packaging
difference.
