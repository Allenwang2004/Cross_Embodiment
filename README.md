# Cross-embodiment latent adaptation for a frozen behavior foundation model

Meta Motivo M-1 (FB-CPR) is a humanoid behavior foundation model trained on an **adult** SMPL body: a frozen policy
π(a | s, z) is steered by a 256-d latent z (|z| = 16). This repository asks how far the latent alone can carry the policy to
**other bodies**: a child (root scale 0.611), scaled bodies (length x width, e.g. the tall thin len1.4_wid0.8), adult-size
variants (a heavy hand = payload, a locked bent leg), changed joint gains, and robots (Unitree G1, H1, Apollo) driven through an
SMPL <-> robot map. The policy's weights never change.

Our method, **LOES**, searches one latent per motion and body with ES + Adam on Motivo's sphere, scored by a cost in Motivo's own
representation (1 - cos of the backward-map embeddings of rollout and reference) plus heading and position terms. It is compared
with the frozen task latent z0, random search, CMA-ES, BFM-Zero's CEM on PHC's tracking reward, BFMTrack's latent-sequence
optimisation (LSO, LSOR) and BFM-Zero's Trajectory Adaptation, on 52 task motions and 68 AMASS locomotion clips.

The pipeline (search -> re-roll -> analyse -> tables and figures): [`docs/workflow.md`](docs/workflow.md). Modules, classes,
data layout, bit-identity rules and the naming table: [`docs/architecture.md`](docs/architecture.md). Adding a body:
[`docs/new_body.md`](docs/new_body.md) (in Chinese).

## Setup

```bash
uv sync                                       # Python >= 3.11; metamotivo from its git repository (pyproject.toml)
git submodule update --init                   # assets/robots/menagerie, assets/robots/limx_humanoid (the robots' meshes)
export CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
```

| Piece | Where |
|---|---|
| Frozen policy and backward map B | `facebook/metamotivo-M-1` (Hugging Face; `HF_HUB_OFFLINE=1` once it is cached) |
| Adult motions and their task latents | `data/origin_motion/<task>/<stem>.npz` (30 fps qpos), `data/origin_z/<task>/<stem>.npy` (z0); AMASS: `amass_data/` |
| Bodies | `assets/robots/<body>/robot.xml` (shape), `assets/robots_torque/<body>/robot_torque_full.xml` (torque-calibrated physics), `data/<body>/retargeting_motion/` (references); registry `configs/bodies.yaml`; robots `model/robots.py` |
| Clip lists, figure configs | `configs/clips/*.txt` (the 52 task motions, the 68 AMASS clips, the survey's), `configs/figures/*.yaml`, `configs/survey.yaml` |
| Blender renders | `scripts/render_blender.py` runs inside Blender (`bpy`), started by the *_blender.py scripts |

`data/`, `amass_data/`, `outputs/` are not in git. Runs share one machine: check `free -g` first, keep >= 50 GB available, use GPU 2.

## Layout

```
crossembodiment/      the library (docs/architecture.md)
  paths.py            every location in the repository; the outputs / results / figures roots ($CE_OUTPUTS_ROOT, ...)
  bodies.py           BodyRegistry (configs/bodies.yaml): bodies, their XMLs and references, the root scale s; RobotSpec
  methods.py          MethodSpec registry: canonical method keys, flags, settings, the run-directory key per experiment
  geometry.py         latent sphere (project_z), ranks, quaternions, heading / yaw
  metrics.py          the evaluation metrics as named functions (Task page and legacy), MetricSuite
  eval_metrics.py     MPJPE / MMPJPE / MPJAE (per-joint acceleration error, m/s^2), EMD, DTW; Body; the eval_metrics CLI's sets
  align_cost.py       L_align / L_phys (compute_batch_cost)
  retiming.py         a clip resampled to another speed
  policy.py           MotivoPolicy: the frozen FBcprModel (latents, actions, B embeddings)
  obs.py              what the actor sees: RawObs, MultiplierObs, ExactObs (adult-equivalent observation)
  sim.py              HumenvSim (humenv's vector env), RobotSim (RobotVecEnv)
  rollout.py          BatchRollout: one latent or one latent sequence per slot
  robot_rollout.py    the robot rollouts' actor call and motion summary
  costs.py            CostTerm objects (BFM similarity, heading, position, anchor, IK, MSE, PHC, DTW, fall) and CompositeCost
  optim.py            ES, CEM, CMAES, RandomSearch, LSO, TrajectoryAdaptation, Adam (ask / tell)
  problems.py         the four searches' problems (human / robot x single latent / sequence)
  search.py           SearchLoop (generations), SequenceSearchLoop (iterations)
  records.py          RunRecorder: every file a search run writes
  experiment.py       ClipSet, RunSpec, Experiment: an experiment's grid, run folders and command lines
  reroll.py           Reroller, RerollIndex, EvalRollouts: re-rolling the latents of finished searches
  results.py          ResultsTable, Checks, ReportWriter: per_run.npz / results.json / means.json / README.md
  queue.py            CommandQueue + ResourceGate (scripts/run_queue.py)
  bodybuild/          BodyBuilder subclasses, Retargeter, TorqueCalibrator, GainScaler, BodyPipeline (scripts/build_body.py)
  palette.py          every method's label and colour (figures, videos, Blender)
  figstyle.py         matplotlib styles, the paper's drawing canvas
  camera.py           ViewSpec: cameras as data, the world-to-pixel projection
  clips.py            ClipBundle: the trajectories of one clip a figure or video draws
  blender.py          BlenderScene: scene.json + meshes.npz for render_blender.py
  strobe.py           StrobeLayout: strobe frames, tints, shifts, headings
  video.py            VideoRenderer: recorded qpos replayed into videos
  sheet.py            ContactSheet: overview pages of rendered panels
  draw.py             small PIL / camera drawing helpers
  survey.py           the user survey's clips, bodies and runs
  multibody_study.py  the layout of the 10/10 multi-body correction experiment
model/                physics helpers: exact_obs, obs_scale, bfm_align, losses, kinematics, dtw, phc_reward, friction, robots,
                      robot_map (RobotMap), robot_env (RobotEnv, RobotVecEnv); legacy adapter training: simple/, networks, dataset
scripts/              the CLIs: searches, experiment drivers, body building, queue, figures, videos (docs/workflow.md)
configs/              bodies.yaml, clip lists, figure and survey configs
tests/                unit / regression tests (tests/test_*.py) and the golden harness (tests/golden/)
archive/              files nothing uses any more, with their history (archive/README.md says why each was archived)
docs/                 workflow.md, architecture.md, new_body.md, refactor/PLAN.md, figures/, reports of earlier phases
```

## Main CLIs

```bash
# LOES on one task motion of the child (the Task page's settings: 64 generations x 16 rollouts)
uv run python scripts/single_z_search.py --clip move-ego-0-2/move-ego-0-2_1 --body child \
    --xml assets/robots_torque/child/robot_torque_full.xml --objective bfm --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 \
    --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/move-ego-0-2/move-ego-0-2_1.npy \
    --anchor-weight 0 --eval-every 0 --init-vel-scale 1 --seed 0 --out outputs/example/loes
# the same on the G1
uv run python scripts/robot_z_search.py --robot g1 --clip move-ego-0-2/move-ego-0-2_1 --fill ridge \
    --gains assets/robots/g1/gains_perjoint.json --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 --sampling independent \
    --init-vel reference --eval-every 0 --algo es --heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0 --ik-weight 1.0 \
    --seed 0 --out outputs/example/g1_loes
# LSO (BFMTrack), a latent per step; --rank for LSOR, --method ta --objective phc for Trajectory Adaptation
uv run python scripts/lso_search.py --ref data/child/retargeting_motion/move-ego-0-2/move-ego-0-2_1.npz --body child \
    --scale 0.611036 --n 16 --iters 64 --init-vel-scale 1 --trace budgets --seed 0 --out outputs/example/lso
uv run python scripts/robot_lso_search.py --robot g1 --ref data/origin_motion/move-ego-0-2/move-ego-0-2_1.npz --fill ridge \
    --gains assets/robots/g1/gains_perjoint.json --n 16 --iters 64 --init-vel reference --trace budgets \
    --heading-weight 0 --pos-weight 0 --ik-weight 0 --seed 0 --out outputs/example/g1_lso     # its defaults add heading / position / IK
# a new body (docs/new_body.md)
uv run python scripts/build_body.py scaled len0.8_wid1.2 --len 0.8 --wid 1.2
# an experiment: print its runs, run them under the resource gate, re-roll, analyse
uv run python scripts/body_52_test.py commands > outputs/body_52/commands.txt
uv run python scripts/run_queue.py outputs/body_52/commands.txt --log outputs/body_52/queue.log --sentinel outputs/body_52/searches.done
uv run python scripts/body_52_test.py prepare && uv run python scripts/body_52_test.py evalcmds > outputs/body_52/evalcmds.txt
uv run python scripts/run_queue.py outputs/body_52/evalcmds.txt --done-file eval_qpos.npz
uv run python scripts/body_52_test.py analyse
```

The experiment drivers (`child_52_rerun.py`, `child_52_cem_phc.py`, `child_52_lso.py`, `loes_noanchor.py`, `body_52_test.py`,
`g1_small_test.py`, `tall_body_52.py`, `amass_baselines.py`, `kpkd_test.py` and the follow-up studies) each print their own
command lines (`commands`, `evalcmds`), re-roll (`prepare`) and `analyse`; docs/workflow.md lists them. The methods' flags are
in `crossembodiment/methods.py` (`METHODS[key].flags`, `COMMON`).

## Where results go

* A search run: `outputs/<experiment>/[<body>/]<stem>_<run key>_s<seed>/` (summary.json, curve.csv, best_z.npy, best.npz,
  origin_z.npz = z0's rollout, z_trace.npz; latent sequences: mu_trace.npz, rollouts.npz). The run keys are the old ones
  (`es`, `cem`, `lsor`, ...): `crossembodiment.methods.method_of(experiment, key)` says which method each is there.
* Re-rolls: `outputs/<experiment>/[<body>/]rollouts/<stem>[_<key>]/` (z.npy, index.json, eval.csv, eval_qpos.npz).
* Analyses: `outputs/<experiment>/{per_run.npz, results.json, means.json, README.md}`; with `CE_OUTPUTS_ROOT` / `CE_RESULTS_ROOT`
  set, read from one tree and written into another (inside the repository).
* Figures: `docs/figures/` (`CE_FIGURES_ROOT`); videos and Blender renders next to their experiment (e.g. `outputs/strobe_task52_blender`).

## Golden check

The project relies on exact re-rolls, so code changes must leave the searches bit-identical:

```bash
tests/golden/check.sh <label>       # 22 small real runs + the metrics of fixed rollouts + the --resume case vs outputs/refactor_golden/baseline
for t in tests/test_*.py; do uv run python $t; done
```

`check.sh` exits 0 only if everything is identical (tests/golden/README.md: what is compared, what is ignored, the resource gate).
The refactor's plan and its rules: docs/refactor/PLAN.md.
