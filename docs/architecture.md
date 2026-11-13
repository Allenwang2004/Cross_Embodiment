# Architecture

How the code is organised after the OOP refactor of 2026-11-11 (docs/refactor/PLAN.md, steps 0-11): the modules and what each
class is for, where data and results live, what must never change a bit, and the one name each concept has in the code (with
the old names that still read). The pipeline itself (search, re-roll, analyse, tables and figures) is in [workflow.md](workflow.md).

## Modules

```
scripts/                       CLIs only: argparse + main(); every number is computed in the package
  searches                     single_z_search.py  robot_z_search.py  lso_search.py  robot_lso_search.py
  experiment drivers           child_52_rerun / child_52_cem_phc / child_52_lso, body_52_test, g1_small_test, loes_noanchor,
                               tall_body_52, amass_baselines, kpkd_test, loes_lso_4096, loes_weight_test, lsor_mpjae_test,
                               cost_correlation, drift_success_child, success_ / fall_definitions   (commands | prepare | evalcmds | analyse)
  bodies, queue                build_body.py (+ thin wrappers), run_queue.py
  figures, videos              plot_*.py, render_panels.py, body_overlay.py, robot_overlay.py, *_blender.py, render_blender.py (standalone)
        |
        v
crossembodiment/               the library (importing it imports nothing heavy; torch / mujoco / humenv load with the modules that need them)
  data            paths ........ every location (REPO from the package's own path), $CE_OUTPUTS_ROOT / $CE_RESULTS_ROOT / $CE_FIGURES_ROOT
                  bodies ....... BodyRegistry over configs/bodies.yaml (+ RobotSpec over model/robots.py)
                  methods ...... MethodSpec registry: canonical method keys, flags, settings, run-directory keys per experiment
  pure functions  geometry ..... latent sphere (project_z), ranks, quaternions, heading / yaw
                  metrics ...... Task-page and legacy metric definitions, MetricSuite;  eval_metrics: MPJPE / MMPJPE / MPJAE, Body
                  align_cost ... L_align / L_phys (compute_batch_cost);  retiming: resample a clip
  simulation      policy ....... MotivoPolicy  ->  obs: ObsMapper (Raw / Multiplier / Exact)  ->  sim: HumenvSim / RobotSim
                  rollout ...... BatchRollout: n slots x T steps, one latent or one latent sequence per slot
  search          costs ........ CostTerm objects, CompositeCost -> (cost, parts)
                  optim ........ ES, CEM, CMAES, RandomSearch, LSO, TrajectoryAdaptation, Adam (ask / tell, own RNG)
                  problems ..... the four searches: Human/Robot x Latent/Sequence Problem (score, baseline, start, save)
                  search ....... SearchLoop (generations), SequenceSearchLoop (iterations)
                  records ...... RunRecorder: every output file of a run, written and read back
  experiments     experiment ... ClipSet, RunSpec, Experiment (grid, run folders, "dir|cmd" lines)
                  reroll ....... Reroller, RerollIndex, EvalRollouts (re-roll the best / centre latents of finished searches)
                  results ...... ResultsTable, Checks, ReportWriter (per_run.npz, results.json, means.json, README.md)
                  queue ........ CommandQueue + ResourceGate (scripts/run_queue.py)
  body pipeline   bodybuild/ ... BodyBuilder subclasses, Retargeter subclasses, TorqueCalibrator, GainScaler, BodyPipeline
  visualisation   palette, figstyle, camera (ViewSpec), clips (ClipBundle), blender (BlenderScene), strobe (StrobeLayout),
                  video (VideoRenderer), sheet (ContactSheet), draw, survey
        |
        v
model/                         physics-side helpers the package imports (kept in place; their names are imported by many scripts)
  exact_obs (ExactObs: adult-equivalent observation)  obs_scale (the old multiplier)  bfm_align (B embeddings)  losses / kinematics
  dtw  phc_reward (PHCCost)  friction (FloorFrictionWrapper)  robots (ROBOTS data)  robot_map (RobotMap)  robot_env (RobotEnv, RobotVecEnv)
  legacy (the adapter-training line): simple/ (train_es.py, train.py, config.py), networks, dataset, bodies (BodyRoster) -- still
  imported by dump_heldout_clips.py, test_exact_obs.py, write_body_splits.py, strength_rollout.py (bodies.up_z)
```

Dependencies point down: scripts import the package, the package imports model/, nothing in model/ imports the package. Inside the
package, a search is `problems` (built from a script's parsed arguments) feeding `search`, which calls `optim` and `costs`
through the problem and writes through `records`; `problems` builds `policy`, `obs`, `sim` and `rollout` in the order the
original scripts did (part of bit-identity: robot workers spawn before CUDA is touched).

## Classes

| Module | Class | Responsibility |
|---|---|---|
| bodies | `BodySpec` | one body: kind, shape / physics XML, references per clip kind, root scale rule, asset folder (`label`) |
| bodies | `BodyRegistry` (`BODIES`) | bodies by name or alias; `scale(body, clip)`, `reference(body, clip)`, `robot(body)` |
| bodies | `RobotSpec` | model/robots.py `ROBOTS[name]` as frozen data |
| methods | `MethodSpec` (`METHODS`) | a method's canonical key, display name, script + flags, identifying summary settings, `run_key` |
| metrics | `MetricSuite`, `Rollout` | named metrics, in order, of many rollouts against one clip's reference on one body |
| eval_metrics | `Body` | a body's MuJoCo model for the metrics (world / local positions, part-floor distances) |
| policy | `MotivoPolicy` | the frozen FBcprModel: `latents()` (float64 -> float32), `act()`, `embed()` (B) |
| obs | `RawObs`, `MultiplierObs`, `ExactObs` | what the actor and B see of a simulated body (strategy; ExactObs in the env workers) |
| sim | `HumenvSim`, `RobotSim`, `SimStep` | reset / step of n bodies: humenv's async vector env, or RobotVecEnv |
| rollout | `BatchRollout`, `Rollout` | one batch rollout of fixed slots; the batch compositions (`copies`, `padded`, `with_mean`) |
| costs | `ScoreContext`, `CostTerm`, `CompositeCost` | one scored batch; a cost term (batch / per-step form); main + weighted terms -> (cost, parts) |
| costs | `BFMSimilarity`, `Heading`, `Position`, `Anchor`, `IKResidual`, `TrackingMSE`, `RobotMSE`, `RobotPoints`, `PHCReward`, `SlowDTW`, `DTW012`, `DTWUnfinished`, `AlignedHeading`, `Fall`, `AlignPhys`, `AsType` | the individual terms (costs.py's table) |
| optim | `ES`, `CEM`, `CMAES`, `RandomSearch` | single-latent optimisers (ask / tell); ES `state_dict()` for `--resume` |
| optim | `LSO`, `TrajectoryAdaptation`, `Adam` | latent-sequence optimisers; Adam's moments |
| problems | `HumanLatentProblem`, `RobotLatentProblem`, `HumanSequenceProblem`, `RobotSequenceProblem`, `Scored` | a search's body, references, mapper, simulator, policy, rollout and cost; score / baseline / start / curve rows / save |
| search | `SearchLoop`, `SequenceSearchLoop` | the generation loop (best so far, mean-z rollouts, z trace, resume) / the iteration loop (budgets, kept rollouts) |
| records | `RunRecorder` | best_z.npy, best.npz, origin_z.npz, z_trace.npz, curve.csv, summary.json, eval.csv, state.npz, mu_trace.npz, rollouts.npz |
| experiment | `ClipSet`, `RunSpec`, `Experiment` | an experiment's clips, grid points, run folders, command lines; `method_spec(key)` / `key(method)` |
| reroll | `Reroller`, `RerollIndex`, `EvalRollouts` | which latents of finished searches are re-rolled (z.npy + index.json, schema 2), reading every index layout, an --eval-z folder |
| results | `ResultsTable`, `Checks`, `ReportWriter` | motions x seeds x budgets x metrics with the drivers' reductions and paired Wilcoxon; consistency checks; README.md |
| queue | `CommandQueue`, `ResourceGate`, `Job` | run "dir\|cmd" lines under the machine's memory / CPU thresholds (summary.json = done) |
| bodybuild | `BodyBuilder`: `ScaledBody`, `SpecBody`, `HeavyLimb`, `LockJoints`, `LimitJoints`, `Elderly`, `ActuatorStrength` | one new body's MJCF + parameter.json |
| bodybuild | `Retargeter`: `SameTopologyRetargeter`, `CopyReference`, `LockedJointReference` | a new body's reference motions |
| bodybuild | `TorqueCalibrator`, `GainScaler`, `BodyPipeline`, `BuildRoots` | torque ratio -> robot_torque_full.xml; kp / kd scaling; the idempotent build steps; where a build writes |
| palette | `Palette`, `MethodColour` | every method's label and colour (hex, video RGBA, Blender sRGB), script variants kept apart |
| figstyle | `FigureStyle` | matplotlib styles (paper / notebook), the 0.01-inch canvas, save to pdf / svg / png |
| camera | `ViewSpec` | a camera as data (MuJoCo, tracking, Blender), the one world-to-pixel projection |
| clips | `ClipBundle` | the trajectories of one clip a figure or video draws (survey format) |
| blender | `BlenderScene` | scene.json + meshes.npz for scripts/render_blender.py |
| strobe | `StrobeLayout` | strobe frames, tints, shifts, headings |
| video | `VideoRenderer` | recorded qpos replayed into side-by-side or overlay videos |
| sheet | `ContactSheet` | overview pages of rendered panels |
| model/exact_obs | `ExactObs`, `ExactObsWrapper` | the adult-equivalent observation of a body state (positions / scale), in the env workers; option `clearance_vel` (default off): + d/dt of the ground-clearance height correction in the root's vertical velocity (single_z_search `--exact-clearance-vel`) |
| model/robot_map | `RobotMap`, `Lowest` | SMPL <-> robot through per-segment orientations (IK), the robot's root scale |
| model/robot_env | `RobotEnv`, `RobotVecEnv` | a robot driven by Motivo's SMPL actions; n of them in worker processes |
| model/phc_reward | `PHCCost` | PHC's tracking reward (BFM-Zero's) as a cost |

## Data layout

```
assets/robots/<label>/            robot.xml (shape; the adult-size variants simulate it directly), parameter.json, skeleton.json
                                  g1/, h1/, apollo/: the robots' MJCF, gains_perjoint.json, smpl_fill.json (menagerie, limx_humanoid: submodules)
assets/robots_torque/<label>/     robot_torque_full.xml: a scaled body's physics (actuators / joint dynamics x its torque ratio)
configs/bodies.yaml               the body registry (keys s / s_rule / s_default = the root scale)
configs/clips/*.txt               clip lists: child_52_task (the 52 task motions), amass_locomotion_5s, user_survey
configs/figures/*.yaml            strobe examples, teaser and method-figure panels;  configs/survey.yaml: the user survey
data/origin_motion/<task>/<stem>.npz        the adult's task clips (qpos, 30 fps)
data/origin_z/<task>/<stem>.npy             Motivo's task latent z0 per task clip
data/<body>/retargeting_motion/<task>/<stem>.npz            a body's own reference;  data/<body>/amass_retargeting_motion/ACCAD/<clip>.npz
amass_data/amass_origin_motion/ACCAD/<clip>.npz             the adult's AMASS clips;  amass_retargeting_motion/ACCAD: the child's
outputs/<robot>/roundtrip/<stem>/{g1.npz | robot.npz}       a task clip mapped onto a robot (its target)
outputs/<experiment>/[<body>/]<stem>_<run key>_s<seed>/     one search run (records.py: best_z.npy, best.npz, origin_z.npz, z_trace.npz,
                                                            curve.csv, summary.json; LSO: mu_trace.npz, rollouts.npz; robots: state.npz)
outputs/<experiment>/[<body>/]rollouts/<stem>[_<key>]/      re-rolls: z.npy, index.json (schema 2), eval.csv, eval_base.json, eval_qpos.npz
outputs/<experiment>/{per_run.npz, results.json, means.json, README.md}      an analysis (written under $CE_RESULTS_ROOT if set)
outputs/refactor_golden/<label>/                            the golden regression runs (tests/golden)
docs/figures/                     the paper figures ($CE_FIGURES_ROOT when set) and their inputs (blender_renders/, *_parts/)
```

In the refactor worktree `data`, `amass_data`, `.venv`, `assets/robots/menagerie`, `assets/robots/limx_humanoid` and some
`outputs/*` folders are symlinks into the main tree (read only).

## Bit-identity constraints (PLAN.md rule 2)

The project relies on exact re-rolls (a 1e-15 change moved a G1 single-rollout cost by 28 %), so a refactor or a rename must not
change a bit; `tests/golden/check.sh <label>` compares 22 golden runs, the metrics file and the `--resume` case against the
baseline made from the snapshot code. Never change:

* batch composition: 2 x pairs slots, padding with z0 (`--eval-z`), 16 copies for mean-z rollouts, LSO's n + 1 slots with slot 0 = the mean;
* RNG draw order and shapes: one `default_rng(seed)` per optimiser, one draw per step; pycma seeded with seed + 1; colored noise draws
  sr then si;
* arithmetic order: candidates `[z + sigma eps, z - sigma eps]` then projected; leave-one-out baselines; Adam with gen + 1; ranks by
  double argsort; the first argmin with strict `<`; float64 z cast to float32 tensors; float32 observations;
* `max_episode_steps`, ExactObs computed inside the env workers, RobotMap's `assume_valid` rotations and the G1 "seg" calibration
  order, the IK warm start;
* the output schema: summary.json keys, curve.csv columns, z_trace with gen -1, state.npz, npz key `qpos`, rollouts.npz keys, file
  names, run-directory keys, and the driver docstrings that analyses copy into README.md / topview.md.

## Names (step 11)

One concept, one name in the code. On disk nothing was renamed: folders, run-directory keys, file names, JSON / npz / CSV keys and
CLI flags keep their old spelling, and every reader still reads them.

| Concept | In the code | Old names (still readable) | On disk (unchanged) |
|---|---|---|---|
| LOES, no anchor (the method since 11/11) | `loes` | `es` | run key `es` in outputs/noanchor/<body>, tall_body_52; `--algo es`; summary `algo: "es"` |
| LOES + 0.3 anchor (Task page before 11/11) | `loes_anchor` | `es` | run key `es` in child_52_rerun, body_52, g1_52, g1_small, walk_bodies, loes_lso_4096, kpkd ("") |
| CEM with BFM-Zero's PHC reward | `cem_phc` | palette `cem` | run key `cem` (child_52_cem_phc, body_52, g1_*, tall_body_52); result key `cem_phc` in fall_ / success_definitions |
| CEM with our cost | `cem_ours` | palette `("cem", "algo_compare")` | run key `cem` in child_52_rerun only |
| LSO + ranks, BFMTrack's cost | `lsor_bfm` | `lsor` | run key `lsor` in child_52_lso |
| LSO + ranks, 0.05 heading + 0.005 position | `lsor_w005` | `lsor`, palette `lsor` | run key `lsor` (body_52, g1_*, amass_baselines), `lsor_w005` (child_52_lso) |
| LSO + ranks, our cost x 1 | `lsor_ours` | | run key `lsor_ours` (child_52_lso) |
| LSO, TA, random search, CMA-ES | `lso`, `ta`, `random`, `cmaes` | | the same keys |
| Motivo's zero-shot tracking | `native` | | rollouts.npz `q_ER8` of a latent-sequence run |
| the frozen latent | method `frozen`; the latent `z0` | `z0` (methods.ALIASES, palette), `origin_z` | data/origin_z/, `origin_z.npz`, summary key `origin_z`, `improvement_vs_origin_z`, curve `z_start` / best `source` value `origin_z` (`paths.task_z0`, `Z0_DIRNAME`, `Z0_ROLLOUT`, `Z0_KEY`) |
| root scale (body / adult size) | `scale` (`BODIES.scale()`, `BodySpec.scale`, `ExactObs.scale`, `RobotMap.scale`, `MetricSuite.scale`, `Position.scale`); `s` only inside formulas | `.s`, `BodySpec.s_rule / s_default`, `obs.CHILD_EXACT_S`, `Position.pos_div` | configs/bodies.yaml `s`, `s_rule`, `s_default`; `lso_search.py --scale`; summary `pos_div` |
| what positions are divided by | problems' `pos_scale` (= scale with `--pos-unit adult`, else 1) | `pos_div` | summary `pos_div` |
| embodiment | `body` (registry name), `robot` only in robot-specific code, `label` = the asset folder (= `--body` / `--robot`) | | assets/robots[_torque]/<label> |
| RobotMap, robot side | `m_robot`, `d_robot`, `robot_jnt`, `robot_jnt_a`, `robot_qadr`, `robot_dadr`, `robot_range`, `robot_bid`, `low_robot`, `_fk_robot`, `_robot_rot` | `mg`, `dg`, `g_jnt`, `g_jnt_a`, `g_qadr`, `g_dadr`, `g_range`, `gid`, `low_g`, `_fk_g`, `_g_rot` (read-only, `robot_map.OLD_NAMES`) | |
| RobotMap, SMPL side | `m_smpl`, `d_smpl`, `smpl_jnt`, `smpl_names`, `smpl_qadr`, `smpl_adr3`, `smpl_bid`, `low_smpl` | `ms`, `ds`, `s_jnt`, `s_names`, `s_qadr`, `s_adr3`, `sid`, `low_s` | |
| RobotEnv, SMPL side | `m_smpl`, `smpl_kp`, `smpl_gain`, `smpl_bias`, `smpl_spring`, `smpl_qadr` | `ms`, `s_kp`, ... (`robot_env.OLD_NAMES`) | |
| robot points / MSE | `RobotPoints(m_robot, ...)`, `RobotMSE(points, target)` | `g1_points`, `gT` | `--objective mse` (choice `g1mse` kept), the G1's `_g1mse` run folders, `g1_cmd_ik_deg` keys |
| search progress | `generation` / `gen` (single latent), `iteration` / `it` (latent sequence) | "step" in the progress lines | summary `generations`, curve `gen` / `iter`, z_trace `gen`, mu_trace `mu_iters`, `q_best_it<k>` |
| MPJAE | per-joint acceleration error (m/s^2) | | metric key `mpjae` |

Lookups: `methods.method_of(experiment, run_key)` -> MethodSpec, `methods.key_in(experiment, method)` / `experiment_keys(experiment)` /
`Experiment.key(method)` -> run key, `methods.canonical(name)` (raises for the ambiguous `es`, `cem`, `lsor`),
`MethodSpec.run_key` (the key most drivers give the method). The experiment drivers take their keys from `experiment_keys(...)`
(`LOES, CEM = K["loes"], K["cem_phc"]`), so a run key in a driver says which method it is.
