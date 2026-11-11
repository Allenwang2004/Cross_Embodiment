"""crossenbodiment -- the library behind scripts/ (OOP refactor of 2026-11-11, docs/refactor/PLAN.md).

Importable when the repository root is on sys.path, as every script arranges (`sys.path.insert(0, str(REPO))`).
Step 2 adds the data layer only; model/ stays where it is (later steps move modules here and leave re-export shims):
  paths     the repository's locations (REPO from this package's own path, relative and absolute spellings)
  bodies    BodySpec / BodyRegistry (configs/bodies.yaml): every body, its XMLs, references and root scale s; RobotSpec
  methods   MethodSpec registry: the methods of the baseline comparison, their flags, expected settings and the
            run-directory key each experiment used for them
Step 3 adds the shared pure functions (the scripts' copies now call these):
  geometry  the latent sphere (project_z, sphere16, unit_sphere, ang_pair), ranks, quaternion product, heading / yaw / facing
  metrics   the evaluation metrics as named functions (Task page: contact_fall_time, phc_success, drift, heading_error; legacy:
            root_success, height_fall, hand_fall), world positions, MetricSuite (one clip's reference, many rollouts), metric_body
Step 4 moves library code out of scripts (the old modules import it back; tests/test_moves.py runs each against its old source):
  align_cost       compute_batch_cost + AlignCostConfig (model/simple's L_align / L_phys cost; single_z_search.py)
  metrics          + legacy_metrics / body_hand_fall / LEGACY_KEYS (child_52_means.py, archived), pose_similarity
                   (mse_vs_align_test.metrics)
  survey, palette  the user survey's clips, bodies and run lookup (configs/survey.yaml), the video colours (survey_body_videos.py)
  robot_rollout    act / motion of the robot rollouts (g1_z0_rollout.py, archived in step 7)
  retiming         resample: a clip at another speed (bfm_velocity_check.py)
  draw             frame_label (rollout_z_trace.label) and the explanation-panel helpers of g1_axes_demo.py (archived in step 7)
  multibody_study  the 10/10 multi-body correction experiment's layout (multibody_corr.py)
  paths            + the clip lists configs/clips/*.txt (read_clips)
Step 5 adds the simulation layer, the one rollout loop every search and evaluation script uses:
  policy    MotivoPolicy: the frozen FBcprModel on its device; latents() (float64 -> float32 tensor), act() (the actor's
            mean action), embed() (B of observations)
  obs       ObsMapper strategies RawObs / MultiplierObs / ExactObs (in the env workers or the main process) and the scripts'
            selection rules (single_z_obs = single_z_search --obs-scale; lso_scale_kw = lso_search --scale as it is)
  sim       HumenvSim (humenv's async vector env) and RobotSim (model/robot_env.RobotVecEnv): reset / step -> SimStep
  rollout   BatchRollout: a fixed number of slots, run(Z) for one latent per slot (n, 256) or a sequence (n, T, 256),
            recording qpos / qvel / actions / obs / SMPL qpos / IK residuals; the batch compositions (copies, padded, with_mean)
Step 6 adds the searches themselves; scripts/single_z_search.py, robot_z_search.py, lso_search.py and robot_lso_search.py
are thin CLIs over them (same flags, defaults and output files):
  costs     CostTerm objects (BFMSimilarity incl. zero-velocity / plain, Heading, Position, Anchor, IKResidual, TrackingMSE,
            RobotMSE, PHCReward, SlowDTW, DTW012, DTWUnfinished, AlignedHeading, Fall, AlignPhys) with their single-latent
            (batch) and per-step forms, and CompositeCost: (cost, parts), the terms summed in the scripts' order
  optim     ask / tell optimisers ES, CEM, CMAES, RandomSearch (single latent), LSO, TrajectoryAdaptation (sequences), Adam;
            each owns its RNG; ES.state_dict() for --resume; colored_noise, return_to_go
  problems  the four searches' problems (HumanLatentProblem, RobotLatentProblem, HumanSequenceProblem, RobotSequenceProblem):
            the parts above put together from a script's parsed arguments, score(), the best records, rows and outputs
  search    SearchLoop (generation loop, best so far, mean-z rollouts, curve rows, z trace, resume) and SequenceSearchLoop
            (iterations, budgets, kept rollouts, mu trace)
  records   RunRecorder: every output file of a search, written (and read back) exactly as the scripts did
Step 7 retires the G1-only line (model/g1_map.py, model/g1_env.py and 13 scripts/g1_*.py, archived; the pre-10/20 "mass"
gains dropped, user 11/12): the G1 goes through model/robot_map.py / robot_env.py like every robot, RobotMap("g1") /
RobotEnv("g1") reproducing G1Map / G1Env(gains="perjoint") bit for bit (tests/test_g1_generic.py).
Step 8 adds the experiment layer the drivers (child_52_*, body_52_test, g1_small_test, kpkd_test, loes_*, lsor_mpjae_test,
tall_body_52, amass_baselines, cost_correlation, drift_success_child, success_ / fall_definitions) are built on:
  experiment    ClipSet, RunSpec, Experiment: the grid (clips, methods, seeds, bodies, budgets), run folders and the "dir|cmd"
                command lines; workers() ($CE_WORKERS), parse_flag / output_arg (the drivers read their flags in main now)
  reroll        Reroller (prepare: z.npy + the schema-2 index.json, best per generation / centre per budget / the budgets) and
                RerollIndex (reads it and the three older layouts); check_vs_curve; EvalRollouts (an --eval-z folder)
  results       ResultsTable (motions x seeds x budgets x metrics; the drivers' reductions, paired Wilcoxon, markdown), Checks,
                ReportWriter, write_json / save_npz (per_run.npz, results.json, means.json, README.md as before)
  queue         CommandQueue + ResourceGate: the shell queues' loop in Python (scripts/run_queue.py)
  eval_metrics  moved from scripts/eval_metrics.py (which re-exports it)
  paths         + outputs_root() / results_root() ($CE_OUTPUTS_ROOT, $CE_RESULTS_ROOT) and writable()
Step 9 adds the body-building pipeline (scripts/build_body.py; docs/new_body.md), the subpackage bodybuild:
  BodyBuilder     ScaledBody (beta: the 8 axes, len x wid, presets), HeavyLimb, LockJoints, LimitJoints, Elderly, ActuatorStrength:
                  load the adult -> edit -> ground -> robot.xml + parameter.json (the old keys + name / kind / parent / builder / edit)
  Retargeter      SameTopologyRetargeter (qpos_retarget.py), CopyReference, LockedJointReference
  TorqueCalibrator  k matrix -> aggregate k -> robot_torque_full.xml with its checks (torque_ratio_across_motions.py,
                  torque_aggregate_motion_k.py, torque_scale_actuators.py's library, archived)
  GainScaler      kp / kd multipliers, MJCF or gains json (scale_body_gains.py, kpkd_test.py prepare)
  BodyPipeline    the idempotent, memory-gated steps; BuildRoots (where a build writes); beta.BETA_AXES, the one list of the 8 axes
Step 10 adds the visualisation layer (scripts/strobe_final_blender.py, export_blender_package.py, topdown / walk / teaser / child_all /
body_all Blender scripts, render_panels.py, body_overlay.py, robot_overlay.py, the survey packagers, the paper figures are built on it):
  palette   Palette: the method table (ref, adult, z0, cem, loes, random, cmaes, lso, lsor, ta, native) -> label + hex / video RGBA /
            render_blender.py's sRGB, VARIANTS where scripts disagree (DISAGREEMENTS), ACCENTS, CONDITIONS, INK token sets
  figstyle  FigureStyle (paper: STIX, the 0.01-inch canvas, save to pdf / svg / png; notebook), context manager or apply();
            txt / arrow / box, the figures' plain drawing functions
  camera    ViewSpec (to_mjv / to_track / to_blender / project) and the presets (survey_track, body_overlay, robot_overlay,
            BLENDER_STROBE); project is the one world-to-pixel formula (draw.project, robot_overlay.py's copies)
  clips     ClipBundle: a survey-format clip folder (from_run / from_files / from_survey, add_cem, save); strobe_source routes
            strobe_final_blender.py's sources ("child_52", "child_52_s<k>", "c540", "tall_body_52", "body_52/", "kpkd:", folders)
  blender   BlenderScene (panels + its own mesh store, write() with a schema version, load, render() through uv + bpy) and the
            pose export (load_model, pose_spec, contact_frames, heading, body_height, panel)
  strobe    StrobeLayout (frames, tints, shifts, fixed / path / orbit headings, align) and the overview pages' frame windows;
            recipe_panels (export_strobe_blender.py's clip_panels, archived)
  video     VideoRenderer: side by side (render_panels' spec) and overlay sharing paint / label_px / legend / write
  sheet     ContactSheet (the overview pages), hcat / vcat / legend_strip, write_index_md
  paths     + read_path / write_path / own_path ($CE_OUTPUTS_ROOT / $CE_RESULTS_ROOT for the visualisation scripts), figures_root()
            ($CE_FIGURES_ROOT), FIGURE_CONFIGS (configs/figures: strobe_examples.yaml, teaser.yaml, method.yaml)
Importing the package imports nothing heavy (no torch, mujoco or humenv); align_cost (model.losses -> mujoco), robot_rollout,
eval_metrics (mujoco, ot), policy (torch), bodybuild's modules, blender and video (mujoco) import them when they are imported.
"""
