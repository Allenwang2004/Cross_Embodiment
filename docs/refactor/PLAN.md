# OOP refactor of crossenbodiment: plan and rules (2026-11-11)

Branch `refactor/oop` in the worktree `/home/allen19/crossenbodiment-refactor`, started from `refactor/base` (a snapshot of the
whole working tree of `feat/multibody`). The user approved the plan and asked for it to run in the background, one commit per
step, reporting after each step and stopping only when a check fails or a decision is theirs.

## Hard rules (every step)

1. **Never touch `/home/allen19/crossenbodiment`** (the main tree): experiments are running there on the old code. In the worktree,
   `data`, `amass_data`, `.venv`, `assets/robots/menagerie`, `assets/robots/limx_humanoid` and `outputs/{g1,map_error,child_52,user_survey}`
   are symlinks into the main tree: READ ONLY. Write only inside the worktree (`outputs/refactor_golden/`, `outputs/refactor_*`).
2. **Bit-identical behaviour.** The project relies on exact re-rolls (a 1e-15 change moved a G1 single-rollout cost by 28%). After every
   step, `tests/golden/check.sh` must report every golden run identical to the baseline. Do not change: batch composition (2·pairs
   slots, z0 padding, 16 copies for mean-z, LSO n+1 slots with slot 0 = mean), RNG draw order and shapes (one `default_rng(seed)`, one
   draw per step; pycma seed+1; colored-noise sr then si), arithmetic order ([z+σε, z−σε] then project; leave-one-out; Adam with gen+1;
   double-argsort ranks; first argmin with strict <; float64 z → float32 tensors; float32 obs), `max_episode_steps`, ExactObs computed in
   the workers, RobotMap `assume_valid` and the G1 "seg" order, IK warm start. Output schema (summary.json keys, curve.csv columns,
   z_trace with gen −1, state.npz, npz key `qpos`, rollouts.npz keys) must stay readable by old analysis code.
3. **Archive, never delete**: move unused files to `archive/` (keep their git history with `git mv`).
4. **Old CLIs keep working**: `scripts/single_z_search.py`, `robot_z_search.py`, `lso_search.py`, `robot_lso_search.py` and the
   experiment drivers stay as thin entry points with the same flags, so the queued experiments (outputs/tall_body_52, amass_baselines)
   and every queue script can run on the new code unchanged.
5. **Resources** (CLAUDE.md, shared 503 GB / 64-core box): before launching anything run `free -g` and
   `top -bn2 -d 2 | grep "%Cpu(s)" | tail -1`; keep >= 50 GB available after launch; small golden runs only, at most 4 at once.
   GPU 2 only: `CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1`.
6. **Git**: one commit per step on `refactor/oop`, message `refactor(step N): ...` ending with
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage explicit paths only (never `git add -A` / `commit -a`); before
   committing check `git diff --cached --stat` contains no `assets/robots/menagerie`, `assets/robots/limx_humanoid` (submodules shown as
   type changes because they are symlinks here), `data`, `amass_data`, `.venv`, `outputs`. Never push. Never touch other branches.
7. If a golden check fails and the cause cannot be found and fixed within the step, STOP: revert the step's uncommitted changes and
   report what differs and why.
8. Waiting for a background job: wait on its pid (`while kill -0 $PID 2>/dev/null; do sleep 30; done`) or on a file it writes when it
   ends. Never `pgrep -f '<text>'` where <text> also appears in your own waiting command: it matches itself and never ends
   (this stalled step 2 for 3 hours).

## Steps (in order)

| # | Step | Verification |
|---|---|---|
| 0 | Golden harness `tests/golden/` (commands, baseline outputs from the snapshot code, comparator) | baseline exists; re-running the snapshot code reproduces it (determinism check) |
| 1 | Archive unused files to `archive/` (see list below) | every remaining script imports (`python -c "import ast"` + import smoke test of modules); golden identical |
| 2 | Package skeleton `crossenbodiment/` (paths.py with REPO from `__file__`), `BodySpec`/`BodyRegistry` (+ s computed once), `RobotSpec`, `MethodSpec` registry; replace literal paths/scales in live code | registry reproduces every s/path used by existing results (assert table); golden identical |
| 3 | Pure-function consolidation: geometry (heading x6, robot yaw, lowest point x5, quaternion), `project_z` x13, metrics (`world/xpos` x9, `stats` x8) into `eval/metrics.py` + `MetricSuite` with named definitions (legacy root-success/height-fall vs PHC-success/contact-fall) | golden metrics file (Task-page metrics of fixed saved rollouts) identical; golden runs identical |
| 4 | Pull library code out of scripts: `compute_batch_cost` out of `model/simple`, `child_52_means.full/KEYS/hand_fall`, `survey_body_videos` CLIPS/BODY/colours, `g1_z0_rollout.act/motion`, `bfm_velocity_check.resample`, `rollout_z_trace.label`, `mse_vs_align_test.metrics` | imports; golden identical |
| 5 | Simulation layer: `MotivoPolicy`, `ObsMapper` (Raw/Multiplier/Exact; scale explicit from BodySpec), `Simulator` (Humenv/Robot) + `BatchRollout` | golden identical (highest risk: batch composition) |
| 6 | Costs (`CostTerm`s + `CompositeCost` returning (cost, parts)), optimisers (`ES`, `CEM`, `CMAES`, `RandomSearch`, `Adam`; `LSO`, `TrajectoryAdaptation`), `SearchLoop`, `RunRecorder`; the four search scripts become thin CLIs; `g1_z_search`, `batch_z_search` retired | golden identical for every algo x body; `--eval-z` identical; robot `--resume` works |
| 7 | Merge `g1_map`/`g1_env` into `robot_map`/`robot_env` (keep or archive the pre-10/20 "mass" gains: ASK the user) and the `g1_*`<->`robot_*` script pairs | G1 golden identical |
| 8 | Experiment layer: `Experiment`/`RunSpec`, `Reroller` (one index schema + readers for the 3 old ones), `ResultsTable`, `Checks`, `CommandQueue` + `ResourceGate` (CLAUDE.md thresholds, 20 GB emergency stop, correct exit codes, summary.json as done marker, sentinel files instead of log greps); port the live drivers | re-running `analyse` of existing results gives identical numbers (written into the worktree); queue dry-run |
| 9 | Body pipeline: `BodyBuilder` (+ subclasses), `TorqueCalibrator`, `GainScaler`, `Retargeter`, `BodyPipeline` (replaces new_body_pipeline.sh, make_grid_bodies.sh, make_morph_path.sh, make_scaled_bodies.py) | rebuild child and one len/wid body into a temp dir: XML, skeleton.json, retargets byte-identical |
| 10 | Visualisation: `Palette`, `FigureStyle`, `ViewSpec`, `ClipBundle`, `StrobeLayout`, `BlenderScene`, `VideoRenderer`, `ContactSheet`; strobe picks (EX/CEM) to YAML; `render_blender.py` stays a standalone file | re-exported scene.json identical; one panel re-rendered pixel-equal; figures re-plotted equal |
| 11 | Naming (es->loes, cem->cem_ours/cem_phc, lsor->lsor_bfm/lsor_w005, z0/origin_z, s/scale/pos_div, body/robot/label) with read aliases for old output files; update docs (workflow.md, new_body.md) | old outputs still readable; golden identical |

Stay plain functions (do not wrap in classes): `project_z`, `rank_normalize`, `colored_noise`, return-to-go, latent distance,
quaternion/heading/yaw, lowest point (a cached function), `losses.*`, `dtw`, PHC kinematics, MPJPE/MMPJPE/MPJAE math (MPJAE is the
per-joint ACCELERATION error, m/s^2), fall windows, Wilcoxon wrapper, torque fitting (`fit_ratio`, `joint_torques`), one-off figures,
CLI `main`s. Lists (strobe picks, clip lists, bodies) are YAML data, not classes.

## Archive candidates (step 1; verify each is not imported / referenced by live code first)

- Old child pipeline: child_52_report, child_52_curves, child_52_mean_z, child_52_sd, child_52_aligned (child_52_cmaes: its output folder is
  still read by child_52_rerun's settings check; child_52_means: move `full/KEYS/hand_fall` first, step 4).
- G1 line (after step 7 only): model/g1_map, model/g1_env, g1_roundtrip, g1_gain_scaling, g1_feasibility_compare, g1_z0_summary, g1_z_search.
- Abandoned training direction: train_sup_lowdim, train_sup_targets, build_b500_sup_dataset, build_dataset, diffusion_adapter(+_eval),
  dist_adapter, rollout_ckpt_on_body, rollout_z_on_body; model/networks unused heads; model/jx (only __pycache__). model/simple only after
  `compute_batch_cost` is extracted (step 4).
- Old roster: model/bodies.py BodyRoster, write_body_splits, pd_track_bodies, replay_actions_on_body.
- Inputs gone: compare_noscale, compare_exact_search, analyze_search_convergence, compare_runs, payload_all_blender, render_crowd,
  stack_crowd, render_bodies, plot_b500_pca, plot_correction_orthogonality, wandb_clip_charts.
- Superseded renderers/figures: g1_overlay, compare_videos, plot_method_diagram, g1_amass_video, checkpoint_gallery,
  export_strobe_blender (keep `clip_panels` until step 10).
- Pilots / older studies with no references: child_new_test_report, child_lso_vs_loes, k_scale_compare, select_clips, analyze_anchor,
  analyze_lowdim_search, analyze_latent_transfer, compare_lowdim_exact, natural_pairs, controlled_pairs, z_sensitivity, correction_basis,
  interface_ladder, pca_subspace_test, body_transfer_test, body_pca_transfer, multibody_pca_test, axis_subspace_test,
  multibody_corr_analyze, slippery_{test,eval,velstart}, dtw_eval, lso_dtw_eval, loss_variant_compare, strength_search_summary,
  z_heading_{bodies,spread}, locked_action_{check,trace}, scan_adult_reward, rollout_budget_analyze, robot_long_rollout,
  scale_robot_l1 main + grid_l0_bodies (library parts of scale_robot_l1 may be imported: check), qpos_retarget_ik, ik_action_from_qpos.
- utils/read_npy.py; duplicate copies of survey_replay.py (keep outputs/user_survey/replay.py, it ships with the survey).
- KEEP (imported or used by docs/new_body.md / pipelines): rank_initial_cost, score_z_matrix, batch_infer_z, infer_z_from_qpos,
  write_body_splits?, dump_heldout_clips, split_tasks, split_tasks_by_fall, multibody_corr (constants imported by 6 scripts),
  mse_vs_align_test (metrics imported), bfm_velocity_check (resample imported), payload_fall_candidates, metamotivo_motion_rollout,
  rollout_z_trace, strength_rollout, all live Task-page drivers.
- tests/test_losses.py: one test loads the deleted assets/robots/giant; fix the test (skip when the asset is missing), do not archive.

## Known traps found in the analysis

- `lso_search.py --scale` defaults to the CHILD's 0.611036 for any body (ExactObs default); single_z_search computes s per body from the
  reference vs the adult clip. The registry must make s explicit everywhere.
- Queue logs: `echo "$(date ...) exit $?"` logs date's status (always 0).
- `cem` means our-cost CEM in child_52_rerun but PHC CEM in body_52 / g1_52 / tall_body_52; `lsor` means BFM+ranks in child_52_lso but
  the w005 variant elsewhere. Two generations of success/fall definitions coexist (old per_run.npz vs the Task page's).
- 0.611036 (child s) appears in ~13 scripts, 0.8212611427612194 (G1 s) in 3; "facebook/metamotivo-M-1" in 9 places; the adult XML literal
  in ~68 files; `slice(214, 358)` duplicates obs_scale.OBS_SEGMENTS.
- `single_z_search` imports `ESConfig` and `compute_batch_cost` from `model.simple` (training code that imports wandb, dataset, networks).
