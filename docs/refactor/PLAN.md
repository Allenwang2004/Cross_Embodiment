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

| # | Step | Verification | Status (commit) |
|---|---|---|---|
| 0 | Golden harness `tests/golden/` (commands, baseline outputs from the snapshot code, comparator) | baseline exists; re-running the snapshot code reproduces it (determinism check) | done, e014fb1 |
| 1 | Archive unused files to `archive/` (see list below) | every remaining script imports (`python -c "import ast"` + import smoke test of modules); golden identical | done, 35f0bcb |
| 2 | Package skeleton `crossenbodiment/` (paths.py with REPO from `__file__`), `BodySpec`/`BodyRegistry` (+ s computed once), `RobotSpec`, `MethodSpec` registry; replace literal paths/scales in live code | registry reproduces every s/path used by existing results (assert table); golden identical | done, 01e511e |
| 3 | Pure-function consolidation: geometry (heading x6, robot yaw, lowest point x5, quaternion), `project_z` x13, metrics (`world/xpos` x9, `stats` x8) into `eval/metrics.py` + `MetricSuite` with named definitions (legacy root-success/height-fall vs PHC-success/contact-fall) | golden metrics file (Task-page metrics of fixed saved rollouts) identical; golden runs identical | done, 68c95f4 |
| 4 | Pull library code out of scripts: `compute_batch_cost` out of `model/simple`, `child_52_means.full/KEYS/hand_fall`, `survey_body_videos` CLIPS/BODY/colours, `g1_z0_rollout.act/motion`, `bfm_velocity_check.resample`, `rollout_z_trace.label`, `mse_vs_align_test.metrics` | imports; golden identical | done, 37ed139 |
| 5 | Simulation layer: `MotivoPolicy`, `ObsMapper` (Raw/Multiplier/Exact; scale explicit from BodySpec), `Simulator` (Humenv/Robot) + `BatchRollout` | golden identical (highest risk: batch composition) | done, fa6c5f9 |
| 6 | Costs (`CostTerm`s + `CompositeCost` returning (cost, parts)), optimisers (`ES`, `CEM`, `CMAES`, `RandomSearch`, `Adam`; `LSO`, `TrajectoryAdaptation`), `SearchLoop`, `RunRecorder`; the four search scripts become thin CLIs; `g1_z_search`, `batch_z_search` retired | golden identical for every algo x body; `--eval-z` identical; robot `--resume` works | done, d299653 |
| 7 | Merge `g1_map`/`g1_env` into `robot_map`/`robot_env` (keep or archive the pre-10/20 "mass" gains: ASK the user) and the `g1_*`<->`robot_*` script pairs | G1 golden identical | done, be4ad1d (after step 8) |
| 8 | Experiment layer: `Experiment`/`RunSpec`, `Reroller` (one index schema + readers for the 3 old ones), `ResultsTable`, `Checks`, `CommandQueue` + `ResourceGate` (CLAUDE.md thresholds, 20 GB emergency stop, correct exit codes, summary.json as done marker, sentinel files instead of log greps); port the live drivers | re-running `analyse` of existing results gives identical numbers (written into the worktree); queue dry-run | done, 543a303 |
| 9 | Body pipeline: `BodyBuilder` (+ subclasses), `TorqueCalibrator`, `GainScaler`, `Retargeter`, `BodyPipeline` (replaces new_body_pipeline.sh, make_grid_bodies.sh, make_morph_path.sh, make_scaled_bodies.py) | rebuild child and one len/wid body into a temp dir: XML, skeleton.json, retargets byte-identical | done, 8e9dc99 |
| 10 | Visualisation: `Palette`, `FigureStyle`, `ViewSpec`, `ClipBundle`, `StrobeLayout`, `BlenderScene`, `VideoRenderer`, `ContactSheet`; strobe picks (EX/CEM) to YAML; `render_blender.py` stays a standalone file | re-exported scene.json identical; one panel re-rendered pixel-equal; figures re-plotted equal | done, eee1938 |
| 11 | Naming (es->loes, cem->cem_ours/cem_phc, lsor->lsor_bfm/lsor_w005, z0/origin_z, s/scale/pos_div, body/robot/label) with read aliases for old output files; update docs (workflow.md, new_body.md) | old outputs still readable; golden identical | done, the commit "refactor(step 11): consistent naming with on-disk compatibility, documentation" |

Also on the branch: a997a23 (the rule on waiting for background jobs) and the "port:" commits (main-tree changes made during and
after the refactor): ded6f5d (the --loes-dir switches of 11/11), 2bdb2fe, 10c8ebd, a2a03a8 (strobe examples of 11/12), and the 11/12
port of ExactObs's clearance_vel option (single_z_search --exact-clearance-vel), kpkd_test --set mild / table, the new analysis and
figure scripts (obs_canon_metrics, phc_reward_tables, tall_kp06_analysis, plot_cost_phc, plot_obs_canon, amass_pickup_figures) and the
paper's LaTeX sections and figures (tests/test_clearance_vel.py; tests/test_cli.py and test_vis.py expect the main tree's additions).
The 11/13 port (feat/multibody 26f3dca, the main tree committed in full) makes the branch a superset of the main tree: amass_baselines
--set bfm (BFMTrack's 128 x 24 budget), kpkd_test --set mild's tall body, amass_pickup_figures --sequence / --overlay-adult,
phc_reward_tables.reward_fn, plot_obs_canon's lower figure, the new scripts (success_thresholds, amass_thresholds, objective_optimizer_test,
objective_optimizer_52, walk_band_labels, latex_accept_changes, latex_split_allen), the gain-variant assets, the paper's LaTeX (tracked
and clean) and .gitignore; outside the code every file equals feat/multibody's except the refactor's own docs.

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

## Step 11 (naming): what was renamed, what was not

The canonical names and their old spellings are tabled in docs/architecture.md ("Names"). Renamed in the code only, with read-only
aliases where anything outside the module used the old name: method keys (crossenbodiment/methods.py `run_key`, `key_in`,
`experiment_keys`, `canonical`; the experiment drivers take their run keys from the registry), palette keys (frozen, cem_phc,
cem_ours, lsor_w005; OLD_KEYS), the root scale (`scale` on BodySpec, ExactObs, RobotMap, MetricSuite, Position; problems'
`pos_scale`), RobotMap / RobotEnv's G1-era attributes (OLD_NAMES), z0's on-disk names behind paths constants (Z0_ROLLOUT, Z0_KEY,
task_z0). Deliberately left as they were:
* everything on disk (folders, run keys, file names, JSON / npz / CSV keys, configs/bodies.yaml keys) and every CLI flag and choice
  (no alias flags were added: tests/test_cli.py pins each parser to its fa6c5f9 version, and no flag name was misleading enough to
  need one; `--objective g1mse` stays a choice);
* the drivers' docstrings (analyse copies them into README.md / topview.md, which must stay byte-identical) and the searches'
  progress lines on stdout (people and old logs read them; "origin_z" and "steps" stay there);
* the robot problems' record keys `qg` / `qs` / `fz` (in-memory dicts and ScoreContext attributes the cost terms read by name) and
  the scripts' own local variables (`mg`, `dg` in robot_pairs.py, robot_gain_scaling.py);
* the scripts outside the experiment drivers (figures, videos, one-off studies) still spell run folders with their literal keys
  (`<stem>_es_s0`): those are on-disk names, and each such script reads one experiment whose keys are not ambiguous there.

## Decisions left to the user

1. **Palette disagreements** (step 10, crossenbodiment/palette.py DISAGREEMENTS; every script keeps its own colours until decided):
   CEM green #1baf7a vs orange #eb6834 in plot_algo_compare.py (step 11 found that this one is CEM with OUR cost, cem_ours, a
   different method from BFM-Zero's cem_phc: it may keep a colour of its own); random search orange (plot_child_convergence) vs
   green (plot_algo_compare); z0 grey #8c8b87 (plots, top views) vs orange (plot_k_scale_bars) vs red (every video); two LOES video
   blues ([0.18, 0.42, 0.85] survey / lso videos vs [0.16, 0.47, 0.84] tall_body_52 videos); label variants: CEM "BFM-Zero" /
   "CEM (BFM-Zero)" / "CEM", LOES "Ours" / "LOES (ours)" / "ES (ours)" / "LOES (best latent)", z0 "$z_0$" / "z0".
2. **robot_z_search --resume**: two derived records differ from a straight run when the best latent was found before the resume
   point (tests/golden KNOWN_RESUME, pre-existing): summary.json `best.motion` is recomputed from best.npz's float32 qpos (~1e-7 m),
   and curve.csv `best_done` is "" instead of "nan". The search itself is exact. Fixing it changes the output of resumed runs.
3. **The child's stored robot_torque_full.xml** was made on 08-24 from the child's previous robot.xml; the step-9 pipeline (old and
   new code alike) rebuilds a file whose compiled numbers differ by <= 1.1e-14 relative. Keep the stored file (every child result
   used it) or rebuild it (re-run the child's experiments)?
4. **lso_search.py --scale** defaults to the child's 0.611036 for ANY body (kept for bit-identity; every driver passes it for other
   bodies). Make it required, or default to the registry's value for --body?
5. **model/simple, model/dataset, model/networks** (the adapter-training line) are kept because dump_heldout_clips.py,
   test_exact_obs.py and model/simple/train_es.py import them (and write_body_splits.py / strength_rollout.py import model/bodies.py).
   Archive them together with those scripts?
6. **Merging refactor/oop back into feat/multibody**: not done. The main tree still runs the old code (experiments running there,
   the tall / AMASS queues held); the refactor's CLIs take the same flags and write the same files, so the queues can move over
   once the user decides.

