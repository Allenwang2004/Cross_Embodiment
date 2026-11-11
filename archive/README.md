# archive/

Files that nothing uses any more, moved here on 2026-11-11 (OOP refactor step 1, docs/refactor/PLAN.md) with `git mv`, keeping
their path under `archive/` (`scripts/x.py` -> `archive/scripts/x.py`) and their history (`git log --follow archive/scripts/x.py`).
They are kept for the record and not maintained. They find the repository from their own path (`Path(__file__).parent.parent`),
so to run one, move it back with `git mv` first.

Checked for every file below on 2026-11-11, besides the reason in its line:
no remaining .py file under scripts/, model/, tests/, utils/ imports it (ast, imports inside functions and the bare-name /
`scripts.` imports included; `tests/golden/import_check.py` passes after the move); no .py / .sh / .md file of the worktree and
no queue script of the main tree (`outputs/**/*.sh`, and its launchers / command lists) launches it by name; docs/workflow.md
and README.md lines that listed it as tooling now point here (docs/new_body.md mentions only `ik_action_from_qpos.py`, as the
history of a removed route). Two files archived together import each other only (noted in the line).
"Gone" = the folder is absent from the main tree on 2026-11-11.

## Old child pipeline (Comparison A, outputs/child_52)

| file | what it was | why archived (2026-11-11) |
|---|---|---|
| scripts/child_52_report.py | Comparison A report (z0 / ES / CEM / random, 52 motions x 5 seeds, medians) | superseded by child_52_means.py (means) and the Task-page drivers (child_52_rerun.py, child_52_lso.py); no references |
| scripts/child_52_curves.py | MPJPE / MMPJPE / MPJAE of every run's best-so-far z at every generation | done: its outputs stay in outputs/child_52_curves (read by child_52_means.py / child_52_cmaes.py); no references |
| scripts/child_52_mean_z.py | rollouts of every run's mean z after 4..64 generations | done: outputs stay in outputs/child_52_mean_z (read by child_52_means.py); no references |
| scripts/child_52_sd.py | mean +- sd figure from outputs/child_52_curves/per_run.npz | one-off figure; no references |
| scripts/child_52_aligned.py | Comparison A re-scored with q[t] against ref[t + 1] | superseded: that alignment is built into the Task-page metrics (eval_metrics.py, tests/golden); no references |
| scripts/child_52_means.py | Comparison A by means: per_run.npz / means.json / convergence_metrics_mean.png in outputs/child_52_curves (archived in step 4) | done: its outputs stay in outputs/child_52_curves (per_run.npz read by child_52_cmaes.py); its inputs' makers (child_52_curves.py, child_52_mean_z.py) are archived; its `full` / `KEYS` / `hand_fall` moved to crossenbodiment/metrics.py (legacy_metrics / LEGACY_KEYS / body_hand_fall), which the 5 drivers and tests/golden/metrics_golden.py import now; no other references |

## Abandoned adapter-training line

| file | what it was | why archived (2026-11-11) |
|---|---|---|
| scripts/train_sup_lowdim.py | supervised z0 -> z* adapter on the 8-d subspace labels | labels outputs/lowdim_b100, outputs/lowdim_train gone; README.md's run line marked archived |
| scripts/train_sup_targets.py | supervised z0 -> z* adapter on the b500_targets labels | input outputs/b500_targets (sup_dataset) gone |
| scripts/build_b500_sup_dataset.py | the supervised dataset built from the b500_targets searches | input outputs/b500_targets gone |
| scripts/diffusion_adapter.py | synthetic tasks around one task + a cross-body adapter G(beta, z0) (10/14) | outputs/diffusion_adapter gone; most of its bodies (teen, giant, ax*) deleted; only importer: diffusion_adapter_eval.py (archived) |
| scripts/diffusion_adapter_eval.py | scored that adapter against z0' | outputs/diffusion_adapter gone |
| scripts/dist_adapter.py | a cloud of good latents per body for one clip, 21 multibody bodies | outputs/dist_adapter gone; 30 of the 32 multibody bodies deleted from assets/robots_torque |
| scripts/rollout_ckpt_on_body.py | rolled a trained LatentAdapter checkpoint out on unseen bodies | checkpoints outputs/simple_es gone (outputs/ckpt_rollout too) |

## Old 11-body roster (model/bodies.py BodyRoster)

| file | what it was | why archived (2026-11-11) |
|---|---|---|
| scripts/pd_track_bodies.py | can each roster body's actuators reach the commanded angle (PD servo) | roster bodies (teen, giant, ...) and assets/robots_cal_alltrain, robots_cal_movetrain gone |
| scripts/replay_actions_on_body.py | open-loop replay of the adult's ctrl stream on the roster bodies | roster bodies and assets/robots_calib, robots_calib_move gone |

## Inputs gone

| file | what it was | why archived (2026-11-11) |
|---|---|---|
| scripts/compare_noscale.py | the b500 search with vs without obs scaling | input outputs/b500_targets gone |
| scripts/compare_exact_search.py | the b500 search, obs multiplier vs exact observations | input outputs/b500_targets gone |
| scripts/analyze_search_convergence.py | after how many evals each b500 search stops paying | input outputs/b500_targets gone |
| scripts/compare_runs.py | per-category cost curves of train_es runs | input outputs/simple_es gone |
| scripts/payload_all_blender.py | Blender strobe panels of all 52 payload motions, to pick examples (11/10) | outputs/payload_all52 deleted on 11/10 after the export (strobe_final_blender.py says so); the picks are made |
| scripts/render_crowd.py | every multibody body in one scene, replaying one clip | 30 of the 32 multibody bodies deleted from assets/robots_torque |
| scripts/stack_crowd.py | stacked render_crowd's z0 and searched-z videos | consumes render_crowd.py's output (archived) |
| scripts/render_bodies.py | contact sheet of a robot directory (default assets/robots_advance) | assets/robots_advance gone |
| scripts/plot_b500_pca.py | PCA of the b500 L_align latents by motion category | input outputs/b500_targets gone |
| scripts/plot_correction_orthogonality.py | what a PCA picture of search corrections hides | input outputs/b500_targets gone |
| scripts/wandb_clip_charts.py | per-clip W&B eval charts for old train_es runs | input outputs/simple_es gone |

## Superseded renderers and figures

| file | what it was | why archived (2026-11-11) |
|---|---|---|
| scripts/compare_videos.py | stacked two rendered .mp4 files side by side | no references; side-by-side videos come from render_panels.py and the survey's replay.py |
| scripts/plot_method_diagram.py | the method diagram that still showed the correction basis | superseded by plot_method_figure.py ("Replaces scripts/plot_method_diagram.py") |
| scripts/checkpoint_gallery.py | a c540 search at 50 / 80 / 90 / 95 / 100 % of its gain | no references; its output outputs/checkpoint_gallery gone |

## Pilots and finished studies

| file | what it was | why archived (2026-11-11) |
|---|---|---|
| scripts/child_new_test_report.py | report of outputs/child_new_test (3-motion pilot of Comparison A) | superseded by the 52-motion runs; no references |
| scripts/child_lso_vs_loes.py | LSO vs LOES on 3 child motions (outputs/lso_vs_loes) | superseded by child_52_lso.py; no references |
| scripts/k_scale_compare.py | search gain with joint gains x0.8 / x1.2 (11/03) | finished; no references (plot_k_scale_bars.py reads scale_body_gains.py's output, stays) |
| scripts/analyze_anchor.py | does an anchor penalty keep nearby starts nearby (10/01-04) | finished; no references |
| scripts/analyze_lowdim_search.py | does a k-d subspace make the correction a function of the start | finished; no references (plot_lowdim_trend.py stays) |
| scripts/analyze_latent_transfer.py | do latents that start close stay close after the search | finished; no references |
| scripts/compare_lowdim_exact.py | the 8-start walking test in k = 8, multiplier vs exact | finished; no references |
| scripts/natural_pairs.py | latent-vs-behaviour step 1 on real clip pairs | finished; no references |
| scripts/controlled_pairs.py | latent-vs-behaviour step 1 at controlled distances | finished; only importer: natural_pairs.py (archived) |
| scripts/z_sensitivity.py | which latent directions the frozen actor uses | finished; output outputs/z_sensitivity gone; no references |
| scripts/correction_basis.py | k-dim correction basis of a batch of single_z_search runs | no references |
| scripts/interface_ladder.py | policy quality as the interface is fixed step by step | input outputs/exact_train gone; no references |
| scripts/pca_subspace_test.py | ES in the principal subspace of past corrections (10/06) | finished: outputs stay in outputs/pca_subspace* (read by plot_pca_subspace.py); no references |
| scripts/body_transfer_test.py | the child's 8-d basis on other bodies | bodies teen, short_limbed, elderly, giant deleted |
| scripts/body_pca_transfer.py | the child's 8-d principal subspace on other bodies | bodies teen, short_limbed, elderly, giant deleted |
| scripts/multibody_pca_test.py | leave-one-body-out shared principal directions | bodies teen, short_limbed, elderly, giant deleted |
| scripts/axis_subspace_test.py | 8 original coordinates vs 8 principal directions (10/13) | finished; no references |
| scripts/multibody_corr_analyze.py | analyses A-D of the multibody correction experiment | finished; no references (multibody_corr.py stays: its constants are imported) |
| scripts/slippery_test.py | the adult on floors of different friction (11/02) | finished; no references |
| scripts/slippery_eval.py | evaluation of a --floor-friction search | finished; no references |
| scripts/slippery_velstart.py | does a moving start help on a slippery floor | finished; no references |
| scripts/dtw_eval.py | evaluation of an --objective dtw search | finished; no references (the dtw objective stays: model/dtw.py, golden case child_dtw) |
| scripts/lso_dtw_eval.py | evaluation of an lso_search.py --dtw run | finished; no references |
| scripts/loss_variant_compare.py | one running clip searched with three costs | finished; no references |
| scripts/strength_search_summary.py | does a search win back what a weaker adult lost | finished; no references (strength_rollout.py stays) |
| scripts/z_heading_bodies.py | why one z turns differently on different bodies | bodies m2c_t250 / t500 / t750, fat, tall deleted |
| scripts/z_heading_spread.py | does Motivo care which way the body faces | finished; no references |
| scripts/locked_action_check.py | does a searched z stop commanding the locked joints | finished; only importer: locked_action_trace.py (archived) |
| scripts/locked_action_trace.py | the same over every step of a search | finished; no references |
| scripts/scan_adult_reward.py | does the adult's own clip earn its humenv reward | finished; no references |
| scripts/rollout_budget_analyze.py | G1 convergence with 16 / 8 / 4 rollouts per step (10/28) | finished; no references |
| scripts/robot_long_rollout.py | a robot result rolled out longer than its clip | no references |
| scripts/scale_robot_l1.py | 30-axis (per-segment) body generator, library of the two below | its body set assets/robots_advance gone; only importers render_bodies.py, grid_l0_bodies.py (archived) |
| scripts/grid_l0_bodies.py | assets/robots filled to a stature x build grid laid out like robots_advance | assets/robots_advance gone; body grids now come from make_grid_bodies.sh / make_scaled_bodies.py |
| scripts/qpos_retarget_ik.py | foot-locked IK refinement on top of qpos_retarget.py | its output data/retargeted_motion_ik gone; the runbook uses qpos_retarget.py; no references |
| scripts/ik_action_from_qpos.py | ctrl stream that commands a pose-only clip | its route (ik_z_from_action.py) was removed (docs/new_body.md); no references |
| utils/read_npy.py | np.load wrapper with a hard-coded path | its path data/retargeting_z gone; no references |

## Step 6 (search scripts as thin CLIs, 2026-11-11)

| file | what it was | why archived (2026-11-11) |
|---|---|---|
| scripts/batch_z_search.py | one antithetic ES search per (clip, body) of a datasets/crossenbodiment-10bodies split in one process (sigma 0.25, lr 0.5, L_align; outputs/batch_z_search{,_seed777}) | an older driver, superseded by single_z_search.py and the experiment drivers; no .py / .sh of the worktree imports or launches it, no queue script of the main tree (`outputs/**/*.sh`) names it, nothing reads its targets.npz; its dataset (datasets/) is not even in the worktree. It imported `rollout` / `project_z` / `rank_normalize` / `device_arg` from scripts.single_z_search, which still exports them |

Kept although step 6 listed it for retirement: **scripts/g1_z_search.py**. `robot_z_search.py --robot g1 --sampling antithetic
--frame-offset 0` reproduces it for the per-joint gains, but its `--gains mass` (model/g1_env.py's first, pre-10/20 gains: kp x
the mass ratio x s, no json) has no other path: model/robot_env.py takes a gains json only. Whether the mass gains stay is the
user's decision in step 7 (PLAN.md: merge g1_map / g1_env into robot_map / robot_env); until then g1_z_search.py is unchanged.

## Candidates kept (PLAN.md list, 2026-11-11)

model/bodies.py (imported by strength_rollout.py and write_body_splits.py), write_body_splits.py (docs/new_body.md Step 1 tells
the reader to run it), build_dataset.py (writes the manifest model/dataset.py and model/simple read; new_body.md describes it;
kept with model/simple, see below), rollout_z_on_body.py (single_z_search.py's docstring renders results with it),
select_clips.py (writes top_clips.json, read by package_survey_picks.py), strobe_figures/replay.py (part of the self-contained
strobe_figures bundle, listed in its README), scripts/survey_replay.py (package_survey.py copies it into the survey),
model/networks.py (removing its unused heads is a code edit, not a move); g1_overlay.py, g1_amass_video.py and the rest of the
G1 line (step 7), export_strobe_blender.py (step 10).

Step 4 (2026-11-11) moved `compute_batch_cost` and the settings single_z_search.py read from model/simple into
crossenbodiment/align_cost.py (model/simple/train.py re-exports it; rollout_z_trace.py, rank_initial_cost.py and score_z_matrix.py
use the package too), but model/simple, model/dataset.py, model/networks.py and scripts/build_dataset.py stay: still live are
scripts/dump_heldout_clips.py (KEEP list: model.dataset and model.simple.train_es.clip_index), scripts/test_exact_obs.py
(model.dataset, model.simple.train make_body_ctx, train_es.set_init_qpos, and unpickling checkpoints' TrainConfig / ESConfig),
(scripts/batch_z_search.py, model.dataset and ESConfig, until it was archived in step 6) and the launchers scripts/run_b500_subspace.sh,
run_balanced500_global_anchor.sh, run_headstand_global_anchor.sh (`python -m model.simple.train_es`); model/simple/train.py
imports model/networks.py; build_dataset.py writes the manifest model/dataset.py reads.
