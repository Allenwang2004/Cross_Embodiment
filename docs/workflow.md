# Workflow and script index

The pipeline in the order it runs, then every script in `scripts/` by what it is for. Conventions used
throughout: latents live on the sphere of radius 16 and distances between them are Euclidean (two random
latents are ~22.6 apart); costs are reported as **cost ÷ the clip's z0 cost** on the same body (1.0 = no change).

## 1. Bodies

A body is an MJCF scaled from `assets/robots/adult/robot.xml` by 8 parameters β (length and girth of leg,
arm, torso, head), then given actuators calibrated to the torque it actually needs. Step-by-step guide (in
Chinese): [`new_body.md`](new_body.md); one-shot driver: `scripts/new_body_pipeline.sh`.

| Script | What it does |
|---|---|
| `scale_robot.py` | Scaled variant of the adult MJCF from β |
| `qpos_retarget.py`, `qpos_retarget_ik.py` | Retarget adult motions to a body: joint angles copied, root position scaled, lifted to the ground (`_ik`: foot-locked IK refinement) |
| `export_skeleton_json.py` | Rest-pose transform of every body part |
| `torque_ratio_per_joint.py`, `torque_ratio_across_motions.py`, `torque_aggregate_motion_k.py` | Inverse-dynamics torque demand of a body vs the adult, per joint and over motions |
| `torque_scale_actuators.py` | Write `robot_torque_full.xml` with actuators scaled by that ratio |
| `torque_capability_check.py`, `pd_track_bodies.py`, `replay_actions_on_body.py`, `plot_k_diagnostics.py` | Checks: can the actuators hold the poses, reach commanded angles, replay the adult's controls |
| `ik_action_from_qpos.py` | Control stream that commands a pose-only motion |
| `write_body_splits.py` | Write each body's train / test split (`model/bodies.py`) into its `parameter.json` |
| `render_bodies.py` | Contact sheet of a robot directory |
| `grid_l0_bodies.py`, `make_grid_bodies.sh`, `make_morph_path.sh`, `make_all_paths.sh` | Bodies on a stature × build grid, a leg × arm grid, or straight paths in β from the adult (morphology study) |
| `scale_robot_l1.py` | Library only now: constants and `describe()` used by `render_bodies.py` / `grid_l0_bodies.py` |

## 2. Data and clip sets

| Script | What it does |
|---|---|
| `build_dataset.py` | The (clip × body) `manifest.jsonl` that `model/dataset.py` reads |
| `batch_infer_z.py`, `infer_z_from_qpos.py` | Metamotivo tracking inference: per-frame z of a motion on a body (`data/*/infer_*`) |
| `metamotivo_motion_rollout.py` | Roll Metamotivo's own skills in humenv |
| `split_tasks.py`, `split_tasks_by_fall.py` | Train / test task splits |
| `rank_initial_cost.py`, `write_clip_list.py`, `write_balanced_clips.py` | z0's cost on a body for every clip; clip lists above a threshold; the balanced 500-clip set + categories |
| `dump_heldout_clips.py` | Reproduce a checkpoint's held-out clip split |

## 3. Searching a latent for one clip

`single_z_search.py` is the workhorse: antithetic ES + Adam on one z for one (clip, body). The flags that
matter:

| Flag | Meaning |
|---|---|
| `--objective bfm \| align` | Cost: bfm, or L_align in joint space |
| `--heading-weight`, `--pos-weight` | Add heading and root-xy terms (the "global" cost) |
| `--z-start`, `--best-from-start` | Start from another latent (e.g. the bfm stage's best) |
| `--anchor`, `--anchor-weight` | Penalty λ (‖z − anchor‖ / 16)² inside the cost |
| `--subspace`, `--subspace-dim` | Move only within start + span of the first k rows of a basis |
| `--obs-scale auto \| none \| exact` | What the policy and B see (see the README) |
| `--ref` | Track another reference (e.g. a start's own adult rollout) |
| `--steps` | Rollout length; the costs only read the reference's frames, so `--steps` = its length gives the same result faster |

The standard recipe is two stages: bfm (4096 evals, lr 0.1) → L_align and/or global (2048 evals, lr 0.03) from
the bfm stage's best; `analyze_search_convergence.py` shows where those budgets can be cut. Batches of searches
are run by small schedulers next to their outputs (`outputs/b500_targets/scheduler.py`,
`outputs/lowdim_*/scheduler.py`, `outputs/exact_train/scheduler.py`): one process per search, a cap per GPU.

| Script | What it does |
|---|---|
| `score_z_matrix.py` | Roll arbitrary latents out on arbitrary (clip, body) and score them like training does |
| `batch_z_search.py` | One search per (clip, body) in a single process (older driver) |
| `segment_z_search.py` | Search a short latent sequence instead of one z |

## 4. Labels for supervised learning

| Output | How it was made |
|---|---|
| `outputs/b500_targets/` | Full latent space, anchor λ = 1, bfm → global and L_align, all 500 clips |
| `outputs/b500_targets/sup_dataset/` | `build_b500_sup_dataset.py`: the clips whose targets beat z0 (488), with the ES runs' held-out split |
| `outputs/lowdim_b100/`, `outputs/lowdim_train/` | 8-d subspace (`outputs/lowdim_search/basis_corrPCA_train.npy`), no anchor, bfm → L_align, all 441 training clips |
| `outputs/exact_train/` | Exact observations, full space, anchor λ = 1, bfm → L_align, 441 training clips (in progress) |

## 5. Training adapters

An adapter maps (β, z0) to a z for the body. `model/simple/train_es.py` trains it by ES on the rollout cost
(no labels); the `run_*.sh` launchers fix the settings of each run.

| Launcher / script | Run |
|---|---|
| `run_b500_subspace.sh <k>` | 500 clips, correction confined to k dims; `OBS=exact`, `BASIS=`, `TAG=` |
| `run_balanced500_global_anchor.sh` | 500 clips, full space, global cost + anchor |
| `run_headstand_global_anchor.sh` | Headstand clips only |
| `train_sup_targets.py` | Supervised adapter on the full-space labels, scored by rollout |
| `train_sup_lowdim.py` | Supervised adapter on the 8-d labels vs full-space, scored by rollout |
| `wandb_clip_charts.py` | Per-clip eval charts for a run started before train_es logged them |
| `compare_runs.py` | Per-category cost curves of several runs |

## 6. Evaluation and videos

| Script | What it does |
|---|---|
| `rollout_ckpt_on_body.py` | Roll a trained adapter out on a body |
| `rollout_z_on_body.py`, `rollout_z_trace.py` | Drive the frozen policy with a z (sequence) from disk; watch a search learn |
| `render_panels.py`, `render_seed_bests.py`, `compare_videos.py` | Side-by-side videos |
| `test_track_z.py` | The child's own backward z, per frame, vs z0 and search targets |
| `test_exact_obs.py` | z0 and targets under raw / multiplier / exact observations |
| `compare_noscale.py`, `compare_exact_search.py` | The same search without obs scaling / with exact observations |

## 7. Analyses (current line, 2026-10-01 to 10-04)

| Script | Question |
|---|---|
| `controlled_pairs.py`, `natural_pairs.py`, `analyze_latent_transfer.py` | Do latents that are close stay close after the search on the child? |
| `plot_seed_pca.py`, `plot_pair_distances.py`, `plot_joint_pca.py`, `z_sensitivity.py` | Spread of the searched latents; which directions the policy uses |
| `analyze_anchor.py`, `plot_anchor_pca.py` | Does an anchor penalty keep nearby starts nearby? |
| `plot_b500_pca.py` | Do the searched latents keep the motion categories' structure? |
| `plot_correction_orthogonality.py`, `plot_correction_learnability.py` | Why corrections from search labels cannot be learned |
| `analyze_lowdim_search.py`, `plot_lowdim_trend.py` | Does a k-d subspace make corrections consistent, and at what cost? |

## 8. Earlier lines still in the repo

**Single-z diagnostics** (2026-09): `plot_single_z.py`, `loss_test.py`, `compare_align_losses.py`,
`plot_seed_consistency.py`, `plot_z_scree.py`, `probe_z_flat.py`, `walk_z_plateau.py`, `cross_body_plateau.py`,
`compare_es_cmaes.py`, `cross_clip_z_transfer.py`, `analyze_z_targets.py`, `rollout_fitted_map.py`,
`geodesic_profile.py`, `cost_profile.py`, `plot_cost_profile.py`, `replay_noise.py`, `analyze_segment_z.py`,
`compare_global_cost.py`, `plot_nonunique.py`, `plot_lowdim.py`.

**Single-body ES adapter variants** (2026-09, none beat the plain run): `run_rowweight_experiments.sh`,
`run_budget_experiments.sh`, `run_memorize_experiments.sh`, `run_head_experiments.sh`,
`run_size_experiments.sh`, `run_size_headstand.sh`, `run_headstand_floor_search.sh`, `run_plateau_pipeline.sh`,
`analyze_collapse.py`.

**Morphology study** (how the best latent changes with β; 2026-09-27 to 09-30, figures on the project page):
`pipeline_morph.sh`, `pipeline_morph_analyses.sh`, `pipeline_axis.sh`, `pipeline_grid.sh`, `queue_paths.sh`,
`launch_when_free.sh`, `run_continuation.sh`, `run_grid_continuation.sh`, `run_path_controls.sh`,
`run_long_control.sh`, `run_warm_start_ab.sh`, `run_warm_start_ab2.sh`, `run_bnn.sh`, `plan_bnn.py`,
`run_beta_prior.sh`, `analyze_continuation.py`, `analyze_bnn.py`, `analyze_interp.py`, `analyze_transfer.py`,
`analyze_map_real.py`, `build_interp_jobs.py`, `build_transfer_jobs.py`, `build_correction_transfer_jobs.py`,
`analyze_correction_transfer.py`, `fit_beta_map.py`, `grid_zero_shot.py`, `plot_path_angles.py`,
`plot_real_bodies.py`, `plot_summary_figs.py`.
