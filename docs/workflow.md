# Workflow and script index

The pipeline in the order it runs, then every script in `scripts/` by what it is for. Conventions used
throughout: latents live on the sphere of radius 16 and distances between them are Euclidean (two random
latents are ~22.6 apart); costs are reported as **cost ÷ the clip's z0 cost** on the same body (1.0 = no change).
Names marked *archived* were moved to `archive/` on 2026-11-11 (unused; the reasons are in [`archive/README.md`](../archive/README.md));
they find the repository from their own path, so move one back with `git mv` before running it.

## 1. Bodies

A body is an MJCF scaled from `assets/robots/adult/robot.xml` by 8 parameters β (length and girth of leg,
arm, torso, head), then given actuators calibrated to the torque it actually needs. Step-by-step guide (in
Chinese): [`new_body.md`](new_body.md); one-shot driver: `scripts/new_body_pipeline.sh`.

| Script | What it does |
|---|---|
| `scale_robot.py` | Scaled variant of the adult MJCF from β |
| `qpos_retarget.py` | Retarget adult motions to a body: joint angles copied, root position scaled, lifted to the ground (foot-locked IK refinement: `archive/scripts/qpos_retarget_ik.py`, *archived 2026-11-11*) |
| `export_skeleton_json.py` | Rest-pose transform of every body part |
| `torque_ratio_per_joint.py`, `torque_ratio_across_motions.py`, `torque_aggregate_motion_k.py` | Inverse-dynamics torque demand of a body vs the adult, per joint and over motions |
| `torque_scale_actuators.py` | Write `robot_torque_full.xml` with actuators scaled by that ratio |
| `torque_capability_check.py`, `plot_k_diagnostics.py` | Checks: can the actuators hold the poses; the old 11-body roster's checks (reach commanded angles, replay the adult's controls) are `archive/scripts/pd_track_bodies.py`, `archive/scripts/replay_actions_on_body.py` (*archived 2026-11-11*) |
| `archive/scripts/ik_action_from_qpos.py` | Control stream that commands a pose-only motion (*archived 2026-11-11*) |
| `write_body_splits.py` | Write each body's train / test split (`model/bodies.py`) into its `parameter.json` |
| `archive/scripts/render_bodies.py` | Contact sheet of a robot directory (default `assets/robots_advance`, gone; *archived 2026-11-11*) |
| `make_grid_bodies.sh`, `make_morph_path.sh`, `make_all_paths.sh` | Bodies on a leg × arm grid, or straight paths in β from the adult (morphology study); the stature × build grid `archive/scripts/grid_l0_bodies.py` is *archived 2026-11-11* |
| `archive/scripts/scale_robot_l1.py` | 30-axis generator; was library only for `render_bodies.py` / `grid_l0_bodies.py` (*archived 2026-11-11* with them) |

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

`single_z_search.py` is the workhorse: ES (16 independent perturbations, ranked; `--sampling antithetic` for the mirrored pairs of every search before 11/04) + Adam on one z for one (clip, body). The flags that
matter:

| Flag | Meaning |
|---|---|
| `--objective bfm \| align` | Cost: bfm, or L_align in joint space |
| `--heading-weight`, `--pos-weight` | Add heading and root-xy terms (the "global" cost); position in adult-equivalent metres, `--pos-unit body` for the body's own (every search before 11/04) |
| `--z-start`, `--best-from-start` | Start from another latent (e.g. the bfm stage's best) |
| `--anchor`, `--anchor-weight` | Penalty λ (‖z − anchor‖ / 16)² inside the cost |
| `--subspace`, `--subspace-dim` | Move only within start + span of the first k rows of a basis |
| `--obs-scale auto \| none \| exact` | What the policy and B see (see the README) |
| `--ref` | Track another reference (e.g. a start's own adult rollout) |
| `--steps` | Rollout length; the costs only read the reference's frames, so `--steps` = its length gives the same result faster |

The standard recipe is two stages: bfm (4096 evals, lr 0.1) → L_align and/or global (2048 evals, lr 0.03) from
the bfm stage's best; `archive/scripts/analyze_search_convergence.py` (*archived 2026-11-11*: `outputs/b500_targets` is gone) showed where those budgets can be cut. Batches of searches
are run by small schedulers next to their outputs (`outputs/b500_targets/scheduler.py`,
`outputs/lowdim_*/scheduler.py`, `outputs/exact_train/scheduler.py`): one process per search, a cap per GPU.

| Script | What it does |
|---|---|
| `score_z_matrix.py` | Roll arbitrary latents out on arbitrary (clip, body) and score them like training does |
| `archive/scripts/batch_z_search.py` | One search per (clip, body) in a single process (older driver; *archived 2026-11-11*, see archive/README.md) |

## 4. Labels for supervised learning

| Output | How it was made |
|---|---|
| `outputs/b500_targets/` | Full latent space, anchor λ = 1, bfm → global and L_align, all 500 clips |
| `outputs/b500_targets/sup_dataset/` | `archive/scripts/build_b500_sup_dataset.py` (*archived 2026-11-11*): the clips whose targets beat z0 (488), with the ES runs' held-out split |
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
| `archive/scripts/train_sup_targets.py` | Supervised adapter on the full-space labels, scored by rollout (*archived 2026-11-11*) |
| `archive/scripts/train_sup_lowdim.py` | Supervised adapter on the 8-d labels vs full-space, scored by rollout (*archived 2026-11-11*) |
| `archive/scripts/wandb_clip_charts.py` | Per-clip eval charts for a run started before train_es logged them (*archived 2026-11-11*) |
| `archive/scripts/compare_runs.py` | Per-category cost curves of several runs (*archived 2026-11-11*) |

## 6. Evaluation and videos

| Script | What it does |
|---|---|
| `archive/scripts/rollout_ckpt_on_body.py` | Roll a trained adapter out on a body (*archived 2026-11-11*) |
| `rollout_z_on_body.py`, `rollout_z_trace.py` | Drive the frozen policy with a z (sequence) from disk; watch a search learn |
| `render_panels.py`, `render_seed_bests.py` | Side-by-side videos (`archive/scripts/compare_videos.py`, *archived 2026-11-11*) |
| `test_exact_obs.py` | z0 and targets under raw / multiplier / exact observations |
| `archive/scripts/compare_noscale.py`, `archive/scripts/compare_exact_search.py` | The same search without obs scaling / with exact observations (*archived 2026-11-11*) |

## 7. Analyses (current line, 2026-10-01 to 10-04)

| Script | Question |
|---|---|
| `archive/scripts/controlled_pairs.py`, `archive/scripts/natural_pairs.py`, `archive/scripts/analyze_latent_transfer.py` (*archived 2026-11-11*) | Do latents that are close stay close after the search on the child? |
| `plot_seed_pca.py`, `plot_pair_distances.py`, `plot_joint_pca.py`, `archive/scripts/z_sensitivity.py` (*archived 2026-11-11*) | Spread of the searched latents; which directions the policy uses |
| `archive/scripts/analyze_anchor.py` (*archived 2026-11-11*), `plot_anchor_pca.py` | Does an anchor penalty keep nearby starts nearby? |
| `archive/scripts/plot_b500_pca.py` (*archived 2026-11-11*) | Do the searched latents keep the motion categories' structure? |
| `archive/scripts/plot_correction_orthogonality.py` (*archived 2026-11-11*), `plot_correction_learnability.py` | Why corrections from search labels cannot be learned |
| `archive/scripts/analyze_lowdim_search.py` (*archived 2026-11-11*), `plot_lowdim_trend.py` | Does a k-d subspace make corrections consistent, and at what cost? |
