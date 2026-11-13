# Workflow

The pipeline as it runs since the OOP refactor (2026-11-11): **search -> re-roll -> analyse -> tables and figures**. Every
step is a script in `scripts/` over the package `crossembodiment/` (module map and class list: [architecture.md](architecture.md)).
Commands run from the repository root with `uv run python ...` on GPU 2 only:

```bash
export CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
```

The box is shared (503 GB, 64 cores): run `free -g` before launching, keep >= 50 GB available, and launch long jobs detached
(`setsid nohup ... &`) so a closed session does not kill them. `scripts/run_queue.py` does the memory check for a whole commands file.

## Conventions

* **Latents** live on Motivo's sphere, |z| = 16; distances are Euclidean (two random latents are ~22.6 apart). `z0` is the task
  latent of a clip (`data/origin_z/<task>/<stem>.npy`, "origin_z" on disk); the method that keeps it is called *frozen*.
* **Methods** have one name each in code and tables (`crossembodiment/methods.py`): `loes` (ours, no anchor since 11/11),
  `loes_anchor`, `cem_phc` (BFM-Zero's CEM), `cem_ours`, `random`, `cmaes` (single latent); `lso`, `lsor_w005`, `lsor_bfm`,
  `lsor_ours`, `ta`, `native` (latent sequence). Run folders keep their old keys (`<stem>_es_s0`, `_cem_s0`, `_lsor_s0`, ...):
  the same key means different methods in different experiments, so look it up with `methods.method_of(experiment, key)`.
* **Root scale** s = body size / adult size (`BODIES.scale(body, clip)`): the child 0.611036, the G1 0.8213, the adult-size
  variants 1, the scaled bodies read off the clip. Positions (the position term, PHC's lengths, drift, the 0.5 m success
  threshold) are in adult-equivalent metres = metres / s. `lso_search.py --scale` defaults to the child's value for *any*
  body: pass it explicitly (`obs.registry_scale(body, clip)`).
* **Progress**: a single-latent search runs *generations* of 2 x pairs = 16 rollouts; a latent-sequence search runs
  *iterations* of n = 16 sequences (+ the mean). Budgets are quoted in rollouts (64 generations = 1,024 rollouts).
* **Metrics** (Task page; `crossembodiment/metrics.py`, `eval_metrics.py`): every rollout q[t] against the reference
  ref[t + 1], on the body itself. Fall Time (s, non-foot contact while the reference has none), PHC success (world-frame mean
  body error never above 0.5 m adult-equivalent), MPJPE / MMPJPE (cm), MPJAE = per-joint acceleration error (m/s^2), drift
  (mean pelvis x, y distance). Older results (per_run.npz of child_52_*) also carry the legacy root success / height fall.

## 1. Bodies

| What | How |
|---|---|
| A new human body (scaled, adult-size variant, gain variant) | `scripts/build_body.py scaled / scaled-set / grid / morph / heavy-limb / lock-joints / limit-joints / elderly / strength / gains ...` -> assets/robots/<body>/robot.xml, data/<body>/retargeting_motion, assets/robots_torque/<body>/robot_torque_full.xml, the registry entry (configs/bodies.yaml). Runbook (in Chinese): [new_body.md](new_body.md) |
| A robot (G1, H1, Apollo) | its entry in `model/robots.py`; `scripts/robot_gain_scaling.py --robot <r>` (per-joint kp / kd json); `scripts/robot_roundtrip.py --robot <r>` (the task clips mapped onto it: outputs/<robot>/roundtrip/<stem>/, the robot's targets); `scripts/fit_smpl_fill.py` (the ridge fill of the SMPL parts no link reports) |
| z0's cost of every clip on a body | `scripts/rank_initial_cost.py`, `scripts/robot_z0.py --robot <r>` |

## 2. Search

| Script | Body | Method |
|---|---|---|
| `single_z_search.py` | SMPL topology (child, scaled bodies, adult-size variants) | one latent per clip: `--algo es` (LOES) / `cem` / `cmaes` / `random`; `--objective bfm` (+ `--heading-weight 1.0 --pos-weight 0.1`; the anchor term is off since 11/11) or `phc` (BFM-Zero's reward) |
| `robot_z_search.py` | a robot of model/robots.py (`--robot g1`) | the same search through RobotVecEnv, + an IK term; `--resume` continues a run exactly (state.npz) |
| `lso_search.py` | SMPL topology | a latent per step: LSO (BFMTrack), `--rank` (LSOR), `--method ta` (BFM-Zero's Trajectory Adaptation) |
| `robot_lso_search.py` | a robot | the same on a robot |

Each run writes one folder (`--out`): `summary.json` last (the done marker), `curve.csv` (one row per generation / iteration),
`best_z.npy`, `best.npz` / `origin_z.npz` (the best and z0's rollouts), `z_trace.npz` (single latent) or `mu_trace.npz` +
`rollouts.npz` (sequences: `q_ER8` = native, `q_best_it<k>` / `q_mean_it<k>` at the `--budgets`). The settings each method uses on
the Task page are `METHODS[...].flags` + `COMMON` in `crossembodiment/methods.py`; the drivers below print the exact lines.

Experiments are grids of runs, one driver each:

| Driver | Experiment |
|---|---|
| `child_52_rerun.py`, `child_52_cem_phc.py`, `child_52_lso.py` | the child on the 52 task motions (configs/clips/child_52_task.txt), 5 seeds: LOES + anchor, CEM (our cost), random, CMA-ES / CEM (BFM-Zero) / the latent-sequence methods |
| `loes_noanchor.py` | LOES without the anchor on the child, the payload, the limited leg and the G1 |
| `body_52_test.py` | the payload (adult_heavyRWristHand_x80) and the limited leg (adult_lockRleg_bent_lim): every method, seed 0, 2,048 rollouts |
| `g1_small_test.py [--set 52]` | the G1: every method (3 clips x 3 seeds, or the 52 x seed 0) |
| `tall_body_52.py` | the tall thin body len1.4_wid0.8: LOES vs CEM (BFM-Zero) |
| `amass_baselines.py` | the latent-sequence methods on the 68 ACCAD locomotion clips (configs/clips/amass_locomotion_5s.txt), child and tall body |
| `kpkd_test.py` | gains kp / kd moved in opposite directions on four bodies |
| `loes_lso_4096.py`, `loes_weight_test.py`, `lsor_mpjae_test.py`, `cost_correlation.py`, `drift_success_child.py`, `success_definitions.py`, `fall_definitions.py` | follow-up questions on the child (each docstring says which) |

A driver's `commands` prints `"<run dir>|<command>"` lines; run them through the queue:

```bash
uv run python scripts/body_52_test.py commands > outputs/body_52/commands.txt
setsid nohup uv run python scripts/run_queue.py outputs/body_52/commands.txt --log outputs/body_52/queue.log \
    --sentinel outputs/body_52/searches.done --max-jobs 8 > /dev/null 2>&1 &
```

`run_queue.py` skips finished folders (summary.json), keeps >= 50 GB available, retries crashes, writes `<dir>/run.out` and
`<dir>/.rc`, and the sentinel file when every line is done (wait on that file, or on the queue's pid; never `pgrep -f` on text that
is also in your own waiting command).

## 3. Re-roll

The analyses score the latents a search found, rolled out again with the search's own cost:

* `<driver> prepare` -> `rollouts/<stem>[_<key>]/z.npy` + `index.json` (schema 2, `crossembodiment/reroll.py`): each run's best
  latent so far per generation (or at the budgets) and its centre (the mean z) at the budgets, each distinct latent once.
* `<driver> evalcmds` -> one `--eval-z <dir>/z.npy --eval-save-qpos` line per folder, run with
  `run_queue.py ... --done-file eval_qpos.npz` (latents padded with z0 to full batches; writes eval.csv, eval_base.json, eval_qpos.npz).
* Latent-sequence runs need no re-roll: their rollouts at the budgets are in rollouts.npz.

## 4. Analyse

`<driver> analyse` reads the runs and re-rolls (under `$CE_OUTPUTS_ROOT`, default outputs/), checks them (settings as intended,
re-rolls reproduce each search's best cost so far and its saved best rollout, z0 / native rollouts identical across runs) and
writes `per_run.npz` / `results.json` / `means.json` / `README.md` (under `$CE_RESULTS_ROOT`, default the same folder) through
`crossembodiment/results.py` (`ResultsTable`: means over motions and seeds per budget, paired Wilcoxon tests per motion).

## 5. Tables and figures

| Output | Script |
|---|---|
| Task-page tables (Notion) | the drivers' `analyse` tables (results.json `mean`, at 1,024 / 2,048 rollouts; best latent and centre) |
| convergence by budget, the child's 52 motions | `plot_child_convergence.py [--loes-dir outputs/noanchor/child]` |
| LOES's MPJPE gain per motion | `plot_child_improvement.py` |
| teaser, method figure | `plot_teaser.py render / draw / band`, `plot_method_figure.py render / draw / simple / first` (panels: configs/figures/*.yaml) |
| strobe figures (Blender) | `strobe_final_blender.py [export] [render] [combine]` (examples: configs/figures/strobe_examples.yaml), `render_blender.py` (standalone, runs inside Blender), `topdown_blender.py`, `walk_band_blender.py`, `child_all_blender.py`, `body_all_blender.py` |
| side-by-side and overlay videos | `render_panels.py <spec.json>`, `body_overlay.py`, `robot_overlay.py`, `lso_videos.py` |
| the user survey | `survey_body_videos.py`, `survey_robot_videos.py`, `package_survey.py`, `package_survey_picks.py`, `package_amass_lso_survey.py` |

Colours and labels come from `crossembodiment/palette.py` (where scripts still disagree, each keeps its own: `DISAGREEMENTS`),
styles from `figstyle.py`; figures go to docs/figures (`$CE_FIGURES_ROOT` to write elsewhere).

## 6. Checks

```bash
tests/golden/check.sh <label>                # 22 golden runs + metrics + the --resume case, bit-identical to outputs/refactor_golden/baseline
for t in tests/test_*.py; do uv run python $t; done
uv run python tests/golden/import_check.py   # every repo-local import resolves
uv run python tests/golden/verify_analyses.py <label>   # the drivers' analyses on the main tree's experiments, byte-identical results
uv run python tests/golden/verify_bodies.py <label>     # the body pipeline rebuilds stored bodies byte for byte
uv run python tests/golden/verify_vis.py <label>        # scenes, pages, figures, videos, renders as before
```

tests/golden/README.md says what each compares and what it ignores (wall-clock fields only).

## History

* Before 11/04 searches used antithetic pairs (`--sampling antithetic`) and positions in the body's own metres (`--pos-unit body`),
  and before the corrected alignment (child_52_rerun) `--frame-offset 0`; the flags still reproduce those runs. Some defaults are
  older than the Task-page settings (`--obs-scale auto`, `--init-vel-scale` unset, `lso_search.py --scale`), so the drivers pass
  every setting explicitly.
* The adapter-training line (Sep-Oct: an adapter (beta, z0) -> z trained by ES on the rollout cost, `model/simple/train_es.py`,
  `scripts/run_b500_subspace.sh`, `run_balanced500_global_anchor.sh`, `run_headstand_global_anchor.sh`, supervised adapters on
  search labels) ended when the per-clip optimum turned out not to be a function of z0. The scripts nothing uses any more are in
  `archive/` (archive/README.md lists each with the reason): that line's trainers and label builders, the analyses of that period
  (b500 labels, low-dimensional subspaces, latent transfer, anchors, the multi-body corrections) and the G1-only code (`g1_*.py`,
  model/g1_map.py, g1_env.py, merged into robot_*.py in step 7). An archived script finds the repository from its own path:
  `git mv` it back before running it.
* Reports of that period (in Chinese): research_journey.md, morphology_latent_report.md.
