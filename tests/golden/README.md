# Golden regression harness (OOP refactor, step 0)

Every refactor step (docs/refactor/PLAN.md) must leave the searches **bit-identical**. This folder holds small but real runs of the
current CLIs, a frozen set of saved rollouts for the metrics, and a comparator. The reference outputs are
`outputs/refactor_golden/baseline/` (made from the snapshot code, commit 1331ef7); `baseline2` is a second run of the same code that
proves the cases are deterministic run to run.

## How to run

```bash
tests/golden/check.sh step1            # all cases -> outputs/refactor_golden/step1, metrics, compare with baseline; exit 0 = identical
tests/golden/run.sh step1 g1_es child_lso         # some cases only
uv run python tests/golden/metrics_golden.py step1 # outputs/refactor_golden/step1/metrics.json
uv run python tests/golden/compare.py baseline step1 [--cases a,b,metrics] [--max-diffs 20]
```

`run.sh` is resource-gated for the shared machine (CLAUDE.md): before every start it reads `free -g` and the CPU idle share; a case
starts only if fewer than `GOLDEN_P` (default 2) of its cases run, available memory minus the case's estimate (G1 30 GB, others
12 GB; CLAUDE.md: child ~9 GB; measured: a G1 case ~17 GB) stays >= 50 GB, CPU idle >= 15 %, and while idle < 25 % only when none of its cases runs.
If available memory falls below 20 GB it kills its newest case and exits 2. Every case runs in its own session with
`CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1` and `UV_NO_SYNC=1` (`uv run` must not
touch `.venv`, a symlink into the main tree). Per case: `<case>/stdout.log`, `<case>/.rc` (the command's real exit code),
`<label>/run.log` (start / done lines with exit code and seconds). A case whose folder has `.rc` 0 is skipped on a re-run
(`GOLDEN_FORCE=1` re-runs it; an old folder is moved to `<label>/_stale/`, never deleted). `check.sh` refuses the label `baseline`.

All cases take 14-17 min with the box busy (CPU idle 15-28 %: mostly one at a time); `metrics_golden.py` about 20 s. Times below are
the baseline's (seconds, start to exit, including the model load).

## Cases (tests/golden/cases.txt)

Clip move-ego-0-2/move-ego-0-2_1 (300 frames, one of the 52 task motions) unless noted. Each command is the line the experiment
driver's builder prints, with only the budget cut (`--evals 64` = 4 steps x 16 rollouts; `--iters 4 --budgets 2 4`) and `--out`.

| case | covers | command from | s |
|---|---|---|---|
| child_es_noanchor | LOES, our cost without anchor (BFM + 1.0 heading + 0.1 position), ExactObs, reference-velocity start | `loes_noanchor.py` `noanchor(child_cmd)` | 31 |
| child_es_anchor | LOES with `--anchor-weight 0.3` (the Task page's anchored runs) | `loes_noanchor.py` `child_cmd` | 33 |
| child_cem_phc | CEM (BFM-Zero): `--algo cem --objective phc`, PHC reward alone, pos_div = child s | `body_52_test.py` `ccmd` on the child (= outputs/child_52_cem_phc) | 33 |
| child_cmaes | CMA-ES (pycma, seed + 1), our cost | `child_cmd` + `--algo cmaes` (= child_52_rerun) | 33 |
| child_random | random search, our cost | `child_cmd` + `--algo random` | 31 |
| child_mse | `--objective mse` (losses.tracking_mse + heading/pos + anchor) | `child_cmd` with mse | 36 |
| child_dtw | `--objective dtw --fall-weight 2 --floor-friction 0.2` (slow-only DTW, fall term, FloorFrictionWrapper) | `child_cmd` with the slippery-floor DTW flags | 40 |
| child_evalz | `--eval-z` re-roll of `inputs/eval_z.npy` (3 latents, padded to 16 with z0) with `--eval-save-qpos --eval-save-act --eval-save-qvel` | `loes_noanchor.py` / `body_52_test.py` `evalcmds` | 19 |
| child_lso | LSO (BFMTrack), colored noise, n+1 slots (slot 0 = mean), ER L=8/5 rollouts | `child_52_lso.py` `commands` (lso) | 39 |
| child_lsor_w005 | LSOR: `--rank --heading-weight 0.05 --pos-weight 0.005` | `child_52_lso.py` (lsor_w005) | 39 |
| child_ta | Trajectory Adaptation: `--method ta --objective phc` | `child_52_lso.py` (ta) | 42 |
| payload_es | LOES on adult_heavyRWristHand_x80 (`--ref` adult clip, s read off the clip = 1) | `body_52_test.py` `zcmd("payload")` | 32 |
| payload_lso | LSO on the payload, `--scale 1` | `body_52_test.py` `lcmd("payload", "lso")` | 43 |
| tall_es | LOES (no anchor) on len1.4_wid0.8 (s = 1.364 read off the clip) | `tall_body_52.py` `commands` (es) | 31 |
| tall_cem_phc | CEM (BFM-Zero) on the tall body (PHC lengths / 1.364) | `tall_body_52.py` `commands` (cem) | 36 |
| tall_amass_lso | LSO on the AMASS clip Male2Walking__B9_-__Walk_turn_left_90 (152 frames, the shortest of outputs/amass_baselines/clips.txt), `--scale 1.364408905613045` | `amass_baselines.py` `commands` (tall, lso) | 27 |
| g1_es | robot_z_search LOES on the G1: ridge fill, per-joint gains, our cost + 1.0 IK, reference-velocity start, RobotVecEnv | `g1_small_test.py --set 52` `zcmd(c, "es")` | 58 |
| g1_es_resume | the same, 32 rollouts, then `--resume` to 64 (state.npz: Adam + RNG state); compare.py also checks `full/` against g1_es | `zcmd` + `--resume` | 70 |
| g1_cem_phc | CEM (BFM-Zero) on the G1, PHC against the G1 target, lengths / G1 s | `zcmd(c, "cem")` | 57 |
| g1_evalz | robot `--eval-z` of `inputs/eval_z.npy` with `--eval-save-qpos` | `g1_small_test.py` `evalcmds` | 29 |
| g1_lsor | robot_lso_search LSOR: ranks, 0.05 heading, 0.005 position, 1.0 IK | `g1_small_test.py --set 52` `lcmd(c, "lsor")` | 101 |
| g1_ta | robot_lso_search TA with PHC against the G1 target | `lcmd(c, "ta")` | 100 |

Not covered by a case: single_z_search on an AMASS clip (it needs the clip's task latent `data/origin_z/<clip>.npy`, which AMASS
clips do not have; the drivers run only latent-sequence methods there), the limited-leg body (same code path as payload with a
different XML / `--ref`), H1 / Apollo.

## What compare.py checks, and what it ignores

For every case, every file under `<label>/<case>` (recursively; the G1 resume case has `half/` and `full/`), BASE vs CAND:

* `.npz` / `.npy`: the same keys, and per array the same dtype, shape and **bytes** (NaN payloads and -0.0 count).
* `.csv`: the same header and the same text in every cell.
* `.json`: the same keys, types (int / float / bool / null) and values (floats exactly, NaN = NaN). Strings are compared after
  replacing the repository root with `<REPO>` and `refactor_golden/<label>/` with `refactor_golden/<LABEL>/`, so a path's spelling
  (relative vs absolute, which file) still has to match: old analysis code compares e.g. `summary["xml"]` with its own literal.
* `metrics.json` (below): as a JSON file, nothing ignored.
* Every case must have exited 0 in both runs, and the file sets must match.

Ignored, and why (each is a wall-clock measurement; nothing else differs between two runs of the same code):

| ignored | where | why |
|---|---|---|
| JSON key `minutes` | summary.json of single_z_search, lso_search, robot_z_search, robot_lso_search | run time |
| CSV column `sec` | curve.csv of lso_search, robot_lso_search | seconds since the start, per iteration |
| `*.log`, `*.txt`, `*.out`, dotfiles, MUJOCO_LOG.TXT | stdout logs, the runner's `.rc` / `.pgid` | text logs print timings |

**Resume check** (inside CAND): `g1_es_resume/full` must equal `g1_es` except the keys `resumed_from`, `resumed_at`, `resume_exact`.
The snapshot code is exact for the search itself (z_trace, best_z, state.npz, best.npz, origin_z.npz, every cost column of
curve.csv), but has two known, pre-existing differences that compare.py counts and reports instead of failing on
(`KNOWN_RESUME`; found in step 0, not fixed because step 0 changes no code): when the best latent was found before the resume
point, (1) `summary.json` `best.motion` (fallen / travel / end_xy / pelvis) is recomputed from best.npz's float32 qpos, while the
straight run used the float64 rollout (differences ~1e-7 m), and (2) curve.csv `best_done` is `""` instead of `"nan"` (the best
record rebuilt from summary.json has no `done`). The `g1_es_resume` case itself is compared strictly against the baseline.

## Determinism (baseline vs baseline2)

`baseline2` (same commit 1331ef7, run right after `baseline`, 12:48-13:05 against 12:33-12:48 UTC, other searches running on
the box both times) against `baseline`: **all 22 cases, metrics.json and the resume check identical** (every npz array bitwise,
every csv cell, every json field). Nothing had to be excluded or replaced: GPU 2's actor / B forward passes and the async env
workers are reproducible, and every RNG is seeded. The ignored fields really are wall-clock: `minutes` differed in all 21
summary.json files and `sec` in all 7 curve.csv files that have it.

## Metrics (metrics_golden.py)

`metrics_golden.py LABEL` computes the Task page's metrics with the current helper functions on fixed saved rollouts
(`inputs/metrics_rollouts.npz`, 35 arrays, 2.9 MB) and writes `<label>/metrics.json`: `eval_metrics.metrics` and
`tracking_metrics` (every body), `child_52_means.full` / `hand_fall`, `success_definitions.tests`, `fall_definitions.falls`,
`loes_noanchor.stats` (child s 0.611036, payload 1, G1 0.8213), `body_52_test.stats` (payload), `g1_small_test.stats` (G1),
`drift_success_child.one` and `tall_body_52.one` (their module folders `RR`, `LS`, `SEEDS`, `OUT` pointed at folders rebuilt from
the inputs). Every function gets q[t] against ref[t + 1]. When step 3 moves these functions, point the calls in `main()` at the new
code and keep the output keys; `compare.py baseline <label>` then proves the numbers are the same.
Step 3 (done) kept the calls as they are: the functions called here now delegate to crossembodiment/metrics.py and
crossembodiment/geometry.py (MetricSuite.task_page and the named definitions); tests/test_metrics.py covers the drivers this file
does not run (loes_weight_test, lsor_mpjae_test, amass_baselines.one, kpkd_test.phc_one / contact_one) against their a997a23 code.
Not covered: `amass_baselines.one` (its `stats` closure is `tall_body_52.one`'s without the PHC reward).
Step 4 archived scripts/child_52_means.py: its `full` / `hand_fall` are crossembodiment.metrics `legacy_metrics` /
`body_hand_fall` now, which metrics_golden.py calls under the old output keys (`child_52_means.full`, `child_52_means.hand_fall`);
tests/test_moves.py runs every helper step 4 moved against its 68c95f4 source.

## Inputs (tests/golden/inputs, 2.9 MB, in git)

* `eval_z.npy` (3, 256) float64: `project_z(z0)` of move-ego-0-2_1 and two latents `project_z(z0 + 0.05 * eps)`,
  `eps ~ default_rng(20261111).standard_normal(256)` (two draws), |z - z0| = 0.77.
* `metrics_rollouts.npz`, `metrics_tall_summaries.json`: `uv run python tests/golden/metrics_golden.py make-inputs baseline` (read
  the main tree's outputs/child_52_rerun, child_52_lso, body_52/payload, g1_52 and the golden baseline's tall_es / tall_cem_phc,
  since no full tall-body runs exist yet). Do not re-make them: the baseline metrics.json was computed from these exact files.

## Notes

* `/tests` is in `.gitignore`: new files under tests/golden need `git add -f`.
* Runs may append to the tracked `MUJOCO_LOG.TXT` at the repository root; never stage it.
* The baseline was made while about 8 other searches ran on the machine (CPU idle 15-28 %); timings above are from that run.
* Step 5 (simulation layer: crossembodiment/policy.py, obs.py, sim.py, rollout.py) replaced the rollout loops the cases run;
  tests/test_rollout.py also runs every replaced loop from its 37ed139 source (single_z_search.rollout in its four modes,
  lso_search's / robot_z_search's / robot_lso_search's loops, child_reward_eval.rollout, rank_initial_cost.rollout_multi,
  test_exact_obs's loop) against BatchRollout on 20-step, 4-slot rollouts of the child and the G1 (np.array_equal; ~20 s on GPU 2).
* Step 6 (costs, optimisers, search loops, run records: crossembodiment/costs.py, optim.py, problems.py, search.py, records.py)
  turned the four search scripts into thin CLIs; every case above now runs through them. tests/test_optim.py runs each script's
  old loop (from its fa6c5f9 source) against the new optimiser + loop on a toy cost (every scored batch, iterate and best
  np.array_equal; ES --resume through state.npz); tests/test_records.py checks RunRecorder (round trips; the baseline's
  summary.json / curve.csv / eval_base.json / best_z.npy rewritten byte-identical, its npz files array- and order-identical);
  tests/test_cli.py checks the four parsers against their fa6c5f9 versions (every flag, default, type, help; --help text).
  Flag paths no case covers (mean-z rollouts --eval-every, --z-start / --best-from-start, --subspace, antithetic / raw fitness,
  objectives align / both / mse / dtw on the human body, zero-velocity B, --init default, obs multiplier / raw, robot dtw / mse /
  cmaes + phc / random with --frame-offset 0, --resume without state.npz, LSO --dtw / --steps-scale / --init-pace / mse /
  antithetic / phc / white noise, robot LSO phc --best-from all) were run once with the fa6c5f9 scripts and the new ones
  (20 short cases, outputs/refactor_tests/step6_extra, compared with compare.py's cmp_dirs): identical.
* Step 7 (the G1-only line retired, user 11/12: the pre-10/20 "mass" gains dropped) archived model/g1_map.py, model/g1_env.py and
  13 scripts/g1_*.py (archive/README.md); no case ran them (the G1 cases are robot_z_search / robot_lso_search), so the golden
  cases and metrics.json must stay identical. tests/test_g1_generic.py runs the two archived modules from their 543a303 source
  against RobotMap("g1") / RobotEnv("g1") (calibration, both map directions incl. joint limits, lowest point, 60 env steps, the
  stored targets of g1_roundtrip / robot_roundtrip) and re-makes assets/robots/g1/gains_perjoint.json with robot_gain_scaling.py
  (np.array_equal / ==; CPU, ~15 s).
* Step 8 (experiment layer: crossembodiment/experiment.py, reroll.py, results.py, queue.py; eval_metrics.py moved into the
  package, scripts/eval_metrics.py re-exports it) changed no search code; the golden cases and metrics.json must stay identical.
  The drivers it ported are checked three ways:
  - tests/test_experiments.py: every command list of the ported drivers (28 builder / flag combinations) byte-identical to its
    d299653 source run as __main__; Reroller.prepare against the three old prepare loops (z.npy bytes, index lists); ResultsTable's
    reductions against the drivers' expressions (bitwise); amass_baselines.analyse old vs new on a synthetic experiment.
  - verify_analyses.py LABEL: each driver's analyse / prepare on the main tree's finished experiments, read only
    (CE_OUTPUTS_ROOT=<main>/outputs), written into outputs/refactor_step8/LABEL (CE_RESULTS_ROOT), compared file by file with the
    main tree's existing results (JSON and markdown byte-identical, npz array- and order-identical, z.npy byte-identical).
  - tests/test_queue.py: scripts/run_queue.py's CommandQueue / ResourceGate with fake commands.
* Step 9 (body-building pipeline: crossembodiment/bodybuild, scripts/build_body.py; the four pipeline copies and the building
  scripts are thin wrappers, scripts/torque_scale_actuators.py archived) changed no search code; the golden cases and
  metrics.json must stay identical. The bodies are checked two ways:
  - verify_bodies.py LABEL: the new pipeline rebuilds len0.8_wid1.2 and the child into outputs/refactor_step9/LABEL/root
    (robot.xml, skeleton.json, parameter.json keys, all 720 retargets, the torque ratio files, robot_torque_full.xml), 3 AMASS
    retargets each of len1.4_wid0.8 and the child, the payload's robot.xml, every gain variant of configs/bodies.yaml (MJCF and the
    G1's gains json), all byte for byte against the stored files (data/ and the main tree's outputs read only); and the same two
    bodies built by the 2bdb2fe scripts in a scratch root against the new files. The one stored file it cannot reproduce is the
    child's robot_torque_full.xml (made 08-24 from the child's previous robot.xml; old and new code agree on the rebuild, the
    compiled numbers differ by <= 1.1e-14 relative).
  - tests/test_bodybuild.py: the variant scripts (heavy_arm_body, lock_joint_body, limit_joint_body, elderly_body,
    strength_rollout's build) run from 2bdb2fe in a scratch root against the builders; the limited leg and adult_s08_* against
    the stored files; the wrappers' CLIs against 2bdb2fe's parsers; one BETA_AXES; the pipeline's skips, prune and registry.
* Step 10 (visualisation layer: crossembodiment/palette.py, figstyle.py, camera.py, clips.py, blender.py, strobe.py, video.py, sheet.py;
  configs/figures/{strobe_examples,teaser,method}.yaml; scripts/export_strobe_blender.py archived) changed no search code; the golden
  cases and metrics.json must stay identical. The visualisation outputs are checked two ways:
  - verify_vis.py LABEL: the new scripts re-make, into outputs/refactor_step10/LABEL/new (CE_OUTPUTS_ROOT = the main tree's outputs,
    read only; CE_RESULTS_ROOT / CE_FIGURES_ROOT inside the worktree), the stored Blender scenes (scene.json JSON-equal but for the new
    "version" field, meshes.npz arrays equal) of strobe_task52_blender (all 15 examples), blender_package, the top view, the walk
    band / rows, teaser_band_tall, child_all52, limit_leg_all52; the ClipBundle trajectories of child_all52, limit_leg_all52 and the
    14 strobe examples whose sources still exist (arrays, video.json bytes); the overview pages, index.md, top views, combined
    strobes and walk figures (bytes); the paper figures against the 8e9dc99 scripts run in a scratch root (PNG bytes, PDF but
    /CreationDate, SVG but <dc:date> and matplotlib's random ids); render_panels.py videos (MuJoCo's EGL frames are not
    bit-reproducible: old vs new within old vs old on the GPU, and exactly equal frames / mp4 bytes with tests/golden/fake_render.py,
    which replaces the rasterisation by a picture of the scene digest); two small Blender panels rendered by the old and the new
    scene + render_blender.py (Cycles on the CPU is deterministic run to run: pixel-equal).
  - tests/test_vis.py: the palette against render_blender.py's COLORS and every script's 8e9dc99 colour table, the figure helpers
    (SVG bytes), project / presets, StrobeLayout, pose_spec, BlenderScene, ClipBundle, ContactSheet, the frame bands, VideoRenderer vs
    the survey's replay.py.
* Step 11 (naming: crossembodiment/methods.py run keys, palette keys, the root scale `scale`, RobotMap / RobotEnv attribute names,
  z0's on-disk names behind paths constants; docs/architecture.md "Names") renamed identifiers only; nothing on disk, no flag. The
  golden cases and metrics.json must stay identical, and so must verify_analyses.py (the ported drivers now take their run keys from
  the registry; their docstrings, copied into README.md, are unchanged) and verify_vis.py (the figure scripts ask the palette by the
  new keys). tests/test_registry.py test_step11_names checks the lookups and every old name still read (palette OLD_KEYS, BodySpec
  s / s_rule / s_default, obs.CHILD_EXACT_S, Position.pos_div, paths.origin_z); tests/test_g1_generic.py the RobotMap aliases.
