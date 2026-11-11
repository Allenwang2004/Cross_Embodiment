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
Not covered: `amass_baselines.one` (its `stats` closure is `tall_body_52.one`'s without the PHC reward).

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
