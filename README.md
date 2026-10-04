# Cross-embodiment latent adaptation for a frozen behavior foundation model

Meta Motivo M-1 (FB-CPR) is a humanoid behavior foundation model trained on an **adult** body: a frozen
policy π(a | s, z) is steered by a 256-d latent z (|z| = 16). This repo asks: **on a different body — mainly
a child (pelvis-height ratio 0.611, ~38 kg) — which z makes the frozen policy perform a given motion, and can
that z be predicted from the adult's own latent z0?** Only z is changed; the policy's weights never are.

The pipeline step by step, with every script: [`docs/workflow.md`](docs/workflow.md).

## Setup

| Piece | Where |
|---|---|
| Frozen policy and backward map B | `facebook/metamotivo-M-1` (pulled by `metamotivo`) |
| Bodies | `assets/robots/<body>/` (scaled MJCF, 8-d body parameters β, `scripts/scale_robot.py`); actuator-calibrated `assets/robots_torque/<body>/robot_torque_full.xml` — the child is the body used now |
| Adult motions, adult latents | `data/origin_motion/<task>/<task>_<trial>.npz` (540 clips); `data/origin_z/...` (one z0 per clip) |
| Child references | `data/child/retargeting_motion/...`: joint angles copied from the adult, root position × 0.611, lifted so the lowest point is at −5 mm |
| Clip sets | `datasets/crossenbodiment-child-balanced/` — manifest + `splits/balanced500_clips.txt` (500 clips, 6 categories: crawl, headstand, jump, move, raisearms, rotate) |
| What the policy sees | humenv's 358-d heading-relative proprio, translated to the adult's units: `--obs-scale auto` = fixed per-feature multiplier (`model/obs_scale.py`, used by every run so far), `exact` = the adult-equivalent observation by reverse retargeting (`model/exact_obs.py`) |
| Costs | bfm = 1 − mean_t cos(B(s_t), B(g_t)) (`model/bfm_align.py`); L_align, joint-space tracking (`model/losses.py`); global terms = heading + root-xy distance; anchor penalty λ (‖z − z0‖ / 16)². Results are reported as **cost ÷ z0's cost** (1.0 = no change) |

## Findings so far (child body)

1. **A per-clip search finds a good z.** Over 500 clips, the searched z reaches L_align ÷ z0 = 0.23–0.26
   (median) — `outputs/b500_targets/`.
2. **That z is not unique, so z0 → correction cannot be learned from search labels.** From one start, different
   seeds give corrections 17.5 apart (each ~16 long, cos 0.39); starts only 0.14 apart do no better; corrections
   of different clips are almost orthogonal (cos 0.06 within a category). Adapters in the full latent space —
   supervised on these labels, or trained by ES on the rollout cost — reach only 0.87–0.94 on held-out clips,
   against 0.26 for the search itself.
3. **Confining the correction to an 8-d subspace makes it consistent.** With the top-8 PCA directions of the
   corrections, nearby starts get nearby corrections (difference ÷ size 0.37 vs 1.09 unconstrained) at ~30%
   higher L_align. With 441 such labels a supervised adapter reaches 0.85 (33 / 47 held-out clips beat z0) and an
   8-d ES adapter reaches bfm 0.82 (34 / 47) — the best adapters so far.
4. **How the observation is translated matters.** Raw child observations break walking; the fixed multiplier is
   2–15% off the adult's observation; exact reverse retargeting reproduces positions and rotations exactly and
   improves z0 alone (L_align 1.12 → 0.90, median over 47 held-out clips), but does not make corrections of
   different clips more alike.
5. **The BFM's own backward z is not a child z.** Computed from the child's reference it equals the adult's
   backward z (cos 0.93): it describes the target motion, not how the child body should achieve it. Fed per
   frame it gives 0.79, against 0.26 for the search.

In progress (2026-10-04): exact-observation search labels for the 441 training clips (`outputs/exact_train/`)
→ an 8-d basis from them → an 8-d ES adapter with exact observations
(`outputs/simple_es/child_balanced/b500_sub8_exactbasis/`).

## Layout

```
model/            shared code
  bfm_align.py      bfm cost (B embeddings of rollout vs reference)
  exact_obs.py      adult-equivalent observation by reverse retargeting
  obs_scale.py      fixed per-feature observation multiplier
  losses.py         L_align (and L_phys), kinematics.py helpers
  networks.py       LatentAdapter (z0 + MLP), SubspaceAdapter (correction in a k-d subspace)
  dataset.py        manifest reader;  bodies.py: the 11-body roster and its split
  simple/train_es.py  ES training of an adapter (the main trainer); config.py; train.py (shared helpers
                      make_body_ctx / compute_batch_cost; its PPO trainer is legacy)
scripts/          search, labels, training launchers, evaluation, analyses — indexed in docs/workflow.md
docs/             workflow.md (this pipeline); new_body.md (adding a body); research_journey.md and
                  morphology_latent_report.md (reports up to 2026-09-30, in Chinese)
project_page/     results web page
outputs/ data/ datasets/ wandb/   git-ignored: results, motions and latents, clip sets, run logs
```

## Running

```bash
uv sync

# one clip: search a z for the child (two-stage: bfm, then L_align from its best)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/single_z_search.py --clip move-ego-0-2/move-ego-0-2_4 \
    --body child --init reference --objective bfm --evals 4096 --out outputs/example/bfm
CUDA_VISIBLE_DEVICES=1 uv run python scripts/single_z_search.py --clip move-ego-0-2/move-ego-0-2_4 \
    --body child --init reference --objective align --evals 2048 --lr 0.03 \
    --z-start outputs/example/bfm/best_z.npy --best-from-start --out outputs/example/align

# an 8-d ES adapter on the 500 clips (10% held out), 500 updates
GPU=1 bash scripts/run_b500_subspace.sh 8            # OBS=exact for exact observations

# a supervised adapter on 8-d search labels, scored by rollout on the held-out clips
uv run python scripts/train_sup_lowdim.py --label-roots outputs/lowdim_b100 outputs/lowdim_train \
    --out outputs/lowdim_train/sup
```

Experiments run on GPUs 1–3; the bottleneck is CPU (MuJoCo), so keep the number of parallel searches
(each with 16 env workers) within what 64 cores can carry.

## History

Removed on 2026-10-04 (all recoverable from git branch `snapshot/pre-cleanup-2026-10-04`): the bilevel
retargeting + RL line (`model/bilevel/`), the PPO-trained adapter, the tracking-inference z-map, and the 30-axis
body generator. Still in the repo but no longer active: the morphology study (continuation along body paths,
β nearest neighbour, interpolation, grid, β → latent maps, warm starts) and the single-body ES variants (row
weights, multi-head, best-point buffer, CMA-ES, larger adapters) — see `docs/workflow.md`.
