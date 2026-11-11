#!/usr/bin/env python3
"""lso_search.py -- BFMTrack's Latent Sequence Optimization (Rupf et al. 2026, paper/BFMTrack.pdf Sec. 5.1) on our bodies.

One latent per control step. Step t (state s_t -> s_t+1) uses z_t; its mean mu_t starts at the goal latent of the
frame it should reach, B(g_t+1), and is optimised by REINFORCE with
  z_t^n = project(mu_t + sigma * eps_t^n),  eps^n = colored noise over time (1/f^beta; beta 1 = pink), per dimension
  r_t^n = cos(B(s_t+1^n), B(g_t+1))          (B fed the adult-equivalent obs, model.exact_obs, as everywhere else)
  R_t^n = sum_j>=t gamma^(j-t) r_j^n          (each latent is credited only with what follows it)
  A_t^n = R_t^n - mean_{i != n} R_t^i         (leave-one-out baseline)
  mu_t += Adam( mean_n A_t^n eps_t^n / sigma ),  then mu_t back onto the sphere
working on the unit sphere as the paper does (the policy gets 16 * z). Slot 0 of every batch rolls out the mean
sequence itself (not used in the gradient), so the curve is the mean's own cost. Before the first step the zero-shot
tracking latents of Motivo are rolled out for reference: ER with windows L (z_t = mean of B(g_t+1..t+L)).
Cost reported = 1 - mean per-step reward: bfm = 1 - mean_t cos, plus --heading-weight * heading + --pos-weight * pos
per step when given -- with 1.0 / 0.1 the same number as single_z_search.py's bfm recipe without the anchor.
--antithetic / --rank swap in single_z_search's gradient estimate (mirrored pairs; centred ranks, per step); off = the paper's.
Time-aligned variant (slower allowed, e.g. on a slippery floor, --floor-friction): --steps-scale k rolls out k x the
clip's steps; --init-pace p starts mu_t at the goal latents of the reference played at p x speed (velocities shrink
with it); --dtw rewards each rollout frame by its similarity to the reference frame model/dtw.py's slow-only alignment
matches it to (a reference frame matched by 2 rollout frames shares its similarity between them, one never reached
gives 0, so 1 - sum r / S is the alignment's cost), heading along the same alignment, and --fall-weight w subtracts w
for every matched frame with the head below --fall-head (spread so it adds w x the fallen share to the cost).
Cost variants: --bfm-zero-vel feeds B both sides' observations with their velocity part (obs[214:358]) set to 0 (the
starting latents B(g_t) stay the full ones); --objective mse makes the per-step reward 1 - tracking_mse at that frame
(model/losses.tracking_mse_frames; --mse-no-xy: the root term is the pelvis height only), frame by frame (no --dtw).
Writes <out>/{curve.csv, summary.json, mu_trace.npz, rollouts.npz}.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/lso_search.py --ref amass_data/amass_retargeting_motion/ACCAD/<clip>.npz \
           --out outputs/lso/<clip> [--n 128 --iters 24 --sigma 0.0125 --lr 0.00625 --gamma 0.97 --beta 1]
"""
import argparse, sys
from pathlib import Path
import mujoco  # noqa: F401  (imported first, as before)
import numpy as np  # noqa: F401
import torch  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
# OOP refactor step 6: the search lives in the package (crossenbodiment/problems.py HumanSequenceProblem: body, reference,
# ExactObs, simulator, policy, per-step reward; costs.py the terms; optim.py LSO / TrajectoryAdaptation and colored_noise;
# search.py SequenceSearchLoop; records.py RunRecorder). colored_noise is re-exported here (robot_lso_search imported it).
from crossenbodiment.optim import colored_noise                 # noqa: E402,F401  (moved; re-exported)


def build_parser():
    """the CLI (flags, defaults and help as they were)"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference qpos .npz on the body (frame 0 is the start pose)")
    ap.add_argument("--body", default="child")
    ap.add_argument("--xml", default=None)
    # !!! --scale defaults to ExactObs's CHILD value 0.611036 for ANY body (crossenbodiment/obs.py lso_scale_kw); kept for
    # !!! bit-identity of old commands. Other bodies must pass it (crossenbodiment.obs.registry_scale(body, clip) gives it)
    ap.add_argument("--scale", type=float, default=None, help="ExactObs root scale (default: ExactObs's child value, for any body)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.0125)
    ap.add_argument("--lr", type=float, default=0.00625)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.8, 0.99))
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--beta", type=float, default=1.0, help="noise colour: 0 white, 1 pink, 2 red")
    ap.add_argument("--er-windows", type=int, nargs="*", default=[8, 5], help="zero-shot ER baselines to roll out")
    ap.add_argument("--heading-weight", type=float, default=0.0,
                    help="per-step reward -= w * (1 - cos(heading - reference heading)) / 2 (single_z_search's heading term)")
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="per-step reward -= w * |root xy - reference root xy| in m (single_z_search's pos term, see --pos-unit)")
    ap.add_argument("--pos-unit", default="adult", choices=["adult", "body"],
                    help="adult (default since 11/04, as robot_lso_search.py): adult-equivalent metres, the body's metres / ExactObs's "
                         "root scale; body: the body's own metres, every run before 11/04")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--antithetic", action="store_true",
                    help="mirrored noise as single_z_search: n/2 sequences eps and their negatives; the gradient from each "
                         "pair's difference (no leave-one-out baseline: the pair cancels it)")
    ap.add_argument("--rank", action="store_true",
                    help="per step t, the n returns-to-go R_t replaced by centred ranks in [-0.5, 0.5] (single_z_search's "
                         "rank_normalize), ranked separately at each step so the credit assignment stays")
    ap.add_argument("--floor-friction", type=float, default=None, help="model/friction.py: the floor's friction")
    ap.add_argument("--steps-scale", type=float, default=1.0, help="roll out this x the clip's steps")
    ap.add_argument("--init-pace", type=float, default=1.0,
                    help="start mu_t at the goal latents of the reference played at this x speed")
    ap.add_argument("--dtw", action="store_true", help="time-aligned per-step reward (model/dtw.py, slower allowed)")
    ap.add_argument("--fall-weight", type=float, default=0.0)
    ap.add_argument("--fall-head", type=float, default=0.8)
    ap.add_argument("--bfm-zero-vel", action="store_true", help="B sees both sides with the velocity part set to 0")
    ap.add_argument("--objective", choices=("bfm", "mse", "phc"), default="bfm",
                    help="phc: the per-step reward is PHC's tracking reward (model/phc_reward.py), BFM-Zero's tracking reward")
    ap.add_argument("--method", choices=("lso", "ta"), default="lso",
                    help="lso: BFMTrack's LSO from B(g_t+1); ta: BFM-Zero's trajectory adaptation (dual-loop annealing as DIAL-MPC, "
                         "softmax average of the samples by whole-rollout cost), from Motivo's ER L=8 sequence, white noise "
                         "--sigma x exp(-(M - i) / (beta1 M) - (H - h) / (beta2 H)), i counting down from M = --iters, h = 1..H steps")
    ap.add_argument("--ta-beta1", type=float, default=0.85, help="--method ta: iteration annealing (BFM-Zero)")
    ap.add_argument("--ta-beta2", type=float, default=0.9, help="--method ta: horizon annealing (BFM-Zero)")
    ap.add_argument("--ta-temp", type=float, default=0.05, help="--method ta: softmax temperature on the costs z-scored over the samples (DIAL-MPC's example config)")
    ap.add_argument("--trace", choices=("all", "budgets"), default="all",
                    help="mu_trace.npz: the mean sequence after every iteration (about 20 MB for 300 steps), or only after 0 and --budgets iterations")
    ap.add_argument("--budgets", type=int, nargs="*", default=[4, 8, 16, 32, 64],
                    help="save the rollouts of the best sequence so far (q_best_it<k>) and of the mean sequence (q_mean_it<k>) after k iterations")
    ap.add_argument("--mse-no-xy", action="store_true", help="--objective mse: root term = pelvis height only")
    ap.add_argument("--init-vel-scale", type=float, default=0.0, help="start at this x the reference's velocity (frames 0 -> 1, as single_z_search.py); 0 = at rest")
    ap.add_argument("--keep-iters", type=int, nargs="*", default=[], help="also save the mean sequence's rollout after this many iterations (rollouts.npz q_it<k>)")
    ap.add_argument("--device", default="cuda")
    return ap


def main():
    """parse -> HumanSequenceProblem (crossenbodiment/problems.py) -> LSO / TrajectoryAdaptation (optim.py) ->
    SequenceSearchLoop (search.py) -> RunRecorder (records.py)"""
    a = build_parser().parse_args()
    from crossenbodiment.problems import HumanSequenceProblem
    from crossenbodiment.records import RunRecorder
    from crossenbodiment.search import SequenceSearchLoop

    P = HumanSequenceProblem(a)                        # asserts the flag combinations, builds everything, as before
    P.baseline()                                       # the zero-shot references B(g_t) and ER L=8 / 5, one batch
    opt, best = P.start()                              # LSO from B(g_t), or TA from ER L=8
    loop = SequenceSearchLoop(opt, P.score, a.iters, P.S, best, budgets=a.budgets, keep=P.keep, keep_iters=a.keep_iters,
                              row=P.curve_row, report=P.report)
    loop.run()
    P.finish(loop)                                     # the final mean and the best sequence, re-rolled
    P.save(RunRecorder(P.out), loop)                   # curve.csv, mu_trace.npz, rollouts.npz, summary.json


if __name__ == "__main__":
    main()
