#!/usr/bin/env python3
"""robot_lso_search.py -- scripts/lso_search.py (BFMTrack's latent sequence optimisation: one latent per control step) on
a robot of model/robots.py, driven through model/robot_env.RobotVecEnv exactly as scripts/robot_z_search.py drives it
(SMPL-equivalent observations in, Motivo's SMPL action mapped onto the robot out).

Reference: the ADULT's SMPL clip (--ref, e.g. amass_data/amass_origin_motion_grounded/ACCAD/<clip>.npz). Goal of step t
(state s_t -> s_t+1) = B of frame t+1 of the adult's clip (bfm_align.obs_from_qpos at 30 Hz, as robot_z_search); every
rollout starts from frame 0 mapped onto the robot, at rest or (--init-vel reference) moving at the reference's velocity
(frames 0 -> 1 mapped onto the robot, as robot_z_search --init-vel reference). Per-step reward
  r_t = cos(B(o_t+1), B(g_t+1)) - heading_weight * (1 - cos(heading - ref heading)) / 2 - pos_weight * |xy - ref xy|
        - ik_weight * ik_t
heading / xy on the robot's SMPL-equivalent root against the adult's clip (as robot_z_search), ik_t = mean IK residual
(rad) over the paired links of the action of step t (RobotEnv's last_resid: how far Motivo's command is from any pose
the robot can make). Cost reported = 1 - mean per-step reward, so with ik_weight 1 it is robot_z_search's cost without
the anchor term. --objective phc (11/09): the cos term is replaced by PHC's tracking reward per step (model/phc_reward.py,
BFM-Zero's tracking reward, as lso_search.py --objective phc) on the robot's own bodies against the robot target (--target,
default model/robots.target_path: the clip mapped onto the robot), lengths / the map's root scale s; the state before the
first step is the rollout's start pose on one side and target frame 0 on the other.
The optimiser is lso_search's: mu_t starts at unit(B(g_t+1)), pink noise over time, discounted return-to-go (--gamma),
per-step ranks (--rank), Adam, mu_t back on the sphere, slot 0 of every batch = the mean sequence. --method ta (11/09):
lso_search.py's BFM-Zero trajectory adaptation, same code: from Motivo's ER L = 8 sequence, white noise
--sigma x exp(-(M - i) / (beta1 M) - (H - h) / (beta2 H)), softmax average of the samples by whole-rollout cost
(z-scored, temperature --ta-temp). Baseline rolled out first: Motivo's zero-shot tracking ER with window 8.
Best so far: the lowest-cost SAMPLE (slot 0, the mean, is logged only), as lso_search.py; --best-from all lets the mean
win too, as every run of this script before 11/09.
Writes <out>/{curve.csv, summary.json, mu_trace.npz, rollouts.npz}: rollouts.npz holds per sequence (ER8, LSO start,
LSO final mean, LSO best) the robot qpos q_* (S x nq), its SMPL-equivalent qs_* and the per-step IK numbers fz_*, and,
as lso_search.py, the robot qpos of the best sequence so far after k iterations (q_best_it<k>) and of the mean sequence
after k updates (q_mean_it<k>) for k in --budgets.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/robot_lso_search.py --robot g1 \
           --ref amass_data/amass_origin_motion_grounded/ACCAD/<clip>.npz --out outputs/g1/lso/<clip> \
           [--n 128 --iters 64 --rank --heading-weight 1 --pos-weight 0.1 --ik-weight 1 --fill ridge]"""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco  # noqa: F401  (imported first, as before)
import numpy as np  # noqa: F401
import torch  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment.optim import colored_noise                     # noqa: E402,F401  (lso_search's, as before)
from model.robots import ROBOTS                                     # noqa: E402
# OOP refactor step 6: the search lives in the package (crossembodiment/problems.py RobotSequenceProblem: robot, reference,
# simulator, policy, per-step reward; costs.py the terms; optim.py LSO / TrajectoryAdaptation; search.py SequenceSearchLoop;
# records.py RunRecorder).


def build_parser():
    """the CLI (flags, defaults and help as they were)"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--ref", required=True, help="the adult's SMPL clip (.npz with qpos)")
    ap.add_argument("--target", default=None, help="--objective phc: the robot target (default model/robots.target_path of --ref's stem)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.0125)
    ap.add_argument("--lr", type=float, default=0.00625)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.8, 0.99))
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--beta", type=float, default=1.0, help="noise colour: 0 white, 1 pink, 2 red")
    ap.add_argument("--rank", action="store_true", help="per-step centred ranks of the returns-to-go (lso_search --rank)")
    ap.add_argument("--heading-weight", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=0.1)
    ap.add_argument("--ik-weight", type=float, default=1.0)
    ap.add_argument("--objective", choices=("bfm", "phc"), default="bfm")
    ap.add_argument("--method", choices=("lso", "ta"), default="lso")
    ap.add_argument("--ta-beta1", type=float, default=0.85, help="--method ta: iteration annealing (BFM-Zero)")
    ap.add_argument("--ta-beta2", type=float, default=0.9, help="--method ta: horizon annealing (BFM-Zero)")
    ap.add_argument("--ta-temp", type=float, default=0.05, help="--method ta: softmax temperature (DIAL-MPC's example config)")
    ap.add_argument("--init-vel", default="rest", choices=["rest", "reference"])
    ap.add_argument("--best-from", default="samples", choices=["samples", "all"],
                    help="samples: the mean sequence (slot 0) never becomes the best, as lso_search.py; all: it can (runs before 11/09)")
    ap.add_argument("--trace", choices=("all", "budgets"), default="all",
                    help="mu_trace.npz: the mean sequence after every iteration, or only after 0 and --budgets iterations")
    ap.add_argument("--budgets", type=int, nargs="*", default=[4, 8, 16, 32, 64])
    ap.add_argument("--fill", default="ridge", choices=["fixed", "ridge"])
    ap.add_argument("--gains", default=None, help="gains json (default: ROBOTS[robot]['gains_json'])")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    return ap


def main():
    """parse -> RobotSequenceProblem (crossembodiment/problems.py) -> LSO / TrajectoryAdaptation (optim.py) ->
    SequenceSearchLoop (search.py) -> RunRecorder (records.py)"""
    a = build_parser().parse_args()
    from crossembodiment.problems import RobotSequenceProblem
    from crossembodiment.records import RunRecorder
    from crossembodiment.search import SequenceSearchLoop

    P = RobotSequenceProblem(a)                        # spawns the robot workers before CUDA is touched, as before
    rec = RunRecorder(P.out)
    P.baseline()                                       # the zero-shot references B(g_t) and ER L=8, one batch
    opt, best = P.start()                              # LSO from B(g_t), or TA from ER L=8
    # curve.csv is written after every iteration: a stop keeps the curve
    loop = SequenceSearchLoop(opt, P.score, a.iters, P.S, best, budgets=a.budgets, keep=P.keep, best_from=a.best_from,
                              row=P.curve_row, report=P.report, on_iter=lambda lp, it: rec.curve(lp.curve))
    loop.run()
    P.finish(loop)                                     # the final mean and the best sequence, re-rolled
    P.save(rec, loop)                                  # mu_trace.npz, rollouts.npz, summary.json


if __name__ == "__main__":
    main()
