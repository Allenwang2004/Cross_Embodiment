#!/usr/bin/env python3
"""robot_z_search.py -- scripts/g1_z_search.py for any robot in model/robots.py (model/robot_env.py RobotVecEnv).

Everything below reads "G1" for the robot chosen with --robot; --objective mse is g1mse against that robot's target,
its hands / feet / head points from ROBOTS[robot]["ee"]. Outputs go to outputs/<robot>/search/<stem>[_mse]_s<seed>
(the G1's MSE runs keep their _g1mse name). With --robot g1 it reproduces g1_z_search.py.

---- g1_z_search.py: single_z_search.py's latent search, on the Unitree G1 driven by an unchanged Meta Motivo.

Same search as single_z_search.py --algo es (--sampling: independent perturbations with LSO's leave-one-out baseline, or
the antithetic pairs of every search before 11/04; rank-normalised costs, Adam on z, z kept on
Motivo's sphere, best = best candidate actually rolled out, the mean z rolled out every --eval-every steps), the
same cost terms, and the same output files. What differs is the body: every rollout runs in model/g1_env.py
(G1 physics, SMPL observations / actions through model/g1_map.py), 2 * pairs of them in parallel processes, each
starting from the clip's frame 0 mapped onto the G1 at rest.

  cost = bfm + heading_weight * heading + pos_weight * pos + anchor_weight * (|z - anchor| / 16)^2 + ik_weight * ik
         bfm      1 - mean_t cos(B(obs_t), B(g_t)), obs = the G1's SMPL-equivalent observation, g = the adult's
                  original clip (bfm_align.obs_from_qpos at 30 Hz); the G1 map alone costs 0.176 on the walk
         heading  mean_t (1 - cos(heading - reference heading)) / 2, pos mean_t |root xy - reference xy| (m), both on
                  the SMPL-equivalent root (G1 x, y / s) against the adult's clip, as single_z_search computes them
         ik       mean over steps and the 13 paired links of the IK residual (rad) of each action's target pose
                  (G1Env.last_resid): how far Motivo's commanded SMPL pose is from any pose the G1 can make --
                  missing DoF and joint limits. 0 by default (--ik-weight); also logged: the arms' share and the
                  fraction of G1 targets that sit at a joint limit
--bfm-zero-vel feeds B the observations with their velocity part (local_body_vel, local_body_ang_vel: obs[214:]) set to
0, on both the rollout's and the reference's side, so bfm compares poses only (scripts/bfm_velocity_check.py); the
plain bfm is logged alongside as bfm_plain.
--objective dtw (first version, paused: its 2-frame steps let the path catch up with the reference, so done does not
show a slower pace) compares in time-aligned order (DTW) instead of frame t against frame t, so doing the motion slower is not
penalised for being late: each rollout frame is matched to one reference frame, the match advancing 0, 1 or 2 reference
frames per step and never 0 twice in a row (0.5x to 2x speed, Itakura's slope constraint; the reference's last frame
may be held), starting at frame 0 of both, end left open. The path maximises the summed cos(B(obs_i), B(g_j)); its end
is chosen to minimise
         dtw + dtw_lambda * (1 - done)     dtw = 1 - mean cos along the path, done = fraction of the reference reached
(dtw_lambda 1: a reference frame never reached counts as completely unlike). heading is then compared along the same
path; pos is not used (pass --pos-weight 0). Logged: dtw, done, the aligned heading head_al; bfm stays the plain
frame-by-frame value. best.npz / origin_z.npz also hold the path ("dtw_path": reference frame per rollout frame).
--objective g1mse puts squared errors against the clip mapped onto the G1 kinematically (outputs/g1/roundtrip/<stem>/
g1.npz, the G1 target) in place of bfm + heading + pos, on the G1's own body, weights 1 as losses.tracking_mse:
         pose     mean (G1 joint angle - target)^2 over the 29 joints, wrapped                            rad^2
         ee       mean |p - p_target|^2 of the hands (wrist_yaw_link), feet (foot sites) and head (the torso_link's
                  top mesh), relative to the pelvis in its heading frame                                  m^2
         root     mean |pelvis - pelvis_target|^2, world x, y, z (--mse-no-xy: the height z only)       m^2
         heading  mean 1 - cos(pelvis yaw - target yaw)
         (logged under either objective as "mse"; anchor and ik terms as above)
--objective phc (11/09, the Task page's CEM (BFM-Zero) on the G1): cost = 1 - mean_t PHC tracking reward
(model/phc_reward.py, as single_z_search.py --objective phc) on the G1's own bodies against the G1 target (the clip mapped
onto the G1, as g1mse), lengths / the map's root scale s (adult-equivalent metres); the state before the first step is
the rollout's start pose on one side and target frame frame_offset - 1 on the other. Pass --heading-weight 0
--pos-weight 0 --anchor-weight 0 --ik-weight 0 for BFM-Zero's reward alone.
--algo (11/09, single_z_search.py's other searches, same code): es (the default, above), random (candidates = the best z
so far + sigma * noise, on the sphere), cem (N(mean, std^2) per dimension, then mean and std of the --cem-elites best,
std floor --cem-sigma-min, start std --sigma), cmaes (pycma, popsize 2 * pairs, sigma0 --sigma, seed + 1, candidates
projected onto the sphere and told back, mean re-projected).
--eval-z (11/09): no search; roll out every latent of an (N, 256) .npy with the run's exact cost (batches of 2 * pairs,
padded with z0) and write <out>/{eval.csv, eval_base.json} and, with --eval-save-qpos, eval_qpos.npz (G1 qpos).
Writes <out>/{best_z.npy, best.npz, origin_z.npz, z_trace.npz, curve.csv, summary.json}; best.npz / origin_z.npz hold
the G1 qpos ("qpos"), its SMPL-equivalent ("qpos_smpl") and per step the 13 IK residuals (deg) + limit count ("fz").
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/robot_z_search.py --robot h1 [--clip ...] [--evals 1024] [--objective mse]
"""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco  # noqa: F401  (imported first, as before)
import numpy as np  # noqa: F401
import torch  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robots import ROBOTS                                     # noqa: E402
# OOP refactor step 6: the search lives in the package (crossenbodiment/problems.py RobotLatentProblem: robot, reference,
# simulator, policy, cost; crossenbodiment/costs.py the terms; optim.py the optimisers; search.py SearchLoop; records.py
# RunRecorder). dtw_align moved to crossenbodiment/costs.py; re-exported here.
from crossenbodiment.costs import dtw_align                         # noqa: E402,F401


def build_parser():
    """the CLI (flags, defaults and help as they were)"""
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    p.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    p.add_argument("--pairs", type=int, default=8, help="2*pairs rollouts per step")
    p.add_argument("--sampling", default="independent", choices=["antithetic", "independent"],
                   help="as single_z_search.py. independent (default since 11/04): 2*pairs independent eps, each weighted by "
                        "its rank minus the mean of the others; antithetic: eps and -eps (every search before 11/04)")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--evals", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heading-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=0.1)
    p.add_argument("--anchor", default=None, help="default: z0 of --clip")
    p.add_argument("--anchor-weight", type=float, default=0.3)
    p.add_argument("--ik-weight", type=float, default=0.0,
                   help="weight on the mean IK residual (rad) of the commanded poses: penalises latents whose actions "
                        "ask the G1 for joint configurations it cannot reach")
    p.add_argument("--objective", default="bfm", choices=["bfm", "mse", "g1mse", "dtw", "phc"],
                   help="mse (= g1mse): squared errors against the robot's target in place of bfm + heading + pos; "
                        "phc: 1 - mean_t PHC tracking reward against the robot's target, in place of bfm + heading + pos")
    p.add_argument("--algo", default="es", choices=["es", "random", "cem", "cmaes"], help="as single_z_search.py --algo")
    p.add_argument("--cem-elites", type=int, default=4)
    p.add_argument("--cem-sigma-min", type=float, default=None, help="--algo cem: std floor (default --sigma / 5)")
    p.add_argument("--eval-z", default=None, help="no search: score every latent in this (N, 256) .npy, write eval.csv, then exit")
    p.add_argument("--eval-save-qpos", action="store_true", help="with --eval-z: also write eval_qpos.npz (robot qpos (N, T, nq))")
    p.add_argument("--mse-no-xy", action="store_true",
                   help="g1mse's root term compares the pelvis height only, not where it is on the floor")
    p.add_argument("--start-frame", type=int, default=0,
                   help="use the clip from this frame on (reference, robot target, rollout start)")
    p.add_argument("--frame-offset", type=int, default=1,
                   help="the state after step t is scored against reference frame t + this. 1: the rollout starts at frame 0, so "
                        "after step t it should be at frame t + 1, as in scripts/robot_lso_search.py. 0 reproduces every run before "
                        "this flag existed, which scored each state one frame late")
    p.add_argument("--init-vel", default="rest", choices=["rest", "reference"],
                   help="every rollout starts from the clip's frame 0 at rest, or moving at the reference's velocity "
                        "(frames 0 -> 1 mapped onto the robot)")
    p.add_argument("--bfm-zero-vel", action="store_true",
                   help="B sees the observations with their velocities set to 0, rollout and reference alike")
    p.add_argument("--dtw-lambda", type=float, default=1.0,
                   help="--objective dtw: cost of not reaching the reference's end, per fraction of the reference")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--resume", default=None,
                   help="continue the search saved in this output dir up to --evals rollouts IN TOTAL (its steps are "
                        "kept, the new ones appended). Exact -- bit-identical to one longer run -- when the dir has "
                        "state.npz (written by every run since 10/29); for older runs: from its last search point with "
                        "the noise stream advanced past its steps, only Adam's moments restart")
    p.add_argument("--fill", default="fixed", choices=["fixed", "ridge"],
                   help="robot -> SMPL fill of the parts no robot link reports (model/robot_map.py): fixed = Motivo's mean "
                        "posture, ridge = predicted from the reported parts")
    p.add_argument("--gains", default=None, help="gains json (default: ROBOTS[robot]['gains_json'])")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    return p


def main():
    """parse -> RobotLatentProblem (crossenbodiment/problems.py) -> the optimiser (optim.py; --resume restores it) ->
    SearchLoop (search.py) -> RunRecorder (records.py)"""
    a = build_parser().parse_args()
    from crossenbodiment.problems import RobotLatentProblem
    from crossenbodiment.records import RunRecorder
    from crossenbodiment.search import SearchLoop

    P = RobotLatentProblem(a)                          # spawns the robot workers before CUDA is touched, as before
    P.baseline()                                       # z0's rollout (origin_z)
    rec = RunRecorder(P.out)
    if a.eval_z:                                       # no search: roll out the given latents with this run's cost
        P.eval_latents(a.eval_z, rec)
        return
    opt, st = P.start()                                # --algo; --resume: the earlier steps, traces, best and Adam / RNG state
    loop = SearchLoop(opt, P.score, P.record, P.n_envs, P.n_gens, st["best"], eval_every=a.eval_every, row=P.curve_row,
                      report=P.report, gen0=st["gen0"], path_len=st["path_len"], curve=st["curve"],
                      trace_mean=st["trace_mean"], trace_best=st["trace_best"])
    loop.run()
    P.close()
    P.save(rec, loop)


if __name__ == "__main__":
    main()
