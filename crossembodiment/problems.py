"""problems.py -- the four searches' problems: what is rolled out and how it is scored (OOP refactor step 6, docs/refactor/PLAN.md).

A problem is built from a search script's parsed arguments (the scripts keep their argparse; nothing here parses) and puts
together the package's parts in the order the script did (that order is part of bit-identity: the robot workers are spawned
before CUDA is touched, the env workers before the reference environments, ...):

  HumanLatentProblem     scripts/single_z_search.py   one latent per rollout, an SMPL-topology body (humenv)
  RobotLatentProblem     scripts/robot_z_search.py    one latent per rollout, a robot (model/robot_env.RobotVecEnv)
  HumanSequenceProblem   scripts/lso_search.py        a latent per step (LSO / TA), an SMPL-topology body
  RobotSequenceProblem   scripts/robot_lso_search.py  a latent per step (LSO / TA), a robot

Each holds the body (paths, references, start state), the observation mapper / simulator / policy / BatchRollout
(crossembodiment/obs.py, sim.py, policy.py, rollout.py), the cost (crossembodiment/costs.py CompositeCost) and offers
  score(Z) -> Scored        one batch rollout and its cost (cost, parts: every logged term, and the rollouts kept)
  record(...)               the best-so-far record of one slot (the dict the summary reads)
  baseline() / start()      z0's rollout (and the kinematic floor); the optimiser (crossembodiment/optim.py) and the start best
  curve_row / report        the curve.csv row and the progress line of a step (crossembodiment/search.py calls them)
  save(rec, loop)           every output file through a RunRecorder (crossembodiment/records.py)
The prints are the scripts' own lines (stdout logs are not compared, but they are what people read).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .costs import (AlignPhys, Anchor, AsType, BFMSimilarity, CompositeCost, Fall, Heading, PHCReward, Position, ScoreContext,
                    SlowDTW, TrackingMSE, AlignedHeading)
from .geometry import ang_pair, project_z
from .paths import REPO, Z0_DIRNAME, Z0_KEY, Z0_ROLLOUT, task_z0
from .search import copies

# (lambda_align, lambda_phys) per single_z_search.py --objective -- fed straight into AlignCostConfig so the cost
# compute_batch_cost returns IS the objective (bfm / mse / dtw / phc replace it; L_align / L_phys are logged)
OBJECTIVES = {
    "align": (1.0, 0.0),
    "phys": (0.0, 1.0),
    "both": (1.0, 1.0),
    "bfm": (1.0, 0.0),      # the "align" slot is the BFM latent loss
    "mse": (1.0, 0.0),      # cost = losses.tracking_mse (pose + ee + root + heading MSE); L_align logged only
    "dtw": (1.0, 0.0),      # cost = 1 - time-aligned B similarity (model/dtw.py: slower allowed) + aligned heading
    "phc": (1.0, 0.0),      # cost = 1 - mean_t PHC tracking reward (model/phc_reward.py; BFM-Zero's trajectory adaptation)
}
DTW_KEYS = ("dtw", "done", "pace", "head_al", "fallen")


class Scored:
    """one scored batch: cost (n,) (single latent) or r (n, S) per-step rewards (sequences), parts (the logged terms),
    q (the qpos the records keep), ro (the Rollout)"""

    def __init__(self, cost=None, parts=None, q=None, r=None, ro=None):
        self.cost, self.parts, self.q, self.r, self.ro = cost, parts if parts is not None else {}, q, r, ro


# ================================================================== single_z_search.py

class HumanLatentProblem:
    """scripts/single_z_search.py: one clip (data/origin_z/<clip>.npy, its reference on the body), one SMPL-topology body,
    one latent per rollout (2 * pairs slots), cost per --objective (+ heading / position / anchor terms)."""

    def __init__(self, args):
        import mujoco
        from humenv import make_humenv

        from .align_cost import AlignCostConfig
        from .obs import ExactObs, MultiplierObs, single_z_obs
        from .policy import MotivoPolicy
        from .rollout import BatchRollout
        from .sim import HumenvSim

        a = self.args = args
        task, stem = a.clip.split("/")
        data_root = Path(a.data_root) if Path(a.data_root).is_absolute() else REPO / a.data_root
        z0_path = data_root / Z0_DIRNAME / task / f"{stem}.npy"
        ref_path = (Path(a.ref) if a.ref else data_root / a.body / "retargeting_motion" / task / f"{stem}.npz")
        for q in (z0_path, ref_path):
            if not q.exists():
                raise SystemExit(f"{q} not found")
        xml = Path(a.xml) if a.xml else (REPO / "assets" / "robots_torque" / a.body / "robot_torque_full.xml")
        if not xml.exists():
            raise SystemExit(f"{xml} not found")
        out_dir = Path(a.out) if a.out else (REPO / "outputs" / "single_z" / f"{stem}_{a.objective}")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.xml, self.ref_path, self.out_dir = xml, ref_path, out_dir

        self.z0 = z0 = project_z(np.load(z0_path).reshape(-1).astype(np.float64))
        self.ref = ref = np.load(ref_path)["qpos"]
        self.goal = goal = ref[a.frame_offset:]                 # goal[t]: what the state after step t is scored against
        self.lam_a, self.lam_p = lam_a, lam_p = OBJECTIVES[a.objective]

        self.cfg = cfg = AlignCostConfig()
        cfg.lambda_align, cfg.lambda_phys = lam_a, lam_p
        cfg.phys_weights = a.phys_weights
        cfg.phys_fall_ref = not a.no_fall_ref
        cfg.align_discount = a.discount

        self.n_envs = n_envs = 2 * a.pairs
        self.n_gens = n_gens = max(a.evals // n_envs, 1)
        print(f"clip {a.clip} on {a.body} ({xml.name})")
        if a.objective == "mse":
            print("objective mse: cost = losses.tracking_mse (pose + ee + root + heading, weights 1)"
                  + (f" + {a.anchor_weight} * (|z - anchor| / 16)^2" if a.anchor_weight else ""))
        elif a.objective == "bfm":
            print("objective bfm: cost = 1 - mean_t cos(B(s_t), B(g_t))  (BFMTrack Eq. 3, "
                  "B fed the same rescaled obs as the actor); L_align / L_phys logged only")
        elif a.objective == "phc":
            print("objective phc: cost = 1 - mean_t PHC tracking reward (Luo et al. 2023 Eq. 2 as PHC's code computes it: "
                  "body positions, rotations, velocities, angular velocities; model/phc_reward.py); L_align / L_phys logged only")
        else:
            print(f"objective {a.objective}: cost = {lam_a} * L_align + {lam_p} * L_phys "
                  f"(L_phys weights={cfg.phys_weights}, L_align discount={cfg.align_discount})")
        print(f"{n_gens} steps x {n_envs} rollouts = {n_gens * n_envs} evals, "
              f"algo {a.algo}, sigma {a.sigma}, lr {a.lr}, reference {len(ref)} frames")
        print(f"init: {a.init}" + (" (rollout starts from the reference's frame 0)"
                                   if a.init == "reference" else
                                   " (humenv standing reset)"))

        self.policy = policy = MotivoPolicy.load(a.metamotivo, a.device)   # FBcprModel.from_pretrained(...).to(device); .eval()
        model = policy.model
        # --obs-scale: exact = the adult-equivalent observation; a body other than the child reads its root scale s off this
        # clip against the adult's (bodies.clip_ratio), the child keeps ExactObs's default; auto / <r> / none: multiplier / raw.
        # --exact-clearance-vel (main tree 11/12): + d/dt of ExactObs's ground-clearance correction (model/exact_obs.py clearance_vel)
        clearance_vel = bool(getattr(a, "exact_clearance_vel", False))
        if clearance_vel and a.obs_scale != "exact":
            raise SystemExit("--exact-clearance-vel needs --obs-scale exact")
        mapper = single_z_obs(a.obs_scale, a.body, xml, ref_path, data_root / "origin_motion" / task / f"{stem}.npz",
                              REPO / a.obs_scale_ref, parts=cfg.obs_scale_parts, clearance_vel=clearance_vel)
        if clearance_vel:
            print("exact obs: + d/dt of the ground-clearance height correction in the root's vertical velocity")
        if isinstance(mapper, ExactObs) and "scale" in mapper.kw:
            print(f"exact obs: root scale {mapper.kw['scale']:.6f} (from the reference vs the adult motion)")
        if a.floor_friction is not None:
            print(f"floor friction {a.floor_friction} (the floor wins every contact)")
        # ExactObs: each env worker computes its own slot's obs right after its physics step; --floor-friction in every worker
        self.sim = HumenvSim.make(xml, n_envs, wrappers=mapper.wrappers(), floor_friction=a.floor_friction,
                                  max_episode_steps=max(a.steps, 300))
        self.fk = fk = mujoco.MjModel.from_xml_path(str(xml))
        self.exact = exact = mapper.build() if isinstance(mapper, ExactObs) else None
        obs_mul = mapper.mul if isinstance(mapper, MultiplierObs) else None
        self.batch = BatchRollout(policy, self.sim, mapper, n_envs, a.steps, nv=fk.nv,
                                  record=("qpos", "obs", "act") + (("qvel",) if a.eval_save_qvel else ()))

        self.init_qpos = init_qpos = ((np.load(a.init_ref)["qpos"][0] if a.init_ref else ref[0])
                                      if a.init == "reference" else None)
        self.init_qvel = None
        if a.init_vel_scale is not None and init_qpos is not None:
            src = np.load(a.init_ref)["qpos"] if a.init_ref else ref
            init_qvel = np.zeros(fk.nv); mujoco.mj_differentiatePos(fk, init_qvel, 1.0 / 30, src[0], src[1])
            init_qvel *= a.init_vel_scale
            self.init_qvel = init_qvel
            print(f"start velocity: {a.init_vel_scale:g} x the reference's (pelvis {np.linalg.norm(init_qvel[:2]):.2f} m/s)")

        Bg = None
        self.env1 = None
        VEL = slice(214, 358)                    # humenv proprio: local_body_vel + local_body_ang_vel
        if a.objective in ("bfm", "dtw"):
            from model import bfm_align
            self.env1, _ = make_humenv(num_envs=1, task=None, xml=str(xml), state_init="Default")
            if exact is None:
                ro = bfm_align.obs_from_qpos(self.env1, ref, bfm_align.DEFAULT_DT, obs_mul)
                if a.bfm_zero_vel:
                    ro[:, VEL] = 0
                Bg = bfm_align.embed(model, ro, a.device)
            else:
                # the reference through the SAME reverse retargeting as the rollout states
                vr, ro = np.zeros(fk.nv), []
                exact.reset()                     # clearance_vel: differenced along the reference, frame 0 adds 0
                for t in range(len(ref)):
                    if t:
                        mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, ref[t - 1], ref[t])
                    ro.append(exact(ref[t], vr))
                ro = np.stack(ro)
                if a.bfm_zero_vel:
                    ro[:, VEL] = 0
                Bg = bfm_align.embed(model, ro, a.device)
        if Bg is not None:
            Bg = Bg[a.frame_offset:]
        self.Bg = Bg
        glob_w = (a.heading_weight, a.pos_weight)
        # what positions (and PHC's lengths) are divided by: the root scale with --pos-unit adult, else 1 (summary.json pos_div)
        self.pos_scale = pos_scale = exact.scale if a.pos_unit == "adult" and exact is not None else 1.0
        if a.pos_weight:
            print(f"position term: metres / {pos_scale:.6f} ({a.pos_unit if exact is not None else 'body, no s without --obs-scale exact'})")
        self.phc = None
        if a.objective == "phc":
            from model.phc_reward import PHCCost
            self.phc = PHCCost(fk, ref[max(a.frame_offset - 1, 0)], goal, scale=pos_scale)
            print(f"phc: lengths / {pos_scale:.6f} ({a.pos_unit if exact is not None else 'body'} metres), {fk.nbody - 1} bodies")
        self.anchor = None
        if a.anchor_weight:
            if not a.anchor:
                raise SystemExit("--anchor-weight needs --anchor")
            self.anchor = project_z(np.load(a.anchor).reshape(-1).astype(np.float64))

        # ---- the cost (crossembodiment/costs.py): the terms, and what is added when, as score() did
        from .geometry import smpl_heading
        ref_h, ref_xy = smpl_heading(fk, goal), goal[..., :2]   # humenv's heading, frame by frame
        self.align_phys = AlignPhys(fk, cfg, goal)
        self.tmse = TrackingMSE(fk, goal, root_xy=not a.mse_no_xy)
        heading, position = Heading(fk, ref_h), Position(ref_xy, pos_scale)
        log, terms = [], []
        if Bg is None and self.phc is None:                    # bfm / dtw / phc only log L_align / L_phys / tracking_mse
            log += [self.align_phys, self.tmse]
        bfm = None
        if Bg is not None:
            bfm = BFMSimilarity(policy, Bg, zero_vel=a.bfm_zero_vel)
            log.append(bfm)
        if a.objective == "dtw":
            dtw = SlowDTW(bfm, Bg)
            path_of = lambda ctx: ctx.parts["path"]                       # noqa: E731
            main = dtw
            # heading along the alignment; pos stays frame by frame; the fall term on the aligned frames
            terms += [(glob_w[0], AlignedHeading(heading, path_of)), (glob_w[1], position),
                      (a.fall_weight, Fall(fk, a.fall_head, path_of))]
        elif self.phc is not None:                             # rollouts start at init_qpos (or wherever humenv put frame 0)
            main = PHCReward(self.phc, lambda ctx: init_qpos if init_qpos is not None else ctx.q[0, 0])
        elif a.objective == "mse":
            main = AsType(self.tmse, np.float32)
        elif bfm is not None:
            main = bfm
        else:
            main = self.align_phys
        log += [heading, position]
        if a.objective != "dtw" and any(glob_w):
            terms += [(glob_w[0], heading), (glob_w[1], position)]
        if self.anchor is not None:
            terms.append((a.anchor_weight, Anchor(self.anchor)))
        self.cost = CompositeCost(main, terms, log)

    # ------------------------------------------------------------------ scoring
    def score(self, zs):
        """zs: (n, 256) -> Scored: cost (n,), parts (align / phys / mse: the joint-space terms, NaN where the objective does
        not need them for every slot; bfm; head, pos, anchor_dist; dtw terms; act, qvel), q the qpos. One batched rollout."""
        rr = self.batch.run(zs, init_qpos=self.init_qpos, init_qvel=self.init_qvel)     # zs float64 -> float32 on the device
        q = rr.qpos
        ctx = ScoreContext(z=zs, ro=rr, q=q, obs32=rr.obs.astype(np.float32))
        cost, parts = self.cost(ctx)
        n = len(zs)
        if "align" not in parts:
            parts["align"] = parts["phys"] = parts["mse"] = np.full(n, np.nan)
        parts.setdefault("bfm", np.full(n, np.nan, dtype=np.float32))
        parts.setdefault("anchor_dist", np.full(n, np.nan))
        parts["act"] = rr.act
        if self.args.eval_save_qvel:
            parts["qvel"] = rr.qvel
        return Scored(cost, parts, q, ro=rr)

    def logged(self, sc, i):
        """(L_align, L_phys, tracking_mse) of slot i -- computed here for that one rollout when the objective did not need
        them for every slot"""
        if self.Bg is None and self.phc is None:
            p = sc.parts
            return float(p["align"][i]), float(p["phys"][i]), float(p["mse"][i])
        la, lp = self.align_phys.one(sc.q[i])
        return la, lp, float(self.tmse.one(sc.q[i]))

    @staticmethod
    def dext(sc, i, with_path=True):
        """the dtw numbers of slot i, for the best / base records"""
        d = {k: float(sc.parts[k][i]) for k in DTW_KEYS if k in sc.parts}
        if with_path and "path" in sc.parts:
            d["path"] = sc.parts["path"][i].copy()
        return d

    def record(self, sc, i, z, gen, source):
        la, lp, lm = self.logged(sc, i)
        p = sc.parts
        return dict(cost=float(sc.cost[i]), align=la, phys=lp, bfm=float(p["bfm"][i]), z=z.copy(), qpos=sc.q[i].copy(), gen=gen,
                    source=source, head=float(p["head"][i]), pos=float(p["pos"][i]), mse=lm, **self.dext(sc, i))

    # ------------------------------------------------------------------ where we start, and the floor we are aiming at
    def baseline(self):
        a, z0, goal = self.args, self.z0, self.goal
        from .align_cost import compute_batch_cost
        self.sc0 = sc0 = self.score(copies(z0, self.n_envs))
        la, lp, lm = self.logged(sc0, 0)
        p = sc0.parts
        self.base = base = dict(cost=float(sc0.cost[0]), align=la, phys=lp, bfm=float(p["bfm"][0]),
                                head=float(p["head"][0]), pos=float(p["pos"][0]), mse=lm, **self.dext(sc0, 0, False))
        self.base_path = p["path"][0].copy() if "path" in p else None
        _, ra, rp = compute_batch_cost(self.fk, self.cfg, goal[None, :a.steps], [goal])
        self.floor = floor = dict(cost=float(self.lam_a * ra[0] + self.lam_p * rp[0]),
                                  align=float(ra[0]), phys=float(rp[0]), bfm=0.0 if self.Bg is not None else float("nan"))
        if self.Bg is not None:
            # the kinematic reference against itself, through obs_from_qpos on both sides -- exactly 0 by construction
            floor["cost"] = 0.0
        if self.phc is not None:                                # the reference against itself, from its own frame 0: 0
            floor["cost"] = float(self.phc(goal[None, :a.steps], self.ref[max(a.frame_offset - 1, 0)])[0])
        print(f"  origin_z : cost {base['cost']:.4f}  L_align {base['align']:.4f}  "
              f"L_phys {base['phys']:.4f}")
        print(f"  reference: cost {floor['cost']:.4f}  L_align {floor['align']:.4f}  "
              f"L_phys {floor['phys']:.4f}   <- kinematic floor")
        return base, floor

    def eval_latents(self, path, rec):
        """--eval-z: score every latent of an (N, 256) .npy with this run's cost (batches of 2 * pairs padded with z0) and
        write eval.csv, eval_base.json and the --eval-save-* arrays"""
        a, z0, n_envs = self.args, self.z0, self.n_envs
        from .rollout import BatchRollout
        Z = np.load(path).reshape(-1, z0.size).astype(np.float64)
        rows, qs, acts, vels = [], [], [], []
        sc = None
        for i in range(0, len(Z), n_envs):
            chunk = Z[i:i + n_envs]
            sc = self.score(BatchRollout.padded(chunk, z0, n_envs))      # padded to 2 * pairs slots with z0
            c, f, p = sc.cost, sc.parts["bfm"], sc.parts
            qs.append(np.asarray(sc.q)[:len(chunk)]); acts.append(p["act"][:len(chunk)])
            if a.eval_save_qvel:
                vels.append(p["qvel"][:len(chunk)])
            for k in range(len(chunk)):
                rows.append((i + k, float(c[k]), float(f[k]), float(p["head"][k]), float(p["pos"][k]),
                             float(p["anchor_dist"][k]))
                            + tuple(float(p[x][k]) for x in DTW_KEYS if x in p))
        keys = sc.parts if sc is not None else {}
        rec.eval_csv(["index", "cost", "bfm", "head", "pos", "anchor_dist"] + [x for x in DTW_KEYS if x in keys], rows, style="single")
        rec.json("eval_base.json", self.base)
        if a.eval_save_qpos:
            rec.npz("eval_qpos.npz", qpos=np.concatenate(qs).astype(np.float32))
        if a.eval_save_act:
            rec.npz("eval_act.npz", act=np.concatenate(acts).astype(np.float32))
        if a.eval_save_qvel:
            rec.npz("eval_qvel.npz", qvel=np.concatenate(vels).astype(np.float32))
        print(f"eval: {len(rows)} latents scored -> {rec.path('eval.csv')}")

    # ------------------------------------------------------------------ the search
    def start(self):
        """the start latent (z0, or --z-start), --subspace, the optimiser (crossembodiment/optim.py) and the best-so-far
        record the search starts from. Returns (optimiser, best, z_start)."""
        from .optim import single_latent_optimizer
        a, z0, base = self.args, self.z0, self.base
        z_start = z0.copy()
        start_best = None
        if a.z_start:
            z_start = project_z(np.load(a.z_start).reshape(-1).astype(np.float64))
            sc = self.score(copies(z_start, self.n_envs))
            la, lp, lm = self.logged(sc, 0)
            p = sc.parts
            start_best = dict(cost=float(sc.cost[0]), align=la, phys=lp, bfm=float(p["bfm"][0]),
                              head=float(p["head"][0]), pos=float(p["pos"][0]), mse=lm, **self.dext(sc, 0),
                              z=z_start.copy(), qpos=sc.q[0].copy(), gen=-1, source="z_start")
            print(f"  warm start: cost {float(sc.cost[0]):.4f} "
                  f"({float(sc.cost[0]) / max(base['cost'], 1e-9):.3f} x origin_z), "
                  f"{ang_pair(z_start, z0):.1f} deg from z0")
        self.z_start = z_start
        U = None
        if a.subspace:
            if a.algo != "es":
                raise SystemExit("--subspace is for --algo es")
            U = np.load(a.subspace).astype(np.float64)
            U = U[: (a.subspace_dim or len(U))]
            assert np.allclose(U @ U.T, np.eye(len(U)), atol=1e-6), "--subspace rows must be orthonormal"
            print(f"subspace: {len(U)} of {z_start.size} dims ({a.subspace}), step scale {np.sqrt(z_start.size / len(U)):.2f}")
        self.U = U
        best = dict(cost=base["cost"], align=base["align"], phys=base["phys"], bfm=base["bfm"],
                    head=base["head"], pos=base["pos"], mse=base["mse"],
                    z=z0.copy(), qpos=self.sc0.q[0].copy(), gen=-1, source=Z0_KEY)
        if a.z_start and a.best_from_start:
            best = start_best
        self.opt = single_latent_optimizer(a.algo, z_start.copy(), a.sigma, a.lr, a.pairs, a.seed, sampling=a.sampling,
                                           fitness=a.fitness, subspace=U, cem_elites=a.cem_elites, cem_sigma_min=a.cem_sigma_min)
        self.t0 = time.time()
        return self.opt, best, z_start

    def curve_row(self, loop, gen, sc):
        a, z, z0 = self.args, loop.z, self.z0
        return dict(gen=gen, evals=(gen + 1) * self.n_envs,
                    gen_best=float(sc.cost.min()), gen_mean=float(sc.cost.mean()),
                    best_so_far=loop.best["cost"], mean_z_cost="",
                    z_start=(a.z_start or Z0_KEY),
                    start_deg_from_z0=ang_pair(self.z_start, z0),
                    cos_z0=float(np.dot(z, z0) / (np.linalg.norm(z) * np.linalg.norm(z0))),
                    path_len_deg=round(loop.path_len, 2), net_deg=round(ang_pair(z0, z), 2),
                    sigma=loop.opt.step_sigma)

    def report(self, loop, gen, sc, sm, row):
        best, el = loop.best, time.time() - self.t0
        cma = self.args.algo == "cmaes"
        print(f"  gen {gen:5d}/{self.n_gens}  best {best['cost']:9.4f}  "
              f"gen_best {sc.cost.min():9.4f}  mean_z {sm.cost[0]:9.4f}  "
              f"cos(z,z0) {row['cos_z0']:.3f}  path {loop.path_len:5.0f} deg / net {row['net_deg']:4.0f} deg"
              + (f"  sigma {loop.opt.es.sigma:.4f}" if cma else "")
              + (f"  | best dtw {best['dtw']:.4f} done {best['done']:.2f} pace {best['pace']:.2f} heading_al {best['head_al']:.4f}"
                 f" fallen {best['fallen']:.3f}"
                 if "pace" in best else "") + f"  [{el / 60:.1f} min]", flush=True)

    def close(self):
        self.sim.close()
        if self.Bg is not None:
            self.env1.close()

    # ------------------------------------------------------------------ what came out
    def save(self, rec, loop):
        a, best, base, floor, z0, cfg = self.args, loop.best, self.base, self.floor, self.z0, self.cfg
        z, U = loop.z, self.U
        rec.best_z(best["z"])
        rec.npz("best.npz", qpos=best["qpos"].astype(np.float32),
                fps=cfg.control_fps, **({"dtw_path": best["path"]} if "path" in best else {}))
        # z0's rollout under the SAME init and env as the search (a falling rollout is chaotic: re-rendering it
        # elsewhere would not reproduce it)
        rec.npz(Z0_ROLLOUT, qpos=self.sc0.q[0].astype(np.float32),
                fps=cfg.control_fps, **({"dtw_path": self.base_path} if self.base_path is not None else {}))
        rec.z_trace(loop.trace_mean, loop.trace_best)          # gen -1 is the start latent
        rec.curve(loop.curve)
        exact = self.exact
        summary = {
            "clip": a.clip, "body": a.body, "xml": str(self.xml), "ref": str(self.ref_path),
            "objective": a.objective, "bfm_zero_vel": a.bfm_zero_vel,
            "fall_weight": a.fall_weight, "fall_head": a.fall_head, "init_vel_scale": a.init_vel_scale, "mse_no_xy": a.mse_no_xy,
            "lambda_align": self.lam_a, "lambda_phys": self.lam_p,
            "phys_weights": cfg.phys_weights, "phys_fall_ref": cfg.phys_fall_ref,
            # "generations" is the on-disk name for what the logs and figures now call steps
            "evals": self.n_gens * self.n_envs, "generations": self.n_gens, "pairs": a.pairs,
            "sigma": a.sigma, "lr": a.lr, "steps": a.steps, "algo": a.algo, "fitness": a.fitness, "sampling": a.sampling,
            "floor_friction": a.floor_friction,
            "path_len_deg": round(loop.path_len, 1), "net_deg": round(ang_pair(z0, z), 1),
            "obs_scale": a.obs_scale, "seed": a.seed, "init": a.init, "init_ref": a.init_ref, "frame_offset": a.frame_offset,
            "align_discount": a.discount, "heading_weight": a.heading_weight, "pos_weight": a.pos_weight,
            "pos_unit": a.pos_unit if exact is not None else "body", "pos_div": self.pos_scale,
            "anchor": a.anchor, "anchor_weight": a.anchor_weight,
            "subspace": a.subspace, "subspace_dim": None if U is None else len(U),
            **({"exact_clearance_vel": True} if getattr(a, "exact_clearance_vel", False) else {}),
            Z0_KEY: base, "reference_floor": floor,
            "best": {k: best[k] for k in ("cost", "align", "phys", "bfm", "gen", "source", "head", "pos", "mse",
                                          "dtw", "done", "pace", "head_al", "fallen") if k in best},
            "improvement_vs_origin_z": (base["cost"] - best["cost"]) / max(abs(base["cost"]), 1e-12),
            "cos_best_z0": float(np.dot(best["z"], z0)
                                 / (np.linalg.norm(best["z"]) * np.linalg.norm(z0))),
            "minutes": (time.time() - self.t0) / 60,
        }
        rec.summary(summary, indent=2)

        print(f"\n=== {a.objective}: {self.n_gens * self.n_envs} evals in "
              f"{summary['minutes']:.1f} min ===")
        print(f"{'':12s}{'cost':>10s}{'L_align':>10s}{'L_phys':>10s}")
        print(f"{'origin_z':12s}{base['cost']:10.4f}{base['align']:10.4f}{base['phys']:10.4f}")
        print(f"{'best z':12s}{best['cost']:10.4f}{best['align']:10.4f}{best['phys']:10.4f}"
              f"   ({100 * summary['improvement_vs_origin_z']:+.1f}%, from {best['source']} "
              f"at gen {best['gen']})")
        print(f"{'reference':12s}{floor['cost']:10.4f}{floor['align']:10.4f}{floor['phys']:10.4f}"
              f"   <- kinematic floor")
        print(f"cos(best z, z0) = {summary['cos_best_z0']:.4f}")
        if "pace" in best:
            for n_, b_ in (("origin_z", base), ("best z", best)):
                print(f"{n_:9s} dtw {b_['dtw']:.4f}  done {b_['done']:.2f}  pace {b_['pace']:.2f}  heading_al {b_['head_al']:.4f}  "
                      f"fallen {b_['fallen']:.3f}  "
                      f"(frame-by-frame bfm {b_['bfm']:.4f})")
        print(f"-> {self.out_dir}")


# ================================================================== robot_z_search.py

class RobotLatentProblem:
    """scripts/robot_z_search.py: single_z_search's latent search on a robot of model/robots.py (model/robot_env.py
    RobotVecEnv: SMPL-equivalent observations in, Motivo's SMPL action mapped onto the robot out), 2 * pairs robots from
    the clip's frame 0 mapped onto the robot; cost = main (bfm | mse | phc | dtw) + heading / position + anchor + IK."""

    def __init__(self, args):
        import mujoco

        from model import bfm_align
        from model.robot_map import RobotMap
        from model.robots import ROBOTS, out_dir, target_path

        from .costs import DTW012, DTWUnfinished, IKResidual, RobotMSE, RobotPoints, robot_ee_points
        from .obs import RawObs
        from .policy import MotivoPolicy
        from .rollout import BatchRollout
        from .sim import RobotSim

        a = self.args = args
        task, stem = a.clip.split("/")
        self.cfg = cfg = ROBOTS[a.robot]
        if a.objective == "g1mse":
            a.objective = "mse"
        tag = {"bfm": "", "dtw": "_dtw", "phc": "_phc"}.get(a.objective, "_g1mse" if a.robot == "g1" else "_mse")
        assert a.algo == "es" or not a.resume, "--resume is for --algo es only"
        self.out = out = Path(a.out) if a.out else out_dir(a.robot) / "search" / f"{stem}{tag}_s{a.seed}"
        out.mkdir(parents=True, exist_ok=True)
        self.n_envs = n_envs = 2 * a.pairs
        self.n_gens = max(a.evals // n_envs, 1)
        self.n_eps = a.pairs if a.sampling == "antithetic" else n_envs

        self.sim = RobotSim(a.robot, n_envs, fill=a.fill, gains=a.gains)          # spawn the workers before CUDA is touched
        rm = RobotMap(a.robot, fill=a.fill)
        n_link, n_jnt = len(rm.pairs) - 1, len(rm.robot_jnt)
        ARM = [i for i, (b, _) in enumerate(rm.pairs[1:]) if b.split("_")[-1] in ("Shoulder", "Elbow", "Hand")]
        from humenv import make_humenv
        self.ref = ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"][a.start_frame:]
        T = len(ref) - a.frame_offset                 # steps
        self.z0 = z0 = project_z(np.load(REPO / task_z0(a.clip)).reshape(-1).astype(np.float64))
        anchor = project_z(np.load(a.anchor).reshape(-1).astype(np.float64)) if a.anchor else z0
        self.policy = policy = MotivoPolicy.load(device=a.device, eval_mode=False)   # this script never called eval()
        model = policy.model
        adult, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
        fk = adult.unwrapped.model; dk = mujoco.MjData(fk)
        VEL = slice(214, 358)                         # humenv proprio: local_body_vel + local_body_ang_vel
        Og = bfm_align.obs_from_qpos(adult, ref)
        Bg_plain = policy.embed(Og)[a.frame_offset:]
        if a.bfm_zero_vel:
            Og = Og.copy(); Og[:, VEL] = 0
        Bg = bfm_align.embed(model, Og, a.device)[a.frame_offset:]
        self.robot_ref = robot_ref = np.load(target_path(a.robot, stem))["qpos"][a.start_frame:]
        floor = json.loads((target_path(a.robot, stem).parent / "summary.json").read_text())["bfm_floor"]
        fc = REPO / "outputs/map_error/fill_compare_motivo.json"            # the ridge fill's floor (scripts/fit_smpl_fill.py)
        if a.fill == "ridge" and fc.exists():
            floor = json.loads(fc.read_text()).get(a.robot, {}).get("ridge", {}).get("bfm_full_per_clip", {}).get(stem, floor)
        self.floor = floor
        m_robot = mujoco.MjModel.from_xml_path(str(REPO / cfg["xml"]))
        root = m_robot.body(cfg["root"]).id
        ee = robot_ee_points(cfg, rm, m_robot)             # per point: (kind, id, body-frame offset for "end")
        robot_goal = robot_ref[a.frame_offset:a.frame_offset + T]
        mse = RobotMSE(RobotPoints(m_robot, root, ee), robot_goal, no_xy=a.mse_no_xy)
        self.phc = None
        if a.objective == "phc":
            from model.phc_reward import PHCCost
            self.phc = PHCCost(m_robot, robot_ref[max(a.frame_offset - 1, 0)], robot_goal, scale=rm.scale)
            print(f"phc: {m_robot.nbody - 1} robot bodies against the robot target, lengths / s = {rm.scale:.6f}", flush=True)

        from .geometry import smpl_heading
        ref_h, ref_xy = smpl_heading(fk, ref[a.frame_offset:], dk, batched=True), ref[a.frame_offset:][..., :2]

        # 2 * pairs robots from ref[0] mapped onto the robot (at rest, or moving: --init-vel reference), T steps, one latent each
        self.batch = BatchRollout(policy, self.sim, RawObs(), n_envs, T, record=("obs", "qpos_smpl", "qpos", "fz"))

        # ---- the cost: main + heading / position (bfm, phc, dtw) + anchor + IK, as score() summed them
        plain = BFMSimilarity(policy, Bg_plain, name="bfm_plain")
        bfm = BFMSimilarity(policy, Bg, zero_vel=True) if a.bfm_zero_vel else plain   # without it the plain one is the bfm
        heading, position = Heading(fk, ref_h, d=dk, batched=True, src="qs"), Position(ref_xy, src="qs")
        anchor_t, ik = Anchor(anchor), IKResidual(n_link, n_jnt, ARM)
        hw, pw = a.heading_weight, a.pos_weight
        self.dtw = None
        if a.objective == "mse":
            main, terms = mse, []
        elif a.objective == "phc":                    # every slot starts from the same pose (checked)
            main = PHCReward(self.phc, lambda ctx: ctx.ro.start.qpos[0], same_start=True)
            terms = [(hw, heading), (pw, position)]
        elif a.objective == "dtw":
            self.dtw = main = DTW012(bfm, Bg, a.dtw_lambda)
            terms = [(a.dtw_lambda, DTWUnfinished(main)), (hw, AlignedHeading(heading, main.path, masked=False)), (pw, position)]
        else:
            main, terms = bfm, [(hw, heading), (pw, position)]
        terms += [(a.anchor_weight, anchor_t), (a.ik_weight, ik)]
        self.cost = CompositeCost(main, terms, log=[plain, bfm, heading, position, anchor_t, mse, ik])

    # ------------------------------------------------------------------ scoring
    def score(self, zs):
        """zs (n, 256) -> Scored: cost, parts (bfm, bfm_plain, mse, head, pos, anchor_dist, ik, ik_arm, atlim, dtw, done,
        head_al, path; qg / qs / fz: the robot qpos, its SMPL-equivalent, the IK numbers per step)"""
        import torch
        a = self.args
        with torch.no_grad():
            ro = self.batch.run(zs, init_qpos=self.ref[0], init_next=self.ref[1] if a.init_vel == "reference" else None)
            ctx = ScoreContext(z=zs, ro=ro, q=ro.qpos, qs=ro.qpos_smpl, fz=ro.fz, obs32=ro.obs.astype(np.float32))
            cost, parts = self.cost(ctx)
        if not a.bfm_zero_vel:
            parts["bfm"] = parts["bfm_plain"]
        if self.dtw is None:
            n_, T_ = ro.obs.shape[:2]
            parts["dtw"] = parts["done"] = parts["head_al"] = np.full(n_, np.nan)
            parts["path"] = np.zeros((n_, T_), dtype=np.int64)
        parts.update(qg=ro.qpos, qs=ro.qpos_smpl, fz=ro.fz)
        return Scored(cost, parts, ro.qpos, ro=ro)

    def record(self, sc, i, z, gen, source):
        r = sc.parts
        return dict(cost=float(sc.cost[i]), bfm=float(r["bfm"][i]), bfm_plain=float(r["bfm_plain"][i]), mse=float(r["mse"][i]),
                    head=float(r["head"][i]), pos=float(r["pos"][i]),
                    ik=float(r["ik"][i]), ik_arm=float(r["ik_arm"][i]), atlim=float(r["atlim"][i]),
                    dtw=float(r["dtw"][i]), done=float(r["done"][i]), head_al=float(r["head_al"][i]), path=r["path"][i].copy(),
                    z=z.copy(), qg=r["qg"][i].copy(), qs=r["qs"][i].copy(), fz=r["fz"][i].copy(), gen=gen, source=source)

    def al(self, b):
        return f"  dtw {b['dtw']:.4f} done {b['done']:.2f} heading_al {b['head_al']:.4f}" if self.args.objective == "dtw" else ""

    def baseline(self):
        a, z0 = self.args, self.z0
        print(f"{self.cfg['name']} latent search on {a.clip}, objective {a.objective}: {self.n_gens} steps x {self.n_envs} rollouts = "
              f"{self.n_gens * self.n_envs} evals, "
              f"sigma {a.sigma}, lr {a.lr}, heading {a.heading_weight}, pos {a.pos_weight}, anchor {a.anchor_weight}, "
              f"ik {a.ik_weight}", flush=True)
        self.t0 = t0 = time.time()
        r0 = self.score(copies(z0, self.n_envs))
        spread = float(np.abs(r0.parts["qg"] - r0.parts["qg"][:1]).max())
        self.base = base = self.record(r0, 0, z0, -1, Z0_KEY)
        print(f"  origin_z:{self.al(base)} cost {base['cost']:.4f}  bfm {base['bfm']:.4f} (plain {base['bfm_plain']:.4f})  mse {base['mse']:.4f}  heading {base['head']:.4f}  pos {base['pos']:.2f}"
              f"  ik {base['ik']:.2f} deg (arms {base['ik_arm']:.2f}, at limit {100 * base['atlim']:.1f}%)"
              f"   (16 copies identical to {spread:.1e}; {time.time() - t0:.0f} s per step)", flush=True)
        return base

    def eval_latents(self, path, rec):
        """--eval-z: roll out the given latents with this run's cost (batches of 2 * pairs padded with z0)"""
        from .rollout import BatchRollout
        a, z0, n_envs, base = self.args, self.z0, self.n_envs, self.base
        Z = np.load(path).reshape(-1, z0.size).astype(np.float64); rows, Q = [], []
        for i0 in range(0, len(Z), n_envs):
            chunk = Z[i0:i0 + n_envs]; r = self.score(BatchRollout.padded(chunk, z0, n_envs))      # padded with z0
            Q.append(r.parts["qg"][:len(chunk)])
            col = lambda x: r.cost if x == "cost" else r.parts[x]                                  # noqa: E731
            rows += [(i0 + k, *(float(col(x)[k]) for x in ("cost", "bfm", "head", "pos", "anchor_dist", "ik", "mse")))
                     for k in range(len(chunk))]
        rec.eval_csv(["index", "cost", "bfm", "head", "pos", "anchor_dist", "ik", "mse"], rows, style="robot")
        rec.json("eval_base.json", {k: base[k] for k in ("cost", "bfm", "head", "pos", "ik", "mse")})
        if a.eval_save_qpos:
            rec.npz("eval_qpos.npz", qpos=np.concatenate(Q).astype(np.float32))
        self.sim.close(); print(f"eval: {len(rows)} latents scored -> {rec.path('eval.csv')}")

    # ------------------------------------------------------------------ the search (and --resume)
    def start(self):
        """the optimiser (crossembodiment/optim.py; ES owns the RNG --resume restores) and the loop's starting state:
        (opt, dict(best, curve, trace_mean, trace_best, path_len, gen0)). --resume continues the search saved in a.resume
        up to --evals in total: exact (bit-identical to one longer run) when it has state.npz, else from its last search
        point with the noise stream advanced past its steps (only Adam's moments restart)."""
        import csv
        from .optim import single_latent_optimizer
        a, z0, base = self.args, self.z0, self.base
        opt = single_latent_optimizer(a.algo, z0.copy(), a.sigma, a.lr, a.pairs, a.seed, sampling=a.sampling,
                                      cem_elites=a.cem_elites, cem_sigma_min=a.cem_sigma_min)
        st = dict(best=base, curve=[], trace_mean=[z0.copy()], trace_best=[z0.copy()], path_len=0.0, gen0=0)
        self.t_adam0, self.exact = 0, None             # Adam steps taken before this run; resumed from state.npz
        if a.resume:
            rd = Path(a.resume) if Path(a.resume).is_absolute() else REPO / a.resume
            ps = json.loads((rd / "summary.json").read_text())
            same = {k: (ps[k], v_) for k, v_ in (("clip", a.clip), ("body", a.robot), ("seed", a.seed), ("pairs", a.pairs),
                                                 ("sigma", a.sigma), ("lr", a.lr), ("ik_weight", a.ik_weight),
                                                 ("objective", a.objective)) if ps[k] != v_}
            assert not same, f"--resume: settings differ from {rd}: {same}"
            ps_sampling = ps.get("sampling", "antithetic")     # runs before 11/04 record none: all antithetic
            assert ps_sampling == a.sampling, f"--resume: {rd} ran --sampling {ps_sampling}, this run has --sampling {a.sampling}"
            assert abs(ps[Z0_KEY]["cost"] - base["cost"]) < 1e-9, f"--resume: z0 costs {base['cost']} now, {ps[Z0_KEY]['cost']} then"
            gen0 = st["gen0"] = ps["generations"]
            assert self.n_gens > gen0, f"--resume: {rd} already has {gen0} steps; --evals is the total budget"
            st["curve"] = list(csv.DictReader(open(rd / "curve.csv")))
            zt = np.load(rd / "z_trace.npz")
            st["trace_mean"], st["trace_best"] = list(zt["z_mean"].astype(np.float64)), list(zt["z_best"].astype(np.float64))
            bq = np.load(rd / "best.npz")
            best = dict(**{k: ps["best"][k] for k in ("cost", "bfm", "mse", "head", "pos", "ik", "ik_arm", "atlim", "gen", "source")},
                        bfm_plain=ps["best"].get("bfm_plain", ps["best"]["bfm"]),   # runs before 11/02 have no bfm_plain (= bfm there)
                        z=np.load(rd / "best_z.npy").reshape(-1).astype(np.float64), qg=bq["qpos"], qs=bq["qpos_smpl"], fz=bq["fz"])
            self.exact = (rd / "state.npz").exists()
            if self.exact:
                S = np.load(rd / "state.npz")
                opt.load_state_dict(dict(z=S["z"], m=S["m"], v=S["v"], t=int(S["t_adam"]), rng=S["rng"]))
                self.t_adam0, st["path_len"] = int(S["t_adam"]), float(S["path_len"])
                best["z"] = S["best_z"]
            else:
                opt.z, st["path_len"] = st["trace_mean"][-1].copy(), ps["path_len_deg"]
                opt.skip(gen0)                                 # the noise the earlier steps drew
            st["best"] = best
            print(f"  resumed {rd.name} at step {gen0} ({'exact: Adam state restored' if self.exact else 'from its last search point, Adam restarted'}), "
                  f"best so far {best['cost']:.4f}", flush=True)
        self.opt, self.gen0 = opt, st["gen0"]
        return opt, st

    def curve_row(self, loop, gen, sc):
        z, z0, best = loop.z, self.z0, loop.best
        return dict(gen=gen, evals=(gen + 1) * self.n_envs, gen_best=float(sc.cost.min()), gen_mean=float(sc.cost.mean()),
                    best_so_far=best["cost"], best_bfm=best["bfm"], best_done=best.get("done", ""), best_ik=best["ik"], mean_z_cost="",
                    cos_z0=float(z @ z0 / (np.linalg.norm(z) * np.linalg.norm(z0))),
                    path_len_deg=round(loop.path_len, 2), net_deg=round(ang_pair(z0, z), 2),
                    sigma=loop.opt.step_sigma)

    def report(self, loop, gen, sc, sm, row):
        best, z, al = loop.best, loop.z, self.al
        print(f"  gen {gen:4d}/{self.n_gens}  best {best['cost']:.4f} ({al(best)[2:] + ', ' if al(best) else ''}bfm {best['bfm']:.4f} (plain {best['bfm_plain']:.4f}), mse {best['mse']:.4f}, heading {best['head']:.4f}, "
              f"pos {best['pos']:.2f}, ik {best['ik']:.2f} deg)  gen_best {sc.cost.min():.4f}  mean_z {sm.cost[0]:.4f}  "
              f"|z-z0| {np.linalg.norm(z - self.z0):.2f}  [{(time.time() - self.t0) / 60:.1f} min]", flush=True)

    def close(self):
        self.sim.close()

    def save(self, rec, loop):
        from .robot_rollout import motion
        a, best, base, z0, opt, robot_ref = self.args, loop.best, self.base, self.z0, loop.opt, self.robot_ref
        n_gens, n_envs, gen0 = self.n_gens, self.n_envs, self.gen0
        rec.best_z(best["z"])
        # everything --resume needs to continue this search exactly (ES's Adam and RNG; the other optimisers have none to
        # restore: their Adam moments stay 0 and the RNG is the one they drew from)
        es = getattr(opt, "adam", None) is not None
        rng = opt.rng if hasattr(opt, "rng") else np.random.default_rng(a.seed)
        rec.state(z=loop.z, m=opt.adam.m if es else np.zeros_like(z0), v=opt.adam.v if es else np.zeros_like(z0),
                  t_adam=n_gens - gen0 + self.t_adam0, path_len=loop.path_len, best_z=best["z"],
                  rng=json.dumps(rng.bit_generator.state))
        for f, b in (("best.npz", best), (Z0_ROLLOUT, base)):
            rec.npz(f, qpos=b["qg"].astype(np.float32), qpos_smpl=b["qs"].astype(np.float32),
                    fz=b["fz"].astype(np.float32), fps=30, **({"dtw_path": b["path"]} if a.objective == "dtw" and "path" in b else {}))
        rec.z_trace(loop.trace_mean, loop.trace_best)
        rec.curve(loop.curve, columns_from=-1, restval="")     # [-1]: a resumed older run's rows lack newer columns
        keys = ("cost", "bfm", "bfm_plain", "mse", "head", "pos", "ik", "ik_arm", "atlim", "gen", "source") + (("dtw", "done", "head_al") if a.objective == "dtw" else ())
        summ = dict(clip=a.clip, body=a.robot, evals=n_gens * n_envs, generations=n_gens, pairs=a.pairs, sampling=a.sampling, sigma=a.sigma, lr=a.lr,
                    algo=a.algo, **({"cem_elites": a.cem_elites, "cem_sigma_min": opt.sigma_min} if a.algo == "cem" else {}),
                    seed=a.seed, **({"dtw_lambda": a.dtw_lambda} if a.objective == "dtw" else {}), bfm_zero_vel=a.bfm_zero_vel, mse_no_xy=a.mse_no_xy, init_vel=a.init_vel, frame_offset=a.frame_offset, start_frame=a.start_frame, heading_weight=a.heading_weight, pos_weight=a.pos_weight, anchor=a.anchor or "z0",
                    anchor_weight=a.anchor_weight, ik_weight=a.ik_weight, gains=a.gains or self.cfg["gains_json"], objective=a.objective, fill=a.fill,
                    bfm_floor_of_map=self.floor,
                    origin_z={**{k: base.get(k) for k in keys}, "motion": motion(base["qg"], robot_ref[a.frame_offset:])},   # Z0_KEY
                    best={**{k: best.get(k) for k in keys}, "motion": motion(best["qg"], robot_ref[a.frame_offset:])},
                    improvement_vs_origin_z=(base["cost"] - best["cost"]) / abs(base["cost"]),
                    dist_best_z0=float(np.linalg.norm(best["z"] - z0)), path_len_deg=round(loop.path_len, 1),
                    resumed_from=a.resume, resumed_at=gen0 if a.resume else None, resume_exact=self.exact,
                    minutes=(time.time() - self.t0) / 60)
        rec.summary(summ, indent=2)
        al = self.al
        print(f"\n=== {n_gens * n_envs} evals in {summ['minutes']:.1f} min ===")
        for n, b in (("origin_z", base), ("best z", best)):
            mo = summ["origin_z" if n == "origin_z" else "best"]["motion"]
            print(f"{n:9s}{al(b)} cost {b['cost']:.4f}  bfm {b['bfm']:.4f} (plain {b['bfm_plain']:.4f})  mse {b['mse']:.4f}  heading {b['head']:.4f}  pos {b['pos']:.2f}  ik {b['ik']:.2f} deg "
                  f"(arms {b['ik_arm']:.2f}, at limit {100 * b['atlim']:.1f}%) | fallen {mo['fallen']}"
                  f"  travel {mo['travel']:.1f} m  end-point off {mo['end_xy']:.2f} m  pelvis {mo['pelvis']:.3f} m")
        print(f"|best z - z0| = {summ['dist_best_z0']:.2f}   -> {self.out}")


# ================================================================== lso_search.py / robot_lso_search.py

class _SequenceRows:
    """the curve rows / budgets shared by the two latent-sequence problems"""

    def best_start(self, start_key, q_start):
        return dict(cost=self.res[start_key]["cost"], z=self.opt.mu.copy(), it=-1, src="start", q=q_start)

    def make_optimizer(self, mu):
        from .optim import LSO, TrajectoryAdaptation
        a = self.args
        if a.method == "ta":
            return TrajectoryAdaptation(mu, a.sigma, a.n, a.iters, seed=a.seed, beta1=a.ta_beta1, beta2=a.ta_beta2, temp=a.ta_temp)
        return LSO(mu, a.sigma, a.lr, a.n, seed=a.seed, beta=a.beta, gamma=a.gamma, betas=a.betas, rank=a.rank,
                   antithetic=getattr(a, "antithetic", False))

    def mu_move(self, loop):
        return float(np.linalg.norm(16 * (loop.opt.mu - loop.trace[-1]), axis=1).mean())


class HumanSequenceProblem(_SequenceRows):
    """scripts/lso_search.py: BFMTrack's LSO (or BFM-Zero's TA) on an SMPL-topology body: n + 1 slots (slot 0 = the mean
    sequence), the per-step reward cos(B(s_t+1), B(g_t+1)) (ExactObs, root scale --scale) minus heading / position, or
    the --objective mse / phc reward, or the time-aligned --dtw reward."""

    def __init__(self, a):
        import mujoco

        from model import bfm_align

        from .obs import ExactObs, lso_scale_kw
        from .policy import MotivoPolicy
        from .rollout import BatchRollout
        from .sim import HumenvSim
        self.args = a
        assert not a.antithetic or a.n % 2 == 0, "--antithetic needs an even --n"
        assert a.dtw or a.steps_scale == 1.0, "--steps-scale needs --dtw (the frame-by-frame reward has one frame per step)"
        assert not (a.dtw and a.objective in ("mse", "phc")), "--objective mse / phc are frame by frame"
        self.out = out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        xml = Path(a.xml) if a.xml else REPO / "assets/robots_torque" / a.body / "robot_torque_full.xml"
        self.ref = ref = np.load(REPO / a.ref)["qpos"]; self.T = T = len(ref); self.S = S = T - 1    # g_1 .. g_T-1 are the goals
        self.SR = SR = int(round(a.steps_scale * S))                           # control steps rolled out
        kw = lso_scale_kw(a.scale)                                             # None: ExactObs's default (the child's s), any body
        self.n_envs = n_envs = a.n + 1                                         # slot 0: the mean sequence, then a.n samples
        # ExactObs computed in the env workers, after FloorFrictionWrapper when given; max_episode_steps: humenv's TimeLimit
        # (300 steps by default) would auto-reset clips longer than 10 s mid-rollout
        mapper = ExactObs(xml, REPO / "assets/robots/adult/robot.xml", **kw)
        self.sim = HumenvSim.make(xml, n_envs, wrappers=mapper.wrappers(), floor_friction=a.floor_friction, max_episode_steps=SR + 1)
        self.policy = policy = MotivoPolicy.load(device=a.device); model = policy.model     # facebook/metamotivo-M-1, eval()
        self.fk = fk = mujoco.MjModel.from_xml_path(str(xml)); nv = fk.nv
        exact = mapper.build()
        self.pos_scale = pos_scale = exact.scale if a.pos_unit == "adult" else 1.0      # summary.json pos_div
        phc = None
        if a.objective == "phc":
            from model.phc_reward import PHCCost
            phc = PHCCost(fk, ref[0], ref[1:], scale=pos_scale)
        vr, ro = np.zeros(nv), []
        for t in range(T):
            if t:
                mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, ref[t - 1], ref[t])
            ro.append(exact(ref[t], vr))
        self.Bg = Bg = policy.embed(np.stack(ro))[1:]                          # (S, 256): goal of step t is frame t+1
        Bstart = Bg
        VEL = slice(214, 358)                    # humenv proprio: local_body_vel + local_body_ang_vel
        Bc = Bg                                  # the reference side of the cost
        if a.bfm_zero_vel:
            ro0 = np.stack(ro); ro0[:, VEL] = 0
            Bc = bfm_align.embed(model, ro0, a.device)[1:]
        if a.init_pace != 1.0:                       # goal latents of the reference played at init_pace x speed
            from .retiming import resample
            rs, vr, ro2 = resample(ref.astype(np.float64), a.init_pace), np.zeros(nv), []
            for t in range(len(rs)):
                if t:
                    mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, rs[t - 1], rs[t])
                ro2.append(exact(rs[t], vr))
            Bstart = bfm_align.embed(model, np.stack(ro2), a.device)[1:]
        self.Bstart = Bstart[np.minimum(np.arange(SR), len(Bstart) - 1)]      # (SR, 256), the last goal held
        bfm = BFMSimilarity(policy, Bc, zero_vel=a.bfm_zero_vel, seq_embed="backward_map")
        bfm.reference_tensor()
        self.v0 = v0 = np.zeros(nv)
        if a.init_vel_scale:
            mujoco.mj_differentiatePos(fk, v0, 1.0 / 30, ref[0], ref[1]); v0 *= a.init_vel_scale
        # one rollout of n_envs latent sequences from ref[0] moving at v0, no get_obs after set_physics (the start observation
        # is ExactObs(ref[0], v0)); 16 * Z on the unit sphere -> float32
        self.batch = BatchRollout(policy, self.sim, mapper, n_envs, SR, z_scale=16.0, fetch_start_obs=False, record=("qpos", "obs"))
        dk = mujoco.MjData(fk)
        from .geometry import smpl_heading
        ref_h, ref_xy = smpl_heading(fk, ref[1:], dk, batched=True), ref[1:, :2]
        heading = Heading(fk, ref_h, d=dk, batched=True)
        ref_pose = None
        if a.objective == "mse":
            import model.kinematics as kin
            ref_pose = kin.batch_forward_pose(fk, ref[1:], kin.EE_BODIES + [kin.ROOT_BODY])

        # ---- the per-step reward (crossembodiment/costs.py, form per_step)
        if a.dtw:
            main = SlowDTW(bfm, Bc, heading=heading, fall=Fall(fk, a.fall_head, d=dk), heading_weight=a.heading_weight,
                           fall_weight=a.fall_weight)
            self.cost = CompositeCost(main, form="per_step")
        else:
            if a.objective == "mse":
                main = TrackingMSE(fk, ref[1:], root_xy=not a.mse_no_xy, ref_pose=ref_pose)
            elif phc is not None:                                               # every rollout starts at ref[0]
                main = PHCReward(phc, lambda ctx: ref[0])
            else:
                main = bfm
            terms = [(a.heading_weight, heading), (a.pos_weight, Position(ref_xy, pos_scale))] \
                if a.heading_weight or a.pos_weight else []
            self.cost = CompositeCost(main, terms, log=[bfm], form="per_step")

    def score(self, Z):
        """Z: (n_envs, SR, 256) on the unit sphere -> Scored: r per-step reward (n_envs, SR), q qpos (n_envs, SR, nq), parts"""
        import torch
        with torch.no_grad():
            ro_ = self.batch.run(Z, init_qpos=self.ref[0], init_qvel=self.v0)
            ctx = ScoreContext(z=Z, ro=ro_, q=ro_.qpos)
            r, parts = self.cost(ctx)
        return Scored(parts=parts, q=ro_.qpos, r=r, ro=ro_)

    def terms(self, sc, i):
        """objective (1 - mean per-step reward) and its parts for slot i"""
        p = sc.parts
        d = dict(cost=float(1 - sc.r[i].sum() / self.S), bfm=float(p["bfm"][i]))
        if "mse_frames" in p:
            d["mse_obj"] = float(p["mse_frames"][i])
        if "dtw" in p:
            d.update({k: float(p[k][i]) for k in ("dtw", "done", "pace", "fallen")}, head=float(p["head"][i]),
                     pos_m=float("nan"))
        elif "head" in p:
            d.update(head=float(p["head"][i]), pos_m=float(p["pos"][i]))
        return d

    @staticmethod
    def line(k, v):
        return (f"  {k:28s} cost {v['cost']:.4f} (bfm {v['bfm']:.4f}" + (f", heading {v['head']:.3f}, pos {v['pos_m']:.2f} m"
                if "head" in v else "") + (f", dtw {v['dtw']:.3f} done {v['done']:.2f} pace {v['pace']:.2f} fallen {v['fallen']:.3f}"
                if "dtw" in v else "") + f")  joint rms {v['joint_rms_deg']:.1f} deg  ee {v['ee_rms_cm']:.1f} cm  "
                f"root {v['root_rms_cm']:.1f} cm  end-point off {v['end_off_m']:.2f} m")

    def report_q(self, q):
        from model import losses
        ref = self.ref
        tot, terms = losses.tracking_mse(self.fk, q, ref[1:])
        return dict(mse=float(tot), joint_rms_deg=float(np.degrees(np.sqrt(terms["pose"]))),
                    ee_rms_cm=float(100 * np.sqrt(terms["ee"])), root_rms_cm=float(100 * np.sqrt(terms["root"])),
                    end_off_m=float(np.linalg.norm(q[-1, :2] - ref[-1, :2])), pelvis_min_m=float(q[:, 2].min()))

    def baseline(self):
        """the zero-shot references, rolled out in one batch: B(g_t) (the LSO start) and Motivo's ER windows"""
        from .optim import unit_sphere as unit
        a, S, SR, Bg = self.args, self.S, self.SR, self.Bg
        mu = unit(self.Bstart.astype(np.float64))
        base = {"B(g_t) (LSO start)": mu}
        hold = np.minimum(np.arange(SR), S - 1)
        for L in a.er_windows:
            base[f"ER L={L} (Motivo zero-shot)"] = unit(np.stack([Bg[t:t + L].mean(0) for t in range(S)]).astype(np.float64))[hold]
        Z = copies(mu, self.n_envs)
        for i, z in enumerate(base.values()):
            Z[i] = z
        self.t0 = time.time(); sc = self.score(Z)
        q = sc.q
        self.res = {k: dict(**self.terms(sc, i), **self.report_q(q[i])) for i, k in enumerate(base)}
        tag = lambda k: "q_Bg" if k.startswith("B(") else "q_ER" + k.split("=")[1].split(" ")[0]      # noqa: E731
        self.keep = {tag(k): q[i].astype(np.float32) for i, k in enumerate(base)}
        self.base, self.q_base = base, q
        print(f"{a.ref}: {self.T} frames, {S} steps ({SR} rolled out), {a.n} sequences per iteration, {a.iters} iterations "
              f"({time.time() - self.t0:.0f} s per batch)")
        for k, v in self.res.items():
            print(self.line(k, v), flush=True)

    def start(self):
        """the optimiser (LSO from B(g_t), or TA from ER L=8) and the start best (the start sequence's rollout)"""
        a = self.args
        start_key = "B(g_t) (LSO start)" if a.method == "lso" else "ER L=8 (Motivo zero-shot)"
        mu = self.base["B(g_t) (LSO start)"]
        if a.method == "ta":
            assert start_key in self.base and not a.antithetic and not a.rank, "--method ta: starts from ER L=8; no --antithetic / --rank"
            mu = self.base[start_key].copy()
        self.opt = self.make_optimizer(mu)
        return self.opt, self.best_start(start_key, self.q_base[list(self.base).index(start_key)].copy())

    def curve_row(self, loop, it, sc, costs):
        return dict(iter=it, mean_seq_cost=float(costs[0]), mean_seq_bfm=float(sc.parts["bfm"][0]),
                    best_sample_cost=float(costs[1:].min()), sample_mean_cost=float(costs[1:].mean()), best_so_far=loop.best["cost"],
                    mu_move=self.mu_move(loop), sec=time.time() - self.t0)

    def report(self, loop, it, sc, costs, row):
        p = sc.parts
        if self.args.method == "ta":
            print(f"  iter {it:3d}  mean sequence cost {costs[0]:.4f}  samples best {costs[1:].min():.4f} mean {costs[1:].mean():.4f}"
                  f"  top weight {loop.opt.w.max():.3f}  [{row['sec'] / 60:.1f} min]", flush=True)
            return
        bi = 1 + int(np.argmin(costs[1:])); dl = (f" | best sample dtw {p['dtw'][bi]:.3f} pace {p['pace'][bi]:.2f} "
                                                    f"fallen {p['fallen'][bi]:.3f}" if "dtw" in p else "")
        print(f"  iter {it:3d}  mean sequence cost {costs[0]:.4f} (bfm {p['bfm'][0]:.4f})  samples best {costs[1:].min():.4f} mean {costs[1:].mean():.4f}"
              f"  |d mu_t| {row['mu_move']:.3f}{dl}  [{row['sec'] / 60:.1f} min]", flush=True)

    def finish(self, loop):
        """the final mean sequence and the best one, re-rolled; then the env closes"""
        a, keep, res, best = self.args, loop.keep, self.res, loop.best
        sc = self.score(copies(loop.opt.mu, self.n_envs))                       # the final mean sequence
        res["LSO final mean"] = dict(**self.terms(sc, 0), **self.report_q(sc.q[0])); keep["q_LSO"] = sc.q[0].astype(np.float32)
        if a.iters in a.budgets:
            keep[f"q_mean_it{a.iters}"] = sc.q[0].astype(np.float32)
        if "path" in sc.parts:
            keep["path_LSO"] = sc.parts["path"][0]
        sb = self.score(copies(best["z"], self.n_envs))
        res["LSO best (re-rolled)"] = dict(**self.terms(sb, 0), it=best["it"], src=best["src"], **self.report_q(sb.q[0]))
        keep["q_LSO_best"] = sb.q[0].astype(np.float32)
        if "path" in sb.parts:
            keep["path_LSO_best"] = sb.parts["path"][0]
        self.sim.close()

    def save(self, rec, loop):
        a, best, res, T = self.args, loop.best, self.res, self.T
        rec.curve(loop.curve)
        rec.mu_trace(loop.trace, loop.traced_iters(a.trace), best["z"])
        rec.rollouts(self.ref, loop.keep)
        rec.summary(dict(ref=a.ref, body=a.body, frames=T, n=a.n, iters=a.iters, sigma=a.sigma,
                         lr=a.lr, gamma=a.gamma, beta=a.beta, seed=a.seed, heading_weight=a.heading_weight,
                         pos_weight=a.pos_weight, pos_unit=a.pos_unit, pos_div=self.pos_scale, antithetic=a.antithetic, rank=a.rank, floor_friction=a.floor_friction,
                         steps_scale=a.steps_scale, init_pace=a.init_pace, dtw=a.dtw, init_vel_scale=a.init_vel_scale,
                         fall_weight=a.fall_weight, fall_head=a.fall_head,
                         bfm_zero_vel=a.bfm_zero_vel, objective=a.objective, mse_no_xy=a.mse_no_xy,
                         method=a.method, ta_beta1=a.ta_beta1, ta_beta2=a.ta_beta2, ta_temp=a.ta_temp, betas=list(a.betas),
                         budgets=a.budgets, best_cost_at=loop.budget_cost, best_it=best["it"], best_src=best["src"],
                         sample_rollouts=a.iters * a.n, results=res,
                         minutes=(time.time() - self.t0) / 60), indent=1)
        print("\n" + "\n".join(self.line(k, v) for k, v in res.items()))
        print(f"-> {self.out}  ({(time.time() - self.t0) / 60:.1f} min)")


class RobotSequenceProblem(_SequenceRows):
    """scripts/robot_lso_search.py: lso_search's LSO / TA on a robot (RobotVecEnv as robot_z_search drives it); per-step
    reward cos(B(o_t+1), B(g_t+1)) (or PHC's against the robot target) - heading - position - IK, always all four."""

    def __init__(self, a):
        import mujoco

        from model import bfm_align
        from model.robot_map import RobotMap
        from model.robots import ROBOTS, target_path

        from .costs import IKResidual
        from .obs import RawObs
        from .policy import MotivoPolicy
        from .rollout import BatchRollout
        from .sim import RobotSim
        self.args = a
        self.out = out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        self.ref = ref = np.load(REPO / a.ref)["qpos"].astype(np.float64); self.T = T = len(ref); self.S = S = T - 1
        self.n_envs = n_envs = a.n + 1
        self.sim = RobotSim(a.robot, n_envs, fill=a.fill, gains=a.gains)          # spawn the workers before CUDA is touched
        rm = RobotMap(a.robot, fill=a.fill); n_link = len(rm.pairs) - 1
        from humenv import make_humenv
        self.policy = policy = MotivoPolicy.load(device=a.device)    # facebook/metamotivo-M-1, eval()
        adult, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
        fk = adult.unwrapped.model; dk = mujoco.MjData(fk)
        self.Bg = Bg = policy.embed(bfm_align.obs_from_qpos(adult, ref))[1:]                          # (S, 256): goal of step t = frame t+1
        bfm = BFMSimilarity(policy, Bg, seq_embed="embed")
        bfm.reference_tensor()
        self.robot_name = ROBOTS[a.robot]["name"]
        self.gains = a.gains or ROBOTS[a.robot].get("gains_json")
        phc = None
        if a.objective == "phc":
            from model.phc_reward import PHCCost
            tgt = Path(a.target) if a.target else target_path(a.robot, Path(a.ref).stem)
            robot_ref = np.load(tgt if tgt.is_absolute() else REPO / tgt)["qpos"].astype(np.float64)
            assert len(robot_ref) == T, (len(robot_ref), T)
            m_robot = mujoco.MjModel.from_xml_path(str(REPO / ROBOTS[a.robot]["xml"]))
            phc = PHCCost(m_robot, robot_ref[0], robot_ref[1:], scale=rm.scale)
            print(f"phc: {m_robot.nbody - 1} robot bodies against {tgt}, lengths / s = {rm.scale:.6f}", flush=True)
        self.phc = phc
        from .geometry import smpl_heading
        ref_h, ref_xy = smpl_heading(fk, ref[1:], dk), ref[1:][..., :2]           # humenv's heading, frame by frame

        # n + 1 robots (slot 0: the mean sequence) from ref[0] mapped onto the robot, S steps, latent z_t = 16 * Z[:, t]
        self.batch = BatchRollout(policy, self.sim, RawObs(), n_envs, S, z_scale=16.0, record=("obs", "qpos_smpl", "qpos", "fz"))
        main = bfm if phc is None else PHCReward(phc, lambda ctx: ctx.ro.start.qpos[0], same_start=True)
        self.n_jnt = len(rm.robot_jnt)
        self.cost = CompositeCost(main, [(a.heading_weight, Heading(fk, ref_h, d=dk, src="qs")),
                                         (a.pos_weight, Position(ref_xy, src="qs")),
                                         (a.ik_weight, IKResidual(n_link, self.n_jnt))], log=[bfm], form="per_step")

    PARTS = ("bfm", "head", "pos", "ik_deg", "atlim", "pelvis_min", "phc")

    def score(self, Z):
        """Z: (n_envs, S, 256) on the unit sphere -> Scored: r per-step reward (n_envs, S), q / qs / fz the trajectories"""
        import torch
        a = self.args
        with torch.no_grad():
            ro = self.batch.run(Z, init_qpos=self.ref[0], init_next=self.ref[1] if a.init_vel == "reference" else None)
            ctx = ScoreContext(z=Z, ro=ro, q=ro.qpos, qs=ro.qpos_smpl, fz=ro.fz)
            r, P = self.cost(ctx)
        P["pelvis_min"] = ro.qpos[:, :, 2].min(1)
        parts = {k: P[k] for k in self.PARTS if k in P}
        sc = Scored(parts=parts, q=ro.qpos, r=r, ro=ro)
        sc.keep = dict(qg=ro.qpos, qs=ro.qpos_smpl, fz=ro.fz)
        return sc

    def terms(self, sc, i):
        return dict(cost=float(1 - sc.r[i].sum() / self.S), **{k: float(v[i]) for k, v in sc.parts.items()})

    @staticmethod
    def line(k, v):
        return (f"  {k:28s} cost {v['cost']:.4f} (bfm {v['bfm']:.4f}" + (f", phc {v['phc']:.4f}" if "phc" in v else "")
                + f", heading {v['head']:.3f}, pos {v['pos']:.2f} m, "
                f"ik {v['ik_deg']:.1f} deg, at limit {100 * v['atlim']:.1f}%, pelvis min {v['pelvis_min']:.2f} m)")

    def store(self, keep, sc, tag, i):
        for k, v in sc.keep.items():
            keep[f"{'q' if k == 'qg' else k}_{tag}"] = v[i].astype(np.float32)

    def baseline(self):
        """the zero-shot references: B(g_t) (the LSO start) and Motivo's ER window 8"""
        from .optim import unit_sphere as unit
        a, S, Bg = self.args, self.S, self.Bg
        self.mu0 = mu = unit(Bg.astype(np.float64))
        self.er8 = er8 = unit(np.stack([Bg[t:t + 8].mean(0) for t in range(S)]).astype(np.float64))
        Z = copies(mu, self.n_envs); Z[1] = er8
        self.t0 = time.time(); sc = self.score(Z)
        self.res = {"B(g_t) (LSO start)": self.terms(sc, 0), "ER L=8 (Motivo zero-shot)": self.terms(sc, 1)}
        self.q_start = {"B(g_t) (LSO start)": sc.keep["qg"][0].copy(), "ER L=8 (Motivo zero-shot)": sc.keep["qg"][1].copy()}
        self.keep = {}
        self.store(self.keep, sc, "Bg", 0); self.store(self.keep, sc, "ER8", 1)
        print(f"{self.robot_name} {a.method.upper()} on {a.ref}: {self.T} frames, {S} steps, {a.n} sequences per iteration, {a.iters} iterations "
              f"({time.time() - self.t0:.0f} s per batch)")
        for k, v in self.res.items():
            print(self.line(k, v), flush=True)

    def start(self):
        a = self.args
        start_key = "B(g_t) (LSO start)" if a.method == "lso" else "ER L=8 (Motivo zero-shot)"
        mu = self.mu0
        if a.method == "ta":
            assert not a.rank, "--method ta: no --rank"
            mu = self.er8.copy()
        self.opt = self.make_optimizer(mu)
        return self.opt, self.best_start(start_key, self.q_start[start_key])

    def curve_row(self, loop, it, sc, costs):
        p = sc.parts
        return dict(iter=it, mean_seq_cost=float(costs[0]), mean_seq_bfm=float(p["bfm"][0]), mean_seq_ik_deg=float(p["ik_deg"][0]),
                    best_sample_cost=float(costs[1:].min()), sample_mean_cost=float(costs[1:].mean()), best_so_far=loop.best["cost"],
                    mu_move=self.mu_move(loop), sec=time.time() - self.t0)

    def report(self, loop, it, sc, costs, row):
        p = sc.parts
        print(f"  iter {it:3d}  mean sequence cost {costs[0]:.4f} (bfm {p['bfm'][0]:.4f}, ik {p['ik_deg'][0]:.1f} deg)  "
              f"samples best {costs[1:].min():.4f} mean {costs[1:].mean():.4f}  |d mu_t| {row['mu_move']:.3f}  [{row['sec'] / 60:.1f} min]", flush=True)

    def finish(self, loop):
        a, keep, res, best = self.args, loop.keep, self.res, loop.best
        sc = self.score(copies(loop.opt.mu, self.n_envs)); res["LSO final mean"] = self.terms(sc, 0); self.store(keep, sc, "LSO", 0)
        if a.iters in a.budgets:
            keep[f"q_mean_it{a.iters}"] = sc.keep["qg"][0].astype(np.float32)
        sc = self.score(copies(best["z"], self.n_envs))
        res["LSO best (re-rolled)"] = dict(**self.terms(sc, 0), it=best["it"], src=best["src"]); self.store(keep, sc, "LSO_best", 0)
        self.sim.close()

    def save(self, rec, loop):
        a, best, res = self.args, loop.best, self.res
        rec.mu_trace(loop.trace, loop.traced_iters(a.trace), best["z"])
        rec.rollouts(self.ref, loop.keep)
        rec.summary(dict(robot=a.robot, ref=a.ref, frames=self.T, n=a.n, iters=a.iters, sigma=a.sigma,
                         lr=a.lr, betas=list(a.betas), gamma=a.gamma, beta=a.beta, rank=a.rank, seed=a.seed,
                         heading_weight=a.heading_weight, pos_weight=a.pos_weight, ik_weight=a.ik_weight,
                         objective=a.objective, target=str(a.target) if a.target else None, method=a.method,
                         ta_beta1=a.ta_beta1, ta_beta2=a.ta_beta2, ta_temp=a.ta_temp, init_vel=a.init_vel,
                         best_from=a.best_from, budgets=a.budgets, best_cost_at=loop.budget_cost,
                         best_it=best["it"], best_src=best["src"], sample_rollouts=a.iters * a.n,
                         fill=a.fill, gains=self.gains, results=res,
                         minutes=(time.time() - self.t0) / 60), indent=1)
        print("\n" + "\n".join(self.line(k, v) for k, v in res.items()))
        print(f"-> {self.out}  ({(time.time() - self.t0) / 60:.1f} min)")
