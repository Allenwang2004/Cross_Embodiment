"""costs.py -- the search costs as CostTerm objects and their CompositeCost (OOP refactor step 6, docs/refactor/PLAN.md).

A term scores one batch rollout (a ScoreContext: the latents, the Rollout of crossenbodiment/rollout.py and arrays derived
from it, plus a memo so a heading or an embedding is computed once per batch). Most terms have two forms:
  batch(ctx)     (n,) one number per rollout: the single-latent searches (single_z_search.py, robot_z_search.py)
  per_step(ctx)  (n, S) one number per control step: the latent-sequence searches' per-step reward (lso_search.py,
                 robot_lso_search.py). A main term's per-step form is a REWARD (a similarity, larger is better); a penalty
                 term's per-step form is the penalty (subtracted with its weight).
Each term also writes what the scripts logged into ctx.parts (the names the scripts used: bfm, bfm_plain, head, pos,
anchor_dist, ik / ik_arm / atlim, ik_deg, mse, mse_frames, phc, dtw / done / pace / path, head_al, fallen, align / phys).

  term            batch form                                         per-step form
  BFMSimilarity   1 - mean_t cos(B(o_t), B(g_t)) (float32), zero_vel: cos(B(o_t), B(g_t)) (lso: B on the GPU tensor,
                  B sees obs[214:358] set to 0 (the "plain" variant   robot_lso: bfm_align.embed then a tensor)
                  is the same term without it, logged as bfm_plain)
  Heading         mean_t (1 - cos(h - h_ref)) / 2 (SMPL pelvis)      (1 - cos(h - h_ref)) / 2
  Position        mean_t |xy - xy_ref| (/ pos_div)                    |xy - xy_ref| (/ pos_div)
  Anchor          (|z - anchor| / 16)^2 (logs anchor_dist)            -
  IKResidual      radians(mean IK residual) (logs ik, ik_arm, atlim)  radians(mean over links) (logs ik_deg, atlim)
  TrackingMSE     losses.tracking_mse                                 1 - losses.tracking_mse_frames (logs mse_frames)
  RobotMSE        robot_z_search's g1mse (pose + ee + root + heading)  -
  PHCReward       1 - mean_t PHC reward (model/phc_reward.PHCCost)   PHC reward r_t (logs phc = 1 - mean r)
  SlowDTW         1 - score of model/dtw.align_slow (logs done,       the per-step reward of lso_search --dtw (its similarity
                  pace, path)                                         share minus heading and fall along the alignment)
  DTW012          robot_z_search's dtw_align (steps 0 / 1 / 2)        -
  DTWUnfinished   1 - done of a DTW term                              -
  AlignedHeading  the heading term along a DTW path (logs head_al)    -
  Fall            share of the aligned frames with the head low       -
  AlignPhys       align_cost.compute_batch_cost (logs align, phys)    -

CompositeCost(main, terms, log, form) returns (cost, parts): cost = main + w_1 t_1 + w_2 t_2 + ... (batch form) or
reward = main - w_1 t_1 - w_2 t_2 - ... (per-step form), added left to right in the given order -- the exact expressions and
summation order of the scripts at fa6c5f9 (which terms a script adds, and when, is the composition the problems in
crossenbodiment/problems.py build). `log` terms are computed for parts only. It replaces single_z_search's `extra`
side-channel dict, lso_search / robot_lso_search's rollout.parts / rollout.keep function attributes and robot_z_search's
score dict. Every term's arithmetic is the scripts' own, operation for operation (golden runs: bit-identical).
"""
from __future__ import annotations

import numpy as np

VEL = slice(214, 358)          # humenv proprio: local_body_vel + local_body_ang_vel (model/obs_scale.OBS_SEGMENTS)


class ScoreContext:
    """one scored batch. z: the latents (n, 256) or sequences; ro: the Rollout; the arrays the terms read are attributes
    (q: the qpos the costs use, qs: a robot's SMPL-equivalent qpos, fz: a robot's IK numbers, obs: the observations)."""

    def __init__(self, z=None, ro=None, **arrays):
        self.z, self.ro = z, ro
        self.parts = {}
        self._memo = {}
        for k, v in arrays.items():
            setattr(self, k, v)

    @property
    def n(self):
        return len(self.z)

    def memo(self, key, make):
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]


class CostTerm:
    """a cost term. batch(ctx) -> (n,), per_step(ctx) -> (n, S); both may write ctx.parts. A term's value is computed once
    per context (value())."""
    name = "?"

    def batch(self, ctx):
        raise NotImplementedError(f"{type(self).__name__} has no single-latent form")

    def per_step(self, ctx):
        raise NotImplementedError(f"{type(self).__name__} has no per-step form")

    def value(self, ctx, form="batch"):
        return ctx.memo((id(self), form), lambda: self.batch(ctx) if form == "batch" else self.per_step(ctx))


class CompositeCost:
    """main + sum of weighted terms, in order (module docstring). terms: [(weight, CostTerm)]; log: terms computed for
    ctx.parts only. Returns (cost (n,) or reward (n, S), ctx.parts)."""

    def __init__(self, main, terms=(), log=(), form="batch"):
        assert form in ("batch", "per_step"), form
        self.main, self.terms, self.log, self.form = main, list(terms), list(log), form

    def __call__(self, ctx):
        for t in self.log:
            t.value(ctx, self.form)
        c = self.main.value(ctx, self.form)
        for w, t in self.terms:
            if self.form == "batch":
                c = c + w * t.value(ctx, self.form)
            else:
                c = c - w * t.value(ctx, self.form)
        return c, ctx.parts


class AsType(CostTerm):
    """another term's value cast to dtype (single_z_search --objective mse: cost = the float64 tracking_mse as float32,
    while parts keep the float64 values)"""

    def __init__(self, term, dtype):
        self.term, self.dtype, self.name = term, dtype, term.name

    def batch(self, ctx):
        return self.term.value(ctx).astype(self.dtype)


# ------------------------------------------------------------------ the BFM term

class BFMSimilarity(CostTerm):
    """BFMTrack's latent-space tracking: cos(B(o_t), B(g_t)), B = Motivo's backward map, o = the observation the actor saw
    (ctx.obs32: the rollout's obs as float32), g = the reference (Bg: its embeddings, the goal of step t at row t).
    zero_vel: B sees the observations with their velocity part obs[214:358] set to 0 (Bg must be embedded the same way).
    seq_embed (per-step form): "backward_map" = policy.backward_map of the float32 tensor on the device (lso_search.py),
    "embed" = bfm_align.embed (numpy) back into a tensor (robot_lso_search.py)."""

    def __init__(self, policy, Bg, zero_vel=False, name="bfm", seq_embed="backward_map"):
        self.policy, self.Bg, self.zero_vel, self.name, self.seq_embed = policy, Bg, zero_vel, name, seq_embed
        self._Bg_t = None

    def embeddings(self, ctx):
        """B of every rollout state, (n, T, 256) float32 numpy: bfm_align.embed of the (n * T, 358) float32 observations
        (single_z_search's batch_bfm_align / its dtw embedding, robot_z_search's Bs_plain / Bs)"""
        def make():
            from model import bfm_align
            o = ctx.obs32
            if self.zero_vel:
                o = np.array(o, copy=True); o[..., VEL] = 0
            n_, T_, D_ = o.shape
            return bfm_align.embed(self.policy.model, o.reshape(n_ * T_, D_), self.policy.device).reshape(n_, T_, -1)
        return ctx.memo(("Bs", self.zero_vel), make)

    def batch(self, ctx):
        from model import bfm_align
        Bs = self.embeddings(ctx)
        v = np.array([bfm_align.bfm_align_loss(Bs[i], self.Bg) for i in range(len(Bs))], dtype=np.float32)
        ctx.parts[self.name] = v
        return v

    def tensor_embeddings(self, ctx):
        """B of every rollout state as a tensor on the device (n, S, 256): the sequence searches' two ways"""
        def make():
            import torch
            if self.seq_embed == "backward_map":                   # lso_search.py
                O = torch.as_tensor(ctx.ro.obs, dtype=torch.float32, device=self.policy.device)
                if self.zero_vel:
                    O[..., VEL] = 0
                return self.policy.backward_map(O.reshape(-1, O.shape[-1])).reshape(O.shape[0], O.shape[1], -1)
            from model import bfm_align                            # robot_lso_search.py
            O = ctx.ro.obs
            n_, S_ = O.shape[:2]
            return torch.as_tensor(bfm_align.embed(self.policy.model, O.astype(np.float32).reshape(n_ * S_, -1), self.policy.device),
                                   device=self.policy.device).reshape(n_, S_, -1)
        return ctx.memo(("Bs_t", self.zero_vel, self.seq_embed), make)

    def reference_tensor(self):
        """the reference embeddings as a tensor on the device (made once; the sequence searches made it at setup)"""
        if self._Bg_t is None:
            import torch
            self._Bg_t = torch.as_tensor(self.Bg, device=self.policy.device)
        return self._Bg_t

    def per_step(self, ctx):
        import torch
        Bs = self.tensor_embeddings(ctx)
        cos = torch.nn.functional.cosine_similarity(Bs, self.reference_tensor()[None], dim=-1).cpu().numpy()
        ctx.parts[self.name] = 1 - cos.mean(1)
        return cos


# ------------------------------------------------------------------ global placement: heading, position

class Heading(CostTerm):
    """humenv's SMPL pelvis heading against the reference's: (1 - cos(h - h_ref)) / 2, per step or its mean over the steps.
    src: the ctx attribute holding the SMPL qpos ("q" for a human body, "qs" for a robot's SMPL-equivalent); the heading
    is crossenbodiment.geometry.smpl_heading(fk, qpos, d, batched) (d: a scratch MjData or None for a new one; batched:
    one calc_heading over all frames, as lso_search / robot_z_search, else frame by frame). The batch form scores the
    first min(T, len(ref_h)) states (single_z_search's global_terms)."""
    name = "head"

    def __init__(self, fk, ref_h, d=None, batched=False, src="q"):
        self.fk, self.ref_h, self.d, self.batched, self.src = fk, ref_h, d, batched, src

    def headings(self, ctx, T=None):
        """(n, T) headings of the first T states (None: all)"""
        X = getattr(ctx, self.src)
        if T is not None and T >= X.shape[1]:
            T = None
        from .geometry import smpl_heading
        return ctx.memo(("heading", self.src, T, self.batched),
                        lambda: smpl_heading(self.fk, X if T is None else X[:, :T], self.d, batched=self.batched))

    def batch(self, ctx):
        X = getattr(ctx, self.src)
        T = min(X.shape[1], len(self.ref_h))
        h = self.headings(ctx, T)
        v = ((1 - np.cos(h - self.ref_h[None, :T])) / 2).mean(1)
        ctx.parts[self.name] = v
        return v

    def per_step(self, ctx):
        hd = (1 - np.cos(self.headings(ctx) - self.ref_h[None])) / 2
        ctx.parts[self.name] = hd.mean(1)
        return hd


class Position(CostTerm):
    """|root xy - reference root xy| per step (metres; / pos_div when given: adult-equivalent metres), or its mean.
    src as Heading's; the batch form scores the first min(T, len(ref_xy)) states."""
    name = "pos"

    def __init__(self, ref_xy, pos_div=None, src="q"):
        self.ref_xy, self.pos_div, self.src = ref_xy, pos_div, src

    def batch(self, ctx):
        X = getattr(ctx, self.src)
        T = min(X.shape[1], len(self.ref_xy))
        xy = (X if T >= X.shape[1] else X[:, :T])[..., :2]
        v = np.linalg.norm(xy - self.ref_xy[None, :T], axis=-1).mean(1)
        if self.pos_div is not None:
            v = v / self.pos_div
        ctx.parts[self.name] = v
        return v

    def per_step(self, ctx):
        X = getattr(ctx, self.src)
        ps = np.linalg.norm(X[..., :2] - self.ref_xy[None], axis=-1)
        if self.pos_div is not None:
            ps = ps / self.pos_div
        ctx.parts[self.name] = ps.mean(1)
        return ps


# ------------------------------------------------------------------ the latent itself

class Anchor(CostTerm):
    """(|z - anchor| / 16)^2: among equally good latents prefer the one nearest the anchor (logs anchor_dist = |z - anchor|)"""
    name = "anchor"

    def __init__(self, anchor):
        self.anchor = anchor

    def batch(self, ctx):
        adist = np.linalg.norm(np.asarray(ctx.z) - self.anchor, axis=1)
        ctx.parts["anchor_dist"] = adist
        return (adist / 16.0) ** 2


# ------------------------------------------------------------------ robots

class IKResidual(CostTerm):
    """a robot's IK residual of the commanded poses (RobotEnv.last_resid, ctx.fz (n, T, n_link + 1): degrees per paired link,
    then the number of joints at a limit), in radians. batch: of the mean over steps and links (logs ik (deg), ik_arm (deg,
    the arm links), atlim (share of joints at a limit)); per step: of the mean over links (logs ik_deg, atlim)."""
    name = "ik"

    def __init__(self, n_link, n_jnt, arm=None):
        self.n_link, self.n_jnt, self.arm = n_link, n_jnt, arm

    def batch(self, ctx):
        FZ = ctx.fz
        ik = FZ[:, :, :self.n_link].mean((1, 2))
        ctx.parts.update(ik=ik, ik_arm=FZ[:, :, self.arm].mean((1, 2)), atlim=FZ[:, :, self.n_link].mean(1) / self.n_jnt)
        return np.radians(ik)

    def per_step(self, ctx):
        FZ = ctx.fz
        ik = np.radians(FZ[:, :, :self.n_link].mean(-1))                     # (n, S) rad
        ctx.parts.update(ik_deg=np.degrees(ik.mean(1)), atlim=FZ[:, :, self.n_link].mean(1) / self.n_jnt)
        return ik


def robot_ee_points(cfg, rm, mg):
    """model/robots.py ROBOTS[robot]["ee"] resolved on the robot model mg: per point (kind, id, body-frame offset for "end")
    (robot_z_search.py's ee list; kind geom_top -> the body's highest geom)"""
    n_jnt = len(rm.g_jnt)
    ee = []
    for kind, nm in cfg["ee"]:
        if kind == "geom_top":
            b = mg.body(nm).id
            ee.append(("geom", max((g for g in range(mg.ngeom) if mg.geom_bodyid[g] == b), key=lambda g: mg.geom_pos[g][2]), None))
        elif kind == "end":
            rm._fk_g(np.zeros(n_jnt)); ee.append(("end", mg.body(nm).id, rm._robot_end(rm.mg.body(nm).id)))
        else:
            ee.append((kind, mg.body(nm).id if kind == "body" else mg.site(nm).id, None))
    return ee


class RobotPoints:
    """forward kinematics of robot qpos: pelvis xyz, pelvis yaw, hands / feet / head relative to the pelvis in its heading
    frame (robot_z_search.py's g1_points)"""

    def __init__(self, mg, root, ee):
        import mujoco
        self.mg, self.root, self.ee = mg, root, ee
        self.gd = mujoco.MjData(mg)

    def _ee_pos(self, kind, i, off):
        gd = self.gd
        if kind == "body":
            return gd.xpos[i]
        if kind == "site":
            return gd.site_xpos[i]
        if kind == "geom":
            return gd.geom_xpos[i]
        return gd.xpos[i] + gd.xmat[i].reshape(3, 3) @ off

    def __call__(self, qg):
        """(..., nq) robot qpos -> pelvis xyz (..., 3), pelvis yaw (...), points (..., n_ee, 3)"""
        import mujoco
        from .geometry import robot_yaw
        mg, gd, root, n_ee = self.mg, self.gd, self.root, len(self.ee)
        flat = qg.reshape(-1, qg.shape[-1]); P = np.empty((len(flat), 3)); H = np.empty(len(flat))
        E = np.empty((len(flat), n_ee, 3))
        for i, qq in enumerate(flat):
            gd.qpos[:] = qq; mujoco.mj_kinematics(mg, gd)
            pel = gd.xpos[root].copy(); h = robot_yaw(gd.xmat[root])
            d = np.stack([self._ee_pos(*e) for e in self.ee]) - pel
            c, sn = np.cos(h), np.sin(h)
            E[i] = np.stack([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1], d[:, 2]], -1); P[i], H[i] = pel, h
        sh = qg.shape[:-1]
        return P.reshape(sh + (3,)), H.reshape(sh), E.reshape(sh + (n_ee, 3))


class RobotMSE(CostTerm):
    """robot_z_search.py --objective mse (g1mse): squared errors against the robot target gT (the clip mapped onto the robot),
    weights 1 as losses.tracking_mse: pose (wrapped joint angles), ee (hands / feet / head, pelvis heading frame), root
    (pelvis xyz; no_xy: height only), heading (1 - cos yaw). ctx.q: the robot qpos."""
    name = "mse"

    def __init__(self, points, gT, no_xy=False):
        self.points, self.gT, self.no_xy = points, gT, no_xy
        self.ref_P, self.ref_H, self.ref_E = points(gT)

    def batch(self, ctx):
        QG, gT = ctx.q, self.gT
        ref_P, ref_H, ref_E = self.ref_P, self.ref_H, self.ref_E
        P, H, E = self.points(QG)
        dq = (QG[..., 7:] - gT[None, :, 7:] + np.pi) % (2 * np.pi) - np.pi
        v = ((dq ** 2).mean((1, 2)) + ((E - ref_E[None]) ** 2).sum(-1).mean((1, 2))
             + (((P[..., 2:] - ref_P[None, :, 2:]) if self.no_xy else (P - ref_P[None])) ** 2).sum(-1).mean(1)
             + (1 - np.cos(H - ref_H[None])).mean(1))
        ctx.parts[self.name] = v
        return v


# ------------------------------------------------------------------ joint-space tracking

class TrackingMSE(CostTerm):
    """losses.tracking_mse (pose + ee + root + heading MSE) against goal; root_xy False: the root term is the height only.
    per step: 1 - losses.tracking_mse_frames (ref_pose: goal's forward kinematics, computed once)."""
    name = "mse"

    def __init__(self, fk, goal, root_xy=True, ref_pose=None):
        self.fk, self.goal, self.root_xy, self.ref_pose = fk, goal, root_xy, ref_pose

    def one(self, qi):
        from model import losses
        return losses.tracking_mse(self.fk, qi, self.goal, root_xy=self.root_xy)[0]

    def batch(self, ctx):
        v = np.array([self.one(qi) for qi in ctx.q])
        ctx.parts[self.name] = v
        return v

    def per_step(self, ctx):
        from model import losses
        q = ctx.q
        mf = np.stack([losses.tracking_mse_frames(self.fk, q[k], self.goal, root_xy=self.root_xy, ref_pose=self.ref_pose)
                       for k in range(len(q))])
        ctx.parts["mse_frames"] = mf.mean(1)
        return 1 - mf


class AlignPhys(CostTerm):
    """crossenbodiment.align_cost.compute_batch_cost: cfg.lambda_align * L_align + cfg.lambda_phys * L_phys against goal
    (float32; logs align, phys). one(qi): (L_align, L_phys) of one rollout."""
    name = "align"

    def __init__(self, fk, cfg, goal):
        self.fk, self.cfg, self.goal = fk, cfg, goal

    def batch(self, ctx):
        from .align_cost import compute_batch_cost
        cost, align, phys = compute_batch_cost(self.fk, self.cfg, ctx.q, [self.goal] * ctx.n)
        ctx.parts.update(align=align, phys=phys)
        return cost

    def one(self, qi):
        from .align_cost import compute_batch_cost
        _, a1, p1 = compute_batch_cost(self.fk, self.cfg, qi[None], [self.goal])
        return float(a1[0]), float(p1[0])


class PHCReward(CostTerm):
    """BFM-Zero's tracking reward (model/phc_reward.PHCCost: PHC's body positions, rotations, velocities, angular velocities).
    batch: 1 - mean_t r_t; per step: r_t (logs phc = 1 - mean r). start(ctx): the state before the first step;
    same_start: assert every slot started from the same pose (the robots')."""
    name = "phc"

    def __init__(self, phc, start, same_start=False, src="q"):
        self.phc, self.start, self.same_start, self.src = phc, start, same_start, src

    def _start(self, ctx):
        if self.same_start:                                    # every slot starts from the same pose (checked)
            qg0 = ctx.ro.start.qpos
            assert np.abs(qg0 - qg0[:1]).max() == 0
        return self.start(ctx)

    def batch(self, ctx):
        return self.phc(getattr(ctx, self.src), self._start(ctx))

    def per_step(self, ctx):
        q, prev = getattr(ctx, self.src), self._start(ctx)
        r = np.stack([self.phc.parts(q[k], prev)[0] for k in range(len(q))])
        ctx.parts[self.name] = 1 - r.mean(1)
        return r


# ------------------------------------------------------------------ time-aligned (DTW) terms

class Fall(CostTerm):
    """the share of the ALIGNED rollout frames (a DTW path's frames >= 0) with the head below fall_head (m); the head's
    height by forward kinematics of every state (heights(ctx), (n, T))"""
    name = "fallen"

    def __init__(self, fk, fall_head, path_of=None, d=None):
        self.fk, self.fall_head, self.path_of, self.d = fk, fall_head, path_of, d

    def heights(self, ctx):
        def make():
            import mujoco
            q = ctx.q
            dk = mujoco.MjData(self.fk) if self.d is None else self.d
            hid = self.fk.body("Head").id
            hz = np.empty(q.shape[:2])
            for k in range(q.shape[0]):
                for i in range(q.shape[1]):
                    dk.qpos[:] = q[k, i]; mujoco.mj_kinematics(self.fk, dk); hz[k, i] = dk.xpos[hid][2]
            return hz
        return ctx.memo("head_z", make)

    def batch(self, ctx):
        hz, P = self.heights(ctx), self.path_of(ctx)
        v = np.array([(hz[k][P[k] >= 0] < self.fall_head).mean() for k in range(ctx.n)])
        ctx.parts[self.name] = v
        return v


class SlowDTW(CostTerm):
    """model/dtw.py's slow-only alignment of the rollout's embeddings B(o_i) to the reference's B(g_j) (the rollout may be
    slower, never faster). batch (single_z_search --objective dtw): 1 - score, logs done (share of the reference reached),
    pace, path. per step (lso_search --dtw): each frame's similarity to the reference frame it is matched to (shared by the
    frames matched to it), minus (heading_weight x the aligned heading term + fall_weight x fallen) spread over the matched
    frames, so 1 - sum r / S is the alignment's cost (logs bfm (frame by frame), dtw, done, pace, head (aligned), fallen,
    path)."""
    name = "dtw"

    def __init__(self, bfm, Bg, heading=None, fall=None, heading_weight=0.0, fall_weight=0.0):
        self.bfm, self.Bg = bfm, Bg
        self.heading, self.fall, self.heading_weight, self.fall_weight = heading, fall, heading_weight, fall_weight

    def path(self, ctx):
        return ctx.parts["path"]

    def batch(self, ctx):
        from model import dtw
        sc, je, ie, path = dtw.align_slow(dtw.cosine_matrix(self.bfm.embeddings(ctx), self.Bg))
        ctx.parts.update(dtw=1 - sc, done=(je + 1) / len(self.Bg), pace=(je + 1) / (ie + 1), path=path)
        return ctx.parts["dtw"]

    def per_step(self, ctx):
        from model import dtw
        q, n_envs, SR, S, Bc = ctx.q, ctx.n, ctx.q.shape[1], len(self.Bg), self.Bg
        Bs = self.bfm.tensor_embeddings(ctx)
        Sm = dtw.cosine_matrix(Bs.cpu().numpy().astype(np.float64), Bc.astype(np.float64))
        sc, je, ie, path = dtw.align_slow(Sm)
        hz = self.fall.heights(ctx)
        hd_all = (1 - np.cos(self.heading.headings(ctx) - self.heading.ref_h[np.maximum(path, 0)])) / 2
        r = np.zeros((n_envs, SR)); fallen, head_al = np.zeros(n_envs), np.zeros(n_envs)
        for k in range(n_envs):
            mk = path[k] >= 0; jk = path[k][mk]; cnt = np.bincount(jk, minlength=S)
            nm = mk.sum()
            fl = hz[k][mk] < self.fall.fall_head
            r[k, mk] = (Sm[k, np.flatnonzero(mk), jk] / cnt[jk]
                        - (self.heading_weight * hd_all[k][mk] + self.fall_weight * fl) * S / nm)
            fallen[k], head_al[k] = fl.mean(), hd_all[k][mk].mean()
        ctx.parts.update(bfm=np.array([1 - (Sm[k, np.arange(min(SR, S)), np.arange(min(SR, S))]).mean() for k in range(n_envs)]),
                         dtw=1 - sc, done=(je + 1) / S, pace=(je + 1) / (ie + 1), head=head_al, fallen=fallen, path=path)
        return r


def dtw_align(Bs, Bg, lam):
    """Bs (n, T, d) rollout embeddings, Bg (R, d) reference -> per rollout: dtw = 1 - mean cos along the best path,
    done = fraction of the reference reached at its end, path (n, T) = reference frame of each rollout frame. Steps of
    0, 1 or 2 reference frames, never 0 twice in a row except on the reference's last frame; the end is the one
    minimising dtw + lam * (1 - done). (robot_z_search.py's, imported from there as before)"""
    a = Bs / (np.linalg.norm(Bs, axis=-1, keepdims=True) + 1e-12)
    b = Bg / (np.linalg.norm(Bg, axis=-1, keepdims=True) + 1e-12)
    n, T, _ = a.shape; R = len(b); S = a @ b.T                     # (n, T, R) cosine of every rollout / reference pair
    NEG = -1e18
    V = np.full((n, R, 2), NEG); V[:, 0, 0] = S[:, 0, 0]          # [j, s]: s = 1 if the last step stayed on j
    bp = np.zeros((T, n, R, 2), dtype=np.int8)                     # 0/1: stayed (from s 0/1); 2+k: advanced, k below
    for i in range(1, T):
        Vn = np.full((n, R, 2), NEG)
        Vn[:, :, 1] = V[:, :, 0]
        hold = V[:, R - 1, 1] > V[:, R - 1, 0]                     # the last frame may be held for longer
        Vn[:, R - 1, 1] = np.where(hold, V[:, R - 1, 1], V[:, R - 1, 0]); bp[i, :, R - 1, 1] = hold
        cand = np.full((4, n, R), NEG)                             # from j-1 (s 0, 1), j-2 (s 0, 1)
        cand[0, :, 1:], cand[1, :, 1:] = V[:, :-1, 0], V[:, :-1, 1]
        cand[2, :, 2:], cand[3, :, 2:] = V[:, :-2, 0], V[:, :-2, 1]
        k = cand.argmax(0)
        Vn[:, :, 0] = np.take_along_axis(cand, k[None], 0)[0]; bp[i, :, :, 0] = k + 2
        V = Vn + S[:, i, :, None]
    tot = V.max(-1)                                                # (n, R): best summed cos ending on each frame
    end_cost = 1 - tot / T + lam * (1 - (np.arange(R) + 1) / R)
    j = end_cost.argmin(1); s = V[np.arange(n), j].argmax(-1)
    dtw = 1 - tot[np.arange(n), j] / T; done = (j + 1) / R
    path = np.empty((n, T), dtype=np.int64)
    for i in range(T - 1, -1, -1):
        path[:, i] = j
        if i:
            c = bp[i, np.arange(n), j, s]
            j, s = j - np.where(c < 2, 0, np.where(c < 4, 1, 2)), c % 2
    assert (path[:, 0] == 0).all()
    return dtw, done, path


class DTW012(CostTerm):
    """robot_z_search.py --objective dtw: dtw_align of the rollout's embeddings (the BFM term's, zero_vel as configured) to
    the reference's, steps of 0 / 1 / 2 reference frames; batch: dtw = 1 - mean cos along the path (logs done, path)"""
    name = "dtw"

    def __init__(self, bfm, Bg, lam):
        self.bfm, self.Bg, self.lam = bfm, Bg, lam

    def path(self, ctx):
        return ctx.parts["path"]

    def batch(self, ctx):
        dtw, done, path = dtw_align(self.bfm.embeddings(ctx), self.Bg, self.lam)
        ctx.parts.update(dtw=dtw, done=done, path=path)
        return dtw


class DTWUnfinished(CostTerm):
    """1 - done of a DTW term (robot_z_search: + dtw_lambda x the share of the reference not reached)"""
    name = "unfinished"

    def __init__(self, dtw_term):
        self.dtw_term = dtw_term

    def batch(self, ctx):
        self.dtw_term.value(ctx)
        return 1 - ctx.parts["done"]


class AlignedHeading(CostTerm):
    """the heading term along a DTW path (logs head_al): masked (single_z_search) = per rollout the mean over its aligned
    frames (path >= 0) of (1 - cos(h_i - h_ref[path_i])) / 2; unmasked (robot_z_search's dtw_align paths have no -1)
    = ((1 - cos(h - h_ref[path])) / 2).mean(1). The headings are the Heading term's, of every state."""
    name = "head_al"

    def __init__(self, heading, path_of, masked=True):
        self.heading, self.path_of, self.masked = heading, path_of, masked

    def batch(self, ctx):
        h, P, ref_h = self.heading.headings(ctx), self.path_of(ctx), self.heading.ref_h
        if self.masked:
            v = np.array([((1 - np.cos(h[k][P[k] >= 0] - ref_h[P[k][P[k] >= 0]])) / 2).mean() for k in range(ctx.n)])
        else:
            v = ((1 - np.cos(h - ref_h[P])) / 2).mean(1)
        ctx.parts[self.name] = v
        return v
