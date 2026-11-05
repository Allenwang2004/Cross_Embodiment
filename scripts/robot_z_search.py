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
Writes <out>/{best_z.npy, best.npz, origin_z.npz, z_trace.npz, curve.csv, summary.json}; best.npz / origin_z.npz hold
the G1 qpos ("qpos"), its SMPL-equivalent ("qpos_smpl") and per step the 13 IK residuals (deg) + limit count ("fz").
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/robot_z_search.py --robot h1 [--clip ...] [--evals 1024] [--objective mse]
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from single_z_search import project_z, rank_normalize, ang_pair   # noqa: E402
from g1_z0_rollout import motion                                    # noqa: E402
from model import bfm_align                                         # noqa: E402
from model.robot_env import RobotVecEnv                             # noqa: E402
from model.robot_map import RobotMap                                # noqa: E402
from model.robots import ROBOTS, out_dir, target_path               # noqa: E402


def dtw_align(Bs, Bg, lam):
    """Bs (n, T, d) rollout embeddings, Bg (R, d) reference -> per rollout: dtw = 1 - mean cos along the best path,
    done = fraction of the reference reached at its end, path (n, T) = reference frame of each rollout frame. Steps of
    0, 1 or 2 reference frames, never 0 twice in a row except on the reference's last frame; the end is the one
    minimising dtw + lam * (1 - done)."""
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


def main():
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
    p.add_argument("--objective", default="bfm", choices=["bfm", "mse", "g1mse", "dtw"],
                   help="mse (= g1mse): squared errors against the robot's target in place of bfm + heading + pos")
    p.add_argument("--mse-no-xy", action="store_true",
                   help="g1mse's root term compares the pelvis height only, not where it is on the floor")
    p.add_argument("--start-frame", type=int, default=0,
                   help="use the clip from this frame on (reference, robot target, rollout start)")
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
    a = p.parse_args()
    task, stem = a.clip.split("/")
    cfg = ROBOTS[a.robot]
    if a.objective == "g1mse":
        a.objective = "mse"
    tag = {"bfm": "", "dtw": "_dtw"}.get(a.objective, "_g1mse" if a.robot == "g1" else "_mse")
    out = Path(a.out) if a.out else out_dir(a.robot) / "search" / f"{stem}{tag}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    n_envs = 2 * a.pairs; n_gens = max(a.evals // n_envs, 1); n_eps = a.pairs if a.sampling == "antithetic" else n_envs

    vec = RobotVecEnv(a.robot, n_envs, fill=a.fill, gains=a.gains)                                    # spawn the workers before CUDA is touched
    rm = RobotMap(a.robot, fill=a.fill)
    n_link, n_jnt = len(rm.pairs) - 1, len(rm.g_jnt)
    ARM = [i for i, (b, _) in enumerate(rm.pairs[1:]) if b.split("_")[-1] in ("Shoulder", "Elbow", "Hand")]
    from humenv import make_humenv
    import humenv.utils as hu
    from metamotivo.fb_cpr.huggingface import FBcprModel
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"][a.start_frame:]
    T = len(ref)
    z0 = project_z(np.load(REPO / f"data/origin_z/{a.clip}.npy").reshape(-1).astype(np.float64))
    anchor = project_z(np.load(a.anchor).reshape(-1).astype(np.float64)) if a.anchor else z0
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device)
    adult, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
    fk = adult.unwrapped.model; dk = mujoco.MjData(fk)
    VEL = slice(214, 358)                         # humenv proprio: local_body_vel + local_body_ang_vel
    Og = bfm_align.obs_from_qpos(adult, ref)
    Bg_plain = bfm_align.embed(model, Og, a.device)
    if a.bfm_zero_vel:
        Og = Og.copy(); Og[:, VEL] = 0
    Bg = bfm_align.embed(model, Og, a.device)
    gref = np.load(target_path(a.robot, stem))["qpos"][a.start_frame:]
    floor = json.loads((target_path(a.robot, stem).parent / "summary.json").read_text())["bfm_floor"]
    fc = REPO / "outputs/map_error/fill_compare_motivo.json"            # the ridge fill's floor (scripts/fit_smpl_fill.py)
    if a.fill == "ridge" and fc.exists():
        floor = json.loads(fc.read_text()).get(a.robot, {}).get("ridge", {}).get("bfm_full_per_clip", {}).get(stem, floor)
    mg = mujoco.MjModel.from_xml_path(str(REPO / cfg["xml"])); gd = mujoco.MjData(mg)
    root = mg.body(cfg["root"]).id
    ee = []                             # per point: (kind, id, body-frame offset for "end")
    for kind, nm in cfg["ee"]:
        if kind == "geom_top":
            b = mg.body(nm).id
            ee.append(("geom", max((g for g in range(mg.ngeom) if mg.geom_bodyid[g] == b), key=lambda g: mg.geom_pos[g][2]), None))
        elif kind == "end":
            rm._fk_g(np.zeros(n_jnt)); ee.append(("end", mg.body(nm).id, rm._robot_end(rm.mg.body(nm).id)))
        else:
            ee.append((kind, mg.body(nm).id if kind == "body" else mg.site(nm).id, None))
    n_ee = len(ee)

    def ee_pos(kind, i, off):
        if kind == "body":
            return gd.xpos[i]
        if kind == "site":
            return gd.site_xpos[i]
        if kind == "geom":
            return gd.geom_xpos[i]
        return gd.xpos[i] + gd.xmat[i].reshape(3, 3) @ off

    def g1_points(qg):
        """(..., nq) robot qpos -> pelvis xyz (..., 3), pelvis yaw (...), hands / feet / head relative to the pelvis in
        its heading frame (..., n_ee, 3)."""
        flat = qg.reshape(-1, qg.shape[-1]); P = np.empty((len(flat), 3)); H = np.empty(len(flat))
        E = np.empty((len(flat), n_ee, 3))
        for i, qq in enumerate(flat):
            gd.qpos[:] = qq; mujoco.mj_kinematics(mg, gd)
            pel = gd.xpos[root].copy(); R = gd.xmat[root].reshape(3, 3); h = np.arctan2(R[1, 0], R[0, 0])
            d = np.stack([ee_pos(*e) for e in ee]) - pel
            c, sn = np.cos(h), np.sin(h)
            E[i] = np.stack([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1], d[:, 2]], -1); P[i], H[i] = pel, h
        sh = qg.shape[:-1]
        return P.reshape(sh + (3,)), H.reshape(sh), E.reshape(sh + (n_ee, 3))
    gT = gref[:T]
    ref_P, ref_H, ref_E = g1_points(gT)

    def g1mse(QG):
        P, H, E = g1_points(QG)
        dq = (QG[..., 7:] - gT[None, :, 7:] + np.pi) % (2 * np.pi) - np.pi
        return ((dq ** 2).mean((1, 2)) + ((E - ref_E[None]) ** 2).sum(-1).mean((1, 2))
                + (((P[..., 2:] - ref_P[None, :, 2:]) if a.mse_no_xy else (P - ref_P[None])) ** 2).sum(-1).mean(1)
                + (1 - np.cos(H - ref_H[None])).mean(1))

    base_rot = hu.quat_conjugate(np.array([[0.5, 0.5, 0.5, 0.5]]))     # hu.remove_base_rot's, SMPL

    def heading_xy(qs):
        flat = qs.reshape(-1, qs.shape[-1]); r = np.empty((len(flat), 4))
        for i, qq in enumerate(flat):
            dk.qpos[:] = qq; mujoco.mj_kinematics(fk, dk); r[i] = dk.xquat[1]
        return hu.calc_heading(hu.quat_mul(r, np.repeat(base_rot, len(r), axis=0))).reshape(qs.shape[:-1]), qs[..., :2]     # hu.remove_base_rot works on one quat only
    ref_h, ref_xy = heading_xy(ref)

    @torch.no_grad()
    def score(zs):
        zt = torch.as_tensor(zs, dtype=torch.float32, device=a.device)
        obs, _, _, _ = vec.reset((ref[0], ref[1]) if a.init_vel == "reference" else ref[0])
        O, QS, QG, FZ = [], [], [], []
        for _ in range(T):
            o = torch.as_tensor(obs, dtype=torch.float32, device=a.device)
            u = model._actor(model._normalize(o), zt, model.cfg.actor_std).mean.cpu().numpy()
            obs, qs, qg, fz = vec.step(u)
            O.append(obs); QS.append(qs); QG.append(qg); FZ.append(fz)
        O, QS, QG, FZ = (np.stack(x, axis=1) for x in (O, QS, QG, FZ))
        ik = FZ[:, :, :n_link].mean((1, 2)); ik_arm = FZ[:, :, ARM].mean((1, 2)); atlim = FZ[:, :, n_link].mean(1) / n_jnt
        n_, T_, D_ = O.shape                       # = bfm_align.batch_bfm_align, keeping the embeddings for dtw
        Of = O.astype(np.float32).reshape(n_ * T_, D_)
        Bs_plain = bfm_align.embed(model, Of, a.device).reshape(n_, T_, -1)
        bfm_plain = np.array([bfm_align.bfm_align_loss(Bs_plain[i], Bg_plain) for i in range(n_)], dtype=np.float32)
        if a.bfm_zero_vel:
            Of = Of.copy(); Of[:, VEL] = 0
            Bs = bfm_align.embed(model, Of, a.device).reshape(n_, T_, -1)
            bfm = np.array([bfm_align.bfm_align_loss(Bs[i], Bg) for i in range(n_)], dtype=np.float32)
        else:
            Bs, bfm = Bs_plain, bfm_plain
        h, xy = heading_xy(QS)
        head = ((1 - np.cos(h - ref_h[None])) / 2).mean(1)
        pos = np.linalg.norm(xy - ref_xy[None], axis=-1).mean(1)
        adist = np.linalg.norm(np.asarray(zs) - anchor, axis=1)
        mse = g1mse(QG)
        if a.objective == "dtw":
            dtw, done, path = dtw_align(Bs, Bg, a.dtw_lambda)
            head_al = ((1 - np.cos(h - ref_h[path])) / 2).mean(1)
        else:
            dtw = done = head_al = np.full(n_, np.nan); path = np.zeros((n_, T_), dtype=np.int64)
        if a.objective == "mse":
            main = mse
        elif a.objective == "dtw":
            main = dtw + a.dtw_lambda * (1 - done) + a.heading_weight * head_al + a.pos_weight * pos
        else:
            main = bfm + a.heading_weight * head + a.pos_weight * pos
        cost = main + a.anchor_weight * (adist / 16.0) ** 2 + a.ik_weight * np.radians(ik)
        return dict(cost=cost, bfm=bfm, bfm_plain=bfm_plain, mse=mse, head=head, pos=pos, anchor_dist=adist, ik=ik, ik_arm=ik_arm,
                    atlim=atlim, qg=QG, qs=QS, fz=FZ, dtw=dtw, done=done, head_al=head_al, path=path)

    def pick(r, i, z, gen, source):
        return dict(cost=float(r["cost"][i]), bfm=float(r["bfm"][i]), bfm_plain=float(r["bfm_plain"][i]), mse=float(r["mse"][i]), head=float(r["head"][i]),
                    pos=float(r["pos"][i]),
                    ik=float(r["ik"][i]), ik_arm=float(r["ik_arm"][i]), atlim=float(r["atlim"][i]),
                    dtw=float(r["dtw"][i]), done=float(r["done"][i]), head_al=float(r["head_al"][i]), path=r["path"][i].copy(),
                    z=z.copy(), qg=r["qg"][i].copy(), qs=r["qs"][i].copy(), fz=r["fz"][i].copy(), gen=gen, source=source)

    print(f"{cfg['name']} latent search on {a.clip}, objective {a.objective}: {n_gens} steps x {n_envs} rollouts = {n_gens * n_envs} evals, "
          f"sigma {a.sigma}, lr {a.lr}, heading {a.heading_weight}, pos {a.pos_weight}, anchor {a.anchor_weight}, "
          f"ik {a.ik_weight}", flush=True)
    t0 = time.time()
    r0 = score(np.repeat(z0[None], n_envs, axis=0))
    spread = float(np.abs(r0["qg"] - r0["qg"][:1]).max())
    base = pick(r0, 0, z0, -1, "origin_z")
    al = (lambda b: f"  dtw {b['dtw']:.4f} done {b['done']:.2f} heading_al {b['head_al']:.4f}") if a.objective == "dtw" else (lambda b: "")
    print(f"  origin_z:{al(base)} cost {base['cost']:.4f}  bfm {base['bfm']:.4f} (plain {base['bfm_plain']:.4f})  mse {base['mse']:.4f}  heading {base['head']:.4f}  pos {base['pos']:.2f}"
          f"  ik {base['ik']:.2f} deg (arms {base['ik_arm']:.2f}, at limit {100 * base['atlim']:.1f}%)"
          f"   (16 copies identical to {spread:.1e}; {time.time() - t0:.0f} s per step)", flush=True)

    rng = np.random.default_rng(a.seed)
    z = z0.copy(); m = np.zeros_like(z); v = np.zeros_like(z); b1, b2, eps_adam = 0.9, 0.999, 1e-8
    best, curve, trace_mean, trace_best, path_len = base, [], [z0.copy()], [z0.copy()], 0.0
    gen0, t_adam0, exact = 0, 0, None              # first step of this run; Adam steps taken before it
    if a.resume:
        rd = Path(a.resume) if Path(a.resume).is_absolute() else REPO / a.resume
        ps = json.loads((rd / "summary.json").read_text())
        same = {k: (ps[k], v_) for k, v_ in (("clip", a.clip), ("body", a.robot), ("seed", a.seed), ("pairs", a.pairs),
                                             ("sigma", a.sigma), ("lr", a.lr), ("ik_weight", a.ik_weight),
                                             ("objective", a.objective)) if ps[k] != v_}
        assert not same, f"--resume: settings differ from {rd}: {same}"
        ps_sampling = ps.get("sampling", "antithetic")     # runs before 11/04 record none: all antithetic
        assert ps_sampling == a.sampling, f"--resume: {rd} ran --sampling {ps_sampling}, this run has --sampling {a.sampling}"
        assert abs(ps["origin_z"]["cost"] - base["cost"]) < 1e-9, f"--resume: z0 costs {base['cost']} now, {ps['origin_z']['cost']} then"
        gen0 = ps["generations"]
        assert n_gens > gen0, f"--resume: {rd} already has {gen0} steps; --evals is the total budget"
        curve = list(csv.DictReader(open(rd / "curve.csv")))
        zt = np.load(rd / "z_trace.npz")
        trace_mean, trace_best = list(zt["z_mean"].astype(np.float64)), list(zt["z_best"].astype(np.float64))
        bq = np.load(rd / "best.npz")
        best = dict(**{k: ps["best"][k] for k in ("cost", "bfm", "mse", "head", "pos", "ik", "ik_arm", "atlim", "gen", "source")},
                    bfm_plain=ps["best"].get("bfm_plain", ps["best"]["bfm"]),   # runs before 11/02 have no bfm_plain (= bfm there)
                    z=np.load(rd / "best_z.npy").reshape(-1).astype(np.float64), qg=bq["qpos"], qs=bq["qpos_smpl"], fz=bq["fz"])
        exact = (rd / "state.npz").exists()
        if exact:
            S = np.load(rd / "state.npz")
            z, m, v, t_adam0, path_len = S["z"], S["m"], S["v"], int(S["t_adam"]), float(S["path_len"])
            best["z"] = S["best_z"]; rng.bit_generator.state = json.loads(str(S["rng"]))
        else:
            z, path_len = trace_mean[-1].copy(), ps["path_len_deg"]
            for _ in range(gen0):                          # the noise the earlier steps drew
                rng.standard_normal((n_eps, z.size))
        print(f"  resumed {rd.name} at step {gen0} ({'exact: Adam state restored' if exact else 'from its last search point, Adam restarted'}), "
              f"best so far {best['cost']:.4f}", flush=True)
    for gen in range(gen0, n_gens):
        eps = rng.standard_normal((n_eps, z.size))
        cand = project_z(np.concatenate([z + a.sigma * eps, z - a.sigma * eps], axis=0) if a.sampling == "antithetic" else z + a.sigma * eps)
        r = score(cand)
        i = int(np.argmin(r["cost"]))
        if r["cost"][i] < best["cost"]:
            best = pick(r, i, cand[i], gen, "sample")
        s = rank_normalize(r["cost"])
        if a.sampling == "antithetic":
            g = ((s[:a.pairs] - s[a.pairs:])[:, None] * eps).sum(0) / (2 * a.pairs * a.sigma)
        else:                                              # leave-one-out baseline, as lso_search.py
            g = ((s - (s.sum() - s) / (len(s) - 1))[:, None] * eps).sum(0) / (2 * a.pairs * a.sigma)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        t = gen - gen0 + t_adam0 + 1                       # Adam's step count (gen + 1 unless resumed without state)
        step = a.lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps_adam)
        z_prev, z = z, project_z(z - step)
        path_len += ang_pair(z_prev, z)
        row = dict(gen=gen, evals=(gen + 1) * n_envs, gen_best=float(r["cost"].min()), gen_mean=float(r["cost"].mean()),
                   best_so_far=best["cost"], best_bfm=best["bfm"], best_done=best.get("done", ""), best_ik=best["ik"], mean_z_cost="",
                   cos_z0=float(z @ z0 / (np.linalg.norm(z) * np.linalg.norm(z0))),
                   path_len_deg=round(path_len, 2), net_deg=round(ang_pair(z0, z), 2))
        if a.eval_every and (gen % a.eval_every == 0 or gen == n_gens - 1):
            rm = score(np.repeat(z[None], n_envs, axis=0))
            row["mean_z_cost"] = float(rm["cost"][0])
            if rm["cost"][0] < best["cost"]:
                best = pick(rm, 0, z, gen, "mean_z")
            print(f"  gen {gen:4d}/{n_gens}  best {best['cost']:.4f} ({al(best)[2:] + ', ' if al(best) else ''}bfm {best['bfm']:.4f} (plain {best['bfm_plain']:.4f}), mse {best['mse']:.4f}, heading {best['head']:.4f}, "
                  f"pos {best['pos']:.2f}, ik {best['ik']:.2f} deg)  gen_best {r['cost'].min():.4f}  mean_z {rm['cost'][0]:.4f}  "
                  f"|z-z0| {np.linalg.norm(z - z0):.2f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        curve.append(row); trace_mean.append(z.copy()); trace_best.append(best["z"].copy())
    vec.close()

    np.save(out / "best_z.npy", best["z"].astype(np.float32).reshape(1, -1))
    # everything --resume needs to continue this search exactly
    np.savez(out / "state.npz", z=z, m=m, v=v, t_adam=n_gens - gen0 + t_adam0, path_len=path_len, best_z=best["z"],
             rng=json.dumps(rng.bit_generator.state))
    for f, b in (("best.npz", best), ("origin_z.npz", base)):
        np.savez_compressed(out / f, qpos=b["qg"].astype(np.float32), qpos_smpl=b["qs"].astype(np.float32),
                            fz=b["fz"].astype(np.float32), fps=30, **({"dtw_path": b["path"]} if a.objective == "dtw" and "path" in b else {}))
    np.savez_compressed(out / "z_trace.npz", gen=np.arange(-1, len(trace_mean) - 1, dtype=np.int32),
                        z_mean=np.stack(trace_mean).astype(np.float32), z_best=np.stack(trace_best).astype(np.float32))
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[-1]), restval=""); w.writeheader(); w.writerows(curve)   # [-1]: a resumed older run's rows lack newer columns
    keys = ("cost", "bfm", "bfm_plain", "mse", "head", "pos", "ik", "ik_arm", "atlim", "gen", "source") + (("dtw", "done", "head_al") if a.objective == "dtw" else ())
    summ = dict(clip=a.clip, body=a.robot, evals=n_gens * n_envs, generations=n_gens, pairs=a.pairs, sampling=a.sampling, sigma=a.sigma, lr=a.lr,
                seed=a.seed, **({"dtw_lambda": a.dtw_lambda} if a.objective == "dtw" else {}), bfm_zero_vel=a.bfm_zero_vel, mse_no_xy=a.mse_no_xy, init_vel=a.init_vel, start_frame=a.start_frame, heading_weight=a.heading_weight, pos_weight=a.pos_weight, anchor=a.anchor or "z0",
                anchor_weight=a.anchor_weight, ik_weight=a.ik_weight, gains=a.gains or cfg["gains_json"], objective=a.objective, fill=a.fill,
                bfm_floor_of_map=floor,
                origin_z={**{k: base.get(k) for k in keys}, "motion": motion(base["qg"], gref)},
                best={**{k: best.get(k) for k in keys}, "motion": motion(best["qg"], gref)},
                improvement_vs_origin_z=(base["cost"] - best["cost"]) / abs(base["cost"]),
                dist_best_z0=float(np.linalg.norm(best["z"] - z0)), path_len_deg=round(path_len, 1),
                resumed_from=a.resume, resumed_at=gen0 if a.resume else None, resume_exact=exact,
                minutes=(time.time() - t0) / 60)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\n=== {n_gens * n_envs} evals in {summ['minutes']:.1f} min ===")
    for n, b in (("origin_z", base), ("best z", best)):
        mo = summ["origin_z" if n == "origin_z" else "best"]["motion"]
        print(f"{n:9s}{al(b)} cost {b['cost']:.4f}  bfm {b['bfm']:.4f} (plain {b['bfm_plain']:.4f})  mse {b['mse']:.4f}  heading {b['head']:.4f}  pos {b['pos']:.2f}  ik {b['ik']:.2f} deg "
              f"(arms {b['ik_arm']:.2f}, at limit {100 * b['atlim']:.1f}%) | fallen {mo['fallen']}"
              f"  travel {mo['travel']:.1f} m  end-point off {mo['end_xy']:.2f} m  pelvis {mo['pelvis']:.3f} m")
    print(f"|best z - z0| = {summ['dist_best_z0']:.2f}   -> {out}")


if __name__ == "__main__":
    main()
