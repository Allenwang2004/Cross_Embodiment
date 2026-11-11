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
import argparse, csv, json, sys, time
from pathlib import Path
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.geometry import rank_normalize_steps, smpl_heading, unit_sphere  # noqa: E402


def colored_noise(beta, shape, rng):
    """Gaussian noise with power spectrum 1/f^beta along the LAST axis, unit variance (Timmer & Koenig 1995, as in
    the colorednoise package that BFMTrack / Eberhard et al. use). beta 0 = white."""
    T = shape[-1]; f = np.fft.rfftfreq(T)
    s = np.maximum(f, 1.0 / T) ** (-beta / 2.0)
    w = s[1:].copy(); w[-1] *= (1 + (T % 2)) / 2.0
    sd = 2 * np.sqrt(np.sum(w ** 2)) / T
    sr = rng.standard_normal(shape[:-1] + (len(f),)) * s
    si = rng.standard_normal(shape[:-1] + (len(f),)) * s
    if T % 2 == 0:
        si[..., -1] = 0; sr[..., -1] *= np.sqrt(2)
    si[..., 0] = 0; sr[..., 0] *= np.sqrt(2)
    return np.fft.irfft(sr + 1j * si, n=T, axis=-1) / sd


def main():
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
    a = ap.parse_args()
    assert not a.antithetic or a.n % 2 == 0, "--antithetic needs an even --n"
    assert a.dtw or a.steps_scale == 1.0, "--steps-scale needs --dtw (the frame-by-frame reward has one frame per step)"
    assert not (a.dtw and a.objective in ("mse", "phc")), "--objective mse / phc are frame by frame"
    from crossenbodiment.obs import ExactObs, lso_scale_kw                  # the simulation layer (refactor step 5)
    from crossenbodiment.policy import MotivoPolicy
    from crossenbodiment.rollout import BatchRollout
    from crossenbodiment.sim import HumenvSim
    from model import bfm_align, losses

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    xml = Path(a.xml) if a.xml else REPO / "assets/robots_torque" / a.body / "robot_torque_full.xml"
    ref = np.load(REPO / a.ref)["qpos"]; T = len(ref); S = T - 1          # S control steps: g_1 .. g_T-1 are the goals
    SR = int(round(a.steps_scale * S))                                     # control steps rolled out
    kw = lso_scale_kw(a.scale)                                             # None: ExactObs's default (the child's s), any body
    n_envs = a.n + 1                                                       # slot 0: the mean sequence, then a.n samples
    # ExactObs computed in the env workers (ExactObsWrapper), after FloorFrictionWrapper when given;
    # max_episode_steps: humenv's TimeLimit (300 steps by default) would auto-reset clips longer than 10 s mid-rollout
    mapper = ExactObs(xml, REPO / "assets/robots/adult/robot.xml", **kw)
    sim = HumenvSim.make(xml, n_envs, wrappers=mapper.wrappers(), floor_friction=a.floor_friction, max_episode_steps=SR + 1)
    policy = MotivoPolicy.load(device=a.device); model = policy.model     # facebook/metamotivo-M-1, eval()
    fk = mujoco.MjModel.from_xml_path(str(xml)); nv = fk.nv
    exact = mapper.build()
    pos_div = exact.s if a.pos_unit == "adult" else 1.0
    phc = None
    if a.objective == "phc":
        from model.phc_reward import PHCCost
        phc = PHCCost(fk, ref[0], ref[1:], scale=pos_div)
    vr, ro = np.zeros(nv), []
    for t in range(T):
        if t:
            mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, ref[t - 1], ref[t])
        ro.append(exact(ref[t], vr))
    Bg = policy.embed(np.stack(ro))[1:]                                     # (S, 256): goal of step t is frame t+1
    Bstart = Bg
    VEL = slice(214, 358)                    # humenv proprio: local_body_vel + local_body_ang_vel
    Bc = Bg                                  # the reference side of the cost
    if a.bfm_zero_vel:
        ro0 = np.stack(ro); ro0[:, VEL] = 0
        Bc = bfm_align.embed(model, ro0, a.device)[1:]
    if a.init_pace != 1.0:                       # goal latents of the reference played at init_pace x speed
        from crossenbodiment.retiming import resample   # bfm_velocity_check's (refactor step 4)
        rs, vr, ro2 = resample(ref.astype(np.float64), a.init_pace), np.zeros(nv), []
        for t in range(len(rs)):
            if t:
                mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, rs[t - 1], rs[t])
            ro2.append(exact(rs[t], vr))
        Bstart = bfm_align.embed(model, np.stack(ro2), a.device)[1:]
    Bstart = Bstart[np.minimum(np.arange(SR), len(Bstart) - 1)]            # (SR, 256), the last goal held
    Bc_t = torch.as_tensor(Bc, device=a.device)
    unit = unit_sphere                                                         # x / |x| along the last axis
    v0 = np.zeros(nv)
    if a.init_vel_scale:
        mujoco.mj_differentiatePos(fk, v0, 1.0 / 30, ref[0], ref[1]); v0 *= a.init_vel_scale
    # one rollout of n_envs latent sequences from ref[0] moving at v0, no get_obs after set_physics (the start observation
    # is ExactObs(ref[0], v0)); 16 * Z on the unit sphere -> float32
    batch = BatchRollout(policy, sim, mapper, n_envs, SR, z_scale=16.0, fetch_start_obs=False, record=("qpos", "obs"))
    dk = mujoco.MjData(fk)

    def heading(q):
        """(..., nq) -> humenv heading (...), as single_z_search.root_heading_xy (batched: crossenbodiment.geometry)."""
        return smpl_heading(fk, q, dk, batched=True)
    ref_h, ref_xy = heading(ref[1:]), ref[1:, :2]
    hid = fk.body("Head").id
    if a.objective == "mse":
        import model.kinematics as kin
        ref_pose = kin.batch_forward_pose(fk, ref[1:], kin.EE_BODIES + [kin.ROOT_BODY])

    @torch.no_grad()
    def rollout(Z):
        """Z: (n_envs, S, 256) on the unit sphere -> per-step reward (n_envs, S) and qpos (n_envs, S, nq)."""
        ro_ = batch.run(Z, init_qpos=ref[0], init_qvel=v0)
        O = torch.as_tensor(ro_.obs, dtype=torch.float32, device=a.device)
        if a.bfm_zero_vel:
            O[..., VEL] = 0
        Bs = policy.backward_map(O.reshape(-1, O.shape[-1])).reshape(n_envs, SR, -1)
        q = ro_.qpos
        if a.dtw:
            from model import dtw
            Sm = dtw.cosine_matrix(Bs.cpu().numpy().astype(np.float64), Bc.astype(np.float64))
            sc, je, ie, path = dtw.align_slow(Sm)
            hz = np.empty(q.shape[:2])
            for k in range(n_envs):
                for i in range(SR):
                    dk.qpos[:] = q[k, i]; mujoco.mj_kinematics(fk, dk); hz[k, i] = dk.xpos[hid][2]
            hd_all = (1 - np.cos(heading(q) - ref_h[np.maximum(path, 0)])) / 2
            r = np.zeros((n_envs, SR)); fallen, head_al = np.zeros(n_envs), np.zeros(n_envs)
            for k in range(n_envs):
                mk = path[k] >= 0; jk = path[k][mk]; cnt = np.bincount(jk, minlength=S)
                nm = mk.sum()
                fl = hz[k][mk] < a.fall_head
                r[k, mk] = (Sm[k, np.flatnonzero(mk), jk] / cnt[jk]
                            - (a.heading_weight * hd_all[k][mk] + a.fall_weight * fl) * S / nm)
                fallen[k], head_al[k] = fl.mean(), hd_all[k][mk].mean()
            rollout.parts = dict(bfm=np.array([1 - (Sm[k, np.arange(min(SR, S)), np.arange(min(SR, S))]).mean() for k in range(n_envs)]),
                                 dtw=1 - sc, done=(je + 1) / S, pace=(je + 1) / (ie + 1), head=head_al, fallen=fallen, path=path)
            return r, q
        cos = torch.nn.functional.cosine_similarity(Bs, Bc_t[None], dim=-1).cpu().numpy()
        parts = dict(bfm=1 - cos.mean(1))
        r = cos
        if a.objective == "mse":
            mf = np.stack([losses.tracking_mse_frames(fk, q[k], ref[1:], root_xy=not a.mse_no_xy, ref_pose=ref_pose)
                           for k in range(n_envs)])
            r = 1 - mf; parts["mse_frames"] = mf.mean(1)
        if phc is not None:                                                            # every rollout starts at ref[0]
            r = np.stack([phc.parts(q[k], ref[0])[0] for k in range(n_envs)]); parts["phc"] = 1 - r.mean(1)
        if a.heading_weight or a.pos_weight:
            hd = (1 - np.cos(heading(q) - ref_h[None])) / 2; ps = np.linalg.norm(q[..., :2] - ref_xy[None], axis=-1) / pos_div
            r = r - a.heading_weight * hd - a.pos_weight * ps
            parts.update(head=hd.mean(1), pos=ps.mean(1))
        rollout.parts = parts
        return r, q

    def terms(r, i):
        """objective (1 - mean per-step reward) and its parts for slot i of the last rollout"""
        d = dict(cost=float(1 - r[i].sum() / S), bfm=float(rollout.parts["bfm"][i]))
        if "mse_frames" in rollout.parts:
            d["mse_obj"] = float(rollout.parts["mse_frames"][i])
        if "dtw" in rollout.parts:
            d.update({k: float(rollout.parts[k][i]) for k in ("dtw", "done", "pace", "fallen")}, head=float(rollout.parts["head"][i]),
                     pos_m=float("nan"))
        elif "head" in rollout.parts:
            d.update(head=float(rollout.parts["head"][i]), pos_m=float(rollout.parts["pos"][i]))
        return d

    def line(k, v):
        return (f"  {k:28s} cost {v['cost']:.4f} (bfm {v['bfm']:.4f}" + (f", heading {v['head']:.3f}, pos {v['pos_m']:.2f} m"
                if "head" in v else "") + (f", dtw {v['dtw']:.3f} done {v['done']:.2f} pace {v['pace']:.2f} fallen {v['fallen']:.3f}"
                if "dtw" in v else "") + f")  joint rms {v['joint_rms_deg']:.1f} deg  ee {v['ee_rms_cm']:.1f} cm  "
                f"root {v['root_rms_cm']:.1f} cm  end-point off {v['end_off_m']:.2f} m")

    def report(q):
        tot, terms = losses.tracking_mse(fk, q, ref[1:])
        return dict(mse=float(tot), joint_rms_deg=float(np.degrees(np.sqrt(terms["pose"]))),
                    ee_rms_cm=float(100 * np.sqrt(terms["ee"])), root_rms_cm=float(100 * np.sqrt(terms["root"])),
                    end_off_m=float(np.linalg.norm(q[-1, :2] - ref[-1, :2])), pelvis_min_m=float(q[:, 2].min()))

    # ---- zero-shot references: B(g_t) (the LSO start) and Motivo's ER windows -------------------------------------
    mu = unit(Bstart.astype(np.float64))
    base = {"B(g_t) (LSO start)": mu}
    hold = np.minimum(np.arange(SR), S - 1)
    for L in a.er_windows:
        base[f"ER L={L} (Motivo zero-shot)"] = unit(np.stack([Bg[t:t + L].mean(0) for t in range(S)]).astype(np.float64))[hold]
    Z = batch.copies(mu, n_envs)
    for i, z in enumerate(base.values()):
        Z[i] = z
    t0 = time.time(); r, q = rollout(Z)
    res = {k: dict(**terms(r, i), **report(q[i])) for i, k in enumerate(base)}
    tag = lambda k: "q_Bg" if k.startswith("B(") else "q_ER" + k.split("=")[1].split(" ")[0]
    keep = {tag(k): q[i].astype(np.float32) for i, k in enumerate(base)}
    print(f"{a.ref}: {T} frames, {S} steps ({SR} rolled out), {a.n} sequences per iteration, {a.iters} iterations ({time.time() - t0:.0f} s per batch)")
    for k, v in res.items():
        print(line(k, v), flush=True)

    # ---- LSO / trajectory adaptation ------------------------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    start_key = "B(g_t) (LSO start)" if a.method == "lso" else "ER L=8 (Motivo zero-shot)"
    if a.method == "ta":
        assert start_key in base and not a.antithetic and not a.rank, "--method ta: starts from ER L=8; no --antithetic / --rank"
        mu = base[start_key].copy()
    m, v = np.zeros_like(mu), np.zeros_like(mu); b1, b2 = a.betas
    disc = a.gamma ** np.arange(SR)
    curve, trace = [], [mu.astype(np.float32)]
    best = dict(cost=res[start_key]["cost"], z=mu.copy(), it=-1, src="start", q=q[list(base).index(start_key)].copy())
    budget_cost = {}
    ta_h = np.exp(-(SR - np.arange(1, SR + 1)) / (a.ta_beta2 * SR))                   # horizon factor, 1 at the last step
    for it in range(a.iters):
        if a.method == "ta":
            sig = a.sigma * np.exp(-(a.iters - (a.iters - it)) / (a.ta_beta1 * a.iters)) * ta_h   # i = M - it counts down from M
            eps = rng.standard_normal((a.n, SR, 256))
            Z = batch.with_mean(mu, unit(mu[None] + sig[None, :, None] * eps))
        elif a.antithetic:                                                               # pairs: eps^i, -eps^i
            h = colored_noise(a.beta, (a.n // 2, 256, SR), rng).transpose(0, 2, 1)
            eps = np.concatenate([h, -h])
        else:
            eps = colored_noise(a.beta, (a.n, 256, SR), rng).transpose(0, 2, 1)       # (n, SR, 256), coloured over time
        if a.method == "lso":
            Z = batch.with_mean(mu, unit(mu[None] + a.sigma * eps))
        r, q = rollout(Z)
        if it in a.keep_iters:                                                         # slot 0: the mean after `it` updates
            keep[f"q_it{it}"] = q[0].astype(np.float32)
        if it in a.budgets:
            keep[f"q_mean_it{it}"] = q[0].astype(np.float32)
        costs = 1 - r.sum(1) / S
        i = 1 + int(np.argmin(costs[1:]))                                              # samples only: slot 0 (the mean) is logged, never the result
        if costs[i] < best["cost"]:
            best = dict(cost=float(costs[i]), z=Z[i].copy(), it=it, src="sample", q=q[i].copy())
        if it + 1 in a.budgets:
            keep[f"q_best_it{it + 1}"] = best["q"].astype(np.float32); budget_cost[it + 1] = best["cost"]
        if a.method == "ta":                                                           # softmax over the z-scored costs of the samples
            c = costs[1:]; zc = (c - c.mean()) / (c.std() + 1e-12); w = np.exp(-(zc - zc.min()) / a.ta_temp); w /= w.sum()
            mu = unit((w[:, None, None] * Z[1:]).sum(0))
            row = dict(iter=it, mean_seq_cost=float(costs[0]), mean_seq_bfm=float(rollout.parts["bfm"][0]),
                       best_sample_cost=float(costs[1:].min()), sample_mean_cost=float(costs[1:].mean()), best_so_far=best["cost"],
                       mu_move=float(np.linalg.norm(16 * (mu - trace[-1]), axis=1).mean()), sec=time.time() - t0)
            curve.append(row); trace.append(mu.astype(np.float32))
            print(f"  iter {it:3d}  mean sequence cost {costs[0]:.4f}  samples best {costs[1:].min():.4f} mean {costs[1:].mean():.4f}"
                  f"  top weight {w.max():.3f}  [{row['sec'] / 60:.1f} min]", flush=True)
            continue
        rs = r[1:]
        R = np.stack([np.convolve(x[::-1], disc)[:SR][::-1] for x in rs])              # discounted return-to-go
        if a.rank:                                                                     # centred ranks, per step
            R = rank_normalize_steps(R)                                                # R has a.n rows
        if a.antithetic:
            k = a.n // 2
            g = ((R[:k] - R[k:])[:, :, None] * eps[:k]).sum(0) / (2 * k * a.sigma)
        else:
            A = R - (R.sum(0, keepdims=True) - R) / (a.n - 1)
            g = (A[:, :, None] * eps).mean(0) / a.sigma
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mu = unit(mu + a.lr * (m / (1 - b1 ** (it + 1))) / (np.sqrt(v / (1 - b2 ** (it + 1))) + 1e-8))
        row = dict(iter=it, mean_seq_cost=float(costs[0]), mean_seq_bfm=float(rollout.parts["bfm"][0]),
                   best_sample_cost=float(costs[1:].min()), sample_mean_cost=float(costs[1:].mean()), best_so_far=best["cost"],
                   mu_move=float(np.linalg.norm(16 * (mu - trace[-1]), axis=1).mean()), sec=time.time() - t0)
        curve.append(row); trace.append(mu.astype(np.float32))
        bi = 1 + int(np.argmin(costs[1:])); dl = (f" | best sample dtw {rollout.parts['dtw'][bi]:.3f} pace {rollout.parts['pace'][bi]:.2f} "
                                                    f"fallen {rollout.parts['fallen'][bi]:.3f}" if "dtw" in rollout.parts else "")
        print(f"  iter {it:3d}  mean sequence cost {costs[0]:.4f} (bfm {rollout.parts['bfm'][0]:.4f})  samples best {costs[1:].min():.4f} mean {costs[1:].mean():.4f}"
              f"  |d mu_t| {row['mu_move']:.3f}{dl}  [{row['sec'] / 60:.1f} min]", flush=True)
    Z = batch.copies(mu, n_envs); r, q = rollout(Z)                                    # the final mean sequence
    res["LSO final mean"] = dict(**terms(r, 0), **report(q[0])); keep["q_LSO"] = q[0].astype(np.float32)
    if a.iters in a.budgets:
        keep[f"q_mean_it{a.iters}"] = q[0].astype(np.float32)
    if "path" in rollout.parts:
        keep["path_LSO"] = rollout.parts["path"][0]
    Zb = batch.copies(best["z"], n_envs); rb, qb = rollout(Zb)
    res["LSO best (re-rolled)"] = dict(**terms(rb, 0), it=best["it"], src=best["src"], **report(qb[0]))
    keep["q_LSO_best"] = qb[0].astype(np.float32)
    if "path" in rollout.parts:
        keep["path_LSO_best"] = rollout.parts["path"][0]
    sim.close()
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
    its = list(range(len(trace))) if a.trace == "all" else [0] + [k for k in a.budgets if k < len(trace)]
    np.savez_compressed(out / "mu_trace.npz", mu=np.stack([trace[k] for k in its]), mu_iters=np.array(its), best=best["z"].astype(np.float32))
    np.savez_compressed(out / "rollouts.npz", ref=ref.astype(np.float32), **keep)
    (out / "summary.json").write_text(json.dumps(dict(ref=a.ref, body=a.body, frames=T, n=a.n, iters=a.iters, sigma=a.sigma,
                                                      lr=a.lr, gamma=a.gamma, beta=a.beta, seed=a.seed, heading_weight=a.heading_weight,
                                                      pos_weight=a.pos_weight, pos_unit=a.pos_unit, pos_div=pos_div, antithetic=a.antithetic, rank=a.rank, floor_friction=a.floor_friction,
                                                      steps_scale=a.steps_scale, init_pace=a.init_pace, dtw=a.dtw, init_vel_scale=a.init_vel_scale,
                                                      fall_weight=a.fall_weight, fall_head=a.fall_head,
                                                      bfm_zero_vel=a.bfm_zero_vel, objective=a.objective, mse_no_xy=a.mse_no_xy,
                                                      method=a.method, ta_beta1=a.ta_beta1, ta_beta2=a.ta_beta2, ta_temp=a.ta_temp, betas=list(a.betas),
                                                      budgets=a.budgets, best_cost_at=budget_cost, best_it=best["it"], best_src=best["src"],
                                                      sample_rollouts=a.iters * a.n, results=res,
                                                      minutes=(time.time() - t0) / 60), indent=1))
    print("\n" + "\n".join(line(k, v) for k, v in res.items()))
    print(f"-> {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
