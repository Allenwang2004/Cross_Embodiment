#!/usr/bin/env python3
"""g1_z_search.py -- single_z_search.py's latent search, on the Unitree G1 driven by an unchanged Meta Motivo.

Same search as single_z_search.py --algo es (antithetic pairs, rank-normalised costs, Adam on z, z kept on
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
--objective g1mse puts squared errors against the clip mapped onto the G1 kinematically (outputs/g1/roundtrip/<stem>/
g1.npz, the G1 target) in place of bfm + heading + pos, on the G1's own body, weights 1 as losses.tracking_mse:
         pose     mean (G1 joint angle - target)^2 over the 29 joints, wrapped                            rad^2
         ee       mean |p - p_target|^2 of the hands (wrist_yaw_link), feet (foot sites) and head (the torso_link's
                  top mesh), relative to the pelvis in its heading frame                                  m^2
         root     mean |pelvis - pelvis_target|^2, world x, y, z                                          m^2
         heading  mean 1 - cos(pelvis yaw - target yaw)
         (logged under either objective as "mse"; anchor and ik terms as above)
Writes <out>/{best_z.npy, best.npz, origin_z.npz, z_trace.npz, curve.csv, summary.json}; best.npz / origin_z.npz hold
the G1 qpos ("qpos"), its SMPL-equivalent ("qpos_smpl") and per step the 13 IK residuals (deg) + limit count ("fz").
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/g1_z_search.py [--clip ...] [--evals 4096]
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.geometry import project_z, rank_normalize, ang_pair   # noqa: E402  single_z_search re-exported it (refactor step 4)
from crossenbodiment.geometry import robot_yaw, smpl_heading       # noqa: E402
from crossenbodiment.robot_rollout import motion   # noqa: E402  g1_z0_rollout's (refactor step 4)
from model import bfm_align                                         # noqa: E402
from model.g1_env import G1VecEnv, G1_XML                           # noqa: E402
from model.g1_map import PAIRS                                      # noqa: E402

ARM = [i for i, (b, _) in enumerate(PAIRS[1:]) if b.split("_")[-1] in ("Shoulder", "Elbow", "Hand")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="move-ego-0-2/move-ego-0-2_4")
    p.add_argument("--pairs", type=int, default=8)
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
    p.add_argument("--gains", default="perjoint", choices=["perjoint", "mass"], help="G1 PD gains, see model/g1_env.py")
    p.add_argument("--objective", default="bfm", choices=["bfm", "g1mse"],
                   help="g1mse: squared errors against the G1 target in place of bfm + heading + pos (see above)")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    task, stem = a.clip.split("/")
    tag = "" if a.objective == "bfm" else f"_{a.objective}"
    out = Path(a.out) if a.out else REPO / "outputs/g1/search" / f"{stem}{tag}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    n_envs = 2 * a.pairs; n_gens = max(a.evals // n_envs, 1)

    vec = G1VecEnv(n_envs, gains=a.gains)                                  # spawn the workers before CUDA is touched
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    ref = np.load(REPO / f"data/origin_motion/{a.clip}.npz")["qpos"]
    T = len(ref)
    z0 = project_z(np.load(REPO / f"data/origin_z/{a.clip}.npy").reshape(-1).astype(np.float64))
    anchor = project_z(np.load(a.anchor).reshape(-1).astype(np.float64)) if a.anchor else z0
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(a.device)
    adult, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
    fk = adult.unwrapped.model; dk = mujoco.MjData(fk)
    Bg = bfm_align.embed(model, bfm_align.obs_from_qpos(adult, ref), a.device)
    gref = np.load(REPO / f"outputs/g1/roundtrip/{stem}/g1.npz")["qpos"]
    mg = mujoco.MjModel.from_xml_path(str(REPO / G1_XML)); gd = mujoco.MjData(mg)
    torso = mg.body("torso_link").id
    head_geom = max((g for g in range(mg.ngeom) if mg.geom_bodyid[g] == torso), key=lambda g: mg.geom_pos[g][2])
    hands = [mg.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    feet = [mg.site(n).id for n in ("left_foot", "right_foot")]

    def g1_points(qg):
        """(..., 36) G1 qpos -> pelvis xyz (..., 3), pelvis yaw (...), hands / feet / head relative to the pelvis in
        its heading frame (..., 5, 3)."""
        flat = qg.reshape(-1, qg.shape[-1]); P = np.empty((len(flat), 3)); H = np.empty(len(flat))
        E = np.empty((len(flat), 5, 3))
        for i, qq in enumerate(flat):
            gd.qpos[:] = qq; mujoco.mj_kinematics(mg, gd)
            pel = gd.xpos[1].copy(); h = robot_yaw(gd.xmat[1])
            d = np.stack([gd.xpos[hands[0]], gd.xpos[hands[1]], gd.site_xpos[feet[0]], gd.site_xpos[feet[1]],
                          gd.geom_xpos[head_geom]]) - pel
            c, sn = np.cos(h), np.sin(h)
            E[i] = np.stack([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1], d[:, 2]], -1); P[i], H[i] = pel, h
        sh = qg.shape[:-1]
        return P.reshape(sh + (3,)), H.reshape(sh), E.reshape(sh + (5, 3))
    gT = gref[:T]
    ref_P, ref_H, ref_E = g1_points(gT)

    def g1mse(QG):
        P, H, E = g1_points(QG)
        dq = (QG[..., 7:] - gT[None, :, 7:] + np.pi) % (2 * np.pi) - np.pi
        return ((dq ** 2).mean((1, 2)) + ((E - ref_E[None]) ** 2).sum(-1).mean((1, 2))
                + ((P - ref_P[None]) ** 2).sum(-1).mean(1) + (1 - np.cos(H - ref_H[None])).mean(1))

    def heading_xy(qs):
        return smpl_heading(fk, qs, dk), qs[..., :2]                     # humenv's heading, frame by frame
    ref_h, ref_xy = heading_xy(ref)

    @torch.no_grad()
    def score(zs):
        zt = torch.as_tensor(zs, dtype=torch.float32, device=a.device)
        obs, _, _, _ = vec.reset(ref[0])
        O, QS, QG, FZ = [], [], [], []
        for _ in range(T):
            o = torch.as_tensor(obs, dtype=torch.float32, device=a.device)
            u = model._actor(model._normalize(o), zt, model.cfg.actor_std).mean.cpu().numpy()
            obs, qs, qg, fz = vec.step(u)
            O.append(obs); QS.append(qs); QG.append(qg); FZ.append(fz)
        O, QS, QG, FZ = (np.stack(x, axis=1) for x in (O, QS, QG, FZ))
        ik = FZ[:, :, :13].mean((1, 2)); ik_arm = FZ[:, :, ARM].mean((1, 2)); atlim = FZ[:, :, 13].mean(1) / 29
        bfm = bfm_align.batch_bfm_align(model, O.astype(np.float32), Bg, a.device)
        h, xy = heading_xy(QS)
        head = ((1 - np.cos(h - ref_h[None])) / 2).mean(1)
        pos = np.linalg.norm(xy - ref_xy[None], axis=-1).mean(1)
        adist = np.linalg.norm(np.asarray(zs) - anchor, axis=1)
        mse = g1mse(QG)
        main = mse if a.objective == "g1mse" else bfm + a.heading_weight * head + a.pos_weight * pos
        cost = main + a.anchor_weight * (adist / 16.0) ** 2 + a.ik_weight * np.radians(ik)
        return dict(cost=cost, bfm=bfm, mse=mse, head=head, pos=pos, anchor_dist=adist, ik=ik, ik_arm=ik_arm,
                    atlim=atlim, qg=QG, qs=QS, fz=FZ)

    def pick(r, i, z, gen, source):
        return dict(cost=float(r["cost"][i]), bfm=float(r["bfm"][i]), mse=float(r["mse"][i]), head=float(r["head"][i]),
                    pos=float(r["pos"][i]),
                    ik=float(r["ik"][i]), ik_arm=float(r["ik_arm"][i]), atlim=float(r["atlim"][i]),
                    z=z.copy(), qg=r["qg"][i].copy(), qs=r["qs"][i].copy(), fz=r["fz"][i].copy(), gen=gen, source=source)

    print(f"G1 latent search on {a.clip}, objective {a.objective}: {n_gens} steps x {n_envs} rollouts = {n_gens * n_envs} evals, "
          f"sigma {a.sigma}, lr {a.lr}, heading {a.heading_weight}, pos {a.pos_weight}, anchor {a.anchor_weight}, "
          f"ik {a.ik_weight}", flush=True)
    t0 = time.time()
    r0 = score(np.repeat(z0[None], n_envs, axis=0))
    spread = float(np.abs(r0["qg"] - r0["qg"][:1]).max())
    base = pick(r0, 0, z0, -1, "origin_z")
    print(f"  origin_z: cost {base['cost']:.4f}  bfm {base['bfm']:.4f}  mse {base['mse']:.4f}  heading {base['head']:.4f}  pos {base['pos']:.2f}"
          f"  ik {base['ik']:.2f} deg (arms {base['ik_arm']:.2f}, at limit {100 * base['atlim']:.1f}%)"
          f"   (16 copies identical to {spread:.1e}; {time.time() - t0:.0f} s per step)", flush=True)

    rng = np.random.default_rng(a.seed)
    z = z0.copy(); m = np.zeros_like(z); v = np.zeros_like(z); b1, b2, eps_adam = 0.9, 0.999, 1e-8
    best, curve, trace_mean, trace_best, path_len = base, [], [z0.copy()], [z0.copy()], 0.0
    for gen in range(n_gens):
        eps = rng.standard_normal((a.pairs, z.size))
        cand = project_z(np.concatenate([z + a.sigma * eps, z - a.sigma * eps], axis=0))
        r = score(cand)
        i = int(np.argmin(r["cost"]))
        if r["cost"][i] < best["cost"]:
            best = pick(r, i, cand[i], gen, "sample")
        s = rank_normalize(r["cost"])
        g = ((s[:a.pairs] - s[a.pairs:])[:, None] * eps).sum(0) / (2 * a.pairs * a.sigma)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        step = a.lr * (m / (1 - b1 ** (gen + 1))) / (np.sqrt(v / (1 - b2 ** (gen + 1))) + eps_adam)
        z_prev, z = z, project_z(z - step)
        path_len += ang_pair(z_prev, z)
        row = dict(gen=gen, evals=(gen + 1) * n_envs, gen_best=float(r["cost"].min()), gen_mean=float(r["cost"].mean()),
                   best_so_far=best["cost"], best_bfm=best["bfm"], best_ik=best["ik"], mean_z_cost="",
                   cos_z0=float(z @ z0 / (np.linalg.norm(z) * np.linalg.norm(z0))),
                   path_len_deg=round(path_len, 2), net_deg=round(ang_pair(z0, z), 2))
        if a.eval_every and (gen % a.eval_every == 0 or gen == n_gens - 1):
            rm = score(np.repeat(z[None], n_envs, axis=0))
            row["mean_z_cost"] = float(rm["cost"][0])
            if rm["cost"][0] < best["cost"]:
                best = pick(rm, 0, z, gen, "mean_z")
            print(f"  gen {gen:4d}/{n_gens}  best {best['cost']:.4f} (bfm {best['bfm']:.4f}, mse {best['mse']:.4f}, heading {best['head']:.4f}, "
                  f"pos {best['pos']:.2f}, ik {best['ik']:.2f} deg)  gen_best {r['cost'].min():.4f}  mean_z {rm['cost'][0]:.4f}  "
                  f"|z-z0| {np.linalg.norm(z - z0):.2f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        curve.append(row); trace_mean.append(z.copy()); trace_best.append(best["z"].copy())
    vec.close()

    np.save(out / "best_z.npy", best["z"].astype(np.float32).reshape(1, -1))
    for f, b in (("best.npz", best), ("origin_z.npz", base)):
        np.savez_compressed(out / f, qpos=b["qg"].astype(np.float32), qpos_smpl=b["qs"].astype(np.float32),
                            fz=b["fz"].astype(np.float32), fps=30)
    np.savez_compressed(out / "z_trace.npz", gen=np.arange(-1, len(trace_mean) - 1, dtype=np.int32),
                        z_mean=np.stack(trace_mean).astype(np.float32), z_best=np.stack(trace_best).astype(np.float32))
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
    keys = ("cost", "bfm", "mse", "head", "pos", "ik", "ik_arm", "atlim", "gen", "source")
    summ = dict(clip=a.clip, body="g1", evals=n_gens * n_envs, generations=n_gens, pairs=a.pairs, sigma=a.sigma, lr=a.lr,
                seed=a.seed, heading_weight=a.heading_weight, pos_weight=a.pos_weight, anchor=a.anchor or "z0",
                anchor_weight=a.anchor_weight, ik_weight=a.ik_weight, gains=a.gains, objective=a.objective, bfm_floor_of_map=0.176,
                origin_z={**{k: base[k] for k in keys}, "motion": motion(base["qg"], gref)},
                best={**{k: best[k] for k in keys}, "motion": motion(best["qg"], gref)},
                improvement_vs_origin_z=(base["cost"] - best["cost"]) / abs(base["cost"]),
                dist_best_z0=float(np.linalg.norm(best["z"] - z0)), path_len_deg=round(path_len, 1),
                minutes=(time.time() - t0) / 60)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\n=== {n_gens * n_envs} evals in {summ['minutes']:.1f} min ===")
    for n, b in (("origin_z", base), ("best z", best)):
        mo = summ["origin_z" if n == "origin_z" else "best"]["motion"]
        print(f"{n:9s} cost {b['cost']:.4f}  bfm {b['bfm']:.4f}  mse {b['mse']:.4f}  heading {b['head']:.4f}  pos {b['pos']:.2f}  ik {b['ik']:.2f} deg "
              f"(arms {b['ik_arm']:.2f}, at limit {100 * b['atlim']:.1f}%) | fallen {mo['fallen']}"
              f"  travel {mo['travel']:.1f} m  end-point off {mo['end_xy']:.2f} m  pelvis {mo['pelvis']:.3f} m")
    print(f"|best z - z0| = {summ['dist_best_z0']:.2f}   -> {out}")


if __name__ == "__main__":
    main()
