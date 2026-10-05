#!/usr/bin/env python3
"""segment_search_upper_bound.py -- how much more does a z that changes every second buy than the best single z?

Per clip, on the child, exact observations (model/exact_obs.py), full 256-d z:
  z0        the clip's adult latent
  single    the best single z of the exact_train search (outputs/exact_train/<clip>/align/best_z.npy)
  segment   a z per second, searched greedily (receding horizon): for segment k = 0..9 the committed z of segments
            < k are replayed from the reference start, and antithetic ES + Adam (single_z_search's update) searches
            the z of segment k on L_align over frames [kL, kL + L + look) -- this second and the next one, the
            candidate held through both. Segment k starts from the better of the previous segment's z and the
            single z (both scored), so every window is no worse than the single z's. The best z is committed.
No anchor: an upper bound, not a learnable target.
Final: full 300-step rollouts of the three, L_align (joint space, all frames of the reference) and per-second.
Writes outputs/segment_search/<stem>/{zseq.npy, summary.json}.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO)); sys.path.append(str(REPO / "scripts"))
DS = REPO / "datasets/crossenbodiment-child-balanced"


def project(z):
    return z * (16 / np.linalg.norm(z, axis=-1, keepdims=True))


def rank_normalize(f):
    return np.argsort(np.argsort(f)) / max(len(f) - 1, 1) - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="task/stem")
    ap.add_argument("--seg", type=int, default=30)
    ap.add_argument("--look", type=int, default=30)
    ap.add_argument("--gens", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import losses
    from model.dataset import CrossEmbodimentDataset
    from model.simple.config import ESConfig
    from model.simple.train import make_body_ctx
    from model.simple.train_es import set_init_qpos
    rng = np.random.default_rng(args.seed); dev = "cuda"
    P, C, L = args.pairs, len(args.clips), args.seg
    cfg = ESConfig(); cfg.device = dev; cfg.obs_scale = "exact"; cfg.batch_size = 2 * P * C; cfg.vectorization_mode = "async"
    W = {"root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
         "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight}
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    ds = CrossEmbodimentDataset(DS)
    at = {(r["reward_name"], r["trial"]): i for i, r in enumerate(ds.rows) if r["morphology_label"] == "child"}
    stems = [c.split("/")[1] for c in args.clips]
    keys = [(s.rsplit("_", 1)[0], int(s.rsplit("_", 1)[1])) for s in stems]
    smp = [ds[at[k]] for k in keys]
    refs = [np.asarray(s["qpos_ref"], dtype=np.float64) for s in smp]
    z0 = np.stack([np.asarray(s["z0"], dtype=np.float64).reshape(-1) for s in smp]); z0 = project(z0)
    zs = project(np.stack([np.load(REPO / f"outputs/exact_train/{s}/align/best_z.npy").reshape(-1).astype(np.float64) for s in stems]))
    ctx = make_body_ctx(cfg, DS, "child", ds.rows[at[keys[0]]]["target_xml"])
    env, X, fk = ctx["env"], ctx["exact"], ctx["fk"]; B, nv, T = cfg.batch_size, fk.nv, cfg.steps_per_episode
    K = (T + L - 1) // L
    owner = np.repeat(np.arange(C), 2 * P)                      # slot -> clip

    def rollout(Z, steps):
        """Z: (B, T, 256) per-step z. Returns qpos (B, steps, nq)."""
        env.reset()
        set_init_qpos(env, [refs[c][0] for c in owner], nv)
        ex = np.stack([X(refs[c][0], np.zeros(nv)) for c in owner])
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev); Q = []
        for t in range(steps):
            with torch.no_grad():
                a = model._actor(model._normalize(torch.as_tensor(ex, dtype=torch.float32, device=dev)), Zt[:, t],
                                 model.cfg.actor_std).mean.cpu().numpy()
            _, _, _, _, info = env.step(a)
            Q.append(info["qpos"].copy())
            ex = np.stack([X(q, v) for q, v in zip(info["qpos"], info["qvel"])])
        return np.stack(Q, 1)

    def l_align(q, ref, a, b):
        b = min(b, len(ref), q.shape[0])
        return float(losses.functional_equivalence(fk, q[a:b], ref[a:b], W, 1 / 30)[0]) if b - a > 1 else float("nan")

    committed = np.repeat(zs[:, None], K, axis=1)               # (C, K, 256)
    win_log = []
    t_all = time.time()
    for k in range(K):
        t0 = time.time()
        a, b = k * L, min(T, (k + 1) * L + args.look); steps = b
        live = [c for c in range(C) if a < len(refs[c]) - 1]
        if not live:
            break

        def score(cand):                                        # cand (B, 256) -> (B,) window L_align
            Z = np.empty((B, T, 256))
            for i in range(B):
                Z[i] = np.repeat(committed[owner[i]], L, axis=0)[:T]
                Z[i, a:] = cand[i]
            Q = rollout(Z, steps)
            return np.array([l_align(Q[i], refs[owner[i]], a, b) for i in range(B)])

        # start: the better of the previous segment's z and the single z, both scored on this window
        prev = committed[:, max(k - 1, 0)]
        start = np.concatenate([np.repeat(prev, 2 * P, axis=0).reshape(C, 2 * P, 256)[:, :P],
                                np.repeat(zs, 2 * P, axis=0).reshape(C, 2 * P, 256)[:, P:]], 1).reshape(B, 256)
        f0 = score(start).reshape(C, 2 * P)
        z = np.where((f0[:, 0] <= f0[:, P])[:, None], prev, zs).copy()
        best_f = np.minimum(f0[:, 0], f0[:, P]); best_z = z.copy(); single_f = f0[:, P].copy()
        m, v = np.zeros_like(z), np.zeros_like(z)
        for g in range(args.gens):
            eps = rng.standard_normal((C, P, 256))
            cand = project(np.concatenate([z[:, None] + args.sigma * eps, z[:, None] - args.sigma * eps], 1)).reshape(B, 256)
            f = score(cand).reshape(C, 2 * P)
            for c in live:
                i = int(np.argmin(f[c]))
                if f[c, i] < best_f[c]:
                    best_f[c], best_z[c] = f[c, i], cand.reshape(C, 2 * P, 256)[c, i]
                s = rank_normalize(f[c])
                grad = ((s[:P] - s[P:])[:, None] * eps[c]).sum(0) / (2 * P * args.sigma)
                m[c] = 0.9 * m[c] + 0.1 * grad; v[c] = 0.999 * v[c] + 0.001 * grad ** 2
                step = args.lr * (m[c] / (1 - 0.9 ** (g + 1))) / (np.sqrt(v[c] / (1 - 0.999 ** (g + 1))) + 1e-8)
                z[c] = project(z[c] - step)
        for c in live:
            committed[c, k:] = best_z[c]
        win_log.append(dict(k=k, single=single_f.tolist(), best=best_f.tolist()))
        print(f"segment {k} frames [{a},{b}) ({(time.time() - t0) / 60:.1f} min): window L_align single -> segment search  "
              + "  ".join(f"{stems[c][:18]} {single_f[c]:.3f}->{best_f[c]:.3f}" for c in live), flush=True)

    # final: full rollouts of z0, single, segment sequence
    Z = np.empty((B, T, 256)); lab = []
    for i in range(B):
        c, j = owner[i], i % (2 * P)
        if j == 0:
            Z[i] = z0[c]; lab.append("z0")
        elif j == 1:
            Z[i] = zs[c]; lab.append("single")
        else:
            Z[i] = np.repeat(committed[c], L, axis=0)[:T]; lab.append("segment")
    Q = rollout(Z, T)
    rows = {}
    for i in range(B):
        c, j = owner[i], i % (2 * P)
        if j > 2:
            continue
        n = min(len(refs[c]), T)
        rows.setdefault(stems[c], {})[lab[i]] = dict(
            L_align=l_align(Q[i], refs[c], 0, n),
            per_second=[l_align(Q[i], refs[c], s, s + L) for s in range(0, n - 1, L)])
    for c, s in enumerate(stems):
        out = REPO / f"outputs/segment_search/{s}"; out.mkdir(parents=True, exist_ok=True)
        np.save(out / "zseq.npy", committed[c].astype(np.float32))
        d = np.linalg.norm(committed[c] - zs[c], axis=1)
        rows[s]["segment_dist_to_single"] = d.tolist()
        rows[s]["segment_step_dist"] = np.linalg.norm(np.diff(committed[c], axis=0), axis=1).tolist()
        json.dump(dict(clip=args.clips[c], windows=[dict(k=w["k"], single=w["single"][c], best=w["best"][c]) for w in win_log],
                       final=rows[s], args=vars(args)), open(out / "summary.json", "w"), indent=1)
    print(f"\n{'clip':30s} | full L_align: z0  single  segment | per-second L_align single / segment")
    for s, r in rows.items():
        print(f"{s:30s} | {r['z0']['L_align']:.3f}  {r['single']['L_align']:.3f}  {r['segment']['L_align']:.3f}"
              f"  | {' '.join(f'{x:.2f}' for x in r['single']['per_second'])}  /  {' '.join(f'{x:.2f}' for x in r['segment']['per_second'])}"
              f"  | |z_k - single| mean {np.mean(r['segment_dist_to_single']):.2f}")
    print(f"total {(time.time() - t_all) / 60:.1f} min")
    env.close()


if __name__ == "__main__":
    main()
