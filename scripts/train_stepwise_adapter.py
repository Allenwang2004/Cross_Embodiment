#!/usr/bin/env python3
"""train_stepwise_adapter.py -- an adapter that re-chooses z every segment from what the body is doing.

The per-clip adapters (train_es.py) give ONE z per clip, from (beta, z0), before the rollout starts. Here
the correction is closed-loop, RMA-style but in the latent: every L steps

    x_k = [B(s_t), B(g_{t+L}), z0]           (current state, the target L steps ahead, the clip's z0;
                                              observations are the adult-equivalent ones, model/exact_obs.py)
    z_k = project(z0 + c_phi(x_k) @ U)       (c: 8 coefficients of the exact-observation correction basis U)

and the frozen policy runs on z_k for the next L steps. The last layer starts at zero, so update 0 IS z0.

Training (no labels): antithetic ES on the OUTPUT of every segment. Each candidate rollout gets an independent
perturbation eps_k per segment (the - rollout of a pair gets -eps_k); the credit of segment k is the bfm cost of
its own frames and the next segment's (frames [kL, (k+2)L)), plus the anchor term lam (|z_k - z0| / 16)^2. The
per-(row, segment) difference is rank-shaped over the pairs, giving g_k in coefficient space, and the surrogate
sum_k g_k . c_phi(x_k) (x_k from both rollouts of the pair) is backpropagated into phi.
Same clip split as the b500 ES runs (10% held out per category). Evals (deterministic, eps = 0) every --eval-every
updates on the 50 held-out clips and 50 training clips: bfm (exact) and L_align, per clip.
Writes outputs/stepwise/<tag>/{eval_history.json, adapter_<u>.pt}.
"""
import argparse, json, random, sys, time
from pathlib import Path
import mujoco
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO)); sys.path.append(str(REPO / "scripts"))
DS = REPO / "datasets/crossenbodiment-child-balanced"


class StepAdapter(nn.Module):
    def __init__(self, U, hidden=256):
        super().__init__()
        self.register_buffer("U", torch.as_tensor(U, dtype=torch.float32))
        self.net = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, U.shape[0]))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seg30")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--seg", type=int, default=30, help="steps per segment (30 = 1 s)")
    ap.add_argument("--clips-per-update", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--basis", default="outputs/lowdim_search/basis_corrPCA_train_exact.npy")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--phase", action="store_true", help="target and cost follow the body's matched reference phase")
    ap.add_argument("--phase-win", type=int, default=10, help="phase search: expected index +- this many frames")
    ap.add_argument("--lag-weight", type=float, default=0.05, help="per segment: w |phase - clock| / seg")
    ap.add_argument("--clip-limit", type=int, default=0, help="debug: use only the first N train/test clips")
    args = ap.parse_args()
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.dataset import CrossEmbodimentDataset
    from model.simple.config import ESConfig
    from model.simple.train import make_body_ctx, compute_batch_cost
    from model.simple.train_es import set_init_qpos
    from build_b500_sup_dataset import heldout
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); random.seed(args.seed)
    dev = "cuda"
    out = REPO / f"outputs/stepwise/{args.tag}"; out.mkdir(parents=True, exist_ok=True)
    cfg = ESConfig(); cfg.device = dev; cfg.obs_scale = "exact"; cfg.batch_size = 2 * args.pairs * args.clips_per_update
    cfg.vectorization_mode = "async"; cfg.init_from_reference = True
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cat = dict(l.split() for l in open(DS / "splits/balanced500_categories.txt") if l.strip())
    clips = sorted((t, int(k)) for t, k in (l.split() for l in open(DS / "splits/balanced500_clips.txt") if l.strip()))
    ds = CrossEmbodimentDataset(DS)
    at = {(r["reward_name"], r["trial"]): i for i, r in enumerate(ds.rows) if r["morphology_label"] == "child"}
    held = heldout(clips, cat); test = sorted(held); train = sorted(set(clips) - held)
    ev_train = sorted(random.Random(1).sample(train, 50))
    if args.clip_limit:
        test, ev_train = test[: args.clip_limit], ev_train[: args.clip_limit]
        train = sorted(set(ev_train) | set(train[: args.clip_limit])); clips = sorted(set(train) | set(test))
    ctx = make_body_ctx(cfg, DS, "child", ds.rows[at[clips[0]]]["target_xml"])
    env, X, fk = ctx["env"], ctx["exact"], ctx["fk"]; nv, B = fk.nv, cfg.batch_size
    U = np.load(REPO / args.basis)[: args.k]; Ut = torch.as_tensor(U, dtype=torch.float32, device=dev)
    scale = (256 / args.k) ** 0.5

    # per clip: reference, z0, B of the reference through the exact translation
    S = {}
    t0 = time.time()
    for c in clips:
        s = ds[at[c]]; ref = np.asarray(s["qpos_ref"], dtype=np.float64)
        v, ob = np.zeros(nv), []
        for t in range(len(ref)):
            if t:
                mujoco.mj_differentiatePos(fk, v, 1 / 30, ref[t - 1], ref[t])
            ob.append(X(ref[t], v))
        with torch.no_grad():
            Bg = model.backward_map(torch.as_tensor(np.array(ob, dtype=np.float32), device=dev))
        S[c] = dict(ref=ref, z0=torch.as_tensor(np.asarray(s["z0"], dtype=np.float32).reshape(-1), device=dev), Bg=Bg)
    print(f"{len(clips)} clips ({len(train)} train / {len(test)} test), reference embeddings in {(time.time() - t0) / 60:.1f} min", flush=True)

    adapter = StepAdapter(U).to(dev)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    T, L = cfg.steps_per_episode, args.seg; K = (T + L - 1) // L

    def match_phase(bs, slots, tau, step):
        """Each slot's reference frame closest (cos of B) to its current state, searched within the expected index
        tau + step +- phase_win; the window keeps periodic motions (gait) from snapping a cycle ahead or back."""
        out = np.empty(len(slots), dtype=np.int64)
        for i, c in enumerate(slots):
            Bg = S[c]["Bg"]; n = len(Bg); e = int(tau[i]) + step
            lo, hi = min(max(e - args.phase_win, 0), n - 1), min(max(e + args.phase_win, 0), n - 1)
            out[i] = lo + int(nn.functional.cosine_similarity(bs[i:i + 1], Bg[lo:hi + 1], dim=-1).argmax())
        return out

    def rollout(slots, eps=None):
        """slots: list of clips (len B). eps: (B, K, k) coefficient-space noise or None. Returns qpos (B,T,nq),
        Bs (B,T,256), inputs (B,K,768), coefficients (B,K,k), phase (B,K+1) at t = 0, L, .., T.
        The phase is matched in every run (a diagnostic); only --phase uses it for the target."""
        tau = np.zeros(B, dtype=np.int64); TAU = []
        env.reset()
        obs = set_init_qpos(env, [S[c]["ref"][0] for c in slots], nv)
        q = np.stack([S[c]["ref"][0] for c in slots]); v = np.zeros((B, nv))
        ex = np.stack([X(q[i], v[i]) for i in range(B)])
        z0 = torch.stack([S[c]["z0"] for c in slots])
        Q, OB, XS, CS = [], [], [], []
        z = z0.clone()
        for t in range(T):
            if t % L == 0:
                k = t // L
                with torch.no_grad():
                    bs = model.backward_map(torch.as_tensor(ex, dtype=torch.float32, device=dev))
                    if k:
                        tau = match_phase(bs, slots, tau, L)
                    TAU.append(tau)
                    src = tau if args.phase else np.full(B, t)
                    bg = torch.stack([S[c]["Bg"][min(int(src[i]) + L, len(S[c]["Bg"]) - 1)] for i, c in enumerate(slots)])
                    x = torch.cat([bs, bg, z0], dim=-1)
                    cf = adapter(x)
                    if eps is not None:
                        cf = cf + args.sigma * scale * eps[:, k]
                    z = z0 + cf @ Ut; z = 16 * z / z.norm(dim=-1, keepdim=True)
                XS.append(x); CS.append(cf)
            with torch.no_grad():
                a = model._actor(model._normalize(torch.as_tensor(ex, dtype=torch.float32, device=dev)), z,
                                 model.cfg.actor_std).mean.cpu().numpy()
            _, _, _, _, info = env.step(a)
            q, v = info["qpos"].copy(), info["qvel"].copy()
            ex = np.stack([X(q[i], v[i]) for i in range(B)])
            Q.append(q); OB.append(ex)
        Q = np.stack(Q, 1); OB = np.stack(OB, 1).astype(np.float32)
        with torch.no_grad():
            Bs = model.backward_map(torch.as_tensor(OB.reshape(-1, 358), device=dev)).reshape(B, T, -1)
            TAU.append(match_phase(Bs[:, -1], slots, tau, T - (K - 1) * L))
        return Q, Bs, torch.stack(XS, 1), torch.stack(CS, 1), np.stack(TAU, 1)

    def seg_cost(slots, Bs, TAU, span):
        """(B, K) phase-aligned bfm: the frames [kL, kL + span) against the reference from the phase matched at kL;
        NaN where the reference has ended. And (B, K) lag |phase at (k+1)L - clock|, in frames, the clock held at
        the reference's last frame."""
        c = torch.full((len(slots), K), float("nan"), device=dev); lag = np.zeros((len(slots), K))
        for i, cl in enumerate(slots):
            Bg = S[cl]["Bg"]; n = len(Bg)
            for k in range(K):
                s0, tk = k * L, int(TAU[i, k]); m = min(span, T - s0, n - tk)
                if m > 0:
                    c[i, k] = (1 - nn.functional.cosine_similarity(Bs[i, s0:s0 + m], Bg[tk:tk + m], dim=-1)).mean()
                lag[i, k] = abs(int(TAU[i, k + 1]) - min((k + 1) * L, n - 1))
        return c, lag

    def frame_cost(slots, Bs):
        """(B, T) 1 - cos(B(s_t), B(g_t)) with NaN past each reference's end."""
        c = torch.full((len(slots), T), float("nan"), device=dev)
        for i, cl in enumerate(slots):
            n = min(T, len(S[cl]["Bg"]))
            c[i, :n] = 1 - nn.functional.cosine_similarity(Bs[i, :n], S[cl]["Bg"][:n], dim=-1)
        return c

    history = []
    if args.wandb:
        import wandb
        from model.simple.train_es import eval_wandb_log
        wandb.init(project=cfg.wandb_project, name=f"child-stepwise-{args.tag}", config=vars(args))

    def evaluate(u):
        slots = test + ev_train; pad = slots + [slots[-1]] * (-len(slots) % B)
        res = {}
        for a0 in range(0, len(pad), B):
            part = pad[a0:a0 + B]
            Q, Bs, _, CS, TAU = rollout(part)
            fc = frame_cost(part, Bs); sc, lag = seg_cost(part, Bs, TAU, L)
            _, la, _ = compute_batch_cost(fk, cfg, Q, [S[c]["ref"] for c in part])
            for i, c in enumerate(part):
                res[f"{c[0]}_{c[1]}"] = dict(bfm=float(torch.nanmean(fc[i])), L_align=float(la[i]),
                                             dz=float((CS[i] @ Ut).norm(dim=-1).mean()),
                                             bfm_phase=float(torch.nanmean(sc[i])), lag=float(lag[i].mean()))
        e = {"update": u}
        for name, cl in (("test", test), ("train", ev_train)):
            e[name] = {f"{c[0]}_{c[1]}": res[f"{c[0]}_{c[1]}"] for c in cl}
        history.append(e); json.dump(history, open(out / "eval_history.json", "w"))
        if args.wandb:   # train_es's per-clip charts: bfm per clip / its own at update 0
            wh = [(h["update"], {**{sp: dict(cost=float(np.mean([v["bfm"] for v in h[sp].values()])),
                                              L_align=float(np.mean([v["L_align"] for v in h[sp].values()])),
                                              dist_mean=float(np.mean([v["dz"] for v in h[sp].values()])))
                                     for sp in ("test", "train")},
                                  **{f"{sp}_clips": {"per_clip": [dict(clip=k, cost=v["bfm"], dist=v["dz"]) for k, v in h[sp].items()]}
                                     for sp in ("test", "train")}}) for h in history]
            wandb.log({**eval_wandb_log(wh, {"train": "train_clips", "test": "test_clips"}), "update": u})
        base = history[0]
        for name in ("test", "train"):
            r_b = np.array([e[name][k]["bfm"] / base[name][k]["bfm"] for k in e[name]])
            r_a = np.array([e[name][k]["L_align"] / base[name][k]["L_align"] for k in e[name]])
            print(f"  [eval @{u:4d}] {name:5s}: bfm {np.median([v['bfm'] for v in e[name].values()]):.3f} (vs update 0 med {np.median(r_b):.2f}, "
                  f"down {int((r_b < 1).sum())}/{len(r_b)}) | L_align {np.median([v['L_align'] for v in e[name].values()]):.3f} "
                  f"(med {np.median(r_a):.2f}, down {int((r_a < 1).sum())}/{len(r_a)}) | |dz| {np.mean([v['dz'] for v in e[name].values()]):.2f}"
                  f" | phase-aligned bfm {np.median([v['bfm_phase'] for v in e[name].values()]):.3f}, lag {np.mean([v['lag'] for v in e[name].values()]):.1f} fr", flush=True)

    evaluate(0)
    P, C = args.pairs, args.clips_per_update
    for u in range(1, args.updates + 1):
        t0 = time.time()
        picks = random.sample(train, C)
        slots = [c for c in picks for _ in range(2 * P)]                          # row-major: clip, then 2P candidates
        e = rng.standard_normal((C, P, K, args.k)).astype(np.float32)
        eps = np.concatenate([e, -e], axis=1).reshape(B, K, args.k)               # [+p0..+pP-1, -p0..-pP-1] per clip
        eps_t = torch.as_tensor(eps, device=dev)
        Q, Bs, XS, CS, TAU = rollout(slots, eps_t)
        fc = frame_cost(slots, Bs)                                                # (B, T)
        dz = (CS @ Ut).norm(dim=-1)                                               # (B, K)
        if args.phase:   # this segment and the next, from where the body actually is in the reference + its lag
            sc, lag = seg_cost(slots, Bs, TAU, 2 * L)
            win = torch.nan_to_num(sc, nan=0.0) + args.lag_weight * torch.as_tensor(lag / L, dtype=torch.float32, device=dev)
        else:
            win = torch.stack([torch.nanmean(fc[:, k * L:(k + 2) * L], dim=1) for k in range(K)], 1)  # (B, K)
            win = torch.nan_to_num(win, nan=0.0)
        win = win + args.anchor * (dz / 16) ** 2
        win = win.view(C, 2 * P, K); d = win[:, :P] - win[:, P:]                  # (C, P, K) + minus -
        # rank-shape the pair differences per (clip, segment)
        r = d.argsort(dim=1).argsort(dim=1).float(); r = r / (P - 1) - 0.5
        g = (r.unsqueeze(-1) * eps_t.view(C, 2 * P, K, -1)[:, :P]).mean(1)       # (C, K, k): uphill
        x = XS.view(C, 2 * P, K, -1)
        cf = adapter(x)                                                           # (C, 2P, K, k) with grad
        loss = (g.unsqueeze(1).detach() * cf).sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); opt.step()
        cost = float(torch.nanmean(fc))
        print(f"[{u:4d}/{args.updates}] cost {cost:.4f} |dz| {dz.mean().item():.2f} |g| {g.norm(dim=-1).mean().item():.3f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if args.wandb:
            wandb.log({"train/bfm": cost, "train/dz": dz.mean().item(), "train/g": g.norm(dim=-1).mean().item(), "update": u})
        if u % args.eval_every == 0:
            evaluate(u); torch.save(adapter.state_dict(), out / f"adapter_{u:04d}.pt")
    env.close()


if __name__ == "__main__":
    main()
