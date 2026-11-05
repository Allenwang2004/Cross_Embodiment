#!/usr/bin/env python3
"""dist_adapter.py -- learn a CLOUD of good latents per body for one clip, from simulation cost alone.

One clip (default move-ego-0-2-raisearms-h-h_6), many bodies: the 21 multibody bodies whose arm / leg length ratio is
within 1.15 (the 10/13 crowd-video selection), 7 of them held out (analysis E's test bodies that survive the cut).
A small network maps a body's shape beta (8 numbers, assets/robots/<body>/parameter.json) to a cloud of latents
around the clip's z0:

    bell   one Gaussian: centre project(z0 + d(beta)), one width per coordinate s(beta)
    mix    --components Gaussians, each with its own centre, widths and weight

Training. Every update, each training body draws --samples latents from its cloud (antithetic pairs, a pair shares
its component), each is rolled out on that body and scored with the multibody search's exact cost
(scripts/multibody_corr.py: bfm + 1.0 heading + 0.1 pos + 0.3 (|z - z0| / 16)^2, exact obs, reference init; the
same number as outputs/multibody_corr/{search,transfer}), divided by the body's z0 cost (ratio r, clipped at
--clip-ratio). The network follows the score-function gradient of

    mean over bodies of   E_z[r]  -  lambda * H(cloud)

with the per-body batch mean as baseline. H is the cloud's entropy (for mix: the weighted component entropies plus
the entropy of the weights). --spread sets lambda in plain units: the average ratio worth paying to double the
cloud's width in every coordinate, lambda = spread / (256 ln 2). spread 0 lets the cloud shrink onto one point;
--fixed-sigma freezes the widths at --sigma0 (the point model: only the centre is learned, as in train_es.py).

Rollouts run in one worker process per body (its own Motivo copy and async env), 16 slots per batch as in every
multibody search, all on the one visible GPU.
Writes <out>/{config.json, log.csv, centres.csv, model.pt, model_u<update>.pt}.
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 uv run python scripts/dist_adapter.py --shape bell --spread 0.02 \
           --out outputs/dist_adapter/bell_s0.02
       ... --verify child      (score the first 16 transfer latents of one body and compare with transfer/eval.csv)
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")

CLIP = "move-ego-0-2-raisearms-h-h/move-ego-0-2-raisearms-h-h_6"
TEST = ["child", "giant", "short_stocky", "gl0500a0550", "gl1500a1400", "ax2m_t1000", "ax4m_t1000"]
SLOTS = 16      # rollouts per batch, as in every multibody search and transfer eval: the actor's batch size changes its
                # float32 output in the last bits, and a rollout that falls is chaotic enough to show it
DIM, R = 256, 16.0
LO, HI = math.log(0.005), math.log(0.5)     # range of each coordinate's width


def kept_bodies():
    from multibody_corr import BODIES, beta
    return [b for b in BODIES if 1 / 1.15 <= beta(b)[1] / beta(b)[0] <= 1.15]


class Scorer:
    """One body's rollouts and cost, exactly as single_z_search.py scores them under the multibody recipe
    (--init reference --steps <clip length> --obs-scale exact --objective bfm --heading-weight 1.0 --pos-weight 0.1
    --anchor z0 --anchor-weight 0.3). The joint-space terms that recipe only logs are skipped."""

    def __init__(self, body, clip, device):
        import mujoco
        import humenv.utils as hu
        from humenv import make_humenv
        from metamotivo.fb_cpr.huggingface import FBcprModel
        from model import bfm_align
        from model.exact_obs import ExactObs
        from single_z_search import project_z, rollout
        self.rollout, self.bfm_align, self.device = rollout, bfm_align, device
        task, stem = clip.split("/")
        self.z0 = project_z(np.load(REPO / f"data/origin_z/{task}/{stem}.npy").reshape(-1).astype(np.float64))
        ref = np.load(REPO / f"data/{body}/retargeting_motion/{task}/{stem}.npz")["qpos"]
        xml = REPO / f"assets/robots_torque/{body}/robot_torque_full.xml"
        self.model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(device)
        self.model.eval()
        self.env, _ = make_humenv(num_envs=SLOTS, vectorization_mode="async", task=None, xml=str(xml),
                                  state_init="Default")
        self.fk = fk = mujoco.MjModel.from_xml_path(str(xml))
        kw = {}
        if body != "child":
            a = np.load(REPO / f"data/origin_motion/{task}/{stem}.npz")["qpos"][:, :2]
            n = min(len(a), len(ref)); m = np.abs(a[:n]) > 1e-3
            kw["scale"] = float(np.median(ref[:n, :2][m] / a[:n][m]))
        self.exact = ExactObs(xml, REPO / "assets/robots/adult/robot.xml", **kw)
        vr, ro = np.zeros(fk.nv), []
        for t in range(len(ref)):
            if t:
                mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, ref[t - 1], ref[t])
            ro.append(self.exact(ref[t], vr))
        self.Bg = bfm_align.embed(self.model, np.stack(ro), device)

        def heading_xy(qs):
            flat = qs.reshape(-1, qs.shape[-1]); dk = mujoco.MjData(fk); h = np.empty(len(flat))
            for i, qq in enumerate(flat):
                dk.qpos[:] = qq; mujoco.mj_kinematics(fk, dk)
                h[i] = hu.calc_heading(hu.remove_base_rot(dk.xquat[1][None].copy(), "smpl"))[0]
            return h.reshape(qs.shape[:-1]), qs[..., :2]

        self.heading_xy, self.ref = heading_xy, ref
        self.ref_h, self.ref_xy = heading_xy(ref)
        self.base = float(self.score(self.z0[None])[0])

    def _batch(self, zs):
        zt = torch.as_tensor(zs, dtype=torch.float32, device=self.device)
        q, o = self.rollout(self.model, self.env, zt, len(self.ref), self.device, None, init_qpos=self.ref[0],
                            nv=self.fk.nv, return_obs=True, exact=self.exact, act_out=[])
        cost = self.bfm_align.batch_bfm_align(self.model, o, self.Bg, self.device)
        T = min(q.shape[1], len(self.ref))
        h, xy = self.heading_xy(q[:, :T])
        head = ((1 - np.cos(h - self.ref_h[None, :T])) / 2).mean(1)
        pos = np.linalg.norm(xy - self.ref_xy[None, :T], axis=-1).mean(1)
        cost = cost + 1.0 * head + 0.1 * pos
        return cost + 0.3 * (np.linalg.norm(np.asarray(zs) - self.z0, axis=1) / 16.0) ** 2

    def score(self, zs):
        """(n, 256) latents on the sphere -> cost (n,), in batches of SLOTS padded with z0."""
        out = []
        for i in range(0, len(zs), SLOTS):
            chunk = zs[i:i + SLOTS]
            pad = np.concatenate([chunk, np.repeat(self.z0[None], SLOTS - len(chunk), 0)])
            out.append(self._batch(pad)[:len(chunk)])
        return np.concatenate(out)


def worker(body, clip, conn):
    torch.set_num_threads(2)
    sc = Scorer(body, clip, "cuda:0")
    conn.send(sc.base)
    while True:
        zs = conn.recv()
        if zs is None:
            break
        conn.send(sc.score(zs))
    sc.env.close()


def proj(x):
    return x * (R / x.norm(dim=-1, keepdim=True).clamp_min(1e-12))


class Cloud(torch.nn.Module):
    """beta (B, 8) -> centres (B, M, 256) on the sphere, log widths (B, M, 256), log weights (B, M)."""

    def __init__(self, z0, mu, sd, m, sigma0, fixed_sigma, hidden=256, seed=0):
        super().__init__()
        nn = torch.nn
        self.register_buffer("z0", torch.as_tensor(z0, dtype=torch.float32))
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.m, self.fixed, self.ls0 = m, fixed_sigma, math.log(sigma0)
        self.trunk = nn.Sequential(nn.Linear(8, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.d, self.s, self.w = nn.Linear(hidden, m * DIM), nn.Linear(hidden, m * DIM), nn.Linear(hidden, m)
        for l in (self.d, self.s, self.w):          # every body starts at z0 with width sigma0 and equal weights
            nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        u0 = (self.ls0 - LO) / (HI - LO)
        nn.init.constant_(self.s.bias, math.log(u0 / (1 - u0)))
        if m > 1:       # components that start together get the same gradient forever: offsets of length ~1
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                self.d.bias.copy_(torch.randn(m * DIM, generator=g) / R)

    def forward(self, beta):
        h = self.trunk((beta - self.mu) / self.sd)
        c = proj(self.z0 + self.d(h).view(-1, self.m, DIM))
        ls = (torch.full_like(c, self.ls0) if self.fixed
              else LO + (HI - LO) * torch.sigmoid(self.s(h).view(-1, self.m, DIM)))
        return c, ls, torch.log_softmax(self.w(h), -1)


def log_q(z, c, ls, lw):
    """log density of unprojected samples z (n, 256) under one body's mixture c, ls (M, 256), lw (M,)."""
    e = (z[:, None] - c[None]) / ls.exp()[None]
    lg = -0.5 * (e ** 2).sum(-1) - ls.sum(-1)[None] - 0.5 * DIM * math.log(2 * math.pi)
    return torch.logsumexp(lw[None] + lg, -1)


def entropy(ls, lw):
    """weighted component entropies + entropy of the weights (up to a constant)."""
    p = lw.exp()
    return (p * ls.sum(-1)).sum() - (p * lw).sum()


def sample(c, ls, lw, n, rng):
    """n unprojected samples (antithetic pairs, a pair shares its component) and their component indices."""
    k = rng.choice(len(lw), size=n // 2, p=lw.exp().numpy().astype(np.float64) / lw.exp().sum().item())
    eps = rng.standard_normal((n // 2, DIM))
    s = ls.exp().numpy()[k] * eps
    cc = c.numpy()[k]
    return np.concatenate([cc + s, cc - s]), np.concatenate([k, k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--shape", choices=["bell", "mix"], default="bell")
    ap.add_argument("--components", type=int, default=4, help="Gaussians of --shape mix")
    ap.add_argument("--spread", type=float, default=0.0,
                    help="average ratio worth paying to double the cloud's width in every coordinate")
    ap.add_argument("--fixed-sigma", action="store_true", help="freeze the widths at --sigma0 (point model)")
    ap.add_argument("--sigma0", type=float, default=0.05, help="starting width per coordinate (0.05 = ES's 0.8 radius)")
    ap.add_argument("--samples", type=int, default=16, help="latents per body per update (antithetic pairs)")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam, trunk and centres")
    ap.add_argument("--lr-head", type=float, default=1e-2, help="Adam, widths and weights")
    ap.add_argument("--clip-ratio", type=float, default=3.0)
    ap.add_argument("--centre-every", type=int, default=25, help="also roll out every body's centres this often")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--bodies", default=None, help="comma list of training bodies (default: the kept 21 minus TEST)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", default=None, help="body: compare this script's cost with transfer/eval.csv and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from multibody_corr import beta

    if a.verify:
        stem = a.clip.split("/")[1]
        d = REPO / f"outputs/multibody_corr/transfer/{a.verify}/{stem}"
        sc = Scorer(a.verify, a.clip, "cuda:0")
        Z = np.load(d / "z.npy")[:SLOTS].astype(np.float64)
        mine = sc.score(Z)
        ref = [float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))][:SLOTS]
        base = json.load(open(d / "eval_base.json"))["cost"]
        print(f"z0 cost: here {sc.base:.6g}, transfer {base:.6g}")
        print("max |cost here - transfer eval.csv| over 16 latents:", float(np.max(np.abs(mine - np.array(ref)))))
        sc.env.close()
        return

    train = a.bodies.split(",") if a.bodies else [b for b in kept_bodies() if b not in TEST]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lam = a.spread / (DIM * math.log(2))
    B = np.array([beta(b) for b in train], dtype=np.float64)
    mu, sd = B.mean(0), B.std(0); sd[sd < 1e-6] = 1.0
    m = a.components if a.shape == "mix" else 1
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    from single_z_search import project_z
    task, stem = a.clip.split("/")
    z0 = project_z(np.load(REPO / f"data/origin_z/{task}/{stem}.npy").reshape(-1).astype(np.float64))
    net = Cloud(z0, mu, sd, m, a.sigma0, a.fixed_sigma, seed=a.seed)
    head = [p for n, p in net.named_parameters() if n.startswith(("s.", "w."))]
    rest = [p for n, p in net.named_parameters() if not n.startswith(("s.", "w."))]
    opt = torch.optim.Adam([dict(params=rest, lr=a.lr), dict(params=head, lr=a.lr_head)])
    cfg = dict(vars(a), train=train, test=TEST, lam=lam, components=m, beta_mu=mu.tolist(), beta_sd=sd.tolist())
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"{a.shape} x{m}, spread {a.spread} (lambda {lam:.3g}){', widths fixed' if a.fixed_sigma else ''}; "
          f"{len(train)} bodies x {a.samples} rollouts x {a.updates} updates -> {out}", flush=True)

    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for b in train:
        p_main, p_w = ctx.Pipe()
        pr = ctx.Process(target=worker, args=(b, a.clip, p_w)); pr.start()      # not daemonic: it has env children
        pipes.append(p_main); procs.append(pr)
    try:
        train_loop(a, net, opt, pipes, train, B, z0, m, lam, rng, out, cfg)
    finally:
        for p, pr in zip(pipes, procs):
            try:
                p.send(None)
            except (BrokenPipeError, OSError):
                pass
        for pr in procs:
            pr.join(timeout=60)
            if pr.is_alive():
                pr.terminate()


def train_loop(a, net, opt, pipes, train, B, z0, m, lam, rng, out, cfg):
    from single_z_search import project_z
    base = np.array([p.recv() for p in pipes])
    print("z0 cost per body: " + ", ".join(f"{b} {c:.3f}" for b, c in zip(train, base)), flush=True)
    Bt = torch.as_tensor(B, dtype=torch.float32)

    def centres_row(u):
        with torch.no_grad():
            c, _, lw = net(Bt)
        for p, cb in zip(pipes, c.numpy().astype(np.float64)):
            p.send(project_z(cb))
        costs = np.array([p.recv() for p in pipes]) / base[:, None]           # (bodies, M)
        with open(out / "centres.csv", "a") as fh:
            if u == 0:
                fh.write("update,body,component,weight,ratio,dist_z0\n")
            for i, b in enumerate(train):
                for k in range(m):
                    fh.write(f"{u},{b},{k},{lw[i, k].exp().item():.4f},{costs[i, k]:.5f},"
                             f"{np.linalg.norm(c[i, k].numpy() - z0):.4f}\n")
        return costs

    log = open(out / "log.csv", "w", newline=""); w = None
    t0 = time.time()
    for u in range(a.updates + 1):
        if u % a.centre_every == 0 or u == a.updates:
            cc = centres_row(u)
            print(f"  u{u:4d} centres: median best-component ratio {np.median(cc.min(1)):.3f}", flush=True)
        if u == a.updates:
            break
        with torch.no_grad():
            c, ls, lw = net(Bt)
        Z, K = zip(*[sample(c[i], ls[i], lw[i], a.samples, rng) for i in range(len(train))])
        for p, z in zip(pipes, Z):
            p.send(project_z(z))
        r = np.minimum(np.array([p.recv() for p in pipes]) / base[:, None], a.clip_ratio)    # (bodies, samples)
        adv = torch.as_tensor(r - r.mean(1, keepdims=True), dtype=torch.float32)
        c, ls, lw = net(Bt)
        loss, H = 0.0, []
        for i in range(len(train)):
            lq = log_q(torch.as_tensor(Z[i], dtype=torch.float32), c[i], ls[i], lw[i])
            Hi = entropy(ls[i], lw[i]); H.append(Hi.item())
            loss = loss + (adv[i] * lq).mean() - lam * Hi
        opt.zero_grad(); (loss / len(train)).backward(); opt.step()
        with torch.no_grad():
            wd = (lw.exp() * ls.exp().mean(-1)).sum(-1)                    # weight-averaged mean width per body
            dist = (lw.exp() * (c - net.z0).norm(dim=-1)).sum(-1)
        row = dict(update=u, minutes=round((time.time() - t0) / 60, 2), mean_ratio=float(r.mean()),
                   median_body_mean=float(np.median(r.mean(1))), median_body_min=float(np.median(r.min(1))),
                   frac_below_0_8=float((r < 0.8).mean()), width=float(wd.mean()), radius=float(wd.mean() * R),
                   dist_z0=float(dist.mean()), entropy=float(np.mean(H)),
                   max_weight=float(lw.exp().max(-1).values.mean()))
        row.update({f"r_{b}": float(r[i].mean()) for i, b in enumerate(train)})
        if w is None:
            w = csv.DictWriter(log, fieldnames=list(row)); w.writeheader()
        w.writerow(row); log.flush()
        if u % 10 == 0:
            print(f"  u{u:4d} [{row['minutes']:.1f} min] mean ratio {row['mean_ratio']:.3f}  body-min median "
                  f"{row['median_body_min']:.3f}  <0.8 {row['frac_below_0_8']:.2f}  width {row['width']:.4f} "
                  f"(radius {row['radius']:.2f})  |c - z0| {row['dist_z0']:.2f}", flush=True)
        if (u + 1) % a.save_every == 0:
            torch.save(dict(state=net.state_dict(), cfg=cfg), out / f"model_u{u + 1}.pt")
    torch.save(dict(state=net.state_dict(), cfg=cfg), out / "model.pt")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
