"""Evolution strategies on z_beta -- no policy gradient, no value function, no
noise in action space at all.

    z_beta   = G_theta(beta, z0)                       # the only thing learned
    eps_k    ~ N(0, I_256)
    F+_k     = cost( rollout(project(z_beta + sigma*eps_k)) )
    F-_k     = cost( rollout(project(z_beta - sigma*eps_k)) )
    g_z      = sum_k shape(F+_k - F-_k) * eps_k        # antithetic estimate
    theta   <- theta - lr * (dz_beta/dtheta)^T g_z     # exact, by autograd

Why this fits the problem better than PPO
------------------------------------------
z_beta is computed ONCE per episode and does not depend on the state, so the
learnable decision is a single 256-dim vector, not a per-step control law.
model/simple/train.py explores that by injecting Gaussian noise into 69 action
dims x 300 steps and hoping the score function routes 20700 dimensions of noise
back to the 256 that matter. ES perturbs the 256 directly.

Three things fall out of that, and they are the point:

  * NO exploration noise in the rollout. Actions are the frozen actor's mean.
    config.py's own comment says the per-step noise "compounds over 300 MuJoCo
    steps and was swamping the L_align/L_phys learning signal" -- here it is gone,
    and the only thing that differs between two rollouts is z.
  * NO per-step credit machinery. window_rewards, GAE, ValueNet, the PPO ratio,
    the KL guard and the log-ratio clamp all exist to get a usable per-step
    advantage; ES needs one scalar per rollout. The fitness is therefore the
    whole-episode cost from the UNCHANGED losses.py, which is also the number
    model/simple/evaluate.py reports -- training and evaluation measure the same
    quantity again.
  * The "some clips are just harder" confound cancels EXACTLY rather than being
    modelled. F+ and F- are the same clip on the same body, so the difference
    is within-pair. That is what ValueNet and bilevel's PairAdvantageNormalizer
    are approximations of.

Common random numbers are what make it work
--------------------------------------------
model/bilevel/upper.py:106 measured that without CRN the ES signal "is buried
under rollout noise". Here CRN is exact and free: humenv's Default init resets
every env slot to a bit-identical state (verified: max spread 0.0 across slots),
and with deterministic actions nothing else is random. So a +eps and a -eps
rollout differ ONLY by the z perturbation.

Slot budget
-----------
One vectorized env has cfg.batch_size slots and one skeleton, so one update is
still one body. Those slots are split into 2 * es_pairs antithetic rollouts per
clip, which fixes how many clips an update can see:

    clips_per_update = batch_size // (2 * es_pairs)

At the defaults (16 slots, 4 pairs) that is 2 clips x 4 directions, in ONE
batched rollout -- the same simulator cost as a PPO update, not 8x it.

Usage (from project root):
    uv run model/simple/train_es.py
    uv run model/simple/train_es.py --updates 400 --run-name es-400-8bodies
    uv run model/simple/train_es.py --sigma 0.5 --pairs 8 --no-wandb

    # one P.fall group only -- see scripts/split_tasks_by_fall.py for why
    uv run model/simple/train_es.py --tasks upright --run-name es-upright
"""

import argparse
import dataclasses
import os
import random
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb

from metamotivo.fb_cpr.huggingface import FBcprModel

from model.dataset import CrossEmbodimentDataset, load_task_list
from model.networks import LatentAdapter
from model.simple.config import ESConfig
from model.simple.train import compute_batch_cost, make_body_ctx

REPO_ROOT = Path(__file__).resolve().parents[2]


def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Evenly spaced ranks in [-0.5, 0.5].

    Copied from model/bilevel/upper.py:129 with its reason: it makes the
    estimator invariant to the fitness's drifting scale. That matters more here
    than it does there -- L_align is heavy-tailed (measured p90 18.9 against a median
    of 6.6, max 136), so one catastrophic clip would otherwise set the step size
    for the whole update.
    """
    n = x.numel()
    if n < 2:
        return torch.zeros_like(x)
    order = x.argsort()
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(n, dtype=x.dtype, device=x.device)
    return ranks / (n - 1) - 0.5


@torch.no_grad()
def rollout_z(model, env, z_env, cfg, obs_mul=None):
    """Deterministic rollout, one z per env slot. Returns qpos (n_envs, T, nq).

    No sampling anywhere: this is what makes the antithetic difference a clean
    measurement of z rather than of the noise draw.
    """
    obs, _ = env.reset()
    qpos_hist = []
    for _ in range(cfg.steps_per_episode):
        proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.as_tensor(proprio, dtype=torch.float32, device=cfg.device)
        mu = model._actor(model._normalize(obs_t), z_env, model.cfg.actor_std).mean
        obs, _, _, _, info = env.step(mu.cpu().numpy())
        qpos_hist.append(info["qpos"].copy())
    return np.stack(qpos_hist, axis=1)


def project_z(z, z_dim, on):
    """The adapter's own projection, applied again after perturbing. The actor
    was trained with norm_z=True and never re-projects what it is given (see
    TrainConfig.adapter_project_z), so a perturbed z has to be put back on the
    sphere before it is shown -- otherwise ES would be measuring the effect of
    leaving the manifold as much as the effect of the direction."""
    return (z_dim ** 0.5) * F.normalize(z, dim=-1) if on else z


def tangent_eps(z_beta, K, sigma, project_on):
    """(R, K, D) perturbation directions, in the TANGENT space of the sphere.

    ES weights each direction by the eps it multiplied in, so the eps in the
    estimator has to be the perturbation that was actually evaluated. It is not,
    if you sample isotropically and then project: the radial component of eps is
    annihilated by project() but still appears in the sum, which biases g_z with
    a direction nothing was ever measured along.

    Projecting eps onto the tangent plane first removes that null direction
    exactly. What is left is a scalar shrink -- project(z + sigma*e_perp) has
    tangential part sigma*e_perp / sqrt(1 + (sigma|e_perp|/|z|)^2) -- plus a
    radial term that depends on |e_perp|^2 and is therefore IDENTICAL for +e and
    -e, so it is common-mode and cancels in the antithetic difference. The
    scalar shrink (measured 0.973 at sigma=0.25) is a uniform rescale that
    rank normalization discards anyway.

    Scale check worth doing whenever sigma changes: eps ~ N(0, I_256) has
    |eps| ~ sqrt(256) = 16, so |sigma*eps| = 16*sigma against |z| = 16 -- i.e.
    sigma IS the fractional perturbation of z, and 0.25 moves z by 25%, not by
    a fraction of a percent.
    """
    eps = torch.randn(z_beta.shape[0], K, z_beta.shape[-1], device=z_beta.device)
    if project_on:
        zhat = F.normalize(z_beta.detach(), dim=-1).unsqueeze(1)      # (R, 1, D)
        eps = eps - (eps * zhat).sum(-1, keepdim=True) * zhat
    return eps


def es_update(cfg, model, adapter, optimizer, ctx, samples, z_dim):
    """One antithetic ES step over `samples` clips of one body."""
    dev = cfg.device
    R, K = len(samples), cfg.es_pairs
    z0 = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=dev)
    beta = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=dev)
    qpos_refs = [s["qpos_ref"] for s in samples]

    z_beta = adapter(beta, z0)                                   # (R, 256), differentiable
    eps = tangent_eps(z_beta, K, cfg.es_sigma, cfg.adapter_project_z)

    with torch.no_grad():
        base = z_beta.unsqueeze(1)                               # (R, 1, 256)
        plus = project_z(base + cfg.es_sigma * eps, z_dim, cfg.adapter_project_z)
        minus = project_z(base - cfg.es_sigma * eps, z_dim, cfg.adapter_project_z)
        # Slot layout: [row0 +k0..+k3, row0 -k0..-k3, row1 +..., row1 -...]
        z_env = torch.cat([torch.cat([plus[r], minus[r]]) for r in range(R)])

    qpos = rollout_z(model, ctx["env"], z_env, cfg, ctx["obs_mul"])
    refs_env = [qpos_refs[r] for r in range(R) for _ in range(2 * K)]
    costs, align_totals, l_physes = compute_batch_cost(ctx["fk"], cfg, qpos, refs_env)

    f = torch.as_tensor(costs, dtype=torch.float32, device=dev).view(R, 2, K)
    delta = f[:, 0] - f[:, 1]                                    # (R, K) = F+ - F-
    if cfg.es_rank_normalize:
        # PER ROW. Ranking across rows would compare "which clip is easier"
        # instead of "which direction is better": a clip whose deltas are O(5)
        # would take every extreme rank and a clip whose deltas are O(0.1) would
        # be assigned arbitrary middle ranks. Only directions within one
        # (clip, body) are commensurable, because only they share a landscape.
        shaped = torch.stack([rank_normalize(delta[r]) for r in range(R)])
    else:
        shaped = delta / (2.0 * cfg.es_sigma)
    # g_z points UPHILL in cost; the surrogate below is minimized, so gradient
    # descent on theta walks z against it.
    g_z = (shaped.unsqueeze(-1) * eps).mean(dim=1)               # (R, 256)

    surrogate = (g_z.detach() * z_beta).sum(-1).mean()
    z_cos = (F.normalize(z_beta, dim=-1) * F.normalize(z0, dim=-1)).sum(-1).mean()
    anchor = cfg.lambda_z * (1.0 - z_cos)
    loss = surrogate + anchor

    # The two terms' gradients w.r.t. z_beta, so their relative size is visible
    # rather than assumed. Rank normalization strips g_z of the cost's units, so
    # lambda_z tuned against the PPO objective carries no meaning here -- if this
    # ratio is far from O(1) the anchor is either inert or in sole charge.
    g_es = torch.autograd.grad(surrogate, z_beta, retain_graph=True)[0]
    g_anchor = torch.autograd.grad(anchor, z_beta, retain_graph=True)[0]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    return {
        "cost": float(costs.mean()), "L_align": float(align_totals.mean()),
        "L_phys": float(l_physes.mean()),
        # |F+ - F-| against the spread of F itself: if the perturbation moves the
        # cost by much less than the clip-to-clip spread, sigma is too small to
        # measure anything and no amount of lr will fix it.
        "delta_abs": float(delta.abs().mean()), "delta_std": float(delta.std()),
        "cost_std": float(costs.std()),
        "grad_norm": float(grad), "z_cos": z_cos.item(), "surrogate": surrogate.item(),
        "z_norm": z_beta.detach().norm(dim=-1).mean().item(),
        "g_es_norm": g_es.norm(dim=-1).mean().item(),
        "g_anchor_norm": g_anchor.norm(dim=-1).mean().item(),
        "g_ratio": (g_anchor.norm(dim=-1).mean() / g_es.norm(dim=-1).mean().clamp(min=1e-12)).item(),
    }


@torch.no_grad()
def run_eval(cfg, model, adapter, ctxs, dataset, eval_idx, splits, z_dim):
    """Unperturbed, deterministic rollout of the fixed eval clips."""
    out = {}
    for b, idxs in eval_idx.items():
        samples = [dataset[i] for i in idxs]
        z0 = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=cfg.device)
        beta = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=cfg.device)
        z_env = adapter(beta, z0)
        qpos = rollout_z(model, ctxs[b]["env"], z_env, cfg, ctxs[b]["obs_mul"])
        costs, la, lp = compute_batch_cost(ctxs[b]["fk"], cfg, qpos, [s["qpos_ref"] for s in samples])
        out[b] = {"cost": float(costs.mean()), "L_align": float(la.mean()), "L_phys": float(lp.mean())}
    for sp in ("train", "test"):
        mem = [b for b in out if splits.get(b) == sp]
        if mem:
            out[sp] = {k: float(np.mean([out[b][k] for b in mem]))
                       for k in ("cost", "L_align", "L_phys")}
    return out


def train(cfg: ESConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.batch_size % (2 * cfg.es_pairs):
        raise SystemExit(
            f"batch_size {cfg.batch_size} must be a multiple of 2*es_pairs "
            f"{2 * cfg.es_pairs}: every env slot carries one antithetic rollout.")
    clips_per_update = cfg.batch_size // (2 * cfg.es_pairs)

    dataset_dir = REPO_ROOT / cfg.dataset_dir
    task_filter = None
    if cfg.task_group != "all":
        path = dataset_dir / "splits" / f"{cfg.task_group}_tasks.txt"
        if not path.exists():
            raise SystemExit(f"{path} not found -- run scripts/split_tasks_by_fall.py")
        task_filter = load_task_list(path)
        print(f"task group '{cfg.task_group}': {len(task_filter)} tasks")
    dataset = CrossEmbodimentDataset(dataset_dir, task_filter=task_filter)
    by_body = dataset.indices_by_body()
    bodies_path = dataset_dir / "splits" / "train_bodies.txt"
    bodies = ([b for b in load_task_list(bodies_path) if b in by_body]
              if bodies_path.exists() else list(by_body))
    held = sorted(set(by_body) - set(bodies))
    if held:
        print(f"held out of training ({len(held)}): {' '.join(held)}")

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()
    z_dim = model.cfg.archi.z_dim

    ctxs = {b: make_body_ctx(cfg, dataset_dir, b, dataset[by_body[b][0]]["target_xml"])
            for b in by_body}
    splits = {b: ("train" if b in bodies else "test") for b in by_body}
    beta_dim = len(dataset[by_body[bodies[0]][0]]["beta"])
    print(f"{len(bodies)} training bodies; {clips_per_update} clips x {cfg.es_pairs} "
          f"antithetic pairs = {cfg.batch_size} slots per update, sigma={cfg.es_sigma}")

    adapter = LatentAdapter(
        beta_dim=beta_dim, z_dim=z_dim, hidden_dims=cfg.adapter_hidden_dims,
        alpha=cfg.adapter_alpha, alpha_learnable=cfg.adapter_alpha_learnable,
        project=cfg.adapter_project_z,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)

    erng = random.Random(cfg.eval_seed)
    eval_idx = {b: [erng.choice(by_body[b]) for _ in range(cfg.batch_size)] for b in by_body}

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config={**dataclasses.asdict(cfg), "train_bodies": bodies,
                           "held_out_bodies": held, "clips_per_update": clips_per_update})

    history = []

    def do_eval(update):
        ev = run_eval(cfg, model, adapter, ctxs, dataset, eval_idx, splits, z_dim)
        history.append((update, ev))
        tr, te = ev.get("train"), ev.get("test")
        tqdm.write(f"  [eval @ {update:04d}] "
                   + (f"train cost={tr['cost']:.4f} (L_align={tr['L_align']:.4f} Lp={tr['L_phys']:.4f})  " if tr else "")
                   + (f"test cost={te['cost']:.4f} (L_align={te['L_align']:.4f} Lp={te['L_phys']:.4f})  " if te else "")
                   + (f"gap={te['cost'] - tr['cost']:+.4f}" if tr and te else ""))
        if cfg.use_wandb:
            log = {}
            for b, v in ev.items():
                pre = f"eval/{b}" if b not in ("train", "test") else f"eval/{b}_bodies"
                log.update({f"{pre}/{k}": val for k, val in v.items()})
            if tr and te:
                log["eval/gap_cost"] = te["cost"] - tr["cost"]
            wandb.log(log, step=update)

    if cfg.eval_every and cfg.eval_at_start:
        do_eval(0)

    pbar = tqdm(range(cfg.num_updates), desc="es", disable=not cfg.progress)
    for update in pbar:
        label = (bodies[update % len(bodies)] if cfg.body_order == "cycle"
                 else random.choice(bodies))
        idx = [random.choice(by_body[label]) for _ in range(clips_per_update)]
        st = es_update(cfg, model, adapter, optimizer, ctxs[label],
                       [dataset[i] for i in idx], z_dim)

        pbar.set_postfix(body=label, L_align=f"{st['L_align']:.4f}", d=f"{st['delta_abs']:.3f}")
        if cfg.use_wandb:
            wandb.log({**{k: v for k, v in st.items()},
                       "body_idx": bodies.index(label),
                       f"by_body/{label}/cost": st["cost"],
                       f"by_body/{label}/L_align": st["L_align"],
                       f"by_body/{label}/L_phys": st["L_phys"]}, step=update)
        if update % cfg.log_every == 0:
            tqdm.write(f"[{update:04d}/{cfg.num_updates}] {label:13s} "
                       f"cost={st['cost']:.4f} |dF|={st['delta_abs']:.4f} "
                       f"sd(dF)={st['delta_std']:.4f} |g_es|={st['g_es_norm']:.4f} "
                       f"|g_anc|={st['g_anchor_norm']:.2e} r={st['g_ratio']:.2e} "
                       f"grad={st['grad_norm']:.3f} "
                       f"1-zcos={1 - st['z_cos']:.2e} L_align={st['L_align']:.4f} Lp={st['L_phys']:.4f}")

        if cfg.eval_every and (update + 1) % cfg.eval_every == 0:
            do_eval(update + 1)
        if (update + 1) % cfg.ckpt_every == 0:
            path = ckpt_dir / f"update_{update + 1:05d}.pt"
            torch.save({"adapter": adapter.state_dict(), "update": update + 1,
                        "cfg": cfg, "bodies": bodies}, path)
            tqdm.write(f"saved checkpoint -> {path}")

    for c in ctxs.values():
        c["env"].close()
    if cfg.use_wandb:
        wandb.finish()

    if history:
        print("\n=== held-out evaluation ===")
        print(f"{'update':>7s} {'train cost':>11s} {'test cost':>10s} {'gap':>9s} "
              f"{'train L_align':>13s} {'test L_align':>13s}")
        for u, ev in history:
            tr, te = ev.get("train"), ev.get("test")
            if tr and te:
                print(f"{u:7d} {tr['cost']:11.4f} {te['cost']:10.4f} "
                      f"{te['cost'] - tr['cost']:+9.4f} {tr['L_align']:13.4f} {te['L_align']:13.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None, help="ES perturbation scale on z")
    p.add_argument("--pairs", type=int, default=None, help="antithetic pairs per clip")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tasks", default=None, choices=["all", "ground", "upright"],
                   help="restrict to a P.fall group (scripts/split_tasks_by_fall.py). "
                        "'upright' is where fall is ~0 on a good rollout, so any fall "
                        "the policy incurs is real signal -- train there first")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--no-rank", action="store_true",
                   help="use the raw (F+ - F-)/(2 sigma) estimate instead of ranks")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--project", default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args()

    cfg = ESConfig(device=a.device or f"cuda:{a.gpu}")
    for attr, val in (("num_updates", a.updates), ("es_sigma", a.sigma),
                      ("es_pairs", a.pairs), ("lr", a.lr), ("ckpt_dir", a.ckpt_dir),
                      ("task_group", a.tasks),
                      ("wandb_project", a.project), ("wandb_run_name", a.run_name)):
        if val is not None:
            setattr(cfg, attr, val)
    if a.no_rank:
        cfg.es_rank_normalize = False
    if a.no_progress:
        cfg.progress = False
    if a.no_wandb:
        cfg.use_wandb = False
    train(cfg)


if __name__ == "__main__":
    main()
