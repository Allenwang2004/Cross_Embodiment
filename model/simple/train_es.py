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

One motion, every body
-----------------------
An update holds the CLIP fixed and varies the BODY: it draws clips_per_update
clips and evaluates each of them on all 8 training bodies. The manifest stores
one origin_z per (task, trial) and every body's row for that clip points at the
same file, so the rows of one batch share z0 EXACTLY and differ only in beta and
in the per-body retargeted reference.

That is the contrast the adapter exists to explain, and it was absent before.
With one body per update beta is a CONSTANT inside the batch, so no single
update's gradient can distinguish G_theta(beta, z0) from a beta-blind
G_theta(z0) -- the beta-dependence had to be assembled ACROSS updates, through
the optimizer state, from batches that each also differed in which clips they
drew. Here it is inside one gradient: same z0, eight betas, eight references.

Slot budget
-----------
One vectorized env has cfg.batch_size slots and one skeleton, so a body's rows
have to be rolled out in that body's OWN env -- one batched rollout per body per
update. Within a body the slots are split into 2 * es_pairs antithetic rollouts
per clip, which fixes how many clips an update can see:

    clips_per_update = batch_size // (2 * es_pairs)

At the defaults (16 slots, 4 pairs, 8 bodies) that is 2 clips x 4 directions x
2 signs = 16 slots per body, 8 batched rollouts, 128 episodes per update -- 8x
the simulator cost of the one-body update this replaces. --bodies N buys that
back at the cost of the within-update beta contrast: --bodies 1 recovers the old
loop exactly, and es_pairs=8 (--pairs 8) makes clips_per_update 1, i.e. the
literal "one motion, eight bodies" batch.

Usage (from project root):
    uv run model/simple/train_es.py
    uv run model/simple/train_es.py --updates 400 --run-name es-400-8bodies
    uv run model/simple/train_es.py --sigma 0.5 --pairs 8 --no-wandb

    # one clip x 8 bodies per update (pairs 8 => clips_per_update 1), or fewer
    # bodies per update when the 8x rollout cost is the binding constraint.
    uv run model/simple/train_es.py --pairs 8 --run-name es-1clip-8bodies
    uv run model/simple/train_es.py --bodies 4 --run-name es-4bodies

    # one task group only -- see scripts/spilt_tasks.py. `upright` is the
    # 30 low-fall tasks; `move` is its 17-task locomotion SUBSET, so the two
    # overlap and `move` is the narrower experiment, not a disjoint one.
    uv run model/simple/train_es.py --category upright --run-name es-upright
    uv run model/simple/train_es.py --category move --run-name es-move

    # which of the two terms the fitness IS. The other one is still measured and
    # logged either way -- it just does not enter the cost -- so running all
    # three is how you find out whether they pull in the same direction:
    for L in both L_align L_phys; do
      uv run model/simple/train_es.py --loss $L --category upright \
          --ckpt-dir outputs/simple_es/$L --run-name es-upright-$L
    done
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

# --loss -> (lambda_align, lambda_phys), fed straight into ESConfig so the cost
# compute_batch_cost already returns IS the objective and this file never gets a
# second definition of it. Same table as scripts/single_z_search.py:OBJECTIVES,
# so the ceiling that script measures and what training minimises are the same
# three quantities.
#
# Only "both" is sensitive to the two terms' relative scale. The single-term
# settings are not: es_rank_normalize replaces the fitness with its rank inside
# each row, and any monotone rescaling of one loss leaves the ranks unchanged.
LOSS_LAMBDAS = {
    "both": (1.0, 1.0),
    "L_align": (1.0, 0.0),
    "L_phys": (0.0, 1.0),
}


def clip_index(dataset):
    """{(reward_name, trial): {body label: row index}} -- a CLIP's column of the
    manifest's (clip x body) cross product.

    CrossEmbodimentDataset.indices_by_body() is the transpose of this, and is
    what a one-body-per-update loop needs. This is what "same motion, different
    bodies" needs. The two rows of one column share origin_z (hence z0) and
    differ in `morphology` (beta) and `retargeted_motion` (qpos_ref).
    """
    out = {}
    for i, r in enumerate(dataset.rows):
        key = (r["reward_name"], r["trial"])
        out.setdefault(key, {})[r.get("morphology_label", "child")] = i
    return out


def select_bodies(bodies, n, order, update):
    """Which bodies one update's batch covers. n >= len(bodies) is all of them,
    which is this file's point and its default; smaller n is the wall-clock
    escape hatch, and body_order then says whether the subset walks the list
    ("cycle", so N updates still give every body N*n/len(bodies) of them) or is
    drawn i.i.d. ("random")."""
    if n >= len(bodies):
        return list(bodies)
    if order == "cycle":
        start = (update * n) % len(bodies)
        return [bodies[(start + j) % len(bodies)] for j in range(n)]
    return random.sample(bodies, n)


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


def update_best_buffer(buf, keys, z_row, costs):
    """{(task, trial, body): (z, cost)} -- the best CANDIDATE ever evaluated for
    each cell, kept across updates.

    Comparing a cost measured 300 updates ago against one measured now is only
    legitimate because the rollout is deterministic and the reset is
    bit-identical (see this module's CRN section): the same z on the same cell
    scores the same number forever, so the running minimum is a real minimum and
    not a record of which update got a lucky draw.

    Note what is NOT in here: z_beta itself. Training rolls out the perturbed
    candidates, never the iterate, so the buffer holds points the simulator
    actually scored -- which is the property that makes it a supervised target
    rather than a second guess.
    """
    hit = 0
    for r, k in enumerate(keys):
        j = int(costs[r].argmin())
        if k not in buf or costs[r, j] < buf[k][1]:
            buf[k] = (z_row[r, j].detach().cpu().numpy().copy(), float(costs[r, j]))
            hit += 1
    return hit


def es_update(cfg, model, adapter, optimizer, ctxs, rows, z_dim, best_buf=None):
    """One antithetic ES step over `rows`, each a (body label, sample) pair.

    Every clip appears once per body, so a row is a (clip, body) cell and the
    batch is a rectangle of them. eps is drawn PER ROW rather than shared by the
    bodies of one clip: the perturbation lives in the tangent plane of that
    row's own z_beta, and betas differ, so a shared direction would not be
    tangent for more than one of them.

    best_buf, when cfg.lambda_bc > 0, is the running per-cell best candidate
    (update_best_buffer). It adds a term that pulls z_beta toward a point the
    simulator MEASURED to be better, with an exact gradient, instead of toward
    whatever the rank-weighted zeroth-order estimate points at. Passing None, or
    leaving lambda_bc at 0, leaves every number in this function bit-identical to
    what it was before the buffer existed -- no RNG is drawn and no branch is
    taken.
    """
    dev = cfg.device
    R, K = len(rows), cfg.es_pairs
    labels = [lab for lab, _ in rows]
    samples = [s for _, s in rows]
    z0 = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=dev)
    beta = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=dev)

    z_beta = adapter(beta, z0)                                   # (R, 256), differentiable
    eps = tangent_eps(z_beta, K, cfg.es_sigma, cfg.adapter_project_z)

    with torch.no_grad():
        base = z_beta.unsqueeze(1)                               # (R, 1, 256)
        plus = project_z(base + cfg.es_sigma * eps, z_dim, cfg.adapter_project_z)
        minus = project_z(base - cfg.es_sigma * eps, z_dim, cfg.adapter_project_z)
        # Per-row slot layout: [+k0..+k3, -k0..-k3]
        z_row = torch.cat([plus, minus], dim=1)                  # (R, 2K, 256)

    # One batched rollout per body: an env carries one skeleton, and scoring a
    # row against another body's forward kinematics would silently measure the
    # adapter on a body it was not asked about (model/dataset.py says the same).
    groups = {}
    for r, lab in enumerate(labels):
        groups.setdefault(lab, []).append(r)
    costs = np.empty((R, 2 * K), dtype=np.float32)
    align_totals = np.empty((R, 2 * K), dtype=np.float32)
    l_physes = np.empty((R, 2 * K), dtype=np.float32)
    for lab, rs in groups.items():
        n_slots = len(rs) * 2 * K
        if n_slots != cfg.batch_size:
            raise SystemExit(f"body {lab} has {len(rs)} rows x {2 * K} = {n_slots} "
                             f"rollouts but its env has {cfg.batch_size} slots")
        z_env = torch.cat([z_row[r] for r in rs])                # (batch_size, 256)
        qpos = rollout_z(model, ctxs[lab]["env"], z_env, cfg, ctxs[lab]["obs_mul"])
        refs_env = [samples[r]["qpos_ref"] for r in rs for _ in range(2 * K)]
        c, a, p = compute_batch_cost(ctxs[lab]["fk"], cfg, qpos, refs_env)
        costs[rs] = c.reshape(len(rs), 2 * K)
        align_totals[rs] = a.reshape(len(rs), 2 * K)
        l_physes[rs] = p.reshape(len(rs), 2 * K)

    f = torch.as_tensor(costs, dtype=torch.float32, device=dev).view(R, 2, K)
    delta = f[:, 0] - f[:, 1]                                    # (R, K) = F+ - F-
    if cfg.es_rank_normalize:
        # PER ROW. Ranking across rows would compare "which clip is easier"
        # instead of "which direction is better": a clip whose deltas are O(5)
        # would take every extreme rank and a clip whose deltas are O(0.1) would
        # be assigned arbitrary middle ranks. Only directions within one
        # (clip, body) are commensurable, because only they share a landscape.
        # Unchanged by the body-major batch, and load-bearing for it: cost scale
        # is a body property (L_phys on `giant` and on `petite` are not the same
        # number for the same quality of rollout), so ranking across the rows of
        # one clip would rank the BODIES.
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

    bc = None
    bc_stats = {}
    if cfg.lambda_bc > 0 and best_buf is not None:
        keys = [(s["reward_name"], s["trial"], lab) for lab, s in rows]
        # Updated BEFORE the loss, so this update's own candidates are eligible:
        # the term is then "move toward the best thing measured so far,
        # including a moment ago", which is the greedy step the ES gradient is a
        # noisy approximation of.
        n_new = update_best_buffer(best_buf, keys, z_row, costs)
        tgt = torch.tensor(np.stack([best_buf[k][0] for k in keys]),
                           dtype=torch.float32, device=dev)
        # Cosine, not Euclidean, for the same reason as the anchor: with
        # adapter_project_z the radius is fixed, so a squared distance would
        # spend part of itself on a gap that cannot close.
        bc_cos = F.cosine_similarity(z_beta, tgt, dim=-1)
        bc = cfg.lambda_bc * (1.0 - bc_cos).mean()
        loss = loss + bc
        bc_stats = {
            "bc_deg": float(torch.rad2deg(torch.arccos(bc_cos.detach().clamp(-1, 1))).mean()),
            "bc_new": n_new / max(len(keys), 1),      # share of cells improved this update
            "buf_size": len(best_buf),
            "buf_cost": float(np.mean([best_buf[k][1] for k in keys])),
        }

    # The two terms' gradients w.r.t. z_beta, so their relative size is visible
    # rather than assumed. Rank normalization strips g_z of the cost's units, so
    # lambda_z tuned against the PPO objective carries no meaning here -- if this
    # ratio is far from O(1) the anchor is either inert or in sole charge.
    g_es = torch.autograd.grad(surrogate, z_beta, retain_graph=True)[0]
    g_anchor = torch.autograd.grad(anchor, z_beta, retain_graph=True)[0]
    if bc is not None:
        # Same diagnostic as the anchor's, and needed for the same reason: this
        # term's gradient is EXACT while the ES term's is a K-dimensional probe
        # of a 256-dimensional landscape, so a bc_ratio well under 1 can still
        # dominate once the noise averages out over updates.
        g_bc = torch.autograd.grad(bc, z_beta, retain_graph=True)[0]
        bc_stats["g_bc_norm"] = g_bc.norm(dim=-1).mean().item()
        bc_stats["bc_ratio"] = (g_bc.norm(dim=-1).mean()
                                / g_es.norm(dim=-1).mean().clamp(min=1e-12)).item()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    per_body = {lab: {"cost": float(costs[rs].mean()),
                      "L_align": float(align_totals[rs].mean()),
                      "L_phys": float(l_physes[rs].mean())}
                for lab, rs in groups.items()}

    return {
        "by_body": per_body,
        "cost": float(costs.mean()), "L_align": float(align_totals.mean()),
        "L_phys": float(l_physes.mean()),
        # Spread of the pooled cost that is pure body, at a fixed set of clips --
        # only meaningful because every body in this batch saw the same clips.
        "body_cost_spread": float(np.std([v["cost"] for v in per_body.values()])),
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
        **bc_stats,
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
            raise SystemExit(f"{path} not found -- run scripts/spilt_tasks.py")
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

    # A clip is (reward_name, trial); its column of the manifest is one row per
    # body. Training may only draw a clip that every SELECTED body has (a batch
    # is a full rectangle -- see es_update's slot check); eval may only draw one
    # that every body has, train and held-out alike, since it scores all of them.
    clips = clip_index(dataset)
    train_clips = sorted(k for k, m in clips.items() if all(b in m for b in bodies))
    eval_clips_pool = sorted(k for k, m in clips.items() if len(m) == len(by_body))
    if not train_clips or not eval_clips_pool:
        raise SystemExit("no clip is present for every body -- the manifest is not a "
                         "full (clip x body) cross product; rebuild it with "
                         "scripts/build_dataset.py --force")
    if len(train_clips) < len(clips):
        print(f"{len(clips) - len(train_clips)} of {len(clips)} clips dropped: "
              f"not present for every training body")

    ctxs = {b: make_body_ctx(cfg, dataset_dir, b, dataset[by_body[b][0]]["target_xml"])
            for b in by_body}
    splits = {b: ("train" if b in bodies else "test") for b in by_body}
    beta_dim = len(dataset[by_body[bodies[0]][0]]["beta"])
    n_bodies = min(cfg.es_bodies_per_update or len(bodies), len(bodies))
    print(f"{len(bodies)} training bodies, {len(train_clips)} clips; per update "
          f"{n_bodies} bodies x {clips_per_update} clips x {cfg.es_pairs} antithetic "
          f"pairs = {n_bodies} rollouts of {cfg.batch_size} slots "
          f"({n_bodies * cfg.batch_size} episodes), sigma={cfg.es_sigma}")
    print(f"objective: cost = {cfg.lambda_align} * L_align + {cfg.lambda_phys} * L_phys "
          f"(the zero-weighted term is still computed and logged)")

    adapter = LatentAdapter(
        beta_dim=beta_dim, z_dim=z_dim, hidden_dims=cfg.adapter_hidden_dims,
        alpha=cfg.adapter_alpha, alpha_learnable=cfg.adapter_alpha_learnable,
        project=cfg.adapter_project_z, residual=cfg.adapter_residual,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)

    # Where the run STARTS, measured rather than assumed. residual=True puts
    # z_beta on top of z0 (cos ~ 1); residual=False with projection on starts it
    # at a uniformly random point of the sphere (cos ~ 0), which is a different
    # experiment and needs to be visible in the log rather than inferred from
    # the flags.
    with torch.no_grad():
        _r = [dataset[by_body[b][0]] for b in bodies]
        _z0 = torch.tensor(np.stack([x["z0"] for x in _r]), dtype=torch.float32, device=cfg.device)
        _be = torch.tensor(np.stack([x["beta"] for x in _r]), dtype=torch.float32, device=cfg.device)
        _c = F.cosine_similarity(adapter(_be, _z0), _z0, dim=-1).mean()
        _a = torch.rad2deg(torch.arccos(_c.clamp(-1, 1)))
    print(f"adapter: " + (f"z0 + {cfg.adapter_alpha:g} * MLP([beta, z0])" if cfg.adapter_residual
                          else "MLP([beta, z0])  (NO z0 residual; alpha unused)")
          + f", project={cfg.adapter_project_z}; "
            f"at init cos(z_beta, z0) = {_c:.4f} ({_a:.1f} deg from z0)")

    erng = random.Random(cfg.eval_seed)
    # The SAME clips for every body. eval's headline number is the train/test
    # BODY gap, and drawing each body's clips independently put the clip
    # difficulty spread -- which is heavy-tailed, L_align p90 18.9 against a
    # median of 6.6 -- straight into it, so one body's unlucky draw could move
    # the gap by more than the adapter does. Shared clips make it a body effect.
    eval_clips = [erng.choice(eval_clips_pool) for _ in range(cfg.batch_size)]
    eval_idx = {b: [clips[c][b] for c in eval_clips] for b in by_body}

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config={**dataclasses.asdict(cfg), "train_bodies": bodies,
                           "held_out_bodies": held, "clips_per_update": clips_per_update,
                           "bodies_per_update": n_bodies})

    history = []
    # None unless the feature is on, so the default path allocates nothing and
    # the checkpoints it writes keep their old shape.
    best_buf = {} if cfg.lambda_bc > 0 else None
    if best_buf is not None:
        print(f"best-point buffer ON: lambda_bc={cfg.lambda_bc} "
              f"(cosine pull toward the best candidate measured per (clip, body); "
              f"watch bc_ratio and bc_deg)")

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
        sel = select_bodies(bodies, n_bodies, cfg.body_order, update)
        picks = [random.choice(train_clips) for _ in range(clips_per_update)]
        # Body-major so es_update's per-body group is contiguous; the (clip,
        # body) rectangle it forms is what makes beta the only thing varying.
        rows = [(b, dataset[clips[c][b]]) for b in sel for c in picks]
        st = es_update(cfg, model, adapter, optimizer, ctxs, rows, z_dim, best_buf)

        # cost, not L_align: under --loss L_phys the L_align column is a
        # bystander and watching it would say nothing about whether ES is working.
        clip_tag = picks[0][0] if clips_per_update == 1 else f"{len(picks)} clips"
        pbar.set_postfix(clip=clip_tag, cost=f"{st['cost']:.4f}", d=f"{st['delta_abs']:.3f}")
        if cfg.use_wandb:
            log = {k: v for k, v in st.items() if k != "by_body"}
            for lab, v in st["by_body"].items():
                log.update({f"by_body/{lab}/{k}": val for k, val in v.items()})
            wandb.log(log, step=update)
        if update % cfg.log_every == 0:
            tqdm.write(f"[{update:04d}/{cfg.num_updates}] {clip_tag:24s} "
                       f"cost={st['cost']:.4f} sd_body={st['body_cost_spread']:.4f} "
                       f"|dF|={st['delta_abs']:.4f} "
                       f"sd(dF)={st['delta_std']:.4f} |g_es|={st['g_es_norm']:.4f} "
                       f"|g_anc|={st['g_anchor_norm']:.2e} r={st['g_ratio']:.2e} "
                       f"grad={st['grad_norm']:.3f} "
                       f"1-zcos={1 - st['z_cos']:.2e} L_align={st['L_align']:.4f} Lp={st['L_phys']:.4f}"
                       + (f" | bc {st['bc_deg']:.1f}deg r={st['bc_ratio']:.2e} "
                          f"new={st['bc_new']:.0%} buf={st['buf_size']}"
                          if "bc_deg" in st else ""))

        if cfg.eval_every and (update + 1) % cfg.eval_every == 0:
            do_eval(update + 1)
        if (update + 1) % cfg.ckpt_every == 0:
            path = ckpt_dir / f"update_{update + 1:05d}.pt"
            blob = {"adapter": adapter.state_dict(), "update": update + 1,
                    "cfg": cfg, "bodies": bodies}
            if best_buf:
                # The buffer IS a labelled (clip, body) -> z dataset, collected
                # for free by training. Saved so a run can be resumed without
                # losing it and so it can be exported, but only when the feature
                # is on -- ~1 MB that a default run has no reason to carry.
                blob["best_buf"] = {"|".join(map(str, k)): (v[0], v[1])
                                    for k, v in best_buf.items()}
            torch.save(blob, path)
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
    p.add_argument("--loss", default="both", choices=["both", "L_align", "L_phys"])
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None, help="ES perturbation scale on z")
    p.add_argument("--pairs", type=int, default=None, help="antithetic pairs per clip")
    p.add_argument("--bodies", type=int, default=None,
                   help="bodies per update (default: all training bodies). One "
                        "batched rollout each, so this multiplies the simulator "
                        "cost of an update -- lower it only if wall clock matters "
                        "more than varying beta inside one gradient")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="residual scale in z_beta = z0 + alpha * MLP. Sets how far "
                        "a given MLP output moves z: tan(angle) = alpha*|MLP|/16, "
                        "so 0.1 needs |MLP| ~ 92 for 30 deg. Ignored under "
                        "--no-residual")
    p.add_argument("--lambda-bc", type=float, default=None,
                   help="weight of the pull toward the best candidate ever "
                        "measured for each (clip, body). 0 (default) is OFF and "
                        "bit-identical to the loop without it. This gradient is "
                        "EXACT while the ES one is estimated, so it can take over "
                        "at a bc_ratio well under 1 -- try 0.1..1.0 and read the "
                        "logged bc_ratio / bc_deg before going higher")
    p.add_argument("--lambda-z", type=float, default=None,
                   help="weight of the (1 - cos(z_beta, z0)) anchor. Inert at the "
                        "default while z sits on z0 (measured g_ratio ~5e-4), but "
                        "under --no-residual z starts ~87 deg away, 1-cos is O(1), "
                        "and 10.0 puts the anchor at ~11%% of the ES gradient "
                        "PULLING BACK TO z0 -- i.e. fighting the ablation. Turn it "
                        "down (0 disables) when running --no-residual")
    p.add_argument("--no-residual", action="store_true",
                   help="z_beta = MLP([beta, z0]) with no z0 skip and no alpha. "
                        "Removes the prior that the answer is near z0 -- and with "
                        "projection on, starts z_beta at a RANDOM point of the "
                        "sphere instead of on z0, so the frozen actor begins from "
                        "a latent it has no reason to like. Check the printed "
                        "'at init cos(z_beta, z0)' before reading the curve")
    p.add_argument("--category", default=None,
                   choices=["all", "ground", "upright", "move"],
                   help="restrict training to one task list written by "
                        "scripts/spilt_tasks.py. 'ground' (24) and "
                        "'upright' (30) are the P.fall partition -- on ground, "
                        "fall is a large per-task constant the policy cannot "
                        "remove, so pooling the two makes most of the objective "
                        "an offset. 'move' (17) is a SUBSET of upright, not a "
                        "fourth disjoint group: the upright tasks whose "
                        "reference root actually travels, where beta acts "
                        "through stride rather than through reach")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--no-rank", action="store_true",
                   help="use the raw (F+ - F-)/(2 sigma) estimate instead of ranks")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--project", default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args()

    cfg = ESConfig(device=a.device or f"cuda:{a.gpu}")
    cfg.lambda_align, cfg.lambda_phys = LOSS_LAMBDAS[a.loss]
    # --category names a TASK group, never a per-clip property:
    # spilt_tasks.py averages P.fall (and root displacement) over a
    # task's clips before cutting, so the dataset carries no per-clip label and
    # "move clips" can only mean "clips of a move task".
    for attr, val in (("num_updates", a.updates), ("es_sigma", a.sigma),
                      ("es_pairs", a.pairs), ("es_bodies_per_update", a.bodies),
                      ("lr", a.lr), ("adapter_alpha", a.alpha),
                      ("lambda_z", a.lambda_z), ("lambda_bc", a.lambda_bc),
                      ("ckpt_dir", a.ckpt_dir),
                      ("task_group", a.category),
                      ("wandb_project", a.project), ("wandb_run_name", a.run_name)):
        if val is not None:
            setattr(cfg, attr, val)
    if a.no_residual:
        cfg.adapter_residual = False
    if a.no_rank:
        cfg.es_rank_normalize = False
    if a.no_progress:
        cfg.progress = False
    if a.no_wandb:
        cfg.use_wandb = False
    train(cfg)


if __name__ == "__main__":
    main()
