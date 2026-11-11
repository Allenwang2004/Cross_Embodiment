"""policy.py -- the frozen Meta Motivo policy: MotivoPolicy (OOP refactor step 5, docs/refactor/PLAN.md).

One object holds the FBcprModel, its device and its mode, and performs the two calls the rollouts and costs make:

    policy = MotivoPolicy.load(paths.MOTIVO_CKPT, device="cuda")      # FBcprModel.from_pretrained(ckpt).to(device); .eval()
    z = policy.latents(Z)                 # float64 numpy (n, 256) / (n, T, 256) -> float32 tensor on the device
    z = policy.latents(Z, scale=16.0)     # the latent-sequence searches: 16 * Z (float64) first, then float32
    u = policy.act(obs, z)                # the actor's mean action, (n, n_act) float32 numpy
    B = policy.embed(obs)                 # the backward map B(obs), (T, 256) float32 numpy (bfm_align.embed)

`act` is EXACTLY the call every rollout loop made (single_z_search.rollout, lso_search / robot_z_search / robot_lso_search's
loops, child_reward_eval.rollout, rank_initial_cost.rollout_multi, test_exact_obs):
    model._actor(model._normalize(torch.as_tensor(obs, dtype=torch.float32, device=device)), z, model.cfg.actor_std).mean
then .cpu().numpy(). The copies differed only in what does not change a bit: torch.no_grad() (decorator, or none with .detach()
in test_exact_obs: the model has requires_grad False, autograd records nothing either way) and calling .cpu().numpy() once or
twice (single_z_search with act_out: two equal copies; here one array serves both). The z tensor is passed through unchanged
(a (n, T, 256) tensor's slice z[:, t] as the sequence loops took it; the actor concatenates it with the observation).
NOT covered: crossenbodiment/robot_rollout.act (the G1 line's single-observation call computed in a batch of 16 identical rows;
its batch size changes the GPU's last bits), which stays as it is.

`eval_mode`: single_z_search, lso_search, robot_lso_search, rank_initial_cost, rollout_z_trace and test_exact_obs call
model.eval(); robot_z_search and child_reward_eval do not. FBModel.__init__ already puts the model in eval mode
(train(False), requires_grad_(False)), so this is a no-op either way; it is kept as each script had it.
"""
from __future__ import annotations

import torch

from . import paths


class MotivoPolicy:
    """the frozen FBcprModel on one device (`model`, `device` stay public: the cost code reads them)."""

    def __init__(self, model, device):
        self.model, self.device = model, device

    @classmethod
    def load(cls, ckpt=paths.MOTIVO_CKPT, device="cuda", eval_mode=True):
        """FBcprModel.from_pretrained(ckpt).to(device), then .eval() when eval_mode (module docstring)."""
        from metamotivo.fb_cpr.huggingface import FBcprModel
        model = FBcprModel.from_pretrained(ckpt).to(device)
        if eval_mode:
            model.eval()
        return cls(model, device)

    def latents(self, Z, scale=None):
        """the latent tensor the actor gets: torch.as_tensor(Z, dtype=float32, device) -- or of scale * Z, the product taken in
        Z's own dtype first (lso_search / robot_lso_search: torch.as_tensor(16.0 * Z, ...)). A tensor is returned unchanged
        (the callers of single_z_search.rollout pass one)."""
        if isinstance(Z, torch.Tensor):
            return Z
        return torch.as_tensor(Z if scale is None else scale * Z, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs, z):
        """the actor's mean action for a batch: obs (n, 358) numpy (float32 or float64), z (n, 256) float32 tensor on the device
        -> (n, n_act) float32 numpy"""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.model._actor(self.model._normalize(o), z, self.model.cfg.actor_std).mean.cpu().numpy()

    def embed(self, obs):
        """B(obs) for (T, 358) observations (already mapped): model/bfm_align.embed, the reference embeddings' call"""
        from model import bfm_align
        return bfm_align.embed(self.model, obs, self.device)

    def backward_map(self, O):
        """B of a float32 tensor already on the device, kept there (lso_search's batched embedding of the rollout)"""
        return self.model.backward_map(O)
