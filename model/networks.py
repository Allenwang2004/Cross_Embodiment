import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, output_dim)]
    return nn.Sequential(*layers)


class LatentAdapter(nn.Module):
    """G_theta: z_beta = z0 + alpha * MLP_theta([beta, z0])   (residual=True)
              z_beta = MLP_theta([beta, z0])                  (residual=False)

    Bottleneck MLP over [beta, z0] (default hidden dims 256->512->512->256).

    `residual` (default True) is the model.md spec: the output is a same-size
    delta added to z0 and scaled by alpha, so z_beta starts ON z0 and the map
    only has to learn the correction the body change calls for. That skip is a
    prior, and a strong one -- z0 is already the latent that produces the right
    motion on the SOURCE body.

    residual=False drops it and lets the MLP name z_beta outright. Worth having
    because scripts/single_z_search.py measured the best z for a single clip at
    79..96 degrees from z0, and with alpha=0.1 reaching even 30 degrees needs an
    MLP output of norm ~92 (per-coordinate ~5.8) -- the residual form makes far
    targets expensive to express, and this is the ablation that says whether
    that is what is holding the map back.

    What it costs, and it is not small: with project=True, F.normalize turns ANY
    output direction into a full-radius point, so at initialization z_beta is a
    uniformly random point on the sphere rather than z0. The frozen actor is
    then being steered by a latent it has no reason to like, and the ES
    estimator -- a K-dimensional random-subspace probe of a 256-dimensional
    non-differentiable landscape -- has to climb out of that with no gradient to
    guide it. train_es.py prints cos(z_beta, z0) at init so which regime a run
    started in is on the record. model/simple/train_zmap.py is the setting where
    dropping the prior is safe: its target is labelled, so it does not need one.

    `project`: re-project the result onto the sphere of radius sqrt(z_dim),
    which is where FB's latents actually live -- every z in data/z/ has norm
    exactly 16.0 = sqrt(256), and metamotivo's FBModel.project_z enforces it
    (metamotivo/fb/model.py:126) because the model was trained with norm_z=True.
    Without this the frozen actor is fed an off-manifold z it has never seen.
    Defaults to False, which is what model/simple/train.py has always run with;
    the bilevel path (model/bilevel/config.py: project_z) turns it on. Worth
    revisiting for the simple path now that z_beta is its only control channel.
    """

    def __init__(self, beta_dim, z_dim, hidden_dims=(256, 512, 512, 256),
                 alpha=0.1, alpha_learnable=False, project=False, residual=True):
        super().__init__()
        self.z_dim = z_dim
        self.project = project
        self.residual = residual
        self.mlp = _mlp(beta_dim + z_dim, list(hidden_dims), z_dim)
        # Built even when residual=False, where nothing reads it: alpha is a
        # state_dict entry, and dropping it would make the two modes'
        # checkpoints structurally incompatible for no gain. Which mode a
        # checkpoint was trained under travels in its pickled cfg, not in its
        # tensor shapes -- so a loader MUST pass `residual` through from there,
        # exactly as it already has to for `project`.
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, beta: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        # beta: (B, beta_dim), z0: (B, z_dim)
        out = self.mlp(torch.cat([beta, z0], dim=-1))
        z = z0 + self.alpha * out if self.residual else out
        if self.project:
            z = (self.z_dim ** 0.5) * F.normalize(z, dim=-1)
        return z


class ActionHead(nn.Module):
    """Residual correction on top of the frozen actor's raw action mean,
    conditioned on beta -- accounts for the target body's different
    actuator/limb response even though the action space size is unchanged
    (robot_<label>.xml keeps the same actuator names/gear ratios).

    Used by model/bilevel/policy.py only. model/simple/train.py dropped it: it
    steers the frozen actor through z_beta alone."""

    def __init__(self, action_dim, beta_dim, hidden_dims=(128, 128)):
        super().__init__()
        self.mlp = _mlp(action_dim + beta_dim, list(hidden_dims), action_dim)

    def forward(self, raw_action: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(torch.cat([raw_action, beta], dim=-1))
        return raw_action + delta


class RootWrenchHead(nn.Module):
    """Extra 30 Hz actuator: a 6-DoF wrench applied to the root body.

    The humanoid's root is a free joint and therefore unactuated (nu=69 vs
    nv=75), so nothing in the action space can stop it falling directly. This
    head emits a force+torque written into data.xfrc_applied[Pelvis] each
    control step, which gives the policy a way to stay up while it is still
    learning to track. It is a TRAINING CRUTCH, not part of the deliverable:
    its magnitude is annealed to zero (BilevelConfig.wrench_scale) and it is
    the most heavily penalized term in the regularization reward (e_ext weight
    8.0), because a helping hand is the cheapest possible way to satisfy every
    other reward term.

    Output is in the ROOT'S OWN FRAME; the caller rotates it to world with the
    current root quaternion. That makes the head rotation-equivariant, which is
    much easier to learn than a world-frame wrench.
    """

    def __init__(self, root_feat_dim, beta_dim, action_dim, hidden_dims=(128, 128)):
        super().__init__()
        self.mlp = _mlp(root_feat_dim + beta_dim + action_dim, list(hidden_dims), 6)
        # Start at zero wrench: the policy should have to learn to reach for the
        # crutch rather than beginning life leaning on a random one.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, root_feats: torch.Tensor, beta: torch.Tensor,
                raw_action: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([root_feats, beta, raw_action], dim=-1))


class ValueNet(nn.Module):
    """V(obs, z_beta, beta, phase) for the bilevel PPO lower level.

    Conditioning on (z_beta, beta) lets one network represent the per-(clip,
    body) baseline directly -- model/simple/diagnose_single_task.py exists because
    task-composition heterogeneity swamped the old scalar EMA baseline.

    `phase` = (t/H, (H-t)/H) is NOT optional. With a 24-step window the value
    function has to know the episode is about to be truncated, or the bootstrap
    at t=H is systematically mis-scaled.
    """

    def __init__(self, obs_dim, z_dim, beta_dim, hidden_dims=(512, 512)):
        super().__init__()
        self.mlp = _mlp(obs_dim + z_dim + beta_dim + 2, list(hidden_dims), 1)

    def forward(self, obs, z_beta, beta, phase):
        return self.mlp(torch.cat([obs, z_beta, beta, phase], dim=-1)).squeeze(-1)


class ActionResidual(nn.Module):
    """Residual correction on top of the frozen actor's raw action mean, with
    NO beta conditioning -- for the single-body, no-adapter, kinematics-only
    exploration experiment (model/simple/train_explore.py) where z is fed to the
    frozen actor unmodified and there is no morphology descriptor to condition
    on."""

    def __init__(self, action_dim, hidden_dims=(128, 128)):
        super().__init__()
        self.mlp = _mlp(action_dim, list(hidden_dims), action_dim)

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        return raw_action + self.mlp(raw_action)
