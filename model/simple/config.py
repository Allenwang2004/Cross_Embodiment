import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class TrainConfig:
    metamotivo_repo: str = "facebook/metamotivo-M-1"
    # 540 clips x 10 bodies, built by scripts/build_dataset.py. The bodies come
    # from the manifest, one per row, and splits/train_bodies.txt says which 8
    # of them are trained on -- train.py never reads target_xml any more.
    dataset_dir: str = "datasets/crossenbodiment-10bodies"
    # One update = one body (see train.py "One body per update"). "cycle" walks
    # the 8 training bodies in order so a run of N updates gives each of them
    # exactly N/8; "random" draws i.i.d., which lets short runs be unbalanced.
    body_order: str = "cycle"
    # Used by model/simple/evaluate.py and by the legacy single-body dataset
    # only. train.py takes the body from each row.
    target_morphology_json: str = "assets/robots/child/parameter.json"
    target_xml: str = "assets/robots/child/robot.xml"
    device: str = "cuda:0"

    # G_theta: z_beta = z0 + alpha * MLP([beta, z0]) -- the ONLY thing trained.
    # The beta-conditioned ActionHead that used to sit on the frozen actor's
    # output is gone; see train.py's "z_beta is the only thing learned".
    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 512, 512, 256])
    adapter_alpha: float = 0.1
    adapter_alpha_learnable: bool = False
    # Re-project z_beta onto the sphere of radius sqrt(z_dim) FB's latents live
    # on. metamotivo applies project_z when it PRODUCES a z (sample_z,
    # reward/goal/tracking_inference) but never inside actor()/_actor(), and
    # Actor.forward just concatenates [obs, z] -- so an off-sphere z is fed
    # straight through. Was False while the ActionHead could absorb the
    # mismatch; now that z_beta is the only channel it is on. Pairs with the
    # COSINE lambda_z below -- do not turn one on without the other.
    adapter_project_z: bool = True

    # What the frozen actor is SHOWN (model/obs_scale.py). "auto" = per-body
    # length ratios measured from the two rest poses, so the target body's
    # metre-carrying obs features read as the adult's -- the actor's obs
    # normalizer holds adult-scale BatchNorm statistics and z_beta should not
    # have to spend itself on a unit conversion. "none" is the ablation (the
    # raw obs this file used before); an explicit float forces one uniform
    # ratio. Affects the actor's input only -- never the physics or D/L_phys.
    obs_scale: str = "auto"
    obs_scale_parts: str = "length"  # "length" also rescales local_body_vel; "pose" does not
    obs_scale_ref_xml: str = "assets/robots/adult/robot.xml"  # the body the actor was trained on

    # loss weights, L = lambda_rtg * D + lambda_z * (1 - cos(z_beta, z0)) + lambda_phys * L_phys
    # (no R_task term -- see train.py module docstring for why)
    lambda_rtg: float = 1.0
    lambda_z: float = 0.1  # COSINE anchor, not Euclidean: with adapter_project_z the radius is
                           # fixed by construction, so ||z_beta - z0||^2 would spend part of
                           # itself penalizing a distance that cannot change. Same form and same
                           # value as model/bilevel/config.py:252. The magnitude is comparable to
                           # the old Euclidean term (both are O(||delta||^2 / z_dim) for a delta
                           # near-orthogonal to z0, which in 256 dims is the typical case), so
                           # 0.1 carries over without a retune.
    lambda_phys: float = 1.0
    d_root_weight: float = 1.0
    d_ee_weight: float = 1.0
    d_contact_weight: float = 1.0
    d_pose_weight: float = 1.0
    d_velocity_weight: float = 1.0

    # PPO, ported from model/bilevel/ppo.py -- see train.py's "Per-step credit,
    # not one scalar per episode". The rollout is still not differentiable; what
    # changed is that the score-function estimator now gets one advantage per
    # WINDOW instead of one per episode.
    window_steps: int = 30      # 1.0 s @ 30 Hz. D/L_phys are computed on each window
                                # separately using the UNCHANGED model/losses.py, so an
                                # episode of 300 steps yields 10 rewards, not 1. Shorter
                                # than this and d_root's heading/curvature terms have too
                                # few frames to mean anything; bilevel's own window is 60.
    gamma: float = 0.97         # model/bilevel/config.py:203 -- effective horizon 33 steps
    gae_lambda: float = 0.95    # model/bilevel/config.py:206
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatches: int = 4
    # Early-stop the epoch loop once the policy has moved this far from the one
    # that collected the data. Not decoration: exploration_std is 0.05, so
    # d logp/d mu = (a - mu)/sigma^2 amplifies by 400 per action dim summed over
    # 69 of them -- model/bilevel/ppo.py flags the same effect at sigma=0.2.
    # Measured on the first smoke run, epoch 4 was running at clip_frac 0.46 and
    # approx_kl 0.19, i.e. almost off-policy. bilevel can afford not to have this
    # because its BC term keeps the policy near a known-good action; this path
    # has no such tether, and one bad update destroys the frozen prior.
    ppo_target_kl: float = 0.02
    value_clip: float = 0.2
    value_coef: float = 0.5
    value_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [512, 512])
    lr: float = 3e-4
    exploration_std: float = 0.05  # PPO's policy sigma: actions are sampled from
                                    # Normal(frozen actor's mean, this), and the ratio is
                                    # computed against the same distribution. Decoupled from
                                    # the frozen model's own actor_std (0.2) because this
                                    # noise compounds over 300 MuJoCo steps and was swamping
                                    # the D/L_phys signal. It being this small is also why
                                    # ppo_target_kl and grad_clip_norm=1.0 are load-bearing:
                                    # d logp/d mu scales as 1/sigma^2.
    batch_size: int = 16  # episodes per update, run as one vectorized HumEnv (see train.py)
    vectorization_mode: str = "sync"  # gymnasium VectorEnv mode for the batched rollout
    num_updates: int = 50
    steps_per_episode: int = 300
    # The scalar EMA baseline is GONE, replaced by ValueNet(obs, z_beta, beta,
    # phase) -- that network's docstring says it exists precisely because
    # task-composition heterogeneity swamped this EMA. Kept only so an old
    # pickled cfg still unpickles; nothing reads it.
    baseline_momentum: float = 0.95
    # 1.0, not the 5.0 this file used under REINFORCE. model/bilevel/config.py:201
    # made the same reduction for the same reason -- with a residual policy on a
    # frozen prior the clip, not lr, is what sets the step, and 5.0 let the first
    # PPO smoke run diverge to NaN inside one update.
    grad_clip_norm: float = 1.0
    seed: int = 0

    # W&B. One update is one body, so the pooled D/L_phys series alternates
    # between bodies with different cost scales -- read `by_body/<label>/*`
    # for a trend within a body and treat the pooled ones as a sanity check.
    use_wandb: bool = True
    wandb_project: str = "crossenbodiment-simple"
    wandb_run_name: Optional[str] = None

    # Held-out evaluation during training. The body axis is the only split, so
    # "test" means the bodies in splits/test_bodies.txt. The TRAIN bodies are
    # scored the same way in the same pass, because a held-out number on its own
    # says nothing -- the gap between the two is the thing worth watching.
    eval_every: int = 50        # updates; 0 disables
    eval_at_start: bool = True  # one pass before update 0, so there is a t=0 reference
    # The eval clips are drawn ONCE with this seed and then reused at every
    # eval. A fresh draw each time would make the curve a resampling of the
    # heavy-tailed cost rather than a measurement of the policy.
    eval_seed: int = 12345

    # tqdm writes every refresh with a bare \r, so a redirected run puts the
    # whole bar on ONE unreadable line in the log. Off for background runs.
    progress: bool = True

    log_every: int = 1
    # Under outputs/, which .gitignore covers -- model/checkpoints/ does not,
    # and a run writing 20 x 2.6 MB of .pt into a tracked directory shows up as
    # untracked noise in every git status afterwards.
    ckpt_dir: str = "outputs/simple/checkpoints"
    ckpt_every: int = 20
    loss_curve_path: str = "outputs/simple/loss_curve.png"
