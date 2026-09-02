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
    # ratio. Affects the actor's input only -- never the physics or L_align/L_phys.
    obs_scale: str = "auto"
    obs_scale_parts: str = "length"  # "length" also rescales local_body_vel; "pose" does not
    obs_scale_ref_xml: str = "assets/robots/adult/robot.xml"  # the body the actor was trained on

    # loss weights, L = lambda_align * L_align + lambda_z * (1 - cos(z_beta, z0)) + lambda_phys * L_phys
    # (no R_task term -- see train.py module docstring for why)
    lambda_align: float = 1.0
    lambda_z: float = 0.1  # COSINE anchor, not Euclidean: with adapter_project_z the radius is
                           # fixed by construction, so ||z_beta - z0||^2 would spend part of
                           # itself penalizing a distance that cannot change. Same form and same
                           # value as model/bilevel/config.py:252. The magnitude is comparable to
                           # the old Euclidean term (both are O(||delta||^2 / z_dim) for a delta
                           # near-orthogonal to z0, which in 256 dims is the typical case), so
                           # 0.1 carries over without a retune.
    lambda_phys: float = 1.0
    # L_phys's fall term: reference the pelvis height to the retargeted clip's
    # own per-frame height instead of the rollout's first frame (see
    # losses._fall_penalty). Only has an effect where a qpos_ref exists -- rows
    # without one fall back to the old behaviour on their own.
    #
    # OFF. It was written to stop the term calling a legitimate crawl/headstand
    # a fall, and it does -- but on the upright tasks that training actually
    # keeps it removes most of what fall was penalising (a rollout that stays
    # near the reference's height scores ~0 whether or not it is about to go
    # over), so the term stops separating a rollout from the reference at all.
    # The ground tasks it fixes are dropped from training anyway. Keep it off
    # unless the ground split comes back, and then fix the tilt half too --
    # tilt is unreferenced and is the larger of the two on ground clips.
    phys_fall_ref: bool = False
    # A key of losses.PHYS_WEIGHT_TABLES:
    #   "default"   the dt-migration weights, what every recorded number used
    #   "equal"     every active term 1.0. An ABLATION, not a trainable
    #               objective: the terms differ by ~3 orders of magnitude, so
    #               this makes L_phys == smooth to within 0.1%
    #   "balanced"  each term scaled to contribute ~1 at a typical rollout --
    #               the one that actually gives every term an equal say
    phys_weights: str = "default"
    # Per-frame weight decay inside every L_align sub-term: frame t counts
    # gamma ** t, normalised by the weights' sum (see losses._discounted_mean).
    # 1.0 is the plain mean and reproduces every number recorded before this
    # existed. Below 1.0 pulls the objective back toward the part of the episode
    # a single z can still influence -- a tracking error is cumulative, so at
    # gamma = 1 most of L_align is drift that no z could have prevented by then.
    # At 30 Hz over 300 frames: 0.995 leaves frame 300 at 22% of frame 0's
    # weight, 0.99 at 5%, 0.97 at 0.01% (i.e. only the first ~2 s count).
    # Applies to L_align only -- L_phys is a penalty, not an accumulating error,
    # and a fall at t=250 should cost exactly what a fall at t=10 costs.
    align_discount: float = 1.0
    # L_align's five sub-terms; the d_ prefix matches losses.py's d_root/d_ee/...
    d_root_weight: float = 1.0
    d_ee_weight: float = 1.0
    d_contact_weight: float = 1.0
    d_pose_weight: float = 1.0
    # 1/900, not 1.0, for the same reason as losses.PHYS_DEFAULT_WEIGHTS' migration
    # note: d_velocity is a squared first time-derivative, so real dt made it 900x
    # larger. This holds the objective at its pre-dt-fix value; 1.0 is the retune.
    d_velocity_weight: float = 1.0 / 30 ** 2
    # Control rate of the clips and of the env (15 physics steps of 1/450 s).
    # Everything scored from qpos is differentiated with dt = 1 / control_fps.
    control_fps: float = 30.0

    # PPO, ported from model/bilevel/ppo.py -- see train.py's "Per-step credit,
    # not one scalar per episode". The rollout is still not differentiable; what
    # changed is that the score-function estimator now gets one advantage per
    # WINDOW instead of one per episode.
    window_steps: int = 30      # 1.0 s @ 30 Hz. L_align/L_phys are computed on each window
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
                                    # the L_align/L_phys signal. It being this small is also why
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

    # W&B. One update is one body, so the pooled L_align/L_phys series alternates
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


@dataclasses.dataclass
class ESConfig:
    """model/simple/train_es.py -- antithetic evolution strategies on z_beta.

    Shares TrainConfig's model-defining fields (adapter, obs canonicalisation,
    loss weights, episode length) so the two are comparable, and drops every
    field that only existed to make a per-step policy gradient work: gamma,
    gae_lambda, ppo_*, value_*, window_steps, exploration_std, baseline_momentum.
    ES needs one scalar per rollout, so none of them have an analogue.
    """
    metamotivo_repo: str = "facebook/metamotivo-M-1"
    dataset_dir: str = "datasets/crossenbodiment-10bodies"
    # How many bodies one update's batch covers, 0 = all training bodies. This
    # is the file's whole premise (train_es.py "One motion, every body"): a batch
    # is clips_per_update clips x this many bodies, same z0, different beta, so
    # beta VARIES inside one gradient instead of being a per-update constant.
    # Each body needs its own batched rollout, so an update costs this many of
    # them -- lower it only to buy wall clock back.
    es_bodies_per_update: int = 0
    # Only read when es_bodies_per_update < the number of training bodies, i.e.
    # when an update covers a SUBSET: "cycle" walks the list so N updates still
    # give every body its share, "random" draws the subset i.i.d.
    body_order: str = "cycle"
    device: str = "cuda:0"
    # "all", or a task list written by scripts/spilt_tasks.py.
    # "ground" (24) and "upright" (30) are the partition: P.fall is 66.6% of the
    # cost and is a per-task CONSTANT on the tasks whose reference motion is
    # legitimately on the ground, so pooling them means most of the objective is
    # an offset the policy cannot move. "move" (17) is a SUBSET of upright, cut
    # on root displacement instead: d_root's heading/velocity terms carry the
    # objective on clips whose root travels and are ~0 by construction on clips
    # whose root stays put, so upright is still two landscapes and move is the
    # locomotion one. Selecting "move" therefore narrows "upright"; it does not
    # pick a disjoint group.
    task_group: str = "all"

    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 512, 512, 256])
    adapter_alpha: float = 0.1
    adapter_alpha_learnable: bool = False
    # True: z_beta = z0 + alpha * MLP([beta, z0]) -- the spec, and a prior that
    # z_beta belongs near z0. False: z_beta = MLP([beta, z0]), no skip and alpha
    # unused. The ablation exists because scripts/single_z_search.py measured the
    # best single-clip z at 79..96 degrees from z0, while alpha=0.1 needs an MLP
    # output of norm ~92 to reach even 30 degrees -- so the residual form may be
    # what is pinning the map near z0 rather than the optimizer.
    #
    # Read train_es.py's "adapter:" startup line before trusting a False run:
    # with adapter_project_z on, normalize gives ANY direction full radius, so
    # z_beta starts as a uniformly random point on the sphere (cos ~ 0) instead
    # of on z0 (cos ~ 1). That is a real cost, not a formality -- the run starts
    # from a latent the frozen actor has no reason to like.
    adapter_residual: bool = True
    adapter_project_z: bool = True

    obs_scale: str = "auto"
    obs_scale_parts: str = "length"
    obs_scale_ref_xml: str = "assets/robots/adult/robot.xml"

    # Fitness = lambda_align * L_align + lambda_phys * L_phys over the WHOLE episode,
    # from the unchanged model/losses.py -- the same number evaluate.py reports.
    lambda_align: float = 1.0
    lambda_phys: float = 1.0
    # NOT 0.1. Rank normalization strips g_z of the cost's units, so nothing the
    # PPO objective tuned carries over. MEASURED at lambda_z = 0.1:
    # |g_es| ~ 1.5 against |g_anchor| ~ 1.3e-5, a ratio of 8e-6 -- the anchor was
    # doing literally nothing. 10.0 is chosen so it stays negligible while
    # z_beta sits on top of z0 (1 - cos ~ 1e-5 today) and reaches ~10% of the ES
    # term once z has drifted to 1 - cos ~ 0.01, i.e. about 8 degrees: a soft
    # barrier against leaving the frozen actor's distribution, not a constant
    # drag. train_es.py logs g_es_norm / g_anchor_norm / g_ratio every update, so
    # re-derive this rather than trusting it if the rank scheme or sigma changes.
    lambda_z: float = 10.0
    # Weight of the "keep the best point" term: lambda_bc * (1 - cos(z_beta,
    # best_z)), where best_z is the lowest-cost CANDIDATE ever rolled out for
    # that (clip, body) cell -- train_es.update_best_buffer.
    #
    # 0.0 = OFF, and off is bit-identical to the loop before this existed: no
    # buffer is built, no branch taken, no RNG drawn. Turn it on only
    # deliberately, and read the logged bc_ratio when you do.
    #
    # What it is for: scripts/single_z_search.py escapes the basin around z0 and
    # the adapter does not, and one of the two things it does differently is
    # keep the best candidate it ever evaluated instead of only following the
    # gradient. This gives the adapter that memory. The term's gradient is
    # EXACT, unlike the ES term's zeroth-order estimate, so it can dominate at a
    # bc_ratio well under 1 -- start at 0.1..1.0, not at lambda_z's 10.
    #
    # Known caveat, measured: two searches of the same cell with different seeds
    # land ~42 degrees apart at equal cost (outputs/zmap_fit/seed_disagreement.png),
    # so "the best z" for a cell is one draw from a broad set, not a point. Here
    # that bites less than it did for the offline targets -- every candidate is
    # within sigma of the CURRENT z_beta, so the buffer tracks the iterate rather
    # than naming an absolute answer -- but a target that stops improving while
    # z_beta moves on will go stale, which is what bc_deg is logged for.
    lambda_bc: float = 0.0
    # Per-frame weight decay inside every L_align sub-term: frame t counts
    # gamma ** t, normalised by the weights' sum (see losses._discounted_mean).
    # 1.0 is the plain mean and reproduces every number recorded before this
    # existed. Below 1.0 pulls the objective back toward the part of the episode
    # a single z can still influence -- a tracking error is cumulative, so at
    # gamma = 1 most of L_align is drift that no z could have prevented by then.
    # At 30 Hz over 300 frames: 0.995 leaves frame 300 at 22% of frame 0's
    # weight, 0.99 at 5%, 0.97 at 0.01% (i.e. only the first ~2 s count).
    # Applies to L_align only -- L_phys is a penalty, not an accumulating error,
    # and a fall at t=250 should cost exactly what a fall at t=10 costs.
    align_discount: float = 1.0
    # L_align's five sub-terms; the d_ prefix matches losses.py's d_root/d_ee/...
    d_root_weight: float = 1.0
    d_ee_weight: float = 1.0
    d_contact_weight: float = 1.0
    d_pose_weight: float = 1.0
    # 1/900, not 1.0, for the same reason as losses.PHYS_DEFAULT_WEIGHTS' migration
    # note: d_velocity is a squared first time-derivative, so real dt made it 900x
    # larger. This holds the objective at its pre-dt-fix value; 1.0 is the retune.
    d_velocity_weight: float = 1.0 / 30 ** 2
    # Control rate of the clips and of the env (15 physics steps of 1/450 s).
    # Everything scored from qpos is differentiated with dt = 1 / control_fps.
    control_fps: float = 30.0

    # eps ~ N(0, I_256) has |eps| ~ sqrt(256) = 16 and |z| = 16, so |sigma*eps| /
    # |z| = sigma: **sigma IS the fractional perturbation of z**. 0.25 moves z by
    # 26% (measured), which is a large step, not a small probe. Too small and
    # |F+ - F-| drowns in the noise of F (train_es.py logs delta_abs and
    # delta_std so that ratio is visible); too large and the difference stops
    # being a local measurement of the landscape.
    es_sigma: float = 0.25
    es_pairs: int = 4            # antithetic pairs per clip; batch_size // (2*this)
                                 # clips fit in one update
    es_rank_normalize: bool = True   # see train_es.py:rank_normalize -- L_align is
                                     # heavy-tailed and one bad clip would
                                     # otherwise set the step size. Applied PER
                                     # ROW; ranking across clips would compare
                                     # clips instead of directions.

    lr: float = 3e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 16         # env slots; must be a multiple of 2*es_pairs
    vectorization_mode: str = "sync"
    num_updates: int = 400
    steps_per_episode: int = 300
    seed: int = 0

    eval_every: int = 50
    eval_at_start: bool = True
    eval_seed: int = 12345

    use_wandb: bool = True
    wandb_project: str = "crossenbodiment-simple"
    wandb_run_name: Optional[str] = None
    progress: bool = True
    log_every: int = 1
    ckpt_dir: str = "outputs/simple_es/checkpoints"
    ckpt_every: int = 50


@dataclasses.dataclass
class ZMapConfig:
    """model/simple/train_zmap.py -- supervised (z_adult, beta) -> z_body.

    No simulator, so nothing from TrainConfig/ESConfig about rollouts, rewards
    or estimators applies. What carries over is only the network itself: this
    trains the SAME LatentAdapter, on a labelled target instead of a physics
    objective, so a checkpoint from here is loadable by the rollout paths.
    """
    dataset_dir: str = "datasets/crossenbodiment-10bodies"
    device: str = "cuda:0"

    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 512, 512, 256])
    # alpha=1.0, not 0.1. The residual scale exists to keep z_beta near z0 when
    # the objective is a noisy rollout; here the target is labelled and the map
    # has to reach it. Measured cos(z_adult, z_body) is well below 1, so a 0.1
    # residual could not span the gap even in principle.
    adapter_alpha: float = 1.0
    adapter_alpha_learnable: bool = False
    adapter_project_z: bool = True   # both endpoints are on the sphere already
                                     # (tracking_inference calls project_z)

    lr: float = 1e-3
    lr_final: float = 1e-5           # cosine schedule over the whole run
    batch_size: int = 4096           # frames, mixed across bodies
    epochs: int = 20
    grad_clip_norm: float = 1.0
    seed: int = 0
    # Put the whole training pool on the GPU. ~1.3M frames x 256 x 4 B x 2
    # (src+dst) plus beta is about 3 GB; set False to stream from host memory.
    preload_device: bool = True

    use_wandb: bool = True
    wandb_project: str = "crossenbodiment-simple"
    wandb_run_name: Optional[str] = None
    progress: bool = True
    log_every: int = 20
    ckpt_dir: str = "outputs/simple_zmap"

