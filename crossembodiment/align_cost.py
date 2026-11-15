"""align_cost.py -- the L_align / L_phys rollout cost (model/losses.py) and the settings it reads, out of the training code.

OOP refactor step 4 (docs/refactor/PLAN.md). scripts/single_z_search.py used to import `ESConfig` (model/simple/config.py) and
`compute_batch_cost` (model/simple/train.py), i.e. the abandoned adapter-training package with its wandb / dataset / networks
imports, for one function and a dozen settings. Both live here now, operation for operation as they were at 68c95f4
(tests/test_moves.py runs the old source against this one); model/simple/train.py re-exports `compute_batch_cost` and
`_phys_weight_table` from here, so the trainer and this module share one definition.

    cfg = AlignCostConfig(lambda_align=1.0, lambda_phys=0.0)
    cost, align, phys = compute_batch_cost(fk, cfg, Q, [goal] * len(Q))    # (B,) float32 each

`compute_batch_cost` duck-types its config: AlignCostConfig, model/simple's ESConfig and TrainConfig all work (it reads the
fields below; phys_weights, phys_fall_ref and align_discount by getattr with the defaults below, as before).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from model import losses


@dataclasses.dataclass
class AlignCostConfig:
    """The fields of model/simple/config.py ESConfig that compute_batch_cost and single_z_search.py read, with ESConfig's
    defaults (phys_weights / phys_fall_ref are not ESConfig fields: single_z_search.py set them on the instance, and
    compute_batch_cost reads them with these defaults)."""
    lambda_align: float = 1.0
    lambda_phys: float = 1.0
    phys_weights: str = "default"           # a key of losses.PHYS_WEIGHT_TABLES
    phys_fall_ref: bool = False             # L_phys's fall term relative to the reference
    align_discount: float = 1.0
    d_root_weight: float = 1.0
    d_ee_weight: float = 1.0
    d_contact_weight: float = 1.0
    d_pose_weight: float = 1.0
    d_velocity_weight: float = 1.0 / 30 ** 2
    control_fps: float = 30.0
    obs_scale_parts: str = "length"         # model/obs_scale.py build_obs_multiplier(parts=...): "length" also rescales local_body_vel


def _phys_weight_table(cfg):
    """cfg.phys_weights -> a weights dict, or None for physics_penalty's own
    default. See losses.PHYS_WEIGHT_TABLES."""
    w = losses.PHYS_WEIGHT_TABLES.get(getattr(cfg, "phys_weights", "default"))
    return None if w is losses.PHYS_DEFAULT_WEIGHTS else w


def compute_batch_cost(fk_model, cfg, qpos_beta, qpos_refs, return_terms=False):
    """qpos_beta: (B, T, nq) numpy. qpos_refs: length-B list, entries may be
    None (see losses.functional_equivalence). L_align/L_phys use forward kinematics
    per-trajectory (not batched), looped here since it's cheap vs simulation.
    Returns per-item cost/L_align/L_phys arrays, shape (B,); with return_terms,
    a fourth element, the length-B list of L_phys's unweighted per-term dicts
    (free -- physics_penalty already returns them, and re-deriving them costs a
    second forward-kinematics pass per trajectory)."""
    B = qpos_beta.shape[0]
    costs = np.empty(B, dtype=np.float32)
    align_totals = np.empty(B, dtype=np.float32)
    l_physes = np.empty(B, dtype=np.float32)

    d_weights = {
        "root": cfg.d_root_weight,
        "ee": cfg.d_ee_weight,
        "contact": cfg.d_contact_weight,
        "pose": cfg.d_pose_weight,
        "velocity": cfg.d_velocity_weight,
    }
    dt = 1.0 / cfg.control_fps
    phys_w = _phys_weight_table(cfg)
    fall_ref = getattr(cfg, "phys_fall_ref", False)
    term_list = []
    for i in range(B):
        align_total, _ = losses.functional_equivalence(
            fk_model, qpos_beta[i], qpos_refs[i], d_weights, dt,
            discount=getattr(cfg, "align_discount", 1.0))
        l_phys, terms = losses.physics_penalty(
            fk_model, qpos_beta[i], weights=phys_w, dt=dt,
            qpos_ref=qpos_refs[i] if fall_ref else None)
        costs[i] = cfg.lambda_align * align_total + cfg.lambda_phys * l_phys
        align_totals[i] = align_total
        l_physes[i] = l_phys
        term_list.append(terms)
    if return_terms:
        return costs, align_totals, l_physes, term_list
    return costs, align_totals, l_physes

