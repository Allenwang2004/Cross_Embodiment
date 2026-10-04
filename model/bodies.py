"""The 11-body roster, its train / test split, and two small body helpers.

Moved here from model/bilevel/config.py and model/bilevel/data.py when the bilevel line was removed
(2026-10-04; the full pre-cleanup tree is on git branch snapshot/pre-cleanup-2026-10-04). Used by the
body-pipeline scripts: scripts/write_body_splits.py, scripts/pd_track_bodies.py,
scripts/replay_actions_on_body.py. The child-only latent study does not use the roster: it names its body
(`child`) directly.
"""
import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class BodyRoster:
    # actuator-calibrated assets (scripts/torque_*.py); assets/robots/ ships the adult's actuators on every body
    robots_dir: str = "assets/robots_calib_move"
    source_body: str = "adult"   # the body the frozen policy was trained on
    # Held-out pair chosen for EXTRAPOLATION: training spans 38-101 kg and 0.59-1.07 m rest height; giant
    # (110.7 kg, 1.17 m) and short_stocky (130.5 kg) sit outside it on the axis they probe.
    train_bodies: List[str] = dataclasses.field(default_factory=lambda: [
        "child", "teen", "petite", "tall_slim",
        "long_limbed", "short_limbed", "athletic", "elderly", "pear_shaped",
    ])
    heldout_bodies: List[str] = dataclasses.field(default_factory=lambda: ["giant", "short_stocky"])
    action_repeat: int = 15      # humenv: simulation_dt = 1/450, control at 30 Hz
    # Termination test of scripts/replay_actions_on_body.py: the root drops below this fraction of the
    # reference's root height, or the up-vector falls more than term_up_margin below the REFERENCE's own
    # (relative, so a correct headstand is not a fall; with an upright reference it equals the absolute 0.2).
    term_root_height_frac: float = 0.5
    term_up_margin: float = 0.8


def split_of(roster: BodyRoster, name: str) -> str:
    """'source', 'train', 'test' or 'unused' -- the one definition scripts/write_body_splits.py writes into
    every body's parameter.json."""
    if name == roster.source_body:
        return "source"
    if name in roster.train_bodies:
        return "train"
    if name in roster.heldout_bodies:
        return "test"
    return "unused"


def up_z(quat_wxyz) -> float:
    """World-z component of the body's up axis: 2 (q_y q_z + q_w q_x), NOT 1 - 2 (q_x^2 + q_y^2) -- this
    asset's Pelvis carries euler="90 0 0", so its local +Y, not +Z, is world up (model/losses.py says the same)."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    return float(2.0 * (y * z + w * x))
