"""Rescale humenv's 358-dim proprio obs so a scaled body reads as the body the
frozen Metamotivo actor was trained on.

The obs DIMENSION does not change with the body -- a scaled skeleton keeps
humenv's 24 rigid bodies -- so nothing needs reshaping. What changes is the
units. Measured on move-ego--90-2, adult vs child at qvel=0:

    root_h_obs             1 dim    child/adult 0.6188   cos 1.0000
    local_body_pos        69 dims               0.6657   cos 0.9965
    local_body_rot_obs   144 dims               1.0000   cos 1.0000

The rotations are bit-identical, because the retarget copies every hinge angle
verbatim; only the features carrying a METRE move, and they move by very nearly
one scalar -- cos 0.9965 says the child's pose vector points the same way and is
simply shorter. So the mismatch the frozen actor sees is a units mismatch, not a
different pose. It is worth removing because the actor's obs normalizer is a
BatchNorm holding adult-scale running statistics, so those 70 features otherwise
arrive at a systematic offset rather than merely "smaller".

This is a canonicalisation and it removes part of the problem rather than
solving it: it tells the actor the body is adult-sized when it is not, so the
motions it commands are still calibrated for adult limb lengths. What is left
after it is the genuine dynamics mismatch -- different masses, inertias and
actuator authority -- which is what the LatentAdapter's z_beta has to explain.

Shared by scripts/rollout_z_on_body.py (where it is an ablation switch, run
with and without to split "how much of the gap is pure scale" from "how much is
real dynamics") and model/simple/train.py + evaluate.py (where it is on by
default, because z_beta is their only control channel and it should not be
spent re-deriving a unit conversion).
"""

import mujoco
import numpy as np


# humenv/env.py:compute_humanoid_self_obs_v2 concatenates an OrderedDict in this
# order over 24 rigid bodies; local_body_pos drops the root's own 3, and
# local_body_rot_obs is 6D tan-norm rather than quaternions.
OBS_SEGMENTS = {
    "root_h":       (0, 1),      # metre
    "body_pos":     (1, 70),     # metre
    "body_rot":     (70, 214),   # unitless
    "body_vel":     (214, 286),  # metre / second
    "body_ang_vel": (286, 358),  # radian / second
}

SCALED_BY_PARTS = {
    "pose":   ("root_h", "body_pos"),
    "length": ("root_h", "body_pos", "body_vel"),
}

DEFAULT_REF_XML = "assets/robots/adult/robot.xml"


def body_scale_ratios(xml: str, ref_xml: str) -> np.ndarray:
    """(24,) per-body length ratio of `xml` against the reference body.

    One scalar is NOT enough. scripts/scale_robot.py scales four groups
    independently and assets/robots/child/parameter.json uses leg 0.62, torso
    0.75, arm 0.65, head 1.05, so the distance from the pelvis to each body
    scales by a different amount -- measured 0.6200 for every leg body, 0.7500
    for the torso chain, 0.7847 for the head, and a gradient 0.7181 -> 0.6676
    down the arm as the chain leaves the torso and accumulates arm segments. The
    rest pelvis height ratio alone is 0.6110, below all of them and 28% wrong at
    the head.

    Measuring at the rest pose is enough because the ratios barely move: across
    100 frames of move-ego--90-2 the leg and torso ratios are constant to
    0.0000 (single-scale chains) and the mixed arm/head chains hold to a std of
    0.005, differing from their rest value by at most 0.019.

    Index 0 is the root, whose local_body_pos is identically zero and carries no
    length of its own; it gets the rest pelvis height ratio instead, which is
    what phi0_retarget scaled the root translation by.
    """
    out = []
    for path in (ref_xml, xml):
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d)
        pos = d.xpos[1:25].copy()
        out.append((pos - pos[0], float(m.qpos0[2])))
    (pa, ha), (pb, hb) = out
    ratios = np.ones(24)
    ratios[0] = hb / ha
    na, nb = np.linalg.norm(pa, axis=1), np.linalg.norm(pb, axis=1)
    ok = na > 1e-9
    ratios[1:][ok[1:]] = (nb[1:] / na[1:])[ok[1:]]
    return ratios


def obs_rescaler(ratios: np.ndarray, parts: str) -> np.ndarray:
    """(358,) multiplier making this body's length features read as the
    reference body's. Every feature carrying a metre is divided by its OWN
    body's ratio; rotations and angular velocities are left at 1.0.

    body_pos covers bodies 1..23 (the root's own offset is dropped by humenv),
    while body_vel covers all 24 -- the sensors are world-frame velocities, not
    root-relative ones, so the root has a real velocity to rescale. That makes
    the velocity term the approximate one: a body's world velocity mixes the
    root's translation with its own local motion, and those two carry different
    ratios. Use parts='pose' to leave it out.

    'length' also rescales local_body_vel, the dimensionally consistent choice
    under a kinematic retarget (same angles, same clock, so linear velocity
    carries the same metre as position while angular velocity does not); 'pose'
    rescales only the static features and leaves all 144 velocity dims alone,
    which is the right choice if the motion is gravity-driven, where speeds
    scale like sqrt(L) rather than L.
    """
    scaled = SCALED_BY_PARTS[parts]
    mul = np.ones(358)
    if "root_h" in scaled:
        mul[0] = 1.0 / ratios[0]
    if "body_pos" in scaled:
        mul[1:70] = 1.0 / np.repeat(ratios[1:24], 3)
    if "body_vel" in scaled:
        mul[214:286] = 1.0 / np.repeat(ratios[0:24], 3)
    return mul


def build_obs_multiplier(xml, ref_xml=DEFAULT_REF_XML, mode="auto", parts="length",
                         verbose=True):
    """(358,) multiplier, or None when `mode` is 'none'.

    `mode`: 'none' (raw obs), 'auto' (per-body ratios measured from the two rest
    poses -- the bodies are scaled by group, not uniformly), or a float string
    forcing ONE uniform ratio, which is the ablation 'auto' replaces.

    Callers that want the raw obs pass mode='none' and get None back rather than
    an all-ones array, so the multiply can be skipped entirely.
    """
    if mode == "none" or mode is None:
        return None
    if mode == "auto":
        ratios = body_scale_ratios(str(xml), str(ref_xml))
        how = f"per-body ratios {ratios.min():.4f}..{ratios.max():.4f} (root {ratios[0]:.4f})"
    else:
        uniform = float(mode)
        if not uniform > 0:
            raise SystemExit(f"obs scale must be positive, got {uniform}")
        ratios = np.full(24, uniform)
        how = f"uniform ratio {uniform:.4f}"
    mul = obs_rescaler(ratios, parts)
    if verbose:
        print(f"obs rescale: {how} -> {int((mul != 1.0).sum())}/358 features "
              f"({', '.join(SCALED_BY_PARTS[parts])}) multiplied by "
              f"{1.0 / ratios.max():.4f}..{1.0 / ratios.min():.4f}")
    return mul
