"""beta.py -- the body parameters beta of the SMPL scaled bodies, in one place (OOP refactor step 9).

BETA_AXES is the 8-axis body vector: the length and girth factor of four body groups. It used to be written twice, as
scripts/scale_robot.py AXES and model/dataset.py BETA_AXES (the order had to match); both import it from here now. A scaled body's
parameter.json carries these 8 keys (crossembodiment/bodybuild/builders.py ScaledBody), which model/dataset.py load_beta reads.

Importing this module imports nothing (no numpy, no mujoco), so model/dataset.py stays light.
"""

# the 8 axes, in the order of beta vectors (model/dataset.py load_beta) and of scale_robot.py's CLI flags
BETA_AXES = ["leg_scale", "arm_scale", "torso_scale", "head_scale",
             "leg_girth", "arm_girth", "torso_girth", "head_girth"]

GROUPS = ("leg", "arm", "torso", "head")

# which bodies of the adult each group's length / girth factor applies to (scale_robot.py BODY_GROUPS); the Pelvis is in no group
# (its contact box follows the torso girth: shape.apply_scale)
BODY_GROUPS = {
    "leg": ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
    "arm": ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
            "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
    "torso": ["Torso", "Spine", "Chest", "Neck"],
    "head": ["Head"],
}
GROUP_OF_BODY = {name: group for group, names in BODY_GROUPS.items() for name in names}

# scale_robot.py's presets (rough illustrative defaults; the child is the one in use)
PRESETS = {
    "adult": dict(leg_scale=1.00, arm_scale=1.00, torso_scale=1.00, head_scale=1.00,
                  leg_girth=1.00, arm_girth=1.00, torso_girth=1.00, head_girth=1.00),
    # Children: proportionally shorter legs/arms/torso, slightly bigger &
    # rounder head, thinner limbs.
    "child": dict(leg_scale=0.62, arm_scale=0.65, torso_scale=0.75, head_scale=1.05,
                  leg_girth=0.70, arm_girth=0.70, torso_girth=0.85, head_girth=1.15),
    # Elderly: mild height loss, thinner limbs (sarcopenia), thicker torso.
    "elderly": dict(leg_scale=0.92, arm_scale=0.97, torso_scale=0.95, head_scale=1.00,
                    leg_girth=0.80, arm_girth=0.78, torso_girth=1.08, head_girth=1.00),
}


def isometric_beta(L, W):
    """the beta of scripts/make_scaled_bodies.py's len<L>_wid<W> bodies (user, 11/11): every segment length x L and every girth
    x L * W (rounded to 6 decimals as make_scaled_bodies passed it on the command line; girth = W alone would leave a short body
    with adult-sized head, feet and limbs)"""
    return {**{f"{g}_scale": L for g in GROUPS}, **{f"{g}_girth": round(L * W, 6) for g in GROUPS}}


def lerp_beta(target, t, ndigits=4):
    """beta(t) = (1 - t) * 1 + t * beta_target, each axis rounded to `ndigits` (scripts/make_morph_path.sh: python3 -c
    "print(round((1-$t)*1.0 + $t*<target>, 4))"); target: 8 values in BETA_AXES order"""
    return {a: round((1 - t) * 1.0 + t * float(v), ndigits) for a, v in zip(BETA_AXES, target)}


def grid_beta(leg, arm):
    """scripts/make_grid_bodies.sh's leg x arm grid point: leg and arm length set, every other axis 1.0"""
    return dict(leg_scale=leg, arm_scale=arm, torso_scale=1.0, head_scale=1.0,
                leg_girth=1.0, arm_girth=1.0, torso_girth=1.0, head_girth=1.0)
