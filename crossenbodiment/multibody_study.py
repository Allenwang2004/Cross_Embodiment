"""multibody_study.py -- the layout of the multi-body correction experiment of 10/10-12 (scripts/multibody_corr.py): its bodies,
seeds, budgets, analysis-E arms and folders. Moved out of the driver in OOP refactor step 4, unchanged; multibody_corr.py imports
it back, plot_multibody_E.py imports it from here.

Not crossenbodiment.bodies: these are the experiment's 32 bodies of 10/10 (assets/robots/<body>, most of them deleted since:
outputs/body_shapes_deleted_2026-11-11.txt), named as in outputs/multibody_corr; BODIES here is that list, not the registry.

  OUT, LOG                  outputs/multibody_corr, outputs/train_logs/multibody_corr (absolute)
  ADULT, GRID, AXIS, NAMED  the reference body, the leg x arm grid, the one-axis bodies, the named bodies; GROUPS {body: group},
                            BODIES (their order), BETA_KEYS (parameter.json's beta), SEEDS, GENS, PAIRS, N_RANDOM
  E_TEST, E_ARMS, E_K       analysis E: the 8 test bodies, the 8-d bases compared, the basis size
  beta(body), clips(), ref_path(body, clip), cell_dir(body, clip, seed)
"""
import json

from .paths import REPO

OUT = REPO / "outputs/multibody_corr"
LOG = REPO / "outputs/train_logs/multibody_corr"
ADULT = "m2c_t000"
GRID = [["gl0500a0550", "ax0m_t1000", "gl0500a1400"],
        ["gl0750a0550", "ax0m_t500", "gl0750a1400"],
        ["ax1m_t1000", ADULT, "ax1p_t1000"],
        ["gl1250a0550", "ax0p_t500", "gl1250a1400"],
        ["gl1500a0550", "ax0p_t1000", "gl1500a1400"]]
AXIS = ["ax2m_t1000", "ax3m_t1000", "ax4m_t1000", "ax5m_t1000", "ax6m_t1000", "ax7m_t1000"]
NAMED = ["child", "teen", "petite", "tall_slim", "long_limbed", "short_limbed", "athletic", "elderly",
         "pear_shaped", "giant", "short_stocky"]
GROUPS = {**{b: "grid" for row in GRID for b in row}, **{b: "axis" for b in AXIS}, **{b: "named" for b in NAMED}}
BODIES = list(GROUPS)
BETA_KEYS = ["leg_scale", "arm_scale", "torso_scale", "head_scale", "leg_girth", "arm_girth", "torso_girth", "head_girth"]
SEEDS = [0, 1]
GENS, PAIRS = 64, 8
N_RANDOM = 16

E_TEST = ["child", "giant", "short_stocky", "gl0500a0550", "gl1500a1400", "gl0500a1400", "ax2m_t1000", "ax4m_t1000"]
E_ARMS = ["multi_same", "multi_loco", "child524", "rand8"]
E_K = 8


def beta(body):
    p = json.load(open(REPO / f"assets/robots/{body}/parameter.json"))
    return [float(p[k]) for k in BETA_KEYS]


def clips():
    return [l.strip() for l in open(OUT / "clips.txt") if l.strip()]


def ref_path(body, clip):
    return REPO / f"data/{body}/retargeting_motion/{clip}.npz"


def cell_dir(body, clip, seed):
    return OUT / "search" / body / f"{clip.split('/')[1]}_s{seed}"
