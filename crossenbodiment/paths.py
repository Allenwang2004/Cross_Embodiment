"""paths.py -- where things live in the repository, in one place (OOP refactor step 2, docs/refactor/PLAN.md).

REPO is found from this file (crossenbodiment/paths.py -> the repository root), as every script finds it from its own path.

Two spellings of every location, because the old code uses both and writes both into its outputs:
  *_REL   the repository-relative POSIX string, as the drivers put it on the command line (and single_z_search.py
          then writes it into summary.json `xml` / `ref`): use it wherever the old code had the relative literal
  plain   the absolute Path (REPO / *_REL), for the old `REPO / "..."` spellings
The helper functions return the relative string; `REPO / helper(...)` gives the absolute path. Nothing here creates,
moves or checks a file.

Clip names: a task clip is "<task>/<stem>" (data/origin_motion/<task>/<stem>.npz); an AMASS clip is the bare ACCAD file
stem (amass_data/amass_origin_motion/ACCAD/<clip>.npz). The robots' targets are keyed by the stem alone.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- bodies
ADULT_XML_REL = "assets/robots/adult/robot.xml"                          # the SMPL adult Motivo was trained on
CHILD_XML_REL = "assets/robots_torque/child/robot_torque_full.xml"       # the child, torque-scaled (every child search)
ADULT_XML = REPO / ADULT_XML_REL
CHILD_XML = REPO / CHILD_XML_REL
ROBOTS_REL = "assets/robots"                                             # <body>/robot.xml: shape (and adult-size variants)
ROBOTS_TORQUE_REL = "assets/robots_torque"                               # <body>/robot_torque_full.xml: scaled bodies' physics

# ---------------------------------------------------------------- motions and latents
DATA_REL = "data"
ORIGIN_MOTION_REL = "data/origin_motion"                                 # the adult's task clips (qpos, 30 fps)
ORIGIN_Z_REL = "data/origin_z"                                           # Motivo's task latent z0 per task clip
AMASS_ORIGIN_MOTION_REL = "amass_data/amass_origin_motion/ACCAD"         # the adult's AMASS (ACCAD) clips
AMASS_RETARGETING_MOTION_REL = "amass_data/amass_retargeting_motion/ACCAD"   # the CHILD's AMASS clips
OUTPUTS_REL = "outputs"
ORIGIN_MOTION = REPO / ORIGIN_MOTION_REL
ORIGIN_Z = REPO / ORIGIN_Z_REL
AMASS_ORIGIN_MOTION = REPO / AMASS_ORIGIN_MOTION_REL
AMASS_RETARGETING_MOTION = REPO / AMASS_RETARGETING_MOTION_REL
OUTPUTS = REPO / OUTPUTS_REL

# ---------------------------------------------------------------- the frozen policy
MOTIVO_CKPT = "facebook/metamotivo-M-1"                                  # FBcprModel.from_pretrained id (HF_HUB_OFFLINE=1)


def origin_motion(clip):
    """the adult's task clip: data/origin_motion/<task>/<stem>.npz"""
    return f"{ORIGIN_MOTION_REL}/{clip}.npz"


def origin_z(clip):
    """the task latent z0 of a task clip: data/origin_z/<task>/<stem>.npy"""
    return f"{ORIGIN_Z_REL}/{clip}.npy"


def retargeting_motion(body, clip):
    """a body's own task clip (scripts/qpos_retarget.py, make_scaled_bodies.py): data/<body>/retargeting_motion/<clip>.npz"""
    return f"{DATA_REL}/{body}/retargeting_motion/{clip}.npz"


def amass_origin_motion(clip):
    """the adult's AMASS clip: amass_data/amass_origin_motion/ACCAD/<clip>.npz"""
    return f"{AMASS_ORIGIN_MOTION_REL}/{clip}.npz"


def amass_retargeting_motion(clip):
    """the CHILD's AMASS clip: amass_data/amass_retargeting_motion/ACCAD/<clip>.npz"""
    return f"{AMASS_RETARGETING_MOTION_REL}/{clip}.npz"


def body_amass_retargeting_motion(body, clip):
    """another body's AMASS clip: data/<body>/amass_retargeting_motion/ACCAD/<clip>.npz (made so far: len1.4_wid0.8)"""
    return f"{DATA_REL}/{body}/amass_retargeting_motion/ACCAD/{clip}.npz"


def shape_xml(body):
    """assets/robots/<body>/robot.xml"""
    return f"{ROBOTS_REL}/{body}/robot.xml"


def torque_xml(body):
    """assets/robots_torque/<body>/robot_torque_full.xml (single_z_search.py's and lso_search.py's default --xml)"""
    return f"{ROBOTS_TORQUE_REL}/{body}/robot_torque_full.xml"


def robot_out(robot):
    """outputs/<robot> (model/robots.py out_dir, relative)"""
    return f"{OUTPUTS_REL}/{robot}"


def roundtrip(robot, stem):
    """the task clip mapped onto a robot (scripts/robot_roundtrip.py; the G1's g1_roundtrip.py):
    outputs/<robot>/roundtrip/<stem>/g1.npz for the G1, .../robot.npz otherwise (model/robots.py target_path)"""
    return f"{robot_out(robot)}/roundtrip/{stem}/{'g1.npz' if robot == 'g1' else 'robot.npz'}"


def stem(clip):
    """'<task>/<stem>' -> '<stem>' (an AMASS clip name is returned unchanged)"""
    return clip.split("/")[-1]


def is_task_clip(clip):
    """task clips are '<task>/<stem>'; AMASS clips are bare ACCAD names"""
    return "/" in clip
