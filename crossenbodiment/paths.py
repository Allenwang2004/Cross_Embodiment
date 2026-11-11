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

Experiment outputs (refactor step 8): the drivers READ their experiment folders under outputs_root() and WRITE their analysis
results (per_run.npz, results.json, means.json, README.md, the re-roll indexes) under results_root(). Both are REPO/outputs unless
the environment says otherwise:
  CE_OUTPUTS_ROOT   where experiment folders are read (e.g. the main tree's outputs, read only, to re-analyse finished runs)
  CE_RESULTS_ROOT   where analyses write (default: the read root, i.e. next to the inputs as before); every results writer checks
                    writable(): inside REPO only, so pointing CE_OUTPUTS_ROOT at another tree without CE_RESULTS_ROOT fails instead
                    of writing there (printing commands still works)
The command lines keep their repository-relative "outputs/..." spellings (they run with the repository as cwd).
"""
import os
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

# ---------------------------------------------------------------- clip lists (configs/clips/<name>.txt, one clip per line)
# Refactor step 4: the drivers read these instead of the experiment folders' clips.txt; each is a byte-identical copy of the
# file it replaces (tests/test_moves.py compares them with the main tree's).
CLIPS_REL = "configs/clips"
CHILD_52_TASK = "child_52_task"             # the 52 task motions of Comparison A and the Task page (outputs/child_52/clips.txt)
AMASS_LOCOMOTION_5S = "amass_locomotion_5s"  # the 68 ACCAD clips with >= 5 s of locomotion (outputs/amass_baselines/clips.txt)
USER_SURVEY = "user_survey"                  # the 11 stems of the user survey's videos (scripts/survey_body_videos.py CLIPS)


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
    """the task clip mapped onto a robot (scripts/robot_roundtrip.py; the G1's first ones by g1_roundtrip.py, archived):
    outputs/<robot>/roundtrip/<stem>/g1.npz for the G1, .../robot.npz otherwise (model/robots.py target_path)"""
    return f"{robot_out(robot)}/roundtrip/{stem}/{'g1.npz' if robot == 'g1' else 'robot.npz'}"


# ---------------------------------------------------------------- experiment outputs: read root and results root (step 8)
OUTPUTS_ROOT_ENV = "CE_OUTPUTS_ROOT"
RESULTS_ROOT_ENV = "CE_RESULTS_ROOT"


def outputs_root():
    """where the drivers read experiment folders: $CE_OUTPUTS_ROOT, else REPO/outputs"""
    v = os.environ.get(OUTPUTS_ROOT_ENV)
    return Path(v) if v else OUTPUTS


def results_root():
    """where the drivers write analysis results: $CE_RESULTS_ROOT, else outputs_root() (writes are checked by writable())"""
    v = os.environ.get(RESULTS_ROOT_ENV)
    return Path(v) if v else outputs_root()


def writable(path):
    """`path` if it lies inside REPO (symlinks resolved), else RuntimeError: the results writers (crossenbodiment/results.py,
    reroll.py) call this, so a CE_OUTPUTS_ROOT pointing at another tree without CE_RESULTS_ROOT cannot write into that tree"""
    path = Path(path)
    if not path.resolve().is_relative_to(REPO):
        raise RuntimeError(f"refusing to write {path}: outside the repository {REPO} (set {RESULTS_ROOT_ENV} to a folder inside it; "
                           f"the read root is {outputs_root()})")
    return path


def output(rel=""):
    """an experiment path to READ: outputs_root() / rel (rel without the leading "outputs/", e.g. "child_52_rerun")"""
    return outputs_root() / rel


def result(rel=""):
    """an analysis path to WRITE: results_root() / rel (parents are created by the writers)"""
    return results_root() / rel


def clips_file(name):
    """configs/clips/<name>.txt (CHILD_52_TASK, AMASS_LOCOMOTION_5S, USER_SURVEY)"""
    return f"{CLIPS_REL}/{name}.txt"


def read_clips(name):
    """the clips of configs/clips/<name>.txt, in file order: one per line, stripped, blank lines skipped. On these files this is
    exactly what the drivers' own readers gave (`[l.strip() for l in open(f)]`, `f.read_text().split()`, the same with
    `if l.strip()`): every line is one clip name without spaces, and the file ends in one newline."""
    with open(REPO / clips_file(name)) as f:
        return [l.strip() for l in f if l.strip()]


def stem(clip):
    """'<task>/<stem>' -> '<stem>' (an AMASS clip name is returned unchanged)"""
    return clip.split("/")[-1]


def is_task_clip(clip):
    """task clips are '<task>/<stem>'; AMASS clips are bare ACCAD names"""
    return "/" in clip
