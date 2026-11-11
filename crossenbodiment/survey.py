"""survey.py -- the user survey's clips, bodies and runs as data (scripts/survey_body_videos.py's config until OOP refactor step 4).

survey_body_videos.py, package_survey.py, body_reward_eval.py, child_reward_eval.py, the robot / LSO / G1 video scripts and the
map-error figures imported these from survey_body_videos.py; they import them from here now, and survey_body_videos.py imports
them back. Every value is the one survey_body_videos.py had (tests/test_moves.py compares them with its 68c95f4 source):

  CLIPS       the 11 survey stems (configs/clips/user_survey.txt)
  ADULT       the adult's XML (paths.ADULT_XML_REL); ADULT_H its stature, 1.74 m
  BODY        {survey name: dict(xml, ref, h[, clips])} built from configs/survey.yaml: xml is the registry body's physics_xml
              (crossenbodiment/bodies.py) for child, heavy_arm (= payload) and limit_leg, and the old literal for the deleted
              survey bodies fat and tall (see configs/survey.yaml: the survey's tall is not the registry's)
  SUFFIXES    the run-folder suffixes run_of chooses from
  run_of      which search the survey videos use for (name, stem)
The colours are in crossenbodiment/palette.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import paths

CONFIG = paths.REPO / "configs" / "survey.yaml"

CLIPS = paths.read_clips(paths.USER_SURVEY)
ADULT = paths.ADULT_XML_REL
SUFFIXES = ("_s0", "_g512_s0", "_sig015_s0", "_sig015_cont_s1", "_sig015_cont2_s2", "_sig015_s1", "_sig015_s2", "_sig015_s3")
CHILD_RUNS = "outputs/c540_bfmglobal_a03/{stem}"                       # the child's survey searches ...
CHILD_RUN_OVERRIDE = {"rotate-x--5-0.8_4": "outputs/user_survey_search/rotate-x--5-0.8_4_g256_s0"}   # ... but this one's 256-step rerun


def _load(path=CONFIG):
    import yaml
    from .bodies import BODIES
    with open(path) as f:
        cfg = yaml.safe_load(f)
    body = {}
    for name, d in cfg["bodies"].items():
        b = dict(xml=BODIES[d["body"]].physics_xml if d.get("body") else d["xml"], ref=d["ref"], h=d["h"])
        if "clips" in d:
            b["clips"] = dict(d["clips"])
        body[name] = b
    return cfg["adult_h"], body


ADULT_H, BODY = _load()


def run_of(name, stem, root=None):
    """the search folder (repository-relative) the survey videos use for body `name` on `stem`: the child's from
    outputs/c540_bfmglobal_a03 (one rerun excepted), the others' the outputs/user_survey_search/<name>/<stem><suffix> with the
    lowest best cost. root: where the runs are looked up (default paths.REPO; scripts/survey_body_videos.py passes its own
    REPO, the main tree's absolute path, as it always did)."""
    root = paths.REPO if root is None else Path(root)
    if name == "child":
        return CHILD_RUN_OVERRIDE.get(stem, CHILD_RUNS.format(stem=stem))
    runs = [f"outputs/user_survey_search/{name}/{stem}{suf}" for suf in SUFFIXES
            if (root / f"outputs/user_survey_search/{name}/{stem}{suf}/summary.json").exists()]
    return min(runs, key=lambda d: json.loads((root / d / "summary.json").read_text())["best"]["cost"])
