"""clips.py -- ClipBundle: the trajectories of one clip that a figure or video draws, in the user survey's format (OOP refactor step 10).

A clip folder (outputs/user_survey/<body>/trajectories/<clip>/, outputs/strobe_task52_blender/trajectories/<example>/, ...) holds one
.npz per trajectory -- qpos (frames x nq, float32) and fps (float32 30) -- and video.json: clip, body, fps, frames (each
trajectory's frame count, in file order), then what made them (search_run, cem_run, the selection / falls / candidate notes) and
how the survey videos draw them (side_by_side, overlay). Trajectory names: reference_adult (the adult's original motion),
reference_body (the body's own target), z0 (the rollout of z0), best (of the searched latent), cem (CEM (BFM-Zero)'s best), and for
the latent-sequence clips zero_shot / lso_best.

  ClipBundle.from_run(clip, body, run, reference_body, ...)   the survey alignment (rollout frame t against reference frame t + 1):
                     the references from their second frame, every rollout cut to the reference's length; run = a single-latent
                     search folder ("outputs/...", read through paths.read_path) -> its origin_z.npz and best.npz
  ClipBundle.from_files(clip, body, {name: path}, **meta)     trajectories as they are (package_survey.py: whole clips)
  ClipBundle.from_survey(folder)                               an existing clip folder
  bundle.add(name, qpos) / add_cem(run)                        another trajectory (cem: the CEM run's best, cut to the reference)
  bundle.save(folder)                                          the .npz files in order, then video.json (indent 1)
  copy_folder(src, dst)                                        a survey-format folder copied file by file (byte-identical)
  strobe_source(src, clip, body)                               strobe_final_blender.py's trajectory sources: "child_52",
                     "child_52_s<k>", "c540", "tall_body_52", "body_52/<body>", "kpkd:<folder>" (a single-latent run + the body's
                     target) or any other string (a survey-format clip folder to copy)
These replace the packaging code of package_survey.py, package_survey_picks.py, package_amass_lso_survey.py, body_all_blender.py,
payload_fall_candidates.py and strobe_final_blender.py (trajectories / add_cem); tests/golden/verify_vis.py re-makes their stored
clip folders (arrays and video.json text identical).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import paths

FILES = ("reference_adult.npz", "reference_body.npz", "z0.npz", "best.npz", "video.json")
# the body's own target (its retargeted task clips) by strobe-figure body name; default the child's
TARGETS = {"payload": "data/origin_motion", "limit_leg": "data/adult_lockRleg_bent_lim/retargeting_motion",
           "tall": "data/len1.4_wid0.8/retargeting_motion"}
CHILD_TARGET = "data/child/retargeting_motion"
RUN_SOURCES = ("child_52", "c540", "tall_body_52")
RUN_PREFIXES = ("body_52/", "child_52_s", "kpkd:")


def task_of(clip):
    """'<task>_<k>' -> '<task>' (the motion folder of a task clip stem)"""
    return clip.rsplit("_", 1)[0]


def origin_clip(clip):
    """the adult's task clip of a stem: data/origin_motion/<task>/<stem>.npz"""
    return f"{paths.ORIGIN_MOTION_REL}/{task_of(clip)}/{clip}.npz"


def is_run_source(src):
    return src in RUN_SOURCES or src.startswith(RUN_PREFIXES)


def run_of_source(src, clip):
    """the single-latent search folder ("outputs/...") behind a strobe source string"""
    if src.startswith("kpkd:"):                                    # kpkd:<folder>: a gain-variant test (user, 11/12), z0 and LOES in <folder>/es_s0
        return f"outputs/{src[len('kpkd:'):]}/es_s0"
    if src.startswith("child_52_s"):                               # child_52_s<k>: the child 52-motion run of seed k
        return f"outputs/child_52_rerun/{clip}_es_s{src[len('child_52_s'):]}"
    # c540: the older child search behind the survey's child videos; body_52/<body>: the 52-motion runs
    return {"child_52": f"outputs/child_52_rerun/{clip}_es_s0", "c540": f"outputs/c540_bfmglobal_a03/{clip}"}.get(src, f"outputs/{src}/{clip}_es_s0")


def strobe_source(src, clip, body):
    """("run", run folder, the body's target clip) or ("copy", the survey-format clip folder) for strobe_final_blender.py"""
    if not is_run_source(src):
        return ("copy", f"{src}/{clip}")
    return ("run", run_of_source(src, clip), f"{TARGETS.get(body, CHILD_TARGET)}/{task_of(clip)}/{clip}.npz")


def copy_folder(src, dst, files=FILES):
    """the files of a survey-format clip folder that exist, copied as they are"""
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        if (Path(src) / f).exists():
            shutil.copy(Path(src) / f, dst / f)
    return dst


def load_qpos(rel):
    return np.load(paths.read_path(rel))["qpos"]


class ClipBundle:
    def __init__(self, clip, body, qpos=None, meta=None, fps=30, order=None):
        self.clip, self.body, self.fps = clip, body, fps
        self.qpos = dict(qpos or {})              # trajectory name -> qpos array, in file order
        self.meta = dict(meta or {})              # video.json after clip / body / fps / frames, in order
        self.order = list(order or [])            # an existing video.json's key order (from_survey), kept when it is rewritten

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_run(cls, clip, body, run, reference_body, reference_adult=None, same_length=False, at_least=False, **meta):
        """the survey alignment of a single-latent search (strobe_final_blender.py, body_all_blender.py, payload_fall_candidates.py,
        package_survey_picks.py): references from frame 1, z0 / best cut to the reference's length. same_length: assert the adult's
        reference is as long as the body's; at_least: assert the rollouts are at least that long."""
        q = dict(reference_adult=load_qpos(reference_adult or origin_clip(clip))[1:], reference_body=load_qpos(reference_body)[1:])
        n = len(q["reference_body"])
        if same_length:
            assert len(q["reference_adult"]) == n
        for k, f in (("z0", paths.Z0_ROLLOUT), ("best", "best.npz")):
            x = np.load(paths.read_path(run) / f)["qpos"]
            if at_least:
                assert len(x) >= n, (clip, k, len(x), n)
            q[k] = x[:n]
        return cls(clip, body, q, dict(search_run=str(run), **meta))

    @classmethod
    def from_files(cls, clip, body, files, **meta):
        """trajectories read as they are ({name: repository-relative path}), e.g. package_survey.py's whole clips"""
        return cls(clip, body, {k: load_qpos(v) for k, v in files.items()}, meta)

    @classmethod
    def from_survey(cls, folder):
        """an existing clip folder: video.json and the .npz files its frames name"""
        folder = Path(folder); v = json.loads((folder / "video.json").read_text())
        q = {k: np.load(folder / f"{k}.npz")["qpos"] for k in v.get("frames", {})}
        meta = {k: x for k, x in v.items() if k not in ("clip", "body", "fps", "frames")}
        return cls(v["clip"], v["body"], q, meta, v.get("fps", 30), order=list(v))

    # ---------------------------------------------------------------- edits
    @property
    def n(self):
        return len(self.qpos["reference_body"])

    def add(self, name, q):
        self.qpos[name] = q
        return self

    def add_cem(self, run):
        """the CEM run's best rollout (run: "outputs/..."), cut to the reference's length (as z0 and best); video.json gets cem_run"""
        n = self.n; q = np.load(paths.read_path(run) / "best.npz")["qpos"]
        assert len(q) >= n, (self.clip, len(q), n)
        self.qpos["cem"] = q[:n]; self.meta["cem_run"] = run
        return self

    def video_json(self):
        out = dict(clip=self.clip, body=self.body, fps=self.fps, frames={k: len(v) for k, v in self.qpos.items()}, **self.meta)
        if self.order:                            # an existing file's key order, new keys after it
            out = {**{k: out[k] for k in self.order if k in out}, **{k: v for k, v in out.items() if k not in self.order}}
        return out

    # ---------------------------------------------------------------- files
    def save_npz(self, folder, names=None):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        for k in names or self.qpos:
            np.savez_compressed(folder / f"{k}.npz", qpos=self.qpos[k].astype(np.float32), fps=np.float32(30))
        return folder

    def save_json(self, folder):
        (Path(folder) / "video.json").write_text(json.dumps(self.video_json(), indent=1))

    def save(self, folder):
        """every trajectory (float32 qpos, fps 30), then video.json"""
        self.save_npz(folder); self.save_json(folder)
        return Path(folder)
