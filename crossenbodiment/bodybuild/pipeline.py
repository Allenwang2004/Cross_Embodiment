"""pipeline.py -- BodyPipeline: a new body from its builder to a searchable body, step by step (OOP refactor step 9).

The steps of docs/new_body.md, in order, each skipped when its outputs exist (force=True runs it again):

  shape         builder.build()                         <robots>/<b>/robot.xml, parameter.json
  skeleton      skeleton.dump_skeleton                  <robots>/<b>/skeleton.json
  retarget      scaled bodies: SameTopologyRetargeter over every motion folder of data/origin_motion (or `motions`), one
                folder per job, `workers` jobs at once -> <data>/<b>/retargeting_motion/<task>/<stem>.npz; adult-size variants:
                their own reference of each of `clips` (CopyReference, LockedJointReference, Elderly.reference)
  amass         optional (amass_clips): the AMASS clips -> <data>/<b>/amass_retargeting_motion/ACCAD/<clip>.npz
  torque_ratio  TorqueCalibrator.ratio_matrix           <outputs>/torque_ratio_across_motions/gravity/<b>/_k_matrix.csv ...
  aggregate     TorqueCalibrator.aggregate              .../_ratios_aggregate.csv
  physics       TorqueCalibrator.write_xml + checks     <robots_torque>/<b>/robot_torque_full.xml; the step-5 check line
  prune         optional (keep_tasks): delete the retarget folders of tasks not listed (make_morph_path.sh KEEP_TASKS, make_grid_bodies.sh:
                torque calibration used every motion, the study keeps only its tasks; a full retarget is ~200 MB)
  registry      is the body in configs/bodies.yaml? if not, its entry is written to <logs>/<b>.bodies.yaml, and with register=True
                (the repository root only) inserted into configs/bodies.yaml before its gain_variants section
The torque steps apply to scaled bodies only (an adult-size variant is simulated on its shape XML; configs/bodies.yaml).

Outputs go under BuildRoots(root): root/assets/robots, root/assets/robots_torque, root/data, root/outputs (default: the repository,
the layout every reader expects). The inputs are always the repository's: the adult's XML and skeleton.json, data/origin_motion,
amass_data/amass_origin_motion/ACCAD.

Resources (CLAUDE.md, a shared box): before any parallel launch (retarget workers; several bodies with run_many) available memory
minus the launch's estimate must stay >= keep_gb (50 GB); otherwise fewer workers, or wait (logged). Log: <logs>/<b>.log, every line
stamped in Taiwan time; the library's prints go there too.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from .. import paths
from ..queue import available_gb, stamp
from .builders import BodyBuilder, HeavyLimb, LimitJoints, LockJoints, ScaledBody
from .skeleton import dump_skeleton

ADULT_SKELETON = paths.REPO / "assets/robots/adult/skeleton.json"
WORKER_GB = 0.5          # one retarget worker (python + mujoco + numpy; measured ~0.2 GB), budgeted
BODY_GB = 1.5            # one body pipeline (main process + its default 2 workers), budgeted
KEEP_GB = 50.0           # CLAUDE.md: keep >= 50 GB available after a launch


@dataclass(frozen=True)
class BuildRoots:
    """where a build writes (module docstring); root defaults to the repository"""
    root: Path = paths.REPO
    mode: str = "gravity"

    @property
    def robots(self):
        return Path(self.root) / paths.ROBOTS_REL

    @property
    def robots_torque(self):
        return Path(self.root) / paths.ROBOTS_TORQUE_REL

    @property
    def data(self):
        return Path(self.root) / paths.DATA_REL

    @property
    def outputs(self):
        return Path(self.root) / paths.OUTPUTS_REL

    @property
    def logs(self):
        return self.outputs / "body_build"

    def matrix(self, body):
        return self.outputs / "torque_ratio_across_motions" / self.mode / body

    def physics_xml(self, body):
        return self.robots_torque / body / "robot_torque_full.xml"

    def retarget_dir(self, body):
        return self.data / body / "retargeting_motion"

    def amass_dir(self, body):
        return self.data / body / "amass_retargeting_motion" / "ACCAD"

    @property
    def is_repo(self):
        return Path(self.root).resolve() == paths.REPO


# ------------------------------------------------------------------ resources

def wait_for_memory(need_gb, keep_gb=KEEP_GB, log=print, poll_s=60, max_wait_s=6 * 3600):
    """block until available memory minus need_gb >= keep_gb (logged while waiting); returns the available GB"""
    t0 = time.time()
    while True:
        av = available_gb()
        if av - need_gb >= keep_gb:
            return av
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"memory: {av:.0f} GB available, {need_gb:.1f} GB needed, {keep_gb:.0f} GB to keep; gave up")
        log(f"memory: {av:.0f} GB available, need {need_gb:.1f} GB + keep {keep_gb:.0f} GB: waiting")
        time.sleep(poll_s)


def fit_workers(want, per_gb=WORKER_GB, keep_gb=KEEP_GB, log=print):
    """the number of workers (<= want, >= 1) that leaves keep_gb available; waits when not even one fits"""
    av = wait_for_memory(per_gb, keep_gb, log)
    n = max(1, min(want, int((av - keep_gb) // per_gb)))
    if n < want:
        log(f"memory: {av:.0f} GB available -> {n} of {want} workers")
    return n


# ------------------------------------------------------------------ pool jobs (top level: picklable)

def _retarget_folder(job):
    from .retarget import SameTopologyRetargeter
    src, dst, src_skel, dst_skel, xml = job
    r = SameTopologyRetargeter(src_skel, dst_skel, xml)
    msgs = []
    n, total = r.write_folder(src, dst, log=msgs.append)
    return Path(src).name, n, total


def _retarget_clips(job):
    from .retarget import SameTopologyRetargeter
    pairs, src_skel, dst_skel, xml = job
    r = SameTopologyRetargeter(src_skel, dst_skel, xml)
    total = 0
    for src, dst in pairs:
        r.write_clip(src, dst); total += r.last["n_corrected"]
    return len(pairs), total


@dataclass
class StepResult:
    step: str
    status: str            # done | skipped | n/a | warning | failed
    note: str = ""


class BodyPipeline:
    """the steps above for one builder; run() -> [StepResult]"""
    STEPS = ("shape", "skeleton", "retarget", "amass", "torque_ratio", "aggregate", "physics", "prune", "registry")

    def __init__(self, builder: BodyBuilder, roots: BuildRoots | None = None, *, motions=None, clips=None, amass_clips=None,
                 workers=2, keep_tasks=None, keep_gb=KEEP_GB, register=False, lift=True, quiet=False):
        self.b = builder
        self.roots = roots or BuildRoots()
        if Path(builder.robots_dir).resolve() != self.roots.robots.resolve():
            raise ValueError(f"builder writes to {builder.robots_dir}, the pipeline's robots root is {self.roots.robots}")
        self.motions, self.clips, self.amass_clips = motions, list(clips or []), list(amass_clips or [])
        self.workers, self.keep_tasks, self.keep_gb, self.register, self.lift = workers, keep_tasks, keep_gb, register, lift
        self.quiet = quiet
        self.name = builder.name
        self.scaled = isinstance(builder, ScaledBody)
        self.log_path = self.roots.logs / f"{self.name}.log"
        self._cal = None

    # ------------------------------------------------------------ logging
    def log(self, msg=""):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            for line in str(msg).splitlines() or [""]:
                f.write(f"{stamp()} {line}\n")

    def say(self, msg):
        self.log(msg)
        if not self.quiet:
            print(f"[{stamp()}] {self.name}: {msg}", flush=True)

    @contextlib.contextmanager
    def captured(self):
        """the library's prints of one step go to the log"""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                yield buf
        finally:
            if buf.getvalue():
                self.log(buf.getvalue().rstrip("\n"))

    # ------------------------------------------------------------ paths of this body
    @property
    def skeleton_json(self):
        return self.b.dir / "skeleton.json"

    @property
    def retarget_dir(self):
        return self.roots.retarget_dir(self.name)

    @property
    def physics_xml(self):
        return self.roots.physics_xml(self.name)

    def motion_folders(self):
        names = sorted(p.name for p in paths.ORIGIN_MOTION.iterdir() if p.is_dir())
        if self.motions is not None:
            want = list(self.motions)
            missing = sorted(set(want) - set(names))
            assert not missing, f"no such motion folders in {paths.ORIGIN_MOTION}: {missing}"
            names = [n for n in names if n in want]
        return names

    def calibrator(self):
        if self._cal is None:
            from .torque import TorqueCalibrator
            self._cal = TorqueCalibrator(self.b.xml, self.retarget_dir, self.roots.matrix(self.name))
        return self._cal

    # ------------------------------------------------------------ which steps apply, which are done
    def applies(self, step):
        if step in ("torque_ratio", "aggregate", "physics"):
            return self.scaled
        if step == "amass":
            return bool(self.amass_clips)
        if step == "prune":
            return self.keep_tasks is not None and self.scaled
        if step == "retarget":
            return self.scaled or bool(self.clips)
        return True

    def missing_folders(self):
        """motion folders whose retarget is incomplete; with keep_tasks, once the physics XML exists only the kept tasks count
        (the others were pruned on purpose after the calibration)"""
        out = []
        folders = self.motion_folders()
        if self.keep_tasks is not None and self.physics_xml.exists():
            keep = self.kept_tasks()
            folders = [m for m in folders if m in keep]
        for m in folders:
            src = sorted(p.stem for p in (paths.ORIGIN_MOTION / m).rglob("*.npz"))
            dst = self.retarget_dir / m
            if not dst.is_dir() or any(not (dst / f"{s}.npz").exists() for s in src):
                out.append(m)
        return out

    def kept_tasks(self):
        return {l.strip() for l in Path(self.keep_tasks).read_text().splitlines() if l.strip()}

    def done(self, step):
        if step == "shape":
            return self.b.done()
        if step == "skeleton":
            return self.skeleton_json.exists()
        if step == "retarget":
            if self.scaled:
                return not self.missing_folders()
            return all((self.retarget_dir / f"{c}.npz").exists() for c in self.clips)
        if step == "amass":
            return all((self.roots.amass_dir(self.name) / f"{c}.npz").exists() for c in self.amass_clips)
        if step == "torque_ratio":
            return self.calibrator().ratio_done()
        if step == "aggregate":
            return self.calibrator().aggregate_csv().exists()
        if step == "physics":
            return self.physics_xml.exists()
        if step == "prune":
            keep = self.kept_tasks()
            return not any(d.is_dir() and d.name not in keep for d in self.retarget_dir.iterdir()) if self.retarget_dir.exists() else True
        if step == "registry":
            return False                                      # cheap: always checked
        raise KeyError(step)

    # ------------------------------------------------------------ the steps
    def step_shape(self):
        with self.captured():
            self.b.build()
        return f"{self.b.xml}"

    def step_skeleton(self):
        with self.captured():
            dump_skeleton(self.b.xml, self.skeleton_json)
        return str(self.skeleton_json)

    def step_retarget(self):
        if not self.scaled:
            return self.variant_references()
        todo = self.missing_folders()
        jobs = [(paths.ORIGIN_MOTION / m, self.retarget_dir / m, ADULT_SKELETON, self.skeleton_json, self.b.xml) for m in todo]
        n = fit_workers(min(self.workers, len(jobs)) or 1, keep_gb=self.keep_gb, log=self.say)
        clips = frames = 0
        if n > 1:
            with Pool(n) as p:
                for m, k, total in p.imap(_retarget_folder, jobs):
                    clips += k; frames += total
                    self.log(f"  {m}: {k} clips, {total} frames ground-corrected")
        else:
            for job in jobs:
                m, k, total = _retarget_folder(job)
                clips += k; frames += total
                self.log(f"  {m}: {k} clips, {total} frames ground-corrected")
        n_dir = sum(1 for p in self.retarget_dir.iterdir() if p.is_dir())
        n_npz = len(list(self.retarget_dir.rglob("*.npz")))
        return f"{len(todo)} folders ({clips} clips, {frames} frames lifted) with {n} workers; now {n_dir} folders, {n_npz} clips"

    def variant_references(self):
        """the adult-size variants' references of self.clips (their scripts' rules)"""
        from .elderly import Elderly
        from .retarget import CopyReference
        import mujoco
        b, out = self.b, []
        for clip in self.clips:
            dst = self.retarget_dir / f"{clip}.npz"
            with self.captured():
                if isinstance(b, Elderly):
                    if b.model is None:                       # shape built earlier: plan again, compile the written XML
                        b.plan(); b.model = mujoco.MjModel.from_xml_path(str(b.xml))
                    b.reference(self.roots.data, self.roots.outputs)
                elif isinstance(b, LimitJoints):
                    CopyReference().write_clip(b.reference_source(clip, self.roots.data), dst)
                elif isinstance(b, LockJoints):
                    if b.model is None:
                        b.model = mujoco.MjModel.from_xml_path(str(b.xml))
                    r = b.reference(lift=self.lift); r.write_clip(paths.REPO / paths.origin_motion(clip), dst)
                elif isinstance(b, HeavyLimb):
                    CopyReference().write_clip(paths.REPO / paths.origin_motion(clip), dst)
                else:
                    raise ValueError(f"{type(b).__name__} has no reference rule (it tracks the adult's own clips)")
            out.append(str(dst))
        return f"{len(out)} reference(s)"

    def step_amass(self):
        src = paths.AMASS_ORIGIN_MOTION; dst = self.roots.amass_dir(self.name)
        todo = [c for c in self.amass_clips if not (dst / f"{c}.npz").exists()]
        n = fit_workers(min(self.workers, len(todo)) or 1, keep_gb=self.keep_gb, log=self.say)
        chunks = [todo[i::n] for i in range(n)]
        jobs = [([(src / f"{c}.npz", dst / f"{c}.npz") for c in ch], ADULT_SKELETON, self.skeleton_json, self.b.xml) for ch in chunks if ch]
        if len(jobs) > 1:
            with Pool(len(jobs)) as p:
                res = p.map(_retarget_clips, jobs)
        else:
            res = [_retarget_clips(j) for j in jobs]
        return f"{sum(r[0] for r in res)} AMASS clips ({sum(r[1] for r in res)} frames lifted) -> {dst}"

    def step_torque_ratio(self):
        with self.captured():
            self.calibrator().ratio_matrix()
        return str(self.roots.matrix(self.name))

    def step_aggregate(self):
        with self.captured():
            self.calibrator().aggregate()
        return str(self.calibrator().aggregate_csv())

    def step_physics(self):
        from .torque import CalibrationError
        cal = self.calibrator()
        warn = ""
        with self.captured() as buf:
            kmap = cal.aggregate(write_csv=False) if cal._agg is None else cal._agg["kmap"]
            print()
            try:
                cal.write_xml(self.physics_xml, kmap)
            except CalibrationError as e:
                # the "zeta off by" check fails after the file is written (make_scaled_bodies.py tolerated exactly that)
                if "zeta off by" not in str(e) or not self.physics_xml.exists():
                    print(str(e))
                    raise
                warn = "warning: " + str(e).strip()
                print(str(e))
            cal.motion_choice()
        lines = [l.strip() for l in buf.getvalue().splitlines()
                 if any(k in l for k in ("matches the law", "tau   = C/K", "omega_n", "dt*sqrt", "mirror"))]
        for l in lines:
            self.say("  " + l)
        chk = cal.check()
        return f"{self.physics_xml} | {chk['line']}" + (f" | {warn}" if warn else "")

    def step_prune(self):
        keep, gone = self.kept_tasks(), []
        for d in sorted(self.retarget_dir.iterdir()):
            if d.is_dir() and d.name not in keep:
                shutil.rmtree(d); gone.append(d.name)
        n = len(list(self.retarget_dir.rglob("*.npz")))
        return f"removed {len(gone)} task folders not in {self.keep_tasks}; {n} clips kept"

    def step_registry(self):
        from ..bodies import BODIES
        if self.name in BODIES:
            spec = BODIES[self.name]
            return f"registered (configs/bodies.yaml: {spec.name}, kind {spec.kind})"
        entry = registry_entry(self.b, amass=bool(self.amass_clips))
        snip = self.roots.logs / f"{self.name}.bodies.yaml"
        snip.parent.mkdir(parents=True, exist_ok=True); snip.write_text(entry)
        if self.register:
            if not self.roots.is_repo:
                return f"not registered (built outside the repository); entry -> {snip}"
            register(entry)
            return f"registered: entry inserted into configs/bodies.yaml (also {snip})"
        return f"not in configs/bodies.yaml; entry -> {snip} (register=True / --register inserts it)"

    # ------------------------------------------------------------ run
    def run(self, steps=None, force=False):
        steps = list(self.STEPS if steps is None else steps)
        unknown = set(steps) - set(self.STEPS)
        assert not unknown, f"unknown steps {sorted(unknown)}; steps: {self.STEPS}"
        self.say(f"start: {type(self.b).__name__} -> {self.roots.root} (steps {', '.join(steps)})")
        results = []
        for step in self.STEPS:
            if step not in steps:
                continue
            if not self.applies(step):
                results.append(StepResult(step, "n/a")); continue
            if not force and step != "registry" and self.done(step):
                self.say(f"{step}: done before, skipped")
                results.append(StepResult(step, "skipped")); continue
            t0 = time.time()
            try:
                note = getattr(self, f"step_{step}")()
            except BaseException as e:
                self.say(f"{step}: FAILED ({type(e).__name__}: {e})")
                results.append(StepResult(step, "failed", str(e)))
                raise
            status = "warning" if "warning:" in str(note) else "done"
            self.say(f"{step}: {status} in {time.time() - t0:.1f} s: {note}")
            results.append(StepResult(step, status, note))
        self.say("finished")
        return results


# ------------------------------------------------------------------ the registry entry

def registry_entry(builder, amass=False):
    """the configs/bodies.yaml entry of a built body (full fields, no YAML anchors)"""
    n = builder.name
    if isinstance(builder, ScaledBody):
        lines = [f"  {n}:", "    kind: smpl_scaled", f"    display: {n}", f"    parent: {builder.parent}",
                 f"    shape_xml: assets/robots/{n}/robot.xml", f"    physics_xml: assets/robots_torque/{n}/robot_torque_full.xml",
                 "    smpl: true", f"    ref_task: data/{n}/retargeting_motion/{{clip}}.npz",
                 f"    ref_amass: {'data/' + n + '/amass_retargeting_motion/ACCAD/{clip}.npz' if amass else 'null'}",
                 "    s_rule: clip_ratio"]
    else:
        ref = "data/origin_motion/{clip}.npz" if isinstance(builder, HeavyLimb) else f"data/{n}/retargeting_motion/{{clip}}.npz"
        lines = [f"  {n}:", "    kind: smpl_variant", f"    display: {n}", f"    label: {n}", f"    parent: {builder.parent}",
                 f"    shape_xml: assets/robots/{n}/robot.xml", f"    physics_xml: assets/robots/{n}/robot.xml", "    smpl: true",
                 f"    ref_task: {ref}", "    ref_amass: null", "    s_rule: clip_ratio", "    s_default: 1.0"]
    return f"  # {type(builder).__name__} (scripts/build_body.py)\n" + "\n".join(lines) + "\n"


def register(entry, config=None):
    """insert `entry` into the bodies: section of configs/bodies.yaml (before the comment block above gain_variants:), after
    checking that the result still loads as a registry"""
    import yaml
    from ..bodies import CONFIG, BodyRegistry
    config = Path(config or CONFIG)
    lines = config.read_text().splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if l.startswith("gain_variants:"))
    while i > 0 and (lines[i - 1].startswith("#") or not lines[i - 1].strip()):
        i -= 1
    new = "".join(lines[:i]) + "\n" + entry + "".join(lines[i:])
    cfg = yaml.safe_load(new)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(new)
    try:
        BodyRegistry.load(f.name)
    finally:
        os.unlink(f.name)
    assert cfg["bodies"], "empty registry"
    config.write_text(new)
    return config


# ------------------------------------------------------------------ several bodies at once

def run_many(pipelines, parallel=1, steps=None, force=False, keep_gb=KEEP_GB, log=print):
    """run each pipeline in its own process, at most `parallel` at once, each launch only when available memory minus BODY_GB
    stays >= keep_gb; returns {name: exit code} (0 = every step done or skipped)"""
    import multiprocessing as mp
    if parallel <= 1:
        out = {}
        for p in pipelines:
            try:
                p.run(steps, force); out[p.name] = 0
            except BaseException as e:            # noqa: BLE001 -- reported, the next body still runs
                log(f"FAILED {p.name}: {type(e).__name__}: {e}"); out[p.name] = 1
        return out
    ctx = mp.get_context("fork")
    running, out, todo = {}, {}, list(pipelines)
    while todo or running:
        for name, proc in list(running.items()):
            if not proc.is_alive():
                proc.join(); out[name] = proc.exitcode; del running[name]
                log(f"[{stamp()}] {name}: exit {proc.exitcode}")
        if todo and len(running) < parallel:
            wait_for_memory(BODY_GB, keep_gb, log)
            p = todo.pop(0)
            proc = ctx.Process(target=_run_one, args=(p, steps, force), name=p.name)
            proc.start(); running[p.name] = proc
            continue
        time.sleep(2)
    return out


def _run_one(p, steps, force):
    try:
        p.run(steps, force)
    except BaseException as e:                      # noqa: BLE001
        print(f"FAILED {p.name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
