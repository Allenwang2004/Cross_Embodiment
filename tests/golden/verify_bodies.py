#!/usr/bin/env python3
"""verify_bodies.py LABEL -- OOP refactor step 9: rebuild existing bodies with the new pipeline (crossembodiment/bodybuild) into the
worktree and compare every artifact with the stored one, byte for byte.

Everything is WRITTEN under outputs/refactor_step9/LABEL (BuildRoots(root=.../root): assets/robots, assets/robots_torque, data,
outputs below it); the stored files are READ ONLY: the worktree's assets (git, identical to the main tree's), data/ (a symlink into
the main tree) and the main tree's outputs/torque_ratio_across_motions (MAIN, read only).

  scaled    BodyPipeline(ScaledBody) of len0.8_wid1.2 (len 0.8 x wid 1.2, scripts/make_scaled_bodies.py) and the child
            (scale_robot.py --preset child --no-actuator-scale), every step: robot.xml, skeleton.json, parameter.json (the old
            keys identical, the added honest keys listed), every retarget of data/<body>/retargeting_motion (54 folders, 720
            clips), the torque ratio files (_k_matrix.csv, _r2_matrix.csv, _corr.csv, _corr.png, _ratios_aggregate.csv) and
            assets/robots_torque/<body>/robot_torque_full.xml
  amass     SameTopologyRetargeter on 3 AMASS clips (outputs/amass_baselines' list) for len1.4_wid0.8 (its stored skeleton / XML)
            -> vs data/len1.4_wid0.8/amass_retargeting_motion/ACCAD, and for the child -> vs amass_data/amass_retargeting_motion/ACCAD
  variant   HeavyLimb(80, R, Wrist + Hand) -> vs assets/robots/adult_heavyRWristHand_x80/robot.xml
  old       the same bodies built by the scripts of BASE (2bdb2fe: scale_robot, export_skeleton_json, qpos_retarget per motion
            folder, torque_ratio_across_motions, torque_aggregate_motion_k --joint-dynamics, called as make_scaled_bodies.py
            called them) in a scratch root (outputs/refactor_step9/LABEL/oldrepo) vs the new pipeline's files, byte for byte
            (needs part scaled); for a stored file the new code cannot reproduce, the compiled difference is printed
  gains     GainScaler.scale_xml for every human gain variant of configs/bodies.yaml (child_k*, child_kp*_kd*, the payload's and the
            limited leg's kp / kd variants) and GainScaler.scale_json for the G1's 6 gains json variants -> vs the stored files
Writes WORK/report.json; exit 0 = every comparison identical.
usage: CUDA_VISIBLE_DEVICES=2 uv run python tests/golden/verify_bodies.py step9 [--parts scaled,amass,variant,gains,old]
           [--bodies len0.8_wid1.2,child] [--workers 6]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
MAIN = Path(os.environ.get("CROSSENBODIMENT_MAIN", "/home/allen19/crossembodiment"))
BASE = "2bdb2fe"                                              # the commit before step 9 (the old scripts)
MATRIX_FILES = ("_k_matrix.csv", "_r2_matrix.csv", "_corr.csv", "_ratios_aggregate.csv", "_corr.png")
AMASS_CLIPS = 3


def same_bytes(a, b):
    a, b = Path(a), Path(b)
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


class Report:
    def __init__(self):
        self.rows = []

    def add(self, part, what, ok, note=""):
        self.rows.append(dict(part=part, what=what, ok=bool(ok), note=note))
        print(f"  {'PASS' if ok else 'FAIL'}  {part:8s} {what}" + (f"  ({note})" if note else ""), flush=True)

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)


def parameter_keys(new, old):
    """the old file's keys must be in the new one with equal values; returns (ok, added keys, differing keys)"""
    n, o = json.loads(Path(new).read_text()), json.loads(Path(old).read_text())
    diff = [k for k in o if k not in n or n[k] != o[k] or type(n[k]) is not type(o[k])]
    return not diff, [k for k in n if k not in o], diff


def scaled(rep, work, names, workers):
    from crossembodiment.bodybuild import BodyPipeline, BuildRoots, ScaledBody
    roots = BuildRoots(work / "root")
    for name in names:
        b = (ScaledBody.from_preset("child", robots_dir=roots.robots) if name == "child" else
             ScaledBody.from_len_wid(float(name[3:].split("_")[0]), float(name.split("wid")[1]), robots_dir=roots.robots))
        assert b.name == name, (b.name, name)
        t0 = time.time()
        res = BodyPipeline(b, roots, workers=workers, quiet=False).run()
        rep.add("scaled", f"{name}: pipeline ran ({', '.join(f'{r.step} {r.status}' for r in res)})",
                all(r.status in ("done", "skipped", "n/a") for r in res), f"{time.time() - t0:.0f} s")
        stored = REPO / "assets/robots" / name
        for f in ("robot.xml", "skeleton.json"):
            rep.add("scaled", f"{name}/{f} byte-identical", same_bytes(b.dir / f, stored / f))
        ok, added, diff = parameter_keys(b.parameter_json, stored / "parameter.json")
        rep.add("scaled", f"{name}/parameter.json: every stored key identical", ok,
                f"added {added}" + (f"; differ {diff}" if diff else ""))
        # retargets: every clip of the stored folder, byte for byte (and the same file set)
        new_dir, old_dir = roots.retarget_dir(name), REPO / "data" / name / "retargeting_motion"
        old_files = sorted(p.relative_to(old_dir) for p in old_dir.rglob("*.npz"))
        new_files = sorted(p.relative_to(new_dir) for p in new_dir.rglob("*.npz"))
        bad = [str(p) for p in old_files if not same_bytes(new_dir / p, old_dir / p)]
        folders = len({p.parts[0] for p in old_files})
        rep.add("scaled", f"{name}: {len(old_files)} retargets in {folders} motion folders byte-identical, same file set",
                not bad and new_files == old_files, f"{len(bad)} differ: {bad[:3]}" if bad else "")
        # torque ratio files vs the main tree's outputs (read only)
        stored_m = MAIN / "outputs/torque_ratio_across_motions/gravity" / name
        for f in MATRIX_FILES:
            rep.add("scaled", f"{name}: torque_ratio_across_motions/gravity/{name}/{f} byte-identical",
                    same_bytes(roots.matrix(name) / f, stored_m / f))
        rep.add("scaled", f"{name}: robots_torque/{name}/robot_torque_full.xml byte-identical",
                same_bytes(roots.physics_xml(name), REPO / "assets/robots_torque" / name / "robot_torque_full.xml"))


def amass(rep, work):
    from crossembodiment import paths
    from crossembodiment.bodybuild import SameTopologyRetargeter
    clips = paths.read_clips(paths.AMASS_LOCOMOTION_5S)[:AMASS_CLIPS]
    adult_sk = REPO / "assets/robots/adult/skeleton.json"
    for body, stored in (("len1.4_wid0.8", REPO / "data/len1.4_wid0.8/amass_retargeting_motion/ACCAD"),
                         ("child", REPO / paths.AMASS_RETARGETING_MOTION_REL)):
        r = SameTopologyRetargeter(adult_sk, REPO / f"assets/robots/{body}/skeleton.json", REPO / f"assets/robots/{body}/robot.xml")
        out = work / "amass" / body
        bad = []
        for c in clips:
            p = r.write_clip(REPO / paths.amass_origin_motion(c), out / f"{c}.npz")
            bad += [] if same_bytes(p, stored / f"{c}.npz") else [c]
        rep.add("amass", f"{body}: {len(clips)} AMASS retargets byte-identical vs {stored.relative_to(REPO)}", not bad, ", ".join(bad))


def variant(rep, work):
    from crossembodiment.bodybuild import BuildRoots, HeavyLimb
    roots = BuildRoots(work / "root")
    b = HeavyLimb(80, "R", ["Wrist", "Hand"], robots_dir=roots.robots)
    assert b.name == "adult_heavyRWristHand_x80", b.name
    b.build(log=lambda *_: None)
    stored = REPO / "assets/robots" / b.name
    rep.add("variant", f"{b.name}/robot.xml byte-identical", same_bytes(b.xml, stored / "robot.xml"))
    ok, added, diff = parameter_keys(b.parameter_json, stored / "parameter.json")
    rep.add("variant", f"{b.name}/parameter.json: every stored key (the adult's copy) identical", ok, f"added {added}")


def gains(rep, work):
    from crossembodiment.bodies import BODIES
    from crossembodiment.bodybuild import GainScaler
    out = work / "gains"
    for spec in BODIES:
        if spec.kind != "gain_variant":
            continue
        g = GainScaler.k(spec.k_scale) if spec.k_scale is not None else GainScaler(spec.kp_scale, spec.kd_scale)
        parent = BODIES[spec.parent]
        if spec.smpl:
            dst = out / spec.physics_xml
            g.scale_xml(parent.physics_xml, dst, check=True, log=lambda *_: None)
            rep.add("gains", f"{spec.physics_xml} byte-identical (GainScaler.{'k' if spec.k_scale else 'kp/kd'} on "
                    f"{parent.physics_xml})", same_bytes(dst, REPO / spec.physics_xml))
        else:
            dst = out / spec.gains
            g.scale_json(parent.gains, dst)
            rep.add("gains", f"{spec.gains} byte-identical (GainScaler.scale_json on {parent.gains})", same_bytes(dst, REPO / spec.gains))


OLD_SCRIPTS = ("scale_robot", "export_skeleton_json", "qpos_retarget", "torque_ratio_across_motions", "torque_ratio_per_joint",
               "torque_aggregate_motion_k", "torque_scale_actuators")


def old_repo(work):
    """a scratch repository root for the old scripts (BASE's source; they find their root from their own path): scripts/ from
    git, assets/robots/adult and data/origin_motion as read-only symlinks to the repository's, everything else a real folder"""
    import subprocess
    root = work / "oldrepo"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    for s in OLD_SCRIPTS:
        src = subprocess.run(["git", "show", f"{BASE}:scripts/{s}.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        (root / "scripts" / f"{s}.py").write_text(src)
    for rel in ("assets/robots/adult", "data/origin_motion"):
        link = root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(REPO / rel)
    (root / "outputs").mkdir(exist_ok=True)
    return root


def old_build(root, name, flags, log):
    """make_scaled_bodies.py's steps with BASE's scripts in `root` (python of this environment, as `uv run python` was)"""
    import subprocess
    py = [sys.executable]
    A = root / "assets/robots" / name; K = root / "outputs/torque_ratio_across_motions/gravity" / name
    cmds = [py + ["scripts/scale_robot.py", "--label", name, "--no-actuator-scale"] + flags,
            py + ["scripts/export_skeleton_json.py", "--input", str(A / "robot.xml"), "--output", str(A / "skeleton.json")]]
    for m in sorted(p.name for p in (REPO / "data/origin_motion").iterdir() if p.is_dir()):
        cmds.append(py + ["scripts/qpos_retarget.py", "--input_dir", f"data/origin_motion/{m}", "--output_dir",
                          f"data/{name}/retargeting_motion/{m}", "--source_skeleton_json", "assets/robots/adult/skeleton.json",
                          "--target_skeleton_json", str(A / "skeleton.json"), "--target_xml", str(A / "robot.xml")])
    cmds.append(py + ["scripts/torque_ratio_across_motions.py", "--origin", "data/origin_motion", "--retarget",
                      f"data/{name}/retargeting_motion", "--adult-xml", "assets/robots/adult/robot.xml", "--child-xml",
                      str(A / "robot.xml"), "--outdir", str(K)])
    out = root / "assets/robots_torque" / name / "robot_torque_full.xml"; out.parent.mkdir(parents=True, exist_ok=True)
    cmds.append(py + ["scripts/torque_aggregate_motion_k.py", "--matrix", str(K), "--src", str(A / "robot.xml"), "--out", str(out),
                      "--joint-dynamics"])
    with open(log, "w") as f:
        for c in cmds:
            f.write("$ " + " ".join(c) + "\n"); f.flush()
            r = subprocess.run(c, cwd=root, stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="1"))
            if r.returncode:
                raise RuntimeError(f"old {c[1]} failed ({r.returncode}); see {log}")
    return A, K, out


def old_vs_new(rep, work, names):
    """the same bodies built by BASE's scripts in a scratch root vs the new pipeline's (part scaled must have run): every artifact"""
    from crossembodiment.bodybuild import BuildRoots
    from crossembodiment.bodybuild.beta import BETA_AXES, PRESETS, isometric_beta
    roots, root = BuildRoots(work / "root"), old_repo(work)
    for name in names:
        beta = dict(PRESETS["child"]) if name == "child" else isometric_beta(float(name[3:].split("_")[0]), float(name.split("wid")[1]))
        flags = sum([[f"--{a.replace('_', '-')}", repr(beta[a])] for a in BETA_AXES], [])
        t0 = time.time()
        A, K, out = old_build(root, name, flags, work / f"old_{name}.log")
        rep.add("old", f"{name}: BASE {BASE}'s make_scaled_bodies steps ran", True, f"{time.time() - t0:.0f} s, log {work / f'old_{name}.log'}")
        nd = roots.robots / name
        for f in ("robot.xml", "skeleton.json"):
            rep.add("old", f"{name}/{f}: new == old", same_bytes(nd / f, A / f))
        ok, added, diff = parameter_keys(nd / "parameter.json", A / "parameter.json")
        rep.add("old", f"{name}/parameter.json: every old key identical", ok, f"added {added}")
        new_dir, old_dir = roots.retarget_dir(name), root / "data" / name / "retargeting_motion"
        files = sorted(p.relative_to(old_dir) for p in old_dir.rglob("*.npz"))
        bad = [str(p) for p in files if not same_bytes(new_dir / p, old_dir / p)]
        rep.add("old", f"{name}: {len(files)} retargets new == old", not bad and len(files) == len(list(new_dir.rglob('*.npz'))),
                f"{len(bad)} differ" if bad else "")
        for f in MATRIX_FILES:
            rep.add("old", f"{name}: {f} new == old", same_bytes(roots.matrix(name) / f, K / f))
        rep.add("old", f"{name}: robot_torque_full.xml new == old", same_bytes(roots.physics_xml(name), out))


def compiled_diff(a, b):
    """the largest relative difference of the physics arrays of two MJCFs (0.0 = the same numbers) and the arrays that differ"""
    import mujoco
    import numpy as np
    ma, mb = mujoco.MjModel.from_xml_path(str(a)), mujoco.MjModel.from_xml_path(str(b))
    worst, which = 0.0, []
    for k in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange", "actuator_gear", "dof_armature",
              "dof_damping", "jnt_stiffness", "jnt_range", "body_mass", "body_inertia", "body_pos", "geom_size", "geom_pos"):
        x, y = getattr(ma, k), getattr(mb, k)
        if x.shape != y.shape:
            return np.inf, [k]
        d = np.abs(x - y) / np.maximum(np.abs(x), 1e-300)
        d = float(np.max(np.where(x == y, 0.0, d))) if d.size else 0.0
        if d > 0:
            which.append(f"{k} {d:.1e}")
        worst = max(worst, d)
    return worst, which


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--parts", default="scaled,amass,variant,gains,old")
    ap.add_argument("--bodies", default="len0.8_wid1.2,child")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    work = REPO / "outputs/refactor_step9" / a.label
    work.mkdir(parents=True, exist_ok=True)
    rep = Report()
    parts = a.parts.split(",")
    if "gains" in parts:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            gains(rep, work)
    if "variant" in parts:
        variant(rep, work)
    if "amass" in parts:
        amass(rep, work)
    if "scaled" in parts:
        scaled(rep, work, a.bodies.split(","), a.workers)
    if "old" in parts:
        old_vs_new(rep, work, a.bodies.split(","))
    for name in a.bodies.split(","):                    # a stored physics XML that differs: how far, numerically
        new_x = work / "root" / "assets/robots_torque" / name / "robot_torque_full.xml"
        old_x = REPO / "assets/robots_torque" / name / "robot_torque_full.xml"
        if "scaled" in parts and new_x.exists() and not same_bytes(new_x, old_x):
            d, which = compiled_diff(old_x, new_x)
            print(f"  note     {name}: stored robot_torque_full.xml vs rebuilt, compiled: largest relative difference {d:.2e} "
                  f"({', '.join(which) or 'none'})")
    (work / "report.json").write_text(json.dumps(rep.rows, indent=1) + "\n")
    n = len(rep.rows); k = sum(r["ok"] for r in rep.rows)
    print(f"\n{k}/{n} identical -> {work / 'report.json'}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
