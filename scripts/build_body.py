#!/usr/bin/env python3
"""build_body.py -- build new bodies: one CLI over crossenbodiment/bodybuild (OOP refactor step 9; docs/new_body.md).

It replaces the four copies of the pipeline (scripts/new_body_pipeline.sh, make_grid_bodies.sh, make_morph_path.sh,
make_scaled_bodies.py, now thin wrappers that print the build_body.py line they run) and the adult-variant scripts' building
(heavy_arm_body.py, lock_joint_body.py, limit_joint_body.py, elderly_body.py keep their flags and call the same builders).

Scaled bodies (the adult's skeleton at another size; steps shape -> skeleton -> retarget every motion folder -> torque ratio ->
aggregate k -> robot_torque_full.xml -> registry check):
  scaled NAME --len L --wid W              make_scaled_bodies.py's isometric body: every length x L, every girth x L * W
  scaled NAME --preset child               scale_robot.py's preset (child, elderly), axis flags override it
  scaled NAME --leg-scale 1.15 ...         the 8 axes (--leg/arm/torso/head-scale, --leg/arm/torso/head-girth; missing ones 1.0)
  scaled-set [all | len0.8_wid1.2 ...]     the 8 bodies len{1.4,1.2,0.8,0.6}_wid{1.2,0.8} (make_scaled_bodies.py)
  grid                                     make_grid_bodies.sh's 64 leg x arm bodies gl<leg*1000>a<arm*1000>
  morph PREFIX N leg arm torso head legG armG torsoG headG [--skip0]
                                           make_morph_path.sh: <PREFIX>_t<1000 i / N> on the line from the adult to the target
Adult-size variants (steps shape -> skeleton -> reference of each --clip -> registry check; simulated on the shape XML):
  heavy-limb --scale 80 [--side R] [--parts Wrist,Hand] [--name ...]                       (heavy_arm_body.py)
  lock-joints --name adult_lockRleg --lock "R_Knee_x=10,..." [--solref 0.005] [--no-lift]  (lock_joint_body.py)
  limit-joints --lock-body adult_lockRleg [--band 1] [--solref 0.005]                      (limit_joint_body.py)
  elderly [--name adult_elderly] [--stoop 40 --slow 2 --restrict joint ...]                 (elderly_body.py; needs --clip)
  strength --name adult_s08_gear --scale 0.8 [--what gear|cap]                             (strength_rollout.py's bodies)
Gain variants (an existing body's gains scaled; one file):
  gains --src assets/robots_torque/child/robot_torque_full.xml --kp 0.8 --kd 1.2 --out ...   (scale_body_gains.py; or --m 1.2)
  gains --src assets/robots/g1/gains_perjoint.json --kp 0.8 --kd 0.8 --out ...              (the robots' gains json)

Common flags: --root DIR (write assets/, data/, outputs/ under DIR instead of the repository; inputs are always the repository's),
--steps shape,skeleton,... (default all), --force (run the selected steps even when their outputs exist), --workers N (parallel
retarget jobs per body, default 2), --parallel N (bodies at once), --motions a,b (motion folders; default all of data/origin_motion),
--clip task/stem (repeatable; variants' references), --amass-clips FILE (also retarget these AMASS clips), --keep-tasks FILE (after
the torque calibration delete the other tasks' retargets), --register (insert a new body into configs/bodies.yaml), --dry-run.
Each launch checks that >= 50 GB stay available (CLAUDE.md). Logs: <root>/outputs/body_build/<body>.log (Taiwan time).
Every step is skipped when its outputs exist, so re-running a command finishes what an interrupted one started.

usage: uv run python scripts/build_body.py scaled len0.8_wid1.2 --len 0.8 --wid 1.2
       uv run python scripts/build_body.py scaled-set all --parallel 4
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossenbodiment.bodybuild.beta import BETA_AXES, grid_beta, lerp_beta   # noqa: E402

SCALED_SET = [f"len{L}_wid{W}" for L in (1.4, 1.2, 0.8, 0.6) for W in (1.2, 0.8)]       # make_scaled_bodies.py BODIES order
GRID_LEGS = (0.5, 0.625, 0.75, 0.875, 1.125, 1.25, 1.375, 1.5)                            # make_grid_bodies.sh LEGS / ARMS
GRID_ARMS = (0.55, 0.6625, 0.775, 0.8875, 1.1, 1.2, 1.3, 1.4)


def common(p):
    p.add_argument("--root", default=None, help="write under this folder instead of the repository")
    p.add_argument("--steps", default=None, help="comma-separated subset of: shape,skeleton,retarget,amass,torque_ratio,"
                                                 "aggregate,physics,prune,registry")
    p.add_argument("--force", action="store_true", help="run the selected steps even when their outputs exist")
    p.add_argument("--workers", type=int, default=2, help="parallel retarget jobs per body")
    p.add_argument("--parallel", type=int, default=1, help="bodies built at once")
    p.add_argument("--motions", default=None, help="comma-separated motion folders (default: every folder of data/origin_motion)")
    p.add_argument("--clip", action="append", default=[], help="task/stem: a variant's reference clip (repeatable)")
    p.add_argument("--amass-clips", default=None, help="file of AMASS clip names to retarget as well")
    p.add_argument("--keep-tasks", default=None, help="file of task names: other tasks' retargets are deleted after calibration")
    p.add_argument("--register", action="store_true", help="insert a body missing from configs/bodies.yaml into it")
    p.add_argument("--keep-gb", type=float, default=50.0, help="memory to keep available after a launch (GB)")
    p.add_argument("--dry-run", action="store_true", help="print the bodies and which steps would run, build nothing")


def axis_flags(p):
    for axis in BETA_AXES:
        p.add_argument(f"--{axis.replace('_', '-')}", type=float, default=None)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="recipe", required=True)
    s = sub.add_parser("scaled", help="one scaled body"); s.add_argument("name"); common(s); axis_flags(s)
    s.add_argument("--len", type=float, default=None); s.add_argument("--wid", type=float, default=None)
    s.add_argument("--preset", default=None, choices=["adult", "child", "elderly"])
    s.add_argument("--scale-actuators", action="store_true",
                   help="scale_robot.py's old load heuristic on the actuators (not for the torque pipeline: it asserts adult actuators)")
    s = sub.add_parser("scaled-set", help="make_scaled_bodies.py's 8 bodies"); s.add_argument("names", nargs="*", default=["all"]); common(s)
    s = sub.add_parser("grid", help="make_grid_bodies.sh's leg x arm grid"); common(s)
    s = sub.add_parser("morph", help="make_morph_path.sh"); s.add_argument("prefix"); s.add_argument("n", type=int)
    s.add_argument("target", nargs=8); s.add_argument("--skip0", action="store_true"); common(s)
    s = sub.add_parser("heavy-limb"); common(s); s.add_argument("--scale", type=float, required=True)
    s.add_argument("--side", default="R", choices=["L", "R"]); s.add_argument("--parts", default="Shoulder,Elbow,Wrist,Hand")
    s.add_argument("--name", default=None)
    s = sub.add_parser("lock-joints"); common(s); s.add_argument("--name", required=True); s.add_argument("--lock", required=True)
    s.add_argument("--solref", type=float, default=None); s.add_argument("--no-lift", action="store_true")
    s = sub.add_parser("limit-joints"); common(s); s.add_argument("--lock-body", required=True)
    s.add_argument("--band", type=float, default=1.0); s.add_argument("--solref", type=float, default=0.005); s.add_argument("--name", default=None)
    s = sub.add_parser("elderly"); common(s); s.add_argument("--name", default="adult_elderly")
    s.add_argument("--stoop", type=float, default=40.0); s.add_argument("--slow", type=float, default=2.0)
    s.add_argument("--solref", type=float, default=0.005); s.add_argument("--spine-band", type=float, default=5.0)
    s.add_argument("--limit-hip-extension", action="store_true"); s.add_argument("--restrict", default="joint", choices=["joint", "action", "both"])
    s = sub.add_parser("strength"); common(s); s.add_argument("--name", required=True); s.add_argument("--scale", type=float, required=True)
    s.add_argument("--what", default="gear", choices=["gear", "cap"])
    s = sub.add_parser("gains", help="scale an existing body's gains (MJCF or robot gains json)")
    s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s.add_argument("--m", type=float, default=None); s.add_argument("--kp", type=float, default=None); s.add_argument("--kd", type=float, default=None)
    return ap


def builders(a, roots):
    """the builders a recipe asks for, writing into roots.robots"""
    from crossenbodiment.bodybuild import (ActuatorStrength, Elderly, HeavyLimb, LimitJoints, LockJoints, ScaledBody)
    kw = dict(robots_dir=roots.robots)
    r = a.recipe
    if r == "scaled":
        axes = {x: getattr(a, x) for x in BETA_AXES if getattr(a, x) is not None}
        if a.len is not None or a.wid is not None:
            assert a.len is not None and a.wid is not None and not axes and not a.preset, "--len with --wid, and nothing else"
            return [ScaledBody.from_len_wid(a.len, a.wid, name=a.name, scale_actuators=a.scale_actuators, **kw)]
        if a.preset:
            return [ScaledBody.from_preset(a.preset, name=a.name, overrides=axes, scale_actuators=a.scale_actuators, **kw)]
        return [ScaledBody(a.name, axes, scale_actuators=a.scale_actuators, edit=dict(beta="axes"), **kw)]
    if r == "scaled-set":
        names = SCALED_SET if a.names in ([], ["all"]) else a.names
        bad = sorted(set(names) - set(SCALED_SET)); assert not bad, f"not in the set: {bad} ({SCALED_SET})"
        return [ScaledBody.from_len_wid(float(n[3:].split("_")[0]), float(n.split("wid")[1]), name=n, **kw) for n in names]
    if r == "grid":
        return [ScaledBody(f"gl{round(leg * 1000):04d}a{round(arm * 1000):04d}", grid_beta(leg, arm), edit=dict(grid=[leg, arm]), **kw)
                for leg in GRID_LEGS for arm in GRID_ARMS]
    if r == "morph":
        out = []
        for i in range(1 if a.skip0 else 0, a.n + 1):
            t = i / a.n
            out.append(ScaledBody(f"{a.prefix}_t{i * 1000 // a.n:03d}", lerp_beta(a.target, t),
                                  edit=dict(morph=a.prefix, t=t, target=[float(v) for v in a.target]), **kw))
        return out
    if r == "heavy-limb":
        return [HeavyLimb(a.scale, a.side, a.parts.split(","), name=a.name, **kw)]
    if r == "lock-joints":
        locks = [(kv.split("=")[0].strip(), float(kv.split("=")[1])) for kv in a.lock.split(",")]
        return [LockJoints(a.name, locks, solref=a.solref, **kw)]
    if r == "limit-joints":
        return [LimitJoints(a.lock_body, band=a.band, solref=a.solref, name=a.name, **kw)]
    if r == "elderly":
        assert len(a.clip) == 1, "elderly: one --clip (the walk it is planned on)"
        return [Elderly(a.clip[0], name=a.name, stoop=a.stoop, slow=a.slow, solref=a.solref, spine_band=a.spine_band,
                        limit_hip_extension=a.limit_hip_extension, restrict=a.restrict, **kw)]
    if r == "strength":
        return [ActuatorStrength(a.name, a.scale, a.what, **kw)]
    raise ValueError(r)


def gains(a):
    from crossenbodiment.bodybuild import GainScaler
    if (a.m is None) == (a.kp is None or a.kd is None):
        raise SystemExit("give --m, or --kp and --kd")
    g = GainScaler.k(a.m) if a.m is not None else GainScaler(a.kp, a.kd)
    out = g.scale_json(a.src, a.out) if a.src.endswith(".json") else g.scale_xml(a.src, a.out)
    print(f"-> {out}")
    return 0


def main(argv=None):
    a = parser().parse_args(argv)
    if a.recipe == "gains":
        return gains(a)
    from crossenbodiment.bodybuild import BodyPipeline, BuildRoots, run_many
    roots = BuildRoots(Path(a.root).resolve()) if a.root else BuildRoots()
    steps = a.steps.split(",") if a.steps else None
    amass = None
    if a.amass_clips:
        amass = [l.strip() for l in (REPO / a.amass_clips).read_text().splitlines() if l.strip()]
    motions = a.motions.split(",") if a.motions else None
    keep = str(REPO / a.keep_tasks) if a.keep_tasks else None
    pipes = [BodyPipeline(b, roots, motions=motions, clips=a.clip, amass_clips=amass, workers=a.workers, keep_tasks=keep,
                          keep_gb=a.keep_gb, register=a.register, lift=not getattr(a, "no_lift", False), quiet=a.parallel > 1)
             for b in builders(a, roots)]
    if a.dry_run:
        for p in pipes:
            todo = [s for s in p.STEPS if (steps is None or s in steps) and p.applies(s) and (a.force or s == "registry" or not p.done(s))]
            print(f"{p.name}: {type(p.b).__name__} -> {roots.root}; would run: {', '.join(todo) or 'nothing'}")
        return 0
    codes = run_many(pipes, parallel=a.parallel, steps=steps, force=a.force, keep_gb=a.keep_gb)
    bad = [n for n, c in codes.items() if c]
    for p in pipes:
        if p.scaled and p.calibrator().aggregate_csv().exists():
            print(f"{p.name}: {p.calibrator().check()['line']}")
    print(f"{len(codes) - len(bad)}/{len(codes)} bodies built" + (f"; FAILED: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
