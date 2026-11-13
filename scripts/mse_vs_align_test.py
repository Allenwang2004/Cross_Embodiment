#!/usr/bin/env python3
"""mse_vs_align_test.py -- does searching on the new MSE cost give motions that look more like the reference
than searching on L_align?

Clips from the 540 (crossembodiment-child-torque), --per-group per motion group (seed 0; "ground" =
lieonground / sitonground / split / crouch, never searched before). Each searched twice from its own z0, same
settings except the objective: exact observations, + 0.3 (|z - z0| / 16)^2, 8 pairs (16 rollouts) per
generation, --gens generations.
  align  old L_align (model/losses.py:functional_equivalence)
  mse    new cost   (model/losses.py:tracking_mse: pose + ee + root + heading, weights fixed at 1)
Both arms' best rollouts (and z0's) are then scored on metrics neither optimises directly:
  joint_deg   mean |joint angle error|, degrees
  mpjpe_loc   mean position error of all 24 bodies relative to the pelvis, in each body's heading frame, cm
  mpjpe_glob  mean world position error of all 24 bodies, cm
  head_deg    mean |heading error|, degrees
  run      launch (round-robin over --gpus) and wait
  report   print the table; write <out>/metrics.json and the video specs <out>/videos/<clip>.json
Writes outputs/mse_vs_align/{clips.txt, align/<clip>, mse/<clip>}.
"""
import argparse, json, os, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
from crossembodiment.geometry import sphere16   # noqa: E402  16 z / |z| (refactor step 3; 16 (z / |z|) is the same bits: 16 = 2^4)
OUT = REPO / "outputs/mse_vs_align"
LOG = REPO / "outputs/train_logs/mse_vs_align"
ARMS = ("align", "mse")
XML = "assets/robots_torque/child/robot_torque_full.xml"


def group(task):
    for g in ("crawl", "headstand", "jump", "rotate", "move"):
        if task.startswith(g):
            return g
    return "raisearms" if task.startswith("raisearms") else "ground"


def pick(per_group):
    man = [json.loads(l) for l in open(REPO / "datasets/crossembodiment-child-torque/manifest.jsonl")]
    by = defaultdict(list)
    for m in man:
        by[group(m["reward_name"])].append((m["reward_name"], str(m["trial"])))
    return [(t, k, g) for g in sorted(by) for t, k in random.Random(0).sample(sorted(by[g]), per_group)]


def ref_of(t, k):
    return np.load(REPO / f"data/child/retargeting_motion/{t}/{t}_{k}.npz")["qpos"].astype(np.float64)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    clips = pick(args.per_group)
    (OUT / "clips.txt").write_text("".join(" ".join(c) + "\n" for c in clips))
    gpus, procs, t0 = args.gpus.split(","), [], time.time()
    for i, ((t, k, g), arm) in enumerate((c, a) for c in clips for a in ARMS):
        stem, out = f"{t}_{k}", OUT / arm / f"{t}_{k}"
        if (out / "summary.json").exists():
            continue
        z0 = f"data/origin_z/{t}/{stem}.npy"
        cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", f"{t}/{stem}", "--body", "child",
               "--init", "reference", "--steps", str(len(ref_of(t, k))), "--pairs", "8", "--sigma", "0.05",
               "--lr", "0.03", "--seed", "0", "--obs-scale", "exact", "--objective", arm,
               "--evals", str(args.gens * 16), "--z-start", z0, "--best-from-start",
               "--anchor", z0, "--anchor-weight", "0.3", "--out", str(out)]
        while sum(p.poll() is None for p in procs) >= args.max_par:
            time.sleep(5)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]},
                                      stdout=open(LOG / f"{arm}_{stem}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print(f"=== mse_vs_align done ({(time.time() - t0) / 60:.1f} min, {len(procs)} runs, "
          f"{sum(p.returncode != 0 for p in procs)} failed) ===", flush=True)


# metrics lives in crossembodiment/metrics.py as pose_similarity since refactor step 4 (unchanged); imported back for report()
from crossembodiment.metrics import pose_similarity as metrics   # noqa: E402,F401


def report(args):
    import mujoco
    from model import losses, kinematics as kin
    fk = mujoco.MjModel.from_xml_path(str(REPO / XML))
    W = {"root": 1.0, "ee": 1.0, "contact": 1.0, "pose": 1.0, "velocity": 1.0 / 30 ** 2}
    clips = [l.split() for l in open(OUT / "clips.txt") if l.strip()]
    rows = {}
    for t, k, g in clips:
        stem = f"{t}_{k}"
        if not all((OUT / a / stem / "summary.json").exists() for a in ARMS):
            continue
        ref = ref_of(t, k)
        traj = {"z0": np.load(OUT / "align" / stem / "origin_z.npz")["qpos"].astype(np.float64)}
        traj.update({a: np.load(OUT / a / stem / "best.npz")["qpos"].astype(np.float64) for a in ARMS})
        rows[stem] = {"group": g}
        for name, q in traj.items():
            m = metrics(fk, q, ref, losses, kin)
            m["L_align"] = float(losses.functional_equivalence(fk, q, ref, W, 1 / 30)[0])
            m["mse"] = float(losses.tracking_mse(fk, q, ref)[0])
            m["dist_z0"] = float(np.linalg.norm(np.load(OUT / name / stem / "best_z.npy").reshape(-1)
                                                - sphere16(np.load(REPO / f"data/origin_z/{t}/{stem}.npy").reshape(-1)))) if name in ARMS else 0.0
            rows[stem][name] = m
    json.dump(rows, open(OUT / "metrics.json", "w"), indent=1)
    keys = ["L_align", "mse", "joint_deg", "mpjpe_loc", "mpjpe_glob", "head_deg", "dist_z0"]
    print(f"{len(rows)} clips. Medians (L_align / mse are the two objectives; the four after them are independent):")
    print(f"{'':16s}" + "".join(f"{k:>11s}" for k in keys))
    for name in ("z0",) + ARMS:
        print(f"{name + (' search' if name in ARMS else ''):16s}" + "".join(f"{np.median([r[name][k] for r in rows.values()]):11.3f}" for k in keys))
    print("clips where the mse search beats the align search:  " + "  ".join(
        f"{k} {sum(r['mse'][k] < r['align'][k] for r in rows.values())}/{len(rows)}" for k in keys[:6]))
    print(f"\n{'clip':34s} {'group':9s} | " + " | ".join(f"{k}: z0 / align / mse" for k in ("mpjpe_glob", "mpjpe_loc", "head_deg")))
    for stem, r in rows.items():
        print(f"{stem:34s} {r['group']:9s} | " + " | ".join(
            f"{r['z0'][k]:5.1f} / {r['align'][k]:5.1f} / {r['mse'][k]:5.1f}" for k in ("mpjpe_glob", "mpjpe_loc", "head_deg")))
    vd = OUT / "videos"; vd.mkdir(exist_ok=True)
    for stem, r in rows.items():
        t = stem.rsplit("_", 1)[0]
        panel = lambda title, sub, q: {"title": title, "sub": sub, "xml": XML, "qpos": q}
        sub = lambda m: f"MPJPE {m['mpjpe_glob']:.0f} cm, heading {m['head_deg']:.0f} deg"
        spec = {"out": f"outputs/mse_vs_align/videos/{stem}.mp4", "cols": 4, "size": 300, "camera": "front_side",
                "panels": [panel("reference", "child retarget", f"data/child/retargeting_motion/{t}/{stem}.npz"),
                           panel("z0", sub(r["z0"]), f"outputs/mse_vs_align/align/{stem}/origin_z.npz"),
                           panel("old L_align search", sub(r["align"]), f"outputs/mse_vs_align/align/{stem}/best.npz"),
                           panel("new MSE search", sub(r["mse"]), f"outputs/mse_vs_align/mse/{stem}/best.npz")]}
        (vd / f"{stem}.json").write_text(json.dumps(spec, indent=1))
    print(f"\nwrote {OUT / 'metrics.json'} and {len(rows)} video specs in {vd}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--gens", type=int, default=32)
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--max-par", type=int, default=28)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
