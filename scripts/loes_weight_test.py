#!/usr/bin/env python3
"""loes_weight_test.py -- would smaller heading / position weights also lower LOES's MPJPE, as x 0.05 did for LSOR
(outputs/lsor_mpjae_test)? Small sample: the 6 child task motions where LOES's MPJPE trails LSO (BFMTrack) most at 1,024 rollouts
(outputs/loes_weight_test/clips.txt), seeds 0-2, LOES as in outputs/child_52_rerun with only the two weights scaled; anchor 0.3 kept.
  w0     heading 0, position 0
  w0.05  heading 0.05, position 0.005
  w1     heading 1.0, position 0.1: the existing runs (outputs/child_52_rerun, best latent after 64 generations)
  commands  "<out dir>|<command>" per search, into outputs/loes_weight_test/<stem>_<cond>_s<seed>
  prepare   outputs/loes_weight_test/rollouts/<stem>/{z.npy, index.json}: each new run's best latent after 64 generations
  evalcmds  one --eval-z command per motion
  analyse   MPJPE, MMPJPE, MPJAE (eval_metrics.tracking_metrics), PHC success (world-frame mean body error never above 0.31 m on the
            child), root drift (mean root xy distance, body metres); checks that the rollouts reproduce each search's best cost.
            Writes outputs/loes_weight_test/results.json.
usage: uv run python scripts/loes_weight_test.py commands | prepare | evalcmds | analyse"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "outputs/loes_weight_test"; ROLL = OUT / "rollouts"; RERUN = REPO / "outputs/child_52_rerun"; S = 0.611036
XML = "assets/robots_torque/child/robot_torque_full.xml"; SEEDS = range(3)
NEW = {"w0": (0.0, 0.0), "w0.05": (0.05, 0.005)}; COND = ("w0", "w0.05", "w1")
body = None


def clips():
    return [l.strip() for l in open(OUT / "clips.txt") if l.strip()]


def base_cmd(c, h, p):
    return (f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {XML} --objective bfm --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 "
            f"--obs-scale exact --heading-weight {h} --pos-weight {p} --anchor data/origin_z/{c}.npy --anchor-weight 0.3 --eval-every 0 --init-vel-scale 1")


def commands():
    for s in SEEDS:
        for k, (h, p) in NEW.items():
            for c in clips():
                d = f"outputs/loes_weight_test/{c.split('/')[1]}_{k}_s{s}"; print(f"{d}|{base_cmd(c, h, p)} --seed {s} --out {d}")


def prepare():
    for c in clips():
        stem = c.split("/")[1]; d = ROLL / stem; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
        for k in NEW:
            for s in SEEDS:
                r = f"{stem}_{k}_s{s}"; t = np.load(OUT / r / "z_trace.npz")["z_best"]; assert len(t) == 65, r; idx[r] = len(Z); Z.append(t[64])
        np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, map=idx)))
    print("prepared")


def evalcmds():
    for c in clips():
        d = f"outputs/loes_weight_test/rollouts/{c.split('/')[1]}"; print(f"{d}|{base_cmd(c, 0.0, 0.0)} --seed 0 --eval-z {d}/z.npy --eval-save-qpos --out {d}")


def world(Q):
    m, d = body.m, body.d; P = np.empty((len(Q), m.nbody - 1, 3))
    for t, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); P[t] = d.xpos[1:]
    return P


def stats(q, ref, Pw):
    import eval_metrics as E
    q = q.astype(np.float64); T = min(len(q), len(ref)); q, r = q[:T], ref[:T]; gw = np.linalg.norm(world(q) - Pw[:T], axis=-1).mean(1)
    return dict(**E.tracking_metrics(body, q, r), success=float(not (gw > 0.5 * S).any()), drift=float(np.linalg.norm(q[:, :2] - r[:, :2], axis=1).mean()))


def clip(c):
    global body
    import eval_metrics as E
    body = body or E.Body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]; Pw = world(ref)
    d = ROLL / stem; ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"]; res = {}; worst = 0.0
    for k in NEW:
        res[k] = []
        for s in SEEDS:
            r = f"{stem}_{k}_s{s}"; res[k].append(stats(Q[ix[r]], ref, Pw))
            b = np.load(OUT / r / "best.npz")                      # the search's own rollout of its best latent
            if "qpos" in b.files:
                worst = max(worst, float(np.abs(b["qpos"][: len(Q[ix[r]])] - Q[ix[r]][: len(b["qpos"])]).max()))
    ox = json.load(open(RERUN / f"rollouts/{stem}/index.json"))["map"]; OQ = np.load(RERUN / f"rollouts/{stem}/eval_qpos.npz")["qpos"]
    res["w1"] = [stats(OQ[ox[f"{stem}_es_s{s}"]["best"][-1]], ref, Pw) for s in SEEDS]
    return stem, res, worst


def analyse():
    cl = clips()
    with Pool(len(cl)) as p:
        out = p.map(clip, cl)
    res = {o[0]: o[1] for o in out}; keys = ("mpjpe", "mmpjpe", "mpjae", "success", "drift")
    print("best rollout re-rolled vs the search's saved best rollout, largest qpos difference:", max(o[2] for o in out))
    mean = {k: {m: float(np.mean([r[m] for st in res for r in res[st][k]])) for m in keys} for k in COND}
    print(f"{len(cl)} motions x {len(SEEDS)} seeds, LOES best latent after 1,024 rollouts")
    print(f"{'weights':8s}" + "".join(f"{m:>9s}" for m in keys))
    for k in COND:
        print(f"{k:8s}" + "".join(f"{mean[k][m] * (100 if m == 'success' else 1):9.2f}" for m in keys))
    for st in res:
        print(f"  {st:30s}" + " | ".join(f"{k}: MPJPE {np.mean([r['mpjpe'] for r in res[st][k]]):5.2f} drift {np.mean([r['drift'] for r in res[st][k]]):4.2f} "
                                          f"succ {sum(r['success'] for r in res[st][k]):.0f}/3" for k in COND))
    (OUT / "results.json").write_text(json.dumps(dict(per_motion=res, mean=mean), indent=1))


if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[sys.argv[1]]()
