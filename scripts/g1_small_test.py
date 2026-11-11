#!/usr/bin/env python3
"""g1_small_test.py -- the Task page's methods on the Unitree G1, small sample before the 52 task motions (user, 11/09):
3 task motions (headstand_4, rotate-x--5-0.8_4: their G1 targets existed; move-ego-0-2_1, a slow walk, target made
11/09 by scripts/robot_roundtrip.py), seeds 0-2, every method with the child's Task-page settings carried over:
  single latent (scripts/robot_z_search.py, 64 x 16 rollouts, start z0, sigma 0.05, reference-velocity start, ridge fill,
                 per-joint gains)
    es      LOES: our cost = BFM term + 1.0 heading + 0.1 position + 0.3 anchor + 1.0 IK residual (the G1 cost of kpkd)
    random  random search, our cost
    cmaes   CMA-ES, our cost
    cem     CEM (BFM-Zero), BFM-Zero's tracking reward (PHC) on the G1's bodies against the G1 target, nothing else
  latent sequence (scripts/robot_lso_search.py, 64 x 16 rollouts + the logged mean, reference-velocity start)
    lso     LSO (BFMTrack): BFMTrack's values, the BFM term alone
    lsor    LSOR: + per-step ranks, 0.05 heading + 0.005 position (x 0.05 as on the child) + 1.0 IK (user, 11/09)
    ta      Trajectory Adaptation (BFM-Zero): from ER L = 8, PHC reward alone
    native  Motivo's ER L = 8 rollout every latent-sequence run makes first
  frozen z0: every single-latent run's own origin_z rollout.
  commands  "<out dir>|<command>" per search, into outputs/g1_small/<stem>_<method>_s<seed>
  prepare   outputs/g1_small/rollouts/<stem>_<algo>/{z.npy, index.json}: each single-latent run's best latent so far and its
            centre after 4, 8, 16, 32, 64 generations (64 to 1,024 rollouts)
  evalcmds  one --eval-z command per motion and single-latent method, with that method's own cost
  analyse   per rollout, against the G1 target (q[t] vs target[t + 1]): MPJPE, MMPJPE, MPJAE (eval_metrics.tracking_metrics
            on the G1), Fall Time (non-foot contact, scripts/fall_definitions.py 'contact'), PHC success (world-frame mean
            body error never above 0.5 m adult-equivalent = 0.41 m on the G1), root drift (mean pelvis xy distance, G1 m).
            Checks: settings, frozen identical across runs, native identical across runs, re-rolled latents reproduce
            the search's best cost so far, latent-sequence best re-roll = saved best. Writes outputs/g1_small/results.json.
  --set 52  the same on the 52 task motions (outputs/child_52/clips.txt; G1 targets of the 47 missing made 11/09 by
            scripts/robot_roundtrip.py), seed 0 only, 128 x 16 = 2,048 rollouts (user, 11/09: 4,096 first, cut to 2,048 after 16 min), budgets 64 to 2,048:
            outputs/g1_52/. Its LOES runs are also checked against the 1,024-rollout runs of the same motion and seed
            (g1_small, kpkd): nothing in the search depends on the total budget, so the first 64 generations must agree.
  --methods es,lso,lsor,ta  only these methods (es and lso always: frozen and native come from their runs)
usage: uv run python scripts/g1_small_test.py [--set small | 52] [--methods a,b,...] commands | prepare | evalcmds | analyse"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.paths import CHILD_52_TASK, read_clips   # noqa: E402  configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
from crossenbodiment.bodies import BODIES
from crossenbodiment.metrics import MetricSuite, metric_body, world_positions
SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "small"
CONF = {"small": dict(out="outputs/g1_small", clips=("headstand/headstand_4", "rotate-x--5-0.8/rotate-x--5-0.8_4", "move-ego-0-2/move-ego-0-2_1"),
                      seeds=(0, 1, 2), G=(4, 8, 16, 32, 64)),
        "52": dict(out="outputs/g1_52", clips=tuple(read_clips(CHILD_52_TASK)), seeds=(0,), G=(4, 8, 16, 32, 64, 128))}[SET]
OUT_REL = CONF["out"]; OUT = REPO / OUT_REL; ROLL = OUT / "rollouts"; S = BODIES.scale("g1")              # RobotMap("g1").s
CLIPS = CONF["clips"]
SEEDS = CONF["seeds"]; G = CONF["G"]; KEYS = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "drift")
GENS = G[-1]; EVALS = 16 * GENS                       # 16 rollouts per generation / iteration
GAINS = "assets/robots/g1/gains_perjoint.json"
OURS = "--heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0.3 --ik-weight 1.0"
SINGLE = dict(es=f"--algo es {OURS}", random=f"--algo random {OURS}", cmaes=f"--algo cmaes {OURS}",
              cem="--algo cem --objective phc --heading-weight 0 --pos-weight 0 --anchor-weight 0 --ik-weight 0")
SEQ = dict(lso="--heading-weight 0 --pos-weight 0 --ik-weight 0", lsor="--rank --heading-weight 0.05 --pos-weight 0.005 --ik-weight 1.0",
           ta="--method ta --objective phc --heading-weight 0 --pos-weight 0 --ik-weight 0")
W_SINGLE = dict(es=(1.0, 0.1, 0.3, 1.0, "bfm", "es"), random=(1.0, 0.1, 0.3, 1.0, "bfm", "random"), cmaes=(1.0, 0.1, 0.3, 1.0, "bfm", "cmaes"),
                cem=(0.0, 0.0, 0.0, 0.0, "phc", "cem"))
W_SEQ = dict(lso=(0.0, 0.0, 0.0, False, "bfm", "lso"), lsor=(0.05, 0.005, 1.0, True, "bfm", "lso"), ta=(0.0, 0.0, 0.0, False, "phc", "ta"))
if "--methods" in sys.argv:                              # e.g. --methods es,lso,lsor,ta (user, 11/09: the 52 with these four first)
    M = set(sys.argv[sys.argv.index("--methods") + 1].split(","))
    assert {"es", "lso"} <= M <= set(SINGLE) | set(SEQ), "frozen z0 comes from the LOES runs and native from the LSO runs"
    SINGLE = {k: v for k, v in SINGLE.items() if k in M}; SEQ = {k: v for k, v in SEQ.items() if k in M}
body = None


def zcmd(c, k):
    return (f"uv run python scripts/robot_z_search.py --robot g1 --clip {c} --fill ridge --gains {GAINS} --evals {EVALS} --pairs 8 --sigma 0.05 "
            f"--lr 0.03 --sampling independent --init-vel reference --eval-every 0 {SINGLE[k]}")


def lcmd(c, k):
    return (f"uv run python scripts/robot_lso_search.py --robot g1 --ref data/origin_motion/{c}.npz --fill ridge --gains {GAINS} --n 16 --iters {GENS} "
            f"--init-vel reference --trace budgets {SEQ[k]}" + ("" if SET == "small" else " --budgets " + " ".join(map(str, G))))


def stem(c):
    return c.split("/")[1]


def commands():
    """the 52: longest clips first (300, 150, 120 frames), every method of a motion together, so the queue ends on short runs"""
    order = CLIPS if SET == "small" else sorted(CLIPS, key=lambda c: -len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]))
    for s in SEEDS:
        for c in order:
            for k in SINGLE:
                d = f"{OUT_REL}/{stem(c)}_{k}_s{s}"; print(f"{d}|{zcmd(c, k)} --seed {s} --out {d}")
            for k in SEQ:
                d = f"{OUT_REL}/{stem(c)}_{k}_s{s}"; print(f"{d}|{lcmd(c, k)} --seed {s} --out {d}")


def prepare():
    for c in CLIPS:
        for k in SINGLE:
            d = ROLL / f"{stem(c)}_{k}"; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
            for s in SEEDS:
                r = f"{stem(c)}_{k}_s{s}"; t = np.load(OUT / r / "z_trace.npz"); b, m = t["z_best"], t["z_mean"]; assert len(b) == GENS + 1 and len(m) == GENS + 1, r
                idx[r] = dict(best=list(range(len(Z), len(Z) + len(G)))); Z += [b[g] for g in G]
                idx[r]["centre"] = list(range(len(Z), len(Z) + len(G))); Z += [m[g] for g in G]
            np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, algo=k, map=idx)))
    print("prepared")


def evalcmds():
    for c in CLIPS:
        for k in SINGLE:
            d = f"{OUT_REL}/rollouts/{stem(c)}_{k}"; print(f"{d}|{zcmd(c, k)} --seed 0 --eval-z {d}/z.npy --eval-save-qpos --out {d}")


def world(Q):
    """world positions of every body of the module's `body` (crossenbodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def stats(q, ref, Pw, Dr):
    """[fall_s, success, mpjpe, mmpjpe, mpjae, drift] of the module's `body` (the G1; MetricSuite.task_page): success threshold
    0.5 m x S (the G1's s), drift in the G1's metres"""
    return MetricSuite.task_page(body, S).reference(ref, Pw, Dr).values(q)


def settings_bad(k, S_, single):
    if single:
        h, p, an, ik, obj, algo = W_SINGLE[k]
        want = dict(evals=EVALS, pairs=8, sigma=0.05, lr=0.03, sampling="independent", init_vel="reference", frame_offset=1, fill="ridge", gains=GAINS,
                    heading_weight=h, pos_weight=p, anchor_weight=an, ik_weight=ik, objective=obj, algo=algo)
    else:
        h, p, ik, rank, obj, method = W_SEQ[k]
        want = dict(n=16, iters=GENS, budgets=list(G), sigma=0.0125, lr=0.00625, betas=[0.8, 0.99], gamma=0.97, beta=1.0, init_vel="reference", fill="ridge", gains=GAINS,
                    heading_weight=h, pos_weight=p, ik_weight=ik, rank=rank, objective=obj, method=method, best_from="samples")
    return [x for x, v in want.items() if S_.get(x) != v]


def clip(c):
    global body
    import eval_metrics as E
    body = body or metric_body(E.XML["g1"], False); st = stem(c)
    ref = np.load(REPO / f"outputs/g1/roundtrip/{st}/g1.npz")["qpos"].astype(np.float64)[1:]; Pw = world(ref); _, Dr = body.part_floor(ref)
    R = {}; chk = dict(bad=[], frozen_diff=0.0, native_diff=0.0, reroll_cost=0.0, reroll_best_qpos=0.0, seq_reroll=0.0)
    z0 = np.load(OUT / f"{st}_es_s0/origin_z.npz")["qpos"]; R["frozen"] = np.array([stats(z0, ref, Pw, Dr)])
    for k in SINGLE:
        d = ROLL / f"{st}_{k}"; ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"]
        cost = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]); B = np.zeros((len(SEEDS), len(G), len(KEYS))); C = np.zeros_like(B)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(OUT / r / "summary.json")); chk["bad"] += [(r, x) for x in settings_bad(k, S_, True)]
            chk["frozen_diff"] = max(chk["frozen_diff"], float(np.abs(np.load(OUT / r / "origin_z.npz")["qpos"] - z0).max()))
            bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(OUT / r / "curve.csv"))])
            chk["reroll_cost"] = max(chk["reroll_cost"], float(np.abs(cost[ix[r]["best"]] - bs[[g - 1 for g in G]]).max()))
            chk["reroll_best_qpos"] = max(chk["reroll_best_qpos"], float(np.abs(Q[ix[r]["best"][-1]] - np.load(OUT / r / "best.npz")["qpos"]).max()))
            for j in range(len(G)):
                B[i, j] = stats(Q[ix[r]["best"][j]], ref, Pw, Dr); C[i, j] = stats(Q[ix[r]["centre"][j]], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = B, C
    nat = np.load(OUT / f"{st}_lso_s0/rollouts.npz")["q_ER8"]; R["native"] = np.array([stats(nat, ref, Pw, Dr)])
    for k in SEQ:
        B = np.zeros((len(SEEDS), len(G), len(KEYS))); C = np.zeros_like(B)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(OUT / r / "summary.json")); chk["bad"] += [(r, x) for x in settings_bad(k, S_, False)]
            X = np.load(OUT / r / "rollouts.npz"); chk["native_diff"] = max(chk["native_diff"], float(np.abs(X["q_ER8"] - nat).max()))
            chk["seq_reroll"] = max(chk["seq_reroll"], float(np.abs(X["q_LSO_best"] - X[f"q_best_it{GENS}"]).max()))
            for j, g in enumerate(G):
                B[i, j] = stats(X[f"q_best_it{g}"], ref, Pw, Dr); C[i, j] = stats(X[f"q_mean_it{g}"], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = B, C
    return st, R, chk


NAMES = dict(frozen="Frozen (z0)", random="Random Search", cem="CEM (BFM-Zero)", cmaes="CMA-ES", native="Native BFM Tracking Inference", lso="LSO (BFMTrack)",
             lsor="LSOR (our method)", ta="Trajectory Adaptation (BFM-Zero)", es="LOES (our method)")
ORDER = tuple(k for k in ("frozen", "random", "cem", "cmaes", "native", "lso", "lsor", "ta", "es") if k in ("frozen", "native") or k in SINGLE or k in SEQ)


def analyse():
    with Pool(min(len(CLIPS), 16)) as p:
        out = p.map(clip, CLIPS)
    chk = {x: max(o[2][x] for o in out) for x in ("frozen_diff", "native_diff", "reroll_cost", "reroll_best_qpos", "seq_reroll")}
    bad = [b for o in out for b in o[2]["bad"]]
    print(f"settings not as intended: {len(bad)} {bad[:6]} | frozen across runs, largest qpos difference: {chk['frozen_diff']} | native across runs: {chk['native_diff']}")
    print(f"single-latent re-rolls vs the search's best cost so far, largest difference: {chk['reroll_cost']:.3g} | best latent re-rolled vs saved best rollout, "
          f"largest qpos difference: {chk['reroll_best_qpos']:.3g} | latent-sequence best re-rolled vs saved: {chk['seq_reroll']:.3g}")
    others = [REPO / "outputs/kpkd/g1_kp1_kd1"] + ([REPO / "outputs/g1_small"] if SET != "small" else [])
    kp = [(st, s, o) for st in (stem(c) for c in CLIPS) for s in SEEDS for o in others if (o / f"{st}_s{s}/z_trace.npz").exists() or (o / f"{st}_es_s{s}/z_trace.npz").exists()]
    def zb(o, st, s):
        f = o / f"{st}_s{s}/z_trace.npz"; return np.load(f if f.exists() else o / f"{st}_es_s{s}/z_trace.npz")["z_best"]
    kpd = max((float(np.abs(np.load(OUT / f"{st}_es_s{s}/z_trace.npz")["z_best"][:len(zb(o, st, s))] - zb(o, st, s)[:GENS + 1]).max()) for st, s, o in kp), default=float("nan"))
    print(f"LOES against the same motion and seed in {', '.join(o.name for o in others)} ({len(kp)} runs), largest best-latent difference over the generations both have: {kpd}")
    chk["loes_vs_kpkd"] = kpd
    res = {o[0]: o[1] for o in out}; rep = dict(checks=dict(chk, bad=bad), per_motion={}, mean={})

    def get(st, k, w):                                   # (seeds, budgets, keys) for searches; (1, 1, keys) for frozen / native
        if k in ("frozen", "native"):
            return np.repeat(res[st][k][:, None], len(G), axis=1)
        return res[st][f"{k}_{w}"]
    for w in ("best", "centre"):
        rep["mean"][w] = {k: {str(g * 16): dict(zip(KEYS, np.mean([get(st, k, w)[:, j].mean(0) for st in res], 0).tolist())) for j, g in enumerate(G)} for k in ORDER}
        rep["per_motion"][w] = {st: {k: dict(zip(KEYS, get(st, k, w)[:, -1].mean(0).tolist())) for k in ORDER} for st in res}
    (OUT / "results.json").write_text(json.dumps(rep, indent=1))
    shown = [b for b in dict.fromkeys((1024, EVALS)) if b // 16 in G]
    for b in shown:
        for w, lab in (("best", "Best latent so far"), ("centre", "Centre latent")):
            print(f"\n{lab}, after {b:,} rollouts: {len(CLIPS)} motions x {len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''} (frozen, native: one rollout per motion)")
            print(f"{'method':34s}{'Fall (s)':>9s}{'Success':>9s}{'MPJPE':>8s}{'MMPJPE':>8s}{'MPJAE':>8s}{'drift m':>9s}")
            for k in ORDER:
                v = rep["mean"][w][k][str(b)]
                print(f"{NAMES[k]:34s}{v['fall_s']:9.3f}{100 * v['success']:8.1f}%{v['mpjpe']:8.2f}{v['mmpjpe']:8.2f}{v['mpjae']:8.2f}{v['drift']:9.2f}")
    for w in ("best", "centre"):
        for m in ("success", "mpjpe", "mmpjpe", "mpjae"):
            print(f"\n{w}, {m} by rollout budget ({', '.join(str(g * 16) for g in G)})")
            for k in ORDER:
                print(f"  {NAMES[k]:34s}" + "".join(f"{(100 if m == 'success' else 1) * rep['mean'][w][k][str(g * 16)][m]:8.2f}" for g in G))
    n = len(SEEDS)
    print(f"\nper motion, best after {EVALS:,}: success (of {n} seed{'s' if n > 1 else ''}) | MPJPE | drift")
    for st in res:
        print(f"  {st:24s}" + " | ".join(f"{k}: {n * rep['per_motion']['best'][st][k]['success']:.0f}/{n} {rep['per_motion']['best'][st][k]['mpjpe']:.2f} "
                                         f"{rep['per_motion']['best'][st][k]['drift']:.2f}" for k in ORDER if k not in ("frozen", "native")))
        print(f"  {'':24s}frozen: success {rep['per_motion']['best'][st]['frozen']['success']:.0f} MPJPE {rep['per_motion']['best'][st]['frozen']['mpjpe']:.2f} | "
              f"native: success {rep['per_motion']['best'][st]['native']['success']:.0f} MPJPE {rep['per_motion']['best'][st]['native']['mpjpe']:.2f}")

if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[[x for x in sys.argv[1:] if x in ("commands", "prepare", "evalcmds", "analyse")][0]]()
