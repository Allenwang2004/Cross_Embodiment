#!/usr/bin/env python3
"""body_52_test.py -- LOES, LSO, LSOR and TA on the 52 task motions (outputs/child_52/clips.txt) for the payload and the limited leg
(user, 11/09: "the same as the G1", after the G1 run of scripts/g1_small_test.py --set 52): seed 0, 128 x 16 = 2,048 rollouts.
  payload    assets/robots/adult_heavyRWristHand_x80 (right hand x 80), tracks the adult's own motion (data/origin_motion)
  limit_leg  assets/robots/adult_lockRleg_bent_lim (right leg held bent by its joint ranges), tracks the bent-leg reference
             (data/adult_lockRleg_bent_lim/retargeting_motion; the 48 missing of the 52 built 11/09 exactly as
             scripts/lock_joint_body.py --user-lock builds them -- the 12 existing ones rebuild to 0.0)
Both bodies are the adult's size: root scale s = 1 (the reference's x, y over the adult's, 1.000000 on every clip), so
adult-equivalent metres are the body's own and PHC's 0.5 m is 0.5 m.
  es    LOES, scripts/single_z_search.py with the settings of scripts/kpkd_test.py (the same bodies' LOES runs) at 2,048 rollouts:
        our cost = BFM term + 1.0 heading + 0.1 position + 0.3 anchor (z0), reference-velocity start, exact obs (s read off the clip)
  lso   LSO (BFMTrack), scripts/lso_search.py with scripts/child_52_lso.py's settings (BFMTrack's values, the BFM term alone),
        --scale 1 (lso_search.py's default is the child's 0.611), reference-velocity start
  lsor  + per-step ranks, 0.05 heading + 0.005 position (as on the child, lsor_w005)
  ta    Trajectory Adaptation (BFM-Zero): --method ta --objective phc, from ER L = 8
  native  Motivo's ER L = 8 rollout of the LSO run; frozen z0: the LOES run's origin_z rollout
  commands  "<out dir>|<command>" per search, into outputs/body_52/<body>/<stem>_<method>_s0 (the two bodies interleaved, longest clips first)
  prepare   <body>/rollouts/<stem>/{z.npy, index.json}: LOES's best latent so far and its centre after 4 ... 128 generations
  evalcmds  one single_z_search.py --eval-z per body and motion
  cem   CEM (BFM-Zero), added 11/10 (user): scripts/single_z_search.py --algo cem --objective phc, BFM-Zero's tracking reward alone (as
        outputs/child_52_cem_phc), 2,048 rollouts; cemcommands, then prepare / evalcmds --algos cem (latents in rollouts/<stem>_cem)
  analyse   per rollout, q[t] against ref[t + 1], on the body itself: Fall Time (non-foot contact), PHC success (world-frame mean
            body error never above 0.5 m), MPJPE, MMPJPE, MPJAE, root drift (mean pelvis xy distance, m). Checks: settings, re-rolled
            LOES latents reproduce the search's best cost so far and its saved best rollout, native identical across a motion's
            runs, latent-sequence best re-roll = saved best, LOES's first 64 generations = the kpkd runs of the same body, motion, seed.
            Writes outputs/body_52/<body>/results.json (the layout of scripts/g1_small_test.py's) and prints the tables.
usage: uv run python scripts/body_52_test.py commands | prepare | evalcmds | analyse [--body payload|limit_leg]"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.metrics import MetricSuite, metric_body, world_positions   # noqa: E402
from crossenbodiment.paths import CHILD_52_TASK, read_clips   # noqa: E402  configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
OUT = REPO / "outputs/body_52"; CLIPS = tuple(read_clips(CHILD_52_TASK))
SEEDS = (0,); G = (4, 8, 16, 32, 64, 128); GENS = G[-1]; EVALS = 16 * GENS; KEYS = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "drift")
BODIES = dict(payload=dict(label="adult_heavyRWristHand_x80", xml="assets/robots/adult_heavyRWristHand_x80/robot.xml", ref="data/origin_motion/{c}.npz",
                           kpkd="heavy_arm_kp1_kd1"),
              limit_leg=dict(label="adult_lockRleg_bent_lim", xml="assets/robots/adult_lockRleg_bent_lim/robot.xml",
                             ref="data/adult_lockRleg_bent_lim/retargeting_motion/{c}.npz", kpkd="limit_leg_kp1_kd1"))
SEQ = dict(lso="", lsor="--rank --heading-weight 0.05 --pos-weight 0.005", ta="--method ta --objective phc")
W_SEQ = dict(lso=dict(method="lso", objective="bfm", rank=False, heading_weight=0.0, pos_weight=0.0),
             lsor=dict(method="lso", objective="bfm", rank=True, heading_weight=0.05, pos_weight=0.005),
             ta=dict(method="ta", objective="phc", rank=False, heading_weight=0.0, pos_weight=0.0))
NAMES = dict(frozen="Frozen (z0)", random="Random Search", cem="CEM (BFM-Zero)", cmaes="CMA-ES", native="Native BFM Tracking Inference", lso="LSO (BFMTrack)", lsor="LSOR (our method)",
             ta="Trajectory Adaptation (BFM-Zero)", es="LOES (our method)")
ORDER = ("frozen", "random", "cem", "cmaes", "native", "lso", "lsor", "ta", "es")   # cem added 11/10, random / cmaes 11/10 (algocommands --algos ...);
SINGLE = ("es", "random", "cem", "cmaes")                                # analyse keeps the methods whose latents were re-rolled for every motion
ALGOS = sys.argv[sys.argv.index("--algos") + 1].split(",") if "--algos" in sys.argv else ["es", "cem"]
body = None


def stem(c):
    return c.split("/")[1]


def zcmd(b, c, k="es"):
    """LOES (es) and, with our cost and LOES's settings, random search and CMA-ES (--algo random / cmaes; user, 11/10)"""
    B = BODIES[b]
    return (f"uv run python scripts/single_z_search.py --clip {c} --body {B['label']} --xml {B['xml']} --ref {B['ref'].format(c=c)} --objective bfm "
            f"--evals {EVALS} --pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy "
            f"--anchor-weight 0.3 --eval-every 0 --init-vel-scale 1" + ("" if k == "es" else f" --algo {k}"))


def lcmd(b, c, k):
    B = BODIES[b]
    return (f"uv run python scripts/lso_search.py --ref {B['ref'].format(c=c)} --body {B['label']} --xml {B['xml']} --scale 1 --n 16 --iters {GENS} "
            f"--init-vel-scale 1 --trace budgets --budgets {' '.join(map(str, G))} {SEQ[k]}").replace("  ", " ")


def ccmd(b, c):
    """CEM (BFM-Zero): BFM-Zero's tracking reward (PHC) alone (no heading, position or anchor), as outputs/child_52_cem_phc (user, 11/10)."""
    B = BODIES[b]
    return (f"uv run python scripts/single_z_search.py --clip {c} --body {B['label']} --xml {B['xml']} --ref {B['ref'].format(c=c)} --objective phc --algo cem "
            f"--pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --eval-every 0 --evals {EVALS} --init-vel-scale 1")


def cemcommands():
    order = sorted(CLIPS, key=lambda c: -len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]))
    for s in SEEDS:
        for c in order:
            for b in BODIES:
                d = f"outputs/body_52/{b}/{stem(c)}_cem_s{s}"; print(f"{d}|{ccmd(b, c)} --seed {s} --out {d}")


def algocommands():
    """--algos random,cmaes (or cem): those single-latent searches, longest clips first, the two bodies interleaved"""
    order = sorted(CLIPS, key=lambda c: -len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]))
    for s in SEEDS:
        for c in order:
            for b in BODIES:
                for k in ALGOS:
                    d = f"outputs/body_52/{b}/{stem(c)}_{k}_s{s}"; print(f"{d}|{ccmd(b, c) if k == 'cem' else zcmd(b, c, k)} --seed {s} --out {d}")


def commands():
    order = sorted(CLIPS, key=lambda c: -len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]))
    for s in SEEDS:
        for c in order:
            for b in BODIES:
                d = f"outputs/body_52/{b}/{stem(c)}_es_s{s}"; print(f"{d}|{zcmd(b, c)} --seed {s} --out {d}")
                for k in SEQ:
                    d = f"outputs/body_52/{b}/{stem(c)}_{k}_s{s}"; print(f"{d}|" + " ".join(f"{lcmd(b, c, k)} --seed {s} --out {d}".split()))


def rdir(b, c, k):
    return OUT / b / "rollouts" / (stem(c) + ("" if k == "es" else f"_{k}"))     # LOES's latents in rollouts/<stem>, CEM's in rollouts/<stem>_cem


def prepare():
    for b in BODIES:
        for c in CLIPS:
          for k in ALGOS:
            d = rdir(b, c, k); d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
            for s in SEEDS:
                r = f"{stem(c)}_{k}_s{s}"; t = np.load(OUT / b / r / "z_trace.npz"); zb, zm = t["z_best"], t["z_mean"]; assert len(zb) == GENS + 1 and len(zm) == GENS + 1, (b, r)
                idx[r] = dict(best=list(range(len(Z), len(Z) + len(G)))); Z += [zb[g] for g in G]
                idx[r]["centre"] = list(range(len(Z), len(Z) + len(G))); Z += [zm[g] for g in G]
            np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(body=b, clip=c, map=idx)))
    print("prepared", ALGOS)


def evalcmds():
    for b in BODIES:
        for c in CLIPS:
            for k in ALGOS:                      # each latent rolled out and scored with its own search's cost
                d = str(rdir(b, c, k).relative_to(REPO)); print(f"{d}|{ccmd(b, c) if k == 'cem' else zcmd(b, c, k)} --seed 0 --eval-z {d}/z.npy --eval-save-qpos --out {d}")


def world(Q):
    """world positions of every body of the module's `body` (crossenbodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def stats(q, ref, Pw, Dr):
    """[fall_s, success, mpjpe, mmpjpe, mpjae, drift] of the module's `body` (the Task page's metrics, MetricSuite.task_page):
    success threshold 0.5 m (s = 1: the payload and the limited leg are adult-sized), drift in metres"""
    return MetricSuite.task_page(body, 1.0).reference(ref, Pw, Dr).values(q)


def settings_bad(b, c, k, S_, s):
    B = BODIES[b]
    if k == "cem":
        want = dict(evals=EVALS, pairs=8, sigma=0.05, lr=0.03, sampling="independent", fitness="rank", init="reference", frame_offset=1, obs_scale="exact",
                    pos_div=1.0, pos_unit="adult", heading_weight=0.0, pos_weight=0.0, anchor=None, anchor_weight=0.0, init_vel_scale=1.0,
                    objective="phc", algo="cem", xml=B["xml"], ref=B["ref"].format(c=c), body=B["label"], seed=s)
    elif k in ("es", "random", "cmaes"):
        want = dict(evals=EVALS, pairs=8, sigma=0.05, lr=0.03, sampling="independent", fitness="rank", init="reference", frame_offset=1, obs_scale="exact",
                    pos_div=1.0, pos_unit="adult", heading_weight=1.0, pos_weight=0.1, anchor=f"data/origin_z/{c}.npy", anchor_weight=0.3, init_vel_scale=1.0,
                    objective="bfm", algo=k, xml=B["xml"], ref=B["ref"].format(c=c), body=B["label"], seed=s)
    else:
        want = dict(n=16, iters=GENS, budgets=list(G), sigma=0.0125, lr=0.00625, betas=[0.8, 0.99], gamma=0.97, beta=1.0, init_vel_scale=1.0, antithetic=False,
                    pos_unit="adult", pos_div=1.0, sample_rollouts=EVALS, ref=B["ref"].format(c=c), body=B["label"], seed=s, **W_SEQ[k])
    return [x for x, v in want.items() if S_.get(x) != v]


def clip(job):
    global body
    import eval_metrics as E
    b, c = job; B = BODIES[b]; D = OUT / b; st = stem(c)
    if body is None or body.xml != B["xml"]:
        body = metric_body(str(REPO / B["xml"]), True); body.xml = B["xml"]
    ref = np.load(REPO / B["ref"].format(c=c))["qpos"].astype(np.float64)[1:]; Pw = world(ref); _, Dr = body.part_floor(ref)
    R = {}; chk = dict(bad=[], native_diff=0.0, reroll_cost=0.0, reroll_best_qpos=0.0, seq_reroll=0.0, cem_z0=0.0)
    R["frozen"] = np.array([stats(np.load(D / f"{st}_es_s0/origin_z.npz")["qpos"], ref, Pw, Dr)])
    for k in [k for k in SINGLE if (rdir(b, c, k) / "eval_qpos.npz").exists()]:   # the single-latent searches: best and centre latents re-rolled
        d = rdir(b, c, k); ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"]
        cost = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]); Bm = np.zeros((len(SEEDS), len(G), len(KEYS))); Cm = np.zeros_like(Bm)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(D / r / "summary.json")); chk["bad"] += [(b, r, x) for x in settings_bad(b, c, k, S_, s)]
            bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(D / r / "curve.csv"))])
            chk["reroll_cost"] = max(chk["reroll_cost"], float(np.abs(cost[ix[r]["best"]] - bs[[g - 1 for g in G]]).max()))
            chk["reroll_best_qpos"] = max(chk["reroll_best_qpos"], float(np.abs(Q[ix[r]["best"][-1]] - np.load(D / r / "best.npz")["qpos"][:Q.shape[1]]).max()))
            for j in range(len(G)):
                Bm[i, j] = stats(Q[ix[r]["best"][j]], ref, Pw, Dr); Cm[i, j] = stats(Q[ix[r]["centre"][j]], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = Bm, Cm
    z0 = np.load(D / f"{st}_es_s0/origin_z.npz")["qpos"]                # every single-latent search starts from the same z0 rollout
    chk["cem_z0"] = max(float(np.abs(np.load(D / f"{st}_{k}_s0/origin_z.npz")["qpos"] - z0).max()) for k in SINGLE if f"{k}_best" in R and k != "es")
    nat = np.load(D / f"{st}_lso_s0/rollouts.npz")["q_ER8"]; R["native"] = np.array([stats(nat, ref, Pw, Dr)])
    for k in SEQ:
        Bm = np.zeros((len(SEEDS), len(G), len(KEYS))); Cm = np.zeros_like(Bm)
        for i, s in enumerate(SEEDS):
            r = f"{st}_{k}_s{s}"; S_ = json.load(open(D / r / "summary.json")); chk["bad"] += [(b, r, x) for x in settings_bad(b, c, k, S_, s)]
            X = np.load(D / r / "rollouts.npz"); chk["native_diff"] = max(chk["native_diff"], float(np.abs(X["q_ER8"] - nat).max()))
            chk["seq_reroll"] = max(chk["seq_reroll"], float(np.abs(X["q_LSO_best"] - X[f"q_best_it{GENS}"]).max()))
            for j, g in enumerate(G):
                Bm[i, j] = stats(X[f"q_best_it{g}"], ref, Pw, Dr); Cm[i, j] = stats(X[f"q_mean_it{g}"], ref, Pw, Dr)
        R[f"{k}_best"], R[f"{k}_centre"] = Bm, Cm
    kp = REPO / f"outputs/kpkd/{B['kpkd']}/{st}_s0/z_trace.npz"
    chk["loes_vs_kpkd"] = (float(np.abs(np.load(D / f"{st}_es_s0/z_trace.npz")["z_best"][:65] - np.load(kp)["z_best"]).max()) if kp.exists() else None)
    return b, st, R, chk


def analyse():
    bodies = [sys.argv[sys.argv.index("--body") + 1]] if "--body" in sys.argv else list(BODIES)
    with Pool(16) as p:
        out = p.map(clip, [(b, c) for b in bodies for c in CLIPS], chunksize=1)
    for b in bodies:
        o = [x for x in out if x[0] == b]; bad = [y for x in o for y in x[3]["bad"]]
        chk = {k: max(x[3][k] for x in o) for k in ("native_diff", "reroll_cost", "reroll_best_qpos", "seq_reroll", "cem_z0")}
        kp = [x[3]["loes_vs_kpkd"] for x in o if x[3]["loes_vs_kpkd"] is not None]; chk["loes_vs_kpkd"] = (max(kp) if kp else None, len(kp))
        print(f"\n===== {b} ({BODIES[b]['label']}) =====")
        print(f"settings not as intended: {len(bad)} {bad[:6]} | native across runs, largest qpos difference: {chk['native_diff']}")
        print(f"z0 rollout of the other single-latent searches vs LOES's, largest qpos difference: {chk['cem_z0']}")
        print(f"LOES and CEM re-rolls vs the search's best cost so far, largest difference: {chk['reroll_cost']:.3g} | best latent re-rolled vs saved best rollout: "
              f"{chk['reroll_best_qpos']:.3g} | latent-sequence best re-rolled vs saved: {chk['seq_reroll']:.3g}")
        print(f"LOES against outputs/kpkd/{BODIES[b]['kpkd']} (same motion, seed 0; {kp and len(kp)} runs), largest best-latent difference over their 64 generations: {chk['loes_vs_kpkd'][0]}")
        res = {x[1]: x[2] for x in o}; rep = dict(checks=dict(chk, bad=bad), per_motion={}, mean={})
        ORDER = [k for k in globals()["ORDER"] if k in ("frozen", "native") or all(f"{k}_best" in res[st] for st in res)]

        def get(st, k, w):
            if k in ("frozen", "native"):
                return np.repeat(res[st][k][:, None], len(G), axis=1)
            return res[st][f"{k}_{w}"]
        for w in ("best", "centre"):
            rep["mean"][w] = {k: {str(g * 16): dict(zip(KEYS, np.mean([get(st, k, w)[:, j].mean(0) for st in res], 0).tolist())) for j, g in enumerate(G)} for k in ORDER}
            rep["per_motion"][w] = {st: {k: dict(zip(KEYS, get(st, k, w)[:, -1].mean(0).tolist())) for k in ORDER} for st in res}
        rep["per_motion_1024"] = {w: {st: {k: dict(zip(KEYS, get(st, k, w)[:, G.index(64)].mean(0).tolist())) for k in ORDER} for st in res} for w in ("best", "centre")}
        (OUT / b / "results.json").write_text(json.dumps(rep, indent=1))
        for n_ in (1024, EVALS):
            for w, lab in (("best", "Best latent so far"), ("centre", "Centre latent")):
                print(f"\n{lab}, after {n_:,} rollouts: {len(res)} motions x 1 seed (frozen, native: one rollout per motion)")
                print(f"{'method':34s}{'Fall (s)':>9s}{'Success':>9s}{'MPJPE':>8s}{'MMPJPE':>8s}{'MPJAE':>8s}{'drift m':>9s}")
                for k in ORDER:
                    v = rep["mean"][w][k][str(n_)]
                    print(f"{NAMES[k]:34s}{v['fall_s']:9.3f}{100 * v['success']:8.1f}%{v['mpjpe']:8.2f}{v['mmpjpe']:8.2f}{v['mpjae']:8.2f}{v['drift']:9.2f}")
        for w in ("best", "centre"):
            for m in ("success", "mpjpe", "mmpjpe", "mpjae"):
                print(f"\n{w}, {m} by rollout budget ({', '.join(str(g * 16) for g in G)})")
                for k in ORDER:
                    print(f"  {NAMES[k]:34s}" + "".join(f"{(100 if m == 'success' else 1) * rep['mean'][w][k][str(g * 16)][m]:8.2f}" for g in G))


if __name__ == "__main__":
    A = {"commands": commands, "cemcommands": cemcommands, "algocommands": algocommands, "prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}
    A[[x for x in sys.argv[1:] if x in A][0]]()
