#!/usr/bin/env python3
"""tall_kp06_analysis.py -- the Task page's single-latent tables (Notion Baseline page, "Task motions: child") for the tall, thin body
len1.4_wid0.8 with weakened actuators (kp x 0.6, kd x 1: assets/robots_torque/len1.4_wid0.8_kp0.6_kd1.0/robot_torque_full.xml; user, 11/12):
the 52 task motions (outputs/child_52/clips.txt) x 5 seeds x 4 searches, 1,024 rollouts each, in outputs/tall_kp06_52/<stem>_<algo>_s<seed>
(commands.txt, queue run_queue2.sh):
  es       LOES (our method): our cost without the anchor (BFM term + 1.0 heading + 0.1 position; --anchor only logs anchor_dist)
  random   random search, the same cost            cmaes   CMA-ES, the same cost
  cem      CEM (BFM-Zero): --objective phc, PHC's tracking reward alone
  frozen   the z0 rollout: each es run's origin_z.npz (identical in every run of a motion; checked)
Root scale s = 1.364408905613045 (summary pos_div: the reference's x, y over the adult's).
  prepare   rollouts/<stem>/{z.npy, index.json}: for every run the best latent so far (z_trace z_best) and the centre (z_mean) after
            4, 8, 16, 32, 64 generations (64 ... 1,024 rollouts). index.json map: run -> best / centre rows of z.npy (loes_noanchor.py's
            non-child schema); rows: algo -> [start, stop), each search's rows contiguous; rollouts/<stem>_<algo>/z.npy = those rows
  evalcmds  "<dir>|<command>": rollouts/<stem>_<algo>/z.npy re-rolled with that search's own command from commands.txt (the first seed's
            line; the seeds' lines differ only in --seed, which --eval-z does not use, and --out) + --eval-z --eval-save-qpos
  analyse   per rollout, q[t] against ref[t + 1] (data/len1.4_wid0.8/retargeting_motion), on the body itself: loes_noanchor.stats (Fall
            Time = seconds with a non-foot part within 1 cm of the floor while the reference's is > 5 cm up; PHC success = world-frame
            mean body error never above 0.5 m x s; MPJPE, MMPJPE (cm), MPJAE (m/s^2); root drift, mean pelvis x, y distance in the
            body's own metres, / s for adult-equivalent) and the PHC reward = 1 - model/phc_reward.PHCCost(model, ref[0], ref[1:],
            scale=s) (as phc_reward_tables.py). Means over the seeds, then over the motions.
            Checks: every run's settings; each re-rolled best latent's cost = curve.csv best_so_far at that budget (to eval.csv's 6
            significant digits); the 1,024 re-roll = best.npz's rollout; the re-roll's z0 cost = summary origin_z cost; the z0 rollout
            identical across a motion's runs; CEM: PHC reward of its re-rolled best = 1 - best_so_far, of z0 = 1 - origin_z cost.
            -> outputs/tall_kp06_52/{results.json, README.md}; exit code 3 if a check fails (the files are still written)
usage: uv run python scripts/tall_kp06_analysis.py prepare | evalcmds | analyse [--seeds 0,1,2,3,4] [--motions <stem>,...] [--procs 8]"""
import csv, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment import paths                                         # noqa: E402
# OOP refactor (ported from the main tree, 11/12): the runs, commands.txt and the re-rolls are read under paths.outputs_root()
# ($CE_OUTPUTS_ROOT, default outputs/); prepare's re-roll index and analyse's results.json / README.md are written under
# $CE_RESULTS_ROOT (paths.write_path, default outputs/). The settings check expects summary "ref" under the tree the runs were made in
# (RUNS_REPO = the read root's parent: this repository by default, as before).
OUT = "outputs/tall_kp06_52"; D = paths.output("tall_kp06_52"); RUNS_REPO = paths.outputs_root().parent
BODY = "len1.4_wid0.8"; XML = f"assets/robots_torque/{BODY}_kp0.6_kd1.0/robot_torque_full.xml"; REF = f"data/{BODY}/retargeting_motion/{{c}}.npz"
S = 1.364408905613045; G = (4, 8, 16, 32, 64); ALGOS = ("es", "random", "cmaes", "cem")
KEYS = ("phc", "success", "mpjpe", "mmpjpe", "mpjae", "fall_s", "drift")
NAME = dict(frozen="Frozen (z0)", random="Random Search", cem="CEM (BFM-Zero)", cmaes="CMA-ES", es="LOES (our method)")
CLIPS = paths.read_clips(paths.CHILD_52_TASK)         # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)


def opt(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


SEEDS = [int(x) for x in opt("--seeds", "0,1,2,3,4").split(",")]
PROCS = int(opt("--procs", "8"))
_want = opt("--motions", None)
MOTIONS = CLIPS if _want is None else [c for c in CLIPS if c.split("/")[1] in _want.split(",") or c in _want.split(",")]
assert _want is None or len(MOTIONS) == len(_want.split(",")), ("unknown motion in --motions", _want)


def st_(c):
    return c.split("/")[1]


def run_(c, k, s):
    return f"{st_(c)}_{k}_s{s}"


def curve(r):
    return [x["best_so_far"] for x in csv.DictReader(open(D / r / "curve.csv"))]


def prepare():
    n = 0
    for c in MOTIONS:
        miss = [run_(c, k, s) for k in ALGOS for s in SEEDS if not (D / run_(c, k, s) / "summary.json").exists()]
        if miss:
            sys.exit(f"prepare: {st_(c)} has unfinished runs: {miss}")
        Z, idx, rows = [], {}, {}
        for k in ALGOS:
            a = len(Z)
            for s in SEEDS:
                r = run_(c, k, s); t = np.load(D / r / "z_trace.npz"); zb, zm = t["z_best"], t["z_mean"]
                assert len(zb) == len(zm) == G[-1] + 1 and len(curve(r)) == G[-1], (r, len(zb), len(zm))
                idx[r] = dict(best=list(range(len(Z), len(Z) + len(G)))); Z += [zb[g] for g in G]
                idx[r]["centre"] = list(range(len(Z), len(Z) + len(G))); Z += [zm[g] for g in G]
            rows[k] = [a, len(Z)]
        W = paths.write_path(OUT)
        d = W / "rollouts" / st_(c); d.mkdir(parents=True, exist_ok=True); Z = np.stack(Z); np.save(d / "z.npy", Z)
        (d / "index.json").write_text(json.dumps(dict(body=BODY, xml=XML, clip=c, seeds=SEEDS, G=list(G), map=idx, rows=rows)))
        for k, (a, b) in rows.items():
            e = W / "rollouts" / f"{st_(c)}_{k}"; e.mkdir(exist_ok=True); np.save(e / "z.npy", Z[a:b])
        n += 1
    print("prepared", n, "motions x", len(ALGOS), "searches x seeds", SEEDS)


def evalcmds():
    C = {}
    for l in open(D / "commands.txt"):
        if l.strip():
            d, cmd = l.strip().split("|", 1); C[d] = cmd
    strip = lambda x: re.sub(r" --(seed|out) \S+", "", x)
    for c in MOTIONS:
        seeds = json.load(open(D / "rollouts" / st_(c) / "index.json"))["seeds"]
        for k in ALGOS:
            cmds = [C[f"{OUT}/{run_(c, k, s)}"] for s in seeds]
            assert all(strip(x) == strip(cmds[0]) for x in cmds), (c, k, "the seeds' commands differ beyond --seed / --out")
            assert cmds[0].count(" --out ") == 1 and cmds[0].endswith(f" --out {OUT}/{run_(c, k, seeds[0])}"), cmds[0]
            e = f"{OUT}/rollouts/{st_(c)}_{k}"
            print(f"{e}|{cmds[0][:cmds[0].rindex(' --out ')]} --eval-z {e}/z.npy --eval-save-qpos --out {e}")


def want(c, k, s):
    w = dict(clip=c, body=BODY, xml=XML, ref=str(RUNS_REPO / REF.format(c=c)), evals=1024, generations=64, pairs=8, sigma=0.05, lr=0.03, obs_scale="exact",
             init_vel_scale=1.0, init="reference", init_ref=None, frame_offset=1, seed=s, fitness="rank", sampling="independent", subspace=None,
             floor_friction=None, fall_weight=0.0, bfm_zero_vel=False, pos_unit="adult", anchor_weight=0.0, algo=k)
    if k == "cem":
        w.update(objective="phc", heading_weight=0.0, pos_weight=0.0, anchor=None)
    else:
        w.update(objective="bfm", heading_weight=1.0, pos_weight=0.1, anchor=f"data/origin_z/{c}.npy")
    return w


_B = {}


def job(c):
    import mujoco
    from crossembodiment import eval_metrics as E             # scripts/eval_metrics.py re-exports this module (refactor step 8)
    import loes_noanchor as LN
    from model.phc_reward import PHCCost
    if "body" not in _B:
        _B["body"] = E.Body(XML, True)
    body = _B["body"]; st = st_(c)
    ref_all = np.load(REPO / REF.format(c=c))["qpos"].astype(np.float64); ref = ref_all[1:]
    Pw = np.empty((len(ref), body.m.nbody - 1, 3))
    for t, q in enumerate(ref):
        body.d.qpos[:] = q; mujoco.mj_kinematics(body.m, body.d); Pw[t] = body.d.xpos[1:]
    _, Dr = body.part_floor(ref); phc = PHCCost(body.m, ref_all[0], ref, scale=S)

    def f(q):                                                   # -> KEYS
        v = LN.stats(body, q, ref, Pw, Dr, S); r = 1.0 - float(phc(np.asarray(q, dtype=np.float64)[None], ref_all[0])[0])
        return [r, v[1], v[2], v[3], v[4], v[0], v[5]]
    ix = json.load(open(D / "rollouts" / st / "index.json")); assert ix["seeds"] == SEEDS, (st, ix["seeds"], SEEDS)
    Zall = np.load(D / "rollouts" / st / "z.npy")
    chk = dict(runs=0, settings=[], cost_mismatch=[], cost_max_rel=0.0, best_npz=0.0, eval_z0_cost=0.0, z0_rollout=0.0, cem_phc_vs_curve=0.0,
               cem_phc_z0=0.0, eval_rows_ok=True)
    z0q = np.load(D / run_(c, "es", SEEDS[0]) / "origin_z.npz")["qpos"]; R = {}
    for k in ALGOS:
        e = D / "rollouts" / f"{st}_{k}"; a, b = ix["rows"][k]; Q = np.load(e / "eval_qpos.npz")["qpos"]
        ev = [x["cost"] for x in csv.DictReader(open(e / "eval.csv"))]; base = json.load(open(e / "eval_base.json"))
        chk["eval_rows_ok"] &= bool(np.array_equal(np.load(e / "z.npy"), Zall[a:b]) and len(Q) == len(ev) == b - a)
        Bm, Cm = np.zeros((len(SEEDS), len(G), len(KEYS))), np.zeros((len(SEEDS), len(G), len(KEYS)))
        for i, s in enumerate(SEEDS):
            r = run_(c, k, s); Sm = json.load(open(D / r / "summary.json")); chk["runs"] += 1
            chk["settings"] += [(r, x, Sm.get(x), v) for x, v in want(c, k, s).items() if Sm.get(x) != v]
            if abs(Sm.get("pos_div", 0) - S) > 1e-12:
                chk["settings"].append((r, "pos_div", Sm.get("pos_div"), S))
            chk["eval_z0_cost"] = max(chk["eval_z0_cost"], abs(base["cost"] - Sm["origin_z"]["cost"]))
            bs = curve(r); m = ix["map"][r]; bi = [x - a for x in m["best"]]; ci = [x - a for x in m["centre"]]
            for j, g in enumerate(G):
                ref_cost = float(bs[g - 1])
                if ev[bi[j]] != f"{ref_cost:.6g}":
                    chk["cost_mismatch"].append((r, 16 * g, ev[bi[j]], bs[g - 1]))
                chk["cost_max_rel"] = max(chk["cost_max_rel"], abs(float(ev[bi[j]]) - ref_cost) / max(abs(ref_cost), 1e-12))
            bq = np.load(D / r / "best.npz")["qpos"]
            chk["best_npz"] = max(chk["best_npz"], float(np.abs(Q[bi[-1]] - bq).max()) if Q[bi[-1]].shape == bq.shape else np.inf)
            chk["z0_rollout"] = max(chk["z0_rollout"], float(np.abs(np.load(D / r / "origin_z.npz")["qpos"] - z0q).max()))
            for j in range(len(G)):
                Bm[i, j] = f(Q[bi[j]]); Cm[i, j] = f(Q[ci[j]])
            if k == "cem":                                      # CEM optimised PHC: its own numbers must come back
                chk["cem_phc_vs_curve"] = max(chk["cem_phc_vs_curve"], max(abs(Bm[i, j, 0] - (1 - float(bs[g - 1]))) for j, g in enumerate(G)))
                chk["cem_phc_z0"] = max(chk["cem_phc_z0"], abs(f(np.load(D / r / "origin_z.npz")["qpos"])[0] - (1 - Sm["origin_z"]["cost"])))
        R[k] = dict(best=Bm.tolist(), centre=Cm.tolist())
    R["frozen"] = [f(np.load(D / run_(c, "es", s) / "origin_z.npz")["qpos"]) for s in SEEDS]
    return st, R, chk


def fails(ch):
    return [n for n, bad in (("settings", ch["settings_differing"] > 0), ("re-rolled cost vs curve.csv best_so_far", ch["cost_mismatch"] > 0),
                             ("1,024 re-roll vs best.npz", ch["best_npz_max_abs"] != 0), ("re-roll z0 cost vs summary", ch["eval_z0_cost_max_abs"] != 0),
                             ("z0 rollout across runs", ch["z0_rollout_max_abs"] != 0), ("CEM PHC reward vs 1 - best_so_far", ch["cem_phc_vs_curve_max_abs"] > 1e-5),
                             ("CEM z0 PHC reward vs 1 - origin_z cost", ch["cem_phc_z0_max_abs"] > 1e-5), ("eval rows", not ch["eval_rows_ok"])) if bad]


def analyse():
    miss = [f"{st_(c)}_{k}" for c in MOTIONS for k in ALGOS if not (D / "rollouts" / f"{st_(c)}_{k}" / "eval_qpos.npz").exists()]
    if miss:
        sys.exit(f"analyse: no re-roll for {len(miss)} of {len(MOTIONS) * len(ALGOS)} (motion, search): {miss[:8]}")
    if PROCS > 1:
        from multiprocessing import Pool
        with Pool(PROCS) as p:
            out = p.map(job, MOTIONS, chunksize=1)
    else:
        out = [job(c) for c in MOTIONS]
    per = {st: R for st, R, _ in out}; C = [x[2] for x in out]
    ch = dict(motions=len(out), seeds=SEEDS, runs=sum(c["runs"] for c in C), settings_differing=sum(len(c["settings"]) for c in C),
              settings_examples=[y for c in C for y in c["settings"]][:10], cost_checked=sum(c["runs"] for c in C) * len(G),
              cost_mismatch=sum(len(c["cost_mismatch"]) for c in C), cost_mismatch_examples=[y for c in C for y in c["cost_mismatch"]][:10],
              cost_max_rel=max(c["cost_max_rel"] for c in C), best_npz_max_abs=max(c["best_npz"] for c in C),
              eval_z0_cost_max_abs=max(c["eval_z0_cost"] for c in C), z0_rollout_max_abs=max(c["z0_rollout"] for c in C),
              cem_phc_vs_curve_max_abs=max(c["cem_phc_vs_curve"] for c in C), cem_phc_z0_max_abs=max(c["cem_phc_z0"] for c in C),
              eval_rows_ok=all(c["eval_rows_ok"] for c in C))
    ch["failed"] = fails(ch)
    mean = {w: {str(16 * g): {k: dict(zip(KEYS, np.mean([np.mean(np.array(per[st][k][w])[:, j], 0) for st in per], 0).tolist())) for k in ALGOS}
                for j, g in enumerate(G)} for w in ("best", "centre")}
    frozen = dict(zip(KEYS, np.mean([np.mean(per[st]["frozen"], 0) for st in per], 0).tolist()))
    rep = dict(body=BODY, xml=XML, s=S, keys=KEYS, budgets=[16 * g for g in G], checks=ch, frozen=frozen, mean=mean,
               per_motion=per, note="per_motion[stem][algo][best|centre] = seeds x budgets x keys; per_motion[stem]['frozen'] = seeds x keys")
    W = paths.write_path(OUT); W.mkdir(parents=True, exist_ok=True)
    (W / "results.json").write_text(json.dumps(rep, indent=1)); (W / "README.md").write_text(readme(rep))
    print(json.dumps(ch, indent=1)); print("frozen", frozen); print("best 1024", mean["best"]["1024"])
    if ch["failed"]:
        print("CHECKS FAILED:", ch["failed"]); sys.exit(3)


FMT = dict(phc=("PHC Reward ↑", "{:.3f}"), success=("Success Rate ↑", None), mpjpe=("MPJPE (cm) ↓", "{:.2f}"), mmpjpe=("MMPJPE (cm) ↓", "{:.2f}"),
           mpjae=("MPJAE (m/s²) ↓", "{:.2f}"), fall_s=("Fall Time (s) ↓", "{:.3f}"), drift=("Drift (m, body) ↓", "{:.2f}"))


def fm(key, v):
    return f"{100 * v:.1f}%" if key == "success" else FMT[key][1].format(v)


def readme(rep):
    ch = rep["checks"]; n = ch["motions"]
    L = [f"# Tall, thin body with weak actuators (len1.4_wid0.8, kp x 0.6, kd x 1): single-latent methods on the task motions", "",
         f"{n} task motions x {len(ch['seeds'])} seeds ({', '.join(map(str, ch['seeds']))}); means over the seeds first, then over the motions. "
         f"Success: PHC's 0.5 m adult-equivalent = {0.5 * S:.3f} m on this body. Drift in the body's own metres (/ {S:.3f} for adult-equivalent). "
         "Frozen (z0): one rollout per motion, the same in every column.", "",
         "## Checks", "", "| check | result |", "|---|---|",
         f"| runs | {ch['runs']} |", f"| settings differing from the plan | {ch['settings_differing']} {ch['settings_examples'][:3] if ch['settings_examples'] else ''} |",
         f"| re-rolled best latent's cost = curve.csv best_so_far (6 significant digits) | {ch['cost_checked'] - ch['cost_mismatch']} of {ch['cost_checked']} "
         f"(largest relative difference {ch['cost_max_rel']:.1e}) |",
         f"| 1,024 re-roll vs best.npz rollout (max abs qpos) | {ch['best_npz_max_abs']:.1e} |",
         f"| re-roll's z0 cost vs summary origin_z cost | {ch['eval_z0_cost_max_abs']:.1e} |",
         f"| z0 rollout across a motion's runs (max abs qpos) | {ch['z0_rollout_max_abs']:.1e} |",
         f"| CEM: PHC reward of re-rolled best vs 1 - best_so_far | {ch['cem_phc_vs_curve_max_abs']:.1e} |",
         f"| CEM: PHC reward of z0 vs 1 - summary origin_z cost | {ch['cem_phc_z0_max_abs']:.1e} |",
         f"| eval rows = prepared latents | {ch['eval_rows_ok']} |", f"| failed | {', '.join(ch['failed']) or 'none'} |"]
    for w, title in (("best", "Best latent so far"), ("centre", "Centre latent")):
        M = rep["mean"][w]["1024"]
        L += ["", f"## {title}, after 1,024 rollouts", "", "| Method | " + " | ".join(FMT[k][0] for k in KEYS) + " | Rollouts ↓ |",
              "|" + "---|" * (len(KEYS) + 2), "| Frozen (z0) | " + " | ".join(fm(k, rep["frozen"][k]) for k in KEYS) + " | 0 adaptation |"]
        L += [f"| {NAME[a]} | " + " | ".join(fm(k, M[a][k]) for k in KEYS) + " | 1,024 |" for a in ("random", "cem", "cmaes", "es")]
    for w, title in (("best", "Best latent so far"), ("centre", "Centre latent")):
        bud = [str(b) for b in rep["budgets"]]
        L += ["", f"## {title}, by rollout budget", "", "| Metric | Method | " + " | ".join(f"{int(b):,}" for b in bud) + " |", "|" + "---|" * (len(bud) + 2)]
        for k in KEYS:
            L += [f"| {FMT[k][0]} | {NAME[a] if a != 'es' else 'LOES'} | " + " | ".join(fm(k, rep["mean"][w][b][a][k]) for b in bud) + " |"
                  for a in ("random", "cem", "cmaes", "es")]
    return "\n".join(L) + "\n\n## Method\n\n```\n" + __doc__.strip() + "\n```\n"


if __name__ == "__main__":
    {"prepare": prepare, "evalcmds": evalcmds, "analyse": analyse}[[x for x in sys.argv[1:] if x in ("prepare", "evalcmds", "analyse")][0]]()
