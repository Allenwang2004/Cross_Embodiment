#!/usr/bin/env python3
"""phc_reward_tables.py -- PHC tracking reward of every rollout behind the Notion Baseline page's result tables (user, 11/12: PHC
reward replaces Fall Time for every baseline). PHC reward = 1 - model/phc_reward.PHCCost(model, ref[0], ref[1:], scale=s)(q[None], ref[0]):
the mean over steps of PHC's per-step reward (0-1, higher is better), the rollout's state after step t against reference frame t + 1,
world frame, lengths / the body's root scale s -- the same number CEM (BFM-Zero) and Trajectory Adaptation optimise
(single_z_search.py / lso_search.py --objective phc) and plot_child_improvement.py / plot_child_convergence.py / cost_correlation.py report.
  child      assets/robots_torque/child/robot_torque_full.xml, data/child/retargeting_motion, s = 0.611036
  payload    assets/robots/adult_heavyRWristHand_x80/robot.xml, data/origin_motion, s = 1
  limit_leg  assets/robots/adult_lockRleg_bent_lim/robot.xml, data/adult_lockRleg_bent_lim/retargeting_motion, s = 1
  g1         robot_z_search.py --objective phc: the G1's own bodies (assets/robots/g1/g1.xml) against the G1 target
             (outputs/g1/roundtrip/<stem>/g1.npz), s = RobotMap('g1').s = 0.8213, the reference frame before the first goal = target
             frame 0, the rollout's previous pose = its reset pose (the clip's frame 0 mapped onto the G1, = target frame 0; checked)
  amass_child / amass_tall   the AMASS runs (outputs/amass_baselines), s = 0.611036 / the clip's root scale (1.364)
Rollouts (the sources of scripts/fall_definitions.py, success_definitions.py, child_52_lso.py, body_52_test.py, g1_small_test.py --set 52,
loes_noanchor.py, amass_baselines.py):
  frozen      the z0 rollout (child: child_52_rerun/<stem>_es_s0/origin_z.npz; bodies: body_52/<b>/<stem>_es_s0; G1: g1_52/<stem>_es_s0)
  es (LOES)   outputs/noanchor/<b>/rollouts (no anchor term); random, cmaes: child_52_rerun / body_52/<b> / g1_52 rollouts;
  cem         child_52_cem_phc / body_52/<b>/rollouts/<stem>_cem / g1_52/rollouts/<stem>_cem
  native      q_ER8 of the LSO run; lso, lsor, ta: q_best_it<k>, q_mean_it<k> (child: lsor = lsor_w005; lsor_ours = full weights x 1)
Means over motions (child: over the 5 seeds first), at every budget of the by-budget tables; best and centre (mean sequence).
Checks: every CEM run (PHC objective) -- the PHC reward of its re-rolled best latent at every generation (child) / budget = 1 - curve.csv
best_so_far; its z0 rollout = 1 - summary origin_z cost; the frozen rollout used = that z0; G1 reset pose = target frame 0; the
Trajectory Adaptation runs (PHC objective) -- 1 - best_cost_at[k] = PHC reward of q_best_it<k>.
-> outputs/phc_reward_tables/{results.json (body -> best|centre -> rollouts -> method -> value), per_motion.json, checks.json, README.md}
usage: uv run python scripts/phc_reward_tables.py"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "outputs/phc_reward_tables"; O = REPO / "outputs"
CLIPS = [l.strip() for l in open(O / "child_52/clips.txt")]; AMASS = [l.strip() for l in open(O / "amass_baselines/clips.txt")]
G5, G6 = (4, 8, 16, 32, 64), (4, 8, 16, 32, 64, 128)
BODY = dict(
    child=dict(xml="assets/robots_torque/child/robot_torque_full.xml", ref="data/child/retargeting_motion/{c}.npz", s=0.611036, seeds=range(5), G=G5,
               single=dict(es=O / "noanchor/child/rollouts/{st}", random=O / "child_52_rerun/rollouts/{st}", cmaes=O / "child_52_rerun/rollouts/{st}",
                           cem=O / "child_52_cem_phc/rollouts/{st}"),
               runs=dict(es=O / "noanchor/child", random=O / "child_52_rerun", cmaes=O / "child_52_rerun", cem=O / "child_52_cem_phc"),
               frozen=O / "child_52_rerun/{st}_es_s0/origin_z.npz", seq=O / "child_52_lso/{st}_{k}_s{s}/rollouts.npz",
               seqk=dict(lso="lso", lsor="lsor_w005", ta="ta", lsor_ours="lsor_ours")),
    payload=dict(xml="assets/robots/adult_heavyRWristHand_x80/robot.xml", ref="data/origin_motion/{c}.npz", s=1.0, seeds=(0,), G=G6,
                 single=dict(es=O / "noanchor/payload/rollouts/{st}", random=O / "body_52/payload/rollouts/{st}_random",
                             cmaes=O / "body_52/payload/rollouts/{st}_cmaes", cem=O / "body_52/payload/rollouts/{st}_cem"),
                 runs=dict(es=O / "noanchor/payload", random=O / "body_52/payload", cmaes=O / "body_52/payload", cem=O / "body_52/payload"),
                 frozen=O / "body_52/payload/{st}_es_s0/origin_z.npz", seq=O / "body_52/payload/{st}_{k}_s{s}/rollouts.npz",
                 seqk=dict(lso="lso", lsor="lsor", ta="ta")),
    limit_leg=dict(xml="assets/robots/adult_lockRleg_bent_lim/robot.xml", ref="data/adult_lockRleg_bent_lim/retargeting_motion/{c}.npz", s=1.0, seeds=(0,), G=G6,
                   single=dict(es=O / "noanchor/limit_leg/rollouts/{st}", random=O / "body_52/limit_leg/rollouts/{st}_random",
                               cmaes=O / "body_52/limit_leg/rollouts/{st}_cmaes", cem=O / "body_52/limit_leg/rollouts/{st}_cem"),
                   runs=dict(es=O / "noanchor/limit_leg", random=O / "body_52/limit_leg", cmaes=O / "body_52/limit_leg", cem=O / "body_52/limit_leg"),
                   frozen=O / "body_52/limit_leg/{st}_es_s0/origin_z.npz", seq=O / "body_52/limit_leg/{st}_{k}_s{s}/rollouts.npz",
                   seqk=dict(lso="lso", lsor="lsor", ta="ta")),
    g1=dict(xml="assets/robots/g1/g1.xml", ref="outputs/g1/roundtrip/{st}/g1.npz", s=None, seeds=(0,), G=G6,
            single=dict(es=O / "noanchor/g1/rollouts/{st}", random=O / "g1_52/rollouts/{st}_random", cmaes=O / "g1_52/rollouts/{st}_cmaes",
                        cem=O / "g1_52/rollouts/{st}_cem"),
            runs=dict(es=O / "noanchor/g1", random=O / "g1_52", cmaes=O / "g1_52", cem=O / "g1_52"),
            frozen=O / "g1_52/{st}_es_s0/origin_z.npz", seq=O / "g1_52/{st}_{k}_s{s}/rollouts.npz", seqk=dict(lso="lso", lsor="lsor", ta="ta")),
    amass_child=dict(xml="assets/robots_torque/child/robot_torque_full.xml", ref="amass_data/amass_retargeting_motion/ACCAD/{c}.npz", s=0.611036,
                     seeds=(0,), G=G5, single={}, frozen=None, seq=O / "amass_baselines/child/{st}_{k}_s{s}/rollouts.npz", seqk=dict(lso="lso", lsor="lsor", ta="ta")),
    amass_tall=dict(xml="assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml", ref="data/len1.4_wid0.8/amass_retargeting_motion/ACCAD/{c}.npz", s="clip",
                    seeds=(0,), G=G5, single={}, frozen=None, seq=O / "amass_baselines/tall/{st}_{k}_s{s}/rollouts.npz", seqk=dict(lso="lso", lsor="lsor", ta="ta")))
_M = {}


def model(xml):
    if xml not in _M:
        _M[xml] = mujoco.MjModel.from_xml_path(str(REPO / xml))
    return _M[xml]


def reward_fn(xml, ref, s):
    """the PHC reward of a rollout q (T, nq) against ref (frames 0..; q[t] against ref[t + 1]) with root scale s:
    1 - PHCCost(model, ref[0], ref[1:], scale=s)(q[None], ref[0]) (also used by amass_baselines.py --set bfm)"""
    from model.phc_reward import PHCCost
    phc = PHCCost(model(xml), ref[0], ref[1:], scale=s)
    return lambda q: 1.0 - float(phc(np.asarray(q, dtype=np.float64)[None], ref[0])[0])


def curve(run):
    return np.array([float(r["best_so_far"]) for r in csv.DictReader(open(run / "curve.csv"))])


def job(arg):
    b, c = arg; B = BODY[b]; st = c.split("/")[1] if b in ("child", "payload", "limit_leg", "g1") else c; G = B["G"]; S = list(B["seeds"])
    chk = dict(cem_curve=[], cem_eval=[], cem_z0=[], frozen_vs_cem_z0=None, ta_budget=[], g1_reset=None)
    if b == "g1":
        from model.robot_map import RobotMap
        if "rm" not in _M:
            _M["rm"] = RobotMap("g1", fill="ridge")
        rm = _M["rm"]; s = rm.s; adult = np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"].astype(np.float64)
        ref = np.load(REPO / B["ref"].format(st=st))["qpos"].astype(np.float64)[:len(adult)]
        chk["g1_reset"] = float(np.abs(rm.smpl_to_robot_qpos(adult[0])[0] - ref[0]).max())     # the rollouts' reset pose vs target frame 0
    else:
        ref = np.load(REPO / B["ref"].format(c=c))["qpos"].astype(np.float64)
        if B["s"] == "clip":                                         # amass_baselines.scale: median ref / adult root x, y
            from amass_baselines import scale
            s = scale("tall", c)
        else:
            s = B["s"]
    rew = reward_fn(B["xml"], ref, s)
    R = {}
    if B["frozen"] is not None:
        R["frozen"] = rew(np.load(str(B["frozen"]).format(st=st))["qpos"])
    for k, d0 in B["single"].items():
        d = Path(str(d0).format(st=st)); ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"]
        ev = np.array([float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))])
        best, cen = np.zeros((len(S), len(G))), np.zeros((len(S), len(G))); cache = {}
        f = lambda i: cache.setdefault(i, rew(Q[i]))
        for a, sd in enumerate(S):
            r = f"{st}_{k}_s{sd}"; m = ix[r]; per_gen = len(m["best"]) == G[-1]          # child: the best latent after every generation
            bi = [m["best"][g - 1] for g in G] if per_gen else m["best"]
            for j in range(len(G)):
                best[a, j] = f(bi[j]); cen[a, j] = f(m["centre"][j])
            if k == "cem":                                           # CEM optimised PHC: its own numbers must come back
                run = B["runs"][k] / r; bs = curve(run); gens = range(1, G[-1] + 1) if per_gen else G
                idx = [m["best"][g - 1] for g in gens] if per_gen else m["best"]
                chk["cem_curve"].append(max(abs(f(i) - (1 - bs[g - 1])) for i, g in zip(idx, gens)))
                chk["cem_eval"].append(max(abs(f(i) - (1 - ev[i])) for i in idx))
                Sm = json.load(open(run / "summary.json")); assert Sm["objective"] == "phc" and abs(Sm["pos_div"] - s) < 1e-9 if "pos_div" in Sm else True, (r, Sm.get("pos_div"), s)
                chk["cem_z0"].append(abs(rew(np.load(run / "origin_z.npz")["qpos"]) - (1 - Sm["origin_z"]["cost"])))
                if sd == S[0] and "frozen" in R:
                    chk["frozen_vs_cem_z0"] = abs(R["frozen"] - (1 - Sm["origin_z"]["cost"]))
        R[k] = dict(best=best.tolist(), centre=cen.tolist())
    nat = None
    for k, kd in B["seqk"].items():
        best, cen = np.zeros((len(S), len(G))), np.zeros((len(S), len(G)))
        for a, sd in enumerate(S):
            p = Path(str(B["seq"]).format(st=st, k=kd, s=sd)); X = np.load(p)
            if nat is None and kd == "lso":
                nat = rew(X["q_ER8"])
            for j, g in enumerate(G):
                best[a, j] = rew(X[f"q_best_it{g}"]); cen[a, j] = rew(X[f"q_mean_it{g}"])
            if k == "ta":                                            # Trajectory Adaptation optimised PHC too
                Sm = json.load(open(p.parent / "summary.json")); assert Sm["objective"] == "phc"
                chk["ta_budget"].append(max(abs(best[a, j] - (1 - Sm["best_cost_at"][str(g)])) for j, g in enumerate(G)))
        R[k] = dict(best=best.tolist(), centre=cen.tolist())
    R["native"] = nat
    return b, st, R, chk, s


def main():
    jobs = [(b, c) for b in ("child", "payload", "limit_leg", "g1") for c in CLIPS] + [(b, c) for b in ("amass_child", "amass_tall") for c in AMASS]
    with Pool(16) as p:
        out = p.map(job, jobs, chunksize=1)
    res, per, checks = {}, {}, {}
    for b, B in BODY.items():
        o = [x for x in out if x[0] == b]; G = B["G"]; bud = [str(16 * g) for g in G]
        mk = [k for k in ("frozen", "random", "cem", "cmaes", "native", "lso", "lsor", "ta", "es", "lsor_ours") if k in o[0][2]]
        res[b] = {w: {n: {} for n in bud} for w in ("best", "centre")}; per[b] = {w: {n: {} for n in bud} for w in ("best", "centre")}
        for k in mk:
            for w in ("best", "centre"):
                for j, n in enumerate(bud):
                    pm = {x[1]: (x[2][k] if k in ("frozen", "native") else float(np.mean(np.array(x[2][k][w])[:, j]))) for x in o}   # seeds first
                    per[b][w][n][k] = pm; res[b][w][n][k] = float(np.mean(list(pm.values())))
        C = [x[3] for x in o]; mx = lambda key: max((v for c in C for v in c[key]), default=None)
        checks[b] = dict(motions=len(o), scale=sorted({round(x[4], 9) for x in o}), cem_runs=sum(len(c["cem_curve"]) for c in C),
                         cem_reward_vs_curve_best_so_far=mx("cem_curve"), cem_reward_vs_eval_csv=mx("cem_eval"), cem_z0_vs_summary_origin_z=mx("cem_z0"),
                         frozen_vs_cem_s0_origin_z=max((c["frozen_vs_cem_z0"] for c in C if c["frozen_vs_cem_z0"] is not None), default=None),
                         ta_runs=sum(len(c["ta_budget"]) for c in C), ta_reward_vs_best_cost_at=mx("ta_budget"),
                         g1_reset_vs_target_frame0=max((c["g1_reset"] for c in C if c["g1_reset"] is not None), default=None))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1)); (OUT / "per_motion.json").write_text(json.dumps(per))
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))
    print(readme())


NAME = dict(frozen="Frozen (z0)", random="Random Search", cem="CEM (BFM-Zero)", cmaes="CMA-ES", native="Native BFM Tracking Inference", lso="LSO (BFMTrack)",
            lsor="LSOR (our method)", ta="Trajectory Adaptation (BFM-Zero)", es="LOES (our method)", lsor_ours="LSOR, full weights (x 1; Method details only)")
LABEL = dict(child="Child, 52 task motions x 5 seeds", payload="Payload, 52 task motions, seed 0", limit_leg="Limited leg, 52 task motions, seed 0",
             g1="Unitree G1, 52 task motions, seed 0", amass_child="AMASS (68 ACCAD clips), child, seed 0", amass_tall="AMASS (68 ACCAD clips), tall body len1.4_wid0.8, seed 0")


def readme():
    res = json.load(open(OUT / "results.json")); chk = json.load(open(OUT / "checks.json"))
    L = ["# PHC reward for the Notion Baseline page's tables", "", "```", __doc__.strip(), "```", "", "## Checks", "",
         "Largest absolute difference (PHC reward; the rollouts are stored as float32, so agreement is to about 1e-8, not to the 16 digits curve.csv prints):", "",
         "| body | motions | s | CEM runs | CEM best (every generation on the child, every budget elsewhere) vs 1 - curve.csv best_so_far | CEM z0 rollout vs 1 - summary origin_z cost | "
         "frozen rollout used vs CEM s0 origin_z | TA runs | TA q_best_it<k> vs 1 - best_cost_at | G1 reset pose vs target frame 0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda v: "-" if v is None else f"{v:.1e}"
    for b, c in chk.items():
        L.append(f"| {b} | {c['motions']} | {', '.join(f'{x:g}' for x in c['scale'])} | {c['cem_runs']} | {g(c['cem_reward_vs_curve_best_so_far'])} | "
                 f"{g(c['cem_z0_vs_summary_origin_z'])} | {g(c['frozen_vs_cem_s0_origin_z'])} | {c['ta_runs']} | {g(c['ta_reward_vs_best_cost_at'])} | "
                 f"{'-' if c['g1_reset_vs_target_frame0'] is None else c['g1_reset_vs_target_frame0']} |")
    L += ["", "The child's frozen, LOES (best after 1,024) and CEM values also equal outputs/noanchor/child/phc_reward.json (plot_child_improvement.py's cache)."]
    for b, R in res.items():
        bud = list(R["best"])
        L += ["", f"## {LABEL[b]}", "", "PHC reward (mean over motions" + (", over the 5 seeds first" if b == "child" else "") + "); best = best latent (sequence) so far, "
              "centre = centre latent / mean sequence. Frozen and native: one rollout per motion, the same in every column.", ""]
        for w in ("best", "centre"):
            L += [f"{w} | " + " | ".join(bud), "|".join(["---"] * (len(bud) + 1))]
            for k in R[w][bud[0]]:
                L.append(f"{NAME[k]} | " + " | ".join(f"{R[w][n][k]:.3f}" for n in bud))
            L.append("")
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    return "\n".join(L)


if __name__ == "__main__":
    readme() if sys.argv[1:] == ["readme"] else main()
