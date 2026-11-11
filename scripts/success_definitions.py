#!/usr/bin/env python3
"""success_definitions.py -- Success Rate under several definitions, for every method on the 52 child task motions (best and
centre after 1,024 rollouts). A clip succeeds if the failure test never triggers; each rollout's state after step t is compared
with reference frame t + 1.
  root       the Task page's definition until 11/08: root more than 1.5 m (horizontal) or 45 deg (rotation) from the reference
  global     chosen before comparing methods: the mean over bodies of the world-frame body position error exceeds 0.5 m
             (adult-equivalent metres, i.e. 0.5 x the body's root scale s on the child); the success test of PHC (Luo et al. 2023), as in
             UHC: joints on average more than 0.5 m from the reference at any point; it counts drift and a collapsed pose alike
  global_raw the same with 0.5 m of the body's own metres
  local      the same 0.5 m (adult-equivalent) on the root-relative, heading-aligned positions of MPJPE: pose only, no drift
Writes outputs/success_definitions/task_motions.json. usage: uv run python scripts/success_definitions.py"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.metrics import mean_body_error, metric_body, phc_success, root_success, world_positions   # noqa: E402
OUT = REPO / "outputs/success_definitions"; RERUN = REPO / "outputs/child_52_rerun"; CEMP = REPO / "outputs/child_52_cem_phc"; LSO = REPO / "outputs/child_52_lso"
# --loes-dir outputs/noanchor/child (user 11/11): LOES ("es") from LOES without the anchor term, everything else unchanged;
# the result goes to task_motions_noanchor.json next to the anchored one
LOESD = REPO / sys.argv[sys.argv.index("--loes-dir") + 1] if "--loes-dir" in sys.argv else None       # None: RERUN, read at call time
SUFFIX = "_noanchor" if "--loes-dir" in sys.argv else ""
DEFS = ("root", "global", "global_raw", "local"); THR = 0.5
body = None; S = None


def clips():
    return [l.strip() for l in open(REPO / "outputs/child_52/clips.txt")]


def world(Q):
    """world positions of every body of the module's `body` (crossenbodiment.metrics.world_positions)"""
    return world_positions(body.m, body.d, Q)


def tests(q, ref, R):
    """q (T, nq) rollout, ref the reference frames 1.., R = (world positions, local positions) of the reference"""
    Pw_r, Pl_r = R; T = min(len(q), len(ref)); q = q[:T].astype(np.float64); r = ref[:T]
    gw = mean_body_error(world(q), Pw_r[:T]); (pl, _) = body.local(q); gl = mean_body_error(pl, Pl_r[:T])
    # crossenbodiment.metrics: the legacy root_success; phc_success on world positions (s, or 1 = raw metres) and on local ones
    return dict(root=float(root_success(q, r)), global_=phc_success(gw, S, THR), global_raw=phc_success(gw, 1.0, THR),
                local=phc_success(gl, S, THR))


def clip(c):
    global body, S
    import eval_metrics as E
    from model.exact_obs import ExactObs
    body = body or metric_body(E.XML["child"], True); S = S or ExactObs(REPO / E.XML["child"], REPO / "assets/robots/adult/robot.xml").s
    stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]
    R = (world(ref), body.local(ref)[0]); res = {"frozen": tests(np.load(RERUN / f"{stem}_es_s0/origin_z.npz")["qpos"], ref, R)}
    self_check = tests(ref, ref, R)
    for src, algos in ((LOESD or RERUN, ("es",)), (RERUN, ("random", "cmaes")), (CEMP, ("cem",))):            # LOES from --loes-dir (default RERUN)
        ix = json.load(open(src / f"rollouts/{stem}/index.json"))["map"]; Q = np.load(src / f"rollouts/{stem}/eval_qpos.npz")["qpos"]
        for a in algos:
            key = "cem_phc" if src == CEMP else a
            res[key + "_best"] = [tests(Q[ix[f"{stem}_{a}_s{s}"]["best"][-1]], ref, R) for s in range(5)]
            res[key + "_centre"] = [tests(Q[ix[f"{stem}_{a}_s{s}"]["centre"][-1]], ref, R) for s in range(5)]
    res["native"] = tests(np.load(LSO / f"{stem}_lso_s0/rollouts.npz")["q_ER8"], ref, R)
    for k in ("lso", "lsor_ours", "ta", "lsor_w005"):
        Rs = [np.load(LSO / f"{stem}_{k}_s{s}/rollouts.npz") for s in range(5)]
        res[k + "_best"] = [tests(x["q_best_it64"], ref, R) for x in Rs]; res[k + "_centre"] = [tests(x["q_mean_it64"], ref, R) for x in Rs]
    return stem, res, self_check


def main():
    from scipy.stats import wilcoxon
    cl = clips()
    with Pool(24) as p:
        out = p.map(clip, cl)
    print("reference against itself, all succeed:", all(all(v == 1.0 for v in o[2].values()) for o in out), "| child root scale s:", S if S else "(in workers)")
    per = {o[0]: o[1] for o in out}; keys = {"root": "root", "global": "global_", "global_raw": "global_raw", "local": "local"}
    motion = lambda m, k: np.array([np.mean([x[k] for x in per[s][m]]) if isinstance(per[s][m], list) else per[s][m][k] for s in per])
    methods = list(out[0][1]); summ = {}
    for m in methods:
        summ[m] = {d: float(motion(m, k).mean()) for d, k in keys.items()}
        if m != "es_best":
            summ[m]["vs_loes_best"] = {}
            for d, k in keys.items():
                L, X = motion("es_best", k), motion(m, k); diff = L - X
                summ[m]["vs_loes_best"][d] = dict(loes_higher=int((diff > 0).sum()), loes_lower=int((diff < 0).sum()), p=float(wilcoxon(L, X).pvalue) if (diff != 0).any() else 1.0)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"task_motions{SUFFIX}.json").write_text(json.dumps(dict(per_clip=per, mean=summ), indent=1))
    print(f"{'method':16s}" + "".join(f"{d:>11s}" for d in keys))
    for m, v in summ.items():
        print(f"{m:16s}" + "".join(f"{100 * v[d]:10.1f}%" for d in keys))
    print("LOES best against each method's best, per motion (LOES higher / lower, Wilcoxon p):")
    for m, v in summ.items():
        if m.endswith("_best") or m in ("frozen", "native"):
            if "vs_loes_best" in v:
                print(f"  {m:16s}" + "  ".join(f"{d}: {x['loes_higher']}/{x['loes_lower']} p {x['p']:.2g}" for d, x in v["vs_loes_best"].items()))


if __name__ == "__main__":
    main()
