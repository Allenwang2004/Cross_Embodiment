#!/usr/bin/env python3
"""amass_thresholds.py -- PHC success on the AMASS runs under looser thresholds (user, 11/13: "what if the success threshold is 0.75?").
For each clip, body and method, the rollout's world-frame mean body error per frame (q[t] against ref[t + 1], all bodies but the world,
exactly as scripts/amass_baselines.py's success), divided by the body's root scale s (adult-equivalent metres). Success at threshold h:
that error never above h. Reported for h = 0.5 (PHC's, = amass_baselines.py's success), 0.75 and 1.0 m, plus the share of frames within h.
Runs: outputs/amass_baselines_bfm (BFMTrack's budget, 128 x 24): native q_ER8, LSO / LSOR / TA q_best_it24 (3,072) and q_best_it8 (1,024).
  uv run python scripts/amass_thresholds.py  -> outputs/amass_baselines_bfm/thresholds.json, thresholds.md"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment import paths                                         # noqa: E402
from crossenbodiment.metrics import mean_body_error, world_positions      # noqa: E402
from crossenbodiment.results import write_json                            # noqa: E402
# OOP refactor (ported from the main tree, 11/13): the runs and amass_baselines.py --set bfm's results.json are read under
# paths.outputs_root() ($CE_OUTPUTS_ROOT, default outputs/), thresholds.json / .md written under $CE_RESULTS_ROOT (paths.write_path)
OUT_REL = "outputs/amass_baselines_bfm"; OUT = paths.read_path(OUT_REL); H = (0.5, 0.75, 1.0)
import amass_baselines as AB                                              # noqa: E402
CLIPS = AB.CLIPS; BODY = AB.BODY
_ctx = {}


def one(job):
    import mujoco
    b, c = job
    if _ctx.get("b") != b:
        m = mujoco.MjModel.from_xml_path(str(REPO / BODY[b]["xml"])); _ctx.update(b=b, m=m, d=mujoco.MjData(m))
    m, d = _ctx["m"], _ctx["d"]; s = AB.scale(b, c)
    ref = np.load(REPO / BODY[b]["ref"].format(c=c))["qpos"].astype(np.float64)[1:]

    def xpos(Q):
        return world_positions(m, d, Q)
    Pr = xpos(ref); out = {}
    z = {k: np.load(OUT / b / f"{c}_{k}_s0/rollouts.npz") for k in ("lso", "lsor", "ta")}
    rolls = dict(native=z["lso"]["q_ER8"], **{f"{k}@3072": z[k]["q_best_it24"] for k in z}, **{f"{k}@1024": z[k]["q_best_it8"] for k in z})
    for k, q in rolls.items():
        q = q.astype(np.float64); T = min(len(q), len(ref))
        e = mean_body_error(xpos(q[:T]), Pr[:T]) / s
        out[k] = dict(max=float(e.max()), **{f"succ{h}": float(e.max() <= h) for h in H}, **{f"within{h}": float((e <= h).mean()) for h in H})
    return b, c, out


def main():
    jobs = [(b, c) for b in BODY for c in CLIPS]
    with Pool(16) as p:
        res = p.map(one, jobs, chunksize=4)
    # check: success at 0.5 and within0.5 equal amass_baselines.py --set bfm's (results.json)
    R = json.load(open(OUT / "results.json")); keys = R["child"]["keys"]; bad = 0
    for b, c, o in res:
        pm = R[b]["per_motion"][c]
        for k in ("lso", "lsor", "ta"):
            v = pm[k]["best"][-1]
            bad += (abs(v[keys.index("success")] - o[f"{k}@3072"]["succ0.5"]) > 0) + (abs(v[keys.index("within")] - o[f"{k}@3072"]["within0.5"]) > 1e-9)
    rep, L = {}, [f"# AMASS success under other thresholds (BFMTrack budget), adult-equivalent metres\n\ncheck vs results.json (success, within at 0.5 m): {bad} mismatches\n"]
    for b in BODY:
        o = [x[2] for x in res if x[0] == b]; rep[b] = {}
        L.append(f"\n## {b} ({len(o)} clips): success rate / share of frames within\n\n| method | 0.5 m | 0.75 m | 1.0 m |\n|---|---|---|---|")
        for k in ("native", "lso@3072", "ta@3072", "lsor@3072", "lso@1024", "ta@1024", "lsor@1024"):
            rep[b][k] = {f"succ{h}": float(np.mean([x[k][f'succ{h}'] for x in o])) for h in H} | {f"within{h}": float(np.mean([x[k][f'within{h}'] for x in o])) for h in H}
            L.append(f"| {k} | " + " | ".join(f"{100 * rep[b][k][f'succ{h}']:.1f}% ({sum(x[k][f'succ{h}'] for x in o):.0f}/{len(o)}) / {100 * rep[b][k][f'within{h}']:.1f}%" for h in H) + " |")
    W = paths.write_path(OUT_REL)
    write_json(W / "thresholds.json", dict(summary=rep, per_clip=[dict(body=b, clip=c, **o) for b, c, o in res], mismatches=int(bad)))
    (W / "thresholds.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
