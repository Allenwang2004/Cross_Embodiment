#!/usr/bin/env python3
"""metrics_golden.py -- the Task page's metrics, computed by the CURRENT helper functions on a fixed set of saved rollouts, so the
metrics consolidation (refactor step 3) can prove it gives the same numbers.

Inputs (tests/golden/inputs/, frozen by `make-inputs`; see tests/golden/README.md):
  metrics_rollouts.npz   key "<body>__<stem>__<rollout>" = qpos (T, nq) of one saved rollout, "ref__<body>__<stem>" = the reference
                         (all frames; every metric gets q[t] against ref[t + 1], i.e. ref[1:], as on the Task page)
    child    headstand_4, rotate-x--5-0.8_4, move-ego-0-2_1: z0 (child_52_rerun <stem>_es_s0/origin_z.npz), es_best (best.npz),
             native / lso_best / lsor_w005_best / ta_best (child_52_lso <stem>_<method>_s0/rollouts.npz q_ER8 / q_best_it64)
    payload  move-ego-0-2_1: z0, es_best (body_52/payload <stem>_es_s0), native, lso_best (<stem>_lso_s0 q_ER8 / q_best_it128)
    g1       move-ego-0-2_1: z0, es_best (g1_52 <stem>_es_s0), native, lsor_best (<stem>_lsor_s0 q_ER8 / q_best_it128)
    tall     move-ego-0-2_1: z0, es_best, cem_best (the golden baseline's tall_es / tall_cem_phc runs: no real tall runs exist yet)
  metrics_tall_summaries.json   those two runs' summary.json (scripts/tall_body_52.py one() reads them)
What is computed, per rollout (functions called as they are, module globals set where they read them):
  eval_metrics.metrics, eval_metrics.tracking_metrics                    every body (child, payload, tall: SMPL heading; g1: yaw)
  child_52_means.full (ratio 1), child_52_means.hand_fall                child
  success_definitions.tests, fall_definitions.falls                      child (their globals body, S set as their clip() does)
  loes_noanchor.stats                                                    child (s 0.611036), payload (1), g1 (its CONF s)
  body_52_test.stats                                                     payload (global body = the payload body)
  g1_small_test.stats                                                    g1 (global body = the G1)
  drift_success_child.one(clip)                                          child; RR / LS / SEEDS pointed at a folder made from
                                                                         the inputs (es "re-roll" = the saved best rollout)
  tall_body_52.one(clip)                                                 tall; OUT pointed at a folder made from the inputs
Writes outputs/refactor_golden/LABEL/metrics.json (compared by tests/golden/compare.py like every other file).
usage: uv run python tests/golden/metrics_golden.py LABEL
       uv run python tests/golden/metrics_golden.py make-inputs [BASELINE_LABEL]   (once: copies the inputs; reads the main tree)"""
import json, shutil, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
INP = REPO / "tests/golden/inputs"
MAIN = Path("/home/allen19/crossenbodiment")           # make-inputs only: the saved runs live in the main tree (read only)
CHILD = ("headstand/headstand_4", "rotate-x--5-0.8/rotate-x--5-0.8_4", "move-ego-0-2/move-ego-0-2_1")
ONE = "move-ego-0-2/move-ego-0-2_1"


def st(c):
    return c.split("/")[1]


def make_inputs(base_label="baseline"):
    out = {}
    q = lambda p, k="qpos": np.load(p)[k]
    for c in CHILD:
        s = st(c); rr = MAIN / "outputs/child_52_rerun" / f"{s}_es_s0"; ls = MAIN / "outputs/child_52_lso"
        out[f"child__{s}__z0"] = q(rr / "origin_z.npz"); out[f"child__{s}__es_best"] = q(rr / "best.npz")
        out[f"child__{s}__native"] = q(ls / f"{s}_lso_s0/rollouts.npz", "q_ER8")
        for k in ("lso", "lsor_w005", "ta"):
            out[f"child__{s}__{k}_best"] = q(ls / f"{s}_{k}_s0/rollouts.npz", "q_best_it64")
        out[f"ref__child__{s}"] = q(REPO / f"data/child/retargeting_motion/{c}.npz")
    s = st(ONE); pl = MAIN / "outputs/body_52/payload"
    out[f"payload__{s}__z0"] = q(pl / f"{s}_es_s0/origin_z.npz"); out[f"payload__{s}__es_best"] = q(pl / f"{s}_es_s0/best.npz")
    out[f"payload__{s}__native"] = q(pl / f"{s}_lso_s0/rollouts.npz", "q_ER8"); out[f"payload__{s}__lso_best"] = q(pl / f"{s}_lso_s0/rollouts.npz", "q_best_it128")
    out[f"ref__payload__{s}"] = q(REPO / f"data/origin_motion/{ONE}.npz")
    g = MAIN / "outputs/g1_52"
    out[f"g1__{s}__z0"] = q(g / f"{s}_es_s0/origin_z.npz"); out[f"g1__{s}__es_best"] = q(g / f"{s}_es_s0/best.npz")
    out[f"g1__{s}__native"] = q(g / f"{s}_lsor_s0/rollouts.npz", "q_ER8"); out[f"g1__{s}__lsor_best"] = q(g / f"{s}_lsor_s0/rollouts.npz", "q_best_it128")
    out[f"ref__g1__{s}"] = q(REPO / f"outputs/g1/roundtrip/{s}/g1.npz")
    b = REPO / "outputs/refactor_golden" / base_label
    out[f"tall__{s}__z0"] = q(b / "tall_es/origin_z.npz"); out[f"tall__{s}__es_best"] = q(b / "tall_es/best.npz")
    out[f"tall__{s}__cem_best"] = q(b / "tall_cem_phc/best.npz")
    out[f"ref__tall__{s}"] = q(REPO / f"data/len1.4_wid0.8/retargeting_motion/{ONE}.npz")
    INP.mkdir(parents=True, exist_ok=True); np.savez_compressed(INP / "metrics_rollouts.npz", **out)
    summ = {"es": json.loads((b / "tall_es/summary.json").read_text()), "cem": json.loads((b / "tall_cem_phc/summary.json").read_text())}
    (INP / "metrics_tall_summaries.json").write_text(json.dumps(summ, indent=1))
    print(f"{len(out)} arrays -> {INP / 'metrics_rollouts.npz'} ({(INP / 'metrics_rollouts.npz').stat().st_size / 1e6:.2f} MB)")


def plain(x):
    """numpy scalars / arrays -> JSON values (floats keep every bit through json's repr)"""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def main(label):
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
    import mujoco
    import eval_metrics as E
    import child_52_means as CM
    import success_definitions as SD
    import fall_definitions as FD
    import loes_noanchor as LN
    import body_52_test as BT
    import g1_small_test as GT
    import drift_success_child as DS
    import tall_body_52 as TB
    from model.exact_obs import ExactObs
    R = np.load(INP / "metrics_rollouts.npz"); keys = R.files
    work = REPO / "outputs/refactor_golden" / label / "_metrics_work"
    if work.exists():
        shutil.rmtree(work)
    res = {}

    def world(body, Q):                                 # loes_noanchor.clip_job's reference body positions
        P = np.empty((len(Q), body.m.nbody - 1, 3))
        for t, q in enumerate(Q):
            body.d.qpos[:] = q; mujoco.mj_kinematics(body.m, body.d); P[t] = body.d.xpos[1:]
        return P

    bodies = dict(child=E.Body(E.XML["child"], True), payload=E.Body(BT.BODIES["payload"]["xml"], True), g1=E.Body(E.XML["g1"], False),
                  tall=E.Body(TB.XML, True))
    s_of = dict(child=LN.CONF["child"]["s"], payload=LN.CONF["payload"]["s"], g1=LN.CONF["g1"]["s"])
    for name, body in bodies.items():
        stems = sorted({k.split("__")[1] for k in keys if k.startswith(name + "__")})
        for s in stems:
            ref = R[f"ref__{name}__{s}"].astype(np.float64); r1 = ref[1:]
            Pw = world(body, r1); _, Dr = body.part_floor(r1)
            if name == "child":
                SD.body = body; SD.S = ExactObs(REPO / E.XML["child"], REPO / "assets/robots/adult/robot.xml").s
                Rsd = (SD.world(r1), body.local(r1)[0])
                FD.body = body; names, Drf = body.part_floor(r1); d = mujoco.MjData(body.m); hid = body.m.body("Head").id; hr = np.empty(len(r1))
                for t in range(len(r1)):
                    d.qpos[:] = r1[t]; mujoco.mj_kinematics(body.m, d); hr[t] = d.xpos[hid][2]
                Rfd = (names, Drf, hr)
            if name == "payload":
                BT.body = body
            if name == "g1":
                GT.body = body
            for k in sorted(k for k in keys if k.startswith(f"{name}__{s}__")):
                q = R[k].astype(np.float64); m = {}
                m["eval_metrics.metrics"] = E.metrics(body, q, r1)
                m["eval_metrics.tracking_metrics"] = E.tracking_metrics(body, q, r1)
                if name in s_of:
                    m["loes_noanchor.stats"] = LN.stats(body, q, r1, Pw, Dr, s_of[name])
                if name == "child":
                    m["child_52_means.full"] = CM.full(body, q, r1, 1.0)
                    m["child_52_means.hand_fall"] = CM.hand_fall(body, q, r1)
                    m["success_definitions.tests"] = SD.tests(q, r1, Rsd)
                    m["fall_definitions.falls"] = FD.falls(q, r1, Rfd)
                if name == "payload":
                    m["body_52_test.stats"] = BT.stats(q, r1, Pw, Dr)
                if name == "g1":
                    m["g1_small_test.stats"] = GT.stats(q, r1, Pw, Dr)
                res[k] = plain(m); print(k, flush=True)

    # drift_success_child.one: its folders rebuilt from the inputs (SEEDS = seed 0 only)
    rr, ls = work / "child_52_rerun", work / "child_52_lso"; DS.RR, DS.LS, DS.SEEDS = rr, ls, range(1)
    for c in CHILD:
        s = st(c); (rr / f"{s}_es_s0").mkdir(parents=True, exist_ok=True); (rr / "rollouts" / s).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(rr / f"{s}_es_s0/origin_z.npz", qpos=R[f"child__{s}__z0"])
        np.savez_compressed(rr / "rollouts" / s / "eval_qpos.npz", qpos=R[f"child__{s}__es_best"][None])
        (rr / "rollouts" / s / "index.json").write_text(json.dumps({"map": {f"{s}_es_s0": {"best": [0]}}}))
        for k in ("lso", "lsor_w005", "ta"):
            (ls / f"{s}_{k}_s0").mkdir(parents=True, exist_ok=True)
            extra = {"q_ER8": R[f"child__{s}__native"]} if k == "lso" else {}
            np.savez_compressed(ls / f"{s}_{k}_s0/rollouts.npz", q_best_it64=R[f"child__{s}__{k}_best"], **extra)
        res[f"drift_success_child.one/{s}"] = plain(DS.one(c)[1]); print("drift_success_child", s, flush=True)

    # tall_body_52.one: its OUT folder rebuilt from the inputs
    TB.OUT = work / "tall_body_52"; S = json.loads((INP / "metrics_tall_summaries.json").read_text()); s = st(ONE)
    for a, z0k, bk in (("es", "z0", "es_best"), ("cem", "z0", "cem_best")):
        d = TB.OUT / f"{s}_{a}_s0"; d.mkdir(parents=True, exist_ok=True); (d / "summary.json").write_text(json.dumps(S[a]))
        np.savez_compressed(d / "origin_z.npz", qpos=R[f"tall__{s}__{z0k}"]); np.savez_compressed(d / "best.npz", qpos=R[f"tall__{s}__{bk}"])
    _, r_, chk = TB.one(ONE); res[f"tall_body_52.one/{s}"] = plain(dict(res=r_, checks=chk)); print("tall_body_52", s, flush=True)

    out = REPO / "outputs/refactor_golden" / label / "metrics.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1)); shutil.rmtree(work)
    print(f"{len(res)} entries -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "make-inputs":
        make_inputs(*sys.argv[2:3])
    else:
        label = sys.argv[1]; sys.argv = sys.argv[:1]          # the drivers imported in main() parse sys.argv at import
        main(label)
