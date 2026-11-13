#!/usr/bin/env python3
"""kpkd_test.py -- does the latent search still work when kp and kd move in opposite directions? The child, the G1, the limited leg (right leg locked, bent; tracks the bent-leg reference) and the payload (right hand x 80; tracks the adult's motion), kp x 0.8 with kd x 1.2, kp x 1.2 with kd x 0.8, and both unchanged, on the 8 clips of the k-scale test (outputs/k_scale), 5 seeds, LOES with 1,024 rollouts and the current settings (reference-velocity start, q[t] scored against ref[t + 1], independent perturbations, no extra rollouts of the mean). Torque limits unchanged. An opposite move changes each joint's damping ratio by kd / sqrt(kp): x 1.34 and x 0.73.
  prepare   writes each human body's XML with the gains moved (scripts/scale_body_gains.py), next to its own folder as <folder>_kp*_kd*, and assets/robots/g1/gains_perjoint_kp*_kd*.json
  commands  prints "<out dir> <command>" per search (run through xargs -P 8), into outputs/kpkd/<body>_<condition>/<stem>_s<seed>
  analyse   cost, metrics (scripts/eval_metrics.py, on the body itself) and PHC reward (as scripts/phc_reward_tables.py) of each run's z0 and best-so-far rollout, for the bodies whose runs are all finished; writes <OUT>/{per_run.npz, means.json} . Checks: settings, z0 rollout identical across seeds, conditions differ, child kp x 1 matches outputs/child_52_rerun (--set mild: outputs/noanchor/child) where the clip is there.
--set mild (11/12, user: 0.8 -> 0.9 and 1.2 -> 1.1, the same 6 motions, 3 seeds, child and G1): kp1_kd1, kp0.9_kd1.1, kp1.1_kd0.9, kp0.9_kd0.9, kp1.1_kd1.1, seeds 0-2, child and g1 only, LOES as used now, without the anchor term
          (--anchor-weight 0 on both; robot_z_search's default is 0.3, so the G1 gets it explicitly), everything else as above -> outputs/kpkd_mild. Without --set: the 11/06 run (outputs/kpkd, anchor 0.3).
          11/13 (user): + the tall body (tall = len1.4_wid0.8) with the paper's gains as its base, kp x 0.6 (assets/robots_torque/len1.4_wid0.8_kp0.6_kd1.0),
          the five conditions multiplying that base again (kp0.9 = 0.6 x 0.9 = 0.54 of the calibrated gains; XMLs in <base folder>_<condition>);
          its searches use the tall runs' own flags (outputs/tall_kp06_52/commands.txt, the _es_ lines) with --xml, --seed and --out changed, so
          tall kp1_kd1 must reproduce outputs/tall_kp06_52/<stem>_es_s<seed> where that run exists; root scale s = 1.364408905613045 (single_z_search's
          median ref / adult root x, y; every run's summary pos_div is checked against it) for the PHC reward and the 0.5 x s success threshold.
          The child and G1 commands are unchanged; the tall's are the last 90 lines of commands.
  table     after analyse, contactfall and phcsuccess: the Notion page's table rows (Fall Time by contact, PHC success, analyse's metrics, PHC reward) -> <OUT>/table.md
usage: uv run python scripts/kpkd_test.py [--set mild] prepare | commands | analyse | contactfall | phcsuccess | table"""
import json, shutil, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossembodiment.bodies import BODIES
from crossembodiment.paths import CHILD_XML_REL
from crossembodiment.metrics import contact_fall_time, mean_body_error, metric_body, phc_success, world_positions
from crossembodiment import paths
from crossembodiment.experiment import Experiment, workers
from crossembodiment.results import Checks, save_npz, write_json
# refactor step 8: runs (and, for summary, per_run.npz) are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under
# paths.results_root() ($CE_RESULTS_ROOT); both REPO/outputs by default
SET = "default"
if "--set" in sys.argv:                                                  # taken out of argv, so summary's --drop list stays clean
    _i = sys.argv.index("--set"); SET = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
assert SET in ("default", "mild"), SET
MILD = SET == "mild"
OUT_REL = "outputs/kpkd_mild" if MILD else "outputs/kpkd"                 # --set mild (main tree 11/12): the second, milder gain test
OUT = paths.read_path(OUT_REL); RES = paths.result(OUT_REL[len("outputs/"):]); SEEDS = range(3) if MILD else range(5)
RUNS_REPO = paths.outputs_root().parent                                 # the tree the runs were made in (summary "ref" is absolute there; this repository by default)
ANCHOR = 0.0 if MILD else 0.3                                            # the anchor weight of every search
SAME_RUN = "outputs/noanchor/child" if MILD else "outputs/child_52_rerun"   # child kp x 1 must reproduce these runs' best_z
BODY = dict(child=dict(xml=CHILD_XML_REL, ref="data/child/retargeting_motion/{c}.npz", label="child"), g1=dict(ref="outputs/g1/roundtrip/{stem}/g1.npz"),
            limit_leg=dict(xml="assets/robots/adult_lockRleg_bent_lim/robot.xml", ref="data/adult_lockRleg_bent_lim/retargeting_motion/{c}.npz", label="adult_lockRleg_bent_lim"),
            heavy_arm=dict(xml="assets/robots/adult_heavyRWristHand_x80/robot.xml", ref="data/origin_motion/{c}.npz", label="adult_heavyRWristHand_x80"))
COND = (("kp1_kd1", 1.0, 1.0), ("kp0.8_kd1.2", 0.8, 1.2), ("kp1.2_kd0.8", 1.2, 0.8), ("kp0.8_kd0.8", 0.8, 0.8), ("kp1.2_kd1.2", 1.2, 1.2))
if MILD:
    COND = (("kp1_kd1", 1.0, 1.0), ("kp0.9_kd1.1", 0.9, 1.1), ("kp1.1_kd0.9", 1.1, 0.9), ("kp0.9_kd0.9", 0.9, 0.9), ("kp1.1_kd1.1", 1.1, 1.1))
    BODY = {b: BODY[b] for b in ("child", "g1")}
    BODY["tall"] = dict(xml="assets/robots_torque/len1.4_wid0.8_kp0.6_kd1.0/robot_torque_full.xml", ref="data/len1.4_wid0.8/retargeting_motion/{c}.npz", label="len1.4_wid0.8",
                        note=" (len1.4_wid0.8; gains relative to its paper base kp × 0.6, kd × 1)")
TALL_RUN = "outputs/tall_kp06_52"                                         # tall kp x 1 must reproduce these runs (<stem>_es_s<seed>) exactly
# 11/08: the user's motion list is these 6; crawl-0.4-2-d_8 and crouch-0_8 were also run for the first three conditions (outputs/kpkd,
# means_8clips.json) and are left out, so the same-ratio conditions (kp and kd x 0.8 or x 1.2 together) were run on the 6 only
CLIPS_8 = ["crawl-0.4-2-d/crawl-0.4-2-d_8", "crouch-0/crouch-0_8", "headstand/headstand_4", "jump-2/jump-2_6", "jump-2/jump-2_7", "move-ego-0-2/move-ego-0-2_2", "move-ego-0-4/move-ego-0-4_8", "rotate-x--5-0.8/rotate-x--5-0.8_4"]
CLIPS = [c for c in CLIPS_8 if not c.startswith(("crawl-0.4-2-d/", "crouch-0/"))]
def XML(b, name):
    p = Path(BODY[b]["xml"])
    return str(p) if name == "kp1_kd1" else str(p.parent.parent / f"{p.parent.name}_{name}" / p.name)


GAINS = lambda name: "assets/robots/g1/gains_perjoint.json" if name == "kp1_kd1" else f"assets/robots/g1/gains_perjoint_{name}.json"
# the run folders: outputs/kpkd[_mild]/<body>_<condition>/<stem>_s<seed> (no method key: every run is LOES)
EXP = Experiment(OUT_REL, clips=CLIPS, methods=[n for n, _, _ in COND], seeds=SEEDS, bodies=tuple(BODY),
                 name=lambda r: f"{r.body}_{r.method}/{r.stem}_s{r.seed}", order=("body", "method", "clip", "seed"), sep=" ")
body = {}


def prepare():
    # refactor step 9: crossembodiment/bodybuild GainScaler (scale_xml = scripts/scale_body_gains.py --kp --kd, which this ran as a
    # subprocess; scale_json = the json this wrote inline): the same files byte for byte (tests/golden/verify_bodies.py, part gains)
    from crossembodiment.bodybuild.gains import GainScaler
    for name, a, b in COND[1:]:
        g = GainScaler(a, b)
        for body_ in [x for x in BODY if x != "g1"]:
            if (REPO / XML(body_, name)).exists():                       # made earlier: keep it as it is
                continue
            g.scale_xml(BODY[body_]["xml"], XML(body_, name))
            for f in ("parameter.json", "skeleton.json"):
                if (REPO / BODY[body_]["xml"]).with_name(f).exists():
                    shutil.copy(REPO / Path(BODY[body_]["xml"]).with_name(f), REPO / Path(XML(body_, name)).with_name(f))
        if (REPO / GAINS(name)).exists():
            continue
        g.scale_json(GAINS("kp1_kd1"), GAINS(name))


def command(r):
    b, name, c, s, d = r.body, r.method, r.clip, r.seed, r.dir
    if b == "tall":                                                       # outputs/tall_kp06_52/commands.txt's _es_ flags, in their order; no --ref (single_z_search finds it)
        return f"uv run python scripts/single_z_search.py --clip {c} --body {BODY[b]['label']} --xml {XML(b, name)} --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --eval-every 0 --init-vel-scale 1 --seed {s} --objective bfm --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy --anchor-weight {ANCHOR:g} --out {d}"
    if b != "g1":
        rf = "" if b == "child" else f" --ref {ref_path(b, c)}"
        return f"uv run python scripts/single_z_search.py --clip {c} --body {BODY[b]['label']} --xml {XML(b, name)}{rf} --objective bfm --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy --anchor-weight {ANCHOR:g} --seed {s} --eval-every 0 --init-vel-scale 1 --out {d}"
    return f"uv run python scripts/robot_z_search.py --robot g1 --clip {c} --fill ridge --ik-weight 1.0{' --anchor-weight 0' if MILD else ''} --evals 1024 --init-vel reference --eval-every 0 --gains {GAINS(name)} --seed {s} --out {d}"


def commands():
    EXP.print_commands(command)


def ref_path(b, c):
    return BODY[b]["ref"].format(c=c, stem=c.split("/")[1])


def ref_of(b, c):
    return np.load(REPO / ref_path(b, c))["qpos"].astype(np.float64)[1:]


_PHC = {}


def phc_reward(b, c, q):
    """PHC reward of rollout q as scripts/phc_reward_tables.py computes it: 1 - PHCCost(model, ref[0], ref[1:], scale=s)(q[None], ref[0]); the
    child and the human bodies on their own XML and reference, s = SCALE; the G1 on assets/robots/g1/g1.xml against its target cut to the
    clip's length, s = RobotMap('g1', fill='ridge').s, the reset pose = target frame 0. The gains do not enter (kinematics only)."""
    import mujoco
    from model.phc_reward import PHCCost
    if (b, c) not in _PHC:
        if b == "g1":
            if "g1_s" not in _PHC:
                from model.robot_map import RobotMap
                _PHC["g1_s"] = RobotMap("g1", fill="ridge").scale
            s, xml = _PHC["g1_s"], "assets/robots/g1/g1.xml"
            ref = np.load(REPO / ref_path(b, c))["qpos"].astype(np.float64)[:len(np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"])]
        else:
            s, xml = SCALE[b], BODY[b]["xml"]; ref = np.load(REPO / ref_path(b, c))["qpos"].astype(np.float64)
        _PHC[b, c] = (PHCCost(mujoco.MjModel.from_xml_path(str(REPO / xml)), ref[0], ref[1:], scale=s), ref[0])
    f, r0 = _PHC[b, c]
    return 1.0 - float(f(np.asarray(q, dtype=np.float64)[None], r0)[0])


def one(job):
    import eval_metrics as E
    from crossembodiment.metrics import LEGACY_KEYS as KEYS, legacy_metrics as full   # child_52_means' (refactor step 4)
    b, name, c, s = job
    if b not in body:
        body[b] = metric_body(E.XML["g1"] if b == "g1" else BODY[b]["xml"], b != "g1")
    d = OUT / f"{b}_{name}/{c.split('/')[1]}_s{s}"; S = json.load(open(d / "summary.json")); ref = ref_of(b, c); base = S["origin_z"]["cost"]; bq = np.load(d / "best.npz")["qpos"].astype(np.float64); zq = np.load(d / "origin_z.npz")["qpos"].astype(np.float64)
    vz, vb = full(body[b], zq, ref, 1.0), full(body[b], bq, ref, S["best"]["cost"] / base)
    bad = Checks.settings(dict(evals=1024, seed=s, frame_offset=1, sampling="independent", sigma=0.05, lr=0.03, heading_weight=1.0, pos_weight=0.1, anchor_weight=ANCHOR), S)
    bad += [k for k, want in (("init_vel_scale", 1.0), ("obs_scale", "exact"), ("xml", XML(b, name) if b != "g1" else None)) if b != "g1" and S.get(k) != want] + [k for k, want in (("init_vel", "reference"), ("fill", "ridge"), ("ik_weight", 1.0), ("gains", GAINS(name))) if b == "g1" and S.get(k) != want]
    if b == "tall":
        bad += [k for k, want in (("body", BODY[b]["label"]), ("ref", str(RUNS_REPO / ref_path(b, c))), ("pos_div", SCALE[b]), ("pos_unit", "adult"), ("pairs", 8), ("objective", "bfm"), ("init", "reference"), ("algo", "es")) if S.get(k) != want]
    return job, [vz[k] for k in KEYS], [vb[k] for k in KEYS], zq[:len(ref)], base, S["best"]["cost"], bad, phc_reward(b, c, zq), phc_reward(b, c, bq)


def analyse():
    from crossembodiment.metrics import LEGACY_KEYS as KEYS   # child_52_means.KEYS (refactor step 4)
    done = lambda b: EXP.all_done(EXP.runs(body=(b,)))
    BODIES = [b for b in BODY if done(b)]; print("bodies finished:", BODIES, flush=True)
    jobs = [(b, n, c, s) for b in BODIES for n, _, _ in COND for c in CLIPS for s in SEEDS]
    with Pool(workers(16)) as p:
        res = p.map(one, jobs)
    R = {r[0]: r for r in res}; shape = (len(CLIPS), len(SEEDS), len(KEYS)); Z = {}; B = {}; CZ = {}; CB = {}; PZ = {}; PB = {}; bad = [(r[0], r[6]) for r in res if r[6]]; spread = 0.0; same_across = 0
    for b in BODIES:
        for n, _, _ in COND:
            z, x = np.zeros(shape), np.zeros(shape)
            for i, c in enumerate(CLIPS):
                for s in SEEDS:
                    r = R[(b, n, c, s)]; z[i, s], x[i, s] = r[1], r[2]; spread = max(spread, float(np.abs(r[3] - R[(b, n, c, 0)][3]).max()))
            Z[b, n], B[b, n] = z, x; CZ[b, n] = np.array([R[(b, n, c, 0)][4] for c in CLIPS]); CB[b, n] = np.array([[R[(b, n, c, s)][5] for s in SEEDS] for c in CLIPS])
            PZ[b, n] = np.array([R[(b, n, c, 0)][7] for c in CLIPS]); PB[b, n] = np.array([[R[(b, n, c, s)][8] for s in SEEDS] for c in CLIPS])
        for c in CLIPS:
            q = [R[(b, n, c, 0)][3] for n, _, _ in COND]; same_across += int(np.abs(q[0] - q[1]).max() == 0 or np.abs(q[0] - q[2]).max() == 0)
    m52 = 0.0; n52 = 0
    for c in CLIPS:
        st = c.split("/")[1]
        for s in SEEDS:
            o = paths.read_path(f"{SAME_RUN}/{st}_es_s{s}/best_z.npy")
            if "child" in BODIES and o.exists():
                m52 = max(m52, float(np.abs(np.load(o) - np.load(OUT / f"child_kp1_kd1/{st}_s{s}/best_z.npy")).max())); n52 += 1
    print("runs:", len(res), "| settings not as intended:", len(bad), bad[:4], "| z0 rollout, largest difference across seeds:", spread, "| clips whose z0 rollout did not change with the gains:", same_across, f"| child kp x 1 vs {SAME_RUN} on {n52} runs, largest z difference:", m52, flush=True)
    if "tall" in BODIES:
        b = "tall"; sp = max(float(np.abs(R[(b, n, c, s)][3] - R[(b, n, c, 0)][3]).max()) for n, _, _ in COND for c in CLIPS for s in SEEDS)
        moved = [float(np.abs(R[(b, n, c, 0)][3] - R[(b, "kp1_kd1", c, 0)][3]).max()) for n, _, _ in COND[1:] for c in CLIPS]
        same, nt = [], 0                                                  # same command apart from --out -> every array and curve.csv identical
        for c in CLIPS:
            for s in SEEDS:
                o, d = paths.read_path(f"{TALL_RUN}/{c.split('/')[1]}_es_s{s}"), OUT / f"tall_kp1_kd1/{c.split('/')[1]}_s{s}"
                if not (o / "summary.json").exists():
                    continue
                nt += 1; ok = (o / "curve.csv").read_bytes() == (d / "curve.csv").read_bytes() and np.array_equal(np.load(o / "best_z.npy"), np.load(d / "best_z.npy"))
                for f in ("best.npz", "origin_z.npz", "z_trace.npz"):
                    x, y = np.load(o / f), np.load(d / f); ok &= sorted(x.files) == sorted(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
                S0, S1 = (json.load(open(p / "summary.json")) for p in (o, d)); ok &= S0["best"] == S1["best"] and S0["origin_z"] == S1["origin_z"]
                same.append(ok)
        print(f"tall: z0 rollout, largest difference across seeds: {sp} | z0 rollout vs kp1_kd1, smallest largest-difference over the other {len(moved)} (condition, clip) pairs: {min(moved):.4g}"
              f" (unchanged: {sum(v == 0 for v in moved)}) | tall kp x 1 vs {TALL_RUN} on {nt} runs: identical (curve.csv bytes, best_z, best/origin_z/z_trace arrays, summary best and origin_z) in {sum(same)}", flush=True)
    i = {k: j for j, k in enumerate(KEYS)}; rep = {}
    for b in BODIES:
        rep[b] = {}
        for n, _, _ in COND:
            z, x = Z[b, n], B[b, n]; zc = z[:, 0]
            rep[b][n] = dict(z0=dict({k: float(zc[:, i[k]].mean()) for k in KEYS}, cost=float(CZ[b, n].mean()), phc_reward=float(PZ[b, n].mean())),
                             best=dict({k: float(x[..., i[k]].mean()) for k in KEYS}, cost=float(CB[b, n].mean()), phc_reward=float(PB[b, n].mean())),
                             per_clip={c.split("/")[1]: dict(z0_cost=float(CZ[b, n][j]), best_cost=float(CB[b, n][j].mean()), z0_phc_reward=float(PZ[b, n][j]), best_phc_reward=float(PB[b, n][j].mean()), z0_success=float(zc[j, i["success"]]), z0_mpjpe=float(zc[j, i["mpjpe"]]), z0_fall=float(zc[j, i["fall_s"]]), best_success=int(x[j, :, i["success"]].sum()), best_mpjpe=float(x[j, :, i["mpjpe"]].mean()), best_fall=float(x[j, :, i["fall_s"]].mean()), improved=int(sum(R[(b, n, c, s)][5] < R[(b, n, c, s)][4] for s in SEEDS))) for j, c in enumerate(CLIPS)},
                             seed_success=[float(x[:, s, i["success"]].mean()) for s in SEEDS], seed_mpjpe=[float(x[:, s, i["mpjpe"]].mean()) for s in SEEDS])
    save_npz(RES / "per_run.npz", clips=np.array(CLIPS), keys=np.array(KEYS), **{f"z0_{b}_{n}": Z[b, n] for b, n in Z}, **{f"best_{b}_{n}": B[b, n] for b, n in B}, **{f"cost_z0_{b}_{n}": CZ[b, n] for b, n in CZ}, **{f"cost_best_{b}_{n}": CB[b, n] for b, n in CB},
             **{f"phc_z0_{b}_{n}": PZ[b, n] for b, n in PZ}, **{f"phc_best_{b}_{n}": PB[b, n] for b, n in PB})
    print("written", write_json(RES / "means.json", rep))


def table():
    """the Notion page's tables: Fall Time from contactfall (means_contactfall_without_none.json), Success Rate from phcsuccess
    (means_phcsuccess.json), MPJPE, MMPJPE, MPJAE, cost and PHC reward from analyse (means.json; its own fall_s / success are the older
    definitions and are not used) -> <OUT>/table.md"""
    m, f, p = (json.load(open(paths.own_path(f"{OUT_REL}/{x}"))) for x in ("means.json", "means_contactfall_without_none.json", "means_phcsuccess.json"))
    lab = {n: f"kp × {a:g}, kd × {d:g}" for n, a, d in COND}; L = []
    for b in [x for x in BODY if x in m]:
        L += [f"\n{b}{BODY[b].get('note', '')}: frozen = mean over {len(CLIPS)} motions, LOES = mean over {len(CLIPS) * len(SEEDS)} searches ({OUT_REL})", "",
              "| Gains | Method | Fall Time (s) ↓ | Success Rate ↑ | MPJPE (cm) ↓ | MMPJPE (cm) ↓ | MPJAE (m/s²) ↓ | Cost ↓ | PHC reward ↑ |", "|---|---|---|---|---|---|---|---|---|"]
        for n, _, _ in COND:
            for w, name in (("z0", "Frozen (z\\_0)"), ("best", "LOES")):
                v = m[b][n][w]
                L.append(f"| {lab[n]} | {name} | {f[b][n][w]:.3f} | {100 * p[b][n][w]:.1f}% | {v['mpjpe']:.2f} | {v['mmpjpe']:.2f} | {v['mpjae']:.2f} | {v['cost']:.3f} | {v['phc_reward']:.3f} |")
    t = paths.writable(RES / "table.md"); t.parent.mkdir(parents=True, exist_ok=True); t.write_text("\n".join(L) + "\n"); print("\n".join(L))


def summary():
    """summary --drop <stem> ...: the tables of analyse() from per_run.npz without the given motions -> means_without_<stems>.json"""
    drop = sys.argv[sys.argv.index("--drop") + 1:] if "--drop" in sys.argv else []
    P = np.load(OUT / "per_run.npz"); keys = list(P["keys"]); cl = [c.split("/")[1] for c in P["clips"]]; keep = np.array([c not in drop for c in cl])
    assert set(drop) <= set(cl), drop
    i = {k: j for j, k in enumerate(keys)}; rep = dict(dropped=drop, clips=[c for c, k in zip(cl, keep) if k]); lab = dict(kp1_kd1="kp × 1, kd × 1", **{n: f"kp × {a:g}, kd × {b:g}" for n, a, b in COND[1:]})
    for b in BODY:
        if f"z0_{b}_kp1_kd1" not in P.files:
            continue
        rep[b] = {}
        for n, _, _ in COND:
            z, x = P[f"z0_{b}_{n}"][keep][:, 0], P[f"best_{b}_{n}"][keep]; cz, cb = P[f"cost_z0_{b}_{n}"][keep], P[f"cost_best_{b}_{n}"][keep]
            rep[b][n] = dict(z0=dict({k: float(z[:, i[k]].mean()) for k in keys}, cost=float(cz.mean())), best=dict({k: float(x[..., i[k]].mean()) for k in keys}, cost=float(cb.mean())),
                             improved=int((cb < cz[:, None]).sum()), searches=int(cb.size), success_by_clip={c: int(v) for c, v in zip(rep["clips"], x[..., i["success"]].sum(1))},
                             seed_success=[float(v) for v in x[..., i["success"]].mean(0)])
    tag = "_".join(drop) or "none"; write_json(RES / f"means_without_{tag}.json", rep); print("written", RES / f"means_without_{tag}.json")
    for b in BODY:
        if b not in rep:
            continue
        print(f"\n{b}\n| Gains | Method | Fall Time (s) ↓ | Success Rate ↑ | MPJPE (cm) ↓ | MMPJPE (cm) ↓ | MPJAE (m/s²) ↓ | Cost ↓ |")
        for n, _, _ in COND:
            for w, name in (("z0", "Frozen (z\\_0)"), ("best", "LOES")):
                v = rep[b][n][w]
                print(f"| {lab[n]} | {name} | {v['fall_s']:.3f} | {100 * v['success']:.1f}% | {v['mpjpe']:.2f} | {v['mmpjpe']:.2f} | {v['mpjae']:.2f} | {v['cost']:.3f} |")


def hand_one(job):
    import eval_metrics as E
    from crossembodiment.metrics import body_hand_fall as hand_fall   # child_52_means.hand_fall (refactor step 4)
    b, name, c, s = job
    if b not in body:
        body[b] = metric_body(E.XML["g1"] if b == "g1" else BODY[b]["xml"], b != "g1")
    d = OUT / f"{b}_{name}/{c.split('/')[1]}_s{s}"; ref = ref_of(b, c)
    return job, hand_fall(body[b], np.load(d / "origin_z.npz")["qpos"].astype(np.float64), ref), hand_fall(body[b], np.load(d / "best.npz")["qpos"].astype(np.float64), ref)


def handfall():
    """handfall --drop <stem> ...: Fall Time by hand contact (crossembodiment.metrics.body_hand_fall), frozen and LOES, by motion (NaN for motions whose
    reference puts the hands on the floor) and by frame -> means_handfall_without_<stems>.json"""
    drop = sys.argv[sys.argv.index("--drop") + 1:] if "--drop" in sys.argv else []
    clips = [c for c in CLIPS if c.split("/")[1] not in drop]
    jobs = [(b, n, c, s) for b in BODY for n, _, _ in COND for c in clips for s in SEEDS]
    with Pool(workers(16)) as p:
        R = {r[0]: r[1:] for r in p.map(hand_one, jobs)}
    rep = dict(dropped=drop, clips=[c.split("/")[1] for c in clips])
    for b in BODY:
        rep[b] = dict(hand_motions=[c.split("/")[1] for c in clips if np.isnan(R[(b, "kp1_kd1", c, 0)][0][0])])
        for n, _, _ in COND:
            z = np.array([[R[(b, n, c, 0)][0][k] for k in (0, 1)] for c in clips]); x = np.array([[[R[(b, n, c, s)][1][k] for k in (0, 1)] for s in SEEDS] for c in clips])
            rep[b][n] = dict(z0_by_motion=float(np.nanmean(z[:, 0])), best_by_motion=float(np.nanmean(x[..., 0])), z0_by_frame=float(z[:, 1].mean()), best_by_frame=float(x[..., 1].mean()),
                             per_clip={c.split("/")[1]: dict(z0=z[i].tolist(), best=x[i].mean(0).tolist()) for i, c in enumerate(clips)})
    tag = "_".join(drop) or "none"; write_json(RES / f"means_handfall_without_{tag}.json", rep)
    for b in BODY:
        print(b, "hand motions:", rep[b]["hand_motions"])
        for n, _, _ in COND:
            v = rep[b][n]; print(f"  {n:12s} by motion: frozen {v['z0_by_motion']:.3f} -> LOES {v['best_by_motion']:.3f} | by frame: frozen {v['z0_by_frame']:.3f} -> LOES {v['best_by_frame']:.3f}")


def contact_one(job):
    import eval_metrics as E
    b, name, c, s = job
    if b not in body:
        body[b] = metric_body(E.XML["g1"] if b == "g1" else BODY[b]["xml"], b != "g1")
    B = body[b]; ref = ref_of(b, c); _, Dr = B.part_floor(ref); d = OUT / f"{b}_{name}/{c.split('/')[1]}_s{s}"; out = []
    for f in ("origin_z", "best"):
        q = np.load(d / f"{f}.npz")["qpos"].astype(np.float64); T = min(len(q), len(ref)); _, D = B.part_floor(q[:T])
        out.append(contact_fall_time(D, Dr[:T]))
    return job, out


def contactfall():
    """contactfall --drop <stem> ...: Fall Time as on the Task page (scripts/fall_definitions.py 'contact'): seconds with a non-foot body
    within 1 cm of the floor while the reference's same body is more than 5 cm above it -> means_contactfall_without_<stems>.json"""
    drop = sys.argv[sys.argv.index("--drop") + 1:] if "--drop" in sys.argv else []
    clips = [c for c in CLIPS if c.split("/")[1] not in drop]
    jobs = [(b, n, c, s) for b in BODY for n, _, _ in COND for c in clips for s in SEEDS]
    with Pool(workers(16)) as p:
        R = dict(p.map(contact_one, jobs))
    rep = dict(dropped=drop, clips=[c.split("/")[1] for c in clips])
    for b in BODY:
        rep[b] = {}
        for n, _, _ in COND:
            z = np.array([R[(b, n, c, 0)][0] for c in clips]); x = np.array([[R[(b, n, c, s)][1] for s in SEEDS] for c in clips])
            rep[b][n] = dict(z0=float(z.mean()), best=float(x.mean()), per_clip={c.split("/")[1]: dict(z0=float(z[i]), best=float(x[i].mean())) for i, c in enumerate(clips)})
            print(f"{b:9s} {n:12s} frozen {z.mean():.3f} -> LOES {x.mean():.3f}")
    tag = "_".join(drop) or "none"; write_json(RES / f"means_contactfall_without_{tag}.json", rep)


SCALE = {b: BODIES.scale(b) for b in ("child", "g1", "limit_leg", "heavy_arm")}   # body / adult: ExactObs's s, RobotMap("g1").s, 1, 1
if MILD: SCALE["tall"] = 1.364408905613045                                   # single_z_search's root scale of len1.4_wid0.8 (the same on all 6 clips; = each run's pos_div, checked)


def phc_one(job):
    import eval_metrics as E
    b, name, c, s = job
    if b not in body:
        body[b] = metric_body(E.XML["g1"] if b == "g1" else BODY[b]["xml"], b != "g1")
    B = body[b]; m, dd = B.m, B.d; ref = ref_of(b, c); d = OUT / f"{b}_{name}/{c.split('/')[1]}_s{s}"
    Pr = world_positions(m, dd, ref); out = []
    for f in ("origin_z", "best"):
        q = np.load(d / f"{f}.npz")["qpos"].astype(np.float64); T = min(len(q), len(ref))
        out.append(phc_success(mean_body_error(world_positions(m, dd, q[:T]), Pr[:T]), SCALE[b]))
    return job, out


def phcsuccess():
    """Success Rate as on the Task page (PHC's test): the clip fails if the mean world-frame body position error ever exceeds
    0.5 m in adult-equivalent metres (0.5 x SCALE on the body) -> means_phcsuccess.json"""
    jobs = [(b, n, c, s) for b in BODY for n, _, _ in COND for c in CLIPS for s in SEEDS]
    with Pool(workers(16)) as p:
        R = dict(p.map(phc_one, jobs))
    rep = dict(clips=[c.split("/")[1] for c in CLIPS], threshold_m={b: 0.5 * v for b, v in SCALE.items()})
    for b in BODY:
        rep[b] = {}
        for n, _, _ in COND:
            z = np.array([R[(b, n, c, 0)][0] for c in CLIPS]); x = np.array([[R[(b, n, c, s)][1] for s in SEEDS] for c in CLIPS])
            rep[b][n] = dict(z0=float(z.mean()), best=float(x.mean()), per_clip={c.split("/")[1]: dict(z0=int(z[i]), best=int(x[i].sum())) for i, c in enumerate(CLIPS)},
                             seed=[float(v) for v in x.mean(0)])
            print(f"{b:9s} {n:12s} frozen {100 * z.mean():5.1f}% -> LOES {100 * x.mean():5.1f}% | per clip {[int(v) for v in x.sum(1)]}")
    write_json(RES / "means_phcsuccess.json", rep)


if __name__ == "__main__":
    {"prepare": prepare, "commands": commands, "analyse": analyse, "summary": summary, "handfall": handfall, "contactfall": contactfall,
     "phcsuccess": phcsuccess, "table": table}[sys.argv[1]]()
