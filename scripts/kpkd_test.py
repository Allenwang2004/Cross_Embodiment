#!/usr/bin/env python3
"""kpkd_test.py -- does the latent search still work when kp and kd move in opposite directions? The child, the G1, the limited leg (right leg locked, bent; tracks the bent-leg reference) and the payload (right hand x 80; tracks the adult's motion), kp x 0.8 with kd x 1.2, kp x 1.2 with kd x 0.8, and both unchanged, on the 8 clips of the k-scale test (outputs/k_scale), 5 seeds, LOES with 1,024 rollouts and the current settings (reference-velocity start, q[t] scored against ref[t + 1], independent perturbations, no extra rollouts of the mean). Torque limits unchanged. An opposite move changes each joint's damping ratio by kd / sqrt(kp): x 1.34 and x 0.73.
  prepare   writes each human body's XML with the gains moved (scripts/scale_body_gains.py), next to its own folder as <folder>_kp*_kd*, and assets/robots/g1/gains_perjoint_kp*_kd*.json
  commands  prints "<out dir> <command>" per search (run through xargs -P 8), into outputs/kpkd/<body>_<condition>/<stem>_s<seed>
  analyse   cost and metrics of each run's z0 and best-so-far rollout (scripts/eval_metrics.py, on the body itself), for the bodies whose runs are all finished; writes outputs/kpkd/{per_run.npz, means.json}. Checks: settings, z0 rollout identical across seeds, conditions differ, child kp x 1 matches outputs/child_52_rerun where the clip is there.
usage: uv run python scripts/kpkd_test.py prepare | commands | analyse"""
import json, shutil, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment.bodies import BODIES
from crossenbodiment.paths import CHILD_XML_REL
from crossenbodiment.metrics import contact_fall_time, mean_body_error, metric_body, phc_success, world_positions
from crossenbodiment import paths
from crossenbodiment.experiment import Experiment, workers
from crossenbodiment.results import Checks, save_npz, write_json
# refactor step 8: runs (and, for summary, per_run.npz) are read under paths.outputs_root() ($CE_OUTPUTS_ROOT), results written under
# paths.results_root() ($CE_RESULTS_ROOT); both REPO/outputs by default
OUT = paths.output("kpkd"); RES = paths.result("kpkd"); SEEDS = range(5)
BODY = dict(child=dict(xml=CHILD_XML_REL, ref="data/child/retargeting_motion/{c}.npz", label="child"), g1=dict(ref="outputs/g1/roundtrip/{stem}/g1.npz"),
            limit_leg=dict(xml="assets/robots/adult_lockRleg_bent_lim/robot.xml", ref="data/adult_lockRleg_bent_lim/retargeting_motion/{c}.npz", label="adult_lockRleg_bent_lim"),
            heavy_arm=dict(xml="assets/robots/adult_heavyRWristHand_x80/robot.xml", ref="data/origin_motion/{c}.npz", label="adult_heavyRWristHand_x80"))
COND = (("kp1_kd1", 1.0, 1.0), ("kp0.8_kd1.2", 0.8, 1.2), ("kp1.2_kd0.8", 1.2, 0.8), ("kp0.8_kd0.8", 0.8, 0.8), ("kp1.2_kd1.2", 1.2, 1.2))
# 11/08: the user's motion list is these 6; crawl-0.4-2-d_8 and crouch-0_8 were also run for the first three conditions (outputs/kpkd,
# means_8clips.json) and are left out, so the same-ratio conditions (kp and kd x 0.8 or x 1.2 together) were run on the 6 only
CLIPS_8 = ["crawl-0.4-2-d/crawl-0.4-2-d_8", "crouch-0/crouch-0_8", "headstand/headstand_4", "jump-2/jump-2_6", "jump-2/jump-2_7", "move-ego-0-2/move-ego-0-2_2", "move-ego-0-4/move-ego-0-4_8", "rotate-x--5-0.8/rotate-x--5-0.8_4"]
CLIPS = [c for c in CLIPS_8 if not c.startswith(("crawl-0.4-2-d/", "crouch-0/"))]
def XML(b, name):
    p = Path(BODY[b]["xml"])
    return str(p) if name == "kp1_kd1" else str(p.parent.parent / f"{p.parent.name}_{name}" / p.name)


GAINS = lambda name: "assets/robots/g1/gains_perjoint.json" if name == "kp1_kd1" else f"assets/robots/g1/gains_perjoint_{name}.json"
# the run folders: outputs/kpkd/<body>_<condition>/<stem>_s<seed> (no method key: every run is LOES)
EXP = Experiment("outputs/kpkd", clips=CLIPS, methods=[n for n, _, _ in COND], seeds=SEEDS, bodies=tuple(BODY),
                 name=lambda r: f"{r.body}_{r.method}/{r.stem}_s{r.seed}", order=("body", "method", "clip", "seed"), sep=" ")
body = {}


def prepare():
    g = json.loads((REPO / GAINS("kp1_kd1")).read_text())
    for name, a, b in COND[1:]:
        for body_ in ("child", "limit_leg", "heavy_arm"):
            if (REPO / XML(body_, name)).exists():                       # made earlier: keep it as it is
                continue
            subprocess.run(["uv", "run", "python", "scripts/scale_body_gains.py", "--src", BODY[body_]["xml"], "--kp", str(a), "--kd", str(b), "--out", XML(body_, name)], cwd=REPO, check=True)
            for f in ("parameter.json", "skeleton.json"):
                if (REPO / BODY[body_]["xml"]).with_name(f).exists():
                    shutil.copy(REPO / Path(BODY[body_]["xml"]).with_name(f), REPO / Path(XML(body_, name)).with_name(f))
        if (REPO / GAINS(name)).exists():
            continue
        h = dict(g, note=f"assets/robots/g1/gains_perjoint.json with kp x {a} and kd x {b} (scripts/kpkd_test.py); torque limits and armature unchanged. " + g["note"], kp_scale=a, kd_scale=b, kp={k: v * a for k, v in g["kp"].items()}, kd={k: v * b for k, v in g["kd"].items()})
        (REPO / GAINS(name)).write_text(json.dumps(h, indent=1))


def command(r):
    b, name, c, s, d = r.body, r.method, r.clip, r.seed, r.dir
    if b != "g1":
        rf = "" if b == "child" else f" --ref {ref_path(b, c)}"
        return f"uv run python scripts/single_z_search.py --clip {c} --body {BODY[b]['label']} --xml {XML(b, name)}{rf} --objective bfm --evals 1024 --pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{c}.npy --anchor-weight 0.3 --seed {s} --eval-every 0 --init-vel-scale 1 --out {d}"
    return f"uv run python scripts/robot_z_search.py --robot g1 --clip {c} --fill ridge --ik-weight 1.0 --evals 1024 --init-vel reference --eval-every 0 --gains {GAINS(name)} --seed {s} --out {d}"


def commands():
    EXP.print_commands(command)


def ref_path(b, c):
    return BODY[b]["ref"].format(c=c, stem=c.split("/")[1])


def ref_of(b, c):
    return np.load(REPO / ref_path(b, c))["qpos"].astype(np.float64)[1:]


def one(job):
    import eval_metrics as E
    from crossenbodiment.metrics import LEGACY_KEYS as KEYS, legacy_metrics as full   # child_52_means' (refactor step 4)
    b, name, c, s = job
    if b not in body:
        body[b] = metric_body(E.XML["g1"] if b == "g1" else BODY[b]["xml"], b != "g1")
    d = OUT / f"{b}_{name}/{c.split('/')[1]}_s{s}"; S = json.load(open(d / "summary.json")); ref = ref_of(b, c); base = S["origin_z"]["cost"]; bq = np.load(d / "best.npz")["qpos"].astype(np.float64); zq = np.load(d / "origin_z.npz")["qpos"].astype(np.float64)
    vz, vb = full(body[b], zq, ref, 1.0), full(body[b], bq, ref, S["best"]["cost"] / base)
    bad = Checks.settings(dict(evals=1024, seed=s, frame_offset=1, sampling="independent", sigma=0.05, lr=0.03, heading_weight=1.0, pos_weight=0.1, anchor_weight=0.3), S)
    bad += [k for k, want in (("init_vel_scale", 1.0), ("obs_scale", "exact"), ("xml", XML(b, name) if b != "g1" else None)) if b != "g1" and S.get(k) != want] + [k for k, want in (("init_vel", "reference"), ("fill", "ridge"), ("ik_weight", 1.0), ("gains", GAINS(name))) if b == "g1" and S.get(k) != want]
    return job, [vz[k] for k in KEYS], [vb[k] for k in KEYS], zq[:len(ref)], base, S["best"]["cost"], bad


def analyse():
    from crossenbodiment.metrics import LEGACY_KEYS as KEYS   # child_52_means.KEYS (refactor step 4)
    done = lambda b: EXP.all_done(EXP.runs(body=(b,)))
    BODIES = [b for b in BODY if done(b)]; print("bodies finished:", BODIES, flush=True)
    jobs = [(b, n, c, s) for b in BODIES for n, _, _ in COND for c in CLIPS for s in SEEDS]
    with Pool(workers(16)) as p:
        res = p.map(one, jobs)
    R = {r[0]: r for r in res}; shape = (len(CLIPS), len(SEEDS), len(KEYS)); Z = {}; B = {}; CZ = {}; CB = {}; bad = [(r[0], r[6]) for r in res if r[6]]; spread = 0.0; same_across = 0
    for b in BODIES:
        for n, _, _ in COND:
            z, x = np.zeros(shape), np.zeros(shape)
            for i, c in enumerate(CLIPS):
                for s in SEEDS:
                    r = R[(b, n, c, s)]; z[i, s], x[i, s] = r[1], r[2]; spread = max(spread, float(np.abs(r[3] - R[(b, n, c, 0)][3]).max()))
            Z[b, n], B[b, n] = z, x; CZ[b, n] = np.array([R[(b, n, c, 0)][4] for c in CLIPS]); CB[b, n] = np.array([[R[(b, n, c, s)][5] for s in SEEDS] for c in CLIPS])
        for c in CLIPS:
            q = [R[(b, n, c, 0)][3] for n, _, _ in COND]; same_across += int(np.abs(q[0] - q[1]).max() == 0 or np.abs(q[0] - q[2]).max() == 0)
    m52 = 0.0; n52 = 0
    for c in CLIPS:
        st = c.split("/")[1]; o = paths.output(f"child_52_rerun/{st}_es_s0/best_z.npy")
        if o.exists():
            for s in SEEDS:
                m52 = max(m52, float(np.abs(np.load(paths.output(f"child_52_rerun/{st}_es_s{s}/best_z.npy")) - np.load(OUT / f"child_kp1_kd1/{st}_s{s}/best_z.npy")).max())); n52 += 1
    print("runs:", len(res), "| settings not as intended:", len(bad), bad[:4], "| z0 rollout, largest difference across seeds:", spread, "| clips whose z0 rollout did not change with the gains:", same_across, f"| child kp x 1 vs outputs/child_52_rerun on {n52} runs, largest z difference:", m52, flush=True)
    i = {k: j for j, k in enumerate(KEYS)}; rep = {}
    for b in BODIES:
        rep[b] = {}
        for n, _, _ in COND:
            z, x = Z[b, n], B[b, n]; zc = z[:, 0]
            rep[b][n] = dict(z0=dict({k: float(zc[:, i[k]].mean()) for k in KEYS}, cost=float(CZ[b, n].mean())), best=dict({k: float(x[..., i[k]].mean()) for k in KEYS}, cost=float(CB[b, n].mean())),
                             per_clip={c.split("/")[1]: dict(z0_cost=float(CZ[b, n][j]), best_cost=float(CB[b, n][j].mean()), z0_success=float(zc[j, i["success"]]), z0_mpjpe=float(zc[j, i["mpjpe"]]), z0_fall=float(zc[j, i["fall_s"]]), best_success=int(x[j, :, i["success"]].sum()), best_mpjpe=float(x[j, :, i["mpjpe"]].mean()), best_fall=float(x[j, :, i["fall_s"]].mean()), improved=int(sum(R[(b, n, c, s)][5] < R[(b, n, c, s)][4] for s in SEEDS))) for j, c in enumerate(CLIPS)},
                             seed_success=[float(x[:, s, i["success"]].mean()) for s in SEEDS], seed_mpjpe=[float(x[:, s, i["mpjpe"]].mean()) for s in SEEDS])
    save_npz(RES / "per_run.npz", clips=np.array(CLIPS), keys=np.array(KEYS), **{f"z0_{b}_{n}": Z[b, n] for b, n in Z}, **{f"best_{b}_{n}": B[b, n] for b, n in B}, **{f"cost_z0_{b}_{n}": CZ[b, n] for b, n in CZ}, **{f"cost_best_{b}_{n}": CB[b, n] for b, n in CB})
    write_json(RES / "means.json", rep); print("written")


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
    from crossenbodiment.metrics import body_hand_fall as hand_fall   # child_52_means.hand_fall (refactor step 4)
    b, name, c, s = job
    if b not in body:
        body[b] = metric_body(E.XML["g1"] if b == "g1" else BODY[b]["xml"], b != "g1")
    d = OUT / f"{b}_{name}/{c.split('/')[1]}_s{s}"; ref = ref_of(b, c)
    return job, hand_fall(body[b], np.load(d / "origin_z.npz")["qpos"].astype(np.float64), ref), hand_fall(body[b], np.load(d / "best.npz")["qpos"].astype(np.float64), ref)


def handfall():
    """handfall --drop <stem> ...: Fall Time by hand contact (crossenbodiment.metrics.body_hand_fall), frozen and LOES, by motion (NaN for motions whose
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
     "phcsuccess": phcsuccess}[sys.argv[1]]()
