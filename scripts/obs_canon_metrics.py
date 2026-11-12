#!/usr/bin/env python3
"""obs_canon_metrics.py -- the paper's "Effect of Geometry-Aware Canonicalization" evidence with the metrics the paper reports
(user, 11/12: the old text used a per-joint RMS angle error; docs/latex/observation_canonicalization.tex).

The 52 child task motions (outputs/child_52/clips.txt), the torque-calibrated child, the frozen task latent z0 (no search), one
rollout per motion and mapping, the Task page's child rollout settings (outputs/child_52_rerun: --pairs 8, start from the
reference's frame 0 with its velocity, the state after step t scored against reference frame t + 1). Mappings of the child's
state to the adult-trained policy (single_z_search.py --obs-scale):
  raw       none    the child's own proprioception
  perbody   auto    per-body length scaling: every body's length features / that body's child-to-adult length ratio
                    (model/obs_scale.py, the code's "multiplier")
  exact     exact   Phi_beta, the geometric canonicalizer: the adult-equivalent state (model/exact_obs.py)
  exact_dz  exact + --exact-clearance-vel: Phi_beta whose root vertical velocity also gets d/dt of the ground-clearance height
                    correction c = qa[2] - qc[2] / s (backward difference over the 1/30 s control step, c of the previous step;
                    0 at the start state; computed in the env workers, model/exact_obs.py clearance_vel). Appendix-B variant.
Rollouts: raw, perbody, exact from outputs/obs_canon_52/<mapping>/<stem>/eval_qpos.npz (scripts/obs_canonicalization_check.py;
re-rolled with the current code on 4 of them: bit-identical); exact_dz from outputs/obs_canon_metrics/rollouts/exact_dz/<stem>.
Metrics (the Task page's, scripts/loes_noanchor.py stats and scripts/phc_reward_tables.py), rollout frame t vs reference frame t + 1:
  success      PHC's criterion: the world-frame mean body-position error never above 0.5 m adult-equivalent (0.5 x 0.611 m on the child)
  phc          PHC tracking reward, 1 - model/phc_reward.PHCCost(child, ref[0], ref[1:], scale=0.611036)(q, ref[0]) (0-1, higher better)
  mpjpe        mean per-joint position error in the root's heading frame (cm)
  mmpjpe       each body's largest position error over the motion, averaged over bodies (cm)
  mpjae        mean per-joint acceleration error (m/s^2; second difference of the heading-frame positions x 30^2)
  cosine       representation check: each motion's reference frames observed on the child under the mapping, embedded by the frozen
               backward map B, cosine against B of the adult's own motion at the same frame, mean over frames
               (outputs/obs_canon_52/cosine.json; exact_dz: computed here, outputs/obs_canon_metrics/cosine_dz.json)
Tests: paired over the 52 motions, Phi_beta (exact) vs each other mapping: Wilcoxon signed-rank (two-sided) for the continuous
metrics, exact McNemar (binomial test on the discordant motions) for success; and on how many motions Phi_beta is better / worse.
exact_dz: also the added vertical-velocity term (adult-equivalent m/s), recomputed from the saved qpos (c depends on qpos only).
Checks: exact rollout = outputs/child_52_rerun/<stem>_es_s0/origin_z.npz bit for bit (and, re-rolled with the modified code, again);
settings equal across mappings but --obs-scale; z0 the same; exact's means = the Task page's z0 numbers (outputs/noanchor,
outputs/phc_reward_tables); ExactObs default unchanged (vs git HEAD) and its clearance_vel option in the workers = a sequential
recomputation (check_dz).
  commands   "<out dir> <command>": exact_dz rollouts and the default-exact re-roll check (outputs/obs_canon_metrics/rollouts/)
  check_dz   ExactObs checks -> outputs/obs_canon_metrics/check_dz.json
  cosine     exact_dz (and exact again) backward-embedding cosine -> outputs/obs_canon_metrics/cosine_dz.json (GPU)
  analyse    -> outputs/obs_canon_metrics/{results.json, README.md}
usage: uv run python scripts/obs_canon_metrics.py commands | check_dz | cosine | analyse"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from crossenbodiment import paths                                         # noqa: E402
# OOP refactor (ported from the main tree, 11/12): inputs are read through paths.read_path ($CE_OUTPUTS_ROOT, default outputs/), this
# script's own earlier files (check_dz.json, cosine_dz.json) through paths.own_path, and its results written under $CE_RESULTS_ROOT
# (paths.write_path, default outputs/); "commands" still prints (and creates) the repository-relative run folders. ExactObs's default
# is checked against model/exact_obs.py at BEFORE, the code before the clearance_vel option (the main tree's HEAD, = refactor/base's).
OLD = paths.read_path("outputs/obs_canon_52"); OUT_REL = "outputs/obs_canon_metrics"; OUT = paths.read_path(OUT_REL); RUN = OUT / "rollouts"
BEFORE = "8cc7d89"
XML = "assets/robots_torque/child/robot_torque_full.xml"; ADULT = "assets/robots/adult/robot.xml"; S = 0.611036; DT = 1.0 / 30
MAPS = ("raw", "perbody", "exact"); ALL = MAPS + ("exact_dz",)
NAMES = dict(raw="raw observations", perbody="per-body length scaling", exact="Phi_beta", exact_dz="Phi_beta + clearance d/dt")
KEYS = ("success", "phc", "mpjpe", "mmpjpe", "mpjae")
HIGHER = dict(success=True, phc=True, mpjpe=False, mmpjpe=False, mpjae=False, cosine=True)


def out_w():
    """the results folder ($CE_RESULTS_ROOT/obs_canon_metrics), checked to lie inside the repository"""
    return paths.write_path(OUT_REL)


def clips():
    return paths.read_clips(paths.CHILD_52_TASK)          # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)


def family(stem):
    t = stem.rsplit("_", 1)[0]
    for k in ("move-ego-0-2-raisearms", "move-ego-low", "move-ego", "raisearms", "rotate", "crawl", "lieonground", "sitonground",
              "split", "headstand", "crouch", "jump"):
        if t.startswith(k):
            return {"move-ego-0-2-raisearms": "move-arms", "raisearms": "raise-arms"}.get(k, k)
    return t


def old_cmds():
    """outputs/obs_canon_52/commands.txt -> {(mapping, stem): command}"""
    rev = dict(raw="raw", perbody="perbody", uniform="uniform", exact="exact"); C = {}
    for line in open(OLD / "commands.txt"):
        d, cmd = line.strip().split(" ", 1); k, stem = Path(d).parts[-2:]
        C[(rev[k], stem)] = cmd
    return C


def commands():
    C = old_cmds()
    for c in clips():
        st = c.split("/")[1]; base = C[("exact", st)]
        assert f"--out outputs/obs_canon_52/exact/{st}" in base and base.count("--obs-scale exact") == 1
        for k, extra in (("exact_dz", " --exact-clearance-vel"), ("exact_check", "")):
            d = Path(OUT_REL) / "rollouts" / k / st; (REPO / d).mkdir(parents=True, exist_ok=True)
            print(d, base.replace(f"--out outputs/obs_canon_52/exact/{st}", f"--out {d}") + extra)


def corr_of(e, q):
    """ExactObs's height correction c of the child pose q (depends on qpos only): e has clearance_vel on; returns c"""
    e.reset(); e.adult_state(q, np.zeros(e.mc.nv)); return e._prev


def added_terms(e, q, q_prev0):
    """the added vertical velocity of each state of the sequence q (T, nq) given the pose before it (adult-equivalent m/s)"""
    c = np.array([corr_of(e, x) for x in np.concatenate([q_prev0[None], q])])
    return np.diff(c) / DT


def check_dz():
    import functools, importlib.util, tempfile
    import mujoco
    from humenv import make_humenv
    from model.exact_obs import ExactObs, ExactObsWrapper
    res = {}
    src = subprocess.run(["git", "show", f"{BEFORE}:model/exact_obs.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("exact_obs_head", f.name); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    fk = mujoco.MjModel.from_xml_path(str(REPO / XML)); new, old = ExactObs(REPO / XML, REPO / ADULT), H.ExactObs(REPO / XML, REPO / ADULT)
    rng = np.random.default_rng(0); n = 0; same = True
    for c in clips()[::4]:                                   # default (clearance_vel off) vs git HEAD, reference states
        ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); v = np.zeros(fk.nv)
        for t in range(1, len(ref), 7):
            mujoco.mj_differentiatePos(fk, v, DT, ref[t - 1], ref[t]); v2 = v + rng.normal(0, 0.1, fk.nv)
            a, b = new(ref[t], v2), old(ref[t], v2); qa, va = new.adult_state(ref[t], v2); qb, vb = old.adult_state(ref[t], v2)
            same &= bool(np.array_equal(a, b) and np.array_equal(qa, qb) and np.array_equal(va, vb)); n += 1
    res["default_vs_head"] = dict(states=n, bit_identical=same)
    c = "crawl-0.4-0-d/crawl-0.4-0-d_1"; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)
    v0 = np.zeros(fk.nv); mujoco.mj_differentiatePos(fk, v0, DT, ref[0], ref[1]); T, N = 60, 2
    env, _ = make_humenv(num_envs=N, vectorization_mode="async", task=None, max_episode_steps=300, xml=str(REPO / XML), state_init="Default",
                         wrappers=[functools.partial(ExactObsWrapper, child_xml=str(REPO / XML), adult_xml=str(REPO / ADULT), clearance_vel=True)])
    A = rng.uniform(-0.4, 0.4, (2, T, N, fk.nu)); ok_obs, ok_add, ok_post, mx = True, True, 0.0, 0.0
    for rep in range(2):                                      # twice: reset must clear the previous rollout's correction
        env.reset(); env.call("set_physics", qpos=ref[0], qvel=v0); W, Q, V, D = [], [], [], []
        for t in range(T):
            _, _, _, _, info = env.step(A[rep, t])
            W.append(np.array(info["exact_proprio"])); Q.append(np.array(info["qpos"])); V.append(np.array(info["qvel"])); D.append(np.array(info["clearance_vel_added"]))
        W, Q, V, D = (np.stack(x, 1) for x in (W, Q, V, D))
        for k in range(N):
            e = ExactObs(REPO / XML, REPO / ADULT, clearance_vel=True); e.adult_state(ref[0], v0)        # primed on the start state
            seq = np.stack([e(Q[k, t], V[k, t]) for t in range(T)]); ok_obs &= bool(np.array_equal(seq, W[k]))
            e2 = ExactObs(REPO / XML, REPO / ADULT, clearance_vel=True); add = []
            e2.adult_state(ref[0], v0)
            for t in range(T):
                e2.adult_state(Q[k, t], V[k, t]); add.append(e2.added)
            ok_add &= bool(np.array_equal(np.array(add), D[k]))
            post = added_terms(ExactObs(REPO / XML, REPO / ADULT, clearance_vel=True), Q[k], ref[0])
            ok_post = max(ok_post, float(np.abs(post - D[k]).max())); mx = max(mx, float(np.abs(D[k]).max()))
    env.close()
    e = ExactObs(REPO / XML, REPO / ADULT, clearance_vel=True); first = e.adult_state(ref[0], v0)[1][2] - ExactObs(REPO / XML, REPO / ADULT).adult_state(ref[0], v0)[1][2]
    res["clearance_vel_in_workers"] = dict(clip=c, steps=T, envs=N, rollouts=2, obs_equal_sequential=ok_obs, added_equal_sequential=ok_add,
                                           added_vs_qpos_recompute_max_abs=ok_post, max_abs_added=mx, first_call_adds=float(first))
    W = out_w(); W.mkdir(parents=True, exist_ok=True); (W / "check_dz.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def cosine():
    import mujoco, torch
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.exact_obs import ExactObs
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda"); model.eval()
    enva, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / ADULT), state_init="Default")
    fk = mujoco.MjModel.from_xml_path(str(REPO / XML)); res = {}
    cos = lambda a, b: float((np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).mean())
    with torch.no_grad():
        for c in clips():
            st = c.split("/")[1]
            rc = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)
            ra = np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"].astype(np.float64)
            Ba = bfm_align.embed(model, bfm_align.obs_from_qpos(enva, ra, bfm_align.DEFAULT_DT, None), "cuda"); r = {}
            for k, on in (("exact", False), ("exact_dz", True)):     # as obs_canonicalization_check.py: the reference, differenced
                ex = ExactObs(REPO / XML, REPO / ADULT, clearance_vel=on); vr, ro = np.zeros(fk.nv), []
                for t in range(len(rc)):
                    if t:
                        mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, rc[t - 1], rc[t])
                    ro.append(ex(rc[t], vr))
                r[k] = cos(bfm_align.embed(model, np.stack(ro), "cuda"), Ba)
            res[st] = r; print(st, {k: round(v, 6) for k, v in r.items()}, flush=True)
    W = out_w(); W.mkdir(parents=True, exist_ok=True); (W / "cosine_dz.json").write_text(json.dumps(res, indent=1))


_C = {}


def job(arg):
    """all metrics of one (mapping, motion) rollout"""
    import mujoco
    from crossenbodiment import eval_metrics as E             # scripts/eval_metrics.py re-exports this module (refactor step 8)
    from loes_noanchor import stats
    from model.phc_reward import PHCCost
    k, c = arg; st = c.split("/")[1]
    if "body" not in _C:
        _C["body"] = E.Body(E.XML["child"], True); _C["m"] = mujoco.MjModel.from_xml_path(str(REPO / XML))
    body = _C["body"]; full = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); ref = full[1:]
    Pw = np.empty((len(ref), body.m.nbody - 1, 3))
    for t, q in enumerate(ref):
        body.d.qpos[:] = q; mujoco.mj_kinematics(body.m, body.d); Pw[t] = body.d.xpos[1:]
    _, Dr = body.part_floor(ref)
    p = (RUN / "exact_dz" / st if k == "exact_dz" else OLD / k / st) / "eval_qpos.npz"; q = np.load(p)["qpos"][0]
    fall, succ, mpjpe, mmpjpe, mpjae, drift = stats(body, q, ref, Pw, Dr, S)
    phc = PHCCost(_C["m"], full[0], full[1:], scale=S); r = dict(success=succ, phc=1.0 - float(phc(q.astype(np.float64)[None], full[0])[0]),
                                                               mpjpe=mpjpe, mmpjpe=mmpjpe, mpjae=mpjae, fall_s=fall, drift=drift)
    chk = {}
    if k == "exact":
        o = np.load(paths.read_path(f"outputs/child_52_rerun/{st}_es_s0/origin_z.npz"))["qpos"]
        chk["vs_rerun_origin_z"] = bool(o.shape == q.shape and o.dtype == q.dtype and np.array_equal(o, q))
        rr = RUN / "exact_check" / st / "eval_qpos.npz"
        chk["reroll_now_vs_rerun"] = bool(np.array_equal(np.load(rr)["qpos"][0], o)) if rr.exists() else None
    if k == "exact_dz":
        from model.exact_obs import ExactObs
        if "ex" not in _C:
            _C["ex"] = ExactObs(REPO / XML, REPO / ADULT, clearance_vel=True)
        r["added_rollout"] = np.abs(added_terms(_C["ex"], q.astype(np.float64), full[0])).tolist()       # per step, adult m/s
        r["added_ref"] = np.abs(added_terms(_C["ex"], full[1:], full[0])).tolist()                        # along the reference
    return k, st, r, chk


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as W
    d = np.asarray(a) - np.asarray(b)
    if not np.any(d):
        return dict(W=None, p=1.0)
    w = W(a, b); return dict(W=float(w.statistic), p=float(w.pvalue))


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(only_first=n10, only_second=n01, p=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0)


def compare(R, C, a, b, keys):
    """paired a vs b over the motions: per metric the test and how many motions a is better / worse / tied"""
    st = sorted(R[a]); out = {}
    for m in keys:
        x = np.array([(C[s][a] if m == "cosine" else R[a][s][m]) for s in st]); y = np.array([(C[s][b] if m == "cosine" else R[b][s][m]) for s in st])
        better = int(((x > y) if HIGHER[m] else (x < y)).sum()); worse = int(((x < y) if HIGHER[m] else (x > y)).sum())
        t = mcnemar(x, y) if m == "success" else wilcoxon(x, y)
        out[m] = dict(mean_a=float(x.mean()), mean_b=float(y.mean()), median_a=float(np.median(x)), median_b=float(np.median(y)),
                      mean_diff=float((x - y).mean()), better=better, worse=worse, tied=len(st) - better - worse,
                      test="McNemar exact" if m == "success" else "Wilcoxon signed-rank", **t)
    return out


def analyse():
    from multiprocessing import Pool
    jobs = [(k, c) for k in ALL for c in clips()]
    with Pool(4) as p:
        res = p.map(job, jobs, chunksize=2)
    R = {k: {} for k in ALL}; chk = dict(exact_vs_rerun_origin_z={}, exact_reroll_now={})
    for k, st, r, ch in res:
        R[k][st] = r
        if k == "exact":
            chk["exact_vs_rerun_origin_z"][st] = ch["vs_rerun_origin_z"]; chk["exact_reroll_now"][st] = ch["reroll_now_vs_rerun"]
    st = sorted(R["exact"]); assert len(st) == 52
    # settings: the three mappings' commands differ only in --obs-scale (and their own paths); z0 identical; the Task page's settings
    Cm = old_cmds(); sett = []
    for s in st:
        norm = {k: Cm[(k, s)].replace(f"obs_canon_52/{k}/", "obs_canon_52/X/").replace(f"--obs-scale {dict(raw='none', perbody='auto', exact='exact')[k]}", "--obs-scale Y") for k in MAPS}
        z = [np.load(OLD / k / s / "z0.npy") for k in MAPS]
        sett.append(len(set(norm.values())) == 1 and all(np.array_equal(z[0], x) for x in z[1:]) and "--pairs 8" in norm["exact"] and "--init-vel-scale 1" in norm["exact"])
    rs = [json.load(open(paths.read_path(f"outputs/child_52_rerun/{s}_es_s0/summary.json"))) for s in st]
    chk["settings_equal_but_obs_scale_and_same_z0"] = all(sett)
    chk["task_page_child_runs"] = dict(pairs=sorted({r["pairs"] for r in rs}), init=sorted({r["init"] for r in rs}), init_vel_scale=sorted({r["init_vel_scale"] for r in rs}),
                                       frame_offset=sorted({r["frame_offset"] for r in rs}), obs_scale=sorted({r["obs_scale"] for r in rs}))
    rr = [bool(np.array_equal(np.load(OUT / "reroll_check" / k / s / "eval_qpos.npz")["qpos"], np.load(OLD / k / s / "eval_qpos.npz")["qpos"]))
          for k in ("raw", "perbody") for s in ("move-ego-0-0_2", "jump-2_3") if (OUT / "reroll_check" / k / s / "eval_qpos.npz").exists()]
    chk["raw_perbody_rerolled_now_bit_identical"] = f"{sum(rr)}/{len(rr)}"
    dzs = [json.loads((RUN / "exact_dz" / s / "eval_base.json").read_text()) for s in st]
    chk["exact_dz_runs"] = len(dzs)
    na = json.load(open(paths.read_path("outputs/noanchor/results.json")))["child"]["frozen"]; ph = json.load(open(paths.read_path("outputs/phc_reward_tables/per_motion.json")))["child"]["best"]["64"]["frozen"]
    mean = lambda k, m: float(np.mean([R[k][s][m] for s in st]))
    chk["exact_means_vs_task_page"] = {m: [mean("exact", m), na[m]] for m in ("success", "mpjpe", "mmpjpe", "mpjae", "fall_s", "drift")}
    chk["exact_phc_vs_phc_reward_tables_max_abs"] = float(max(abs(R["exact"][s]["phc"] - ph[s]) for s in st))
    f_chk, f_cz = paths.own_path(f"{OUT_REL}/check_dz.json"), paths.own_path(f"{OUT_REL}/cosine_dz.json")
    chk["check_dz"] = json.loads(f_chk.read_text()) if f_chk.exists() else None
    C = json.loads((OLD / "cosine.json").read_text()); Cz = json.loads(f_cz.read_text()) if f_cz.exists() else {}
    for s in st:
        if Cz:
            C[s]["exact_dz"] = Cz[s]["exact_dz"]
    chk["cosine_exact_recomputed_max_abs"] = float(max(abs(Cz[s]["exact"] - C[s]["exact"]) for s in st)) if Cz else None
    cos_keys = [k for k in ALL if all(k in C[s] for s in st)]
    means = {k: {**{m: mean(k, m) for m in KEYS + ("fall_s", "drift")}, **({"cosine": float(np.mean([C[s][k] for s in st])), "cosine_min": float(min(C[s][k] for s in st))} if k in cos_keys else {})} for k in ALL}
    tests = {f"exact_vs_{b}": compare(R, C, "exact", b, KEYS + (("cosine",) if b in cos_keys else ())) for b in ("raw", "perbody", "exact_dz")}
    tests["perbody_vs_raw"] = compare(R, C, "perbody", "raw", KEYS + ("cosine",))
    fam = {s: family(s) for s in st}
    spikes = {}
    for w in ("rollout", "ref"):
        mx = {s: float(max(R["exact_dz"][s][f"added_{w}"])) for s in st}; allv = np.concatenate([R["exact_dz"][s][f"added_{w}"] for s in st])
        spikes[w] = dict(per_motion_max=mx, family_max={f: max(mx[s] for s in st if fam[s] == f) for f in sorted(set(fam.values()))},
                         median_of_motion_max=float(np.median(list(mx.values()))), max=float(max(mx.values())), argmax=max(mx, key=mx.get),
                         median_all_steps=float(np.median(allv)), p99_all_steps=float(np.percentile(allv, 99)), frac_steps_above_0p5=float((allv > 0.5).mean()))
    per_motion = {k: {s: {m: v for m, v in R[k][s].items() if not m.startswith("added")} for s in st} for k in ALL}
    W = out_w(); W.mkdir(parents=True, exist_ok=True)
    (W / "results.json").write_text(json.dumps(dict(names=NAMES, means=means, tests=tests, spikes={w: {k: v for k, v in x.items() if k != "per_motion_max"} for w, x in spikes.items()},
                                                    spikes_per_motion={w: x["per_motion_max"] for w, x in spikes.items()}, per_motion=per_motion, cosine=C, family=fam, checks=chk), indent=1))
    # README
    L = ["Paper: docs/latex/observation_canonicalization.tex (old version: observation_canonicalization_old.tex), its figure environment "
         "docs/latex/fig_obs_canon.tex; figure docs/figures/obs_canonicalization.{pdf,png} by scripts/plot_obs_canon.py. "
         "Rollouts: exact_dz and the default-exact re-roll check in rollouts/ (commands.txt, run_queue.sh); reroll_check/: 4 raw / per-body re-rolls.",
         "\n## Checks", f"- exact rollout = child_52_rerun origin_z bit for bit: {sum(chk['exact_vs_rerun_origin_z'].values())}/52; re-rolled now (modified code, "
         f"clearance_vel off) = origin_z: {sum(1 for v in chk['exact_reroll_now'].values() if v)}/{sum(1 for v in chk['exact_reroll_now'].values() if v is not None)}",
         f"- raw / per-body rollouts re-rolled with the current code, bit-identical: {chk['raw_perbody_rerolled_now_bit_identical']}",
         f"- commands equal but --obs-scale, z0 identical: {chk['settings_equal_but_obs_scale_and_same_z0']}; Task page child runs: {chk['task_page_child_runs']}",
         "- exact means vs the Task page's z0 (outputs/noanchor child frozen): " + ", ".join(f"{m} {a:.6g} / {b:.6g}" for m, (a, b) in chk["exact_means_vs_task_page"].items()),
         f"- exact PHC reward vs phc_reward_tables per motion, max |diff|: {chk['exact_phc_vs_phc_reward_tables_max_abs']:.2g}",
         f"- exact cosine recomputed vs obs_canon_52/cosine.json, max |diff|: {chk['cosine_exact_recomputed_max_abs']}",
         f"- ExactObs checks (check_dz.json): {json.dumps(chk['check_dz'])}",
         "\n## Means over the 52 motions (z0, one rollout each)\n",
         "| mapping | success | PHC reward | MPJPE (cm) | MMPJPE (cm) | MPJAE (m/s^2) | B cosine (min) |", "|---|---|---|---|---|---|---|"]
    for k in ALL:
        v = means[k]
        L.append(f"| {NAMES[k]} | {100 * v['success']:.1f}% ({round(52 * v['success'])}/52) | {v['phc']:.3f} | {v['mpjpe']:.2f} | {v['mmpjpe']:.2f} | {v['mpjae']:.2f} | "
                 + (f"{v['cosine']:.4f} ({v['cosine_min']:.4f}) |" if "cosine" in v else "- |"))
    L.append("\n## Paired tests (52 motions)\n\n| comparison | metric | mean A | mean B | A better / worse / tied | test | p |\n|---|---|---|---|---|---|---|")
    for name, T in tests.items():
        a, b = name.split("_vs_")
        for m, t in T.items():
            f = (lambda x: f"{100 * x:.1f}%") if m == "success" else (lambda x: f"{x:.4f}") if m in ("phc", "cosine") else (lambda x: f"{x:.2f}")
            extra = f" (only A {t['only_first']}, only B {t['only_second']})" if m == "success" else ""
            L.append(f"| {NAMES[a]} vs {NAMES[b]} | {m} | {f(t['mean_a'])} | {f(t['mean_b'])} | {t['better']} / {t['worse']} / {t['tied']}{extra} | {t['test']} | {t['p']:.2g} |")
    L.append("\n## By task family (means)\n\n| family | n | " + " | ".join(f"{NAMES[k]}: success, PHC, MPJPE" for k in ALL) + " |\n|---|---|" + "---|" * len(ALL))
    for f in sorted(set(fam.values())):
        ss = [s for s in st if fam[s] == f]
        L.append(f"| {f} | {len(ss)} | " + " | ".join(f"{100 * np.mean([R[k][s]['success'] for s in ss]):.0f}%, {np.mean([R[k][s]['phc'] for s in ss]):.3f}, {np.mean([R[k][s]['mpjpe'] for s in ss]):.2f}" for k in ALL) + " |")
    L.append("\n## Clearance derivative: size of the added vertical-velocity term |dc/dt| (adult-equivalent m/s)\n")
    for w, lab in (("rollout", "z0 rollouts with it"), ("ref", "the child references (frame 0 adds 0)")):
        x = spikes[w]
        L.append(f"- {lab}: median over motions of the per-motion max {x['median_of_motion_max']:.3f}; largest {x['max']:.3f} ({x['argmax']}); "
                 f"median over all steps {x['median_all_steps']:.4f}, 99th percentile {x['p99_all_steps']:.3f}, steps above 0.5 m/s {100 * x['frac_steps_above_0p5']:.2f}%")
        L.append("  - per family, max: " + ", ".join(f"{f} {v:.2f}" for f, v in x["family_max"].items()))
    L.append("\n## Per motion: PHC reward, MPJPE (cm), success -- raw / per-body / Phi_beta / Phi_beta + d/dt\n")
    for s in st:
        L.append(f"- {s}: " + " / ".join(f"{R[k][s]['phc']:.3f}" for k in ALL) + " | " + " / ".join(f"{R[k][s]['mpjpe']:.2f}" for k in ALL)
                 + " | " + "".join("S" if R[k][s]["success"] else "-" for k in ALL))
    (W / "README.md").write_text("# Observation canonicalization with the reported metrics\n\n" + __doc__ + "\n\n" + "\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    {"commands": commands, "check_dz": check_dz, "cosine": cosine, "analyse": analyse}[sys.argv[1]]()
