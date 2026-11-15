"""Tests of OOP refactor step 5 (docs/refactor/PLAN.md): the simulation layer (crossembodiment/policy.py, obs.py, sim.py,
rollout.py) against the rollout loops it replaced, run from their own source at BASE (37ed139, the commit before step 5).

Every copy is executed from `git show BASE:<file>`: a top-level function by its definition (ast, as tests/test_moves.py), a
loop that lived inside a script's main() by its verbatim lines (from the line holding a start marker to the line holding an
end marker, dedented). Each runs a short rollout (STEPS steps, N slots, the golden clip) on the same simulators as the new
code, which must give the same arrays (dtype, shape, every bit: np.array_equal):
  single_z_search.rollout        exact in the workers (the searches), exact in the main process (rollout_z_trace), the obs
                                 multiplier, raw obs from humenv's own reset (batch_z_search): qpos, obs, actions, qvel
  lso_search's rollout loop      latent sequences (16 * Z), no get_obs, ExactObs in the workers: qpos, obs (and B(obs))
  child_reward_eval.rollout      exact and raw bodies: qpos, qvel, actions
  rank_initial_cost.rollout_multi  per-slot start poses: obs
  test_exact_obs's loop          none / mult / exact (main process), per-slot starts, recorded mult and exact obs
  robot_z_search's score loop    the G1 (RobotVecEnv, ridge fill, per-joint gains, --init-vel reference): obs, SMPL qpos,
                                 robot qpos, IK residuals, start qpos
  robot_lso_search's rollout loop  the G1, latent sequences, at rest
plus the batch compositions (copies / padded / with_mean) and the observation factories (single_z_obs, lso_scale_kw).
The G1 workers are spawned before CUDA is touched, as the robot scripts do. Needs GPU 2 (CUDA_VISIBLE_DEVICES=2) and the
golden clip's files (data/, a symlink into the main tree, read only); ~1-2 min.

Run standalone (no pytest in the environment), like tests/test_moves.py:
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 uv run python tests/test_rollout.py
"""
import ast
import os
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

BASE = "37ed139"                                             # the commit before step 5
CLIP = "move-ego-0-2/move-ego-0-2_1"
STEPS, N = 20, 4
ADULT_XML = REPO / "assets/robots/adult/robot.xml"
CHILD_XML = REPO / "assets/robots_torque/child/robot_torque_full.xml"
G1_GAINS = "assets/robots/g1/gains_perjoint.json"
F = {}                                                        # fixtures, made by setup() in main()


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def check(name, a, b):
    assert same(a, b), f"{name}: differs (dtype {np.asarray(a).dtype} / {np.asarray(b).dtype}, shape " \
                       f"{np.asarray(a).shape} / {np.asarray(b).shape}, max |d| " \
                       f"{np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max() if np.asarray(a).shape == np.asarray(b).shape else 'n/a'})"


def old_source(rel):
    return subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def old_defs(rel, names, ns=None):
    """the top-level functions `names` of file `rel` at BASE, executed into a namespace (ns: the globals they need)"""
    ns = dict(ns or {}); ns.setdefault("__name__", f"old_{Path(rel).stem}")
    tree = ast.parse(old_source(rel)); found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), f"{BASE}:{rel}", "exec"), ns); found.add(node.name)
    assert found == set(names), f"{rel}: not found at {BASE}: {sorted(set(names) - found)}"
    return ns


def block(src, start, end, label):
    """the lines of `src` from the first one containing `start` to the next one containing `end`, dedented"""
    lines = src.split("\n")
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next(k for k in range(i, len(lines)) if end in lines[k])
    return compile(textwrap.dedent("\n".join(lines[i:j + 1])) + "\n", label, "exec")


def run_block(rel, start, end, ns, src=None):
    """execute that block of `rel` (at BASE, or `src`) in ns, gradients off (the loops ran under torch.no_grad())"""
    import torch
    code = block(old_source(rel) if src is None else src, start, end, f"{BASE if src is None else 'HEAD'}:{rel}")
    with torch.no_grad():
        exec(code, ns)
    return ns


# ------------------------------------------------------------------ fixtures

def setup():
    from crossembodiment.sim import RobotSim
    F["g1"] = RobotSim("g1", N, fill="ridge", gains=G1_GAINS)            # spawn the robot workers before CUDA is touched
    import functools
    import mujoco
    from humenv import make_humenv
    from crossembodiment.geometry import project_z, unit_sphere
    from crossembodiment.policy import MotivoPolicy
    from model.exact_obs import ExactObs, ExactObsWrapper
    from model.obs_scale import build_obs_multiplier
    F["policy"] = MotivoPolicy.load(device="cuda")
    F["dev"] = "cuda"
    F["ref"] = np.load(REPO / f"data/child/retargeting_motion/{CLIP}.npz")["qpos"]
    F["ref_adult"] = np.load(REPO / f"data/origin_motion/{CLIP}.npz")["qpos"]
    fk = mujoco.MjModel.from_xml_path(str(CHILD_XML)); F["nv"] = fk.nv
    v0 = np.zeros(fk.nv); mujoco.mj_differentiatePos(fk, v0, 1.0 / 30, F["ref"][0], F["ref"][1]); F["v0"] = v0
    z0 = project_z(np.load(REPO / f"data/origin_z/{CLIP}.npy").reshape(-1).astype(np.float64))
    rng = np.random.default_rng(20261111)
    F["Z"] = project_z(z0[None] + 0.8 * rng.standard_normal((N, 256)))                         # (N, 256) float64, |z| = 16
    F["Zseq"] = unit_sphere(z0[None, None] / 16 + 0.0125 * rng.standard_normal((N, STEPS, 256)))  # (N, STEPS, 256), unit
    F["exact"] = ExactObs(CHILD_XML, ADULT_XML)
    F["mul"] = build_obs_multiplier(CHILD_XML, ADULT_XML, mode="auto", parts="length", verbose=False)
    F["env_exact"], _ = make_humenv(num_envs=N, vectorization_mode="async", task=None, max_episode_steps=300, xml=str(CHILD_XML),
                                    state_init="Default", wrappers=[functools.partial(ExactObsWrapper, child_xml=str(CHILD_XML),
                                                                                      adult_xml=str(ADULT_XML))])
    F["env_raw"], _ = make_humenv(num_envs=N, vectorization_mode="async", task=None, xml=str(CHILD_XML), state_init="Default")


def teardown():
    for k in ("env_exact", "env_raw"):
        if k in F:
            F[k].close()
    if "g1" in F:
        F["g1"].close()


def zt():
    import torch
    return torch.as_tensor(F["Z"], dtype=torch.float32, device=F["dev"])


# ------------------------------------------------------------------ 1. single_z_search.rollout

def _single_z(env_key, **kw):
    import torch
    import single_z_search as sz_new
    from crossembodiment.obs import ExactObs, multiplier_or_raw
    from crossembodiment.rollout import BatchRollout
    from crossembodiment.sim import HumenvSim
    old = old_defs("scripts/single_z_search.py", ["rollout"], {"np": np, "torch": torch})["rollout"]
    model, env, dev = F["policy"].model, F[env_key], F["dev"]
    outs = []
    for fn in (old, sz_new.rollout):
        A, V = [], []
        q, o = fn(model, env, zt(), STEPS, dev, kw.get("obs_mul"), init_qpos=kw.get("init_qpos"), nv=F["nv"], return_obs=True,
                  exact=kw.get("exact"), act_out=A, exact_in_env=kw.get("exact_in_env", False), init_qvel=kw.get("init_qvel"),
                  qvel_out=V)
        outs.append((q, o, np.stack(A, axis=1), np.stack(V, axis=1)))
    ex = kw.get("exact")
    mapper = ExactObs.wrap(ex, in_env=kw.get("exact_in_env", False)) if ex is not None else multiplier_or_raw(kw.get("obs_mul"))
    ro = BatchRollout(F["policy"], HumenvSim(env), mapper, N, STEPS, record=("qpos", "obs", "act", "qvel"), nv=F["nv"]).run(
        F["Z"], init_qpos=kw.get("init_qpos"), init_qvel=kw.get("init_qvel"))
    outs.append((ro.qpos, ro.obs.astype(np.float32), ro.act, ro.qvel))
    (q0, o0, a0, v0), rest = outs[0], outs[1:]
    for who, (q, o, a, v) in zip(("single_z_search.rollout", "BatchRollout"), rest):
        check(f"{who} qpos", q, q0); check(f"{who} obs", o, o0); check(f"{who} act", a, a0); check(f"{who} qvel", v, v0)
    assert q0.shape == (N, STEPS, 76) and o0.dtype == np.float32
    return q0


def test_single_z_exact_in_env():
    """the searches: ExactObsWrapper in the workers, start from ref[0] moving (--init-vel-scale 1)"""
    _single_z("env_exact", init_qpos=F["ref"][0], init_qvel=F["v0"], exact=F["exact"], exact_in_env=True)


def test_single_z_exact_main_process():
    """rollout_z_trace: ExactObs computed in the main process, start at rest"""
    _single_z("env_raw", init_qpos=F["ref"][0], exact=F["exact"], exact_in_env=False)


def test_single_z_multiplier():
    """--obs-scale auto: raw obs x the per-body multiplier"""
    _single_z("env_raw", init_qpos=F["ref"][0], obs_mul=F["mul"])


def test_single_z_raw_default_init():
    """batch_z_search: raw obs from humenv's own (T-pose) reset, no start pose"""
    _single_z("env_raw")


# ------------------------------------------------------------------ 2. lso_search's loop

def test_lso_loop():
    import torch
    from crossembodiment.obs import ExactObs
    from crossembodiment.rollout import BatchRollout
    from crossembodiment.sim import HumenvSim
    model, ref, v0 = F["policy"].model, F["ref"], F["v0"]
    ns = dict(np=np, torch=torch, a=types.SimpleNamespace(device=F["dev"], bfm_zero_vel=False), Z=F["Zseq"], env=F["env_exact"],
              ref=ref, v0=v0, ex0=F["exact"](ref[0], v0), n_envs=N, SR=STEPS, model=model, VEL=slice(214, 358))
    run_block("scripts/lso_search.py", "zt = torch.as_tensor(16.0 * Z", "q = np.stack(qs, 1)", ns)
    batch = BatchRollout(F["policy"], HumenvSim(F["env_exact"]), ExactObs.wrap(F["exact"]), N, STEPS, z_scale=16.0,
                         fetch_start_obs=False, record=("qpos", "obs"))
    ro = batch.run(F["Zseq"], init_qpos=ref[0], init_qvel=v0)
    check("qpos", ro.qpos, ns["q"])
    check("obs", torch.as_tensor(ro.obs, dtype=torch.float32, device=F["dev"]).cpu().numpy(), ns["O"].cpu().numpy())
    O = torch.as_tensor(ro.obs, dtype=torch.float32, device=F["dev"])
    check("B(obs)", F["policy"].backward_map(O.reshape(-1, O.shape[-1])).reshape(N, STEPS, -1).cpu().numpy(), ns["Bs"].cpu().numpy())


# ------------------------------------------------------------------ 3. child_reward_eval.rollout

def test_child_reward_eval_rollout():
    import torch
    import child_reward_eval as cre_new
    old = old_defs("scripts/child_reward_eval.py", ["rollout"], {"np": np, "torch": torch})["rollout"]
    model, ref, zs = F["policy"].model, F["ref"], [F["Z"][k] for k in range(N)]
    for env, ex, qv in ((F["env_exact"], F["exact"], None), (F["env_exact"], F["exact"], F["v0"]), (F["env_raw"], None, None)):
        a = old(model, env, zs, ref[0], STEPS, F["nv"], ex, qv)
        b = cre_new.rollout(model, env, zs, ref[0], STEPS, F["nv"], ex, qv)
        for k, x, y in zip(("qpos", "qvel", "act"), b, a):
            check(f"{'exact' if ex is not None else 'raw'} {k}", x, y)


# ------------------------------------------------------------------ 4. rank_initial_cost.rollout_multi

def test_rank_initial_cost_rollout_multi():
    import torch
    import rank_initial_cost as ric_new
    ns = old_defs("scripts/rank_initial_cost.py", ["set_physics_per_slot", "rollout_multi"], {"np": np, "torch": torch})
    inits = [F["ref"][5 * k] for k in range(N)]
    a = ns["rollout_multi"](F["policy"].model, F["env_raw"], zt(), STEPS, F["dev"], F["mul"], inits, F["nv"])
    b = ric_new.rollout_multi(F["policy"].model, F["env_raw"], zt(), STEPS, F["dev"], F["mul"], inits, F["nv"])
    check("obs", b, a)
    assert a.dtype == np.float32 and a.shape == (N, STEPS, 358)


# ------------------------------------------------------------------ 5. test_exact_obs's loop

def test_test_exact_obs_loop():
    import torch
    from crossembodiment.obs import ExactObs as ExactObsMapper, MultiplierObs, RawObs
    from crossembodiment.rollout import BatchRollout
    from crossembodiment.sim import HumenvSim
    set_init_qpos = old_defs("model/simple/train_es.py", ["set_init_qpos"], {"np": np})["set_init_qpos"]
    pad = [(k, "l") for k in range(N)]
    common = dict(np=np, torch=torch, labels={"l": {k: F["Z"][k] for k in range(N)}}, pad=pad, dev=F["dev"],
                  ctx={"env": F["env_raw"]}, smp={k: {"qpos_ref": F["ref"][5 * k:]} for k in range(N)}, bs=N, nv=F["nv"],
                  cfg=types.SimpleNamespace(steps_per_episode=STEPS), model=F["policy"].model, X=F["exact"], mul=F["mul"])
    new_src = (REPO / "scripts/test_exact_obs.py").read_text()
    for mode in ("none", "mult", "exact"):
        old = run_block("scripts/test_exact_obs.py", "Z = torch.tensor(", "Q, OM, OE = np.stack(Q, 1)",
                        dict(common, mode=mode, set_init_qpos=set_init_qpos))
        mappers = {"none": RawObs(), "mult": MultiplierObs(F["mul"]), "exact": ExactObsMapper.wrap(F["exact"], in_env=False)}
        new = run_block("scripts/test_exact_obs.py", "Z = torch.tensor(", "Q, OM, OE = ro.qpos", dict(
            common, mode=mode, policy=F["policy"], mappers=mappers, BatchRollout=BatchRollout, HumenvSim=HumenvSim), src=new_src)
        for k in ("Q", "OM", "OE"):
            check(f"{mode} {k}", new[k], old[k])


# ------------------------------------------------------------------ 6. the G1: robot_z_search / robot_lso_search loops

def test_robot_z_loop_g1():
    import torch
    from crossembodiment.obs import RawObs
    from crossembodiment.rollout import BatchRollout
    ref, sim = F["ref_adult"], F["g1"]
    ns = run_block("scripts/robot_z_search.py", "zt = torch.as_tensor(zs, dtype=torch.float32, device=a.device)",
                   "O, QS, QG, FZ = (np.stack(x, axis=1) for x in (O, QS, QG, FZ))",
                   dict(np=np, torch=torch, zs=F["Z"], vec=sim.vec, a=types.SimpleNamespace(device=F["dev"], init_vel="reference"),
                        ref=ref, T=STEPS, model=F["policy"].model))
    ro = BatchRollout(F["policy"], sim, RawObs(), N, STEPS, record=("obs", "qpos_smpl", "qpos", "fz")).run(
        F["Z"], init_qpos=ref[0], init_next=ref[1])
    for k, x in (("O", ro.obs), ("QS", ro.qpos_smpl), ("QG", ro.qpos), ("FZ", ro.fz), ("qg_start", ro.start.qpos)):
        check(k, x, ns[k])


def test_robot_lso_loop_g1():
    import torch
    from crossembodiment.obs import RawObs
    from crossembodiment.rollout import BatchRollout
    ref, sim = F["ref_adult"].astype(np.float64), F["g1"]
    ns = run_block("scripts/robot_lso_search.py", "zt = torch.as_tensor(16.0 * Z", "O, QS, QG, FZ = (np.stack(x, axis=1)",
                   dict(np=np, torch=torch, Z=F["Zseq"], vec=sim.vec, a=types.SimpleNamespace(device=F["dev"], init_vel="rest"),
                        ref=ref, S=STEPS, model=F["policy"].model))
    ro = BatchRollout(F["policy"], sim, RawObs(), N, STEPS, z_scale=16.0, record=("obs", "qpos_smpl", "qpos", "fz")).run(
        F["Zseq"], init_qpos=ref[0])
    for k, x in (("O", ro.obs), ("QS", ro.qpos_smpl), ("QG", ro.qpos), ("FZ", ro.fz), ("qg0", ro.start.qpos)):
        check(k, x, ns[k])


# ------------------------------------------------------------------ 7. batch compositions, observation factories (no simulation)

def test_batch_compositions():
    from crossembodiment.rollout import BatchRollout as B
    rng = np.random.default_rng(3)
    z, Z, mu, S = rng.standard_normal(256), rng.standard_normal((3, 256)), rng.standard_normal((7, 256)), rng.standard_normal((5, 7, 256))
    check("copies", B.copies(z, 16), np.repeat(z[None], 16, axis=0))
    check("padded", B.padded(Z, z, 16), np.concatenate([Z, np.repeat(z[None], 16 - len(Z), axis=0)]))
    check("with_mean", B.with_mean(mu, S), np.concatenate([mu[None], S]))
    check("copies (sequence)", B.copies(mu, 6), np.repeat(mu[None], 6, 0))


def test_obs_factories():
    from crossembodiment.bodies import BODIES
    from crossembodiment.obs import ExactObs, MultiplierObs, RawObs, lso_scale_kw, registry_scale, single_z_obs
    from model.obs_scale import build_obs_multiplier
    task, stem = CLIP.split("/")
    m = single_z_obs("exact", "child", CHILD_XML, REPO / f"data/child/retargeting_motion/{CLIP}.npz", REPO / f"data/origin_motion/{CLIP}.npz", ADULT_XML)
    assert isinstance(m, ExactObs) and m.kw == {} and m.in_env
    tall = "len1.4_wid0.8"
    ref_path, adult = REPO / f"data/{tall}/retargeting_motion/{CLIP}.npz", REPO / f"data/origin_motion/{task}/{stem}.npz"
    if not ref_path.exists():
        raise unittest.SkipTest(f"{ref_path} missing")
    m = single_z_obs("exact", tall, REPO / f"assets/robots_torque/{tall}/robot_torque_full.xml", ref_path, adult, ADULT_XML)
    ref = np.load(ref_path)["qpos"]                                       # single_z_search.py's lines at BASE, verbatim
    a = np.load(adult)["qpos"][:, :2]
    n = min(len(a), len(ref)); mk = np.abs(a[:n]) > 1e-3
    s_old = float(np.median(ref[:n, :2][mk] / a[:n][mk]))
    assert m.kw == {"scale": s_old} and s_old == BODIES.scale(tall, CLIP) == 1.364408905613045
    for mode in ("auto", "0.6"):
        m = single_z_obs(mode, "child", CHILD_XML, ref_path, adult, ADULT_XML)
        assert isinstance(m, MultiplierObs) and same(m.mul, build_obs_multiplier(CHILD_XML, ADULT_XML, mode=mode, parts="length", verbose=False))
    for mode in ("none", None):
        assert isinstance(single_z_obs(mode, "child", CHILD_XML, ref_path, adult, ADULT_XML), RawObs)
    assert lso_scale_kw(None) == {} and lso_scale_kw(1.0) == {"scale": 1.0}
    assert registry_scale("child") == 0.611036 and registry_scale("payload") == 1.0
    src = old_source("scripts/lso_search.py")
    assert 'kw = {} if a.scale is None else {"scale": a.scale}' in src          # the rule lso_scale_kw keeps


# --------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    try:
        setup()
        for name, fn in tests:
            try:
                fn()
                print(f"  PASS  {name}", flush=True)
            except unittest.SkipTest as e:
                skipped += 1
                print(f"  SKIP  {name}: {e}", flush=True)
            except (AssertionError, KeyError, ValueError, OSError, ImportError, subprocess.CalledProcessError, StopIteration) as e:
                failed += 1
                print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    finally:
        teardown()
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
