"""Tests of the port of the main tree's ExactObs clearance_vel option (main tree 11/12: model/exact_obs.py ExactObs(clearance_vel=True),
single_z_search.py --exact-clearance-vel) into the refactor (model/exact_obs.py, crossembodiment/obs.py ExactObs / single_z_obs,
crossembodiment/problems.py HumanLatentProblem).

MAIN: the main tree's model/exact_obs.py and the rollout() of its scripts/single_z_search.py (/home/allen19/crossembodiment, read only:
their text is executed into temporary modules, nothing is imported from or written into that tree; the MAIN tests are skipped when
those files are not there). BEFORE: model/exact_obs.py at 8cc7d89 (the main tree's HEAD, = refactor/base's: the code every run used
before the option existed), from `git show`. Every comparison is bit for bit (dtype, shape, np.array_equal).
  default_bit_identical      clearance_vel off: the new ExactObs = BEFORE's on reference states with perturbed velocities (obs, adult
                             qpos / qvel), and the new ExactObsWrapper in async workers = BEFORE's (sync env) over two short rollouts
                             (random actions from the reference's frame 0, moving), with no "clearance_vel_added" in info
  clearance_vel_sequential   clearance_vel on, ExactObs called along a rollout's states after reset() (and again after a second
                             reset()): new = MAIN (obs, adult qpos / qvel, the added term); the first call after reset() adds 0, and the
                             option does add something on this clip
  clearance_vel_wrapper      two rollouts of N slots: the new ExactObsWrapper(clearance_vel=True) in async workers vs MAIN's in a sync
                             env -- exact_proprio, clearance_vel_added, qpos (the second rollout: reset() clears the first one's
                             correction)
  policy_rollout             the frozen policy on N latents (GPU): the main tree's single_z_search.rollout with MAIN's ExactObs
                             (clearance_vel=True, main process: start pose) and MAIN's ExactObsWrapper (sync env) vs the refactor's path
                             for --exact-clearance-vel (single_z_obs(..., clearance_vel=True) -> wrappers() in async workers, build(),
                             BatchRollout), each main-process ExactObs first used on other states (as the reference embedding leaves
                             it), two rollouts in a row: qpos, obs, actions
  mapper_option_off          single_z_obs without the option passes no clearance_vel keyword (wrappers(), build()), as before
Needs GPU 2 for policy_rollout (CUDA_VISIBLE_DEVICES=2) and the clip's files (data/, a symlink into the main tree, read only); ~1 min.

Run standalone (no pytest in the environment), like tests/test_rollout.py:
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 uv run python tests/test_clearance_vel.py
"""
import ast
import functools
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

MAIN_TREE = Path("/home/allen19/crossembodiment")             # read only
BEFORE = "8cc7d89"                                            # model/exact_obs.py before the option (main tree HEAD = refactor/base)
CLIP = "crawl-0.4-0-d/crawl-0.4-0-d_1"                        # ground contact changes a lot: the clearance correction moves
STEPS, N = 40, 4
ADULT_XML = REPO / "assets/robots/adult/robot.xml"
CHILD_XML = REPO / "assets/robots_torque/child/robot_torque_full.xml"
F = {}


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def check(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert same(a, b), f"{name}: differs (dtype {a.dtype} / {b.dtype}, shape {a.shape} / {b.shape}" + \
        (f", max |d| {np.abs(a.astype(np.float64) - b.astype(np.float64)).max()})" if a.shape == b.shape else ")")


def module_from_source(name, src, label):
    """a temporary module executed from source text (no import machinery: nothing is cached or written next to the file)"""
    mod = types.ModuleType(name)
    mod.__file__ = label
    exec(compile(src, label, "exec"), mod.__dict__)
    return mod


def main_file(rel):
    f = MAIN_TREE / rel
    if not f.exists():
        raise unittest.SkipTest(f"{f} not found")
    return f.read_text(), str(f)


def main_mod():
    if "main" not in F:
        F["main"] = module_from_source("exact_obs_main_tree", *main_file("model/exact_obs.py"))
    return F["main"]


def before_mod():
    if "before" not in F:
        src = subprocess.run(["git", "show", f"{BEFORE}:model/exact_obs.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        F["before"] = module_from_source("exact_obs_before", src, f"{BEFORE}:model/exact_obs.py")
    return F["before"]


def setup():
    import mujoco
    F["ref"] = np.load(REPO / f"data/child/retargeting_motion/{CLIP}.npz")["qpos"].astype(np.float64)
    fk = mujoco.MjModel.from_xml_path(str(CHILD_XML)); F["fk"], F["nv"], F["nu"] = fk, fk.nv, fk.nu
    v0 = np.zeros(fk.nv); mujoco.mj_differentiatePos(fk, v0, 1.0 / 30, F["ref"][0], F["ref"][1]); F["v0"] = v0
    F["A"] = np.random.default_rng(20261112).uniform(-0.4, 0.4, (2, STEPS, N, fk.nu))


def make_env(wrapper_cls, mode, **kw):
    from humenv import make_humenv
    env, _ = make_humenv(num_envs=N, vectorization_mode=mode, task=None, max_episode_steps=300, xml=str(CHILD_XML), state_init="Default",
                         wrappers=[functools.partial(wrapper_cls, child_xml=str(CHILD_XML), adult_xml=str(ADULT_XML), **kw)])
    return env


def random_rollouts(env):
    """two rollouts with F["A"] from the reference's frame 0 (moving): per rollout exact_proprio, clearance_vel_added (or None), qpos, qvel"""
    out = []
    for rep in range(2):
        env.reset(); env.call("set_physics", qpos=F["ref"][0], qvel=F["v0"]); W, D, Q, V = [], [], [], []
        for t in range(STEPS):
            _, _, _, _, info = env.step(F["A"][rep, t])
            W.append(np.array(info["exact_proprio"])); Q.append(np.array(info["qpos"])); V.append(np.array(info["qvel"]))
            D.append(np.array(info["clearance_vel_added"]) if "clearance_vel_added" in info else None)
        out.append(dict(W=np.stack(W, 1), Q=np.stack(Q, 1), V=np.stack(V, 1), D=None if D[0] is None else np.stack(D, 1)))
    return out


# ------------------------------------------------------------------ tests

def test_default_bit_identical():
    import mujoco
    from model.exact_obs import ExactObs, ExactObsWrapper
    B = before_mod(); fk, ref = F["fk"], F["ref"]
    new, old = ExactObs(CHILD_XML, ADULT_XML), B.ExactObs(CHILD_XML, ADULT_XML)
    assert new.clearance_vel is False
    rng = np.random.default_rng(0); v = np.zeros(fk.nv); n = 0
    for t in range(1, len(ref), 5):
        mujoco.mj_differentiatePos(fk, v, 1.0 / 30, ref[t - 1], ref[t]); v2 = v + rng.normal(0, 0.1, fk.nv)
        check(f"obs t={t}", new(ref[t], v2), old(ref[t], v2))
        qa, va = new.adult_state(ref[t], v2); qb, vb = old.adult_state(ref[t], v2)
        check(f"adult qpos t={t}", qa, qb); check(f"adult qvel t={t}", va, vb); n += 1
    assert n > 10
    env_new, env_old = make_env(ExactObsWrapper, "async"), make_env(B.ExactObsWrapper, "sync")
    try:
        a, b = random_rollouts(env_new), random_rollouts(env_old)
    finally:
        env_new.close(); env_old.close()
    for r, (x, y) in enumerate(zip(a, b)):
        assert x["D"] is None and y["D"] is None, "clearance_vel_added in info with the option off"
        check(f"rollout {r} qpos", x["Q"], y["Q"]); check(f"rollout {r} exact_proprio", x["W"], y["W"])


def test_clearance_vel_sequential():
    from model.exact_obs import ExactObs
    M = main_mod()
    if "seq" not in F:                                       # the states of a rollout (any simulator: only (qpos, qvel) matter here)
        from model.exact_obs import ExactObsWrapper
        env = make_env(ExactObsWrapper, "async")
        try:
            F["seq"] = random_rollouts(env)
        finally:
            env.close()
    Q, V = F["seq"][0]["Q"][0], F["seq"][0]["V"][0]
    new, old = ExactObs(CHILD_XML, ADULT_XML, clearance_vel=True), M.ExactObs(CHILD_XML, ADULT_XML, clearance_vel=True)
    biggest = 0.0
    for rep in range(2):                                      # reset() between the two passes: the first call adds 0 again
        new.reset(); old.reset()
        for t in range(len(Q)):
            check(f"pass {rep} obs t={t}", new(Q[t], V[t]), old(Q[t], V[t]))
            assert new.added == old.added and type(new.added) is type(old.added), (rep, t, new.added, old.added)
            if t == 0:
                assert new.added == 0.0
            biggest = max(biggest, abs(new.added))
        new.reset(); old.reset()
        for t in range(len(Q)):
            qa, va = new.adult_state(Q[t], V[t]); qb, vb = old.adult_state(Q[t], V[t])
            check(f"pass {rep} adult qpos t={t}", qa, qb); check(f"pass {rep} adult qvel t={t}", va, vb)
    assert biggest > 1e-3, f"the option added at most {biggest} m/s on {CLIP}: not exercised"


def test_clearance_vel_wrapper():
    from model.exact_obs import ExactObsWrapper
    M = main_mod()
    env_new, env_old = make_env(ExactObsWrapper, "async", clearance_vel=True), make_env(M.ExactObsWrapper, "sync", clearance_vel=True)
    try:
        a, b = random_rollouts(env_new), random_rollouts(env_old)
    finally:
        env_new.close(); env_old.close()
    for r, (x, y) in enumerate(zip(a, b)):
        check(f"rollout {r} qpos", x["Q"], y["Q"]); check(f"rollout {r} exact_proprio", x["W"], y["W"])
        check(f"rollout {r} clearance_vel_added", x["D"], y["D"])
        assert np.abs(x["D"]).max() > 1e-3, f"rollout {r}: nothing added"


def test_policy_rollout():
    import torch
    from crossembodiment.geometry import project_z
    from crossembodiment.obs import ExactObs as ExactMapper, single_z_obs
    from crossembodiment.policy import MotivoPolicy
    from crossembodiment.rollout import BatchRollout
    from crossembodiment.sim import HumenvSim
    M = main_mod()
    src, label = main_file("scripts/single_z_search.py")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "rollout")
    ns = {"np": np, "torch": torch, "__name__": "single_z_search_main_tree"}
    exec(compile(ast.Module([node], []), label, "exec"), ns)
    rollout_main = ns["rollout"]
    policy = MotivoPolicy.load(device="cuda"); model, ref, v0 = policy.model, F["ref"], F["v0"]
    z0 = project_z(np.load(REPO / f"data/origin_z/{CLIP}.npy").reshape(-1).astype(np.float64))
    Z = project_z(z0[None] + 0.8 * np.random.default_rng(20261112).standard_normal((N, 256)))
    zt = torch.as_tensor(Z, dtype=torch.float32, device="cuda")
    # the main tree's search: ExactObs(clearance_vel=True) in the main process and ExactObsWrapper(clearance_vel=True) in the env
    exact_main = M.ExactObs(CHILD_XML, ADULT_XML, clearance_vel=True)
    env_main = make_env(M.ExactObsWrapper, "sync", clearance_vel=True)
    # the refactor's: what HumanLatentProblem builds for --obs-scale exact --exact-clearance-vel on the child
    mapper = single_z_obs("exact", "child", CHILD_XML, REPO / f"data/child/retargeting_motion/{CLIP}.npz",
                          REPO / f"data/origin_motion/{CLIP}.npz", ADULT_XML, clearance_vel=True)
    assert isinstance(mapper, ExactMapper) and mapper.clearance_vel and mapper.kw == {}
    (w,) = mapper.wrappers(); assert w.keywords.get("clearance_vel") is True, w.keywords
    sim = HumenvSim.make(CHILD_XML, N, wrappers=mapper.wrappers(), max_episode_steps=300)
    exact_new = mapper.build(); assert exact_new.clearance_vel is True
    batch = BatchRollout(policy, sim, mapper, N, STEPS, record=("qpos", "obs", "act"), nv=F["nv"])
    try:
        for e in (exact_main, exact_new):                    # used on other states first (as the reference embedding leaves it)
            e.reset(); e(ref[5], v0); e(ref[6], v0)
        for rep in range(2):
            A = []
            with torch.no_grad():
                q0, o0 = rollout_main(model, env_main, zt, STEPS, "cuda", None, init_qpos=ref[0], nv=F["nv"], return_obs=True,
                                      exact=exact_main, act_out=A, exact_in_env=True, init_qvel=v0)
            ro = batch.run(Z, init_qpos=ref[0], init_qvel=v0)
            check(f"rollout {rep} qpos", ro.qpos, q0); check(f"rollout {rep} obs", ro.obs.astype(np.float32), o0)
            check(f"rollout {rep} act", ro.act, np.stack(A, axis=1))
            assert q0.shape == (N, STEPS, 76) and o0.dtype == np.float32
    finally:
        env_main.close(); sim.close()


def test_mapper_option_off():
    from crossembodiment.obs import single_z_obs
    m = single_z_obs("exact", "child", CHILD_XML, REPO / f"data/child/retargeting_motion/{CLIP}.npz", REPO / f"data/origin_motion/{CLIP}.npz",
                     ADULT_XML)
    assert m.clearance_vel is False and m._option_kw() == {} and m.kw == {}
    (w,) = m.wrappers(); assert set(w.keywords) == {"child_xml", "adult_xml"}, w.keywords
    assert m.build().clearance_vel is False


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    setup()
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except unittest.SkipTest as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}", flush=True)
        except (AssertionError, KeyError, ValueError, OSError, ImportError, AttributeError, subprocess.CalledProcessError, StopIteration) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
