#!/usr/bin/env python3
"""diffusion_adapter.py -- synthetic tasks diffused around one real task, to test whether a CROSS-BODY adapter
G(beta, z0) can learn when the neighbourhood of one task is filled densely.

  tasks     125 new task latents z0' grown outward from raisearms-l-l_8's z0 in rings at Euclidean distance
            1..--radius (see tasks()); each point grows from a point of the previous ring by a random step.
            25 are held out, stratified by ring.
  adult     roll every z0' out on the ADULT exactly as data/origin_motion was made
            (scripts/metamotivo_motion_rollout.py: default humenv adult, Default reset, actor mean, CPU,
            1 torch thread, 120 steps); first checks that the real z0 reproduces data/origin_motion bit-for-bit.
  retarget  retarget every new adult motion to every body exactly as data/<body>/retargeting_motion was made
            (scripts/qpos_retarget.py: root scaled by the skeleton rest-height ratio, ground correction); first
            checks the real clip reproduces data/child/retargeting_motion.
  manifest  datasets/diffusion-raisearms: one row per (task, body) in model/dataset.py's format, task names
            diffraise<k> (trial 0), plus splits/{train,test}_tasks.txt and train_clips.txt for train_es.
Writes outputs/diffusion_adapter/{tasks.json, data/...}.
"""
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "outputs/diffusion_adapter"
DATA = OUT / "data"
DS = REPO / "datasets/diffusion-raisearms"
SRC_TASK, SRC_STEM = "raisearms-l-l", "raisearms-l-l_8"
T = 120
TRAIN_BODIES = ["m2c_t000", "child", "teen", "petite", "giant", "short_stocky", "tall_slim", "ax4m_t1000"]
TEST_BODIES = ["short_limbed", "long_limbed", "athletic", "elderly", "pear_shaped", "ax2m_t1000", "ax3m_t1000",
               "ax5m_t1000", "ax6m_t1000", "ax7m_t1000", "gl1250a1400", "gl1500a1400"]


def u16(v):
    v = np.asarray(v, np.float64).reshape(-1)
    return 16 * v / np.linalg.norm(v)


def name(k):
    return f"diffraise{k:03d}"


def tasks(a):
    """Rings: a.rings rings at distance 1, 2, ..., a.radius from z0 (Euclidean), a.n / a.rings points each. A point
    of ring r grows from a random point of ring r - 1 (z0 for ring 1) by one step in a random direction tangent to
    the sphere at its parent, its length set (bisection) so the new point lands at distance r from z0. In 256
    dimensions an unconstrained random walk barely moves outward (random steps are nearly orthogonal to the
    displacement so far), so the ring constraint is what makes the cloud actually reach the requested radius."""
    rng = np.random.default_rng(a.seed)
    z0 = u16(np.load(REPO / f"data/origin_z/{SRC_TASK}/{SRC_STEM}.npy"))
    radii = np.linspace(a.radius / a.rings, a.radius, a.rings)
    sizes = [len(c) for c in np.array_split(np.arange(a.n), a.rings)]
    P, parent, ring = [], [], []
    prev = [z0]; prev_idx = [-1]
    for r_i, (R, m) in enumerate(zip(radii, sizes)):
        cur, cur_idx = [], []
        while len(cur) < m:
            j = int(rng.integers(len(prev)))
            p0 = prev[j]
            g = rng.standard_normal(256); g -= (g @ p0) / 256 * p0; g /= np.linalg.norm(g)
            lo, hi = 0.0, 4 * a.radius
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if np.linalg.norm(u16(p0 + mid * g) - z0) < R else (lo, mid)
            q = u16(p0 + hi * g)
            if min([np.linalg.norm(q - x) for x in P + cur] + [np.inf]) < a.min_gap:
                continue
            cur.append(q); cur_idx.append(len(P) + len(cur) - 1); parent.append(prev_idx[j]); ring.append(r_i + 1)
        P += cur; prev, prev_idx = cur, cur_idx
    Z = np.array(P); d = np.linalg.norm(Z - z0, axis=1); ring = np.array(ring)
    rs = np.random.default_rng(a.seed + 1)
    n_test = np.array_split(np.arange(25), a.rings)
    test = sorted(int(k) for r_i in range(a.rings) for k in rs.choice(np.nonzero(ring == r_i + 1)[0], len(n_test[r_i]), replace=False))
    for k in range(a.n):
        p = DATA / "origin_z" / name(k); p.mkdir(parents=True, exist_ok=True)
        np.save(p / f"{name(k)}_0.npy", Z[k][None].astype(np.float32))
    step = [float(np.linalg.norm(Z[k] - (Z[parent[k]] if parent[k] >= 0 else z0))) for k in range(a.n)]
    meta = dict(source=f"{SRC_TASK}/{SRC_STEM}", radius=a.radius, rings=a.rings, min_gap=a.min_gap, seed=a.seed,
                tasks=[dict(k=k, name=name(k), dist=float(d[k]), ring=int(ring[k]), parent=int(parent[k]), step=step[k],
                            split="test" if k in test else "train") for k in range(a.n)])
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "tasks.json").write_text(json.dumps(meta, indent=1))
    reals = [u16(np.load(REPO / f"data/origin_z/{SRC_TASK}/{SRC_TASK}_{i}.npy")) for i in range(10) if i != 8]
    pd = np.linalg.norm(Z[:, None] - Z[None], axis=-1); np.fill_diagonal(pd, np.inf)
    print(f"{a.n} tasks in {a.rings} rings at distance {', '.join(f'{r:g}' for r in radii)} (sizes {sizes}); "
          f"step length {min(step):.2f}-{max(step):.2f}; nearest-neighbour distance median {np.median(pd.min(1)):.2f}; "
          f"test {len(test)} ({[int(np.sum(ring[test] == r + 1)) for r in range(a.rings)]} per ring)")
    print("for scale: the other 9 real raisearms-l-l trials sit at " + ", ".join(f"{np.linalg.norm(r - z0):.1f}" for r in reals))


def _adult_rollout(z, seed=8):
    import torch
    torch.set_num_threads(1)
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    global _M, _E
    if "_M" not in globals():
        _M = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cpu").eval()
        _E, _ = make_humenv(num_envs=1, task=None, state_init="Default")
    torch.manual_seed(seed)
    obs, _ = _E.reset(seed=seed)
    zt = torch.as_tensor(np.asarray(z, np.float32).reshape(1, -1))
    q = []
    for _ in range(T):
        with torch.no_grad():
            act = _M.act(torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0), zt, mean=True)
        obs, _, _, _, info = _E.step(act.cpu().numpy().ravel())
        q.append(info["qpos"].copy())
    return np.stack(q)


def _adult_job(k):
    z = np.load(DATA / "origin_z" / name(k) / f"{name(k)}_0.npy")
    out = DATA / "origin_motion" / name(k); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name(k)}_0.npz", qpos=_adult_rollout(z))
    return k


def adult(a):
    ref = np.load(REPO / f"data/origin_motion/{SRC_TASK}/{SRC_STEM}.npz")["qpos"]
    q = _adult_rollout(np.load(REPO / f"data/origin_z/{SRC_TASK}/{SRC_STEM}.npy"))
    err = np.abs(q - ref).max()
    print(f"check: real z0 on the adult vs data/origin_motion: max |dqpos| {err:.3g}")
    if err > 1e-6:
        raise SystemExit("origin motion does not reproduce -- stop")
    n = len(json.load(open(OUT / "tasks.json"))["tasks"])
    with Pool(a.workers) as pool:
        for i, k in enumerate(pool.imap_unordered(_adult_job, range(n))):
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{n} adult rollouts", flush=True)


def _retarget(body, qpos):
    import mujoco
    from qpos_retarget import ground_correct_qpos, load_root_rest_height, retarget_qpos
    s = load_root_rest_height(REPO / f"assets/robots/{body}/skeleton.json") / load_root_rest_height(REPO / "assets/robots/adult/skeleton.json")
    out = retarget_qpos(qpos, s)
    out, _, _ = ground_correct_qpos(mujoco.MjModel.from_xml_path(str(REPO / f"assets/robots/{body}/robot.xml")), out)
    return out


def _retarget_job(args):
    body, k = args
    q = np.load(DATA / "origin_motion" / name(k) / f"{name(k)}_0.npz")["qpos"]
    out = DATA / body / "retargeting_motion" / name(k); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name(k)}_0.npz", qpos=_retarget(body, q), fps=30.0)
    return body, k


def retarget(a):
    src = np.load(REPO / f"data/origin_motion/{SRC_TASK}/{SRC_STEM}.npz")["qpos"]
    for b in ("child", "giant", "ax4m_t1000"):
        ref = np.load(REPO / f"data/{b}/retargeting_motion/{SRC_TASK}/{SRC_STEM}.npz")["qpos"]
        err = np.abs(_retarget(b, src) - ref).max()
        print(f"check: retargeting the real clip to {b} vs data/{b}/retargeting_motion: max |dqpos| {err:.3g}")
        if err > 1e-6:
            raise SystemExit("retargeting does not reproduce -- stop")
    n = len(json.load(open(OUT / "tasks.json"))["tasks"])
    jobs = [(b, k) for b in TRAIN_BODIES + TEST_BODIES for k in range(n)]
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_retarget_job, jobs)):
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)} retargets", flush=True)


def manifest(a):
    meta = json.load(open(OUT / "tasks.json"))
    DS.mkdir(parents=True, exist_ok=True); (DS / "splits").mkdir(exist_ok=True)
    for link, target in (("data", os.path.relpath(DATA, DS)), ("robots", "../../assets/robots"),
                         ("robots_torque", "../../assets/robots_torque")):
        if not (DS / link).exists():
            (DS / link).symlink_to(target)
    rows = []
    for b in TRAIN_BODIES + TEST_BODIES:
        for t in meta["tasks"]:
            nm = t["name"]
            rows.append({"id": len(rows), "reward_name": nm, "trial": 0, "morphology_label": b,
                         "body_split": "train" if b in TRAIN_BODIES else "test", "task_split": t["split"],
                         "origin_z": f"data/origin_z/{nm}/{nm}_0.npy", "morphology": f"robots/{b}/parameter.json",
                         "target_xml": f"robots_torque/{b}/robot_torque_full.xml",
                         "retargeted_motion": f"data/{b}/retargeting_motion/{nm}/{nm}_0.npz"})
    (DS / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tr = [t["name"] for t in meta["tasks"] if t["split"] == "train"]; te = [t["name"] for t in meta["tasks"] if t["split"] == "test"]
    (DS / "splits/train_tasks.txt").write_text("".join(f"{t}\n" for t in tr))
    (DS / "splits/test_tasks.txt").write_text("".join(f"{t}\n" for t in te))
    (DS / "splits/train_bodies.txt").write_text("".join(f"{b}\n" for b in TRAIN_BODIES))
    (DS / "splits/test_bodies.txt").write_text("".join(f"{b}\n" for b in TEST_BODIES))
    (DS / "splits/train_clips.txt").write_text("".join(f"{t} 0\n" for t in tr))   # "<task> <trial>"
    print(f"{DS}: {len(rows)} rows = {len(meta['tasks'])} tasks x {len(TRAIN_BODIES) + len(TEST_BODIES)} bodies; "
          f"train {len(tr)} / test {len(te)} tasks, {len(TRAIN_BODIES)} train / {len(TEST_BODIES)} test bodies")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tasks", "adult", "retarget", "manifest"])
    ap.add_argument("--n", type=int, default=125)
    ap.add_argument("--rings", type=int, default=6)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--min-gap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    {"tasks": tasks, "adult": adult, "retarget": retarget, "manifest": manifest}[a.cmd](a)
