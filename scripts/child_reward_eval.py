#!/usr/bin/env python3
"""child_reward_eval.py -- humenv's own task rewards as a second metric for the child searches (the survey's 11 clips).

Each clip is a rollout of a humenv reward task (data/origin_motion/<task>/...), so its reward asks whether the body
still does the TASK, independent of the bfm the search optimised. The rewards are written for the adult (head >= 1.4 m
to count as standing, 2 m/s, pelvis >= 0.8 m, ...), so the child's state is first mapped to its adult-equivalent state
(model.exact_obs.ExactObs.adult_state: joint angles and root orientation as they are, root position and root linear
velocity / s with s = 0.611036, the scale the child references were retargeted with; heights come out of the adult
skeleton) and humenv's reward function is evaluated on that, with Motivo's action as ctrl.

Per clip, three rollouts from the clip's first frame at rest, 300 steps, return = sum of per-step rewards (max 300):
  adult z0          z0 on the adult body (the reference value)
  child z0          z0 on the child
  child searched    the survey's searched z on the child (scripts/survey_body_videos.py run_of)
Rollouts run in batches of 16 like the search (the actor's GPU numerics depend on the batch size). The child
rollouts are checked against the qpos the search saved (origin_z.npz, best.npz): chaotic ones saved on another GPU
(the c540 runs predate GPU 2) can still differ; returns.json records the deviation per rollout.
Writes outputs/reward/child/{returns.json, returns.png}.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/child_reward_eval.py [--curve | --mean]   (in that order: --curve needs
       returns.json, --mean needs return_curve.json)
"""
import functools, inspect, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.exact_obs import ExactObs, ExactObsWrapper, ADULT_XML       # noqa: E402
from survey_body_videos import CLIPS, run_of                            # noqa: E402

OUT = REPO / "outputs/reward/child"
CHILD_XML = REPO / "assets/robots_torque/child/robot_torque_full.xml"
GREY, RED, BLUE = "#8c8f9e", "#d93833", "#2e6bd9"                       # the survey videos' colours
INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"


def reward_of(task):
    from humenv import rewards as R
    for _, c in inspect.getmembers(R, inspect.isclass):
        if hasattr(c, "reward_from_name") and c is not R.RewardFunction:
            r = c.reward_from_name(task)
            if r is not None:
                return r
    raise SystemExit(f"no humenv reward for {task}")


@torch.no_grad()
def rollout(model, env, zs, init, steps, nv, exact=None, qvel0=None):
    """single_z_search.rollout (deterministic actor mean, start at rest from init), also returning qvel and actions.
    qvel0: start with this velocity instead of at rest."""
    v0 = np.zeros(nv) if qvel0 is None else np.asarray(qvel0, dtype=np.float64)
    env.reset()
    env.call("set_physics", qpos=init, qvel=v0)
    if exact is not None:
        obs = np.repeat(exact(init, v0)[None], len(zs), 0)
    else:
        obs = np.stack([o["proprio"] for o in env.call("get_obs")])
    zt = torch.as_tensor(np.asarray(zs), dtype=torch.float32, device="cuda")
    Q, V, A = [], [], []
    for _ in range(steps):
        mu = model._actor(model._normalize(torch.as_tensor(obs, dtype=torch.float32, device="cuda")), zt,
                          model.cfg.actor_std).mean
        a = mu.cpu().numpy()
        o, _, _, _, info = env.step(a)
        Q.append(info["qpos"].copy()); V.append(info["qvel"].copy()); A.append(a)
        obs = info["exact_proprio"].copy() if exact is not None else o["proprio"]
    return np.stack(Q, 1), np.stack(V, 1), np.stack(A, 1)


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    OUT.mkdir(parents=True, exist_ok=True)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT_XML))
    exact = ExactObs(CHILD_XML)
    print(f"adult-equivalent scale s = {exact.s}")
    env_c, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(CHILD_XML), state_init="Default",
                           wrappers=[functools.partial(ExactObsWrapper, child_xml=str(CHILD_XML), adult_xml=str(REPO / ADULT_XML))])
    env_a, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(REPO / ADULT_XML), state_init="Default")
    nv_c, nv_a = mujoco.MjModel.from_xml_path(str(CHILD_XML)).nv, ma.nv
    res = {}
    for stem in CLIPS:
        task = stem.rsplit("_", 1)[0]; rew = reward_of(task)
        run = REPO / run_of("child", stem); summ = json.loads((run / "summary.json").read_text())
        steps = summ["steps"]
        z0 = np.load(REPO / f"data/origin_z/{task}/{stem}.npy").reshape(-1)
        zb = np.load(run / "best_z.npy").reshape(-1)
        ref_c = np.load(summ["ref"])["qpos"]; ref_a = np.load(REPO / f"data/origin_motion/{task}/{stem}.npz")["qpos"]
        Qc, Vc, Ac = rollout(model, env_c, [z0] * 8 + [zb] * 8, ref_c[0], steps, nv_c, exact)   # the search's batch of 16
        Qc, Vc, Ac = Qc[[0, 8]], Vc[[0, 8]], Ac[[0, 8]]
        Qa, Va, Aa = rollout(model, env_a, [z0] * 16, ref_a[0], steps, nv_a)
        dev = [float(np.abs(Qc[0] - np.load(run / "origin_z.npz")["qpos"][:steps]).max()),
               float(np.abs(Qc[1] - np.load(run / "best.npz")["qpos"][:steps]).max())]
        per = {}
        for name, Q, V, A, child in (("child z0", Qc[0], Vc[0], Ac[0], True), ("child searched", Qc[1], Vc[1], Ac[1], True),
                                     ("adult z0", Qa[0], Va[0], Aa[0], False)):
            r = []
            for q, v, a in zip(Q, V, A):
                qa, va = exact.adult_state(q, v) if child else (q, v)
                r.append(float(rew(ma, qa, va, a)))
            per[name] = r
        res[stem] = dict(task=task, reward=repr(rew), run=str(run.relative_to(REPO)), rollout_vs_saved_max_dq=dev,
                         returns={k: float(np.sum(v)) for k, v in per.items()}, per_step=per)
        R_ = res[stem]["returns"]
        print(f"{stem:20s} adult z0 {R_['adult z0']:6.1f} | child z0 {R_['child z0']:6.1f} | child searched "
              f"{R_['child searched']:6.1f}   (replay vs saved: z0 {dev[0]:.1e}, searched {dev[1]:.1e})", flush=True)
    env_c.close(); env_a.close()
    (OUT / "returns.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURF); ax.set_facecolor(SURF)
    x = np.arange(len(CLIPS)); w = 0.27
    for i, (k, c, lab) in enumerate((("adult z0", GREY, "adult, z0 (reference)"), ("child z0", RED, "child, z0"),
                                     ("child searched", BLUE, "child, searched z"))):
        v = [res[s]["returns"][k] for s in CLIPS]
        ax.bar(x + (i - 1) * w, v, w - 0.03, color=c, label=lab, zorder=3)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 2, f"{vi:.0f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, CLIPS, rotation=25, ha="right", fontsize=9, color=INK)
    ax.set_ylabel("return (sum over 300 steps, max 300)", fontsize=10, color=INK)
    ax.set_ylim(0, 320)
    ax.grid(axis="y", color=GRID, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=3)
    ax.set_title("humenv task return on the child (adult-equivalent state, s = 0.611)", fontsize=13, color=INK, loc="left",
                 pad=24)
    ax.text(0, 1.03, "each clip's own humenv reward; child states mapped to the adult skeleton before scoring; "
            "all rollouts start from the clip's first frame at rest", transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "returns.png", dpi=200)
    print("->", OUT / "returns.png")


def curve():
    """--curve: the return of the search's best-so-far z at every step (z_trace.npz z_best; step 0 = z0), rolled out
    16 at a time like the search; only the steps where the best changed are rolled out. One panel per clip, the adult's
    z0 return (returns.json) as the dashed reference. Writes outputs/reward/child/return_curve.{json,png}."""
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT_XML))
    exact = ExactObs(CHILD_XML)
    env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(CHILD_XML), state_init="Default",
                         wrappers=[functools.partial(ExactObsWrapper, child_xml=str(CHILD_XML), adult_xml=str(REPO / ADULT_XML))])
    nv = mujoco.MjModel.from_xml_path(str(CHILD_XML)).nv
    adult = json.loads((OUT / "returns.json").read_text())
    res = {}
    for stem in CLIPS:
        task = stem.rsplit("_", 1)[0]; rew = reward_of(task)
        run = REPO / run_of("child", stem); summ = json.loads((run / "summary.json").read_text())
        steps, init = summ["steps"], np.load(summ["ref"])["qpos"][0]
        Zb = np.load(run / "z_trace.npz")["z_best"]
        change = np.r_[True, np.abs(np.diff(Zb, axis=0)).max(1) > 0]
        uniq = np.flatnonzero(change)                                     # the steps where best-so-far changed
        R = {}
        for i in range(0, len(uniq), 16):
            idx = list(uniq[i:i + 16]); zs = [Zb[j] for j in idx] + [Zb[idx[-1]]] * (16 - len(idx))
            Q, V, A = rollout(model, env, zs, init, steps, nv, exact)
            for k, j in enumerate(idx):
                R[int(j)] = float(sum(rew(ma, *exact.adult_state(q, v), a) for q, v, a in zip(Q[k], V[k], A[k])))
        ret = [R[int(uniq[np.searchsorted(uniq, j, side="right") - 1])] for j in range(len(Zb))]
        res[stem] = dict(task=task, returns=ret, adult_z0=adult[stem]["returns"]["adult z0"], n_rolled=len(uniq))
        print(f"{stem:20s} step 0 {ret[0]:6.1f} -> last {ret[-1]:6.1f} (adult z0 {res[stem]['adult_z0']:.1f}; "
              f"{len(uniq)} distinct z rolled out)", flush=True)
    env.close()
    (OUT / "return_curve.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), facecolor=SURF)
    for ax, stem in zip(axes.flat, CLIPS):
        r = res[stem]; y = np.array(r["returns"]); x = np.arange(len(y))
        ax.set_facecolor(SURF)
        ax.axhline(r["adult_z0"], color=GREY, lw=1.5, ls="--", zorder=2)
        ax.plot(x, y, color=BLUE, lw=2, zorder=3)
        ax.plot([0], [y[0]], "o", color=RED, ms=7, zorder=4)
        ax.text(x[-1], y[-1], f" {y[-1]:.0f}", color=INK, fontsize=9, va="center")
        low = r["adult_z0"] < 50
        ax.set_title(stem + ("  (adult z0 also low)" if low else ""), fontsize=10, color=MUTED if low else INK, loc="left")
        ax.set_ylim(0, max(300, y.max() * 1.1)); ax.set_xlim(0, len(y) - 1 + max(4, len(y) // 8))
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax = axes.flat[-1]; ax.axis("off")
    ax.plot([], [], color=BLUE, lw=2, label="child, best z so far"); ax.plot([], [], "o", color=RED, label="child, z0 (step 0)")
    ax.plot([], [], color=GREY, lw=1.5, ls="--", label="adult, z0 (reference)")
    ax.legend(frameon=False, fontsize=10, loc="center")
    for a in axes[:, 0]:
        a.set_ylabel("return (max 300)", fontsize=9, color=INK)
    for a in axes[-1, :3]:
        a.set_xlabel("search step (16 rollouts each)", fontsize=9, color=INK)
    fig.suptitle("humenv task return of the child during the single-z search (adult-equivalent state, s = 0.611)",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "return_curve.png", dpi=200)
    print("->", OUT / "return_curve.png")


def explore(threshold=50.0):
    """--explore: for the clips mean_curve uses, the return of every z the search tried: per step the 16 candidates
    project_z(z_mean[t] +- sigma eps_t), eps_t redrawn from the search's own generator (default_rng(seed), one
    (pairs, 256) draw per step, nothing else draws from it). Checked: every new best-so-far z is one of its step's
    candidates (to ~1e-5: z_trace is float32). Each step's 16 are rolled out as one batch, as in the search. Writes outputs/reward/child/return_explore.json."""
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from single_z_search import project_z
    curve_res = json.loads((OUT / "return_curve.json").read_text())
    keep = [s for s in CLIPS if curve_res[s]["adult_z0"] > threshold]
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT_XML))
    exact = ExactObs(CHILD_XML)
    env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(CHILD_XML), state_init="Default",
                         wrappers=[functools.partial(ExactObsWrapper, child_xml=str(CHILD_XML), adult_xml=str(REPO / ADULT_XML))])
    nv = mujoco.MjModel.from_xml_path(str(CHILD_XML)).nv
    res = {}
    for stem in keep:
        task = stem.rsplit("_", 1)[0]; rew = reward_of(task)
        run = REPO / run_of("child", stem); summ = json.loads((run / "summary.json").read_text())
        assert summ["algo"] == "es" and not summ.get("subspace") and summ.get("sampling", "antithetic") == "antithetic"
        steps, init = summ["steps"], np.load(summ["ref"])["qpos"][0]
        tr = np.load(run / "z_trace.npz"); Zm, Zb = tr["z_mean"].astype(np.float64), tr["z_best"].astype(np.float64)
        rng = np.random.default_rng(summ["seed"])
        rets, worst_match = [], 0.0
        for t in range(len(Zm) - 1):
            eps = rng.standard_normal((summ["pairs"], Zm.shape[1]))
            cand = project_z(np.concatenate([Zm[t] + summ["sigma"] * eps, Zm[t] - summ["sigma"] * eps]))
            if np.abs(Zb[t + 1] - Zb[t]).max() > 0:                     # a new best: must be one of these candidates
                worst_match = max(worst_match, float(np.abs(cand - Zb[t + 1]).max(1).min()))
            Q, V, A = rollout(model, env, cand, init, steps, nv, exact)
            rets.append([float(sum(rew(ma, *exact.adult_state(q, v), a) for q, v, a in zip(Q[k], V[k], A[k])))
                         for k in range(len(cand))])
        res[stem] = dict(candidate_returns=rets, new_best_vs_candidates_max_dz=worst_match)
        r = np.array(rets)
        print(f"{stem:20s} {r.shape[0]} steps x {r.shape[1]} candidates | new-best match max |dz| {worst_match:.1e} | "
              f"candidate return min {r.min():.1f} median {np.median(r):.1f} max {r.max():.1f}", flush=True)
    env.close()
    (OUT / "return_explore.json").write_text(json.dumps(res))


def mean_curve(threshold=50.0):
    """--mean: from return_curve.json, the clips whose adult z0 return is above threshold (tasks Motivo itself does),
    their best-so-far returns averaged per search step; with return_explore.json (--explore) also the range of the 16
    z each step tried (per clip min and max over the 16, averaged over the clips), at the step they are tried from
    (the 16 around z0 at step 0).
    Writes outputs/reward/child/return_curve_mean.png."""
    res = json.loads((OUT / "return_curve.json").read_text())
    keep = [s for s in CLIPS if res[s]["adult_z0"] > threshold]
    n = min(len(res[s]["returns"]) for s in keep)
    Y = np.array([res[s]["returns"][:n] for s in keep]); y = Y.mean(0); adult = np.mean([res[s]["adult_z0"] for s in keep])
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
    ex_path = OUT / "return_explore.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        C = np.array([ex[s]["candidate_returns"][:n - 1] for s in keep])        # (clips, steps, 16)
        lo, hi = C.min(2).mean(0), C.max(2).mean(0)
        ax.fill_between(np.arange(n - 1), lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=1,     # at the step they start from
                        label="range of the 16 z tried at each step")
    ax.plot(np.arange(n), y, color=BLUE, lw=2.5, zorder=3, label="child, best z so far")
    ax.plot([0], [y[0]], "o", color=RED, ms=8, zorder=4, label=f"child, z0 (step 0): {y[0]:.0f}")
    ax.text(n - 1, y[-1] - 6, f"{y[-1]:.0f}", color=INK, fontsize=10, va="top", ha="right")
    ax.set_xlim(0, n - 1); ax.set_ylim(0, max(200, ax.get_ylim()[1]))
    ax.set_xlabel("search step (16 rollouts each)", fontsize=10, color=INK)
    ax.set_ylabel("mean return (max 300)", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("Child task return during the single-z search", fontsize=13, color=INK, loc="left", pad=8)
    fig.tight_layout(); fig.savefig(OUT / "return_curve_mean.png", dpi=200)
    print(f"{len(keep)} clips: {keep}\nstep 0 {y[0]:.1f}, step 16 {y[16]:.1f}, step 32 {y[32]:.1f}, last {y[-1]:.1f}; adult {adult:.1f}")
    print("->", OUT / "return_curve_mean.png")


if __name__ == "__main__":
    (mean_curve() if "--mean" in sys.argv else explore() if "--explore" in sys.argv else
     curve() if "--curve" in sys.argv else main())
