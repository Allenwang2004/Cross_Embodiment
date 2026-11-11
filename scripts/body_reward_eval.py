#!/usr/bin/env python3
"""body_reward_eval.py -- scripts/child_reward_eval.py's --curve / --explore / --mean for any humanoid body of the
survey (child, fat, tall, heavy_arm, limit_leg), on the 5 clips whose task the adult's own z0 does (adult z0 return
> 50 on the child's evaluation): walk, run, rotate-y, headstand, crouch (heavy_arm walks move-ego-0-2_4).

For each clip, the run the survey videos use (scripts/survey_body_videos.py run_of) is replayed exactly as the search
ran it: the body's XML, its reference's first frame at rest, the adult-equivalent observation with the scale the search
used (ExactObs: the child's 0.611036; the others' scale read off the reference against the adult's clip, as
single_z_search does), batches of 16. Rewards: humenv's own, on the adult-equivalent state, Motivo's action as ctrl.
  curve     the best-so-far z at every step (only the steps where it changed are rolled out)
  explore   the 16 z each step tried, rebuilt from the run's z_mean and its generator (default_rng(seed), one
            draw per step, as the run's --sampling drew it); every new best is checked to be one of its step's candidates
Steps beyond --max-steps (default 256) are not evaluated (the 512-step runs are cut to their first 256).
Writes outputs/reward/<body>/{return_curve.json, return_explore.json, return_curve_mean.png}; --all draws
outputs/reward/all_bodies_64.png (every body's mean best-so-far over the first 64 steps).
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/body_reward_eval.py <body> [--max-steps 256]   (evaluate + plot)
       CUDA_VISIBLE_DEVICES=2 uv run python scripts/body_reward_eval.py <body> --fix-records   (after evaluate; see fix_records)
       uv run python scripts/body_reward_eval.py <body> --plot | --all
"""
import argparse, functools, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.exact_obs import ExactObs, ExactObsWrapper, ADULT_XML       # noqa: E402
from crossenbodiment.survey import BODY, run_of   # noqa: E402  survey_body_videos' (refactor step 4)
from child_reward_eval import reward_of, rollout, BLUE, RED, INK, MUTED, GRID, SURF   # noqa: E402

OUT = REPO / "outputs/reward"
CLIPS = ["move-ego-0-2_2", "move-ego-0-4_8", "rotate-y-5-0.8_7", "headstand_4", "crouch-0_8"]
BODIES = ["child", "fat", "tall", "heavy_arm", "limit_leg"]
COLORS = {"child": "#2a78d6", "fat": "#eb6834", "tall": "#1baf7a", "heavy_arm": "#eda100", "limit_leg": "#e87ba4"}
YLABEL = "mean return over the 5 tasks\n(300-step rollouts; 120 for headstand, crouch)"


def tried(rng, summ, z):
    """the candidates the search scored at its point z, drawn as it drew them (--sampling; older runs: antithetic)."""
    from crossenbodiment.geometry import project_z   # single_z_search re-exported it (refactor step 4)
    if summ.get("sampling", "antithetic") == "antithetic":
        eps = rng.standard_normal((summ["pairs"], z.size))
        return project_z(np.concatenate([z + summ["sigma"] * eps, z - summ["sigma"] * eps]))
    return project_z(z + summ["sigma"] * rng.standard_normal((2 * summ["pairs"], z.size)))


def clips_of(body):
    return [BODY[body].get("clips", {}).get(c, c) for c in CLIPS]


def setup(body, stem):
    """the run, its summary, the body XML, the reference and the ExactObs scale kwargs, as single_z_search set them."""
    run = REPO / run_of(body, stem); summ = json.loads((run / "summary.json").read_text())
    assert summ["algo"] == "es" and not summ.get("subspace") and not summ.get("z_start") and summ["init"] == "reference"
    assert summ["obs_scale"] == "exact"
    ref = np.load(REPO / summ["ref"])["qpos"]
    kw = {}
    if body != "child":
        task = stem.rsplit("_", 1)[0]
        a = np.load(REPO / f"data/origin_motion/{task}/{stem}.npz")["qpos"][:, :2]
        n = min(len(a), len(ref)); m = np.abs(a[:n]) > 1e-3
        kw["scale"] = float(np.median(ref[:n, :2][m] / a[:n][m]))
    return run, summ, REPO / summ["xml"], ref, kw


def evaluate(body, max_steps):
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from crossenbodiment.geometry import project_z   # single_z_search re-exported it (refactor step 4)
    out = OUT / body; out.mkdir(parents=True, exist_ok=True)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT_XML))
    curve, explore = {}, {}
    for stem in clips_of(body):
        rew = reward_of(stem.rsplit("_", 1)[0])
        run, summ, xml, ref, kw = setup(body, stem)
        exact = ExactObs(xml, **kw); nv = mujoco.MjModel.from_xml_path(str(xml)).nv
        env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(xml), state_init="Default",
                             wrappers=[functools.partial(ExactObsWrapper, child_xml=str(xml), adult_xml=str(REPO / ADULT_XML), **kw)])
        steps, init = summ["steps"], ref[0]

        def returns(zs):
            Q, V, A = rollout(model, env, zs, init, steps, nv, exact)
            return [float(sum(rew(ma, *exact.adult_state(q, v), a) for q, v, a in zip(Q[k], V[k], A[k]))) for k in range(len(zs))]

        tr = np.load(run / "z_trace.npz"); n = min(len(tr["z_mean"]) - 1, max_steps)
        Zm, Zb = tr["z_mean"][:n + 1].astype(np.float64), tr["z_best"][:n + 1].astype(np.float64)
        # best so far, at every step 0..n (step 0 = z0)
        uniq = np.flatnonzero(np.r_[True, np.abs(np.diff(Zb, axis=0)).max(1) > 0]); R = {}
        for i in range(0, len(uniq), 16):
            idx = list(uniq[i:i + 16]); zs = [Zb[j] for j in idx] + [Zb[idx[-1]]] * (16 - len(idx))
            for j, r in zip(idx, returns(zs)):
                R[int(j)] = r
        best = [R[int(uniq[np.searchsorted(uniq, j, side="right") - 1])] for j in range(n + 1)]
        # the 16 tried at each step 0..n-1
        rng = np.random.default_rng(summ["seed"]); cands, worst = [], 0.0
        for t in range(n):
            cand = tried(rng, summ, Zm[t])
            if np.abs(Zb[t + 1] - Zb[t]).max() > 0:
                worst = max(worst, float(np.abs(cand - Zb[t + 1]).max(1).min()))
            cands.append(returns(cand))
        env.close()
        curve[stem] = dict(returns=best, run=str(run.relative_to(REPO)), steps=steps, scale=exact.s, n_rolled=len(uniq))
        explore[stem] = dict(candidate_returns=cands, new_best_vs_candidates_max_dz=worst)
        print(f"[{body}] {stem:20s} ({Path(curve[stem]['run']).name}, s {exact.s:.6f}) best so far {best[0]:.1f} -> {best[-1]:.1f} | "
              f"new-best match {worst:.1e} | tried: min {np.min(cands):.1f} max {np.max(cands):.1f}", flush=True)
        (out / "return_curve.json").write_text(json.dumps(curve, indent=1))
        (out / "return_explore.json").write_text(json.dumps(explore))


def fix_records(body):
    """--fix-records: the best-so-far returns taken from the rollouts the search itself made, so the line and the band
    agree. The actor's GPU numerics depend on which z share a batch, and chaotic clips (walk, run, rotate) turn that into
    different rollouts: re-rolling the records in batches of their own (evaluate's curve) gave returns up to ~50 off the
    same z's return in its search batch. Now a record that was one of a step's 16 candidates takes that candidate's
    return from return_explore.json (its own search batch); z0 and a record that was the mean z (checked every 25 steps)
    are rolled out as the search rolled them, 16 copies in one batch. The old numbers stay as returns_rerolled."""
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from crossenbodiment.geometry import project_z   # single_z_search re-exported it (refactor step 4)
    d = OUT / body; cu = json.loads((d / "return_curve.json").read_text()); ex = json.loads((d / "return_explore.json").read_text())
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    ma = mujoco.MjModel.from_xml_path(str(REPO / ADULT_XML))
    for stem in clips_of(body):
        rew = reward_of(stem.rsplit("_", 1)[0]); run, summ, xml, ref, kw = setup(body, stem)
        C = ex[stem]["candidate_returns"]; n = len(C)
        tr = np.load(run / "z_trace.npz"); Zm, Zb = tr["z_mean"][:n + 1].astype(np.float64), tr["z_best"][:n + 1].astype(np.float64)
        rng = np.random.default_rng(summ["seed"]); rec, own = [None] * (n + 1), {0: Zb[0]}
        for t in range(n):
            cand = tried(rng, summ, Zm[t])
            if np.abs(Zb[t + 1] - Zb[t]).max() > 0:
                dist = np.abs(cand - Zb[t + 1]).max(1); k = int(dist.argmin())
                if dist[k] < 1e-5:
                    rec[t + 1] = C[t][k]
                else:                                                      # the mean z: roll it as the search did
                    own[t + 1] = Zb[t + 1]
        if own:
            exact = ExactObs(xml, **kw); nv = mujoco.MjModel.from_xml_path(str(xml)).nv
            env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(xml), state_init="Default",
                                 wrappers=[functools.partial(ExactObsWrapper, child_xml=str(xml), adult_xml=str(REPO / ADULT_XML), **kw)])
            for j, z in own.items():
                Q, V, A = rollout(model, env, [z] * 16, ref[0], summ["steps"], nv, exact)
                rec[j] = float(sum(rew(ma, *exact.adult_state(q, v), a) for q, v, a in zip(Q[0], V[0], A[0])))
            env.close()
        for t in range(1, n + 1):                                          # carried forward while the record stands
            rec[t] = rec[t] if rec[t] is not None else rec[t - 1]
        old = cu[stem].get("returns_rerolled", cu[stem]["returns"])
        cu[stem].update(returns_rerolled=old, returns=rec + old[n + 1:], records_rolled_alone=sorted(own))
        print(f"[{body}] {stem:20s} records from candidates' own rollouts; {len(own)} rolled alone (z0 + mean z) | "
              f"max |old - new| {np.max(np.abs(np.array(old[:n + 1]) - np.array(rec))):.1f}", flush=True)
    (d / "return_curve.json").write_text(json.dumps(cu, indent=1))


def style(ax):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)


def body_mean(body, n=None):
    """mean over the body's 5 clips: best-so-far return per step (steps 0..n) and the per-step range of the 16 tried
    (per clip min / max over the 16, averaged over the clips), at the step they are tried from."""
    d = OUT / body; cu = json.loads((d / "return_curve.json").read_text()); ex = json.loads((d / "return_explore.json").read_text())
    clips = clips_of(body)
    n = min([len(cu[s]["returns"]) - 1 for s in clips] + ([n] if n else []))
    y = np.mean([cu[s]["returns"][:n + 1] for s in clips], 0)
    C = np.array([ex[s]["candidate_returns"][:n] for s in clips])
    return y, C.min(2).mean(0), C.max(2).mean(0)


def plot_body(body):
    y, lo, hi = body_mean(body); n = len(y) - 1
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF); style(ax)
    ax.fill_between(np.arange(n), lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=1, label="range of the 16 z tried at each step")
    ax.plot(np.arange(n + 1), y, color=BLUE, lw=2.5, zorder=3, label="best z so far")
    ax.plot([0], [y[0]], "o", color=RED, ms=8, zorder=4, label=f"z0 (step 0): {y[0]:.0f}")
    ax.text(n, y[-1] - 6, f"{y[-1]:.0f}", color=INK, fontsize=10, va="top", ha="right")
    ax.set_xlim(0, n); ax.set_ylim(0, max(200, ax.get_ylim()[1]))
    ax.set_xlabel("search step (16 rollouts each)", fontsize=10, color=INK); ax.set_ylabel(YLABEL, fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title(f"{body}: task return during the single-z search", fontsize=13, color=INK, loc="left", pad=8)
    fig.tight_layout(); fig.savefig(OUT / body / "return_curve_mean.png", dpi=200); plt.close(fig)
    print(f"{body}: step 0 {y[0]:.1f} -> step {n} {y[-1]:.1f} -> {OUT / body / 'return_curve_mean.png'}")


def plot_all(n=64):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURF); style(ax)
    for b in BODIES:
        if not (OUT / b / "return_explore.json").exists():
            continue
        y, _, _ = body_mean(b, n)
        ax.plot(np.arange(len(y)), y, color=COLORS[b], lw=2.2, zorder=3, label=b)
        ax.plot([0], [y[0]], "o", color=COLORS[b], ms=6, zorder=4)
        ax.text(len(y) - 1 + 0.6, y[-1], f"{b} {y[-1]:.0f}", color=INK, fontsize=8.5, va="center")
    ax.set_xlim(0, n + 9); ax.set_ylim(0, max(200, ax.get_ylim()[1]))
    ax.set_xlabel("search step (16 rollouts each)", fontsize=10, color=INK); ax.set_ylabel(YLABEL, fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", ncol=2)
    ax.set_title(f"Task return during the single-z search, every body (first {n} steps, best z so far)", fontsize=12.5,
                 color=INK, loc="left", pad=8)
    fig.tight_layout(); fig.savefig(OUT / f"all_bodies_{n}.png", dpi=200); plt.close(fig)
    print("->", OUT / f"all_bodies_{n}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("body", nargs="?", choices=BODIES)
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fix-records", action="store_true")
    a = ap.parse_args()
    if a.all:
        plot_all()
    elif a.fix_records:
        fix_records(a.body); plot_body(a.body)
    else:
        if not a.plot:
            evaluate(a.body, a.max_steps)
        plot_body(a.body)
