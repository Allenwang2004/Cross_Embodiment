#!/usr/bin/env python3
"""plot_child_convergence.py -- the child's single-latent searches by rollout budget, every one of the 52 task motions (user, 11/10):
cost convergence and PHC-reward rise of LOES, random search, CEM (BFM-Zero) and CMA-ES, 16 ... 1,024 rollouts (64 generations x 16),
the best latent so far, 5 search seeds; z0 as a flat line.

Cost = OUR cost (what LOES, random search and CMA-ES search with: BFM term + 1.0 heading + 0.1 position + 0.3 anchor):
  LOES / random / CMA-ES: each run's best_so_far (outputs/child_52_rerun/<stem>_<algo>_s<seed>/curve.csv);
  CEM (BFM-Zero) searched with PHC's reward instead, so its best latent of each generation (outputs/child_52_cem_phc/rollouts/<stem>)
  is re-scored with our cost (outputs/child_52_cem_phc/rollouts_ourcost/<stem>/eval.csv, run_child_cemphc_ourcost.sh).
Reward = PHC's tracking reward (1 - model/phc_reward.py's cost; CEM's objective):
  CEM: 1 - its run's best_so_far; LOES / random / CMA-ES: their best latent of each generation, re-rolled
  (outputs/child_52_rerun/rollouts/<stem>/eval_qpos.npz), scored with PHCCost.
z0: our cost from the runs' summaries, reward from its rollout. A motion's curve is the mean over the 5 seeds; the summary is the mean
over the 52 motions (band: +- one standard error over motions).
Checks: LOES's reward after 1,024 = outputs/child_52_rerun/phc_reward.json (from its best.npz); CEM's re-rolled latents reproduce its
best_so_far reward; the re-scoring's z0 cost = the runs' z0 cost.
Out: docs/figures/child_convergence_mean.{pdf,png} (5.5 in, 10 pt, cost | reward), outputs/child_convergence/{cost,reward}_all.png
(52 panels) and data.npz. Colours and labels: crossenbodiment/palette.py; styles: crossenbodiment/figstyle.py (OOP refactor step 10).
Runs are read through crossenbodiment.paths.read_path ($CE_OUTPUTS_ROOT), outputs/... written under $CE_RESULTS_ROOT and the figure
under $CE_FIGURES_ROOT (defaults: outputs/, docs/figures).
usage: uv run python scripts/plot_child_convergence.py [--no-cem-cost]   (--no-cem-cost: draw before CEM's re-scoring has finished)"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossenbodiment import paths                                          # noqa: E402
from crossenbodiment.figstyle import FigureStyle                           # noqa: E402
from crossenbodiment.palette import INK, Palette                           # noqa: E402
RR, CP = paths.read_path("outputs/child_52_rerun"), paths.read_path("outputs/child_52_cem_phc"); OUT = paths.write_path("outputs/child_convergence")
# legend names (user, 11/10): LOES -> Ours, CEM -> BFM-Zero
METHODS = tuple((a, Palette.label(k), Palette.hex(k)) for a, k in (("es", "loes"), ("random", "random"), ("cem", "cem"), ("cmaes", "cmaes")))
Z0 = Palette.hex("z0"); AXIS, GRID, GRID_ALL = INK["curves"]["axis"], INK["curves"]["grid"], INK["curves"]["grid_all"]
# the mean figure's style; the 52-panel grids are drawn on top of it (DejaVu Sans 8 pt, keeping its axis colours, as they always were)
MEAN_STYLE = FigureStyle.paper({"font.size": 10, "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS}, svg=False)
GRID_STYLE = MEAN_STYLE.updated({"font.family": "DejaVu Sans", "font.size": 8})
SEEDS, GENS = range(5), 64
NOCEM = "--no-cem-cost" in sys.argv
# --loes-dir outputs/noanchor/child (user 11/11: LOES without the anchor term): LOES's runs and re-rolls from there, and the cost panel
# shows the TRACKING cost of every method's best latent so far (BFM term + 1.0 heading + 0.1 position, no anchor; from the re-rolls'
# eval.csv parts), since the searches' own costs no longer share a definition (LOES without the anchor, random / CMA-ES with it)
RL = paths.read_path(sys.argv[sys.argv.index("--loes-dir") + 1]) if "--loes-dir" in sys.argv else RR; TRACK = RL != RR
if TRACK:
    OUT = paths.write_path("outputs/child_convergence_noanchor")


def col(f, k):
    return np.array([float(r[k]) for r in csv.DictReader(open(f))])


def one(c):
    import mujoco
    from model.phc_reward import PHCCost
    st = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)
    m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots_torque/child/robot_torque_full.xml"))
    pos_div = json.load(open(CP / f"{st}_cem_s0/summary.json"))["pos_div"]; phc = PHCCost(m, ref[0], ref[1:], scale=pos_div)
    cost, rew, chk = {}, {}, {}
    track = lambda f: col(f, "bfm") + 1.0 * col(f, "head") + 0.1 * col(f, "pos")          # our tracking cost, no anchor (eval.csv parts)
    for a in ("es", "random", "cmaes"):
        D_ = RL if a == "es" else RR
        ix = json.load(open(D_ / "rollouts" / st / "index.json"))["map"]; Q = np.load(D_ / "rollouts" / st / "eval_qpos.npz")["qpos"].astype(np.float64)
        rew[a] = np.stack([[1 - float(phc(Q[i][None], ref[0])[0]) for i in ix[f"{st}_{a}_s{s}"]["best"]] for s in SEEDS])
        if TRACK:
            T = track(D_ / "rollouts" / st / "eval.csv"); cost[a] = np.stack([T[ix[f"{st}_{a}_s{s}"]["best"]] for s in SEEDS])
        else:
            cost[a] = np.stack([col(RR / f"{st}_{a}_s{s}/curve.csv", "best_so_far") for s in SEEDS])
    P = json.load(open(RL / "phc_reward.json"))[st]; chk["loes_reward_vs_best_npz"] = float(np.abs(rew["es"][:, -1] - np.array(P["loes"])).max())
    rew["cem"] = 1 - np.stack([col(CP / f"{st}_cem_s{s}/curve.csv", "best_so_far") for s in SEEDS])
    ic = json.load(open(CP / "rollouts" / st / "index.json"))["map"]; Qc = np.load(CP / "rollouts" / st / "eval_qpos.npz")["qpos"].astype(np.float64)
    chk["cem_reroll_vs_curve"] = float(max(abs(1 - float(phc(Qc[ic[f"{st}_cem_s{s}"]["best"][-1]][None], ref[0])[0]) - rew["cem"][s, -1]) for s in SEEDS))
    z0c = json.load(open(RR / f"{st}_es_s0/summary.json"))["origin_z"]["cost"]
    e = CP / "rollouts_ourcost" / st / "eval.csv"
    if e.exists():
        ce = track(e) if TRACK else col(e, "cost"); cost["cem"] = np.stack([ce[ic[f"{st}_cem_s{s}"]["best"]] for s in SEEDS])
        chk["ourcost_z0"] = abs(json.load(open(CP / "rollouts_ourcost" / st / "eval_base.json"))["cost"] - z0c)
    else:
        cost["cem"] = np.full((len(SEEDS), GENS), np.nan)
    return st, cost, rew, dict(cost=z0c, reward=P["z0"]), chk


def plot_mean(D, x):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    MEAN_STYLE.apply()
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.3)); fig.subplots_adjust(left=0.1, right=0.965, bottom=0.2, top=0.8, wspace=0.32)
    for ax, key, lab in ((axs[0], "cost", "Tracking cost" if TRACK else "Cost (ours)"), (axs[1], "reward", "Reward")):
        z = np.mean([D[st]["z0"][key] for st in D]); ax.axhline(z, color=Z0, lw=1.2, ls="--", zorder=2)
        for a, name, c in METHODS:
            Y = np.stack([D[st][key][a].mean(0) for st in D])                 # motions x generations (seed means)
            if np.isnan(Y).all():
                continue
            mu, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y)); ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.15, lw=0, zorder=3)
            ax.plot(x, mu, color=c, lw=1.6, zorder=4, label=name)
        ax.set_xscale("log", base=2); ax.set_xticks([16, 64, 256, 1024]); ax.set_xticklabels(["16", "64", "256", "1024"]); ax.set_xlim(16, 1024)
        ax.set_xlabel("Rollouts"); ax.set_ylabel(lab); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axs[1].get_legend_handles_labels(); h.append(plt.Line2D([], [], color=Z0, ls="--", lw=1.2)); l.append(Palette.label("z0"))
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
    FigureStyle.save(fig, paths.figures_root() / "child_convergence_mean", formats=("pdf", "png"), dpi={"png": 300})
    plt.close(fig)


def plot_grid(D, x, key, ylab):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    GRID_STYLE.apply()
    names = sorted(D); fig, axs = plt.subplots(8, 7, figsize=(21, 22)); axs = axs.ravel()
    for k, st in enumerate(names):
        ax = axs[k]; ax.axhline(D[st]["z0"][key], color=Z0, lw=1, ls="--")
        for a, name, c in METHODS:
            Y = D[st][key][a]
            if np.isnan(Y).all():
                continue
            ax.fill_between(x, Y.min(0), Y.max(0), color=c, alpha=0.12, lw=0); ax.plot(x, Y.mean(0), color=c, lw=1.3, label=name)
        ax.set_xscale("log", base=2); ax.set_xticks([16, 64, 256, 1024]); ax.set_xticklabels(["16", "64", "256", "1024"]); ax.set_xlim(16, 1024)
        ax.set_title(st, fontsize=9); ax.grid(True, color=GRID_ALL, lw=0.5)
        if k % 7 == 0:
            ax.set_ylabel(ylab)
    for ax in axs[len(names):]:
        ax.axis("off")
    h, l = axs[0].get_legend_handles_labels(); h.append(plt.Line2D([], [], color=Z0, ls="--")); l.append(Palette.label("z0", "plain"))
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=5, fontsize=12, frameon=False)
    fig.suptitle(f"Child, 52 task motions: {ylab} of the best latent so far (mean of 5 seeds; band = min-max over seeds) by rollouts", y=0.997, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.962)); fig.savefig(OUT / f"{key}_all.png", dpi=110); plt.close(fig)


def main():
    from crossenbodiment.paths import CHILD_52_TASK, read_clips   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)
    clips = read_clips(CHILD_52_TASK); OUT.mkdir(parents=True, exist_ok=True)
    with Pool(8) as p:
        res = p.map(one, clips, chunksize=1)
    D = {st: dict(cost=c, reward=r, z0=z) for st, c, r, z, _ in res}; chk = {k: max(x[4].get(k, 0) for x in res) for k in ("loes_reward_vs_best_npz", "cem_reroll_vs_curve", "ourcost_z0")}
    missing = [st for st in D if np.isnan(D[st]["cost"]["cem"]).all()]
    print("checks (largest differences):", chk, "| CEM not yet re-scored with our cost:", len(missing))
    if missing and not NOCEM:
        raise SystemExit("CEM's re-scoring has not finished (pass --no-cem-cost to draw without it)")
    x = 16 * np.arange(1, GENS + 1)
    plot_mean(D, x); plot_grid(D, x, "cost", "Tracking cost" if TRACK else "Cost (ours)"); plot_grid(D, x, "reward", "Reward")
    np.savez_compressed(OUT / "data.npz", clips=np.array(sorted(D)), rollouts=x, **{f"{k}_{a}": np.stack([D[st][k][a] for st in sorted(D)]) for k in ("cost", "reward") for a, _, _ in METHODS},
                        z0_cost=np.array([D[st]["z0"]["cost"] for st in sorted(D)]), z0_reward=np.array([D[st]["z0"]["reward"] for st in sorted(D)]))
    for k in ("cost", "reward"):
        print(k, "after 1,024 (mean over motions):", {a: round(float(np.nanmean([D[st][k][a][:, -1].mean() for st in D])), 4) for a, _, _ in METHODS},
              "z0", round(float(np.mean([D[st]["z0"][k] for st in D])), 4))


if __name__ == "__main__":
    main()
