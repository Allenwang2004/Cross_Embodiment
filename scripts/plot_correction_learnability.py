#!/usr/bin/env python3
"""plot_correction_learnability.py -- why z_start -> correction cannot be learned from search labels.

Walking (move-ego-0-2_4). s = a start latent, z* = what the child search found from it, d = z* - s
(the correction), all at radius 16, Euclidean distances.
  8 starts   z0 and z0 rotated by 0.5..30 deg, each tracking its own adult rollout, two-stage L_align
             (lambda 0: outputs/latent_transfer_own; lambda 0.03 / 0.1 / 0.3: outputs/latent_transfer_anchor)
  20 seeds   20 searches from the same z0, only the random seed differs, two-stage L_align
             (outputs/single_z_seeds_twostage_align)
A: the three numbers for one pair (z0 and its 0.5-deg rotation), drawn to scale in the plane of d0, d1.
B: start distance |s_i - s_j| against correction difference |d_i - d_j|, every pair, lambda 0.
C: the same difference divided by the corrections' size, for each penalty lambda.
Writes outputs/latent_transfer_anchor/correction_learnability.png.
"""
import itertools
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
STEM = "move-ego-0-2_4"
RUNS = {"0": f"outputs/latent_transfer_own/{STEM}/align"}
RUNS.update({l: f"outputs/latent_transfer_anchor/{STEM}/lam{l}/align" for l in ("0.03", "0.1", "0.3")})
LCOL = {"0": ORANGE, "0.03": BLUE, "0.1": AQUA, "0.3": YELLOW}


def ld(p):
    v = np.load(REPO / p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def style(ax):
    ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(INK3)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    S = np.stack([ld(f"outputs/latent_transfer/{STEM}/starts/{n}.npy") for n in N])
    P = list(itertools.combinations(range(8), 2))
    xs = np.array([np.linalg.norm(S[i] - S[j]) for i, j in P])
    run = {}
    for lam, root in RUNS.items():
        D = np.stack([ld(f"{root}/{n}/best_z.npy") for n in N]) - S
        diff = np.array([np.linalg.norm(D[i] - D[j]) for i, j in P])
        rms = np.array([np.sqrt((D[i] @ D[i] + D[j] @ D[j]) / 2) for i, j in P])
        run[lam] = dict(D=D, diff=diff, norm=diff / rms, size=float(np.linalg.norm(D, axis=1).mean()))
    z0 = ld(f"data/origin_z/move-ego-0-2/{STEM}.npy")
    seeds = sorted((REPO / "outputs/single_z_seeds_twostage_align").glob(f"{STEM}_s*"))
    Ds = np.stack([ld((d / "best_z.npy").relative_to(REPO)) for d in seeds if (d / "best_z.npy").exists()]) - z0
    sd = np.array([np.linalg.norm(Ds[i] - Ds[j]) for i, j in itertools.combinations(range(len(Ds)), 2)])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 6.6)); fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.22)
    gb = gs[1].subgridspec(1, 2, width_ratios=[0.17, 1.0], wspace=0.04)
    aA, aS = fig.add_subplot(gs[0]), fig.add_subplot(gb[0])
    aB = fig.add_subplot(gb[1], sharey=aS); aC = fig.add_subplot(gs[2])

    # A: one pair to scale, in the plane spanned by d0 and d1
    D = run["0"]["D"]; d0, d1 = D[0], D[1]
    l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1); c = d0 @ d1 / (l0 * l1)
    a0 = np.radians(48); a1 = a0 + np.arccos(c)
    t0, t1 = l0 * np.array([np.cos(a0), np.sin(a0)]), l1 * np.array([np.cos(a1), np.sin(a1)])
    gap = np.linalg.norm(S[1] - S[0])
    aA.annotate("", xy=t0, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, mutation_scale=16))
    aA.annotate("", xy=t1, xytext=(gap, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2, mutation_scale=16))
    g = 15.8 * np.array([np.cos(a0 - 0.08), np.sin(a0 - 0.08)])
    aA.annotate("", xy=g, xytext=(gap, 0), arrowprops=dict(arrowstyle="-|>", color=INK3, lw=1.4, ls=(0, (4, 3)), mutation_scale=12))
    aA.text(g[0] + 0.7, g[1] - 1.0, "what a learnable map\nwould need: d₁ ≈ d₀",
            fontsize=8.5, color=INK2, ha="left", va="top")
    aA.plot(*zip(t0, t1), color=INK, lw=1.2, ls=(0, (2, 2)))
    mid = (t0 + t1) / 2
    aA.text(mid[0], max(t0[1], t1[1]) + 0.9, f"correction difference\n|d₁ − d₀| = {np.linalg.norm(d1 - d0):.1f}",
            ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    aA.text(*(t0 * 0.42 + np.array([2.2, -1.8])), f"correction size\n|d₀| = {l0:.1f}", fontsize=9, color=BLUE, ha="left")
    aA.text(*(t1 * 0.5 + np.array([-0.9, -0.6])), f"correction size\n|d₁| = {l1:.1f}", fontsize=9, color=ORANGE, ha="right")
    aA.scatter([0, gap], [0, 0], s=40, color=[BLUE, ORANGE], edgecolor=INK, lw=.6, zorder=5)
    aA.text(0.2, -1.1, f"start distance |s₁ − s₀| = {gap:.2f}\n(s₀ = z0, s₁ = z0 rotated 0.5°:"
            " the two dots overlap)", fontsize=9, color=INK, ha="center", va="top")
    aA.text(t0[0] - 0.4, t0[1] + 0.6, "z₀* (found from s₀)", fontsize=8.5, color=BLUE, ha="left")
    aA.text(t1[0] - 0.3, t1[1] + 0.6, "z₁* (found from s₁)", fontsize=8.5, color=ORANGE, ha="right")
    aA.set_xlim(-12, 21); aA.set_ylim(-4.2, 20); aA.set_aspect("equal")
    aA.set_title("A. one pair of starts, drawn to scale (d = z* − s)\ninputs 0.14 apart, the labels to learn 17.6 apart",
                 fontsize=10, color=INK)
    aA.set_xticks([]); aA.set_yticks([]); aA.set_facecolor(SURF)
    for sp in aA.spines.values(): sp.set_visible(False)

    # B: start distance vs correction difference (lambda 0) + same-start seeds
    aS.scatter(np.random.default_rng(0).uniform(-.3, .3, len(sd)), sd, s=10, color=INK3, alpha=.6, lw=0)
    aS.set_xlim(-.6, .6); aS.set_xticks([0]); aS.set_xticklabels(["0\n(same z0,\n20 seeds)"], fontsize=8.5)
    aS.set_ylabel("correction difference |d_i − d_j|", fontsize=9.5, color=INK2)
    aS.set_ylim(0, 25)
    aB.scatter(xs, run["0"]["diff"], s=50, color=ORANGE, edgecolor=SURF, lw=1, zorder=4,
               label="8 starts, every pair (28), no penalty")
    xx = np.logspace(-1.1, 1.1, 50)
    aB.plot(xx, xx, color=INK3, ls=(0, (4, 3)), lw=1.3, label="a smooth map: labels differ no more than inputs")
    aB.axhline(run["0"]["size"], color=INK2, lw=1, ls=":")
    aB.text(0.105, run["0"]["size"] - 0.5, f"size of one correction ({run['0']['size']:.1f})", fontsize=8.5,
            color=INK2, ha="left", va="top")
    aB.axhline(16 * np.sqrt(2), color=INK3, lw=1, ls=":")
    aB.text(9.5, 16 * np.sqrt(2) + 0.3, "two perpendicular corrections (22.6)", fontsize=8.5, color=INK2, ha="right")
    aB.set_xscale("log"); aB.set_xlim(0.1, 10)
    aB.set_xlabel("start distance |s_i − s_j|  (log scale)", fontsize=9.5, color=INK2)
    plt.setp(aB.get_yticklabels(), visible=False)
    aB.set_title(f"B. the labels differ by ~{run['0']['diff'].mean():.0f} however close the inputs are,\n"
                 f"as much as rerunning from the very same z0 ({sd.mean():.1f})", fontsize=10, color=INK)
    aB.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)

    # C: normalized difference per lambda
    for lam, r in run.items():
        aC.scatter(xs, r["norm"], s=34, color=LCOL[lam], edgecolor=SURF, lw=.8, zorder=4,
                   label=("no penalty" if lam == "0" else f"λ = {lam}") +
                         f":  correction size {r['size']:.1f},  difference {r['diff'].mean():.1f}")
    aC.axhline(np.sqrt(2), color=INK3, lw=1, ls=":"); aC.text(9.5, np.sqrt(2) + .03, "perpendicular (1.41)",
                                                               fontsize=8.5, color=INK2, ha="right")
    aC.axhline(0, color=INK3, lw=1, ls=":"); aC.text(9.5, .03, "same direction (0): what learning needs",
                                                     fontsize=8.5, color=INK2, ha="right")
    aC.set_xscale("log"); aC.set_xlim(0.1, 10); aC.set_ylim(-0.05, 1.6)
    aC.set_xlabel("start distance |s_i − s_j|  (log scale)", fontsize=9.5, color=INK2)
    aC.set_ylabel("correction difference ÷ correction size", fontsize=9.5, color=INK2)
    aC.set_title("C. the penalty shrinks the corrections, not how random their\n"
                 "direction is: difference ÷ size stays ~1.1 for every λ", fontsize=10, color=INK)
    aC.legend(frameon=False, fontsize=8.3, loc="lower left", bbox_to_anchor=(0, 0.1), labelcolor=INK)
    for ax in (aS, aB, aC):
        style(ax)
    out = REPO / "outputs/latent_transfer_anchor/correction_learnability.png"
    fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight")
    for lam, r in run.items():
        print(f"lambda {lam}: size {r['size']:.2f}, difference {r['diff'].mean():.2f}, ratio {r['norm'].mean():.2f}")
    print(f"same-start seeds: difference {sd.mean():.2f}  |  pair A: cos {c:+.2f}\n-> {out}")


if __name__ == "__main__":
    main()
