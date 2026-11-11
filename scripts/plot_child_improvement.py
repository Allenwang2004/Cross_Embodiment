#!/usr/bin/env python3
"""plot_child_improvement.py -- bar chart of LOES's MPJPE gain on the child, one bar per task motion (user, 11/10, after a
CEM-improvement bar chart they showed): the solid bar is LOES's MPJPE, the hatched part above it the error LOES removed
(z0's MPJPE minus LOES's), so the whole bar is z0's MPJPE. The motions with the largest gain, largest first.

Data: the child 52-motion comparison (outputs/child_52_rerun/per_run.npz): LOES = best latent after 1,024 rollouts, mean over
the 5 search seeds; z0 = its one rollout. MPJPE in cm as on the Task page. Labels: the task name (the clip's index dropped;
one clip per task).

Text size: every text in the figure is FONT pt at the figure's printed width WIDTH in (default 10 pt at 5.5 in: the
existing paper figures' width, a 5.5 in \\textwidth with 10 pt body text), so include it with
\\includegraphics[width=\\textwidth] (no scaling). The font is STIX (Times-like, as LaTeX's Times body text). The number of
motions is the largest that fits: adjacent 45-degree labels do not touch (tick spacing >= label height / sin 45 + GAP) and
no label runs past the figure's left edge. The PDF is saved at exactly WIDTH in (no tight bounding box, which would widen it).
--metric phc (user, 11/10): the PHC tracking reward instead (BFM-Zero's / CEM's objective, model/phc_reward.py: 1 - its cost, the mean over
frames of the per-frame reward, 0-1, higher is better): the solid bar is z0's reward, the hatched part LOES's gain on top of it (5-seed mean),
ranked by that gain; values cached in outputs/child_52_rerun/phc_reward.json (z0 checked against CEM's searches' own z0 cost) ->
docs/figures/child_improvement_bars_phc.*
--short: abbreviated task names (short()); --rot 90: vertical labels (spacing = label height, more motions fit).
Legend: our method is "Ours" (user, 11/10). Both bar parts have the same 0.6 pt blue outline, so the solid and hatched parts are equally wide.
usage: uv run python scripts/plot_child_improvement.py [--width 5.5] [--font 10] [--n N] [--metric mpjpe|phc] [--short] [--rot 45|90]   (N: fewer motions than the maximum)"""
import argparse, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent; OUT = REPO / "docs/figures"
# --loes-dir: where LOES's runs are (default the anchored outputs/child_52_rerun; outputs/noanchor/child = LOES without the anchor term,
# user 11/11); z0 and CEM stay the same runs. The PHC cache is written next to LOES's runs.
import sys as _sys
LOES = REPO / (_sys.argv[_sys.argv.index("--loes-dir") + 1] if "--loes-dir" in _sys.argv else "outputs/child_52_rerun")
BLUE, TINT, INK, MUTED, GRID = "#2a78d6", "#d4e4f7", "#0b0b0b", "#52514e", "#e6e6e3"
GAP = 1.5                                                    # pt between adjacent rotated labels
SIN = math.sin(math.radians(45))
ABBR = (("move-ego-low-", "mvl-"), ("move-ego-", "mv-"), ("raisearms", "ra"), ("rotate-", "rot-"), ("lieonground-", "lie-"), ("sitonground", "sit"))


def short(name):
    """--short: the task names abbreviated (move-ego -> mv, move-ego-low -> mvl, raisearms -> ra, rotate -> rot, lieonground -> lie,
    sitonground -> sit; the caption must say so)"""
    for a, b in ABBR:
        name = name.replace(a, b)
    return name


def phc_one(c):
    """PHC tracking reward (1 - the cost of model/phc_reward.py, as single_z_search.py --objective phc scores it) of z0, LOES's best after
    1,024 rollouts (outputs/child_52_rerun, seeds 0-4) and CEM (BFM-Zero)'s (outputs/child_52_cem_phc, its own search cost), one motion"""
    import json, sys, mujoco
    sys.path.insert(0, str(REPO)); from model.phc_reward import PHCCost
    st = c.split("/")[1]; m = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots_torque/child/robot_torque_full.xml"))
    ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); S0 = json.load(open(REPO / f"outputs/child_52_cem_phc/{st}_cem_s0/summary.json"))
    cost = PHCCost(m, ref[0], ref[1:], scale=S0["pos_div"])
    loes = [1 - float(cost(np.load(LOES / f"{st}_es_s{s}/best.npz")["qpos"].astype(np.float64)[None], ref[0])[0]) for s in range(5)]
    cem = [1 - json.load(open(REPO / f"outputs/child_52_cem_phc/{st}_cem_s{s}/summary.json"))["best"]["cost"] for s in range(5)]
    z0 = 1 - float(cost(np.load(REPO / f"outputs/child_52_rerun/{st}_es_s0/origin_z.npz")["qpos"].astype(np.float64)[None], ref[0])[0])
    assert abs(z0 - (1 - S0["origin_z"]["cost"])) < 1e-6, (c, z0, S0["origin_z"]["cost"])          # same z0 rollout and scoring as CEM's search
    return st, dict(z0=z0, loes=loes, cem=cem)


def phc_data():
    import json
    f = LOES / "phc_reward.json"
    if not f.exists():
        from multiprocessing import Pool
        clips = [l.strip() for l in open(REPO / "outputs/child_52/clips.txt")]
        with Pool(16) as p:
            f.write_text(json.dumps(dict(p.map(phc_one, clips)), indent=1))
    return json.load(open(f))


def data(metric="mpjpe"):
    if metric == "phc":                                       # higher is better: ranked by LOES's gain over z0 (5-seed mean)
        D = phc_data(); names = list(D); z0 = np.array([D[c]["z0"] for c in names]); es = np.array([np.mean(D[c]["loes"]) for c in names])
        o = np.argsort(-(es - z0)); return [names[j].rsplit("_", 1)[0] for j in o], z0[o], es[o]
    z = np.load(REPO / "outputs/child_52_rerun/per_run.npz", allow_pickle=True); i = list(z["keys"]).index("mpjpe")
    names = [str(c).split("/")[0] for c in z["clips"]]; z0 = z["z0"][:, i].astype(float); es = z["best_es"][:, :, -1, i].mean(1).astype(float)
    assert z["best_es"].shape[2] == 64                       # 64 generations x 16 = 1,024 rollouts
    o = np.argsort(-(z0 - es))
    return [names[j] for j in o], z0[o], es[o]


def text_size(fig, s, fs):
    t = fig.text(0, 0, s, fontsize=fs); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width * 72 / fig.dpi, bb.height * 72 / fig.dpi   # pt


def layout(fig, labels, fs, W, n, rot=45):
    """left margin (in) and tick spacing (in) for the first n labels at width W, or None if they do not fit"""
    sn, cs = math.sin(math.radians(rot)), math.cos(math.radians(rot))
    h = max(text_size(fig, s, fs)[1] for s in labels[:n]); dmin = (h / sn + GAP) / 72
    ylab = text_size(fig, "MPJPE (cm)", fs)[1] + text_size(fig, "0.25", fs)[0] + 10     # y label + tick labels + ticks, pt
    L, R = ylab / 72, 0.04
    for _ in range(20):                                       # the leftmost labels reach left of their ticks by w cos 45
        d = (W - L - R) / n; need = max(text_size(fig, s, fs)[0] * cs / 72 + 3 / 72 - (i + 0.5) * d for i, s in enumerate(labels[:n]))
        L2 = max(ylab / 72, need)
        if abs(L2 - L) < 1e-4:
            break
        L = L2
    d = (W - L - R) / n
    return (L, d, dmin, h) if d >= dmin else None


def main():
    a = argparse.ArgumentParser(); a.add_argument("--width", type=float, default=5.5); a.add_argument("--font", type=float, default=10.0)
    a.add_argument("--n", type=int, default=None); a.add_argument("--metric", choices=["mpjpe", "phc"], default="mpjpe")
    a.add_argument("--short", action="store_true"); a.add_argument("--rot", type=float, default=45.0); a.add_argument("--loes-dir", default="outputs/child_52_rerun"); a = a.parse_args()
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": a.font, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "hatch.linewidth": 0.8})
    labels, z0, es = data(a.metric); W = a.width; phc = a.metric == "phc"
    if a.short:
        labels = [short(x) for x in labels]; assert len(set(labels)) == len(labels)
    probe = plt.figure(figsize=(W, 3))
    nmax = max(n for n in range(2, len(labels) + 1) if layout(probe, labels, a.font, W, n, a.rot))
    n = min(a.n or nmax, nmax); L, d, dmin, h = layout(probe, labels, a.font, W, n, a.rot); plt.close(probe)
    print(f"width {W} in, {a.font} pt: at most {nmax} motions (label height {h:.1f} pt, tick spacing needed {dmin:.3f} in); drawing {n}, "
          f"spacing {d:.3f} in, left margin {L:.2f} in")
    fig = plt.figure(figsize=(W, 3)); lab_h = max(text_size(fig, s, a.font)[0] for s in labels[:n]) * math.sin(math.radians(a.rot)) / 72 + 0.12
    PH, TOP, BOT = 1.55, 0.28, 0.04; H = TOP + PH + lab_h + BOT; fig.set_size_inches(W, H)
    ax = fig.add_axes([L / W, (BOT + lab_h) / H, (W - L - 0.04) / W, PH / H])
    x = np.arange(n); bw = 0.62
    lo, hi = (z0, es) if phc else (es, z0)                    # solid: z0's reward / LOES's error; hatched: up to the other
    ax.bar(x, lo[:n], bw, color=BLUE, edgecolor=BLUE, linewidth=0.6, zorder=3)   # same outline as the hatched part, so both are equally wide (user, 11/10)
    ax.bar(x, hi[:n] - lo[:n], bw, bottom=lo[:n], color=TINT, edgecolor=BLUE, hatch="////", linewidth=0.6, zorder=3)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylabel("Reward" if phc else "MPJPE (cm)")
    ax.set_xticks(x); ax.set_xticklabels(labels[:n], rotation=a.rot, ha="right", va="center" if a.rot == 90 else "top", rotation_mode="anchor", color=INK)
    ax.tick_params(axis="x", length=2, pad=2); ax.tick_params(axis="y", length=2.5, pad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor=BLUE, linewidth=0.6, label=r"$z_0$" if phc else "Ours"),
                       Patch(facecolor=TINT, edgecolor=BLUE, hatch="////", linewidth=0.6, label="Ours" if phc else r"reduction from $z_0$")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, handlelength=1.4, borderaxespad=0.1, columnspacing=1.2)   # above the plot, clear of the bars
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    lo, hi = ax.get_ylim(); yt = [t for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if lo <= v <= hi]     # ticks matplotlib keeps outside the view are not drawn
    for t in ax.get_xticklabels() + yt + [ax.yaxis.label] + ax.get_legend().get_texts():
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W * fig.dpi + 0.5 or bb.y1 > H * fig.dpi + 0.5:
            bad.append(t.get_text())
    print("text outside the figure:", bad or "none", f"| figure {W:.2f} x {H:.2f} in")
    stem = OUT / (("child_improvement_bars_phc" if phc else "child_improvement_bars") + ("_short" if a.short else "") + ("" if a.rot == 45 else f"_rot{a.rot:g}"))
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    for lab, a_, b_ in zip(labels[:n], z0[:n], es[:n]):
        print(f"  {lab:30s} z0 {a_:6.3f}  LOES {b_:6.3f}  gain {b_ - a_:6.3f}" if phc else f"  {lab:30s} z0 {a_:6.2f}  LOES {b_:6.2f}  gain {a_ - b_:6.2f} cm")


if __name__ == "__main__":
    main()
