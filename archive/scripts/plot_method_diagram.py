#!/usr/bin/env python3
"""plot_method_diagram.py -- architecture diagram of the current method (frozen FB-CPR + interface alignment +
inference-time latent search in a correction subspace). Writes docs/figures/method_overview.{png,pdf,svg}.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

STYLE = {   # fill, edge, tag colour
    "input":   ("#F1F3F6", "#6B7686", "#4A5363"),
    "offline": ("#E6EFFA", "#2F6DB5", "#2F6DB5"),
    "frozen":  ("#E9E9E9", "#4D4D4D", "#333333"),
    "env":     ("#FFFFFF", "#4D4D4D", "#4D4D4D"),
    "op":      ("#FFFFFF", "#7A5AA6", "#7A5AA6"),
    "optim":   ("#FDEBDA", "#C8611A", "#B4530F"),
    "output":  ("#E4F3E9", "#2E8B57", "#22704A"),
}

fig = plt.figure(figsize=(17.8, 9.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 178); ax.set_ylim(-3, 91); ax.axis("off")


def box(x0, y0, x1, y1, kind, title, body, tag):
    fc, ec, tc = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=2.6 if kind == "frozen" else 1.6, zorder=2))
    cx = (x0 + x1) / 2
    ax.text(cx, y1 - 2.4, tag, ha="center", va="center", fontsize=7.6, color=tc, fontweight="bold", zorder=3)
    ax.text(cx, y1 - 5.6, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1E1E1E", zorder=3)
    ax.text(cx, (y0 + y1 - 7.4) / 2, body, ha="center", va="center", fontsize=9.2, color="#2B2B2B",
            linespacing=1.45, zorder=3)


def arrow(pts, label=None, at=None, color="#333333", ls="-", lw=1.5, ha="left", va="bottom", fs=9):
    for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, ls=ls, zorder=1.5, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1.6,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text(*at, label, fontsize=fs, color=color, ha=ha, va=va, zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


# ---- panel: the closed-loop rollout
ax.add_patch(FancyBboxPatch((33, 34), 107, 28, boxstyle="round,pad=0,rounding_size=1.5", fc="#FAFAFC",
                            ec="#9AA3AF", lw=1.1, ls=(0, (4, 3)), zorder=0.5))
ax.text(52, 64.0, "Closed-loop rollout, t = 1 … T   (16 candidate latents per generation, simulated in parallel)",
        fontsize=9.3, color="#5A6270", va="center", style="italic",
        bbox=dict(fc="white", ec="none", pad=0.5))

# ---- boxes
box(2, 66, 30, 86, "input", "Body parameters β", "8-d: length and girth of\nleg, arm, torso, head", "INPUT")
box(36, 66, 66, 86, "offline", "Body builder", "scale the adult MJCF by β;\nactuators × torque ratio $k_j$", "OFFLINE · ONCE PER BODY")
box(144, 66, 174, 86, "offline", "Correction basis U", "top-8 right singular vectors of\npast corrections $z^*_i - z_{0,i}$",
    "OFFLINE · ONCE")
box(36, 38, 66, 56, "env", "MuJoCo, body β", "simulator state $s_t$\n(control at 30 Hz)", "ENVIRONMENT")
box(72, 38, 102, 56, "op", "Exact observation $\\Phi_\\beta$", "reverse-retarget $s_t$ to the adult\n→ humenv 358-d proprio $o_t$",
    "FIXED MAP · NO PARAMETERS")
box(108, 38, 138, 56, "frozen", "Actor  π(a | o, z)", "Meta Motivo M-1 (FB-CPR)\ndeterministic mean action", "PRETRAINED · FROZEN")
box(151, 38, 174, 56, "output", "Output", "$z^*$ = best evaluated\ncandidate → motion\non body β", "RESULT")
box(2, 4, 30, 28, "input", "Adult clip", "task latent $z_0$ (reward-inferred)\n+ its adult rollout $q^{A}_{1:T}$", "INPUT")
box(36, 4, 66, 28, "offline", "Retarget", "joint angles copied,\nroot × $s_\\beta$, ground lift\n→ reference $g_{1:T}$",
    "OFFLINE · ONCE PER CLIP")
box(72, 4, 102, 28, "frozen", "Backward map B", "$B(o_t)$ of the rollout;\n$B(\\hat g_t)$, $\\hat g_t=\\Phi_\\beta(g_t)$, of the\nreference (precomputed); ∈ $\\mathbb{R}^{256}$",
    "PRETRAINED · FROZEN")
box(108, 4, 138, 28, "op", "Tracking cost J(z)", "$1-\\mathrm{mean}_t\\,\\cos(B(o_t), B(\\hat g_t))$\n+ heading + root-xy\n+ $\\lambda(\\|z-z_0\\|/16)^2$",
    "SCALAR PER ROLLOUT")
box(144, 4, 174, 28, "optim", "Latent search", "antithetic ES + Adam on $c\\in\\mathbb{R}^8$\n$z = 16\\cdot\\mathrm{normalize}(z_0 + U^\\top c)$\n5–8 generations × 16 rollouts",
    "INFERENCE-TIME OPTIMIZATION")

# ---- arrows
arrow([(30, 76), (36, 76)])
arrow([(45, 66), (45, 56)], "body MJCF", (46, 61.6), fs=8.6)
arrow([(30, 16), (36, 16)], "$q^{A}_{1:T}$", (30.4, 17), fs=8.6)
arrow([(51, 28), (51, 38)], "init $s_1 = g_1$", (52, 31.2), fs=8.6)
arrow([(66, 47), (72, 47)], "$s_t$", (67.2, 47.6))
arrow([(102, 47), (108, 47)], "$o_t$", (103.2, 47.6))
arrow([(123, 56), (123, 59), (57, 59), (57, 56)], "$a_t$  (PD joint targets)", (78, 59.3), fs=8.8)
arrow([(87, 38), (87, 28)], "$o_{1:T}$", (88, 31.2), fs=8.8)
arrow([(66, 16), (72, 16)], "$g_{1:T}$", (66.2, 17), fs=8.6)
arrow([(102, 16), (108, 16)])
arrow([(138, 16), (144, 16)], "J", (139.4, 17), fs=9)
arrow([(148, 28), (148, 47), (138, 47)], "z", (144.6, 41), color="#B4530F", fs=10.5)
arrow([(166, 28), (166, 38)], "after\nconvergence", (166.8, 29.5), color="#22704A", fs=8.2)
arrow([(174, 76), (176.6, 76), (176.6, 22), (174, 22)], "U", (175.2, 57), color="#2F6DB5", fs=10, ha="center")
arrow([(16, 4), (16, 0.6), (159, 0.6), (159, 4)], "$z_0$  (start point and anchor of the search)", (64, 1.0),
      color="#B4530F", fs=8.8)

# ---- title and legend
ax.text(72, 86.5, "Latent adaptation of a frozen BFM to a new body", fontsize=13.5,
        fontweight="bold", va="top", color="#1E1E1E")
ax.text(72, 82.0, "Only the 256-d latent z is changed; no network weight is trained.", fontsize=10, va="top",
        color="#444444")
legend = [("input", "input data"), ("offline", "offline preprocessing (no learning)"),
          ("frozen", "pretrained, frozen network"), ("op", "fixed computation"),
          ("optim", "inference-time optimization (of z)"), ("output", "output")]
for i, (k, name) in enumerate(legend):
    x, y = 72 + (i % 2) * 34, 76 - (i // 2) * 3.6
    fc, ec, _ = STYLE[k]
    ax.add_patch(FancyBboxPatch((x, y - 1.1), 3.4, 2.2, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec,
                                lw=2.2 if k == "frozen" else 1.2))
    ax.text(x + 4.4, y, name, fontsize=9, va="center", color="#333333")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"method_overview.{ext}", dpi=200 if ext == "png" else None, facecolor="white")
print(OUT / "method_overview.png")
