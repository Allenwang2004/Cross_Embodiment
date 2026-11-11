#!/usr/bin/env python3
"""child_topview.py -- top views of the pelvis path for all 52 child task motions (user, 11/11: the Blender strobes put the poses at
even spacing with the root's x, y removed and turn each panel side-on to its own path, so drift and heading cannot be compared there).

Per motion: the child's reference (the target), z0, BFM-Zero (CEM on PHC's reward) and ours (LOES), seed 0, the same trajectories as
outputs/child_all52/trajectories (survey alignment: rollout frame t against reference frame t + 1, 299 frames). Lines = pelvis x, y over
the clip; arrows = the body's facing every 2 s and at the end (facing = root yaw - 90 deg: on the forward-walking references the walking
direction is root yaw - 84 ... -93 deg); dot = end. Numbers: drift = mean pelvis x, y distance to the reference (child metres) and
heading error = mean |facing - reference facing| (deg), over the clip.
Order and numbering: as outputs/child_all52/index.md / overview_<k>.jpg (CEM's MPJPE minus LOES's, largest first), 9 a page.
Out: outputs/child_all52/{topview_<k>.jpg, topview_all.jpg, topview.md}.
usage: uv run python scripts/child_topview.py"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent; OUT = REPO / "outputs/child_all52"; T = OUT / "trajectories"
sys.path.insert(0, str(REPO))
from crossenbodiment.geometry import facing                    # noqa: E402  root yaw - 90 deg (refactor step 3)
LINES = (("reference_body", "Reference", "#1a1a1a", "-", 2.0), ("z0", r"$z_0$", "#8c8b87", "--", 1.4),
         ("cem", "BFM-Zero", "#1baf7a", "-", 1.6), ("best", "Ours", "#2a78d6", "-", 1.6))
EVERY = 60                                                     # facing arrows every 2 s (30 fps)


def order():
    return [m.group(2) for m in re.finditer(r"^\| (\d+) \| `([^`]+)` \|", (OUT / "index.md").read_text(), re.M)]


def load(c):
    Q = {k: np.load(T / f"child_{c}" / f"{k}.npz")["qpos"].astype(np.float64) for k, *_ in LINES}
    n = {len(q) for q in Q.values()}; assert len(n) == 1, (c, n)
    r = Q["reference_body"]; M = {}
    for k in ("z0", "cem", "best"):
        q = Q[k]; M[k] = dict(drift=float(np.linalg.norm(q[:, :2] - r[:, :2], axis=1).mean()),
                              end=float(np.linalg.norm(q[-1, :2] - r[-1, :2])),
                              head=float(np.degrees(np.abs(np.angle(np.exp(1j * (facing(q) - facing(r)))))).mean()))
    return Q, M


def panel(ax, c, num, Q, M, small=False):
    P = np.concatenate([q[:, :2] for q in Q.values()]); lo, hi = P.min(0), P.max(0); span = max(float((hi - lo).max()), 0.6)
    L = 0.07 * span
    for k, lab, col, ls, lw in LINES:
        q = Q[k]; ax.plot(q[:, 0], q[:, 1], color=col, ls=ls, lw=lw, label=lab, zorder=3 if k != "reference_body" else 2)
        idx = list(range(0, len(q), EVERY)) + [len(q) - 1]; f = facing(q)[idx]
        ax.quiver(q[idx, 0], q[idx, 1], L * np.cos(f), L * np.sin(f), color=col, angles="xy", scale_units="xy", scale=1, width=0.006,
                  headwidth=4, headlength=5, zorder=4)
        ax.plot(q[-1, 0], q[-1, 1], "o", color=col, ms=4 if small else 5, zorder=5)
    ax.plot(Q["reference_body"][0, 0], Q["reference_body"][0, 1], "s", color="#1a1a1a", ms=4 if small else 5, mfc="white", zorder=6)
    mid = (lo + hi) / 2; h = 0.5 * span * 1.12
    ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[1] - h, mid[1] + h); ax.set_aspect("equal")
    ax.grid(True, color="#ececea", lw=0.5); ax.tick_params(labelsize=7 if small else 8)
    fs = 8 if small else 10
    ax.set_title(f"#{num} {c}\ndrift m: z0 {M['z0']['drift']:.2f} | BFM-Zero {M['cem']['drift']:.2f} | Ours {M['best']['drift']:.2f}\n"
                 f"heading err deg: z0 {M['z0']['head']:.0f} | BFM-Zero {M['cem']['head']:.0f} | Ours {M['best']['head']:.0f}", fontsize=fs)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    clips = order(); assert len(clips) == 52, len(clips); D = {c: load(c) for c in clips}
    note = "square = start, dot = end, arrows = facing every 2 s; seed 0"
    for p in range(0, 52, 9):
        cs = clips[p:p + 9]; fig, axs = plt.subplots(3, 3, figsize=(15, 16.2)); axs = axs.ravel()
        for i, c in enumerate(cs):
            panel(axs[i], c, p + i + 1, *D[c])
        for ax in axs[len(cs):]:
            ax.axis("off")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"Child, top view of the pelvis path (page {p // 9 + 1}; same order as overview_{p // 9 + 1}.jpg). {note}", y=0.963, fontsize=12)
        fig.subplots_adjust(left=0.05, right=0.985, bottom=0.03, top=0.905, hspace=0.42, wspace=0.25); fig.savefig(OUT / f"topview_{p // 9 + 1}.jpg", dpi=100, pil_kwargs=dict(quality=90)); plt.close(fig)
    fig, axs = plt.subplots(8, 7, figsize=(30, 38)); axs = axs.ravel()
    for i, c in enumerate(clips):
        panel(axs[i], c, i + 1, *D[c], small=True)
    for ax in axs[52:]:
        ax.axis("off")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=18, frameon=False, bbox_to_anchor=(0.5, 0.998))
    fig.suptitle(f"Child, 52 task motions: top view of the pelvis path. {note}", y=0.985, fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.955, hspace=0.5, wspace=0.28); fig.savefig(OUT / "topview_all.jpg", dpi=70, pil_kwargs=dict(quality=88)); plt.close(fig)
    rows = ["| # | clip | drift z0 | drift BFM-Zero | drift Ours | end dist z0 | end dist BFM-Zero | end dist Ours | heading err z0 | heading err BFM-Zero | heading err Ours |",
            "|" + "---|" * 11]
    for i, c in enumerate(clips):
        M = D[c][1]
        rows.append(f"| {i + 1} | `{c}` | " + " | ".join(f"{M[k]['drift']:.2f}" for k in ("z0", "cem", "best")) + " | "
                    + " | ".join(f"{M[k]['end']:.2f}" for k in ("z0", "cem", "best")) + " | " + " | ".join(f"{M[k]['head']:.0f}" for k in ("z0", "cem", "best")) + " |")
    mean = {k: {x: float(np.mean([D[c][1][k][x] for c in clips])) for x in ("drift", "end", "head")} for k in ("z0", "cem", "best")}
    closer = sum(D[c][1]["best"]["drift"] < D[c][1]["cem"]["drift"] for c in clips)
    head = (f"# Child, top view of the pelvis path\n\n{__doc__}\n\nMean over the 52 motions: drift (m) z0 {mean['z0']['drift']:.2f}, BFM-Zero "
            f"{mean['cem']['drift']:.2f}, Ours {mean['best']['drift']:.2f}; end distance (m) z0 {mean['z0']['end']:.2f}, BFM-Zero {mean['cem']['end']:.2f}, "
            f"Ours {mean['best']['end']:.2f}; heading error (deg) z0 {mean['z0']['head']:.0f}, BFM-Zero {mean['cem']['head']:.0f}, Ours {mean['best']['head']:.0f}. "
            f"Ours drifts less than BFM-Zero on {closer} of 52.\n\n")
    (OUT / "topview.md").write_text(head + "\n".join(rows) + "\n"); print(head)


if __name__ == "__main__":
    main()
