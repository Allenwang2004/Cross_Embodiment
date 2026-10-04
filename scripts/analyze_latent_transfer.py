#!/usr/bin/env python3
"""analyze_latent_transfer.py -- step 3 of the latent-vs-behaviour study: do latents
that start close to z0 stay close after each is searched on the child body?

Inputs (outputs/latent_transfer/<stem>/, from outputs/latent_transfer/driver.sh):
  starts/<name>.npy           z0 and z0 rotated 0.5..30 deg (direction 0), the adult study's starts
  {global,align}/<name>/      the two-stage result on the child: bfm stage (4096 evals, not
                              analysed here), then 2048 evals on bfm + heading + root-xy
                              ("global") or on L_align ("align")

For each (clip, version) the reference point after search is z0', the result of the
search that started from z0 itself. Per start:
  start angle     angle(start, z0)            (and Euclidean distance, |z| = 16)
  after angle     angle(z', z0')
  L_align, joint  the z' rollout against the z0' rollout, both on the child from the
                  child reference's frame 0 (joint = mean |dq| over the 69 joints, deg)

Writes per (clip, version): a render spec and table, then (with --render) the video
outputs/latent_transfer/<stem>/<version>_child.mp4. Then the PCA figure for the group
whose 8 searched latents are closest together (smallest mean pairwise angle):
outputs/latent_transfer/pca_<stem>_<version>.png.
"""
import argparse, csv, itertools, json, subprocess, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
NAMES = ["z0"] + [f"rot{a}_0" for a in ("0.5", "1", "2", "5", "10", "20", "30")]
CLIPS = {"headstand_3": "headstand", "move-ego-0-2_4": "move-ego-0-2"}
VERSIONS = {"global": "bfm + heading + root position", "align": "L_align"}
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / np.linalg.norm(v)


def deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def eu(d):
    return 2 * 16 * np.sin(np.radians(d) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--root", default="outputs/latent_transfer",
                    help="outputs/latent_transfer (every start tracks the clip's reference) or "
                         "outputs/latent_transfer_own (every start tracks its own adult rollout, <root>/<stem>/targets/)")
    ap.add_argument("--pca", default=None, help="<stem>:<version> to draw instead of the most concentrated group")
    a = ap.parse_args()
    import mujoco
    from model import losses
    from model.simple.config import ESConfig
    c = ESConfig()
    W = {"root": c.d_root_weight, "ee": c.d_ee_weight, "contact": c.d_contact_weight,
         "pose": c.d_pose_weight, "velocity": c.d_velocity_weight}
    X = "assets/robots_torque/child/robot_torque_full.xml"
    fk = mujoco.MjModel.from_xml_path(str(REPO / X))
    groups = {}
    for stem, task in CLIPS.items():
        R = REPO / a.root / stem
        S = {n: np.load(REPO / "outputs/latent_transfer" / stem / "starts" / f"{n}.npy") for n in NAMES}
        own = {n: np.load(R / "targets" / f"{n}.npz")["qpos"] for n in NAMES} if (R / "targets").exists() else None
        for v, vname in VERSIONS.items():
            if not all((R / v / n / "summary.json").exists() for n in NAMES):
                print(f"{stem} {v}: not complete yet"); continue
            Z = {n: np.load(R / v / n / "best_z.npy").reshape(-1) for n in NAMES}
            Q = {n: np.load(R / v / n / "best.npz")["qpos"] for n in NAMES}
            cost = {n: json.loads((R / v / n / "summary.json").read_text())["best"]["cost"] for n in NAMES}
            rows, panels = [], []
            for n in NAMES:
                d0, d1 = deg(S[n], S["z0"]), deg(Z[n], Z["z0"])
                T = min(len(Q[n]), len(Q["z0"]))
                la, _ = losses.functional_equivalence(fk, Q[n][:T], Q["z0"][:T], W, 1 / c.control_fps)
                jd = float(np.degrees(np.abs(Q[n][:T, 7:] - Q["z0"][:T, 7:])).mean())
                extra = {}
                if own is not None:
                    To = min(len(Q[n]), len(own[n]))
                    lo, _ = losses.functional_equivalence(fk, Q[n][:To], own[n][:To], W, 1 / c.control_fps)
                    extra = dict(l_align_vs_own_target=float(lo),
                                 joint_vs_own_target=float(np.degrees(np.abs(Q[n][:To, 7:] - own[n][:To, 7:])).mean()))
                rows.append(dict(start=n, start_deg=d0, start_eu=eu(d0), after_deg=d1, after_eu=eu(d1),
                                 moved_deg=deg(S[n], Z[n]), l_align_vs_z0p=float(la), joint_vs_z0p=jd, cost=cost[n], **extra))
                title = "from z0" if n == "z0" else f"from z0 + {n[3:-2]} deg"
                panels.append({"title": title,
                               "sub": f"{d0:.1f} deg/{eu(d0):.2f} -> {d1:.1f} deg/{eu(d1):.2f}, L_align {la:.3f}, joint {jd:.2f} deg",
                               "xml": X, "qpos": str(R / v / n / "best.npz")})
            with open(R / f"{v}_child.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            spec = REPO / "outputs/video_specs" / f"{Path(a.root).name}_{stem}_{v}.json"
            json.dump({"out": str(R / f"{v}_child.mp4"), "cols": 4, "size": 320, "panels": panels,
                       "track": {"distance": 3.3, "elevation": -8, "azimuth": 45, "lookat_z": 0.55}},
                      open(spec, "w"), indent=1)
            after = [deg(Z[i], Z[j]) for i, j in itertools.combinations(NAMES, 2)]
            before = [deg(S[i], S[j]) for i, j in itertools.combinations(NAMES, 2)]
            groups[(stem, v)] = (np.mean(after), S, Z)
            print(f"\n== {stem} / {vname}: pairwise angle among the 8 latents, before {np.mean(before):.1f} deg -> after {np.mean(after):.1f} deg")
            print(f"{'start':10s} {'start->z0':>16s} {'after->z0p':>16s} {'moved':>6s} {'L_align':>8s} {'joint':>7s} {'cost':>6s}"
                  + ("  | vs own target: L_align  joint" if own is not None else ""))
            for r in rows:
                print(f"{r['start']:10s} {r['start_deg']:6.1f} ({r['start_eu']:5.2f}) {r['after_deg']:6.1f} ({r['after_eu']:5.2f}) "
                      f"{r['moved_deg']:6.1f} {r['l_align_vs_z0p']:8.3f} {r['joint_vs_z0p']:7.2f} {r['cost']:6.3f}"
                      + (f"  |              {r['l_align_vs_own_target']:7.3f} {r['joint_vs_own_target']:6.2f}" if own is not None else ""))
            if a.render:
                subprocess.run([sys.executable, str(REPO / "scripts/render_panels.py"), str(spec)], check=True,
                               stdout=subprocess.DEVNULL)
                (R / f"{v}_child.mp4.png").rename(R / f"{v}_child_sheet.png")
                print(f"-> {R / f'{v}_child.mp4'}")
    if not groups:
        return
    if a.pca:
        stem, v = a.pca.split(":"); spread, S, Z = groups[(stem, v)]
        print(f"\nPCA group (requested): {stem} / {VERSIONS[v]}, mean pairwise angle after {spread:.1f} deg")
    else:
        (stem, v), (spread, S, Z) = min(groups.items(), key=lambda kv: kv[1][0])
        print(f"\nPCA group (closest after search): {stem} / {VERSIONS[v]}, mean pairwise angle after {spread:.1f} deg")
    P = np.stack([unit(S[n]) for n in NAMES] + [unit(Z[n]) for n in NAMES])
    mu = P.mean(0); U, sv, Vt = np.linalg.svd(P - mu, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum(); Y = (P - mu) @ Vt[:2].T
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(7.2, 6.2)); fig.patch.set_facecolor(SURF)
    angs = [0.0, 0.5, 1, 2, 5, 10, 20, 30]; col = cm.viridis(np.linspace(0, .9, len(NAMES)))
    for i, n in enumerate(NAMES):
        b, e = Y[i], Y[i + len(NAMES)]
        ax.annotate("", xy=e, xytext=b, arrowprops=dict(arrowstyle="->", color=col[i], lw=1.2, alpha=.8))
        ax.scatter(*b, s=70, color=col[i], marker="o", edgecolor=INK, lw=.6, zorder=3)
        ax.scatter(*e, s=90, color=col[i], marker="^", edgecolor=INK, lw=.6, zorder=3)
        ax.annotate(f"{angs[i]:g}°", e, textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK2)
    ax.scatter([], [], s=60, marker="o", color=INK3, label="start (adult latent: z0 rotated by the angle shown)")
    ax.scatter([], [], s=70, marker="^", color=INK3, label="after search on the child body")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.set_title(f"{stem}, {VERSIONS[v]}: the 16 latents (unit vectors), PCA fit on all 16", fontsize=10.5)
    ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); out = REPO / a.root / f"pca_{stem}_{v}.png"
    fig.savefig(out, dpi=150, facecolor=SURF); print(f"-> {out}  (PC1+PC2 explain {ev[0] + ev[1]:.0%})")


if __name__ == "__main__":
    main()
