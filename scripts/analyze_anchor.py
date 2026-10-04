#!/usr/bin/env python3
"""analyze_anchor.py -- does pulling the search toward its own start keep nearby starts nearby?

Walking (move-ego-0-2_4), the 8 starts (z0 + rotations 0.5..30 deg along one direction), each
tracking its own adult rollout on the child body, two stages (bfm 4096 + L_align 2048), with
+ lambda * (|z - start| / 16)^2 in both stages. lambda = 0 is outputs/latent_transfer_own (no
penalty); 0.03 / 0.1 / 0.3 are outputs/latent_transfer_anchor/move-ego-0-2_4/lam<lambda>.

Per lambda: Euclidean distances (latents at radius 16) between the 8 results and from each result
to its own start; tracking quality = L_align and joint difference of each result's rollout against
its OWN target (no penalty in these numbers); behaviour difference against the z0 start's result.
Writes outputs/latent_transfer_anchor/anchor_summary.csv and pair_distances_anchor.png.
"""
import csv, itertools, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
N = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
STEM = "move-ego-0-2_4"
RUNS = {"0": REPO / f"outputs/latent_transfer_own/{STEM}/align"}
RUNS.update({lam: REPO / f"outputs/latent_transfer_anchor/{STEM}/lam{lam}/align" for lam in ("0.03", "0.1", "0.3")})
COL = {"0": "#8a8983", "0.03": "#2a78d6", "0.1": "#1baf7a", "0.3": "#eb6834"}


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float64)
    return 16 * v / np.linalg.norm(v)


def main():
    import mujoco
    from model import losses
    from model.simple.config import ESConfig
    c = ESConfig()
    W = {"root": c.d_root_weight, "ee": c.d_ee_weight, "contact": c.d_contact_weight,
         "pose": c.d_pose_weight, "velocity": c.d_velocity_weight}
    fk = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots_torque/child/robot_torque_full.xml"))
    S = [ld(REPO / f"outputs/latent_transfer/{STEM}/starts/{n}.npy") for n in N]
    T = [np.load(REPO / f"outputs/latent_transfer_own/{STEM}/targets/{n}.npz")["qpos"] for n in N]
    P = list(itertools.combinations(range(8), 2))
    xs = np.array([np.linalg.norm(S[i] - S[j]) for i, j in P])
    rows, pairs = [], {}
    for lam, D in RUNS.items():
        if not all((D / n / "summary.json").exists() for n in N):
            print(f"lambda {lam}: not complete"); continue
        Z = [ld(D / n / "best_z.npy") for n in N]
        Q = [np.load(D / n / "best.npz")["qpos"][:300] for n in N]
        la = [float(losses.functional_equivalence(fk, Q[i], T[i][:300], W, 1 / c.control_fps)[0]) for i in range(8)]
        jt = [float(np.degrees(np.abs(Q[i][:, 7:] - T[i][:300, 7:])).mean()) for i in range(8)]
        jz = [float(np.degrees(np.abs(Q[i][:, 7:] - Q[0][:, 7:])).mean()) for i in range(1, 8)]
        moved = [float(np.linalg.norm(Z[i] - S[i])) for i in range(8)]
        ys = np.array([np.linalg.norm(Z[i] - Z[j]) for i, j in P]); pairs[lam] = ys
        rows.append(dict(lam=lam, pair_mean=ys.mean(), pair_min=ys.min(), pair_max=ys.max(),
                         moved_mean=np.mean(moved), moved_min=min(moved), moved_max=max(moved),
                         l_align_own=np.mean(la), l_align_own_max=max(la), joint_own=np.mean(jt),
                         joint_vs_z0result=np.mean(jz), corr_before_after=float(np.corrcoef(xs, ys)[0, 1])))
    out = REPO / "outputs/latent_transfer_anchor"
    with open(out / "anchor_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"starts: pairwise distance {xs.mean():.2f} [{xs.min():.2f}-{xs.max():.2f}]")
    print(f"{'lambda':>6s} | {'pairwise after':>20s} | {'moved from start':>18s} | {'L_align own':>14s} | {'joint own':>9s} | {'joint vs z0 result':>18s} | corr(before, after)")
    for r in rows:
        print(f"{r['lam']:>6s} | {r['pair_mean']:6.2f} [{r['pair_min']:5.2f}-{r['pair_max']:5.2f}] | {r['moved_mean']:5.2f} [{r['moved_min']:4.1f}-{r['moved_max']:4.1f}] | "
              f"{r['l_align_own']:.3f} (max {r['l_align_own_max']:.3f}) | {r['joint_own']:7.2f}   | {r['joint_vs_z0result']:16.2f}   | {r['corr_before_after']:+.2f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0); R = rng.standard_normal((4000, 256)); R = 16 * R / np.linalg.norm(R, axis=1, keepdims=True)
    rand = float(np.linalg.norm(R[:2000] - R[2000:], axis=1).mean())
    fig, ax = plt.subplots(figsize=(7.6, 6.9)); fig.patch.set_facecolor(SURF)
    ax.plot([0, 9], [0, 9], color=INK3, ls=(0, (4, 3)), lw=1.2)
    ax.text(6.6, 5.6, "after = before", ha="left", va="top", fontsize=9, color=INK2)
    ax.axhline(rand, color="#e34948", ls=(0, (4, 3)), lw=1.2)
    ax.text(8.9, rand + 0.3, f"two random latents: {rand:.1f}", ha="right", va="bottom", fontsize=9, color="#e34948")
    for lam, ys in pairs.items():
        ax.scatter(xs, ys, s=40, color=COL[lam], edgecolor=INK, lw=.4, zorder=3,
                   label=("no penalty" if lam == "0" else f"lambda {lam}") + f": after {ys.mean():.2f} on average")
    ax.set_xlim(0, 9); ax.set_ylim(0, 24)
    ax.set_xlabel("distance between the two starts (adult latents)", fontsize=10)
    ax.set_ylabel("distance between their latents after the search on the child", fontsize=10)
    ax.set_title("walking, each start tracks its own adult rollout, two-stage L_align\n"
                 "+ lambda (|z - start| / 16)^2 in both stages; 28 pairs of the 8 starts per setting", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_facecolor(SURF); ax.grid(alpha=.22, color=INK3, lw=.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "pair_distances_anchor.png", dpi=150, facecolor=SURF)
    print(f"-> {out / 'anchor_summary.csv'}\n-> {out / 'pair_distances_anchor.png'}")


if __name__ == "__main__":
    main()
