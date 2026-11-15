#!/usr/bin/env python3
"""locked_action_trace.py -- over a search's 256 steps, does the policy's command to the LIMITED joints narrow?

locked_action_check.py compares z0 with the final best z. This follows every step: each row of the run's z_trace.npz
(z_mean = the search point after k steps, z_best = the best sample so far) is re-rolled through
single_z_search.py --eval-z with the run's own settings (actions saved), and for the run's limited joints it reports
the commanded PD target angle target = (gain * u + bias0) / kp as
  range   max_t - min_t of the target (deg), averaged over the limited joints -- how far the command still swings
  off     mean_t |target - lock angle| (deg), averaged over the limited joints
  within  share of (frame, joint) whose target is within --tol deg of the lock angle
plus the joints' actual angle off the lock (they are held to +-1 deg by the joint limit either way) and the cost,
which is checked against the run's own z0 and final best.
Writes outputs/lock_knee/<run>/act_trace/{trace.csv, ...} and outputs/lock_knee/locked_action_trace.png.
usage: uv run python scripts/locked_action_trace.py [adult_lockRleg_bent_lim_s0 adult_lockRleg_bent_lim_origref_s0]
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "outputs/lock_knee"
sys.path.insert(0, str(REPO / "scripts"))
from locked_action_check import locks_of, targets   # noqa: E402


def eval_run(run, gpu):
    d = RUNS / run; s = json.loads((d / "summary.json").read_text())
    out = d / "act_trace"; out.mkdir(exist_ok=True)
    zt = np.load(d / "z_trace.npz")
    np.save(out / "z.npy", np.concatenate([zt["z_mean"], zt["z_best"]]).astype(np.float64))
    if not (out / "eval_act.npz").exists():
        cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", s["clip"], "--body", s["body"],
               "--xml", s["xml"], "--ref", s["ref"], "--init", s["init"], "--obs-scale", s["obs_scale"],
               "--objective", s["objective"], "--pairs", str(s["pairs"]), "--steps", str(s["steps"]),
               "--heading-weight", str(s["heading_weight"]), "--pos-weight", str(s["pos_weight"]),
               "--anchor", s["anchor"], "--anchor-weight", str(s["anchor_weight"]),
               "--eval-z", str(out / "z.npy"), "--eval-save-qpos", "--eval-save-act", "--out", str(out)]
        if s.get("init_ref"):
            cmd += ["--init-ref", s["init_ref"]]
        subprocess.run(cmd, cwd=REPO, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "HF_HUB_OFFLINE": "1"},
                       stdout=open(out / "eval.log", "w"), stderr=subprocess.STDOUT)
    return s, zt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["adult_lockRleg_bent_lim_s0", "adult_lockRleg_bent_lim_origref_s0"])
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--gpu", default="2")
    a = ap.parse_args()
    m_adult = mujoco.MjModel.from_xml_path(str(REPO / "assets/robots/adult/robot.xml"))
    R = {}
    for run in a.runs:
        s, zt, out = eval_run(run, a.gpu)
        n = len(zt["z_mean"])
        cost = np.array([float(r["cost"]) for r in csv.DictReader(open(out / "eval.csv"))])
        # the re-rolls must be the search's own rollouts: z0 and the final best reproduce its costs
        for i, ref in ((0, s["origin_z"]["cost"]), (2 * n - 1, s["best"]["cost"])):
            assert abs(cost[i] - ref) < 1e-5, f"{run}: row {i} cost {cost[i]} vs the search's {ref}"
        q = np.load(out / "eval_qpos.npz")["qpos"]; act = np.load(out / "eval_act.npz")["act"]
        lk = locks_of(s["body"]); J = list(lk); lock = np.array([lk[j] for j in J])
        mb = mujoco.MjModel.from_xml_path(str(REPO / s["xml"]))
        jq = [mb.joint(j).qposadr[0] for j in J]
        tg = np.degrees(np.stack([targets(mb, act[i], J) for i in range(len(act))]))      # (rows, T, joints)
        dev = np.abs(tg - np.degrees(lock))
        met = dict(cost=cost, range=(tg.max(1) - tg.min(1)).mean(1), off=dev.mean((1, 2)),
                   within=(dev < a.tol).mean((1, 2)), joint_off=np.degrees(np.abs(q[:, :, jq] - lock)).mean((1, 2)),
                   per_joint_off=dev.mean(1))
        ad = np.degrees(targets(m_adult, np.load(REPO / f"data/origin_action/{s['clip']}.npz")["action"], J))
        R[run] = dict(J=J, n=n, met=met, adult_range=float((ad.max(0) - ad.min(0)).mean()))
        with open(out / "trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "which", "cost", "target_range_deg", "target_off_deg", f"within_{a.tol:g}deg",
                        "joint_off_deg"] + [f"off_{j}" for j in J])
            for i in range(2 * n):
                w.writerow([i % n, "mean" if i < n else "best"] + [f"{met[k][i]:.6g}" for k in
                           ("cost", "range", "off", "within", "joint_off")] + [f"{v:.4g}" for v in met["per_joint_off"][i]])
        for which, sl in (("search point", slice(0, n)), ("best so far", slice(n, 2 * n))):
            m = {k: v[sl] for k, v in met.items()}
            print(f"{run} [{which}] step 0 -> 64 -> 128 -> 256: range " +
                  " -> ".join(f"{m['range'][k]:.1f}" for k in (0, 64, 128, n - 1)) + " deg | off " +
                  " -> ".join(f"{m['off'][k]:.1f}" for k in (0, 64, 128, n - 1)) + f" deg | within {a.tol:g} deg " +
                  " -> ".join(f"{m['within'][k]:.0%}" for k in (0, 64, 128, n - 1)) + " | cost " +
                  " -> ".join(f"{m['cost'][k]:.3f}" for k in (0, 64, 128, n - 1)) +
                  f" | joint off lock max {m['joint_off'].max():.2f} deg")
        print(f"   adult's own walk: target range {R[run]['adult_range']:.1f} deg on the same joints")
    (RUNS / "locked_action_trace.json").write_text(json.dumps(
        {r: {"joints": v["J"], "adult_range": v["adult_range"],
             **{k: v["met"][k].tolist() for k in ("cost", "range", "off", "within", "joint_off")}} for r, v in R.items()}))
    plot(R, a.tol)


def plot(R, tol):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"adult_lockRleg_bent_lim_s0": "#2a78d6", "adult_lockRleg_bent_lim_origref_s0": "#eb6834"}
    lab = {"adult_lockRleg_bent_lim_s0": "target: bent-leg walk",
           "adult_lockRleg_bent_lim_origref_s0": "target: the adult's original walk"}
    fig, axs = plt.subplots(1, 4, figsize=(26, 6.4), dpi=120)
    panels = (("cost", "cost (lower is better)", "Search cost"),
              ("range", "max - min of the target angle (deg)", "How far the command to the limited joints swings"),
              ("off", "mean |target - lock angle| (deg)", "How far the command is from the lock angle"),
              ("within", f"share of frames within {tol:g} deg of the lock", "How often the command sits at the lock"))
    for run, v in R.items():
        n, m = v["n"], v["met"]; x = np.arange(n)
        for ax, (k, yl, t) in zip(axs, panels):
            y = m[k] * (100 if k == "within" else 1)
            ax.plot(x, y[:n], color=col.get(run), lw=1.2, alpha=.9, label=f"{lab.get(run, run)}: search point")
            ax.plot(x, y[n:], color=col.get(run), lw=2.4, ls="--", label=f"{lab.get(run, run)}: best so far")
    ref = next(iter(R.values()))["adult_range"]
    axs[1].axhline(ref, color="#888", ls=":", lw=1.2); axs[1].text(256, ref, "adult's own walk", ha="right", va="bottom",
                                                                   fontsize=11, color="#555")
    axs[1].axhline(2.0, color="#888", ls=":", lw=1.2); axs[1].text(256, 2.0, "joint limit band (+-1 deg)", ha="right",
                                                                   va="bottom", fontsize=11, color="#555")
    for ax, (k, yl, t) in zip(axs, panels):
        ax.set_title(t, fontsize=14); ax.set_xlabel("search step", fontsize=13); ax.set_ylabel(yl + (" (%)" if k == "within" else ""), fontsize=12)
        ax.grid(alpha=.25); ax.set_xlim(0, 256)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].legend(fontsize=10.5, loc="upper right")
    fig.suptitle("Right leg held bent by joint limits (12 joints, motors kept): does the policy stop commanding them?",
                 fontsize=16)
    fig.tight_layout(); p = RUNS / "locked_action_trace.png"; fig.savefig(p); print("->", p)


if __name__ == "__main__":
    main()
