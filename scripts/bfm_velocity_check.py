#!/usr/bin/env python3
"""bfm_velocity_check.py -- B sees velocities (144 of its 358 inputs). Two ways around that for comparing motions done
at a different speed, checked against the plain B:
  full    B(obs) as everywhere else
  zero    the velocity part (local_body_vel, local_body_ang_vel: obs[214:]) set to 0 on BOTH sides
  scaled  the reference's velocity part multiplied by the speed ratio r (the reference played r times as fast)
A. does B still tell poses apart? the running clip's frame x frame cosine for full and zero, against a pose-only
   matrix (distance of the body parts relative to the pelvis, obs[1:70], metres): the spread of the off-diagonal
   cosines and the rank correlation with the pose distance.
B. is a slowed copy of the same motion penalised? the adult's running clip resampled at r = 0.75 and 0.5 (root
   position and hinge angles linear, root rotation slerp; velocities by the same 30 Hz difference, so they shrink),
   compared on the frames where the two are exactly aligned (slow frame 4k <-> original 3k, 2k <-> k): 1 - mean cos
   for full, zero and scaled; and a frozen copy (frame 0's pose held) as the cheat that must stay expensive.
C. do they agree with comparing poses? the H1 survey searches (z0 and best of each clip, SMPL-equivalent qpos) against
   their adult clips, frame by frame: full and zero B losses vs the pose error, rank correlation over all rollouts and
   per clip whether z0 -> best moves the same way. (scaled needs a speed ratio, which only an alignment gives: n/a.)
Observations are rebuilt from qpos at 30 Hz (bfm_align.obs_from_qpos) on both sides, so a rollout's full-B loss here
differs a little from the search's logged bfm (velocities from one substep there).
-> outputs/dtw/velocity_check/{similarity.png, results.json}
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/bfm_velocity_check.py"""
import glob, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model import bfm_align                                            # noqa: E402

FONT = Path.home() / ".local/share/fonts/NotoSansCJKtc-Regular.otf"
font_manager.fontManager.addfont(str(FONT))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
OUT = REPO / "outputs/dtw/velocity_check"
VEL = slice(214, 358)                                                   # local_body_vel (72) + local_body_ang_vel (72)
POS = slice(1, 70)                                                      # body positions relative to the pelvis (69)
CLIP = "move-ego-0-4/move-ego-0-4_8"


def unit(B):
    return B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12)


# resample lives in crossembodiment/retiming.py (refactor step 4, unchanged); imported here for main() and old importers
from crossembodiment.retiming import resample                          # noqa: E402,F401


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    env, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
    obs = lambda q: bfm_align.obs_from_qpos(env, q)
    assert obs(np.load(REPO / f"data/origin_motion/{CLIP}.npz")["qpos"][:2]).shape[1] == 358

    def emb(O, mode="full", r=1.0):
        O = np.array(O, dtype=np.float64)
        if mode == "zero":
            O[:, VEL] = 0
        elif mode == "scaled":
            O[:, VEL] *= r
        return bfm_align.embed(model, O.astype(np.float32), "cuda")
    res = {}

    # A. the running clip against itself
    ref = np.load(REPO / f"data/origin_motion/{CLIP}.npz")["qpos"]; Or = obs(ref)
    P = Or[:, POS].reshape(len(Or), 23, 3)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1).mean(-1)          # mean body-part distance, metres
    iu = np.triu_indices(len(Or), 1)
    S = {m: unit(emb(Or, m)) @ unit(emb(Or, m)).T for m in ("full", "zero")}
    res["A_tell_poses_apart"] = {m: dict(offdiag_cos_mean=float(S[m][iu].mean()), offdiag_cos_p5=float(np.percentile(S[m][iu], 5)),
                                         spearman_vs_pose_distance=float(-spearmanr(S[m][iu], D[iu])[0])) for m in S}

    # B. slowed copies and a frozen copy of the same clip
    Bfull = emb(Or)
    res["B_slowed_copy"] = {}
    for r, (a, b) in ((0.75, (4, 3)), (0.5, (2, 1))):
        Os = obs(resample(ref, r))
        n = min(len(Os) // a, len(Or) // b); i_s, i_o = np.arange(n) * a, np.arange(n) * b
        loss = lambda X, Y: float(1 - (unit(X) * unit(Y)).sum(-1).mean())
        res["B_slowed_copy"][f"speed {r}"] = dict(
            full=loss(emb(Os)[i_s], Bfull[i_o]), zero=loss(emb(Os, "zero")[i_s], emb(Or, "zero")[i_o]),
            scaled=loss(emb(Os)[i_s], emb(Or, "scaled", r)[i_o]), frames=int(n))
    Of = obs(np.repeat(ref[:1], len(ref), 0))
    res["B_frozen_copy"] = dict(full=float(1 - (unit(emb(Of)) * unit(Bfull)).sum(-1).mean()),
                                zero=float(1 - (unit(emb(Of, "zero")) * unit(emb(Or, "zero"))).sum(-1).mean()))

    # C. H1 rollouts vs their clips
    rows = []
    for f in sorted(glob.glob(str(REPO / "outputs/user_survey_search/h1_ridge/*/summary.json"))):
        s = json.loads(Path(f).read_text()); q0 = np.load(REPO / f"data/origin_motion/{s['clip']}.npz")["qpos"]; O0 = obs(q0)
        Bf0, Bz0 = unit(emb(O0)), unit(emb(O0, "zero"))
        for k in ("origin_z", "best"):
            qs = np.load(Path(f).parent / f"{k}.npz")["qpos_smpl"].astype(np.float64); n = min(len(qs), len(q0))
            O = obs(qs[:n])
            pe = np.linalg.norm(O[:, POS].reshape(n, 23, 3) - O0[:n, POS].reshape(n, 23, 3), axis=-1).mean()
            rows.append(dict(clip=s["clip"].split("/")[1], z=k, pose_err_m=float(pe),
                             full=float(1 - (unit(emb(O)) * Bf0[:n]).sum(-1).mean()),
                             zero=float(1 - (unit(emb(O, "zero")) * Bz0[:n]).sum(-1).mean())))
    pe = np.array([r["pose_err_m"] for r in rows])
    res["C_h1_rollouts"] = dict(
        rows=rows, **{f"spearman_{m}_vs_pose_err": float(spearmanr([r[m] for r in rows], pe)[0]) for m in ("full", "zero")},
        **{f"same_direction_as_pose_{m}": int(sum(np.sign(b[m] - a[m]) == np.sign(b["pose_err_m"] - a["pose_err_m"])
                                                  for a, b in zip(rows[::2], rows[1::2]))) for m in ("full", "zero")},
        clips=len(rows) // 2)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4), dpi=120)
    for a_, M, t, cm in ((ax[0], -D, "只比姿勢（各部位相對骨盆的位置差）", "viridis"), (ax[1], S["full"], "原本的 B（含速度）", "viridis"),
                         (ax[2], S["zero"], "速度設 0 的 B", "viridis")):
        im = a_.imshow(M, cmap=cm, origin="lower", vmin=np.percentile(M, 2), vmax=np.percentile(M, 99.5))
        a_.set_title(t, fontsize=15); a_.set_xlabel("第幾幀", fontsize=12); a_.set_ylabel("第幾幀", fontsize=12)
        fig.colorbar(im, ax=a_, fraction=0.046, pad=0.03).set_label("越亮越像", fontsize=11)
    fig.suptitle("大人跑步（move-ego-0-4_8）每兩幀之間有多像：斜的亮條紋 = 每隔一個步伐週期姿勢重複", fontsize=15)
    fig.tight_layout(); fig.savefig(OUT / "similarity.png")
    print(json.dumps({k: v for k, v in res.items() if k != "C_h1_rollouts"}, indent=1))
    c = res["C_h1_rollouts"]
    print({k: v for k, v in c.items() if k != "rows"})
    for r in rows:
        print(f"  {r['clip']:22s} {r['z']:8s} pose err {r['pose_err_m']:.3f} m | full {r['full']:.3f} | zero {r['zero']:.3f}")


if __name__ == "__main__":
    main()
