#!/usr/bin/env python3
"""child_lso_vs_loes.py -- the child on three motions, five seeds: per-step latents (native tracking, LSO, outputs/lso_vs_loes) against one latent (LOES, outputs/child_new_test), on the metrics of Comparison A (scripts/child_52_means.py), never on cost (LOES's also has the anchor term).
LOES (outputs/child_new_test) was searched with --frame-offset 0, LOES aligned (outputs/loes_aligned) with --frame-offset 1. A rollout's q[t] is the state after step t, which goes with ref[t + 1]. "next" compares them that way; "same" compares q[t] with ref[t], as the tables of Comparison A do.
Native tracking and B(g_t) are the same for every seed when the file's rollouts are identical (checked below). LSO at <k> rollouts is the mean sequence after k / 128 updates.
Writes outputs/lso_vs_loes/report.json. usage: uv run python scripts/child_lso_vs_loes.py"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import eval_metrics as E                                                  # noqa: E402
from child_52_means import full                                           # noqa: E402
OUT = REPO / "outputs/lso_vs_loes"; LOES = REPO / "outputs/child_new_test"; CLIPS = ["move-ego-0-2/move-ego-0-2_2", "move-ego-0-4/move-ego-0-4_8", "headstand/headstand_4"]; SEEDS = range(5)
KEYS = ("fall_s", "success", "mpjpe", "mmpjpe", "mpjae", "root_cm")
METHODS = [("z0", "z0 (1 rollout)"), ("loes", "LOES best z (1,024)"), ("loes_aligned", "LOES aligned search (1,024)"), ("q_ER8", "Native tracking, ER L=8"), ("q_ER5", "Native tracking, ER L=5"), ("q_Bg", "Native tracking, B(g_t)"), ("q_it8", "LSO mean (1,024)"), ("q_it16", "LSO mean (2,048)"), ("q_it32", "LSO mean (4,096)"), ("q_it64", "LSO mean (8,192)"), ("q_LSO", "LSO mean (16,384)"), ("q_LSO_best", "LSO best (16,384)")]


def score(body, q, ref, offset):
    qr = ref[offset:]; T = min(len(q), len(qr)); m = full(body, q[:T], qr[:T], 1.0); m["root_cm"] = float(100 * np.linalg.norm(q[:T, :2] - qr[:T, :2], axis=1).mean()); return [m[k] for k in KEYS]


def main():
    body = E.Body(E.XML["child"], True); V = {o: {m: np.full((len(CLIPS), 5, len(KEYS)), np.nan) for m, _ in METHODS} for o in ("next", "same")}; checks = {}
    for ci, c in enumerate(CLIPS):
        stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); first = {}
        for s in SEEDS:
            R = np.load(OUT / f"{stem}_s{s}/rollouts.npz"); assert np.abs(R["ref"] - ref).max() < 1e-5, (stem, s)
            d = LOES / f"{stem}_es_s{s}"; Q = {k: R[k].astype(np.float64) for k, _ in METHODS if k in R.files}; Q["loes"] = np.load(d / "eval/eval_qpos.npz")["qpos"][0].astype(np.float64); Q["loes_aligned"] = np.load(REPO / f"outputs/loes_aligned/{stem}_s{s}/eval/eval_qpos.npz")["qpos"][0].astype(np.float64); Q["z0"] = np.load(d / "origin_z.npz")["qpos"].astype(np.float64).reshape(-1, ref.shape[1])
            for o, off in (("next", 1), ("same", 0)):
                for k, q in Q.items():
                    V[o][k][ci, s] = score(body, q, ref, off)
            for k in ("q_ER8", "q_ER5", "q_Bg", "z0"):
                first.setdefault(k, Q[k]); checks[f"{stem} {k} max change over seeds"] = max(checks.get(f"{stem} {k} max change over seeds", 0.0), float(np.abs(Q[k] - first[k]).max()))
            if s == 0:
                checks[f"{stem} first frame"] = {k: dict(to_ref0=float(np.abs(Q[k][0] - ref[0]).max()), to_ref1=float(np.abs(Q[k][0] - ref[1]).max()), frames=len(Q[k])) for k in ("z0", "loes", "q_ER8", "q_it8", "q_LSO")}
        print(stem, flush=True)
    rep = dict(keys=KEYS, clips=CLIPS, checks=checks, table={o: {m: dict(all=dict(zip(KEYS, np.nanmean(V[o][m], (0, 1)).tolist())), per_clip={c.split("/")[1]: dict(zip(KEYS, np.nanmean(V[o][m][i], 0).tolist())) for i, c in enumerate(CLIPS)}, seed_range=dict(mpjpe=[float(V[o][m][:, :, 2].mean(0).min()), float(V[o][m][:, :, 2].mean(0).max())], mpjae=[float(V[o][m][:, :, 4].mean(0).min()), float(V[o][m][:, :, 4].mean(0).max())])) for m, _ in METHODS} for o in V})
    (OUT / "report.json").write_text(json.dumps(rep, indent=1)); np.savez_compressed(OUT / "per_run.npz", keys=np.array(KEYS), clips=np.array(CLIPS), **{f"{o}_{m}": V[o][m] for o in V for m in V[o]})
    print(json.dumps(checks, indent=0))
    for o in ("next", "same"):
        print(f"\n===== q[t] vs ref[t{'+1' if o == 'next' else ''}], mean over 3 clips x 5 seeds\n{'':28s}" + "".join(f"{k:>10s}" for k in KEYS))
        for m, name in METHODS:
            print(f"{name:28s}" + "".join(f"{x:10.3f}" for x in np.nanmean(V[o][m], (0, 1))))
        for i, c in enumerate(CLIPS):
            print(f"-- {c.split('/')[1]}")
            for m, name in METHODS:
                print(f"{name:28s}" + "".join(f"{x:10.3f}" for x in np.nanmean(V[o][m][i], 0)))


if __name__ == "__main__":
    main()
