#!/usr/bin/env python3
"""child_52_cem_phc.py -- Comparison A's CEM (BFM-Zero) searched again with BFM-Zero's tracking reward
(scripts/single_z_search.py --objective phc: 1 - mean_t PHC tracking reward, model/phc_reward.py) instead of our cost:
no heading, position or anchor term; every other setting as outputs/child_52_rerun's CEM (52 motions of
outputs/child_52/clips.txt, 5 seeds, 64 x 16 rollouts, sigma 0.05, 4 elites, std floor 0.01, the reference velocity at the
start, q[t] scored against ref[t + 1]). The old CEM runs stay in outputs/child_52_rerun.
  commands       "<out dir> <command>" per search, into outputs/child_52_cem_phc/<stem>_cem_s<seed>
  prepare        outputs/child_52_cem_phc/rollouts/<stem>/{z.npy, index.json}: each run's distinct best z so far at every
                 generation and its centre after 4, 8, 16, 32, 64 generations (as child_52_rerun.py)
  eval_commands  "<out dir> <command>" per motion: those latents rolled out with --eval-z --eval-save-qpos
  analyse        metrics (q[t] against ref[t + 1]) -> per_run.npz (best_cem, mean_cem: motions x seeds x gens x metrics, z0)
                 and means.json (best / centre by budget; LOES and the old CEM of child_52_rerun against this one).
                 Checks: settings match the old CEM runs apart from the cost; rollouts reproduce each search's best_so_far;
                 the z0 rollout equals child_52_rerun's.
usage: uv run python scripts/child_52_cem_phc.py commands | prepare | eval_commands | analyse"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
SRC = REPO / "outputs/child_52"; OLD = REPO / "outputs/child_52_rerun"; OUT = REPO / "outputs/child_52_cem_phc"; ROLL = OUT / "rollouts"
SEEDS = range(5); G = (4, 8, 16, 32, 64); XML = "assets/robots_torque/child/robot_torque_full.xml"
SAME = ("evals", "generations", "pairs", "sigma", "lr", "steps", "fitness", "sampling", "init_vel_scale", "pos_unit", "pos_div",
        "obs_scale", "init", "bfm_zero_vel", "fall_weight", "seed", "algo", "frame_offset")
WANT = dict(objective="phc", heading_weight=0.0, pos_weight=0.0, anchor_weight=0.0)
body = None


def clips():
    return [l.strip() for l in open(SRC / "clips.txt")]


def base_cmd(c):
    return (f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {XML} --objective phc --algo cem --evals 1024 "
            f"--pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --eval-every 0 --init-vel-scale 1")


def commands():
    for c in clips():
        for s in SEEDS:
            d = f"outputs/child_52_cem_phc/{c.split('/')[1]}_cem_s{s}"
            print(d, f"{base_cmd(c)} --seed {s} --out {d}")


def prepare():
    for c in clips():
        stem = c.split("/")[1]; d = ROLL / stem; d.mkdir(parents=True, exist_ok=True); Z, idx = [], {}
        for s in SEEDS:
            r = f"{stem}_cem_s{s}"; t = np.load(OUT / r / "z_trace.npz"); b, m = t["z_best"], t["z_mean"]; assert len(b) == 65 and len(m) == 65, r; best = []
            for g in range(1, 65):
                if not best or np.abs(b[g] - b[g - 1]).max() > 0:
                    Z.append(b[g]); best.append(len(Z) - 1)
                else:
                    best.append(best[-1])
            centre = list(range(len(Z), len(Z) + len(G))); Z += [m[g] for g in G]; idx[r] = dict(best=best, centre=centre)
        np.save(d / "z.npy", np.stack(Z)); (d / "index.json").write_text(json.dumps(dict(clip=c, map=idx)))
    print("prepared")


def eval_commands():
    for c in clips():
        d = f"outputs/child_52_cem_phc/rollouts/{c.split('/')[1]}"
        print(d, f"{base_cmd(c)} --seed 0 --eval-z {d}/z.npy --eval-save-qpos --out {d}")


def clip(c):
    global body
    import eval_metrics as E
    from child_52_means import full, KEYS
    body = body or E.Body(E.XML["child"], True); stem = c.split("/")[1]; ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)[1:]; d = ROLL / stem
    ix = json.load(open(d / "index.json"))["map"]; Q = np.load(d / "eval_qpos.npz")["qpos"].astype(np.float64); cost = [float(r["cost"]) for r in csv.DictReader(open(d / "eval.csv"))]
    base = json.load(open(d / "eval_base.json"))["cost"]; V = {}
    def m(i):
        if i not in V:
            v = full(body, Q[i], ref, cost[i] / base); V[i] = [v[k] for k in KEYS]
        return V[i]
    B = np.zeros((5, 64, len(KEYS))); C = np.zeros((5, len(G), len(KEYS))); worst = 0.0; bad = []
    for s in SEEDS:
        r = f"{stem}_cem_s{s}"; old = json.load(open(OLD / r / "summary.json")); new = json.load(open(OUT / r / "summary.json"))
        bad += [(r, k, old.get(k), new.get(k)) for k in SAME if old.get(k) != new.get(k)] + [(r, k, new.get(k)) for k, v in WANT.items() if new.get(k) != v]
        for j, i in enumerate(ix[r]["best"]):
            B[s, j] = m(i)
        for j, i in enumerate(ix[r]["centre"]):
            C[s, j] = m(i)
        bs = np.array([float(x["best_so_far"]) for x in csv.DictReader(open(OUT / r / "curve.csv"))]); worst = max(worst, float(np.abs(np.array(cost)[ix[r]["best"]] - bs).max()))
    z0q = np.load(OUT / f"{stem}_cem_s0/origin_z.npz")["qpos"]; same_z0 = float(np.abs(z0q - np.load(OLD / f"{stem}_es_s0/origin_z.npz")["qpos"]).max())
    return stem, B, C, worst, bad, same_z0


def analyse():
    from child_52_means import KEYS
    cl = clips()
    with Pool(24) as p:
        res = p.map(clip, cl)
    B = np.stack([r[1] for r in res]); C = np.stack([r[2] for r in res]); bad = [x for r in res for x in r[4]]
    print("largest rollout vs search best_so_far:", max(r[3] for r in res), "| settings not as intended:", len(bad), bad[:6],
          "| z0 rollout against child_52_rerun's, largest difference:", max(r[5] for r in res), flush=True)
    old = np.load(OLD / "per_run.npz"); assert list(old["clips"]) == cl and list(old["keys"]) == list(KEYS)
    np.savez_compressed(OUT / "per_run.npz", clips=np.array(cl), keys=np.array(KEYS), gens_mean_z=np.array(G), z0=old["z0"], best_cem=B, mean_cem=C)
    m = lambda A: {k: float(A[..., i].mean()) for i, k in enumerate(KEYS)}
    test = lambda X, Y: {k: dict(lower=int((X[..., i].mean(1) < Y[..., i].mean(1)).sum()), p=float(wilcoxon(X[..., i].mean(1), Y[..., i].mean(1)).pvalue)) for i, k in enumerate(KEYS) if k != "ratio"}
    rep = dict(z0=m(old["z0"]), best={str(g * 16): m(B[:, :, g - 1]) for g in G}, centre={str(g * 16): m(C[:, :, j]) for j, g in enumerate(G)},
               loes_vs_cem_phc=dict(best=test(old["best_es"][:, :, -1], B[:, :, -1]), centre=test(old["mean_es"][:, :, -1], C[:, :, -1])),
               cem_phc_vs_cem_ours=dict(best=test(B[:, :, -1], old["best_cem"][:, :, -1]), centre=test(C[:, :, -1], old["mean_cem"][:, :, -1])))
    (OUT / "means.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    {"commands": commands, "prepare": prepare, "eval_commands": eval_commands, "analyse": analyse}[sys.argv[1]]()
