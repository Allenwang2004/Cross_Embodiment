#!/usr/bin/env python3
"""test_exact_obs.py -- z0 on the child with three versions of what the actor sees.

  none    the child's raw proprio
  mult    raw x the fixed per-feature multiplier (model/obs_scale.py) -- what every run so far used
  exact   the adult-equivalent observation by reverse retargeting (model/exact_obs.py)

Rolled out on the 47 b500 test clips (child torque XML, reference init, 300 steps, the b500 ES run's cfg),
for z0 and for the b500 search target z_align (found under `mult`). Scored with
  L_align     joint space against the child reference -- independent of the observation, the common metric
  bfm_exact   1 - mean_t cos(B(adult-equivalent of the rollout state), B(the ADULT's own motion))
  bfm_mult    the cost the searches used: B(mult x rollout obs) vs B(mult x child reference obs)
both bfm columns are computed for every rollout whatever the actor saw.
Writes outputs/exact_obs/scores.csv and prints the summary.
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
DS = REPO / "datasets/crossenbodiment-child-balanced"
OUT = REPO / "outputs/exact_obs"


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.dataset import CrossEmbodimentDataset
    from model.exact_obs import ExactObs
    from model.simple.train import make_body_ctx, compute_batch_cost
    from model.simple.train_es import set_init_qpos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    ck = sorted((REPO / "outputs/simple_es/child_balanced/b500_global_anchor").glob("update_*.pt"))[-1]
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]; cfg.device = dev
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    rows = [json.loads(l) for l in open(REPO / "outputs/b500_targets/sup_dataset/index.jsonl")]
    D = np.load(REPO / "outputs/b500_targets/sup_dataset/targets.npz")
    te = [i for i, r in enumerate(rows) if r["split"] == "test"]
    ds = CrossEmbodimentDataset(DS)
    at = {(r["reward_name"], r["trial"]): i for i, r in enumerate(ds.rows) if r["morphology_label"] == "child"}
    xml_rel = ds.rows[at[(rows[te[0]]["task"], rows[te[0]]["trial"])]]["target_xml"]
    ctx = make_body_ctx(cfg, DS, "child", xml_rel)
    X = ExactObs(DS / xml_rel)
    env_c1, _ = make_humenv(num_envs=1, task=None, xml=str(DS / xml_rel), state_init="Default")
    env_a1, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / "assets/robots/adult/robot.xml"), state_init="Default")
    mul = ctx["obs_mul"]; nv = ctx["fk"].nv; bs = cfg.batch_size

    smp = {i: ds[at[(rows[i]["task"], rows[i]["trial"])]] for i in te}
    Bg_mult, Bg_exact = {}, {}
    for i in te:
        Bg_mult[i] = bfm_align.reference_embeddings(model, env_c1, smp[i]["qpos_ref"], dev, mul)
        qa = np.load(REPO / "data/origin_motion" / rows[i]["task"] / f"{rows[i]['clip']}.npz")["qpos"]
        Bg_exact[i] = bfm_align.embed(model, bfm_align.obs_from_qpos(env_a1, qa), dev)

    labels = {"z0": D["z0"], "target": D["z_align"]}
    res = {}
    for mode in ("none", "mult", "exact"):
        jobs = [(i, l) for i in te for l in labels]
        for a in range(0, len(jobs), bs):
            t0 = time.time()
            part = jobs[a:a + bs]; pad = part + [part[-1]] * (bs - len(part))
            Z = torch.tensor(np.stack([labels[l][i] for i, l in pad]), dtype=torch.float32, device=dev)
            env = ctx["env"]; env.reset()
            obs = set_init_qpos(env, [smp[i]["qpos_ref"][0] for i, _ in pad], nv)
            qpos = np.stack([smp[i]["qpos_ref"][0] for i, _ in pad]); qvel = np.zeros((bs, nv))
            Q, OM, OE = [], [], []
            for t in range(cfg.steps_per_episode):
                raw = obs["proprio"]
                if mode == "none":
                    o = raw
                elif mode == "mult":
                    o = raw * mul
                else:
                    o = np.stack([X(qpos[k], qvel[k]) for k in range(bs)])
                mu = model._actor(model._normalize(torch.as_tensor(o, dtype=torch.float32, device=dev)), Z,
                                  model.cfg.actor_std).mean
                obs, _, _, _, info = env.step(mu.detach().cpu().numpy())
                qpos, qvel = info["qpos"].copy(), info["qvel"].copy()
                Q.append(qpos); OM.append(obs["proprio"] * mul)
                OE.append(np.stack([X(qpos[k], qvel[k]) for k in range(bs)]))
            Q, OM, OE = np.stack(Q, 1), np.stack(OM, 1).astype(np.float32), np.stack(OE, 1)
            _, la, _ = compute_batch_cost(ctx["fk"], cfg, Q, [smp[i]["qpos_ref"] for i, _ in pad])
            Bm = bfm_align.embed(model, OM.reshape(-1, 358), dev).reshape(bs, Q.shape[1], -1)
            Be = bfm_align.embed(model, OE.reshape(-1, 358), dev).reshape(bs, Q.shape[1], -1)
            for k, (i, l) in enumerate(part):
                res[(i, l, mode)] = dict(L_align=float(la[k]), bfm_exact=bfm_align.bfm_align_loss(Be[k], Bg_exact[i]),
                                         bfm_mult=bfm_align.bfm_align_loss(Bm[k], Bg_mult[i]))
            print(f"  {mode}: rollouts {min(a + bs, len(jobs))}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ctx["env"].close()

    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "category", "label", "obs", "L_align", "bfm_exact", "bfm_mult"])
        for (i, l, m), r in sorted(res.items()):
            w.writerow([rows[i]["clip"], rows[i]["category"], l, m, r["L_align"], r["bfm_exact"], r["bfm_mult"]])
    cats = sorted({rows[i]["category"] for i in te})
    for metric in ("L_align", "bfm_exact", "bfm_mult"):
        print(f"\n=== {metric}, 47 test clips: median (mean) | by category median ===")
        for l in labels:
            for m in ("none", "mult", "exact"):
                v = np.array([res[(i, l, m)][metric] for i in te])
                print(f"  {l:6s} obs={m:5s} {np.median(v):.3f} ({v.mean():.3f}) | " + "  ".join(
                    f"{c} {np.median([res[(i, l, m)][metric] for i in te if rows[i]['category'] == c]):.3f}" for c in cats))
        for l in labels:
            v = np.array([res[(i, l, 'exact')][metric] / res[(i, l, 'mult')][metric] for i in te])
            print(f"  {l}: exact / mult per clip: median {np.median(v):.2f}, exact better on {int((v < 1).sum())}/47")
    print(f"-> {OUT / 'scores.csv'}")


if __name__ == "__main__":
    main()
