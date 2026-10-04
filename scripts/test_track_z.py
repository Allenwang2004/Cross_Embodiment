#!/usr/bin/env python3
"""test_track_z.py -- the child's own backward z, fed per frame (FB's tracking mode), on the child.

For each clip, from the child's retargeted reference g_0..g_{T-1}:
  obs_t  = proprio of g_t written into the child env (bfm_align.obs_from_qpos), raw or scaled to
           adult units (model/obs_scale.py, the scaling the actor and the bfm cost use)
  z_t    = metamotivo tracking_inference(next_obs = obs_1..obs_{T-1}): project(mean of B over the next
           seq_length frames); step t of the rollout uses z_t (Metamotivo's tracking eval), the last held
and as before (9/27) the mean-pooled version, one z per clip. Compared in the same rollout batches
(train_es.score_rollouts, the b500 ES run's cfg: child torque XML, reference init, 300 steps) with
z0 and the b500 search targets (z_align, z_global).

Clips: the 47 test clips of outputs/b500_targets/sup_dataset + move-ego-0-2_4.
Writes outputs/track_z/scores.csv and prints ratios to z0.
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
DS = REPO / "datasets/crossenbodiment-child-balanced"
OUT = REPO / "outputs/track_z"


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.dataset import CrossEmbodimentDataset
    from model.simple.train import make_body_ctx
    from model.simple import train_es as T_es
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    ck = sorted((REPO / "outputs/simple_es/child_balanced/b500_global_anchor").glob("update_*.pt"))[-1]
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]; cfg.device = dev
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()

    rows = [json.loads(l) for l in open(REPO / "outputs/b500_targets/sup_dataset/index.jsonl")]
    D = np.load(REPO / "outputs/b500_targets/sup_dataset/targets.npz")
    sel = [i for i, r in enumerate(rows) if r["split"] == "test" or r["clip"] == "move-ego-0-2_4"]
    ds = CrossEmbodimentDataset(DS)
    at = {(r["reward_name"], r["trial"]): i for i, r in enumerate(ds.rows) if r["morphology_label"] == "child"}
    xml_rel = ds.rows[at[(rows[sel[0]]["task"], rows[sel[0]]["trial"])]]["target_xml"]
    ctx = make_body_ctx(cfg, DS, "child", xml_rel)
    ctx["env1"], _ = make_humenv(num_envs=1, task=None, xml=str(DS / xml_rel), state_init="Default")
    ctx["Bg"] = {}

    def track(ref, obs_mul):
        obs = bfm_align.obs_from_qpos(ctx["env1"], ref, obs_mul=obs_mul)
        with torch.no_grad():
            return model.tracking_inference(next_obs=torch.as_tensor(obs[1:], device=dev)).cpu().numpy()

    proj = lambda v: 16 * v / np.linalg.norm(v)
    Z, refs, samples = {}, {}, {}
    chk = []
    for i in sel:
        r = rows[i]; s = ds[at[(r["task"], r["trial"])]]; samples[i] = s; ref = s["qpos_ref"]
        zr, zs = track(ref, None), track(ref, ctx["obs_mul"])
        Z[(i, "z0")] = D["z0"][i][None]; Z[(i, "target_align")] = D["z_align"][i][None]
        Z[(i, "target_global")] = D["z_global"][i][None]
        Z[(i, "track_raw")] = zr; Z[(i, "track_scaled")] = zs
        Z[(i, "mean_raw")] = proj(zr.mean(0))[None]; Z[(i, "mean_scaled")] = proj(zs.mean(0))[None]
        stored = np.load(REPO / "data/child/infer_retargeting_z" / r["task"] / f"{r['clip']}.npy")
        n = min(len(stored) - 1, len(zr))
        chk.append(np.mean(np.sum(stored[1:n + 1] * zr[:n], 1) / 256))
    print(f"recomputed raw track z vs stored infer_retargeting_z (shifted one frame): mean cos {np.mean(chk):+.3f}")
    labels = ["z0", "target_align", "target_global", "track_raw", "track_scaled", "mean_raw", "mean_scaled"]
    jobs = [(i, l) for i in sel for l in labels]
    Tm = max(z.shape[0] for z in Z.values())
    res = {}; bs = cfg.batch_size; t0 = time.time()
    for a in range(0, len(jobs), bs):
        part = jobs[a:a + bs]; pad = part + [part[-1]] * (bs - len(part))
        zb = np.zeros((bs, Tm, 256), dtype=np.float32)
        for k, j in enumerate(pad):
            z = Z[j]; zb[k, :len(z)] = z; zb[k, len(z):] = z[-1]
        terms = {}
        c, la, _ = T_es.score_rollouts(cfg, model, ctx, torch.as_tensor(zb, device=dev),
                                       [samples[i]["qpos_ref"] for i, _ in pad],
                                       [T_es.clip_key(samples[i]) for i, _ in pad], extra=True, terms=terms)
        for k, j in enumerate(part):
            h, p = float(terms["heading"][k]), float(terms["pos"][k])
            res[j] = dict(bfm=float(c[k]) - cfg.heading_weight * h - cfg.pos_weight * p, heading=h, pos=p,
                          L_align=float(la[k]))
        print(f"  rollouts {min(a + bs, len(jobs))}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ctx["env"].close(); ctx["env1"].close()

    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "category", "label", "bfm", "heading", "pos", "L_align"])
        for (i, l), r in sorted(res.items()):
            w.writerow([rows[i]["clip"], rows[i]["category"], l, r["bfm"], r["heading"], r["pos"], r["L_align"]])
    test = [i for i in sel if rows[i]["split"] == "test"]
    for metric in ("L_align", "bfm"):
        print(f"\n=== {metric} / z0's {metric}, 47 test clips: median [clips below z0] (mean) ===")
        for l in labels[1:]:
            v = np.array([res[(i, l)][metric] / res[(i, "z0")][metric] for i in test])
            print(f"  {l:14s} {np.median(v):5.2f} [{int((v < 1).sum()):2d}/{len(v)}]  (mean {v.mean():.2f})")
        cats = sorted({rows[i]["category"] for i in test})
        for l in ("track_raw", "track_scaled"):
            print(f"  {l} by category: " + "  ".join(
                f"{c} {np.median([res[(i, l)][metric] / res[(i, 'z0')][metric] for i in test if rows[i]['category'] == c]):.2f}"
                for c in cats))
    w = [i for i in sel if rows[i]["clip"] == "move-ego-0-2_4"][0]
    print("\nmove-ego-0-2_4 (L_align / bfm):", "  ".join(f"{l} {res[(w, l)]['L_align']:.3f}/{res[(w, l)]['bfm']:.3f}" for l in labels))
    print(f"-> {OUT / 'scores.csv'}")


if __name__ == "__main__":
    main()
