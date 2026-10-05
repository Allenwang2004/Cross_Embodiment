#!/usr/bin/env python3
"""stat_backward_vs_z0.py -- z0 vs the per-frame backward z, on the child, exact observations, all 500 clips.

  z0         the clip's single adult latent
  backward   Metamotivo tracking_inference on the child reference translated to the adult (model/exact_obs.py)
             -- identical to the adult's own backward z (cos 1.000); step t uses the z of frame t + 1
Both rolled out on the child (torque-calibrated XML, reference init, 300 steps) with the policy and B seeing the
adult-equivalent observation (train_es obs_scale "exact"). Scored with train_es.score_rollouts: bfm against the
child reference through the same translation, L_align in joint space, heading / root-xy terms.
Writes outputs/backward_vs_z0/scores.csv and prints the summary.
"""
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
DS = REPO / "datasets/crossenbodiment-child-balanced"
OUT = REPO / "outputs/backward_vs_z0"


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.dataset import CrossEmbodimentDataset
    from model.simple.train import make_body_ctx
    from model.simple import train_es as T_es
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    ck = sorted((REPO / "outputs/simple_es/child_balanced/b500_global_anchor").glob("update_*.pt"))[-1]
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]; cfg.device = dev; cfg.obs_scale = "exact"
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    cat = dict(l.split() for l in open(DS / "splits/balanced500_categories.txt") if l.strip())
    clips = [(t, int(k)) for t, k in (l.split() for l in open(DS / "splits/balanced500_clips.txt") if l.strip())]
    ds = CrossEmbodimentDataset(DS)
    at = {(r["reward_name"], r["trial"]): i for i, r in enumerate(ds.rows) if r["morphology_label"] == "child"}
    xml_rel = ds.rows[at[clips[0]]]["target_xml"]
    ctx = make_body_ctx(cfg, DS, "child", xml_rel)
    ctx["env1"], _ = make_humenv(num_envs=1, task=None, xml=str(DS / xml_rel), state_init="Default"); ctx["Bg"] = {}
    X, fk = ctx["exact"], ctx["fk"]

    smp, Z = {}, {}
    t0 = time.time()
    for c in clips:
        s = ds[at[c]]; smp[c] = s; ref = s["qpos_ref"]
        v, ob = np.zeros(fk.nv), []
        for t in range(len(ref)):
            if t:
                mujoco.mj_differentiatePos(fk, v, 1 / 30, ref[t - 1], ref[t])
            ob.append(X(ref[t], v))
        with torch.no_grad():
            Z[(c, "backward")] = model.tracking_inference(
                next_obs=torch.as_tensor(np.array(ob[1:], dtype=np.float32), device=dev)).cpu().numpy()
        Z[(c, "z0")] = np.asarray(s["z0"], dtype=np.float32).reshape(1, -1)
    print(f"backward z for {len(clips)} clips ({(time.time() - t0) / 60:.1f} min)", flush=True)

    jobs = [(c, l) for c in clips for l in ("z0", "backward")]
    Tm = max(z.shape[0] for z in Z.values()); bs = cfg.batch_size; res = {}
    for a in range(0, len(jobs), bs):
        t0 = time.time()
        part = jobs[a:a + bs]; pad = part + [part[-1]] * (bs - len(part))
        zb = np.zeros((bs, Tm, 256), dtype=np.float32)
        for k, j in enumerate(pad):
            z = Z[j]; zb[k, :len(z)] = z; zb[k, len(z):] = z[-1]
        terms = {}
        cost, la, _ = T_es.score_rollouts(cfg, model, ctx, torch.as_tensor(zb, device=dev),
                                          [smp[c]["qpos_ref"] for c, _ in pad], [T_es.clip_key(smp[c]) for c, _ in pad],
                                          extra=True, terms=terms)
        for k, j in enumerate(part):
            h, p = float(terms["heading"][k]), float(terms["pos"][k])
            res[j] = dict(bfm=float(cost[k]) - cfg.heading_weight * h - cfg.pos_weight * p, L_align=float(la[k]), heading=h, pos=p)
        print(f"  rollouts {min(a + bs, len(jobs))}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ctx["env"].close(); ctx["env1"].close()

    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "category", "label", "L_align", "bfm", "heading", "pos"])
        for (c, l), r in sorted(res.items()):
            w.writerow([f"{c[0]}_{c[1]}", cat[c[0]], l, r["L_align"], r["bfm"], r["heading"], r["pos"]])
    by = defaultdict(list)
    for c in clips:
        by[cat[c[0]]].append(c); by["ALL"].append(c)
    for metric in ("L_align", "bfm", "pos"):
        print(f"\n=== {metric} on the child, exact observations: z0 vs backward z (median), backward better on, "
              f"median of backward / z0 per clip ===")
        for g in sorted(by, key=lambda g: (g == "ALL", g)):
            a = np.array([res[(c, "z0")][metric] for c in by[g]]); b = np.array([res[(c, "backward")][metric] for c in by[g]])
            print(f"  {g:10s} n={len(a):3d}  z0 {np.median(a):.3f}  backward {np.median(b):.3f}  | backward better {int((b < a).sum()):3d}/{len(a)}"
                  f"  | ratio {np.median(b / a):.2f}")
    print(f"-> {OUT / 'scores.csv'}")


if __name__ == "__main__":
    main()
