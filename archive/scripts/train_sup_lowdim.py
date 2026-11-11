#!/usr/bin/env python3
"""train_sup_lowdim.py -- supervised z0 -> z* with k = 8 subspace labels, on 100 b500 train clips.

Labels: <label root>/<clip>/align/best_z.npy -- the k = 8 subspace search (bfm -> L_align, no penalty,
basis = top-8 of the b500 TRAIN-clip correction PCA); --label-roots outputs/lowdim_b100 (100 clips) and
outputs/lowdim_train (the other 341). 90% train / 10% val (stratified, seed 0, val picks the epoch). Models, same MLP and loss 1 - cos(pred, target):
  sup8      SubspaceAdapter (correction confined to the same 8 dims) on the k = 8 labels
  supfull   LatentAdapter (full 256) on the SAME 100 clips' b500_targets z_align labels (full space, lambda 1)
First the labels themselves: how alike are the corrections of different clips, k = 8 vs full.
Then rollouts (train_es.score_rollouts, the b500 ES run's cfg) on the 47 test clips and the 10 val clips:
  z0 | sup8 | supfull | es_sub8 (b500_sub8, latest checkpoint) | es_full (b500_global_anchor, final)
  | target8 (val only: the clip's own k = 8 search) | target_full (b500 z_align)
Writes outputs/lowdim_b100/sup/{sup8,supfull}.pt, rollout_scores.csv; prints the summary.
"""
import argparse, collections, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
DS = REPO / "datasets/crossenbodiment-child-balanced"
B100 = REPO / "outputs/lowdim_b100"
BASIS = REPO / "outputs/lowdim_search/basis_corrPCA_train.npy"


def ld(p):
    v = np.load(p).reshape(-1).astype(np.float32)
    return 16 * v / np.linalg.norm(v)


def make(cfg, beta_dim, sub=None):
    from model.networks import LatentAdapter, SubspaceAdapter
    if sub is not None:
        return SubspaceAdapter(beta_dim, 256, sub, hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
                               project=cfg.adapter_project_z)
    return LatentAdapter(beta_dim=beta_dim, z_dim=256, hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
                         alpha_learnable=cfg.adapter_alpha_learnable, project=cfg.adapter_project_z,
                         residual=cfg.adapter_residual, head=cfg.adapter_head,
                         theta_max_deg=cfg.adapter_theta_max_deg, n_heads=1)


def load_ckpt_adapter(path, beta_dim, dev):
    b = torch.load(path, map_location="cpu", weights_only=False); c = b["cfg"]
    sub = (np.load(REPO / c.adapter_subspace)[: c.adapter_subspace_dim or None]
           if getattr(c, "adapter_subspace", "") else None)
    ad = make(c, beta_dim, sub).to(dev); ad.load_state_dict(b["adapter"]); ad.eval()
    return ad, b["update"]


def fit(ad, beta, z0, tgt, tr, va, args, dev):
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(ad.parameters(), lr=args.lr, weight_decay=args.wd)
    B, Z0, T = (torch.tensor(x, device=dev) for x in (beta, z0, tgt))
    err = lambda idx: float((ad(B[idx], Z0[idx]) - T[idx]).norm(dim=-1).mean())
    best, hist = (np.inf, None, -1), []
    for ep in range(args.epochs):
        ad.train(); perm = np.random.default_rng(args.seed + ep).permutation(tr)
        for a in range(0, len(perm), args.batch):
            b = torch.tensor(perm[a:a + args.batch], device=dev)
            loss = (1 - F.cosine_similarity(ad(B[b], Z0[b]), T[b], dim=-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            ad.eval()
            with torch.no_grad():
                e = (err(tr), err(va))
            hist.append((ep, *e))
            if e[1] < best[0]:
                best = (e[1], {k: v.detach().clone() for k, v in ad.state_dict().items()}, ep)
    ad.load_state_dict(best[1]); ad.eval()
    return best[2], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-roots", nargs="+", default=[str(B100)],
                    help="dirs holding <clip>/align/best_z.npy and a clips.txt; outputs/lowdim_b100 (100 clips, "
                         "4096 + 2048 evals) and outputs/lowdim_train (the other 341 train clips, 1024 + 1024)")
    ap.add_argument("--out", default=str(B100 / "sup"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(REPO / "outputs/b500_targets/sup_dataset/index.jsonl")]
    D = np.load(REPO / "outputs/b500_targets/sup_dataset/targets.npz")
    idx = {r["clip"]: i for i, r in enumerate(rows)}
    root = {}
    for r in map(Path, args.label_roots):
        for t, k, _ in (l.split() for l in open(r / "clips.txt") if l.strip()):
            root[f"{t}_{k}"] = r
    clips = sorted(root)
    S = {c: json.loads((root[c] / c / "align/summary.json").read_text()) for c in clips}
    lab = [idx[c] for c in clips]                                    # rows of the 100 labelled clips
    te = [i for i, r in enumerate(rows) if r["split"] == "test"]
    cats = [r["category"] for r in rows]
    z0, beta = D["z0"], D["beta"]
    z8 = np.zeros_like(z0); z8[lab] = np.stack([ld(root[c] / c / "align/best_z.npy") for c in clips])
    U = np.load(BASIS)[:8]

    # the labels: L_align vs z0 (search's own score), and how alike corrections of different clips are
    r8 = np.array([S[c]["best"]["align"] / S[c]["origin_z"]["align"] for c in clips])
    print(f"k = 8 labels, {len(clips)} clips: L_align / z0 median {np.median(r8):.2f}, below z0 {int((r8 < 1).sum())}/{len(clips)} | "
          + "  ".join(f"{c} {np.median([r8[j] for j, i in enumerate(lab) if cats[i] == c]):.2f}" for c in sorted(set(cats))))
    rf = np.array([rows[i]["align_ratio"] for i in lab])
    print(f"full-space labels (b500_targets, lambda 1), same clips: L_align / z0 median {np.median(rf):.2f}")
    for name, Z in (("k = 8", z8), ("full", D["z_align"])):
        d = Z[lab] - z0[lab]; u = d / np.linalg.norm(d, axis=1, keepdims=True); C = u @ u.T
        iu = np.triu_indices(len(lab), 1); same = (np.array(cats)[lab][:, None] == np.array(cats)[lab][None])[iu]
        print(f"  {name:6s} corrections: |d| {np.linalg.norm(d, axis=1).mean():.2f} | cos same category {C[iu][same].mean():+.2f}, "
              f"different {C[iu][~same].mean():+.2f} | |mean d| / mean |d| {np.linalg.norm(d.mean(0)) / np.linalg.norm(d, axis=1).mean():.2f}")

    # fit
    rng = random.Random(args.seed); g = collections.defaultdict(list)
    for i in lab: g[cats[i]].append(i)
    va = sorted(sum((rng.sample(v, max(1, round(.1 * len(v)))) for _, v in sorted(g.items())), []))
    tr = sorted(set(lab) - set(va))
    ck_full = sorted((REPO / "outputs/simple_es/child_balanced/b500_global_anchor").glob("update_*.pt"))[-1]
    cfg = torch.load(ck_full, map_location="cpu", weights_only=False)["cfg"]; cfg.device = dev
    preds = {}
    for name, tgt, sub in (("sup8", z8, U), ("supfull", D["z_align"], None)):
        ad = make(cfg, beta.shape[1], sub).to(dev)
        ep, hist = fit(ad, beta, z0, tgt, tr, va, args, dev)
        torch.save({"adapter": ad.state_dict(), "cfg": cfg, "epoch": ep}, out / f"{name}.pt")
        with torch.no_grad():
            preds[name] = ad(torch.tensor(beta, device=dev), torch.tensor(z0, device=dev)).cpu().numpy()
        e = lambda X, ix: float(np.linalg.norm(X[ix] - tgt[ix], axis=1).mean())
        print(f"[{name}] best epoch {ep}: |pred - target| train {e(preds[name], tr):.2f} (z0 {e(z0, tr):.2f}), "
              f"val {e(preds[name], va):.2f} (z0 {e(z0, va):.2f})")
        print("   curve (epoch: train/val):", "  ".join(f"{h[0]}: {h[1]:.2f}/{h[2]:.2f}" for h in hist[::15]))

    ck8 = sorted((REPO / "outputs/simple_es/child_balanced/b500_sub8").glob("update_*.pt"))
    with torch.no_grad():
        Bt, Zt = torch.tensor(beta, device=dev), torch.tensor(z0, device=dev)
        ad, u_full = load_ckpt_adapter(ck_full, beta.shape[1], dev); preds["es_full"] = ad(Bt, Zt).cpu().numpy()
        if ck8:
            ad, u8 = load_ckpt_adapter(ck8[-1], beta.shape[1], dev); preds["es_sub8"] = ad(Bt, Zt).cpu().numpy()
    print(f"ES adapters: b500_global_anchor @ {u_full}" + (f", b500_sub8 @ {u8}" if ck8 else ", b500_sub8: no checkpoint yet"))

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.dataset import CrossEmbodimentDataset
    from model.simple.train import make_body_ctx
    from model.simple import train_es as T_es
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    ds = CrossEmbodimentDataset(DS)
    at = {(r["reward_name"], r["trial"]): i for i, r in enumerate(ds.rows) if r["morphology_label"] == "child"}
    xml_rel = ds.rows[at[(rows[te[0]]["task"], rows[te[0]]["trial"])]]["target_xml"]
    ctx = make_body_ctx(cfg, DS, "child", xml_rel)
    ctx["env1"], _ = make_humenv(num_envs=1, task=None, xml=str(DS / xml_rel), state_init="Default"); ctx["Bg"] = {}
    zs = {"z0": z0, "target_full": D["z_align"], "target8": z8, **preds}
    labels = [l for l in ("z0", "sup8", "supfull", "es_sub8", "es_full", "target_full") if l in zs]
    jobs = [(i, l) for i in te + va for l in labels] + [(i, "target8") for i in va]
    res, bs, t0 = {}, cfg.batch_size, time.time()
    for a in range(0, len(jobs), bs):
        part = jobs[a:a + bs]; pad = part + [part[-1]] * (bs - len(part))
        smp = [ds[at[(rows[i]["task"], rows[i]["trial"])]] for i, _ in pad]
        Z = torch.tensor(np.stack([zs[l][i] for i, l in pad]), dtype=torch.float32, device=dev)
        c, la, _ = T_es.score_rollouts(cfg, model, ctx, Z, [s["qpos_ref"] for s in smp], [T_es.clip_key(s) for s in smp])
        for k, j in enumerate(part):
            res[j] = dict(bfm=float(c[k]), L_align=float(la[k]))
        print(f"  rollouts {min(a + bs, len(jobs))}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ctx["env"].close(); ctx["env1"].close()
    with open(out / "rollout_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "category", "set", "label", "bfm", "L_align"])
        for (i, l), r in sorted(res.items()):
            w.writerow([rows[i]["clip"], cats[i], "test" if i in te else "val", l, r["bfm"], r["L_align"]])
    for metric in ("L_align", "bfm"):
        print(f"\n=== {metric} / z0's {metric}: median [clips below z0] ===")
        for sname, ix, labs in ((f"test ({len(te)})", te, labels[1:]), (f"val ({len(va)})", va, labels[1:] + ["target8"])):
            print(f"  {sname:9s} " + " | ".join(
                f"{l} {np.median([res[(i, l)][metric] / res[(i, 'z0')][metric] for i in ix]):.2f} "
                f"[{sum(res[(i, l)][metric] < res[(i, 'z0')][metric] for i in ix)}/{len(ix)}]" for l in labs))
        print("  test by category (median):")
        for l in labels[1:]:
            print(f"    {l:11s} " + "  ".join(
                f"{c} {np.median([res[(i, l)][metric] / res[(i, 'z0')][metric] for i in te if cats[i] == c]):.2f}"
                for c in sorted(set(cats))))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
