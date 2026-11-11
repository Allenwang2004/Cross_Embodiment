#!/usr/bin/env python3
"""train_sup_targets.py -- supervised z0 -> z* on the b500_targets dataset, then scored by rollout.

Data: outputs/b500_targets/sup_dataset (build_b500_sup_dataset.py): 488 child clips, the b500 ES
run's held-out split (441 train / 47 test). Two targets, one model each:
  align  : z_align  (stage-2 L_align search)       -> judged on L_align
  global : z_global (stage-2 bfm + heading + pos)  -> judged on bfm (heading / pos reported)

Model: the ES adapter's LatentAdapter (z0 + 1.0 * MLP([beta, z0]), projected to radius 16,
hidden 256-512-512-256), loss 1 - cos(pred, target). Epochs are picked on a validation 10% of
the TRAIN clips (stratified by category, seed 0); the test clips are never looked at in training.

Rollout scoring, all in one batch configuration so the numbers compare (train_es.score_rollouts,
the ES run's own cfg: child torque XML, reference init, 300 steps, bfm against the child reference,
L_align in joint space, heading / root-xy terms):
  z0 | target (the search's z) | sup (this model at the val-best epoch) | full (the same model
  after all epochs, i.e. fitted to the train targets) | es (the b500_global_anchor ES adapter,
  latest checkpoint)
on the 47 test clips and 47 train clips (stratified sample, seed 0).
Writes <out>/{align,global}.pt, <out>/rollout_scores.csv, prints the summary.
"""
import argparse, collections, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
DATA = REPO / "outputs/b500_targets/sup_dataset"
DS = REPO / "datasets/crossenbodiment-child-balanced"
ES_DIR = REPO / "outputs/simple_es/child_balanced/b500_global_anchor"


def strat_sample(idx, cats, frac, seed):
    rng = random.Random(seed); g = collections.defaultdict(list)
    for i in idx:
        g[cats[i]].append(i)
    out = []
    for c in sorted(g):
        out += rng.sample(g[c], max(1, round(frac * len(g[c]))))
    return sorted(out)


def make_adapter(beta_dim, cfg):
    from model.networks import LatentAdapter
    return LatentAdapter(beta_dim=beta_dim, z_dim=256, hidden_dims=cfg.adapter_hidden_dims,
                         alpha=cfg.adapter_alpha, alpha_learnable=cfg.adapter_alpha_learnable,
                         project=cfg.adapter_project_z, residual=cfg.adapter_residual,
                         head=cfg.adapter_head, theta_max_deg=cfg.adapter_theta_max_deg,
                         n_heads=getattr(cfg, "adapter_heads", 1))


def fit(cfg, beta, z0, tgt, tr, va, args, dev):
    torch.manual_seed(args.seed)
    ad = make_adapter(beta.shape[1], cfg).to(dev)
    opt = torch.optim.AdamW(ad.parameters(), lr=args.lr, weight_decay=args.wd)
    B, Z0, T = (torch.tensor(x, device=dev) for x in (beta, z0, tgt))
    err = lambda idx: float((ad(B[idx], Z0[idx]) - T[idx]).norm(dim=-1).mean())
    best = (np.inf, None, -1); hist = []
    for ep in range(args.epochs):
        ad.train(); perm = torch.tensor(np.random.default_rng(args.seed + ep).permutation(tr), device=dev)
        for a in range(0, len(perm), args.batch):
            b = perm[a:a + args.batch]
            loss = (1 - F.cosine_similarity(ad(B[b], Z0[b]), T[b], dim=-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            ad.eval()
            with torch.no_grad():
                e_tr, e_va = err(tr), err(va)
            hist.append((ep, e_tr, e_va))
            if e_va < best[0]:
                best = (e_va, {k: v.detach().clone() for k, v in ad.state_dict().items()}, ep)
    last = make_adapter(beta.shape[1], cfg).to(dev); last.load_state_dict(ad.state_dict()); last.eval()
    ad.load_state_dict(best[1]); ad.eval()
    return ad, last, best[2], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs/b500_targets/sup_train"))
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train-eval", type=int, default=47)
    ap.add_argument("--no-rollout", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    D = np.load(DATA / "targets.npz"); rows = [json.loads(l) for l in open(DATA / "index.jsonl")]
    z0, beta, cats, split = D["z0"], D["beta"], list(D["category"]), list(D["split"])
    tr_all = [i for i, s in enumerate(split) if s == "train"]; te = [i for i, s in enumerate(split) if s == "test"]
    va = strat_sample(tr_all, cats, 0.1, args.seed); tr = sorted(set(tr_all) - set(va))
    ck = sorted(ES_DIR.glob("update_*.pt"))[-1]
    blob = torch.load(ck, map_location="cpu", weights_only=False); cfg = blob["cfg"]; cfg.device = dev
    print(f"{len(tr)} train / {len(va)} val / {len(te)} test clips | ES adapter: {ck.name}")

    preds, mshift = {}, {}
    for name, key in (("align", "z_align"), ("global", "z_global")):
        T = D[key]
        ad, last, ep, hist = fit(cfg, beta, z0, T, tr, va, args, dev)
        torch.save({"adapter": ad.state_dict(), "cfg": cfg, "epoch": ep, "target": key}, out / f"{name}.pt")
        torch.save({"adapter": last.state_dict(), "cfg": cfg, "epoch": args.epochs - 1, "target": key},
                   out / f"{name}_full.pt")
        with torch.no_grad():
            P = ad(torch.tensor(beta, device=dev), torch.tensor(z0, device=dev)).cpu().numpy()
            preds[name + "_full"] = last(torch.tensor(beta, device=dev), torch.tensor(z0, device=dev)).cpu().numpy()
        preds[name] = P
        shift = (T[tr] - z0[tr]).mean(0); M = z0 + shift; M = 16 * M / np.linalg.norm(M, axis=1, keepdims=True)
        mshift[name] = M
        e = lambda X, idx: float(np.linalg.norm(X[idx] - T[idx], axis=1).mean())
        print(f"\n[{name}] best epoch {ep} (val) | mean |pred - target| (Euclidean, radius 16):")
        for lab, idx in (("train", tr), ("val", va), ("test", te)):
            print(f"  {lab:5s}  z0 {e(z0, idx):5.2f}   mean shift {e(M, idx):5.2f}   sup {e(P, idx):5.2f}"
                  f"   sup_full {e(preds[name + '_full'], idx):5.2f}"
                  f"   | |sup - z0| {np.linalg.norm(P[idx] - z0[idx], axis=1).mean():.2f}")
        print("  curve (epoch: train / val):", "  ".join(f"{h[0]}: {h[1]:.2f}/{h[2]:.2f}" for h in hist[::15]))
    if args.no_rollout:
        return

    # the ES adapter's prediction for every clip
    with torch.no_grad():
        es = make_adapter(beta.shape[1], cfg).to(dev); es.load_state_dict(blob["adapter"]); es.eval()
        Pes = es(torch.tensor(beta, device=dev), torch.tensor(z0, device=dev)).cpu().numpy()

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.dataset import CrossEmbodimentDataset
    from model.simple.train import make_body_ctx
    from model.simple import train_es as T_es
    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(dev); model.eval()
    ds = CrossEmbodimentDataset(DS)
    at = {(r["reward_name"], r["trial"]): i for i, r in enumerate(ds.rows) if r["morphology_label"] == "child"}
    ev_tr = strat_sample(tr_all, cats, args.n_train_eval / len(tr_all), args.seed + 1)
    ctx = make_body_ctx(cfg, DS, "child", ds.rows[at[(rows[0]["task"], rows[0]["trial"])]]["target_xml"])
    ctx["env1"], _ = make_humenv(num_envs=1, task=None, xml=str(DS / ds.rows[at[(rows[0]["task"], rows[0]["trial"])]]["target_xml"]),
                                 state_init="Default")
    ctx["Bg"] = {}
    zs = {"z0": z0, "target_align": D["z_align"], "target_global": D["z_global"],
          "sup_align": preds["align"], "sup_global": preds["global"],
          "full_align": preds["align_full"], "full_global": preds["global_full"], "es": Pes}
    jobs = [(i, lab) for i in te + ev_tr for lab in zs]
    res = {}; t0 = time.time(); bs = cfg.batch_size
    for a in range(0, len(jobs), bs):
        part = jobs[a:a + bs]; pad = part + [part[-1]] * (bs - len(part))
        samples = [ds[at[(rows[i]["task"], rows[i]["trial"])]] for i, _ in pad]
        Z = torch.tensor(np.stack([zs[lab][i] for i, lab in pad]), dtype=torch.float32, device=dev)
        terms = {}
        c, la, _ = T_es.score_rollouts(cfg, model, ctx, Z, [s["qpos_ref"] for s in samples],
                                       [T_es.clip_key(s) for s in samples], extra=True, terms=terms)
        for k, (i, lab) in enumerate(part):
            h, p = float(terms["heading"][k]), float(terms["pos"][k])
            res[(i, lab)] = dict(bfm=float(c[k]) - cfg.heading_weight * h - cfg.pos_weight * p,
                                 heading=h, pos=p, L_align=float(la[k]))
        print(f"  rollouts {min(a + bs, len(jobs))}/{len(jobs)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    ctx["env"].close(); ctx["env1"].close()

    with open(out / "rollout_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "category", "split", "label", "bfm", "heading", "pos", "L_align"])
        for (i, lab), r in sorted(res.items()):
            w.writerow([rows[i]["clip"], cats[i], "test" if i in te else "train", lab, r["bfm"], r["heading"], r["pos"], r["L_align"]])
    for metric, labs in (("L_align", ("target_align", "sup_align", "full_align", "es")),
                         ("bfm", ("target_global", "sup_global", "full_global", "es"))):
        print(f"\n=== {metric} / z0's {metric} (same rollout batch), median [clips below z0] ===")
        for sname, idx in (("test", te), ("train", ev_tr)):
            cells = []
            for lab in labs:
                r = np.array([res[(i, lab)][metric] / res[(i, "z0")][metric] for i in idx])
                cells.append(f"{lab} {np.median(r):.2f} [{int((r < 1).sum())}/{len(r)}]")
            print(f"  {sname:5s} " + " | ".join(cells))
        lab = "sup_align" if metric == "L_align" else "sup_global"
        print(f"  test by category ({lab}, median ratio [below z0]):", "  ".join(
            f"{c} {np.median([res[(i, lab)][metric] / res[(i, 'z0')][metric] for i in te if cats[i] == c]):.2f}"
            f" [{sum(res[(i, lab)][metric] < res[(i, 'z0')][metric] for i in te if cats[i] == c)}/{sum(cats[i] == c for i in te)}]"
            for c in sorted(set(cats))))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
