#!/usr/bin/env python3
"""interface_ladder.py -- how well does the frozen policy drive the child as the interface is fixed step by step?

Same clips as a fast_search run (default outputs/fast_search_c60/clips.txt), each rolled out once per setting:
  adult        z0 on the ADULT body tracking the adult's own motion (Meta Motivo on the body it was trained on)
  raw          z0, child geometry with the adult's actuators (assets/robots/child), raw child observation
  actuator     z0, torque-calibrated child (robots_torque/child/robot_torque_full.xml), raw observation
  multiplier   z0, calibrated child, observation x a fixed per-feature multiplier (model/obs_scale.py)
  exact        z0, calibrated child, the adult-equivalent observation (reverse retargeting, model/exact_obs.py)
  fast5_k8     the fast search's best z after 5 generations (80 rollouts) in the 8-dim basis, as `exact`
  fast5_full   the same in all 256 dims
  search       exact_train's best z (bfm 1024 + L_align 1024 rollouts), as `exact`
Child settings are scored against the child's retargeted reference, `adult` against the adult motion: L_align
(joint space, the reference's frames), mean joint-angle error (deg) and root-xy distance at the reference's end (m).
Writes outputs/interface_ladder/{<setting>.json, <setting>_qpos.npz}.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
DS = REPO / "datasets/crossenbodiment-child-balanced"
OUT = REPO / "outputs/interface_ladder"
XML = {"adult": "robots/adult/robot.xml", "raw": "robots/child/robot.xml"}
TORQUE = "robots_torque/child/robot_torque_full.xml"
OBS = {"adult": "none", "raw": "none", "actuator": "none", "multiplier": "auto"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["adult", "raw", "actuator", "multiplier", "exact"])
    ap.add_argument("--fast-root", default="outputs/fast_search_c60")
    ap.add_argument("--fast-gens", type=int, default=5)
    ap.add_argument("--out", default="outputs/interface_ladder")
    args = ap.parse_args()
    global OUT
    OUT = REPO / args.out
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model.simple.config import ESConfig
    from model.simple.train import make_body_ctx, compute_batch_cost
    from model.simple.train_es import rollout_z
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    clips = [l.split() for l in open(REPO / args.fast_root / "clips.txt") if l.strip()]
    stems = [f"{t}_{k}" for t, k, _, _ in clips]
    cfg0 = ESConfig()
    model = FBcprModel.from_pretrained(cfg0.metamotivo_repo).to(dev); model.eval()
    z0 = np.stack([np.load(REPO / f"data/origin_z/{t}/{t}_{k}.npy").reshape(-1) for t, k, _, _ in clips])
    child_ref = [np.load(REPO / f"data/child/retargeting_motion/{t}/{t}_{k}.npz")["qpos"].astype(np.float64) for t, k, _, _ in clips]
    adult_ref = [np.load(REPO / f"data/origin_motion/{t}/{t}_{k}.npz")["qpos"].astype(np.float64) for t, k, _, _ in clips]
    B = 64 if len(clips) <= 64 else len(clips)

    def z_for(setting):
        if setting in ("adult", "raw", "actuator", "multiplier", "exact"):
            return z0
        if setting == "search":
            return np.stack([np.load(REPO / f"outputs/exact_train/{s}/align/best_z.npy").reshape(-1) for s in stems])
        arm = setting.split("_")[1]
        zs = []
        for s in stems:
            tr = np.load(REPO / args.fast_root / arm / s / "z_trace.npz")
            zs.append(tr["z_best"][list(tr["gen"]).index(args.fast_gens - 1)])
        return np.stack(zs)

    for setting in args.settings:
        t0 = time.time()
        cfg = ESConfig(); cfg.device = dev; cfg.batch_size = B; cfg.vectorization_mode = "async"
        cfg.obs_scale = OBS.get(setting, "exact")
        ctx = make_body_ctx(cfg, DS, setting, XML.get(setting, TORQUE))
        refs = adult_ref if setting == "adult" else child_ref
        z = z_for(setting); z = 16 * z / np.linalg.norm(z, axis=1, keepdims=True)
        zb = np.concatenate([z, np.repeat(z[-1:], B - len(z), 0)]).astype(np.float32)
        init = [r[0] for r in refs] + [refs[-1][0]] * (B - len(refs))
        q = rollout_z(model, ctx["env"], torch.as_tensor(zb, device=dev), cfg, ctx["obs_mul"], init_qpos=init,
                      nv=ctx["fk"].nv, exact=ctx.get("exact"))[: len(clips)]
        _, la, _ = compute_batch_cost(ctx["fk"], cfg, q, refs)
        rows = {}
        for i, (s, r) in enumerate(zip(stems, refs)):
            n = min(len(r), q.shape[1])
            jd = float(np.degrees(np.abs(np.angle(np.exp(1j * (q[i, :n, 7:] - r[:n, 7:]))))).mean())
            rows[s] = dict(category=clips[i][2], L_align=float(la[i]), joint_deg=jd,
                           end_xy_m=float(np.linalg.norm(q[i, n - 1, :2] - r[n - 1, :2])))
        json.dump(rows, open(OUT / f"{setting}.json", "w"), indent=1)
        np.savez_compressed(OUT / f"{setting}_qpos.npz", **{s: q[i].astype(np.float32) for i, s in enumerate(stems)})
        ctx["env"].close()
        print(f"{setting:11s} L_align median {np.median(la[:len(clips)]):.3f} mean {np.mean(la[:len(clips)]):.3f} | joint err "
              f"{np.median([r['joint_deg'] for r in rows.values()]):.1f} deg | end xy {np.median([r['end_xy_m'] for r in rows.values()]):.2f} m"
              f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
