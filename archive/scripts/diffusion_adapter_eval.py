#!/usr/bin/env python3
"""diffusion_adapter_eval.py -- score a trained cross-body adapter G(beta, z0) (scripts/diffusion_adapter.py,
model/simple/train_es.py) against doing nothing (z0'), with the search's own cost.

Sets (synthetic tasks from outputs/diffusion_adapter/tasks.json; 25 train tasks are a fixed seeded sample):
  E0  train tasks (25) x train bodies (8)    fit: what training saw
  E1  test tasks (25)  x train bodies (8)    new tasks, known bodies
  E2  train tasks (25) x test bodies (12)    known tasks, new bodies
  E3  test tasks (25)  x test bodies (12)    both new
  E4  the 10 REAL raisearms-l-l trials x all 20 bodies (trial 8 is the cloud's centre, the others sit 11-15 away)
Per cell: the adapter's z and the cell's own z0 are scored in one single_z_search.py --eval-z run (exact obs,
reference init, bfm + 1.0 heading + 0.1 xy + 0.3 (|z - z0| / 16)^2 with the cell's z0 as anchor), so both see the
identical cost. Ratio = cost(adapter z) / cost(z0) (< 1: the adapter helped).
  run    --ckpt <update_xxxxx.pt> --tag <name>   compute latents, launch the scoring jobs (GPU --gpus)
  report --tag <name>
Writes outputs/diffusion_adapter/eval/<tag>/...
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from diffusion_adapter import DATA, OUT, TEST_BODIES, TRAIN_BODIES, name   # noqa: E402

EV = OUT / "eval"
REAL_TASK = "raisearms-l-l"


def cells():
    meta = json.load(open(OUT / "tasks.json"))
    tr = [t["name"] for t in meta["tasks"] if t["split"] == "train"]
    te = [t["name"] for t in meta["tasks"] if t["split"] == "test"]
    tr25 = sorted(np.random.default_rng(0).choice(tr, 25, replace=False).tolist())
    S = {"E0": [(b, t, "synth") for t in tr25 for b in TRAIN_BODIES],
         "E1": [(b, t, "synth") for t in te for b in TRAIN_BODIES],
         "E2": [(b, t, "synth") for t in tr25 for b in TEST_BODIES],
         "E3": [(b, t, "synth") for t in te for b in TEST_BODIES],
         "E4": [(b, f"{REAL_TASK}_{k}", "real") for k in range(10) for b in TRAIN_BODIES + TEST_BODIES]}
    return S, {t["name"]: t for t in meta["tasks"]}


def paths(kind, t):
    if kind == "synth":
        return DATA, f"{t}/{t}_0", DATA / "origin_z" / t / f"{t}_0.npy"
    return REPO / "data", f"{REAL_TASK}/{t}", REPO / "data/origin_z" / REAL_TASK / f"{t}.npy"


def run(a):
    import torch
    from model.dataset import load_beta
    from model.networks import LatentAdapter
    from multibody_corr import run_jobs
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"] if isinstance(ck["cfg"], dict) else vars(ck["cfg"])
    ad = LatentAdapter(beta_dim=8, z_dim=256, hidden_dims=c["adapter_hidden_dims"], alpha=c["adapter_alpha"],
                       alpha_learnable=c.get("adapter_alpha_learnable", False), project=c["adapter_project_z"],
                       residual=c.get("adapter_residual", True), head=c.get("adapter_head") or "residual",
                       theta_max_deg=c.get("adapter_theta_max_deg", 60.0), n_heads=c.get("adapter_heads", 1))
    ad.load_state_dict(ck["adapter"]); ad.eval()
    S, _ = cells()
    jobs = []
    for s in a.sets.split(","):
        for b, t, kind in S[s]:
            root, clip, z0p = paths(kind, t)
            z0 = np.load(z0p).reshape(-1).astype(np.float32)
            beta = load_beta(REPO / f"assets/robots/{b}/parameter.json")
            with torch.no_grad():
                za = ad(torch.tensor(beta)[None], torch.tensor(z0)[None])[0].numpy()
            out = EV / a.tag / s / b / t
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "z.npy", np.stack([16 * z0 / np.linalg.norm(z0), za]))
            n = len(np.load(root / b / "retargeting_motion" / f"{clip}.npz")["qpos"])
            cmd = ["uv", "run", "python", "scripts/single_z_search.py", "--clip", clip, "--body", b, "--data-root", str(root),
                   "--init", "reference", "--steps", str(n), "--pairs", "8", "--obs-scale", "exact", "--objective", "bfm",
                   "--heading-weight", "1.0", "--pos-weight", "0.1", "--anchor", str(z0p), "--anchor-weight", "0.3",
                   "--eval-z", str(out / "z.npy"), "--out", str(out)]
            jobs.append((out, cmd))
    print(f"{len(jobs)} cells ({a.sets}) for {a.ckpt}")
    (EV / a.tag).mkdir(parents=True, exist_ok=True)
    (EV / a.tag / "ckpt.txt").write_text(str(a.ckpt) + "\n")
    run_jobs(jobs, a, "eval.csv")


def report(a):
    S, meta = cells()
    out = {}
    for s, cl in S.items():
        R = []
        for b, t, kind in cl:
            f = EV / a.tag / s / b / t / "eval.csv"
            if not f.exists():
                continue
            c = [float(r["cost"]) for r in csv.DictReader(open(f))]
            R.append(dict(body=b, task=t, z0=c[0], ad=c[1], r=c[1] / c[0],
                          ring=meta[t]["ring"] if kind == "synth" else (0 if t.endswith("_8") else 13)))
        if not R:
            continue
        r = np.array([x["r"] for x in R])
        out[s] = dict(n=len(R), median_ratio=float(np.median(r)), helped=float(np.mean(r < 1)),
                      mean_z0=float(np.mean([x["z0"] for x in R])), mean_adapter=float(np.mean([x["ad"] for x in R])))
        line = f"{s} n {len(R):3d}: cost(adapter) / cost(z0) median {np.median(r):.3f}, adapter better in {np.mean(r < 1):.0%}"
        if s != "E4":
            rings = sorted({x["ring"] for x in R})
            line += " | by distance from the centre: " + " ".join(
                f"{g}:{np.median([x['r'] for x in R if x['ring'] == g]):.2f}" for g in rings)
        else:
            line += f" | trial 8 (centre) {np.median([x['r'] for x in R if x['ring'] == 0]):.3f}, other trials {np.median([x['r'] for x in R if x['ring'] == 13]):.3f}"
        print(line)
        bodies = sorted({x["body"] for x in R})
        print("   by body: " + " ".join(f"{b}:{np.median([x['r'] for x in R if x['body'] == b]):.2f}" for b in bodies))
    (EV / a.tag / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="E0,E1,E2,E3,E4")
    ap.add_argument("--gpus", default="2"); ap.add_argument("--max-par", type=int, default=8)
    ap.add_argument("--min-free-mib", type=int, default=2500); ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--launch-pause", type=float, default=10)
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)
