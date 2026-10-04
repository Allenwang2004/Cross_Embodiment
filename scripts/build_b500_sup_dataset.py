#!/usr/bin/env python3
"""build_b500_sup_dataset.py -- supervised (z0 -> z*) dataset from the b500_targets searches.

Source: outputs/b500_targets/<clip>/{bfm,global,align} (child body, each clip tracking its own
child reference, anchored to its own z0 with 1.0 * (|z - z0| / 16)^2; see the scheduler there).

A clip is KEPT only when both stage-2 targets beat z0 on their tracking term (penalty excluded):
  z_align :  L_align(z_align) < L_align(z0)      (align stage's own z0 score)
  z_global:  bfm(z_global)    < bfm(z0)          (global stage's own z0 score)
Split: the held-out 10% of model/simple/train_es.py --heldout-clip-frac 0.1 with
balanced500_categories.txt (stratified, eval_seed 12345) -- the same 50 clips the b500 ES runs
hold out; recomputed here and checked against b500_global_anchor's eval_history.json.

Writes <out>/targets.npz (arrays, row i = index.jsonl line i), <out>/index.jsonl, <out>/dropped.jsonl.
"""
import argparse, collections, json, random, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
DS = REPO / "datasets/crossenbodiment-child-balanced"
SRC = REPO / "outputs/b500_targets"


def heldout(clips, cat, frac=0.1, seed=12345):
    """train_es's stratified held-out sample, same RNG calls in the same order."""
    n_held = max(1, round(frac * len(clips)))
    rng = random.Random(seed)
    groups = {}
    for c in clips:
        groups.setdefault(cat[c[0]], []).append(c)
    base, extra = divmod(n_held, len(groups))
    out = []
    for i, g in enumerate(sorted(groups, key=lambda g: -len(groups[g]))):
        out += rng.sample(groups[g], min(base + (1 if i < extra else 0), len(groups[g])))
    return set(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SRC / "sup_dataset"))
    a = ap.parse_args()
    from model.dataset import load_beta
    cat = dict(l.split() for l in open(DS / "splits/balanced500_categories.txt") if l.strip())
    want = {(t, int(k)) for t, k in (l.split() for l in open(DS / "splits/balanced500_clips.txt") if l.strip())}
    man = {(r["reward_name"], r["trial"]): r for r in map(json.loads, open(DS / "manifest.jsonl"))
           if r["morphology_label"] == "child"}
    clips = sorted(c for c in man if c in want)
    assert len(clips) == len(want) == 500, (len(clips), len(want))
    held = heldout(clips, cat)
    ev = json.load(open(REPO / "outputs/simple_es/child_balanced/b500_global_anchor/eval_history.json"))[0]
    seen = {c["clip"] for c in ev["child:unseen"]["per_clip"]}
    assert {f"{t}_{k}" for t, k in held} == seen, "split differs from the b500 ES run's held-out clips"

    rows, arrs, dropped = [], collections.defaultdict(list), []
    for t, k in clips:
        stem = f"{t}_{k}"; d = SRC / stem
        S = {v: json.loads((d / v / "summary.json").read_text()) for v in ("bfm", "global", "align")
             if (d / v / "summary.json").exists()}
        if len(S) < 3:
            raise SystemExit(f"{stem}: search not finished ({sorted(S)}) -- wait for b500_targets")
        ga, al = S["global"], S["align"]
        ok_g = ga["best"]["bfm"] < ga["origin_z"]["bfm"]
        ok_a = al["best"]["align"] < al["origin_z"]["align"]
        rec = dict(clip=stem, task=t, trial=k, category=cat[t], split="test" if (t, k) in held else "train",
                   global_bfm_ratio=ga["best"]["bfm"] / ga["origin_z"]["bfm"],
                   align_ratio=al["best"]["align"] / al["origin_z"]["align"])
        if not (ok_g and ok_a):
            dropped.append(dict(rec, reason=" & ".join(([] if ok_g else ["global bfm >= z0"])
                                                       + ([] if ok_a else ["L_align >= z0"]))))
            continue
        r = man[(t, k)]
        z0 = np.load(DS / r["origin_z"]).reshape(-1).astype(np.float32)
        z = {v: np.load(d / v / "best_z.npy").reshape(-1).astype(np.float32) for v in ("bfm", "global", "align")}
        arrs["z0"].append(z0); arrs["z_align"].append(z["align"]); arrs["z_global"].append(z["global"])
        arrs["z_stage1"].append(z["bfm"]); arrs["beta"].append(load_beta(DS / r["morphology"]).astype(np.float32))
        rec.update(
            dist_align=float(np.linalg.norm(z["align"] - z0)), dist_global=float(np.linalg.norm(z["global"] - z0)),
            align_z0=al["origin_z"]["align"], align_best=al["best"]["align"],
            align_cost_z0=al["origin_z"]["cost"], align_cost_best=al["best"]["cost"],
            global_bfm_z0=ga["origin_z"]["bfm"], global_bfm_best=ga["best"]["bfm"],
            global_head_best=ga["best"].get("head"), global_pos_best=ga["best"].get("pos"),
            global_cost_z0=ga["origin_z"]["cost"], global_cost_best=ga["best"]["cost"],
            stage1_bfm_z0=S["bfm"]["origin_z"]["bfm"], stage1_bfm_best=S["bfm"]["best"]["bfm"],
            ref_motion=str(Path("datasets/crossenbodiment-child-balanced") / r["retargeted_motion"]),
            origin_z=str(Path("datasets/crossenbodiment-child-balanced") / r["origin_z"]),
            search_dir=str(d.relative_to(REPO)), manifest_id=r["id"])
        rows.append(rec)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "targets.npz", **{k: np.stack(v) for k, v in arrs.items()},
                        clip=np.array([r["clip"] for r in rows]), category=np.array([r["category"] for r in rows]),
                        split=np.array([r["split"] for r in rows]))
    with open(out / "index.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / "dropped.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in dropped)
    sp = collections.Counter(r["split"] for r in rows)
    print(f"kept {len(rows)} / 500 clips (train {sp['train']}, test {sp['test']}), dropped {len(dropped)}:")
    for r in dropped:
        print(f"  {r['clip']:30s} {r['split']:5s} global bfm {r['global_bfm_ratio']:.3f}x z0, L_align {r['align_ratio']:.3f}x z0")
    for c in sorted(set(cat.values())):
        n = {s: sum(1 for r in rows if r["category"] == c and r["split"] == s) for s in ("train", "test")}
        print(f"  {c:10s} train {n['train']:3d}  test {n['test']:2d}")
    print(f"-> {out}/targets.npz, index.jsonl, dropped.jsonl")


if __name__ == "__main__":
    main()
