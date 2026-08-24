"""Apply a fitted cross-body latent map to a whole tree of z, mirroring layout.

Takes the W.npz written by `fit_cross_body_z_map.py` and pushes every .npy in
an input tree through it:

    z_out = project_z( unit(z_in) @ W )

Works on any (..., 256) array, so it handles both the per-frame tracking trees
((T, 256)) and the per-clip reward z0 trees ((1, 256)).

DIRECTION. W.npz stores two maps, and picking the wrong one silently produces
well-formed garbage: the output lands back on the sqrt(d) sphere either way, so
shape and norm checks will NOT catch it. Measured on latents this map does not
fit, that mistake reads as cosine +0.21 -> -0.35 against the target body.

So there is no default: name the map with --key.

    --key W_origin_to_child     --key W_child_to_origin

fit_cross_body_z_map.py writes those directional names alongside the positional
W_full / W_full_reverse, and they are the ones to use -- W_full only means
something once you know which tree was src, which is exactly the fact that goes
missing. A W.npz fitted before those names existed still works: --key takes any
matrix in the file, and running with no --key prints what that file holds.

EXTRAPOLATION WARNING. The shipped map was fitted on TRACKING embeddings
(`tracking_inference`, per frame). Applying it to a reward-inferred z0 tree
(`data/origin_z`) is out of distribution: per fit_z_map.py, a reward-like z is
a limit cycle while a tracking-like z is closer to a fixed point, and the two
do not occupy the same region. The map will still return a unit-sphere vector.
Use --check-against to get a number for whether it actually moved the latents
toward the target body rather than trusting that it did.

Usage (from project root):
    uv run scripts/apply_z_map.py --in-dir data/origin_z --out-dir data/origin_z_w \
        --key W_origin_to_child
    uv run scripts/apply_z_map.py --in-dir data/origin_z --out-dir data/origin_z_w \
        --key W_origin_to_child \
        --check-against data/child/infer_retargeting_z
"""

import argparse
from pathlib import Path

import numpy as np

# Shares the sphere geometry with the fitting script so the two cannot drift.
from fit_cross_body_z_map import apply_map, unit, row_cosine

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", default="outputs/fit_cross_body_z_map/W.npz")
    ap.add_argument("--task-map", default=None,
                    help="W_by_task.npz; uses each clip's per-task correction, "
                         "keyed on the FIRST path component under --in-dir. Clips "
                         "whose task is not in the bank fall back to the global map")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--key", default=None,
                    help="which matrix in the map file to apply, e.g. "
                         "W_origin_to_child. Run without it to see what the "
                         "file holds. No default -- see the DIRECTION note")
    ap.add_argument("--check-against", default=None,
                    help="tree of the TARGET body's z; reports whether the map "
                         "moved each clip closer to it")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    absolute = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    map_path, in_dir, out_dir = map(absolute, (args.map, args.in_dir, args.out_dir))

    z = np.load(map_path, allow_pickle=False)
    src, dst = str(z["src"]), str(z["dst"])
    kf = str(z["key_forward"]) if "key_forward" in z.files else "W_full"
    kr = str(z["key_reverse"]) if "key_reverse" in z.files else "W_full_reverse"
    # W_heldout is the same direction as W_full, fitted without fold 0's tasks.
    fwd, rev_ = {"W_full", "W_heldout", kf}, {"W_full_reverse", kr}

    # No default: a wrong direction is not catchable downstream (see docstring).
    if args.key is None or args.key not in z.files:
        avail = [k for k in z.files if k.startswith("W_")]
        lead = ("pick a map with --key" if args.key is None
                else f"--key {args.key!r} is not in this file")
        raise SystemExit(
            f"{lead}. {map_path.name} was fitted {src} -> {dst} and holds "
            f"{', '.join(avail)}\n"
            f"    --key {kf}   ({src} -> {dst})\n"
            f"    --key {kr}   ({dst} -> {src})\n"
            "There is no safe default -- the output is on the sqrt(d) sphere "
            "either way, so a wrong direction passes every shape and norm check.")

    key = args.key
    W = z[key]
    if key in rev_:
        direction, maps_from, maps_to = "reverse", dst, src
    elif key in fwd:
        direction, maps_from, maps_to = "forward", src, dst
    else:
        raise SystemExit(f"--key {key!r} is in the file but is not one of its "
                         f"maps ({kf}, {kr}, W_full, W_full_reverse, W_heldout)")
    print(f"map   {map_path}  [{key}]  lam={float(z['lam'])}")
    print(f"      fitted {src} -> {dst}; this run maps {maps_from} -> {maps_to}")
    print(f"in    {in_dir}\nout   {out_dir}")

    bank = None
    if args.task_map:
        tm = np.load(absolute(args.task_map), allow_pickle=False)
        sfx = "_reverse" if direction == "reverse" else ""
        if f"delta{sfx}" not in tm.files:
            raise SystemExit(f"{args.task_map} has no 'delta{sfx}'")
        gkey = f"W_global{sfx}"
        bank = {"idx": {str(t): i for i, t in enumerate(tm["tasks"])},
                "delta": tm[f"delta{sfx}"],
                "W": tm[gkey] if gkey in tm.files else W}
        W = bank["W"]  # global fallback must match the bank's base map
        print(f"bank  {absolute(args.task_map)}  "
              f"{len(bank['idx'])} per-task corrections, lam={float(tm[f'lam{sfx}']):g}")

    npys = sorted(in_dir.rglob("*.npy"))
    if not npys:
        raise SystemExit(f"no .npy under {in_dir}")

    rows, n_written, n_task_specific = [], 0, 0
    for p in npys:
        a = np.load(p)
        if a.shape[-1] != W.shape[0]:
            raise SystemExit(f"{p}: last dim {a.shape[-1]} != map dim {W.shape[0]}")
        rel = p.relative_to(in_dir)
        Wc = W
        if bank is not None:
            ti = bank["idx"].get(rel.parts[0] if len(rel.parts) > 1 else "")
            if ti is not None:
                Wc = bank["W"] + bank["delta"][ti]
                n_task_specific += 1
        out = apply_map(a, Wc).astype(a.dtype)
        if not args.dry_run:
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / rel, out)
        n_written += 1

        if args.check_against:
            tgt = absolute(args.check_against) / rel
            if tgt.exists():
                # Compare against the target body's trajectory mean: does the
                # map pull the latent toward that body's region of the sphere?
                t = unit(np.load(tgt).astype(np.float64).reshape(-1, W.shape[0]).mean(0))
                rows.append((float(row_cosine(unit(a.reshape(-1)), t)),
                             float(row_cosine(unit(out.reshape(-1)), t))))

    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {n_written} files -> {out_dir}")
    if bank is not None:
        print(f"  {n_task_specific}/{n_written} used a per-task correction, "
              f"{n_written - n_task_specific} fell back to the global map")
    print(f"  shape/dtype preserved (e.g. {np.load(npys[0]).shape} "
          f"{np.load(npys[0]).dtype}); every output back on the sqrt(d) sphere")

    if rows:
        b, a_ = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        print(f"\ncheck against {args.check_against}  ({len(rows)} clips matched)")
        print(f"  cosine to target-body mean z   before {b.mean():.4f}   after {a_.mean():.4f}")
        print(f"  improved on {int((a_ > b).sum())}/{len(rows)} clips")
        if a_.mean() <= b.mean():
            print("  WARNING: the map did NOT move these latents toward the target "
                  "body. Check --direction, and see the extrapolation note in the "
                  "module docstring.")


if __name__ == "__main__":
    main()
