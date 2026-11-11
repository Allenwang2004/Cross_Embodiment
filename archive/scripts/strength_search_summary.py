#!/usr/bin/env python3
"""strength_search_summary.py -- does a latent search win back what the weaker adult lost?

Per clip, three rollouts of the same motion, all from humenv's Default reset and scored against the adult's original
recording (data/origin_motion):
  adult    full strength, z0                (outputs/strength/<label>/adult/<stem>.npz, strength_rollout.py)
  z0       strength x scale, z0             (outputs/strength/<label>/search/<stem>/origin_z.npz)
  search   strength x scale, best z found   (outputs/strength/<label>/search/<stem>/best.npz)
Columns: bfm cost, heading and xy terms (from single_z_search's summary.json), end-point distance, mean pelvis height,
fallen frames (as strength_rollout.py), mean joint-angle difference, |z - z0| (Euclidean, |z| = 16).
Writes outputs/strength/<label>/search_summary.json and, with --spec, a render_panels.py spec (3 rows x the --show clips).
usage: uv run python scripts/strength_search_summary.py [--label s08] [--spec]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model.bodies import up_z               # noqa: E402

SHOW = {"move-ego-0-2_4": "walk 2 m/s", "move-ego-90-4_9": "run sideways 4 m/s", "jump-2_8": "jump",
        "crawl-0.4-0-d_6": "crawl"}


def motion_stats(q, ref):
    T = min(len(q), len(ref)); q, ref = q[:T], ref[:T]
    u = np.array([up_z(x) for x in q[:, 3:7]]); ur = np.array([up_z(x) for x in ref[:, 3:7]])
    down = (u < ur - 0.8) | (q[:, 2] < 0.5 * ref[:, 2])
    return dict(end_xy=float(np.linalg.norm(q[-1, :2] - ref[-1, :2])), pelvis=float(q[:, 2].mean()),
                ref_pelvis=float(ref[:, 2].mean()), down=int(down.sum()),
                joint_deg=float(np.degrees(np.abs(q[:, 7:] - ref[:, 7:]).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="s08")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--spec", action="store_true")
    a = ap.parse_args()
    root = REPO / "outputs/strength" / a.label
    clips = (root / "clips.txt").read_text().split()
    res = {}
    for c in clips:
        stem = c.split("/")[1]; d = root / "search" / stem
        if not (d / "summary.json").exists():
            print(f"{stem}: no search yet"); continue
        s = json.loads((d / "summary.json").read_text())
        ref = np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]
        z0 = np.load(REPO / f"data/origin_z/{c}.npy").reshape(-1); zb = np.load(d / "best_z.npy").reshape(-1)
        n = lambda z: 16 * z / np.linalg.norm(z)   # noqa: E731
        r = {"adult": motion_stats(np.load(root / "adult" / f"{stem}.npz")["qpos"], ref),
             "z0": {**motion_stats(np.load(d / "origin_z.npz")["qpos"], ref),
                    **{k: s["origin_z"][k] for k in ("cost", "bfm", "head", "pos")}},
             "search": {**motion_stats(np.load(d / "best.npz")["qpos"], ref),
                        **{k: s["best"][k] for k in ("cost", "bfm", "head", "pos")}, "gen": s["best"]["gen"],
                        "dz": float(np.linalg.norm(n(zb) - n(z0)))}}
        r["adult"]["bfm"] = json.loads((root / "summary.json").read_text())[c]["adult"]["cost"]
        res[c] = r
    (root / "search_summary.json").write_text(json.dumps(res, indent=1))
    hdr = f"{'clip':34s} {'bfm adult/z0/search':>22s} {'end-point m':>18s} {'pelvis m (ref)':>22s} {'fallen':>12s} {'joint deg':>15s} {'|dz|':>5s}"
    print(hdr)
    for c, r in res.items():
        A, Z, S = r["adult"], r["z0"], r["search"]
        print(f"{c.split('/')[1]:34s} {A['bfm']:6.3f} {Z['bfm']:6.3f} {S['bfm']:6.3f}   "
              f"{Z['end_xy']:6.1f} -> {S['end_xy']:6.1f}   {Z['pelvis']:.2f} -> {S['pelvis']:.2f} ({A['ref_pelvis']:.2f})   "
              f"{Z['down']:4d} -> {S['down']:4d}   {Z['joint_deg']:5.1f} -> {S['joint_deg']:5.1f} {S['dz']:5.1f}")
    if res:
        med = lambda k, key: float(np.median([r[k][key] for r in res.values()]))   # noqa: E731
        zb, sb = np.array([r["z0"]["bfm"] for r in res.values()]), np.array([r["search"]["bfm"] for r in res.values()])
        print(f"\n{len(res)} clips. median bfm: adult {med('adult', 'bfm'):.3f} | x{a.scale} z0 {med('z0', 'bfm'):.3f} | "
              f"x{a.scale} search {med('search', 'bfm'):.3f}; search lower on {int((sb < zb).sum())}/{len(res)}; "
              f"gap to the adult closed (median over clips) "
              f"{np.median([(r['z0']['bfm'] - r['search']['bfm']) / max(r['z0']['bfm'] - r['adult']['bfm'], 1e-9) for r in res.values()]):.0%}")
        for k in ("end_xy", "pelvis", "joint_deg"):
            print(f"  median {k:9s} adult {med('adult', k):6.2f} | z0 {med('z0', k):6.2f} | search {med('search', k):6.2f}")
        print(f"  clips with > 10 fallen frames: z0 {sum(r['z0']['down'] > 10 for r in res.values())} | "
              f"search {sum(r['search']['down'] > 10 for r in res.values())}")
    if a.spec:
        xml_w = f"assets/robots/adult_{a.label}_gear/robot.xml"
        stems = {c.split("/")[1]: c for c in res}
        rows = [("adult, full strength", "assets/robots/adult/robot.xml", "adult", lambda st: str(root.relative_to(REPO) / "adult" / f"{st}.npz")),
                (f"strength x {a.scale}, z0", xml_w, "z0", lambda st: str(root.relative_to(REPO) / "search" / st / "origin_z.npz")),
                (f"strength x {a.scale}, searched z", xml_w, "search", lambda st: str(root.relative_to(REPO) / "search" / st / "best.npz"))]
        panels = []
        for title, xml, key, path in rows:
            for st, name in SHOW.items():
                r = res[stems[st]][key]
                sub = f"bfm cost {r['bfm']:.3f}, end-point off {r['end_xy']:.1f} m, pelvis {r['pelvis']:.2f} m"
                if key == "search":
                    sub += f", |z - z0| {r['dz']:.1f}"
                panels.append(dict(title=f"{name}: {title}", sub=sub, xml=xml, qpos=path(st)))
        spec = dict(out=str(root.relative_to(REPO) / f"strength{a.label[1:]}_search.mp4"), cols=len(SHOW), size=440,
                    track=dict(distance=3.4, elevation=-12, azimuth=140, lookat_z=0.8), panels=panels)
        (root / "spec_search.json").write_text(json.dumps(spec, indent=1))
        print(f"-> {root / 'spec_search.json'}")


if __name__ == "__main__":
    main()
