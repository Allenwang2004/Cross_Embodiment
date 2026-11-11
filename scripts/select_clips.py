"""Pick the clips where LOES helps most, per body (user, 11/09).

Kept: LOES passes PHC success (user, 11/09 09:22), z0 (frozen) fails PHC success or falls (Fall Time > 0), and TA (BFM-Zero's trajectory
adaptation; CEM not run yet) fails PHC success and has a higher MPJPE than LOES. Ranked by z0 MPJPE - LOES MPJPE, best latent at the final
budget (2,048 rollouts), seed 0. This is a selection: the top clips are where LOES did best, not typical clips.
Contact Fall is about the whole clip for every method on crawl motions (the body is on the floor by design), so 'z0 falls' holds on every
crawl: z0_phc_ok marks a kept clip whose z0 passes PHC and is kept only through Fall.

usage: select_clips.py results.json [results.json ...]  (results.json of scripts/g1_small_test.py or scripts/body_52_test.py;
       prints a table per file, writes <dir>/top_clips.json)
"""
import json, sys
from pathlib import Path

for f in sys.argv[1:]:
    f = Path(f); R = json.load(open(f))["per_motion"]["best"]; rows = []
    for st, v in R.items():
        z, e, ta = v["frozen"], v["es"], v["ta"]
        z_bad = z["success"] < 0.5 or z["fall_s"] > 0
        ta_bad = ta["success"] < 0.5 and ta["mpjpe"] > e["mpjpe"]
        rows.append(dict(clip=st, gain=z["mpjpe"] - e["mpjpe"], es_ok=e["success"] >= 0.5, z_bad=z_bad, ta_bad=ta_bad, z0_phc_ok=z["success"] >= 0.5,
                         **{f"{k}_{m}": v[k][m] for k in ("frozen", "es", "ta", "lso", "lsor", "native") for m in ("success", "fall_s", "mpjpe", "mpjae", "drift")}))
    keep = sorted((r for r in rows if r["es_ok"] and r["z_bad"] and r["ta_bad"]), key=lambda r: -r["gain"])
    print(f"\n{f}: {len(rows)} motions | LOES passes: {sum(r['es_ok'] for r in rows)} | z0 fails or falls: {sum(r['z_bad'] for r in rows)} | "
          f"TA fails and worse than LOES: {sum(r['ta_bad'] for r in rows)} | all three: {len(keep)}")
    print(f"{'#':>2s} {'clip':30s}{'gain cm':>8s} | {'z0 MPJPE':>8s} {'ok':>3s} {'fall':>5s} | {'LOES':>6s} {'ok':>3s} {'fall':>5s} | {'TA':>6s} {'ok':>3s} | "
          f"{'LSO':>6s} {'ok':>3s} {'LSOR':>6s} {'ok':>3s}")
    for i, r in enumerate(keep):
        print(f"{i + 1:2d} {r['clip']:30s}{r['gain']:8.1f} | {r['frozen_mpjpe']:8.1f} {r['frozen_success']:3.0f} {r['frozen_fall_s']:5.2f} | {r['es_mpjpe']:6.1f} "
              f"{r['es_success']:3.0f} {r['es_fall_s']:5.2f} | {r['ta_mpjpe']:6.1f} {r['ta_success']:3.0f} | {r['lso_mpjpe']:6.1f} {r['lso_success']:3.0f} "
              f"{r['lsor_mpjpe']:6.1f} {r['lsor_success']:3.0f}")
    (f.parent / "top_clips.json").write_text(json.dumps(dict(rule=__doc__.split("\n\n")[1], n=len(rows), n_kept=len(keep), kept=keep, top5=keep[:5]), indent=1))
