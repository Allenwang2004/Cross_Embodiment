#!/usr/bin/env python3
"""accad_locomotion.py -- the ACCAD clips of outputs/amass_render/ACCAD with locomotion (user, 11/11: as outputs/amass_render/selec_clips,
the clips where the body moves on its feet; feet that do not move: out). Per clip (kinematic adult replay, as the renders):
#   root_path / root_net   pelvis horizontal path length / start-to-end distance (m)
#   steps    foot placements: a foot (ankle or toe, the lower one) is planted when within 5 cm of its lowest height in the clip and
#            slower than 0.3 m/s horizontally; a new placement = planted again more than 0.25 m (horizontally) from where it was last planted
Classes:
  locomotion   at least 3 foot placements, and the pelvis ends at least 1.5 m from its start or travels at least 3 m (or the pelvis
               ends at least 3 m from its start, whatever the placements: fast runs whose feet never pass the planted test); not a crawl
               (the user's list skips Female1General's crawls)          -> outputs/amass_render/locomotion_clips
  borderline   feet move (at least 2 placements) but less travel: single side steps, turning on the spot, martial-arts advances
               and bounces, kicks that end a step away; and the crawls  -> outputs/amass_render/locomotion_borderline
  Clips shorter than 3 s are dropped from both lists (user, 11/11).
  the rest     feet stay put (standing, swaying, gestures, punches, lying down)
usage: uv run python scripts/accad_locomotion.py [--min-sec 5]   (another minimum length: files locomotion_clips_5s, ...)"""
import glob, re, json, os, sys, numpy as np, mujoco
from pathlib import Path
from multiprocessing import Pool
R = "/home/allen19/crossembodiment"
def one(mp4):
    name = os.path.basename(mp4)[:-4]; q = np.load(f"{R}/amass_data/amass_origin_motion/ACCAD/{name}.npz")["qpos"]
    m = mujoco.MjModel.from_xml_path(R + "/assets/robots/adult/robot.xml"); d = mujoco.MjData(m)
    ids = {s: [m.body(f"{s}_Ankle").id, m.body(f"{s}_Toe").id] for s in "LR"}; F = {s: np.empty((len(q), 3)) for s in "LR"}
    for t in range(len(q)):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(m, d)
        for s in "LR":
            P = d.xpos[ids[s]]; F[s][t] = P[np.argmin(P[:, 2])]
    steps = 0
    for s in "LR":
        h = F[s][:, 2]; v = np.r_[0, np.linalg.norm(np.diff(F[s][:, :2], axis=0), axis=1) * 30]; planted = (h < h.min() + 0.05) & (v < 0.3); last = None
        for t in np.flatnonzero(planted):
            if last is None:
                last = F[s][t, :2]
            elif np.linalg.norm(F[s][t, :2] - last) > 0.25:
                steps += 1; last = F[s][t, :2]
    r = q[:, :2]; return dict(clip=name, path=mp4, sec=len(q) / 30, root_path=float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum()), root_net=float(np.linalg.norm(r[-1] - r[0])), steps=steps)
mp4s = sorted(glob.glob(R + "/outputs/amass_render/ACCAD/*.mp4"))
with Pool(16) as p:
    res = p.map(one, mp4s)
nat = lambda r: [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", r["clip"])]
res.sort(key=nat); crawl = lambda r: "crawl" in r["clip"].lower()
MIN = float(sys.argv[sys.argv.index("--min-sec") + 1]) if "--min-sec" in sys.argv else 3.0; SUF = "" if MIN == 3.0 else f"_{MIN:g}s"
short = [r for r in res if r["sec"] < MIN]; res = [r for r in res if r["sec"] >= MIN]       # user, 11/11: no clip shorter than 3 s (--min-sec 5: *_5s files)
loco = [r for r in res if not crawl(r) and ((r["steps"] >= 3 and (r["root_net"] >= 1.5 or r["root_path"] >= 3.0)) or r["root_net"] >= 3.0)]
border = [r for r in res if r not in loco and (r["steps"] >= 2 or crawl(r))]
O = Path(R) / "outputs/amass_render"
(O / f"locomotion_clips{SUF}").write_text("".join(r["path"] + "\n" for r in loco)); (O / f"locomotion_borderline{SUF}").write_text("".join(r["path"] + "\n" for r in border))
(O / f"locomotion_stats{SUF}.json").write_text(json.dumps(res, indent=1))
sel = [l.strip() for l in open(O / "selec_clips") if l.strip()]
print(f"dropped {len(short)} clips shorter than {MIN:g} s; ", end="")
print(f"{len(res)} clips: locomotion {len(loco)}, borderline {len(border)}, feet still {len(res) - len(loco) - len(border)}; "
      f"the user's {len(sel)} picks in locomotion: {sum(p in {r['path'] for r in loco} for p in sel)}")
print("\nborderline:"); [print(f"  {r['clip'][:60]:60s} {r['sec']:5.1f}s placements {r['steps']:3d} path {r['root_path']:4.1f} net {r['root_net']:4.1f} m") for r in border]
