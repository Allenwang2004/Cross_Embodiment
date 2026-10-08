#!/usr/bin/env python3
"""checkpoint_gallery.py -- what does a search look like at 50 / 80 / 90 / 95 / 100% of its gain?

For a few clips of a cosine + heading c540 run (--src, default outputs/c540_bfmglobal_a03; one clip per category,
two for move, each the clip with the category's median relative gain), takes the best-so-far z at the first
generation whose anchored cost reaches each share of the run's own gain (100% = the run's best), replays it with
the search's own batch of 16 (exact observations, from the reference's frame 0; reproduces the recorded rollout),
and scores every checkpoint term by term:
  cos      1 - mean_t cos(B(s_t), B(g_t))                    the cost's local-pose term
  head_deg mean |heading error|, degrees                     (the cost uses (1 - cos) / 2 of it, weight 1.0)
  pos_m    mean pelvis xy distance, metres                   (weight 0.1 in the cost)
  dz       |z - z0|                                          (anchor 0.3 (|z - z0| / 16)^2)
  cost     the search's own cost, recomputed (checks the replay against curve.csv)
and, outside the cost: joint_deg (mean |joint angle error|), mpjpe_loc / mpjpe_glob (all 24 bodies, cm), fallen
(share of frames whose pelvis is below half the reference's).
Writes <out>/{metrics.json, <clip>/<checkpoint>.npz, videos/<clip>.json} -> render with scripts/render_panels.py.
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
XML = "assets/robots_torque/child/robot_torque_full.xml"
SHARES = (0.5, 0.8, 0.9, 0.95)


def pick(src):
    by = defaultdict(list)
    for l in open(src / "clips.txt"):
        t, k, cat, sp = l.split(); s = json.load(open(src / f"{t}_{k}" / "summary.json"))
        c0, cb = s["origin_z"]["cost"], s["best"]["cost"]
        if c0 > cb:
            by[cat].append(((c0 - cb) / c0, t, k, cat))
    out = []
    for cat, rows in sorted(by.items()):
        rows.sort(); n = 2 if cat == "move" else 1
        for q in ([0.5] if n == 1 else [0.3, 0.7]):
            out.append(rows[int(q * (len(rows) - 1))][1:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/c540_bfmglobal_a03")
    ap.add_argument("--out", default="outputs/checkpoint_gallery")
    a = ap.parse_args()
    import mujoco, torch
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from single_z_search import rollout, project_z
    from model.exact_obs import ExactObs
    from model import bfm_align, losses, kinematics as kin
    from mse_vs_align_test import metrics
    src, out = REPO / a.src, REPO / a.out
    out.mkdir(parents=True, exist_ok=True); (out / "videos").mkdir(exist_ok=True)
    dev = "cuda"
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev); model.eval()
    env, _ = make_humenv(num_envs=16, vectorization_mode="async", task=None, xml=str(REPO / XML), state_init="Default")
    fk = mujoco.MjModel.from_xml_path(str(REPO / XML)); ex = ExactObs(REPO / XML, REPO / "assets/robots/adult/robot.xml")
    allm = {}
    for t, k, cat in pick(src):
        stem = f"{t}_{k}"; d = src / stem
        ref = np.load(REPO / f"data/child/retargeting_motion/{t}/{stem}.npz")["qpos"].astype(np.float64)
        z0 = project_z(np.load(REPO / f"data/origin_z/{t}/{stem}.npy").reshape(-1).astype(np.float64))
        summ = json.load(open(d / "summary.json")); c0 = summ["origin_z"]["cost"]
        best = np.minimum.accumulate([min(float(r["best_so_far"]), c0) for r in csv.DictReader(open(d / "curve.csv"))])
        frac = (c0 - best) / (c0 - best[-1])
        tr = np.load(d / "z_trace.npz"); gens = list(tr["gen"])
        cps = [("z0", 0, z0)]
        for s in SHARES:
            g = int(np.argmax(frac >= s)) + 1                     # generations completed
            cps.append((f"{int(s * 100)}%", g, tr["z_best"][gens.index(g - 1)].astype(np.float64)))
        g = summ["best"]["gen"] + 1
        cps.append(("100%", g, np.load(d / "best_z.npy").reshape(-1).astype(np.float64)))
        # reference embeddings through the same exact observation
        vr, ro = np.zeros(fk.nv), []
        for i in range(len(ref)):
            if i:
                mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, ref[i - 1], ref[i])
            ro.append(ex(ref[i], vr))
        Bg = bfm_align.embed(model, np.stack(ro), dev)
        _, rq = kin.batch_forward_pose(fk, ref, [kin.ROOT_BODY]); rh = losses.root_heading(rq[kin.ROOT_BODY])
        rows = []
        (out / stem).mkdir(exist_ok=True)
        for name, g, z in cps:
            zb = torch.as_tensor(np.repeat(project_z(z)[None], 16, 0), dtype=torch.float32, device=dev)
            q, o = rollout(model, env, zb, len(ref), dev, None, init_qpos=ref[0], nv=fk.nv, return_obs=True, exact=ex)
            q, o = q[0].astype(np.float64), o[:1]
            T = min(len(q), len(ref))
            cosv = float(bfm_align.batch_bfm_align(model, o, Bg, dev)[0])
            _, qq = kin.batch_forward_pose(fk, q[:T], [kin.ROOT_BODY]); h = losses.root_heading(qq[kin.ROOT_BODY])
            dh = np.angle(np.exp(1j * (h - rh[:T])))
            head = float(np.mean((1 - np.cos(dh)) / 2)); pos = float(np.mean(np.linalg.norm(q[:T, :2] - ref[:T, :2], axis=1)))
            dz = float(np.linalg.norm(project_z(z) - z0))
            m = metrics(fk, q, ref, losses, kin)
            m.update(name=name, gen=g, rollouts=16 * g, cos=cosv, head_deg=float(np.degrees(np.abs(dh)).mean()), pos_m=pos, dz=dz,
                     cost=cosv + head + 0.1 * pos + 0.3 * (dz / 16) ** 2,
                     cost_logged=float(c0 if g == 0 else best[g - 1]),
                     fallen=float(np.mean(q[:T, 2] < 0.5 * ref[:T, 2])))
            rows.append(m)
            np.savez_compressed(out / stem / f"{name.rstrip('%')}.npz", qpos=q.astype(np.float32), fps=30)
            print(f"{stem:32s} {name:5s} gen {g:2d} | cost {m['cost']:.4f} (logged {m['cost_logged']:.4f}) | cos {cosv:.3f} "
                  f"head {m['head_deg']:5.1f} deg pos {pos:.2f} m | joint {m['joint_deg']:.1f} deg mpjpe {m['mpjpe_glob']:.0f} cm | fallen {m['fallen']:.0%}", flush=True)
        allm[stem] = dict(category=cat, rows=rows)
        panels = [{"title": "reference", "sub": f"{stem} (camera follows this)", "xml": XML,
                   "qpos": f"data/child/retargeting_motion/{t}/{stem}.npz"}]
        for r in rows:
            g = "" if r["name"] == "z0" else f"gen {r['gen']} · {r['rollouts']} rollouts"
            panels.append({"title": f"{r['name']}  {g}", "sub": f"cos {r['cos']:.3f} · heading {r['head_deg']:.0f} deg · pos {r['pos_m']:.2f} m\n"
                           f"joint {r['joint_deg']:.1f} deg · MPJPE {r['mpjpe_glob']:.0f} cm", "xml": XML,
                           "qpos": f"{a.out}/{stem}/{r['name'].rstrip('%')}.npz"})
        (out / "videos" / f"{stem}.json").write_text(json.dumps(
            {"out": f"{a.out}/videos/{stem}.mp4", "cols": 7, "size": 260,
             "track": {"distance": 3.2, "elevation": -15, "azimuth": 135, "lookat_z": 0.5, "follow": "first"}, "panels": panels}, indent=1))
    json.dump(allm, open(out / "metrics.json", "w"), indent=1)
    env.close()


if __name__ == "__main__":
    main()
