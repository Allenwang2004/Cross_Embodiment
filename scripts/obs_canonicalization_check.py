#!/usr/bin/env python3
"""obs_canonicalization_check.py -- re-measure the observation-canonicalization ablation (paper paragraph
"Observation canonicalization (Phi_beta)", docs/latex/observation_canonicalization.tex) with the current pipeline
(user, 11/10: the paragraph's numbers had no data in the repo; re-run and verify).

The 52 child task motions of outputs/child_52/clips.txt, the torque-calibrated child (robots_torque/child), the frozen
nominal latent z0 (no search), one rollout per motion and observation setting (16 envs, --pairs 8: with 2 the GPU batch differs and the rollout drifts by chaos), everything else as outputs/child_52_rerun
(start from the reference's frame 0 with its velocity, q[t] scored against ref[t + 1]):
  raw       --obs-scale none       the child's own proprioception
  perbody   --obs-scale auto       every length feature / its own body's length ratio (model/obs_scale.py "auto";
                                   what the code and outputs/interface_ladder call "multiplier")
  uniform   --obs-scale 0.611036   every length feature / ONE ratio, the child/adult root scale s
  exact     --obs-scale exact      Phi_beta: the adult-equivalent state by reverse retargeting (model/exact_obs.py)
Joint error: wrapped hinge-angle error over the 69 joint DOFs, per motion as the per-joint RMS (sqrt of the mean over
frames and DOFs of the squared error) and as the error vector's norm averaged over frames (the paragraph's "mean joint error").
Backward-space similarity: each motion's reference frames observed on the child under each setting, embedded with the frozen
backward map B, cosine against B of the adult's own motion at the same frame (data/origin_motion), mean over frames.
Checks: exact's z0 rollout = outputs/child_52_rerun/<stem>_es_s0/origin_z.npz; frames of the adult and child references match.
  commands   "<out dir> <command>" per motion and setting (out: outputs/obs_canon_52/<setting>/<stem>)
  cosine     -> outputs/obs_canon_52/cosine.json (GPU)
  analyse    -> outputs/obs_canon_52/results.json, README.md
usage: uv run python scripts/obs_canonicalization_check.py commands | cosine | analyse"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossenbodiment.geometry import sphere16   # noqa: E402  16 z / |z| (refactor step 3)
from crossenbodiment.bodies import BODIES
from crossenbodiment.paths import ADULT_XML_REL, CHILD_52_TASK, CHILD_XML_REL, MOTIVO_CKPT, read_clips
OUT = REPO / "outputs/obs_canon_52"; XML = CHILD_XML_REL; ADULT = ADULT_XML_REL
S = BODIES.scale("child")
SETTINGS = dict(raw="none", perbody="auto", uniform=str(S), exact="exact")
NAMES = dict(raw="raw", perbody="per-body multiplier", uniform="uniform multiplier (s)", exact="Phi_beta (exact)")


def clips():
    return read_clips(CHILD_52_TASK)   # configs/clips/child_52_task.txt = outputs/child_52/clips.txt (refactor step 4)


def family(stem):
    t = stem.rsplit("_", 1)[0]
    for k in ("move-ego-0-2-raisearms", "move-ego-low", "move-ego", "raisearms", "rotate", "crawl", "lieonground", "sitonground",
              "split", "headstand", "crouch", "jump"):
        if t.startswith(k):
            return {"move-ego-0-2-raisearms": "move-arms", "raisearms": "raise-arms"}.get(k, k)
    return t


def commands():
    for c in clips():
        task, stem = c.split("/")
        z = np.load(REPO / f"data/origin_z/{task}/{stem}.npy").reshape(-1).astype(np.float64); z = sphere16(z)
        for k, v in SETTINGS.items():
            d = OUT / k / stem; d.mkdir(parents=True, exist_ok=True); np.save(d / "z0.npy", z[None])
            print(d.relative_to(REPO), f"uv run python scripts/single_z_search.py --clip {c} --body child --xml {XML} --objective bfm --pairs 8 "
                  f"--obs-scale {v} --eval-every 0 --init-vel-scale 1 --eval-z {(d / 'z0.npy').relative_to(REPO)} --eval-save-qpos --out {d.relative_to(REPO)}")


def cosine():
    import torch
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    from model.exact_obs import ExactObs
    from model.obs_scale import build_obs_multiplier
    model = FBcprModel.from_pretrained(MOTIVO_CKPT).to("cuda"); model.eval()
    envc, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / XML), state_init="Default")
    enva, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / ADULT), state_init="Default")
    mul = dict(raw=None, perbody=build_obs_multiplier(str(REPO / XML), str(REPO / ADULT), mode="auto", verbose=False),
               uniform=build_obs_multiplier(str(REPO / XML), str(REPO / ADULT), mode=str(S), verbose=False))
    ex = ExactObs(REPO / XML, REPO / ADULT); res = {}
    import mujoco
    fk = mujoco.MjModel.from_xml_path(str(REPO / XML))

    def cos(a, b):
        return float((np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).mean())
    with torch.no_grad():
        for c in clips():
            task, stem = c.split("/")
            rc = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64)
            ra = np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"].astype(np.float64)
            assert rc.shape == ra.shape and np.abs(rc[:, 7:] - ra[:, 7:]).max() < 1e-6, c        # same frames, same joint angles
            Ba = bfm_align.embed(model, bfm_align.obs_from_qpos(enva, ra, bfm_align.DEFAULT_DT, None), "cuda"); r = {}
            for k, m in mul.items():
                r[k] = cos(bfm_align.embed(model, bfm_align.obs_from_qpos(envc, rc, bfm_align.DEFAULT_DT, m), "cuda"), Ba)
            vr, ro = np.zeros(fk.nv), []
            for t in range(len(rc)):
                if t:
                    mujoco.mj_differentiatePos(fk, vr, bfm_align.DEFAULT_DT, rc[t - 1], rc[t])
                ro.append(ex(rc[t], vr))
            r["exact"] = cos(bfm_align.embed(model, np.stack(ro), "cuda"), Ba); res[stem] = r
            print(stem, {k: round(v, 5) for k, v in r.items()}, flush=True)
    (OUT / "cosine.json").write_text(json.dumps(res, indent=1))


def analyse():
    wrap = lambda x: np.angle(np.exp(1j * x))
    R, chk = {}, {}
    for c in clips():
        task, stem = c.split("/"); ref = np.load(REPO / f"data/child/retargeting_motion/{c}.npz")["qpos"].astype(np.float64); R[stem] = {}
        for k in SETTINGS:
            q = np.load(OUT / k / stem / "eval_qpos.npz")["qpos"][0].astype(np.float64); n = min(len(q), len(ref) - 1)
            e = wrap(q[:n, 7:] - ref[1:n + 1, 7:])
            R[stem][k] = dict(rms=float(np.sqrt((e ** 2).mean())), norm=float(np.linalg.norm(e, axis=1).mean()))
            if k == "exact":
                o = np.load(REPO / f"outputs/child_52_rerun/{stem}_es_s0/origin_z.npz")["qpos"].astype(np.float64)
                chk[stem] = float(np.abs(o[:len(q)] - q[:len(o)]).max())
    C = json.loads((OUT / "cosine.json").read_text()) if (OUT / "cosine.json").exists() else {}
    st = sorted(R); fam = {s: family(s) for s in st}; L = []
    m = lambda k, x: float(np.mean([R[s][k][x] for s in st]))
    L.append(f"52 child task motions, z0, one rollout each. exact rollout vs child_52_rerun origin_z, largest qpos difference: {max(chk.values()):.2e}")
    L.append("\n| setting | per-joint RMS (rad) | (deg) | mean joint-error norm (rad) | B cosine vs adult: mean | min-max over motions |\n|---|---|---|---|---|---|")
    for k in SETTINGS:
        cs = [C[s][k] for s in st] if C else [np.nan]
        L.append(f"| {NAMES[k]} | {m(k, 'rms'):.3f} | {np.degrees(m(k, 'rms')):.1f} | {m(k, 'norm'):.3f} | {np.mean(cs):.4f} | {np.min(cs):.4f}-{np.max(cs):.4f} |")
    L.append("\nPhi_beta (exact) vs each, per-joint RMS: mean reduction; motions where exact is lower")
    for k in ("raw", "perbody", "uniform"):
        L.append(f"- vs {NAMES[k]}: {100 * (1 - m('exact', 'rms') / m(k, 'rms')):.1f}% (norm: {100 * (1 - m('exact', 'norm') / m(k, 'norm')):.1f}%); "
                 f"lower on {sum(R[s]['exact']['rms'] < R[s][k]['rms'] for s in st)}/{len(st)}")
    for k in ("perbody", "uniform"):
        L.append(f"- {NAMES[k]} vs raw: lower on {sum(R[s][k]['rms'] < R[s]['raw']['rms'] for s in st)}/{len(st)}")
    L.append("\nBy task family: mean per-joint RMS (rad) [mean norm]\n\n| family | n | raw | per-body | uniform | exact |\n|---|---|---|---|---|---|")
    for f in sorted(set(fam.values())):
        ss = [s for s in st if fam[s] == f]
        L.append(f"| {f} | {len(ss)} | " + " | ".join(f"{np.mean([R[s][k]['rms'] for s in ss]):.3f} [{np.mean([R[s][k]['norm'] for s in ss]):.2f}]" for k in SETTINGS) + " |")
    L.append("\nPer motion: per-joint RMS (rad) raw / per-body / uniform / exact\n")
    L += [f"- {s}: " + " / ".join(f"{R[s][k]['rms']:.3f}" for k in SETTINGS) + (f" | cos " + " / ".join(f"{C[s][k]:.4f}" for k in SETTINGS) if C else "") for s in st]
    (OUT / "results.json").write_text(json.dumps(dict(joint=R, cosine=C, check_exact_vs_rerun=chk), indent=1))
    (OUT / "README.md").write_text("# Observation canonicalization, re-measured\n\n" + __doc__ + "\n\n" + "\n".join(L) + "\n"); print("\n".join(L[:20]))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
