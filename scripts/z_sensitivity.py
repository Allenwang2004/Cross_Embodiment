#!/usr/bin/env python3
"""z_sensitivity.py -- which directions of the 256-d latent does the frozen actor actually use?

Hypothesis under test: many directions of z barely change the actions, the ES search drifts
along them, and that is why latents found from nearby starts end up far apart. If so the
drift (differences between searched latents that produce the same motion) should lie in the
directions the actor is insensitive to.

  1  ACTOR SENSITIVITY. Roll out latents on a body, take every 5th visited state, and compute
     J = d mean_action / d z (69 x 256) at the latent that produced it, projected to z's
     tangent plane (the search keeps |z| = 16, so the radial direction never matters).
     C = sum_s J^T J is the action-change energy per latent direction; its eigenvectors,
     sorted, are the latent directions from most to least action-relevant.
     Latents: z0 and a searched latent of each of 10 clips (8 path clips + headstand_3 +
     move-ego-0-2_4), on the child and on the adult (m2c_t000). Obs are rescaled exactly as
     in the searches.
  2  REFERENCE SPECTRA. PCA of B(s) over reference frames of every clip on the adult, and PCA
     of the 720 clips' z0; overlap of their top subspaces with the top sensitivity subspace.
  3  WHERE DOES THE SPREAD LIE. Differences between searched latents that track (nearly) the
     same motion: the 8 walking starts (own target, two-stage L_align) and the 20 walking
     seeds (two-stage L_align). For each difference d: share of |d|^2 inside the top-k
     sensitivity directions, and d^T C d / |d|^2 relative to a random tangent direction
     (1 = as action-relevant as a random direction, << 1 = in the insensitive part).

usage: CUDA_VISIBLE_DEVICES=1 uv run scripts/z_sensitivity.py
"""
import glob, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.append(str(REPO))
import numpy as np
import torch

PATH8 = [l.strip() for l in open(REPO / "outputs/continuation_clips.txt") if l.strip()]
CLIPS = PATH8 + ["headstand/headstand_3", "move-ego-0-2/move-ego-0-2_4"]
OUT = REPO / "outputs/z_sensitivity"


def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / np.linalg.norm(v)


def searched_latent(clip):
    stem = clip.split("/")[1]
    p = REPO / f"outputs/latent_transfer_own/{stem}/align/z0/best_z.npy"
    if p.exists():
        return np.load(p)
    return np.load(REPO / f"outputs/continuation/m2c/cont/{stem}__m2c_t1000/best_z.npy")


def k_for(ev, f):
    c = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(c, f) + 1)


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    import mujoco
    from model import bfm_align
    from model.obs_scale import build_obs_multiplier
    from model.simple.config import ESConfig
    from single_z_search import rollout, project_z
    dev = "cuda:0"
    cfg = ESConfig(device=dev)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev); model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    std = model.cfg.actor_std

    def jac(obs, z):
        """obs (n, 358) rescaled, z (256,) -> (n, 69, 256) tangent-projected Jacobian of the mean action."""
        o = model._normalize(torch.as_tensor(obs, dtype=torch.float32, device=dev))
        zt = torch.as_tensor(z, dtype=torch.float32, device=dev)
        f = lambda zz: model._actor(o, zz.expand(o.shape[0], -1), std).mean
        J = torch.autograd.functional.jacobian(f, zt).cpu().numpy().astype(np.float64)
        zh = unit(z)
        return J - np.einsum("nad,d,e->nae", J, zh, zh)

    C, Cclip = {}, {}
    for body in ("child", "m2c_t000"):
        xml = REPO / "assets/robots_torque" / body / "robot_torque_full.xml"
        obs_mul = build_obs_multiplier(xml, REPO / "assets/robots/adult/robot.xml", mode="auto",
                                       parts=cfg.obs_scale_parts, verbose=False)
        nv = mujoco.MjModel.from_xml_path(str(xml)).nv
        env, _ = make_humenv(num_envs=2, vectorization_mode="async", task=None, xml=str(xml), state_init="Default")
        C[body] = np.zeros((256, 256))
        for clip in CLIPS:
            task, stem = clip.split("/")
            ref = np.load(REPO / "data" / body / "retargeting_motion" / task / f"{stem}.npz")["qpos"]
            z0 = project_z(np.load(REPO / "data/origin_z" / task / f"{stem}.npy").reshape(-1).astype(np.float64))
            zs = project_z(searched_latent(clip).reshape(-1).astype(np.float64))
            Z = np.stack([z0, zs])
            _, o = rollout(model, env, torch.as_tensor(Z, dtype=torch.float32, device=dev), 300, dev, obs_mul,
                           init_qpos=ref[0], nv=nv, return_obs=True)
            Ck = np.zeros((256, 256))
            for i in range(2):
                J = jac(o[i, ::5], Z[i])
                Ck += np.einsum("nad,nae->de", J, J)
            Cclip[(body, stem)] = Ck; C[body] += Ck
        env.close()
        print(f"  {body}: sensitivity from {len(CLIPS)} clips x 2 latents x 60 states", flush=True)

    rep = []
    def say(s):
        print(s, flush=True); rep.append(s)

    say("== 1  actor sensitivity spectrum (eigenvalues of sum J^T J, tangent plane, 255 usable directions)")
    E = {}
    for body in C:
        w, V = np.linalg.eigh(C[body]); w, V = w[::-1].clip(0), V[:, ::-1]; E[body] = (w, V)
        say(f"  {body:9s}: directions holding 90/95/99% of the action-change energy: {k_for(w, .9)}/{k_for(w, .95)}/{k_for(w, .99)}"
            f" | largest/median eigenvalue {w[0] / np.median(w[:255]):.0f}x | participation ratio {w.sum() ** 2 / (w ** 2).sum():.1f}")
    ks = {}
    for (body, stem), Ck in Cclip.items():
        w = np.linalg.eigvalsh(Ck)[::-1].clip(0); ks.setdefault(body, []).append(k_for(w, .95))
    for body, v in ks.items():
        say(f"  {body:9s}: per clip, 95% of the energy in {min(v)}-{max(v)} directions (median {int(np.median(v))})")
    Vc, Va = E["child"][1], E["m2c_t000"][1]
    for k in (16, 32, 64):
        s = np.linalg.svd(Vc[:, :k].T @ Va[:, :k], compute_uv=False)
        say(f"  top-{k} subspace, child vs adult: mean cos^2 of principal angles {np.mean(s ** 2):.2f} (random ~{k / 256:.2f})")

    say("== 2  reference spectra")
    from humenv import make_humenv as mk
    import mujoco
    xml = REPO / "assets/robots_torque/m2c_t000/robot_torque_full.xml"
    obs_mul = build_obs_multiplier(xml, REPO / "assets/robots/adult/robot.xml", mode="auto", parts=cfg.obs_scale_parts, verbose=False)
    env1, _ = mk(num_envs=1, task=None, xml=str(xml), state_init="Default")
    Bs = []
    for f in sorted(glob.glob(str(REPO / "data/m2c_t000/retargeting_motion/*/*.npz"))):
        q = np.load(f)["qpos"][::10]
        Bs.append(bfm_align.reference_embeddings(model, env1, q, dev, obs_mul))
    env1.close()
    B = np.concatenate(Bs).astype(np.float64); B /= np.linalg.norm(B, axis=1, keepdims=True)
    Z0 = np.stack([unit(np.load(f)) for f in glob.glob(str(REPO / "data/origin_z/*/*.npy"))])
    for name, X in ((f"B(s), {len(B)} reference frames of {len(Bs)} clips (adult)", B), (f"z0 of {len(Z0)} clips", Z0)):
        Xc = X - X.mean(0); _, sv, Vt = np.linalg.svd(Xc, full_matrices=False); ev = sv ** 2
        line = f"  {name}: 90/95/99% of variance in {k_for(ev, .9)}/{k_for(ev, .95)}/{k_for(ev, .99)} directions"
        for k in (16, 32, 64):
            s = np.linalg.svd(Vt[:k] @ Vc[:, :k], compute_uv=False)
            line += f" | top-{k} overlap with child sensitivity {np.mean(s ** 2):.2f}"
        say(line)

    say("== 3  where does the spread between equally good latents lie (child sensitivity)")
    w, V = E["child"]
    rng = np.random.default_rng(0)
    def spread_report(name, Zs):
        Zs = [16 * unit(z) for z in Zs]
        D = [Zs[i] - Zs[j] for i in range(len(Zs)) for j in range(i + 1, len(Zs))]
        D = np.stack(D)
        rel = []
        for d in D:
            r = rng.standard_normal(256); m = unit(Zs[0]); r -= (r @ m) * m
            rel.append((d @ C["child"] @ d / (d @ d)) / (r @ C["child"] @ r / (r @ r)))
        line = f"  {name}: {len(D)} pairs, distance {np.linalg.norm(D, axis=1).mean():.2f}"
        for k in (16, 32, 64):
            share = np.mean([np.sum((V[:, :k].T @ d) ** 2) / (d @ d) for d in D])
            line += f" | in top-{k}: {share:.0%} (random {k / 256:.0%})"
        line += f" | action relevance vs a random direction: {np.median(rel):.2f}x"
        say(line)
    N8 = ["z0", "rot0.5_0", "rot1_0", "rot2_0", "rot5_0", "rot10_0", "rot20_0", "rot30_0"]
    spread_report("walking, 8 starts, own target, L_align", [np.load(REPO / f"outputs/latent_transfer_own/move-ego-0-2_4/align/{n}/best_z.npy") for n in N8])
    spread_report("walking, 20 seeds, two-stage L_align  ", [np.load(REPO / f"outputs/single_z_seeds_twostage_align/move-ego-0-2_4_s{k}/best_z.npy") for k in range(20)])
    spread_report("headstand, 8 starts, own target, L_align", [np.load(REPO / f"outputs/latent_transfer_own/headstand_3/align/{n}/best_z.npy") for n in N8])
    np.savez(OUT / "sensitivity.npz", **{f"C_{b}": C[b] for b in C}, **{f"eig_{b}": E[b][0] for b in E}, **{f"V_{b}": E[b][1] for b in E})
    (OUT / "report.txt").write_text("\n".join(rep) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3)); fig.patch.set_facecolor("#fcfcfb")
    for body, col in (("child", "#2a78d6"), ("m2c_t000", "#eb6834")):
        w = E[body][0][:255]
        ax[0].semilogy(np.arange(1, 256), w / w[0], color=col, lw=2, label=body)
        ax[1].plot(np.arange(1, 256), np.cumsum(w) / w.sum(), color=col, lw=2, label=body)
    ax[1].plot([1, 255], [1 / 255, 1], color="#8a8983", ls=(0, (4, 3)), lw=1.1, label="if every direction mattered equally")
    ax[0].set_xlabel("latent direction (sorted by action sensitivity)"); ax[0].set_ylabel("sensitivity / largest (log)")
    ax[1].set_xlabel("number of directions"); ax[1].set_ylabel("cumulative share of action-change energy")
    ax[0].set_title("how much each latent direction changes the actions", fontsize=11); ax[1].set_title("cumulative", fontsize=11)
    for x in ax:
        x.set_facecolor("#fcfcfb"); x.grid(alpha=.22, color="#8a8983", lw=.7); x.legend(frameon=False, fontsize=9)
        for sp in ("top", "right"): x.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "spectrum.png", dpi=150, facecolor="#fcfcfb")
    print(f"-> {OUT}/report.txt, spectrum.png, sensitivity.npz")


if __name__ == "__main__":
    main()
