#!/usr/bin/env python3
"""fit_smpl_fill.py -- step 1 of the robot pipeline (robot -> SMPL, model/robot_map.py) fills the SMPL parts no robot link
reports with fixed values: the waist bend split evenly over Torso / Spine / Chest, Neck / Head / Thorax / Toe (and Hand
on robots without a hand link) at Motivo's mean posture, the wrist at 0. This fits a ridge regression from what the robot
does report (the paired parts' orientations, relative to each other, plus gravity in the pelvis / chest frames) to those
parts' local rotations, on human motion, and compares it with the fixed fill:

  per part   world-orientation error of every SMPL body after step 1 alone, fed the SMPL body's own paired orientations
             (a perfect IK), on the survey's 11 clips (deg)
  bfm        1 - mean_t cos(B(obs), B(obs_original)) for step 1 alone and for the full round trip SMPL -> robot -> SMPL
             (robot poses from outputs/<robot>/roundtrip, the IK the searches use), G1 / H1 / Apollo

Training data: data/origin_motion (Motivo's clips) and amass_data/amass_origin_motion/ACCAD, every 3rd frame, without any
clip of the survey clips' tasks; the ridge strength is picked on 10% of the training clips. Two models: robots with a
hand link (G1, Apollo) and without (H1, which also has to predict the hand). The fixed-fill path reproduces
RobotMap.robot_to_smpl_qpos exactly (checked on the round trips).
Also fits a small MLP (two hidden layers of 256, SiLU) on the same data, stopped early on the validation clips.
Writes assets/robots/smpl_fill_{ridge,mlp}_{hand,nohand}.npz and outputs/map_error/fill_compare.json (fixed / ridge / mlp).
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/fit_smpl_fill.py [--data all|motivo]
"""
import glob, json, sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robot_map import (RobotMap, SMPL_SIDES, SMPL_XML, _quat, _rotvec, FILL_REL, FILL_REL_HAND,  # noqa: E402
                             FILL_TARGETS, fill_feats, fill_targets)
from model.robots import target_path                                          # noqa: E402
from crossenbodiment.survey import CLIPS   # noqa: E402  survey_body_videos' (refactor step 4)
from map_error_figs import ang_err, body_rots                                 # noqa: E402

OUT = REPO / "outputs/map_error"
SIDES = SMPL_SIDES
# the features and targets robot_map.RobotMap(fill="ridge") uses
REL, REL_HAND, TARGETS, feats, targets = FILL_REL, FILL_REL_HAND, FILL_TARGETS, fill_feats, fill_targets


class Ridge:
    def __init__(self, X, Y, alpha):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-8; self.my = Y.mean(0)
        Z = (X - self.mx) / self.sx
        self.W = np.linalg.solve(Z.T @ Z + alpha * len(Z) * np.eye(Z.shape[1]), Z.T @ (Y - self.my))

    def __call__(self, X):
        return ((X - self.mx) / self.sx) @ self.W + self.my


class MLP:
    """two hidden layers (SiLU), standardised inputs and outputs, trained with Adam on the training clips and stopped
    early on the validation clips; the forward pass runs in numpy (no torch needed to use it)."""

    def __init__(self, X, Y, Xv, Yv, width=256, epochs=300, patience=20, seed=0):
        import torch
        torch.manual_seed(seed)
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-8; self.my, self.sy = Y.mean(0), Y.std(0) + 1e-8
        t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")
        Z, T, Zv, Tv = t((X - self.mx) / self.sx), t((Y - self.my) / self.sy), t((Xv - self.mx) / self.sx), t((Yv - self.my) / self.sy)
        net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], width), torch.nn.SiLU(), torch.nn.Linear(width, width),
                                  torch.nn.SiLU(), torch.nn.Linear(width, T.shape[1])).cuda()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            for idx in torch.randperm(len(Z), device="cuda").split(1024):
                opt.zero_grad(); torch.nn.functional.mse_loss(net(Z[idx]), T[idx]).backward(); opt.step()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(net(Zv), Tv).item()
            if v < best - 1e-5:
                best, bad, best_state = v, 0, [p.detach().cpu().numpy().copy() for p in net.parameters()]
            else:
                bad += 1
                if bad >= patience:
                    break
        self.params, self.epochs = best_state, ep + 1

    def __call__(self, X):
        h = (X - self.mx) / self.sx
        W1, b1, W2, b2, W3, b3 = self.params
        silu = lambda a: a / (1 + np.exp(-a))
        h = silu(h @ W1.T + b1); h = silu(h @ W2.T + b2)
        return (h @ W3.T + b3) * self.sy + self.my


def smpl_qpos(M, R, xy, clearance, pred=None):
    """RobotMap.robot_to_smpl_qpos from the paired SMPL-frame orientations R, root xy and the height of the lowest point
    (SMPL units). pred = {body: local 3x3} for the unreported parts, None = the fixed fill (bit-exact with robot_map)."""
    rel = {}
    pc = R["Pelvis"].T @ R["Chest"]
    if pred is None:
        rel["Torso"] = rel["Spine"] = rel["Chest"] = Rotation.from_rotvec(_rotvec(pc) / 3).as_matrix()
    else:
        rel["Torso"], rel["Spine"] = pred["Torso"], pred["Spine"]
        rel["Chest"] = (pred["Torso"] @ pred["Spine"]).T @ pc
        for b in ("Neck", "Head") + tuple(f"{S}_Toe" for S in SIDES):
            rel[b] = pred[b]
    for S in SIDES:
        rel[f"{S}_Hip"] = R["Pelvis"].T @ R[f"{S}_Hip"]
        rel[f"{S}_Knee"] = R[f"{S}_Hip"].T @ R[f"{S}_Knee"]
        rel[f"{S}_Ankle"] = R[f"{S}_Knee"].T @ R[f"{S}_Ankle"]
        thor = M.fill_R[f"{S}_Thorax"] if pred is None else pred[f"{S}_Thorax"]
        if pred is not None:
            rel[f"{S}_Thorax"] = thor
        rel[f"{S}_Shoulder"] = thor.T @ R["Chest"].T @ R[f"{S}_Shoulder"]
        rel[f"{S}_Elbow"] = R[f"{S}_Shoulder"].T @ R[f"{S}_Elbow"]
        wr = np.eye(3) if pred is None else pred[f"{S}_Wrist"]
        rel[f"{S}_Wrist"] = wr
        if M.has_hand:
            rel[f"{S}_Hand"] = wr.T @ R[f"{S}_Elbow"].T @ R[f"{S}_Hand"]
        elif pred is not None:
            rel[f"{S}_Hand"] = pred[f"{S}_Hand"]
    q = M.ms.qpos0.copy()
    q[3:7] = _quat(R["Pelvis"])
    for body, v in M.fill.items():
        q[[M.s_qadr[M.s_names.index(f"{body}_{ax}")] for ax in "xyz"]] = v
    bodies = list(rel)
    ang = Rotation.from_matrix(np.stack([rel[b] for b in bodies])).as_euler("XYZ")
    q[[M.s_qadr[M.s_names.index(f"{b}_{ax}")] for b in bodies for ax in "xyz"]] = ang.ravel()
    q[:2] = xy; q[2] = 0.0
    M.ds.qpos[:] = q; mujoco.mj_kinematics(M.ms, M.ds)
    q[2] = clearance - M.low_s(M.ds)
    return q


def predict_all(model, Rs, hand):
    """per frame {body: local 3x3} for the unreported parts, from a list of per-frame R dicts (one batched call)."""
    Rb = {b: np.stack([R[b] for R in Rs]) for b in Rs[0]}
    Y = model(feats(Rb, hand)).reshape(len(Rs), -1, 3)
    return [{b: Rotation.from_rotvec(v).as_matrix() for b, v in zip(targets(hand), y)} for y in Y]


def clip_arrays(M, ds, Q, hand):
    """per frame: the paired parts' world orientations (features) and the unreported parts' local rotations (targets)."""
    W = body_rots(M.ms, ds, Q)                                            # (T, nbody-1, 3, 3)
    R = {b: W[:, M.ms.body(b).id - 1] for b, _ in M.pairs + ([] if hand else [])}
    if hand:
        R.update({f"{S}_Hand": W[:, M.ms.body(f"{S}_Hand").id - 1] for S in SIDES})
    Y = np.concatenate([Rotation.from_euler("XYZ", Q[:, [M.s_qadr[M.s_names.index(f"{b}_{ax}")] for ax in "xyz"]]).as_rotvec()
                        for b in targets(hand)], 1)
    return feats(R, hand), Y


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["all", "motivo"], default="all",
                    help="all = Motivo's clips + AMASS ACCAD; motivo = data/origin_motion only (outputs get a _motivo suffix)")
    a = ap.parse_args()
    sfx = "" if a.data == "all" else "_motivo"
    rng = np.random.default_rng(0)
    survey_tasks = {s.rsplit("_", 1)[0] for s in CLIPS}
    files = [f for f in sorted(glob.glob(str(REPO / "data/origin_motion/*/*.npz"))) if Path(f).parent.name not in survey_tasks]
    if a.data == "all":
        files += sorted(glob.glob(str(REPO / "amass_data/amass_origin_motion/ACCAD/*.npz")))
    val = set(rng.choice(len(files), len(files) // 10, replace=False))
    models, out = {}, {}
    for hand, key in ((True, "g1"), (False, "h1")):
        M = RobotMap(key); ds = mujoco.MjData(M.ms)
        Xt, Yt, Xv, Yv = [], [], [], []
        for i, f in enumerate(files):
            Q = np.load(f)["qpos"][::3]
            if Q.shape[1] != M.ms.nq or len(Q) < 2:
                continue
            X, Y = clip_arrays(M, ds, Q, hand)
            (Xv if i in val else Xt).append(X); (Yv if i in val else Yt).append(Y)
        Xt, Yt, Xv, Yv = map(np.concatenate, (Xt, Yt, Xv, Yv))
        errs = {}
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            errs[alpha] = float(np.degrees(np.sqrt(((Ridge(Xt, Yt, alpha)(Xv) - Yv) ** 2).mean())))
        alpha = min(errs, key=errs.get)
        mdl = Ridge(np.concatenate([Xt, Xv]), np.concatenate([Yt, Yv]), alpha)
        mlp = MLP(Xt, Yt, Xv, Yv)
        models[hand] = {"ridge": mdl, "mlp": mlp}
        name = "hand" if hand else "nohand"
        meta = dict(targets=np.array(targets(hand)), features=np.array([f"{a}->{b}" for a, b in REL + (REL_HAND if hand else [])]))
        np.savez(REPO / f"assets/robots/smpl_fill_ridge_{name}{sfx}.npz", mx=mdl.mx, sx=mdl.sx, my=mdl.my, W=mdl.W, alpha=alpha, **meta)
        np.savez(REPO / f"assets/robots/smpl_fill_mlp_{name}{sfx}.npz", mx=mlp.mx, sx=mlp.sx, my=mlp.my, sy=mlp.sy,
                 **{f"p{i}": p for i, p in enumerate(mlp.params)}, **meta)
        mlp_err = float(np.degrees(np.sqrt(((mlp(Xv) - Yv) ** 2).mean())))
        print(f"[{name}] {len(Xt) + len(Xv)} frames from {len(files)} clips, {Xt.shape[1]} features -> {Yt.shape[1]} targets; "
              f"validation RMSE (deg) by ridge strength: " + ", ".join(f"{a:g}: {e:.1f}" for a, e in errs.items()) +
              f" | MLP {mlp_err:.1f} ({mlp.epochs} epochs)", flush=True)

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    env, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / SMPL_XML), state_init="Default")
    bfm = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    emb = lambda Q: bfm_align.embed(bfm, bfm_align.obs_from_qpos(env, Q), "cuda")
    for key in ("g1", "h1", "apollo"):
        M = RobotMap(key); ds = mujoco.MjData(M.ms); hand = M.has_hand
        names = [M.ms.body(i).name for i in range(1, M.ms.nbody)]
        modes = ("fixed", "ridge", "mlp")
        part_err, b1, bfull = ({m: [] for m in modes} for _ in range(3))
        for stem in CLIPS:
            ref = np.load(REPO / f"data/origin_motion/{stem.rsplit('_', 1)[0]}/{stem}.npz")["qpos"]
            qg = np.load(target_path(key, stem))["qpos"]
            Wref = body_rots(M.ms, ds, ref); B0 = emb(ref)
            Rs, clr, Rrs, clr_r, xy_r = [], [], [], [], []
            for t in range(len(ref)):
                M.ds.qpos[:] = ref[t]; mujoco.mj_kinematics(M.ms, M.ds)
                Rs.append({b: M.ds.xmat[M.ms.body(b).id].reshape(3, 3).copy() for b, _ in M.pairs}); clr.append(M.low_s(M.ds))
                Rrs.append({b: r.copy() for b, r in M.smpl_rots_from_robot(qg[t]).items()})
                clr_r.append(M.low_g(M.dg) / M.s); xy_r.append(qg[t][M.free_q:M.free_q + 2] / M.s)
            for mode in modes:
                p1 = [None] * len(ref) if mode == "fixed" else predict_all(models[hand][mode], Rs, hand)
                pr = [None] * len(ref) if mode == "fixed" else predict_all(models[hand][mode], Rrs, hand)
                st1 = np.stack([smpl_qpos(M, Rs[t], ref[t][:2], clr[t], p1[t]) for t in range(len(ref))])
                rt = np.stack([smpl_qpos(M, Rrs[t], xy_r[t], clr_r[t], pr[t]) for t in range(len(ref))])
                if mode == "fixed" and stem == CLIPS[0]:
                    ref_rt = np.load(target_path(key, stem).parent / "smpl_rt.npz")["qpos"]
                    print(f"  {key}: fixed fill vs robot_map round trip, max |dq| = {np.abs(rt - ref_rt).max():.2e}")
                part_err[mode].append(ang_err(Wref, body_rots(M.ms, ds, st1)).mean(0))
                b1[mode].append(bfm_align.bfm_align_loss(emb(st1), B0)); bfull[mode].append(bfm_align.bfm_align_loss(emb(rt), B0))
        res = {}
        for mode in modes:
            pe = np.mean(part_err[mode], 0)
            res[mode] = dict(bfm_step1=float(np.mean(b1[mode])), bfm_full=float(np.mean(bfull[mode])),
                             part_deg={n: float(v) for n, v in zip(names, pe)},
                             bfm_full_per_clip=dict(zip(CLIPS, map(float, bfull[mode]))))
        out[key] = res
        show = ["Torso", "Spine", "Neck", "Head", "L_Thorax", "L_Wrist", "L_Hand", "L_Toe"]
        arrow = lambda f: " -> ".join(f(res[m]) for m in modes)
        print(f"{key} (fixed -> ridge -> mlp): bfm step 1 alone {arrow(lambda r: format(r['bfm_step1'], '.3f'))} | full round trip "
              f"{arrow(lambda r: format(r['bfm_full'], '.3f'))}\n    part error (deg): " +
              ", ".join(f"{b} {arrow(lambda r: format(r['part_deg'][b], '.1f'))}" for b in show), flush=True)
    (OUT / f"fill_compare{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
