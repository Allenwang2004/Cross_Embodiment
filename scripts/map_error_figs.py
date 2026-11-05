#!/usr/bin/env python3
"""map_error_figs.py -- how much the robot pipeline's two maps lose, on the adult's original clips (no physics):

  step 4  SMPL -> robot (IK)   per paired body part, the orientation the robot cannot reach (RobotMap IK residual, deg)
  step 1  robot -> SMPL        on its own: the back-map fed the SMPL body's exact paired orientations (as if the IK were
                               perfect); per SMPL body part, the world-orientation error against the original (deg)
  both    the full round trip SMPL -> robot -> SMPL, and step 1 alone, as Motivo sees them: bfm = 1 - mean_t cos(B(obs), B(obs_original))

Averages are over the survey's 11 clips (clip means) and left / right. Writes outputs/map_error/{ik_step4, backmap_step1,
bfm_floor}.png and numbers.json. Nothing in outputs/<robot>/roundtrip is touched.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/map_error_figs.py
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robot_map import RobotMap, SMPL_XML, SMPL_SIDES, _quat, _rotvec   # noqa: E402
from survey_body_videos import CLIPS                                          # noqa: E402

OUT = REPO / "outputs/map_error"
ROBOTS = [("g1", "G1", "#2a78d6"), ("h1", "H1", "#eb6834"), ("apollo", "Apollo", "#1baf7a")]   # dataviz categorical 1-3
INK, MUTED, GRID, SURF = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"
PART = {"Chest": "胸", "Hip": "大腿", "Knee": "小腿", "Ankle": "腳掌", "Shoulder": "上臂", "Elbow": "前臂", "Hand": "手",
        "Pelvis": "骨盆", "Torso": "腰（下）", "Spine": "腰（上）", "Neck": "脖子", "Head": "頭", "Thorax": "鎖骨",
        "Wrist": "手腕", "Toe": "腳趾"}
part = lambda b: PART[b.split("_", 1)[-1]]


def back_ideal(M, qs):
    """RobotMap.robot_to_smpl_qpos with the robot's paired link orientations replaced by the SMPL body's own (a perfect IK):
    what the back-map alone loses."""
    M.ds.qpos[:] = qs; mujoco.mj_kinematics(M.ms, M.ds)
    R = {b: M.ds.xmat[M.sid[b]].reshape(3, 3).copy() for b, _ in M.pairs}
    clearance = M.low_s(M.ds)
    rel = {}
    third = Rotation.from_rotvec(_rotvec(R["Pelvis"].T @ R["Chest"]) / 3).as_matrix()
    rel["Torso"] = rel["Spine"] = rel["Chest"] = third
    for S in SMPL_SIDES:
        rel[f"{S}_Hip"] = R["Pelvis"].T @ R[f"{S}_Hip"]
        rel[f"{S}_Knee"] = R[f"{S}_Hip"].T @ R[f"{S}_Knee"]
        rel[f"{S}_Ankle"] = R[f"{S}_Knee"].T @ R[f"{S}_Ankle"]
        rel[f"{S}_Shoulder"] = M.fill_R[f"{S}_Thorax"].T @ R["Chest"].T @ R[f"{S}_Shoulder"]
        rel[f"{S}_Elbow"] = R[f"{S}_Shoulder"].T @ R[f"{S}_Elbow"]
        rel[f"{S}_Wrist"] = np.eye(3)
        if M.has_hand:
            rel[f"{S}_Hand"] = R[f"{S}_Elbow"].T @ R[f"{S}_Hand"]
    q = M.ms.qpos0.copy()
    q[3:7] = _quat(R["Pelvis"])
    for body, v in M.fill.items():
        q[[M.s_qadr[M.s_names.index(f"{body}_{ax}")] for ax in "xyz"]] = v
    bodies = list(rel)
    ang = Rotation.from_matrix(np.stack([rel[b] for b in bodies])).as_euler("XYZ")
    q[[M.s_qadr[M.s_names.index(f"{b}_{ax}")] for b in bodies for ax in "xyz"]] = ang.ravel()
    q[:2] = qs[:2]; q[2] = 0.0
    M.ds.qpos[:] = q; mujoco.mj_kinematics(M.ms, M.ds)
    q[2] = clearance - M.low_s(M.ds)
    return q


def body_rots(ms, ds, Q):
    out = np.zeros((len(Q), ms.nbody - 1, 3, 3))
    for t, q in enumerate(Q):
        ds.qpos[:] = q; mujoco.mj_kinematics(ms, ds); out[t] = ds.xmat[1:].reshape(-1, 3, 3)
    return out


def ang_err(Ra, Rb):
    """per-frame, per-body angle (deg) of Ra^T Rb."""
    return np.degrees(np.linalg.norm(Rotation.from_matrix((Ra.transpose(0, 1, 3, 2) @ Rb).reshape(-1, 3, 3)).as_rotvec(), axis=1)
                      ).reshape(Ra.shape[:2])


def group(d):
    """{body name: value} -> {part: mean over left / right}."""
    g = {}
    for b, v in d.items():
        g.setdefault(part(b), []).append(v)
    return {k: float(np.mean(v)) for k, v in g.items()}


def bars(ax, cats, series, ylabel, fmt="{:.1f}"):
    n = len(series); w = 0.8 / n; x = np.arange(len(cats))
    for i, (label, color, vals) in enumerate(series):
        for j, c in enumerate(cats):
            v = vals.get(c)
            if v is None:
                continue
            xi = x[j] - 0.4 + w * (i + 0.5)
            ax.bar(xi, v, w - 0.03, color=color, label=label if j == 0 or label not in ax.get_legend_handles_labels()[1] else None,
                   zorder=3)
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, cats, fontsize=11, color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (Path.home() / ".local/share/fonts").glob("NotoSansCJK*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Noto Sans CJK TC"
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    clips = [f"{s.rsplit('_', 1)[0]}/{s}" for s in CLIPS]
    res = json.loads((OUT / "numbers.json").read_text()) if "--plot-only" in sys.argv else {}
    if not res:
        env, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / SMPL_XML), state_init="Default")
        model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    for key, name, _ in ([] if res else ROBOTS):
        M = RobotMap(key); ms = M.ms; ds = mujoco.MjData(ms)
        names = [ms.body(i).name for i in range(1, ms.nbody)]
        links = [b for b, _ in M.pairs[1:]]
        ik, e1, bfm1, bfm14 = [], [], [], []
        for clip in clips:
            ref = np.load(REPO / f"data/origin_motion/{clip}.npz")["qpos"]
            T = len(ref); qg = np.zeros((T, M.mg.nq)); resid = np.zeros((T, len(links))); q = None
            for t in range(T):
                qg[t], resid[t] = M.smpl_to_robot_qpos(ref[t], q); q = qg[t][M.g_qadr]
            rt = np.stack([M.robot_to_smpl_qpos(g) for g in qg])
            st1 = np.stack([back_ideal(M, r) for r in ref])
            ik.append(resid.mean(0))
            e1.append(ang_err(body_rots(ms, ds, ref), body_rots(ms, ds, st1)).mean(0))
            B0 = bfm_align.embed(model, bfm_align.obs_from_qpos(env, ref), "cuda")
            bfm1.append(bfm_align.bfm_align_loss(bfm_align.embed(model, bfm_align.obs_from_qpos(env, st1), "cuda"), B0))
            bfm14.append(bfm_align.bfm_align_loss(bfm_align.embed(model, bfm_align.obs_from_qpos(env, rt), "cuda"), B0))
        res[key] = dict(ik=group(dict(zip(links, np.mean(ik, 0)))), step1=group(dict(zip(names, np.mean(e1, 0)))),
                        bfm_step1=float(np.mean(bfm1)), bfm_full=float(np.mean(bfm14)),
                        bfm_full_per_clip=dict(zip(CLIPS, map(float, bfm14))), ik_mean=float(np.mean(ik)))
        print(f"{name:7s} IK mean {res[key]['ik_mean']:.1f} deg  " + "  ".join(f"{k} {v:.1f}" for k, v in res[key]["ik"].items()))
        print(f"        step 1 alone (deg): " + "  ".join(f"{k} {v:.1f}" for k, v in res[key]["step1"].items()))
        print(f"        bfm: step 1 alone {res[key]['bfm_step1']:.3f}, full round trip {res[key]['bfm_full']:.3f}", flush=True)
        (OUT / "numbers.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # 1. step 4: IK residual per paired part
    cats = ["胸", "大腿", "小腿", "腳掌", "上臂", "前臂", "手"]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
    bars(ax, cats, [(f"{n}（平均 {res[k]['ik_mean']:.1f}°）", c, res[k]["ik"]) for k, n, c in ROBOTS], "IK 之後還差的角度（度）")
    ax.set_ylim(0, 1.3 * max(max(res[k]["ik"].values()) for k, _, _ in ROBOTS))
    ax.set_title("第 4 步 IK：機器人擺不出人體要求的方向，還差幾度", fontsize=14, color=INK, loc="left", pad=26)
    ax.text(0, 1.02, "11 段問卷動作的平均，左右平均；H1 沒有手部連桿，所以「手」沒有對應", transform=ax.transAxes,
            fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "ik_step4.png", dpi=200); plt.close(fig)

    # 2. step 1 alone: only the parts it actually changes. The same on every robot except the hand (H1 has no hand link),
    #    so one grey series, plus H1's hand in its own colour
    allp = ["骨盆", "腰（下）", "腰（上）", "胸", "鎖骨", "脖子", "頭", "大腿", "小腿", "腳掌", "腳趾", "上臂", "前臂", "手腕", "手"]
    why = {"腰（下）": "平均分配", "腰（上）": "平均分配", "鎖骨": "固定姿勢", "脖子": "固定姿勢", "頭": "固定姿勢",
           "腳趾": "固定姿勢", "手腕": "設為 0", "手": "H1：固定姿勢"}
    cats = [p for p in allp if max(res[k]["step1"].get(p, 0) for k, _, _ in ROBOTS) > 0.05]
    zero = [p for p in allp if p not in cats]
    same = {p: res["g1"]["step1"][p] for p in cats if p != "手"}
    assert all(abs(res[k]["step1"][p] - v) < 1e-6 for k, _, _ in ROBOTS for p, v in same.items())
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
    x = np.arange(len(cats))
    for j, p in enumerate(cats):
        v, c = (res["h1"]["step1"][p], ROBOTS[1][2]) if p == "手" else (same[p], "#8a8984")
        ax.bar(x[j], v, 0.6, color=c, zorder=3)
        ax.text(x[j], v, f"{v:.1f}", ha="center", va="bottom", fontsize=10, color=MUTED)
    ax.set_xticks(x, [f"{p}\n{why.get(p, '')}" for p in cats], fontsize=11, color=INK)
    ax.set_ylabel("換回人體後差的角度（度）", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)
    ax.set_title("第 1 步 機器人 → 人體：就算 IK 完美，換回人體也會丟掉的部分", fontsize=14, color=INK, loc="left", pad=26)
    ax.text(0, 1.02, f"三台機器人相同，只有橘色的「手」是 H1（G1、Apollo 有手部連桿，為 0）；誤差為 0 的部位：{'、'.join(zero)}",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "backmap_step1.png", dpi=200); plt.close(fig)

    # 3. what Motivo sees: bfm of step 1 alone vs the full round trip
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURF); ax.set_facecolor(SURF)
    x = np.arange(len(ROBOTS))
    for i, (k, n, c) in enumerate(ROBOTS):
        for j, (v, a, lab) in enumerate([(res[k]["bfm_step1"], 0.4, "只有第 1 步（IK 完美）"), (res[k]["bfm_full"], 1.0, "第 4 步 + 第 1 步（完整來回）")]):
            ax.bar(x[i] - 0.2 + 0.4 * j, v, 0.37, color=c, alpha=a, zorder=3, label=lab if i == 0 else None)
            ax.text(x[i] - 0.2 + 0.4 * j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=MUTED)
    ax.set_xticks(x, [n for _, n, _ in ROBOTS], fontsize=12, color=INK)
    ax.set_ylabel("BFM（0 = Motivo 看起來完全一樣）", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, length=0)
    lg = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for h in lg.legend_handles:
        h.set_color("#8a8984")
    lg.legend_handles[0].set_alpha(0.4)
    ax.set_title("Motivo 看到的差異：大人原本的動作，經過換算後", fontsize=14, color=INK, loc="left", pad=26)
    ax.text(0, 1.02, "11 段問卷動作的平均；完整來回 = 任何控制器在這台機器人上能達到的最低 BFM", transform=ax.transAxes,
            fontsize=9.5, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "bfm_floor.png", dpi=200); plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    main()
