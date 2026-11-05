#!/usr/bin/env python3
"""g1_fill_compare.py -- the G1 survey searches run with the fixed fill (outputs/user_survey_search/g1) and with the
ridge fill (.../g1_ridge) measure bfm through different robot -> SMPL maps, so their own numbers are not comparable.
This re-measures the saved G1 trajectories (origin_z.npz and best.npz, robot qpos) of both searches through ONE map,
each fill in turn: bfm = 1 - mean_t cos(B(obs(map(G1 qpos))), B(obs(adult original))), obs from the qpos sequence
(bfm_align.obs_from_qpos), no simulation. Writes outputs/map_error/g1_fill_compare.json.
usage: CUDA_VISIBLE_DEVICES=2 uv run python scripts/g1_fill_compare.py
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from model.robot_map import RobotMap, SMPL_XML     # noqa: E402
from survey_body_videos import CLIPS               # noqa: E402


def main():
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from model import bfm_align
    env, _ = make_humenv(num_envs=1, task=None, xml=str(REPO / SMPL_XML), state_init="Default")
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cuda")
    emb = lambda Q: bfm_align.embed(model, bfm_align.obs_from_qpos(env, Q), "cuda")
    maps = {f: RobotMap("g1", fill=f) for f in ("fixed", "ridge")}
    res = {}
    for stem in CLIPS:
        B0 = emb(np.load(REPO / f"data/origin_motion/{stem.rsplit('_', 1)[0]}/{stem}.npz")["qpos"])
        res[stem] = {}
        for search in ("g1", "g1_ridge"):
            for roll in ("origin_z", "best"):
                qg = np.load(REPO / f"outputs/user_survey_search/{search}/{stem}_s0/{roll}.npz")["qpos"].astype(np.float64)
                for f, M in maps.items():
                    Q = np.stack([M.robot_to_smpl_qpos(q) for q in qg])
                    res[stem][f"{search}/{roll}/measured_{f}"] = bfm_align.bfm_align_loss(emb(Q), B0[:len(Q)])
        r = res[stem]
        print(f"{stem:20s} " + " | ".join(f"measured {f}: fixed-search best {r[f'g1/best/measured_{f}']:.3f}, ridge-search best "
                                          f"{r[f'g1_ridge/best/measured_{f}']:.3f}" for f in ("fixed", "ridge")), flush=True)
    for f in ("fixed", "ridge"):
        a = np.mean([res[s][f"g1/best/measured_{f}"] for s in CLIPS]); b = np.mean([res[s][f"g1_ridge/best/measured_{f}"] for s in CLIPS])
        z = np.mean([res[s][f"g1/origin_z/measured_{f}"] for s in CLIPS])
        print(f"mean, measured with the {f} fill: z0 {z:.3f} | fixed-fill search best {a:.3f} | ridge-fill search best {b:.3f}")
    (REPO / "outputs/map_error/g1_fill_compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
