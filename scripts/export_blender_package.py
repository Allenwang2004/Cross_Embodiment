#!/usr/bin/env python3
"""export_blender_package.py -- the paper figures' motion panels as a folder that renders on any machine with Blender, needing nothing from this repo: outputs/blender_package/ and outputs/blender_package.zip.
scene.json holds, per panel and pose, every visible geom's world pose, shape and colour; meshes.npz the robot's meshes; render_blender.py (scripts/render_blender.py) builds and renders them. Rollouts are the same recorded ones as scripts/plot_teaser.py and scripts/plot_method_figure.py use; run this again when they change.
The results go in docs/figures/blender_renders/<panel>.png; those two scripts then draw them instead of the MuJoCo cut-outs.
Which rollouts, frames and panel sizes: configs/figures/teaser.yaml and method.yaml (OOP refactor step 10); the pose export and the
scene: crossembodiment/blender.py (BlenderScene). Inputs are read through crossembodiment.paths.read_path ($CE_OUTPUTS_ROOT), the
package is written under $CE_RESULTS_ROOT (default: outputs/).
usage: uv run python scripts/export_blender_package.py"""
import shutil, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossembodiment import paths                                         # noqa: E402
from crossembodiment.blender import BlenderScene, panel                   # noqa: E402
from crossembodiment.camera import BLENDER_AZIMUTH                        # noqa: E402

TEASER = yaml.safe_load((paths.FIGURE_CONFIGS / "teaser.yaml").read_text())
METHOD = yaml.safe_load((paths.FIGURE_CONFIGS / "method.yaml").read_text())
OUT_REL = "outputs/blender_package"


def teaser_band(sc, T, groups_cfg, xml_of):
    """the band's poses and group labels; groups_cfg: the band's groups (label, model, qpos, tag[, payload, red_leg])"""
    B = T["band"]; R = lambda p: paths.read_path(p.format(walk2=T["walk2"], walk4=T["walk4"])); poses_ = []; groups = []; n = len(B["alpha"])
    for i, g in enumerate(groups_cfg):
        kw = {k: g[k] for k in ("payload", "red_leg") if k in g}; label = g["label"]
        shift = (i - (len(groups_cfg) - 1) / 2) * B["dx"] - B["left"]; frames = [B["start"] + (i * n + k) * B["dt"] - (B["h1_back"] if label == "H1" else 0) for k in range(n)]
        groups.append(dict(label=label, x_m=shift + (n - 1) / 2 * B["pose_dx"] * np.sin(np.radians(BLENDER_AZIMUTH))))        # over the solid pose
        for po, a in zip(sc.strip(xml_of(g["model"]), R(g["qpos"]), frames, (0.0,) * n, B["pose_dx"], g["tag"], ground=True, **kw), B["alpha"]):
            po["shift"] = shift; po["alpha"] = a; poses_.append(po)
    return poses_, groups


def teaser_panels(sc, T=TEASER):
    """teaser_adult, teaser_<condition> and the band (teaser_band, teaser_band_grey) into scene `sc`"""
    P = sc.panels; R = lambda p: paths.read_path(p.format(walk2=T["walk2"], walk4=T["walk4"])); mdl = {k: paths.read_path(v) for k, v in T["models"].items()}
    U = T["units_per_m"]; dx = T["pose_dx_u"] / U; o = T["origin"]
    P["teaser_adult"] = panel("teaser", *o["panel_u"], U, sc.strip(mdl[o["model"]], R(o["qpos"]), T["frames"], T["tints"], dx, "adult"))
    for p in T["panels"]:
        kw = {k: p[k] for k in ("payload", "red_leg") if k in p}
        q = R(f"outputs/user_survey/{p['body']}/trajectories/{T[p['clip']]}/best.npz")
        P[f"teaser_{p['key']}"] = panel("teaser", *T["panel_u"], U, sc.strip(mdl[p["body"]], q, T["frames"], T["tints"], dx, p["body"], **kw))
    poses_, groups = teaser_band(sc, T, T["band"]["groups"], mdl.__getitem__)
    P["teaser_band"] = dict(panel("teaser", *T["band"]["units"], poses_, white=True), cam_distance=T["band"]["cam_distance"], groups=groups)
    P["teaser_band_grey"] = dict(P["teaser_band"], white=False)


def method_panels(sc, M=METHOD):
    """method_fall_*, method_{adult,child,g1} (the result strips) and the first layout's cut-outs method_first_* into scene `sc`"""
    P = sc.panels; R = lambda p: paths.read_path(p.format(run=M["run"])); mdl = {k: paths.read_path(v) for k, v in M["models"].items()}; B = M["blender"]
    U = M["units_per_m"]; RESULT = M["result"]; tints = tuple(0.62 * (1 - i / (len(RESULT) - 1)) for i in range(len(RESULT)))
    for f in B["fall"]:
        P[f["name"]] = panel("method", *B["fall_u"], U, sc.fall_strip(mdl[f["model"]], R(f["qpos"]), f["tag"]), white=True)
    for s in B["strips"]:
        P[s["name"]] = panel("method", *s["u"], U, sc.strip(mdl[s["model"]], R(s["qpos"]), RESULT, tints, M["pose_dx_u"] / U, s["tag"], ground=True), white=True)
    FIRST = M["first"]
    for f in B["first"]:
        for t, po in zip(FIRST, sc.strip(mdl[f["model"]], R(f["qpos"]), FIRST, (f["tint"],) * len(FIRST), 0.0, f["name"], ground=True)):
            P[f"method_first_{f['name']}_{t}"] = dict(panel("method", *B["first_u"], U, [po], white=True), cutout=True)
    # the method figure's "New body" box: each body standing still (user, 11/11), frame new_body_t of the stand-still task motion
    # move-ego-0-0_2 (the child's retargeted reference; the G1's target from the SMPL round trip), arms down, as the first layout's other cut-outs
    for s in B["stand"]:
        po = sc.strip(mdl[s["model"]], R(s["qpos"]), (M["new_body_t"],), (0.0,), 0.0, s["name"], ground=True)[0]
        P[f"method_first_{s['name']}_stand"] = dict(panel("method", *B["first_u"], U, [po], white=True), cutout=True)


def main():
    OUT = paths.write_path(OUT_REL); sc = BlenderScene()
    teaser_panels(sc); method_panels(sc)
    sc.write(OUT); shutil.copy(REPO / "scripts/blender_package_README.md", OUT / "README.md")
    shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)
    for k, v in sc.panels.items():
        print(k, "poses", len(v["poses"]), "geoms per pose", len(v["poses"][0]["geoms"]), "px", v["px"], "m", round(v["width_m"], 2), flush=True)
    print("meshes", len(sc.meshes) // 2, "scene MB", round((OUT / "scene.json").stat().st_size / 1e6, 2), "meshes MB", round((OUT / "meshes.npz").stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
