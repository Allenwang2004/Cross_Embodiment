#!/usr/bin/env python3
"""verify_vis.py LABEL -- OOP refactor step 10: the visualisation layer (crossembodiment palette / figstyle / camera / clips / blender /
strobe / video / sheet and the scripts on them) re-makes the stored scenes, trajectories, pages, figures, videos and renders.

Everything is WRITTEN under outputs/refactor_step10/LABEL: the new code's outputs in new/ ($CE_RESULTS_ROOT = new/outputs,
$CE_FIGURES_ROOT = new/figures, $CE_OUTPUTS_ROOT = the main tree's outputs, READ ONLY), the old code's in oldrepo/ (a scratch
repository with BASE's scripts, crossembodiment, model and configs; its inputs are read-only symlinks into the main tree / the worktree,
and every folder those scripts write is a real folder of the scratch root).

  exports   new vs the main tree's stored files: scene.json (JSON equal, the new "version" field ignored) + meshes.npz (every array) of
            strobe_task52_blender (all examples, `strobe_final_blender.py scene`), blender_package, topdown_16384_noanchor
            (--ours outputs/child_moveego04_2048/move-ego-0-4_9_es_s0_noanchor_16384 --tag 16384_noanchor --layout row --panel-px 1500
            --poses 6 --uniform-color), walk_bodies/band, walk_bodies/blender, teaser_band_tall, child_all52, limit_leg_all52/blender;
            the trajectories (ClipBundle) of child_all52 / limit_leg_all52 and of every strobe example whose sources still exist (npz
            arrays, video.json bytes), plan.json / the recipe configs (bytes); the old code on the same inputs against the same files
  pages     new vs stored: child_all52 / limit_leg_all52 overview_*.jpg + index.md, the strobe combined/*.png, the top view row,
            walk_band(_labeled).png, walk_bodies_rows.png, child_all52 topview_*.jpg + topview.md (bytes, or pixels)
  figures   new vs old code: plot_method_figure draw / simple / first, plot_teaser draw / band / band_grey, teaser_band_tall draw,
            plot_child_improvement (default; --metric phc --short --rot 90 --loes-dir outputs/noanchor/child),
            plot_child_convergence --loes-dir outputs/noanchor/child (+ its outputs/child_convergence_noanchor): PNG bytes, PDF bytes
            but /CreationDate, SVG bytes but <dc:date> and matplotlib's random ids (svg.hashsalt unset: uuid4 per process)
  videos    render_panels.py, old vs new, on 30-frame cuts of two stored specs (outputs/tall_body_52/videos/_specs/move-ego-low-0-2_9.json:
            captions; outputs/survey_preview/spec_child_move-ego-0-2_2.json: no captions). MuJoCo's GPU (EGL) frames are not
            bit-reproducible here (two renders of one scene differ by 1 level on a few thousand pixels), so: on the GPU, old vs new
            within old vs old; through tests/golden/fake_render.py (rasterisation replaced by a picture of the scene digest), decoded
            frames np.array_equal, mp4 and contact sheet bytes
  render    two small Blender panels (blender_package method_first_child_stand at --scale 0.25, strobe_task52_blender g1_jump-2_3_cem
            (the G1's meshes, drawn black) at --scale 0.2; --samples 16 --threads 16, CPU, one render at a time): the old scene + old
            render_blender.py twice (Cycles' run-to-run determinism) and the new scene + new render_blender.py: pixels
Writes WORK/report.json; exit 0 = every comparison identical.
usage: CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl uv run python tests/golden/verify_vis.py step10 [--parts exports,pages,figures,videos,render]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("CROSSENBODIMENT_MAIN", "/home/allen19/crossembodiment"))
BASE = "8e9dc99"                                              # the commit before step 10 (the old scripts)
PY = sys.executable
ENV0 = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="2", CUDA_VISIBLE_DEVICES="2", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
for k in ("CE_OUTPUTS_ROOT", "CE_RESULTS_ROOT", "CE_FIGURES_ROOT"):
    ENV0.pop(k, None)
TOPDOWN = ["--ours", "outputs/child_moveego04_2048/move-ego-0-4_9_es_s0_noanchor_16384", "--tag", "16384_noanchor", "--layout", "row",
           "--panel-px", "1500", "--poses", "6", "--uniform-color"]
SCENES = ("strobe_task52_blender", "blender_package", "strobe_task52_blender/topdown_16384_noanchor", "walk_bodies/band", "walk_bodies/blender",
          "teaser_band_tall", "child_all52", "limit_leg_all52/blender")
OLD_INPUTS = ("user_survey", "user_survey_search", "g1", "child_52_rerun", "child_52_cem_phc", "noanchor", "child_moveego04_2048",
              "tall_body_52", "child_kpkd_headstand", "tall_kpkd_180", "body_52", "g1_52")
SPECS = ("outputs/tall_body_52/videos/_specs/move-ego-low-0-2_9.json", "outputs/survey_preview/spec_child_move-ego-0-2_2.json")
VIDEO_FRAMES = 30
RENDERS = (("blender_package", "method_first_child_stand", ["--scale", "0.25", "--samples", "16", "--threads", "16", "--device", "CPU"]),
           ("strobe_task52_blender", "g1_jump-2_3_cem", ["--scale", "0.2", "--samples", "16", "--threads", "16", "--device", "CPU"]))


class Report:
    def __init__(self):
        self.rows = []

    def add(self, part, what, ok, note=""):
        self.rows.append(dict(part=part, what=what, ok=bool(ok), note=note))
        print(f"  {'PASS' if ok else 'FAIL'}  {part:8s} {what}" + (f"  ({note})" if note else ""), flush=True)

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)


def run(cmd, cwd, env, log):
    with open(log, "a") as f:
        f.write(f"$ (cd {cwd}) {' '.join(map(str, cmd))}\n"); f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{cmd[:3]} failed ({r.returncode}); see {log}")


def new_env(work):
    return dict(ENV0, CE_OUTPUTS_ROOT=str(MAIN / "outputs"), CE_RESULTS_ROOT=str(work / "new/outputs"), CE_FIGURES_ROOT=str(work / "new/figures"))


# ------------------------------------------------------------------ comparisons
def same_npz(a, b):
    A, B = np.load(a), np.load(b)
    return list(A.keys()) == list(B.keys()) and all(A[k].dtype == B[k].dtype and A[k].shape == B[k].shape and A[k].tobytes() == B[k].tobytes() for k in A)


def same_meshes(a, b):
    A, B = np.load(a), np.load(b)
    return sorted(A.keys()) == sorted(B.keys()) and all(A[k].dtype == B[k].dtype and A[k].shape == B[k].shape and A[k].tobytes() == B[k].tobytes() for k in A)


def same_scene(a, b):
    """scene.json equal as JSON with the version field dropped from both; meshes.npz the same arrays"""
    A, B = json.loads((Path(a) / "scene.json").read_text()), json.loads((Path(b) / "scene.json").read_text())
    va, vb = A.pop("version", None), B.pop("version", None)
    return A == B and same_meshes(Path(a) / "meshes.npz", Path(b) / "meshes.npz"), f"version {va} / {vb}, {len(A.get('panels', {}))} panels"


def same_folder(new, stored):
    """every file under `stored` in `new` (npz: arrays; else bytes) and the same file set"""
    new, stored = Path(new), Path(stored)
    fs = sorted(p.relative_to(stored) for p in stored.rglob("*") if p.is_file()); fn = sorted(p.relative_to(new) for p in new.rglob("*") if p.is_file())
    bad = [str(p) for p in fs if not (new / p).exists() or not (same_npz(new / p, stored / p) if p.suffix == ".npz" else (new / p).read_bytes() == (stored / p).read_bytes())]
    return not bad and fs == fn, f"{len(fs)} files" + (f", {len(bad)} differ: {bad[:3]}" if bad else "") + ("" if fs == fn else ", file sets differ")


def same_image(a, b):
    from PIL import Image
    A, B = Path(a).read_bytes(), Path(b).read_bytes()
    if A == B:
        return True, "bytes"
    return np.array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(b))), "pixels (bytes differ)"


SVG_IDS = re.compile(rb"\b(p|m|h|image)[0-9a-f]{10}\b")


def same_figure(a, b):
    A, B = Path(a).read_bytes(), Path(b).read_bytes()
    if A == B:
        return True, "bytes"
    sfx = Path(a).suffix
    if sfx == ".png":
        return same_image(a, b)
    if sfx == ".pdf":
        s = lambda x: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", x)
        return s(A) == s(B), "bytes but /CreationDate"
    if sfx == ".svg":
        s = lambda x: SVG_IDS.sub(rb"\1ID", re.sub(rb"<dc:date>[^<]*</dc:date>", b"", x))
        return s(A) == s(B), "bytes but <dc:date> and matplotlib's random ids"
    return False, "differ"


# ------------------------------------------------------------------ the old code's scratch root
def old_repo(work):
    root = work / "oldrepo"
    if not (root / "scripts").exists():
        root.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "archive", BASE, "scripts", "crossembodiment", "model", "configs"], cwd=REPO, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(root)], input=tar, check=True)
    links = {"assets": REPO / "assets", "data": MAIN / "data", "amass_data": MAIN / "amass_data"}
    links.update({f"outputs/{o}": MAIN / "outputs" / o for o in OLD_INPUTS})
    links["outputs/strobe_task52_blender/trajectories"] = MAIN / "outputs/strobe_task52_blender/trajectories"
    links.update({f"outputs/walk_bodies/{p.name}": p for p in (MAIN / "outputs/walk_bodies").glob("*_es_s0")})
    links.update({f"docs/figures/{d}": REPO / "docs/figures" / d for d in ("blender_renders", "method_parts", "teaser_parts")})
    for rel, target in links.items():
        link = root / rel; link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(target)
    return root


def old_exports(root, log):
    env = ENV0; O = root / "outputs/strobe_task52_blender"
    (O / "scene.json").write_text(json.dumps(dict(panels={}))); np.savez_compressed(O / "meshes.npz")
    run([PY, "-c", "import strobe_final_blender as F; F.export([e[0] for e in F.EX])"], root / "scripts", env, log)
    for cmd in (["scripts/export_blender_package.py"], ["scripts/topdown_blender.py", "export"] + TOPDOWN, ["scripts/walk_band_blender.py", "export"],
                ["scripts/walk_bodies_blender.py", "export"], ["scripts/teaser_band_tall.py", "export"], ["scripts/child_all_blender.py", "export"],
                ["scripts/body_all_blender.py", "limit_leg", "export"]):
        run([PY] + cmd, root, env, log)


# ------------------------------------------------------------------ parts
def exports(rep, work, log):
    env = new_env(work); N = work / "new/outputs"; M = MAIN / "outputs"
    for cmd in (["scripts/strobe_final_blender.py", "scene"], ["scripts/export_blender_package.py"], ["scripts/topdown_blender.py", "export"] + TOPDOWN,
                ["scripts/walk_band_blender.py", "export"], ["scripts/walk_bodies_blender.py", "export"], ["scripts/teaser_band_tall.py", "export"],
                ["scripts/child_all_blender.py", "export"], ["scripts/body_all_blender.py", "limit_leg", "export"]):
        t0 = time.time(); run([PY] + cmd, REPO, env, log); print(f"    new {' '.join(cmd[:2])}: {time.time() - t0:.0f} s", flush=True)
    for s in SCENES:
        ok, note = same_scene(N / s, M / s); rep.add("exports", f"new {s}: scene.json + meshes.npz = stored", ok, note)
    for t in ("child_all52/trajectories", "limit_leg_all52/trajectories", "limit_leg_all52/configs"):
        ok, note = same_folder(N / t, M / t); rep.add("exports", f"new {t} = stored (ClipBundle)", ok, note)
    for f in ("child_all52/plan.json", "limit_leg_all52/plan.json"):
        rep.add("exports", f"new {f} = stored (bytes)", (N / f).read_bytes() == (M / f).read_bytes())
    # the strobe examples' trajectories, re-made by ClipBundle (strobe_final_blender.trajectories + add_cem) where the sources still exist
    cfg = __import__("yaml").safe_load((REPO / "configs/figures/strobe_examples.yaml").read_text())
    have = [e["name"] for e in cfg["examples"] if (not e["source"].startswith("outputs/") or (M / e["source"][len("outputs/"):] / e["clip"]).exists())
            and (M / cfg["cem"][e["name"]][len("outputs/"):]).exists()]
    gone = [e["name"] for e in cfg["examples"] if e["name"] not in have]
    run([PY, "-c", "import sys, strobe_final_blender as F\nfor n, b, c, s, *_ in F.EX:\n    if n in sys.argv[1:]: F.trajectories(n, b, c, s); F.add_cem(n)", *have],
        REPO / "scripts", env, log)
    for n in have:
        ok, note = same_folder(N / "strobe_task52_blender/trajectories" / n, M / "strobe_task52_blender/trajectories" / n)
        rep.add("exports", f"new strobe trajectories/{n} = stored (ClipBundle)", ok, note)
    if gone:
        print(f"  note     strobe examples whose sources were deleted (not re-made): {gone}")
    # the old code on the same inputs (it made the stored files)
    root = old_repo(work); t0 = time.time(); old_exports(root, log); print(f"    old exports: {time.time() - t0:.0f} s", flush=True)
    for s in SCENES:
        ok, note = same_scene(root / "outputs" / s, M / s); rep.add("exports", f"old {s}: scene.json + meshes.npz = stored", ok, note)


def pages(rep, work, log):
    env = new_env(work); N = work / "new/outputs"; M = MAIN / "outputs"
    run([PY, "scripts/child_all_blender.py", "pages"], REPO, env, log); run([PY, "scripts/body_all_blender.py", "limit_leg", "pages"], REPO, env, log)
    for d in ("child_all52", "limit_leg_all52"):
        for p in sorted((M / d).glob("overview_*.jpg")):
            ok, note = same_image(N / d / p.name, p); rep.add("pages", f"new {d}/{p.name} = stored", ok, note)
        rep.add("pages", f"new {d}/index.md = stored (bytes)", (N / d / "index.md").read_bytes() == (M / d / "index.md").read_bytes())
    run([PY, "scripts/child_topview.py"], REPO, env, log)
    for p in sorted((M / "child_all52").glob("topview*.jpg")) + [M / "child_all52/topview.md"]:
        ok, note = same_image(N / "child_all52" / p.name, p) if p.suffix == ".jpg" else ((N / "child_all52" / p.name).read_bytes() == p.read_bytes(), "bytes")
        rep.add("pages", f"new child_all52/{p.name} = stored (child_topview.py)", ok, note)
    run([PY, "-c", f"import strobe_final_blender as F; F.combine(panels={str(M / 'strobe_task52_blender/panels')!r})"], REPO / "scripts", env, log)
    for p in sorted((N / "strobe_task52_blender/combined").glob("*.png")):
        s = M / "strobe_task52_blender/combined" / p.name
        if s.exists():
            ok, note = same_image(p, s); rep.add("pages", f"new strobe combined/{p.name} = stored", ok, note)
        else:
            print(f"  note     strobe combined/{p.name}: no stored copy")
    run([PY, "scripts/topdown_blender.py", "combine"] + TOPDOWN, REPO, env, log)
    for p in sorted((N / "strobe_task52_blender/combined").glob("*topdown*.png")) + sorted((N / "strobe_task52_blender/panels").glob("*topdown*.png")):
        s = M / "strobe_task52_blender" / p.parent.name / p.name
        if s.exists():
            ok, note = same_image(p, s); rep.add("pages", f"new strobe {p.parent.name}/{p.name} (topdown combine) = stored", ok, note)
        else:
            print(f"  note     strobe {p.parent.name}/{p.name}: no stored copy")
    run([PY, "scripts/walk_band_blender.py", "combine"], REPO, env, log); run([PY, "scripts/walk_bodies_blender.py", "combine"], REPO, env, log)
    for f in ("walk_band.png", "walk_band_labeled.png", "walk_bodies_rows.png"):
        ok, note = same_image(N / "walk_bodies" / f, M / "walk_bodies" / f); rep.add("pages", f"new walk_bodies/{f} = stored", ok, note)


FIGURE_CMDS = (["scripts/plot_method_figure.py", "draw"], ["scripts/plot_method_figure.py", "simple"], ["scripts/plot_method_figure.py", "first"],
               ["scripts/plot_teaser.py", "draw"], ["scripts/plot_teaser.py", "band"], ["scripts/plot_teaser.py", "band_grey"],
               ["scripts/teaser_band_tall.py", "draw"], ["scripts/plot_child_improvement.py"],
               ["scripts/plot_child_improvement.py", "--metric", "phc", "--short", "--rot", "90", "--loes-dir", "outputs/noanchor/child"],
               ["scripts/plot_child_convergence.py", "--loes-dir", "outputs/noanchor/child"])


def figures(rep, work, log):
    for f in ("child_52_rerun", "noanchor/child"):                # the PHC cache must exist: the old code would write it into the main tree
        assert (MAIN / "outputs" / f / "phc_reward.json").exists(), f
    root = old_repo(work); env = new_env(work); (work / "new/figures").mkdir(parents=True, exist_ok=True)
    if not (root / "outputs/blender_package/scene.json").exists() or not (root / "outputs/teaser_band_tall/scene.json").exists():
        old_exports(root, log)
    if not (work / "new/outputs/teaser_band_tall/scene.json").exists():
        run([PY, "scripts/teaser_band_tall.py", "export"], REPO, env, log)
    for cmd in FIGURE_CMDS:
        run([PY] + cmd, root, ENV0, log); run([PY] + cmd, REPO, env, log)
    old, new = root / "docs/figures", work / "new/figures"
    for p in sorted(old.glob("*.*")):
        ok, note = same_figure(new / p.name, p) if (new / p.name).exists() else (False, "missing")
        rep.add("figures", f"docs/figures/{p.name}: new = old code", ok, note)
    o, n = root / "outputs/child_convergence_noanchor", work / "new/outputs/child_convergence_noanchor"
    for p in sorted(o.iterdir()):
        ok, note = (same_npz(n / p.name, p), "arrays") if p.suffix == ".npz" else same_figure(n / p.name, p)
        rep.add("figures", f"outputs/child_convergence_noanchor/{p.name}: new = old code", ok, note)


def videos(rep, work, log):
    """GPU (MuJoCo EGL) renders are not bit-reproducible run to run, so each spec is rendered (1) on the GPU twice by the old code and
    once by the new (old vs new may differ only as much as old vs old) and (2) through tests/golden/fake_render.py (the GPU
    rasterisation replaced by a picture of the scene digest), old vs new: decoded frames, mp4 bytes and the contact sheet exactly equal"""
    import imageio.v2 as imageio
    root = old_repo(work); env = new_env(work); d = work / "videos"; d.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO)); from crossembodiment import paths
    os.environ["CE_OUTPUTS_ROOT"] = str(MAIN / "outputs"); fake = REPO / "tests/golden/fake_render.py"
    frames = lambda f: imageio.mimread(f, memtest=False)
    for i, rel in enumerate(SPECS):
        spec = json.loads(paths.read_path(rel).read_text()); name = Path(rel).name
        for j, p in enumerate(spec["panels"]):                     # 30-frame cuts of every panel's trajectory, absolute paths
            q = np.load(paths.read_path(p["qpos"]))["qpos"][:VIDEO_FRAMES]; f = d / f"spec{i}_panel{j}.npz"; np.savez(f, qpos=q); p["qpos"] = str(f)
        for who, cwd, e, pre in (("old", root, ENV0, []), ("old2", root, ENV0, []), ("new", REPO, env, []),
                                 ("fake_old", root, ENV0, [fake]), ("fake_new", REPO, env, [fake])):
            s = dict(spec, out=str(d / f"spec{i}_{who}.mp4")); sp = d / f"spec{i}_{who}.json"; sp.write_text(json.dumps(s, indent=1))
            run([PY] + pre + ["scripts/render_panels.py", sp], cwd, e, log)
        F = {w: frames(d / f"spec{i}_{w}.mp4") for w in ("old", "old2", "new", "fake_old", "fake_new")}
        dev = lambda a, b: (max(int(np.abs(x.astype(int) - y.astype(int)).max()) for x, y in zip(F[a], F[b])), sum(int((x != y).any(-1).sum()) for x, y in zip(F[a], F[b])))
        (m_oo, n_oo), (m_on, n_on) = dev("old", "old2"), dev("old", "new")
        from PIL import Image                                       # the contact sheets: raw rendered frames (3 moments), no encoder
        S = {w: np.asarray(Image.open(d / f"spec{i}_{w}.mp4.png")).astype(int) for w in ("old", "old2", "new")}
        s_oo, s_on = int(np.abs(S["old"] - S["old2"]).max()), int(np.abs(S["old"] - S["new"]).max())
        rep.add("videos", f"render_panels {name} on the GPU ({VIDEO_FRAMES} frames, {F['old'][0].shape[1]}x{F['old'][0].shape[0]}): old vs new within "
                f"the GPU's run-to-run noise", len(F["new"]) == VIDEO_FRAMES and m_on <= max(m_oo, 1) and s_on <= max(s_oo, 1),
                f"raw frames (contact sheet): old vs old max |diff| {s_oo}, old vs new {s_on}; decoded mp4: old vs old max {m_oo} on {n_oo} "
                f"pixels, old vs new max {m_on} on {n_on} (the encoder amplifies the 1-level render noise)")
        same = len(F["fake_old"]) == len(F["fake_new"]) == VIDEO_FRAMES and all(np.array_equal(x, y) for x, y in zip(F["fake_old"], F["fake_new"]))
        a, b = d / f"spec{i}_fake_old.mp4", d / f"spec{i}_fake_new.mp4"
        rep.add("videos", f"render_panels {name}, the scene each frame draws (fake_render.py): decoded frames old = new", same,
                f"mp4 bytes {'equal' if a.read_bytes() == b.read_bytes() else 'differ'}")
        ok, note = same_image(Path(str(b) + ".png"), Path(str(a) + ".png")); rep.add("videos", f"render_panels {name} (fake_render.py): contact sheet old = new", ok, note)


def render(rep, work, log):
    """the old scene + old render_blender.py twice (Cycles' run-to-run determinism) and the new scene + the new render_blender.py"""
    from PIL import Image
    root = old_repo(work); env = new_env(work)
    if not (root / "outputs/blender_package/scene.json").exists() or not (root / "outputs/strobe_task52_blender/render_blender.py").exists():
        old_exports(root, log)
    if not (work / "new/outputs/blender_package/scene.json").exists():
        run([PY, "scripts/export_blender_package.py"], REPO, env, log)
    if not (work / "new/outputs/strobe_task52_blender/scene.json").exists():
        run([PY, "scripts/strobe_final_blender.py", "scene"], REPO, env, log)
    for folder, panel, args in RENDERS:
        out = work / "render" / panel; out.mkdir(parents=True, exist_ok=True); ims = {}
        bl = ["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--", "--panel", panel] + args
        for who, f in (("old1", root / "outputs" / folder), ("old2", root / "outputs" / folder), ("new", work / "new/outputs" / folder)):
            t0 = time.time(); run(bl + ["--out", out / who], f, dict(ENV0, OMP_NUM_THREADS="1"), log)
            ims[who] = np.asarray(Image.open(out / who / f"{panel}.png")).astype(int); print(f"    render {panel} {who}: {time.time() - t0:.0f} s, {ims[who].shape}", flush=True)
        det = np.array_equal(ims["old1"], ims["old2"]); same = np.array_equal(ims["old1"], ims["new"])
        rep.add("render", f"Cycles run to run: two old renders of {folder} {panel} pixel-equal", det,
                "" if det else f"max |diff| {int(np.abs(ims['old1'] - ims['old2']).max())}")
        rep.add("render", f"{folder} {panel} ({' '.join(args)}, {ims['new'].shape[1]}x{ims['new'].shape[0]}): new scene + render_blender.py = old, pixels",
                same, f"max |diff| {int(np.abs(ims['old1'] - ims['new']).max())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--parts", default="exports,pages,figures,videos,render")
    a = ap.parse_args()
    work = REPO / "outputs/refactor_step10" / a.label; work.mkdir(parents=True, exist_ok=True); log = work / "verify.log"
    rep = Report(); parts = a.parts.split(",")
    for name in ("exports", "pages", "figures", "videos", "render"):
        if name in parts:
            print(f"== {name}", flush=True); t0 = time.time(); globals()[name](rep, work, log); print(f"   ({time.time() - t0:.0f} s)", flush=True)
    (work / "report.json").write_text(json.dumps(rep.rows, indent=1) + "\n")
    n = len(rep.rows); k = sum(r["ok"] for r in rep.rows)
    print(f"\n{k}/{n} identical -> {work / 'report.json'}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
