"""Tests of OOP refactor step 10 (docs/refactor/PLAN.md): the visualisation layer against its 8e9dc99 sources, exact comparisons.

  palette     render_blender.py's COLORS = palette.BLENDER; every script's colour / label table equals its BASE literal (the scripts that
              import safely are imported, the others' values compared through the palette); the forms of one colour agree within
              their rounding
  figstyle    txt / arrow / box and the canvas vs the teaser's / method figure's closures (SVG bytes, svg.hashsalt fixed); rc_context
              restores rcParams
  camera      project / project_point vs draw.project and robot_overlay.py's two inline copies (fake GL camera); the presets vs the
              scripts' expressions
  strobe      StrobeLayout headings / tints / shifts vs strobe_final_blender.py's and export_strobe_blender.py's loops; stretches,
              around_fall, whole_clip vs child_all_blender.py
  blender     pose_spec vs export_blender_package.pose (meshes store), BlenderScene write / load / render argv
  clips       strobe_source vs strobe_final_blender.trajectories' routing; ClipBundle.from_run / add_cem vs the old code on synthetic
              clips (arrays, video.json text)
  sheet       ContactSheet / hcat / write_index_md vs export_strobe_blender.overview, strobe_final_blender.combine, child_all_blender's
              index (JPEG / PNG bytes, text)
  video       label_px / legend vs render_panels.py / robot_overlay.py; VideoRenderer side by side and overlay vs the survey's
              replay.py (frames np.array_equal, 6 frames at 240 px on GPU 2)
Run standalone (no pytest in the environment), like tests/test_moves.py:
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl uv run python tests/test_vis.py
"""
import ast
import io
import json
import math
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "scripts"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

BASE = "8e9dc99"                                              # the commit before step 10
MAIN = Path(os.environ.get("CROSSENBODIMENT_MAIN", "/home/allen19/crossenbodiment"))   # the main tree (read only)
WORK = REPO / "outputs/refactor_tests/test_vis"
RNG = lambda k=0: np.random.default_rng(20261114 + k)        # noqa: E731


def old_source(rel):
    return subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def _lit(node):
    try:
        return ast.literal_eval(node)
    except ValueError:                                        # dict(...) spellings
        return eval(compile(ast.Expression(node), "literal", "eval"), {"__builtins__": {}, "dict": dict})


def old_literals(rel, names):
    """module-level constants `names` of file `rel` at BASE (ast.literal_eval of their assignments; dict(...) allowed)"""
    out = {}
    for node in ast.parse(old_source(rel)).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = _lit(node.value)
                elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for n, v in zip(t.elts, node.value.elts):
                        if isinstance(n, ast.Name) and n.id in names:
                            out[n.id] = _lit(v)
    assert set(out) == set(names), (rel, set(names) - set(out))
    return out


def old_defs(rel, names, ns=None, inner=None):
    """top-level definitions `names` of `rel` at BASE executed into a namespace; inner: {function: [nested function names]} to lift"""
    ns = dict(ns or {}); ns.setdefault("__name__", f"old_{Path(rel).stem}")
    tree = ast.parse(old_source(rel)); found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(compile(ast.Module([node], []), f"{BASE}:{rel}", "exec"), ns); found.add(node.name)
        if inner and isinstance(node, ast.FunctionDef) and node.name in inner:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in inner[node.name]:
                    exec(compile(ast.Module([sub], []), f"{BASE}:{rel}", "exec"), ns); found.add(sub.name)
    want = set(names) | {n for v in (inner or {}).values() for n in v}
    assert found == want, f"{rel}: not found at {BASE}: {sorted(want - found)}"
    return ns


def import_script(name, argv=None):
    """a script imported fresh as a module, with sys.argv = [name] + argv while it loads"""
    import importlib
    saved = sys.argv; sys.argv = [name] + list(argv or [])
    try:
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        sys.argv = saved


# ------------------------------------------------------------------ palette

def test_palette_render_blender_colors():
    from crossenbodiment.palette import BLENDER, BLENDER_BG
    tree = ast.parse((REPO / "scripts/render_blender.py").read_text()); colors = {}; bg = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "COLORS":
                colors.update({k.arg: ast.literal_eval(k.value) for k in node.value.keywords})
            if node.targets[0].id == "BG":
                bg = ast.literal_eval(node.value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "attr", "") == "update":
            colors.update({k.arg: ast.literal_eval(k.value) for k in node.value.keywords})
    assert colors == dict(BLENDER), (colors, dict(BLENDER))
    assert bg == BLENDER_BG


def test_palette_forms_agree():
    """the hex, video RGBA and Blender sRGB forms of one entry are the same colour within the rounding their scripts used"""
    from crossenbodiment.palette import METHODS, VARIANTS, hex_rgb
    for m in list(METHODS.values()) + list(VARIANTS.values()):
        if m.hex and m.blender and m.blender != "body":
            assert np.abs(np.array(m.blender_rgb) - np.array(hex_rgb(m.hex)) / 255).max() <= 0.0051, m
        if m.hex and m.video is not None and m.key in ("cem", "loes") and m.video in ([0.11, 0.69, 0.48, 1.0], [0.16, 0.47, 0.84, 1.0]):
            assert np.abs(np.array(m.video[:3]) - np.array(hex_rgb(m.hex)) / 255).max() <= 0.0051, m


def test_palette_script_tables():
    """every script's colours / labels as computed now = its 8e9dc99 literals"""
    from crossenbodiment import palette as P
    old = old_literals("scripts/plot_child_convergence.py", ["METHODS"])
    assert import_script("plot_child_convergence").METHODS == old["METHODS"]
    old = old_literals("scripts/child_topview.py", ["LINES"])
    assert import_script("child_topview").LINES == old["LINES"]
    old = old_literals("scripts/topdown_blender.py", ["BODIES", "RGB"])
    td = import_script("topdown_blender")
    assert td.BODIES == old["BODIES"] and td.RGB == old["RGB"], (td.BODIES, td.RGB)
    old = old_literals("scripts/plot_algo_compare.py", ["INK", "MUTED", "GRID", "SURF", "ALGOS"])
    ac = import_script("plot_algo_compare")
    assert (ac.INK, ac.MUTED, ac.GRID, ac.SURF, ac.ALGOS) == (old["INK"], old["MUTED"], old["GRID"], old["SURF"], old["ALGOS"])
    old = old_literals("scripts/plot_k_scale_bars.py", ["INK", "MUTED", "GRID", "SURF", "C_Z0", "C_BEST"])   # runs at import: values via the palette
    assert (P.INK["dataviz"]["ink"], P.INK["dataviz"]["muted"], P.INK["dataviz"]["grid"], P.INK["dataviz"]["surface"],
            P.Palette.hex("z0", "k_scale"), P.Palette.hex("loes", "k_scale")) == tuple(old[k] for k in ("INK", "MUTED", "GRID", "SURF", "C_Z0", "C_BEST"))
    old = old_literals("scripts/plot_teaser.py", ["FRAMES", "INK", "MUTED", "RULE", "AMBER", "RED", "BLUE", "GREEN", "U_PER_M", "WALK2", "WALK4"])
    t = import_script("plot_teaser")
    assert (t.FRAMES, t.INK_, t.MUTED, t.RULE, t.AMBER, t.RED, t.U_PER_M, t.WALK2, t.WALK4) == tuple(old[k] for k in ("FRAMES", "INK", "MUTED", "RULE", "AMBER", "RED", "U_PER_M", "WALK2", "WALK4"))
    W2, W4 = old["WALK2"], old["WALK4"]
    assert t.PANELS == [("payload", "Payload (+38 kg)", "heavy_arm", W4, old["AMBER"]), ("joint", "Restricted joint", "limit_leg", W2, old["RED"]),
                        ("child", "Reshaped body", "child", W2, old["BLUE"]), ("g1", "Robot (G1)", "g1", W2, old["GREEN"])], t.PANELS
    old = old_literals("scripts/plot_method_figure.py", ["INK", "MUTED", "RULE", "FROZEN", "ACCENT", "ACCENT_BG", "U_PER_M", "RUN", "STAND", "RESULT"])
    mf = import_script("plot_method_figure")
    assert tuple(getattr(mf, k) for k in old) == tuple(old.values())
    o = ast.parse(old_source("scripts/plot_method_figure.py")); RUN = old["RUN"]
    assert mf.SOURCES == {"adult": dict(xml="assets/robots/adult/robot.xml", qpos=f"outputs/user_survey/child/trajectories/{RUN}/reference_adult.npz"),
                          "child": dict(xml="assets/robots_torque/child/robot_torque_full.xml", qpos=f"outputs/user_survey/child/trajectories/{RUN}/best.npz"),
                          "g1": dict(xml="assets/robots/g1/scene.xml", qpos=f"outputs/user_survey/g1/trajectories/{RUN}/best.npz")} and o
    old = old_literals("scripts/plot_child_improvement.py", ["BLUE", "TINT", "INK", "MUTED", "GRID"])
    ci = import_script("plot_child_improvement")
    assert tuple(getattr(ci, k) for k in old) == tuple(old.values())
    old = old_literals("scripts/package_survey.py", ["GHOST", "RED", "BLUE"])
    assert (P.GHOST, P.RED, P.BLUE) == (old["GHOST"], old["RED"], old["BLUE"])
    old = old_literals("scripts/package_amass_lso_survey.py", ["GREY", "GHOST", "RED", "BLUE"])
    assert (P.GREY, P.GHOST, P.RED, P.BLUE) == (old["GREY"], old["GHOST"], old["RED"], old["BLUE"])
    old = old_literals("scripts/robot_overlay.py", ["COLORS", "W", "H"])
    assert P.OVERLAY == old["COLORS"] and type(P.OVERLAY[0]) is tuple
    from crossenbodiment.video import OVERLAY_SIZE
    assert OVERLAY_SIZE == (old["W"], old["H"])
    old = old_literals("scripts/figure_bodies.py", ["BG", "BODY_RGBA"])
    fb = import_script("figure_bodies")
    assert (fb.BG, fb.BODY_RGBA) == (old["BG"], old["BODY_RGBA"])
    old = old_literals("scripts/child_all_blender.py", ["LABELS"])
    assert import_script("child_all_blender").LABELS == old["LABELS"]


def test_strobe_examples_yaml():
    """configs/figures/strobe_examples.yaml = strobe_final_blender.py's EX / CEM / MODEL at BASE (values and types)"""
    src = old_source("scripts/strobe_final_blender.py"); tree = ast.parse(src); old = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("EX", "CEM"):
            old[node.targets[0].id] = eval(compile(ast.Expression(node.value), "EX", "eval"), {"dict": dict})
    F = import_script("strobe_final_blender")
    assert len(F.EX) == len(old["EX"])
    for a, b in zip(F.EX, old["EX"]):
        assert a[:4] == b[:4] and a[5] == b[5] and a[6] == b[6] and all(type(a[6][k]) is type(b[6][k]) for k in b[6]), (a, b)
        assert (a[4] is None and b[4] is None) or tuple(a[4]) == tuple(b[4]), (a, b)
    assert F.CEM == old["CEM"]
    S = MAIN / "outputs/user_survey"
    assert {k: v.resolve() for k, v in F.MODEL.items()} == {k: v.resolve() for k, v in dict(
        payload=S / "models/heavy_arm.xml", limit_leg=S / "models/limit_leg.xml", g1=S / "models/g1/scene.xml", child=S / "models/child.xml",
        tall=REPO / "assets/robots_torque/len1.4_wid0.8/robot_torque_full.xml", adult=S / "models/adult.xml").items()}


# ------------------------------------------------------------------ figstyle

def _svg(fig):
    b = io.BytesIO(); fig.savefig(b, format="svg", metadata={"Date": None}); return b.getvalue()


def test_figstyle_helpers_match_closures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from functools import partial
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from crossenbodiment.figstyle import FigureStyle, arrow, box, txt
    INK = "#22252a"; FS = 9.0

    def old_fig():
        plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"})
        W, Y0, Y1 = 550, 6, 243; fig = plt.figure(figsize=(W / 100, (Y1 - Y0) / 100)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Y0, Y1); ax.axis("off")

        def t(x, y, s, size=FS, color=INK, ha="center", va="center", **k):
            ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=6, **k)

        def a(pts, color=INK, lw=0.9, ls="-", head=True):
            for p, q in zip(pts[:-2], pts[1:-1]):
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=3)
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=7, lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0, zorder=3))

        def a2(pts, color=INK, lw=0.9, style="-|>"):
            for p, q in zip(pts[:-2], pts[1:-1]):
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle="round", zorder=3)
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=7, lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=6))

        def b(x0, y0, x1, y1, ec=INK, lw=0.9, z=1, r=6):
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec=ec, lw=lw, zorder=z))
        return fig, t, a, a2, b

    def new_fig():
        FigureStyle.paper().apply(); fig, ax = FigureStyle.canvas(550, 243, 6)
        t = partial(txt, ax, size=FS, color=INK, zorder=6)

        def a(pts, color=INK, lw=0.9, ls="-", head=True):
            arrow(ax, pts, color, lw, ls=ls, style="-|>" if head else "-")

        def a2(pts, color=INK, lw=0.9, style="-|>"):
            arrow(ax, pts, color, lw, style=style, head_z=6)
        return fig, t, a, a2, partial(box, ax, ec=INK)

    out = []
    for make in (old_fig, new_fig):
        plt.rcParams["svg.hashsalt"] = "test_vis"
        fig, t, a, a2, b = make()
        t(49, 188, "LOES", size=FS + 1, color="#d2601a", fontweight="bold"); t(36, 118, r"$z^*$", ha="left")
        t(10, 20, "label", size=8, ha="left", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.5", fc="white", ec="none", alpha=0.8))
        a([(426, 140), (426, 108), (62, 108), (62, 164)], color="#d2601a", lw=1.5); a([(369, 240), (369, 209)], ls=(0, (2.5, 2))); a([(1, 2), (30, 40)], head=False)
        a2([(270, 96), (306, 96)], style="<|-|>"); a2([(288, 96), (288, 10), (16, 10), (16, 85)])
        b(6, 6, 544, 162, ec="#a4a7ad", lw=0.8, z=0.5, r=7); b(10, 10, 20, 20)
        out.append(_svg(fig)); plt.close(fig)
    assert out[0] == out[1]


def test_figstyle_context_restores():
    import matplotlib
    from crossenbodiment.figstyle import FigureStyle
    before = dict(matplotlib.rcParams)
    with FigureStyle.paper({"font.size": 10}, svg=False) as s:
        assert matplotlib.rcParams["font.family"] == ["STIXGeneral"] and matplotlib.rcParams["font.size"] == 10 and "svg.fonttype" not in s.rc
    assert dict(matplotlib.rcParams) == before
    g = FigureStyle.paper({"axes.edgecolor": "#52514e"}).updated({"font.family": "DejaVu Sans", "font.size": 8})
    assert g.rc["axes.edgecolor"] == "#52514e" and g.rc["font.family"] == "DejaVu Sans" and g.rc["mathtext.fontset"] == "stix"


# ------------------------------------------------------------------ camera

def _gl(k):
    r = RNG(k); f = r.standard_normal(3); f /= np.linalg.norm(f); u = np.cross(f, r.standard_normal(3)); u /= np.linalg.norm(u)
    cam = SimpleNamespace(forward=f.tolist(), up=u.tolist(), pos=(r.standard_normal(3) * 3).tolist(), frustum_near=0.01, frustum_top=0.01 * math.tan(math.radians(22.5)))
    return SimpleNamespace(scene=SimpleNamespace(camera=[cam])), f, r


def test_camera_project_vs_copies():
    from crossenbodiment.camera import ViewSpec, project, project_point
    ns = old_defs("crossenbodiment/draw.py", ["project"], {"np": np})
    for k in range(5):
        r, f, rng = _gl(k); cpos = np.array(r.scene.camera[0].pos); pts = [cpos + 4 * f + rng.standard_normal(3) for _ in range(7)]
        W, H = 1920, 1080
        assert project(r, pts, W, H) == ns["project"](r, pts, W, H)
        c = r.scene.camera[0]; fw, up = np.array(c.forward), np.array(c.up); rt = np.cross(fw, up)
        for pt in pts:
            # robot_overlay.py main view (its project(p)) and --split panels (P x P), as written at BASE
            kk = c.frustum_near / c.frustum_top; v = np.asarray(pt) - np.array(c.pos); z = v @ fw
            assert project_point(r, pt, W, H) == (W / 2 + (v @ rt) / z * kk * H / 2, H / 2 - (v @ up) / z * kk * H / 2)
            P = 1080; v = pt - np.array(c.pos); z = v @ fw
            assert ViewSpec.project_point(r, pt, P, P) == (P / 2 + (v @ rt) / z * kk * P / 2, P / 2 - (v @ up) / z * kk * P / 2)


def test_camera_presets():
    from crossenbodiment.camera import BLENDER_STROBE, BODY_OVERLAY_DEFAULT, body_overlay, robot_overlay, survey_track
    for h in (1.26, 1.74, 2.07, 2.47, 1.32 * 1.26, 0.9):
        k = max(h, 1.74) / 1.26
        assert survey_track(h).to_track() == dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2))
        assert json.dumps(survey_track(h).to_track()) == json.dumps(dict(distance=round(2.8 * k, 2), elevation=-12, azimuth=140, lookat_z=round(0.55 * k, 2)))
        ko = h / 1.26
        assert body_overlay(h).to_track() == dict(distance=float(f"{3.1 * ko:.2f}"), elevation=-12, azimuth=140, lookat_z=float(f"{0.5 * ko:.2f}"))
        assert (f"{body_overlay(h).distance:.2f}", f"{body_overlay(h).lookat_z:.2f}") == (f"{3.1 * ko:.2f}", f"{0.5 * ko:.2f}")
        assert robot_overlay(ko).to_track() == dict(distance=3.3 * ko, elevation=-12, azimuth=140, lookat_z=0.7 * ko)
    assert survey_track(1.26).to_track() == dict(distance=3.87, elevation=-12, azimuth=140, lookat_z=0.76)   # package_amass_lso_survey's child literal
    assert BODY_OVERLAY_DEFAULT.to_track() == dict(distance=3.1, elevation=-12, azimuth=140, lookat_z=0.5)
    assert (BLENDER_STROBE.azimuth, BLENDER_STROBE.elevation, BLENDER_STROBE.distance) == (125.0, -8.0, 14.0)
    rb = old_literals("scripts/render_blender.py", ["CAM_DISTANCE", "AZIMUTH", "ELEVATION"])
    assert (rb["AZIMUTH"], rb["ELEVATION"], rb["CAM_DISTANCE"]) == (BLENDER_STROBE.azimuth, BLENDER_STROBE.elevation, BLENDER_STROBE.distance)


# ------------------------------------------------------------------ strobe layout

class _Scene:
    """records BlenderScene.pose calls"""
    def __init__(self):
        self.calls = []

    def pose(self, m, q, t, x, tint, heading, red_leg, tag):
        self.calls.append((t, x, tint, heading, red_leg, tag)); return dict(x=x, tint=tint, geoms=[dict(color="body")])


def test_strobe_layout_vs_old_loops():
    from crossenbodiment.geometry import root_yaw as yaw
    from crossenbodiment.strobe import StrobeLayout, path_heading, recipe_layout, strobe_size
    AZ = 125.0; r = RNG(1); T = 300
    q = np.concatenate([np.cumsum(r.standard_normal((T, 2)) * 0.05, 0), r.random((T, 1)), r.standard_normal((T, 4))], 1)
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True); ya = yaw(q[::-1].copy())
    cases = [((120, 170, 10), "fixed", {}), ((150, 192, 7), "fixed", dict(align=True)), ((2, 102, 20), "path", {}),
             (None, "fixed", dict(frames=[137, 150, 163, 175, 183, 191], az=-2.4)), (None, "fixed", dict(frames=[120, 163, 187, 263], az=136.7, H=3.0, dx=0.8)),
             ((37, 72, 7), "path", dict(align=True, nudge=0.1))]
    for clip in ("move-ego-0-2_2", "crawl-0.4-0-d_1"):
        for rng, view, x in cases:
            # strobe_final_blender.py export, at BASE
            frames = list(x["frames"]) if "frames" in x else list(range(rng[0], rng[1] + 1, rng[2]))
            n = len(frames); crawl = "crawl" in clip; dx = x.get("dx", 0.8 if crawl else 0.6)
            tints = [0.62 * (1 - i / (n - 1)) for i in range(n)]; W = (n - 1) * dx + (2.4 if crawl else 1.6) + x.get("wider", 0.0); H = x.get("H", 1.4 if crawl else 2.45)
            y = yaw(q); old = []
            for i, (t, k) in enumerate(zip(frames, tints)):
                if view == "fixed":
                    h = math.radians(x.get("az", 140.0) - AZ)
                else:
                    v = q[frames[-1], :2] - q[frames[0], :2]; h = math.atan2(v[1], v[0]) + math.radians(90.0 - AZ)
                if x.get("align"):
                    h += float(y[t] - ya[t])
                old.append((t, (i - (n - 1) / 2) * dx + x.get("nudge", 0.0), k, h))
            lay = StrobeLayout(frames if "frames" in x else StrobeLayout.frame_range(*rng), dx, view, azimuth=x.get("az", 140.0), align=bool(x.get("align")), nudge=x.get("nudge", 0.0))
            sc = _Scene(); ps = lay.poses(sc, None, q, False, "human", ref_yaw=ya if lay.align else None)
            assert [(c[0], c[2], c[3]) for c in sc.calls] == [(o[0], o[2], o[3]) for o in old] and [p["shift"] for p in ps] == [o[1] for o in old]
            assert strobe_size(clip, n, dx, wider=x.get("wider", 0.0), H=x.get("H")) == (W, H) and lay.tints == tints
    # export_strobe_blender.py clip_panels (archived), at BASE: fixed = 15 deg, orbit_auto = path heading + offset - 125
    ns = old_defs("scripts/export_strobe_blender.py", ["path_heading"], {"np": np})
    for cfg in (dict(strobe=dict(start_step=10, end_step=100, interval=15), camera=dict(type="fixed", azimuth_offset=140.0)),
                dict(strobe=dict(start_step=1, end_step=251, interval=50), camera=dict(type="orbit_auto", azimuth_offset=90.0))):
        lay, view = recipe_layout(cfg, 0.6); st = cfg["strobe"]; frames = list(range(st["start_step"], st["end_step"] + 1, st["interval"]))
        still = cfg["camera"]["type"] == "fixed"
        h = math.radians(15.0) if still else ns["path_heading"](q, frames) + math.radians(cfg["camera"]["azimuth_offset"] - AZ)
        assert lay.frames == frames and all(lay.heading(q, t) == h for t in frames) and path_heading(q, frames) == ns["path_heading"](q, frames)
        assert view == ("fixed 140 deg" if still else f"side, path + {cfg['camera']['azimuth_offset']} deg")


def test_strobe_windows_vs_child_all():
    from crossenbodiment.strobe import around_fall, mid_falls, stretches, whole_clip
    ns = old_defs("scripts/child_all_blender.py", ["stretches"], {})
    N, STEP, BEFORE, START, MINLEN = 6, 15, 40, 45, 9
    for k in range(20):
        r = RNG(10 + k); n = int(r.integers(80, 400)); ref = 0.9 + 0.05 * r.random(n); z = ref * (0.3 + 0.9 * (r.random(n) > 0.1 * (k % 4)))
        below = z < 0.5 * ref
        assert stretches(below) == ns["stretches"](below)
        old_mid = [(a, b) for a, b in ns["stretches"](z < 0.5 * ref) if a >= START and b - a >= MINLEN]
        assert mid_falls(z, ref, START, MINLEN) == old_mid
        if old_mid:
            a, b = old_mid[0]; s = int(np.clip(a - BEFORE, 1, n - 1 - STEP * (N - 1))); assert around_fall(old_mid[0], n, N, STEP, BEFORE) == (s, s + STEP * (N - 1), STEP)
        iv = (n - 2) // (N - 1); assert whole_clip(n, N) == (1, 1 + iv * (N - 1), iv)


# ------------------------------------------------------------------ blender

def test_pose_spec_vs_export_blender_package():
    import mujoco
    from crossenbodiment.blender import BlenderScene, contact_frames, heading, load_model
    src = old_source("scripts/export_blender_package.py")
    ns = {"np": np, "mujoco": mujoco, "sys": sys, "REPO": REPO, "meshes": {},
          "GEOM": {mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
                   mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid", mujoco.mjtGeom.mjGEOM_MESH: "mesh"}}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("load", "color_of", "pose", "contact_frames", "heading", "strip"):
            exec(compile(ast.Module([node], []), "old", "exec"), ns)
    S = MAIN / "outputs/user_survey"
    for xml, qp, payload, red, tag in ((S / "models/heavy_arm.xml", S / "heavy_arm/trajectories/move-ego-0-2_4/best.npz", True, False, "heavy_arm"),
                                        (S / "models/limit_leg.xml", S / "limit_leg/trajectories/move-ego-0-2_2/best.npz", False, True, "limit_leg"),
                                        (S / "models/g1/scene.xml", S / "g1/trajectories/move-ego-0-2_2/best.npz", False, False, "g1")):
        if not qp.exists():
            raise unittest.SkipTest(f"{qp} missing")
        m_old, m_new = ns["load"](xml, payload), load_model(xml, payload); q = np.load(qp)["qpos"].astype(np.float64)
        sc = BlenderScene()
        for t, x, tint, h in ((10, 0.0, 0.3, 0.2), (100, -0.5, 0.0, -1.3), (299, 0.33, 0.62, 2.0)):
            assert ns["pose"](m_old, q, t, x, tint, h, red, tag) == sc.pose(m_new, q, t, x, tint, h, red, tag)
        if tag == "g1":
            assert list(ns["meshes"]) == list(sc.meshes)
        assert all(np.array_equal(ns["meshes"][k], sc.meshes[k]) for k in sc.meshes)
        fr = [40, 52, 64]
        assert ns["heading"](q, fr) == heading(q, fr)
        a, b = ns["contact_frames"](m_old, q, fr), contact_frames(m_new, q, fr); assert a[0] == b[0] and np.array_equal(a[1], b[1])
        assert json.dumps(ns["strip"](xml, qp, fr, (0.0, 0.2, 0.4), 0.57, tag, payload=payload, red_leg=red, ground=True)) == \
            json.dumps(sc.strip(xml, qp, fr, (0.0, 0.2, 0.4), 0.57, tag, payload=payload, red_leg=red, ground=True))


def test_blender_scene_files_and_render_argv():
    from unittest import mock
    from crossenbodiment import blender
    d = WORK / "scene"; d.mkdir(parents=True, exist_ok=True)
    sc = blender.BlenderScene(frames=[0, 20], body_scale=1.0); sc.add_panel("p", dict(figure="x", poses=[])); sc.meshes["g1_a__v"] = np.ones((3, 3), np.float32)
    sc.write(d, placeholder=True)
    s = json.loads((d / "scene.json").read_text())
    assert list(s) == ["version", "panels", "frames", "body_scale"] and s["version"] == blender.SCHEMA_VERSION
    assert list(np.load(d / "meshes.npz").keys()) == ["_", "g1_a__v"] and (d / "render_blender.py").read_bytes() == (REPO / "scripts/render_blender.py").read_bytes()
    back = blender.BlenderScene.load(d); assert back.panels == sc.panels and back.extra == dict(frames=[0, 20], body_scale=1.0) and list(back.base_meshes) == ["_", "g1_a__v"]
    calls = []
    with mock.patch.object(blender.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw))):
        blender.BlenderScene.render(d, samples=256, threads=32, panels=["a_adult", "a_z0"], out="panels")           # strobe_final_blender render(only)
        blender.BlenderScene.render(d, samples=96, threads=32, scale=0.6, out="renders")                             # child_all / body_all
        blender.BlenderScene.render(d, samples=24, threads=32, scale=0.35, out="preview")                            # walk_band preview
    bl = ["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "python", "render_blender.py", "--"]
    assert calls[0][0] == bl + ["--samples", "256", "--device", "CPU", "--threads", "32", "--out", "panels", "--panel", "a_adult", "--panel", "a_z0"]
    assert calls[1][0] == bl + ["--samples", "96", "--scale", "0.6", "--device", "CPU", "--threads", "32", "--out", "renders"]
    assert calls[2][0] == bl + ["--samples", "24", "--scale", "0.35", "--device", "CPU", "--threads", "32", "--out", "preview"]
    assert all(kw["cwd"] == d and kw["check"] and kw["env"]["OMP_NUM_THREADS"] == "1" for _, kw in calls)


# ------------------------------------------------------------------ clips

def test_strobe_source_routing():
    from crossenbodiment.clips import strobe_source
    def old(src, clip, body):            # strobe_final_blender.trajectories at BASE, the paths only
        if src not in ("child_52", "c540", "tall_body_52") and not src.startswith(("body_52/", "child_52_s", "kpkd:")):
            return ("copy", f"{src}/{clip}")
        t = clip.rsplit("_", 1)[0]; target = {"payload": "data/origin_motion", "limit_leg": "data/adult_lockRleg_bent_lim/retargeting_motion",
                                               "tall": "data/len1.4_wid0.8/retargeting_motion"}.get(body, "data/child/retargeting_motion")
        run = {"child_52": f"outputs/child_52_rerun/{clip}_es_s0", "c540": f"outputs/c540_bfmglobal_a03/{clip}"}.get(src, f"outputs/{src}/{clip}_es_s0")
        if src.startswith("kpkd:"):
            run = "outputs/" + src[len("kpkd:"):] + "/es_s0"
        if src.startswith("child_52_s"):
            run = f"outputs/child_52_rerun/{clip}_es_s{src[len('child_52_s'):]}"
        return ("run", run, f"{target}/{t}/{clip}.npz")
    for src in ("child_52", "child_52_s2", "child_52_s4", "c540", "tall_body_52", "body_52/payload", "body_52/limit_leg",
                "kpkd:child_kpkd_headstand/kp0.85_kd1.0", "outputs/user_survey/g1_task52/trajectories", "outputs/payload_all52/trajectories"):
        for clip, body in (("headstand_4", "child"), ("move-ego-low-0-2_9", "tall"), ("rotate-y-5-0.8_7", "payload"), ("move-ego-0-2_2", "limit_leg")):
            assert strobe_source(src, clip, body) == old(src, clip, body), (src, clip)


def test_clip_bundle_vs_old_trajectories():
    from crossenbodiment.clips import ClipBundle
    d = WORK / "clips"; (d / "run").mkdir(parents=True, exist_ok=True); r = RNG(3)
    ref_a, ref_b = r.standard_normal((101, 76)), r.standard_normal((101, 76)).astype(np.float32)
    np.savez(d / "adult.npz", qpos=ref_a); np.savez(d / "body.npz", qpos=ref_b)
    np.savez(d / "run/origin_z.npz", qpos=r.standard_normal((104, 76))); np.savez(d / "run/best.npz", qpos=r.standard_normal((100, 76)).astype(np.float32))
    (d / "cemrun").mkdir(exist_ok=True); np.savez(d / "cemrun/best.npz", qpos=r.standard_normal((120, 76)))
    # the old code (strobe_final_blender.trajectories + add_cem at BASE), inline
    q = dict(reference_adult=np.load(d / "adult.npz")["qpos"][1:], reference_body=np.load(d / "body.npz")["qpos"][1:]); n = len(q["reference_body"])
    q.update(z0=np.load(d / "run/origin_z.npz")["qpos"][:n], best=np.load(d / "run/best.npz")["qpos"][:n])
    old = d / "old"; old.mkdir(exist_ok=True)
    for k, v in q.items():
        np.savez_compressed(old / f"{k}.npz", qpos=v.astype(np.float32), fps=np.float32(30))
    (old / "video.json").write_text(json.dumps(dict(clip="x_1", body="child", fps=30, frames={k: len(v) for k, v in q.items()}, search_run="outputs/run_x"), indent=1))
    nn = len(np.load(old / "reference_body.npz")["qpos"]); c = np.load(d / "cemrun/best.npz")["qpos"]
    np.savez_compressed(old / "cem.npz", qpos=c[:nn].astype(np.float32), fps=np.float32(30))
    v = json.loads((old / "video.json").read_text()); v.setdefault("frames", {})["cem"] = nn; v["cem_run"] = str(d / "cemrun"); (old / "video.json").write_text(json.dumps(v, indent=1))
    # new: one bundle; and a copied folder + add_cem (from_survey keeps the key order)
    new = d / "new"; b = ClipBundle.from_run("x_1", "child", run=d / "run", reference_body=d / "body.npz", reference_adult=d / "adult.npz")
    b.meta["search_run"] = "outputs/run_x"; b.save(new); b2 = ClipBundle.from_survey(new).add_cem(str(d / "cemrun")); b2.save_npz(new, ["cem"]); b2.save_json(new)
    for f in ("reference_adult", "reference_body", "z0", "best", "cem"):
        A, B = np.load(old / f"{f}.npz"), np.load(new / f"{f}.npz")
        assert list(A.keys()) == list(B.keys()) and all(A[k].dtype == B[k].dtype and np.array_equal(A[k], B[k]) for k in A)
    assert (old / "video.json").read_text() == (new / "video.json").read_text()


# ------------------------------------------------------------------ sheet

def test_contact_sheet_vs_old_pages():
    from PIL import Image, ImageDraw, ImageFont
    from crossenbodiment.sheet import ContactSheet, hcat, write_index_md
    d = WORK / "sheet"; d.mkdir(parents=True, exist_ok=True); r = RNG(5); paths = []
    for i in range(6):
        p = d / f"in{i}.png"; Image.fromarray((r.random((90 + 7 * i, 160 + 11 * i, 3)) * 255).astype(np.uint8)).save(p); paths.append(p)
    rows = [paths[0:3], paths[3:6]]; caps = ["#1 a: MPJPE 1.0 -> 2.0", "#2 b"]
    for first, (Wc, L, fs, q) in ((True, (640, 40, 22, 90)), (False, (480, 36, 20, 88))):
        # export_strobe_blender.overview (row height = the first image) / child_all_blender.pages (the tallest), at BASE
        F = ImageFont.truetype("DejaVuSans.ttf", fs); ims = [[Image.open(p).convert("RGB") for p in row] for row in rows]
        ims = [[im.resize((Wc, round(Wc * im.height / im.width))) for im in row] for row in ims]; hs = [row[0].height if first else max(im.height for im in row) for row in ims]
        sheet = Image.new("RGB", (3 * Wc, sum(hs) + L * (len(rows) + 1)), "white"); dd = ImageDraw.Draw(sheet); y = L
        for j, lab in enumerate(("reference (adult)", "z0", "LOES (best latent)")):
            dd.text((j * Wc + 10, 8), lab, fill=(0, 0, 0), font=F)
        for i, (c, row) in enumerate(zip(caps, ims)):
            dd.text((10, y + 6), c, fill=(0, 0, 0), font=F)
            for j, im in enumerate(row):
                sheet.paste(im, (j * Wc, y + L))
            y += L + hs[i]
        sheet.save(d / "old.jpg", quality=q)
        s = ContactSheet(("reference (adult)", "z0", "LOES (best latent)"), col_width=Wc, label_h=L, font_size=fs, row_height="first" if first else "max", quality=q)
        for c, row in zip(caps, rows):
            s.add_row(c, row)
        s.save(d / "new.jpg")
        assert (d / "old.jpg").read_bytes() == (d / "new.jpg").read_bytes()
    ims = [Image.open(p).convert("RGB") for p in paths[:4]]; gap = 24          # strobe_final_blender.combine at BASE
    out = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), max(im.height for im in ims)), "white"); x = 0
    for im in ims:
        out.paste(im, (x, 0)); x += im.width + gap
    assert np.array_equal(np.asarray(out), np.asarray(hcat(paths[:4], 24)))
    lines = ["| 1 | `a` | x |", "| 2 | `b` | y |"]; hdr = ["#", "clip", "v"]
    write_index_md(d / "index.md", "T", "intro text.", hdr, lines)
    assert (d / "index.md").read_text() == "# T\n\nintro text.\n\n| # | clip | v |\n|" + "---|" * 3 + "\n" + "\n".join(lines) + "\n"


# ------------------------------------------------------------------ video

def test_frame_bands_vs_old():
    from PIL import Image, ImageDraw, ImageFont
    from crossenbodiment import video
    ns = old_defs("scripts/render_panels.py", ["label_px"], {"Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont, "np": np, "FONT": video.FONT})
    ns2 = old_defs("scripts/robot_overlay.py", ["legend"], {"Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont, "np": np, "FONT": video.FONT})
    img = (RNG(7).random((540, 960, 3)) * 255).astype(np.uint8)
    assert np.array_equal(ns["label_px"](img, "z0  (move)", "MPJPE 5.0 cm | fall 0.5 s", 36), video.label_px(img, "z0  (move)", "MPJPE 5.0 cm | fall 0.5 s", 36))
    items = [("ghost", (0.55, 0.57, 0.62, 0.35)), ("z0", (0.85, 0.22, 0.20, 1.0)), ("searched", (0.18, 0.42, 0.85, 1.0))]
    assert np.array_equal(ns2["legend"](img, "clip on the G1", items), video.legend(img, "clip on the G1", items))


def test_video_renderer_vs_survey_replay():
    """VideoRenderer's side by side (render_panels' spec) and overlay = the survey's standalone replay.py, frame for frame. The GPU's
    frames are not bit-reproducible run to run, so the rasterisation is tests/golden/fake_render.py's picture of the scene digest"""
    import importlib.util
    import mujoco
    saved = mujoco.Renderer.render                               # loading tests/golden/fake_render.py patches mujoco.Renderer.render
    fs = importlib.util.spec_from_file_location("fake_render_test", REPO / "tests/golden/fake_render.py"); fs.loader.exec_module(importlib.util.module_from_spec(fs))
    try:
        _video_vs_replay()
    finally:
        mujoco.Renderer.render = saved


def _video_vs_replay():
    import importlib.util
    from crossenbodiment.camera import ViewSpec
    from crossenbodiment.video import VideoRenderer
    top = MAIN / "outputs/user_survey"; clip = top / "child/trajectories/move-ego-0-2_2"
    if not (clip / "video.json").exists():
        raise unittest.SkipTest("the survey's child clip is missing")
    spec = json.loads((clip / "video.json").read_text()); d = WORK / "replay"; d.mkdir(parents=True, exist_ok=True)
    for p in spec["side_by_side"]["panels"]:                     # 6-frame cuts
        f = d / p["qpos"]; np.savez(f, qpos=np.load(clip / p["qpos"])["qpos"][:6]); p["qpos"] = str(f)
    for l in spec["overlay"]["layers"]:
        f = d / l["qpos"]; np.savez(f, qpos=np.load(clip / l["qpos"])["qpos"][:6]); l["qpos"] = str(f)
    spec["side_by_side"]["size"] = 240; spec["overlay"].update(width=320, height=180)
    sp = importlib.util.spec_from_file_location("survey_replay_test", REPO / "scripts/survey_replay.py"); R = importlib.util.module_from_spec(sp); sp.loader.exec_module(R)
    old = R.side_by_side(top, clip, spec["side_by_side"])
    s = spec["side_by_side"]; c = s["camera"]
    grid, _ = VideoRenderer().side_by_side(dict(cols=len(s["panels"]), size=s["size"], captions=False, track=dict(c),
                                                 panels=[dict(xml=str(top / p["model"]), qpos=p["qpos"], rgba=p["rgba"]) for p in s["panels"]]))
    assert len(old) == len(grid) and all(np.array_equal(a, b) for a, b in zip(old, grid))
    import mujoco
    o = spec["overlay"]; old = R.overlay(top, clip, o)
    m = mujoco.MjModel.from_xml_path(str(top / o["model"])); trajs = [np.load(l["qpos"])["qpos"].astype(np.float64) for l in o["layers"]]
    new = VideoRenderer.overlay(m, trajs, ViewSpec.from_track(o["camera"]), o["root"], [l["rgba"] for l in o["layers"]], (o["width"], o["height"]), o.get("flat", True))
    assert len(old) == len(new) and all(np.array_equal(a, b) for a, b in zip(old, new))


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}")
        except (AssertionError, KeyError, ValueError, OSError, ImportError, TypeError, AttributeError, subprocess.CalledProcessError) as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
