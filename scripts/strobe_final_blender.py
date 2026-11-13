#!/usr/bin/env python3
"""strobe_final_blender.py -- the strobe figures the user kept (11/10), rendered in Blender at final quality into
outputs/strobe_task52_blender: per example four panels (adult reference, z0, CEM (BFM-Zero)'s best, LOES's best) and the four side by side.

Examples, frames and views are the ones the user chose from the previews: configs/figures/strobe_examples.yaml (the lists EX and CEM
of this script until OOP refactor step 10, with the user's reasons for each pick), read into EX / CEM below. Runs: payload / limited
leg / G1 from the 52-motion runs of 11/09 (seed 0, LOES's best after 2,048 rollouts); child from outputs/child_52_rerun (the
corrected-alignment runs of the Task page; outputs/child_52 holds the older runs, used by mistake until 11/10). Trajectories are in the
survey's frame alignment (references from their second frame; a rollout's frame t against the reference's frame t + 1), copied or
packaged into trajectories/<example>/ (crossembodiment.clips: ClipBundle, strobe_source routes "child_52", "child_52_s<k>", "c540",
"tall_body_52", "body_52/<body>", "kpkd:<folder>" or a survey-format folder).
View: "fixed" = one world view for every panel (azimuth 140, the survey videos'); "path" = side-on to each panel's own path
(path heading + 90 deg). split-1_7 (limited leg): z0's and LOES's poses are each turned about the root to the reference's heading at
that frame (LOES had turned about 90 deg); likewise move-ego-low-90-2_9 (the reference turns left about a full circle, z0 and LOES
turn right). The user labels both in the figure. (crossembodiment.strobe.StrobeLayout)
Look: scripts/render_blender.py (Cycles, 256 samples): one body colour (the G1 black), payload weight amber, the limited leg's locked right leg red,
poses at one depth fading from light (early) to full colour (late).
The payload and limited-leg sources (outputs/payload_all52, outputs/limit_leg_all52) were deleted on 11/10 after the export; their copies are in
outputs/strobe_task52_blender/trajectories, and scene.json + meshes.npz there re-render everything with render_blender.py.
Inputs are read through crossembodiment.paths.read_path ($CE_OUTPUTS_ROOT), outputs written under $CE_RESULTS_ROOT (both default outputs/).
usage: uv run python scripts/strobe_final_blender.py [export] [render] [combine]   (none: all three)
       uv run python scripts/strobe_final_blender.py update <example> ...   (re-make single examples in place)
       uv run python scripts/strobe_final_blender.py cem [<example> ...]      (add the CEM panel to examples already made)
       uv run python scripts/strobe_final_blender.py scene [<example> ...]    (a fresh scene.json + meshes.npz of these examples, default
                                                                              all, from the existing trajectories; nothing rendered)"""
import os, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from crossembodiment import paths                                         # noqa: E402
from crossembodiment.blender import BlenderScene, load_model              # noqa: E402
from crossembodiment.clips import ClipBundle, copy_folder, strobe_source  # noqa: E402
from crossembodiment.geometry import root_yaw as yaw                      # noqa: E402  root quaternion yaw (refactor step 3)
from crossembodiment.palette import ACCENTS                               # noqa: E402
from crossembodiment.sheet import hcat                                    # noqa: E402
from crossembodiment.strobe import StrobeLayout, strobe_size              # noqa: E402

OUT_REL = "outputs/strobe_task52_blender"; OUT = paths.write_path(OUT_REL); THREADS = int(os.environ.get("STROBE_THREADS", 32))
CONFIG = yaml.safe_load((paths.FIGURE_CONFIGS / "strobe_examples.yaml").read_text())
_X = ("name", "body", "clip", "source", "range", "view")
# name, body, clip, trajectory source, frames (start, end, step) or None, view, extra (frames / az / align / H / dx / ...)
EX = [(e["name"], e["body"], e["clip"], e["source"], tuple(e["range"]) if "range" in e else None, e["view"], {k: v for k, v in e.items() if k not in _X})
      for e in CONFIG["examples"]]
CEM = dict(CONFIG["cem"])
MODEL = {k: paths.read_path(v) for k, v in CONFIG["models"].items()}
FILES = (("adult", "reference_adult.npz"), ("z0", "z0.npz"), ("cem", "cem.npz"), ("zstar", "best.npz"))
G1_COLOR = ACCENTS["g1_strobe"]["blender"]                               # the G1 drawn black (user, 11/10)


def add_cem(name):
    """trajectories/<name>/cem.npz: the CEM run's best rollout, cut to the reference's length (as z0 and best)."""
    d = OUT / "trajectories" / name; b = ClipBundle.from_survey(d).add_cem(CEM[name]); b.save_npz(d, ["cem"]); b.save_json(d)


def trajectories(name, body, clip, src):
    d = OUT / "trajectories" / name; d.mkdir(parents=True, exist_ok=True); s = strobe_source(src, clip, body)
    if s[0] == "copy":
        return copy_folder(paths.read_path(s[1]), d)
    ClipBundle.from_run(clip, body, run=s[1], reference_body=s[2]).save(d)
    return d


def example_panels(sc, name, body, clip, rng, view, x, d):
    """the four panels of one example from its trajectory folder d into scene sc"""
    frames = list(x["frames"]) if "frames" in x else StrobeLayout.frame_range(*rng)       # an explicit frame list, or (start, end, step)
    crawl = "crawl" in clip; dx = x.get("dx", 0.8 if crawl else 0.6)                     # a crawl lies along the floor: a lower, wider frame
    lay = StrobeLayout(frames, dx, view, azimuth=x.get("az", 140.0), align=bool(x.get("align")), nudge=x.get("nudge", 0.0))
    W, H = strobe_size(clip, lay.n, dx, wider=x.get("wider", 0.0), H=x.get("H")); pxw = x.get("px", 1500)
    ya = yaw(np.load(d / "reference_adult.npz")["qpos"].astype(np.float64))
    for label, f in FILES:
        if not (d / f).exists():
            print("  no", f, "yet: panel", label, "skipped", flush=True); continue
        xml = MODEL["adult"] if label == "adult" else MODEL[body]; m = load_model(xml, payload=(body == "payload" and label != "adult"))
        q = np.load(d / f)["qpos"].astype(np.float64); assert q.shape[1] == m.nq, (name, label, q.shape, m.nq)
        red = body == "limit_leg" and label != "adult"; tag = "g1" if (body == "g1" and label != "adult") else "human"
        poses = lay.poses(sc, m, q, red, tag, recolor=G1_COLOR if tag == "g1" else None, ref_yaw=ya if lay.align and label != "adult" else None)
        sc.add_panel(f"{name}_{label}", lay.panel(poses, W, H, view, pxw=pxw, feet=x.get("feet", 0.3), elevation=x.get("elev")))
    print(name, frames, view, "aligned" if x.get("align") else "", flush=True)


def export(only=None, traj=None, fresh=False):
    """only: rebuild just these examples into the existing scene.json, from the trajectories already in OUT/trajectories (traj: another
    trajectories folder); fresh: into an empty scene instead (the `scene` command)"""
    if OUT.exists() and only is None:
        raise SystemExit(f"{OUT} exists: move it away first, or update single examples")
    sc = BlenderScene.load(OUT) if only and not fresh else BlenderScene()
    for name, body, clip, src, rng, view, x in EX:
        if only and name not in only:
            continue
        d = Path(traj or OUT / "trajectories") / name if only else trajectories(name, body, clip, src)
        if not only and name in CEM:
            add_cem(name)
        example_panels(sc, name, body, clip, rng, view, x, d)
    sc.write(OUT)


def render(only=None, labels=None):
    sel = [f"{n}_{k}" for n in only for k, _ in FILES if labels is None or k in labels] if only else None
    BlenderScene.render(OUT, samples=256, threads=THREADS, panels=sel, out="panels")


def combine(gap=24, only=None, panels=None):
    """panels: the folder of rendered panels (default OUT/panels; under $CE_OUTPUTS_ROOT if OUT has none)"""
    src = Path(panels or paths.own_path(f"{OUT_REL}/panels")); (OUT / "combined").mkdir(parents=True, exist_ok=True)
    for name, *_ in EX:
        if only and name not in only:
            continue
        ims = [src / f"{name}_{k}.png" for k, _ in FILES if (src / f"{name}_{k}.png").exists()]
        if not ims:                                      # not rendered yet (the old loop stopped here with an error)
            print(name, "no panels rendered: skipped", flush=True); continue
        out = hcat(ims, gap); out.save(OUT / "combined" / f"{name}.png"); print(name, out.size, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["update"]:                      # update NAME ...: re-export, render and combine only these examples
        names = sys.argv[2:]; assert names and all(n in [e[0] for e in EX] for n in names), names
        export(names); render(names); combine(only=names); sys.exit()
    if sys.argv[1:2] == ["cem"]:                         # cem [NAME ...]: add the CEM panel (cem.npz, export, render the CEM panels only, combine)
        names = sys.argv[2:] or [e[0] for e in EX]; assert all(n in CEM for n in names), names
        for n in names:
            add_cem(n)
        export(names); render(names, labels=("cem",)); combine(only=names); sys.exit()
    if sys.argv[1:2] == ["scene"]:                       # scene [NAME ...]: a fresh scene from the existing trajectories (read root)
        names = sys.argv[2:] or [e[0] for e in EX]; assert all(n in [e[0] for e in EX] for n in names), names
        export(names, traj=paths.read_path(OUT_REL) / "trajectories", fresh=True); sys.exit()
    steps = set(sys.argv[1:]) or {"export", "render", "combine"}
    for s in ("export", "render", "combine"):
        if s in steps:
            globals()[s]()
