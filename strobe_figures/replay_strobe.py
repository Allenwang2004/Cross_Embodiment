#!/usr/bin/env python
"""Strobe figures and comparison video from RECORDED trajectories (no rollout).

Same artifacts, same YAML, same knobs as ``render_strobe.py`` -- the only
difference is where the motion comes from: instead of driving the frozen
Metamotivo actor, this replays ``qpos`` dumps kinematically (``data.qpos`` +
``mj_forward``). Nothing is integrated, so no environment, no actor, no GPU.

Inputs are the shared clip folders described in ``replay_README.md``::

    outputs/heavy_arm/trajectories/headstand_4/
        reference_adult.npz   the adult doing the original motion   -> "adult"
        reference_body.npz    the motion on the target body         -> "body" (optional)
        z0.npz                rollout under the clip's own latent   -> "z0"
        best.npz              rollout under the searched latent     -> "zstar"
        video.json            models, colours, camera, fps

Each ``.npz`` holds ``qpos`` (frames x nq) and ``fps``.

Outputs (identical naming to ``render_strobe.py``)::

    outputs/figures/strobe/<task>_<camera>_<source>_strobe.png
    outputs/figures/strobe/<task>_<camera>_compare_<labels>.mp4

``task_name`` comes from the YAML (``--task-name`` overrides); when the YAML
omits it, the clip folder name is used so clips do not overwrite each other.

The clip folder and the motion sources can live in the YAML instead of the
command line::

    replay:
      clip: outputs/heavy_arm/trajectories/headstand_4
      sources:
        - label: adult
          qpos: reference_adult.npz      # file inside the clip folder
          xml: models/adult.xml          # optional: force this model
        - label: z0
          qpos: z0.npz
        - label: zstar
          qpos: best.npz

With ``replay.clip`` set the positional argument is unnecessary. ``sources``
replaces the file-name convention: any label, any file, any panel order. Model
resolution per source is then: YAML ``sources[].xml`` -> ``video.json`` ->
``models/<body>.xml`` -> the config's ``xml_path`` / ``baseline.xml`` (with a
printed warning for the last two).

Camera, strobe window, colours and the burn-in badges all come from the same
``configs/render_strobe.yaml`` (never edited here) and every key can be
overridden on the CLI, exactly as in ``render_strobe.py``.

Example::

    MUJOCO_GL=egl uv run replay_strobe.py \
        outputs/heavy_arm/trajectories/headstand_4 \
        --config configs/render_strobe.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Headless machines have no X11 display: use EGL for offscreen rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_strobe import deep_set, load_config, resolve_path, save_config  # noqa: E402
from tools.strobe_wrapper import (  # noqa: E402
    StrobeStudio,
    capture_source,
    composite_strobe,
    motion_heading,
    strobe_window,
)

# Our source labels -> trajectory file names in a shared clip folder.
_SOURCE_FILES = {
    "adult": "reference_adult.npz",
    "z0": "z0.npz",
    "zstar": "best.npz",
    "body": "reference_body.npz",
}
_DEFAULT_SOURCES = ("adult", "z0", "zstar")


def source_specs(cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Motion sources: the YAML ``replay.sources`` block when present, else the
    ``--sources`` labels mapped through the shared file-name convention.

    ``--sources`` also accepts ``label=file.npz`` entries, which is how you try
    an odd file name without editing the YAML.
    """
    block = (cfg.get("replay") or {}).get("sources")
    if block:
        if isinstance(block, dict):
            block = [{"label": k, **v} for k, v in block.items()]
        specs: list[dict] = []
        for i, entry in enumerate(block):
            if not isinstance(entry, dict) or not entry.get("label"):
                raise SystemExit(f"replay.sources[{i}] needs at least a label: "
                                 f"{entry!r}")
            label = str(entry["label"])
            qpos = entry.get("qpos") or _SOURCE_FILES.get(label)
            if not qpos:
                raise SystemExit(f"replay.sources[{i}] ({label!r}) has no qpos "
                                 "file and is not a known label: "
                                 f"{sorted(_SOURCE_FILES)}")
            specs.append({"label": label, "qpos": str(qpos),
                          "xml": entry.get("xml")})
        dupes = {s["label"] for s in specs if [x["label"] for x in specs].count(s["label"]) > 1}
        if dupes:
            raise SystemExit(f"replay.sources labels repeat: {sorted(dupes)}")
        return specs

    labels = ([s.strip() for s in args.sources.split(",") if s.strip()]
              if args.sources else list(_DEFAULT_SOURCES))
    specs = []
    for item in labels:
        label, _, file = item.partition("=")
        label = label.strip()
        qpos = file.strip() or _SOURCE_FILES.get(label)
        if not qpos:
            raise SystemExit(f"unknown source {label!r}; choose from "
                             f"{sorted(_SOURCE_FILES)} or write label=file.npz")
        specs.append({"label": label, "qpos": qpos, "xml": None})
    return specs


# ---------------------------------------------------------------------------
# clip folder -> trajectories and models
# ---------------------------------------------------------------------------

def load_trajectory(path: Path) -> tuple[np.ndarray, float | None]:
    """``((T, nq), fps)`` from a shared trajectory dump."""
    if not path.exists():
        raise SystemExit(f"trajectory not found: {path}")
    with np.load(path) as data:
        if "qpos" not in data:
            raise SystemExit(f"{path} has no 'qpos' (keys={list(data)})")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        fps = float(data["fps"]) if "fps" in data else None
    return qpos, fps


def load_video_spec(clip_dir: Path) -> dict:
    """``video.json`` next to the trajectories, or ``{}`` when absent."""
    spec_path = clip_dir / "video.json"
    if not spec_path.exists():
        print(f"note: no video.json in {clip_dir}: models fall back to "
              "models/<body>.xml and the config stays in charge of the camera")
        return {}
    try:
        return json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{spec_path} is not valid JSON: {exc}") from exc


def body_name(clip_dir: Path, spec: dict) -> str:
    """Body variant of a clip: video.json's ``body``, else the folder layout
    (``<body>/trajectories/<clip>``), else the clip folder itself."""
    if spec.get("body"):
        return str(spec["body"])
    if clip_dir.parent.name == "trajectories" and clip_dir.parent.parent.name:
        return clip_dir.parent.parent.name
    return clip_dir.name


def models_root(clip_dir: Path, body: str) -> Path:
    """Nearest folder that holds ``models/<body>.xml`` (repo root by default)."""
    for candidate in (REPO_ROOT, *list(clip_dir.parents)[:4]):
        if (candidate / "models" / f"{body}.xml").exists():
            return candidate
    return REPO_ROOT


def model_for(label: str, traj_file: str, clip_dir: Path, spec: dict,
              body: str, root: Path, cfg: dict,
              forced: str | None = None) -> tuple[Path, str]:
    """MuJoCo model for one source, with the reason it was picked."""
    # 0. an explicit model in the YAML sources block always wins
    if forced:
        path = (root / forced) if not Path(forced).is_absolute() \
            else Path(forced)
        if not path.exists():
            raise SystemExit(f"replay.sources xml for {label!r} does not exist: "
                             f"{path}")
        return path.resolve(), "config sources[]"

    # 1. whatever the shared video.json used for this exact trajectory
    for section in ("side_by_side", "overlay"):
        block = spec.get(section) or {}
        entries = list(block.get("panels") or []) + list(block.get("layers") or [])
        for entry in entries:
            if entry.get("qpos") != traj_file or not entry.get("model"):
                continue
            candidate = root / entry["model"]
            if candidate.exists():
                return candidate.resolve(), f"video.json {section}"
            print(f"  warning: video.json model {entry['model']} is missing "
                  f"under {root}; trying a fallback")

    # 2. the sibling convention: models/adult.xml, models/<body>.xml
    guess = "adult.xml" if label == "adult" else f"{body}.xml"
    candidate = root / "models" / guess
    if candidate.exists():
        return candidate.resolve(), f"models/{guess}"

    # 3. the YAML, which is the last thing standing between us and a crash
    fallback = (cfg.get("baseline") or {}).get("xml") if label == "adult" \
        else cfg.get("xml_path")
    if fallback:
        path = resolve_path(fallback)
        if path.exists():
            print(f"  warning: no model for {label!r} in this clip folder; "
                  f"using the config's {fallback} instead")
            return path, "config yaml"
    raise SystemExit(f"no MuJoCo model found for {label!r} ({traj_file}): "
                     f"looked for video.json entries, {candidate} and the "
                     "config fallback")


def camera_settings(cfg: dict, spec: dict, qpos: np.ndarray, start: int,
                    end: int, use_video_camera: bool) -> dict:
    """Camera block for one source: the YAML, optionally re-aimed at the
    framing recorded in ``video.json``.

    ``video.json`` stores an *absolute* free-camera azimuth; our controller
    places the camera relative to the motion heading, so the offset is
    ``video.azimuth - heading`` over the captured window. The result reproduces
    their framing at the start of the window and then follows the body.
    """
    settings = dict(cfg.get("camera", {}))
    for stray in ("start_color", "final_color", "middle_color"):
        settings.pop(stray, None)
    if not use_video_camera:
        return settings
    video_cam = (spec.get("side_by_side") or {}).get("camera") or {}
    if not video_cam:
        print("  note: --video-camera asked for but video.json has no camera")
        return settings
    heading = np.degrees(motion_heading(np.asarray(qpos[:, :2], dtype=np.float64)))
    settings["type"] = "tracking"          # theirs follows the body horizontally
    settings["distance"] = float(video_cam["distance"])
    settings["elevation"] = float(video_cam["elevation"])
    settings["lookat_height"] = float(video_cam.get("lookat_z",
                                                   settings.get("lookat_height", 0.9)))
    settings["azimuth_offset"] = float(video_cam["azimuth"]) - float(heading)
    print(f"  camera from video.json: distance={settings['distance']} "
          f"elevation={settings['elevation']} azimuth={video_cam['azimuth']} "
          f"-> azimuth_offset={settings['azimuth_offset']:.1f} "
          f"(heading {heading:.1f} deg)")
    return settings


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_source(cfg: dict, label: str, qpos: np.ndarray, out_dir: Path,
                  xml_path: Path, why: str, spec: dict,
                  use_video_camera: bool) -> tuple[list, Path]:
    """One recorded trajectory -> strobe PNG + footage for the compare video."""
    strobe_cfg = cfg.get("strobe", {})
    render_cfg = cfg.get("render", {})
    start, end = strobe_window(strobe_cfg, len(qpos))
    camera = camera_settings(cfg, spec, qpos, start, end, use_video_camera)

    print(f"[{label}] replaying qpos trajectory {tuple(qpos.shape)} "
          f"({len(qpos)} frames)  model: {xml_path} [{why}]")
    with StrobeStudio(
        str(xml_path),
        width=int(render_cfg.get("width", 1280)),
        height=int(render_cfg.get("height", 720)),
        start_color=strobe_cfg.get("start_color", [0.98, 0.55, 0.25, 1.0]),
        final_color=strobe_cfg.get("final_color", [0.16, 0.30, 0.65, 1.0]),
        middle_color=strobe_cfg.get("middle_color"),
        camera=camera,
        white_background=bool(render_cfg.get("white_background", True)),
    ) as studio:
        # orbit planning follows camera.plan_window: "strobe" frames only, so a
        # long clip cannot drag the camera away from the poses captured.
        studio.plan_camera(np.asarray(qpos[:, :2], dtype=np.float64),
                           window=(start, end))
        burn_in = bool(render_cfg.get("burn_in_frame_index", False))
        video_frames, strobe_frames, background, steps = capture_source(
            studio, qpos, strobe_cfg, label=label, burn_in=burn_in)
        composite = composite_strobe(
            strobe_frames,
            background,
            threshold=float(strobe_cfg.get("threshold", 5.0)),
            shift_px=int(strobe_cfg.get("shift_px", 14)),
            x_offset_px=int(strobe_cfg.get("x_offset_px") or 0),
            crop_floor=bool(strobe_cfg.get("crop_floor", True)),
            crop_floor_frac=float(strobe_cfg.get("crop_floor_frac", 0.12)),
        )

    task = cfg.get("task_name", "task")
    cam = camera.get("type", "orbit_auto")
    png_path = out_dir / f"{task}_{cam}_{label}_strobe.png"
    imageio.imwrite(png_path, composite)
    print(f"  [{label}] {len(strobe_frames)} ghost poses at steps {steps}")
    print(f"  [{label}] strobe figure -> {png_path}")
    return video_frames, png_path


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("clip", nargs="?", default=None,
                        help="folder holding the trajectory dumps, e.g. "
                             "outputs/heavy_arm/trajectories/headstand_4 "
                             "(default: the YAML's replay.clip)")
    parser.add_argument("--config", action="append", default=None, metavar="PATH",
                        help="YAML config; repeat to layer overrides (last wins)")
    parser.add_argument("--task-name", help="name used in output file names "
                                            "(default: the clip folder name)")
    parser.add_argument("--sources", help="comma separated sources: labels "
                                          "(adult,z0,zstar,body) or label=file.npz "
                                          "entries (default adult,z0,zstar, "
                                          "overridden by the YAML sources block)")
    parser.add_argument("--models-root", help="folder that holds models/ "
                                              "(default: nearest parent that does)")
    parser.add_argument("--video-camera", action="store_true",
                        help="aim the camera with video.json's recorded "
                             "distance/elevation/azimuth instead of the YAML orbit")
    parser.add_argument("--burn-in", dest="burn_in", action="store_true", default=None,
                        help="stamp each video panel with its step index")
    parser.add_argument("--no-burn-in", dest="burn_in", action="store_false", default=None)
    parser.add_argument("--camera-type", choices=["orbit_auto", "tracking", "fixed"])
    parser.add_argument("--distance", type=float)
    parser.add_argument("--elevation", type=float)
    parser.add_argument("--azimuth-offset", type=float)
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--shift-px", type=int)
    parser.add_argument("--x-offset-px", type=int,
                        help="slide the whole ghost trail; 0 = trail centred on "
                             "the camera (default), positive = right")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--crop-floor", type=lambda s: s.lower() not in ("0", "false", "no"))
    parser.add_argument("--fps", type=float, help="video fps (default: the dump's fps)")
    parser.add_argument("--out-dir")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="generic config override, e.g. strobe.interval=25")
    parser.add_argument("--save-config", metavar="PATH",
                        help="write the resolved config here, then reuse it with "
                             "--config base.yaml --config this.yaml")
    return parser


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Named flags beat the YAML; ``--set`` beats the named flags.

    Keys match ``render_strobe.py`` so a tuned figure re-renders with the same
    command line in both scripts.
    """
    named = {
        "camera.type": args.camera_type,
        "camera.distance": args.distance,
        "camera.elevation": args.elevation,
        "camera.azimuth_offset": args.azimuth_offset,
        "strobe.start_step": args.start_step,
        "strobe.end_step": args.end_step,
        "strobe.interval": args.interval,
        "strobe.shift_px": args.shift_px,
        "strobe.x_offset_px": args.x_offset_px,
        "strobe.threshold": args.threshold,
        "strobe.crop_floor": args.crop_floor,
        "render.burn_in_frame_index": args.burn_in,
        "render.fps": args.fps,
        "render.out_dir": args.out_dir,
    }
    for key, value in named.items():
        if value is not None:
            deep_set(cfg, key, value)
    for item in args.set or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        deep_set(cfg, key.strip(), yaml.safe_load(raw))


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config or ["configs/render_strobe.yaml"])
    apply_cli_overrides(cfg, args)
    if args.save_config:
        save_config(cfg, args.save_config)

    clip_from_cfg = (cfg.get("replay") or {}).get("clip")
    clip_arg = args.clip or clip_from_cfg
    if not clip_arg:
        raise SystemExit("no clip folder: pass it as an argument or set "
                         "replay.clip in the config")
    clip_dir = resolve_path(clip_arg)
    if not clip_dir.is_dir():
        raise SystemExit(f"not a clip folder: {clip_dir}")
    spec = load_video_spec(clip_dir)
    body = body_name(clip_dir, spec)
    root = (resolve_path(args.models_root) if args.models_root
            else models_root(clip_dir, body))

    specs = source_specs(cfg, args)
    labels = [s["label"] for s in specs]

    cfg.setdefault("task_name", clip_dir.name)
    if args.task_name:
        cfg["task_name"] = args.task_name
    out_dir = resolve_path(cfg.get("render", {}).get("out_dir", "outputs/figures/strobe"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"clip={clip_dir}  task={cfg['task_name']}  body={body}  "
          f"camera={cfg.get('camera', {}).get('type')}  "
          f"sources={'|'.join(labels)}")

    footage: dict[str, list] = {}
    figures: dict[str, Path] = {}
    order: list[str] = []
    fps: float | None = None
    for entry in specs:
        label, traj_file = entry["label"], entry["qpos"]
        qpos, dump_fps = load_trajectory(clip_dir / traj_file)
        if fps is None and dump_fps:
            fps = dump_fps
        xml_path, why = model_for(label, traj_file, clip_dir, spec, body, root,
                                  cfg, forced=entry["xml"])
        footage[label], figures[label] = render_source(
            cfg, label, qpos, out_dir, xml_path, why, spec,
            bool(args.video_camera))
        order.append(label)

    labels = [s["label"] for s in specs]
    panels = [footage[label] for label in labels]
    n = min(len(panel) for panel in panels)
    if len({len(panel) for panel in panels}) > 1:
        lengths = {label: len(panel) for label, panel in zip(labels, panels, strict=True)}
        print(f"video: panels differ in length {lengths} -> keeping the first {n} "
              "frames; the burn-in shows each panel's own step index")
    side = [np.concatenate([panel[i] for panel in panels], axis=1)
            for i in range(n)]
    task = cfg["task_name"]
    cam = cfg.get("camera", {}).get("type", "orbit_auto")
    mp4_path = out_dir / f"{task}_{cam}_compare_{'_'.join(labels)}.mp4"
    video_fps = float(cfg.get("render", {}).get("fps") or fps or 30.0)
    imageio.mimsave(mp4_path, side, fps=video_fps)
    print(f"side-by-side video ({' | '.join(labels)}, {video_fps:g} fps) -> {mp4_path}")

    print(f"\nAll outputs in {out_dir}")
    print(f"view the figure:   xdg-open {figures[labels[-1]]}")
    print(f"play the video:    xdg-open {mp4_path}")


if __name__ == "__main__":
    main()
