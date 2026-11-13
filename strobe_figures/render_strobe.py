#!/usr/bin/env python
"""Strobe figure generator for qualitative humanoid figures.

Renders two artifacts from one motion task and up to three motion sources
(optional original adult motion, nominal z0, adapted z*):

  1. ``<task>_<camera>_compare_<labels>.mp4`` -- side-by-side video, one panel
     per configured source, same camera and same step window. Each panel is
     badged with its own step index (``render.burn_in_frame_index``, green when
     the step is one the strobe window captures) so you can read frame numbers
     straight off the footage and feed them back as start_step / end_step /
     interval. The PNGs are never badged.
  2. ``<task>_<camera>_<source>_strobe.png`` -- multi-pose strobe figure: ghost
     poses shifted across a clean white plate, one PNG per source.

The ``baseline:`` block adds the untouched source motion -- the adult skeleton
replaying the default latent, from ``data/origin_motion`` -- as the left-most
panel and its own figure, so a comparison reads original adult | z0 on the
target body | z* after search. It renders on its own model (``baseline.xml``,
e.g. ``assets/robots/adult/robot.xml``) and adult clips run longer than child
rollouts, so ``strobe.end_step: null`` lets every source use its full window.

Configuration lives in ``configs/render_strobe.yaml`` and every key can be
overridden on the CLI (``--camera-type``, ``--shift-px``, ... or generic
``--set key=value``). Nothing about the model, the latents or the camera is
hardcoded here.

Each source is resolved from, in order of preference:
  * ``qpos``   -- a trajectory .npz (e.g. an evaluation clip's ``best.npz`` /
    ``origin_z.npz``) replayed kinematically, or
  * ``latent`` -- a z vector (.npy / .npz / inline) rolled out through the
    frozen Metamotivo actor.

Rendering is fully decoupled from the physics: ``tools/strobe_wrapper.py`` colors
robot geoms around each render call and restores the defaults afterwards. The
library's base ``env.py`` is never modified.

Example (one evaluation clip)::

    uv run render_strobe.py --config configs/render_strobe.yaml \
        --nominal-qpos outputs/my_search/jump-2_2/origin_z.npz \
        --adapted-qpos outputs/my_search/jump-2_2/best.npz \
        --set strobe.end_step=null
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless machines have no X11 display: use EGL for offscreen rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.strobe_wrapper import (  # noqa: E402
    StrobeRenderWrapper,
    capture_source,
    composite_strobe,
    strobe_window,
)

Z_RADIUS = 16.0                      # |z| = sqrt(256) for every latent here
_Z_KEYS = ("z", "best_z", "latent")  # npz keys that may hold a latent
_QPOS_KEYS = ("qpos",)


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------

def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay one mapping on another (overlay wins)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | list[str | Path]) -> dict:
    """Read one YAML config, or layer several: later files override earlier ones.

    Keeps a single full base config plus small per-body / per-task overlays,
    so variants never drift the way copied files do.
    """
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if not paths:
        raise SystemExit("load_config needs at least one config path")
    cfg: dict = {}
    for one in paths:
        try:
            with open(one) as fh:
                loaded = yaml.safe_load(fh) or {}
        except OSError as exc:
            raise SystemExit(f"cannot read config {one}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SystemExit(f"{one} did not parse into a mapping")
        deep_merge(cfg, loaded)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write the resolved config so a tuned figure can be re-created later."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "w") as fh:
            fh.write("# resolved by render_strobe.py: base config + overlays + "
                     "command line\n")
            yaml.safe_dump(cfg, fh, sort_keys=False)
    except OSError as exc:
        raise SystemExit(f"cannot write config {out}: {exc}") from exc
    print(f"resolved config saved -> {out}")


def deep_set(cfg: dict, dotted_key: str, value) -> None:
    """Generic override, e.g. ``--set strobe.interval=25``."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Named flags beat the YAML; ``--set`` beats the named flags."""
    named = {
        "task_name": args.task_name,
        "xml_path": args.xml,
        "seed": args.seed,
        "nominal_latent": args.nominal_latent,
        "adapted_latent": args.adapted_latent,
        "nominal_qpos": args.nominal_qpos,
        "adapted_qpos": args.adapted_qpos,
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
        "baseline.label": args.baseline_label,
        "baseline.qpos": args.baseline_qpos,
        "baseline.latent": args.baseline_latent,
        "baseline.xml": args.baseline_xml,
        "render.burn_in_frame_index": args.burn_in,
        "rollout.steps": args.steps,
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


# ---------------------------------------------------------------------------
# sources: latents and qpos trajectories
# ---------------------------------------------------------------------------

def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def load_latent(spec, project: bool) -> np.ndarray:
    """``spec`` is a .npy/.npz path or an inline list; returns shape (256,)."""
    if isinstance(spec, (list, tuple)):
        z = np.asarray(spec, dtype=np.float64).ravel()
    else:
        path = resolve_path(spec)
        if path.suffix == ".npy":
            z = np.load(path)
        elif path.suffix == ".npz":
            data = np.load(path)
            key = next((k for k in _Z_KEYS if k in data), None)
            if key is None:
                raise KeyError(f"{path} holds none of {_Z_KEYS}; keys={list(data)}")
            z = data[key]
        else:
            raise ValueError(f"latent must be .npy/.npz or a list, got {spec!r}")
        z = np.asarray(z, dtype=np.float64).ravel()
    if project:
        z = z / np.linalg.norm(z) * Z_RADIUS
    return z


def load_qpos(spec) -> np.ndarray | None:
    """``(T, nq)`` trajectory from a .npz holding qpos, else None."""
    if spec is None or isinstance(spec, (list, tuple)):
        return None
    path = resolve_path(spec)
    if path.suffix != ".npz":
        return None
    data = np.load(path)
    key = next((k for k in _QPOS_KEYS if k in data), None)
    return np.asarray(data[key], dtype=np.float64) if key is not None else None


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def make_env(cfg: dict, start_color, final_color, middle_color, xml=None):
    from humenv import make_humenv

    render_cfg = cfg.get("render", {})
    env, _ = make_humenv(
        num_envs=1,
        task=None,
        xml=str(resolve_path(xml or cfg["xml_path"])),
        state_init="Default",
        render_mode="rgb_array",
        render_width=int(render_cfg.get("width", 1280)),
        render_height=int(render_cfg.get("height", 720)),
    )
    camera_cfg = dict(cfg.get("camera", {}))
    for stray in ("start_color", "final_color", "middle_color"):
        camera_cfg.pop(stray, None)
    wrapped = StrobeRenderWrapper(
        env,
        start_color=start_color,
        final_color=final_color,
        middle_color=middle_color,
        camera=camera_cfg,
        white_background=bool(render_cfg.get("white_background", True)),
    )
    return wrapped


# ---------------------------------------------------------------------------
# motion sources
# ---------------------------------------------------------------------------

def rollout_qpos(cfg: dict, z: np.ndarray) -> np.ndarray:
    """Deterministic actor rollout under one latent; returns ``(T, nq)``."""
    from metamotivo.fb_cpr.huggingface import FBcprModel

    rollout_cfg = cfg.get("rollout", {})
    steps = int(rollout_cfg.get("steps", 120))
    obs_scale = str(rollout_cfg.get("obs_scale", "none"))
    device = str(rollout_cfg.get("device", "cpu"))
    torch.set_num_threads(1)  # batch-1 actor: threads are pure overhead here

    model = FBcprModel.from_pretrained(
        cfg.get("metamotivo", "facebook/metamotivo-M-1")).to(device)
    model.eval()
    env = make_env(cfg, [0, 0, 0, 1], [0, 0, 0, 1], None)
    base = env.unwrapped

    env.reset(seed=int(cfg.get("seed", 0)))
    ref_qpos = load_qpos(cfg.get("reference_clip"))
    if cfg.get("init_from_reference") and ref_qpos is not None:
        base.set_physics(qpos=ref_qpos[0], qvel=np.zeros(base.model.nv))

    # actor observation of the current state (reverse-retargeted if requested)
    exact = None
    if obs_scale == "exact":
        from model.exact_obs import ExactObs
        exact = ExactObs(str(resolve_path(cfg["xml_path"])))

    def actor_obs() -> np.ndarray:
        if exact is not None:
            return exact(base.data.qpos, base.data.qvel).astype(np.float32)
        return np.asarray(base.get_obs()["proprio"], dtype=np.float32)

    z_t = torch.tensor(z[None], dtype=torch.float32, device=device)
    hist = []
    for _ in range(steps):
        obs_t = torch.tensor(actor_obs()[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.act(obs_t, z_t, mean=True).cpu().numpy().ravel()
        _, _, _, _, info = env.step(action)
        hist.append(np.asarray(info["qpos"]).copy())
    env.close()
    return np.stack(hist)


# ---------------------------------------------------------------------------
# rendering pipeline
# ---------------------------------------------------------------------------

def window_bounds(strobe_cfg: dict, total: int) -> tuple[int, int]:
    """Inclusive (start, end) steps of the strobe window for a ``total``-frame
    trajectory. ``end_step: null`` runs through the last frame of *this*
    source, so adult clips (longer than the child rollouts) keep their tail.
    """
    return strobe_window(strobe_cfg, total)


def capture(env: StrobeRenderWrapper, qpos: np.ndarray, strobe_cfg: dict,
            label: str | None = None, burn_in: bool = False):
    """Play a trajectory back kinematically; capture video + strobe frames.

    Thin alias for :func:`tools.strobe_wrapper.capture_source`, which is shared
    with ``replay_strobe.py`` (recorded dumps render through ``StrobeStudio``).
    Returns ``(video_frames, strobe_frames, background, steps)``.
    """
    return capture_source(env, qpos, strobe_cfg, label=label, burn_in=burn_in)


def render_source(cfg: dict, label: str, qpos: np.ndarray, out_dir: Path,
                  xml=None) -> tuple[list, Path]:
    """One motion source -> strobe PNG + footage for the comparison video."""
    strobe_cfg = cfg.get("strobe", {})
    env = make_env(
        cfg,
        strobe_cfg.get("start_color", [0.98, 0.55, 0.25, 1.0]),
        strobe_cfg.get("final_color", [0.16, 0.30, 0.65, 1.0]),
        strobe_cfg.get("middle_color"),
        xml=xml,
    )

    # camera orbit: look-ahead rollout -> root trajectory (x, y) -> heading.
    # Planning follows camera.plan_window: "strobe" (default) frames only, so a
    # long clip cannot drag the camera away from the poses actually captured.
    start, end = window_bounds(strobe_cfg, len(qpos))
    env.camera_ctl.plan_for_trajectory(np.asarray(qpos[:, :2], dtype=np.float64),
                                       start, end)

    burn_in = bool(cfg.get("render", {}).get("burn_in_frame_index", False))
    video_frames, strobe_frames, background, steps = capture(
        env, qpos, strobe_cfg, label=label, burn_in=burn_in)
    composite = composite_strobe(
        strobe_frames,
        background,
        threshold=float(strobe_cfg.get("threshold", 5.0)),
        shift_px=int(strobe_cfg.get("shift_px", 14)),
        x_offset_px=int(strobe_cfg.get("x_offset_px") or 0),
        crop_floor=bool(strobe_cfg.get("crop_floor", True)),
        crop_floor_frac=float(strobe_cfg.get("crop_floor_frac", 0.12)),
    )
    env.close()

    task = cfg.get("task_name", "task")
    cam = cfg.get("camera", {}).get("type", "orbit_auto")
    png_path = out_dir / f"{task}_{cam}_{label}_strobe.png"
    imageio.imwrite(png_path, composite)
    print(f"  [{label}] {len(strobe_frames)} ghost poses at steps {steps}")
    print(f"  [{label}] strobe figure -> {png_path}")
    return video_frames, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", action="append", default=None, metavar="PATH",
                        help="YAML config; repeat to layer overrides over the "
                             "base config (last file wins)")
    parser.add_argument("--task-name")
    parser.add_argument("--xml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nominal-latent", help="z0: .npy/.npz or inline vector")
    parser.add_argument("--adapted-latent", help="z*: .npy/.npz or inline vector")
    parser.add_argument("--nominal-qpos", help="z0 trajectory .npz (qpos inside)")
    parser.add_argument("--adapted-qpos", help="z* trajectory .npz (qpos inside)")
    parser.add_argument("--baseline-qpos",
                        help="original-motion trajectory .npz, e.g. data/origin_motion/...")
    parser.add_argument("--baseline-latent",
                        help="original-motion latent .npy/.npz (rollout instead of replay)")
    parser.add_argument("--baseline-xml",
                        help="model for the baseline panel, e.g. assets/robots/adult/robot.xml")
    parser.add_argument("--baseline-label")
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
    parser.add_argument("--steps", type=int, help="rollout length for latent sources")
    parser.add_argument("--out-dir")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="generic config override, e.g. strobe.interval=25")
    parser.add_argument("--save-config", metavar="PATH",
                        help="write the resolved config here, then reuse it with "
                             "--config base.yaml --config this.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config or ["configs/render_strobe.yaml"])
    apply_cli_overrides(cfg, args)
    if args.save_config:
        save_config(cfg, args.save_config)

    out_dir = resolve_path(cfg.get("render", {}).get("out_dir", "outputs/figures/strobe"))
    out_dir.mkdir(parents=True, exist_ok=True)
    project = bool(cfg.get("project_latent", False))

    print(f"task={cfg.get('task_name')}  xml={cfg.get('xml_path')}  "
          f"camera={cfg.get('camera', {}).get('type')}  seed={cfg.get('seed', 0)}")

    src_specs = []
    baseline = cfg.get("baseline") or {}
    if baseline.get("qpos") is not None or baseline.get("latent") is not None:
        src_specs.append({
            "label": str(baseline.get("label") or "adult"),
            "qpos": baseline.get("qpos"),
            "latent": baseline.get("latent"),
            "xml": baseline.get("xml"),
        })
    src_specs.append({"label": "z0", "qpos": cfg.get("nominal_qpos"),
                      "latent": cfg.get("nominal_latent"), "xml": None})
    src_specs.append({"label": "zstar", "qpos": cfg.get("adapted_qpos"),
                      "latent": cfg.get("adapted_latent"), "xml": None})

    footage: dict[str, list] = {}
    figures: dict[str, Path] = {}
    order: list[str] = []
    for spec in src_specs:
        label = spec["label"]
        qpos = load_qpos(spec["qpos"])
        if qpos is None:
            if spec["latent"] is None:
                raise SystemExit(f"no qpos or latent source configured for {label!r}")
            z = load_latent(spec["latent"], project)
            print(f"[{label}] rollout of z ({z.shape[0]}-D, |z|={np.linalg.norm(z):.2f}, "
                  f"{cfg.get('rollout', {}).get('steps', 120)} steps)")
            qpos = rollout_qpos(cfg, z)
        else:
            print(f"[{label}] replaying qpos trajectory {tuple(qpos.shape)} from {spec['qpos']}")
            if spec["xml"]:
                print(f"[{label}] model: {spec['xml']}")
        footage[label], figures[label] = render_source(
            cfg, label, qpos, out_dir, xml=spec["xml"])
        order.append(label)

    labels = order
    panels = [footage[label] for label in labels]
    n = min(len(panel) for panel in panels)
    if len({len(panel) for panel in panels}) > 1:
        lengths = {label: len(panel) for label, panel in zip(labels, panels, strict=True)}
        print(f"video: panels differ in length {lengths} -> keeping the first {n} "
              "frames; the burn-in shows each panel's own step index")
    side = [np.concatenate([panel[i] for panel in panels], axis=1)
            for i in range(n)]
    task = cfg.get("task_name", "task")
    cam = cfg.get("camera", {}).get("type", "orbit_auto")
    mp4_path = out_dir / f"{task}_{cam}_compare_{'_'.join(labels)}.mp4"
    imageio.mimsave(mp4_path, side,
                    fps=float(cfg.get("render", {}).get("fps", 30)))
    print(f"side-by-side video ({' | '.join(labels)}) -> {mp4_path}")

    print(f"\nAll outputs in {out_dir}")
    print(f"view the figure:   xdg-open {figures[labels[-1]]}")
    print(f"play the video:    xdg-open {mp4_path}")


if __name__ == "__main__":
    main()
