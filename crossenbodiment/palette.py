"""palette.py -- every colour the figures, videos and Blender renders use, in one table (OOP refactor step 4; step 10: Palette).

Each colour is stored in the exact spelling its users wrote (a hex string for matplotlib, sRGB 0-255 ints for PIL, sRGB 0-1 floats for
render_blender.py's COLORS -- which converts them to linear itself --, RGBA 0-1 lists for the MuJoCo videos, whose specs go through
json.dumps, so the spelling is part of the output). Nothing is derived by rounding: where two forms of one colour were rounded
differently by their scripts (Blender's z0 (0.55, 0.545, 0.53) for #8c8b87), both are kept as written; tests/test_vis.py checks that
the forms of one entry are the same colour within that rounding, that render_blender.py's COLORS equal BLENDER, and that every
script's colour table equals its 8e9dc99 literals.

  Palette.method(key[, variant])   MethodColour(key, label, hex, video, blender): the method table, keys ref, adult, z0, cem
                                   (BFM-Zero: CEM on PHC's reward), loes (ours), random, cmaes, lso, lsor, ta, native
  Palette.label / hex / rgb / video / blender_rgb(key[, variant])
  VARIANTS                         where scripts disagree today, each script's colour / label under a named variant (not unified:
                                   the user decides, see DISAGREEMENTS)
  ACCENTS, CONDITIONS              body accents (the payload's amber weight, the limited leg's locked red leg, the G1 drawn black),
                                   the teaser's condition colours
  INK                              the ink token sets of the figure families (text, muted text, rules, grid, surface, video bands)
  BLENDER                          render_blender.py's COLORS (sRGB 0-1; render_blender.py stays a standalone copy of them)
The survey videos' three colours stay importable as GREY, RED, BLUE (scripts/survey_body_videos.py's names).
"""
from __future__ import annotations

import types
from dataclasses import dataclass


def _ro(d=(), **kw):
    return types.MappingProxyType({**dict(d), **kw})


# ------------------------------------------------------------------ the MuJoCo videos (RGBA 0-1, list spelling as json.dumps writes it)
GREY, RED, BLUE = [0.55, 0.57, 0.62, 1.0], [0.85, 0.22, 0.20, 1.0], [0.18, 0.42, 0.85, 1.0]   # robot_overlay.COLORS, grey opaque
GHOST = [0.55, 0.57, 0.62, 0.35]                              # the overlay's translucent reference (package_survey.py, robot_overlay.py)
GREEN, ORANGE = [0.20, 0.65, 0.30, 1.0], [0.95, 0.60, 0.10, 1.0]   # robot_overlay.py's 4th and 5th trajectory
OVERLAY = [tuple(GHOST), tuple(RED), tuple(BLUE), tuple(GREEN), tuple(ORANGE)]   # robot_overlay.COLORS (tuples), body_overlay.py's too

# ------------------------------------------------------------------ render_blender.py's COLORS (sRGB 0-1; the renderer linearises them)
BLENDER = _ro(body=(0.47, 0.61, 0.70), red=(0.78, 0.15, 0.13), amber=(0.90, 0.60, 0.06), black=(0.09, 0.09, 0.10),
              ref=(0.10, 0.10, 0.10), z0=(0.55, 0.545, 0.53), bfmzero=(0.106, 0.686, 0.478), ours=(0.165, 0.471, 0.839))
BLENDER_BG = (0.737, 0.737, 0.737)                            # #bcbcbc, what render_blender.py's grey-floor poses fade towards

# ------------------------------------------------------------------ ink tokens of the figure families
INK = _ro(
    paper=_ro(ink="#22252a", muted="#62666e", rule="#a4a7ad", frozen="#e4e5e7", accent="#d2601a", accent_bg="#fbe9dc"),  # teaser, method figure
    bars=_ro(ink="#0b0b0b", muted="#52514e", grid="#e6e6e3", tint="#d4e4f7"),           # plot_child_improvement.py
    curves=_ro(axis="#52514e", grid="#e6e6e3", grid_all="#ececea"),                      # plot_child_convergence.py (+ child_topview's grid)
    dataviz=_ro(ink="#1f1e1c", muted="#6b6a64", grid="#e4e3de", surface="#fcfcfb"),     # plot_algo_compare / es_sampling / k_scale_bars ...
    dataviz2=_ro(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8983"),     # plot_anchor_pca / seed_pca / lowdim_trend ...
    video=_ro(band=(252, 252, 251), title=(11, 11, 11), sub=(82, 81, 78), legend=(60, 60, 58), ghost_fill=(200, 202, 208),
              ghost_edge=(90, 90, 95)),                                                   # the title / legend bands burnt into frames
    figure_bg="#bcbcbc",                                       # figure_bodies.BG: BFMTrack's grey behind the MuJoCo cut-outs
    figure_body=(0.467, 0.602, 0.683, 1),                      # figure_bodies.BODY_RGBA: ReActor's body colour
)
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")   # the dataviz categorical order the plot scripts use (1-5)


@dataclass(frozen=True)
class MethodColour:
    key: str
    label: str                       # the name in the paper figures / legends
    hex: str | None = None           # matplotlib (None: no plot draws it)
    video: tuple | list | None = None   # MuJoCo video RGBA 0-1, as the specs spell it
    blender: str | None = None       # render_blender.py COLORS key (BLENDER)
    note: str = ""

    @property
    def rgb(self):
        """sRGB 0-255 ints (PIL; topdown_blender.py's legend squares)"""
        return hex_rgb(self.hex)

    @property
    def blender_rgb(self):
        return BLENDER[self.blender]


METHODS = _ro({m.key: m for m in (
    MethodColour("ref", "Reference", "#1a1a1a", GREY, "ref", "the body's own reference (target): black lines / poses in the top views; "
                 "the grey ghost (GHOST) in the overlay videos"),
    MethodColour("adult", "Origin", None, GREY, "body", "the adult's original motion: the side-by-side videos' grey left panel; drawn "
                 "in the body colour in the strobe panels; 'reference (adult)' on the overview pages"),
    MethodColour("z0", r"$z_0$", "#8c8b87", RED, "z0", "Motivo's own latent: dashed grey line in the plots, red in the videos"),
    MethodColour("cem", "BFM-Zero", "#1baf7a", [0.11, 0.69, 0.48, 1.0], "bfmzero", "CEM on BFM-Zero's (PHC's) tracking reward; video "
                 "colour: outputs/tall_body_52/videos/_specs"),
    MethodColour("loes", "Ours", "#2a78d6", BLUE, "ours", "LOES; blue in the survey videos (BLUE), [0.16, 0.47, 0.84, 1.0] in "
                 "outputs/tall_body_52/videos (VARIANTS)"),
    MethodColour("random", "Random", "#eb6834", None, None, "random search (plot_child_convergence.py)"),
    MethodColour("cmaes", "CMA-ES", "#eda100", None, None, "pycma (plot_child_convergence.py)"),
    MethodColour("lso", "LSO", None, BLUE, None, "BFMTrack's latent-sequence optimisation: the searched (blue) panel of lso_videos.py "
                 "and package_amass_lso_survey.py"),
    MethodColour("lsor", "LSO with ranks", None, BLUE, None, "lso_videos.py's title for a --rank run; no plot colour"),
    MethodColour("ta", "Trajectory Adaptation (BFM-Zero)", None, None, None, "no figure or video colours it (tables only)"),
    MethodColour("native", "Motivo zero-shot tracking (window L=8)", None, RED, None, "Motivo's ER L = 8 rollout: the red panel of "
                 "lso_videos.py, package_amass_lso_survey.py's zero_shot.npz"),
)})

# (key, variant) -> MethodColour: a script's own colour / label where it disagrees with the table above (kept, not unified)
VARIANTS = _ro({
    ("cem", "algo_compare"): MethodColour("cem", "CEM", "#eb6834", note="plot_algo_compare.py: CEM with our cost (cem_ours), orange"),
    ("random", "algo_compare"): MethodColour("random", "Random search", "#1baf7a", note="plot_algo_compare.py: random search green"),
    ("loes", "algo_compare"): MethodColour("loes", "ES (ours)", "#2a78d6", note="plot_algo_compare.py's label"),
    ("z0", "k_scale"): MethodColour("z0", "z0", "#eb6834", note="plot_k_scale_bars.py: z0 orange (C_Z0), not grey"),
    ("loes", "k_scale"): MethodColour("loes", "searched", "#2a78d6", note="plot_k_scale_bars.py C_BEST"),
    ("cem", "video_tall"): MethodColour("cem", "CEM (BFM-Zero)", "#1baf7a", [0.11, 0.69, 0.48, 1.0],
                                        note="outputs/tall_body_52/videos/_specs (spec files only, no script)"),
    ("loes", "video_tall"): MethodColour("loes", "LOES (ours)", "#2a78d6", [0.16, 0.47, 0.84, 1.0],
                                         note="outputs/tall_body_52/videos/_specs: not the survey's BLUE"),
    ("z0", "plain"): MethodColour("z0", "z0", "#8c8b87", None, "z0", "topdown_blender.py's legend / the pages: plain 'z0'"),
    ("adult", "pages"): MethodColour("adult", "reference (adult)", note="the overview pages' column titles"),
    ("cem", "pages"): MethodColour("cem", "CEM (BFM-Zero)", note="child_all_blender.py's column title"),
    ("loes", "pages"): MethodColour("loes", "LOES (ours)", note="child_all_blender.py's column title"),
    ("loes", "pages_best"): MethodColour("loes", "LOES (best latent)", note="export_strobe / body_all / payload_fall pages"),
})

# where scripts disagree today (each keeps its own colour through VARIANTS; for the user to decide)
DISAGREEMENTS = (
    "cem: green #1baf7a (BFM-Zero: plot_child_convergence, child_topview, topdown_blender, render_blender bfmzero) vs orange "
    "#eb6834 (plot_algo_compare.py, CEM with our cost); video green [0.11, 0.69, 0.48] (tall_body_52 specs)",
    "random: orange #eb6834 (plot_child_convergence) vs green #1baf7a (plot_algo_compare)",
    "z0: grey #8c8b87 (plots, top views; Blender z0 (0.55, 0.545, 0.53)) vs orange #eb6834 (plot_k_scale_bars) vs red "
    "[0.85, 0.22, 0.20] (every video)",
    "loes: video blue [0.18, 0.42, 0.85] (survey, lso videos) vs [0.16, 0.47, 0.84] (tall_body_52 videos, = #2a78d6); Blender: the "
    "single body colour (0.47, 0.61, 0.70) in the strobe panels vs 'ours' (0.165, 0.471, 0.839) in the top views",
    "locked leg / payload accents: matplotlib #d6322e / #e39a00 (plot_teaser) vs Blender (0.78, 0.15, 0.13) / (0.90, 0.60, 0.06)",
    "labels: CEM 'BFM-Zero' (plots) / 'CEM (BFM-Zero)' (pages, video titles) / 'CEM' (plot_algo_compare); LOES 'Ours' / "
    "'LOES (ours)' / 'LOES (best latent)' / 'ES (ours)'; z0 '$z_0$' / 'z0'",
)

# body accents: the payload's hand weight, the limited leg's locked right leg, the G1 in the strobe figures
ACCENTS = _ro(
    payload=_ro(hex="#e39a00", blender="amber", note="plot_teaser.py AMBER; the drawn-only weight sphere on the right hand"),
    locked_leg=_ro(hex="#d6322e", blender="red", bodies=("R_Hip", "R_Knee", "R_Ankle", "R_Toe"), note="plot_teaser.py RED"),
    g1_strobe=_ro(blender="black", note="strobe_final_blender.py: the G1 drawn black (user, 11/10)"),
)
# the teaser's condition colours (plot_teaser.py PANELS): frame edges and search paths
CONDITIONS = _ro(payload="#e39a00", joint="#d6322e", child="#2a78d6", g1="#1baf7a")


def hex_rgb(h):
    """'#rrggbb' -> (r, g, b) ints 0-255"""
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def hex_rgba01(h):
    """'#rrggbb' -> (r, g, b, 1) floats 0-1, as plot_teaser.py's to_rgba (h / 255)"""
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (1,)


def srgb_to_linear(c):
    """render_blender.py's conversion of its sRGB COLORS (for checks; the renderer keeps its own copy)"""
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


class Palette:
    """lookups over METHODS / VARIANTS: Palette.hex("cem") -> '#1baf7a'; Palette.hex("cem", "algo_compare") -> '#eb6834'"""
    methods, variants, accents, ink, blender, conditions = METHODS, VARIANTS, ACCENTS, INK, BLENDER, CONDITIONS

    @staticmethod
    def method(key, variant=None):
        return METHODS[key] if variant is None else VARIANTS[(key, variant)]

    @classmethod
    def label(cls, key, variant=None):
        return cls.method(key, variant).label

    @classmethod
    def hex(cls, key, variant=None):
        return cls.method(key, variant).hex

    @classmethod
    def rgb(cls, key, variant=None):
        return cls.method(key, variant).rgb

    @classmethod
    def video(cls, key, variant=None):
        return cls.method(key, variant).video

    @classmethod
    def blender_rgb(cls, key, variant=None):
        return cls.method(key, variant).blender_rgb
