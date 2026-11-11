"""figstyle.py -- matplotlib styles of the figures and the paper's 0.01-inch drawing canvas (OOP refactor step 10).

FigureStyle holds one set of rcParams:
  FigureStyle.paper(rc)             STIX (Times-like, as the paper's body text), stix math, TrueType fonts in the PDF (fonttype 42) and,
                                    unless svg=False, text kept as text in the SVG: the teaser, the method figure, the bands
  FigureStyle.notebook(rc)          DejaVu Sans: the 52-panel overview grids
  style.updated(rc)                 another style on top of this one (the old scripts' successive plt.rcParams.update calls)
  with style: ...                   applied for the block (matplotlib.rc_context) and restored after it
  style.apply()                     applied for good (plt.rcParams.update), as the scripts did before step 10
  FigureStyle.canvas(W, y1, y0=0)   the paper figures' canvas: a figure W x (y1 - y0) units of 0.01 in (W = 550: the 5.5 in single
                                    column), one axes filling it in figure units, axis off -> (fig, ax)
  FigureStyle.save(fig, stem, formats=("pdf", "svg", "png"), dpi=600, **kw)   every format of one figure (dpi may be a dict per format)
The figures' drawing helpers are plain functions on an axes (the scripts bind their own sizes / colours with functools.partial):
  txt(ax, x, y, s, size, color, ...)        text centred on (x, y) at a z-order
  arrow(ax, pts, color, ...)                a polyline whose last segment carries the head (FancyArrowPatch)
  box(ax, x0, y0, x1, y1, ec, ...)          a rounded, unfilled frame
Every call reproduces the scripts' matplotlib calls in their order and keyword spelling (the figures stay byte-identical up to the
PDF / SVG dates: tests/golden/verify_vis.py).
"""
from __future__ import annotations

PAPER_RC = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"}
NOTEBOOK_RC = {"font.family": "DejaVu Sans"}


class FigureStyle:
    def __init__(self, rc=None):
        self.rc = dict(rc or {})
        self._ctx = None

    @classmethod
    def paper(cls, rc=None, svg=True):
        """the paper figures' fonts + rc (further rcParams); svg=False leaves svg.fonttype at matplotlib's default
        (plot_child_convergence.py never set it)"""
        out = dict(PAPER_RC)
        if not svg:
            del out["svg.fonttype"]
        out.update(rc or {})
        return cls(out)

    @classmethod
    def notebook(cls, rc=None):
        return cls({**NOTEBOOK_RC, **(rc or {})})

    def updated(self, rc):
        return FigureStyle({**self.rc, **dict(rc)})

    def apply(self):
        import matplotlib.pyplot as plt
        plt.rcParams.update(self.rc)
        return self

    def __enter__(self):
        import matplotlib
        self._ctx = matplotlib.rc_context(self.rc)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        ctx, self._ctx = self._ctx, None
        return ctx.__exit__(*exc)

    @staticmethod
    def canvas(W, y1, y0=0):
        """plt.figure(figsize=(W / 100, (y1 - y0) / 100)), one axes [0, 0, 1, 1] with xlim (0, W), ylim (y0, y1), axis off"""
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(W / 100, (y1 - y0) / 100)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(y0, y1)
        ax.axis("off")
        return fig, ax

    @staticmethod
    def save(fig, stem, formats=("pdf", "svg", "png"), dpi=600, **kw):
        """fig.savefig(f"{stem}.{fmt}", **kw, dpi=...) for every format; dpi: a number for all, or {fmt: dpi} (missing: None)"""
        for f in formats:
            d = dpi.get(f) if isinstance(dpi, dict) else dpi
            fig.savefig(f"{stem}.{f}", **kw, dpi=d)


# ------------------------------------------------------------------ plain drawing functions on the 0.01-inch canvas

def txt(ax, x, y, s, size, color, ha="center", va="center", zorder=6, **k):
    """ax.text at (x, y): fontsize, colour, alignment and z-order in the scripts' keyword order"""
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=zorder, **k)


def arrow(ax, pts, color, lw=0.9, ls=None, style="-|>", line_z=3, head_z=3, mutation_scale=7):
    """the segments of pts but the last as lines (round caps), the last as a FancyArrowPatch with `style` (ls passed only if given,
    as the method figure did and the short version did not)"""
    from matplotlib.patches import FancyArrowPatch
    kw = {} if ls is None else dict(ls=ls)
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, **kw, solid_capstyle="round", zorder=line_z)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=mutation_scale, lw=lw, **kw, color=color, shrinkA=0,
                                 shrinkB=0, zorder=head_z))


def box(ax, x0, y0, x1, y1, ec, lw=0.9, z=1, r=6):
    """a rounded frame (no fill) from (x0, y0) to (x1, y1)"""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec=ec, lw=lw, zorder=z))
