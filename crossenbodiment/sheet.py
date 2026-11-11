"""sheet.py -- contact sheets and index pages of rendered panels (OOP refactor step 10).

ContactSheet(columns, col_width, label_h, font_size, row_height="first", quality=90): a white page, one titled column per
source and one captioned row per clip; every image is resized to the column width (its aspect kept). The page is laid out as the
overview pages were: the column titles at (j * col_width + 10, 8), each row's caption at (10, y + 6) in a band label_h high, the
images below it; a row is as high as its first image ("first") or its tallest ("max"). It replaces the copies in
export_strobe_blender.py (overview, archived), body_all_blender.py (pages), child_all_blender.py (pages) and
payload_fall_candidates.py (overview).
  sheet.add_row(caption, images)   images: PIL images or paths
  sheet.render() -> Image; sheet.save(path) (JPEG at `quality`)
  ContactSheet.pages(rows, per_page, ...)   the rows split into pages of per_page rows
hcat(images, gap=24)              images side by side, top-aligned, on white (strobe_final_blender.py / topdown_blender.py combine)
vcat(images)                      images stacked, left-aligned
legend_strip(width, items, font)  a 70 px white strip with a colour square and a label per item (topdown_blender.py)
write_index_md(path, title, intro, header, rows)   "# title", the intro paragraph, then one markdown table
"""
from __future__ import annotations

from pathlib import Path

FONT = "DejaVuSans.ttf"


def font(size, fallback=False):
    from PIL import ImageFont
    if not fallback:
        return ImageFont.truetype(FONT, size)
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _open(im):
    from PIL import Image
    return im if hasattr(im, "convert") and not isinstance(im, (str, Path)) else Image.open(im).convert("RGB")


class ContactSheet:
    def __init__(self, columns, col_width=640, label_h=40, font_size=22, row_height="first", quality=90, font_fallback=False,
                 n_columns=None):
        self.columns, self.Wc, self.L, self.row_height, self.quality = list(columns), col_width, label_h, row_height, quality
        self.ncols = n_columns if n_columns is not None else len(self.columns)
        self.font = font(font_size, font_fallback)
        self.rows = []

    def add_row(self, caption, images):
        ims = [_open(im) for im in images]
        self.rows.append((caption, [im.resize((self.Wc, round(self.Wc * im.height / im.width))) for im in ims]))
        return self

    def _h(self, row):
        return row[0].height if self.row_height == "first" else max(im.height for im in row)

    def render(self):
        from PIL import Image, ImageDraw
        Wc, L = self.Wc, self.L
        sheet = Image.new("RGB", (self.ncols * Wc, sum(self._h(r) for _, r in self.rows) + L * (len(self.rows) + 1)), "white")
        d = ImageDraw.Draw(sheet); y = L
        for j, lab in enumerate(self.columns):
            d.text((j * Wc + 10, 8), lab, fill=(0, 0, 0), font=self.font)
        for caption, row in self.rows:
            d.text((10, y + 6), caption, fill=(0, 0, 0), font=self.font)
            for j, im in enumerate(row):
                sheet.paste(im, (j * Wc, y + L))
            y += L + self._h(row)
        return sheet

    def save(self, path):
        im = self.render(); im.save(path, quality=self.quality)
        return im


def hcat(images, gap=24):
    from PIL import Image
    ims = [_open(im) for im in images]
    out = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), max(im.height for im in ims)), "white"); x = 0
    for im in ims:
        out.paste(im, (x, 0)); x += im.width + gap
    return out


def vcat(images):
    from PIL import Image
    ims = [_open(im) for im in images]
    out = Image.new("RGB", (max(im.width for im in ims), sum(im.height for im in ims)), "white"); y = 0
    for im in ims:
        out.paste(im, (0, y)); y += im.height
    return out


def legend_strip(width, items, fnt, height=70):
    """topdown_blender.py's legend: items (label, sRGB fill) left to right from x = 20"""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, height), "white"); dr = ImageDraw.Draw(im); x = 20
    for lab, rgb in items:
        dr.rectangle([x, 22, x + 46, 46], fill=rgb); dr.text((x + 58, 16), lab, fill=(0, 0, 0), font=fnt); x += 90 + int(dr.textlength(lab, font=fnt))
    return im


def write_index_md(path, title, intro, header, rows):
    """the overview pages' index.md: '# title', the intro, one table (header cells, a ---| per column, the rows as written)"""
    Path(path).write_text(f"# {title}\n\n{intro}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join(rows) + "\n")
