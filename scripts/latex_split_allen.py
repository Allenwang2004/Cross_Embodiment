#!/usr/bin/env python3
"""latex_split_allen.py -- make tracked-change LaTeX robust to how \\allen is defined (user, 11/13: with a \\textcolor-based \\allen,
an \\allen{...} group that spans paragraphs fails and its \\label is lost, giving "??"). Every top-level \\allen{...} (not a deletion)
that contains a blank line, a sectioning command, a \\label or a float is rewritten so that:
  - \\section / \\subsection / \\paragraph stay outside, with only their title in \\allen;
  - \\label lines and figure/table environments stay outside \\allen (captions keep their own marks);
  - each remaining paragraph is wrapped in its own \\allen{...}.
The accepted (clean) text is unchanged; check with latex_accept_changes.py.  usage: python3 scripts/latex_split_allen.py file.tex ..."""
import re, sys
from pathlib import Path


def group_end(s, i):                                   # s[i] == '{'
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\": d += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            d -= 1
            if d == 0: return j + 1
    raise ValueError(i)


FLOAT = re.compile(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", re.S)
HEAD = re.compile(r"\A\s*(?:(\\(?:sub)*section|\\paragraph)\{|\\label\{)")


def wrap(body):
    body = body.strip("\n")
    if not body.strip() or all(l.lstrip().startswith("%") or not l.strip() for l in body.split("\n")): return body
    return "\\allen{" + body + ("\n}" if body.split("\n")[-1].lstrip().startswith("%") else "}")


def split_unit(u):
    out = []
    while True:
        m = HEAD.match(u)
        if not m: break
        k = u.index("{", m.start(0) + (len(m.group(1)) if m.group(1) else 0)) if m.group(1) else u.index("{", m.start(0))
        e = group_end(u, k)
        if m.group(1):
            title = u[k + 1:e - 1]
            out.append(m.group(1) + "{" + (title if title.startswith("\\allen{") else "\\allen{" + title + "}") + "}")
        else:
            out.append(u[m.start(0):e].strip())
        u = u[e:]
    rest = wrap(u)
    return "\n".join(out + ([rest] if rest.strip() else []))


def transform(inner):
    floats = []
    def keep(m): floats.append(m.group(0)); return f"\x01{len(floats) - 1}\x01"
    t = FLOAT.sub(keep, inner)
    units = re.split(r"\n[ \t]*\n", t)
    res = []
    for u in units:
        parts = re.split(r"(\x01\d+\x01)", u)
        seg = []
        for p in parts:
            m = re.fullmatch(r"\x01(\d+)\x01", p)
            if m: seg.append(floats[int(m.group(1))])
            elif p.strip(): seg.append(split_unit(p))
        res.append("\n".join(seg))
    return "\n\n".join(x for x in res if x.strip())


def process(s):
    out, i = [], 0
    while True:
        k = s.find("\\allen{", i)
        if k < 0: out.append(s[i:]); break
        a = k + len("\\allen"); e = group_end(s, a); inner = s[a + 1:e - 1]
        needs = (not inner.startswith("\\sout{")) and (re.search(r"\n[ \t]*\n", inner) or re.search(r"\\(?:sub)*section\{|\\paragraph\{|\\label\{|\\begin\{(?:figure|table)", inner))
        out.append(s[i:k]); out.append(transform(inner) if needs else s[k:e]); i = e
    return "".join(out)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        p = Path(f); p.write_text(process(p.read_text())); print(f)
