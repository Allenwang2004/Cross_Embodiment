#!/usr/bin/env python3
"""latex_accept_changes.py -- the clean text of a tracked-changes LaTeX file (user, 11/13: additions in \\allen{...}, deletions in
\\allen{\\sout{...}}): drops every \\allen{\\sout{...}} and every line starting with "%DEL" (deleted display equations), unwraps every other \\allen{...}, removes the \\providecommand{\\allen} line,
then tidies the spaces and blank lines the deletions leave.
  usage: python3 scripts/latex_accept_changes.py docs/latex/introduction.tex [out.tex]   (default out: <name>_clean.tex)"""
import re, sys
from pathlib import Path


def arg_end(s, i):
    """index just past the brace group that starts at s[i] == '{'"""
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            d += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            d -= 1
            if d == 0:
                return j + 1
    raise ValueError(f"unbalanced braces from {i}")


def accept(s):
    out, i = [], 0
    while True:
        k = s.find("\\allen{", i)
        if k < 0:
            out.append(s[i:]); break
        out.append(s[i:k]); a = k + len("\\allen"); e = arg_end(s, a); inner = s[a + 1:e - 1]
        out.append("\x00" if inner.startswith("\\sout{") else accept(inner))      # \x00 marks a deletion
        i = e
    return "".join(out)


def main():
    src = Path(sys.argv[1]); dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_clean.tex")
    # "%DEL " lines: deleted source that \sout cannot strike (display equations), kept as comments in the tracked file; dropped here
    lines = [l for l in src.read_text().split("\n") if not l.startswith("\\providecommand{\\allen}") and not l.lstrip().startswith("%DEL")]
    s = accept("\n".join(lines))
    # a line that held only deletions disappears instead of leaving a blank line (which LaTeX would read as a paragraph break)
    s = "\n".join(l.replace("\x00", "") for l in s.split("\n") if not (l.strip() and not l.replace("\x00", "").strip()))
    s = "\n".join(l.rstrip() for l in s.split("\n"))
    s = re.sub(r"(?m)^[ \t]+\n", "\n", s); s = re.sub(r"\n{3,}", "\n\n", s); s = re.sub(r"(?<=\S)  +", " ", s)
    s = re.sub(r" ([.,;:])", r"\1", s); s = re.sub(r"(?m)^ (?=\S)", "", s)
    s = re.sub(r"(?m)^% (?=Tracked changes|Preamble needs|\(the deleted text)", "% [tracked version only] ", s)
    dst.write_text(f"% Clean text of {src.name} (tracked changes accepted by scripts/latex_accept_changes.py); edit the tracked file, then rerun.\n" + s)
    print(dst)


if __name__ == "__main__":
    main()
