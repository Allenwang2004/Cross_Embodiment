#!/usr/bin/env python3
"""compare.py BASE CAND -- are two golden runs (outputs/refactor_golden/<label>, or any two folders) bit-identical?

For every case of tests/golden/cases.txt, every file under BASE/<case> and CAND/<case> (recursively) is compared:
  .npz   same keys; per array the same dtype, the same shape and the same bytes (bit-identical: NaN payloads and -0.0 count)
  .npy   as one npz array
  .csv   the same header and the same text in every cell, except the wall-clock columns IGNORED_CSV_COLUMNS
  .json  the same structure, keys, types (int vs float vs bool) and values (floats exactly; NaN = NaN), except the wall-clock
         keys IGNORED_JSON_KEYS; strings are compared after replacing the repository root with "<REPO>" and
         "refactor_golden/<label>/" with "refactor_golden/<LABEL>/", so the label of the run and the checkout's location do not
         count but every path's spelling (relative or absolute, which file) does
  other  the same bytes
  ignored: text logs (*.log, *.txt, *.out), files starting with "." (the runner's .rc / .pgid), MUJOCO_LOG.TXT
The file sets must match and each case must have exited 0 (.rc) in both runs. metrics.json (tests/golden/metrics_golden.py) is
compared as a JSON file with no ignored keys. Within CAND, RESUME_PAIRS checks that a resumed search equals the straight run.
Prints a PASS/FAIL table and the first differences of every failing case; exit code 1 on any FAIL.
usage: uv run python tests/golden/compare.py BASE CAND [--cases a,b,...] [--max-diffs 5]"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
GOLDEN = REPO / "outputs/refactor_golden"
# wall-clock measurements, the only things allowed to differ between two runs of the same code:
IGNORED_JSON_KEYS = {"minutes"}        # summary.json of single_z_search / lso_search / robot_z_search / robot_lso_search: run time
IGNORED_CSV_COLUMNS = {"sec"}          # curve.csv of lso_search / robot_lso_search: seconds since the start, per iteration
IGNORED_SUFFIXES = (".log", ".txt", ".out")
# (case, sub-folder) pairs whose outputs must agree inside one run (CAND): robot_z_search.py --resume (half the budget, then the
# rest) against one straight run. Ignored: the keys that say it was resumed. KNOWN_RESUME: differences the snapshot code already
# has (found 11/11 in step 0; reported, not fixed, since step 0 changes no code). The search itself (z_trace, best_z, state.npz,
# best.npz, every cost in curve.csv) is bit-identical; only two derived records are not, when the best latent was found before the
# resume point: summary.json best.motion is recomputed from best.npz's float32 qpos (the straight run used the float64 rollout;
# ~1e-7 m), and curve.csv best_done is "" instead of "nan" (the best record rebuilt from summary.json has no "done").
# A pair PASSES if every other file and field is identical; the known differences are counted in the table.
RESUME_PAIRS = [(("g1_es", ""), ("g1_es_resume", "full"), {"resumed_from", "resumed_at", "resume_exact"})]
KNOWN_RESUME = ("summary.json.best.motion.", "column best_done:")


def run_dir(x):
    p = Path(x)
    return p if p.is_dir() else GOLDEN / x


def cases():
    out = []
    for line in (REPO / "tests/golden/cases.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(line.split("|", 1)[0])
    return out


def files(d):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and not any(s.startswith(".") for s in p.relative_to(d).parts)
                  and not p.name.endswith(IGNORED_SUFFIXES) and p.name != "MUJOCO_LOG.TXT")


class Norm:
    """string normalisation: repository root and run label"""
    def __init__(self, run):
        self.run = run.resolve(); self.label = run.name

    def __call__(self, s):
        for root in {str(REPO), str(REPO.resolve())}:
            s = s.replace(root, "<REPO>")
        return s.replace(f"refactor_golden/{self.label}/", "refactor_golden/<LABEL>/")


def first_diff(a, b):
    """where two same-shape arrays first differ (bitwise), and how much"""
    if a.dtype.kind in "fc":
        neq = ~((a == b) | (np.isnan(a) & np.isnan(b))) | (np.signbit(a) != np.signbit(b)) if a.dtype.kind == "f" else a != b
    else:
        neq = a != b
    idx = np.argwhere(np.atleast_1d(neq))
    n = int(np.atleast_1d(neq).sum())
    if not n:                                   # equal values, different bits (NaN payload): say so
        return "values equal but bytes differ (NaN payload?)"
    i = tuple(int(x) for x in idx[0]) if a.ndim else ()
    msg = f"{n} of {a.size} elements differ; first at {list(i)}: {a[i]!r} vs {b[i]!r}"
    if a.dtype.kind in "fiu":
        with np.errstate(invalid="ignore"):
            d = np.abs(a.astype(np.float64) - b.astype(np.float64)); msg += f"; max |diff| {np.nanmax(d):.3g}"
    return msg


def cmp_arrays(name, a, b, out):
    if a.dtype != b.dtype:
        out.append(f"{name}: dtype {a.dtype} vs {b.dtype}"); return
    if a.shape != b.shape:
        out.append(f"{name}: shape {a.shape} vs {b.shape}"); return
    if a.dtype.kind == "O":
        out.append(f"{name}: object array, cannot compare bitwise"); return
    if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
        out.append(f"{name}: {first_diff(a, b)}")


def cmp_npz(rel, fa, fb, out):
    A, B = np.load(fa, allow_pickle=False), np.load(fb, allow_pickle=False)
    if sorted(A.files) != sorted(B.files):
        out.append(f"{rel}: keys {sorted(set(A.files) - set(B.files))} only in BASE, {sorted(set(B.files) - set(A.files))} only in CAND")
    for k in sorted(set(A.files) & set(B.files)):
        cmp_arrays(f"{rel}[{k}]", A[k], B[k], out)


def cmp_json_obj(path, a, b, na, nb, out, ignore):
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} {a!r} vs {type(b).__name__} {b!r}"); return
    if isinstance(a, dict):
        ka, kb = set(a) - ignore, set(b) - ignore
        if ka != kb:
            out.append(f"{path}: keys {sorted(ka - kb)} only in BASE, {sorted(kb - ka)} only in CAND")
        for k in sorted(ka & kb, key=str):
            cmp_json_obj(f"{path}.{k}", a[k], b[k], na, nb, out, ignore)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            cmp_json_obj(f"{path}[{i}]", x, y, na, nb, out, ignore)
    elif isinstance(a, float):
        if not (a == b and math.copysign(1, a) == math.copysign(1, b)) and not (math.isnan(a) and math.isnan(b)):
            out.append(f"{path}: {a!r} vs {b!r} (diff {b - a:.3g})")
    elif isinstance(a, str):
        if na(a) != nb(b):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def cmp_json(rel, fa, fb, na, nb, out, ignore=IGNORED_JSON_KEYS):
    cmp_json_obj(rel, json.loads(Path(fa).read_text()), json.loads(Path(fb).read_text()), na, nb, out, ignore)


def cmp_csv(rel, fa, fb, out, ignore=IGNORED_CSV_COLUMNS):
    A, B = list(csv.reader(open(fa, newline=""))), list(csv.reader(open(fb, newline="")))
    if not A or not B or A[0] != B[0]:
        out.append(f"{rel}: header {A[:1]} vs {B[:1]}"); return
    if len(A) != len(B):
        out.append(f"{rel}: {len(A) - 1} vs {len(B) - 1} rows")
    keep = [j for j, h in enumerate(A[0]) if h not in ignore]
    for i, (ra, rb) in enumerate(zip(A[1:], B[1:])):
        if len(ra) != len(rb):
            out.append(f"{rel} row {i}: {len(ra)} vs {len(rb)} cells"); continue
        for j in keep:
            if j < len(ra) and ra[j] != rb[j]:
                out.append(f"{rel} row {i} column {A[0][j]}: {ra[j]!r} vs {rb[j]!r}")


def cmp_file(rel, fa, fb, na, nb, out, json_ignore=IGNORED_JSON_KEYS):
    try:
        if rel.endswith(".npz"):
            cmp_npz(rel, fa, fb, out)
        elif rel.endswith(".npy"):
            cmp_arrays(rel, np.load(fa, allow_pickle=False), np.load(fb, allow_pickle=False), out)
        elif rel.endswith(".csv"):
            cmp_csv(rel, fa, fb, out)
        elif rel.endswith(".json"):
            cmp_json(rel, fa, fb, na, nb, out, json_ignore)
        elif Path(fa).read_bytes() != Path(fb).read_bytes():
            out.append(f"{rel}: bytes differ")
    except Exception as e:                       # unreadable file = FAIL, never a silent pass
        out.append(f"{rel}: could not compare ({type(e).__name__}: {e})")


def cmp_dirs(da, db, na, nb, json_ignore=IGNORED_JSON_KEYS):
    out = []
    Fa, Fb = files(da), files(db)
    if Fa != Fb:
        out.append(f"files only in BASE {sorted(set(Fa) - set(Fb))}, only in CAND {sorted(set(Fb) - set(Fa))}")
    for rel in sorted(set(Fa) & set(Fb)):
        cmp_file(rel, da / rel, db / rel, na, nb, out, json_ignore)
    return out, len(set(Fa) & set(Fb))


def rc(d):
    f = d / ".rc"
    return f.read_text().strip() if f.exists() else "missing"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("cand")
    ap.add_argument("--cases", default=None, help="comma-separated case names (and/or 'metrics'); default: all")
    ap.add_argument("--max-diffs", type=int, default=5, help="differences printed per failing case")
    a = ap.parse_args()
    A, B = run_dir(a.base), run_dir(a.cand); na, nb = Norm(A), Norm(B)
    only = a.cases.split(",") if a.cases else None
    md = a.max_diffs
    rows, details, bad = [], [], 0
    for c in cases():
        if only and c not in only:
            continue
        da, db = A / c, B / c
        if not da.is_dir() or not db.is_dir() or rc(da) != "0" or rc(db) != "0":
            diffs, n = [f"BASE exit {rc(da) if da.is_dir() else 'no folder'}, CAND exit {rc(db) if db.is_dir() else 'no folder'}"], 0
        else:
            diffs, n = cmp_dirs(da, db, na, nb)
        rows.append((c, n, diffs))
    if not only or "metrics" in only:            # tests/golden/metrics_golden.py
        fa, fb = A / "metrics.json", B / "metrics.json"
        diffs = []
        if fa.exists() and fb.exists():
            cmp_json("metrics.json", fa, fb, na, nb, diffs, ignore=set())
        else:
            diffs.append(f"metrics.json missing in {'BASE' if not fa.exists() else ''} {'CAND' if not fb.exists() else ''}")
        rows.append(("metrics", 1, diffs))
    for (c1, s1), (c2, s2), ign in RESUME_PAIRS:   # inside CAND: resumed search = straight search
        if only and not ({c1, c2} & set(only)):
            continue
        d1, d2 = B / c1 / s1, B / c2 / s2
        known = []
        if d1.is_dir() and d2.is_dir():
            diffs, n = cmp_dirs(d1, d2, nb, nb, json_ignore=IGNORED_JSON_KEYS | ign)
            known = [d for d in diffs if any(k in d for k in KNOWN_RESUME)]; diffs = [d for d in diffs if d not in known]
        else:
            diffs, n = [f"{d1} or {d2} missing"], 0
        rows.append((f"CAND {c2}/{s2} = {c1}", n, diffs, known))
    w = max(len(r[0]) for r in rows)
    print(f"golden comparison: BASE {A}  vs  CAND {B}")
    print(f"{'case':{w}s}  files  result")
    for c, n, diffs, *known in rows:
        ok = not diffs; bad += not ok
        kn = f"  ({len(known[0])} known --resume differences, see KNOWN_RESUME)" if known and known[0] else ""
        print(f"{c:{w}s}  {n:5d}  {'PASS' if ok else f'FAIL ({len(diffs)} differences)'}{kn}")
        if not ok:
            details.append((c, diffs))
    for c, diffs in details:
        print(f"\n--- {c}: first differences")
        for d in diffs[:md]:
            print(f"  {d}")
        if len(diffs) > md:
            print(f"  ... {len(diffs) - md} more")
    print(f"\n{len(rows) - bad} PASS, {bad} FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
