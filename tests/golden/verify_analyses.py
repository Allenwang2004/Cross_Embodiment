#!/usr/bin/env python3
"""verify_analyses.py -- OOP refactor step 8's check of the ported drivers on real data: each driver's analyse (and prepare) is run on
the MAIN tree's finished experiments, READ ONLY (CE_OUTPUTS_ROOT=<main>/outputs), writing into the worktree
(CE_RESULTS_ROOT=outputs/refactor_step8/<label>), and every file it writes is compared with the main tree's existing one:
  .json   byte-identical (json.dumps of the same objects in the same order); on a difference the first differing values are listed
  .npz    the same keys in the same order, and per array the same dtype, shape and bytes
  .md     byte-identical text (the docstring included: the drivers' docstrings are unchanged)
  prepare z.npy byte-identical to the main tree's, and the schema-2 index.json read back gives the same lists as the main tree's
          old-layout index (best per generation / per budget / last, centre per budget) for every run
Nothing under the main tree is written: the drivers' writers refuse paths outside the worktree (crossenbodiment/paths.writable).

usage: uv run python tests/golden/verify_analyses.py LABEL [case ...] [--workers N] [--compare-only] [--list]
       (cases run one after the other; each case's stdout/stderr in outputs/refactor_step8/<label>/_logs/<case>.log)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("CROSSENBODIMENT_MAIN", "/home/allen19/crossenbodiment"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

# case -> (driver, argv, [files written, relative to outputs/]); the files are compared with the main tree's
CASES = {
    "child_52_rerun": ("child_52_rerun", ["analyse"], ["child_52_rerun/per_run.npz", "child_52_rerun/means.json"]),
    "child_52_cem_phc": ("child_52_cem_phc", ["analyse"], ["child_52_cem_phc/per_run.npz", "child_52_cem_phc/means.json"]),
    "child_52_lso": ("child_52_lso", ["analyse"], ["child_52_lso/per_run.npz", "child_52_lso/means.json"]),
    "body_52": ("body_52_test", ["analyse"], ["body_52/payload/results.json", "body_52/limit_leg/results.json"]),
    "g1_52": ("g1_small_test", ["--set", "52", "analyse"], ["g1_52/results.json"]),
    "g1_small": ("g1_small_test", ["analyse"], ["g1_small/results.json"]),
    "kpkd": ("kpkd_test", ["analyse"], ["kpkd/per_run.npz", "kpkd/means.json"]),
    "kpkd_contactfall": ("kpkd_test", ["contactfall"], ["kpkd/means_contactfall_without_none.json"]),
    "kpkd_phcsuccess": ("kpkd_test", ["phcsuccess"], ["kpkd/means_phcsuccess.json"]),
    "noanchor": ("loes_noanchor", ["analyse"], ["noanchor/results.json", "noanchor/README.md"]),
    "tall_body_52": ("tall_body_52", ["analyse"], ["tall_body_52/results.json", "tall_body_52/README.md"]),
    "cost_corr": ("cost_correlation", ["analyse"], ["cost_corr/results.json", "cost_corr/per_latent.npz", "cost_corr/README.md"]),
    "success_definitions": ("success_definitions", [], ["success_definitions/task_motions.json"]),
    "success_definitions_noanchor": ("success_definitions", ["--loes-dir", "outputs/noanchor/child"], ["success_definitions/task_motions_noanchor.json"]),
    "fall_definitions": ("fall_definitions", [], ["fall_definitions/task_motions.json"]),
    "fall_definitions_noanchor": ("fall_definitions", ["--loes-dir", "outputs/noanchor/child"], ["fall_definitions/task_motions_noanchor.json"]),
    "drift_success_child": ("drift_success_child", [], ["drift_success_child.json"]),
    "drift_success_child_noanchor": ("drift_success_child", ["--loes-dir", "outputs/noanchor/child"], ["drift_success_child_noanchor.json"]),
    "loes_weight_test": ("loes_weight_test", ["analyse"], ["loes_weight_test/results.json"]),
    "lsor_mpjae_test": ("lsor_mpjae_test", [], ["lsor_mpjae_test/results.json"]),
}
# prepare cases: driver, argv, [(rollouts dir relative to outputs/, budgets, gens)] -> compared folder by folder
PREPARES = {
    "prepare_child_52_rerun": ("child_52_rerun", ["prepare"], "child_52_rerun/rollouts", (4, 8, 16, 32, 64), 64),
    "prepare_child_52_cem_phc": ("child_52_cem_phc", ["prepare"], "child_52_cem_phc/rollouts", (4, 8, 16, 32, 64), 64),
    "prepare_body_52": ("body_52_test", ["prepare", "--algos", "es,cem,random,cmaes"], "body_52/*/rollouts", (4, 8, 16, 32, 64, 128), 128),
    "prepare_g1_52": ("g1_small_test", ["--set", "52", "prepare"], "g1_52/rollouts", (4, 8, 16, 32, 64, 128), 128),
    "prepare_g1_small": ("g1_small_test", ["prepare"], "g1_small/rollouts", (4, 8, 16, 32, 64), 64),
    "prepare_noanchor_child": ("loes_noanchor", ["prepare", "--bodies", "child"], "noanchor/child/rollouts", (4, 8, 16, 32, 64), 64),
    "prepare_noanchor_bodies": ("loes_noanchor", ["prepare", "--bodies", "payload,limit_leg,g1"], "noanchor/[plg]*/rollouts",
                                (4, 8, 16, 32, 64, 128), 128),
    "prepare_loes_weight_test": ("loes_weight_test", ["prepare"], "loes_weight_test/rollouts", (64,), 64),
}
ENV = dict(MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="2", MUJOCO_EGL_DEVICE_ID="2", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1")


def first_diffs(a, b, path="", out=None, n=8):
    """the first n differing leaves of two parsed JSON values (key order included)"""
    out = [] if out is None else out
    if len(out) >= n:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            out.append((path + " keys", list(a)[:8], list(b)[:8]))
        for k in a:
            if k in b:
                first_diffs(a[k], b[k], f"{path}/{k}", out, n)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + " len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            first_diffs(x, y, f"{path}[{i}]", out, n)
    elif not (a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)) or type(a) is not type(b):
        out.append((path, a, b))
    return out


def compare_file(new, old):
    if not new.exists():
        return False, "not written"
    if new.suffix == ".npz":
        A, B = np.load(new), np.load(old)
        if A.files != B.files:
            return False, f"keys {A.files[:6]} vs {B.files[:6]}"
        for k in A.files:
            x, y = A[k], B[k]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                return False, f"array {k}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}" + (
                    f", largest difference {np.nanmax(np.abs(x.astype(float) - y.astype(float)))}" if x.shape == y.shape and x.dtype.kind == "f" else "")
        return True, f"{len(A.files)} arrays identical"
    a, b = new.read_bytes(), old.read_bytes()
    if a == b:
        return True, f"{len(a)} bytes identical"
    if new.suffix == ".json":
        d = first_diffs(json.loads(a), json.loads(b))
        return False, f"JSON differs: {d}" if d else "JSON equal as values but not as text"
    la, lb = a.decode().splitlines(), b.decode().splitlines()
    i = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
    return False, f"text differs at line {i + 1}: {la[i:i + 1]} vs {lb[i:i + 1]}"


def compare_prepare(res, rel, budgets, gens):
    from crossenbodiment.reroll import Reroller
    n, bad = 0, []
    for old_dir in sorted((MAIN / "outputs").glob(rel + "/*")):
        if not (old_dir / "index.json").exists():
            continue
        new_dir = res / old_dir.relative_to(MAIN / "outputs")
        if not (new_dir / "index.json").exists():
            bad.append((str(new_dir), "not written")); continue
        if (new_dir / "z.npy").read_bytes() != (old_dir / "z.npy").read_bytes():
            bad.append((old_dir.name, "z.npy differs")); continue
        O, N = Reroller.read(old_dir, budgets, gens), Reroller.read(new_dir)
        if N.schema != 2 or O.runs != N.runs or {k: v for k, v in O.data.items() if k != "map"} != {k: v for k, v in N.data.items() if k in O.data and k != "map"}:
            bad.append((old_dir.name, "runs / meta differ")); continue
        for r in O.runs:
            same = (O.best_generations(r) == N.best_generations(r) and O.best_last(r) == N.best_last(r) and O.rerolled(r) == N.rerolled(r)
                    and (not isinstance(O.map[r], dict) or (O.centre(r) == N.centre(r) and O.best_budgets(r) == N.best_budgets(r))))
            if not same:
                bad.append((old_dir.name, r)); break
        n += 1
    return not bad and n > 0, f"{n} folders: z.npy byte-identical, index lists equal" if not bad else f"{len(bad)} differ: {bad[:4]}"


def main():
    argv = sys.argv[1:]
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and not (i and argv[i - 1] == "--workers")]
    if "--list" in sys.argv:
        print("\n".join(list(CASES) + list(PREPARES))); return 0
    label, want = args[0], args[1:] or list(CASES) + list(PREPARES)
    assert label not in ("baseline",), label
    res = REPO / "outputs/refactor_step8" / label; logs = res / "_logs"; logs.mkdir(parents=True, exist_ok=True)
    workers = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "12"
    env = dict(os.environ, **ENV, CE_OUTPUTS_ROOT=str(MAIN / "outputs"), CE_RESULTS_ROOT=str(res), CE_WORKERS=workers, UV_NO_SYNC="1")
    summary, ok_all = {}, True
    for case in want:
        driver, argv, files = (CASES[case] if case in CASES else PREPARES[case][:2] + (None,))
        t0 = time.time()
        if "--compare-only" not in sys.argv:
            with open(logs / f"{case}.log", "w") as f:
                rc = subprocess.run([sys.executable, str(REPO / "scripts" / f"{driver}.py"), *argv], cwd=REPO, env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
        else:
            rc = 0
        if case in CASES:
            checks = [(rel, *compare_file(res / rel, MAIN / "outputs" / rel)) for rel in files]
        else:
            _, _, rel, budgets, gens = PREPARES[case]; checks = [(rel, *compare_prepare(res, rel, budgets, gens))]
        ok = rc == 0 and all(c[1] for c in checks); ok_all &= ok
        summary[case] = dict(exit=rc, seconds=round(time.time() - t0), ok=ok, files={c[0]: c[2] for c in checks})
        print(f"{'PASS' if ok else 'FAIL'} {case} (exit {rc}, {time.time() - t0:.0f} s)", flush=True)
        for rel, good, why in checks:
            print(f"    {'ok ' if good else 'BAD'} {rel}: {why}", flush=True)
        prev = json.loads((res / "verify.json").read_text()) if (res / "verify.json").exists() else {}
        (res / "verify.json").write_text(json.dumps({**prev, case: summary[case]}, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
