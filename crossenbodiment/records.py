"""records.py -- RunRecorder: the files a search run writes, and reads back (OOP refactor step 6, docs/refactor/PLAN.md).

Every writer produces the file the search scripts wrote at fa6c5f9, byte for byte where the format is deterministic (.npy, .csv,
.json; an .npz is a zip whose member timestamps are the write time, so its ARRAYS are what stays identical: keys in the same
order, dtype, shape, bytes). Old analysis code reads these files, so nothing here may change a name, a column, a key order or a
number format (tests/test_records.py: write-then-read round trips, and bytes equal to a golden run's files).

  file             written by                                       content
  best_z.npy       single_z_search, robot_z_search                  (1, 256) float32: the best latent
  best.npz         single_z_search (qpos, fps 30.0, [dtw_path])     the best latent's rollout
                   robot_z_search (qpos, qpos_smpl, fz, fps 30, [dtw_path])
  origin_z.npz     the same, of z0 (crossenbodiment.paths.Z0_ROLLOUT; "origin_z" is z0's on-disk name, summary.json key too)
  z_trace.npz      both: gen (int32, from -1), z_mean, z_best (float32)
  curve.csv        csv.DictWriter, one row per generation (single latent) / iteration (latent sequence) (floats as str():
                   repr), the columns of the first row
                   (robot_z_search: of the LAST row, restval "", since a resumed older run's rows lack newer columns)
  summary.json     json.dumps(summary, indent=2) (single / robot latent) or indent=1 (lso_search, robot_lso_search)
  eval.csv         --eval-z: index + the cost terms; single_z_search formats floats "%.6g" (str() otherwise), robot_z_search
                   ints with str() and everything else "%.9g"
  eval_base.json   --eval-z: json.dumps(base terms), no indent
  eval_*.npz       --eval-z with --eval-save-*: one array each (qpos / act / qvel, float32)
  state.npz        robot_z_search: np.savez (not compressed) z, m, v, t_adam, path_len, best_z, rng (JSON of the bit generator)
  mu_trace.npz     lso_search, robot_lso_search: mu (the mean sequence after the traced iterations), mu_iters, best
  rollouts.npz     lso_search, robot_lso_search: ref, then the kept rollouts in the order they were made
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


class RunRecorder:
    """one run's output directory (created). The writers return the path written."""

    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.out / name

    # ------------------------------------------------------------------ writers
    def best_z(self, z):
        np.save(self.path("best_z.npy"), z.astype(np.float32).reshape(1, -1))
        return self.path("best_z.npy")

    def npz(self, name, compressed=True, **arrays):
        """np.savez_compressed (or np.savez) of the arrays, in the order given"""
        (np.savez_compressed if compressed else np.savez)(self.path(name), **arrays)
        return self.path(name)

    def z_trace(self, trace_mean, trace_best):
        """gen -1 is the start latent, so row i is "after i steps" (scripts/rollout_z_trace.py indexes it that way)"""
        return self.npz("z_trace.npz", gen=np.arange(-1, len(trace_mean) - 1, dtype=np.int32),
                        z_mean=np.stack(trace_mean).astype(np.float32), z_best=np.stack(trace_best).astype(np.float32))

    def curve(self, rows, columns_from=0, restval=""):
        """curve.csv: csv.DictWriter with the columns of rows[columns_from] (0 = the first row, -1 = the last)"""
        with open(self.path("curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[columns_from]), restval=restval)
            w.writeheader()
            w.writerows(rows)
        return self.path("curve.csv")

    def json(self, name, obj, indent=None):
        self.path(name).write_text(json.dumps(obj, indent=indent))
        return self.path(name)

    def summary(self, summ, indent=2):
        return self.json("summary.json", summ, indent=indent)

    def eval_csv(self, columns, rows, style="single"):
        """--eval-z's eval.csv. style single: float -> "%.6g", anything else str(); robot: int -> str(), anything else "%.9g"."""
        if style == "single":
            fmt = lambda x: f"{x:.6g}" if isinstance(x, float) else str(x)     # noqa: E731
        elif style == "robot":
            fmt = lambda x: str(x) if isinstance(x, int) else f"{x:.9g}"      # noqa: E731
        else:
            raise ValueError(style)
        with open(self.path("eval.csv"), "w") as fh:
            fh.write(",".join(columns) + "\n")
            fh.writelines(",".join(fmt(x) for x in r) + "\n" for r in rows)
        return self.path("eval.csv")

    def state(self, z, m, v, t_adam, path_len, best_z, rng):
        """robot_z_search.py's state.npz: everything --resume needs to continue the search exactly (np.savez, not compressed;
        rng: the bit generator's state as a JSON string)"""
        return self.npz("state.npz", compressed=False, z=z, m=m, v=v, t_adam=t_adam, path_len=path_len, best_z=best_z, rng=rng)

    def mu_trace(self, trace, its, best):
        return self.npz("mu_trace.npz", mu=np.stack([trace[k] for k in its]), mu_iters=np.array(its), best=best.astype(np.float32))

    def rollouts(self, ref, keep):
        return self.npz("rollouts.npz", ref=ref.astype(np.float32), **keep)

    # ------------------------------------------------------------------ readers
    def read_json(self, name):
        return json.loads(self.path(name).read_text())

    def read_summary(self):
        return self.read_json("summary.json")

    def read_curve(self):
        """the rows as csv.DictReader gives them (every value a str)"""
        with open(self.path("curve.csv")) as f:
            return list(csv.DictReader(f))

    def read_npz(self, name):
        with np.load(self.path(name)) as d:
            return {k: d[k] for k in d.files}

    def read_z_trace(self):
        return self.read_npz("z_trace.npz")

    def read_state(self):
        return self.read_npz("state.npz")

    def read_best_z(self):
        return np.load(self.path("best_z.npy"))

    def has(self, name):
        return self.path(name).exists()
