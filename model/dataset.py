"""Reads a manifest.jsonl written by scripts/build_dataset.py.

Two datasets exist and they are not interchangeable:

`datasets/crossenbodiment-10bodies` (current)
    540 clips x 10 bodies = 5400 rows, built over the per-body artifacts from
    docs/new_body.md. Every row carries a retargeted_motion, so qpos_ref is
    real and the D term in model/simple/train.py's objective is live. The rows
    also carry `morphology_label` / `target_xml`, because beta now VARIES --
    a trainer has to roll each row out on its own body's MJCF, not on one
    global cfg.target_xml.

`datasets/crossenbodiment-1-datasets` (legacy, what the published baseline used)
    1530 rows, all `child`, and retargeted_motion was deleted from it in favour
    of model/bilevel's runtime retargeting -- so qpos_ref is None for every row,
    functional_equivalence returns 0.0, and D is identically zero there. The
    constructor says so out loud rather than letting the loss quietly collapse.
    Its task balance is also 1000:10 (`move-ego--90-2` alone holds 1000 of the
    1530 rows); those 990 extra trials are exactly the rows that never had a
    retargeted_motion.

Rebuild the current one with:
    uv run scripts/build_dataset.py --force
"""

import json
from pathlib import Path

import numpy as np

# Must match scripts/scale_robot.py's AXES order -- beta is this 8-dim
# vector read straight out of a robot_<label>_parameter.json (NOT
# robot_<label>.json -- that name now means the skeleton-export schema with
# a "bodies" list, see scripts/export_skeleton_json.py).
BETA_AXES = ["leg_scale", "arm_scale", "torso_scale", "head_scale",
             "leg_girth", "arm_girth", "torso_girth", "head_girth"]


def load_beta(morphology_json_path) -> np.ndarray:
    d = json.loads(Path(morphology_json_path).read_text())
    return np.array([d[a] for a in BETA_AXES], dtype=np.float32)


def load_task_list(path) -> list:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


class CrossEmbodimentDataset:
    def __init__(self, dataset_dir, task_filter=None):
        """task_filter: optional iterable of reward_name strings (e.g. from
        datasets/crossenbodiment-1-datasets/splits/train_tasks.txt or
        test_tasks.txt, see scripts/split_tasks.py) -- only rows whose
        reward_name is in this set are kept. Use this to keep train/test
        task splits from leaking into each other."""
        self.dataset_dir = Path(dataset_dir)
        manifest_path = self.dataset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found -- run scripts/build_dataset.py first"
            )
        self.rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if task_filter is not None:
            task_filter = set(task_filter)
            self.rows = [r for r in self.rows if r["reward_name"] in task_filter]
        if not self.rows:
            raise ValueError(f"no rows in {manifest_path} (after task_filter)")

        if not any(r.get("retargeted_motion") for r in self.rows):
            print(
                "WARNING: this dataset has no retargeted_motion -- it was removed in favour of "
                "model/bilevel's runtime retargeting.\n"
                "         qpos_ref will be None for every row, so functional_equivalence "
                "returns 0.0 and the\n"
                "         D term of model/simple/train.py's loss is identically zero. See this "
                "module's docstring to regenerate."
            )

    def __len__(self):
        return len(self.rows)

    def bodies(self) -> list:
        """Distinct morphology labels present, in manifest order."""
        seen = {}
        for r in self.rows:
            seen.setdefault(r.get("morphology_label", "child"), None)
        return list(seen)

    def indices_by_body(self) -> dict:
        """{label: [row index, ...]} -- one vectorized env per body means the
        sampler has to draw within a body, not across all rows."""
        out = {}
        for i, r in enumerate(self.rows):
            out.setdefault(r.get("morphology_label", "child"), []).append(i)
        return out

    def __getitem__(self, idx):
        row = self.rows[idx]
        z0 = np.load(self.dataset_dir / row["origin_z"]).reshape(-1).astype(np.float32)
        beta = load_beta(self.dataset_dir / row["morphology"])

        # Kept as an optional field so an older manifest (or a regenerated one)
        # still works; the current dataset has none, see the module docstring.
        qpos_ref = None
        rel = row.get("retargeted_motion")
        if rel:
            qpos_ref = np.load(self.dataset_dir / rel)["qpos"]

        return {
            "reward_name": row["reward_name"],
            "trial": row["trial"],
            # Which body this row is FOR. Absent from the legacy single-body
            # manifest, where it was always "child"; a multi-body trainer must
            # group by it, since rolling a row out on the wrong MJCF silently
            # scores the adapter against a body it was not asked about.
            "morphology_label": row.get("morphology_label"),
            "target_xml": row.get("target_xml"),
            "z0": z0,
            "beta": beta,
            "qpos_ref": qpos_ref,
        }
