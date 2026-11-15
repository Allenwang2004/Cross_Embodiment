#!/usr/bin/env python3
"""make_scaled_bodies.py -- the 8 uniformly scaled bodies (user, 11/11: replace the earlier shape designs, child kept): the whole adult
scaled isometrically by L and then made wider or narrower by W: every segment length (leg, arm, torso, head) x L and every girth x L*W
(scale_robot.py's girth multiplies geom sizes absolutely, so girth = W alone would leave a short body with adult-sized head, feet and
limbs; user chose L*W on 11/11), L in {1.4, 1.2, 0.8, 0.6}, W in {1.2, 0.8} -> assets/robots/len<L>_wid<W>.

OOP refactor step 9: a thin wrapper. The steps of docs/new_body.md (shape, skeleton, retarget every motion folder, torque ratio,
aggregate k, robot_torque_full.xml, the measured / predicted check) are crossembodiment/bodybuild BodyPipeline, run here as
    uv run python scripts/build_body.py scaled-set <bodies> --parallel <n> [--workers 1]
(one process per body as before; finished steps are skipped). Logs: outputs/body_build/<body>.log (the old runs' logs stay in
outputs/scaled_bodies).
usage: uv run python scripts/make_scaled_bodies.py [all | <body> ...]
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BODIES = {f"len{L}_wid{W}": (L, W) for L in (1.4, 1.2, 0.8, 0.6) for W in (1.2, 0.8)}

if __name__ == "__main__":
    names = list(BODIES) if sys.argv[1:] in ([], ["all"]) else sys.argv[1:]
    cmd = ["uv", "run", "python", "scripts/build_body.py", "scaled-set", *names, "--parallel", str(len(names)), "--workers", "1"]
    print("$ " + " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, cwd=REPO).returncode)
