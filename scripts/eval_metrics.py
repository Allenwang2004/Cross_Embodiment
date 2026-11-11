#!/usr/bin/env python3
"""eval_metrics.py -- moved to crossenbodiment/eval_metrics.py (OOP refactor step 8, docs/refactor/PLAN.md); this file re-exports it
so the scripts' `import eval_metrics as E` (and `from eval_metrics import ...`) get the package module itself (one Body class), and
keeps the CLI: uv run python scripts/eval_metrics.py [child g1 child_amass g1_amass] -> outputs/eval_metrics/<set>.json"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import crossenbodiment.eval_metrics as _E  # noqa: E402
from crossenbodiment.eval_metrics import *  # noqa: E402,F401,F403
from crossenbodiment.eval_metrics import METRICS, SETS, XML, Body, dtw, main, metrics, tracking_metrics  # noqa: E402,F401

if __name__ == "__main__":
    main()
else:
    sys.modules[__name__] = _E              # `import eval_metrics` is crossenbodiment.eval_metrics (the same module object)
