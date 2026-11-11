"""crossenbodiment -- the library behind scripts/ (OOP refactor of 2026-11-11, docs/refactor/PLAN.md).

Importable when the repository root is on sys.path, as every script arranges (`sys.path.insert(0, str(REPO))`).
Step 2 adds the data layer only; model/ stays where it is (later steps move modules here and leave re-export shims):
  paths     the repository's locations (REPO from this package's own path, relative and absolute spellings)
  bodies    BodySpec / BodyRegistry (configs/bodies.yaml): every body, its XMLs, references and root scale s; RobotSpec
  methods   MethodSpec registry: the methods of the baseline comparison, their flags, expected settings and the
            run-directory key each experiment used for them
Importing the package imports nothing heavy (no torch, mujoco or humenv).
"""
