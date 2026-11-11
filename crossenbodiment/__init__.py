"""crossenbodiment -- the library behind scripts/ (OOP refactor of 2026-11-11, docs/refactor/PLAN.md).

Importable when the repository root is on sys.path, as every script arranges (`sys.path.insert(0, str(REPO))`).
Step 2 adds the data layer only; model/ stays where it is (later steps move modules here and leave re-export shims):
  paths     the repository's locations (REPO from this package's own path, relative and absolute spellings)
  bodies    BodySpec / BodyRegistry (configs/bodies.yaml): every body, its XMLs, references and root scale s; RobotSpec
  methods   MethodSpec registry: the methods of the baseline comparison, their flags, expected settings and the
            run-directory key each experiment used for them
Step 3 adds the shared pure functions (the scripts' copies now call these):
  geometry  the latent sphere (project_z, sphere16, unit_sphere, ang_pair), ranks, quaternion product, heading / yaw / facing
  metrics   the evaluation metrics as named functions (Task page: contact_fall_time, phc_success, drift, heading_error; legacy:
            root_success, height_fall, hand_fall), world positions, MetricSuite (one clip's reference, many rollouts), metric_body
Importing the package imports nothing heavy (no torch, mujoco or humenv).
"""
