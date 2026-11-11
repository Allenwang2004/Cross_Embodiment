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
Step 4 moves library code out of scripts (the old modules import it back; tests/test_moves.py runs each against its old source):
  align_cost       compute_batch_cost + AlignCostConfig (model/simple's L_align / L_phys cost; single_z_search.py)
  metrics          + legacy_metrics / body_hand_fall / LEGACY_KEYS (child_52_means.py, archived), pose_similarity
                   (mse_vs_align_test.metrics)
  survey, palette  the user survey's clips, bodies and run lookup (configs/survey.yaml), the video colours (survey_body_videos.py)
  robot_rollout    act / motion of the robot rollouts (g1_z0_rollout.py)
  retiming         resample: a clip at another speed (bfm_velocity_check.py)
  draw             frame_label (rollout_z_trace.label) and the explanation-panel helpers of g1_axes_demo.py
  multibody_study  the 10/10 multi-body correction experiment's layout (multibody_corr.py)
  paths            + the clip lists configs/clips/*.txt (read_clips)
Importing the package imports nothing heavy (no torch, mujoco or humenv); align_cost (model.losses -> mujoco) and robot_rollout
(torch) import them when they are imported.
"""
