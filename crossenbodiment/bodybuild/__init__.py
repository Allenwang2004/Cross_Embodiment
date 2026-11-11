"""crossenbodiment.bodybuild -- making a new body (OOP refactor step 9; the runbook is docs/new_body.md, the CLI scripts/build_body.py).

One class per job:
  BodyBuilder (builders.py)   load the adult -> edit -> ground the pelvis -> robot.xml + an honest parameter.json
      ScaledBody                the 8 body parameters beta (beta.py BETA_AXES; len x wid; presets)   scale_robot.py, make_scaled_bodies.py
      HeavyLimb                 one arm's geoms heavier                                              heavy_arm_body.py
      LockJoints                joints held by equality constraints, their motors off                lock_joint_body.py
      LimitJoints               a lock body's locks as narrow joint ranges, motors kept              limit_joint_body.py
      Elderly (elderly.py)      bands per joint + the stooped, slowed reference walk                 elderly_body.py
      ActuatorStrength          every actuator's gear or torque limit x scale                        strength_rollout.py
  Retargeter (retarget.py)    the adult's clip -> the body's reference
      SameTopologyRetargeter    root x pelvis-height ratio, joints copied, lifted out of the floor   qpos_retarget.py
      CopyReference, LockedJointReference                                                            heavy / limit / lock_joint_body.py
      (robots: model/robot_map.py RobotMap + scripts/robot_roundtrip.py, a different topology: unchanged)
  TorqueCalibrator (torque.py) the k matrix -> aggregate k -> robot_torque_full.xml with its checks  torque_ratio_across_motions.py,
                                                                                                    torque_aggregate_motion_k.py,
                                                                                                    torque_scale_actuators.py's library
                                                                                                    (its one-clip CLI archived)
  GainScaler (gains.py)       kp / kd multipliers, MJCF (human bodies) or gains json (robots)       scale_body_gains.py, kpkd_test.py prepare
  BodyPipeline (pipeline.py)  the idempotent steps shape -> skeleton -> retarget -> torque ratio -> aggregate -> physics -> registry,
                              BuildRoots (where a build writes), run_many (several bodies, memory-gated)
Plain functions stay functions: the fitting math (torque.joint_torques, fit_ratio, predicted_ratios), the geometry edits
(shape.apply_scale, measure_min_z, ground_pelvis), the ground correction (retarget.ground_correct_qpos, min_geom_z),
skeleton.dump_skeleton / load_root_rest_height. Every one is the old code moved verbatim: the outputs are byte-identical
(tests/test_bodybuild.py, tests/golden/verify_bodies.py).

Importing the package imports nothing heavy: the names below are loaded from their module on first use (module __getattr__), so
`from crossenbodiment.bodybuild.beta import BETA_AXES` (model/dataset.py) stays light; the other modules import mujoco and numpy.
"""
import importlib

_WHERE = {
    "beta": ("BETA_AXES", "BODY_GROUPS", "GROUP_OF_BODY", "PRESETS", "grid_beta", "isometric_beta", "lerp_beta"),
    "builders": ("ActuatorStrength", "BodyBuilder", "HeavyLimb", "LimitJoints", "LockJoints", "ScaledBody", "SpecBody", "body_dir"),
    "elderly": ("Elderly",),
    "gains": ("GainScaler",),
    "pipeline": ("BodyPipeline", "BuildRoots", "StepResult", "register", "registry_entry", "run_many"),
    "retarget": ("CopyReference", "LockedJointReference", "Retargeter", "SameTopologyRetargeter", "ground_correct_qpos",
                 "min_geom_z", "retarget_qpos"),
    "skeleton": ("dump_skeleton", "load_root_rest_height"),
    "torque": ("CalibrationError", "TorqueCalibrator", "fit_ratio", "joint_torques", "predicted_ratios"),
}
_NAME = {n: m for m, names in _WHERE.items() for n in names}
__all__ = sorted(_NAME)


def __getattr__(name):
    if name in _NAME:
        return getattr(importlib.import_module(f".{_NAME[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
