"""Independent scratch experiment: original still recipe plus leg/waist pose.

No project-package imports are needed, so this identity is inspectable by the
CPU exporter without importing Isaac or constructing a training environment.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


VERSION = "elf3_dwaq_still_v2"
PARENT_TASK = "elf3_dwaq_still"
REFERENCE_TASK = "elf3_dwaq_still_pose_v1"


def pose_settings():
    return {
        "version": VERSION,
        "parent_task": PARENT_TASK,
        "reference_task": REFERENCE_TASK,
        "command_epsilon": 1e-6,
        "leg_joint_names": ["l_hip_y_joint", "l_knee_y_joint", "r_hip_y_joint", "r_knee_y_joint"],
        "leg_incremental_weight": -.13,
        "leg_baseline_weight": -.02,
        "leg_total_zero_weight": -.15,
        "waist_joint_names": ["waist_y_joint", "waist_x_joint", "waist_z_joint"],
        "waist_incremental_weight": -.10,
        "waist_baseline_weight": -.20,
        "waist_total_zero_weight": -.30,
        "target": "asset.default_joint_pos",
    }


def pose_experiment_contract(settings):
    expected = pose_settings()
    if json.dumps(settings, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unsupported still v2 pose settings; version changes explicitly")
    directory = Path(__file__).resolve().parent
    names = ("still_v2_contract.py", "still_v2_rewards.py", "elf3_still_v2_config.py",
             "still_v2_runner.py", "still_pose_rewards.py")
    return {
        **deepcopy(expected),
        "activation": "current finite strict-zero vx/vy/wz; no settled/contact/terrain/posture/timer gate",
        "cost": "sum absolute actual q-default q; four hip-pitch/knee and three waist joints",
        "initialization": "from_scratch",
        "moving_leg_weight": -.02,
        "moving_waist_weight": -.20,
        "still_preprocessing_and_motion_cost": "unchanged elf3_dwaq_still_v1",
        "height_braking_PD_asset_and_actor_changes": False,
        "sources_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                           for name in names},
    }
