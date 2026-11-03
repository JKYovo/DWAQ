"""Versioned, zero-command-only default-leg-pose continuation experiment."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from legged_lab.envs.elf3.still_pose_rewards import STAND_POSE_JOINT_NAMES


VERSION = "elf3_dwaq_still_pose_v1"
PARENT_ITERATION = 100000
PARENT_SHA256 = "a5b191bdde7b61a4d9016d4fabe7b181e1675f12048c9de791b7cf711702769e"


def pose_settings():
    return {
        "version": VERSION,
        "command_epsilon": 1e-6,
        "joint_names": list(STAND_POSE_JOINT_NAMES),
        "incremental_weight": -0.08,
        "baseline_weight": -0.02,
        "total_zero_weight": -0.10,
        "target": "asset.default_joint_pos",
    }


def pose_experiment_contract(settings):
    expected = pose_settings()
    if json.dumps(settings, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unsupported zero-command pose settings; version changes explicitly")
    directory = Path(__file__).resolve().parent
    names = ("still_pose_contract.py", "still_pose_rewards.py", "elf3_still_pose_config.py",
             "still_pose_runner.py")
    return {
        **deepcopy(expected),
        "activation": "current strict-zero vx/vy/wz only; no settled, contact, terrain or timer gate",
        "cost": "sum absolute actual joint-position error; four hip-pitch/knee joints",
        "moving_pose_weight": -0.02,
        "still_preprocessing_and_motion_cost": "unchanged elf3_dwaq_still_v1",
        "height_braking_PD_asset_and_actor_changes": False,
        "sources_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                           for name in names},
    }
